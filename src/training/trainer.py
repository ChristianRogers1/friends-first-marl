"""
Two-phase training pipeline for cooperative pretraining experiments.

Phase 1 (cooperative): Agents train in isolation on the referential
    communication task. No adversary, private messages.
Phase 2 (adversarial): Eavesdropper introduced, public channel,
    all agents train simultaneously.

Key design decisions (v2):
  - Pretraining and Phase 2 durations are specified independently
    (``pretraining_episodes`` and ``phase2_episodes``) so that changing
    the amount of pretraining does not alter the evaluation window.
  - Exploration noise is reset to its initial value at the Phase 2
    boundary so both conditions get an equal exploration budget.
  - The replay buffer supports three phase-transition strategies
    (``none``, ``flush``, ``label``) to control whether Phase 1
    experience contaminates Phase 2 training.
  - CIC is computed periodically during training (controlled by
    ``cic_eval_every``) and logged to TensorBoard.
"""

import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from ..environments.adversarial_crypto import AdversarialCryptoEnv
from ..agents.maddpg import MADDPGAgent
from ..utils.replay_buffer import ReplayBuffer


def _quick_cic(
    env: AdversarialCryptoEnv,
    agents: dict[str, MADDPGAgent],
    config: dict,
    num_episodes: int = 100,
    num_counterfactuals: int = 5,
) -> float:
    """Lightweight CIC estimate for periodic logging.

    Uses fewer episodes and counterfactuals than the full post-hoc
    evaluation to keep the cost manageable during training.
    """
    vocab_size = config["env"]["vocab_size"]
    msg_length = config["env"]["msg_length"]

    prev_phase = env.adversary_active
    env.set_phase(2)
    for agent in agents.values():
        agent.eval_mode()

    cic_values = []
    for _ in range(num_episodes):
        obs = env.reset()
        speaker_obs = obs["speaker"]

        speaker_action = agents["speaker"].select_action(
            speaker_obs, explore=False
        )
        actual_msg = TwoPhaseTrainer._onehot_to_discrete(
            speaker_action, vocab_size, msg_length
        )
        post_obs = env.step_speaker(actual_msg)
        listener_obs_actual = post_obs["listener"]

        with torch.no_grad():
            obs_t = torch.FloatTensor(listener_obs_actual).unsqueeze(0)
            actual_dist = (
                agents["listener"]
                .actor(obs_t.to(agents["listener"].device))
                .cpu()
                .numpy()
                .flatten()
            )

        kls = []
        for _ in range(num_counterfactuals):
            cf_msg = np.random.randint(0, vocab_size, size=msg_length)
            cf_post = env.step_speaker(cf_msg)
            cf_obs = cf_post["listener"]
            with torch.no_grad():
                cf_t = torch.FloatTensor(cf_obs).unsqueeze(0)
                cf_dist = (
                    agents["listener"]
                    .actor(cf_t.to(agents["listener"].device))
                    .cpu()
                    .numpy()
                    .flatten()
                )
            kls.append(_kl_div(actual_dist, cf_dist))
        cic_values.append(np.mean(kls))

    # Restore previous phase
    env.set_phase(2 if prev_phase else 1)
    for agent in agents.values():
        agent.train_mode()

    return float(np.mean(cic_values))


def _kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


class TwoPhaseTrainer:
    """Manages the two-phase cooperative pretraining pipeline."""

    def __init__(self, config: dict):
        self.config = config
        self.device = config["experiment"]["device"]

        # Build environment
        self.env = AdversarialCryptoEnv(**config["env"])

        # Compute dimensions for centralized critic
        # Total obs = speaker_obs + listener_obs + adversary_obs
        self.obs_dims = {
            name: self.env.get_obs_dim(name)
            for name in AdversarialCryptoEnv.AGENT_NAMES
        }
        self.action_dims = {
            name: self.env.get_action_dim(name)
            for name in AdversarialCryptoEnv.AGENT_NAMES
        }

        total_obs_dim = sum(self.obs_dims.values())
        total_action_dim = sum(self.action_dims.values())

        # Build agents
        self.agents: dict[str, MADDPGAgent] = {}
        for name in AdversarialCryptoEnv.AGENT_NAMES:
            self.agents[name] = MADDPGAgent(
                agent_name=name,
                obs_dim=self.obs_dims[name],
                action_dim=self.action_dims[name],
                total_obs_dim=total_obs_dim,
                total_action_dim=total_action_dim,
                hidden_dim=config["agent"]["hidden_dim"],
                lr_actor=config["agent"]["lr_actor"],
                lr_critic=config["agent"]["lr_critic"],
                gamma=config["agent"]["gamma"],
                tau=config["agent"]["tau"],
                discrete=True,
                vocab_size=config["env"]["vocab_size"],
                msg_length=config["env"]["msg_length"],
                comm_hidden_dim=config["agent"]["comm_hidden_dim"],
                gumbel_temperature=config["agent"]["gumbel_temperature"],
                device=self.device,
            )

        # Replay buffer (phase-aware)
        phase_strategy = config["training"].get(
            "phase_transition_buffer", "none"
        )
        self.buffer = ReplayBuffer(
            config["training"]["buffer_capacity"],
            phase_strategy=phase_strategy,
        )

        # ---- Episode counts (decoupled) ----
        training_cfg = config["training"]

        # Support both the old ``pretraining_ratio`` / ``total_episodes``
        # interface and the new decoupled ``pretraining_episodes`` /
        # ``phase2_episodes`` interface.
        if "pretraining_episodes" in training_cfg:
            self.phase1_episodes = int(training_cfg["pretraining_episodes"])
        else:
            total = int(training_cfg["total_episodes"])
            ratio = float(training_cfg.get("pretraining_ratio", 0.0))
            self.phase1_episodes = int(total * ratio)

        if "phase2_episodes" in training_cfg:
            self.phase2_episodes = int(training_cfg["phase2_episodes"])
        else:
            total = int(training_cfg["total_episodes"])
            self.phase2_episodes = total - self.phase1_episodes

        # Noise schedule
        self.noise_initial = training_cfg["noise_scale"]
        self.noise_scale = self.noise_initial
        self.noise_decay = training_cfg["noise_decay"]
        self.min_noise = training_cfg["min_noise"]
        self.reset_noise_on_phase2 = training_cfg.get(
            "reset_noise_on_phase2", True
        )

        # CIC tracking
        self.cic_eval_every = training_cfg.get("cic_eval_every", 0)

    def _collect_episode(self, phase: int) -> dict:
        """Run one episode and store the transition in the replay buffer.

        Args:
            phase: Current training phase (1 = cooperative, 2 = adversarial).

        Returns:
            Episode info dictionary.
        """
        self.env.set_phase(phase)
        obs = self.env.reset()

        # Speaker acts
        speaker_obs = obs["speaker"]
        speaker_action = self.agents["speaker"].select_action(
            speaker_obs, explore=True, noise_scale=self.noise_scale
        )

        # Get observations after speaker message
        # Decode the one-hot message back to discrete symbols for the env
        msg_discrete = self._onehot_to_discrete(
            speaker_action,
            self.config["env"]["vocab_size"],
            self.config["env"]["msg_length"],
        )
        post_speaker_obs = self.env.step_speaker(msg_discrete)

        # Listener acts
        listener_obs = post_speaker_obs["listener"]
        listener_action = self.agents["listener"].select_action(
            listener_obs, explore=True, noise_scale=self.noise_scale
        )
        listener_selection = np.argmax(listener_action)
        self.env.step_listener(listener_selection)

        # Adversary acts (only if in phase 2)
        if phase >= 2 and "adversary" in post_speaker_obs:
            adversary_obs = post_speaker_obs["adversary"]
            adversary_action = self.agents["adversary"].select_action(
                adversary_obs, explore=True, noise_scale=self.noise_scale
            )
            adversary_recon = self._onehot_to_discrete(
                adversary_action,
                self.config["env"]["num_values"],
                self.config["env"]["num_attributes"],
            )
            rewards = self.env.step_adversary(adversary_recon)
        else:
            # Phase 1: no adversary
            adversary_obs = np.zeros(
                self.obs_dims["adversary"], dtype=np.float32
            )
            adversary_action = np.zeros(
                self.action_dims["adversary"], dtype=np.float32
            )
            rewards = {
                "speaker": float(self.env._listener_correct),
                "listener": float(self.env._listener_correct),
                "adversary": 0.0,
                "listener_correct": self.env._listener_correct,
                "adversary_accuracy": 0.0,
            }

        # Next obs (episode is one-step, so next_obs = terminal)
        # For this referential game, each episode is a single step
        next_obs_reset = self.env.reset()

        transition = {
            "obs": {
                "speaker": speaker_obs,
                "listener": listener_obs,
                "adversary": adversary_obs,
            },
            "actions": {
                "speaker": speaker_action,
                "listener": listener_action,
                "adversary": adversary_action,
            },
            "rewards": {
                "speaker": rewards["speaker"],
                "listener": rewards["listener"],
                "adversary": rewards["adversary"],
            },
            "next_obs": {
                "speaker": next_obs_reset["speaker"],
                "listener": np.zeros(
                    self.obs_dims["listener"], dtype=np.float32
                ),
                "adversary": np.zeros(
                    self.obs_dims["adversary"], dtype=np.float32
                ),
            },
            "done": True,
        }

        self.buffer.push(transition)

        return {
            "listener_correct": rewards["listener_correct"],
            "adversary_accuracy": rewards["adversary_accuracy"],
            "speaker_reward": rewards["speaker"],
        }

    def _update_agents(self, phase: int):
        """Sample a batch and update all active agents.

        Args:
            phase: Current training phase.
        """
        batch_size = self.config["training"]["batch_size"]
        if len(self.buffer) < batch_size:
            return

        # When using "label" strategy, only sample from the current phase
        sample_phase = phase if self.buffer.phase_strategy == "label" else None
        batch = self.buffer.sample(batch_size, phase=sample_phase)

        if len(batch) < batch_size:
            return

        # Prepare batched tensors
        agent_names = ["speaker", "listener"]
        if phase >= 2:
            agent_names.append("adversary")

        obs_batch = {
            name: torch.FloatTensor(
                np.array([t["obs"][name] for t in batch])
            ).to(self.device)
            for name in AdversarialCryptoEnv.AGENT_NAMES
        }
        action_batch = {
            name: torch.FloatTensor(
                np.array([t["actions"][name] for t in batch])
            ).to(self.device)
            for name in AdversarialCryptoEnv.AGENT_NAMES
        }
        reward_batch = {
            name: torch.FloatTensor(
                np.array([[t["rewards"][name]] for t in batch])
            ).to(self.device)
            for name in AdversarialCryptoEnv.AGENT_NAMES
        }
        next_obs_batch = {
            name: torch.FloatTensor(
                np.array([t["next_obs"][name] for t in batch])
            ).to(self.device)
            for name in AdversarialCryptoEnv.AGENT_NAMES
        }
        done_batch = torch.FloatTensor(
            np.array([[float(t["done"])] for t in batch])
        ).to(self.device)

        # Concatenate all obs and actions for centralized critic
        all_obs = torch.cat(
            [obs_batch[n] for n in AdversarialCryptoEnv.AGENT_NAMES], dim=-1
        )
        all_actions = torch.cat(
            [action_batch[n] for n in AdversarialCryptoEnv.AGENT_NAMES], dim=-1
        )
        all_next_obs = torch.cat(
            [next_obs_batch[n] for n in AdversarialCryptoEnv.AGENT_NAMES],
            dim=-1,
        )

        # Compute target actions
        target_actions = []
        for name in AdversarialCryptoEnv.AGENT_NAMES:
            target_actions.append(
                self.agents[name].get_target_action(next_obs_batch[name])
            )
        all_next_actions = torch.cat(target_actions, dim=-1)

        # Update each active agent
        for name in agent_names:
            agent = self.agents[name]

            # Update critic
            agent.update_critic(
                all_obs,
                all_actions,
                reward_batch[name],
                all_next_obs,
                all_next_actions,
                done_batch,
            )

            # Update actor: replace this agent's action with current policy output
            current_action = agent.get_action_tensor(obs_batch[name])
            actions_list = []
            for n in AdversarialCryptoEnv.AGENT_NAMES:
                if n == name:
                    actions_list.append(current_action)
                else:
                    actions_list.append(action_batch[n].detach())
            all_actions_with_current = torch.cat(actions_list, dim=-1)

            agent.update_actor(all_obs, all_actions_with_current)

            # Soft update targets
            agent.soft_update()

    def train(self, seed: int | None = None, log_suffix: str = "",
              quiet: bool = False):
        """Run the full two-phase training pipeline.

        Args:
            seed: Random seed for reproducibility.
            log_suffix: Suffix for log directory name.
            quiet: If True, suppress tqdm progress bar (for parallel runs).

        Returns:
            Dictionary with training history.
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            self.env.seed(seed)

        log_dir = os.path.join(
            self.config["experiment"]["log_dir"],
            f"seed_{seed}{log_suffix}",
        )
        writer = SummaryWriter(log_dir)

        checkpoint_dir = os.path.join(
            self.config["experiment"]["checkpoint_dir"],
            f"seed_{seed}{log_suffix}",
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        history = {
            "listener_accuracy": [],
            "adversary_accuracy": [],
            "speaker_reward": [],
            "phase": [],
        }

        total_ep = self.phase1_episodes + self.phase2_episodes
        update_every = self.config["training"]["update_every"]
        eval_every = self.config["training"]["eval_every"]
        checkpoint_every = self.config["training"]["checkpoint_every"]

        # Reset noise to initial value
        self.noise_scale = self.noise_initial

        # Notify buffer of initial phase
        initial_phase = 1 if self.phase1_episodes > 0 else 2
        self.buffer.notify_phase(initial_phase)

        pbar = tqdm(range(total_ep), desc="Training", disable=quiet)
        for ep in pbar:
            phase = 1 if ep < self.phase1_episodes else 2

            # ---- Phase transition handling ----
            if ep == self.phase1_episodes and self.phase1_episodes > 0:
                # Notify buffer of phase change (may flush)
                self.buffer.notify_phase(2)

                # Reset noise so Phase 2 starts with full exploration
                if self.reset_noise_on_phase2:
                    self.noise_scale = self.noise_initial

            # Set training mode
            for agent in self.agents.values():
                agent.train_mode()

            # Collect episode
            info = self._collect_episode(phase)

            # Update agents
            if ep % update_every == 0:
                self._update_agents(phase)

            # Decay noise
            self.noise_scale = max(
                self.min_noise, self.noise_scale * self.noise_decay
            )

            # Logging
            history["listener_accuracy"].append(info["listener_correct"])
            history["adversary_accuracy"].append(info["adversary_accuracy"])
            history["speaker_reward"].append(info["speaker_reward"])
            history["phase"].append(phase)

            writer.add_scalar("train/listener_correct", info["listener_correct"], ep)
            writer.add_scalar("train/adversary_accuracy", info["adversary_accuracy"], ep)
            writer.add_scalar("train/speaker_reward", info["speaker_reward"], ep)
            writer.add_scalar("train/phase", phase, ep)
            writer.add_scalar("train/noise_scale", self.noise_scale, ep)

            # Evaluation
            if ep % eval_every == 0:
                eval_info = self._evaluate(phase)
                writer.add_scalar("eval/listener_accuracy", eval_info["listener_accuracy"], ep)
                writer.add_scalar("eval/adversary_accuracy", eval_info["adversary_accuracy"], ep)

                phase_str = "coop" if phase == 1 else "adv"
                pbar.set_postfix(
                    phase=phase_str,
                    listener=f"{eval_info['listener_accuracy']:.3f}",
                    adversary=f"{eval_info['adversary_accuracy']:.3f}",
                )

            # Periodic CIC evaluation
            if self.cic_eval_every > 0 and ep % self.cic_eval_every == 0 and ep > 0:
                cic_val = _quick_cic(
                    self.env, self.agents, self.config,
                    num_episodes=100,
                    num_counterfactuals=5,
                )
                writer.add_scalar("eval/cic", cic_val, ep)
                history.setdefault("cic", []).append(
                    {"episode": ep, "cic": cic_val}
                )

            # Checkpoint
            if ep % checkpoint_every == 0 and ep > 0:
                self._save_checkpoint(checkpoint_dir, ep)

        # Final CIC measurement
        if self.cic_eval_every > 0:
            cic_val = _quick_cic(
                self.env, self.agents, self.config,
                num_episodes=100,
                num_counterfactuals=5,
            )
            writer.add_scalar("eval/cic", cic_val, total_ep)
            history.setdefault("cic", []).append(
                {"episode": total_ep, "cic": cic_val}
            )

        # Final checkpoint
        self._save_checkpoint(checkpoint_dir, total_ep)
        writer.close()

        return history

    def _evaluate(self, phase: int) -> dict:
        """Run evaluation episodes without exploration.

        Args:
            phase: Current training phase.

        Returns:
            Averaged evaluation metrics.
        """
        eval_episodes = self.config["training"]["eval_episodes"]
        self.env.set_phase(phase)

        listener_correct_total = 0
        adversary_accuracy_total = 0.0

        for agent in self.agents.values():
            agent.eval_mode()

        for _ in range(eval_episodes):
            obs = self.env.reset()
            speaker_obs = obs["speaker"]

            # Speaker
            speaker_action = self.agents["speaker"].select_action(
                speaker_obs, explore=False
            )
            msg_discrete = self._onehot_to_discrete(
                speaker_action,
                self.config["env"]["vocab_size"],
                self.config["env"]["msg_length"],
            )
            post_obs = self.env.step_speaker(msg_discrete)

            # Listener
            listener_obs = post_obs["listener"]
            listener_action = self.agents["listener"].select_action(
                listener_obs, explore=False
            )
            listener_selection = np.argmax(listener_action)
            self.env.step_listener(listener_selection)

            listener_correct_total += self.env._listener_correct

            # Adversary
            if phase >= 2 and "adversary" in post_obs:
                adversary_obs = post_obs["adversary"]
                adversary_action = self.agents["adversary"].select_action(
                    adversary_obs, explore=False
                )
                adversary_recon = self._onehot_to_discrete(
                    adversary_action,
                    self.config["env"]["num_values"],
                    self.config["env"]["num_attributes"],
                )
                rewards = self.env.step_adversary(adversary_recon)
                adversary_accuracy_total += rewards["adversary_accuracy"]

        return {
            "listener_accuracy": listener_correct_total / eval_episodes,
            "adversary_accuracy": adversary_accuracy_total / eval_episodes,
        }

    def _save_checkpoint(self, checkpoint_dir: str, episode: int):
        """Save all agent checkpoints."""
        for name, agent in self.agents.items():
            path = os.path.join(checkpoint_dir, f"{name}_ep{episode}.pt")
            agent.save(path)

    @staticmethod
    def _onehot_to_discrete(
        onehot: np.ndarray, num_classes: int, num_slots: int
    ) -> np.ndarray:
        """Convert a flat one-hot vector to discrete indices.

        Args:
            onehot: Flat one-hot array of shape (num_slots * num_classes,).
            num_classes: Number of classes per slot.
            num_slots: Number of slots.

        Returns:
            Array of shape (num_slots,) with integer indices.
        """
        reshaped = onehot.reshape(num_slots, num_classes)
        return np.argmax(reshaped, axis=-1)
