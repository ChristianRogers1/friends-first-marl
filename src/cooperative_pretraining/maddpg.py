from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import erf, sqrt
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from cooperative_pretraining.envs.public_eavesdrop_env import (
    ADVERSARY,
    LISTENER,
    SPEAKER,
    PublicEavesdropParallelEnv,
)


AGENT_ORDER = [SPEAKER, LISTENER, ADVERSARY]


@dataclass(frozen=True)
class MADDPGConfig:
    train_episodes: int = 2500
    eval_episodes: int = 200
    gamma: float = 0.95
    tau: float = 0.01
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    batch_size: int = 128
    replay_size: int = 50_000
    warmup_steps: int = 500
    updates_per_step: int = 1
    hidden_dim: int = 128
    gumbel_tau: float = 1.0
    policy_noise: float = 0.05
    device: str = "cpu"


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def mean_confidence_interval(values: np.ndarray, z: float = 1.96) -> tuple[float, float]:
    if values.size == 1:
        value = float(values[0])
        return value, value
    mean = float(values.mean())
    stderr = float(values.std(ddof=1) / np.sqrt(values.size))
    return mean - z * stderr, mean + z * stderr


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = ((x.size - 1) * vx + (y.size - 1) * vy) / max(x.size + y.size - 2, 1)
    if pooled <= 0:
        return 0.0
    return float((x.mean() - y.mean()) / np.sqrt(pooled))


def two_sided_pvalue(diff: np.ndarray) -> float:
    if diff.size < 2:
        return 1.0
    mean = float(diff.mean())
    std = float(diff.std(ddof=1))
    if std == 0:
        return 0.0 if mean != 0 else 1.0
    z = abs(mean / (std / np.sqrt(diff.size)))
    return float(2 * (1 - _normal_cdf(z)))


def flatten_space_value(space: spaces.Space, value: Any) -> np.ndarray:
    if isinstance(space, spaces.Dict):
        parts = []
        for key in sorted(space.spaces.keys()):
            parts.append(flatten_space_value(space.spaces[key], value[key]))
        return np.concatenate(parts).astype(np.float32)
    if isinstance(space, spaces.Box):
        return np.asarray(value, dtype=np.float32).reshape(-1)
    if isinstance(space, spaces.Discrete):
        out = np.zeros(space.n, dtype=np.float32)
        out[int(value)] = 1.0
        return out
    if isinstance(space, spaces.MultiDiscrete):
        value = np.asarray(value, dtype=np.int64).reshape(-1)
        parts = []
        for idx, n in enumerate(space.nvec):
            one_hot = np.zeros(int(n), dtype=np.float32)
            one_hot[int(value[idx])] = 1.0
            parts.append(one_hot)
        return np.concatenate(parts)
    raise TypeError(f"Unsupported space type: {type(space)}")


def action_to_onehot(space: spaces.Space, action: Any) -> np.ndarray:
    return flatten_space_value(space, action)


def sample_random_action(space: spaces.Space, rng: np.random.Generator):
    if isinstance(space, spaces.Discrete):
        return int(rng.integers(0, space.n))
    if isinstance(space, spaces.MultiDiscrete):
        return np.asarray([rng.integers(0, int(n)) for n in space.nvec], dtype=np.int64)
    raise TypeError(f"Unsupported action space: {type(space)}")


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition: dict[str, Any]) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[dict[str, Any]]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class MultiHeadActor(nn.Module):
    def __init__(self, obs_dim: int, action_dims: list[int], hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, dim) for dim in action_dims])
        self.action_dims = action_dims

    def forward(self, obs: torch.Tensor) -> list[torch.Tensor]:
        hidden = self.net(obs)
        return [head(hidden) for head in self.heads]


class Critic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AgentSpec:
    def __init__(self, name: str, obs_space: spaces.Space, action_space: spaces.Space, hidden_dim: int, device: torch.device):
        self.name = name
        self.obs_space = obs_space
        self.action_space = action_space
        self.obs_dim = flatten_space_value(obs_space, obs_space.sample()).shape[0]
        if isinstance(action_space, spaces.Discrete):
            self.action_dims = [action_space.n]
        elif isinstance(action_space, spaces.MultiDiscrete):
            self.action_dims = [int(n) for n in action_space.nvec]
        else:
            raise TypeError(f"Unsupported action space: {type(action_space)}")
        self.action_dim = int(sum(self.action_dims))

        self.actor = MultiHeadActor(self.obs_dim, self.action_dims, hidden_dim).to(device)
        self.target_actor = MultiHeadActor(self.obs_dim, self.action_dims, hidden_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.critic: Critic | None = None
        self.target_critic: Critic | None = None
        self.actor_optimizer: torch.optim.Optimizer | None = None
        self.critic_optimizer: torch.optim.Optimizer | None = None


class MADDPG:
    def __init__(self, env: PublicEavesdropParallelEnv, config: MADDPGConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.agent_specs = {
            agent: AgentSpec(
                agent,
                env.observation_space(agent),
                env.action_space(agent),
                hidden_dim=config.hidden_dim,
                device=self.device,
            )
            for agent in AGENT_ORDER
        }
        self.global_obs_dim = sum(self.agent_specs[agent].obs_dim for agent in AGENT_ORDER)
        self.global_action_dim = sum(self.agent_specs[agent].action_dim for agent in AGENT_ORDER)
        critic_input_dim = self.global_obs_dim + self.global_action_dim

        for agent in AGENT_ORDER:
            spec = self.agent_specs[agent]
            spec.critic = Critic(critic_input_dim, config.hidden_dim).to(self.device)
            spec.target_critic = Critic(critic_input_dim, config.hidden_dim).to(self.device)
            spec.target_critic.load_state_dict(spec.critic.state_dict())
            spec.actor_optimizer = torch.optim.Adam(spec.actor.parameters(), lr=config.actor_lr)
            spec.critic_optimizer = torch.optim.Adam(spec.critic.parameters(), lr=config.critic_lr)

        self.replay = ReplayBuffer(config.replay_size)
        self.total_steps = 0
        self.training_log: list[dict[str, Any]] = []

    def select_actions(self, observations: dict[str, Any], explore: bool, rng: np.random.Generator) -> dict[str, Any]:
        actions = {}
        for agent in AGENT_ORDER:
            spec = self.agent_specs[agent]
            if agent not in observations:
                if isinstance(spec.action_space, spaces.Discrete):
                    actions[agent] = 0
                else:
                    actions[agent] = np.zeros(len(spec.action_dims), dtype=np.int64)
                continue
            obs_vec = flatten_space_value(spec.obs_space, observations[agent])
            obs_tensor = torch.tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = spec.actor(obs_tensor)
            action = self._sample_action_from_logits(spec, logits, explore=explore, rng=rng)
            actions[agent] = action
        return actions

    def _sample_action_from_logits(
        self,
        spec: AgentSpec,
        logits_list: list[torch.Tensor],
        explore: bool,
        rng: np.random.Generator,
    ):
        discrete_values = []
        for logits in logits_list:
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
            if explore:
                probs = 0.95 * probs + 0.05 / probs.shape[0]
                idx = int(rng.choice(probs.shape[0], p=probs))
            else:
                idx = int(np.argmax(probs))
            discrete_values.append(idx)
        if isinstance(spec.action_space, spaces.Discrete):
            return discrete_values[0]
        return np.asarray(discrete_values, dtype=np.int64)

    def store_transition(
        self,
        observations: dict[str, Any],
        actions: dict[str, Any],
        rewards: dict[str, float],
        next_observations: dict[str, Any],
        terminations: dict[str, bool],
    ) -> None:
        transition = {
            "obs": {
                agent: flatten_space_value(self.agent_specs[agent].obs_space, observations[agent])
                if agent in observations
                else np.zeros(self.agent_specs[agent].obs_dim, dtype=np.float32)
                for agent in AGENT_ORDER
            },
            "actions": {
                agent: action_to_onehot(self.agent_specs[agent].action_space, actions[agent])
                if agent in actions
                else np.zeros(self.agent_specs[agent].action_dim, dtype=np.float32)
                for agent in AGENT_ORDER
            },
            "rewards": {agent: float(rewards.get(agent, 0.0)) for agent in AGENT_ORDER},
            "next_obs": {
                agent: flatten_space_value(self.agent_specs[agent].obs_space, next_observations[agent])
                if agent in next_observations
                else np.zeros(self.agent_specs[agent].obs_dim, dtype=np.float32)
                for agent in AGENT_ORDER
            },
            "dones": {agent: float(terminations.get(agent, True)) for agent in AGENT_ORDER},
        }
        self.replay.add(transition)
        self.total_steps += 1

    def maybe_update(self) -> None:
        if len(self.replay) < max(self.config.batch_size, self.config.warmup_steps):
            return
        for _ in range(self.config.updates_per_step):
            batch = self.replay.sample(self.config.batch_size)
            self._update_from_batch(batch)

    def _update_from_batch(self, batch: list[dict[str, Any]]) -> None:
        obs = {
            agent: torch.tensor(np.stack([item["obs"][agent] for item in batch]), dtype=torch.float32, device=self.device)
            for agent in AGENT_ORDER
        }
        actions = {
            agent: torch.tensor(np.stack([item["actions"][agent] for item in batch]), dtype=torch.float32, device=self.device)
            for agent in AGENT_ORDER
        }
        rewards = {
            agent: torch.tensor([item["rewards"][agent] for item in batch], dtype=torch.float32, device=self.device).unsqueeze(-1)
            for agent in AGENT_ORDER
        }
        next_obs = {
            agent: torch.tensor(np.stack([item["next_obs"][agent] for item in batch]), dtype=torch.float32, device=self.device)
            for agent in AGENT_ORDER
        }
        dones = {
            agent: torch.tensor([item["dones"][agent] for item in batch], dtype=torch.float32, device=self.device).unsqueeze(-1)
            for agent in AGENT_ORDER
        }

        obs_cat = torch.cat([obs[agent] for agent in AGENT_ORDER], dim=-1)
        actions_cat = torch.cat([actions[agent] for agent in AGENT_ORDER], dim=-1)
        critic_input = torch.cat([obs_cat, actions_cat], dim=-1)

        next_obs_cat = torch.cat([next_obs[agent] for agent in AGENT_ORDER], dim=-1)
        next_actions = []
        for agent in AGENT_ORDER:
            spec = self.agent_specs[agent]
            next_actions.append(self._actor_output_onehot(spec.target_actor(next_obs[agent]), tau=self.config.gumbel_tau, hard=True))
        next_actions_cat = torch.cat(next_actions, dim=-1)
        target_critic_input = torch.cat([next_obs_cat, next_actions_cat], dim=-1)

        for agent in AGENT_ORDER:
            spec = self.agent_specs[agent]
            assert spec.critic is not None and spec.target_critic is not None
            assert spec.actor_optimizer is not None and spec.critic_optimizer is not None

            with torch.no_grad():
                target_q = rewards[agent] + self.config.gamma * (1.0 - dones[agent]) * spec.target_critic(target_critic_input)

            critic_q = spec.critic(critic_input)
            critic_loss = F.mse_loss(critic_q, target_q)
            spec.critic_optimizer.zero_grad()
            critic_loss.backward()
            spec.critic_optimizer.step()

            current_actions = []
            for other in AGENT_ORDER:
                other_spec = self.agent_specs[other]
                if other == agent:
                    current_actions.append(
                        self._actor_output_onehot(other_spec.actor(obs[other]), tau=self.config.gumbel_tau, hard=True)
                    )
                else:
                    current_actions.append(actions[other].detach())
            actor_action_cat = torch.cat(current_actions, dim=-1)
            actor_input = torch.cat([obs_cat, actor_action_cat], dim=-1)
            actor_loss = -spec.critic(actor_input).mean()

            spec.actor_optimizer.zero_grad()
            actor_loss.backward()
            spec.actor_optimizer.step()

            self._soft_update(spec.actor, spec.target_actor)
            self._soft_update(spec.critic, spec.target_critic)

    def _actor_output_onehot(self, logits_list: list[torch.Tensor], tau: float, hard: bool) -> torch.Tensor:
        parts = []
        for logits in logits_list:
            parts.append(F.gumbel_softmax(logits, tau=tau, hard=hard))
        return torch.cat(parts, dim=-1)

    def _soft_update(self, src: nn.Module, dst: nn.Module) -> None:
        for dst_param, src_param in zip(dst.parameters(), src.parameters()):
            dst_param.data.copy_(self.config.tau * src_param.data + (1.0 - self.config.tau) * dst_param.data)


def run_training_regime(
    env_factory,
    pretraining_ratio: float,
    seed: int,
    config: MADDPGConfig,
) -> MADDPG:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = env_factory()
    maddpg = MADDPG(env, config)
    rng = np.random.default_rng(seed)

    pretraining_episodes = int(config.train_episodes * pretraining_ratio)
    public_episodes = config.train_episodes - pretraining_episodes
    phases = ["cooperative_pretraining"] * pretraining_episodes + ["public_adversarial"] * public_episodes

    for episode_idx, phase in enumerate(phases):
        observations, _ = env.reset(seed=int(rng.integers(0, 1_000_000)), options={"phase": phase})
        done = False
        episode_rewards = {agent: 0.0 for agent in AGENT_ORDER}
        last_infos: dict[str, Any] = {}
        while not done:
            actions = maddpg.select_actions(observations, explore=True, rng=rng)
            if int(observations[SPEAKER]["step_id"]) == 0:
                actions[LISTENER] = 0
                if ADVERSARY in env.agents:
                    actions[ADVERSARY] = np.zeros(env.action_space(ADVERSARY).nvec.shape[0], dtype=np.int64)
            else:
                actions[SPEAKER] = np.zeros(env.action_space(SPEAKER).nvec.shape[0], dtype=np.int64)
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            maddpg.store_transition(observations, actions, rewards, next_observations, terminations)
            maddpg.maybe_update()
            for agent, reward in rewards.items():
                episode_rewards[agent] += float(reward)
            observations = next_observations
            done = all(terminations.values()) or all(truncations.values())
            last_infos = infos
        transition_episode = pretraining_episodes
        if abs(episode_idx - transition_episode) <= 10 or episode_idx % 250 == 0:
            maddpg.training_log.append(
                {
                    "episode": episode_idx,
                    "phase": phase,
                    "speaker_reward": episode_rewards[SPEAKER],
                    "listener_reward": episode_rewards[LISTENER],
                    "adversary_reward": episode_rewards[ADVERSARY],
                    "listener_correct": bool(last_infos.get(LISTENER, {}).get("listener_correct", False)),
                    "adversary_accuracy": float(
                        last_infos.get(ADVERSARY, {}).get("adversary_attribute_accuracy", 0.0)
                    ),
                }
            )
    env.close()
    return maddpg


def evaluate_maddpg(
    env_factory,
    trainer: MADDPG,
    episodes: int,
    seed: int,
) -> dict[str, np.ndarray]:
    env = env_factory()
    rng = np.random.default_rng(seed)
    listener_accuracy = []
    adversary_accuracy = []
    cic = []

    for _ in range(episodes):
        observations, _ = env.reset(seed=int(rng.integers(0, 1_000_000)), options={"phase": "public_adversarial"})
        actions = trainer.select_actions(observations, explore=False, rng=rng)
        actions[LISTENER] = 0
        actions[ADVERSARY] = np.zeros(env.action_space(ADVERSARY).nvec.shape[0], dtype=np.int64)
        observations, _, _, _, _ = env.step(actions)
        message = np.asarray(observations[LISTENER]["public_message"], dtype=np.int64)
        final_actions = trainer.select_actions(observations, explore=False, rng=rng)
        final_actions[SPEAKER] = np.zeros(env.action_space(SPEAKER).nvec.shape[0], dtype=np.int64)
        _, _, _, _, infos = env.step(final_actions)

        listener = float(infos[LISTENER]["listener_correct"])
        adv = float(infos[ADVERSARY]["adversary_attribute_accuracy"])
        listener_accuracy.append(listener)
        adversary_accuracy.append(adv)
        cic.append(estimate_cic_from_trainer(trainer, observations, message))
    env.close()
    return {
        "listener_accuracy": np.asarray(listener_accuracy, dtype=np.float64),
        "adversary_accuracy": np.asarray(adversary_accuracy, dtype=np.float64),
        "cic": np.asarray(cic, dtype=np.float64),
    }


def estimate_cic_from_trainer(trainer: MADDPG, observations: dict[str, Any], message: np.ndarray) -> float:
    listener_spec = trainer.agent_specs[LISTENER]
    base_obs = {
        key: (value.copy() if isinstance(value, np.ndarray) else value)
        for key, value in observations[LISTENER].items()
    }
    obs_vec = flatten_space_value(listener_spec.obs_space, base_obs)
    obs_tensor = torch.tensor(obs_vec, dtype=torch.float32, device=trainer.device).unsqueeze(0)
    base_logits = listener_spec.actor(obs_tensor)[0]
    base_probs = torch.softmax(base_logits, dim=-1).detach().cpu().numpy()[0]

    vocab_size = trainer.agent_specs[SPEAKER].action_dims[0]
    total = 0.0
    count = 0
    for token0 in range(vocab_size):
        for token1 in range(vocab_size):
            for token2 in range(vocab_size):
                cf_message = np.asarray([token0, token1, token2], dtype=np.int64)
                if np.array_equal(cf_message, message):
                    continue
                cf_obs = {
                    key: (value.copy() if isinstance(value, np.ndarray) else value)
                    for key, value in observations[LISTENER].items()
                }
                cf_obs["public_message"] = cf_message
                cf_vec = flatten_space_value(listener_spec.obs_space, cf_obs)
                cf_tensor = torch.tensor(cf_vec, dtype=torch.float32, device=trainer.device).unsqueeze(0)
                cf_logits = listener_spec.actor(cf_tensor)[0]
                cf_probs = torch.softmax(cf_logits, dim=-1).detach().cpu().numpy()[0]
                total += 0.5 * float(np.abs(base_probs - cf_probs).sum())
                count += 1
    return total / max(count, 1)


def summarize_group(runs: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    summary = {}
    for key in runs[0]:
        per_seed = np.asarray([run[key].mean() for run in runs], dtype=np.float64)
        ci_low, ci_high = mean_confidence_interval(per_seed)
        summary[key] = {
            "mean": float(per_seed.mean()),
            "std": float(per_seed.std(ddof=0)),
            "ci95": [float(ci_low), float(ci_high)],
            "per_seed": per_seed.tolist(),
        }
    leakage_per_seed = np.asarray(
        [
            run["adversary_accuracy"].mean() / max(run["listener_accuracy"].mean(), 1e-6)
            for run in runs
        ],
        dtype=np.float64,
    )
    ci_low, ci_high = mean_confidence_interval(leakage_per_seed)
    summary["leakage_ratio"] = {
        "mean": float(leakage_per_seed.mean()),
        "std": float(leakage_per_seed.std(ddof=0)),
        "ci95": [float(ci_low), float(ci_high)],
        "per_seed": leakage_per_seed.tolist(),
    }
    return summary


def compare_groups(baseline_runs: list[dict[str, np.ndarray]], pretrained_runs: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    out = {}
    for key in ("listener_accuracy", "cic", "adversary_accuracy", "leakage_ratio"):
        if key == "leakage_ratio":
            baseline = np.asarray(
                [
                    run["adversary_accuracy"].mean() / max(run["listener_accuracy"].mean(), 1e-6)
                    for run in baseline_runs
                ],
                dtype=np.float64,
            )
            pretrained = np.asarray(
                [
                    run["adversary_accuracy"].mean() / max(run["listener_accuracy"].mean(), 1e-6)
                    for run in pretrained_runs
                ],
                dtype=np.float64,
            )
        else:
            baseline = np.asarray([run[key].mean() for run in baseline_runs], dtype=np.float64)
            pretrained = np.asarray([run[key].mean() for run in pretrained_runs], dtype=np.float64)
        diff = pretrained - baseline
        ci_low, ci_high = mean_confidence_interval(diff)
        out[key] = {
            "baseline_mean": float(baseline.mean()),
            "pretrained_mean": float(pretrained.mean()),
            "difference_mean": float(diff.mean()),
            "difference_ci95": [float(ci_low), float(ci_high)],
            "cohens_d": float(cohens_d(pretrained, baseline)),
            "p_value_normal_approx": float(two_sided_pvalue(diff)),
        }
    return out


def run_screen_and_confirm_maddpg(
    env_factory,
    screening_seeds: int = 5,
    confirmation_seeds: int = 12,
    ratios: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75),
    base_seed: int = 7,
    config: MADDPGConfig | None = None,
) -> dict[str, Any]:
    config = MADDPGConfig() if config is None else config
    screening = {}

    for ratio in ratios:
        runs = []
        logs = []
        for offset in range(screening_seeds):
            seed = base_seed + offset
            trainer = run_training_regime(env_factory, pretraining_ratio=ratio, seed=seed, config=config)
            runs.append(evaluate_maddpg(env_factory, trainer, episodes=config.eval_episodes, seed=seed + 10_000))
            logs.append({"seed": seed, "training_log": trainer.training_log})
        screening[ratio] = summarize_group(runs)
        screening[ratio]["transition_logs"] = logs

    best_ratio = max(ratios, key=lambda ratio: screening[ratio]["listener_accuracy"]["mean"])

    baseline_runs = []
    pretrained_runs = []
    baseline_logs = []
    pretrained_logs = []
    for offset in range(confirmation_seeds):
        seed = base_seed + 100 + offset
        baseline_trainer = run_training_regime(env_factory, pretraining_ratio=0.0, seed=seed, config=config)
        baseline_runs.append(evaluate_maddpg(env_factory, baseline_trainer, episodes=config.eval_episodes, seed=seed + 20_000))
        baseline_logs.append({"seed": seed, "training_log": baseline_trainer.training_log})
        pretrained_trainer = run_training_regime(env_factory, pretraining_ratio=best_ratio, seed=seed, config=config)
        pretrained_runs.append(evaluate_maddpg(env_factory, pretrained_trainer, episodes=config.eval_episodes, seed=seed + 30_000))
        pretrained_logs.append({"seed": seed, "training_log": pretrained_trainer.training_log})

    return {
        "screening": screening,
        "selected_ratio": best_ratio,
        "confirmation": {
            "baseline": summarize_group(baseline_runs),
            "pretrained": summarize_group(pretrained_runs),
            "comparisons": compare_groups(baseline_runs, pretrained_runs),
            "transition_logs": {
                "baseline": baseline_logs,
                "pretrained": pretrained_logs,
            },
        },
    }
