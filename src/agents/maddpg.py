"""
MADDPG (Multi-Agent Deep Deterministic Policy Gradient) agent.

Implements centralized training with decentralized execution for
multi-agent reinforcement learning with communication.

Reference:
    Lowe et al., "Multi-Agent Actor-Critic for Mixed
    Cooperative-Competitive Environments" (2017)
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .networks import Actor, Critic, CommunicationModule


class MADDPGAgent:
    """A single MADDPG agent with optional communication capability.

    Each agent has its own actor, critic, target networks, and optionally
    a communication module (for the speaker agent).
    """

    def __init__(
        self,
        agent_name: str,
        obs_dim: int,
        action_dim: int,
        total_obs_dim: int,
        total_action_dim: int,
        hidden_dim: int = 128,
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        gamma: float = 0.95,
        tau: float = 0.01,
        discrete: bool = True,
        # Communication params (only used if agent_name == "speaker")
        vocab_size: int = 10,
        msg_length: int = 3,
        comm_hidden_dim: int = 64,
        gumbel_temperature: float = 1.0,
        device: str = "cpu",
    ):
        self.agent_name = agent_name
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.discrete = discrete
        self.device = torch.device(device)

        # Communication module for speaker
        self.comm_module = None
        if agent_name == "speaker":
            self.comm_module = CommunicationModule(
                input_dim=obs_dim,
                vocab_size=vocab_size,
                msg_length=msg_length,
                hidden_dim=comm_hidden_dim,
                temperature=gumbel_temperature,
            ).to(self.device)
            # Speaker's actor takes obs and outputs message (handled by comm_module)
            # We still have an actor for consistency but the comm_module is primary
            actor_action_dim = vocab_size * msg_length
        else:
            actor_action_dim = action_dim

        # Actor and critic networks
        self.actor = Actor(
            obs_dim=obs_dim,
            action_dim=actor_action_dim,
            hidden_dim=hidden_dim,
            discrete=discrete,
        ).to(self.device)

        self.critic = Critic(
            total_obs_dim=total_obs_dim,
            total_action_dim=total_action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # Target networks
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        # Optimizers
        actor_params = list(self.actor.parameters())
        if self.comm_module is not None:
            actor_params += list(self.comm_module.parameters())

        self.actor_optimizer = torch.optim.Adam(actor_params, lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr_critic
        )

    def select_action(
        self, obs: np.ndarray, explore: bool = True, noise_scale: float = 0.1
    ) -> np.ndarray:
        """Select an action given an observation.

        For the speaker, this produces a discrete message via the comm module.
        For the listener/adversary, uses the actor network.

        Args:
            obs: Observation array.
            explore: Whether to add exploration noise.
            noise_scale: Scale of exploration noise.

        Returns:
            Action as numpy array (one-hot encoded for discrete actions).
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.comm_module is not None:
                # Speaker: use communication module
                message_onehot, _ = self.comm_module(obs_t, hard=True)
                action = message_onehot.squeeze(0).cpu().numpy()
            else:
                action_probs = self.actor(obs_t).squeeze(0).cpu().numpy()

                if self.discrete:
                    if explore:
                        # Sample from distribution with noise
                        noisy_probs = action_probs + noise_scale * np.random.randn(
                            len(action_probs)
                        )
                        noisy_probs = np.maximum(noisy_probs, 1e-8)
                        noisy_probs /= noisy_probs.sum()
                        idx = np.random.choice(len(noisy_probs), p=noisy_probs)
                    else:
                        idx = np.argmax(action_probs)
                    action = np.zeros(self.action_dim, dtype=np.float32)
                    action[idx] = 1.0
                else:
                    action = action_probs
                    if explore:
                        action += noise_scale * np.random.randn(*action.shape)
                        action = np.clip(action, -1, 1)

        return action

    def get_action_tensor(
        self, obs: torch.Tensor, gumbel_hard: bool = True
    ) -> torch.Tensor:
        """Get differentiable action tensor for training.

        Args:
            obs: Batched observation tensor.
            gumbel_hard: Whether to use straight-through estimator for Gumbel-Softmax.

        Returns:
            Action tensor.
        """
        if self.comm_module is not None:
            message_onehot, _ = self.comm_module(obs, hard=gumbel_hard)
            return message_onehot
        else:
            return self.actor(obs)

    def get_target_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get action from target actor (for critic target computation)."""
        if self.comm_module is not None:
            # Use comm module with hard discretization for target
            with torch.no_grad():
                message_onehot, _ = self.comm_module(obs, hard=True)
            return message_onehot
        return self.target_actor(obs)

    def update_critic(
        self,
        all_obs: torch.Tensor,
        all_actions: torch.Tensor,
        reward: torch.Tensor,
        all_next_obs: torch.Tensor,
        all_next_actions: torch.Tensor,
        done: torch.Tensor,
    ) -> float:
        """Update the critic network.

        Returns:
            Critic loss value.
        """
        with torch.no_grad():
            target_q = self.target_critic(all_next_obs, all_next_actions)
            y = reward + self.gamma * (1 - done) * target_q

        current_q = self.critic(all_obs, all_actions)
        critic_loss = F.mse_loss(current_q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        return critic_loss.item()

    def update_actor(
        self,
        all_obs: torch.Tensor,
        all_actions_with_current: torch.Tensor,
    ) -> float:
        """Update the actor network using the critic's policy gradient.

        Args:
            all_obs: Concatenated observations from all agents.
            all_actions_with_current: All actions, with this agent's action
                replaced by the current policy output (for gradient flow).

        Returns:
            Actor loss value.
        """
        actor_loss = -self.critic(all_obs, all_actions_with_current).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        if self.comm_module is not None:
            nn.utils.clip_grad_norm_(self.comm_module.parameters(), 0.5)
        self.actor_optimizer.step()

        return actor_loss.item()

    def soft_update(self):
        """Soft update target networks toward current networks."""
        for target_param, param in zip(
            self.target_actor.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def save(self, path: str):
        """Save agent state to disk."""
        state = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }
        if self.comm_module is not None:
            state["comm_module"] = self.comm_module.state_dict()
        torch.save(state, path)

    def load(self, path: str):
        """Load agent state from disk."""
        state = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_actor.load_state_dict(state["target_actor"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        if self.comm_module is not None and "comm_module" in state:
            self.comm_module.load_state_dict(state["comm_module"])

    def train_mode(self):
        """Set networks to training mode."""
        self.actor.train()
        self.critic.train()
        if self.comm_module is not None:
            self.comm_module.train()

    def eval_mode(self):
        """Set networks to evaluation mode."""
        self.actor.eval()
        self.critic.eval()
        if self.comm_module is not None:
            self.comm_module.eval()
