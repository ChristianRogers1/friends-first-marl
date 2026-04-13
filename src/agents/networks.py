"""
Neural network architectures for MADDPG agents with communication.

Actor: Maps observation -> action (decentralized execution).
Critic: Maps (all observations, all actions) -> Q-value (centralized training).
CommunicationModule: Discrete message production using Gumbel-Softmax.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CommunicationModule(nn.Module):
    """Produces discrete messages using Gumbel-Softmax relaxation.

    During training, uses Gumbel-Softmax for differentiable discrete sampling.
    During evaluation, uses argmax discretization.

    References:
        Tilbury et al., "Revisiting the Gumbel-Softmax in MADDPG" (2023)
    """

    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        msg_length: int,
        hidden_dim: int = 64,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.msg_length = msg_length
        self.temperature = temperature

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size * msg_length),
        )

    def forward(
        self, obs: torch.Tensor, hard: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce a message given speaker observation.

        Args:
            obs: Speaker observation tensor.
            hard: If True, use straight-through (argmax in forward, soft in backward).

        Returns:
            message_onehot: One-hot encoded message (batch, msg_length * vocab_size).
            message_probs: Softmax probabilities for each symbol position.
        """
        logits = self.net(obs)
        # Reshape to (batch, msg_length, vocab_size)
        logits = logits.view(-1, self.msg_length, self.vocab_size)

        if self.training:
            # Gumbel-Softmax for differentiable discrete sampling
            message_soft = F.gumbel_softmax(
                logits, tau=self.temperature, hard=hard, dim=-1
            )
        else:
            # Argmax discretization at evaluation
            indices = logits.argmax(dim=-1)
            message_soft = F.one_hot(indices, self.vocab_size).float()

        probs = F.softmax(logits, dim=-1)
        # Flatten for downstream: (batch, msg_length * vocab_size)
        message_flat = message_soft.view(-1, self.msg_length * self.vocab_size)

        return message_flat, probs

    def get_discrete_message(self, obs: torch.Tensor) -> torch.Tensor:
        """Get the discrete (argmax) message indices.

        Returns:
            Tensor of shape (batch, msg_length) with integer symbol indices.
        """
        logits = self.net(obs)
        logits = logits.view(-1, self.msg_length, self.vocab_size)
        return logits.argmax(dim=-1)


class Actor(nn.Module):
    """Actor network for MADDPG.

    Maps agent observation to action logits / continuous action.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        discrete: bool = True,
    ):
        """
        Args:
            obs_dim: Observation dimension.
            action_dim: Action dimension (number of discrete choices, or
                        total one-hot dimension for MultiDiscrete).
            hidden_dim: Hidden layer dimension.
            discrete: Whether actions are discrete (softmax output).
        """
        super().__init__()
        self.discrete = discrete

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute action logits or values from observation."""
        out = self.net(obs)
        if self.discrete:
            return F.softmax(out, dim=-1)
        return torch.tanh(out)


class Critic(nn.Module):
    """Centralized critic for MADDPG.

    Takes the full state (all observations) and all actions as input,
    outputs a Q-value estimate.
    """

    def __init__(
        self,
        total_obs_dim: int,
        total_action_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim + total_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, all_obs: torch.Tensor, all_actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute Q-value.

        Args:
            all_obs: Concatenated observations from all agents.
            all_actions: Concatenated actions from all agents.

        Returns:
            Q-value estimate of shape (batch, 1).
        """
        x = torch.cat([all_obs, all_actions], dim=-1)
        return self.net(x)
