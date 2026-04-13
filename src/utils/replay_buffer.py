"""
Experience replay buffer for multi-agent training.
"""

import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """Fixed-size replay buffer storing transitions for all agents."""

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: dict):
        """Store a transition.

        Args:
            transition: Dictionary with keys for each agent's
                obs, action, reward, next_obs, and a shared 'done' flag.
                Expected structure:
                {
                    'obs': {agent_name: np.ndarray, ...},
                    'actions': {agent_name: np.ndarray, ...},
                    'rewards': {agent_name: float, ...},
                    'next_obs': {agent_name: np.ndarray, ...},
                    'done': bool,
                }
        """
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[dict]:
        """Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            List of transition dictionaries.
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self):
        """Clear the replay buffer."""
        self.buffer.clear()
