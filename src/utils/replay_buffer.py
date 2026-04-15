"""
Experience replay buffer for multi-agent training.

Supports phase-aware transition storage: each transition can be tagged
with a phase label, and sampling can optionally be restricted to the
current phase.
"""

import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """Fixed-size replay buffer storing transitions for all agents.

    Supports three phase-transition strategies (set via ``phase_strategy``):
      - ``"none"``:  Buffer is shared across phases with no distinction.
      - ``"flush"``: Buffer is cleared at every phase transition.
      - ``"label"``: Transitions are tagged with their phase; sampling
        can be restricted to the current phase only.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        phase_strategy: str = "none",
    ):
        """
        Args:
            capacity: Maximum number of stored transitions.
            phase_strategy: One of ``"none"``, ``"flush"``, ``"label"``.
        """
        if phase_strategy not in ("none", "flush", "label"):
            raise ValueError(
                f"phase_strategy must be 'none', 'flush', or 'label', "
                f"got '{phase_strategy}'"
            )
        self.capacity = capacity
        self.phase_strategy = phase_strategy
        self.buffer = deque(maxlen=capacity)
        self._current_phase: int | None = None

    def notify_phase(self, phase: int):
        """Inform the buffer that training has entered a new phase.

        For ``"flush"`` strategy this clears the buffer.
        For ``"label"`` strategy this updates the phase tag for future
        transitions.

        Args:
            phase: The new training phase (e.g. 1 or 2).
        """
        prev = self._current_phase
        self._current_phase = phase
        if self.phase_strategy == "flush" and prev is not None and phase != prev:
            self.clear()

    def push(self, transition: dict):
        """Store a transition.

        Args:
            transition: Dictionary with keys for each agent's
                obs, action, reward, next_obs, and a shared 'done' flag.
                Expected structure::

                    {
                        'obs': {agent_name: np.ndarray, ...},
                        'actions': {agent_name: np.ndarray, ...},
                        'rewards': {agent_name: float, ...},
                        'next_obs': {agent_name: np.ndarray, ...},
                        'done': bool,
                    }

                When using the ``"label"`` strategy, a ``'phase'`` key is
                added automatically.
        """
        if self.phase_strategy == "label" and self._current_phase is not None:
            transition = {**transition, "phase": self._current_phase}
        self.buffer.append(transition)

    def sample(
        self, batch_size: int, phase: int | None = None
    ) -> list[dict]:
        """Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample.
            phase: If given and strategy is ``"label"``, only sample
                transitions from this phase.  Ignored otherwise.

        Returns:
            List of transition dictionaries.
        """
        if phase is not None and self.phase_strategy == "label":
            eligible = [t for t in self.buffer if t.get("phase") == phase]
        else:
            eligible = list(self.buffer)

        return random.sample(eligible, min(batch_size, len(eligible)))

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self):
        """Clear the replay buffer."""
        self.buffer.clear()
