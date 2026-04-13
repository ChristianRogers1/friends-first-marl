"""
Custom adversarial crypto environment for studying emergent communication
under adversarial pressure.

Modified from PettingZoo's simple_crypto to support a team-based
public-eavesdropping setting:
  - Cooperative Team: Speaker observes target state, sends discrete message
    to Listener who must identify the correct target from distractors.
  - Adversary: Eavesdropper observes the cooperative team's messages on a
    public channel and attempts to reconstruct the target state.
  - Messages are discrete sequences of `msg_length` symbols from a vocabulary
    of `vocab_size`. All messages are publicly observable.

Supports two training phases:
  Phase 1 (cooperative): No adversary, messages are private.
  Phase 2 (adversarial): Eavesdropper introduced, public channel.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class AdversarialCryptoEnv:
    """Team-based referential communication game with public eavesdropping.

    Agents:
        speaker  - observes the target, produces a discrete message
        listener - observes distractors + message, selects target
        adversary - observes message, tries to reconstruct target

    The environment cycles: speaker -> listener -> adversary each step.
    """

    AGENT_NAMES = ["speaker", "listener", "adversary"]

    def __init__(
        self,
        num_attributes: int = 3,
        num_values: int = 5,
        num_distractors: int = 2,
        vocab_size: int = 10,
        msg_length: int = 3,
        adversary_active: bool = True,
        cooperative_reward_weight: float = 1.0,
        leakage_penalty_weight: float = 0.5,
    ):
        """
        Args:
            num_attributes: Number of attributes composing the target state.
            num_values: Number of possible values per attribute.
            num_distractors: Number of distractor targets presented to listener.
            vocab_size: Size of the discrete communication vocabulary.
            msg_length: Number of symbols in each message.
            adversary_active: Whether the adversary is active (Phase 2).
            cooperative_reward_weight: Weight for listener accuracy reward.
            leakage_penalty_weight: Weight for information leakage penalty.
        """
        self.num_attributes = num_attributes
        self.num_values = num_values
        self.num_distractors = num_distractors
        self.vocab_size = vocab_size
        self.msg_length = msg_length
        self.adversary_active = adversary_active
        self.cooperative_reward_weight = cooperative_reward_weight
        self.leakage_penalty_weight = leakage_penalty_weight

        self.num_candidates = 1 + num_distractors  # target + distractors

        # Define observation and action spaces for each agent
        # Speaker observes the target state (one-hot encoded attributes)
        speaker_obs_dim = num_attributes * num_values
        self.observation_spaces = {
            "speaker": spaces.Box(
                low=0, high=1, shape=(speaker_obs_dim,), dtype=np.float32
            ),
            # Listener sees all candidates (one-hot) + the message
            "listener": spaces.Box(
                low=0,
                high=1,
                shape=(
                    self.num_candidates * num_attributes * num_values
                    + msg_length * vocab_size,
                ),
                dtype=np.float32,
            ),
            # Adversary sees only the message
            "adversary": spaces.Box(
                low=0,
                high=1,
                shape=(msg_length * vocab_size,),
                dtype=np.float32,
            ),
        }

        self.action_spaces = {
            # Speaker produces a message: msg_length discrete symbols
            "speaker": spaces.MultiDiscrete([vocab_size] * msg_length),
            # Listener selects one of the candidates
            "listener": spaces.Discrete(self.num_candidates),
            # Adversary tries to reconstruct target (one value per attribute)
            "adversary": spaces.MultiDiscrete([num_values] * num_attributes),
        }

        # Internal state
        self._target = None
        self._distractors = None
        self._candidates = None  # target + distractors in shuffled order
        self._target_idx = None  # index of the target in candidates
        self._message = None
        self._rng = np.random.default_rng()

    def seed(self, seed: int):
        """Set the random seed."""
        self._rng = np.random.default_rng(seed)

    def set_phase(self, phase: int):
        """Switch between cooperative (1) and adversarial (2) phases."""
        self.adversary_active = phase >= 2

    def _encode_state(self, state: np.ndarray) -> np.ndarray:
        """One-hot encode a target state vector."""
        encoded = np.zeros(self.num_attributes * self.num_values, dtype=np.float32)
        for i, val in enumerate(state):
            encoded[i * self.num_values + val] = 1.0
        return encoded

    def _encode_message(self, message: np.ndarray) -> np.ndarray:
        """One-hot encode a discrete message."""
        encoded = np.zeros(self.msg_length * self.vocab_size, dtype=np.float32)
        for i, sym in enumerate(message):
            encoded[i * self.vocab_size + sym] = 1.0
        return encoded

    def _generate_target(self) -> np.ndarray:
        """Generate a random target state."""
        return self._rng.integers(0, self.num_values, size=self.num_attributes)

    def _generate_distractors(self, target: np.ndarray) -> list[np.ndarray]:
        """Generate distractors that differ from the target."""
        distractors = []
        for _ in range(self.num_distractors):
            while True:
                d = self._rng.integers(
                    0, self.num_values, size=self.num_attributes
                )
                if not np.array_equal(d, target) and not any(
                    np.array_equal(d, existing) for existing in distractors
                ):
                    distractors.append(d)
                    break
        return distractors

    def reset(self) -> dict[str, np.ndarray]:
        """Reset environment and return initial observations.

        Returns:
            Dictionary with speaker observation (listener and adversary
            observations are not available until the speaker acts).
        """
        self._target = self._generate_target()
        self._distractors = self._generate_distractors(self._target)
        self._message = None

        # Shuffle candidates and track target index
        candidates = [self._target] + self._distractors
        indices = list(range(len(candidates)))
        self._rng.shuffle(indices)
        self._candidates = [candidates[i] for i in indices]
        self._target_idx = indices.index(0)

        speaker_obs = self._encode_state(self._target)
        return {"speaker": speaker_obs}

    def step_speaker(self, message: np.ndarray) -> dict[str, np.ndarray]:
        """Speaker sends a message. Returns observations for listener (and adversary).

        Args:
            message: Array of shape (msg_length,) with integer symbols.

        Returns:
            Dictionary with listener and adversary observations.
        """
        self._message = np.clip(message, 0, self.vocab_size - 1)
        encoded_msg = self._encode_message(self._message)

        # Listener observation: candidates (shuffled) + message
        candidate_encodings = np.concatenate(
            [self._encode_state(c) for c in self._candidates]
        )
        listener_obs = np.concatenate([candidate_encodings, encoded_msg])

        obs = {"listener": listener_obs}

        if self.adversary_active:
            obs["adversary"] = encoded_msg.copy()

        return obs

    def step_listener(self, selection: int) -> dict[str, float]:
        """Listener selects a candidate.

        Args:
            selection: Index of chosen candidate.

        Returns:
            Partial rewards dict (adversary reward added in step_adversary).
        """
        self._listener_correct = int(selection == self._target_idx)
        return {"listener_correct": self._listener_correct}

    def step_adversary(self, reconstruction: np.ndarray) -> dict[str, float]:
        """Adversary tries to reconstruct the target.

        Args:
            reconstruction: Array of shape (num_attributes,) with predicted values.

        Returns:
            Full rewards dictionary for all agents this episode.
        """
        reconstruction = np.clip(reconstruction, 0, self.num_values - 1)

        # Adversary accuracy: fraction of attributes correctly reconstructed
        adversary_correct = np.mean(reconstruction == self._target)

        # Cooperative team reward
        coop_reward = self.cooperative_reward_weight * self._listener_correct

        if self.adversary_active:
            # Penalize cooperative team for information leakage
            leakage_penalty = self.leakage_penalty_weight * adversary_correct
            speaker_reward = coop_reward - leakage_penalty
            listener_reward = coop_reward - leakage_penalty
            adversary_reward = adversary_correct
        else:
            speaker_reward = coop_reward
            listener_reward = coop_reward
            adversary_reward = 0.0

        return {
            "speaker": float(speaker_reward),
            "listener": float(listener_reward),
            "adversary": float(adversary_reward),
            "listener_correct": self._listener_correct,
            "adversary_accuracy": float(adversary_correct),
        }

    def step(
        self,
        speaker_message: np.ndarray,
        listener_selection: int,
        adversary_reconstruction: np.ndarray | None = None,
    ) -> dict:
        """Run a full episode step (all three agents act).

        Convenience method that calls step_speaker, step_listener,
        step_adversary in sequence.

        Returns:
            Dictionary with rewards and info for all agents.
        """
        obs = self.step_speaker(speaker_message)
        self.step_listener(listener_selection)

        if self.adversary_active and adversary_reconstruction is not None:
            rewards = self.step_adversary(adversary_reconstruction)
        else:
            # No adversary: return cooperative rewards only
            rewards = {
                "speaker": float(
                    self.cooperative_reward_weight * self._listener_correct
                ),
                "listener": float(
                    self.cooperative_reward_weight * self._listener_correct
                ),
                "adversary": 0.0,
                "listener_correct": self._listener_correct,
                "adversary_accuracy": 0.0,
            }

        return {
            "observations": obs,
            "rewards": rewards,
        }

    def get_obs_dim(self, agent_name: str) -> int:
        """Return the observation dimension for a given agent."""
        return self.observation_spaces[agent_name].shape[0]

    def get_action_dim(self, agent_name: str) -> int:
        """Return the action dimension for a given agent."""
        space = self.action_spaces[agent_name]
        if isinstance(space, spaces.Discrete):
            return space.n
        elif isinstance(space, spaces.MultiDiscrete):
            return int(np.sum(space.nvec))
        return 0

    def get_action_shape(self, agent_name: str) -> tuple:
        """Return the raw action shape for a given agent."""
        space = self.action_spaces[agent_name]
        if isinstance(space, spaces.Discrete):
            return (space.n,)
        elif isinstance(space, spaces.MultiDiscrete):
            return tuple(space.nvec)
        return ()
