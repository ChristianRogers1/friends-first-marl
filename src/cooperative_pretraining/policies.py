from __future__ import annotations

import numpy as np

from cooperative_pretraining.envs.public_eavesdrop_env import ADVERSARY, LISTENER, SPEAKER


def speaker_heuristic_action(observation: dict[str, np.ndarray | int]) -> np.ndarray:
    target = np.asarray(observation["speaker_target"], dtype=np.int64)
    message_length = int(np.asarray(observation["public_message"]).shape[0])
    return target[:message_length].copy()


def listener_heuristic_action(observation: dict[str, np.ndarray | int]) -> int:
    candidates = np.asarray(observation["listener_candidates"], dtype=np.int64)
    message = np.asarray(observation["public_message"], dtype=np.int64)
    encoded = candidates[:, : len(message)]
    distances = np.not_equal(encoded, message).sum(axis=1)
    return int(np.argmin(distances))


def adversary_heuristic_action(
    observation: dict[str, np.ndarray | int], num_attributes: int
) -> np.ndarray:
    message = np.asarray(observation["public_message"], dtype=np.int64)
    reconstruction = np.zeros(num_attributes, dtype=np.int64)
    reconstruction[: len(message)] = message
    return reconstruction


def heuristic_actions(observations: dict[str, dict[str, np.ndarray | int]]) -> dict[str, np.ndarray | int]:
    step_id = int(observations[SPEAKER]["step_id"])
    message_len = int(np.asarray(observations[SPEAKER]["public_message"]).shape[0])
    num_attributes = int(len(np.asarray(observations[SPEAKER]["speaker_target"])))
    if step_id == 0:
        return {
            SPEAKER: speaker_heuristic_action(observations[SPEAKER]),
            LISTENER: 0,
            ADVERSARY: np.zeros(num_attributes, dtype=np.int64),
        }

    return {
        SPEAKER: np.zeros(message_len, dtype=np.int64),
        LISTENER: listener_heuristic_action(observations[LISTENER]),
        ADVERSARY: adversary_heuristic_action(observations[ADVERSARY], num_attributes),
    }


def random_actions(
    observations: dict[str, dict[str, np.ndarray | int]], rng: np.random.Generator
) -> dict[str, np.ndarray | int]:
    speaker_obs = observations[SPEAKER]
    listener_obs = observations[LISTENER]
    step_id = int(speaker_obs["step_id"])
    message_len = int(np.asarray(speaker_obs["public_message"]).shape[0])
    num_candidates = int(np.asarray(listener_obs["listener_candidates"]).shape[0])
    num_attributes = int(len(np.asarray(speaker_obs["speaker_target"])))

    if step_id == 0:
        return {
            SPEAKER: rng.integers(0, 10, size=message_len, dtype=np.int64),
            LISTENER: 0,
            ADVERSARY: np.zeros(num_attributes, dtype=np.int64),
        }

    return {
        SPEAKER: np.zeros(message_len, dtype=np.int64),
        LISTENER: int(rng.integers(0, num_candidates)),
        ADVERSARY: rng.integers(0, 10, size=num_attributes, dtype=np.int64),
    }
