"""Tests for the AdversarialCryptoEnv custom environment."""

import numpy as np
import pytest
from src.environments.adversarial_crypto import AdversarialCryptoEnv


def test_env_creation():
    env = AdversarialCryptoEnv()
    assert env.num_attributes == 3
    assert env.num_values == 5
    assert env.vocab_size == 10
    assert env.msg_length == 3


def test_reset_returns_speaker_obs():
    env = AdversarialCryptoEnv()
    obs = env.reset()
    assert "speaker" in obs
    expected_dim = env.num_attributes * env.num_values
    assert obs["speaker"].shape == (expected_dim,)
    # One-hot: exactly num_attributes ones
    assert np.sum(obs["speaker"]) == env.num_attributes


def test_speaker_step():
    env = AdversarialCryptoEnv(adversary_active=True)
    env.reset()
    message = np.array([0, 1, 2])
    obs = env.step_speaker(message)
    assert "listener" in obs
    assert "adversary" in obs
    assert obs["listener"].shape == (env.observation_spaces["listener"].shape[0],)
    assert obs["adversary"].shape == (env.observation_spaces["adversary"].shape[0],)


def test_speaker_step_no_adversary():
    env = AdversarialCryptoEnv(adversary_active=False)
    env.reset()
    message = np.array([0, 1, 2])
    obs = env.step_speaker(message)
    assert "listener" in obs
    assert "adversary" not in obs


def test_full_episode_cooperative():
    env = AdversarialCryptoEnv(adversary_active=False)
    env.seed(42)
    obs = env.reset()
    msg = np.array([3, 5, 7])
    result = env.step(msg, listener_selection=0)
    assert "rewards" in result
    assert result["rewards"]["adversary"] == 0.0


def test_full_episode_adversarial():
    env = AdversarialCryptoEnv(adversary_active=True)
    env.seed(42)
    obs = env.reset()
    msg = np.array([3, 5, 7])
    recon = np.array([0, 0, 0])
    result = env.step(msg, listener_selection=0, adversary_reconstruction=recon)
    assert "rewards" in result
    assert "adversary" in result["rewards"]


def test_phase_switching():
    env = AdversarialCryptoEnv()
    env.set_phase(1)
    assert not env.adversary_active
    env.set_phase(2)
    assert env.adversary_active


def test_obs_action_dims():
    env = AdversarialCryptoEnv()
    assert env.get_obs_dim("speaker") == 3 * 5  # num_attributes * num_values
    assert env.get_obs_dim("adversary") == 3 * 10  # msg_length * vocab_size
    assert env.get_action_dim("listener") == 3  # num_candidates (1 target + 2 distractors)


def test_distractors_differ_from_target():
    env = AdversarialCryptoEnv()
    env.seed(0)
    for _ in range(50):
        env.reset()
        target = env._target
        for d in env._distractors:
            assert not np.array_equal(target, d)
