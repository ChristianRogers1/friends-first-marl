"""Tests for MADDPG agents and network architectures."""

import numpy as np
import torch
import pytest
from src.agents.networks import Actor, Critic, CommunicationModule
from src.agents.maddpg import MADDPGAgent


def test_communication_module():
    comm = CommunicationModule(input_dim=15, vocab_size=10, msg_length=3)
    obs = torch.randn(4, 15)  # batch of 4
    comm.train()
    msg, probs = comm(obs, hard=False)
    assert msg.shape == (4, 30)  # msg_length * vocab_size
    assert probs.shape == (4, 3, 10)
    # Probs should sum to 1 per slot
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4, 3), atol=1e-5)


def test_communication_module_eval():
    comm = CommunicationModule(input_dim=15, vocab_size=10, msg_length=3)
    obs = torch.randn(4, 15)
    comm.eval()
    msg, probs = comm(obs, hard=False)
    # In eval mode, output should be one-hot (argmax)
    reshaped = msg.view(4, 3, 10)
    assert torch.all(reshaped.sum(dim=-1) == 1.0)


def test_discrete_message():
    comm = CommunicationModule(input_dim=15, vocab_size=10, msg_length=3)
    obs = torch.randn(2, 15)
    indices = comm.get_discrete_message(obs)
    assert indices.shape == (2, 3)
    assert (indices >= 0).all() and (indices < 10).all()


def test_actor():
    actor = Actor(obs_dim=20, action_dim=5, discrete=True)
    obs = torch.randn(4, 20)
    action = actor(obs)
    assert action.shape == (4, 5)
    # Softmax output should sum to 1
    assert torch.allclose(action.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_critic():
    critic = Critic(total_obs_dim=50, total_action_dim=30)
    obs = torch.randn(4, 50)
    actions = torch.randn(4, 30)
    q = critic(obs, actions)
    assert q.shape == (4, 1)


def test_maddpg_agent_creation():
    agent = MADDPGAgent(
        agent_name="listener",
        obs_dim=20,
        action_dim=5,
        total_obs_dim=50,
        total_action_dim=30,
    )
    assert agent.comm_module is None


def test_maddpg_speaker_has_comm():
    agent = MADDPGAgent(
        agent_name="speaker",
        obs_dim=15,
        action_dim=30,
        total_obs_dim=50,
        total_action_dim=30,
        vocab_size=10,
        msg_length=3,
    )
    assert agent.comm_module is not None


def test_maddpg_select_action():
    agent = MADDPGAgent(
        agent_name="listener",
        obs_dim=20,
        action_dim=5,
        total_obs_dim=50,
        total_action_dim=30,
    )
    obs = np.random.randn(20).astype(np.float32)
    action = agent.select_action(obs, explore=True)
    assert action.shape == (5,)
    # One-hot: exactly one 1
    assert np.sum(action) == 1.0


def test_maddpg_speaker_select_action():
    agent = MADDPGAgent(
        agent_name="speaker",
        obs_dim=15,
        action_dim=30,
        total_obs_dim=50,
        total_action_dim=30,
        vocab_size=10,
        msg_length=3,
    )
    obs = np.random.randn(15).astype(np.float32)
    action = agent.select_action(obs, explore=False)
    assert action.shape == (30,)  # msg_length * vocab_size
