"""Tests for the two-phase training pipeline."""

import pytest
from src.utils.config import load_config
from src.training.trainer import TwoPhaseTrainer


@pytest.fixture
def quick_config():
    """Minimal config for fast testing."""
    config = load_config()
    config["training"]["total_episodes"] = 100
    config["training"]["pretraining_ratio"] = 0.5
    config["training"]["batch_size"] = 16
    config["training"]["buffer_capacity"] = 500
    config["training"]["update_every"] = 2
    config["training"]["eval_every"] = 50
    config["training"]["eval_episodes"] = 10
    config["training"]["checkpoint_every"] = 200  # no checkpoint during test
    config["experiment"]["log_dir"] = "/tmp/test_runs"
    config["experiment"]["checkpoint_dir"] = "/tmp/test_checkpoints"
    config["experiment"]["results_dir"] = "/tmp/test_results"
    return config


def test_trainer_creation(quick_config):
    trainer = TwoPhaseTrainer(quick_config)
    assert "speaker" in trainer.agents
    assert "listener" in trainer.agents
    assert "adversary" in trainer.agents
    assert trainer.phase1_episodes == 50
    assert trainer.phase2_episodes == 50


def test_trainer_collect_episode(quick_config):
    trainer = TwoPhaseTrainer(quick_config)
    # Phase 1
    info = trainer._collect_episode(phase=1)
    assert "listener_correct" in info
    assert info["adversary_accuracy"] == 0.0
    assert len(trainer.buffer) == 1

    # Phase 2
    info = trainer._collect_episode(phase=2)
    assert "listener_correct" in info
    assert len(trainer.buffer) == 2


def test_trainer_short_run(quick_config):
    trainer = TwoPhaseTrainer(quick_config)
    history = trainer.train(seed=42)
    assert len(history["listener_accuracy"]) == 100
    assert len(history["phase"]) == 100
    # Check phase transition
    assert history["phase"][0] == 1
    assert history["phase"][-1] == 2
