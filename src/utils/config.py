"""
Configuration loading and management.
"""

import yaml
from pathlib import Path


DEFAULT_CONFIG = {
    # Environment
    "env": {
        "num_attributes": 3,
        "num_values": 5,
        "num_distractors": 2,
        "vocab_size": 10,
        "msg_length": 3,
        "cooperative_reward_weight": 1.0,
        "leakage_penalty_weight": 0.5,
    },
    # Agent
    "agent": {
        "hidden_dim": 128,
        "comm_hidden_dim": 64,
        "lr_actor": 1e-3,
        "lr_critic": 1e-3,
        "gamma": 0.95,
        "tau": 0.01,
        "gumbel_temperature": 1.0,
    },
    # Training
    "training": {
        "total_episodes": 50_000,
        "pretraining_ratio": 0.25,
        "batch_size": 256,
        "buffer_capacity": 100_000,
        "update_every": 4,
        "eval_every": 500,
        "eval_episodes": 100,
        "checkpoint_every": 5000,
        "noise_scale": 0.1,
        "noise_decay": 0.9999,
        "min_noise": 0.01,
    },
    # Experiment
    "experiment": {
        "seed": 42,
        "num_seeds": 5,
        "device": "cpu",
        "log_dir": "runs",
        "checkpoint_dir": "checkpoints",
        "results_dir": "results",
    },
}


def load_config(config_path: str | None = None) -> dict:
    """Load configuration from YAML file, falling back to defaults.

    Args:
        config_path: Path to YAML config file. If None, uses defaults.

    Returns:
        Configuration dictionary.
    """
    config = _deep_copy_dict(DEFAULT_CONFIG)

    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                user_config = yaml.safe_load(f)
            if user_config:
                _deep_update(config, user_config)

    return config


def _deep_copy_dict(d: dict) -> dict:
    """Deep copy a nested dictionary."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _deep_copy_dict(v)
        else:
            result[k] = v
    return result


def _deep_update(base: dict, update: dict):
    """Recursively update base dict with values from update dict."""
    for k, v in update.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
