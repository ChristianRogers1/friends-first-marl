#!/usr/bin/env python3
"""
Main training entry point.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --seed 42
    python scripts/train.py --config configs/baseline.yaml --seed 0
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.training.trainer import TwoPhaseTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train MADDPG agents with cooperative pretraining"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--pretraining-ratio",
        type=float,
        default=None,
        help="Pretraining ratio (overrides config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cpu or cuda (overrides config)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    if args.pretraining_ratio is not None:
        config["training"]["pretraining_ratio"] = args.pretraining_ratio
    if args.device is not None:
        config["experiment"]["device"] = args.device

    seed = config["experiment"]["seed"]
    ratio = config["training"]["pretraining_ratio"]
    print(f"Training with seed={seed}, pretraining_ratio={ratio}")
    print(f"Device: {config['experiment']['device']}")
    print(f"Total episodes: {config['training']['total_episodes']}")
    print(f"Phase 1 (cooperative): {int(config['training']['total_episodes'] * ratio)} episodes")
    print(f"Phase 2 (adversarial): {int(config['training']['total_episodes'] * (1 - ratio))} episodes")
    print()

    trainer = TwoPhaseTrainer(config)
    history = trainer.train(
        seed=seed,
        log_suffix=f"_ratio{ratio}",
    )

    # Save results summary
    results_dir = config["experiment"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    import json
    results_path = os.path.join(results_dir, f"history_seed{seed}_ratio{ratio}.json")
    with open(results_path, "w") as f:
        json.dump(history, f)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
