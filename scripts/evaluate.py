#!/usr/bin/env python3
"""
Evaluate trained agents on all three hypotheses metrics.

Usage:
    python scripts/evaluate.py --config configs/default.yaml --checkpoint-dir checkpoints/seed_42_ratio0.25
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.training.trainer import TwoPhaseTrainer
from src.evaluation.metrics import (
    evaluate_task_performance,
    compute_cic,
    compute_information_leakage,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained agents")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--num-episodes", type=int, default=500)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    # Build trainer and load checkpoints
    trainer = TwoPhaseTrainer(config)

    # Find latest checkpoint
    checkpoint_files = [f for f in os.listdir(args.checkpoint_dir) if f.endswith(".pt")]
    if not checkpoint_files:
        print(f"No checkpoints found in {args.checkpoint_dir}")
        return

    # Load latest checkpoint for each agent
    for agent_name in ["speaker", "listener", "adversary"]:
        agent_files = sorted(
            [f for f in checkpoint_files if f.startswith(agent_name)],
            key=lambda x: int(x.split("ep")[1].split(".")[0]),
        )
        if agent_files:
            path = os.path.join(args.checkpoint_dir, agent_files[-1])
            trainer.agents[agent_name].load(path)
            print(f"Loaded {agent_name} from {agent_files[-1]}")

    # H1: Task Performance
    print("\n--- H1: Task Performance ---")
    perf = evaluate_task_performance(
        trainer.env, trainer.agents, num_episodes=args.num_episodes, config=config
    )
    print(f"  Listener accuracy: {perf['accuracy']:.4f}")
    print(f"  95% CI: [{perf['ci_lower']:.4f}, {perf['ci_upper']:.4f}]")

    # H2: Communication Robustness (CIC)
    print("\n--- H2: Communication Robustness (CIC) ---")
    cic = compute_cic(
        trainer.env, trainer.agents, num_episodes=min(200, args.num_episodes), config=config
    )
    print(f"  Mean CIC: {cic['mean_cic']:.4f}")
    print(f"  Std CIC:  {cic['std_cic']:.4f}")

    # H3: Information Leakage
    print("\n--- H3: Information Leakage ---")
    leakage = compute_information_leakage(
        trainer.env, trainer.agents, num_episodes=args.num_episodes, config=config
    )
    print(f"  Adversary accuracy: {leakage['adversary_accuracy']:.4f}")
    print(f"  Leakage ratio:      {leakage['leakage_ratio']:.4f}")

    # Save results
    results = {
        "task_performance": {k: v for k, v in perf.items() if k != "per_episode"},
        "cic": {k: v for k, v in cic.items() if k != "per_episode"},
        "information_leakage": {
            k: v
            for k, v in leakage.items()
            if k not in ("listener_per_episode", "adversary_per_episode")
        },
    }

    output_path = args.output or os.path.join(
        args.checkpoint_dir, "evaluation_results.json"
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
