#!/usr/bin/env python3
"""
Run the screening stage: train across multiple pretraining ratios
and seeds to identify the best configuration.

Usage:
    python scripts/run_screening.py --config configs/screening.yaml
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from src.utils.config import load_config
from src.training.trainer import TwoPhaseTrainer
from src.evaluation.metrics import evaluate_task_performance, compute_cic, compute_information_leakage


def main():
    parser = argparse.ArgumentParser(description="Run screening stage")
    parser.add_argument(
        "--config", type=str, default="configs/screening.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Load screening-specific config
    with open(args.config) as f:
        screening_config = yaml.safe_load(f)

    ratios = screening_config.get("screening", {}).get(
        "pretraining_ratios", [0.1, 0.25, 0.5, 0.75]
    )
    seeds_per_ratio = screening_config.get("screening", {}).get(
        "seeds_per_ratio", 5
    )

    all_results = {}

    for ratio in ratios:
        print(f"\n{'='*60}")
        print(f"Screening: pretraining_ratio = {ratio}")
        print(f"{'='*60}")

        ratio_results = []
        config["training"]["pretraining_ratio"] = ratio

        for seed_idx in range(seeds_per_ratio):
            seed = seed_idx * 100 + 42
            print(f"\n  Seed {seed} ({seed_idx + 1}/{seeds_per_ratio})")

            config["experiment"]["log_dir"] = f"runs/screening/ratio{ratio}"
            config["experiment"]["checkpoint_dir"] = f"checkpoints/screening/ratio{ratio}"

            trainer = TwoPhaseTrainer(config)
            history = trainer.train(seed=seed, log_suffix=f"_ratio{ratio}")

            # Evaluate
            eval_result = evaluate_task_performance(
                trainer.env, trainer.agents, num_episodes=200, phase=2, config=config
            )
            leakage_result = compute_information_leakage(
                trainer.env, trainer.agents, num_episodes=200, config=config
            )

            ratio_results.append({
                "seed": seed,
                "listener_accuracy": eval_result["accuracy"],
                "adversary_accuracy": leakage_result["adversary_accuracy"],
                "leakage_ratio": leakage_result["leakage_ratio"],
            })

        all_results[str(ratio)] = ratio_results

    # Save screening results
    results_dir = config["experiment"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "screening_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("SCREENING RESULTS SUMMARY")
    print(f"{'='*60}")
    for ratio, results in all_results.items():
        accuracies = [r["listener_accuracy"] for r in results]
        mean_acc = sum(accuracies) / len(accuracies)
        print(f"  Ratio {ratio}: mean listener accuracy = {mean_acc:.4f}")
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
