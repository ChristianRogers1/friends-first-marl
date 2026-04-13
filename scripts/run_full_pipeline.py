#!/usr/bin/env python3
"""
Full experimental pipeline: screening -> selection -> confirmation -> evaluation.

Runs the complete two-stage screen-and-confirm procedure described in the
project proposal:

  1. SCREENING -- Train 5 seeds x 4 pretraining ratios (0.1, 0.25, 0.5, 0.75).
     Evaluate each, pick the ratio with the highest mean listener accuracy.

  2. CONFIRMATION -- Train 12 fresh seeds under the best ratio AND 12 seeds
     under the baseline (ratio = 0.0).  Evaluate all 24 runs on H1/H2/H3.

  3. EXPORT -- Write per-seed and aggregate results to CSV for MATLAB analysis.

Usage:
    python scripts/run_full_pipeline.py --device cuda
    python scripts/run_full_pipeline.py --device cuda --total-episodes 50000
    python scripts/run_full_pipeline.py --device cpu  --total-episodes 5000   # quick test
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.utils.config import load_config
from src.training.trainer import TwoPhaseTrainer
from src.evaluation.metrics import (
    evaluate_task_performance,
    compute_cic,
    compute_information_leakage,
    compute_effect_size,
)

# ------------------------------------------------------------------ #
#  Constants matching the project proposal                            #
# ------------------------------------------------------------------ #
SCREENING_RATIOS = [0.1, 0.25, 0.5, 0.75]
SCREENING_SEEDS = 5          # seeds per ratio in screening stage
CONFIRMATION_SEEDS = 12      # seeds per condition in confirmation stage
EVAL_EPISODES = 500          # episodes for H1/H3 evaluation
CIC_EPISODES = 200           # episodes for H2 (CIC) evaluation
CIC_COUNTERFACTUALS = 10     # counterfactual messages per CIC episode


def make_seed_list(n: int, offset: int = 0) -> list[int]:
    """Deterministic seed list: 0, 100, 200, ..."""
    return [i * 100 + offset for i in range(n)]


# ================================================================== #
#  SCREENING STAGE                                                    #
# ================================================================== #
def run_screening(base_config: dict) -> tuple[float, dict]:
    """Run the screening stage and return (best_ratio, all_results)."""
    print("\n" + "=" * 70)
    print("  STAGE 1 : SCREENING")
    print("=" * 70)

    all_results: dict[str, list[dict]] = {}

    for ratio in SCREENING_RATIOS:
        print(f"\n--- Ratio {ratio} ---")
        ratio_results = []
        seeds = make_seed_list(SCREENING_SEEDS)

        for idx, seed in enumerate(seeds):
            tag = f"[ratio={ratio} seed={seed} ({idx+1}/{SCREENING_SEEDS})]"
            print(f"\n{tag}  Training ...")

            cfg = _clone(base_config)
            cfg["training"]["pretraining_ratio"] = ratio
            cfg["experiment"]["log_dir"] = f"runs/screening/ratio{ratio}"
            cfg["experiment"]["checkpoint_dir"] = f"checkpoints/screening/ratio{ratio}"
            cfg["experiment"]["results_dir"] = f"results/screening/ratio{ratio}"

            trainer = TwoPhaseTrainer(cfg)
            trainer.train(seed=seed, log_suffix=f"_ratio{ratio}")

            # Quick evaluation (H1 only for screening)
            perf = evaluate_task_performance(
                trainer.env, trainer.agents,
                num_episodes=EVAL_EPISODES, phase=2, config=cfg,
            )
            leakage = compute_information_leakage(
                trainer.env, trainer.agents,
                num_episodes=EVAL_EPISODES, config=cfg,
            )

            result = {
                "seed": seed,
                "ratio": ratio,
                "listener_accuracy": perf["accuracy"],
                "listener_ci_lower": perf["ci_lower"],
                "listener_ci_upper": perf["ci_upper"],
                "adversary_accuracy": leakage["adversary_accuracy"],
                "leakage_ratio": leakage["leakage_ratio"],
            }
            ratio_results.append(result)
            print(f"{tag}  listener_acc={perf['accuracy']:.4f}  "
                  f"adv_acc={leakage['adversary_accuracy']:.4f}")

        all_results[str(ratio)] = ratio_results

    # Select best ratio
    ratio_means = {}
    for ratio_str, results in all_results.items():
        mean_acc = np.mean([r["listener_accuracy"] for r in results])
        ratio_means[float(ratio_str)] = mean_acc

    best_ratio = max(ratio_means, key=ratio_means.get)

    print(f"\n{'=' * 70}")
    print("  SCREENING SUMMARY")
    print(f"{'=' * 70}")
    for ratio in SCREENING_RATIOS:
        marker = " <-- BEST" if ratio == best_ratio else ""
        print(f"  ratio {ratio:>4}  ->  mean listener accuracy = "
              f"{ratio_means[ratio]:.4f}{marker}")
    print(f"\n  Selected ratio: {best_ratio}")

    # Save screening results
    os.makedirs("results/screening", exist_ok=True)
    with open("results/screening/screening_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return best_ratio, all_results


# ================================================================== #
#  CONFIRMATION STAGE                                                 #
# ================================================================== #
def run_confirmation(
    base_config: dict,
    best_ratio: float,
) -> tuple[list[dict], list[dict]]:
    """Run confirmation stage: best ratio vs baseline, 12 seeds each."""
    print(f"\n{'=' * 70}")
    print("  STAGE 2 : CONFIRMATION")
    print(f"{'=' * 70}")

    pretrained_results = _run_condition(
        base_config, ratio=best_ratio, label="pretrained",
        seeds=make_seed_list(CONFIRMATION_SEEDS, offset=1000),
    )
    baseline_results = _run_condition(
        base_config, ratio=0.0, label="baseline",
        seeds=make_seed_list(CONFIRMATION_SEEDS, offset=2000),
    )

    return pretrained_results, baseline_results


def _run_condition(
    base_config: dict,
    ratio: float,
    label: str,
    seeds: list[int],
) -> list[dict]:
    """Train + evaluate one experimental condition across all seeds."""
    print(f"\n--- Condition: {label} (ratio={ratio}, {len(seeds)} seeds) ---")
    results = []

    for idx, seed in enumerate(seeds):
        tag = f"[{label} seed={seed} ({idx+1}/{len(seeds)})]"
        print(f"\n{tag}  Training ...")

        cfg = _clone(base_config)
        cfg["training"]["pretraining_ratio"] = ratio
        cfg["experiment"]["log_dir"] = f"runs/confirmation/{label}"
        cfg["experiment"]["checkpoint_dir"] = f"checkpoints/confirmation/{label}"
        cfg["experiment"]["results_dir"] = f"results/confirmation/{label}"

        trainer = TwoPhaseTrainer(cfg)
        history = trainer.train(seed=seed, log_suffix=f"_ratio{ratio}")

        # Full evaluation (H1, H2, H3)
        print(f"{tag}  Evaluating H1 (task performance) ...")
        perf = evaluate_task_performance(
            trainer.env, trainer.agents,
            num_episodes=EVAL_EPISODES, phase=2, config=cfg,
        )

        print(f"{tag}  Evaluating H2 (CIC) ...")
        cic = compute_cic(
            trainer.env, trainer.agents,
            num_episodes=CIC_EPISODES,
            num_counterfactuals=CIC_COUNTERFACTUALS,
            config=cfg,
        )

        print(f"{tag}  Evaluating H3 (information leakage) ...")
        leakage = compute_information_leakage(
            trainer.env, trainer.agents,
            num_episodes=EVAL_EPISODES, config=cfg,
        )

        result = {
            "condition": label,
            "seed": seed,
            "ratio": ratio,
            # H1
            "listener_accuracy": perf["accuracy"],
            "listener_ci_lower": perf["ci_lower"],
            "listener_ci_upper": perf["ci_upper"],
            "listener_std": perf["std"],
            # H2
            "mean_cic": cic["mean_cic"],
            "std_cic": cic["std_cic"],
            # H3
            "adversary_accuracy": leakage["adversary_accuracy"],
            "leakage_ratio": leakage["leakage_ratio"],
        }
        results.append(result)

        print(f"{tag}  listener={perf['accuracy']:.4f}  "
              f"cic={cic['mean_cic']:.4f}  "
              f"adv={leakage['adversary_accuracy']:.4f}  "
              f"leak={leakage['leakage_ratio']:.4f}")

    return results


# ================================================================== #
#  EXPORT FOR MATLAB                                                  #
# ================================================================== #
def export_results(
    screening_results: dict,
    pretrained_results: list[dict],
    baseline_results: list[dict],
    best_ratio: float,
    output_dir: str = "results",
):
    """Write all results to CSV files for MATLAB analysis."""
    print(f"\n{'=' * 70}")
    print("  EXPORTING RESULTS FOR MATLAB")
    print(f"{'=' * 70}")

    os.makedirs(output_dir, exist_ok=True)

    # --- screening_results.csv ---
    screening_path = os.path.join(output_dir, "screening_results.csv")
    screening_rows = []
    for ratio_str, results in screening_results.items():
        for r in results:
            screening_rows.append(r)
    _write_csv(screening_path, screening_rows)
    print(f"  Wrote {screening_path}  ({len(screening_rows)} rows)")

    # --- confirmation_results.csv ---
    confirm_path = os.path.join(output_dir, "confirmation_results.csv")
    all_confirm = pretrained_results + baseline_results
    _write_csv(confirm_path, all_confirm)
    print(f"  Wrote {confirm_path}  ({len(all_confirm)} rows)")

    # --- pretrained_results.csv ---
    pre_path = os.path.join(output_dir, "pretrained_results.csv")
    _write_csv(pre_path, pretrained_results)
    print(f"  Wrote {pre_path}  ({len(pretrained_results)} rows)")

    # --- baseline_results.csv ---
    base_path = os.path.join(output_dir, "baseline_results.csv")
    _write_csv(base_path, baseline_results)
    print(f"  Wrote {base_path}  ({len(baseline_results)} rows)")

    # --- effect_size.csv ---
    effect = compute_effect_size(
        [r["listener_accuracy"] for r in pretrained_results],
        [r["listener_accuracy"] for r in baseline_results],
    )
    effect["best_ratio"] = best_ratio
    effect_path = os.path.join(output_dir, "effect_size.csv")
    _write_csv(effect_path, [effect])
    print(f"  Wrote {effect_path}")

    # --- summary.json (full dump) ---
    summary = {
        "best_ratio": best_ratio,
        "effect_size": effect,
        "pretrained": pretrained_results,
        "baseline": baseline_results,
        "screening": screening_results,
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Wrote {summary_path}")


def _write_csv(path: str, rows: list[dict]):
    """Write a list of dicts to a CSV file."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _clone(d: dict) -> dict:
    """Deep-clone a nested dict (no numpy/torch objects expected)."""
    return json.loads(json.dumps(d))


# ================================================================== #
#  MAIN                                                               #
# ================================================================== #
def main():
    parser = argparse.ArgumentParser(
        description="Run full experimental pipeline: screening -> confirmation -> export"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--total-episodes", type=int, default=None,
                        help="Override total_episodes (for quick testing)")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--skip-screening", action="store_true",
                        help="Skip screening, use --best-ratio instead")
    parser.add_argument("--best-ratio", type=float, default=None,
                        help="Manually specify best ratio (skips screening)")
    args = parser.parse_args()

    config = load_config(args.config)
    config["experiment"]["device"] = args.device
    if args.total_episodes is not None:
        config["training"]["total_episodes"] = args.total_episodes

    print("=" * 70)
    print("  COOPERATIVE PRETRAINING EXPERIMENT PIPELINE")
    print("=" * 70)
    print(f"  Device          : {args.device}")
    print(f"  Total episodes  : {config['training']['total_episodes']}")
    print(f"  Screening seeds : {SCREENING_SEEDS} per ratio")
    print(f"  Confirm seeds   : {CONFIRMATION_SEEDS} per condition")
    print(f"  Eval episodes   : {EVAL_EPISODES}")
    print(f"  CIC episodes    : {CIC_EPISODES}")

    # Stage 1: Screening
    if args.skip_screening and args.best_ratio is not None:
        best_ratio = args.best_ratio
        screening_results = {}
        print(f"\n  Skipping screening, using ratio={best_ratio}")
    else:
        best_ratio, screening_results = run_screening(config)

    # Stage 2: Confirmation
    pretrained_results, baseline_results = run_confirmation(config, best_ratio)

    # Print confirmation summary
    print(f"\n{'=' * 70}")
    print("  CONFIRMATION SUMMARY")
    print(f"{'=' * 70}")

    pre_accs  = [r["listener_accuracy"] for r in pretrained_results]
    base_accs = [r["listener_accuracy"] for r in baseline_results]
    pre_cics  = [r["mean_cic"] for r in pretrained_results]
    base_cics = [r["mean_cic"] for r in baseline_results]
    pre_leak  = [r["leakage_ratio"] for r in pretrained_results]
    base_leak = [r["leakage_ratio"] for r in baseline_results]

    effect = compute_effect_size(pre_accs, base_accs)

    print(f"\n  H1 - Task Performance (listener accuracy):")
    print(f"    Pretrained : {np.mean(pre_accs):.4f} +/- {np.std(pre_accs):.4f}")
    print(f"    Baseline   : {np.mean(base_accs):.4f} +/- {np.std(base_accs):.4f}")
    print(f"    Cohen's d  : {effect['cohens_d']:.4f}")

    print(f"\n  H2 - Communication Robustness (CIC):")
    print(f"    Pretrained : {np.mean(pre_cics):.4f} +/- {np.std(pre_cics):.4f}")
    print(f"    Baseline   : {np.mean(base_cics):.4f} +/- {np.std(base_cics):.4f}")

    print(f"\n  H3 - Information Leakage (leakage ratio):")
    print(f"    Pretrained : {np.mean(pre_leak):.4f} +/- {np.std(pre_leak):.4f}")
    print(f"    Baseline   : {np.mean(base_leak):.4f} +/- {np.std(base_leak):.4f}")

    # Stage 3: Export
    export_results(
        screening_results, pretrained_results, baseline_results,
        best_ratio, output_dir=args.output_dir,
    )

    print(f"\n{'=' * 70}")
    print("  PIPELINE COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Results exported to: {args.output_dir}/")
    print(f"  Run analysis: python analysis/analyze_results.py")


if __name__ == "__main__":
    main()
