#!/usr/bin/env python3
"""
Full experimental pipeline: screening -> selection -> confirmation -> evaluation.

Supports two sweep modes:
  - "decoupled" (default): Sweep over pretraining_episodes with a fixed
    phase2_episodes window.  This cleanly separates the pretraining dose from
    the evaluation horizon.
  - "legacy": Sweep over pretraining_ratio (fraction of total_episodes).

Supports multi-GPU parallelism: when --num-gpus > 1, independent training runs
are distributed across GPUs via a worker pool (one job per GPU at a time).

Usage:
    # Decoupled sweep (default) -- uses configs/decoupled_sweep.yaml
    python scripts/run_full_pipeline.py --config configs/decoupled_sweep.yaml \\
        --device cuda --num-gpus 0

    # Legacy ratio sweep
    python scripts/run_full_pipeline.py --sweep-mode legacy --device cuda

    # CPU quick test
    python scripts/run_full_pipeline.py --device cpu --phase2-episodes 100
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.multiprocessing as mp

from src.utils.config import load_config
from src.evaluation.metrics import compute_effect_size

# ------------------------------------------------------------------ #
#  Constants matching the project proposal                            #
# ------------------------------------------------------------------ #
# Legacy (ratio-based) sweep defaults
SCREENING_RATIOS = [0.1, 0.25, 0.5, 0.75]
SCREENING_SEEDS = 5          # seeds per ratio in screening stage
CONFIRMATION_SEEDS = 12      # seeds per condition in confirmation stage

# Decoupled sweep defaults (overridden by config screening section)
DECOUPLED_PRETRAINING_SWEEP = [0, 2500, 5000, 10000, 25000]
DECOUPLED_SCREENING_SEEDS = 5
DECOUPLED_CONFIRMATION_SEEDS = 12

# Shared evaluation constants
EVAL_EPISODES = 500          # episodes for H1/H3 evaluation
CIC_EPISODES = 200           # episodes for H2 (CIC) evaluation
CIC_COUNTERFACTUALS = 10     # counterfactual messages per CIC episode


def make_seed_list(n: int, offset: int = 0) -> list[int]:
    """Deterministic seed list: 0, 100, 200, ..."""
    return [i * 100 + offset for i in range(n)]


# ================================================================== #
#  WORKER FUNCTION  (runs in a spawned subprocess for multi-GPU)      #
# ================================================================== #

# Module-level GPU queue, set by _init_gpu_pool in each worker process.
_gpu_queue = None


def _init_gpu_pool(queue):
    """Initializer for worker processes: store the shared GPU queue."""
    global _gpu_queue
    _gpu_queue = queue


def _run_job(job: dict) -> dict:
    """Train + evaluate a single (seed, ratio) run.

    In multi-GPU mode this runs in a spawned process. It acquires a GPU ID
    from the shared queue, trains on cuda:<id>, then releases the GPU.

    In single-GPU / CPU mode it is called directly in the main process.

    Args:
        job: Dictionary with keys:
            seed, ratio, label, eval_mode ("screening" | "confirmation"),
            base_config (serialized dict), quiet (bool).

    Returns:
        Result dictionary.
    """
    # ---- GPU acquisition ----
    global _gpu_queue
    gpu_id = None
    if _gpu_queue is not None:
        gpu_id = _gpu_queue.get()

    try:
        return _run_job_inner(job, gpu_id)
    finally:
        if _gpu_queue is not None and gpu_id is not None:
            _gpu_queue.put(gpu_id)


def _run_job_inner(job: dict, gpu_id: int | None) -> dict:
    """Core logic for a single training + evaluation run."""
    # Imports here so spawned processes can find everything
    from src.training.trainer import TwoPhaseTrainer
    from src.evaluation.metrics import (
        evaluate_task_performance,
        compute_cic,
        compute_information_leakage,
    )

    seed = job["seed"]
    label = job["label"]
    eval_mode = job["eval_mode"]
    quiet = job.get("quiet", False)
    cfg = json.loads(job["config_json"])  # deserialize config

    # Set device
    if gpu_id is not None:
        cfg["experiment"]["device"] = f"cuda:{gpu_id}"

    # ---- Configure pretraining amount ----
    # Support both decoupled (pretraining_episodes) and legacy (ratio) modes.
    if "pretraining_episodes" in job:
        pretraining_ep = job["pretraining_episodes"]
        cfg["training"]["pretraining_episodes"] = pretraining_ep
        # Remove ratio keys so the trainer uses the decoupled path
        cfg["training"].pop("pretraining_ratio", None)
        run_tag = f"pretrain{pretraining_ep}"
    else:
        ratio = job["ratio"]
        cfg["training"]["pretraining_ratio"] = ratio
        # Remove decoupled keys so the trainer uses the legacy path
        cfg["training"].pop("pretraining_episodes", None)
        cfg["training"].pop("phase2_episodes", None)
        pretraining_ep = None
        run_tag = f"ratio{ratio}"

    device_tag = cfg["experiment"]["device"]
    tag = f"[{label} seed={seed} {run_tag} {device_tag}]"

    cfg["experiment"]["log_dir"] = f"runs/{label}"
    cfg["experiment"]["checkpoint_dir"] = f"checkpoints/{label}"
    cfg["experiment"]["results_dir"] = f"results/{label}"

    print(f"{tag}  Training ...", flush=True)
    t0 = time.time()

    trainer = TwoPhaseTrainer(cfg)
    trainer.train(seed=seed, log_suffix=f"_{run_tag}", quiet=quiet)

    elapsed = time.time() - t0
    print(f"{tag}  Training done ({elapsed:.0f}s). Evaluating ...", flush=True)

    # ---- Evaluation ----
    perf = evaluate_task_performance(
        trainer.env, trainer.agents,
        num_episodes=EVAL_EPISODES, phase=2, config=cfg,
    )

    result = {
        "seed": seed,
        "label": label,
        "listener_accuracy": perf["accuracy"],
        "listener_ci_lower": perf["ci_lower"],
        "listener_ci_upper": perf["ci_upper"],
        "listener_std": perf["std"],
    }

    # Store whichever parameterisation was used
    if pretraining_ep is not None:
        result["pretraining_episodes"] = pretraining_ep
    else:
        result["ratio"] = ratio

    if eval_mode == "screening":
        leakage = compute_information_leakage(
            trainer.env, trainer.agents,
            num_episodes=EVAL_EPISODES, config=cfg,
        )
        result["adversary_accuracy"] = leakage["adversary_accuracy"]
        result["leakage_ratio"] = leakage["leakage_ratio"]

    elif eval_mode == "confirmation":
        cic = compute_cic(
            trainer.env, trainer.agents,
            num_episodes=CIC_EPISODES,
            num_counterfactuals=CIC_COUNTERFACTUALS,
            config=cfg,
        )
        leakage = compute_information_leakage(
            trainer.env, trainer.agents,
            num_episodes=EVAL_EPISODES, config=cfg,
        )
        result.update({
            "condition": label,
            "mean_cic": cic["mean_cic"],
            "std_cic": cic["std_cic"],
            "adversary_accuracy": leakage["adversary_accuracy"],
            "leakage_ratio": leakage["leakage_ratio"],
        })

    total = time.time() - t0
    print(f"{tag}  Done ({total:.0f}s)  listener={perf['accuracy']:.4f}",
          flush=True)

    return result


# ================================================================== #
#  JOB DISPATCHER                                                     #
# ================================================================== #
def run_jobs(jobs: list[dict], num_gpus: int) -> list[dict]:
    """Execute a list of jobs, optionally in parallel across GPUs.

    Args:
        jobs: List of job dicts (see _run_job).
        num_gpus: Number of GPUs. 0 or 1 = sequential, >1 = parallel.

    Returns:
        List of result dicts (same order as jobs).
    """
    if num_gpus <= 1:
        # Sequential: run in main process
        return [_run_job(job) for job in jobs]

    # Parallel: spawn worker pool with GPU semaphore queue
    ctx = mp.get_context("spawn")
    gpu_queue = ctx.Queue()
    for i in range(num_gpus):
        gpu_queue.put(i)

    results = []
    # Pool size = num_gpus: at most one job per GPU at any time
    with ctx.Pool(
        num_gpus,
        initializer=_init_gpu_pool,
        initargs=(gpu_queue,)
    ) as pool:
        results = pool.map(_run_job, jobs)

    return results


# ================================================================== #
#  SCREENING STAGE                                                    #
# ================================================================== #
def run_screening(base_config: dict, num_gpus: int) -> tuple[float, dict]:
    """Run the screening stage and return (best_ratio, all_results)."""
    print(f"\n{'=' * 70}")
    print("  STAGE 1 : SCREENING")
    print(f"  {len(SCREENING_RATIOS)} ratios x {SCREENING_SEEDS} seeds"
          f" = {len(SCREENING_RATIOS) * SCREENING_SEEDS} jobs"
          f"  |  GPUs: {max(num_gpus, 1)}")
    print("=" * 70)

    config_json = json.dumps(base_config)
    quiet = num_gpus > 1

    jobs = []
    for ratio in SCREENING_RATIOS:
        for seed in make_seed_list(SCREENING_SEEDS):
            jobs.append({
                "seed": seed,
                "ratio": ratio,
                "label": f"screening/ratio{ratio}",
                "eval_mode": "screening",
                "config_json": config_json,
                "quiet": quiet,
            })

    results = run_jobs(jobs, num_gpus)

    # Group by ratio
    all_results: dict[str, list[dict]] = {}
    for r in results:
        key = str(r["ratio"])
        all_results.setdefault(key, []).append(r)

    # Select best ratio
    ratio_means = {}
    for ratio_str, ratio_results in all_results.items():
        ratio_means[float(ratio_str)] = np.mean(
            [r["listener_accuracy"] for r in ratio_results]
        )

    best_ratio = max(ratio_means, key=ratio_means.get)

    print(f"\n{'=' * 70}")
    print("  SCREENING SUMMARY")
    print(f"{'=' * 70}")
    for ratio in SCREENING_RATIOS:
        marker = " <-- BEST" if ratio == best_ratio else ""
        print(f"  ratio {ratio:>4}  ->  mean listener accuracy = "
              f"{ratio_means[ratio]:.4f}{marker}")
    print(f"\n  Selected ratio: {best_ratio}")

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
    num_gpus: int,
) -> tuple[list[dict], list[dict]]:
    """Run confirmation stage: best ratio vs baseline, 12 seeds each."""
    print(f"\n{'=' * 70}")
    print("  STAGE 2 : CONFIRMATION")
    print(f"  2 conditions x {CONFIRMATION_SEEDS} seeds"
          f" = {2 * CONFIRMATION_SEEDS} jobs"
          f"  |  GPUs: {max(num_gpus, 1)}")
    print("=" * 70)

    config_json = json.dumps(base_config)
    quiet = num_gpus > 1

    jobs = []
    # Pretrained condition
    for seed in make_seed_list(CONFIRMATION_SEEDS, offset=1000):
        jobs.append({
            "seed": seed,
            "ratio": best_ratio,
            "label": "pretrained",
            "eval_mode": "confirmation",
            "config_json": config_json,
            "quiet": quiet,
        })
    # Baseline condition
    for seed in make_seed_list(CONFIRMATION_SEEDS, offset=2000):
        jobs.append({
            "seed": seed,
            "ratio": 0.0,
            "label": "baseline",
            "eval_mode": "confirmation",
            "config_json": config_json,
            "quiet": quiet,
        })

    results = run_jobs(jobs, num_gpus)

    pretrained_results = [r for r in results if r.get("label") == "pretrained"]
    baseline_results   = [r for r in results if r.get("label") == "baseline"]

    return pretrained_results, baseline_results


# ================================================================== #
#  DECOUPLED SCREENING STAGE                                          #
# ================================================================== #
def run_decoupled_screening(
    base_config: dict,
    num_gpus: int,
) -> tuple[int, dict]:
    """Sweep over pretraining_episodes with fixed phase2_episodes.

    Returns (best_pretraining_episodes, all_results_by_condition).
    """
    # Read sweep values from config, fall back to defaults
    sweep_values = (
        base_config.get("screening", {})
        .get("pretraining_episodes_sweep", DECOUPLED_PRETRAINING_SWEEP)
    )
    seeds_per = (
        base_config.get("screening", {})
        .get("seeds_per_condition", DECOUPLED_SCREENING_SEEDS)
    )
    phase2_ep = base_config["training"]["phase2_episodes"]

    total_jobs = len(sweep_values) * seeds_per
    print(f"\n{'=' * 70}")
    print("  STAGE 1 : DECOUPLED SCREENING")
    print(f"  {len(sweep_values)} pretraining doses x {seeds_per} seeds"
          f" = {total_jobs} jobs"
          f"  |  GPUs: {max(num_gpus, 1)}")
    print(f"  phase2_episodes fixed at {phase2_ep}")
    print("=" * 70)

    config_json = json.dumps(base_config)
    quiet = num_gpus > 1

    jobs = []
    for pretrain_ep in sweep_values:
        for seed in make_seed_list(seeds_per):
            jobs.append({
                "seed": seed,
                "pretraining_episodes": pretrain_ep,
                "label": f"screening/pretrain{pretrain_ep}",
                "eval_mode": "screening",
                "config_json": config_json,
                "quiet": quiet,
            })

    results = run_jobs(jobs, num_gpus)

    # Group by pretraining_episodes
    all_results: dict[str, list[dict]] = {}
    for r in results:
        key = str(r["pretraining_episodes"])
        all_results.setdefault(key, []).append(r)

    # Select best pretraining_episodes by mean listener accuracy
    ep_means = {}
    for ep_str, ep_results in all_results.items():
        ep_means[int(ep_str)] = np.mean(
            [r["listener_accuracy"] for r in ep_results]
        )

    best_ep = max(ep_means, key=ep_means.get)

    print(f"\n{'=' * 70}")
    print("  DECOUPLED SCREENING SUMMARY")
    print(f"{'=' * 70}")
    for ep in sweep_values:
        marker = " <-- BEST" if ep == best_ep else ""
        print(f"  pretrain {ep:>6} ep  ->  mean listener accuracy = "
              f"{ep_means[ep]:.4f}{marker}")
    print(f"\n  Selected pretraining_episodes: {best_ep}")

    os.makedirs("results/screening", exist_ok=True)
    with open("results/screening/decoupled_screening_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return best_ep, all_results


# ================================================================== #
#  DECOUPLED CONFIRMATION STAGE                                       #
# ================================================================== #
def run_decoupled_confirmation(
    base_config: dict,
    best_pretrain_ep: int,
    num_gpus: int,
) -> tuple[list[dict], list[dict]]:
    """Confirmation: best pretraining dose vs baseline (0), 12 seeds each."""
    seeds_per = (
        base_config.get("confirmation", {})
        .get("seeds_per_condition", DECOUPLED_CONFIRMATION_SEEDS)
    )
    phase2_ep = base_config["training"]["phase2_episodes"]

    print(f"\n{'=' * 70}")
    print("  STAGE 2 : DECOUPLED CONFIRMATION")
    print(f"  2 conditions x {seeds_per} seeds"
          f" = {2 * seeds_per} jobs"
          f"  |  GPUs: {max(num_gpus, 1)}")
    print(f"  Pretrained : {best_pretrain_ep} pretraining episodes")
    print(f"  Baseline   : 0 pretraining episodes")
    print(f"  Both share : {phase2_ep} phase2 episodes")
    print("=" * 70)

    config_json = json.dumps(base_config)
    quiet = num_gpus > 1

    jobs = []
    # Pretrained condition
    for seed in make_seed_list(seeds_per, offset=1000):
        jobs.append({
            "seed": seed,
            "pretraining_episodes": best_pretrain_ep,
            "label": "pretrained",
            "eval_mode": "confirmation",
            "config_json": config_json,
            "quiet": quiet,
        })
    # Baseline condition (0 pretraining)
    for seed in make_seed_list(seeds_per, offset=2000):
        jobs.append({
            "seed": seed,
            "pretraining_episodes": 0,
            "label": "baseline",
            "eval_mode": "confirmation",
            "config_json": config_json,
            "quiet": quiet,
        })

    results = run_jobs(jobs, num_gpus)

    pretrained_results = [r for r in results if r.get("label") == "pretrained"]
    baseline_results   = [r for r in results if r.get("label") == "baseline"]

    return pretrained_results, baseline_results


# ================================================================== #
#  EXPORT                                                             #
# ================================================================== #
def export_results(
    screening_results: dict,
    pretrained_results: list[dict],
    baseline_results: list[dict],
    best_param: float | int,
    output_dir: str = "results",
    sweep_mode: str = "decoupled",
):
    """Write all results to CSV files for analysis."""
    print(f"\n{'=' * 70}")
    print("  EXPORTING RESULTS")
    print(f"{'=' * 70}")

    os.makedirs(output_dir, exist_ok=True)

    # screening_results.csv
    screening_rows = []
    for key_str, results in screening_results.items():
        for r in results:
            screening_rows.append(r)
    if screening_rows:
        _write_csv(os.path.join(output_dir, "screening_results.csv"),
                    screening_rows)
        print(f"  Wrote screening_results.csv  ({len(screening_rows)} rows)")

    # confirmation_results.csv
    all_confirm = pretrained_results + baseline_results
    _write_csv(os.path.join(output_dir, "confirmation_results.csv"), all_confirm)
    print(f"  Wrote confirmation_results.csv  ({len(all_confirm)} rows)")

    # pretrained_results.csv / baseline_results.csv
    _write_csv(os.path.join(output_dir, "pretrained_results.csv"),
               pretrained_results)
    _write_csv(os.path.join(output_dir, "baseline_results.csv"),
               baseline_results)
    print(f"  Wrote pretrained_results.csv  ({len(pretrained_results)} rows)")
    print(f"  Wrote baseline_results.csv  ({len(baseline_results)} rows)")

    # effect_size.csv
    effect = compute_effect_size(
        [r["listener_accuracy"] for r in pretrained_results],
        [r["listener_accuracy"] for r in baseline_results],
    )
    if sweep_mode == "decoupled":
        effect["best_pretraining_episodes"] = best_param
    else:
        effect["best_ratio"] = best_param
    _write_csv(os.path.join(output_dir, "effect_size.csv"), [effect])
    print(f"  Wrote effect_size.csv")

    # summary.json
    summary = {
        "sweep_mode": sweep_mode,
        "effect_size": effect,
        "pretrained": pretrained_results,
        "baseline": baseline_results,
        "screening": screening_results,
    }
    if sweep_mode == "decoupled":
        summary["best_pretraining_episodes"] = best_param
    else:
        summary["best_ratio"] = best_param
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Wrote summary.json")


def _write_csv(path: str, rows: list[dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ================================================================== #
#  MAIN                                                               #
# ================================================================== #
def _print_confirmation_summary(pretrained_results, baseline_results):
    """Print H1/H2/H3 confirmation summary shared by both sweep modes."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Run full experimental pipeline: "
                    "screening -> confirmation -> export"
    )
    parser.add_argument("--config", type=str,
                        default="configs/decoupled_sweep.yaml")
    parser.add_argument("--sweep-mode", type=str, default="decoupled",
                        choices=["decoupled", "legacy"],
                        help="Sweep mode: 'decoupled' sweeps over "
                             "pretraining_episodes with fixed phase2_episodes; "
                             "'legacy' sweeps over pretraining_ratio. "
                             "Default: decoupled.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: 'cpu' or 'cuda'")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Number of GPUs to parallelize across. "
                             "0 = auto-detect all, 1 = sequential, "
                             "N = use N GPUs. Default: 1 for cpu, "
                             "auto-detect for cuda.")
    parser.add_argument("--total-episodes", type=int, default=None,
                        help="Override total_episodes (legacy mode)")
    parser.add_argument("--phase2-episodes", type=int, default=None,
                        help="Override phase2_episodes (decoupled mode)")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--skip-screening", action="store_true",
                        help="Skip screening, use --best-ratio/--best-pretrain "
                             "instead")
    parser.add_argument("--best-ratio", type=float, default=None,
                        help="Manually specify best ratio (legacy mode)")
    parser.add_argument("--best-pretrain", type=int, default=None,
                        help="Manually specify best pretraining_episodes "
                             "(decoupled mode)")
    args = parser.parse_args()

    config = load_config(args.config)
    config["experiment"]["device"] = args.device
    if args.total_episodes is not None:
        config["training"]["total_episodes"] = args.total_episodes
    if args.phase2_episodes is not None:
        config["training"]["phase2_episodes"] = args.phase2_episodes

    # Ensure decoupled keys exist when using decoupled sweep mode.
    # If the loaded config only has legacy keys (total_episodes +
    # pretraining_ratio), derive sensible decoupled defaults so users
    # don't have to rewrite their config file.
    if args.sweep_mode == "decoupled":
        training = config["training"]
        if "phase2_episodes" not in training:
            total = training.get("total_episodes", 50000)
            ratio = training.get("pretraining_ratio", 0.25)
            training["phase2_episodes"] = int(total * (1 - ratio))
            print(f"  NOTE: phase2_episodes not in config; derived "
                  f"{training['phase2_episodes']} from total_episodes={total}"
                  f" * (1 - pretraining_ratio={ratio})")
        if "pretraining_episodes" not in training:
            total = training.get("total_episodes", 50000)
            ratio = training.get("pretraining_ratio", 0.25)
            training["pretraining_episodes"] = int(total * ratio)

    # Determine GPU count
    if args.device == "cpu":
        num_gpus = 0
    elif args.num_gpus is None or args.num_gpus == 0:
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = args.num_gpus

    print("=" * 70)
    print("  COOPERATIVE PRETRAINING EXPERIMENT PIPELINE")
    print("=" * 70)
    print(f"  Sweep mode      : {args.sweep_mode}")
    print(f"  Config          : {args.config}")
    print(f"  Device          : {args.device}")
    print(f"  GPUs            : {max(num_gpus, 1)}"
          f"{'  (parallel)' if num_gpus > 1 else '  (sequential)'}")
    print(f"  Eval episodes   : {EVAL_EPISODES}")
    print(f"  CIC episodes    : {CIC_EPISODES}")

    if args.sweep_mode == "decoupled":
        _run_decoupled_pipeline(config, num_gpus, args)
    else:
        _run_legacy_pipeline(config, num_gpus, args)

    print(f"\n{'=' * 70}")
    print("  PIPELINE COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Results exported to: {args.output_dir}/")
    print(f"  Run analysis: python analysis/analyze_results.py")


def _run_decoupled_pipeline(config, num_gpus, args):
    """Run the decoupled (pretraining_episodes) sweep pipeline."""
    sweep_values = (
        config.get("screening", {})
        .get("pretraining_episodes_sweep", DECOUPLED_PRETRAINING_SWEEP)
    )
    screen_seeds = (
        config.get("screening", {})
        .get("seeds_per_condition", DECOUPLED_SCREENING_SEEDS)
    )
    confirm_seeds = (
        config.get("confirmation", {})
        .get("seeds_per_condition", DECOUPLED_CONFIRMATION_SEEDS)
    )
    phase2_ep = config["training"]["phase2_episodes"]

    print(f"  phase2_episodes : {phase2_ep}")
    print(f"  Screening sweep : {sweep_values}")
    print(f"  Screening jobs  : {len(sweep_values) * screen_seeds}"
          f"  ({len(sweep_values)} doses x {screen_seeds} seeds)")
    print(f"  Confirm jobs    : {2 * confirm_seeds}"
          f"  (2 conditions x {confirm_seeds} seeds)")
    if num_gpus > 1:
        print(f"\n  Multi-GPU: jobs distributed across {num_gpus} GPUs")

    # Stage 1: Screening
    if args.skip_screening and args.best_pretrain is not None:
        best_ep = args.best_pretrain
        screening_results = {}
        print(f"\n  Skipping screening, using pretraining_episodes={best_ep}")
    else:
        best_ep, screening_results = run_decoupled_screening(config, num_gpus)

    # Stage 2: Confirmation
    pretrained_results, baseline_results = run_decoupled_confirmation(
        config, best_ep, num_gpus
    )

    _print_confirmation_summary(pretrained_results, baseline_results)

    # Stage 3: Export
    export_results(
        screening_results, pretrained_results, baseline_results,
        best_ep, output_dir=args.output_dir, sweep_mode="decoupled",
    )


def _run_legacy_pipeline(config, num_gpus, args):
    """Run the legacy (pretraining_ratio) sweep pipeline."""
    total_ep = config["training"].get("total_episodes", 50000)
    print(f"  Total episodes  : {total_ep}")
    print(f"  Screening jobs  : {len(SCREENING_RATIOS) * SCREENING_SEEDS}"
          f"  ({len(SCREENING_RATIOS)} ratios x {SCREENING_SEEDS} seeds)")
    print(f"  Confirm jobs    : {2 * CONFIRMATION_SEEDS}"
          f"  (2 conditions x {CONFIRMATION_SEEDS} seeds)")
    if num_gpus > 1:
        print(f"\n  Multi-GPU: jobs distributed across {num_gpus} GPUs")

    # Stage 1: Screening
    if args.skip_screening and args.best_ratio is not None:
        best_ratio = args.best_ratio
        screening_results = {}
        print(f"\n  Skipping screening, using ratio={best_ratio}")
    else:
        best_ratio, screening_results = run_screening(config, num_gpus)

    # Stage 2: Confirmation
    pretrained_results, baseline_results = run_confirmation(
        config, best_ratio, num_gpus
    )

    _print_confirmation_summary(pretrained_results, baseline_results)

    # Stage 3: Export
    export_results(
        screening_results, pretrained_results, baseline_results,
        best_ratio, output_dir=args.output_dir, sweep_mode="legacy",
    )


if __name__ == "__main__":
    main()
