#!/usr/bin/env python3
"""
Export experiment results from JSON to CSV for analysis.

Reads the summary.json produced by run_full_pipeline.py and writes
individual CSV files that can be loaded by the analysis script
or any other tool (pandas, matplotlib, etc.).

Usage:
    python scripts/export_to_csv.py --input results/summary.json
    python scripts/export_to_csv.py --input results/summary.json --output-dir results
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def write_csv(path: str, rows: list[dict]):
    """Write a list of dicts to CSV."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Export results to CSV for MATLAB")
    parser.add_argument("--input", type=str, default="results/summary.json")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    with open(args.input) as f:
        summary = json.load(f)

    output_dir = args.output_dir or os.path.dirname(args.input)
    os.makedirs(output_dir, exist_ok=True)

    # screening_results.csv
    screening_rows = []
    for ratio_str, results in summary.get("screening", {}).items():
        for r in results:
            screening_rows.append(r)
    if screening_rows:
        path = os.path.join(output_dir, "screening_results.csv")
        write_csv(path, screening_rows)
        print(f"Wrote {path}  ({len(screening_rows)} rows)")

    # pretrained_results.csv
    pretrained = summary.get("pretrained", [])
    if pretrained:
        path = os.path.join(output_dir, "pretrained_results.csv")
        write_csv(path, pretrained)
        print(f"Wrote {path}  ({len(pretrained)} rows)")

    # baseline_results.csv
    baseline = summary.get("baseline", [])
    if baseline:
        path = os.path.join(output_dir, "baseline_results.csv")
        write_csv(path, baseline)
        print(f"Wrote {path}  ({len(baseline)} rows)")

    # confirmation_results.csv  (pretrained + baseline combined)
    if pretrained or baseline:
        path = os.path.join(output_dir, "confirmation_results.csv")
        write_csv(path, pretrained + baseline)
        print(f"Wrote {path}  ({len(pretrained) + len(baseline)} rows)")

    # effect_size.csv
    effect = summary.get("effect_size", {})
    if effect:
        path = os.path.join(output_dir, "effect_size.csv")
        write_csv(path, [effect])
        print(f"Wrote {path}")

    # training_curves.csv -- flatten training histories if present
    # (These come from the per-seed history JSON files)
    history_dir = os.path.join(output_dir, "confirmation")
    if os.path.isdir(history_dir):
        curve_rows = []
        for label_dir in ["pretrained", "baseline"]:
            hdir = os.path.join(history_dir, label_dir)
            if not os.path.isdir(hdir):
                continue
            for fname in sorted(os.listdir(hdir)):
                if fname.startswith("history_") and fname.endswith(".json"):
                    with open(os.path.join(hdir, fname)) as hf:
                        hist = json.load(hf)
                    seed_str = fname.split("seed")[1].split("_")[0]
                    for ep_idx in range(len(hist.get("listener_accuracy", []))):
                        curve_rows.append({
                            "condition": label_dir,
                            "seed": seed_str,
                            "episode": ep_idx,
                            "listener_accuracy": hist["listener_accuracy"][ep_idx],
                            "adversary_accuracy": hist["adversary_accuracy"][ep_idx],
                            "speaker_reward": hist["speaker_reward"][ep_idx],
                            "phase": hist["phase"][ep_idx],
                        })
        if curve_rows:
            path = os.path.join(output_dir, "training_curves.csv")
            write_csv(path, curve_rows)
            print(f"Wrote {path}  ({len(curve_rows)} rows)")

    print("\nDone. Load in MATLAB with:")
    print("  T = readtable('results/confirmation_results.csv');")


if __name__ == "__main__":
    main()
