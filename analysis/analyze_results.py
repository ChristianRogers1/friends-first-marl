#!/usr/bin/env python3
"""
Matplotlib analysis for:
  "Cooperative Pretraining for Robust Emergent Communication
   Under Adversarial Pressure"

Reads CSV files produced by run_full_pipeline.py and generates:
  Figure 1 -- Screening: listener accuracy vs pretraining ratio
  Figure 2 -- H1: Task performance comparison (pretrained vs baseline)
  Figure 3 -- H2: Communication robustness (CIC comparison)
  Figure 4 -- H3: Information leakage comparison
  Figure 5 -- Combined dashboard of all three hypotheses

Usage:
    python analysis/analyze_results.py
    python analysis/analyze_results.py --results-dir results --figures-dir results/figures
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for HPC
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #
def load_csv(path: str) -> list[dict]:
    """Load a CSV file into a list of dicts, casting numeric strings."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            rows.append(parsed)
    return rows


def col(rows: list[dict], key: str) -> np.ndarray:
    """Extract a numeric column from a list-of-dicts."""
    return np.array([r[key] for r in rows])


def welch_ttest(a: np.ndarray, b: np.ndarray):
    """Welch's t-test returning (t, p, df)."""
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    # Welch-Satterthwaite degrees of freedom
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    num = (va / na + vb / nb) ** 2
    den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    df = num / den if den > 0 else na + nb - 2
    return t_stat, p_val, df


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d (pooled std)."""
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                     / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0


# ------------------------------------------------------------------ #
#  Colors                                                             #
# ------------------------------------------------------------------ #
CLR_PRE  = "#3366CC"
CLR_BASE = "#DA4427"
CLR_SCR  = "#4DAF4A"


# ================================================================== #
#  FIGURE 1 -- SCREENING                                              #
# ================================================================== #
def fig_screening(screen_rows: list[dict], figures_dir: str):
    """Listener accuracy and adversary accuracy vs pretraining ratio."""
    ratios = sorted(set(r["ratio"] for r in screen_rows))
    mean_acc, std_acc = [], []
    mean_adv, std_adv = [], []

    for ratio in ratios:
        accs = [r["listener_accuracy"] for r in screen_rows if r["ratio"] == ratio]
        advs = [r["adversary_accuracy"] for r in screen_rows if r["ratio"] == ratio]
        mean_acc.append(np.mean(accs));  std_acc.append(np.std(accs))
        mean_adv.append(np.mean(advs));  std_adv.append(np.std(advs))

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    ax1.errorbar(ratios, mean_acc, yerr=std_acc, fmt="-o",
                 color=CLR_PRE, markerfacecolor=CLR_PRE, capsize=8,
                 linewidth=1.8, markersize=8, label="Listener Accuracy")
    ax1.set_xlabel(r"Pretraining Ratio ($\alpha / T$)")
    ax1.set_ylabel("Listener Accuracy", color=CLR_PRE)
    ax1.set_ylim(0, 1)
    ax1.set_xticks(ratios)
    ax1.tick_params(axis="y", labelcolor=CLR_PRE)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.errorbar(ratios, mean_adv, yerr=std_adv, fmt="-s",
                 color=CLR_BASE, markerfacecolor=CLR_BASE, capsize=8,
                 linewidth=1.8, markersize=8, label="Adversary Accuracy")
    ax2.set_ylabel("Adversary Accuracy", color=CLR_BASE)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis="y", labelcolor=CLR_BASE)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.suptitle("Screening Stage: Performance vs Pretraining Ratio", fontsize=13)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figures_dir, f"fig1_screening.{ext}"), dpi=200)
    plt.close(fig)
    print("  Saved fig1_screening.{png,pdf}")


# ================================================================== #
#  FIGURE 2 -- H1: TASK PERFORMANCE                                   #
# ================================================================== #
def fig_h1(pre_rows: list[dict], base_rows: list[dict], figures_dir: str):
    """Bar chart with error bars and individual seed points."""
    pre_acc  = col(pre_rows, "listener_accuracy")
    base_acc = col(base_rows, "listener_accuracy")

    t, p, df = welch_ttest(pre_acc, base_acc)
    d = cohens_d(pre_acc, base_acc)

    means = [pre_acc.mean(), base_acc.mean()]
    sems  = [pre_acc.std() / np.sqrt(len(pre_acc)),
             base_acc.std() / np.sqrt(len(base_acc))]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    bars = ax.bar([0, 1], means, width=0.45, color=[CLR_PRE, CLR_BASE],
                  edgecolor="black", linewidth=0.7, zorder=3)
    ax.errorbar([0, 1], means, yerr=sems, fmt="none", ecolor="black",
                capsize=10, linewidth=1.5, zorder=4)

    rng = np.random.default_rng(0)
    ax.scatter(0 + 0.06 * rng.standard_normal(len(pre_acc)),  pre_acc,
               s=35, color=CLR_PRE, alpha=0.5, edgecolors="none", zorder=5)
    ax.scatter(1 + 0.06 * rng.standard_normal(len(base_acc)), base_acc,
               s=35, color=CLR_BASE, alpha=0.5, edgecolors="none", zorder=5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pretrained", "Baseline"])
    ax.set_ylabel("Listener Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("H1: Task Performance")
    ax.grid(axis="y", alpha=0.3)

    ypos = max(means) + max(sems) + 0.06
    ax.text(0.5, ypos, f"Cohen's d = {d:.3f}\np = {p:.4f}",
            ha="center", fontsize=10, transform=ax.get_xaxis_transform())

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figures_dir, f"fig2_h1_task_performance.{ext}"), dpi=200)
    plt.close(fig)
    print(f"  Saved fig2_h1_task_performance.{{png,pdf}}")
    print(f"  Welch t({df:.1f}) = {t:.3f}, p = {p:.4f}, Cohen's d = {d:.3f}")


# ================================================================== #
#  FIGURE 3 -- H2: CIC                                                #
# ================================================================== #
def fig_h2(pre_rows: list[dict], base_rows: list[dict], figures_dir: str):
    """Box plot with individual points for CIC."""
    pre_cic  = col(pre_rows, "mean_cic")
    base_cic = col(base_rows, "mean_cic")

    t, p, df = welch_ttest(pre_cic, base_cic)

    fig, ax = plt.subplots(figsize=(5.5, 5))

    bp = ax.boxplot([pre_cic, base_cic], positions=[0, 1], widths=0.35,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.5))
    bp["boxes"][0].set(facecolor=CLR_PRE, alpha=0.4)
    bp["boxes"][1].set(facecolor=CLR_BASE, alpha=0.4)

    rng = np.random.default_rng(1)
    ax.scatter(0 + 0.04 * rng.standard_normal(len(pre_cic)),  pre_cic,
               s=35, color=CLR_PRE, alpha=0.6, edgecolors="none", zorder=5)
    ax.scatter(1 + 0.04 * rng.standard_normal(len(base_cic)), base_cic,
               s=35, color=CLR_BASE, alpha=0.6, edgecolors="none", zorder=5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pretrained", "Baseline"])
    ax.set_ylabel("Causal Influence of Communication (CIC)")
    ax.set_title("H2: Communication Robustness")
    ax.grid(axis="y", alpha=0.3)

    ymax = max(pre_cic.max(), base_cic.max())
    ax.text(0.5, ymax * 1.08, f"p = {p:.4f}",
            ha="center", fontsize=10, transform=ax.get_xaxis_transform())

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figures_dir, f"fig3_h2_cic.{ext}"), dpi=200)
    plt.close(fig)
    print(f"  Saved fig3_h2_cic.{{png,pdf}}")
    print(f"  Welch t({df:.1f}) = {t:.3f}, p = {p:.4f}")


# ================================================================== #
#  FIGURE 4 -- H3: INFORMATION LEAKAGE                                #
# ================================================================== #
def fig_h3(pre_rows: list[dict], base_rows: list[dict], figures_dir: str):
    """Two-panel: adversary accuracy + leakage ratio."""
    pre_adv   = col(pre_rows, "adversary_accuracy")
    base_adv  = col(base_rows, "adversary_accuracy")
    pre_leak  = col(pre_rows, "leakage_ratio")
    base_leak = col(base_rows, "leakage_ratio")

    t_adv, p_adv, df_adv    = welch_ttest(pre_adv, base_adv)
    t_leak, p_leak, df_leak = welch_ttest(pre_leak, base_leak)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    rng = np.random.default_rng(2)

    # Panel A: adversary accuracy
    for ax, pre_data, base_data, ylabel, title_str, p_val in [
        (ax1, pre_adv, base_adv, "Adversary Accuracy",
         "Eavesdropper Accuracy", p_adv),
        (ax2, pre_leak, base_leak, "Leakage Ratio\n(Adv. Acc. / Listener Acc.)",
         "Information Leakage Ratio", p_leak),
    ]:
        means = [pre_data.mean(), base_data.mean()]
        sems  = [pre_data.std() / np.sqrt(len(pre_data)),
                 base_data.std() / np.sqrt(len(base_data))]

        ax.bar([0, 1], means, width=0.45, color=[CLR_PRE, CLR_BASE],
               edgecolor="black", linewidth=0.7, zorder=3)
        ax.errorbar([0, 1], means, yerr=sems, fmt="none", ecolor="black",
                    capsize=10, linewidth=1.5, zorder=4)
        ax.scatter(0 + 0.06 * rng.standard_normal(len(pre_data)),  pre_data,
                   s=30, color=CLR_PRE, alpha=0.5, edgecolors="none", zorder=5)
        ax.scatter(1 + 0.06 * rng.standard_normal(len(base_data)), base_data,
                   s=30, color=CLR_BASE, alpha=0.5, edgecolors="none", zorder=5)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pretrained", "Baseline"])
        ax.set_ylabel(ylabel)
        ax.set_title(title_str)
        ax.grid(axis="y", alpha=0.3)
        ypos = max(means) + max(sems) + 0.05
        ax.text(0.5, ypos, f"p = {p_val:.4f}",
                ha="center", fontsize=10, transform=ax.get_xaxis_transform())

    fig.suptitle("H3: Information Leakage", fontsize=13)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figures_dir, f"fig4_h3_leakage.{ext}"), dpi=200)
    plt.close(fig)
    print(f"  Saved fig4_h3_leakage.{{png,pdf}}")
    print(f"  Adv. acc:  Welch t({df_adv:.1f})  = {t_adv:.3f}, p = {p_adv:.4f}")
    print(f"  Leak ratio: Welch t({df_leak:.1f}) = {t_leak:.3f}, p = {p_leak:.4f}")


# ================================================================== #
#  FIGURE 5 -- COMBINED DASHBOARD                                     #
# ================================================================== #
def fig_dashboard(pre_rows: list[dict], base_rows: list[dict], figures_dir: str):
    """Three-panel summary: H1, H2, H3."""
    pre_acc   = col(pre_rows, "listener_accuracy")
    base_acc  = col(base_rows, "listener_accuracy")
    pre_cic   = col(pre_rows, "mean_cic")
    base_cic  = col(base_rows, "mean_cic")
    pre_leak  = col(pre_rows, "leakage_ratio")
    base_leak = col(base_rows, "leakage_ratio")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    datasets = [
        (pre_acc,  base_acc,  "Listener Accuracy", "H1: Task Performance"),
        (pre_cic,  base_cic,  "CIC",               "H2: Comm. Robustness"),
        (pre_leak, base_leak, "Leakage Ratio",     "H3: Info. Leakage"),
    ]

    for ax, (pre_data, base_data, ylabel, title_str) in zip(axes, datasets):
        means = [pre_data.mean(), base_data.mean()]
        sems  = [pre_data.std() / np.sqrt(len(pre_data)),
                 base_data.std() / np.sqrt(len(base_data))]
        ax.bar([0, 1], means, width=0.45, color=[CLR_PRE, CLR_BASE],
               edgecolor="black", linewidth=0.7, zorder=3)
        ax.errorbar([0, 1], means, yerr=sems, fmt="none", ecolor="black",
                    capsize=10, linewidth=1.5, zorder=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pre", "Base"])
        ax.set_ylabel(ylabel)
        ax.set_title(title_str)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Cooperative Pretraining: Pretrained vs Baseline", fontsize=14)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figures_dir, f"fig5_dashboard.{ext}"), dpi=200)
    plt.close(fig)
    print("  Saved fig5_dashboard.{png,pdf}")


# ================================================================== #
#  PRINT SUMMARY                                                      #
# ================================================================== #
def print_summary(pre_rows: list[dict], base_rows: list[dict]):
    pre_acc  = col(pre_rows, "listener_accuracy")
    base_acc = col(base_rows, "listener_accuracy")
    pre_cic  = col(pre_rows, "mean_cic")
    base_cic = col(base_rows, "mean_cic")
    pre_adv  = col(pre_rows, "adversary_accuracy")
    base_adv = col(base_rows, "adversary_accuracy")
    pre_leak = col(pre_rows, "leakage_ratio")
    base_leak = col(base_rows, "leakage_ratio")

    t1, p1, df1 = welch_ttest(pre_acc, base_acc)
    t2, p2, df2 = welch_ttest(pre_cic, base_cic)
    t3a, p3a, df3a = welch_ttest(pre_adv, base_adv)
    t3b, p3b, df3b = welch_ttest(pre_leak, base_leak)
    d = cohens_d(pre_acc, base_acc)

    print()
    print("=" * 60)
    print("  STATISTICAL SUMMARY")
    print("=" * 60)
    print()
    print("  H1 -- Task Performance (Listener Accuracy)")
    print(f"    Pretrained : {pre_acc.mean():.4f} +/- {pre_acc.std():.4f}  (n={len(pre_acc)})")
    print(f"    Baseline   : {base_acc.mean():.4f} +/- {base_acc.std():.4f}  (n={len(base_acc)})")
    print(f"    Welch t({df1:.1f}) = {t1:.3f}, p = {p1:.4f}")
    print(f"    Cohen's d  = {d:.4f}")
    print()
    print("  H2 -- Communication Robustness (CIC)")
    print(f"    Pretrained : {pre_cic.mean():.4f} +/- {pre_cic.std():.4f}")
    print(f"    Baseline   : {base_cic.mean():.4f} +/- {base_cic.std():.4f}")
    print(f"    Welch t({df2:.1f}) = {t2:.3f}, p = {p2:.4f}")
    print()
    print("  H3 -- Information Leakage")
    print(f"    Adversary accuracy:")
    print(f"      Pretrained : {pre_adv.mean():.4f} +/- {pre_adv.std():.4f}")
    print(f"      Baseline   : {base_adv.mean():.4f} +/- {base_adv.std():.4f}")
    print(f"      Welch t({df3a:.1f}) = {t3a:.3f}, p = {p3a:.4f}")
    print(f"    Leakage ratio:")
    print(f"      Pretrained : {pre_leak.mean():.4f} +/- {pre_leak.std():.4f}")
    print(f"      Baseline   : {base_leak.mean():.4f} +/- {base_leak.std():.4f}")
    print(f"      Welch t({df3b:.1f}) = {t3b:.3f}, p = {p3b:.4f}")
    print()
    print("=" * 60)


# ================================================================== #
#  MAIN                                                               #
# ================================================================== #
def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--figures-dir", type=str, default=None)
    args = parser.parse_args()

    results_dir = args.results_dir
    figures_dir = args.figures_dir or os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print(f"Loading results from {results_dir}/ ...")

    # Load required files
    pre_path  = os.path.join(results_dir, "pretrained_results.csv")
    base_path = os.path.join(results_dir, "baseline_results.csv")
    scr_path  = os.path.join(results_dir, "screening_results.csv")

    if not os.path.isfile(pre_path) or not os.path.isfile(base_path):
        print(f"ERROR: Missing pretrained_results.csv or baseline_results.csv "
              f"in {results_dir}/")
        sys.exit(1)

    pre_rows  = load_csv(pre_path)
    base_rows = load_csv(base_path)
    print(f"  pretrained_results.csv : {len(pre_rows)} rows")
    print(f"  baseline_results.csv   : {len(base_rows)} rows")

    # Screening (optional)
    if os.path.isfile(scr_path):
        scr_rows = load_csv(scr_path)
        print(f"  screening_results.csv  : {len(scr_rows)} rows")
        fig_screening(scr_rows, figures_dir)
    else:
        print("  screening_results.csv  : not found, skipping Figure 1")

    # Generate figures
    fig_h1(pre_rows, base_rows, figures_dir)
    fig_h2(pre_rows, base_rows, figures_dir)
    fig_h3(pre_rows, base_rows, figures_dir)
    fig_dashboard(pre_rows, base_rows, figures_dir)

    # Print statistical summary
    print_summary(pre_rows, base_rows)

    print(f"\n  Figures saved to: {figures_dir}/")


if __name__ == "__main__":
    main()
