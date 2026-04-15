#!/usr/bin/env python3
"""
Attractor analysis for curriculum learning in emergent communication.

Hypothesis: Curriculum learning (cooperative pretraining) may decrease or
remove the prevalence of a "communicative" attractor — where the listener
performs above chance and symbols carry causal influence — leaving seeds
more likely to converge to a "jabbering" attractor where messages are
meaningless noise.

This script generates:
  Figure 6  -- CIC vs Listener Accuracy scatter (attractor identification)
  Figure 7  -- Attractor prevalence comparison (bar chart)
  Figure 8  -- CIC distribution comparison (strip + violin)
  Figure 9  -- Long-term listener accuracy trajectories (pretrained seeds)
  Figure 10 -- Long-term listener accuracy trajectories (baseline seeds)
  Figure 11 -- Combined trajectory comparison panel
  Figure 12 -- Long-term adversary accuracy trajectories (both conditions)

Usage:
    python analysis/attractor_analysis.py
    python analysis/attractor_analysis.py --results-dir results --runs-dir runs
"""

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# TensorBoard event reading
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ------------------------------------------------------------------ #
#  Constants                                                          #
# ------------------------------------------------------------------ #
CLR_PRE = "#3366CC"
CLR_BASE = "#DA4427"
CLR_JABBER = "#888888"
CLR_COMM = "#2CA02C"

# CIC threshold: seeds with CIC below this are classified as "jabbering"
CIC_THRESHOLD = 1.0

# Chance level for 3-choice task
CHANCE_LEVEL = 1.0 / 3.0

# Smoothing window for training curves (number of eval points)
SMOOTH_WINDOW = 5


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
    return np.array([r[key] for r in rows])


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Simple centered rolling mean, handling edges with shrinking window."""
    if window <= 1:
        return values
    out = np.empty_like(values, dtype=float)
    half = window // 2
    n = len(values)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.mean(values[lo:hi])
    return out


def read_eval_curves(event_dir: str) -> dict:
    """Read eval/listener_accuracy and eval/adversary_accuracy from TensorBoard.

    Returns dict with keys 'steps', 'listener_accuracy', 'adversary_accuracy',
    and 'phase2_start' (episode where phase 2 begins).
    """
    ea = EventAccumulator(event_dir)
    ea.Reload()

    listener_events = ea.Scalars("eval/listener_accuracy")
    adversary_events = ea.Scalars("eval/adversary_accuracy")

    steps_l = np.array([e.step for e in listener_events])
    acc_l = np.array([e.value for e in listener_events])
    acc_a = np.array([e.value for e in adversary_events])

    # Find phase 2 start from train/phase
    phase_events = ea.Scalars("train/phase")
    phase2_start = 0
    for e in phase_events:
        if e.value >= 2.0:
            phase2_start = e.step
            break

    return {
        "steps": steps_l,
        "listener_accuracy": acc_l,
        "adversary_accuracy": acc_a,
        "phase2_start": phase2_start,
    }


def classify_attractor(cic: float) -> str:
    """Classify a seed as 'jabbering' or 'communicative' based on CIC."""
    return "jabbering" if cic < CIC_THRESHOLD else "communicative"


# ------------------------------------------------------------------ #
#  FIGURE 6 -- CIC vs Listener Accuracy Scatter                       #
# ------------------------------------------------------------------ #
def fig_attractor_scatter(pre_rows, base_rows, figures_dir):
    """Scatter plot of CIC vs listener accuracy, revealing attractor structure."""
    pre_acc = col(pre_rows, "listener_accuracy")
    pre_cic = col(pre_rows, "mean_cic")
    base_acc = col(base_rows, "listener_accuracy")
    base_cic = col(base_rows, "mean_cic")

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(pre_acc, pre_cic, s=80, color=CLR_PRE, alpha=0.7,
               edgecolors="black", linewidth=0.5, zorder=5,
               label="Pretrained (curriculum)")
    ax.scatter(base_acc, base_cic, s=80, color=CLR_BASE, alpha=0.7,
               edgecolors="black", linewidth=0.5, zorder=5, marker="^",
               label="Baseline (no curriculum)")

    # Chance level vertical line
    ax.axvline(CHANCE_LEVEL, color="gray", linestyle="--", alpha=0.5,
               linewidth=1, label=f"Chance ({CHANCE_LEVEL:.3f})")

    # CIC threshold horizontal line
    ax.axhline(CIC_THRESHOLD, color="gray", linestyle=":", alpha=0.5,
               linewidth=1, label=f"CIC threshold ({CIC_THRESHOLD})")

    # Annotate attractor regions
    ax.annotate("Jabbering\nAttractor",
                xy=(0.35, -1.2), fontsize=10, fontstyle="italic",
                color=CLR_JABBER, ha="center", weight="bold")
    ax.annotate("Communicative\nAttractor",
                xy=(0.48, 10.5), fontsize=10, fontstyle="italic",
                color=CLR_COMM, ha="center", weight="bold")

    ax.set_xlabel("Listener Accuracy", fontsize=12)
    ax.set_ylabel("Causal Influence of Communication (CIC)", fontsize=12)
    ax.set_title("Attractor Identification:\nCIC vs. Listener Accuracy by Seed",
                 fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0.28, 0.58)
    ax.set_ylim(-2.5, None)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figures_dir, f"fig6_attractor_scatter.{ext}"),
                    dpi=200)
    plt.close(fig)
    print("  Saved fig6_attractor_scatter.{png,pdf}")


# ------------------------------------------------------------------ #
#  FIGURE 7 -- Attractor Prevalence                                    #
# ------------------------------------------------------------------ #
def fig_attractor_prevalence(pre_rows, base_rows, figures_dir):
    """Bar chart showing proportion of seeds in each attractor by condition."""
    pre_cic = col(pre_rows, "mean_cic")
    base_cic = col(base_rows, "mean_cic")

    pre_jabber = np.sum(pre_cic < CIC_THRESHOLD)
    pre_comm = np.sum(pre_cic >= CIC_THRESHOLD)
    base_jabber = np.sum(base_cic < CIC_THRESHOLD)
    base_comm = np.sum(base_cic >= CIC_THRESHOLD)

    n_pre = len(pre_cic)
    n_base = len(base_cic)

    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.array([0, 1])
    width = 0.35

    bars_comm = ax.bar(x - width / 2, [pre_comm / n_pre, base_comm / n_base],
                       width, color=CLR_COMM, alpha=0.8, edgecolor="black",
                       linewidth=0.7, label="Communicative (CIC > 1)", zorder=3)
    bars_jab = ax.bar(x + width / 2, [pre_jabber / n_pre, base_jabber / n_base],
                      width, color=CLR_JABBER, alpha=0.8, edgecolor="black",
                      linewidth=0.7, label="Jabbering (CIC \u2248 0)", zorder=3)

    # Add count annotations
    for bar, count, total in zip(bars_comm, [pre_comm, base_comm],
                                 [n_pre, n_base]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{count}/{total}", ha="center", fontsize=11, weight="bold")
    for bar, count, total in zip(bars_jab, [pre_jabber, base_jabber],
                                 [n_pre, n_base]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{count}/{total}", ha="center", fontsize=11, weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["Pretrained\n(curriculum)", "Baseline\n(no curriculum)"],
                       fontsize=11)
    ax.set_ylabel("Proportion of Seeds", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title("Attractor Prevalence: Curriculum vs. Baseline", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figures_dir, f"fig7_attractor_prevalence.{ext}"),
                    dpi=200)
    plt.close(fig)
    print("  Saved fig7_attractor_prevalence.{png,pdf}")

    print(f"  Pretrained: {pre_jabber}/{n_pre} jabbering, "
          f"{pre_comm}/{n_pre} communicative")
    print(f"  Baseline:   {base_jabber}/{n_base} jabbering, "
          f"{base_comm}/{n_base} communicative")


# ------------------------------------------------------------------ #
#  FIGURE 8 -- CIC Distribution                                       #
# ------------------------------------------------------------------ #
def fig_cic_distribution(pre_rows, base_rows, figures_dir):
    """Strip plot with violin overlay showing CIC distribution per condition."""
    pre_cic = col(pre_rows, "mean_cic")
    base_cic = col(base_rows, "mean_cic")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    # Violin plots
    parts = ax.violinplot([pre_cic, base_cic], positions=[0, 1], widths=0.6,
                          showmeans=True, showmedians=True, showextrema=False)
    parts["bodies"][0].set_facecolor(CLR_PRE)
    parts["bodies"][0].set_alpha(0.25)
    parts["bodies"][1].set_facecolor(CLR_BASE)
    parts["bodies"][1].set_alpha(0.25)
    parts["cmeans"].set_color("black")
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linestyle("--")

    # Strip plot (individual points)
    rng = np.random.default_rng(42)
    jitter_pre = 0.06 * rng.standard_normal(len(pre_cic))
    jitter_base = 0.06 * rng.standard_normal(len(base_cic))

    # Color pretrained points by attractor type
    for i, (cic, jit) in enumerate(zip(pre_cic, jitter_pre)):
        color = CLR_JABBER if cic < CIC_THRESHOLD else CLR_PRE
        ax.scatter(0 + jit, cic, s=50, color=color, alpha=0.8,
                   edgecolors="black", linewidth=0.5, zorder=5)

    ax.scatter(1 + jitter_base, base_cic, s=50, color=CLR_BASE, alpha=0.7,
               edgecolors="black", linewidth=0.5, zorder=5)

    # CIC threshold line
    ax.axhline(CIC_THRESHOLD, color="gray", linestyle=":", alpha=0.5,
               linewidth=1)
    ax.text(1.55, CIC_THRESHOLD + 0.3, f"CIC = {CIC_THRESHOLD}",
            fontsize=9, color="gray")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pretrained\n(curriculum)", "Baseline\n(no curriculum)"],
                       fontsize=11)
    ax.set_ylabel("Causal Influence of Communication (CIC)", fontsize=12)
    ax.set_title("CIC Distribution: Bimodality in Curriculum Learning",
                 fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    # Custom legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CLR_PRE,
               markersize=8, markeredgecolor="black", markeredgewidth=0.5,
               label="Communicative seed"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CLR_JABBER,
               markersize=8, markeredgecolor="black", markeredgewidth=0.5,
               label="Jabbering seed (CIC \u2248 0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CLR_BASE,
               markersize=8, markeredgecolor="black", markeredgewidth=0.5,
               label="Baseline seed"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figures_dir, f"fig8_cic_distribution.{ext}"),
                    dpi=200)
    plt.close(fig)
    print("  Saved fig8_cic_distribution.{png,pdf}")


# ------------------------------------------------------------------ #
#  FIGURE 9 -- Pretrained Listener Accuracy Trajectories               #
# ------------------------------------------------------------------ #
def fig_pretrained_trajectories(pre_rows, runs_dir, figures_dir):
    """Per-seed listener accuracy over training for pretrained condition.

    Lines colored by final attractor state (communicative vs jabbering).
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))

    legend_handles = []
    added_jabber = False
    added_comm = False

    for row in pre_rows:
        seed = int(row["seed"])
        ratio = row["ratio"]
        cic = row["mean_cic"]
        attractor = classify_attractor(cic)

        event_dir = os.path.join(
            runs_dir, "confirmation", "pretrained",
            f"seed_{seed}_ratio{ratio}",
        )
        if not os.path.isdir(event_dir):
            print(f"  WARNING: missing {event_dir}")
            continue

        curves = read_eval_curves(event_dir)
        steps = curves["steps"]
        acc = smooth(curves["listener_accuracy"], SMOOTH_WINDOW)
        phase2 = curves["phase2_start"]

        # Only show from phase 2 onward
        mask = steps >= phase2
        steps_p2 = steps[mask]
        acc_p2 = acc[mask]

        if attractor == "jabbering":
            color = CLR_JABBER
            alpha = 0.7
            lw = 1.5
            ls = "--"
            if not added_jabber:
                label = "Jabbering (CIC \u2248 0)"
                added_jabber = True
            else:
                label = None
        else:
            color = CLR_COMM
            alpha = 0.6
            lw = 1.5
            ls = "-"
            if not added_comm:
                label = "Communicative (CIC > 0)"
                added_comm = True
            else:
                label = None

        ax.plot(steps_p2, acc_p2, color=color, alpha=alpha, linewidth=lw,
                linestyle=ls, label=label)

    # Chance level
    ax.axhline(CHANCE_LEVEL, color="gray", linestyle=":", alpha=0.5,
               linewidth=1, label=f"Chance ({CHANCE_LEVEL:.3f})")

    ax.set_xlabel("Training Episode", fontsize=12)
    ax.set_ylabel("Listener Accuracy (eval, smoothed)", fontsize=12)
    ax.set_title("Pretrained Seeds: Listener Accuracy Trajectories\n"
                 "(Phase 2 onward, colored by final attractor)",
                 fontsize=13)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0.15, 0.70)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            os.path.join(figures_dir, f"fig9_pretrained_trajectories.{ext}"),
            dpi=200)
    plt.close(fig)
    print("  Saved fig9_pretrained_trajectories.{png,pdf}")


# ------------------------------------------------------------------ #
#  FIGURE 10 -- Baseline Listener Accuracy Trajectories                #
# ------------------------------------------------------------------ #
def fig_baseline_trajectories(base_rows, runs_dir, figures_dir):
    """Per-seed listener accuracy over training for baseline condition."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, row in enumerate(base_rows):
        seed = int(row["seed"])
        ratio = row["ratio"]

        event_dir = os.path.join(
            runs_dir, "confirmation", "baseline",
            f"seed_{seed}_ratio{ratio}",
        )
        if not os.path.isdir(event_dir):
            print(f"  WARNING: missing {event_dir}")
            continue

        curves = read_eval_curves(event_dir)
        steps = curves["steps"]
        acc = smooth(curves["listener_accuracy"], SMOOTH_WINDOW)

        label = "Baseline seeds" if i == 0 else None
        ax.plot(steps, acc, color=CLR_BASE, alpha=0.45, linewidth=1.5,
                label=label)

    ax.axhline(CHANCE_LEVEL, color="gray", linestyle=":", alpha=0.5,
               linewidth=1, label=f"Chance ({CHANCE_LEVEL:.3f})")

    ax.set_xlabel("Training Episode", fontsize=12)
    ax.set_ylabel("Listener Accuracy (eval, smoothed)", fontsize=12)
    ax.set_title("Baseline Seeds: Listener Accuracy Trajectories\n"
                 "(all seeds communicative, CIC > 0)",
                 fontsize=13)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0.15, 0.70)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            os.path.join(figures_dir, f"fig10_baseline_trajectories.{ext}"),
            dpi=200)
    plt.close(fig)
    print("  Saved fig10_baseline_trajectories.{png,pdf}")


# ------------------------------------------------------------------ #
#  FIGURE 11 -- Combined Trajectory Panel                              #
# ------------------------------------------------------------------ #
def fig_combined_trajectories(pre_rows, base_rows, runs_dir, figures_dir):
    """Side-by-side comparison of pretrained vs baseline trajectories."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)

    # ---- Left panel: Pretrained ----
    added_jabber = False
    added_comm = False
    for row in pre_rows:
        seed = int(row["seed"])
        ratio = row["ratio"]
        cic = row["mean_cic"]
        attractor = classify_attractor(cic)

        event_dir = os.path.join(
            runs_dir, "confirmation", "pretrained",
            f"seed_{seed}_ratio{ratio}",
        )
        if not os.path.isdir(event_dir):
            continue

        curves = read_eval_curves(event_dir)
        steps = curves["steps"]
        acc = smooth(curves["listener_accuracy"], SMOOTH_WINDOW)
        phase2 = curves["phase2_start"]

        mask = steps >= phase2
        steps_p2 = steps[mask]
        acc_p2 = acc[mask]

        if attractor == "jabbering":
            color, ls = CLR_JABBER, "--"
            label = "Jabbering" if not added_jabber else None
            added_jabber = True
        else:
            color, ls = CLR_COMM, "-"
            label = "Communicative" if not added_comm else None
            added_comm = True

        ax1.plot(steps_p2, acc_p2, color=color, alpha=0.6, linewidth=1.5,
                 linestyle=ls, label=label)

    ax1.axhline(CHANCE_LEVEL, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlabel("Training Episode", fontsize=12)
    ax1.set_ylabel("Listener Accuracy (eval, smoothed)", fontsize=12)
    ax1.set_title("Pretrained (Curriculum)", fontsize=13)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.2)

    # ---- Right panel: Baseline ----
    for i, row in enumerate(base_rows):
        seed = int(row["seed"])
        ratio = row["ratio"]

        event_dir = os.path.join(
            runs_dir, "confirmation", "baseline",
            f"seed_{seed}_ratio{ratio}",
        )
        if not os.path.isdir(event_dir):
            continue

        curves = read_eval_curves(event_dir)
        steps = curves["steps"]
        acc = smooth(curves["listener_accuracy"], SMOOTH_WINDOW)

        label = "All seeds communicative" if i == 0 else None
        ax2.plot(steps, acc, color=CLR_BASE, alpha=0.45, linewidth=1.5,
                 label=label)

    ax2.axhline(CHANCE_LEVEL, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Training Episode", fontsize=12)
    ax2.set_title("Baseline (No Curriculum)", fontsize=13)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.2)

    for ax in (ax1, ax2):
        ax.set_ylim(0.15, 0.70)

    fig.suptitle(
        "Long-term Listener Accuracy: Curriculum Learning Attractor Divergence",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            os.path.join(figures_dir, f"fig11_combined_trajectories.{ext}"),
            dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig11_combined_trajectories.{png,pdf}")


# ------------------------------------------------------------------ #
#  FIGURE 12 -- Adversary Accuracy Trajectories                        #
# ------------------------------------------------------------------ #
def fig_adversary_trajectories(pre_rows, base_rows, runs_dir, figures_dir):
    """Per-seed adversary accuracy over training for both conditions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)

    # ---- Left panel: Pretrained ----
    added_jabber = False
    added_comm = False
    for row in pre_rows:
        seed = int(row["seed"])
        ratio = row["ratio"]
        cic = row["mean_cic"]
        attractor = classify_attractor(cic)

        event_dir = os.path.join(
            runs_dir, "confirmation", "pretrained",
            f"seed_{seed}_ratio{ratio}",
        )
        if not os.path.isdir(event_dir):
            continue

        curves = read_eval_curves(event_dir)
        steps = curves["steps"]
        adv = smooth(curves["adversary_accuracy"], SMOOTH_WINDOW)
        phase2 = curves["phase2_start"]

        mask = steps >= phase2
        steps_p2 = steps[mask]
        adv_p2 = adv[mask]

        if attractor == "jabbering":
            color, ls = CLR_JABBER, "--"
            label = "Jabbering" if not added_jabber else None
            added_jabber = True
        else:
            color, ls = CLR_COMM, "-"
            label = "Communicative" if not added_comm else None
            added_comm = True

        ax1.plot(steps_p2, adv_p2, color=color, alpha=0.6, linewidth=1.5,
                 linestyle=ls, label=label)

    ax1.set_xlabel("Training Episode", fontsize=12)
    ax1.set_ylabel("Adversary Accuracy (eval, smoothed)", fontsize=12)
    ax1.set_title("Pretrained (Curriculum)", fontsize=13)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.2)

    # ---- Right panel: Baseline ----
    for i, row in enumerate(base_rows):
        seed = int(row["seed"])
        ratio = row["ratio"]

        event_dir = os.path.join(
            runs_dir, "confirmation", "baseline",
            f"seed_{seed}_ratio{ratio}",
        )
        if not os.path.isdir(event_dir):
            continue

        curves = read_eval_curves(event_dir)
        steps = curves["steps"]
        adv = smooth(curves["adversary_accuracy"], SMOOTH_WINDOW)

        label = "Baseline seeds" if i == 0 else None
        ax2.plot(steps, adv, color=CLR_BASE, alpha=0.45, linewidth=1.5,
                 label=label)

    ax2.set_xlabel("Training Episode", fontsize=12)
    ax2.set_title("Baseline (No Curriculum)", fontsize=13)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.2)

    for ax in (ax1, ax2):
        ax.set_ylim(0.0, 0.5)

    fig.suptitle(
        "Long-term Adversary Accuracy: Curriculum vs. Baseline",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            os.path.join(figures_dir, f"fig12_adversary_trajectories.{ext}"),
            dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig12_adversary_trajectories.{png,pdf}")


# ------------------------------------------------------------------ #
#  PRINT ATTRACTOR SUMMARY                                             #
# ------------------------------------------------------------------ #
def print_attractor_summary(pre_rows, base_rows):
    pre_cic = col(pre_rows, "mean_cic")
    pre_acc = col(pre_rows, "listener_accuracy")
    base_cic = col(base_rows, "mean_cic")
    base_acc = col(base_rows, "listener_accuracy")

    pre_jabber_mask = pre_cic < CIC_THRESHOLD
    base_jabber_mask = base_cic < CIC_THRESHOLD

    print()
    print("=" * 65)
    print("  ATTRACTOR ANALYSIS SUMMARY")
    print("=" * 65)
    print()
    print(f"  CIC threshold for jabbering classification: {CIC_THRESHOLD}")
    print(f"  Chance level (3-choice): {CHANCE_LEVEL:.4f}")
    print()
    print("  PRETRAINED (curriculum learning, ratio=0.1)")
    print(f"    Total seeds: {len(pre_cic)}")
    print(f"    Jabbering attractor:      {pre_jabber_mask.sum()} seeds")
    print(f"      Mean listener acc:      {pre_acc[pre_jabber_mask].mean():.4f}")
    print(f"      Mean CIC:               {pre_cic[pre_jabber_mask].mean():.2e}")
    print(f"    Communicative attractor:   {(~pre_jabber_mask).sum()} seeds")
    print(f"      Mean listener acc:      {pre_acc[~pre_jabber_mask].mean():.4f}")
    print(f"      Mean CIC:               {pre_cic[~pre_jabber_mask].mean():.4f}")
    print()
    print("  BASELINE (no curriculum, ratio=0.0)")
    print(f"    Total seeds: {len(base_cic)}")
    print(f"    Jabbering attractor:      {base_jabber_mask.sum()} seeds")
    print(f"    Communicative attractor:   {(~base_jabber_mask).sum()} seeds")
    print(f"      Mean listener acc:      {base_acc[~base_jabber_mask].mean():.4f}")
    print(f"      Mean CIC:               {base_cic[~base_jabber_mask].mean():.4f}")
    print()

    # Per-seed detail for pretrained
    print("  PRETRAINED -- Per-seed detail:")
    print(f"  {'Seed':>6}  {'Acc':>8}  {'CIC':>12}  {'Attractor':<15}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*12}  {'-'*15}")
    for row in sorted(pre_rows, key=lambda r: r["mean_cic"]):
        att = classify_attractor(row["mean_cic"])
        print(f"  {int(row['seed']):>6}  {row['listener_accuracy']:>8.3f}  "
              f"{row['mean_cic']:>12.4e}  {att:<15}")
    print()
    print("=" * 65)


# ------------------------------------------------------------------ #
#  MAIN                                                               #
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="Attractor analysis for curriculum learning"
    )
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--figures-dir", type=str, default=None)
    args = parser.parse_args()

    results_dir = args.results_dir
    runs_dir = args.runs_dir
    figures_dir = args.figures_dir or os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print(f"Loading results from {results_dir}/ ...")

    pre_path = os.path.join(results_dir, "pretrained_results.csv")
    base_path = os.path.join(results_dir, "baseline_results.csv")

    if not os.path.isfile(pre_path) or not os.path.isfile(base_path):
        print(f"ERROR: Missing pretrained_results.csv or baseline_results.csv "
              f"in {results_dir}/")
        sys.exit(1)

    pre_rows = load_csv(pre_path)
    base_rows = load_csv(base_path)
    print(f"  pretrained_results.csv : {len(pre_rows)} rows")
    print(f"  baseline_results.csv   : {len(base_rows)} rows")

    # ---- Attractor identification figures ----
    print("\nGenerating attractor analysis figures...")
    fig_attractor_scatter(pre_rows, base_rows, figures_dir)
    fig_attractor_prevalence(pre_rows, base_rows, figures_dir)
    fig_cic_distribution(pre_rows, base_rows, figures_dir)

    # ---- Long-term trajectory figures ----
    print("\nGenerating long-term trajectory figures...")
    print(f"  Reading TensorBoard logs from {runs_dir}/ ...")
    fig_pretrained_trajectories(pre_rows, runs_dir, figures_dir)
    fig_baseline_trajectories(base_rows, runs_dir, figures_dir)
    fig_combined_trajectories(pre_rows, base_rows, runs_dir, figures_dir)
    fig_adversary_trajectories(pre_rows, base_rows, runs_dir, figures_dir)

    # ---- Summary ----
    print_attractor_summary(pre_rows, base_rows)

    print(f"\n  All figures saved to: {figures_dir}/")


if __name__ == "__main__":
    main()
