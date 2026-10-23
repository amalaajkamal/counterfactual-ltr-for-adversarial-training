"""
================================================================================
Regenerate Performance Comparison Figures from Table 1 Data (seed 42 run)
================================================================================
The figures plot the Yahoo! LETOR Set 2 column of Table 1 (seed 42 run,
results_yahoo_seed42.json): beta-VAE NDCG@10=0.7271, IPS-DCG=8.5753.

This script regenerates both figures directly from the numbers already in
the manuscript's Table 1 (Yahoo! LETOR Set 2 column), hardcoded below
exactly as they appear there -- no retraining or pipeline access needed,
since these are already-verified numbers, not new results.

DIFFERENCE FROM THE ORIGINAL FIGURES: the originals showed 6 methods
(Naive, IPS, Regression-EM, Bayes-IPS, Affine, beta-VAE). The current
Table 1 has 8, since Supervised-Only and InfoRank were added as baselines
during this revision specifically at reviewer request. Both new baselines
are included here; omitting them would mean regenerating a figure that is
current on its numbers but stale on its scope.

HOW TO USE:
  Run: python regenerate_performance_figures.py
  Produces: performance_comparison_yahoo.png, ips_dcg_comparison_yahoo.png
================================================================================
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# DATA -- copied exactly from the manuscript's current Table 1, Yahoo! LETOR
# Set 2 column (main_results table, subsec:main_results). Do not edit these
# without updating Table 1 to match, or the figure and the table will
# disagree with each other again, exactly the problem this script fixes.
# ============================================================================
methods = ["Naive\nBaseline", "IPS", "Regression-EM", "Bayes-IPS",
           "Affine", "InfoRank", "Supervised-\nOnly", "$\\beta$-VAE\n(Ours)"]

ndcg = [0.6186, 0.6390, 0.6613, 0.6329, 0.6269, 0.6236, 0.7345, 0.7271]
ndcg_delta_pct = [0.00, 3.30, 6.91, 2.31, 1.35, 0.82, 18.74, 17.55]

ips_dcg = [6.9830, 6.9508, 7.4505, 6.9714, 7.1512, 6.9681, 8.5938, 8.5753]
ips_delta_pct = [0.00, -0.46, 6.69, -0.17, 2.41, -0.21, 23.07, 22.80]

# Colors: baselines in neutral tones, InfoRank marked as most recent baseline,
# Supervised-Only in its own color (not a debiasing method), beta-VAE in
# green/highlighted since it is the proposed method.
colors = ["#999999", "#ADD8E6", "#90EE90", "#FFA500", "#FFD700",
          "#DDA0DD", "#87CEEB", "#2E8B57"]


def make_performance_figure(output_path="performance_comparison_yahoo.png"):
    """Two-panel figure: raw NDCG@10 (left), relative improvement vs. naive
    baseline in percent (right). Mirrors the original figure's layout."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    bars = axes[0].bar(methods, ndcg, color=colors, edgecolor="black")
    axes[0].set_title("Test Set Performance Comparison", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("NDCG@10", fontsize=12)
    axes[0].set_ylim(0, max(ndcg) * 1.15)
    for bar, val in zip(bars, ndcg):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    bar_colors_delta = ["#999999" if d == 0 else ("#d62728" if d < 0 else "#2ca02c")
                         for d in ndcg_delta_pct]
    bars2 = axes[1].bar(methods, ndcg_delta_pct, color=bar_colors_delta, edgecolor="black")
    axes[1].set_title("Improvement vs Baseline", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Relative Improvement (%)", fontsize=12)
    axes[1].axhline(0, color="black", linewidth=0.8)
    for bar, val in zip(bars2, ndcg_delta_pct):
        y_pos = bar.get_height() + (0.3 if val >= 0 else -0.6)
        axes[1].text(bar.get_x() + bar.get_width() / 2, y_pos,
                     f"{val:+.2f}%", ha="center", va="bottom" if val >= 0 else "top",
                     fontsize=10, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    for ax in axes:
        ax.tick_params(axis="x", labelsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def make_ips_dcg_figure(output_path="ips_dcg_comparison_yahoo.png"):
    """Single-panel figure: IPS-DCG comparison with percent labels relative
    to the naive baseline, mirroring the original figure's layout. """
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(methods, ips_dcg, color=colors, edgecolor="black")
    ax.set_title("Counterfactual Evaluation: IPS-DCG Comparison", fontsize=16, fontweight="bold")
    ax.set_ylabel("IPS-DCG Score", fontsize=13)
    ax.set_ylim(0, max(ips_dcg) * 1.2)

    for bar, val, delta in zip(bars, ips_dcg, ips_delta_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        label_color = "#d4f7d4" if delta > 0 else ("#f7d4d4" if delta < 0 else "#e0e0e0")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.5,
                f"{delta:+.1f}%", ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=label_color, edgecolor="none"))

    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


if __name__ == "__main__":
    make_performance_figure()
    make_ips_dcg_figure()
    print("\nBoth figures regenerated from the current, verified Table 1 data.")
