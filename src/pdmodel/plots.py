"""Validation charts (matplotlib). One consistent style for every figure."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

# Fixed colour roles - the same entity always gets the same colour.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SAMPLE_COLORS = {"train": BLUE, "test": AQUA, "oot": ORANGE}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
    "lines.linewidth": 2,
})


def roc_plot(curves: dict, ax=None):
    """curves = {"train": (y, pd), "oot": (y, pd), ...}"""
    ax = ax or plt.subplots(figsize=(5.2, 4.6))[1]
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        auc = np.trapezoid(tpr, fpr)
        ax.plot(fpr, tpr, color=SAMPLE_COLORS.get(name, BLUE), label=f"{name}  (Gini {2*auc-1:.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="random")
    ax.set(xlabel="False positive rate (goods flagged)", ylabel="True positive rate (bads caught)",
           title="ROC curve", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="lower right")
    return ax


def calibration_plot(tables: dict, ax=None):
    """tables = {"train": calibration_table(...), "oot": ...}"""
    ax = ax or plt.subplots(figsize=(5.2, 4.6))[1]
    hi = max(max(t["mean_pd"].max(), t["observed_dr"].max()) for t in tables.values()) * 1.05
    ax.plot([0, hi], [0, hi], color=MUTED, lw=1, ls="--", label="perfect calibration")
    for name, t in tables.items():
        ax.plot(t["mean_pd"], t["observed_dr"], marker="o", ms=6,
                color=SAMPLE_COLORS.get(name, BLUE), label=name)
    ax.set(xlabel="Mean predicted PD (decile)", ylabel="Observed default rate",
           title="Calibration by PD decile", xlim=(0, hi), ylim=(0, hi))
    ax.legend(loc="upper left")
    return ax


def score_distribution(score, y, ax=None):
    ax = ax or plt.subplots(figsize=(6.4, 4))[1]
    y = np.asarray(y)
    bins = np.linspace(np.percentile(score, 0.5), np.percentile(score, 99.5), 40)
    ax.hist(score[y == 0], bins=bins, density=True, color=BLUE, alpha=0.55, label="good (fully paid)")
    ax.hist(score[y == 1], bins=bins, density=True, color=ORANGE, alpha=0.55, label="bad (charged off)")
    ax.set(xlabel="Score (higher = safer)", ylabel="Density", title="Score distribution by outcome")
    ax.legend()
    return ax


def bad_rate_by_band(rank_table, ax=None, title="Default rate by score decile"):
    ax = ax or plt.subplots(figsize=(6.4, 4))[1]
    x = np.arange(len(rank_table))
    ax.bar(x, rank_table["bad_rate"] * 100, color=BLUE, width=0.75)
    ax.set_xticks(x, [f"{int(a)}–{int(b)}" for a, b in zip(rank_table["min_score"], rank_table["max_score"])],
                  rotation=45, ha="right", fontsize=8)
    ax.set(xlabel="Score band", ylabel="Default rate (%)", title=title)
    ax.grid(axis="x", visible=False)
    return ax


def woe_plot(table, ax=None):
    """Bad rate per bin for one feature (from WoEBinner.bins[f].table)."""
    ax = ax or plt.subplots(figsize=(6.4, 3.6))[1]
    x = np.arange(len(table))
    colors = [MUTED if b == "Missing" else BLUE for b in table["bin"]]
    ax.bar(x, table["bad_rate"] * 100, color=colors, width=0.75)
    ax.set_xticks(x, table["bin"], rotation=35, ha="right", fontsize=8)
    ax.set(ylabel="Default rate (%)", title=f"{table['feature'].iloc[0]}  (IV {table['iv_contrib'].sum():.3f})")
    ax.grid(axis="x", visible=False)
    return ax


def default_rate_by_vintage(df, ax=None):
    ax = ax or plt.subplots(figsize=(6.4, 3.6))[1]
    g = df.groupby("issue_year")["default"].agg(["mean", "size"])
    ax.plot(g.index, g["mean"] * 100, marker="o", ms=6, color=BLUE)
    ax.set(xlabel="Issue year", ylabel="Default rate (%)", title="Default rate by vintage (36m loans)")
    ax.set_ylim(bottom=0)
    return ax


def psi_bar(csi, ax=None, title="Feature stability (CSI): train vs OOT"):
    ax = ax or plt.subplots(figsize=(6.4, 3.8))[1]
    s = csi.sort_values()
    ax.barh(s.index, s.values, color=BLUE, height=0.7)
    for thr, lab in [(0.10, "monitor"), (0.25, "significant")]:
        ax.axvline(thr, color=MUTED, lw=1, ls="--")
        ax.text(thr, 0.02, f" {lab}", color=MUTED, fontsize=8, va="bottom",
                transform=ax.get_xaxis_transform())
    ax.set(xlabel="CSI", title=title, xlim=(0, max(0.3, float(s.max()) * 1.1)))
    ax.grid(axis="y", visible=False)
    return ax
