"""Validation charts (matplotlib). One consistent style for every figure."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

# Fixed colour roles - the same entity always gets the same colour.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SAMPLE_COLORS = {"train": BLUE, "test": AQUA, "calib": MUTED, "oot": ORANGE}
METHOD_COLORS = {"none": ORANGE, "intercept": BLUE, "intercept_slope": AQUA}
MODEL_COLORS = {"scorecard": BLUE, "challenger": ORANGE}

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


def calibration_plot(tables: dict, ax=None, colors: dict | None = None, title="Calibration by PD decile"):
    """tables = {"train": calibration_table(...), "oot": ...}"""
    ax = ax or plt.subplots(figsize=(5.2, 4.6))[1]
    colors = colors or SAMPLE_COLORS
    hi = max(max(t["mean_pd"].max(), t["observed_dr"].max()) for t in tables.values()) * 1.05
    ax.plot([0, hi], [0, hi], color=MUTED, lw=1, ls="--", label="perfect calibration")
    for name, t in tables.items():
        ax.plot(t["mean_pd"], t["observed_dr"], marker="o", ms=6,
                color=colors.get(name, BLUE), label=name)
    ax.set(xlabel="Mean predicted PD (decile)", ylabel="Observed default rate",
           title=title, xlim=(0, hi), ylim=(0, hi))
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


def ci_plot(ci, metric="gini", ax=None, title="Out-of-time Gini with 95% bootstrap intervals"):
    """ci: rows (scheme, model) with estimate / lower / upper for one metric."""
    ax = ax or plt.subplots(figsize=(6.4, 3.0))[1]
    rows = ci.xs(metric, level="metric").reset_index()
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows.itertuples()):
        c = MODEL_COLORS.get(r.model, BLUE)
        ax.plot([r.lower, r.upper], [yi, yi], color=c, lw=3, solid_capstyle="round")
        ax.plot(r.estimate, yi, "o", color=c, ms=7, mec="white")
    ax.set_yticks(y, [f"{r.model} · {r.scheme} resampling" for r in rows.itertuples()])
    ax.set(xlabel=metric.capitalize() if metric != "gini" else "Gini", title=title)
    ax.grid(axis="y", visible=False)
    return ax


def loss_distribution(losses_pct, measures_pct: dict, realised_pct=None, ax=None,
                      title="Simulated lifetime loss, 2015 portfolio"):
    """Histogram of portfolio loss (% of exposure) with VaR / ES marked."""
    ax = ax or plt.subplots(figsize=(6.8, 4))[1]
    ax.hist(losses_pct, bins=80, color=BLUE, alpha=0.8)
    marks = [("EL", measures_pct["expected_loss"], MUTED, "-"),
             ("VaR 99%", measures_pct["var_0.99"], INK, "--"),
             ("ES 99%", measures_pct["es_0.99"], INK, ":")]
    if realised_pct is not None:
        marks.append(("realised", realised_pct, ORANGE, "-"))
    for lab, v, c, ls in marks:
        ax.axvline(v, color=c, lw=1.6, ls=ls, label=f"{lab}  {v:.2f}%")
    ax.set(xlabel="Portfolio loss (% of funded amount)", ylabel="Scenarios", title=title)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    return ax


def correlation_sensitivity(table, ax=None):
    """table indexed by rho with expected_loss / var_0.99 / es_0.99 in % of exposure."""
    ax = ax or plt.subplots(figsize=(6.4, 4))[1]
    for col, lab, c in [("expected_loss", "Expected loss", MUTED), ("var_0.99", "VaR 99%", BLUE),
                        ("es_0.99", "ES 99%", ORANGE)]:
        ax.plot(table.index, table[col], marker="o", ms=5, color=c, label=lab)
    ax.set(xlabel="Asset correlation ρ", ylabel="Loss (% of funded amount)",
           title="Tail loss rises with default correlation")
    ax.legend()
    return ax


def convergence_plot(table, ax=None):
    ax = ax or plt.subplots(figsize=(6.4, 4))[1]
    x = table.index.to_numpy()
    for col, lab, c in [("var_0.99", "VaR 99%", BLUE), ("es_0.99", "ES 99%", ORANGE),
                        ("expected_loss", "Expected loss", MUTED)]:
        se = table["el_se" if col == "expected_loss" else f"{col}_se"].to_numpy()
        ax.plot(x, table[col], marker="o", ms=5, color=c, label=lab)
        ax.fill_between(x, table[col] - 2 * se, table[col] + 2 * se, color=c, alpha=0.15, lw=0)
    ax.set_xscale("log")
    ax.set(xlabel="Scenarios", ylabel="Loss (% of funded amount)", title="Monte Carlo convergence (±2 s.e.)")
    ax.legend()
    return ax


def concentration_plot(table, ax=None):
    """table indexed by portfolio size with var_0.99 and es_0.99 in % of exposure."""
    ax = ax or plt.subplots(figsize=(6.4, 4))[1]
    x = table.index.to_numpy()
    ax.plot(x, table["var_0.99"], marker="o", ms=5, color=BLUE, label="VaR 99%")
    ax.plot(x, table["es_0.99"], marker="o", ms=5, color=ORANGE, label="ES 99%")
    ax.plot(x, table["expected_loss"], marker="o", ms=5, color=MUTED, label="Expected loss")
    ax.set_xscale("log")
    ax.set(xlabel="Loans in portfolio", ylabel="Loss (% of funded amount)",
           title="Smaller portfolios carry more tail risk")
    ax.legend()
    return ax


def el_by_grade(table, ax=None):
    """Expected vs realised loss rate by rating grade."""
    ax = ax or plt.subplots(figsize=(6.4, 4))[1]
    x = np.arange(len(table))
    ax.bar(x - 0.2, table["el_rate"] * 100, width=0.4, color=BLUE, label="expected loss")
    ax.bar(x + 0.2, table["realised_loss_rate"] * 100, width=0.4, color=ORANGE, label="realised loss")
    ax.set_xticks(x, [str(g) for g in table.index])
    ax.set(xlabel="Rating grade (1 = best)", ylabel="Loss (% of funded amount)",
           title="Expected vs realised lifetime loss by grade")
    ax.grid(axis="x", visible=False)
    ax.legend()
    return ax
