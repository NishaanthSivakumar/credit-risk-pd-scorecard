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
PURPLE = "#7d5bd6"
SCENARIO_COLORS = {"Base": BLUE, "Mild recession": AQUA, "Severe recession": ORANGE,
                   "Interest rate shock": PURPLE}

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


def stress_distributions(losses_pct: dict, ax=None, title="Lifetime loss distribution by scenario"):
    """Overlaid loss distributions (% of funded amount), one outline per scenario."""
    ax = ax or plt.subplots(figsize=(6.8, 4))[1]
    hi = max(np.percentile(v, 99.9) for v in losses_pct.values())
    bins = np.linspace(0, hi, 90)
    for name, v in losses_pct.items():
        c = SCENARIO_COLORS.get(name, MUTED)
        ax.hist(v, bins=bins, histtype="step", lw=2, color=c, density=True)
        ax.axvline(np.quantile(v, 0.99), color=c, lw=1, ls="--")
        ax.plot([], [], color=c, lw=2, label=f"{name}  (VaR 99% {np.quantile(v, 0.99):.1f}%)")
    ax.set(xlabel="Portfolio loss (% of funded amount)", ylabel="Density", title=title)
    ax.set_yticks([])
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=8)
    return ax


def economy_curve_plot(curves: dict, points: dict | None = None, ax=None,
                       title="Expected loss as the economy worsens"):
    """curves: {label: Series of EL rate indexed by economy quantile}; points: {label: (quantile, el_rate)}."""
    ax = ax or plt.subplots(figsize=(6.8, 4.2))[1]
    for (label, s), c in zip(curves.items(), [BLUE, ORANGE, AQUA]):
        ax.plot(1 / s.index.to_numpy(), s.to_numpy() * 100, marker="o", ms=4, color=c, label=label)
    for label, (q, v) in (points or {}).items():
        c = SCENARIO_COLORS.get(label, INK)
        ax.plot(1 / q, v * 100, "D", ms=8, color=c, mec="white")
        ax.annotate(label, (1 / q, v * 100), textcoords="offset points", xytext=(6, -12), fontsize=8, color=c)
    ax.set_xscale("log")
    ax.set(xlabel="Economy: one-in-N outcome over the loans' life", ylabel="Expected loss (% of funded amount)",
           title=title)
    ax.legend(loc="upper left")
    return ax


def tornado_plot(table, ax=None, title="Severe recession: sensitivity to each assumption"):
    """table from stress.sensitivity (one row per assumption). Blue = low value, orange = high."""
    ax = ax or plt.subplots(figsize=(7.2, 3.8))[1]
    names = {"economy_quantile": ("economy", lambda v: f"1 in {1 / v:g}"),
             "pd_scale": ("PD level", lambda v: f"×{v:g}"),
             "ead_add": ("EAD add-on", lambda v: f"+{v * 100:.0f} pts"),
             "lgd_add": ("LGD add-on", lambda v: f"+{v * 100:.0f} pts"),
             "rho_mult": ("correlation", lambda v: f"×{v:g}")}
    t = table.iloc[::-1]
    y = np.arange(len(t))
    centre = t["el_rate_central"].iloc[0] * 100
    labels = []
    for yi, r in zip(y, t.itertuples()):
        ax.barh(yi, r.el_rate_low * 100 - centre, left=centre, color=BLUE, height=0.6)
        ax.barh(yi, r.el_rate_high * 100 - centre, left=centre, color=ORANGE, height=0.6)
        name, f = names.get(r.Index, (r.Index.replace("_", " "), "{:g}".format))
        labels.append(f"{name}\n{f(r.low)} to {f(r.high)}")
    ax.axvline(centre, color=INK, lw=1)
    ax.set_yticks(y, labels, fontsize=8)
    ax.set(xlabel="Expected loss (% of funded amount)", title=title)
    ax.grid(axis="y", visible=False)
    return ax


def implied_factor_plot(table, ax=None, title="Economy implied by each vintage's default rate"):
    """table from stress.implied_factor_by_vintage."""
    ax = ax or plt.subplots(figsize=(6.4, 3.8))[1]
    z = table["implied_z"]
    ax.bar(z.index.astype(str), z.to_numpy(), color=[ORANGE if v < 0 else BLUE for v in z], width=0.7)
    ax.axhline(0, color=INK, lw=1)
    ax.set(xlabel="Issue year", ylabel="Implied economy factor z (negative = worse)", title=title)
    ax.grid(axis="x", visible=False)
    return ax


def vulnerability_plot(table, label="Severe recession", ax=None,
                       title="Expected loss by grade, base vs severe recession"):
    """table from stress.segment_vulnerability by grade."""
    ax = ax or plt.subplots(figsize=(6.4, 4))[1]
    x = np.arange(len(table))
    ax.bar(x - 0.2, table["el_rate_base"] * 100, width=0.4, color=BLUE, label="Base")
    ax.bar(x + 0.2, table["el_rate_stress"] * 100, width=0.4, color=ORANGE, label=label)
    for xi, m in zip(x, table["multiple"]):
        ax.text(xi + 0.2, table["el_rate_stress"].iloc[xi] * 100, f"×{m:.1f}", ha="center", va="bottom",
                fontsize=8, color=MUTED)
    ax.set_xticks(x, [str(g) for g in table.index])
    ax.set(xlabel="Rating grade (1 = best)", ylabel="Expected loss (% of funded amount)", title=title)
    ax.grid(axis="x", visible=False)
    ax.legend()
    return ax


STRATEGY_COLORS = {"Whole book": INK, "Equal weight": MUTED, "Max expected return": ORANGE,
                   "Min expected loss": AQUA, "Min CVaR": PURPLE, "Max return, CVaR ≤ book": BLUE,
                   "Max return, EL ≤ book": "#c9a227"}
BAND_COLORS = {"A": "#1baf7a", "B": "#2a78d6", "C": "#7d5bd6", "D": "#eb6834", "E–G": "#b23a1e"}


def frontier_plot(front, strategies, seg=None, alpha=0.99, ax=None, title="Expected return vs tail risk (ex ante)"):
    """Segments (dots, by LendingClub band), the frontier and the strategies.
    Strategies at the same point are labelled together."""
    ax = ax or plt.subplots(figsize=(7.2, 4.6))[1]
    col = f"cvar_{alpha:g}"
    if seg is not None:
        for b, g in seg.groupby("lc_band"):
            ax.scatter(g[col] * 100, g["expected_return"] * 100, s=28, color=BAND_COLORS.get(b, MUTED),
                       alpha=0.6, edgecolor="white", label=f"segment, LendingClub {b}")
    ax.plot(front[col] * 100, front["expected_return"] * 100, color=INK, lw=1.5, label="frontier")
    pts = strategies[[col, "expected_return"]].round(5)
    for (x, y), names in pts.groupby([col, "expected_return"]).groups.items():
        ax.plot(x * 100, y * 100, "D", ms=8, color=INK, mec="white")
        ax.annotate("\n".join(names), (x * 100, y * 100), textcoords="offset points", xytext=(8, -4),
                    fontsize=7, color=INK, va="top")
    ax.set(xlabel=f"{alpha:.0%} CVaR (loss, % of funded amount; negative = still a gain)",
           ylabel="Expected lifetime net return (%)", title=title)
    ax.legend(fontsize=7, loc="lower left")
    return ax


def allocation_plot(by_band, ax=None, title="Allocation by LendingClub grade"):
    """by_band: strategies x LendingClub band weights."""
    ax = ax or plt.subplots(figsize=(6.8, 4.2))[1]
    t = by_band.iloc[::-1]
    left = np.zeros(len(t))
    for b in t.columns:
        ax.barh(t.index, t[b] * 100, left=left, color=BAND_COLORS.get(b, MUTED), height=0.65, label=b)
        left += t[b].to_numpy() * 100
    ax.set(xlabel="Share of portfolio (%)", title=title, xlim=(0, 100))
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=5, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    return ax


def strategy_outcomes_plot(table, ax=None, title="Expected vs realised lifetime net return, 2015 loans"):
    ax = ax or plt.subplots(figsize=(7.2, 4.2))[1]
    y = np.arange(len(table))[::-1]
    ax.barh(y + 0.2, table["expected_return"] * 100, height=0.4, color=MUTED, label="expected")
    ax.barh(y - 0.2, table["realised_return"] * 100, height=0.4, color=BLUE, label="realised")
    book = table.loc["Whole book", "realised_return"] * 100 if "Whole book" in table.index else None
    if book is not None:
        ax.axvline(book, color=INK, lw=1, ls="--", label="whole book, realised")
    ax.set_yticks(y, table.index)
    ax.set(xlabel="Lifetime net return per dollar (%)", title=title)
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.4, -0.3))
    return ax


def segment_scatter(seg, ax=None, title="Segments: expected vs realised return"):
    """Each segment (our grade x LendingClub band), sized by exposure."""
    ax = ax or plt.subplots(figsize=(6.4, 4.6))[1]
    size = seg["exposure"] / seg["exposure"].max() * 400
    for b, g in seg.groupby("lc_band"):
        ax.scatter(g["expected_return"] * 100, g["realised_return"] * 100, s=size[g.index], alpha=0.75,
                   color=BAND_COLORS.get(b, MUTED), edgecolor="white", label=f"LendingClub {b}")
    lo = min(seg["expected_return"].min(), seg["realised_return"].min()) * 100 - 1
    hi = max(seg["expected_return"].max(), seg["realised_return"].max()) * 100 + 1
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--", label="realised = expected")
    ax.set(xlabel="Expected lifetime net return (%)", ylabel="Realised lifetime net return (%)", title=title)
    ax.legend(fontsize=7, loc="upper left")
    return ax


RAG_COLORS = {"GREEN": "#1baf7a", "AMBER": "#e8a33d", "RED": "#d64545", "": "#f2f1ee"}


def monitoring_stability(stab, ax=None, title="Population stability by issue quarter"):
    ax = ax or plt.subplots(figsize=(7.2, 3.8))[1]
    x = np.arange(len(stab))
    ax.plot(x, stab["score_psi"], marker="o", ms=4, color=BLUE, label="score PSI")
    ax.plot(x, stab["max_csi"], marker="o", ms=4, color=ORANGE, label="largest feature CSI")
    for thr, lab in [(0.10, "amber"), (0.25, "red")]:
        ax.axhline(thr, color=MUTED, lw=1, ls="--")
        ax.text(len(x) - 0.5, thr, f" {lab}", color=MUTED, fontsize=8, va="bottom", ha="right")
    ax.set_xticks(x, stab.index, rotation=60, ha="right", fontsize=7)
    ax.set(ylabel="Index vs training sample", title=title, ylim=(0, max(0.3, float(stab["max_csi"].max()) * 1.1)))
    ax.legend(loc="upper left")
    return ax


def monitoring_early_warning(early, thresholds, ax=None,
                             title="Early warning: first-year defaults relative to the model's PDs"):
    ax = ax or plt.subplots(figsize=(7.2, 3.8))[1]
    x = np.arange(len(early))
    a, r = thresholds
    ax.axhspan(-a * 100, a * 100, color=RAG_COLORS["GREEN"], alpha=0.12, lw=0)
    for lo, hi in [(a, r), (-r, -a)]:
        ax.axhspan(lo * 100, hi * 100, color=RAG_COLORS["AMBER"], alpha=0.12, lw=0)
    ax.plot(x, early["ratio_shift"] * 100, marker="o", ms=5, color=BLUE)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(x, early.index, rotation=60, ha="right", fontsize=7)
    ax.set(ylabel="Change vs development benchmark (%)", title=title)
    return ax


def monitoring_calibration(perf, ax=None, title="Matured vintages: predicted vs observed default rate"):
    ax = ax or plt.subplots(figsize=(7.2, 3.8))[1]
    x = np.arange(len(perf))
    ax.plot(x, perf["observed_dr"] * 100, marker="o", ms=5, color=ORANGE, label="observed default rate")
    ax.plot(x, perf["mean_pd"] * 100, marker="o", ms=5, color=BLUE, label="mean PD (deployed model)")
    ax.set_xticks(x, perf.index, fontsize=8)
    ax.set(ylabel="Lifetime default rate (%)", title=title)
    ax.legend(loc="upper left")
    return ax


def alert_heatmap(grid, ax=None, title="Monitoring alerts by issue quarter"):
    """grid: quarters x metrics of GREEN / AMBER / RED / blank."""
    ax = ax or plt.subplots(figsize=(7.6, 0.32 * len(grid) + 1.4))[1]
    for i, (q, row) in enumerate(grid.iterrows()):
        for j, v in enumerate(row):
            ax.add_patch(plt.Rectangle((j, i), 0.95, 0.9, color=RAG_COLORS.get(v, "#f2f1ee")))
    ax.set_xlim(0, grid.shape[1])
    ax.set_ylim(grid.shape[0], 0)
    ax.set_xticks(np.arange(grid.shape[1]) + 0.475, grid.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(grid.shape[0]) + 0.45, grid.index, fontsize=7)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(title)
    handles = [plt.Rectangle((0, 0), 1, 1, color=RAG_COLORS[k]) for k in ("GREEN", "AMBER", "RED", "")]
    ax.legend(handles, ["green", "amber", "red", "not yet measurable"], ncol=4, fontsize=7,
              loc="lower center", bbox_to_anchor=(0.5, 1.03))
    return ax
