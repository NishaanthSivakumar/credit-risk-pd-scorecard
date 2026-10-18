"""Ongoing model monitoring, quarter by quarter.

Three questions, each answerable at a different lag:
  1. Inputs and predictions (immediately): has the population moved?
     Score PSI, per-feature CSI, missing values, out-of-range inputs, mean PD, grade mix.
  2. Early performance (after about 15 months): are borrowers stopping paying
     faster than the model's PDs imply? Early-default rate / mean PD, against the
     same ratio on the development vintages, and Gini on early default.
  3. Full performance (after 36 months): discrimination and calibration on
     matured vintages.

Every measure gets a RAG status from thresholds in config.py; the alert log and
the recalibration / redevelopment triggers are built from those statuses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import calibration as K
from . import config as C
from . import validation as V

ORDER = {"GREEN": 0, "AMBER": 1, "RED": 2}


def rag(value: float, thresholds: tuple[float, float]) -> str:
    """GREEN below the first threshold, AMBER below the second, else RED (on |value|)."""
    if pd.isna(value):
        return ""
    v = abs(value)
    return "GREEN" if v < thresholds[0] else ("AMBER" if v < thresholds[1] else "RED")


def quarter(issue_d: pd.Series) -> pd.Series:
    return pd.to_datetime(issue_d).dt.to_period("Q").astype(str)


# ----------------------------------------------------------------------------
# 1. Inputs and predictions
# ----------------------------------------------------------------------------
def stability_by_quarter(model, train: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Score PSI and per-feature CSI against the training sample, each quarter."""
    card = model.card
    ref_score = card.score(train)
    q = quarter(df["issue_d"])
    rows = {}
    for k, g in df.groupby(q):
        csi = V.csi_by_feature(card.binner, train, g, card.features)
        rows[k] = {"loans": len(g), "score_psi": V.psi(ref_score, card.score(g)),
                   **{f"csi_{f}": v for f, v in csi.items()},
                   "max_csi": csi.max(), "max_csi_feature": csi.idxmax()}
    return pd.DataFrame(rows).T.rename_axis("quarter")


def data_quality_by_quarter(train: pd.DataFrame, df: pd.DataFrame, features) -> pd.DataFrame:
    """Rise in each feature's missing share against train, and the share of values
    outside the range seen in training (numeric features)."""
    q = quarter(df["issue_d"])
    num = [f for f in features if f in C.NUMERIC_FEATURES]
    lo, hi = train[num].min(), train[num].max()
    base_missing = train[features].isna().mean()
    rows = {}
    for k, g in df.groupby(q):
        miss = g[features].isna().mean() - base_missing
        x = g[num]
        oor = ((x.lt(lo) | x.gt(hi)) & x.notna()).mean()
        rows[k] = {**{f"missing_change_{f}": v for f, v in miss.items()},
                   **{f"out_of_range_{f}": v for f, v in oor.items()},
                   "max_missing_change": miss.max(), "max_missing_feature": miss.idxmax(),
                   "max_out_of_range": oor.max(), "max_out_of_range_feature": oor.idxmax()}
    return pd.DataFrame(rows).T.rename_axis("quarter")


def predictions_by_quarter(model, df: pd.DataFrame, reference_pd: float) -> pd.DataFrame:
    """Mean PD and the share of loans in each rating grade, each quarter."""
    q = quarter(df["issue_d"])
    p, g = model.predict_pd(df), model.grade(df)
    d = pd.DataFrame({"q": q.to_numpy(), "pd": p, "grade": g})
    t = d.groupby("q").agg(loans=("pd", "size"), mean_pd=("pd", "mean"))
    t["mean_pd_shift"] = t["mean_pd"] / reference_pd - 1
    mix = pd.crosstab(d["q"], d["grade"], normalize="index")
    mix.columns = [f"grade_{c}" for c in mix.columns]
    return t.join(mix).rename_axis("quarter")


# ----------------------------------------------------------------------------
# 2-3. Performance
# ----------------------------------------------------------------------------
def performance_by_quarter(model, df: pd.DataFrame, validated_gini: float) -> pd.DataFrame:
    """Discrimination and calibration on matured vintages, each quarter."""
    q = quarter(df["issue_d"])
    rows = {}
    for k, g in df.groupby(q):
        y, p = g["default"].to_numpy(), model.predict_pd(g)
        d = V.discrimination(y, p)
        a, b = K.calibration_slope(y, p)
        binom = V.binomial_test_by_grade(y, p, model.grade(g))
        rows[k] = {"loans": len(g), "observed_dr": y.mean(), "mean_pd": p.mean(), "cal_gap": p.mean() - y.mean(),
                   "eo_ratio": p.mean() / y.mean(), "gini": d["gini"], "ks": d["ks"],
                   "gini_drop": (validated_gini - d["gini"]) / validated_gini,
                   "cal_intercept": K.calibration_intercept(y, p), "cal_slope": b,
                   "grades_failing": int(binom["result"].str.startswith("RED").sum())}
    return pd.DataFrame(rows).T.rename_axis("quarter")


def early_performance_by_quarter(model, df: pd.DataFrame, benchmark_ratio: float | None = None,
                                 min_months: int = C.EARLY_MIN_MONTHS_OBSERVED) -> pd.DataFrame:
    """Early-default rate against mean PD, each quarter observed for at least
    min_months. If no benchmark is given, the mean ratio of the input is used."""
    if "months_observed" in df:              # matured loans have no months_observed: keep them
        df = df[df["months_observed"].isna() | (df["months_observed"] >= min_months)]
    q = quarter(df["issue_d"])
    rows = {}
    for k, g in df.groupby(q):
        y, p = g["early_default"].to_numpy(), model.predict_pd(g)
        rows[k] = {"loans": len(g), "early_dr": y.mean(), "mean_pd": p.mean(), "ratio": y.mean() / p.mean(),
                   "gini_early": V.discrimination(y, p)["gini"] if 0 < y.sum() < len(y) else np.nan}
    t = pd.DataFrame(rows).T.rename_axis("quarter")
    bench = benchmark_ratio if benchmark_ratio is not None else float(t["ratio"].mean())
    t["ratio_shift"] = t["ratio"] / bench - 1
    return t


def early_ratio_benchmark(model, reference: pd.DataFrame) -> float:
    """Early-default rate / mean PD on a reference sample (the development
    vintages): how much of the model's lifetime PD normally shows up as a default
    in the first year."""
    return float(reference["early_default"].mean() / model.predict_pd(reference).mean())


def segment_early(model, df: pd.DataFrame, segments: dict[str, np.ndarray], benchmark_ratio: float) -> pd.DataFrame:
    y, p = df["early_default"].to_numpy(), model.predict_pd(df)
    rows = {}
    for name, m in segments.items():
        if m.sum() == 0:
            continue
        rows[name] = {"loans": int(m.sum()), "early_dr": y[m].mean(), "mean_pd": p[m].mean(),
                      "ratio": y[m].mean() / p[m].mean(),
                      "ratio_shift": y[m].mean() / p[m].mean() / benchmark_ratio - 1,
                      "gini_early": V.discrimination(y[m], p[m])["gini"] if 0 < y[m].sum() < m.sum() else np.nan}
    return pd.DataFrame(rows).T.rename_axis("segment")


# ----------------------------------------------------------------------------
# Alerts and triggers
# ----------------------------------------------------------------------------
METRICS = {
    # name: (table, column, thresholds, what it watches)
    "Score PSI": ("stability", "score_psi", C.MON_PSI, "shift in the score distribution"),
    "Largest feature CSI": ("stability", "max_csi", C.MON_PSI, "shift in any one input"),
    "Missing values": ("quality", "max_missing_change", C.MON_MISSING_CHANGE, "rise in any input's missing share"),
    "Out-of-range inputs": ("quality", "max_out_of_range", C.MON_OUT_OF_RANGE, "values outside the training range"),
    "Mean PD": ("predictions", "mean_pd_shift", C.MON_MEAN_PD_SHIFT, "change in average predicted risk"),
    "Early default vs PD": ("early", "ratio_shift", C.MON_EARLY_RATIO_SHIFT, "first-year defaults vs PDs"),
    "Calibration gap": ("performance", "cal_gap", C.MON_CAL_GAP, "mean PD vs default rate, matured"),
    "Gini fall": ("performance", "gini_drop", C.MON_GINI_DROP, "ranking power vs validation, matured"),
}


def alert_grid(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """RAG status of every metric in every quarter (blank where not yet measurable)."""
    quarters = sorted(set().union(*[t.index for t in tables.values()]))
    grid = pd.DataFrame(index=quarters, columns=list(METRICS), dtype=object).fillna("")
    for name, (tab, col, thr, _) in METRICS.items():
        t = tables[tab]
        for q, v in t[col].items():
            if name == "Gini fall" and v < 0:      # better than validation is fine
                v = 0.0
            grid.loc[q, name] = rag(v, thr)
    return grid.rename_axis("quarter")


def alert_log(tables: dict[str, pd.DataFrame], grid: pd.DataFrame) -> pd.DataFrame:
    """Every AMBER and RED reading, with its value and threshold."""
    rows = []
    for name, (tab, col, thr, what) in METRICS.items():
        for q, status in grid[name].items():
            if status in ("AMBER", "RED"):
                rows.append({"quarter": q, "metric": name, "status": status,
                             "value": tables[tab].loc[q, col], "amber_at": thr[0], "red_at": thr[1],
                             "watches": what})
    return pd.DataFrame(rows).set_index(["quarter", "metric"]) if rows else pd.DataFrame()


def triggers(grid: pd.DataFrame, performance: pd.DataFrame) -> pd.DataFrame:
    """Recalibration and redevelopment triggers, tied to the validation decision rules:
    a level problem with a sound ranking means recalibrate; a ranking or stability
    problem means redevelop."""
    def consecutive(col, statuses, n=2):
        s = grid[col].isin(statuses).to_numpy()
        return any(s[i:i + n].all() for i in range(len(s) - n + 1))

    last_year = performance.index[-4:]
    rules = [
        ("Recalibrate", "Calibration gap AMBER or RED in the latest matured year",
         bool(grid.loc[last_year, "Calibration gap"].isin(["AMBER", "RED"]).any())),
        ("Recalibrate", "Any grade failing the binomial test at 1% in the latest matured year",
         bool((performance.loc[last_year, "grades_failing"] > 0).any())),
        ("Recalibrate", "Early default vs PD AMBER or RED for two quarters in a row",
         consecutive("Early default vs PD", ["AMBER", "RED"])),
        ("Redevelop", "Score PSI or largest feature CSI RED in any quarter",
         bool(grid[["Score PSI", "Largest feature CSI"]].eq("RED").any().any())),
        ("Redevelop", "Gini fall RED in any matured quarter", bool(grid["Gini fall"].eq("RED").any())),
        ("Redevelop", "Early default vs PD RED for four quarters in a row (recalibration not keeping up)",
         consecutive("Early default vs PD", ["RED"], 4)),
    ]
    return pd.DataFrame(rules, columns=["action", "rule", "fired"]).set_index(["action", "rule"])


def thresholds_table() -> pd.DataFrame:
    rows = [{"metric": n, "watches": w, "amber_from": thr[0], "red_from": thr[1], "table": tab}
            for n, (tab, _, thr, w) in METRICS.items()]
    freq = {"stability": "quarterly, at origination", "quality": "quarterly, at origination",
            "predictions": "quarterly, at origination", "early": f"quarterly, from {C.EARLY_MIN_MONTHS_OBSERVED} months",
            "performance": "quarterly, at 36 months"}
    t = pd.DataFrame(rows).set_index("metric")
    t["when"] = t.pop("table").map(freq)
    return t
