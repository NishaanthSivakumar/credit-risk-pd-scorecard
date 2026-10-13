"""Bootstrap confidence intervals for validation metrics.

Each resample is a vector of frequency weights over the original loans, so every
metric and every model is evaluated on the *same* resample. That makes paired
comparisons (challenger minus scorecard) straightforward.

Two resampling schemes:
  loan   - draw loans with replacement. Treats loans as independent.
  month  - draw whole issue months with replacement (block bootstrap). Loans
           issued in the same month face the same economy, so their outcomes are
           correlated; resampling months keeps that dependence intact. With few
           months the intervals are wider, and that width is real uncertainty.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import calibration as K


class _Ranked:
    """Pre-sorts one model's predictions so each resample's ROC is O(n), and keeps
    the full-sample calibration fits as starting values for each resample."""

    def __init__(self, y, p):
        self.slope_start = K.calibration_slope(y, p)
        self.intercept_start = K.calibration_intercept(y, p)
        order = np.argsort(-np.asarray(p), kind="mergesort")
        ps = np.asarray(p)[order]
        self.order, self.y = order, np.asarray(y, float)[order]
        self.ends = np.r_[np.flatnonzero(np.diff(ps)), len(ps) - 1]   # last index of each tie group

    def auc_ks(self, w) -> tuple[float, float]:
        ws = w[self.order]
        tp = np.cumsum(ws * self.y)[self.ends]
        fp = np.cumsum(ws * (1 - self.y))[self.ends]
        tpr, fpr = np.r_[0.0, tp / tp[-1]], np.r_[0.0, fp / fp[-1]]
        return float(np.trapezoid(tpr, fpr)), float(np.max(tpr - fpr))


def metrics_weighted(y, p, w, ranked: _Ranked | None = None) -> dict:
    ranked = ranked or _Ranked(y, p)
    auc, ks = ranked.auc_ks(w)
    _, slope = K.calibration_slope(y, p, w, start=ranked.slope_start)
    return {
        "auc": auc, "gini": 2 * auc - 1, "ks": ks,
        "brier": K.brier(y, p, w), "log_loss": K.log_loss(y, p, w),
        "cal_slope": slope, "cal_intercept": K.calibration_intercept(y, p, w, start=ranked.intercept_start),
        "eo_ratio": float(np.average(p, weights=w) / np.average(y, weights=w)),
    }


def resample_weights(n: int, groups: np.ndarray | None, rng: np.random.Generator) -> np.ndarray:
    if groups is None:
        return np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
    n_groups = groups.max() + 1
    counts = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)
    return counts[groups].astype(float)


def bootstrap(y, preds: dict[str, np.ndarray], groups: np.ndarray | None = None,
              n_draws: int = 1000, seed: int = 0) -> pd.DataFrame:
    """One row per (draw, model) with every metric. groups = integer block id per loan."""
    y = np.asarray(y, float)
    ranked = {m: _Ranked(y, p) for m, p in preds.items()}
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_draws):
        w = resample_weights(len(y), groups, rng)
        for m, p in preds.items():
            rows.append({"draw": b, "model": m, **metrics_weighted(y, p, w, ranked[m])})
    return pd.DataFrame(rows)


def point_estimates(y, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    w = np.ones(len(y))
    return pd.DataFrame({m: metrics_weighted(y, p, w) for m, p in preds.items()}).T


def intervals(draws: pd.DataFrame, point: pd.DataFrame, level: float = 0.95) -> pd.DataFrame:
    """Percentile intervals, one row per (model, metric)."""
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    long = draws.drop(columns="draw").melt(id_vars="model", var_name="metric")
    g = long.groupby(["model", "metric"])["value"]
    t = pd.DataFrame({"lower": g.quantile(lo), "upper": g.quantile(hi), "std_error": g.std()})
    t["estimate"] = [point.loc[m, k] for m, k in t.index]
    return t[["estimate", "lower", "upper", "std_error"]]


def paired_difference(draws: pd.DataFrame, point: pd.DataFrame, model_a: str, model_b: str,
                      metrics=("gini", "ks", "brier", "log_loss"), level: float = 0.95) -> pd.DataFrame:
    """Interval for metric(model_b) - metric(model_a) using the shared resamples."""
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    a = draws[draws["model"] == model_a].set_index("draw")
    b = draws[draws["model"] == model_b].set_index("draw")
    rows = []
    for k in metrics:
        d = b[k] - a[k]
        rows.append({"metric": k, "difference": point.loc[model_b, k] - point.loc[model_a, k],
                     "lower": d.quantile(lo), "upper": d.quantile(hi),
                     "share_of_draws_b_better": float((d > 0).mean() if k in ("gini", "ks", "auc")
                                                      else (d < 0).mean())})
    t = pd.DataFrame(rows).set_index("metric")
    t["significant"] = (t["lower"] > 0) | (t["upper"] < 0)
    return t


def month_groups(issue_d: pd.Series) -> np.ndarray:
    return pd.factorize(pd.to_datetime(issue_d).dt.to_period("M"), sort=True)[0]
