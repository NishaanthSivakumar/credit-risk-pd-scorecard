"""Model validation: discrimination, calibration and stability.

These are the three questions a model validation team asks of any PD model:
  1. Discrimination - does it rank good borrowers above bad ones?  (AUC, Gini, KS)
  2. Calibration    - are the predicted PDs the right *level*?     (Brier, HL, binomial)
  3. Stability      - is today's population like the one we built on? (PSI, CSI)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from . import config as C


# ----------------------------------------------------------------------------
# 1. Discrimination
# ----------------------------------------------------------------------------
def discrimination(y, pd_pred) -> dict:
    """AUC: P(random bad gets a higher PD than random good). 0.5 = coin flip.
    Gini = 2*AUC - 1 (the number banks usually quote).
    KS  = max gap between the cumulative score distributions of goods and bads."""
    auc = roc_auc_score(y, pd_pred)
    fpr, tpr, _ = roc_curve(y, pd_pred)
    return {"auc": auc, "gini": 2 * auc - 1, "ks": float(np.max(tpr - fpr))}


def rank_ordering(y, score, n_bands=10) -> pd.DataFrame:
    """Default rate by score decile. A working scorecard shows a bad rate that
    falls steadily as the score rises."""
    d = pd.DataFrame({"y": np.asarray(y), "score": score})
    d["band"] = pd.qcut(d["score"].rank(method="first"), n_bands, labels=range(1, n_bands + 1))
    t = d.groupby("band", observed=True).agg(min_score=("score", "min"), max_score=("score", "max"),
                                              n=("y", "size"), bads=("y", "sum"))
    t["bad_rate"] = t["bads"] / t["n"]
    t["monotonic"] = t["bad_rate"].diff().fillna(-1).le(0)
    return t


# ----------------------------------------------------------------------------
# 2. Calibration
# ----------------------------------------------------------------------------
def calibration_table(y, pd_pred, n_groups=10) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y), "pd": pd_pred})
    d["group"] = pd.qcut(d["pd"].rank(method="first"), n_groups, labels=range(1, n_groups + 1))
    t = d.groupby("group", observed=True).agg(n=("y", "size"), observed_dr=("y", "mean"),
                                               mean_pd=("pd", "mean"))
    return t


def hosmer_lemeshow(y, pd_pred, n_groups=10) -> dict:
    """Chi-square test of observed vs expected defaults across PD deciles.
    Caveat for the write-up: with 100k+ loans HL rejects almost any model
    because tiny deviations become 'significant' - read it alongside the
    calibration plot, not on its own."""
    t = calibration_table(y, pd_pred, n_groups)
    exp_bad = t["mean_pd"] * t["n"]
    obs_bad = t["observed_dr"] * t["n"]
    exp_good, obs_good = t["n"] - exp_bad, t["n"] - obs_bad
    chi2 = float((((obs_bad - exp_bad) ** 2) / exp_bad + ((obs_good - exp_good) ** 2) / exp_good).sum())
    return {"hl_chi2": chi2, "hl_pvalue": float(stats.chi2.sf(chi2, n_groups - 2))}


def calibration_summary(y, pd_pred) -> dict:
    y = np.asarray(y)
    return {"mean_pd": float(np.mean(pd_pred)), "observed_dr": float(y.mean()),
            "brier": brier_score_loss(y, pd_pred), **hosmer_lemeshow(y, pd_pred)}


def grade_edges(train_score, n_grades=C.N_GRADES) -> np.ndarray:
    """Rating grades = score bands with equal population on the TRAIN sample.
    Grade 1 = best (highest score)."""
    return np.unique(np.quantile(train_score, np.linspace(0, 1, n_grades + 1)[1:-1]))


def assign_grade(score, edges) -> np.ndarray:
    return len(edges) + 1 - np.searchsorted(edges, score, side="right")


def binomial_test_by_grade(y, pd_pred, grade) -> pd.DataFrame:
    """For each grade: is the observed number of defaults consistent with the
    grade's average PD? One-sided test (H1: true PD is HIGHER than predicted),
    because under-estimating risk is the dangerous direction for a bank."""
    d = pd.DataFrame({"y": np.asarray(y), "pd": pd_pred, "grade": grade})
    t = d.groupby("grade").agg(n=("y", "size"), defaults=("y", "sum"), mean_pd=("pd", "mean"))
    t["observed_dr"] = t["defaults"] / t["n"]
    t["p_value"] = [stats.binomtest(int(r.defaults), int(r.n), r.mean_pd, alternative="greater").pvalue
                    for r in t.itertuples()]
    t["result"] = np.where(t["p_value"] < 0.01, "RED: PD too low",
                   np.where(t["p_value"] < 0.05, "AMBER", "GREEN"))
    return t


# ----------------------------------------------------------------------------
# 3. Stability
# ----------------------------------------------------------------------------
def psi(expected, actual, n_bins=10) -> float:
    """Population Stability Index between a reference sample (train) and a new one.
    PSI = sum (a% - e%) * ln(a% / e%)   over bins defined on the reference sample.
    <0.10 stable, 0.10-0.25 monitor, >0.25 significant shift."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)[1:-1]))
    e = np.bincount(np.searchsorted(edges, expected), minlength=len(edges) + 1) / len(expected)
    a = np.bincount(np.searchsorted(edges, actual), minlength=len(edges) + 1) / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def csi_by_feature(binner, train: pd.DataFrame, other: pd.DataFrame, features) -> pd.Series:
    """Characteristic Stability Index: PSI computed on each feature's WoE bins.
    Tells you WHICH inputs drifted when the score PSI moves."""
    out = {}
    for f in features:
        e = binner.bins[f].bin_labels(train[f]).value_counts(normalize=True)
        a = binner.bins[f].bin_labels(other[f]).value_counts(normalize=True)
        idx = e.index.union(a.index)
        e, a = e.reindex(idx, fill_value=1e-6).clip(lower=1e-6), a.reindex(idx, fill_value=1e-6).clip(lower=1e-6)
        out[f] = float(((a - e) * np.log(a / e)).sum())
    return pd.Series(out, name="csi").sort_values(ascending=False)


# ----------------------------------------------------------------------------
# Traffic lights
# ----------------------------------------------------------------------------
def rag_gini(g):
    return "GREEN" if g >= C.GINI_GREEN else ("AMBER" if g >= C.GINI_AMBER else "RED")


def rag_psi(p):
    return "GREEN" if p < C.PSI_GREEN else ("AMBER" if p < C.PSI_AMBER else "RED")


def benchmark_auc(y, sub_grade: pd.Series) -> float:
    """AUC of LendingClub's own sub-grade (A1 best ... G5 worst) as a risk ranking."""
    s = sub_grade.astype(str).str.strip()
    order = {f"{g}{i}": k for k, (g, i) in enumerate(((g, i) for g in "ABCDEFG" for i in range(1, 6)))}
    rank = s.map(order)
    ok = rank.notna().to_numpy()
    return roc_auc_score(np.asarray(y)[ok], rank[ok])
