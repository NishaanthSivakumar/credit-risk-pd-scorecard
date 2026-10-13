"""Bootstrap engine: fast ROC, resampling schemes, intervals."""
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score, roc_curve

from pdmodel import bootstrap as B


def _data(n=20_000, seed=0, month_effect=0.0):
    r = np.random.default_rng(seed)
    month = r.integers(0, 12, n)
    shock = r.normal(0, month_effect, 12)[month]
    x = r.normal(size=n)
    p = expit(-2 + 0.8 * x)
    y = (r.random(n) < expit(-2 + 0.8 * x + shock)).astype(int)
    return y, np.round(p, 3), month          # rounding creates ties


def test_fast_auc_and_ks_match_sklearn_with_ties():
    y, p, _ = _data()
    auc, ks = B._Ranked(y, p).auc_ks(np.ones(len(y)))
    fpr, tpr, _ = roc_curve(y, p)
    assert np.isclose(auc, roc_auc_score(y, p))
    assert np.isclose(ks, np.max(tpr - fpr))


def test_frequency_weights_equal_duplicated_rows():
    y, p, _ = _data(3_000)
    w = np.random.default_rng(2).integers(0, 3, len(y)).astype(float)
    rep = np.repeat(np.arange(len(y)), w.astype(int))
    auc_w, _ = B._Ranked(y, p).auc_ks(w)
    assert np.isclose(auc_w, roc_auc_score(y[rep], p[rep]))


def test_month_weights_are_constant_within_month():
    _, _, month = _data(5_000)
    w = B.resample_weights(len(month), month, np.random.default_rng(0))
    assert pd.Series(w).groupby(month).nunique().max() == 1
    assert w.sum() > 0


def test_intervals_and_paired_difference():
    y, p, month = _data(20_000)
    preds = {"scorecard": p, "challenger": p}
    point = B.point_estimates(y, preds)
    draws = B.bootstrap(y, preds, None, n_draws=50, seed=1)
    ci = B.intervals(draws, point)
    g = ci.loc[("scorecard", "gini")]
    assert g["lower"] <= g["estimate"] <= g["upper"]
    diff = B.paired_difference(draws, point, "scorecard", "challenger")
    assert np.allclose(diff[["difference", "lower", "upper"]].to_numpy(), 0)


def test_month_blocks_widen_calibration_intervals_when_months_share_shocks():
    y, p, month = _data(60_000, month_effect=0.3)
    point = B.point_estimates(y, {"m": p})
    width = {}
    for scheme, g in {"loan": None, "month": month}.items():
        ci = B.intervals(B.bootstrap(y, {"m": p}, g, n_draws=200, seed=3), point)
        width[scheme] = ci.loc[("m", "eo_ratio"), "upper"] - ci.loc[("m", "eo_ratio"), "lower"]
    assert width["month"] > 2 * width["loan"]
