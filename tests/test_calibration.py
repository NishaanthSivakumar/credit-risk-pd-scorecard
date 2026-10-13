"""Calibration measures and recalibration."""
import numpy as np
from scipy.special import expit

from pdmodel import calibration as K
from pdmodel import validation as V


def _data(n=200_000, seed=0):
    r = np.random.default_rng(seed)
    lp = r.normal(-2, 0.8, n)
    p = expit(lp)
    y = (r.random(n) < p).astype(int)
    return y, p


def test_calibrated_predictions_have_zero_intercept_and_unit_slope():
    y, p = _data()
    assert abs(K.calibration_intercept(y, p)) < 0.03
    assert abs(K.calibration_slope(y, p)[1] - 1) < 0.03


def test_intercept_recalibration_fixes_a_level_shift():
    y, p = _data()
    low = expit(K.logit(p) - 0.5)                  # PDs too low
    assert abs(K.calibration_intercept(y, low) - 0.5) < 0.03
    r = K.Recalibrator("intercept").fit(y, low)
    assert np.isclose(r.transform(low).mean(), y.mean(), rtol=1e-6)   # MLE matches the mean exactly


def test_slope_recalibration_fixes_overspread_pds():
    y, p = _data()
    spread = expit(2 * K.logit(p))
    _, b = K.calibration_slope(y, spread)
    assert abs(b - 0.5) < 0.03
    r = K.Recalibrator("intercept_slope").fit(y, spread)
    assert abs(K.calibration_slope(y, r.transform(spread))[1] - 1) < 1e-6


def test_recalibration_keeps_ranking():
    y, p = _data(50_000)
    low = expit(2 * K.logit(p) - 0.3)
    r = K.Recalibrator("intercept_slope").fit(y, low)
    g0 = V.discrimination(y, low)["gini"]
    assert np.isclose(V.discrimination(y, r.transform(low))["gini"], g0)


def test_weights_equal_duplicated_rows():
    y, p = _data(5_000)
    w = np.random.default_rng(1).integers(0, 3, len(y)).astype(float)
    rep = np.repeat(np.arange(len(y)), w.astype(int))
    assert np.isclose(K.calibration_slope(y, p, w)[1], K.calibration_slope(y[rep], p[rep])[1])


def test_choose_method():
    y, p = _data()
    assert K.choose_method(y, p, 0.10)[0] == "none"
    assert K.choose_method(y, expit(K.logit(p) - 0.1), 0.10)[0] == "intercept"
    assert K.choose_method(y, expit(2 * K.logit(p)), 0.10)[0] == "intercept_slope"
