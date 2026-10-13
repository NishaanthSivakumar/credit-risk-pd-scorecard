"""PD calibration measures and recalibration.

Calibration asks whether predicted PDs are the right *level*, separately from
whether they rank borrowers correctly. Two standard summaries:

  calibration intercept (calibration-in-the-large)
      a in  logit P(default) = a + logit(PD)          ideal 0; > 0 means PDs too low
  calibration slope
      b in  logit P(default) = a + b * logit(PD)      ideal 1; < 1 means PDs too spread out

Recalibration re-uses the same two regressions as a fix: fit them on a dedicated
calibration sample, then map every PD through them. Both are monotonic, so the
ranking (AUC, Gini, KS) is unchanged; only the PD levels move.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

EPS = 1e-9


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None,
                 offset: np.ndarray | None = None, beta0=None, iters: int = 50,
                 tol: float = 1e-8) -> np.ndarray:
    """Weighted logistic regression by Newton-Raphson, for the 1-2 parameter fits
    used here (fast enough to repeat inside a bootstrap)."""
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] != len(y):
        X = X.T
    y = np.asarray(y, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, float)
    if (w == 0).any():               # bootstrap resamples: skip loans that were not drawn
        keep = w > 0
        X, y, w, off = X[keep], y[keep], w[keep], off[keep]
    beta = np.zeros(X.shape[1]) if beta0 is None else np.array(beta0, float)

    def loglik(bt):
        eta = X @ bt + off
        return float(np.sum(w * (y * eta - np.logaddexp(0, eta))))

    ll = loglik(beta)
    for _ in range(iters):
        p = expit(X @ beta + off)
        grad = X.T @ (w * (y - p))
        hess = (X * (w * p * (1 - p))[:, None]).T @ X
        step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        t = 1.0                      # step-halving keeps Newton from overshooting
        new_ll = loglik(beta + step)
        while t > 1e-6 and new_ll < ll - 1e-12:
            t /= 2
            new_ll = loglik(beta + t * step)
        beta, ll = beta + t * step, new_ll
        if np.max(np.abs(t * step)) < tol:
            break
    return beta


def calibration_intercept(y, pd_pred, w=None, start: float = 0.0) -> float:
    """Calibration-in-the-large: intercept with logit(PD) as a fixed offset."""
    n = len(y)
    return float(fit_logistic(np.ones((n, 1)), y, w, offset=logit(pd_pred), beta0=[start])[0])


def calibration_slope(y, pd_pred, w=None, start=(0.0, 1.0)) -> tuple[float, float]:
    """(intercept, slope) of the outcome regressed on logit(PD)."""
    lp = logit(pd_pred)
    a, b = fit_logistic(np.column_stack([np.ones(len(lp)), lp]), y, w, beta0=list(start))
    return float(a), float(b)


class Recalibrator:
    """method = "none" | "intercept" | "intercept_slope"."""

    def __init__(self, method: str = "intercept"):
        if method not in {"none", "intercept", "intercept_slope"}:
            raise ValueError(f"unknown recalibration method: {method}")
        self.method, self.a, self.b = method, 0.0, 1.0

    def fit(self, y, pd_pred):
        if self.method == "intercept":
            self.a = calibration_intercept(y, pd_pred)
        elif self.method == "intercept_slope":
            self.a, self.b = calibration_slope(y, pd_pred)
        return self

    def transform(self, pd_pred) -> np.ndarray:
        if self.method == "none":
            return np.asarray(pd_pred, float)
        return expit(self.a + self.b * logit(pd_pred))

    def __repr__(self):
        return f"Recalibrator({self.method}, a={self.a:+.4f}, b={self.b:.4f})"


def log_loss(y, p, w=None) -> float:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), EPS, 1 - EPS)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.average(ll, weights=w))


def brier(y, p, w=None) -> float:
    return float(np.average((np.asarray(p, float) - np.asarray(y, float)) ** 2, weights=w))


def calibration_metrics(y, p) -> dict:
    y = np.asarray(y, float)
    a, b = calibration_slope(y, p)
    return {
        "mean_pd": float(np.mean(p)),
        "observed_dr": float(y.mean()),
        "eo_ratio": float(np.mean(p) / y.mean()),
        "brier": brier(y, p),
        "log_loss": log_loss(y, p),
        "cal_intercept": calibration_intercept(y, p),
        "cal_slope": b,
    }


def decile_table(y, p, n_groups=10) -> pd.DataFrame:
    """Expected vs observed default rate by PD decile."""
    d = pd.DataFrame({"y": np.asarray(y), "pd": np.asarray(p)})
    d["decile"] = pd.qcut(d["pd"].rank(method="first"), n_groups, labels=range(1, n_groups + 1))
    t = d.groupby("decile", observed=True).agg(n=("y", "size"), mean_pd=("pd", "mean"),
                                                observed_dr=("y", "mean"))
    t["eo_ratio"] = t["mean_pd"] / t["observed_dr"]
    return t


def slope_standard_error(y, pd_pred) -> float:
    """Standard error of the calibration slope (inverse Hessian at the fit)."""
    lp = logit(pd_pred)
    X = np.column_stack([np.ones(len(lp)), lp])
    a, b = calibration_slope(y, pd_pred)
    p = expit(X @ np.array([a, b]))
    hess = (X * (p * (1 - p))[:, None]).T @ X
    return float(np.sqrt(np.linalg.inv(hess)[1, 1]))


def intercept_standard_error(y, pd_pred) -> float:
    """Standard error of the calibration intercept (logit(PD) as offset)."""
    a = calibration_intercept(y, pd_pred)
    p = expit(a + logit(pd_pred))
    return float(1 / np.sqrt(np.sum(p * (1 - p))))


def choose_method(y_cal, pd_cal, slope_material: float) -> tuple[str, str]:
    """Pick the recalibration from the calibration sample alone (never the test sample):
      intercept_slope  if the slope differs from 1 by more than `slope_material`
                       and significantly (|z| > 1.96)
      intercept        else, if the calibration intercept differs significantly from 0
      none             otherwise
    A detected bias in the PD level is always corrected: validation tolerances decide
    whether a model passes, not whether a measured bias is worth removing."""
    _, b = calibration_slope(y_cal, pd_cal)
    b_se = slope_standard_error(y_cal, pd_cal)
    a = calibration_intercept(y_cal, pd_cal)
    a_se = intercept_standard_error(y_cal, pd_cal)
    if abs(b - 1) > slope_material and abs(b - 1) / b_se > 1.96:
        return "intercept_slope", f"calibration slope {b:.3f} (s.e. {b_se:.3f}) differs materially from 1"
    if abs(a) / a_se > 1.96:
        return "intercept", (f"calibration intercept {a:+.3f} (s.e. {a_se:.3f}) differs significantly from 0, "
                             f"while the slope {b:.3f} is within {slope_material} of 1")
    return "none", f"calibration intercept {a:+.3f} (s.e. {a_se:.3f}) and slope {b:.3f} consistent with no change"
