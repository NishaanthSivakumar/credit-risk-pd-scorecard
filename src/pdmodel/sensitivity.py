"""Sensitivity analysis: how the model's outputs respond when its inputs,
parameters or modelling choices change. Each function returns a table.

`model` is anything with predict_pd(df) and grade(df), normally a
model.CalibratedScorecard.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

from . import calibration as K
from . import config as C
from .model import CalibratedScorecard, develop_scorecard


def _gini(y, p) -> float:
    return 2 * roc_auc_score(y, p) - 1


# ----------------------------------------------------------------------------
# 1. Borrower characteristics
# ----------------------------------------------------------------------------
def _income_down(df, f=0.9):
    d = df.copy()
    d["annual_inc"] = d["annual_inc"] * f
    d["loan_to_income"] = d["loan_to_income"] / f
    return d


def _shift(col, delta, lo=0.0):
    def f(df):
        d = df.copy()
        d[col] = (d[col] + delta).clip(lower=lo)
        return d
    return f


DEFAULT_SHIFTS: dict[str, Callable] = {
    "FICO 20 points lower": _shift("fico", -20, 300),
    "Debt-to-income 5 points higher": _shift("dti", 5),
    "Income 10% lower": _income_down,
    "Revolving utilisation 10 points higher": _shift("revol_util", 10),
    "One more credit inquiry": _shift("inq_last_6mths", 1),
    "Credit history 2 years shorter": _shift("credit_hist_years", -2),
}


def characteristic_shifts(model, df: pd.DataFrame, shifts: dict[str, Callable] | None = None) -> pd.DataFrame:
    """Apply one adverse change to every borrower and measure the PD response.
    The last row applies all changes together."""
    shifts = dict(shifts or DEFAULT_SHIFTS)
    base_pd, base_grade = model.predict_pd(df), model.grade(df)
    singles = list(shifts.values())

    def combined(d):
        for f in singles:
            d = f(d)
        return d
    shifts["All of the above together"] = combined
    rows = []
    for name, f in shifts.items():
        d = f(df)
        p, g = model.predict_pd(d), model.grade(d)
        rows.append({"change": name, "mean_pd": p.mean(), "relative_change": p.mean() / base_pd.mean() - 1,
                     "loans_with_higher_pd": (p > base_pd + 1e-12).mean(),
                     "loans_downgraded": (g > base_grade).mean()})
    out = pd.DataFrame(rows).set_index("change")
    out.loc["Baseline (no change)"] = [base_pd.mean(), 0.0, 0.0, 0.0]
    return out.loc[["Baseline (no change)", *shifts]]


# ----------------------------------------------------------------------------
# 2. Missing inputs
# ----------------------------------------------------------------------------
def missing_inputs(model: CalibratedScorecard, df: pd.DataFrame, y) -> pd.DataFrame:
    """Blank one model input for every loan. The scorecard then uses that input's
    'Missing' WoE (numeric) or the pooled/unseen WoE (categorical)."""
    base_p = model.predict_pd(df)
    rows = [{"input_removed": "None (baseline)", "woe_used": np.nan, "gini": _gini(y, base_p),
             "mean_pd": base_p.mean()}]
    binner = model.card.binner
    for f in model.card.features:
        d = df.copy()
        d[f] = np.nan
        p = model.predict_pd(d)
        b = binner.bins[f]
        woe = b.missing_woe if b.kind == "numeric" else b.cat_woe.get("OTHER", 0.0)
        rows.append({"input_removed": f, "woe_used": woe, "gini": _gini(y, p), "mean_pd": p.mean()})
    t = pd.DataFrame(rows).set_index("input_removed")
    t["gini_change"] = t["gini"] - t.loc["None (baseline)", "gini"]
    t["mean_pd_change"] = t["mean_pd"] / t.loc["None (baseline)", "mean_pd"] - 1
    return t


# ----------------------------------------------------------------------------
# 3. Coefficients
# ----------------------------------------------------------------------------
def coefficient_sensitivity(model: CalibratedScorecard, df: pd.DataFrame, y,
                            n_draws: int = 500, seed: int = C.RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(a) Move each coefficient by +/- 1 standard error, one at a time.
    (b) Draw whole coefficient vectors from their estimated covariance (parameter
    uncertainty) and report the spread of mean PD and Gini."""
    card = model.card
    X = sm.add_constant(card.binner.transform(df, card.features), has_constant="add")[["const", *card.features]]
    X = X.to_numpy()
    beta, cov = card.res.params.to_numpy(), card.res.cov_params().to_numpy()
    se, names = np.sqrt(np.diag(cov)), list(card.res.params.index)

    def pd_of(b):
        return model.recal.transform(1 / (1 + np.exp(-(X @ b))))

    base = pd_of(beta)
    rows = []
    for i, name in enumerate(names):
        for sign in (-1, 1):
            b = beta.copy()
            b[i] += sign * se[i]
            p = pd_of(b)
            rows.append({"coefficient": name, "shift": f"{sign:+d} SE", "mean_pd": p.mean(),
                         "mean_pd_change": p.mean() / base.mean() - 1, "gini": _gini(y, p)})
    one_at_a_time = pd.DataFrame(rows).set_index(["coefficient", "shift"])

    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(beta, cov, size=n_draws)
    stats = []
    for b in draws:
        p = pd_of(b)
        stats.append((p.mean(), _gini(y, p)))
    s = pd.DataFrame(stats, columns=["mean_pd", "gini"])
    joint = pd.DataFrame({
        "base": [base.mean(), _gini(y, base)],
        "p2.5": s.quantile(0.025), "p97.5": s.quantile(0.975),
    }, index=["mean_pd", "gini"])
    joint["relative_half_width"] = (joint["p97.5"] - joint["p2.5"]) / 2 / joint["base"]
    return one_at_a_time, joint


# ----------------------------------------------------------------------------
# 4. Probability thresholds
# ----------------------------------------------------------------------------
def threshold_table(y, p, cutoffs=(0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30)) -> pd.DataFrame:
    """Approve when PD < cut-off. What a lender gives up and gains at each cut-off."""
    y, p = np.asarray(y), np.asarray(p)
    rows = []
    for c in cutoffs:
        ok = p < c
        rows.append({"pd_cutoff": c, "approval_rate": ok.mean(),
                     "bad_rate_approved": y[ok].mean() if ok.any() else np.nan,
                     "bad_rate_declined": y[~ok].mean() if (~ok).any() else np.nan,
                     "defaults_avoided": y[~ok].sum() / y.sum(),
                     "goods_declined": (~ok & (y == 0)).sum() / (y == 0).sum()})
    return pd.DataFrame(rows).set_index("pd_cutoff")


# ----------------------------------------------------------------------------
# 5. Sub-populations with shifted feature distributions
# ----------------------------------------------------------------------------
def default_segments(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "All loans": np.ones(len(df), bool),
        "FICO below 680": (df["fico"] < 680).to_numpy(),
        "FICO 740 and above": (df["fico"] >= 740).to_numpy(),
        "Debt-to-income above 25": (df["dti"] > 25).to_numpy(),
        "Income below $40k": (df["annual_inc"] < 40_000).to_numpy(),
        "Income $100k and above": (df["annual_inc"] >= 100_000).to_numpy(),
        "Renters": (df["home_ownership"] == "RENT").to_numpy(),
        "Mortgage holders": (df["home_ownership"] == "MORTGAGE").to_numpy(),
        "Debt consolidation": (df["purpose"] == "debt_consolidation").to_numpy(),
        "Credit card refinancing": (df["purpose"] == "credit_card").to_numpy(),
        "Small business": (df["purpose"] == "small_business").to_numpy(),
        "Loan $25k and above": (df["loan_amnt"] >= 25_000).to_numpy(),
    }


def segment_performance(y, p, segments: dict[str, np.ndarray]) -> pd.DataFrame:
    y, p = np.asarray(y), np.asarray(p)
    rows = []
    for name, m in segments.items():
        if m.sum() == 0:
            continue
        ys, ps = y[m], p[m]
        rows.append({"segment": name, "n": int(m.sum()), "share": m.mean(), "mean_pd": ps.mean(),
                     "observed_dr": ys.mean(), "eo_ratio": ps.mean() / ys.mean(),
                     "gini": _gini(ys, ps) if 0 < ys.sum() < len(ys) else np.nan})
    return pd.DataFrame(rows).set_index("segment")


# ----------------------------------------------------------------------------
# 6. Alternative modelling assumptions
# ----------------------------------------------------------------------------
def alternative_specifications(S: dict[str, pd.DataFrame], recal_method: str = "intercept") -> pd.DataFrame:
    """Re-run the whole development with one modelling choice changed, recalibrate
    on the calibration sample the same way, and score the out-of-time sample."""
    dev = pd.concat([S["train"], S["test"]], ignore_index=True)
    specs = {
        "Baseline (as developed)": dict(train=S["train"]),
        "At most 5 starting bins": dict(train=S["train"], binner_kwargs={"max_bins": 5}),
        "Minimum bin size 10%": dict(train=S["train"], binner_kwargs={"min_bin_frac": 0.10}),
        "No monotonic constraint": dict(train=S["train"], binner_kwargs={"monotonic": False}),
        "Keep mostly-missing features (mort_acc)": dict(train=S["train"], max_missing=1.0),
        "Train on all 2007–13 loans": dict(train=dev),
        "Train on 2007–12 only (chronological)": dict(train=dev[dev["issue_year"] <= C.DEV_LAST_YEAR - 1]),
    }
    y_cal, y_oot = S["calib"]["default"].to_numpy(), S["oot"]["default"].to_numpy()
    rows = []
    for name, spec in specs.items():
        train = spec.pop("train")
        dev_out = develop_scorecard(train, **spec)
        card = dev_out["scorecard"]
        recal = K.Recalibrator(recal_method).fit(y_cal, card.predict_pd(S["calib"]))
        p = recal.transform(card.predict_pd(S["oot"]))
        m = K.calibration_metrics(y_oot, p)
        rows.append({"specification": name, "train_loans": len(train),
                     "n_features": len(dev_out["features"]), "gini_oot": _gini(y_oot, p),
                     "brier_oot": m["brier"], "eo_ratio_oot": m["eo_ratio"], "cal_slope_oot": m["cal_slope"],
                     "features": ", ".join(dev_out["features"])})
    return pd.DataFrame(rows).set_index("specification")
