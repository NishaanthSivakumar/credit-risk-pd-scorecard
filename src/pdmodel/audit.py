"""Data audit: target definition, leakage, sample maturity and data quality.

Every check returns evidence (a table) and a pass/fail flag, so the audit can be
re-run on each data refresh and its results feed the validation assessment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C

# Plausible ranges for application fields. Values outside are counted, not
# removed: WoE binning puts extremes in the end bins, so they cannot distort the fit.
PLAUSIBLE_RANGES = {
    "loan_amnt": (500, 40_000),
    "annual_inc": (1, 10_000_000),
    "loan_to_income": (0, 5),
    "dti": (0, 100),
    "fico": (300, 850),
    "credit_hist_years": (0, 80),
    "emp_length_yrs": (0, 10),
    "inq_last_6mths": (0, 40),
    "open_acc": (0, 100),
    "total_acc": (0, 200),
    "revol_bal": (0, 5_000_000),
    "revol_util": (0, 200),
    "delinq_2yrs": (0, 50),
    "pub_rec": (0, 100),
    "mort_acc": (0, 100),
}


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    dimension: str  # "data quality" or "target definition"
    critical: bool = True  # a failed critical check is RED; a failed minor check is AMBER


def _thirty_six_month(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw[["issue_d", "term", "loan_status"]].copy()
    d["issue_d"] = pd.to_datetime(d["issue_d"], format="%b-%Y", errors="coerce")
    d = d.dropna(subset=["issue_d"])
    d["issue_year"] = d["issue_d"].dt.year
    d["loan_status"] = d["loan_status"].astype(str).str.strip()
    return d[d["term"].astype(str).str.strip() == C.TERM]


def _outcome_class(status: pd.Series) -> pd.Series:
    return pd.Series(np.where(status.isin(C.BAD_STATUSES), "default",
                     np.where(status.isin(C.GOOD_STATUSES), "non-default", "unresolved (excluded)")),
                     index=status.index)


def target_mapping(raw: pd.DataFrame) -> pd.DataFrame:
    """Every loan status in the modelled vintages and how the target treats it."""
    d = _thirty_six_month(raw)
    d = d[d["issue_year"].between(C.FIRST_ISSUE_YEAR, C.LAST_ISSUE_YEAR)]
    t = d["loan_status"].value_counts().rename("n_loans").to_frame()
    t["treatment"] = _outcome_class(t.index.to_series())
    t["share"] = t["n_loans"] / t["n_loans"].sum()
    return t[["treatment", "n_loans", "share"]]


def resolution_by_vintage(raw: pd.DataFrame) -> pd.DataFrame:
    """Resolved vs unresolved 36-month loans by issue year, including the vintages
    excluded because they had not matured by the snapshot."""
    d = _thirty_six_month(raw)
    d["outcome"] = _outcome_class(d["loan_status"])
    t = pd.crosstab(d["issue_year"], d["outcome"])
    for c in ["default", "non-default", "unresolved (excluded)"]:
        if c not in t:
            t[c] = 0
    t["n_loans"] = t.sum(axis=1)
    t["unresolved_share"] = t["unresolved (excluded)"] / t["n_loans"]
    t["default_rate_resolved"] = t["default"] / (t["default"] + t["non-default"]).replace(0, np.nan)
    t["modelled"] = t.index.to_series().between(C.FIRST_ISSUE_YEAR, C.LAST_ISSUE_YEAR)
    return t[["n_loans", "default", "non-default", "unresolved (excluded)", "unresolved_share",
              "default_rate_resolved", "modelled"]]


def leakage_check(features: list[str]) -> pd.DataFrame:
    """Every model input against the list of post-origination fields."""
    banned = C.LEAKAGE_COLUMNS | set(C.OUTCOME_COLUMNS)
    return pd.DataFrame({"feature": features,
                         "post_origination": [f in banned for f in features]}).set_index("feature")


def observation_window(df: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    """Months between the latest issue date of each vintage and the data snapshot."""
    g = df.groupby("issue_year")["issue_d"].agg(first_issue="min", last_issue="max", n_loans="size")
    g["months_to_snapshot"] = ((snapshot.year - g["last_issue"].dt.year) * 12
                               + (snapshot.month - g["last_issue"].dt.month))
    g["full_term_observed"] = g["months_to_snapshot"] >= 36
    return g


def prepayment_profile(df: pd.DataFrame) -> pd.DataFrame:
    """How long loans ran before they were repaid or charged off, by vintage.
    A loan repaid early is a non-default: it met its obligation in full."""
    months = ((df["last_pymnt_d"].dt.year - df["issue_d"].dt.year) * 12
              + (df["last_pymnt_d"].dt.month - df["issue_d"].dt.month))
    d = pd.DataFrame({"issue_year": df["issue_year"], "default": df["default"], "months": months})
    paid, bad = d[d["default"] == 0], d[d["default"] == 1]
    t = pd.DataFrame({
        "fully_paid": paid.groupby("issue_year").size(),
        "repaid_3m_early_share": paid.groupby("issue_year")["months"].apply(lambda m: (m < 33).mean()),
        "median_months_to_payoff": paid.groupby("issue_year")["months"].median(),
        "median_months_to_last_payment_defaults": bad.groupby("issue_year")["months"].median(),
    })
    t.loc["all"] = [len(paid), (paid["months"] < 33).mean(), paid["months"].median(),
                    bad["months"].median()]
    return t


def feature_quality(samples: dict[str, pd.DataFrame],
                    features=C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES) -> pd.DataFrame:
    """Missing share per sample, plus out-of-range counts across all samples."""
    allx = pd.concat(samples.values(), ignore_index=True)
    rows = []
    for f in features:
        r = {"feature": f, "dtype": "numeric" if f in C.NUMERIC_FEATURES else "categorical"}
        for k, d in samples.items():
            r[f"missing_{k}"] = d[f].isna().mean()
        if f in PLAUSIBLE_RANGES:
            lo, hi = PLAUSIBLE_RANGES[f]
            x = allx[f].dropna()
            r["range"] = f"[{lo:,}, {hi:,}]"
            r["out_of_range"] = int(((x < lo) | (x > hi)).sum())
        else:
            r["range"], r["out_of_range"] = "", 0
            r["n_levels"] = allx[f].nunique()
        rows.append(r)
    return pd.DataFrame(rows).set_index("feature")


def sample_boundaries(samples: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({k: {"first_issue": d["issue_d"].min(), "last_issue": d["issue_d"].max(),
                             "n_loans": len(d)} for k, d in samples.items()}).T


def run_checks(raw: pd.DataFrame, df: pd.DataFrame, samples: dict[str, pd.DataFrame],
               model_features: list[str]) -> tuple[list[Check], dict[str, pd.DataFrame]]:
    """Run every audit check. Returns the checks and their evidence tables."""
    # Data snapshot = the latest payment recorded anywhere in the file
    snapshot = pd.to_datetime(raw["last_pymnt_d"], format="%b-%Y", errors="coerce").max()
    tabs = {
        "target_mapping": target_mapping(raw),
        "resolution_by_vintage": resolution_by_vintage(raw),
        "leakage": leakage_check(sorted(set(C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES))),
        "observation_window": observation_window(df, snapshot),
        "prepayment": prepayment_profile(df),
        "feature_quality": feature_quality(samples),
        "sample_boundaries": sample_boundaries(samples),
    }
    res, tm = tabs["resolution_by_vintage"], tabs["target_mapping"]
    modelled = res[res["modelled"]]
    unresolved = modelled["unresolved (excluded)"].sum() / modelled["n_loans"].sum()
    worst_year = modelled["unresolved_share"].idxmax()
    b = tabs["sample_boundaries"]
    chrono = (max(b.loc["train", "last_issue"], b.loc["test", "last_issue"]) < b.loc["calib", "first_issue"]
              and b.loc["calib", "last_issue"] < b.loc["oot", "first_issue"])
    fq = tabs["feature_quality"]
    n_all = sum(len(d) for d in samples.values())
    oor = fq["out_of_range"].sum() / n_all
    paid_ok = all(s in C.GOOD_STATUSES for s in tm.index[tm["treatment"] == "non-default"])
    bad_ok = set(tm.index[tm["treatment"] == "default"]) <= C.BAD_STATUSES

    checks = [
        Check("Fully paid loans are non-defaults",
              paid_ok and bool((df.loc[df["default"] == 0, "total_rec_prncp"]
                                >= df.loc[df["default"] == 0, "funded_amnt"] - 1).mean() > 0.99),
              "Both 'Fully Paid' statuses map to 0, and over 99% of them repaid the funded principal",
              "target definition"),
        Check("Charged-off and defaulted loans are defaults", bad_ok,
              ", ".join(sorted(C.BAD_STATUSES)) + " map to 1", "target definition"),
        Check("Unresolved loans excluded, and rare in modelled vintages",
              unresolved < C.UNRESOLVED_GREEN,
              f"{unresolved:.2%} of 2007–{C.LAST_ISSUE_YEAR} 36-month loans unresolved "
              f"(highest: {worst_year}, {modelled.loc[worst_year, 'unresolved_share']:.2%})",
              "target definition", critical=False),
        Check("Every modelled vintage observed for its full 36-month term",
              bool(tabs["observation_window"]["full_term_observed"].all()),
              f"Snapshot {snapshot:%b %Y}; shortest window "
              f"{int(tabs['observation_window']['months_to_snapshot'].min())} months",
              "target definition"),
        Check("No post-origination field among the candidate features",
              not tabs["leakage"]["post_origination"].any(),
              f"{len(tabs['leakage'])} candidate features checked against "
              f"{len(C.LEAKAGE_COLUMNS | set(C.OUTCOME_COLUMNS))} outcome fields",
              "data quality"),
        Check("Final model uses no post-origination field",
              not set(model_features) & (C.LEAKAGE_COLUMNS | set(C.OUTCOME_COLUMNS)),
              f"{len(model_features)} model features checked", "data quality"),
        Check("Samples separated in time (train/test < calibration < out-of-time)", chrono,
              " · ".join(f"{k} {b.loc[k, 'first_issue']:%Y-%m}–{b.loc[k, 'last_issue']:%Y-%m}" for k in b.index),
              "data quality"),
        Check("Out-of-range values below 0.1% of loans", oor < 0.001,
              f"{int(fq['out_of_range'].sum()):,} out-of-range values across {n_all:,} loans", "data quality",
              critical=False),
        Check("Final features at most 10% missing in training",
              bool((fq.loc[[f for f in model_features if f in fq.index], "missing_train"]
                    <= C.MAX_MISSING_FRAC).all()),
              "Highest: " + ", ".join(f"{f} {v:.1%}" for f, v in
                                      fq.loc[model_features, "missing_train"].nlargest(2).items()),
              "data quality", critical=False),
    ]
    return checks, tabs


def checks_table(checks: list[Check]) -> pd.DataFrame:
    return pd.DataFrame([{"check": c.name, "result": "PASS" if c.passed else "FAIL",
                          "evidence": c.detail, "dimension": c.dimension,
                          "severity": "critical" if c.critical else "minor"} for c in checks]).set_index("check")
