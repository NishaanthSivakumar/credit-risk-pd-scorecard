"""Target definition, sample design and leakage checks."""
import numpy as np
import pytest
from make_synthetic_data import make

from pdmodel import audit, data
from pdmodel import config as C


@pytest.fixture(scope="module")
def raw():
    return make(40_000, seed=3)


@pytest.fixture(scope="module")
def prepared(raw):
    return data.prepare(raw)


def test_target_maps_statuses(raw):
    r = raw.copy()
    r.loc[r.index[:4], "loan_status"] = ["Fully Paid", "Charged Off", "Default", "Current"]
    r.loc[r.index[:4], ["term", "issue_d"]] = [" 36 months", "Jan-2014"]
    df = data.prepare(r.iloc[:4])
    assert df["default"].tolist() == [0, 1, 1]          # Current dropped


def test_only_modelled_term_and_vintages(prepared):
    assert prepared["issue_year"].between(C.FIRST_ISSUE_YEAR, C.LAST_ISSUE_YEAR).all()
    assert set(prepared["default"].unique()) <= {0, 1}


def test_features_are_not_post_origination():
    feats = set(C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES)
    assert not feats & C.LEAKAGE_COLUMNS
    assert not feats & set(C.OUTCOME_COLUMNS)


def test_leakage_check_flags_outcome_fields():
    t = audit.leakage_check(["fico", "recoveries", "total_rec_prncp"])
    assert t["post_origination"].tolist() == [False, True, True]


def test_samples_are_chronological_and_disjoint(prepared):
    S = data.split_samples(prepared)
    assert S["train"]["issue_year"].max() <= C.DEV_LAST_YEAR
    assert (S["calib"]["issue_year"] == C.CALIB_YEAR).all()
    assert (S["oot"]["issue_year"] > C.CALIB_YEAR).all()
    assert sum(len(v) for v in S.values()) == len(prepared)


def test_audit_critical_checks_pass_on_clean_data(raw, prepared):
    S = data.split_samples(prepared)
    checks, tabs = audit.run_checks(raw, prepared, S, ["fico", "dti"])
    failed = [c.name for c in checks if c.critical and not c.passed]
    assert failed == []
    assert np.isclose(tabs["target_mapping"]["share"].sum(), 1)
    assert set(tabs["target_mapping"]["treatment"]) <= {"default", "non-default", "unresolved (excluded)"}


def test_prepayment_profile_counts_early_payers(prepared):
    t = audit.prepayment_profile(prepared)
    assert 0 <= t.loc["all", "repaid_3m_early_share"] <= 1
    assert t.loc["all", "fully_paid"] == (prepared["default"] == 0).sum()
