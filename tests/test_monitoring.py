"""Monitoring: early default, production sample, stability, alerts and triggers."""
import numpy as np
import pandas as pd
import pytest
from make_synthetic_data import make

from pdmodel import calibration as K
from pdmodel import data, model
from pdmodel import monitoring as MON
from pdmodel import validation as V


@pytest.fixture(scope="module")
def setup():
    raw = make(60_000, seed=7)
    S = data.split_samples(data.prepare(raw))
    card = model.develop_scorecard(S["train"])["scorecard"]
    recal = K.Recalibrator("intercept").fit(S["calib"]["default"], card.predict_pd(S["calib"]))
    cm = model.CalibratedScorecard(card, recal, V.grade_edges(card.score(S["train"])))
    return raw, S, cm


def test_rag_bands():
    assert MON.rag(0.05, (0.1, 0.25)) == "GREEN"
    assert MON.rag(-0.15, (0.1, 0.25)) == "AMBER"
    assert MON.rag(0.3, (0.1, 0.25)) == "RED" and MON.rag(np.nan, (0.1, 0.25)) == ""


def test_early_default_flag():
    df = pd.DataFrame({
        "issue_d": pd.to_datetime(["2016-01-01"] * 4),
        "last_pymnt_d": pd.to_datetime(["2016-08-01", "2017-06-01", "2016-05-01", pd.NaT]),
        "loan_status": ["Charged Off", "Charged Off", "Current", "Late (31-120 days)"],
    })
    assert data.early_default_flag(df, 12).tolist() == [1, 0, 0, 1]


def test_production_sample_keeps_every_status(setup):
    raw, _, _ = setup
    prod = data.prepare_production(raw)
    assert prod["issue_year"].between(2016, 2018).all()
    assert "Current" in set(prod["loan_status"]) and "default" not in prod
    assert (prod["months_observed"] > 0).all()


def test_identical_population_is_stable(setup):
    _, S, cm = setup
    same = S["train"].assign(issue_d=pd.Timestamp("2013-01-01"))
    t = MON.stability_by_quarter(cm, S["train"], same)
    assert t["score_psi"].max() < 1e-9 and t["max_csi"].max() < 1e-9


def test_out_of_range_detection():
    train = pd.DataFrame({"issue_d": pd.to_datetime(["2013-01-01"] * 3), "dti": [5.0, 20.0, 35.0]})
    new = pd.DataFrame({"issue_d": pd.to_datetime(["2016-01-01"] * 4), "dti": [10.0, 30.0, 40.0, 50.0]})
    t = MON.data_quality_by_quarter(train, new, ["dti"])
    assert np.isclose(t.loc["2016Q1", "out_of_range_dti"], 0.5)


def test_early_performance_keeps_matured_and_drops_young_quarters(setup):
    _, S, cm = setup
    matured = S["oot"].assign(months_observed=np.nan)
    young = S["oot"].assign(issue_d=pd.Timestamp("2018-10-01"), months_observed=5)
    t = MON.early_performance_by_quarter(cm, pd.concat([matured, young]), benchmark_ratio=0.4)
    assert "2018Q4" not in t.index and len(t) > 0


def test_triggers_follow_the_grid():
    q = ["2015Q1", "2015Q2", "2015Q3", "2015Q4"]
    grid = pd.DataFrame({m: ["GREEN"] * 4 for m in MON.METRICS}, index=q)
    perf = pd.DataFrame({"grades_failing": [0, 0, 0, 0]}, index=q)
    assert not MON.triggers(grid, perf)["fired"].any()
    grid.loc["2015Q4", "Calibration gap"] = "AMBER"
    grid.loc[["2015Q2", "2015Q3"], "Early default vs PD"] = "RED"
    t = MON.triggers(grid, perf)["fired"]
    assert t.loc["Recalibrate"].sum() == 2 and not t.loc["Redevelop"].any()
