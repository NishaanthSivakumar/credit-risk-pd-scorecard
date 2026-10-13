"""Sensitivity analysis on a small synthetic development."""
import numpy as np
import pytest
from make_synthetic_data import make

from pdmodel import calibration as K
from pdmodel import data, model, sensitivity as SE, validation as V


@pytest.fixture(scope="module")
def fitted():
    S = data.split_samples(data.prepare(make(60_000, seed=5)))
    dev = model.develop_scorecard(S["train"])
    card = dev["scorecard"]
    recal = K.Recalibrator("intercept").fit(S["calib"]["default"], card.predict_pd(S["calib"]))
    cm = model.CalibratedScorecard(card, recal, V.grade_edges(card.score(S["train"])))
    return S, cm


def test_adverse_changes_raise_pd(fitted):
    S, cm = fitted
    t = SE.characteristic_shifts(cm, S["oot"])
    assert t.loc["FICO 20 points lower", "relative_change"] > 0
    assert t.loc["All of the above together", "mean_pd"] >= t["mean_pd"].drop("All of the above together").max()


def test_missing_inputs_table_has_every_feature(fitted):
    S, cm = fitted
    t = SE.missing_inputs(cm, S["oot"], S["oot"]["default"])
    assert set(cm.card.features) <= set(t.index)
    assert t.loc["None (baseline)", "gini_change"] == 0


def test_coefficient_uncertainty_is_small_but_positive(fitted):
    S, cm = fitted
    single, joint = SE.coefficient_sensitivity(cm, S["oot"], S["oot"]["default"], n_draws=50)
    assert 0 < joint.loc["mean_pd", "relative_half_width"] < 0.5
    assert len(single) == 2 * (len(cm.card.features) + 1)


def test_threshold_table_is_monotonic():
    r = np.random.default_rng(0)
    p = r.uniform(0, 0.4, 10_000)
    y = (r.random(10_000) < p).astype(int)
    t = SE.threshold_table(y, p)
    assert t["approval_rate"].is_monotonic_increasing
    assert t["defaults_avoided"].is_monotonic_decreasing


def test_calibrated_scorecard_grades_match_validation(fitted):
    S, cm = fitted
    g = cm.grade(S["oot"])
    assert np.array_equal(g, V.assign_grade(cm.score(S["oot"]), cm.edges))
