"""Fast unit tests:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdmodel import validation as V  # noqa: E402
from pdmodel.woe import WoEBinner  # noqa: E402


def _toy(n=20_000, seed=0):
    r = np.random.default_rng(seed)
    x = r.normal(size=n)
    y = (r.random(n) < 1 / (1 + np.exp(-(-1.5 + 0.8 * x)))).astype(int)
    x[r.random(n) < 0.03] = np.nan
    cat = r.choice(list("ABCD"), n)
    return pd.DataFrame({"x": x, "cat": cat}), pd.Series(y)


def test_woe_is_monotonic_and_missing_has_own_bin():
    X, y = _toy()
    b = WoEBinner().fit(X, y, ["x"], ["cat"])
    t = b.bins["x"].table
    rates = t.loc[t["bin"] != "Missing", "bad_rate"].to_numpy()
    assert np.all(np.diff(rates) >= 0)          # risk rises with x
    assert "Missing" in set(t["bin"])
    assert b.transform(X).notna().all().all()


def test_psi_zero_for_identical_and_positive_for_shift():
    a = np.random.default_rng(1).normal(size=10_000)
    assert V.psi(a, a) < 1e-9
    assert V.psi(a, a + 0.5) > 0.1


def test_gini_of_perfect_model_is_one():
    y = np.array([0, 0, 1, 1])
    assert abs(V.discrimination(y, np.array([0.1, 0.2, 0.8, 0.9]))["gini"] - 1) < 1e-12


def test_grades_one_is_best():
    edges = V.grade_edges(np.arange(100), n_grades=4)
    g = V.assign_grade(np.array([0, 99]), edges)
    assert g[0] == 4 and g[1] == 1
