"""Loss measurement and expected loss."""
import numpy as np
import pandas as pd

from pdmodel import portfolio as P


def _loans():
    return pd.DataFrame({
        "default":                 [0,      1,      1,      1],
        "funded_amnt":             [10_000, 10_000, 20_000, 5_000],
        "total_rec_prncp":         [10_000, 4_000,  5_000,  5_000],
        "recoveries":              [0,      1_000,  3_000,  0],
        "collection_recovery_fee": [0,      100,    500,    0],
        "issue_year":              [2014] * 4,
    })


def test_realised_loss_fields():
    f = P.realised_loss_fields(_loans())
    assert f["loss"].tolist() == [0, 6_000 - 900, 15_000 - 2_500, 0]
    assert (f["loss"] <= f["ead"]).all()


def test_parameters_are_exposure_weighted():
    t = P.estimate_loss_parameters(_loans(), np.array([1, 1, 1, 1]))
    ead, funded, loss = 6_000 + 15_000 + 0, 35_000, 5_100 + 12_500
    assert np.isclose(t.loc[1, "ead_ratio"], ead / funded)
    assert np.isclose(t.loc[1, "lgd"], loss / ead)


def _portfolio(n=1_000, seed=0):
    r = np.random.default_rng(seed)
    return pd.DataFrame({"exposure": r.uniform(1_000, 30_000, n), "pd": r.uniform(0.02, 0.3, n),
                         "lgd": 0.9, "ead_ratio": 0.6, "purpose": r.choice(["a", "b"], n),
                         "grade": r.integers(1, 8, n), "size_band": "x"}).assign(
        ead=lambda d: d["exposure"] * d["ead_ratio"], el=lambda d: d["pd"] * d["lgd"] * d["ead"])


def test_el_sensitivity_base_cell_equals_el_rate():
    p = _portfolio()
    s = P.el_sensitivity(p)
    assert np.isclose(s.loc["PD x1", "LGD +0%"], p["el"].sum() / p["exposure"].sum())
    assert s.loc["PD x1.5", "LGD +10%"] > s.loc["PD x1", "LGD +0%"]


def test_equal_exposures_give_hhi_of_one_over_n():
    p = _portfolio(500).assign(exposure=1.0)
    c = P.concentration(p)
    assert np.isclose(c.loc["Herfindahl index, loans", "value"], 1 / 500)
