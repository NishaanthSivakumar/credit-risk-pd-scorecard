"""Stress scenarios on the one-factor model."""
import numpy as np
import pandas as pd
from scipy.special import ndtri

from pdmodel import simulation as MC
from pdmodel import stress as ST


class _DtiModel:
    """Toy model: PD rises with debt-to-income."""
    def predict_pd(self, df):
        return np.clip(0.05 + 0.004 * df["dti"].to_numpy(), 0, 1)


def _book(n=5_000, seed=0):
    r = np.random.default_rng(seed)
    df = pd.DataFrame({"dti": r.uniform(5, 35, n), "revol_bal": r.uniform(0, 30_000, n),
                       "annual_inc": r.uniform(20_000, 150_000, n)})
    port = pd.DataFrame({"pd": _DtiModel().predict_pd(df), "lgd": 0.9, "ead_ratio": 0.6,
                         "exposure": r.uniform(2_000, 30_000, n), "grade": r.integers(1, 8, n)})
    return port, df


def test_scenario_quantile_maps_to_z():
    assert ST.Scenario("x", economy_quantile=0.5).z == 0
    assert np.isclose(ST.Scenario("x", economy_quantile=0.1).z, ndtri(0.1))
    assert ST.Scenario("x").z is None and ST.Scenario("x", economy_quantile=0.02).return_period == 50


def test_base_scenario_leaves_inputs_unchanged():
    port, df = _book()
    x = ST.stressed_inputs(port, df, _DtiModel(), 0.05, ST.Scenario("Base"))
    assert np.allclose(x["pd"], port["pd"]) and np.allclose(x["ead"], port["exposure"] * 0.6)


def test_conditional_pds_average_back_to_unconditional():
    port, _ = _book(500)
    z, w = np.polynomial.hermite_e.hermegauss(60)            # integrate over Z ~ N(0, 1)
    avg = sum(wi * MC.conditional_pd(port["pd"].to_numpy(), 0.08, zi) for zi, wi in zip(z, w)) / w.sum()
    assert np.allclose(avg, port["pd"], atol=1e-6)


def test_add_ons_are_capped():
    port, df = _book()
    x = ST.stressed_inputs(port, df, _DtiModel(), 0.05, ST.Scenario("x", lgd_add=0.5, ead_add=0.6))
    assert x["lgd"].max() == 1 and np.allclose(x["ead"], port["exposure"])


def test_rate_shock_raises_dti_by_extra_interest():
    df = pd.DataFrame({"dti": [10.0, 20.0], "revol_bal": [10_000, 0], "annual_inc": [50_000, 80_000]})
    d = ST.apply_rate_shock(df, 0.03)
    assert np.allclose(d["dti"], [10 + 100 * 0.03 * 10_000 / 50_000, 20.0])


def test_worse_economy_means_more_loss():
    port, df = _book()
    scen = [ST.Scenario("Base"), ST.Scenario("Mild", 0.1), ST.Scenario("Severe", 0.02, 0.04, 0.06, 1.25)]
    inputs = {s.name: ST.stressed_inputs(port, df, _DtiModel(), 0.05, s) for s in scen}
    t = ST.scenario_table(port, inputs, scen)
    assert t.loc["Base", "vs_base"] == 1 and t.loc["Severe", "el_rate"] > t.loc["Mild", "el_rate"] > t.loc["Base", "el_rate"]
    curve = ST.economy_curve(port, 0.05, [0.5, 0.1, 0.01])
    assert curve.is_monotonic_increasing


def test_reverse_stress_hits_the_threshold():
    port, _ = _book()
    rev = ST.reverse_stress(port, 0.05, [0.10, 0.99])
    z = rev.loc[0.10, "z"]
    el = (MC.conditional_pd(port["pd"].to_numpy(), 0.05, z) * 0.9 * port["exposure"] * 0.6).sum() / port["exposure"].sum()
    assert np.isclose(el, 0.10) and np.isnan(rev.loc[0.99, "z"])     # 99% loss is out of reach


def test_implied_factor_recovers_the_economy():
    port, _ = _book()
    dr = MC.conditional_pd(port["pd"].to_numpy(), 0.06, -1.5).mean()
    assert np.isclose(ST.implied_factor(port["pd"], 0.06, dr), -1.5)


def test_vulnerability_shares_sum_to_one():
    port, df = _book()
    base = ST.stressed_inputs(port, df, _DtiModel(), 0.05, ST.Scenario("Base"))
    sev = ST.stressed_inputs(port, df, _DtiModel(), 0.05, ST.Scenario("Severe", 0.02))
    v = ST.segment_vulnerability(port, base, sev, "grade")
    assert np.isclose(v["share_of_increase"].sum(), 1) and (v["multiple"] > 1).all()
