"""Returns per dollar, scenario returns and the allocation linear programmes."""
import numpy as np
import pandas as pd

from pdmodel import optimisation as O


def test_dollar_years_match_the_amortisation_schedule():
    P, r = 10_000.0, 0.12
    i, n = r / 12, 36
    pay = P * i / (1 - (1 + i) ** -n)
    bal, total = P, 0.0
    for _ in range(20):
        total += bal
        bal = bal * (1 + i) - pay
    assert np.isclose(O.dollar_years(P, r, 20), total / 12)
    assert O.dollar_years(P, r, 0) == 0


def test_cash_terms_bound_the_expected_return():
    port = pd.DataFrame({"int_rate": [12.0], "ead_ratio": [0.6], "lgd": [0.9]})
    carry = pd.DataFrame({"k_good": [1.3], "k_bad": [1.0]}, index=["B"])
    t = O.loan_cash_terms(port, pd.Series(["B"]), carry, fee=0.01)
    assert np.isclose(t["good"][0], 0.99 * (1 + 1.3 * 0.12))
    assert np.isclose(t["bad"][0], 0.99 * (0.4 + 0.12) + 0.6 * 0.1)
    assert t["bad"][0] < 1 < t["good"][0]


def test_realised_cash_nets_off_fees():
    df = pd.DataFrame({"total_pymnt": [1_200.0], "recoveries": [100.0], "collection_recovery_fee": [20.0]})
    assert np.isclose(O.realised_net_cash(df, 0.01)[0], 0.99 * 1_100 + 80)


def test_cvar_of_known_returns():
    r = np.linspace(-0.99, 0.0, 100)          # worst return -0.99
    assert np.isclose(O.cvar(r, 0.99), 0.99)
    assert np.isclose(O.cvar(r, 0.98), np.mean([0.99, 0.98]))


def _problem(seed=0, K=8, S=4_000):
    r = np.random.default_rng(seed)
    z = r.standard_normal(S)
    mu = r.uniform(0.02, 0.10, K)
    beta = r.uniform(0.01, 0.06, K)            # sensitivity to the economy
    R = mu + np.outer(z, beta) + r.normal(0, 0.01, (S, K))
    el = r.uniform(0.02, 0.10, K)
    seg = pd.DataFrame({"lc_band": list("AABBCCDD")[:K]})
    return mu, el, R, seg


def test_weights_respect_every_limit():
    mu, el, R, seg = _problem()
    cons = O.Constraints(w_max=0.3, groups=O.group_caps(seg, "lc_band", 0.4), el_cap=float(el.mean()))
    for obj in ("max_return", "min_el", "min_cvar"):
        w = O.optimise(mu, el, R, obj, cons, alpha=0.95)
        assert np.isclose(w.sum(), 1) and w.max() <= 0.3 + 1e-9 and w @ el <= el.mean() + 1e-9
        for members, cap in cons.groups.values():
            assert w[members].sum() <= cap + 1e-9


def test_each_objective_wins_on_its_own_measure():
    mu, el, R, seg = _problem(1)
    cons = O.Constraints(w_max=0.5)
    w = {o: O.optimise(mu, el, R, o, cons, alpha=0.95) for o in ("max_return", "min_el", "min_cvar")}
    eq = np.full(len(mu), 1 / len(mu))
    assert w["max_return"] @ mu >= max(v @ mu for v in [*w.values(), eq]) - 1e-9
    assert w["min_el"] @ el <= min(v @ el for v in [*w.values(), eq]) + 1e-9
    assert O.cvar(R @ w["min_cvar"], 0.95) <= min(O.cvar(R @ v, 0.95) for v in [*w.values(), eq]) + 1e-9


def test_cvar_limit_binds():
    mu, el, R, seg = _problem(2)
    cons = O.Constraints(w_max=0.5)
    lo = O.cvar(R @ O.optimise(mu, el, R, "min_cvar", cons, 0.95), 0.95)
    hi = O.cvar(R @ O.optimise(mu, el, R, "max_return", cons, 0.95), 0.95)
    cap = (lo + hi) / 2
    w = O.optimise(mu, el, R, "max_return_cvar", cons, 0.95, cvar_cap=cap)
    assert O.cvar(R @ w, 0.95) <= cap + 1e-6


def test_scenario_returns_average_to_expected_returns():
    r = np.random.default_rng(3)
    n = 3_000
    port = pd.DataFrame({"exposure": r.uniform(1_000, 20_000, n), "pd": r.uniform(0.03, 0.3, n)})
    terms = pd.DataFrame({"good": 1.15, "bad": 0.55}, index=port.index)
    seg = pd.Series(np.where(port["pd"] < 0.15, "low", "high"), index=port.index)
    R = O.scenario_returns(port, terms, seg, pd.Index(["low", "high"]), 0.06, 40_000, seed=4)
    for k, name in enumerate(["low", "high"]):
        m = seg == name
        e = port.loc[m, "exposure"]
        mu = np.average((1 - port.loc[m, "pd"]) * 1.15 + port.loc[m, "pd"] * 0.55, weights=e) - 1
        assert abs(R[:, k].mean() - mu) < 0.002


def test_price_informed_pd_scales_by_history():
    hist_cells = pd.Series(["a"] * 4 + ["b"] * 4)
    pd_new = O.price_informed_pd(np.array([0.1, 0.1]), pd.Series(["a", "b"]), hist_cells,
                                 [1, 0, 0, 0, 1, 1, 0, 0], [0.25] * 8)
    assert np.allclose(pd_new, [0.1, 0.2])
