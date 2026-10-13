"""One-factor Gaussian Monte Carlo engine."""
import numpy as np
import pandas as pd

from pdmodel import simulation as M


def _book(n=2_000, seed=0):
    r = np.random.default_rng(seed)
    return r.uniform(0.02, 0.25, n), np.full(n, 0.9), r.uniform(2_000, 20_000, n)


def test_simulated_el_matches_analytical():
    pd_, lgd, ead = _book()
    L, _, _ = M.simulate_losses(pd_, lgd, ead, 0.08, 20_000, seed=1)
    se = L.std() / np.sqrt(len(L))
    assert abs(L.mean() - M.analytical_el(pd_, lgd, ead)) < 4 * se


def test_zero_correlation_gives_binomial_default_rates():
    pd_ = np.full(5_000, 0.1)
    _, dr, _ = M.simulate_losses(pd_, np.ones(5_000), np.ones(5_000), 0.0, 4_000, seed=2)
    assert abs(dr.std() - np.sqrt(0.1 * 0.9 / 5_000)) < 0.0005


def test_correlation_fattens_the_tail():
    pd_, lgd, ead = _book()
    v = {rho: M.risk_measures(M.simulate_losses(pd_, lgd, ead, rho, 10_000, seed=3)[0])["var_0.99"]
         for rho in (0.0, 0.15)}
    assert v[0.15] > 1.5 * v[0.0]


def test_risk_measures_on_known_losses():
    m = M.risk_measures(np.arange(1, 1001, dtype=float))
    assert np.isclose(m["expected_loss"], 500.5)
    assert m["es_0.99"] == np.mean(np.arange(991, 1001))
    assert m["es_0.99"] >= m["var_0.99"] >= m["var_0.95"]


def test_large_portfolio_formula_matches_simulation():
    pd_, lgd, ead = _book(20_000)
    L, _, _ = M.simulate_losses(pd_, lgd, ead, 0.05, 20_000, seed=4)
    assert abs(np.quantile(L, 0.99) / M.asrf_quantile(pd_, lgd, ead, 0.05, 0.99) - 1) < 0.03


def test_correlation_estimate_recovers_true_rho():
    r = np.random.default_rng(5)
    n, p, rho = 50_000, 0.12, 0.05
    z = r.standard_normal(60)
    dr = r.binomial(n, M.conditional_pd(np.full(60, p), rho, z)) / n
    v = pd.DataFrame({"n": n, "observed_dr": dr, "mean_pd": p})
    est, _ = M.estimate_correlation(v)
    assert abs(est["rho_raw"] - rho) < 0.025


def test_basel_retail_correlation_range():
    r = M.basel_retail_correlation(np.array([0.0, 0.01, 0.05, 0.5]))
    assert np.isclose(r[0], 0.16) and np.isclose(r[-1], 0.03, atol=1e-6)
    assert np.all(np.diff(r) < 0)


def test_annualised_pd_compounds_back():
    p1 = M.annualised_pd(np.array([0.15]), 3)
    assert np.isclose(1 - (1 - p1) ** 3, 0.15)
