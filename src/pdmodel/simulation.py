"""Monte Carlo credit portfolio simulation with a one-factor Gaussian model.

For borrower i in each scenario:
    A_i = sqrt(rho) * Z + sqrt(1 - rho) * eps_i        Z, eps_i ~ N(0, 1), independent
    default if A_i < Phi^-1(PD_i)
Z is the economy, shared by every borrower; eps_i is borrower-specific. rho sets
how strongly defaults move together. Portfolio loss = sum of LGD_i * EAD_i over
defaulted loans. The horizon is the loan's 36-month life, like the PD.

Limitations of the Gaussian copula (documented in the portfolio report): one
factor only (no sector or regional factors), no tail dependence beyond what
rho implies, LGD independent of the economy, and a single rho for all loans.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import ndtr, ndtri


def simulate_losses(pd_, lgd, ead, rho, n_sims: int, seed: int = 0, chunk: int | None = None):
    """Returns (loss per scenario, default rate per scenario, Z per scenario)."""
    pd_ = np.clip(np.asarray(pd_, float), 1e-12, 1 - 1e-12)
    sev = (np.asarray(lgd, float) * np.asarray(ead, float)).astype(np.float32)
    c = ndtri(pd_).astype(np.float32)
    rho = np.broadcast_to(np.asarray(rho, float), pd_.shape)
    a, b = np.sqrt(rho).astype(np.float32), np.sqrt(1 - rho).astype(np.float32)
    n = len(pd_)
    chunk = chunk or max(1, min(n_sims, 12_000_000 // max(n, 1)))
    rng = np.random.default_rng(seed)
    losses, dr, zs = np.empty(n_sims), np.empty(n_sims), np.empty(n_sims)
    for start in range(0, n_sims, chunk):
        k = min(chunk, n_sims - start)
        z = rng.standard_normal(k)
        eps = rng.standard_normal((k, n), dtype=np.float32)
        defaults = (a * z[:, None].astype(np.float32) + b * eps) < c
        losses[start:start + k] = defaults.astype(np.float32) @ sev
        dr[start:start + k] = defaults.mean(axis=1)
        zs[start:start + k] = z
    return losses, dr, zs


def risk_measures(losses: np.ndarray, var_levels=(0.95, 0.99), es_levels=(0.975, 0.99)) -> dict:
    """EL, standard deviation, VaR (loss quantile) and Expected Shortfall
    (average loss in the worst 1 - alpha of scenarios)."""
    s = np.sort(np.asarray(losses, float))
    out = {"expected_loss": s.mean(), "std": s.std(ddof=1)}
    for a in var_levels:
        out[f"var_{a:g}"] = float(np.quantile(s, a))
    for a in es_levels:
        k = max(1, int(np.ceil(round(len(s) * (1 - a), 9))))
        out[f"es_{a:g}"] = float(s[-k:].mean())
    return out


def analytical_el(pd_, lgd, ead) -> float:
    return float(np.sum(np.asarray(pd_) * np.asarray(lgd) * np.asarray(ead)))


def conditional_pd(pd_, rho, z):
    """PD given the economy factor Z = z (Vasicek). Negative z = bad economy."""
    return ndtr((ndtri(np.clip(pd_, 1e-12, 1 - 1e-12)) - np.sqrt(rho) * z) / np.sqrt(1 - rho))


def asrf_quantile(pd_, lgd, ead, rho, alpha: float) -> float:
    """Large-portfolio (Vasicek / Basel ASRF) loss quantile: the loss when the
    economy is at its (1 - alpha) worst, with idiosyncratic risk diversified away."""
    return float(np.sum(np.asarray(lgd) * np.asarray(ead) * conditional_pd(pd_, rho, -ndtri(alpha))))


def convergence(losses: np.ndarray, sizes, n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """Estimates from the first n scenarios, with bootstrap standard errors."""
    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        x = losses[:n]
        m = risk_measures(x)
        boot = np.array([[(r := risk_measures(xb, (0.99,), (0.99,)))["var_0.99"], r["es_0.99"]]
                         for xb in (x[rng.integers(0, n, n)] for _ in range(n_boot))])
        rows.append({"scenarios": n, "expected_loss": m["expected_loss"],
                     "el_se": x.std(ddof=1) / np.sqrt(n),
                     "var_0.99": m["var_0.99"], "var_0.99_se": boot[:, 0].std(ddof=1),
                     "es_0.99": m["es_0.99"], "es_0.99_se": boot[:, 1].std(ddof=1)})
    return pd.DataFrame(rows).set_index("scenarios")


def annualised_pd(pd_life, years: float) -> np.ndarray:
    """One-year PD with the same cumulative PD over `years`, assuming a constant hazard."""
    return 1 - (1 - np.asarray(pd_life, float)) ** (1 / years)


def basel_retail_correlation(pd_1y) -> np.ndarray:
    """Basel II/III 'other retail' asset correlation: 16% for the safest loans,
    falling to 3% as the one-year PD rises."""
    w = (1 - np.exp(-35 * np.asarray(pd_1y, float))) / (1 - np.exp(-35))
    return 0.03 * w + 0.16 * (1 - w)


# ----------------------------------------------------------------------------
# Default correlation from vintage default rates
# ----------------------------------------------------------------------------
def _vasicek_dr_variance(p: float, rho: float) -> float:
    """Variance of the default rate of a large portfolio under the one-factor model."""
    c = ndtri(p)
    joint = stats.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([c, c])
    return float(joint - p ** 2)


def implied_correlation(mean_dr: float, systematic_var: float) -> float:
    """rho such that the model's default-rate variance matches the observed one."""
    if systematic_var <= 0:
        return 0.0
    f = lambda r: _vasicek_dr_variance(mean_dr, r) - systematic_var
    return float(optimize.brentq(f, 1e-6, 0.95)) if f(0.95) > 0 else 0.95


def estimate_correlation(vintages: pd.DataFrame, min_loans: int = 1_000) -> tuple[dict, pd.DataFrame]:
    """Method of moments on quarterly vintages.

    vintages: one row per vintage with n, observed_dr and mean_pd (model PD).
    Two estimates:
      raw          - variance of observed default rates
      mix-adjusted - default rates rescaled by the model's mean PD for the vintage,
                     which removes changes in borrower mix the model already explains;
                     the leftover variation is the shared, systematic part
    Binomial noise p(1-p)/n is subtracted in both."""
    v = vintages[vintages["n"] >= min_loans].copy()
    p_bar = float(np.average(v["observed_dr"], weights=v["n"]))
    v["mix_adjusted_dr"] = v["observed_dr"] / v["mean_pd"] * np.average(v["mean_pd"], weights=v["n"])
    noise = float(np.mean(p_bar * (1 - p_bar) / v["n"]))
    out = {"vintages": len(v), "mean_dr": p_bar, "binomial_noise_var": noise}
    for col, name in [("observed_dr", "raw"), ("mix_adjusted_dr", "mix_adjusted")]:
        var = float(v[col].var(ddof=1))
        out[f"{name}_var"] = var
        out[f"rho_{name}"] = implied_correlation(p_bar, var - noise)
    return out, v
