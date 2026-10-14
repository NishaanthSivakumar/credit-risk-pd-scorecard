"""Macroeconomic stress testing on the one-factor credit model.

A scenario fixes how bad the economy is over the loans' life, through the
systematic factor Z of the portfolio model (simulation.py). Under the Vasicek
model each loan's PD given Z = z is

    PD(z) = Phi( (Phi^-1(PD) - sqrt(rho) * z) / sqrt(1 - rho) )

so a scenario is set by its severity: `economy_quantile` = the probability that
the economy is at least this bad (0.10 = a 1-in-10 outcome, z = Phi^-1(0.10)).
On top of that a scenario can add downturn LGD and EAD, raise the asset
correlation, scale the model's PD level, and raise interest rates on borrowers'
revolving debt (which feeds through the scorecard via debt-to-income).

Scenario severities and add-ons are assumptions, anchored to LendingClub's own
history where possible (implied_factor_by_vintage, loss_parameters_by_year).
They are not estimated macroeconomic relationships.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import ndtr, ndtri

from . import simulation as MC


@dataclass(frozen=True)
class Scenario:
    name: str
    economy_quantile: float | None = None   # None: no economy shock (unconditional PDs)
    lgd_add: float = 0.0                    # added to LGD (absolute), capped at 100%
    ead_add: float = 0.0                    # added to the EAD ratio (absolute), capped at 100%
    rho_mult: float = 1.0                   # multiplier on asset correlation
    rate_shock: float = 0.0                 # rise in rates on revolving debt (0.03 = 300bp)
    pd_scale: float = 1.0                   # multiplier on the model's PD before the economy shock

    @property
    def z(self) -> float | None:
        return None if self.economy_quantile is None else float(ndtri(self.economy_quantile))

    @property
    def return_period(self) -> float | None:
        return None if self.economy_quantile is None else 1 / self.economy_quantile


def scenarios_from_config(cfg: dict) -> list[Scenario]:
    return [Scenario(name, **params) for name, params in cfg.items()]


# ----------------------------------------------------------------------------
# Applying a scenario
# ----------------------------------------------------------------------------
def apply_rate_shock(df: pd.DataFrame, shock: float) -> pd.DataFrame:
    """Raise each borrower's debt-to-income by the extra interest on their revolving
    balance: monthly payment rises by shock * balance / 12, so DTI (in points) rises by
    100 * shock * revol_bal / annual_inc. LendingClub loans are fixed-rate and its DTI
    excludes mortgages, so neither channel is included."""
    d = df.copy()
    inc = d["annual_inc"].where(d["annual_inc"] > 0)
    d["dti"] = d["dti"] + (100 * shock * d["revol_bal"] / inc).fillna(0.0)
    return d


def stressed_inputs(port: pd.DataFrame, df: pd.DataFrame, model, rho, sc: Scenario) -> pd.DataFrame:
    """Per-loan PD, LGD, EAD and rho under a scenario.
    port: portfolio table (portfolio.build_portfolio); df: the same loans' raw inputs;
    model: anything with predict_pd(df), used only when rates are shocked."""
    base_pd = model.predict_pd(apply_rate_shock(df, sc.rate_shock)) if sc.rate_shock else port["pd"].to_numpy()
    base_pd = np.clip(np.asarray(base_pd, float) * sc.pd_scale, 0, 1 - 1e-9)
    rho_s = np.clip(np.broadcast_to(np.asarray(rho, float), len(port)) * sc.rho_mult, 0, 0.99)
    pd_s = base_pd if sc.z is None else MC.conditional_pd(base_pd, rho_s, sc.z)
    lgd = np.clip(port["lgd"].to_numpy() + sc.lgd_add, 0, 1)
    ead = port["exposure"].to_numpy() * np.clip(port["ead_ratio"].to_numpy() + sc.ead_add, 0, 1)
    return pd.DataFrame({"pd": pd_s, "lgd": lgd, "ead": ead, "rho": rho_s}, index=port.index)


def scenario_table(port: pd.DataFrame, inputs: dict[str, pd.DataFrame], scenarios: list[Scenario],
                   base: str = "Base") -> pd.DataFrame:
    """Expected loss under each scenario, against the base case."""
    exp = port["exposure"].sum()
    sc = {s.name: s for s in scenarios}
    rows = {}
    for name, x in inputs.items():
        el = float((x["pd"] * x["lgd"] * x["ead"]).sum())
        s = sc[name]
        rows[name] = {"economy": "average" if s.z is None else f"1 in {s.return_period:g}",
                      "z": s.z, "mean_pd": x["pd"].mean(), "expected_defaults": x["pd"].sum(),
                      "ead": x["ead"].sum(), "expected_loss": el, "el_rate": el / exp}
    t = pd.DataFrame(rows).T
    t["vs_base"] = t["expected_loss"] / t.loc[base, "expected_loss"]
    return t


def simulate_scenarios(inputs: dict[str, pd.DataFrame], exposure: float, n_sims: int, seed: int,
                       var_levels=(0.95, 0.99), es_levels=(0.975, 0.99)) -> tuple[dict, pd.DataFrame]:
    """Loss distribution with each scenario's parameters as the starting point: the
    economy is still uncertain, but PDs, LGD, EAD and correlation are at stressed
    levels. Returns ({scenario: losses}, risk measures as % of funded amount)."""
    losses, rows = {}, {}
    for name, x in inputs.items():
        L, _, _ = MC.simulate_losses(x["pd"].to_numpy(), x["lgd"].to_numpy(), x["ead"].to_numpy(),
                                     x["rho"].to_numpy(), n_sims, seed=seed)
        losses[name] = L
        rows[name] = {k: v / exposure * 100 for k, v in MC.risk_measures(L, var_levels, es_levels).items()}
    return losses, pd.DataFrame(rows).T


# ----------------------------------------------------------------------------
# Vulnerability, sensitivity and reverse stress
# ----------------------------------------------------------------------------
def segment_vulnerability(port: pd.DataFrame, base: pd.DataFrame, stressed: pd.DataFrame, by: str) -> pd.DataFrame:
    """Expected loss rate by segment, before and after stress, and each segment's
    share of the total increase."""
    d = pd.DataFrame({"seg": port[by], "exposure": port["exposure"],
                      "el_base": base["pd"] * base["lgd"] * base["ead"],
                      "el_stress": stressed["pd"] * stressed["lgd"] * stressed["ead"]})
    t = d.groupby("seg", observed=True)[["exposure", "el_base", "el_stress"]].sum()
    t["exposure_share"] = t["exposure"] / t["exposure"].sum()
    t["el_rate_base"] = t["el_base"] / t["exposure"]
    t["el_rate_stress"] = t["el_stress"] / t["exposure"]
    t["multiple"] = t["el_stress"] / t["el_base"]
    inc = t["el_stress"] - t["el_base"]
    t["share_of_increase"] = inc / inc.sum()
    return t.rename_axis(by)[["exposure_share", "el_rate_base", "el_rate_stress", "multiple", "share_of_increase"]]


def el_rate(port: pd.DataFrame, df: pd.DataFrame, model, rho, sc: Scenario) -> float:
    x = stressed_inputs(port, df, model, rho, sc)
    return float((x["pd"] * x["lgd"] * x["ead"]).sum() / port["exposure"].sum())


def sensitivity(port: pd.DataFrame, df: pd.DataFrame, model, rho, sc: Scenario,
                ranges: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Tornado: expected loss rate when one scenario assumption moves to a low and a
    high value. ranges: {field: (low, high)}, fields of Scenario."""
    centre = el_rate(port, df, model, rho, sc)
    rows = []
    for field, (lo, hi) in ranges.items():
        r_lo = el_rate(port, df, model, rho, replace(sc, **{field: lo}))
        r_hi = el_rate(port, df, model, rho, replace(sc, **{field: hi}))
        rows.append({"assumption": field, "central": getattr(sc, field), "low": lo, "high": hi,
                     "el_rate_low": r_lo, "el_rate_high": r_hi, "el_rate_central": centre,
                     "range": abs(r_hi - r_lo)})
    return pd.DataFrame(rows).set_index("assumption").sort_values("range", ascending=False)


def economy_curve(port: pd.DataFrame, rho, quantiles, lgd_add=0.0, ead_add=0.0, rho_mult=1.0) -> pd.Series:
    """Expected loss rate when the economy sits at each quantile (no other change
    unless add-ons are given)."""
    pd0 = port["pd"].to_numpy()
    r = np.clip(np.broadcast_to(np.asarray(rho, float), len(port)) * rho_mult, 0, 0.99)
    sev = (np.clip(port["lgd"].to_numpy() + lgd_add, 0, 1)
           * port["exposure"].to_numpy() * np.clip(port["ead_ratio"].to_numpy() + ead_add, 0, 1))
    exp = port["exposure"].sum()
    return pd.Series({q: float((MC.conditional_pd(pd0, r, ndtri(q)) * sev).sum() / exp) for q in quantiles},
                     name="el_rate").rename_axis("economy_quantile")


def reverse_stress(port: pd.DataFrame, rho, thresholds, lgd_add=0.0, ead_add=0.0, rho_mult=1.0) -> pd.DataFrame:
    """How bad must the economy be for the lifetime loss rate to reach each threshold?"""
    pd0 = port["pd"].to_numpy()
    r = np.clip(np.broadcast_to(np.asarray(rho, float), len(port)) * rho_mult, 0, 0.99)
    sev = (np.clip(port["lgd"].to_numpy() + lgd_add, 0, 1)
           * port["exposure"].to_numpy() * np.clip(port["ead_ratio"].to_numpy() + ead_add, 0, 1))
    exp = port["exposure"].sum()
    f = lambda z, thr: (MC.conditional_pd(pd0, r, z) * sev).sum() / exp - thr
    rows = []
    for thr in thresholds:
        if f(-8.0, thr) < 0 or f(8.0, thr) > 0:
            z = np.nan
        else:
            z = optimize.brentq(f, -8.0, 8.0, args=(thr,))
        q = float(ndtr(z)) if np.isfinite(z) else np.nan
        rows.append({"loss_threshold": thr, "z": z, "economy_quantile": q,
                     "return_period_years": 1 / q if q and np.isfinite(q) else np.nan})
    return pd.DataFrame(rows).set_index("loss_threshold")


# ----------------------------------------------------------------------------
# Historical anchor
# ----------------------------------------------------------------------------
def implied_factor(pd_, rho, default_rate: float) -> float:
    """The economy factor z at which the model's PDs average to the observed default rate."""
    pd_ = np.asarray(pd_, float)
    f = lambda z: MC.conditional_pd(pd_, rho, z).mean() - default_rate
    return float(optimize.brentq(f, -8.0, 8.0))


def implied_factor_by_vintage(df: pd.DataFrame, pd_, rho, by: str = "issue_year") -> pd.DataFrame:
    """Implied economy factor for each vintage. A negative z is a worse-than-average
    economy. It also absorbs anything else that moved the default rate, such as
    changes in underwriting, so it is an upper bound on the economy's role."""
    d = pd.DataFrame({"g": df[by].to_numpy(), "y": df["default"].to_numpy(),
                      "pd": np.asarray(pd_, float), "rho": np.broadcast_to(np.asarray(rho, float), len(df))})
    rows = []
    for g, x in d.groupby("g"):
        z = implied_factor(x["pd"].to_numpy(), x["rho"].to_numpy(), x["y"].mean())
        q = float(ndtr(z))
        rows.append({by: g, "loans": len(x), "observed_dr": x["y"].mean(), "mean_pd": x["pd"].mean(),
                     "implied_z": z, "economy_quantile": q,
                     "return_period_years": 1 / q if z < 0 else np.nan})
    return pd.DataFrame(rows).set_index(by)
