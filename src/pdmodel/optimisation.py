"""Credit portfolio allocation across loan segments, backtested on realised returns.

An investor buys LendingClub notes at par and holds them to the end. For each
loan, per dollar funded and over the loan's life:

  if repaid:    borrower pays the principal back plus interest ~ k_good * rate
  if defaulted: borrower pays (1 - EAD ratio) of principal plus interest ~ k_bad * rate,
                then recoveries return EAD ratio * (1 - LGD)

k_good and k_bad capture prepayment and the timing of default; they are measured
on earlier loans. A servicing fee is charged on every borrower payment. So the
expected lifetime net return is linear in the PD:

  E[R] = (1 - PD) * good + PD * bad - 1

Allocation is over segments (our rating grade x LendingClub grade), not single
loans. Tail risk comes from the one-factor model: in a scenario where the
economy factor is z, each loan's PD becomes PD(z) (Vasicek), so each segment's
return is a function of z. CVaR is minimised or constrained with the
Rockafellar-Uryasev linear programme over sampled economies.

Everything used to choose a portfolio is known at origination; outcomes are used
only to score the choice afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from . import simulation as MC

LC_BANDS = {"A": "A", "B": "B", "C": "C", "D": "D", "E": "E–G", "F": "E–G", "G": "E–G"}


# ----------------------------------------------------------------------------
# Cash flows per dollar funded
# ----------------------------------------------------------------------------
def interest_yield(df: pd.DataFrame) -> pd.Series:
    """Interest and late fees received per dollar funded, over the loan's life."""
    return (df["total_pymnt"] - df["total_rec_prncp"] - df["recoveries"]) / df["funded_amnt"]


def carry_factors(hist: pd.DataFrame, by: str = "grade") -> pd.DataFrame:
    """Interest earned per unit of interest rate, for repaid and for defaulted loans,
    by LendingClub grade band. Captures prepayment and time-to-default."""
    d = pd.DataFrame({"band": hist[by].map(LC_BANDS), "default": hist["default"],
                      "y": interest_yield(hist), "r": hist["int_rate"] / 100})
    rows = {}
    for band, g in d.groupby("band"):
        good, bad = g[g["default"] == 0], g[g["default"] == 1]
        rows[band] = {"loans": len(g), "k_good": good["y"].sum() / good["r"].sum(),
                      "k_bad": bad["y"].sum() / bad["r"].sum()}
    return pd.DataFrame(rows).T.rename_axis("lc_band")


def loan_cash_terms(port: pd.DataFrame, lc_band: pd.Series, carry: pd.DataFrame, fee: float) -> pd.DataFrame:
    """Net cash per dollar funded if the loan is repaid (good) or defaults (bad)."""
    r = port["int_rate"].to_numpy() / 100
    k = carry.loc[lc_band.to_numpy()]
    ead, lgd = port["ead_ratio"].to_numpy(), port["lgd"].to_numpy()
    good = (1 - fee) * (1 + k["k_good"].to_numpy() * r)
    bad = (1 - fee) * ((1 - ead) + k["k_bad"].to_numpy() * r) + ead * (1 - lgd)
    return pd.DataFrame({"good": good, "bad": bad}, index=port.index)


def realised_net_cash(df: pd.DataFrame, fee: float) -> pd.Series:
    """Cash an investor actually received per loan: borrower payments less the
    servicing fee, plus recoveries less the collection fee."""
    borrower = df["total_pymnt"] - df["recoveries"]
    return (1 - fee) * borrower + df["recoveries"] - df["collection_recovery_fee"]


def dollar_years(funded, rate, months, term: int = 36) -> np.ndarray:
    """Principal outstanding x time, following the amortisation schedule for
    `months` months. Used to turn lifetime gains into an approximate annual return."""
    P, i, m = np.asarray(funded, float), np.asarray(rate, float) / 12, np.clip(np.asarray(months, float), 0, None)
    g = (1 + i) ** term
    m = np.minimum(m, term)
    total = P / (g - 1) * (m * g - ((1 + i) ** m - 1) / i)    # sum of scheduled balances over m months
    return total / 12


# ----------------------------------------------------------------------------
# Segments
# ----------------------------------------------------------------------------
def segment_inputs(port: pd.DataFrame, df: pd.DataFrame, carry: pd.DataFrame, fee: float,
                   min_loans: int, chargeoff_lag: int) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Per segment: ex-ante expected return and expected loss (origination-time
    inputs only), and the realised lifetime and annualised returns.
    Returns (segment table, segment label per loan or NaN if dropped, loan cash terms)."""
    band = df["grade"].map(LC_BANDS)
    seg = port["grade"].astype(str) + " / " + band
    terms = loan_cash_terms(port, band, carry, fee)
    e = port["exposure"]
    months = ((df["last_pymnt_d"].dt.year - df["issue_d"].dt.year) * 12
              + (df["last_pymnt_d"].dt.month - df["issue_d"].dt.month)).fillna(0)
    months = months + np.where(df["default"] == 1, chargeoff_lag, 0)
    cash = realised_net_cash(df, fee)
    d = pd.DataFrame({
        "seg": seg, "grade": port["grade"], "lc_band": band, "e": e, "pd": port["pd"], "rate": port["int_rate"],
        "exp_cash": e * ((1 - port["pd"]) * terms["good"] + port["pd"] * terms["bad"]),
        "el": e * port["pd"] * port["lgd"] * port["ead_ratio"],
        "cash": cash, "default": port["default"], "realised_loss": port["realised_loss"],
        "dy": dollar_years(e, port["int_rate"] / 100, months),
    })
    t = d.groupby("seg").agg(grade=("grade", "first"), lc_band=("lc_band", "first"), loans=("e", "size"),
                             exposure=("e", "sum"), mean_pd=("pd", "mean"), mean_rate=("rate", "mean"),
                             exp_cash=("exp_cash", "sum"), el=("el", "sum"), cash=("cash", "sum"),
                             observed_dr=("default", "mean"), realised_loss=("realised_loss", "sum"),
                             dollar_years=("dy", "sum"))
    t = t[t["loans"] >= min_loans].copy()
    t["expected_return"] = t["exp_cash"] / t["exposure"] - 1
    t["el_rate"] = t["el"] / t["exposure"]
    t["realised_return"] = t["cash"] / t["exposure"] - 1
    t["realised_loss_rate"] = t["realised_loss"] / t["exposure"]
    t["gain_per_dollar"] = (t["cash"] - t["exposure"]) / t["exposure"]
    t["years_per_dollar"] = t["dollar_years"] / t["exposure"]
    t["realised_annualised"] = t["gain_per_dollar"] / t["years_per_dollar"]
    t = t.sort_values(["grade", "lc_band"])
    keep = seg.where(seg.isin(t.index))
    cols = ["grade", "lc_band", "loans", "exposure", "mean_pd", "mean_rate", "expected_return", "el_rate",
            "observed_dr", "realised_return", "realised_loss_rate", "realised_annualised",
            "gain_per_dollar", "years_per_dollar"]
    return t[cols], keep, terms


def scenario_returns(port: pd.DataFrame, terms: pd.DataFrame, seg: pd.Series, segments: pd.Index, rho,
                     n_scenarios: int, seed: int, grid=np.linspace(-6, 6, 601)) -> np.ndarray:
    """Segment lifetime returns in sampled economies (n_scenarios x segments).
    Each segment's return is computed exactly on a grid of economy values z and
    interpolated at the sampled z; idiosyncratic risk is diversified away within a
    segment."""
    ok = seg.notna().to_numpy()
    idx = pd.Index(segments).get_indexer(seg[ok])
    e = port["exposure"].to_numpy()[ok]
    pd0 = port["pd"].to_numpy()[ok]
    rho = np.broadcast_to(np.asarray(rho, float), len(port))[ok]
    good, bad = terms["good"].to_numpy()[ok], terms["bad"].to_numpy()[ok]
    E = np.bincount(idx, weights=e, minlength=len(segments))
    base = np.bincount(idx, weights=e * good, minlength=len(segments))
    drop = e * (good - bad)
    R = np.empty((len(grid), len(segments)))
    for j, z in enumerate(grid):
        p = MC.conditional_pd(pd0, rho, z)
        R[j] = (base - np.bincount(idx, weights=drop * p, minlength=len(segments))) / E - 1
    zs = np.random.default_rng(seed).standard_normal(n_scenarios)
    return np.column_stack([np.interp(zs, grid, R[:, k]) for k in range(len(segments))])


# ----------------------------------------------------------------------------
# Optimisation
# ----------------------------------------------------------------------------
@dataclass
class Constraints:
    w_max: float = 0.10                                   # per segment (concentration limit)
    w_min: float = 0.0                                    # per segment (minimum allocation)
    groups: dict[str, tuple[list[int], float]] = field(default_factory=dict)   # name -> (members, cap)
    el_cap: float | None = None                           # optional limit on expected loss rate


def group_caps(segments: pd.DataFrame, by: str, cap: float) -> dict:
    return {f"{by} {g}": (list(np.flatnonzero(segments[by].to_numpy() == g)), cap)
            for g in segments[by].unique()}


def cvar(returns: np.ndarray, alpha: float) -> float:
    """Expected loss (negative return) in the worst 1 - alpha share of scenarios."""
    loss = np.sort(-np.asarray(returns))
    k = max(1, int(np.ceil(round(len(loss) * (1 - alpha), 9))))
    return float(loss[-k:].mean())


def optimise(mu: np.ndarray, el: np.ndarray, R: np.ndarray, objective: str, cons: Constraints,
             alpha: float = 0.99, cvar_cap: float | None = None) -> np.ndarray:
    """objective: "max_return" | "min_el" | "min_cvar" | "max_return_cvar".
    Variables: weights (K), then for CVaR objectives zeta (1) and scenario
    shortfalls u (S), as in Rockafellar & Uryasev (2000)."""
    K = len(mu)
    use_cvar = objective in ("min_cvar", "max_return_cvar")
    S = R.shape[0] if use_cvar else 0
    n = K + (1 + S if use_cvar else 0)
    c = np.zeros(n)
    if objective in ("max_return", "max_return_cvar"):
        c[:K] = -mu
    elif objective == "min_el":
        c[:K] = el
    elif objective == "min_cvar":
        c[K] = 1.0
        c[K + 1:] = 1 / ((1 - alpha) * S)
    else:
        raise ValueError(objective)

    rows, b = [], []
    for members, cap in cons.groups.values():
        r = np.zeros(n)
        r[members] = 1
        rows.append(r)
        b.append(cap)
    if cons.el_cap is not None:
        r = np.zeros(n)
        r[:K] = el
        rows.append(r)
        b.append(cons.el_cap)
    if objective == "max_return_cvar":
        r = np.zeros(n)
        r[K] = 1.0
        r[K + 1:] = 1 / ((1 - alpha) * S)
        rows.append(r)
        b.append(cvar_cap)
    A = sparse.csr_matrix(np.array(rows)) if rows else sparse.csr_matrix((0, n))
    if use_cvar:                                   # loss_s - zeta - u_s <= 0, loss_s = -R_s . w
        scen = sparse.hstack([sparse.csr_matrix(-R), -np.ones((S, 1)), -sparse.identity(S)])
        A = sparse.vstack([A, scen])
        b = b + [0.0] * S
    bounds = [(cons.w_min, cons.w_max)] * K + ([(None, None)] + [(0, None)] * S if use_cvar else [])
    res = linprog(c, A_ub=A if A.shape[0] else None, b_ub=b if b else None,
                  A_eq=np.r_[np.ones(K), np.zeros(n - K)][None, :], b_eq=[1.0], bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"{objective}: {res.message}")
    return np.clip(res.x[:K], 0, None) / np.clip(res.x[:K], 0, None).sum()


def evaluate(w: np.ndarray, seg: pd.DataFrame, R: np.ndarray, alpha: float) -> dict:
    """Expected and realised outcomes of a weight vector over segments."""
    w = np.asarray(w, float)
    port_R = R @ w
    return {
        "expected_return": float(w @ seg["expected_return"]),
        "el_rate": float(w @ seg["el_rate"]),
        "mean_pd": float(w @ seg["mean_pd"]),
        "mean_rate": float(w @ seg["mean_rate"]) / 100,
        f"var_{alpha:g}": float(-np.quantile(port_R, 1 - alpha)),
        f"cvar_{alpha:g}": cvar(port_R, alpha),
        "realised_return": float(w @ seg["realised_return"]),
        "realised_loss_rate": float(w @ seg["realised_loss_rate"]),
        "realised_annualised": float((w @ seg["gain_per_dollar"]) / (w @ seg["years_per_dollar"])),
        "segments_held": int((w > 1e-4).sum()),
        "largest_weight": float(w.max()),
        "effective_segments": float(1 / np.sum(w ** 2)),
    }


def frontier(mu, el, R, cons: Constraints, alpha: float, n_points: int = 12) -> pd.DataFrame:
    """Highest expected return for a range of CVaR limits, from the minimum-CVaR
    portfolio to the maximum-return one."""
    lo = cvar(R @ optimise(mu, el, R, "min_cvar", cons, alpha), alpha)
    hi = cvar(R @ optimise(mu, el, R, "max_return", cons, alpha), alpha)
    rows = []
    for cap in np.linspace(lo, hi, n_points):
        w = optimise(mu, el, R, "max_return_cvar", cons, alpha, cvar_cap=cap + 1e-9)
        rows.append({"cvar_limit": cap, f"cvar_{alpha:g}": cvar(R @ w, alpha), "expected_return": float(w @ mu)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Alternative return inputs and the strategy set
# ----------------------------------------------------------------------------
def segment_labels(our_grade, lc_grade: pd.Series) -> pd.Series:
    return pd.Series(np.asarray(our_grade), index=lc_grade.index).astype(str) + " / " + lc_grade.map(LC_BANDS)


def price_informed_pd(pd_, cells: pd.Series, hist_cells: pd.Series, hist_default, hist_pd) -> np.ndarray:
    """Scale each loan's PD by its segment's actual-to-predicted default ratio on
    earlier vintages. This brings in the risk information in LendingClub's grade,
    which the scorecard leaves out by design."""
    h = pd.DataFrame({"cell": hist_cells.to_numpy(), "y": np.asarray(hist_default), "pd": np.asarray(hist_pd)})
    g = h.groupby("cell")[["y", "pd"]].sum()
    ratio = cells.map(g["y"] / g["pd"]).fillna(1.0).to_numpy()
    return np.clip(np.asarray(pd_, float) * ratio, 0, 0.99)


def historical_segment_returns(hist: pd.DataFrame, hist_cells: pd.Series, segments: pd.Index, fee: float) -> pd.Series:
    """Each segment's realised lifetime net return on earlier vintages."""
    d = pd.DataFrame({"cell": hist_cells.to_numpy(), "cash": realised_net_cash(hist, fee).to_numpy(),
                      "e": hist["funded_amnt"].to_numpy()})
    g = d.groupby("cell")[["cash", "e"]].sum()
    return (g["cash"] / g["e"] - 1).reindex(segments)


STRATEGIES = {
    "Whole book": "every loan, in proportion to its size (benchmark)",
    "Equal weight": "the same amount in every segment",
    "Max expected return": "highest expected return within the limits",
    "Min expected loss": "lowest expected loss within the limits",
    "Min CVaR": "lowest 99% CVaR within the limits",
    "Max return, CVaR ≤ book": "highest expected return with 99% CVaR no worse than the whole book's",
    "Max return, EL ≤ book": "highest expected return with expected loss no higher than the whole book's",
}


def run_strategies(mu: np.ndarray, el: np.ndarray, R: np.ndarray, seg: pd.DataFrame, cons: Constraints,
                   alpha: float) -> dict[str, np.ndarray]:
    """Weights for every strategy in STRATEGIES, from one set of return inputs."""
    book = (seg["exposure"] / seg["exposure"].sum()).to_numpy()
    w = {"Whole book": book, "Equal weight": np.full(len(seg), 1 / len(seg)),
         "Max expected return": optimise(mu, el, R, "max_return", cons, alpha),
         "Min expected loss": optimise(mu, el, R, "min_el", cons, alpha),
         "Min CVaR": optimise(mu, el, R, "min_cvar", cons, alpha),
         "Max return, CVaR ≤ book": optimise(mu, el, R, "max_return_cvar", cons, alpha,
                                             cvar_cap=cvar(R @ book, alpha) + 1e-9)}
    capped = Constraints(cons.w_max, cons.w_min, cons.groups, el_cap=float(el @ book) + 1e-12)
    w["Max return, EL ≤ book"] = optimise(mu, el, R, "max_return", capped, alpha)
    return w


def strategy_table(weights: dict, mu: np.ndarray, el: np.ndarray, R: np.ndarray, seg: pd.DataFrame,
                   alpha: float) -> pd.DataFrame:
    rows = {}
    for name, w in weights.items():
        m = evaluate(w, seg, R, alpha)
        m["expected_return"], m["el_rate"] = float(w @ mu), float(w @ el)
        m[f"cvar_{alpha:g}"] = cvar(R @ w, alpha)
        rows[name] = m
    return pd.DataFrame(rows).T.rename_axis("strategy")


def gap_by_band(port: pd.DataFrame, df: pd.DataFrame, carry: pd.DataFrame, fee: float) -> pd.DataFrame:
    """Why expected and realised returns differ, by LendingClub grade band: default
    rate, interest earned on repaid loans, and cash recovered from defaults."""
    band = df["grade"].map(LC_BANDS)
    terms = loan_cash_terms(port, band, carry, fee)
    cash = realised_net_cash(df, fee) / df["funded_amnt"]
    d = pd.DataFrame({"band": band, "e": df["funded_amnt"], "pd": port["pd"], "y": df["default"],
                      "good": terms["good"], "bad": terms["bad"], "cash": cash,
                      "r": df["int_rate"] / 100, "iy": interest_yield(df)})
    rows = {}
    for b, g in d.groupby("band"):
        good, bad = g[g["y"] == 0], g[g["y"] == 1]
        dr = g["y"].mean()
        rows[b] = {"loans": len(g), "mean_rate": g["r"].mean(), "mean_pd": g["pd"].mean(), "observed_dr": dr,
                   "k_good_assumed": carry.loc[b, "k_good"], "k_good_realised": good["iy"].sum() / good["r"].sum(),
                   "default_cash_assumed": np.average(bad["bad"], weights=bad["e"]),
                   "default_cash_realised": np.average(bad["cash"], weights=bad["e"]),
                   "expected_return": np.average((1 - g["pd"]) * g["good"] + g["pd"] * g["bad"], weights=g["e"]) - 1,
                   "expected_return_at_actual_dr": np.average((1 - dr) * g["good"] + dr * g["bad"], weights=g["e"]) - 1,
                   "realised_return": np.average(g["cash"], weights=g["e"]) - 1}
    return pd.DataFrame(rows).T.rename_axis("lc_band")
