"""Portfolio expected loss: PD x LGD x EAD per loan, by segment and in total.

All quantities are over the loan's 36-month life, matching the PD target:
  PD   lifetime probability of default (recalibrated scorecard)
  EAD  principal still owed when the loan defaults
  LGD  share of that principal not recovered, net of collection fees
  EL   PD x LGD x EAD

LGD and EAD are *measured* on charged-off loans from earlier vintages, by rating
grade. Recoveries are not reported for every charged-off loan, so the measured
LGD is an upper bound for loans whose recoveries were still in progress.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def realised_loss_fields(df: pd.DataFrame) -> pd.DataFrame:
    """EAD, net recovery and loss for each loan, from the outcome fields.
    Non-defaults have zero loss by definition."""
    funded = df["funded_amnt"].astype(float)
    ead = (funded - df["total_rec_prncp"].astype(float)).clip(lower=0, upper=funded)
    net_rec = (df["recoveries"].astype(float) - df["collection_recovery_fee"].astype(float)).clip(lower=0)
    loss = (ead - net_rec).clip(lower=0)
    bad = df["default"].to_numpy() == 1
    return pd.DataFrame({"funded": funded, "ead": np.where(bad, ead, 0.0),
                         "net_recovery": np.where(bad, np.minimum(net_rec, ead), 0.0),
                         "loss": np.where(bad, loss, 0.0)}, index=df.index)


def estimate_loss_parameters(hist: pd.DataFrame, grade: np.ndarray) -> pd.DataFrame:
    """EAD ratio and LGD per rating grade, exposure-weighted, from charged-off loans.
      ead_ratio = sum(EAD) / sum(funded amount)     for defaulted loans
      lgd       = sum(loss) / sum(EAD)"""
    f = realised_loss_fields(hist)
    bad = hist["default"].to_numpy() == 1
    d = f[bad].assign(grade=np.asarray(grade)[bad])
    t = d.groupby("grade").agg(defaults=("ead", "size"), funded=("funded", "sum"), ead=("ead", "sum"),
                               loss=("loss", "sum"),
                               no_recovery=("net_recovery", lambda s: (s <= 0).mean()))
    t.loc["all"] = [len(d), d["funded"].sum(), d["ead"].sum(), d["loss"].sum(), (d["net_recovery"] <= 0).mean()]
    t["ead_ratio"] = t["ead"] / t["funded"]
    t["lgd"] = t["loss"] / t["ead"]
    t["defaults"] = t["defaults"].astype(int)
    return t[["defaults", "ead_ratio", "lgd", "no_recovery"]]


def loss_parameters_by_year(hist: pd.DataFrame) -> pd.DataFrame:
    """Same measures by issue year: checks the parameters are stable over time."""
    f = realised_loss_fields(hist)
    bad = hist["default"].to_numpy() == 1
    d = f[bad].assign(issue_year=hist.loc[bad, "issue_year"].to_numpy())
    t = d.groupby("issue_year").agg(defaults=("ead", "size"), funded=("funded", "sum"), ead=("ead", "sum"),
                                    loss=("loss", "sum"),
                                    no_recovery=("net_recovery", lambda s: (s <= 0).mean()))
    t["ead_ratio"], t["lgd"] = t["ead"] / t["funded"], t["loss"] / t["ead"]
    return t[["defaults", "ead_ratio", "lgd", "no_recovery"]]


def size_band(amount: pd.Series) -> pd.Series:
    return pd.cut(amount, [0, 5_000, 10_000, 15_000, 25_000, np.inf],
                  labels=["≤ $5k", "$5k–10k", "$10k–15k", "$15k–25k", "> $25k"])


def build_portfolio(df: pd.DataFrame, pd_pred: np.ndarray, grade: np.ndarray,
                    params: pd.DataFrame) -> pd.DataFrame:
    """One row per loan with PD, LGD, EAD, EL and the realised outcome."""
    g = pd.Series(np.asarray(grade), index=df.index)
    real = realised_loss_fields(df)
    p = pd.DataFrame({
        "issue_d": df["issue_d"], "grade": g, "purpose": df["purpose"], "home_ownership": df["home_ownership"],
        "size_band": size_band(df["funded_amnt"]), "int_rate": df["int_rate"],
        "exposure": df["funded_amnt"].astype(float), "pd": np.asarray(pd_pred, float),
        "ead_ratio": g.map(params["ead_ratio"]).astype(float), "lgd": g.map(params["lgd"]).astype(float),
        "default": df["default"].to_numpy(), "realised_loss": real["loss"].to_numpy(),
        "total_pymnt": df["total_pymnt"].astype(float),
    }, index=df.index)
    p["ead"] = p["exposure"] * p["ead_ratio"]
    p["el"] = p["pd"] * p["lgd"] * p["ead"]
    return p


def el_by_segment(port: pd.DataFrame, by: str) -> pd.DataFrame:
    t = port.groupby(by, observed=True).agg(loans=("pd", "size"), exposure=("exposure", "sum"),
                                            mean_pd=("pd", "mean"), el=("el", "sum"),
                                            realised_loss=("realised_loss", "sum"), defaults=("default", "mean"))
    t["exposure_share"] = t["exposure"] / t["exposure"].sum()
    t["el_share"] = t["el"] / t["el"].sum()
    t["el_rate"] = t["el"] / t["exposure"]
    t["realised_loss_rate"] = t["realised_loss"] / t["exposure"]
    t = t.rename(columns={"defaults": "observed_dr"})
    return t[["loans", "exposure", "exposure_share", "mean_pd", "observed_dr", "el", "el_share",
              "el_rate", "realised_loss_rate"]]


def summary(port: pd.DataFrame) -> dict:
    exp = port["exposure"].sum()
    return {"loans": len(port), "exposure": exp, "mean_pd": port["pd"].mean(),
            "exposure_weighted_pd": np.average(port["pd"], weights=port["exposure"]),
            "ead": port["ead"].sum(), "el": port["el"].sum(), "el_rate": port["el"].sum() / exp,
            "observed_dr": port["default"].mean(), "realised_loss": port["realised_loss"].sum(),
            "realised_loss_rate": port["realised_loss"].sum() / exp}


def concentration(port: pd.DataFrame) -> pd.DataFrame:
    """Name concentration (single loans) and segment concentration (Herfindahl index)."""
    e = port["exposure"].to_numpy()
    s = np.sort(e / e.sum())[::-1]
    hhi = float(np.sum(s ** 2))
    rows = {
        "Loans": len(e),
        "Largest single exposure ($)": e.max(),
        "Largest single exposure (share)": s[0],
        "Top 1% of loans (share of exposure)": s[: max(1, len(s) // 100)].sum(),
        "Herfindahl index, loans": hhi,
        "Effective number of loans (1 / HHI)": 1 / hhi,
    }
    for col in ("purpose", "grade", "size_band"):
        w = port.groupby(col, observed=True)["exposure"].sum()
        w = w / w.sum()
        rows[f"Herfindahl index, {col.replace('_', ' ')}"] = float((w ** 2).sum())
        rows[f"Largest {col.replace('_', ' ')} (share)"] = f"{w.idxmax()} ({w.max():.1%})"
    return pd.Series(rows, name="value").to_frame()


def format_concentration(conc: pd.DataFrame) -> pd.DataFrame:
    """Readable copy of concentration(): shares as percentages, tiny indices in scientific notation."""
    def fmt(name, v):
        if not isinstance(v, (int, float, np.floating, np.integer)):
            return str(v)
        if "share" in name:
            return f"{v:.3%}" if v < 0.01 else f"{v:.1%}"
        if "Herfindahl" in name:
            return f"{v:.2e}" if v < 0.001 else f"{v:.3f}"
        return f"{v:,.0f}"
    return pd.DataFrame({"value": [fmt(k, v) for k, v in conc["value"].items()]}, index=conc.index)


def el_sensitivity(port: pd.DataFrame, pd_multipliers=C.PD_MULTIPLIERS,
                   lgd_shifts=C.LGD_SHIFTS) -> pd.DataFrame:
    """Portfolio EL as % of exposure for scaled PDs (rows) and shifted LGDs (columns)."""
    exp = port["exposure"].sum()
    out = {}
    for s in lgd_shifts:
        lgd = (port["lgd"] + s).clip(0, 1)
        out[f"LGD {s:+.0%}"] = {f"PD x{m:g}": float((np.clip(port["pd"] * m, 0, 1) * lgd * port["ead"]).sum() / exp)
                                for m in pd_multipliers}
    return pd.DataFrame(out)
