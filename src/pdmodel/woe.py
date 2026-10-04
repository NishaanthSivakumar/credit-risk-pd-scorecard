"""Weight-of-Evidence (WoE) binning and Information Value (IV), written from scratch.

Why WoE?  A scorecard is a logistic regression on *binned* features. Binning
  - handles outliers and missing values explicitly (missing gets its own bin),
  - lets non-linear relationships enter a linear model,
  - and lets us force each feature's risk pattern to be monotonic, which makes
    the model explainable to credit officers and validators.

Conventions (standard in credit scoring):
  WoE_i = ln( %good_i / %bad_i )       -> positive WoE = safer than average
  IV    = sum_i (%good_i - %bad_i) * WoE_i
  IV rule of thumb: <0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, >0.3 strong,
  >0.5 suspiciously strong (check for leakage).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C

SMOOTH = 0.5  # added to good/bad counts so an empty cell never gives log(0)


def _woe_iv(goods: np.ndarray, bads: np.ndarray, tot_good: float, tot_bad: float):
    g = (goods + SMOOTH) / tot_good
    b = (bads + SMOOTH) / tot_bad
    woe = np.log(g / b)
    return woe, (g - b) * woe


@dataclass
class FeatureBins:
    name: str
    kind: str                          # "numeric" or "categorical"
    edges: np.ndarray | None = None    # interior cut points, right-closed bins
    woe: np.ndarray | None = None      # one value per non-missing bin
    missing_woe: float = 0.0
    cat_map: dict = field(default_factory=dict)   # category -> group label
    cat_woe: dict = field(default_factory=dict)   # group label -> WoE
    table: pd.DataFrame | None = None

    @property
    def iv(self) -> float:
        return float(self.table["iv_contrib"].sum())

    # -- mapping raw values to bins --------------------------------------
    def bin_labels(self, x: pd.Series) -> pd.Series:
        if self.kind == "numeric":
            labels = _interval_labels(self.edges)
            idx = np.searchsorted(self.edges, x.to_numpy(dtype=float), side="left")
            out = pd.Series(np.array(labels, dtype=object)[np.minimum(idx, len(labels) - 1)],
                            index=x.index)
            out[x.isna()] = "Missing"
            return out
        groups = x.astype(str).map(self.cat_map)
        fallback = "OTHER" if "OTHER" in self.cat_woe else None
        return groups.fillna(fallback if fallback else "UNSEEN")

    def transform(self, x: pd.Series) -> pd.Series:
        if self.kind == "numeric":
            idx = np.searchsorted(self.edges, x.to_numpy(dtype=float), side="left")
            out = pd.Series(self.woe[np.minimum(idx, len(self.woe) - 1)], index=x.index)
            out[x.isna()] = self.missing_woe
            return out
        return self.bin_labels(x).map(self.cat_woe).fillna(0.0).astype(float)


def _interval_labels(edges: np.ndarray) -> list[str]:
    pts = [-np.inf, *edges, np.inf]
    fmt = lambda v: "-inf" if v == -np.inf else ("inf" if v == np.inf else f"{v:,.4g}")
    return [f"({fmt(a)}, {fmt(b)}]" for a, b in zip(pts[:-1], pts[1:])]


def _bin_counts(xv: np.ndarray, yv: np.ndarray, edges: np.ndarray):
    idx = np.searchsorted(edges, xv, side="left")
    k = len(edges) + 1
    n = np.bincount(idx, minlength=k).astype(float)
    bad = np.bincount(idx, weights=yv, minlength=k).astype(float)
    return n, bad


def _fit_numeric(name, x, y, tot_good, tot_bad, max_bins, min_n, monotonic) -> FeatureBins:
    mask = x.notna().to_numpy()
    xv = x.to_numpy(dtype=float)[mask]
    yv = y.to_numpy(dtype=float)[mask]

    # 1) start from quantile cut points (duplicates collapse for discrete features)
    qs = np.linspace(0, 1, max_bins + 1)[1:-1]
    edges = np.unique(np.quantile(xv, qs)) if len(xv) else np.array([])
    edges = edges[edges < xv.max()] if len(xv) else edges  # last bin must not be empty

    # 2) merge bins that are too small (into the neighbour with the closest bad rate)
    while len(edges) > 0:
        n, bad = _bin_counts(xv, yv, edges)
        small = np.where(n < min_n)[0]
        if len(small) == 0:
            break
        j = small[np.argmin(n[small])]
        rate = bad / np.maximum(n, 1)
        if j == 0:
            drop = 0
        elif j == len(n) - 1:
            drop = j - 1
        else:
            drop = j - 1 if abs(rate[j] - rate[j - 1]) <= abs(rate[j] - rate[j + 1]) else j
        edges = np.delete(edges, drop)

    # 3) enforce a monotonic bad rate (direction taken from the overall trend)
    if monotonic and len(edges) > 0:
        direction = np.sign(np.corrcoef(xv, yv)[0, 1]) or 1.0
        while len(edges) > 0:
            n, bad = _bin_counts(xv, yv, edges)
            rate = bad / n
            viol = np.where(direction * np.diff(rate) < 0)[0]
            if len(viol) == 0:
                break
            # merge the violating pair whose bad rates are closest
            i = viol[np.argmin(np.abs(np.diff(rate)[viol]))]
            edges = np.delete(edges, i)

    n, bad = _bin_counts(xv, yv, edges)
    woe, ivc = _woe_iv(n - bad, bad, tot_good, tot_bad)
    labels = _interval_labels(edges)
    rows = pd.DataFrame({"bin": labels, "n": n, "bads": bad})
    rows["woe"], rows["iv_contrib"] = woe, ivc

    miss_n = float((~mask).sum())
    missing_woe = 0.0
    if miss_n > 0:
        miss_bad = float(y.to_numpy()[~mask].sum())
        mw, mivc = _woe_iv(np.array([miss_n - miss_bad]), np.array([miss_bad]), tot_good, tot_bad)
        missing_woe = float(mw[0])
        rows = pd.concat([rows, pd.DataFrame({"bin": ["Missing"], "n": [miss_n], "bads": [miss_bad],
                                              "woe": mw, "iv_contrib": mivc})], ignore_index=True)

    return FeatureBins(name, "numeric", edges=edges, woe=woe,
                       missing_woe=missing_woe, table=_finish_table(rows, name))


def _fit_categorical(name, x, y, tot_good, tot_bad, rare_frac) -> FeatureBins:
    xs = x.astype(str)
    freq = xs.value_counts(normalize=True)
    cat_map = {c: (c if f >= rare_frac else "OTHER") for c, f in freq.items()}
    g = xs.map(cat_map)
    agg = pd.DataFrame({"g": g, "y": y.to_numpy()}).groupby("g")["y"].agg(["count", "sum"])
    woe, ivc = _woe_iv((agg["count"] - agg["sum"]).to_numpy(), agg["sum"].to_numpy(), tot_good, tot_bad)
    rows = pd.DataFrame({"bin": agg.index, "n": agg["count"].to_numpy(dtype=float),
                         "bads": agg["sum"].to_numpy(dtype=float), "woe": woe, "iv_contrib": ivc})
    rows = rows.sort_values("woe").reset_index(drop=True)
    return FeatureBins(name, "categorical", cat_map=cat_map,
                       cat_woe=dict(zip(rows["bin"], rows["woe"])), table=_finish_table(rows, name))


def _finish_table(rows: pd.DataFrame, name: str) -> pd.DataFrame:
    rows = rows.copy()
    rows.insert(0, "feature", name)
    rows["pct_of_sample"] = rows["n"] / rows["n"].sum()
    rows["bad_rate"] = rows["bads"] / rows["n"]
    return rows[["feature", "bin", "n", "pct_of_sample", "bads", "bad_rate", "woe", "iv_contrib"]]


class WoEBinner:
    """Fit on the training sample only; then transform any sample with the same bins."""

    def __init__(self, max_bins=C.MAX_BINS, min_bin_frac=C.MIN_BIN_FRAC,
                 rare_cat_frac=C.RARE_CAT_FRAC, monotonic=True):
        self.max_bins, self.min_bin_frac = max_bins, min_bin_frac
        self.rare_cat_frac, self.monotonic = rare_cat_frac, monotonic
        self.bins: dict[str, FeatureBins] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, numeric: list[str], categorical: list[str]):
        y = pd.Series(np.asarray(y), index=X.index)
        tot_bad = float(y.sum())
        tot_good = float(len(y) - tot_bad)
        min_n = self.min_bin_frac * len(y)
        for f in numeric:
            self.bins[f] = _fit_numeric(f, X[f], y, tot_good, tot_bad,
                                        self.max_bins, min_n, self.monotonic)
        for f in categorical:
            self.bins[f] = _fit_categorical(f, X[f], y, tot_good, tot_bad, self.rare_cat_frac)
        return self

    def transform(self, X: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
        features = features or list(self.bins)
        return pd.DataFrame({f: self.bins[f].transform(X[f]) for f in features}, index=X.index)

    def iv_table(self) -> pd.DataFrame:
        iv = pd.Series({f: b.iv for f, b in self.bins.items()}, name="iv").sort_values(ascending=False)
        strength = pd.cut(iv, [-np.inf, 0.02, 0.1, 0.3, 0.5, np.inf],
                          labels=["useless", "weak", "medium", "strong", "suspicious"])
        return pd.DataFrame({"iv": iv, "strength": strength,
                             "n_bins": [len(self.bins[f].table) for f in iv.index]})

    def woe_table(self, features: list[str] | None = None) -> pd.DataFrame:
        features = features or list(self.bins)
        return pd.concat([self.bins[f].table for f in features], ignore_index=True)
