"""Loading the LendingClub file, defining the default flag and splitting samples."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


def find_raw_file(raw_dir: Path = C.RAW_DIR) -> Path:
    """Find the Kaggle 'accepted_2007_to_2018Q4' file wherever it was unzipped."""
    hits = sorted(
        p for p in Path(raw_dir).rglob("accepted_2007_to_2018*")
        if p.is_file() and p.suffix in {".csv", ".gz"}
    )
    if not hits:
        raise FileNotFoundError(
            f"No 'accepted_2007_to_2018Q4.csv(.gz)' found under {raw_dir}. "
            "See README.md -> 'Getting the data'."
        )
    return hits[0]


def load_raw(path: Path | None = None, nrows: int | None = None) -> pd.DataFrame:
    """Read only the columns we need (the full file has 151 columns, ~2.2M rows)."""
    path = Path(path) if path else find_raw_file()
    wanted = set(C.RAW_COLUMNS)
    df = pd.read_csv(path, usecols=lambda c: c in wanted, nrows=nrows, low_memory=False)
    missing = wanted - set(df.columns)
    if missing:
        raise ValueError(f"Raw file is missing columns: {sorted(missing)}")
    return df


def _to_pct_number(s: pd.Series) -> pd.Series:
    """'13.5%' -> 13.5 ; already-numeric columns pass through."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype(str).str.strip().str.rstrip("%"), errors="coerce")


def _emp_length_years(s: pd.Series) -> pd.Series:
    """'< 1 year' -> 0, '10+ years' -> 10, 'n/a' -> NaN."""
    txt = s.astype("string").str.strip()
    out = txt.str.extract(r"(\d+)")[0].astype(float)
    out[txt.str.startswith("<", na=False)] = 0.0
    return out


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the sample definition, build the target and engineer features.

    Returns one row per loan with: issue_d, issue_year, default (0/1),
    model features, and the benchmark column (sub_grade).
    """
    df = raw.copy()

    # Parse dates. The Kaggle file ends with two summary rows that have no date.
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    df = df.dropna(subset=["issue_d"])
    df["issue_year"] = df["issue_d"].dt.year

    # Sample definition
    df["term"] = df["term"].astype(str).str.strip()
    df = df[(df["term"] == C.TERM)
            & df["issue_year"].between(C.FIRST_ISSUE_YEAR, C.LAST_ISSUE_YEAR)]

    # Target: 1 = default (charged off), 0 = fully paid, everything else dropped
    status = df["loan_status"].astype(str).str.strip()
    df["default"] = np.where(status.isin(C.BAD_STATUSES), 1,
                     np.where(status.isin(C.GOOD_STATUSES), 0, -1))
    df = df[df["default"] >= 0].copy()

    # Feature engineering (application-time information only)
    df["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df["credit_hist_years"] = (df["issue_d"] - df["earliest_cr_line"]).dt.days / 365.25
    df["emp_length_yrs"] = _emp_length_years(df["emp_length"])
    df["revol_util"] = _to_pct_number(df["revol_util"])
    df["int_rate"] = _to_pct_number(df["int_rate"])
    inc = df["annual_inc"].where(df["annual_inc"] > 0)
    df["loan_to_income"] = df["loan_amnt"] / inc

    df["home_ownership"] = (df["home_ownership"].astype(str).str.upper()
                            .replace({"NONE": "OTHER", "ANY": "OTHER"}))
    for c in C.CATEGORICAL_FEATURES:
        df[c] = df[c].astype(str).str.strip()

    keep = (["issue_d", "issue_year", "default", C.BENCHMARK_COLUMN, "grade", "int_rate"]
            + C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES)
    return df[keep].reset_index(drop=True)


def split_samples(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Train / test from the development vintages, plus an out-of-time sample."""
    dev = df[df["issue_year"] <= C.DEV_LAST_YEAR]
    oot = df[df["issue_year"] > C.DEV_LAST_YEAR]
    train, test = train_test_split(
        dev, test_size=C.TEST_SIZE, stratify=dev["default"], random_state=C.RANDOM_STATE
    )
    return {"train": train.reset_index(drop=True),
            "test": test.reset_index(drop=True),
            "oot": oot.reset_index(drop=True)}


def sample_summary(samples: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Size, default count, default rate and vintage range per sample."""
    rows = []
    for name, d in samples.items():
        rows.append({
            "sample": name,
            "n_loans": len(d),
            "n_defaults": int(d["default"].sum()),
            "default_rate": d["default"].mean(),
            "first_issue": d["issue_d"].min().strftime("%Y-%m"),
            "last_issue": d["issue_d"].max().strftime("%Y-%m"),
        })
    return pd.DataFrame(rows).set_index("sample")
