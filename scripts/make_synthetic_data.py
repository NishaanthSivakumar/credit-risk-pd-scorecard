"""Generate a FAKE LendingClub-shaped file, only for smoke-testing the code.

    python scripts/make_synthetic_data.py --n 200000

Writes data/raw/synthetic/accepted_2007_to_2018Q4_SYNTHETIC.csv.gz
Never report results from this file - use the real Kaggle data.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def make(n: int, seed: int = 0) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    year = r.choice(np.arange(2007, 2019), n, p=np.array([1, 2, 3, 5, 8, 12, 18, 22, 25, 20, 18, 16]) / 150)
    month = r.integers(1, 13, n)
    issue = pd.to_datetime(dict(year=year, month=month, day=1))
    fico_low = np.clip(r.normal(695, 30, n), 660, 845).round(0) // 5 * 5
    annual_inc = np.exp(r.normal(11.1, 0.55, n)).round(-2)
    loan_amnt = np.clip(r.lognormal(9.4, 0.6, n), 1000, 40000).round(-2)
    dti = np.clip(r.gamma(4, 4.5, n), 0, 45).round(2)
    revol_util = np.clip(r.normal(55, 24, n), 0, 130).round(1)
    inq = r.poisson(0.8, n)
    hist_years = np.clip(r.gamma(5, 3.2, n), 1, 50)
    earliest = issue - pd.to_timedelta((hist_years * 365.25).astype(int), unit="D")
    emp_opts = np.array(["< 1 year"] + [f"{i} year" + ("s" if i > 1 else "") for i in range(1, 10)] + ["10+ years", np.nan], dtype=object)
    emp = r.choice(emp_opts, n, p=np.array([8, 7, 9, 8, 6, 6, 5, 5, 5, 4, 32, 5]) / 100)
    home = r.choice(["MORTGAGE", "RENT", "OWN", "OTHER"], n, p=[0.49, 0.40, 0.105, 0.005])
    purpose = r.choice(["debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase",
                        "small_business", "car", "medical", "wedding"], n,
                       p=[0.59, 0.22, 0.06, 0.06, 0.02, 0.02, 0.015, 0.01, 0.005])
    verif = r.choice(["Not Verified", "Source Verified", "Verified"], n, p=[0.35, 0.38, 0.27])
    term = np.where(r.random(n) < 0.75, " 36 months", " 60 months")
    mort_acc = np.where(year < 2012, np.nan, r.poisson(1.6, n))

    # True default log-odds (risk drifts up for later vintages)
    z = (-1.75 - 0.018 * (fico_low - 695) + 0.035 * (dti - 18) + 0.012 * (revol_util - 55)
         + 0.20 * inq - 0.25 * (np.log(annual_inc) - 11.1) + 0.25 * (np.log(loan_amnt) - 9.4)
         - 0.015 * (hist_years - 16) + 0.30 * (home == "RENT") + 0.55 * (purpose == "small_business")
         - 0.15 * (purpose == "credit_card") + 0.15 * (verif == "Verified")
         + 0.06 * (year - 2012).clip(0, None) + r.normal(0, 0.35, n))
    p = 1 / (1 + np.exp(-z))
    defaulted = r.random(n) < p
    unresolved = (year >= 2016) & (r.random(n) < (year - 2015) * 0.3)
    status = np.where(unresolved, "Current", np.where(defaulted, "Charged Off", "Fully Paid"))
    status[(r.random(n) < 0.01) & (year < 2010)] = "Does not meet the credit policy. Status:Fully Paid"

    rate = np.clip(6 + 0.9 * (z + 3) * 4 + r.normal(0, 1.5, n), 5.3, 30.9).round(2)
    gi = np.clip(((rate - 5) / 26 * 35).astype(int), 0, 34)
    sub_grade = np.array([f"{g}{i}" for g in "ABCDEFG" for i in range(1, 6)])[gi]

    return pd.DataFrame({
        "id": np.arange(n), "loan_amnt": loan_amnt, "term": term, "int_rate": rate,
        "installment": (loan_amnt / 30).round(2), "grade": [s[0] for s in sub_grade], "sub_grade": sub_grade,
        "emp_length": emp, "home_ownership": home, "annual_inc": annual_inc, "verification_status": verif,
        "issue_d": issue.dt.strftime("%b-%Y"), "loan_status": status, "purpose": purpose,
        "addr_state": "CA", "dti": dti, "delinq_2yrs": r.poisson(0.3, n),
        "earliest_cr_line": earliest.dt.strftime("%b-%Y"), "fico_range_low": fico_low,
        "fico_range_high": fico_low + 4, "inq_last_6mths": inq, "open_acc": r.poisson(11, n),
        "pub_rec": r.poisson(0.2, n), "revol_bal": (r.lognormal(9.3, 1.0, n)).round(0),
        "revol_util": revol_util, "total_acc": r.poisson(24, n), "mort_acc": mort_acc,
        "total_pymnt": 0.0,  # post-origination field present in the real file; never used
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    a = ap.parse_args()
    out = ROOT / "data" / "raw" / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    df = make(a.n)
    path = out / "accepted_2007_to_2018Q4_SYNTHETIC.csv.gz"
    df.to_csv(path, index=False)
    print(f"Wrote {len(df):,} synthetic rows -> {path}")
