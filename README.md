# Credit Risk PD Scorecard — LendingClub

A probability-of-default (PD) scorecard for unsecured consumer loans, built and **validated**
the way a bank's credit risk and model validation teams would do it:

- **Sample design** that avoids leakage and resolution bias (36-month loans, vintages 2007–2015)
- **WoE binning + IV**, implemented from scratch with monotonic bins and an explicit missing bin
- **Logistic regression scorecard** with significance, sign and VIF checks, scaled to points (PDO)
- **Challenger model** (gradient boosting) and an **external benchmark** (LendingClub's own sub-grade)
- **Validation** on an **out-of-time** sample: Gini/KS, rank ordering, calibration, Hosmer–Lemeshow,
  binomial test per rating grade, PSI and CSI, with traffic-light (RAG) status
- An auto-generated **validation report** (`reports/validation_report.md`)

## Results

> ✍️ Fill this in after running on the real data (numbers are in `reports/metrics.json`).

| Sample | Loans | Default rate | Gini | KS | Score PSI vs train |
|---|---|---|---|---|---|
| Train (≤2013) | | | | | – |
| Test (≤2013) | | | | | |
| **Out-of-time (2014–15)** | | | | | |

Scorecard Gini vs. gradient-boosting challenger: … · vs. LendingClub sub-grade: …

Key finding: …

## Project structure

```
├── notebooks/
│   ├── 01_data_and_target.ipynb      sample definition, target, vintage analysis, splits
│   ├── 02_woe_binning_iv.ipynb       WoE bins, IV ranking
│   ├── 03_scorecard_model.ipynb      selection, logistic regression, points, challenger
│   └── 04_model_validation.ipynb     discrimination, calibration, stability
├── src/pdmodel/
│   ├── config.py      every setting (dates, thresholds, scaling) in one place
│   ├── data.py        loading, target definition, feature engineering, splits
│   ├── woe.py         WoE binning and IV (from scratch)
│   ├── model.py       feature selection, logistic regression, scorecard, challenger
│   ├── validation.py  Gini/KS, calibration tests, PSI/CSI, RAG thresholds
│   ├── plots.py       charts
│   └── pipeline.py    end-to-end run + report writer
├── scripts/
│   ├── run_pipeline.py          run everything, write reports/
│   └── make_synthetic_data.py   fake data for smoke-testing only
├── tests/test_core.py
└── reports/                     generated tables, figures, validation_report.md
```

## Getting the data

1. Create a free Kaggle account and download **Lending Club Loan Data** by *wordsforthewise*:
   <https://www.kaggle.com/datasets/wordsforthewise/lending-club>
   (or `kaggle datasets download -d wordsforthewise/lending-club -p data/raw --unzip`).
2. Leave `accepted_2007_to_2018Q4.csv(.gz)` anywhere under `data/raw/` — the code finds it.

The file is ~1.6 GB unzipped; only 24 columns are loaded, which needs a few GB of RAM.
`data/` is git-ignored — don't commit the data.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_pipeline.py              # full run, a few minutes
python scripts/run_pipeline.py --nrows 300000   # quick first run
python -m pytest -q                          # unit tests
```

Then open the notebooks in order (01 → 04). Each one saves what the next one needs to `data/processed/`.

No data yet? `python scripts/make_synthetic_data.py` writes a fake LendingClub-shaped file for
checking the code runs. Point the pipeline at it with
`--data data/raw/synthetic/accepted_2007_to_2018Q4_SYNTHETIC.csv.gz`. **Never report synthetic results.**

## Methodology notes

| Decision | Why |
|---|---|
| Default = Charged Off / Default; Current & Late excluded | Only loans with a known final outcome |
| 36m loans issued ≤ 2015 only | Every loan had time to mature in the 2018Q4 snapshot (no resolution bias) |
| No post-origination fields | `total_pymnt`, `recoveries`, … reveal the outcome (leakage) |
| `grade`, `sub_grade`, `int_rate` excluded | They are LendingClub's own risk view; used as a benchmark instead |
| `addr_state` excluded | Geographic variables raise fair-lending concerns |
| Monotonic WoE bins, ≥5% per bin | Explainable, stable points; no noisy tiny bins |
| All WoE coefficients must be negative | Higher WoE = safer; a positive sign means collinearity |
| Out-of-time validation | Tests the model on future vintages, as it would be used |

## Limitations

- **Reject inference:** only approved loans are observed, so the model learns from a filtered population.
- **Lifetime vs 12-month PD:** this is a 36-month lifetime PD; Basel IRB and IFRS 9 stage 1 use 12-month PD.
- **Through-the-cycle:** no macroeconomic drivers, so PDs don't move with the economy.
