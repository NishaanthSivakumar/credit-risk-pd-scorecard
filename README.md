# Credit Risk PD Scorecard — LendingClub

A probability-of-default (PD) scorecard for unsecured consumer loans, built and **validated**
the way a bank's credit risk and model validation teams would do it, and extended into a
**credit portfolio risk engine**:

- **Sample design** that avoids leakage and resolution bias (36-month loans, vintages 2007–2015),
  checked by an automated **data audit**
- **WoE binning + IV**, implemented from scratch with monotonic bins and an explicit missing bin
- **Logistic regression scorecard** with significance, sign and VIF checks, scaled to points (PDO)
- **Challenger model** (gradient boosting) and an **external benchmark** (LendingClub's own sub-grade)
- **Chronological validation**: develop on 2007–13, recalibrate on 2014, test once on 2015
- **Validation**: Gini/KS, rank ordering, calibration intercept and slope, binomial test per grade,
  PSI/CSI, **PD recalibration**, **block-bootstrap confidence intervals**, a paired challenger test,
  **sensitivity analysis** and a ten-dimension assessment with fixed **decision rules**
- **Portfolio risk**: LGD and EAD measured from recoveries, expected loss by segment, and a
  **one-factor Gaussian Monte Carlo** engine for lifetime VaR and Expected Shortfall
- Auto-generated reports: [`validation_report.md`](reports/validation_report.md),
  [`data_audit.md`](reports/data_audit.md), [`portfolio_risk_report.md`](reports/portfolio_risk_report.md)
  and a [limitations register](reports/limitations_register.md)

## Results

Full run on 621,022 resolved 36-month loans (numbers from `reports/metrics.json` and
`reports/tables/`).

| Sample | Loans | Default rate | Gini | KS | Score PSI vs train |
|---|---|---|---|---|---|
| Train (2007–13) | 122,798 | 12.6% | 0.316 | 0.227 | – |
| Test (2007–13) | 52,628 | 12.6% | 0.305 | 0.221 | 0.000 |
| Calibration (2014) | 162,570 | 13.7% | 0.293 | 0.213 | 0.007 |
| **Out-of-time (2015)** | **283,026** | **14.9%** | **0.296** | **0.213** | **0.001** |

Out-of-time Gini: scorecard 0.296 (95% interval 0.292–0.300) · gradient-boosting challenger 0.318 ·
LendingClub sub-grade 0.357.

**Key finding: the scorecard ranks risk consistently, but its PD level drifts year to year, and a
recalibration fitted on the previous year is not enough.** Out of time, Gini is only 6% below
train, default rates fall monotonically from 27.2% in the worst score decile to 4.3% in the best,
and the challenger's edge (+0.022 Gini) is below the materiality threshold. But on 2015 loans the
model predicts 12.54% against 14.89% observed. Re-fitting the intercept on 2014 loans closes only a
quarter of that gap (to 13.18%), because the default rate kept rising while the borrowers' profile
looked *safer* on paper. The validation decision is **Recalibrate**: on the latest mature vintage,
confirmed on the next, with a time-varying calibration to follow. The validation also found that
leaving income or credit inquiries blank *lowers* a borrower's PD, because those fields were
almost never missing in training.

**Portfolio:** on the 2015 book ($3.62bn funded), expected lifetime loss is 6.35% of funded
amount, against 7.56% realised; the shortfall is the PD gap, as loss severity held within 2%. The
99% lifetime VaR is 13.45% and the 99% Expected Shortfall 15.0%, with the Basel retail
correlation; the correlation assumption moves the 99% VaR from 10% to 21%.

![Default rate by score decile, out-of-time](reports/figures/rank_ordering_oot.png)

## Project structure

```
├── IMPLEMENTATION.md                 improvement roadmap and progress tracker
├── notebooks/
│   ├── 01_data_and_target.ipynb      sample definition, target, vintage analysis, splits
│   ├── 02_woe_binning_iv.ipynb       WoE bins, IV ranking
│   ├── 03_scorecard_model.ipynb      selection, logistic regression, points, challenger
│   ├── 04_model_validation.ipynb     discrimination, calibration, stability
│   ├── 05_recalibration_and_uncertainty.ipynb   recalibration, bootstrap intervals, challenger test
│   ├── 06_sensitivity_and_decision.ipynb        data audit, sensitivity, validation decision
│   └── 07_portfolio_risk.ipynb                  LGD/EAD, expected loss, Monte Carlo VaR/ES
├── src/pdmodel/
│   ├── config.py        every setting (dates, thresholds, scaling) in one place
│   ├── data.py          loading, target definition, feature engineering, splits
│   ├── audit.py         data audit: target, leakage, maturity, data quality
│   ├── woe.py           WoE binning and IV (from scratch)
│   ├── model.py         feature selection, logistic regression, scorecard, challenger
│   ├── validation.py    Gini/KS, calibration tests, PSI/CSI, RAG thresholds
│   ├── calibration.py   calibration intercept/slope, recalibration
│   ├── bootstrap.py     loan-level and month-block bootstrap, paired comparison
│   ├── sensitivity.py   inputs, missing values, coefficients, thresholds, alternatives
│   ├── assessment.py    ten validation dimensions and the decision rules
│   ├── portfolio.py     LGD/EAD measurement, expected loss, concentration
│   ├── simulation.py    one-factor Gaussian Monte Carlo, VaR/ES, correlation
│   ├── plots.py         charts
│   ├── reports.py       report writers
│   └── pipeline.py      end-to-end run
├── scripts/
│   ├── run_pipeline.py          run everything, write reports/
│   └── make_synthetic_data.py   fake data for smoke-testing only
├── tests/                       unit tests for every module
└── reports/                     reports, commentary, limitations register, tables, figures
```

## Getting the data

1. Create a free Kaggle account and download **Lending Club Loan Data** by *wordsforthewise*:
   <https://www.kaggle.com/datasets/wordsforthewise/lending-club>
   (or `kaggle datasets download -d wordsforthewise/lending-club -p data/raw --unzip`).
2. Leave `accepted_2007_to_2018Q4.csv(.gz)` anywhere under `data/raw/` — the code finds it.

The file is ~1.6 GB unzipped; only 30 columns are loaded (24 application fields and 6 outcome
fields used only to measure losses), which needs a few GB of RAM.
`data/` is git-ignored.

## How to run

Tested with Python 3.13.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_pipeline.py           # full run, about 10 minutes
python scripts/run_pipeline.py --quick   # fewer bootstrap draws and scenarios, for a fast check
python -m pytest -q                      # unit tests
```

The pipeline writes every table and figure plus the three reports, inserting the analysis from
`reports/commentary.md` and `reports/portfolio_commentary.md`. Random seeds are fixed in
`config.py`, so a re-run reproduces every number.

The notebooks (01 → 07) run the same steps one at a time and display every result, with the
findings at the end of each. They call the code in `src/pdmodel/`, so the notebooks, the pipeline
and the tests all use the same calculations. Run them in order: each saves what the next one needs
to `data/processed/`.

`python scripts/make_synthetic_data.py` writes a fake LendingClub-shaped file for testing the code
without the real data (run the pipeline with
`--data data/raw/synthetic/accepted_2007_to_2018Q4_SYNTHETIC.csv.gz`). Its results are meaningless.

## Methodology notes

| Decision | Why |
|---|---|
| Default = Charged Off / Default; Current & Late excluded | Only loans with a known final outcome |
| 36m loans issued ≤ 2015 only | Every loan had time to mature in the 2018Q4 snapshot (no resolution bias) |
| No post-origination fields | `total_pymnt`, `recoveries`, … reveal the outcome (leakage) |
| `grade`, `sub_grade`, `int_rate` excluded | They are LendingClub's own risk view; used as a benchmark instead |
| `addr_state` excluded | Geographic variables raise fair-lending concerns |
| Features >10% missing in train dropped | `mort_acc` only exists from 2012, so its "Missing" bin was really a vintage flag |
| Monotonic WoE bins, ≥5% per bin | Explainable, stable points; no noisy tiny bins |
| All WoE coefficients must be negative | Higher WoE = safer; a positive sign means collinearity |
| Out-of-time validation | Tests the model on future vintages, as it would be used |
| Recalibrate on 2014, test on 2015 | The recalibration is fitted and tested on different vintages, and the test year is used once |
| Recalibration method chosen on 2014 alone | Intercept if it differs significantly from 0; slope only if it differs from 1 by over 0.10 |
| Bootstrap by issue month | Loans issued in the same month share the economy, so their outcomes are not independent |
| Decision rules fixed in code | `assessment.py` maps the ten dimensions to approve / conditionally approve / recalibrate / redevelop / reject |
| LGD and EAD measured on 2007–14 charge-offs | Real recoveries rather than assumptions, without using the 2015 book's own outcomes |
| Lifetime (36-month) loss measures | Match the lifetime PD; not comparable with one-year capital |
| Basel retail correlation, not the data estimate | The data only covers 2009–15, a recovery without a recession |

## Limitations

The main ones, in full in the [limitations register](reports/limitations_register.md):

- **Reject inference:** only approved loans are observed, so the model learns from a filtered population.
- **Lifetime vs 12-month PD:** this is a 36-month lifetime PD; Basel IRB and IFRS 9 stage 1 use 12-month PD.
- **No macroeconomic drivers:** PDs do not move with the economy, and a lagged recalibration trails a rising default rate.
- **Missing values:** rare missing bins carry a WoE estimated from a handful of loans.
- **One-factor Gaussian dependence:** no sector factors, and LGD does not move with the economy.
