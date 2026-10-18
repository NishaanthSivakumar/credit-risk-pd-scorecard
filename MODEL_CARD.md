# Model Card — LendingClub 36-Month PD Scorecard

| | |
|---|---|
| **Model** | Probability-of-default scorecard for 36-month unsecured LendingClub loans |
| **Version** | 1.1: version 1.0 as developed, with the intercept recalibrated on 2014 loans |
| **Status** | Validation decision **Recalibrate**; monitoring triggers point to **redevelopment** (sections 11–14) |
| **Developer** | Nishaanth Sivakumar |
| **Validation** | Self-validation following the structure of an independent review ([validation report](reports/validation_report.md)) |
| **Data** | Kaggle *Lending Club Loan Data* (2007–2018Q4 accepted loans), snapshot March 2019 |
| **Code** | `src/pdmodel/`; reproduce with `python scripts/run_pipeline.py` |

## 1. Model name and version

LendingClub 36-month PD scorecard, **version 1.1**.

| Version | Change | Evidence |
|---|---|---|
| 1.0 | Developed on 2007–13 loans: WoE binning, logistic regression, scaled to points | [Validation report](reports/validation_report.md), sections 4–5 |
| 1.1 | Intercept re-estimated on 2014 loans (+0.059 on the log-odds scale); bins, coefficients and points unchanged | Validation report, section 6.2 |

## 2. Business objective

Estimate, at application, the probability that a 36-month unsecured consumer loan will be charged off
over its life, and rank applicants by that risk. The PD feeds approval cut-offs and the lifetime
expected-loss, stress-testing and allocation analysis in the
[portfolio risk report](reports/portfolio_risk_report.md).

## 3. Intended use

**Intended:**

- Ranking applicants for 36-month unsecured LendingClub-style loans, with a cut-off set on observed
  bad rates.
- Lifetime PD input to portfolio expected loss, stress testing and allocation, *after* a current
  recalibration (section 14).

**Not intended:**

- 60-month loans, secured lending or other products.
- One-year PDs for Basel IRB capital or IFRS 9 stage 1 provisions: the target is a lifetime default.
- Populations that did not pass LendingClub's approval (no reject inference; limitation L1).
- Decisions based on protected characteristics, or any use as evidence that the model is fair
  (limitation L8).
- Automated decisions without human review of declined or borderline applications.

## 4. Target population

Applicants approved by LendingClub for a 36-month unsecured personal loan in the United States, with
the application data LendingClub recorded (FICO band, income, debt-to-income, revolving utilisation,
credit inquiries, credit history, home ownership, loan purpose).

## 5. Training data

| Sample | Vintages | Loans | Default rate | Use |
|---|---|---|---|---|
| Train | 2007–13, random 70% | 122,798 | 12.6% | Bins, feature selection, coefficients |
| Test | 2007–13, random 30% | 52,628 | 12.6% | In-time overfitting check |
| Calibration | 2014 | 162,570 | 13.7% | Intercept recalibration only |
| Out-of-time | 2015 | 283,026 | 14.9% | Final test, used once |

Only loans with a final outcome are used; 147 loans (0.02%) in these vintages were still open at the
snapshot and are excluded. Full audit: [data audit](reports/data_audit.md).

## 6. Target variable

**Default = 1** if the loan was charged off or in default at any point in its 36-month life;
**0** if it was repaid in full, including early repayment (57% of repaid loans finished at least three
months early). The "Does not meet the credit policy" statuses follow the same rule.

## 7. Model methodology

1. Candidate features (18) binned from training data: up to 10 quantile bins, merged until every bin
   holds at least 5% of loans and the bad rate is monotonic; missing values get their own bin.
2. Each bin replaced by its Weight of Evidence, WoE = ln(%good / %bad).
3. Features kept if IV ≥ 0.02, at most 10% missing, and not correlated above 0.7 with a stronger feature.
4. Logistic regression on the WoE values, with backward elimination until every coefficient is
   significant (p < 0.05) and negative.
5. Points scaling: 600 points at 30:1 good:bad odds; every 20 points doubles the odds.
6. Seven rating grades: equal-population score bands on the training sample.
7. Recalibration (version 1.1): logit(PD) shifted by an intercept fitted on 2014 loans; ranking unchanged.

## 8. Input variables

| Feature | Meaning | IV | Coefficient | Points range |
|---|---|---|---|---|
| `fico` | Midpoint of the FICO band at application | 0.142 | −0.814 | 52–85 |
| `annual_inc` | Self-reported annual income | 0.093 | −0.660 | 53–71 |
| `loan_to_income` | Loan amount ÷ annual income | 0.041 | −0.710 | 54–67 |
| `inq_last_6mths` | Credit inquiries in the last 6 months | 0.040 | −1.106 | 48–70 |
| `purpose` | Stated loan purpose | 0.035 | −1.063 | 41–71 |
| `dti` | Debt-to-income ratio, excluding mortgage (%) | 0.035 | −0.482 | 58–65 |
| `revol_util` | Revolving credit utilisation (%) | 0.034 | −0.318 | 59–66 |
| `home_ownership` | Rent, mortgage, own or other | 0.031 | −0.429 | 57–64 |
| `credit_hist_years` | Years since the earliest credit line | 0.021 | −0.185 | 60–63 |

Every coefficient is negative (higher WoE = safer), significant (largest p-value 0.004) and has a VIF
below 1.6. Excluded on purpose: post-origination fields (leakage), LendingClub's own `grade`,
`sub_grade` and `int_rate` (kept as a benchmark), `addr_state` (fair-lending concerns) and `mort_acc`
(only recorded from 2012).

## 9. Model assumptions

- The relationship between application data and default is stable over time; only the overall default
  level drifts, and a recalibrated intercept corrects it. **Phase 1 and monitoring show this assumption
  does not hold beyond about a year** (sections 11 and 13).
- Approved applicants are representative of the population the model is used on.
- A missing value carries the risk of the loans that were missing in training. **This fails for inputs
  that were almost never missing** (limitation L11).
- Inputs stay within the ranges seen in training. **From late 2014, up to 6.6% of loans in a quarter have
  a debt-to-income ratio above the training maximum** (section 13).

## 10. Performance metrics

| Sample | Gini | KS | Mean PD | Default rate | Brier |
|---|---|---|---|---|---|
| Train (2007–13) | 0.316 | 0.227 | 12.63% | 12.63% | 0.1065 |
| Test (2007–13) | 0.305 | 0.221 | 12.61% | 12.63% | 0.1069 |
| Calibration (2014), v1.0 | 0.293 | 0.213 | 13.07% | 13.73% | 0.1148 |
| Out-of-time (2015), v1.0 | 0.296 | 0.213 | 12.54% | 14.89% | 0.1230 |
| Out-of-time (2015), **v1.1** | **0.296** (95% CI 0.292–0.300) | 0.213 | **13.18%** | 14.89% | 0.1227 |

Benchmarks on 2015 loans: gradient-boosting challenger Gini 0.318 (+0.022, below the 0.03 materiality
threshold); LendingClub sub-grade 0.357 (uses more data). Default rates fall monotonically across all
ten score deciles, from 27.2% to 4.3%.

## 11. Validation results

**Decision: Recalibrate** (for both version 1.0 and version 1.1). From the
[validation report](reports/validation_report.md):

| Dimension | Status |
|---|---|
| Conceptual soundness, data quality, target definition | GREEN |
| Discrimination | RED (Gini 0.296 against a 0.30 threshold; no materially better challenger) |
| Calibration | RED: version 1.1 still under-predicts 2015 by 1.71 points; 6 of 7 grades fail |
| Population stability, out-of-time performance, challenger, sensitivity | GREEN |
| Limitations | AMBER (12 in the register, 3 rated High) |

The ranking holds out of time, but the PD level drifts upward faster than a one-year-lagged
recalibration can follow. Sensitivity analysis found one weakness: a blank income or inquiry count
lowers the PD by 13–19%.

## 12. Model limitations

Full register: [limitations register](reports/limitations_register.md). The most important:

| ID | Limitation | Severity |
|---|---|---|
| L1 | Only approved loans observed (no reject inference) | High |
| L3 | No macroeconomic drivers; default rates rose from 10.6% (2011) to 14.9% (2015) | High |
| L4 | Recalibration lags a drifting default rate | High |
| L5 | Modest discrimination from application data only | Medium |
| L11 | Rare missing values get an unreliable WoE | Medium |
| L12 | Debt-to-income outside the training range for up to 6.6% of loans | Medium |

## 13. Monitoring requirements

Monitoring framework and results: [monitoring report](reports/monitoring_report.md).

| Measure | Frequency | Amber | Red |
|---|---|---|---|
| Score PSI, largest feature CSI (vs training) | Quarterly, at origination | 0.10 | 0.25 |
| Rise in a model input's missing share | Quarterly, at origination | 2 points | 5 points |
| Inputs outside the training range | Quarterly, at origination | 0.5% | 2% |
| Mean PD (vs validated 2015 level) | Quarterly, at origination | ±10% | ±20% |
| First-year defaults ÷ mean PD (vs development) | Quarterly, from 15 months | ±10% | ±20% |
| Calibration gap (mean PD − default rate) | Quarterly, at 36 months | 1 point | 3 points |
| Gini fall (vs validated out-of-time Gini) | Quarterly, at 36 months | 10% | 20% |

**Status at the March 2019 snapshot:** the score distribution is stable (PSI below 0.10 until 2018Q3),
but revolving utilisation has shifted (CSI above 0.25 from 2017Q4), debt-to-income is outside the
training range for up to 6.6% of loans in a quarter, and first-year defaults on 2016–17 loans run 30–50% above what the
PDs imply against the development benchmark.

## 14. Recalibration conditions

**Recalibrate** (re-fit the intercept on the latest mature vintage, then confirm on the next) when any
of these holds:

- the calibration gap is AMBER or RED in the latest matured year;
- any rating grade fails the binomial test at 1% in the latest matured year;
- first-year defaults ÷ mean PD is AMBER or RED for two quarters in a row.

**Redevelop** (new bins, features and coefficients) when any of these holds:

- score PSI or any feature CSI is RED;
- Gini falls by 20% or more against the validated out-of-time Gini;
- first-year defaults ÷ mean PD is RED for four quarters in a row, meaning recalibration is not
  keeping up.

At the snapshot, all three recalibration triggers and two of the three redevelopment triggers have
fired.

## 15. Governance recommendations

1. **Redevelop before further use of the PDs.** Include LendingClub's grade or interest rate as an
   input or a segmentation, add a time-varying or macro-linked intercept, cap the WoE of rare missing
   bins, and re-bin debt-to-income on data covering the current range.
2. **Until then, use the scorecard for ranking only**, with cut-offs set on recent observed bad rates,
   and refer applications with blank income or inquiry fields.
3. **Run monitoring quarterly** against the thresholds in section 13, with the early-default measure as
   the main early warning, and report triggers to model governance.
4. **Commission an independent validation** of the redeveloped model; this card's validation was done
   by the developer.
5. **Re-approve annually**, or on any trigger in section 14.

## Governance checklist

| Item | Status | Evidence |
|---|---|---|
| Model purpose, scope and intended use documented | Done | Sections 2–4 |
| Data sources, sample design and target documented and audited | Done | Sections 5–6; [data audit](reports/data_audit.md) |
| Leakage controls in code and tests | Done | `config.LEAKAGE_COLUMNS`; `tests/test_data.py` |
| Methodology and assumptions documented | Done | Sections 7–9 |
| Out-of-time validation with confidence intervals | Done | [Validation report](reports/validation_report.md), sections 5–6 |
| Challenger and benchmark comparison | Done | Validation report, section 8 |
| Sensitivity analysis | Done | Validation report, section 9 |
| Decision rules fixed before assessment | Done | `src/pdmodel/assessment.py` |
| Limitations register maintained | Done | [Limitations register](reports/limitations_register.md) |
| Monitoring framework with thresholds and triggers | Done | Section 13; [monitoring report](reports/monitoring_report.md) |
| Reproducible end to end, with fixed seeds | Done | `scripts/run_pipeline.py`; `config.py` |
| Unit tests | Done | `tests/` |
| Independent validation | **Open** | Self-validation only |
| Fairness assessment on protected attributes | **Not possible** | No protected attributes in the data (L8) |
| Recalibration on the latest mature vintage | **Open** | Validation decision: Recalibrate |
| Redevelopment | **Open** | Monitoring triggers fired (section 14) |
