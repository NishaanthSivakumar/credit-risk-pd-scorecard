# Implementation Plan and Tracker

Tracks the **Credit Risk PD Scorecard — Improvement Roadmap** phase by phase. Each task has a
status, the files that deliver it and how it is checked.

**Status key:** ✅ done · 🔄 in progress · ⬜ not started · ⏸ optional

**Scope:** credit risk modelling, model validation and credit portfolio risk.

**Layout:** each phase gets a notebook that runs it step by step and displays every result; the
calculations live in `src/pdmodel/`, where they are tested and shared with the pipeline.

---

## Progress summary

| Phase | Title | Priority | Status |
|---|---|---|---|
| 1 | Strengthen model validation | Critical | ✅ |
| 2 | Credit portfolio risk engine | High | ✅ |
| 3 | Macroeconomic stress testing | High | ✅ |
| 4 | Credit portfolio optimisation | High | ✅ |
| 5 | Model monitoring and governance | Medium | ✅ |
| 6 | Engineering and presentation | Medium | ✅ |
| 7 | Documentation and recruiter presentation | High | ✅ |

---

## Schedule

| Step | Work | Deliverable |
|---|---|---|
| 1 | Phase 1: audit, chronological split, recalibration, block bootstrap, sensitivity tests, validation decision | Updated validation framework and report |
| 2 | Phase 2: measured LGD/EAD, expected loss, Monte Carlo engine, VaR/ES | Credit portfolio risk engine, first portfolio risk report |
| 3 | Phase 3: stress scenarios with Vasicek conditional PD | Stress-testing module and scenario report |
| 4 | Phase 4: realised-return backtest, segment-level optimisation | Portfolio optimisation module and report |
| 5 | Phase 5: monitoring on 2016–18 loans, model card | Monitoring module, `MODEL_CARD.md` |
| 6 | Phases 6 and 7: CI, pinned dependencies, README, final reports, optional dashboard | Recruiter-ready repository |

---

## Phase 1: Strengthen model validation

**Priority:** Critical · **Status:** ✅

### 1.1 Target definition and data leakage

| Task | Status | How |
|---|---|---|
| Fully paid loans classified as non-default | ✅ | Automated check in `src/pdmodel/audit.py`; unit test |
| Charged-off and defaulted loans classified as default | ✅ | Same |
| Treatment of early repayments | ✅ | Measure months to payoff for fully paid loans; document why early payers count as good |
| All predictors available at origination | ✅ | Leakage blocklist in `config.py`, checked automatically and in tests |
| Remove post-origination variables | ✅ | Loss fields (recoveries, principal repaid) are loaded only for LGD/EAD and audit, never as features; enforced by test |
| Consistent observation window across vintages | ✅ | Resolution rate and months-to-outcome by vintage |
| Selection bias from resolved-loan filtering | ✅ | Share of loans excluded as unresolved, by vintage |
| Chronological train / calibration / test separation | ✅ | `data.split_samples`: train and test from 2007–13, calibration 2014, out-of-time test 2015 |

Deliverables:

- `notebooks/06_sensitivity_and_decision.ipynb` (section 1): the audit run step by step
- `reports/data_audit.md`: leakage audit, target definition and temporal validation methodology
- `src/pdmodel/audit.py`: data-quality and leakage checks run by the pipeline
- Updated preprocessing in `src/pdmodel/data.py`
- `tests/test_data.py`: data-integrity tests

### 1.2 Probability of default recalibration

Starting point: predicted 12.73% vs observed 14.46% on 2014–15 loans, a 1.73-point gap.

| Task | Status | How |
|---|---|---|
| Calibration intercept and slope | ✅ | Logistic regression of the outcome on logit(PD), in `src/pdmodel/calibration.py` |
| Intercept-only recalibration | ✅ | Fitted on 2014 loans |
| Intercept-and-slope recalibration | ✅ | Fitted on 2014 loans |
| Compare original and recalibrated predictions | ✅ | All metrics on 2015 loans |
| Calibration across risk deciles | ✅ | Decile table per method |
| Reliability diagrams | ✅ | `reports/figures/reliability_oot.png` |
| Brier score and log loss | ✅ | Per method |
| Expected vs observed default rates | ✅ | Expected-to-observed ratio and binomial test by grade |
| Parameters from a dedicated calibration sample | ✅ | 2014 only |
| Final performance on an untouched test sample | ✅ | 2015 only, used once |

Metrics: Gini, ROC-AUC, KS, Brier score, log loss, calibration intercept, calibration slope,
expected-to-observed ratio.

Deliverables: recalibration module, calibration comparison plots, updated validation report and a
recommendation on whether recalibration is appropriate. Notebook:
`notebooks/05_recalibration_and_uncertainty.ipynb`.

### 1.3 Statistical confidence intervals

| Task | Status | How |
|---|---|---|
| Bootstrap engine | ✅ | `src/pdmodel/bootstrap.py`, weighted resampling so every metric uses the same draws |
| 95% intervals for AUC, Gini, KS, Brier score, calibration slope | ✅ | Percentile intervals |
| Paired scorecard vs challenger comparison | ✅ | Same draws for both models; interval for the difference |
| Temporal dependence | ✅ | Block bootstrap by issue month (main result) plus loan-level bootstrap for comparison |

Deliverables: bootstrap module, confidence-interval chart, statistical model comparison and an
updated validation report. Notebook: `notebooks/05_recalibration_and_uncertainty.ipynb`.

### 1.4 Structured model validation

| Task | Status | How |
|---|---|---|
| Assess the 10 dimensions (conceptual soundness → limitations) | ✅ | `src/pdmodel/assessment.py`, each with a RAG status and evidence |
| Sensitivity: borrower characteristics | ✅ | Shift one input at a time (e.g. FICO −20) and measure the PD change, in `src/pdmodel/sensitivity.py` |
| Sensitivity: feature distributions | ✅ | Performance on shifted sub-populations |
| Sensitivity: missing inputs | ✅ | Blank each input in turn |
| Sensitivity: coefficients | ✅ | ±1 standard error and draws from the coefficient covariance |
| Sensitivity: probability thresholds | ✅ | Approval rate and bad rate at each PD cut-off |
| Sensitivity: alternative modelling assumptions | ✅ | Refit with different binning and sample choices |
| Decision criteria for approve / conditionally approve / recalibrate / redevelop / reject | ✅ | Written down before being applied |

Deliverables: updated validation report, `reports/limitations_register.md`, validation findings and
the final model decision. Notebook: `notebooks/06_sensitivity_and_decision.ipynb`. This is a
self-validation that follows the structure of an independent
review; the report says so.

### Phase 1 tests

`tests/test_data.py`, `tests/test_calibration.py`, `tests/test_bootstrap.py`,
`tests/test_sensitivity.py`, `tests/test_assessment.py`.

### Phase 1 outcome

- All nine audit checks pass. Only 147 loans (0.02%) in the modelled vintages were unresolved.
- Out-of-time (2015) Gini 0.296 (95% interval 0.292–0.300), 6% below train; the challenger is
  +0.022 better, below the 0.03 materiality threshold.
- The intercept recalibration fitted on 2014 raises the 2015 mean PD from 12.54% to 13.18% against
  14.89% observed: it closes only about a quarter of the gap, and 6 of 7 grades still fail.
- Month-block bootstrap intervals for calibration are about twice as wide as loan-level ones; for
  Gini they are similar.
- New weakness found: a blank income or inquiry count lowers the PD by 13–19% (limitation L11).
- Decision: **Recalibrate**, for both the model as developed and the 2014 recalibration.

---

## Phase 2: Credit portfolio risk engine

**Priority:** High · **Status:** ✅

The portfolio is the 2015 vintage: loans the model never saw during development or recalibration,
whose actual outcomes are known. That allows the engine's predictions to be compared with the
losses that really happened.

### 2.1 Portfolio expected loss

| Task | Status | How |
|---|---|---|
| Per-loan PD | ✅ | Recalibrated scorecard PD (lifetime, 36 months) |
| Per-loan LGD | ✅ | Measured on 2007–14 charge-offs by rating grade: 1 − net recoveries / EAD |
| Per-loan EAD | ✅ | Measured on 2007–14 charge-offs: share of the funded amount still owed at charge-off |
| Expected loss per loan and in total | ✅ | EL = PD × LGD × EAD, in `src/pdmodel/portfolio.py` |
| EL by borrower segment | ✅ | Rating grade, loan purpose, loan size |
| Exposure concentration | ✅ | Herfindahl index, largest exposures, segment shares |
| Sensitivity to PD and LGD | ✅ | EL across a grid of PD and LGD shifts |
| Back-test against realised 2015 losses | ✅ | Predicted EL vs actual loss |

Deliverables: expected loss module, EL by segment, concentration analysis and an EL summary in the
portfolio risk report. Notebook: `notebooks/07_portfolio_risk.ipynb`.

### 2.2 Monte Carlo credit portfolio simulation

| Task | Status | How |
|---|---|---|
| Correlated default simulation | ✅ | One-factor Gaussian model in `src/pdmodel/simulation.py` |
| Thousands of scenarios | ✅ | 10,000 by default |
| Portfolio loss distribution | ✅ | Histogram with VaR and ES marked |
| Default correlation | ✅ | Basel retail correlation per loan as the base case; data estimate from vintage default rates reported; sensitivity grid 0–0.20 |
| Portfolio concentration | ✅ | Loss distribution for portfolios of different sizes |
| Simulated vs analytical EL | ✅ | With a Monte Carlo standard error |
| Convergence | ✅ | Estimates against the number of scenarios |
| Gaussian dependence limitations | ✅ | Documented in the report |

Metrics: expected loss, loss standard deviation, 95% and 99% VaR, 97.5% and 99% Expected Shortfall
and concentration measures, all over the 36-month lifetime.

Deliverables: Monte Carlo engine, loss distribution plots, VaR and ES, correlation sensitivity and
a convergence check, in `reports/portfolio_risk_report.md`. Notebook: `notebooks/07_portfolio_risk.ipynb`.

### Phase 2 tests

`tests/test_portfolio.py`, `tests/test_simulation.py`.

### Phase 2 outcome

- LGD 89.2% and EAD 57.7% of the funded amount, measured on 44,472 charge-offs from 2007–14.
- 2015 book: expected lifetime loss 6.35% of $3.62bn funded, against 7.56% realised. The gap is
  the PD shortfall (13% more defaults than predicted); loss severity was within 2%.
- With the Basel retail correlation (average ρ 0.065): 99% VaR 13.45%, 99% ES 15.0%; the realised
  loss sits at the 73rd percentile. The simulation matches the analytical results within 0.15%.
- The correlation estimated from the data (0.004–0.013) reflects a 2009–15 recovery with no
  recession, so it is reported but not used.

---

## Phase 3: Macroeconomic stress testing

**Priority:** High · **Status:** ✅

| Task | Status | How |
|---|---|---|
| Define base, mild recession, severe recession and interest rate shock scenarios | ✅ | `STRESS_SCENARIOS` in `config.py`: each sets how rare the economy is (1-in-10, 1-in-50, 1-in-5), plus LGD, EAD, correlation and rate add-ons |
| Specify the effect on PD | ✅ | Vasicek conditional PD at the scenario's economy factor, in `src/pdmodel/stress.py`; the rate shock also raises debt-to-income through the scorecard |
| Specify the effect on LGD | ✅ | Downturn add-ons to LGD and EAD, anchored on 2007–11 charge-offs |
| Recalculate expected loss | ✅ | Per loan and in total, against the base case |
| Simulate stressed loss distributions | ✅ | Phase 2 engine with each scenario's parameters, same random draws for every scenario |
| Stressed VaR and ES | ✅ | 95% and 99% VaR, 97.5% and 99% ES per scenario |
| Compare with the base case | ✅ | Multiple of base expected loss, and each scenario's percentile in the base distribution |
| Sensitivity to scenario assumptions | ✅ | Tornado chart over severity, PD level, LGD, EAD and correlation; expected loss across economies |
| Document limitations | ✅ | Scenarios are assumptions, not estimated macroeconomic relationships |
| Historical anchor (added) | ✅ | Economy factor implied by each vintage's default rate |
| Reverse stress test (added) | ✅ | How rare an economy must be for losses to reach 8–20% |

Deliverables: `notebooks/08_stress_testing.ipynb`, `src/pdmodel/stress.py`, section 7 of
`reports/portfolio_risk_report.md` (scenario comparison, stressed loss distributions, portfolio
vulnerability). Tests: `tests/test_stress.py`.

### Phase 3 outcome

- The 2007 and 2008 vintages, which ran through the 2008–09 recession, imply 1-in-10.6 and 1-in-6.5
  economies: the mild scenario matches them, and the severe one goes beyond anything in the data.
- Lifetime expected loss rises from 6.35% of funded amount to 10.32% in a mild recession and 15.29%
  in a severe one ($554m against $230m), beyond the base 99% VaR of 13.45%.
- A 300bp rate rise barely moves borrowers' payments (+0.04 points of expected loss); the rate
  scenario's impact comes almost entirely from the slower economy.
- The safest grade's losses rise 3.8 times in a severe recession, against 1.9 times for the riskiest.
- Severity and the PD level drive the result: removing the calibration gap adds as much loss as
  moving from a 1-in-50 to a 1-in-100 economy.

---

## Phase 4: Credit portfolio optimisation

**Priority:** High · **Status:** ✅

### 4.1 Allocation framework

| Task | Status | How |
|---|---|---|
| Measure realised return per loan | ✅ | Net cash received (borrower payments less the 1% servicing fee, plus recoveries less collection fees) per dollar funded; approximate annual return from the amortisation schedule |
| Build inputs from origination-time information only | ✅ | Model PD, interest rate (as the return input), measured LGD and EAD, and interest factors measured on 2007–14 loans |
| Document assumptions | ✅ | Servicing fee, prepayment (in the interest factors), holding to maturity (no liquidity cost), no purchase costs |

### 4.2 Strategies, compared on 2015 loans

| Task | Status |
|---|---|
| Equal-weighted portfolio | ✅ |
| Maximum expected net return | ✅ |
| Minimum expected loss | ✅ |
| Minimum portfolio CVaR | ✅ |
| Maximum expected return subject to a CVaR limit | ✅ |
| Constraints: weights sum to 100%, segment limits, minimum/maximum allocations, concentration limits, optional EL limit | ✅ |
| Compare each strategy's expected and realised outcomes | ✅ |
| Whole-book benchmark and hindsight optimum (added) | ✅ |
| Sensitivity to the return inputs: price-informed PD and segment history (added) | ✅ |

Deliverables: `notebooks/09_portfolio_optimisation.ipynb`, `src/pdmodel/optimisation.py`, section 8 of
`reports/portfolio_risk_report.md` (strategy comparison, risk-return charts). Tests: `tests/test_optimisation.py`.

### Phase 4 outcome

- Segments: our rating grade × LendingClub grade (32 segments, 99.6% of the 2015 book).
- Minimising expected loss returned 6.31% over the loans' life, a point above the whole book (5.30%)
  and within 0.03 points of the hindsight optimum (6.34%).
- Maximising expected return forecast 14.91% and earned 3.74%. Under the scorecard model the
  CVaR-minimising and risk-limited strategies chose the same portfolio, because the model saw no
  trade-off between return and tail risk.
- The cause is concentrated input error in high-rate loans: a PD too flat across LendingClub's grades
  (34% defaulted in E–G against 21% predicted), more prepayment, and earlier defaults.
- Neither a price-informed PD nor segment history rescues return seeking (best 4.80% and 5.08%).

---

## Phase 5: Model monitoring and governance

**Priority:** Medium · **Status:** ✅

### 5.1 Monitoring

| Task | Status | How |
|---|---|---|
| PSI and CSI by quarter | ✅ | 2016–18 loans as the production stream, against the training sample, in `src/pdmodel/monitoring.py` |
| AUC, Gini, KS, calibration intercept and slope | ✅ | Matured vintages (2014–15), by quarter |
| Expected vs observed default rates | ✅ | Matured vintages; first-year defaults vs PD as the early warning for 2016–17 |
| Performance by borrower segment | ✅ | First-year defaults by segment on 2016–17 loans |
| Input distribution change detection and alerts | ✅ | Missing values and out-of-range inputs; RAG thresholds (`MON_*` in `config.py`); alert grid and log |
| Recalibration and redevelopment triggers | ✅ | Linked to the Phase 1 decision rules |

### 5.2 Model card

| Task | Status |
|---|---|
| `MODEL_CARD.md` with the 15 required sections | ✅ |
| Model limitations documentation | ✅ (register updated: L12, inputs outside the training range) |
| Model governance checklist | ✅ (in `MODEL_CARD.md`) |

Deliverables: `notebooks/10_monitoring.ipynb`, `src/pdmodel/monitoring.py`, `reports/monitoring_report.md`,
`MODEL_CARD.md`. Tests: `tests/test_monitoring.py`.

### Phase 5 outcome

- The score PSI stays below 0.10 until 2018Q3, but first-year defaults run RED against the PDs from
  2015Q4 onward (+30% to +50% across 2016–17), while the mean PD falls 15% by 2018.
- Debt-to-income exceeds the training maximum for up to 6.6% of loans in a quarter; revolving
  utilisation turns RED from 2017Q4.
- Recalibration is triggered by late 2016 and redevelopment by late 2017, years before the loans mature.

---

## Phase 6: Engineering and presentation

**Priority:** Medium · **Status:** ✅

### 6.1 Testing and CI

| Task | Status | How |
|---|---|---|
| Module tests (written during Phases 1–5) | ✅ | `tests/`, one file per module, synthetic data only |
| GitHub Actions workflow running the tests on every push and pull request | ✅ | `.github/workflows/ci.yml`: lint, unit tests, and the full pipeline on synthetic data |
| Pinned dependencies | ✅ | `requirements.txt` and `dashboard/requirements.txt`, exact versions; tested in a clean environment |
| Random seeds for every stochastic step | ✅ | `config.py`; coefficient draws use a Cholesky factor so they match across platforms |
| Reproducibility instructions | ✅ | README section 14, including the dataset's size and SHA-256 |

### 6.2 Dashboard

| Task | Status | How |
|---|---|---|
| Streamlit app reading precomputed results: model performance, stability, portfolio risk, stress testing, optimisation | ✅ | `dashboard/app.py`, six tabs, reads `reports/tables/` only; tested with Streamlit's AppTest |
| Screenshots and usage notes | ✅ | `reports/figures/dashboard_*.png`; README section 13 |
| Hosting on Streamlit Community Cloud | ✅ | Live at [credit-risk-pd-scorecard.streamlit.app](https://credit-risk-pd-scorecard.streamlit.app) |

---

## Phase 7: Documentation and recruiter presentation

**Priority:** High · **Status:** ✅

| Task | Status | How |
|---|---|---|
| README with the 15 recommended sections | ✅ | Rewritten: overview and lifecycle diagram, results, methods, installation, reproduction, future work |
| Final model validation report (13 sections) | ✅ | Section 9 now covers sensitivity and stress testing; section 12 adds monitoring since validation |
| Portfolio risk report (10 sections), built up from Phase 2 onward | ✅ | All 10 sections; internal roadmap references removed from every report |
