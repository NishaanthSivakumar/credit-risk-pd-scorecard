# Implementation Plan and Tracker

Tracks the **Credit Risk PD Scorecard — Improvement Roadmap** phase by phase. Each task has a
status, the files that deliver it and how it is checked.

**Status key:** ✅ done · 🔄 in progress · ⬜ not started · ⏸ optional

**Scope:** credit risk modelling, model validation and credit portfolio risk.

---

## Progress summary

| Phase | Title | Priority | Status |
|---|---|---|---|
| 1 | Strengthen model validation | Critical | ✅ |
| 2 | Credit portfolio risk engine | High | ✅ |
| 3 | Macroeconomic stress testing | High | ⬜ |
| 4 | Credit portfolio optimisation | High | ⬜ |
| 5 | Model monitoring and governance | Medium | ⬜ |
| 6 | Engineering and presentation | Medium | ⬜ |
| 7 | Documentation and recruiter presentation | High | ⬜ |

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
recommendation on whether recalibration is appropriate.

### 1.3 Statistical confidence intervals

| Task | Status | How |
|---|---|---|
| Bootstrap engine | ✅ | `src/pdmodel/bootstrap.py`, weighted resampling so every metric uses the same draws |
| 95% intervals for AUC, Gini, KS, Brier score, calibration slope | ✅ | Percentile intervals |
| Paired scorecard vs challenger comparison | ✅ | Same draws for both models; interval for the difference |
| Temporal dependence | ✅ | Block bootstrap by issue month (main result) plus loan-level bootstrap for comparison |

Deliverables: bootstrap module, confidence-interval chart, statistical model comparison and an
updated validation report.

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
the final model decision. This is a self-validation that follows the structure of an independent
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
portfolio risk report.

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
a convergence check, in `reports/portfolio_risk_report.md`.

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

**Priority:** High · **Status:** ⬜

| Task | Status | How |
|---|---|---|
| Define base, mild recession, severe recession and interest rate shock scenarios | ⬜ | Each scenario is a systematic-factor shock z, plus LGD and correlation changes |
| Specify the effect on PD | ⬜ | Vasicek conditional PD |
| Specify the effect on LGD | ⬜ | Downturn LGD add-on, documented assumption |
| Recalculate expected loss | ⬜ | |
| Simulate stressed loss distributions | ⬜ | Reuses the Phase 2 engine |
| Stressed VaR and ES | ⬜ | |
| Compare with the base case | ⬜ | |
| Sensitivity to scenario assumptions | ⬜ | |
| Document limitations | ⬜ | Scenarios are assumptions, not estimated macroeconomic relationships |

Deliverables: stress-testing module, scenario comparison report, stressed loss distribution plots,
portfolio vulnerability analysis. Tests: `tests/test_stress.py`.

---

## Phase 4: Credit portfolio optimisation

**Priority:** High · **Status:** ⬜

### 4.1 Allocation framework

| Task | Status | How |
|---|---|---|
| Measure realised return per loan | ⬜ | From total payments received vs amount funded, over the loan's life |
| Build inputs from origination-time information only | ⬜ | Model PD, interest rate (as the return input, not a PD predictor), LGD, EAD |
| Document assumptions | ⬜ | Servicing fee, prepayment, liquidity and transaction costs |

### 4.2 Strategies, compared on 2015 loans

| Task | Status |
|---|---|
| Equal-weighted portfolio | ⬜ |
| Maximum expected net return | ⬜ |
| Minimum expected loss | ⬜ |
| Minimum portfolio CVaR | ⬜ |
| Maximum expected return subject to a CVaR limit | ⬜ |
| Constraints: weights sum to 100%, segment limits, minimum/maximum allocations, concentration limits, optional EL limit | ⬜ |
| Compare each strategy's expected and realised outcomes | ⬜ |

Deliverables: optimisation module, strategy comparison, risk-return charts, allocation section of
the portfolio risk report. Tests: `tests/test_optimisation.py`.

---

## Phase 5: Model monitoring and governance

**Priority:** Medium · **Status:** ⬜

### 5.1 Monitoring

| Task | Status | How |
|---|---|---|
| PSI and CSI by quarter | ⬜ | 2016–18 loans as the production stream |
| AUC, Gini, KS, calibration intercept and slope | ⬜ | Matured vintages only |
| Expected vs observed default rates | ⬜ | Matured vintages only |
| Performance by borrower segment | ⬜ | |
| Input distribution change detection and alerts | ⬜ | Thresholds in `config.py` |
| Recalibration and redevelopment triggers | ⬜ | Linked to the Phase 1 decision criteria |

### 5.2 Model card

| Task | Status |
|---|---|
| `MODEL_CARD.md` with the 15 required sections | ⬜ |
| Model limitations documentation | ⬜ |
| Model governance checklist | ⬜ |

Tests: `tests/test_monitoring.py`.

---

## Phase 6: Engineering and presentation

**Priority:** Medium · **Status:** ⬜

### 6.1 Testing and CI

| Task | Status |
|---|---|
| Module tests (written during Phases 1–5) | 🔄 Phases 1–2 done |
| GitHub Actions workflow running the tests on every push and pull request | ⬜ |
| Pinned dependencies | ⬜ |
| Random seeds for every stochastic step | 🔄 Phases 1–2 done |
| Reproducibility instructions | ⬜ |

### 6.2 Dashboard

| Task | Status |
|---|---|
| ⏸ Streamlit app reading precomputed results: model performance, stability, portfolio risk, stress testing, optimisation | ⏸ |
| ⏸ Hosting on Streamlit Community Cloud, screenshots and usage notes | ⏸ |

---

## Phase 7: Documentation and recruiter presentation

**Priority:** High · **Status:** ⬜

| Task | Status |
|---|---|
| README with the 15 recommended sections | ⬜ |
| Final model validation report (13 sections) | ⬜ |
| Portfolio risk report (10 sections), built up from Phase 2 onward | 🔄 Sections 1–6, 9–10 drafted |
