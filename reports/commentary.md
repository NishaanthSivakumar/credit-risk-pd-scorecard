# Validation commentary

Analysis of the full run. The pipeline inserts each `## <key>` section into the matching section
of `validation_report.md` (keys without the `audit` prefix) or `data_audit.md` (keys starting
with `audit`).

## conclusion

**Decision: Recalibrate. The scorecard ranks risk consistently, but its PD level does not hold
from one year to the next, and a recalibration fitted on 2014 loans is not enough for 2015.**

- *Ranking holds up out of time.* On 2015 loans, which the model never saw, Gini is 0.296 (95%
  interval 0.292–0.300), only 6% below train. Default rates fall monotonically across all ten score
  deciles, from 27.2% to 4.3%, and the score distribution is stable (PSI 0.001).
- *Discrimination is modest.* 0.296 is just below this project's 0.30 threshold. Gradient boosting
  on the same data does better by only 0.022 Gini, below the 0.03 materiality threshold, so a more
  complex model would not fix this.
- *The PD level drifts.* The model as developed predicts 12.54% on 2015 loans against 14.89%
  observed. Re-fitting the intercept on 2014 loans raises the prediction to 13.18%, which closes
  only about a quarter of the gap: 6 of 7 grades still fail the binomial test.
- *A robustness weakness.* Leaving income or recent credit inquiries blank lowers a borrower's PD
  by 13–19%, because those fields were almost never missing in training (section 9).

The PDs should not be used for pricing, provisioning or capital until a recalibration has been
shown to hold on a later, unseen vintage. See section 12.

## scope

**Purpose.** Estimate the probability that a 36-month unsecured LendingClub loan is charged off
over its life, from information available at application, and rank applicants by that risk.

**Intended uses.** Ranking and approving applicants against a cut-off; the PD input to lifetime
expected-loss and portfolio-risk estimates (`reports/portfolio_risk_report.md`).

**Out of scope.** 60-month loans; 12-month PDs for Basel IRB capital or IFRS 9 stage 1; any
population that LendingClub's approval process did not filter (see the limitations register, L1).

**Validation scope.** Data and target, conceptual soundness, discrimination, calibration,
stability, out-of-time performance, challenger comparison and sensitivity. The test sample is
2015 loans, used once, after the recalibration method was chosen on 2014 loans alone.

## data

All nine audit checks pass. 621,022 resolved 36-month loans issued 2007–2015 are used; only 147
loans (0.02%) in those vintages were still open at the data snapshot (March 2019), so resolving
loans by outcome introduces no meaningful selection bias. Vintages 2016–2018 are excluded: 28–88%
of their loans were still running, and the resolved ones are dominated by early defaults (the
resolved default rate for 2016 loans is 19.8%). Full audit: [`data_audit.md`](data_audit.md).

## performance

- *Train vs test:* Gini 0.316 vs 0.305 and KS 0.227 vs 0.221. The scorecard is not overfitting,
  as expected for a 9-feature logistic regression on monotonic bins.
- *Out of time:* Gini 0.293 on 2014 loans and 0.296 on 2015 loans, so ranking power is steady
  across the two later years.
- *Uncertainty:* resampling whole issue months and resampling single loans give similar Gini
  intervals (standard errors 0.0022 and 0.0028), so ranking is not sensitive to month-level
  shocks. Calibration is: the month-level standard error of the expected-to-observed ratio is
  0.0086, more than twice the loan-level 0.0039. Treating loans as independent would overstate
  how precisely the PD level is known.
- *LendingClub sub-grade:* 0.357 Gini on 2015 loans against the scorecard's 0.296. The sub-grade
  uses more data than these 24 columns, so it is a demanding external benchmark rather than an
  achievable target.

## calibration

**Before recalibration.** Mean PD 12.54% against 14.89% observed on 2015 loans (−2.34 points;
expected-to-observed ratio 0.84). All seven grades fail the binomial test.

**Method choice.** On 2014 loans alone, the calibration intercept is +0.059 (standard error
0.007) and the slope 0.946, so the rule re-estimates the intercept only. A slope within 0.10 of 1
does not justify a second parameter, and the test results agree: intercept-and-slope improves the
2015 Brier score by less than 0.00001 over intercept-only.

**After recalibration.** Mean PD rises to 13.18%, Brier falls from 0.1230 to 0.1227 and log loss
from 0.4055 to 0.4043. But the gap is still −1.71 points (ratio 0.885, 95% interval 0.869–0.902),
and 6 of 7 grades still fail. The shortfall is largest in the middle deciles (ratio about 0.84)
and smallest at the extremes (1.01 in the safest decile, 0.97 in the riskiest).

**Why it is not enough.** The 2014 intercept is a year out of date by the time it is applied. The
default rate rose from 13.73% on 2014 loans to 14.89% on 2015 loans, while the mean PD of the model
as developed *fell*, from 13.07% to 12.54%. Within 2015 it fell further, from 13.0% in Q1 to 12.3% in
Q4, while quarterly default rates stayed between 14.6% and 15.4%. Borrowers looked safer on the application data but defaulted more, so the
drift comes from something the inputs do not capture: underwriting, the economy or the loan mix
within each risk band. A one-year-lagged intercept will keep trailing a trend like this.

## stability

Score PSI against train is 0.001 for 2015 and 0.007 for 2014, and every feature's CSI is below
0.10. The largest are `purpose` (0.098) and `dti` (0.086, as borrowers became more indebted);
both should be watched. The input population is stable even though default
rates moved, which is consistent with the calibration finding: the drift is not in the inputs.

## challenger

The challenger (gradient boosting on the raw features, intercept-recalibrated on 2014 like the
scorecard) beats the scorecard on every resample: +0.022 Gini (95% interval +0.018 to +0.024),
+0.015 KS, and slightly better Brier score and log loss. The difference is real but below the 0.03
Gini materiality threshold, and the challenger's calibration fails too (ratio 0.905). It has
the same blind spot over time, so it is not a reason to replace the interpretable scorecard. It
also uses `mort_acc`, which the scorecard excludes because it was only recorded from 2012.

## sensitivity

- *Borrower characteristics:* the PD responds most to one more credit inquiry (+18.9%) and a
  20-point lower FICO (+12.7%), as the coefficients suggest; all six adverse changes together raise
  the mean PD by 49%. Every response has the expected direction.
- *Missing inputs — a weakness:* if `annual_inc`, `loan_to_income` or `inq_last_6mths` is blank,
  the PD *falls* by 13–19%. These fields were missing for only 4 to 24 training loans, almost none
  of which defaulted, so their "Missing" bin carries a safe-looking WoE of +0.26 estimated from
  almost no data. An applicant could lower their PD by leaving a field empty. Blank `fico` instead
  gets a neutral WoE (no missing values in training) and costs 0.046 Gini.
- *Coefficients:* parameter uncertainty moves the mean PD by ±1.7% and Gini by under 0.003. The
  estimates are precise; the calibration problem is not one of estimation noise.
- *Sub-populations:* ranking is weakest where borrowers are riskiest: Gini 0.20 for FICO below
  680, 0.21 for incomes below $40k and 0.17 for small-business loans, against 0.35 for FICO 740
  and above. PDs are most understated for large loans (ratio 0.76 for $25k and above) and high
  earners (0.78 for $100k and above).
- *Thresholds:* approving below a 15% PD accepts 66% of applicants at an 11.1% bad rate and
  avoids 51% of defaults, while declining 31% of good borrowers.
- *Alternative assumptions:* across seven re-developments the out-of-time Gini stays between
  0.282 and 0.297. Training on 2007–12 only gives the lowest (0.282), from 39% fewer loans.
  Binning choices barely matter. The model is not fragile to development choices.

## recommendations

1. **Recalibrate on the most recent mature data and re-validate forward.** Re-fit the intercept
   on 2015 loans (the latest fully matured vintage), then confirm the result on the next vintage
   as it matures. Repeat every year. The 2014 → 2015 test shows that a recalibration more than a
   year old will understate PDs while default rates are rising.
2. **Add a time-varying calibration.** Because the drift is not explained by the inputs, test an
   intercept linked to a macroeconomic or vintage indicator (this feeds the Phase 3 stress
   scenarios) rather than relying on lagged recalibration alone.
3. **Fix the treatment of missing values.** Any "Missing" bin with fewer than 5% of training
   loans should take the WoE of the riskiest bin, or the application should be referred, so a
   blank field can never lower the PD.
4. **Keep the scorecard as the model; do not switch to the challenger.** Its ranking edge is real
   but small, and it drifts in the same way.
5. **Monitor:** calibration-in-the-large on each new mature vintage (trigger: gap above 1 point or
   any grade failing at 1%), Gini (below 0.28), and PSI and CSI quarterly, with `purpose` and `dti`
   on a watch list.

## decision

The model as developed fails calibration with a sound ranking, so the rules return
**Recalibrate**. The intercept recalibration fitted on 2014 loans was then tested on 2015 loans
and still fails calibration, so the rules return **Recalibrate** again: the recalibration has to be
refreshed on more recent data and shown to hold out of time before the PDs are approved. Every
other dimension except discrimination is GREEN; discrimination is RED against the 0.30 threshold,
but no challenger on the same data is materially better, so it does not trigger redevelopment.

## audit summary

Every check passes. The target is defined consistently, no post-origination field can reach the
model, every modelled vintage was observed for at least 39 months, and the four samples are
separated in time.

## audit target

99.6% of the 621,169 loans in the modelled vintages are either "Fully Paid" or "Charged Off". The
"Does not meet the credit policy" statuses (2,438 loans, from LendingClub's early years) follow the
same rule as the standard statuses. The 147 loans still open at the snapshot are excluded; at
0.02% of loans they cannot move the default rate.

## audit prepayment

57% of fully paid loans were repaid at least three months early, and the share rose from 34% for
2007 loans to 58–59% for 2014–15 loans. These loans count as non-defaults: the borrower repaid
the full principal. The target therefore measures default *before* the loan ends, whenever that
is. Because early repayment ends a loan's exposure, a vintage with more prepayment has less time
in which to default. The rise in prepayment over time would, if anything, lower recent default
rates, so it cannot explain the rising default rate seen in the calibration analysis. Defaulted
loans made their last payment after a median of 16 months.

## audit maturity

A 36-month loan issued in December 2015 was due to finish in December 2018, and the data snapshot
is March 2019, so every modelled loan had at least 39 months to resolve. The 2016–2018 vintages
show why the cut-off matters: their resolved default rates (19.8% for 2016) are inflated because
the loans that had already finished are disproportionately early defaults.

## audit leakage

No candidate feature is a post-origination field. The outcome fields (`funded_amnt`,
`total_rec_prncp`, `recoveries`, `collection_recovery_fee`, `total_pymnt`, `last_pymnt_d`) are
loaded only to measure loss given default, exposure at default and prepayment. Unit tests fail if
any of them, or any other field in `LEAKAGE_COLUMNS`, is added to the feature lists. No feature
has an IV above 0.5, which would also have pointed to leakage.

## audit temporal

The train and test samples are a random split of 2007–13 loans: an in-time sample for fitting and
for checking overfitting. Chronological separation starts at the next boundary: 2014 loans are
used only to fit the recalibration, and 2015 loans only for the final test, the bootstrap,
sensitivity analysis and the portfolio. Keeping the original train/test split also keeps the
developed model identical to the one documented before this audit.

## audit quality

Only 9 values in 621,022 loans fall outside the plausible ranges, all of them extreme incomes,
debt-to-income ratios or utilisation rates, which the WoE end bins absorb. Missing values are rare
in the final features (at most 0.1%, for `revol_util`). `mort_acc` is 21% missing in training
because it was not recorded before 2012, and is excluded for that reason. `emp_length_yrs` is 4–7%
missing but has too little IV to be selected.

## audit limitations

- Only approved loans are observed (no reject inference).
- The target is a 36-month lifetime default, not a 12-month default.
- Rare missing values get a WoE estimated from very few loans (see the sensitivity analysis in the
  validation report).
- Recoveries are recorded less completely for recent charge-offs: 25% of 2014 charge-offs show no
  recovery, against under 8% for every vintage from 2007 to 2013. This affects loss measurement, not the PD target.
