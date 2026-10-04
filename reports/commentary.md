# Validation commentary

Analysis of the full run. The pipeline inserts each `## <key>` section into the matching
section of `validation_report.md`.

## conclusion

**Fit with conditions: usable for ranking applicants; not usable for absolute PDs until recalibrated.**

- *Ranking holds up out of time.* Gini falls only 7% from train (0.316) to OOT (0.294), the score
  distribution is stable (PSI 0.001) and default rates fall monotonically across all ten OOT score
  deciles, from 26.6% in the worst to 4.3% in the best.
- *Discrimination is modest.* OOT Gini of 0.294 sits just below this project's 0.30 amber threshold,
  which is expected for a model limited to application data with no detailed bureau history.
- *PDs are too low on recent loans.* Mean predicted PD on 2014–15 loans is 12.7% against an observed
  14.5%, and all seven rating grades fail the binomial test.

Conditions: (1) re-fit the intercept on recent vintages before the PDs are used for pricing,
provisioning or capital; (2) monitor Gini and PSI quarterly against the thresholds in `config.py`.

## performance

- *Train vs test:* Gini 0.316 vs 0.305 and KS 0.227 vs 0.221. The gap is small, so the scorecard is
  not overfitting, as expected for a 9-feature logistic regression on monotonic bins.
- *Test vs OOT:* Gini 0.305 vs 0.294. The small drop reflects the shift in the borrower mix across
  vintages (see section 6), not a breakdown of the risk drivers: every feature's CSI is below 0.10.
- *Scorecard vs challenger:* gradient boosting reaches 0.441 on train but only 0.325 on test and
  0.316 OOT, so most of its in-sample edge is overfitting. Out of time it beats the scorecard by just
  0.02 Gini. That is the price of interpretability here, and it is small.
- *Scorecard vs LendingClub sub-grade:* the scorecard ranks better than LendingClub's sub-grade on
  2007–13 loans (0.305 vs 0.274 on test) but worse on 2014–15 loans (0.294 vs 0.343). LendingClub's
  grading uses more data than these 24 columns and was itself refined over time, so it is a
  demanding benchmark on the later vintages.

## calibration

The OOT default rate (14.5%) is above the development rate (12.6%), and the model, calibrated on
development loans, under-predicts in every grade by 0.3 to 2.4 percentage points.

The drift comes from changes in who was lending and borrowing, not from the model breaking down.
Default rates bottomed at 10.6% for 2011 loans, when post-crisis underwriting was tight, then rose
every year to 14.9% for 2015 loans. Over the same period LendingClub grew quickly (from 14k loans in
this sample in 2011 to 283k in 2015) and accepted more indebted borrowers: average debt-to-income
rose from 13.5% to 18.5%, while average FICO fell from 718 to 696. Because the scorecard's binned
features capture most of that shift, the ranking survives, but the base default rate moved.

**Recommendation:** keep the WoE bins and coefficients and re-fit only the intercept on the most
recent mature vintages, so the mean PD matches the recent default rate. Then repeat the binomial
test by grade on a later hold-out. A full re-development is not justified, because rank ordering
and stability are both satisfactory.

## limitations

- **Reject inference:** only approved loans are observed, so the model learns from a population
  LendingClub had already filtered. Its PDs would be too low if applied to all applicants.
- **Lifetime, not 12-month, PD:** the target is default at any point over a 36-month life. Basel IRB
  and IFRS 9 stage 1 need a 12-month PD, which would require payment-level data to build.
- **Through-the-cycle:** there are no macroeconomic variables, so PDs do not move with the economy,
  and the development window (2007–13) is dominated by the post-crisis recovery.
- **Feature availability:** `mort_acc` was excluded because LendingClub only recorded it from 2012.
  The gradient-boosting challenger still sees it, with missing values for older loans, so its
  train and test Gini are slightly flattered.
- **Monitoring plan:** track score PSI and per-feature CSI quarterly (amber 0.10, red 0.25), Gini
  on each new mature vintage (amber below 0.30), and calibration-in-the-large each year, with
  intercept recalibration whenever the gap exceeds 1 percentage point.
