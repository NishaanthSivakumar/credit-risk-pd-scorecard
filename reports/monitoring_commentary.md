# Monitoring commentary

Analysis of the monitoring run. The pipeline inserts each `## <key>` section into the matching section
of `monitoring_report.md`.

## summary

**Monitoring calls for recalibration from late 2016 and for redevelopment from late 2017.** Both calls
come well before the affected loans mature.

- *The score distribution is stable* (PSI below 0.10 until 2018Q3), so a check on the score alone would
  have missed the problem.
- *First-year defaults outran the PDs:* against the development benchmark, borrowers stopped paying
  within a year 23% more often than the PDs imply on 2015Q4 loans, and 30–50% more often on every
  quarter of 2016–17 loans.
- *Inputs left the training range:* debt-to-income exceeded the training maximum for up to 6.6% of
  loans in a quarter, and revolving utilisation shifted enough to turn RED from 2017Q4.
- *Borrowers looked safer while defaulting more:* the mean PD fell 15% by 2018, the same pattern the
  validation found on 2015 loans.

All three recalibration triggers and two of the three redevelopment triggers fired. Ranking power is
the one thing that held: Gini on matured quarters stayed within 6% of its validated level.

## stability

The score PSI stays below 0.025 until 2017Q3, then rises to 0.06–0.12 in 2017Q4–2018, reaching AMBER
in the last two quarters. The feature view is more telling. `purpose` drifts in and out of AMBER
(CSI 0.10–0.14) throughout. `revol_util` jumps to RED from 2017Q4 (0.25–0.28): the share of borrowers
using less than 21% of their revolving credit doubled, from 10% in training to 21%. Lower utilisation
earns more points, so this shift alone pushes PDs down.

## quality

Missing values are not a concern: no model input's missing share rose by more than 0.21 points.
Inputs outside the training range are. Debt-to-income was capped at 35 in the training data;
LendingClub accepted higher ratios from late 2014, and 3–5% of loans in 2014Q4–2016Q2 and 6–7% in 2018
have a ratio the model never saw; the share fell back to 0.1–1.4% in 2016Q3–2017Q2 before rising again.
The scorecard puts all of these loans in its top debt-to-income bin, which assumes the risk stops rising
at the training edge.

## predictions

The mean PD of the deployed model fell steadily, from 13.7% on 2014 loans to 11.2–11.4% in 2018 (−13% to
−15% against the validated 2015 level; AMBER from 2017Q4). The grade mix shows why: the share of loans
in grade 1 rose from 15% at the end of 2015 to 20–22% in 2018, and the share in grade 7 fell from 13% to
7%. On the application data, LendingClub's borrowers were getting safer.

## early

They were not. On the development loans, 34.6% of the mean PD showed up as a default in the first
year. That ratio was close to the benchmark for most of 2014 (2014Q1 was AMBER, but below it), AMBER for
2015Q2–Q3, and RED from 2015Q4 onward:
+23% for 2015Q4, +35% to +50% across 2016, and +30% to +40% across 2017. The validation found the same
direction on matured 2015 loans: mean PD down, default rate up.

The ratio mixes level and timing: it rises if more loans default, or if the same number default sooner.
The matured 2015 quarters, where both are visible, show the level effect is real: version 1.1
under-predicts every 2015 quarter by 1.2–2.0 points. The 2016–17 ratios run higher still, which suggests
the gap kept widening.

By segment, the shift is largest for loans of $25k and above (+91%) and high earners (+67% for incomes
of $100k and above), the same groups whose PDs the validation found most understated. Ranking on
first-year defaults holds up: Gini 0.25–0.30 across 2016–17.

## performance

On matured vintages, 2014 is the recalibration sample, so its calibration is good by construction
(gaps within 0.9 points). From 2014Q4 the model under-predicts, and from 2015Q1 every quarter is
AMBER, with gaps of 1.2–2.0 points and 5–7 of 7 grades failing the binomial test. Gini stays between
0.280 and 0.304, within 6% of the validated 0.296.

## actions

Monitoring would have called for action as follows, at the earliest date each signal could be measured
(early defaults need 15 months of observation):

| When | Signal | Trigger | Action |
|---|---|---|---|
| Late 2016 | First-year defaults AMBER for 2015Q2–Q3 loans | Recalibrate | Re-fit the intercept on the latest data |
| Late 2017 | First-year defaults RED for four quarters (2015Q4–2016Q3 loans) | Redevelop | Recalibration is not keeping up |
| Q4 2017 | Revolving utilisation CSI RED at origination | Redevelop | The input population has moved |
| Late 2018 | 2015 vintage matures: calibration AMBER, grades failing | Recalibrate | Confirms the early warning |

The recommended response is the one in the model card: redevelop. Bring in LendingClub's grade or
interest rate, add a time-varying intercept, re-bin debt-to-income on current data, and
cap the WoE of rare missing bins. Until then, use the scorecard for ranking only.

## framework

Every measure is computed by issue quarter and compared with a fixed reference: the training sample for
inputs, the validated out-of-time results for performance, and the development sample for first-year
defaults. Stability and data quality are available as soon as loans are issued; first-year defaults
after 15 months; full performance after 36 months. The early-default measure is the one that matters
most for a lifetime PD model, because it is the first evidence about outcomes.

Thresholds sit in `config.py` (`MON_*`); a quarter's statuses feed the triggers, which map onto the
validation decision rules: a level problem with a sound ranking means recalibrate, and a stability or
ranking problem, or a level problem that recalibration cannot fix, means redevelop.
