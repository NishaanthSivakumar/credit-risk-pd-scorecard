# Portfolio risk commentary

Analysis of the full run. The pipeline inserts each `## <key>` section into the matching section
of `portfolio_risk_report.md`.

## composition

The 2015 book holds 283,026 loans and $3.62bn of funded principal. The average PD is 13.18%, but
weighted by amount it is 12.38%, because larger loans went to lower-risk borrowers. Exposure is
spread evenly across the seven rating grades (11–17% each), so expected loss is concentrated in
the riskiest grades: grade 7 holds 11% of the exposure and 24% of the expected loss.

## lgd ead

- *Exposure at default:* a defaulted loan still owes 57.7% of its funded amount on average. The
  share rises from 53.5% in grade 1 to 61.3% in grade 7, because riskier loans default earlier,
  before much principal has been repaid.
- *Loss given default:* 89.2% of the outstanding principal is lost after recoveries and
  collection fees, and LGD is almost flat across grades (88.6–89.8%). Unsecured consumer loans
  recover little whoever the borrower is.
- *Over time:* LGD was about 93% for 2007–11 charge-offs and 89% for 2013–14. The share of
  charge-offs with no recorded recovery jumped from under 8% to 25% for 2014 loans, so recoveries
  on recent charge-offs are concentrated in fewer loans, and some may still have been in progress
  at the snapshot. Measuring on 2007–14 charge-offs keeps the 2015 portfolio's own outcomes out of
  its loss parameters.
- Together, a default costs about 51% of the funded amount (57.7% × 89.2%).

## expected loss

**Expected loss is $230m, 6.35% of funded principal. The 2015 book actually lost $274m (7.56%),
19% more.** The gap is almost entirely the PD shortfall found in the validation: 42,132 loans
defaulted against 37,304 expected (13% more). Loss severity held up: the realised loss per
default was within 2% of the measured LGD × EAD. The remainder comes from defaults falling on
larger loans than average.

Every grade lost more than expected, by 8% (grade 7) to 28% (grade 4). By purpose, debt
consolidation is 59% of exposure and 64% of expected loss; small-business loans carry the highest
loss rate (11.3% expected, 14.4% realised). Loans above $25k show the largest relative miss (5.1%
expected vs 7.5% realised), matching the validation finding that their PDs are the most
understated.

## el sensitivity

Expected loss moves almost proportionally with both inputs: scaling every PD by 1.25 raises it from
6.35% to 7.94%, and adding 10 points to LGD raises it to 7.06%. The realised 7.56% corresponds to
PDs about 19% higher than the model's; the PD shortfall (13% more defaults than predicted) accounts
for most of that. The PD is the input that matters most here, both because it is the least
accurate and because LGD is already close to its ceiling.

## distribution

The simulation reproduces the analytical results: simulated expected loss is within 0.14% of
PD × LGD × EAD (0.5 Monte Carlo standard errors), and the simulated 99% VaR is within 0.1% of the
large-portfolio (Vasicek) formula. With 280,000 small loans, idiosyncratic risk diversifies away
and the loss distribution is driven almost entirely by the economy factor. At 20,000 scenarios the
standard error of the 99% VaR is 0.09 points of funded amount, and 0.15 points for the 99% ES.

## var es

At the 99% level the book could lose 13.45% of its funded amount over the loans' life, against an
expected 6.35%. The 7.1-point difference (about $257m) is the unexpected loss that capital would
have to absorb. The 99% Expected Shortfall is 15.0%: in the worst 1% of scenarios, losses average
2.4 times the expected loss.

The realised 2015 loss sits at the 73rd percentile of the simulated distribution: a worse than
expected but unremarkable outcome under this model. That is the result of two offsetting errors,
not a confirmation of the model: the PDs are about 13% too low, and the correlation is an
assumption, not an estimate (section 5).

## correlation

**The default correlation drives the tail more than anything else.** Raising ρ from 0.02 to 0.20
moves the 99% VaR from 10.0% to 21.4% of funded amount, while expected loss stays at 6.35%
(it does not depend on ρ).

The data cannot pin ρ down. Matching the variation of quarterly vintage default rates gives only
0.013, or 0.004 after removing changes in borrower mix that the model already explains. But every
vintage with enough loans was issued between 2009 and 2015, during a steady recovery, so the
estimate measures how much default rates varied in good times only. Used in the simulation, it
would put the realised 2015 loss at about the 96th percentile, a one-in-25 outcome for what was
an ordinary year, which suggests it understates the risk.

The engine therefore uses the Basel other-retail correlation for each loan, a regulatory
benchmark built from data across economic cycles: 0.030–0.136 here, 0.065 on average. It is
evaluated at each loan's annualised PD, assuming a constant default rate over the 36 months.

## concentration

Name concentration is negligible: the largest loan is $35,000 (0.001% of the book), the largest 1%
of loans hold 2.7% of the exposure, and the effective number of equal-sized loans is about
203,000. The real concentration is by purpose: debt consolidation is 59% of exposure (Herfindahl
index 0.43).

Small portfolios carry more tail risk: a random 100-loan portfolio has a 99% VaR of 15.2% of
funded amount against 13.6% for the full book. The add-on fades quickly and is gone by about 500
loans, beyond which systematic risk dominates. Adding loans does not reduce the tail of a large
book; only lower correlation or lower PDs would.

The one-factor model cannot see the purpose concentration: every loan shares the same economy
factor, so a book made entirely of debt-consolidation loans would show no extra correlation. A
model with sector factors would be needed to price that risk.

## stress scenarios

- **Mild recession:** a 1-in-10 economy over the loans' life, about what LendingClub's 2007 loans
  went through (section 7.2). Unemployment rises moderately, recoveries worsen slightly (LGD +2
  points) and loans default earlier, with more principal still owed (EAD +3 points).
- **Severe recession:** a 1-in-50 economy, worse than anything in LendingClub's history. LGD and
  EAD return to the levels seen on 2007–10 loans (+4 and +6 points), and defaults move together
  more (correlation ×1.25).
- **Interest rate shock:** rates on borrowers' revolving debt rise by 300bp, which raises their
  debt-to-income and feeds through the scorecard, alongside the 1-in-5 slowdown that sharply
  tighter policy usually brings. LGD and EAD are unchanged.

The scenarios describe unemployment and borrowing costs, but the model does not translate
unemployment into the economy factor: nine annual vintages, one of them in a recession, are not
enough to estimate that relationship. Each scenario is set by how rare it is instead, which is
how the portfolio model measures the economy.

## historical anchor

The 2007 and 2008 vintages, issued just before and during the 2008–09 recession, default at
rates that imply 1-in-10.6 and 1-in-6.5 economies. By this measure the recession, as LendingClub's
borrowers experienced it, was roughly a 1-in-10 event, which is where the mild scenario sits; the
severe scenario goes well beyond anything in the data. The anchor is rough: the two vintages are
small (603 and 2,393 loans), they come from LendingClub's first years, when underwriting was
different, and the implied factor absorbs anything else that moved their default rates.

2015 implies a 1-in-3.1 economy: worse than average, although US unemployment fell throughout
2015–18. This is the drift found in the validation, seen through the portfolio model.
The realised 2015 loss of 7.56% sits between the 1-in-3.3 and 1-in-5 points of the economy curve,
which matches.

## stress results

**A mild recession raises lifetime expected loss from 6.35% to 10.32% of funded amount (×1.6);
a severe recession to 15.29% (×2.4), or $554m against $230m.** The interest rate shock gives 8.26%
(×1.3), almost all of it from the slower economy rather than the higher payments (section 7.5).

The severe-recession loss lies at the 99.7th percentile of the base-case distribution, beyond the
base 99% VaR of 13.45%. Capital sized on the base 99% VaR would absorb a mild recession, but not a
severe one with downturn LGD and EAD: on the downturn curve, 13.45% corresponds to roughly a
1-in-20 economy, not 1-in-100.

## stressed distributions

If the book entered each scenario and the economy were still uncertain from there, the 99% VaR
would rise from 13.4% of funded amount (base) to 16.6% (rate shock), 20.1% (mild) and 29.2%
(severe). These figures answer "how bad could it get from here?", so they stack the scenario's
stress on top of further downside. They are the stressed counterpart of the base VaR, not an
estimate of the scenario's own loss, which is the expected loss in section 7.3.

## rate shock

The direct channel barely registers: a 300bp rise on revolving debt adds 0.54 points to the median
borrower's debt-to-income, and lifts expected loss from 6.35% to 6.39%. LendingClub loans are
fixed-rate and its debt-to-income excludes mortgages, so higher rates do not reach most of these
borrowers' payments. Almost all of the scenario's impact (8.21% of the 8.26%) comes from the
economic slowdown that a sharp rate rise brings. A rate shock is a risk to this book through
unemployment, not through payments. The channel modelled here is the narrowest one: variable-rate
mortgages and other floating debt are not in the data.

## vulnerability

- *By grade:* the safest borrowers are hit hardest in relative terms: grade 1's expected loss
  rises 3.8 times, grade 7's 1.9 times. Each grade contributes 12–15% of the extra loss, although
  grade 7 alone holds 24% of the base expected loss. A recession spreads losses across the book,
  so a high-quality book offers less protection than its base expected loss suggests.
- *By purpose:* debt consolidation accounts for 61% of the extra loss (59% of exposure).
  Credit-card refinancing loans are more sensitive (×2.6) than debt consolidation (×2.3), because
  their borrowers are safer to start with.
- *By size:* loans above $25k rise most (×2.6, against ×2.3 for loans under $10k). That compounds
  the validation finding that these loans' PDs are already the most understated.

## stress sensitivity

Two assumptions matter most for the severe scenario. **Severity** (1-in-25 to 1-in-100) moves the
expected loss between 13.7% and 16.8%. The **PD level** moves it between 14.1% and 16.8%: removing
the 2015 calibration gap (PDs ×1.13) adds as much loss as moving from a 1-in-50 to a 1-in-100
economy. The downturn EAD (13.8–16.3%) and correlation (14.3–16.3%) come next. LGD matters least
(14.6–16.0%) because it is already close to 100%.

## reverse stress

Lifetime losses reach 10% of funded amount in a 1-in-12 economy with today's LGD and EAD, or a
1-in-6 economy with downturn LGD, EAD and correlation. 15% needs a 1-in-280 economy, or 1-in-44
with downturn assumptions. 20% needs a 1-in-11,606 or 1-in-497 economy. The downturn assumptions
shorten these return periods by a factor of 2 to 23, so they matter as much as the severity of the
economy itself.

## optimisation inputs

Repaid loans earned 1.33–1.34 times their interest rate per dollar over their life, in every band:
less than the 1.6 or so a 36-month loan pays if it runs to term, because many borrowers repaid early.
Defaulted loans earned 0.95–1.10 times their rate before they stopped paying, and less in the riskier
bands, where loans default sooner. Both factors come from 2007–14 loans.

The backtest is out of sample by vintage, not a real-time simulation: parameters come from loans issued
up to 2014, but an investor in 2015 could not yet have seen how 2013–14 loans turned out. The same
convention runs through the validation (the recalibration is fitted on 2014 outcomes).

## optimisation results

**Every strategy that reached for expected return did worse than buying the whole book; the one that
minimised expected loss did best.**

- *Whole book:* 5.30% lifetime net return (about 4.0% a year), against 6.86% expected.
- *Min expected loss:* 6.31%, a point above the book and within 0.03 points of the best result
  possible with hindsight under the same limits (6.34%). It holds our three best grades, almost all in
  LendingClub's A–C bands.
- *Max expected return:* 3.74%, against 14.91% expected: 1.6 points below the book and 11 points
  below its own forecast. It put all its money in LendingClub's C, D and E–G bands (the 35% limit in
  each of D and E–G), and 14.1% of its money was lost to defaults, against 7.5% for the whole book.
- *Min CVaR, and max return with a CVaR or expected-loss limit:* the same portfolio as max return.
  Under the scorecard model, the high-rate segments have the highest expected return *and* the lowest
  tail risk, because their rates appear to cushion losses even in bad economies. The model sees no
  trade-off between risk and return (section 8.1 chart), so the risk limits never bind.
- *Equal weight:* 4.45%, because it gives small, risky segments the same weight as large, safe ones.

## why return chasing failed

Three things went wrong in the riskier bands, and all three get worse as the rate rises:

1. **The PD is too flat across LendingClub's grades.** For E–G loans the scorecard predicts 20.8% and
   34.3% defaulted; for A loans it predicts 8.1% and 5.4% defaulted. LendingClub's grade contains risk
   information the scorecard leaves out, and the default-rate drift found in the validation was
   concentrated in the riskiest bands (E–G defaults rose from 24% on 2007–13 loans to 34% on 2015 loans).
2. **High-rate borrowers repaid early more often,** earning 1.27 times their rate instead of the
   assumed 1.34.
3. **High-rate loans defaulted earlier,** so less principal had been repaid: 0.57 of each dollar came
   back from E–G defaults, against 0.65 assumed.

For E–G loans, the expected 13.1% falls to 4.6% with the actual default rate, and to 0.1% once
prepayment and recoveries are as realised. An optimiser maximises exactly where its inputs are most
wrong, so these errors end up concentrated in its portfolio.

## input sensitivity

Neither alternative input rescues the return-seeking strategies. With the **price-informed model**, the
expected-loss limit now binds and lifts the result from 3.74% to 4.80%, the best of the return seekers,
but still half a point below the whole book. With **segment history**, minimising CVaR returns 5.08%,
also below the book. History is no guide here: the D and E–G segments returned 6–14% on 2007–14 loans
and between −3% and 6% on 2015 loans, so history's expected returns are uncorrelated with what happened
(−0.01 across segments), and both model-based inputs are negatively correlated (−0.35 and −0.26).

Minimising expected loss returns 6.31% under both PD inputs (the history input reuses the scorecard's
expected loss), because the lowest-loss segments, our best grades in LendingClub's A and B bands, are
much the same either way.

## implications

1. **Expected loss is understated by about 16%** (it would have needed to be 19% higher to match
   the realised loss) for as long as the PD level lags the default rate. Provisions based on these
   PDs would have been too low for 2015. The validation recommendations (recalibrate on the latest
   mature vintage, add a time-varying calibration) apply directly here.
2. **Capital is a correlation decision.** At the Basel retail correlation the 99% lifetime
   unexpected loss is 7.1% of funded amount; at ρ = 0.10 it would be about 9.6%. ρ should be set
   and reviewed as a governed assumption.
3. **Purpose concentration is the main unmodelled risk.** It is worth a limit, or a two-factor
   model, rather than more diversification by loan count.
4. **LGD is stable and high**, so there is little to gain from refining it. Effort is better spent
   on the PD level.
5. **Plan for a mild recession as a realistic case.** It raises lifetime losses by about 60% (to
   10.3%), and the base 99% VaR covers it. A severe recession (15.3%) exceeds that buffer once LGD
   and EAD worsen. Buffer and limit-setting should be based on the downturn curve, not the base
   case alone.
6. **The safest loans are not a hedge.** Grade 1 losses rise almost four times in a severe
   recession, and every grade contributes a similar share of the extra loss.
7. **Interest-rate risk runs through the economy.** Monitoring for this book should track
   unemployment and credit conditions rather than borrowers' payment burden.
8. **Select on low expected loss, not high expected return.** In the 2015 backtest, minimising expected
   loss beat the whole book by a point and matched the hindsight optimum; every return-seeking strategy
   lost to the book.
9. **Treat optimiser output as a stress test of the inputs.** When an optimiser concentrates in the
   highest-rate loans and reports no trade-off between return and tail risk, that is a sign of input
   error. Risk limits measured with the same inputs do not protect against it; limits measured
   independently (here, a price-informed PD) help a little.

## limitations

- *Lifetime horizon:* every figure covers the loans' 36-month life, not one year, so it cannot be
  compared directly with Basel one-year capital or IFRS 9 12-month expected loss.
- *Gaussian one-factor dependence:* one economy factor, no sector or regional factors, no tail
  dependence beyond what ρ implies, and the same ρ formula for every purpose.
- *LGD independent of the economy:* in a downturn, recoveries typically fall as defaults rise. The
  base case has no downturn LGD; the stress scenarios add fixed downturn add-ons to LGD and EAD,
  anchored on 2007–11 charge-offs rather than estimated.
- *Scenario severities are assumptions:* they are set by how rare they are, anchored on two small
  early vintages, not on an estimated link from unemployment or rates to defaults.
- *Rate shock covers revolving debt only:* the data does not show mortgages or other floating-rate
  debt, so the payment channel is a lower bound.
- *Correlation is assumed:* the Basel formula is calibrated for one-year PDs on bank retail books,
  and is applied here at annualised PDs under a constant-hazard assumption.
- *Loss parameters by grade only:* LGD and EAD vary by rating grade but not by purpose or loan size.
- *Recoveries may be incomplete:* 25% of 2014 charge-offs show no recovery, so recent LGDs may be
  slightly overstated.
- *PD inherited from the scorecard:* the expected loss carries the scorecard's calibration gap.
- *Optimisation is a one-vintage backtest:* one year of loans in one economic environment, with
  parameters that an investor in 2015 could not yet have fully seen. It shows how the strategies
  behaved in 2015, not how they would behave in general.
- *Returns are lifetime per dollar,* not time-weighted; annual figures are approximate. Strategies
  hold different mixes of loans that run for different lengths of time, and reinvestment is ignored.
- *Segments are bought pro rata:* every loan in a chosen segment, at the same weight, held to maturity.
  No secondary-market trading, purchase limits or taxes.

