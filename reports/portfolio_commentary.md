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

## limitations

- *Lifetime horizon:* every figure covers the loans' 36-month life, not one year, so it cannot be
  compared directly with Basel one-year capital or IFRS 9 12-month expected loss.
- *Gaussian one-factor dependence:* one economy factor, no sector or regional factors, no tail
  dependence beyond what ρ implies, and the same ρ formula for every purpose.
- *LGD independent of the economy:* in a downturn, recoveries typically fall as defaults rise. This
  base case has no downturn LGD; Phase 3 stress scenarios add one.
- *Correlation is assumed:* the Basel formula is calibrated for one-year PDs on bank retail books,
  and is applied here at annualised PDs under a constant-hazard assumption.
- *Loss parameters by grade only:* LGD and EAD vary by rating grade but not by purpose or loan size.
- *Recoveries may be incomplete:* 25% of 2014 charge-offs show no recovery, so recent LGDs may be
  slightly overstated.
- *PD inherited from the scorecard:* the expected loss carries the scorecard's calibration gap.
