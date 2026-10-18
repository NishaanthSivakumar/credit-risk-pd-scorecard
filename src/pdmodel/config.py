"""Project-wide settings. Change things here, not inside the modules."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
TABLE_DIR = REPORTS_DIR / "tables"

RANDOM_STATE = 42

# --- Sample definition -------------------------------------------------------
# Only 36-month loans, so every loan has the same outcome horizon.
# Only loans issued up to 2015: the Kaggle snapshot ends 2018Q4, so a 36m loan
# issued in Dec-2015 has just reached maturity. Later vintages are still
# "Current" and only their *early* outcomes are visible (resolution bias).
TERM = "36 months"
FIRST_ISSUE_YEAR = 2007
LAST_ISSUE_YEAR = 2015

# Chronological samples. This mimics how a bank would build a model on older
# vintages and then check it on newer ones:
#   train / test  issued <= DEV_LAST_YEAR, random 70/30 split (in-time development sample)
#   calib         issued in CALIB_YEAR: used only to fit the PD recalibration
#   oot           issued after CALIB_YEAR: untouched out-of-time test, used once
DEV_LAST_YEAR = 2013
CALIB_YEAR = 2014
TEST_SIZE = 0.30

# --- Target definition -------------------------------------------------------
GOOD_STATUSES = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}
BAD_STATUSES = {
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
}
# Anything else (Current, Late, In Grace Period) is unresolved -> excluded.

# Early default, for monitoring vintages that have not matured: the borrower
# stopped paying within the first EARLY_DEFAULT_MONTHS months.
EARLY_DEFAULT_MONTHS = 12
EARLY_BAD_STATUSES = BAD_STATUSES | {"Late (31-120 days)"}

# --- Columns -----------------------------------------------------------------
# Only information known at application time. Anything observed after the loan
# was issued (total_pymnt, recoveries, last_pymnt_d, out_prncp, ...) would leak
# the outcome and must never be used.
RAW_COLUMNS = [
    "loan_amnt", "term", "int_rate", "grade", "sub_grade", "emp_length",
    "home_ownership", "annual_inc", "verification_status", "issue_d",
    "loan_status", "purpose", "dti", "delinq_2yrs", "earliest_cr_line",
    "fico_range_low", "fico_range_high", "inq_last_6mths", "open_acc",
    "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc",
]

# Outcome fields, recorded after the loan was issued. They are loaded ONLY to
# measure loss given default and exposure at default (portfolio module) and for
# the data audit. They must never be model features: tests enforce this.
OUTCOME_COLUMNS = [
    "funded_amnt", "total_rec_prncp", "recoveries", "collection_recovery_fee",
    "total_pymnt", "last_pymnt_d",
]

# Post-origination fields in the Kaggle file. None of these may ever appear in
# NUMERIC_FEATURES or CATEGORICAL_FEATURES (checked by audit.leakage_check).
LEAKAGE_COLUMNS = {
    "loan_status", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
    "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low",
    "hardship_flag", "debt_settlement_flag", "settlement_status", "pymnt_plan",
    "funded_amnt_inv", "early_default", "months_observed",
}

NUMERIC_FEATURES = [
    "loan_amnt", "annual_inc", "loan_to_income", "dti", "fico",
    "credit_hist_years", "emp_length_yrs", "inq_last_6mths", "open_acc",
    "total_acc", "revol_bal", "revol_util", "delinq_2yrs", "pub_rec", "mort_acc",
]
CATEGORICAL_FEATURES = ["home_ownership", "purpose", "verification_status"]

# LendingClub's own grade is their PD estimate and int_rate is priced off it.
# We deliberately keep them OUT of the model and use sub_grade as a benchmark:
# "does our scorecard rank risk as well as LendingClub's own grading?"
BENCHMARK_COLUMN = "sub_grade"

# --- WoE binning --------------------------------------------------------------
MAX_BINS = 10           # starting number of quantile bins per numeric feature
MIN_BIN_FRAC = 0.05     # every bin must hold >= 5% of the training sample
RARE_CAT_FRAC = 0.01    # categories below 1% are pooled into "OTHER"

# --- Feature selection ---------------------------------------------------------
MIN_IV = 0.02           # IV < 0.02 = not predictive
SUSPICIOUS_IV = 0.50    # IV > 0.5 = "too good", check for leakage
MAX_CORR = 0.70         # drop the weaker of two WoE features above this |corr|
MAX_PVALUE = 0.05
# A feature missing for a large share of training loans is dropped. In this data
# heavy missingness comes from when a field was introduced, not from the
# borrower (mort_acc is blank for every loan issued before 2012), so its
# "Missing" bin would really be a vintage flag.
MAX_MISSING_FRAC = 0.10

# --- Scorecard scaling ---------------------------------------------------------
# Score 600 corresponds to good:bad odds of 30:1, and every 20 points doubles the odds.
BASE_SCORE = 600
BASE_ODDS = 30
PDO = 20

# --- Rating grades for calibration tests --------------------------------------
N_GRADES = 7

# --- Validation thresholds (traffic-light / RAG) ------------------------------
GINI_GREEN, GINI_AMBER = 0.40, 0.30
PSI_GREEN, PSI_AMBER = 0.10, 0.25

# --- Recalibration and uncertainty --------------------------------------------
RECALIBRATION_METHODS = ["none", "intercept", "intercept_slope"]
# The slope is only re-estimated when it differs from 1 by more than this (and significantly)
RECAL_SLOPE_MATERIAL = 0.10
BOOTSTRAP_DRAWS = 1000        # resamples per scheme
CI_LEVEL = 0.95

# --- Validation decision ------------------------------------------------------
# Calibration-in-the-large: |mean PD - observed default rate| in percentage points
CAL_GAP_GREEN, CAL_GAP_AMBER = 0.01, 0.03
# Grades failing the one-sided binomial test at 1%
FAILED_GRADES_GREEN, FAILED_GRADES_AMBER = 0, 2
# Relative Gini fall from train to out-of-time
GINI_DROP_GREEN, GINI_DROP_AMBER = 0.10, 0.20
# Challenger Gini advantage that would be material (about 10% of a 0.30 Gini)
CHALLENGER_GINI_AMBER, CHALLENGER_GINI_RED = 0.03, 0.05
# Sensitivity: spread of out-of-time Gini across alternative specifications,
# and relative change in mean PD across the coefficient-uncertainty band
ALT_SPEC_GINI_GREEN, ALT_SPEC_GINI_AMBER = 0.02, 0.05
COEF_PD_GREEN, COEF_PD_AMBER = 0.05, 0.10
# Share of loans still unresolved in the modelled vintages (target maturity)
UNRESOLVED_GREEN, UNRESOLVED_AMBER = 0.01, 0.03

# --- Portfolio risk (lifetime, 36-month horizon) ------------------------------
# LGD and EAD are measured on charged-off loans issued up to CALIB_YEAR, so the
# 2015 portfolio's own outcomes are never used to set its loss parameters.
# Asset correlation rho for the one-factor model:
#   "basel_retail" - Basel other-retail formula per loan (3%-16%, falling with PD),
#                    evaluated at the loan's annualised PD
#   "estimate"     - matched to the variation of quarterly vintage default rates
#   a number       - one rho for every loan
# The estimate is reported but not used by default: the vintages with enough loans
# (2009-2015) contain no recession, so they understate systematic risk.
ASSET_CORRELATION = "basel_retail"
TERM_YEARS = 3
MC_SIMULATIONS = 20_000
MC_SEED = 7
VAR_LEVELS = [0.95, 0.99]
ES_LEVELS = [0.975, 0.99]
CORRELATION_GRID = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]
CONCENTRATION_SIZES = [100, 500, 2_000, 10_000]
PD_MULTIPLIERS = [0.8, 0.9, 1.0, 1.1, 1.25, 1.5]
LGD_SHIFTS = [-0.10, -0.05, 0.0, 0.05, 0.10]

# --- Stress testing (lifetime, 36-month horizon) ------------------------------
# economy_quantile: probability that the economy over the loans' life is at least
#   this bad (0.10 = a 1-in-10 outcome); None = average economy (unconditional PDs).
#   Anchor: LendingClub's 2007 and 2008 vintages, which ran through the 2008-09
#   recession, imply roughly 1-in-10 and 1-in-6.5 (stress.implied_factor_by_vintage).
# lgd_add / ead_add: downturn add-ons. Anchor: charge-offs from 2007-11 loans lost
#   about 93% (vs 89% since), and those from 2007-10 loans owed 62-66% of the funded
#   amount (vs 57% overall).
# rho_mult: defaults move together more in a crisis.
# rate_shock: rise in interest rates on revolving debt, passed through debt-to-income.
STRESS_SCENARIOS = {
    "Base": dict(),
    "Mild recession": dict(economy_quantile=0.10, lgd_add=0.02, ead_add=0.03),
    "Severe recession": dict(economy_quantile=0.02, lgd_add=0.04, ead_add=0.06, rho_mult=1.25),
    "Interest rate shock": dict(economy_quantile=0.20, rate_shock=0.03),
}
# Low and high values for the tornado chart, around the severe recession.
# pd_scale 1.13 undoes the 2015 calibration gap (expected-to-observed ratio 0.885).
STRESS_SENSITIVITY = {
    "economy_quantile": (0.04, 0.01),
    "lgd_add": (0.0, 0.08),
    "ead_add": (0.0, 0.10),
    "rho_mult": (1.0, 1.5),
    "pd_scale": (0.9, 1.13),
}
STRESS_CURVE_QUANTILES = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
REVERSE_STRESS_THRESHOLDS = [0.08, 0.10, 0.15, 0.20]

# --- Portfolio optimisation (lifetime, per dollar funded) ---------------------
SERVICING_FEE = 0.01           # LendingClub's investor fee: 1% of every borrower payment
CHARGEOFF_LAG_MONTHS = 5       # principal stays outstanding ~5 months after the last payment
OPT_MIN_LOANS = 500            # segments (our grade x LendingClub band) with fewer loans are left out
OPT_MAX_WEIGHT = 0.10          # concentration limit per segment
OPT_BAND_CAP = 0.35            # limit per LendingClub grade band
OPT_ALPHA = 0.99               # CVaR level
OPT_SCENARIOS = 10_000
OPT_SEED = 11
OPT_FRONTIER_POINTS = 12

# --- Monitoring -----------------------------------------------------------------
# Loans issued after the modelled vintages act as the production stream.
MONITOR_FIRST_YEAR, MONITOR_LAST_YEAR = 2016, 2018
# Early default is only measured once a vintage has been observed this long, so
# that a borrower who stopped paying in month 12 has had time to reach 31+ days late.
EARLY_MIN_MONTHS_OBSERVED = 15
# (green-below, amber-below) thresholds; above amber is RED
MON_PSI = (PSI_GREEN, PSI_AMBER)                  # score PSI and per-feature CSI against train
MON_MISSING_CHANGE = (0.02, 0.05)                 # rise in a feature's missing share vs train
MON_OUT_OF_RANGE = (0.005, 0.02)                  # share of values outside the training range
MON_MEAN_PD_SHIFT = (0.10, 0.20)                  # relative change in mean PD vs the validated 2015 level
MON_CAL_GAP = (CAL_GAP_GREEN, CAL_GAP_AMBER)      # |mean PD - observed default rate|, matured vintages
MON_GINI_DROP = (GINI_DROP_GREEN, GINI_DROP_AMBER)  # relative fall vs validated out-of-time Gini
MON_EARLY_RATIO_SHIFT = (0.10, 0.20)              # relative change in early-default rate / mean PD vs development
