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

# Development sample = issued <= DEV_LAST_YEAR (split into train / test).
# Out-of-time (OOT) sample = issued after that. This mimics how a bank would
# build a model on older vintages and then check it on newer ones.
DEV_LAST_YEAR = 2013
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
