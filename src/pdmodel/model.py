"""Feature selection, logistic-regression scorecard and a gradient-boosting challenger."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config as C
from .woe import WoEBinner


# ----------------------------------------------------------------------------
# Feature selection
# ----------------------------------------------------------------------------
def select_by_iv_and_correlation(binner: WoEBinner, X_woe: pd.DataFrame,
                                 min_iv=C.MIN_IV, suspicious_iv=C.SUSPICIOUS_IV,
                                 max_corr=C.MAX_CORR) -> tuple[list[str], pd.DataFrame]:
    """Step 1: keep features with min_iv <= IV < suspicious_iv.
    Step 2: walk features from highest IV down; drop any whose WoE is highly
    correlated with a feature already kept (keeps the stronger of each pair).
    Returns the kept features and a log explaining every decision."""
    iv = binner.iv_table()["iv"]
    log, kept = [], []
    corr = X_woe.corr().abs()
    for f, v in iv.items():
        if v < min_iv:
            log.append((f, v, "dropped", f"IV {v:.3f} < {min_iv}"))
            continue
        if v >= suspicious_iv:
            log.append((f, v, "dropped", f"IV {v:.3f} >= {suspicious_iv}: check for leakage"))
            continue
        clash = [k for k in kept if corr.loc[f, k] > max_corr]
        if clash:
            log.append((f, v, "dropped", f"|corr| > {max_corr} with {clash[0]}"))
            continue
        kept.append(f)
        log.append((f, v, "kept", ""))
    return kept, pd.DataFrame(log, columns=["feature", "iv", "decision", "reason"])


def fit_logit(X_woe: pd.DataFrame, y: pd.Series):
    return sm.Logit(np.asarray(y), sm.add_constant(X_woe, has_constant="add")).fit(disp=0)


def backward_eliminate(X_woe: pd.DataFrame, y: pd.Series, features: list[str],
                       max_p=C.MAX_PVALUE) -> tuple[list[str], pd.DataFrame]:
    """Refit repeatedly, removing the worst feature until every coefficient is
    (a) significant and (b) NEGATIVE.

    Why negative? WoE is ln(%good/%bad), so higher WoE = safer. A model of
    P(default) must therefore put a negative weight on every WoE feature.
    A positive weight means multicollinearity is flipping the sign, and the
    scorecard would give *more* points to riskier attributes - a validator
    would reject it."""
    feats, log = list(features), []
    while feats:
        res = fit_logit(X_woe[feats], y)
        coefs, pvals = res.params.drop("const"), res.pvalues.drop("const")
        wrong_sign = coefs[coefs >= 0]
        if len(wrong_sign):
            worst = pvals[wrong_sign.index].idxmax()
            log.append((worst, "wrong sign", coefs[worst], pvals[worst]))
        elif pvals.max() > max_p:
            worst = pvals.idxmax()
            log.append((worst, f"p > {max_p}", coefs[worst], pvals[worst]))
        else:
            break
        feats.remove(worst)
    return feats, pd.DataFrame(log, columns=["feature", "reason", "coef", "p_value"])


def coefficient_table(res, X_woe: pd.DataFrame) -> pd.DataFrame:
    """Coefficients, p-values and VIF (variance inflation factor, >5 = collinearity concern)."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    Xc = sm.add_constant(X_woe, has_constant="add")
    vif = {c: variance_inflation_factor(Xc.values, i) for i, c in enumerate(Xc.columns) if c != "const"}
    t = pd.DataFrame({"coef": res.params, "std_err": res.bse, "p_value": res.pvalues})
    t["vif"] = pd.Series(vif)
    return t


# ----------------------------------------------------------------------------
# Scorecard
# ----------------------------------------------------------------------------
class Scorecard:
    """Turns the logistic model into points.

    Scaling: score = offset + factor * ln(odds_good)
      factor = PDO / ln(2)        -> +PDO points doubles the good:bad odds
      offset = base_score - factor * ln(base_odds)
    Because ln(odds_good) = -(b0 + sum_j b_j * WoE_j), the score splits into
    one points value per (feature, bin) - which is what a credit officer sees."""

    def __init__(self, binner: WoEBinner, features: list[str], logit_result,
                 base_score=C.BASE_SCORE, base_odds=C.BASE_ODDS, pdo=C.PDO):
        self.binner, self.features, self.res = binner, list(features), logit_result
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)

    # probabilities -----------------------------------------------------------
    def predict_pd(self, X_raw: pd.DataFrame) -> np.ndarray:
        X = sm.add_constant(self.binner.transform(X_raw, self.features), has_constant="add")
        return np.asarray(self.res.predict(X[["const", *self.features]]))

    def score(self, X_raw: pd.DataFrame) -> np.ndarray:
        pd_ = np.clip(self.predict_pd(X_raw), 1e-9, 1 - 1e-9)
        return self.offset + self.factor * np.log((1 - pd_) / pd_)

    # points table -------------------------------------------------------------
    def points_table(self) -> pd.DataFrame:
        b0, n = self.res.params["const"], len(self.features)
        rows = []
        for f in self.features:
            bj = self.res.params[f]
            for _, r in self.binner.bins[f].table.iterrows():
                pts = self.offset / n - self.factor * (b0 / n + bj * r["woe"])
                rows.append({"feature": f, "bin": r["bin"], "woe": r["woe"],
                             "points": round(pts), "pct_of_sample": r["pct_of_sample"],
                             "bad_rate": r["bad_rate"]})
        return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Challenger model
# ----------------------------------------------------------------------------
def fit_challenger(train: pd.DataFrame, numeric=C.NUMERIC_FEATURES, categorical=C.CATEGORICAL_FEATURES):
    """Gradient boosting on raw features. Used to answer: 'how much predictive
    power does the simple, explainable scorecard give up?'"""
    X = _challenger_matrix(train, numeric, categorical)
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
        l2_regularization=1.0, categorical_features="from_dtype",
        early_stopping=True, validation_fraction=0.15, random_state=C.RANDOM_STATE,
    )
    model.fit(X, train["default"])
    model._cat_levels = {c: X[c].cat.categories for c in categorical}
    return model


def _challenger_matrix(df, numeric, categorical, levels=None):
    X = df[numeric].astype(float).copy()
    for c in categorical:
        cats = levels[c] if levels else sorted(df[c].astype(str).unique())
        X[c] = pd.Categorical(df[c].astype(str), categories=cats)
    return X


def challenger_pd(model, df, numeric=C.NUMERIC_FEATURES, categorical=C.CATEGORICAL_FEATURES):
    X = _challenger_matrix(df, numeric, categorical, levels=model._cat_levels)
    return model.predict_proba(X)[:, 1]
