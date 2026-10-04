"""End-to-end run: data -> WoE -> scorecard -> challenger -> validation -> report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from . import data, model, plots, validation as V
from .woe import WoEBinner


def run(raw_path=None, nrows=None, reports_dir: Path = C.REPORTS_DIR, verbose=True) -> dict:
    say = print if verbose else (lambda *a, **k: None)
    fig_dir, tab_dir = Path(reports_dir) / "figures", Path(reports_dir) / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. data --------------------------------------------------------------
    say("1/6 Loading and preparing data ...")
    df = data.prepare(data.load_raw(raw_path, nrows=nrows))
    S = data.split_samples(df)
    summary = data.sample_summary(S)
    summary.to_csv(tab_dir / "sample_summary.csv")
    say(summary.to_string())

    # ---- 2. WoE binning (fit on TRAIN only) -------------------------------------
    say("2/6 WoE binning ...")
    binner = WoEBinner().fit(S["train"], S["train"]["default"], C.NUMERIC_FEATURES, C.CATEGORICAL_FEATURES)
    iv = binner.iv_table()
    iv.to_csv(tab_dir / "iv_table.csv")
    binner.woe_table().to_csv(tab_dir / "woe_bins.csv", index=False)
    Xw = {k: binner.transform(v) for k, v in S.items()}

    # ---- 3. feature selection + logistic regression ---------------------------
    say("3/6 Feature selection and model fit ...")
    shortlist, sel_log = model.select_by_iv_and_correlation(binner, Xw["train"])
    final, elim_log = model.backward_eliminate(Xw["train"], S["train"]["default"], shortlist)
    sel_log.to_csv(tab_dir / "selection_log.csv", index=False)
    elim_log.to_csv(tab_dir / "elimination_log.csv", index=False)
    res = model.fit_logit(Xw["train"][final], S["train"]["default"])
    coefs = model.coefficient_table(res, Xw["train"][final])
    coefs.to_csv(tab_dir / "coefficients.csv")
    card = model.Scorecard(binner, final, res)
    card.points_table().to_csv(tab_dir / "scorecard_points.csv", index=False)
    say(f"   final features ({len(final)}): {final}")

    # ---- 4. challenger ------------------------------------------------------------
    say("4/6 Challenger (gradient boosting) ...")
    gbm = model.fit_challenger(S["train"])

    # ---- 5. validation ----------------------------------------------------------
    say("5/6 Validation ...")
    pd_ = {k: card.predict_pd(v) for k, v in S.items()}
    sc = {k: card.score(v) for k, v in S.items()}
    y = {k: v["default"].to_numpy() for k, v in S.items()}
    gpd = {k: model.challenger_pd(gbm, v) for k, v in S.items()}

    metrics = {}
    for k in S:
        metrics[k] = {
            "n": int(len(y[k])),
            **V.discrimination(y[k], pd_[k]),
            **V.calibration_summary(y[k], pd_[k]),
            "challenger_gini": V.discrimination(y[k], gpd[k])["gini"],
            "lc_subgrade_gini": 2 * V.benchmark_auc(y[k], S[k][C.BENCHMARK_COLUMN]) - 1,
        }
    for k in ("test", "oot"):
        metrics[k]["score_psi_vs_train"] = V.psi(sc["train"], sc[k])
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv(tab_dir / "metrics_by_sample.csv")

    edges = V.grade_edges(sc["train"])
    binom = {k: V.binomial_test_by_grade(y[k], pd_[k], V.assign_grade(sc[k], edges)) for k in S}
    for k, t in binom.items():
        t.to_csv(tab_dir / f"binomial_test_{k}.csv")
    rank = {k: V.rank_ordering(y[k], sc[k]) for k in S}
    for k, t in rank.items():
        t.to_csv(tab_dir / f"rank_ordering_{k}.csv")
    csi = V.csi_by_feature(binner, S["train"], S["oot"], final)
    csi.to_csv(tab_dir / "csi_train_vs_oot.csv")

    # ---- 6. figures + report -----------------------------------------------------
    say("6/6 Figures and report ...")
    _save(plots.default_rate_by_vintage(df), fig_dir / "default_rate_by_vintage.png")
    _save(plots.roc_plot({k: (y[k], pd_[k]) for k in S}), fig_dir / "roc.png")
    _save(plots.calibration_plot({k: V.calibration_table(y[k], pd_[k]) for k in S}), fig_dir / "calibration.png")
    _save(plots.score_distribution(sc["oot"], y["oot"]), fig_dir / "score_distribution_oot.png")
    _save(plots.bad_rate_by_band(rank["oot"], title="Default rate by score decile (OOT)"), fig_dir / "rank_ordering_oot.png")
    _save(plots.psi_bar(csi), fig_dir / "csi_oot.png")
    for f in final:
        _save(plots.woe_plot(binner.bins[f].table), fig_dir / f"woe_{f}.png")

    report = _write_report(summary, iv, sel_log, elim_log, coefs, metrics_df, binom, rank, csi, final, card)
    (Path(reports_dir) / "validation_report.md").write_text(report)
    (Path(reports_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    say(f"Done. Report: {Path(reports_dir) / 'validation_report.md'}")
    return {"samples": S, "binner": binner, "features": final, "logit": res, "scorecard": card,
            "challenger": gbm, "metrics": metrics_df, "binomial": binom, "csi": csi}


def _save(ax, path):
    ax.figure.savefig(path)
    plt.close(ax.figure)


def _fmt_table(df: pd.DataFrame, floatfmt="{:.4f}") -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: floatfmt.format(v) if pd.notna(v) else "")
    d = d.reset_index()
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "---|" * len(d.columns)
    body = ["| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def _write_report(summary, iv, sel_log, elim_log, coefs, m, binom, rank, csi, final, card) -> str:
    oot, train = m.loc["oot"], m.loc["train"]
    gini_drop = (train["gini"] - oot["gini"]) / train["gini"]
    psi_oot = oot["score_psi_vs_train"]
    red_grades = (binom["oot"]["result"].str.startswith("RED")).sum()
    mono_oot = bool(rank["oot"]["monotonic"].all())
    cal_gap = oot["mean_pd"] - oot["observed_dr"]

    findings = [
        ("Discrimination (OOT Gini)", f"{oot['gini']:.3f}", V.rag_gini(oot["gini"])),
        ("Gini deterioration train → OOT", f"{gini_drop:.1%}",
         "GREEN" if gini_drop < 0.10 else ("AMBER" if gini_drop < 0.20 else "RED")),
        ("Score PSI train → OOT", f"{psi_oot:.3f}", V.rag_psi(psi_oot)),
        ("Rank ordering monotonic (OOT deciles)", "Yes" if mono_oot else "No", "GREEN" if mono_oot else "AMBER"),
        ("Calibration-in-the-large (OOT mean PD − observed DR)", f"{cal_gap:+.2%}",
         "GREEN" if abs(cal_gap) < 0.01 else ("AMBER" if abs(cal_gap) < 0.03 else "RED")),
        ("Grades failing binomial test (OOT, PD too low)", f"{red_grades} of {len(binom['oot'])}",
         "GREEN" if red_grades == 0 else ("AMBER" if red_grades <= 2 else "RED")),
    ]
    fnd = pd.DataFrame(findings, columns=["Test", "Result", "Status"]).set_index("Test")

    metric_cols = ["n", "auc", "gini", "ks", "mean_pd", "observed_dr", "brier", "hl_pvalue",
                   "challenger_gini", "lc_subgrade_gini", "score_psi_vs_train"]
    mm = m[[c for c in metric_cols if c in m.columns]].copy()
    mm["n"] = mm["n"].astype(int)

    return f"""# PD Scorecard — Model Validation Report

*Auto-generated by `scripts/run_pipeline.py`. Numbers below come from the latest run;
the commentary sections marked ✍️ are for you to write after reviewing the results.*

## 1. Executive summary

| Item | Value |
|---|---|
| Model | Logistic regression on WoE-binned application features (scorecard) |
| Portfolio | LendingClub 36-month unsecured consumer loans |
| Target | Default = charged off over the loan's life (36m horizon) |
| Final features | {len(final)}: {", ".join(f"`{f}`" for f in final)} |
| Scaling | {C.BASE_SCORE} points at {C.BASE_ODDS}:1 good:bad odds, PDO = {C.PDO} |

**Validation outcome**

{_fmt_table(fnd)}

✍️ *Overall conclusion (fit for purpose / fit with conditions / not fit), and the main conditions.*

## 2. Data and sample definition

{_fmt_table(summary)}

- Only 36-month loans issued {C.FIRST_ISSUE_YEAR}–{C.LAST_ISSUE_YEAR}, so every loan has had the
  chance to reach maturity in the data snapshot (avoids resolution bias).
- Loans still Current / Late / In Grace Period are excluded (outcome unknown).
- Development sample = vintages ≤ {C.DEV_LAST_YEAR} (70/30 train/test split, stratified).
  Out-of-time (OOT) sample = later vintages.
- Excluded on purpose: any post-origination field (leakage) and LendingClub's own
  `grade` / `sub_grade` / `int_rate` (kept only as a benchmark).

![Default rate by vintage](figures/default_rate_by_vintage.png)

## 3. Feature engineering and selection

Information Value (train):

{_fmt_table(iv)}

Selection by IV and correlation:

{_fmt_table(sel_log.set_index("feature"))}

Backward elimination (significance and sign checks):

{_fmt_table(elim_log.set_index("feature")) if len(elim_log) else "_No features removed._"}

## 4. Final model

{_fmt_table(coefs)}

All WoE coefficients are negative (higher WoE = safer = lower PD), as required. VIF > 5 would flag collinearity.

Full points table: `reports/tables/scorecard_points.csv`.

## 5. Performance by sample

{_fmt_table(mm)}

- `challenger_gini`: gradient-boosting model on the raw features — the price of interpretability.
- `lc_subgrade_gini`: LendingClub's own sub-grade used as a ranking — an external benchmark.

![ROC](figures/roc.png)
![Calibration](figures/calibration.png)
![Rank ordering OOT](figures/rank_ordering_oot.png)

✍️ *Discuss: train vs test (overfitting?), test vs OOT (time drift?), scorecard vs challenger vs LC grade.*

## 6. Calibration — binomial test by rating grade (OOT)

{_fmt_table(binom["oot"])}

✍️ *If the OOT default rate drifted away from the development rate, explain why (macro / underwriting
changes across vintages) and recommend a recalibration (e.g. re-fit the intercept on recent vintages).*

## 7. Stability

Score PSI train → test: **{m.loc['test', 'score_psi_vs_train']:.3f}**; train → OOT: **{psi_oot:.3f}**.

{_fmt_table(csi.to_frame())}

![CSI](figures/csi_oot.png)

## 8. Limitations and recommendations

✍️ *Ideas: reject inference (only approved loans are observed); PD here is lifetime-36m, not
12-month (IFRS 9 / Basel need 12m PD); no macro variables, so PD is not point-in-time; monitoring
plan with PSI and Gini thresholds; recalibration frequency.*
"""
