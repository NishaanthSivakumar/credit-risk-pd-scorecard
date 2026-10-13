"""End-to-end run: data -> audit -> scorecard -> challenger -> validation ->
recalibration -> uncertainty -> sensitivity -> decision -> portfolio risk -> reports."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import assessment as A
from . import audit, data, model, plots, reports
from . import bootstrap as B
from . import calibration as K
from . import config as C
from . import portfolio as P
from . import sensitivity as SE
from . import simulation as MC
from . import validation as V


def run(raw_path=None, reports_dir: Path = C.REPORTS_DIR, quick: bool = False, verbose=True) -> dict:
    """quick=True cuts bootstrap draws and Monte Carlo scenarios, for smoke tests only."""
    say = print if verbose else (lambda *a, **k: None)
    rd = Path(reports_dir)
    fig_dir, tab_dir = rd / "figures", rd / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    n_boot = 100 if quick else C.BOOTSTRAP_DRAWS
    n_sims = 2_000 if quick else C.MC_SIMULATIONS
    R: dict = {"quick": quick}

    # ---- 1. data ------------------------------------------------------------------
    say("1/9 Loading and preparing data ...")
    raw = data.load_raw(raw_path)
    df = data.prepare(raw)
    S = data.split_samples(df)
    R["summary"] = data.sample_summary(S)
    R["summary"].to_csv(tab_dir / "sample_summary.csv")
    say(R["summary"].to_string())

    # ---- 2. development (train only) -----------------------------------------------------
    say("2/9 Scorecard development and challenger ...")
    dev = model.develop_scorecard(S["train"])
    binner, final, res, card = dev["binner"], dev["features"], dev["logit"], dev["scorecard"]
    R.update(binner=binner, final=final, card=card, iv=binner.iv_table(), sel_log=dev["selection_log"],
             elim_log=dev["elimination_log"])
    R["iv"].to_csv(tab_dir / "iv_table.csv")
    binner.woe_table().to_csv(tab_dir / "woe_bins.csv", index=False)
    R["sel_log"].to_csv(tab_dir / "selection_log.csv", index=False)
    R["elim_log"].to_csv(tab_dir / "elimination_log.csv", index=False)
    R["coefs"] = model.coefficient_table(res, binner.transform(S["train"], final))
    R["coefs"].to_csv(tab_dir / "coefficients.csv")
    card.points_table().to_csv(tab_dir / "scorecard_points.csv", index=False)
    gbm = model.fit_challenger(S["train"])
    say(f"   final features ({len(final)}): {final}")

    # ---- 3. data audit ---------------------------------------------------------------------
    say("3/9 Data audit ...")
    checks, audit_tabs = audit.run_checks(raw, df, S, final)
    del raw
    R.update(checks=checks, audit=audit_tabs)
    for k, t in audit_tabs.items():
        t.to_csv(tab_dir / f"audit_{k}.csv")
    audit.checks_table(checks).to_csv(tab_dir / "audit_checks.csv")

    # ---- 4. validation of the model as developed ------------------------------------------------
    say("4/9 Validation ...")
    y = {k: v["default"].to_numpy() for k, v in S.items()}
    pd_ = {k: card.predict_pd(v) for k, v in S.items()}
    sc = {k: card.score(v) for k, v in S.items()}
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
        if k != "train":
            metrics[k]["score_psi_vs_train"] = V.psi(sc["train"], sc[k])
    R["metrics"] = pd.DataFrame(metrics).T
    R["metrics"].to_csv(tab_dir / "metrics_by_sample.csv")
    edges = V.grade_edges(sc["train"])
    R["edges"] = edges
    grade = {k: V.assign_grade(sc[k], edges) for k in S}
    R["binom"] = {k: V.binomial_test_by_grade(y[k], pd_[k], grade[k]) for k in S}
    for k, t in R["binom"].items():
        t.to_csv(tab_dir / f"binomial_test_{k}.csv")
    R["rank"] = {k: V.rank_ordering(y[k], sc[k]) for k in S}
    for k, t in R["rank"].items():
        t.to_csv(tab_dir / f"rank_ordering_{k}.csv")
    R["csi"] = V.csi_by_feature(binner, S["train"], S["oot"], final)
    R["csi"].to_csv(tab_dir / "csi_train_vs_oot.csv")

    # ---- 5. recalibration: fit on calib, evaluate once on oot -------------------------------------
    say("5/9 Recalibration ...")
    rc = K.compare_methods(y["calib"], pd_["calib"], y["oot"], pd_["oot"], grade["oot"],
                           C.RECALIBRATION_METHODS, C.RECAL_SLOPE_MATERIAL)
    method, why, recals, deciles = rc["method"], rc["reason"], rc["recalibrators"], rc["deciles"]
    R.update(recal_method=method, recal_reason=why, recals=recals, recal_table=rc["table"],
             recal_deciles=deciles)
    R["recal_table"].to_csv(tab_dir / "recalibration_oot.csv")
    pd.concat(deciles, names=["method"]).to_csv(tab_dir / "recalibration_deciles_oot.csv")
    cm = model.CalibratedScorecard(card, recals[method], edges)
    pdc = {k: cm.predict_pd(v) for k, v in S.items()}
    R["cm"] = cm
    R["binom_recal"] = V.binomial_test_by_grade(y["oot"], pdc["oot"], grade["oot"])
    R["binom_recal"].to_csv(tab_dir / "binomial_test_oot_recalibrated.csv")
    g_recal = K.Recalibrator("intercept").fit(y["calib"], gpd["calib"])
    say(f"   method: {method} ({why})")

    # ---- 6. bootstrap confidence intervals on oot -------------------------------------------------
    say(f"6/9 Bootstrap ({n_boot} draws x 2 schemes) ...")
    preds = {"scorecard": pdc["oot"], "challenger": g_recal.transform(gpd["oot"])}
    point = B.point_estimates(y["oot"], preds)
    groups = {"month": B.month_groups(S["oot"]["issue_d"]), "loan": None}
    ci, paired = {}, {}
    for scheme, g in groups.items():
        draws = B.bootstrap(y["oot"], preds, g, n_draws=n_boot, seed=C.RANDOM_STATE)
        ci[scheme] = B.intervals(draws, point, C.CI_LEVEL)
        paired[scheme] = B.paired_difference(draws, point, "scorecard", "challenger", level=C.CI_LEVEL)
    R["ci"] = pd.concat(ci, names=["scheme"])
    R["paired"] = pd.concat(paired, names=["scheme"])
    R["ci"].to_csv(tab_dir / "bootstrap_intervals_oot.csv")
    R["paired"].to_csv(tab_dir / "bootstrap_challenger_vs_scorecard_oot.csv")

    # ---- 7. sensitivity analysis on oot -------------------------------------------------------
    say("7/9 Sensitivity analysis ...")
    oot = S["oot"]
    sens = {
        "characteristics": SE.characteristic_shifts(cm, oot),
        "missing": SE.missing_inputs(cm, oot, y["oot"]),
        "thresholds": SE.threshold_table(y["oot"], pdc["oot"]),
        "segments": SE.segment_performance(y["oot"], pdc["oot"], SE.default_segments(oot)),
        "alternatives": SE.alternative_specifications(S, method),
    }
    sens["coef_single"], sens["coef_joint"] = SE.coefficient_sensitivity(cm, oot, y["oot"],
                                                                          n_draws=100 if quick else 500)
    R["sens"] = sens
    for k, t in sens.items():
        t.to_csv(tab_dir / f"sensitivity_{k}.csv")

    # ---- 8. structured assessment and decision -------------------------------------------------
    say("8/9 Validation assessment ...")
    register = A.read_register(rd / "limitations_register.md")
    common = A.gather_evidence(checks, R["coefs"], final, binner, R["metrics"], R["csi"], R["rank"]["oot"],
                               R["ci"], R["paired"], sens, register)
    R["dims"], R["decision"] = A.assess_versions(common, y["oot"],
                                                 {"as developed": pd_["oot"], "recalibrated": pdc["oot"]},
                                                 grade["oot"])
    for version, dims in R["dims"].items():
        A.dimensions_table(dims).to_csv(tab_dir / f"assessment_{version.replace(' ', '_')}.csv")
    R["register"] = register
    say(f"   as developed: {R['decision']['as developed'][0]}; recalibrated: {R['decision']['recalibrated'][0]}")

    # ---- figures and Phase 1 reports ------------------------------------------------------------
    _save(plots.default_rate_by_vintage(df), fig_dir / "default_rate_by_vintage.png")
    _save(plots.roc_plot({k: (y[k], pd_[k]) for k in S}), fig_dir / "roc.png")
    _save(plots.calibration_plot({k: V.calibration_table(y[k], pd_[k]) for k in S}), fig_dir / "calibration.png")
    _save(plots.calibration_plot({m: t for m, t in deciles.items()}, colors=plots.METHOD_COLORS,
                                 title="Reliability by PD decile, 2015 loans"), fig_dir / "reliability_oot.png")
    _save(plots.ci_plot(R["ci"]), fig_dir / "bootstrap_gini_oot.png")
    _save(plots.score_distribution(sc["oot"], y["oot"]), fig_dir / "score_distribution_oot.png")
    _save(plots.bad_rate_by_band(R["rank"]["oot"], title="Default rate by score decile (out-of-time, 2015)"),
          fig_dir / "rank_ordering_oot.png")
    _save(plots.psi_bar(R["csi"]), fig_dir / "csi_oot.png")
    for f in final:
        _save(plots.woe_plot(binner.bins[f].table), fig_dir / f"woe_{f}.png")
    notes = reports.load_commentary(rd / "commentary.md")
    (rd / "data_audit.md").write_text(reports.data_audit_report(R, notes))
    (rd / "validation_report.md").write_text(reports.validation_report(R, notes))
    (rd / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    # ---- 9. portfolio risk (2015 vintage) ----------------------------------------------------------
    say(f"9/9 Portfolio risk ({n_sims:,} scenarios) ...")
    hist = pd.concat([S["train"], S["test"], S["calib"]], ignore_index=True)
    vint = pd.concat([S[k].assign(pd=pdc[k]) for k in S], ignore_index=True)
    R["portfolio"] = run_portfolio(oot, pdc["oot"], grade["oot"], hist, cm.grade(hist), vint,
                                   n_sims, fig_dir, tab_dir)
    pnotes = reports.load_commentary(rd / "portfolio_commentary.md")
    (rd / "portfolio_risk_report.md").write_text(reports.portfolio_report(R["portfolio"], pnotes))
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({"calibrated_model": cm, "challenger": gbm, "portfolio": R["portfolio"]["loans"]},
                 C.PROCESSED_DIR / "pipeline_outputs.pkl")
    say(f"Done. Reports in {rd}")
    return R


def run_portfolio(oot: pd.DataFrame, pd_oot, grade_oot, hist: pd.DataFrame, grade_hist,
                  vintages: pd.DataFrame, n_sims: int, fig_dir: Path, tab_dir: Path) -> dict:
    """Expected loss, Monte Carlo loss distribution, correlation and concentration
    analysis for the 2015 vintage, with LGD/EAD measured on earlier charge-offs."""
    params = P.estimate_loss_parameters(hist, grade_hist)
    by_year = P.loss_parameters_by_year(hist)
    port = P.build_portfolio(oot, pd_oot, grade_oot, params)
    summ = P.summary(port)
    exp = summ["exposure"]
    seg = {c: P.el_by_segment(port, c) for c in ("grade", "purpose", "size_band", "home_ownership")}
    conc = P.concentration(port)
    sens = P.el_sensitivity(port)

    # default correlation from quarterly vintages, mix-adjusted by the model PD
    q = vintages.assign(q=vintages["issue_d"].dt.to_period("Q")).groupby("q").agg(
        n=("default", "size"), observed_dr=("default", "mean"), mean_pd=("pd", "mean"))
    rho_est, rho_tab = MC.estimate_correlation(q)
    if C.ASSET_CORRELATION == "basel_retail":
        rho = MC.basel_retail_correlation(MC.annualised_pd(port["pd"].to_numpy(), C.TERM_YEARS))
    elif C.ASSET_CORRELATION == "estimate":
        rho = rho_est["rho_mix_adjusted"]
    else:
        rho = float(C.ASSET_CORRELATION)
    rho_avg = float(np.average(np.broadcast_to(rho, len(port)), weights=port["ead"]))

    pdv, lgd, ead = port["pd"].to_numpy(), port["lgd"].to_numpy(), port["ead"].to_numpy()
    losses, dr, _ = MC.simulate_losses(pdv, lgd, ead, rho, n_sims, seed=C.MC_SEED)
    rm = MC.risk_measures(losses, C.VAR_LEVELS, C.ES_LEVELS)
    el_an = MC.analytical_el(pdv, lgd, ead)
    check = pd.DataFrame({
        "simulated": [rm["expected_loss"], rm["var_0.99"]],
        "analytical": [el_an, MC.asrf_quantile(pdv, lgd, ead, rho, 0.99)],
    }, index=["Expected loss", "VaR 99% (large-portfolio formula)"])
    check["difference"] = check["simulated"] / check["analytical"] - 1
    check["mc_standard_errors"] = [(rm["expected_loss"] - el_an) / (rm["std"] / np.sqrt(n_sims)), np.nan]
    realised = summ["realised_loss"]
    realised_pct = float((losses <= realised).mean())
    L_est, _, _ = MC.simulate_losses(pdv, lgd, ead, rho_est["rho_mix_adjusted"], max(2_000, n_sims // 2),
                                     seed=C.MC_SEED + 3)
    rho_est["realised_percentile"] = float((L_est <= realised).mean())
    rho_est["var_0.99_pct"] = float(np.quantile(L_est, 0.99) / exp * 100)

    sizes = [s for s in (1_000, 2_000, 5_000, 10_000, 20_000, 50_000) if s <= n_sims]
    conv_pct = MC.convergence(losses, sizes) / exp * 100

    corr_rows = {}
    for r_ in C.CORRELATION_GRID:
        L, _, _ = MC.simulate_losses(pdv, lgd, ead, r_, max(2_000, n_sims // 2), seed=C.MC_SEED + 1)
        corr_rows[r_] = {k: v / exp * 100 for k, v in MC.risk_measures(L, C.VAR_LEVELS, C.ES_LEVELS).items()}
    corr = pd.DataFrame(corr_rows).T.rename_axis("rho")

    rng = np.random.default_rng(C.MC_SEED)
    conc_rows = {}
    for n in [*C.CONCENTRATION_SIZES, len(port)]:
        idx = rng.choice(len(port), n, replace=False) if n < len(port) else np.arange(len(port))
        rho_n = rho[idx] if np.ndim(rho) else rho
        L, _, _ = MC.simulate_losses(pdv[idx], lgd[idx], ead[idx], rho_n, max(2_000, n_sims // 2) if n < len(port)
                                     else n_sims, seed=C.MC_SEED + 2)
        e = port["exposure"].to_numpy()[idx].sum()
        conc_rows[n] = {k: v / e * 100 for k, v in MC.risk_measures(L, C.VAR_LEVELS, C.ES_LEVELS).items()}
    conc_sim = pd.DataFrame(conc_rows).T.rename_axis("loans")

    rm_pct = {k: v / exp * 100 for k, v in rm.items()}
    _save(plots.loss_distribution(losses / exp * 100, rm_pct, realised / exp * 100), fig_dir / "loss_distribution.png")
    _save(plots.correlation_sensitivity(corr), fig_dir / "correlation_sensitivity.png")
    _save(plots.convergence_plot(conv_pct), fig_dir / "mc_convergence.png")
    _save(plots.concentration_plot(conc_sim), fig_dir / "concentration_risk.png")
    _save(plots.el_by_grade(seg["grade"]), fig_dir / "el_by_grade.png")

    out = {"params": params, "params_by_year": by_year, "summary": summ, "segments": seg,
           "concentration": conc, "el_sensitivity": sens, "rho": rho_avg, "rho_loans": rho,
           "rho_method": C.ASSET_CORRELATION, "rho_estimate": rho_est,
           "rho_vintages": rho_tab, "risk": rm, "risk_pct": rm_pct, "analytic_check": check,
           "realised_percentile": realised_pct, "convergence": conv_pct, "correlation": corr,
           "concentration_sim": conc_sim, "n_sims": n_sims, "loans": port,
           "scenario_default_rate": dr}
    params.to_csv(tab_dir / "portfolio_lgd_ead_by_grade.csv")
    by_year.to_csv(tab_dir / "portfolio_lgd_ead_by_year.csv")
    pd.Series(summ).to_csv(tab_dir / "portfolio_summary.csv")
    for k, t in seg.items():
        t.to_csv(tab_dir / f"portfolio_el_by_{k}.csv")
    conc.to_csv(tab_dir / "portfolio_concentration.csv")
    sens.to_csv(tab_dir / "portfolio_el_sensitivity.csv")
    pd.Series(rm).to_csv(tab_dir / "portfolio_risk_measures.csv")
    check.to_csv(tab_dir / "portfolio_mc_vs_analytical.csv")
    corr.to_csv(tab_dir / "portfolio_correlation_sensitivity.csv")
    conv_pct.to_csv(tab_dir / "portfolio_mc_convergence.csv")
    conc_sim.to_csv(tab_dir / "portfolio_concentration_simulation.csv")
    rho_tab.to_csv(tab_dir / "portfolio_vintage_default_rates.csv")
    return out


def _save(ax, path):
    ax.figure.savefig(path)
    plt.close(ax.figure)
