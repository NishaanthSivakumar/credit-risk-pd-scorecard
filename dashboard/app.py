"""Interactive summary of the project's results.

    streamlit run dashboard/app.py

Reads only the precomputed tables in reports/tables/ (written by
scripts/run_pipeline.py), so it needs neither the Kaggle data nor the models.
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
REPO = "https://github.com/NishaanthSivakumar/credit-risk-pd-scorecard/blob/main"

BLUE, ORANGE, AQUA, PURPLE, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#7d5bd6", "#8a8984"
RAG = {"GREEN": "#1baf7a", "AMBER": "#e8a33d", "RED": "#d64545"}


@st.cache_data
def table(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{name}.csv", **kw)


@st.cache_data
def metrics() -> dict:
    return json.loads((ROOT / "reports" / "metrics.json").read_text())


def pct(v, d=1):
    return f"{v * 100:.{d}f}%"


def rag_style(v):
    return f"background-color: {RAG[v]}; color: white" if v in RAG else ""


st.set_page_config(page_title="PD Scorecard and Credit Portfolio Risk", layout="wide")
st.title("PD scorecard and credit portfolio risk")
st.caption("LendingClub 36-month loans · developed on 2007–13, recalibrated on 2014, tested on 2015, "
           "monitored on 2016–18 · all losses and returns are over the loans' 36-month life")

tabs = st.tabs(["Overview", "Validation", "Monitoring", "Portfolio risk", "Stress testing", "Optimisation"])

# ----------------------------------------------------------------------------
with tabs[0]:
    m = metrics()
    summ = table("portfolio_summary", index_col=0).iloc[:, 0]
    risk = table("portfolio_risk_measures", index_col=0).iloc[:, 0]
    strat = table("optimisation_strategies", index_col=0)
    c = st.columns(4)
    c[0].metric("Out-of-time Gini (2015)", f"{m['oot']['gini']:.3f}", f"{m['oot']['gini'] - m['train']['gini']:+.3f} vs train")
    c[1].metric("2015 default rate vs PD (as developed)", pct(m["oot"]["observed_dr"]),
                f"model {pct(m['oot']['mean_pd'])}", delta_color="off")
    c[2].metric("Expected vs realised lifetime loss", pct(summ["el_rate"], 2), f"realised {pct(summ['realised_loss_rate'], 2)}",
                delta_color="off")
    c[3].metric("Best strategy vs whole book", pct(strat.loc["Min expected loss", "realised_return"], 2),
                f"{(strat.loc['Min expected loss', 'realised_return'] - strat.loc['Whole book', 'realised_return']) * 100:+.2f} pts")
    st.markdown(f"""
**Findings**

- The scorecard ranks risk consistently on loans it never saw, but its PD level drifts: a recalibration fitted
  on 2014 closes only a quarter of the 2015 gap. Validation decision: **Recalibrate**.
- On the 2015 book, expected lifetime loss is {pct(summ['el_rate'], 2)} of the funded amount against
  {pct(summ['realised_loss_rate'], 2)} realised; the 99% lifetime VaR is {pct(risk['var_0.99'] / summ['exposure'], 2)}.
- A severe recession (1-in-50 economy) would raise expected loss to about 15%.
- Minimising expected loss beat buying the whole book; every strategy that reached for expected return lost to it.
- Monitoring of 2016–18 loans calls for redevelopment by late 2017, years before those loans matured.

Full write-ups: [validation report]({REPO}/reports/validation_report.md) ·
[portfolio risk report]({REPO}/reports/portfolio_risk_report.md) ·
[monitoring report]({REPO}/reports/monitoring_report.md) · [model card]({REPO}/MODEL_CARD.md)
""")

# ----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Discrimination by sample")
    mbs = table("metrics_by_sample", index_col=0)
    mbs.index = mbs.index.map({"train": "Train 2007–13", "test": "Test 2007–13", "calib": "Calibration 2014",
                               "oot": "Out-of-time 2015"})
    st.dataframe(mbs[["n", "gini", "ks", "mean_pd", "observed_dr", "challenger_gini", "lc_subgrade_gini"]]
                 .style.format({"n": "{:,.0f}", "gini": "{:.3f}", "ks": "{:.3f}", "mean_pd": "{:.2%}",
                                "observed_dr": "{:.2%}", "challenger_gini": "{:.3f}", "lc_subgrade_gini": "{:.3f}"}),
                 use_container_width=True)

    st.subheader("Calibration on 2015 loans")
    dec = table("recalibration_deciles_oot")
    method = st.radio("Recalibration fitted on 2014", ["none", "intercept", "intercept_slope"], index=1, horizontal=True)
    d = (dec[dec["method"] == method].rename(columns={"mean_pd": "predicted", "observed_dr": "observed"})
         .melt(id_vars="decile", value_vars=["predicted", "observed"], var_name="series", value_name="rate"))
    st.altair_chart(alt.Chart(d).mark_bar().encode(
        x=alt.X("decile:O", title="PD decile (1 = safest)", axis=alt.Axis(labelAngle=0)), xOffset="series:N",
        y=alt.Y("rate:Q", title="Lifetime default rate", axis=alt.Axis(format="%")),
        color=alt.Color("series:N", scale=alt.Scale(range=[BLUE, ORANGE]), title=None),
        tooltip=["decile", "series", alt.Tooltip("rate:Q", format=".2%")]), use_container_width=True)

    st.subheader("95% bootstrap intervals, out-of-time")
    ci = table("bootstrap_intervals_oot")
    ci = ci[ci["metric"].isin(["gini", "eo_ratio", "brier"])]
    st.dataframe(ci.pivot_table(index=["model", "metric"], columns="scheme", values=["lower", "upper"]).round(4),
                 use_container_width=True)

    st.subheader("Validation assessment (recalibrated model)")
    a = table("assessment_recalibrated", index_col=0)
    st.dataframe(a.style.map(rag_style, subset=["status"]), use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Alerts by issue quarter")
    grid = table("monitoring_alert_grid", index_col=0).fillna("")
    st.dataframe(grid.style.map(rag_style), use_container_width=True, height=740)
    measure = st.selectbox("Measure over time", ["First-year defaults vs PD", "Score PSI and largest CSI", "Mean PD"])
    if measure == "First-year defaults vs PD":
        e = table("monitoring_early")
        st.altair_chart(alt.Chart(e).mark_line(point=True, color=BLUE).encode(
            x=alt.X("quarter:N", title="Issue quarter"),
            y=alt.Y("ratio_shift:Q", title="Change vs development benchmark", axis=alt.Axis(format="%")),
            tooltip=["quarter", alt.Tooltip("early_dr:Q", format=".2%"), alt.Tooltip("ratio_shift:Q", format="+.1%")]),
            use_container_width=True)
        st.caption("Borrowers who stopped paying within 12 months, relative to the model's PDs, against the same ratio on "
                   "the development loans. Amber at ±10%, red at ±20%.")
    elif measure == "Score PSI and largest CSI":
        s = table("monitoring_stability").melt(id_vars="quarter", value_vars=["score_psi", "max_csi"],
                                               var_name="index", value_name="value")
        st.altair_chart(alt.Chart(s).mark_line(point=True).encode(
            x=alt.X("quarter:N", title="Issue quarter"), y=alt.Y("value:Q", title="Index vs training sample"),
            color=alt.Color("index:N", scale=alt.Scale(range=[ORANGE, BLUE]), title=None)), use_container_width=True)
        st.caption("Amber at 0.10, red at 0.25.")
    else:
        p = table("monitoring_predictions")
        st.altair_chart(alt.Chart(p).mark_line(point=True, color=PURPLE).encode(
            x=alt.X("quarter:N", title="Issue quarter"), y=alt.Y("mean_pd:Q", title="Mean PD", axis=alt.Axis(format="%"),
                                                                 scale=alt.Scale(zero=False))), use_container_width=True)
    st.subheader("Triggers")
    st.dataframe(table("monitoring_triggers"), use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------------
with tabs[3]:
    corr = table("portfolio_correlation_sensitivity")
    rho = st.select_slider("Asset correlation ρ (one value for every loan)", options=corr["rho"].tolist(), value=0.05)
    r = corr[corr["rho"] == rho].iloc[0]
    c = st.columns(4)
    c[0].metric("Expected loss", f"{r['expected_loss']:.2f}%")
    c[1].metric("99% VaR", f"{r['var_0.99']:.2f}%")
    c[2].metric("99% Expected Shortfall", f"{r['es_0.99']:.2f}%")
    c[3].metric("Unexpected loss (VaR − EL)", f"{r['var_0.99'] - r['expected_loss']:.2f}%")
    st.caption("Lifetime loss as a share of funded amount, 2015 book, 10,000 scenarios per ρ. "
               "The report's base case uses the Basel retail correlation per loan (average 0.065).")
    g = (table("portfolio_el_by_grade").rename(columns={"el_rate": "expected", "realised_loss_rate": "realised"})
         .melt(id_vars="grade", value_vars=["expected", "realised"], var_name="series", value_name="rate"))
    st.subheader("Expected vs realised lifetime loss by rating grade")
    st.altair_chart(alt.Chart(g).mark_bar().encode(
        x=alt.X("grade:O", title="Rating grade (1 = best)", axis=alt.Axis(labelAngle=0)), xOffset="series:N",
        y=alt.Y("rate:Q", title="Loss / funded amount", axis=alt.Axis(format="%")),
        color=alt.Color("series:N", scale=alt.Scale(range=[BLUE, ORANGE]), title=None)), use_container_width=True)
    st.subheader("Portfolio size and tail risk")
    size = table("portfolio_concentration_simulation", index_col=0)[["expected_loss", "var_0.99", "es_0.99"]]
    size.index = [f"{n:,} loans" if n < size.index.max() else f"full book ({n:,} loans)" for n in size.index]
    size.columns = ["expected loss (%)", "99% VaR (%)", "99% ES (%)"]
    st.dataframe(size.round(2), use_container_width=True)

# ----------------------------------------------------------------------------
with tabs[4]:
    el = table("stress_expected_loss", index_col=0)
    rm = table("stress_risk_measures", index_col=0)
    sc = st.selectbox("Scenario", el.index.tolist(), index=2)
    c = st.columns(4)
    c[0].metric("Economy", el.loc[sc, "economy"])
    c[1].metric("Expected lifetime loss", pct(el.loc[sc, "el_rate"], 2), f"×{el.loc[sc, 'vs_base']:.2f} base",
                delta_color="inverse")
    c[2].metric("Stressed 99% VaR", f"{rm.loc[sc, 'var_0.99']:.1f}%")
    c[3].metric("Stressed 99% ES", f"{rm.loc[sc, 'es_0.99']:.1f}%")
    comp = el.reset_index(names="scenario")
    st.altair_chart(alt.Chart(comp).mark_bar().encode(
        x=alt.X("scenario:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("el_rate:Q", title="Expected lifetime loss",
                                                              axis=alt.Axis(format="%")),
        color=alt.condition(alt.datum.scenario == sc, alt.value(ORANGE), alt.value(MUTED))), use_container_width=True)
    curve = table("stress_economy_curve")
    curve["one_in_n"] = 1 / curve["economy_quantile"]
    cl = curve.melt(id_vars=["one_in_n"], value_vars=[k for k in curve.columns if k not in ("economy_quantile", "one_in_n")],
                    var_name="assumptions", value_name="el_rate")
    st.subheader("Expected loss as the economy worsens")
    st.altair_chart(alt.Chart(cl).mark_line(point=True).encode(
        x=alt.X("one_in_n:Q", scale=alt.Scale(type="log"), title="Economy: one-in-N outcome over the loans' life"),
        y=alt.Y("el_rate:Q", title="Expected lifetime loss", axis=alt.Axis(format="%")),
        color=alt.Color("assumptions:N", scale=alt.Scale(range=[BLUE, ORANGE]), title=None,
                        legend=alt.Legend(orient="bottom", labelLimit=400))), use_container_width=True)
    st.subheader("Severe recession by rating grade")
    st.dataframe(table("stress_vulnerability_by_grade", index_col=0).style.format(
        {"exposure_share": "{:.1%}", "el_rate_base": "{:.2%}", "el_rate_stress": "{:.2%}", "multiple": "×{:.2f}",
         "share_of_increase": "{:.1%}"}), use_container_width=True)

# ----------------------------------------------------------------------------
with tabs[5]:
    strat = table("optimisation_strategies", index_col=0)
    w = table("optimisation_weights", index_col=0)
    name = st.selectbox("Strategy", strat.index.tolist(), index=3)
    s = strat.loc[name]
    c = st.columns(4)
    c[0].metric("Expected lifetime net return", pct(s["expected_return"], 2))
    c[1].metric("Realised on 2015 loans", pct(s["realised_return"], 2),
                f"{(s['realised_return'] - strat.loc['Whole book', 'realised_return']) * 100:+.2f} pts vs whole book")
    c[2].metric("Expected loss", pct(s["el_rate"], 2))
    c[3].metric("Segments held", int(s["segments_held"]))
    out = (strat.reset_index().rename(columns={"expected_return": "expected", "realised_return": "realised"})
           .melt(id_vars="strategy", value_vars=["expected", "realised"], var_name="series", value_name="return"))
    st.altair_chart(alt.Chart(out).mark_bar().encode(
        y=alt.Y("strategy:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)), yOffset="series:N",
        x=alt.X("return:Q", title="Lifetime net return per dollar", axis=alt.Axis(format="%")),
        color=alt.Color("series:N", scale=alt.Scale(range=[MUTED, BLUE]), title=None)), use_container_width=True)
    st.subheader(f"Allocation: {name}")
    seg = w[[name]].rename(columns={name: "weight"}).reset_index()
    seg[["grade", "lc_band"]] = seg["seg"].str.split(" / ", expand=True)
    st.altair_chart(alt.Chart(seg, height=300).mark_rect(stroke="white").encode(
        x=alt.X("lc_band:N", title="LendingClub grade", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("grade:O", title="Our rating grade (1 = best)"),
        color=alt.Color("weight:Q", scale=alt.Scale(domain=[0, 0.12], range=["#f4f7fb", BLUE]),
                        legend=alt.Legend(format="%", title="weight")),
        tooltip=["seg", alt.Tooltip("weight:Q", format=".1%")]), use_container_width=True)
    st.caption("Limits: at most 10% per segment and 35% per LendingClub band. Expected returns use only information "
               "available when the loans were issued; realised returns are what the 2015 loans actually paid.")
