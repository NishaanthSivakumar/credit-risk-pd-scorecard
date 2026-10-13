"""Structured validation assessment: ten dimensions, each with a RAG status and
the evidence behind it, and a final decision from fixed rules.

The rules are defined here, before any result is seen by them, so the decision
cannot be tuned to the outcome. Thresholds live in config.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

ORDER = {"GREEN": 0, "AMBER": 1, "RED": 2}


@dataclass
class Dimension:
    key: str
    name: str
    status: str
    evidence: str


def _worst(*statuses: str) -> str:
    return max(statuses, key=ORDER.get)


def _band(value: float, green: float, amber: float) -> str:
    """Lower is better: GREEN below `green`, AMBER below `amber`, else RED."""
    return "GREEN" if value < green else ("AMBER" if value < amber else "RED")


def _checks_status(checks) -> str:
    failed = [c for c in checks if not c.passed]
    if any(c.critical for c in failed):
        return "RED"
    return "AMBER" if failed else "GREEN"


def assess(ev: dict) -> list[Dimension]:
    """ev holds the evidence gathered by the pipeline (see pipeline._evidence)."""
    dq = [c for c in ev["checks"] if c.dimension == "data quality"]
    tg = [c for c in ev["checks"] if c.dimension == "target definition"]
    failed = lambda cs: "; ".join(f"failed: {c.name}" for c in cs if not c.passed)

    concept = ("RED" if ev["n_wrong_sign"] or ev["max_pvalue"] > C.MAX_PVALUE
               else "AMBER" if ev["max_vif"] > 5 or not ev["bins_monotonic"] else "GREEN")
    gini_status = ("GREEN" if ev["gini_oot"] >= C.GINI_GREEN
                   else "AMBER" if ev["gini_oot"] >= C.GINI_AMBER else "RED")
    cal_status = _worst(_band(abs(ev["cal_gap"]), C.CAL_GAP_GREEN, C.CAL_GAP_AMBER),
                        _band(ev["red_grades"], C.FAILED_GRADES_GREEN + 1, C.FAILED_GRADES_AMBER + 1))
    stab_status = _worst(_band(ev["score_psi"], C.PSI_GREEN, C.PSI_AMBER),
                         _band(ev["max_csi"], C.PSI_GREEN, C.PSI_AMBER))
    oot_status = _worst(_band(ev["gini_drop"], C.GINI_DROP_GREEN, C.GINI_DROP_AMBER),
                        "GREEN" if ev["rank_monotonic"] else "AMBER")
    d, lo = ev["challenger_gini_diff"], ev["challenger_gini_lower"]
    chal_status = ("RED" if d >= C.CHALLENGER_GINI_RED and lo > 0
                   else "AMBER" if d >= C.CHALLENGER_GINI_AMBER and lo > 0 else "GREEN")
    sens_status = _worst(_band(ev["alt_spec_gini_range"], C.ALT_SPEC_GINI_GREEN, C.ALT_SPEC_GINI_AMBER),
                         _band(ev["coef_pd_half_width"], C.COEF_PD_GREEN, C.COEF_PD_AMBER))

    return [
        Dimension("conceptual", "Conceptual soundness", concept,
                  f"{ev['n_wrong_sign']} coefficients with the wrong sign; largest p-value "
                  f"{ev['max_pvalue']:.3f}; largest VIF {ev['max_vif']:.2f}; monotonic WoE bins: "
                  f"{'yes' if ev['bins_monotonic'] else 'no'}"),
        Dimension("data", "Data quality", _checks_status(dq),
                  f"{sum(c.passed for c in dq)} of {len(dq)} checks passed. {failed(dq)}".strip()),
        Dimension("target", "Target definition", _checks_status(tg),
                  f"{sum(c.passed for c in tg)} of {len(tg)} checks passed. {failed(tg)}".strip()),
        Dimension("discrimination", "Model discrimination", gini_status,
                  f"Out-of-time Gini {ev['gini_oot']:.3f} (95% CI {ev['gini_oot_lower']:.3f}–"
                  f"{ev['gini_oot_upper']:.3f}); thresholds GREEN ≥ {C.GINI_GREEN}, AMBER ≥ {C.GINI_AMBER}"),
        Dimension("calibration", "Model calibration", cal_status,
                  f"Mean PD minus observed default rate {ev['cal_gap']:+.2%}; "
                  f"{ev['red_grades']} of {ev['n_grades']} grades fail the binomial test"),
        Dimension("stability", "Population stability", stab_status,
                  f"Score PSI {ev['score_psi']:.3f}; largest feature CSI {ev['max_csi']:.3f}"),
        Dimension("oot", "Out-of-time performance", oot_status,
                  f"Gini falls {ev['gini_drop']:.1%} from train to out-of-time; rank ordering "
                  f"{'monotonic' if ev['rank_monotonic'] else 'not monotonic'} across deciles"),
        Dimension("challenger", "Challenger comparison", chal_status,
                  f"Challenger Gini minus scorecard {d:+.3f} (95% CI {lo:+.3f} to "
                  f"{ev['challenger_gini_upper']:+.3f}); material at ≥ {C.CHALLENGER_GINI_AMBER}"),
        Dimension("sensitivity", "Sensitivity analysis", sens_status,
                  f"Out-of-time Gini range across alternative specifications "
                  f"{ev['alt_spec_gini_range']:.3f}; coefficient uncertainty moves mean PD by "
                  f"±{ev['coef_pd_half_width']:.1%}"),
        Dimension("limitations", "Model limitations", "AMBER" if ev["n_high_limitations"] else "GREEN",
                  f"{ev['n_limitations']} limitations in the register, {ev['n_high_limitations']} rated High"),
    ]


DECISION_RULES = [
    ("Reject", "RED in conceptual soundness, data quality or target definition",
     "A flaw in what the model is or what it learns from cannot be fixed by tuning it"),
    ("Redevelop", "RED in population stability, out-of-time performance, challenger comparison or "
     "sensitivity; or RED discrimination when a challenger on the same data is materially better",
     "The ranking itself is broken, unstable or clearly beatable, so a new model is needed"),
    ("Recalibrate", "RED in calibration, with none of the above",
     "The ranking is sound and only the PD level is wrong, which a recalibration fixes; the "
     "recalibrated model is then re-assessed"),
    ("Conditionally approve", "No RED, except discrimination when no challenger on the same data is "
     "materially better; or any AMBER in dimensions 1–9",
     "Fit for use with stated conditions and monitoring"),
    ("Approve", "GREEN in dimensions 1–9 (limitations may be AMBER if documented)",
     "No conditions beyond standard monitoring"),
]


def decide(dims: list[Dimension]) -> tuple[str, str]:
    s = {d.key: d.status for d in dims}
    red = [k for k, v in s.items() if v == "RED"]
    if any(k in red for k in ("conceptual", "data", "target")):
        return "Reject", "RED in " + ", ".join(k for k in red if k in ("conceptual", "data", "target"))
    redev = [k for k in ("stability", "oot", "challenger", "sensitivity") if k in red]
    if redev:
        return "Redevelop", "RED in " + ", ".join(redev)
    if "discrimination" in red and s["challenger"] != "GREEN":
        return "Redevelop", "RED discrimination and a materially better challenger"
    if "calibration" in red:
        return "Recalibrate", "RED calibration with a sound ranking"
    core = [d for d in dims if d.key != "limitations"]
    if "discrimination" in red:
        return ("Conditionally approve",
                "Discrimination below threshold, but no challenger on the same data is materially better")
    if any(d.status == "AMBER" for d in core):
        return "Conditionally approve", "AMBER in " + ", ".join(d.name.lower() for d in core if d.status == "AMBER")
    return "Approve", "All dimensions GREEN"


def dimensions_table(dims: list[Dimension]) -> pd.DataFrame:
    return pd.DataFrame([{"#": i + 1, "dimension": d.name, "status": d.status, "evidence": d.evidence}
                         for i, d in enumerate(dims)]).set_index("#")


def rules_table() -> pd.DataFrame:
    return pd.DataFrame(DECISION_RULES, columns=["decision", "criterion", "rationale"]).set_index("decision")


# ----------------------------------------------------------------------------
# Gathering the evidence
# ----------------------------------------------------------------------------
def read_register(path: Path) -> pd.DataFrame:
    """Rows of the markdown table in reports/limitations_register.md."""
    head = ["id", "limitation", "impact", "severity", "mitigation"]
    if not Path(path).exists():
        return pd.DataFrame(columns=head)
    rows = []
    for line in Path(path).read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and re.fullmatch(r"L\d+", cells[0]):
            rows.append(cells[:len(head)])
    return pd.DataFrame(rows, columns=head)


def bins_monotonic(binner, features) -> bool:
    """True if every numeric feature's bad rate moves in one direction across its bins."""
    ok = True
    for f in features:
        b = binner.bins[f]
        if b.kind != "numeric":
            continue
        r = b.table.loc[b.table["bin"] != "Missing", "bad_rate"].diff().dropna()
        ok &= bool((r >= -1e-12).all() or (r <= 1e-12).all())
    return ok


def gather_evidence(checks, coefs: pd.DataFrame, features, binner, metrics: pd.DataFrame,
                    csi: pd.Series, rank_oot: pd.DataFrame, ci: pd.DataFrame, paired: pd.DataFrame,
                    sens: dict, register: pd.DataFrame) -> dict:
    """Evidence shared by every model version (everything except calibration).
    Intervals come from the month-block bootstrap, the more conservative scheme."""
    gini_ci = ci.loc[("month", "scorecard", "gini")]
    diff = paired.loc[("month", "gini")]
    m = metrics
    return {
        "checks": checks,
        "n_wrong_sign": int((coefs.loc[features, "coef"] >= 0).sum()),
        "max_pvalue": float(coefs.loc[features, "p_value"].max()),
        "max_vif": float(coefs.loc[features, "vif"].max()),
        "bins_monotonic": bins_monotonic(binner, features),
        "gini_oot": float(m.loc["oot", "gini"]),
        "gini_oot_lower": gini_ci["lower"], "gini_oot_upper": gini_ci["upper"],
        "score_psi": float(m.loc["oot", "score_psi_vs_train"]), "max_csi": float(csi.max()),
        "gini_drop": float((m.loc["train", "gini"] - m.loc["oot", "gini"]) / m.loc["train", "gini"]),
        "rank_monotonic": bool(rank_oot["monotonic"].all()),
        "challenger_gini_diff": diff["difference"], "challenger_gini_lower": diff["lower"],
        "challenger_gini_upper": diff["upper"],
        "alt_spec_gini_range": float(np.ptp(sens["alternatives"]["gini_oot"])),
        "coef_pd_half_width": float(sens["coef_joint"].loc["mean_pd", "relative_half_width"]),
        "n_limitations": len(register), "n_high_limitations": int((register["severity"] == "High").sum()),
    }


def assess_versions(common: dict, y_oot, versions: dict, grade_oot) -> tuple[dict, dict]:
    """Assess each model version {name: out-of-time PDs}. Returns (dimensions, decisions)."""
    from .validation import binomial_test_by_grade

    y = np.asarray(y_oot)
    dims, decisions = {}, {}
    for name, p in versions.items():
        b = binomial_test_by_grade(y, p, grade_oot)
        ev = {**common, "cal_gap": float(np.mean(p) - y.mean()),
              "red_grades": int(b["result"].str.startswith("RED").sum()), "n_grades": len(b)}
        dims[name] = assess(ev)
        decisions[name] = decide(dims[name])
    return dims, decisions
