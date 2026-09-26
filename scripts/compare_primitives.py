#!/usr/bin/env python
"""KRUN-011 gates, selection and report — generated only from result files (missing -> N/A, never invented).

    .venv/bin/python scripts/compare_primitives.py select      # validation gates + selection (configs/krun_one_v0_2.yaml)
    .venv/bin/python scripts/compare_primitives.py report      # after the final evaluation: PRIMITIVES_REPORT tables

select -> results/primitives_candidates/selection.{json,md}
report -> results/primitives_final.json (+ tables printed, pasted into results/PRIMITIVES_REPORT.md)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))
from compare_with_laya import get, headline  # noqa: E402

ROUNDS = {  # round -> (variants, candidate tag prefix, baseline tag suffix, extra gates / selection terms)
    "v02": (("M50_CE", "M40_CE", "M50_RPS"), "v02", "", False),
    "v03": (("A", "B"), "v03", "_p2", True),
}
ROUND = "v02"
VARIANTS, PREFIX, BSUF, UNSEEN = ROUNDS[ROUND]


def set_round(r: str) -> None:
    global ROUND, VARIANTS, PREFIX, BSUF, UNSEEN
    ROUND = r
    VARIANTS, PREFIX, BSUF, UNSEEN = ROUNDS[r]
SCORE, TARGET = "top1_top2_margin", "0.98"


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def prim(tag: str):
    for d in ("primitives_baseline", "primitives_candidates"):
        r = load(RES / d / f"eval_{tag}.json")
        if r:
            return r
    return None


def pm(rep, q, part, key="all", metric=None, cal="calibrated"):
    return get(rep, q, part, cal, key, metric)


def choice_val(tag_file: str):
    return headline(load(RES / tag_file), SCORE, TARGET, final=False)


def best_laya(q: str, part: str, metric: str, higher: bool, key: str = "in_domain"):
    vals = [pm(prim(t + BSUF), q, part, key, metric) for t in ("laya", "laya_default")]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return max(vals) if higher else min(vals)


def fmt(v, nd=4):
    return "N/A" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def gates(tag: str, choice_file: str, part: str = "val_eval") -> dict:
    v0c, cc = choice_val("eval_krun_one_B.json"), choice_val(choice_file)
    v0p, cp = prim("krun_one_v0" + BSUF), prim(tag)
    g = {}
    for k in ("intent_acc_kb_val", "intent_hard_acc", "tool_acc"):
        g[f"choice:{k} drop<=0.5pp"] = (cc.get(k) is not None and v0c.get(k) is not None
                                        and v0c[k] - cc[k] <= 0.005, cc.get(k), v0c.get(k))
    for k in ("kb_val_ood_auroc", "val_tools_ood_auroc"):
        g[f"choice:{k} drop<=0.01"] = (cc.get(k) is not None and v0c[k] - cc[k] <= 0.01, cc.get(k), v0c.get(k))
    for k in ("kb_val_ece", "val_hard_ece", "val_tools_ece"):
        g[f"choice:{k} rise<=0.005"] = (cc.get(k) is not None and cc[k] - v0c[k] <= 0.005, cc.get(k), v0c.get(k))
    for q, metric, higher in (("noul", "roc_auc", True), ("noul", "brier", False), ("noul", "ece", False),
                              ("score", "mae", False), ("score", "spearman", True), ("score", "ece", False),
                              ("score", "nll", False)):
        c = pm(cp, q, part, "in_domain", metric)
        refs = {"laya_best": best_laya(q, part, metric, higher), "krun_one_v0": pm(v0p, q, part, "in_domain", metric)}
        for name, r in refs.items():
            strict = metric in ("roc_auc", "brier", "mae", "spearman")
            ok = c is not None and r is not None and ((c > r) if higher else (c < r) if strict else (c <= r))
            g[f"{q}:{metric} vs {name}"] = (ok, c, r)
    if UNSEEN:  # v0.3: generalization to never-trained propositions
        c = pm(cp, "noul", part, "unseen_proposition", "roc_auc")
        refs = [pm(prim(t + BSUF), "noul", part, "unseen_proposition", "roc_auc") for t in ("laya", "laya_default", "krun_one_v0")]
        r = max(x for x in refs if x is not None)
        g["noul:unseen_proposition roc_auc >= best(Laya, v0)"] = (c is not None and c >= r, c, r)
    return g


def select() -> int:
    out, rows = {}, []
    for v in VARIANTS:
        tag = f"{PREFIX}_{v}"
        if not (RES / f"eval_{tag}.json").exists() or prim(tag) is None:
            out[v] = {"available": False}
            continue
        g = gates(tag, f"eval_{tag}.json")
        cc = choice_val(f"eval_{tag}.json")
        cp = prim(tag)
        v0c = choice_val("eval_krun_one_B.json")
        ch = (cc["intent_acc_kb_val"] + cc["tool_acc"]) / 2
        terms = [pm(cp, "noul", "val_eval", "in_domain", "roc_auc"), pm(cp, "score", "val_eval", "in_domain", "spearman"), ch]
        if UNSEEN:
            terms.append(pm(cp, "noul", "val_eval", "unseen_proposition", "roc_auc"))
        primary = sum(terms) / len(terms)
        regression = max(0.0, v0c["intent_acc_kb_val"] - cc["intent_acc_kb_val"], v0c["tool_acc"] - cc["tool_acc"],
                         v0c["intent_hard_acc"] - cc["intent_hard_acc"])
        out[v] = {"available": True, "eligible": all(x[0] for x in g.values()), "primary": primary,
                  "choice_mean": ch, "choice_regression": regression,
                  "gates": {k: {"pass": bool(a), "candidate": b, "reference": c} for k, (a, b, c) in g.items()}}
    elig = [v for v in VARIANTS if out[v].get("eligible")]
    chosen = None
    if elig:
        best = max(out[v]["primary"] for v in elig)
        near = [v for v in elig if best - out[v]["primary"] < 0.003]
        chosen = min(near, key=lambda v: (round(out[v]["choice_regression"], 6), 0 if v == VARIANTS[0] else 1))
    res = {"round": ROUND, "rule": f"configs/krun_one_{ROUND.replace('v0', 'v0_')}.yaml decision_rules", "variants": out, "eligible": elig, "selected": chosen}
    (RES / "primitives_candidates").mkdir(exist_ok=True)
    (RES / "primitives_candidates" / f"selection_{ROUND}.json").write_text(json.dumps(res, indent=2) + "\n")
    md = [f"# KRUN-011 — validation gates and selection ({ROUND})", "",
          "Generated by `scripts/compare_primitives.py select` from `results/eval_v02_*.json` (choice) and "
          "`results/primitives_*/eval_*.json` (noul/score, val_eval halves, calibrated). Rules: "
          f"`configs/krun_one_{ROUND.replace('v0', 'v0_')}.yaml` (registered before training).", ""]
    keys = next((list(o["gates"]) for o in out.values() if o.get("available")), [])
    md.append("| gate | " + " | ".join(VARIANTS) + " | reference |")
    md.append("|---|" + "---|" * (len(VARIANTS) + 1))
    for k in keys:
        cells = []
        ref = None
        for v in VARIANTS:
            gg = out[v].get("gates", {}).get(k)
            if gg is None:
                cells.append("N/A")
                continue
            ref = gg["reference"]
            cells.append(f"{'✅' if gg['pass'] else '❌'} {fmt(gg['candidate'])}")
        md.append(f"| {k} | " + " | ".join(cells) + f" | {fmt(ref)} |")
    md += ["", "| variant | eligible | primary (mean of noul AUROC, score Spearman, choice acc" + (", unseen noul AUROC" if UNSEEN else "") + ") | choice regression |",
           "|---|---|---|---|"]
    for v in VARIANTS:
        o = out[v]
        if o.get("available"):
            md.append(f"| {v} | {'yes' if o['eligible'] else 'no'} | {fmt(o['primary'])} | {o['choice_regression'] * 100:.2f} pp |")
        else:
            md.append(f"| {v} | not available | | |")
    md += ["", f"**Selected: {chosen or 'NONE — no variant passed every gate (MODEL_PRIMITIVES_READY=false)'}**", ""]
    (RES / "primitives_candidates" / f"selection_{ROUND}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


def report() -> int:
    dec = load(RES / "primitives_decisions.json")
    if not dec:
        raise SystemExit("results/primitives_decisions.json missing")
    tag = dec["selected_tag"]
    fin = {}
    for name, f in (("laya", "eval_laya.json"), ("laya_default", "eval_laya_default.json"),
                    ("krun_one_v0", "eval_krun_one_B.json"), ("candidate", f"eval_{tag}.json")):
        fin[name] = {"choice": headline(load(RES / f), SCORE, TARGET, final=True),
                     "prim": prim(name + BSUF if name != "candidate" else tag)}
    (RES / "primitives_final.json").write_text(json.dumps({"selected": tag, "models": {
        k: {"choice": v["choice"]} for k, v in fin.items()}}, indent=2) + "\n")
    rows = []

    def row(prim_name, suite, metric, fn, pct=False):
        vals = []
        for m in ("laya", "laya_default", "krun_one_v0", "candidate"):
            try:
                x = fn(fin[m])
            except Exception:  # noqa: BLE001
                x = None
            vals.append("N/A" if x is None else (f"{x * 100:.2f}%" if pct else f"{x:.4f}"))
        rows.append(f"| {prim_name} | {suite} | {metric} | " + " | ".join(vals) + " |")

    for k, lab in (("kb_test_intent_acc", "KrunBench test intents (B77+MASSIVE)"), ("bfcl_acc", "BFCL"),
                   ("holdout_tool_acc", "holdout_tools"), ("intent_acc_kb_val", "KrunBench val intents"),
                   ("tool_acc", "val_tools")):
        row("Choice", lab, "accuracy", lambda d, k=k: d["choice"][k], pct=True)
    for k, lab in (("kb_test_ood_auroc", "KrunBench test"), ("holdout_tool_ood_auroc", "holdout_tools"),
                   ("bfcl_irrelevance_auroc", "BFCL irrelevance")):
        row("Choice", lab, "OOD AUROC", lambda d, k=k: d["choice"][k])
    row("Choice", "KrunBench test", "ECE", lambda d: d["choice"]["kb_test_ece"])
    for key, lab in (("in_domain", "test in-domain (intents, tools, Civil, MHS)"), ("source:boolq", "BoolQ (external)"),
                     ("positive_vs_hard_negative", "positives vs hard negatives"),
                     ("positive_vs_cross_domain", "positives vs cross-domain negatives"),
                     ("unseen_proposition", "unseen propositions (never trained)")):
        for metric in ("roc_auc", "pr_auc", "accuracy@0.5", "nll", "brier", "ece"):
            if key != "in_domain" and metric in ("pr_auc", "nll"):
                continue
            row("Noul", lab, metric, lambda d, key=key, metric=metric: pm(d["prim"], "noul", "test", key, metric))
    for key, lab in (("in_domain", "test in-domain (HelpSteer2 + MHS)"), ("source:helpsteer2", "HelpSteer2"),
                     ("source:mhs", "MHS"), ("source:sst5", "SST-5 (external)")):
        for metric in ("mae", "rmse", "accuracy", "spearman", "kendall_tau_b", "nll", "brier", "ece", "rps",
                       "distance_weighted_error"):
            if key != "in_domain" and metric in ("rmse", "brier", "rps", "kendall_tau_b"):
                continue
            row("Score", lab, metric, lambda d, key=key, metric=metric: pm(d["prim"], "score", "test", key, metric))
    for q, metric in (("choice", "accuracy"), ("noul", "roc_auc"), ("score", "mae"), ("score", "spearman")):
        row("Mixed", f"MHS mixed requests ({q})", metric, lambda d, q=q, metric=metric: get(d["prim"], "mixed", "quality", q, metric))
    row("Mixed", "all questions correct", "rate", lambda d: get(d["prim"], "mixed", "all_questions_correct_rate"))
    for q in ("choice", "noul", "score"):
        row("Mixed", f"invariance alone vs mixed ({q})", "max |Δp|",
            lambda d, q=q: get(d["prim"], "mixed", "invariance_alone_vs_mixed", q, "max_abs_dp"))
    hdr = ["| Primitive | Dataset/Suite | Metric | Laya (1024/896) | Laya (512/192) | Krun One v0 | " + tag + " |",
           "|---|---|---|---:|---:|---:|---:|"]
    print("\n".join(hdr + rows))
    (RES / "primitives_final_table.md").write_text("\n".join(hdr + rows) + "\n")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 2:
        set_round(sys.argv[2])
    sys.exit({"select": select, "report": report}[sys.argv[1]]())
