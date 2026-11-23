"""Apply the registered 1B -> 3B gate (configs/v1/mlm_gate.yaml): MLM held-out loss + decision probe, 0B vs 1B.

    uv run python scripts/v1/probe_compare.py --mlm-run runs/v1/mlm/k1-continued-mlm-1b --probe-runs runs/v1/probe
    -> results/v1/mlm_gate_1b.{json,md}
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("bk", ROOT / "scripts/v1/answerability_bakeoff.py")
bk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bk)
GATE = yaml.safe_load((ROOT / "configs/v1/mlm_gate.yaml").read_text())
TH = GATE["thresholds"]
SEEDS = GATE["decision_probe"]["seeds"]
LANGS = ("en", "pt", "es", "fr")


def probe_metrics(run: Path) -> dict:
    rows = [r for r in bk.load(run) if r["type"] == "choice"]
    bad = [r["id"] for r in rows if "kbv1/final_protected" in r["id"] or r["id"].startswith("v1/bfcl") or "/test/" in r["id"]]
    if bad:   # the probe decides 0B vs 1B on train/val only; protected/final data may never be read here
        raise SystemExit(f"probe signals contain protected/final items (e.g. {bad[0]}); refusing to decide")
    fit = [r for r in rows if bk.is_fit(r["split_group_id"])]
    ev = [r for r in rows if not bk.is_fit(r["split_group_id"])]
    T = {f: bk.fit_temperature([r for r in fit if r["answerable"] and r["family"] == f]) for f in ("intent", "tool")}
    ff, fe = [bk.featurize(r, T[r["family"]]) for r in fit], [bk.featurize(r, T[r["family"]]) for r in ev]
    tau = {}
    for fam in ("intent", "tool"):
        idx = [i for i, r in enumerate(fit) if r["family"] == fam]
        tau[fam] = bk.threshold(bk.native_scores(ff, "margin")[idx], np.array([ff[i]["correct"] for i in idx], dtype=float))
    sig = bk.evaluate_signal(ev, fe, bk.native_scores(fe, "margin"), tau)
    mm = bk.model_metrics(ev, fe)

    def acc(pred):
        v = [f["correct"] for r, f in zip(ev, fe) if r["answerable"] and pred(r)]
        return float(np.mean(v)) if v else None
    out = {"accuracy_intent": mm["accuracy_intent"], "accuracy_tool": mm["accuracy_tool"], "ece": mm["ece"], "nll": mm["nll"],
           "permutation_consistency": mm["permutation_consistency"], "macro_fa": sig["macro_fa"], "auroc": sig["auroc"],
           "aurc": sig["aurc"], "coverage": sig["coverage"], "false_acceptance": sig["false_acceptance"],
           "acc_cross_language": acc(lambda r: "cross_language" in r["slices"]),
           "acc_unseen_tool": acc(lambda r: "unseen_tool" in r["slices"]),
           "acc_hard": acc(lambda r: r.get("difficulty") == "hard")}
    for lang in LANGS:
        out[f"acc_{lang}"] = acc(lambda r, lang=lang: r["lang"].split("-")[0] == lang)
    e = json.loads((run / "eval_validation.json").read_text())
    out.update({k: e.get(k) for k in ("noul_auroc", "score_mae", "score_spearman", "multi_micro_f1")})
    parts = [out["accuracy_intent"], out["accuracy_tool"], out["noul_auroc"], out["score_spearman"], out["multi_micro_f1"],
             out["acc_cross_language"], out["acc_unseen_tool"], out["acc_hard"]]
    out["aggregate_score"] = 100 * float(np.mean([p for p in parts if p is not None]))
    return out


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlm-run", default=str(ROOT / "runs/v1/mlm/k1-continued-mlm-1b"))
    ap.add_argument("--probe-runs", default=str(ROOT / "runs/v1/probe"))
    ap.add_argument("--tag", default="1b")
    args = ap.parse_args()
    mlm = Path(args.mlm_run)
    l0 = json.loads((mlm / "milestone-0/eval.json").read_text())["heldout_mlm_loss"]
    l1 = json.loads((mlm / "milestone-1000000000/eval.json").read_text())["heldout_mlm_loss"]
    probes = {init: {s: probe_metrics(Path(args.probe_runs) / f"{init}_s{s}") for s in SEEDS} for init in ("k1_0b", "k1_1b")}
    keys = [k for k in probes["k1_0b"][SEEDS[0]] if k != "false_acceptance"]
    m0 = {k: mean([probes["k1_0b"][s][k] for s in SEEDS]) for k in keys}
    m1 = {k: mean([probes["k1_1b"][s][k] for s in SEEDS]) for k in keys}
    rel = {k: (l0[k] - l1[k]) / l0[k] for k in LANGS}
    g1 = all(rel[k] >= TH["mlm_rel_gain_min"] for k in LANGS)
    per_seed = [probes["k1_1b"][s]["aggregate_score"] - probes["k1_0b"][s]["aggregate_score"] for s in SEEDS]
    g2 = (m1["aggregate_score"] - m0["aggregate_score"]) >= TH["aggregate_gain_min"] and all(d > 0 for d in per_seed)
    g3 = all(m1[f"acc_{k}"] >= m0[f"acc_{k}"] - TH["lang_drop_max_pp"] / 100 for k in LANGS if m0.get(f"acc_{k}") is not None)
    g4 = (m1["ece"] <= m0["ece"] + TH["ece_rise_max"] and m1["macro_fa"] <= m0["macro_fa"] + TH["fa_rise_max_pp"] / 100
          and m1["auroc"] >= m0["auroc"] - TH["auroc_drop_max"])
    g5 = (m1["accuracy_tool"] >= m0["accuracy_tool"] - TH["tool_drop_max_pp"] / 100
          and m1["acc_cross_language"] >= m0["acc_cross_language"] - TH["crosslang_drop_max_pp"] / 100
          and (m0["permutation_consistency"] is None or m1["permutation_consistency"] >= m0["permutation_consistency"] - TH["permutation_drop_max_pp"] / 100))
    gates = {"G1_mlm": g1, "G2_decision": g2, "G3_multilingual": g3, "G4_calibration_ood": g4, "G5_tool_crosslang_permutation": g5}
    decision = "CONTINUE_TO_3B" if all(gates.values()) else "STOP_AT_1B"
    res = {"mlm_loss": {"0B": l0, "1B": l1, "rel_gain": rel,
                        "perplexity": {"0B": {k: math.exp(v) for k, v in l0.items()}, "1B": {k: math.exp(v) for k, v in l1.items()}}},
           "probe": {"0B": m0, "1B": m1, "per_seed": probes, "aggregate_delta_per_seed": per_seed},
           "gates": gates, "decision": decision,
           "stage2_init": "1B" if g2 else ("0B" if m0["aggregate_score"] >= m1["aggregate_score"] else "1B")}
    out = ROOT / "results/v1"
    (out / f"mlm_gate_{args.tag}.json").write_text(json.dumps(res, indent=2, default=str))
    f = lambda x, pct=True: "—" if x is None else (f"{100 * x:.1f}" if pct else f"{x:.3f}")  # noqa: E731
    md = ["# Continued MLM gate: 0B vs 1B (rule: configs/v1/mlm_gate.yaml, registered before training)\n",
          "| language | MLM loss 0B | MLM loss 1B | relative gain | perplexity 0B → 1B |", "|---|---|---|---|---|"]
    for k in LANGS:
        md.append(f"| {k} | {l0[k]:.3f} | {l1[k]:.3f} | {100 * rel[k]:+.1f}% | {math.exp(l0[k]):.1f} → {math.exp(l1[k]):.1f} |")
    md += ["", "| decision probe (val_eval, mean of 2 seeds) | 0B | 1B | Δ |", "|---|---|---|---|"]
    for k, lab, pct in (("aggregate_score", "aggregate score", False), ("accuracy_intent", "choice intent acc", True),
                        ("accuracy_tool", "tool acc", True), ("noul_auroc", "noul ROC-AUC", False),
                        ("score_spearman", "score Spearman", False), ("score_mae", "score MAE", False),
                        ("multi_micro_f1", "multi micro-F1", True), ("acc_cross_language", "cross-language acc", True),
                        ("acc_unseen_tool", "unseen-tool acc", True), ("acc_hard", "hard-negative acc", True),
                        ("acc_en", "EN acc", True), ("acc_pt", "PT acc", True), ("acc_es", "ES acc", True),
                        ("acc_fr", "FR acc", True), ("auroc", "OOD AUROC (margin)", False),
                        ("macro_fa", "macro OOD FA @0.98", True), ("coverage", "coverage @0.98", True),
                        ("ece", "ECE", False), ("nll", "NLL", False), ("permutation_consistency", "permutation consistency", True)):
        a, b = m0.get(k), m1.get(k)
        d = None if a is None or b is None else b - a
        if k == "aggregate_score":
            md.append(f"| **{lab}** | {a:.2f} | {b:.2f} | {d:+.2f} |")
        else:
            md.append(f"| {lab} | {f(a, pct)} | {f(b, pct)} | {'—' if d is None else (f'{100 * d:+.1f}' if pct else f'{d:+.3f}')} |")
    md += ["", "| gate | pass |", "|---|---|"] + [f"| {k} | {'✅' if v else '❌'} |" for k, v in gates.items()]
    md += ["", f"**Decision: {decision}** (aggregate Δ per seed: {', '.join(f'{d:+.2f}' for d in per_seed)})"]
    (out / f"mlm_gate_{args.tag}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
