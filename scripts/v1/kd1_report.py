"""Krun One v1 candidate vs v0.3: metric tables + registered gates (configs/v1/ft_gates.yaml) -> markdown + json.

    python scripts/v1/kd1_report.py --eval results/v1/ft/eval_v1_ft.json --latency results/v1/ft/latency_rtx4090.json
    -> results/v1/ft/comparison_vs_v03.{md,json}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def f(x, pct=True, nd=1):
    if x is None:
        return "—"
    return f"{100 * x:.{nd}f}" if pct else f"{x:.3f}"


def latency_table(lat: dict) -> tuple[list[str], dict]:
    rows = {}
    for r in lat["results"]:
        if r["dtype"] != "bf16":
            continue
        k = (r["workload"], r["locale"])
        rows.setdefault(k, {})["v03" if r["spec"].startswith("krun-one-v0.3") else "v1"] = r["total_p50_ms"]
    wl = ["typical_1q_k4_ctx48", "typical_1q_tools8_ctx48", "typical_4q_mixed_ctx96", "k_32_tools_ctx64",
          "k_64_tools_ctx64", "k_128_tools_ctx64", "stress_16q_tools32_ctx512"]
    md = ["| workload (RTX 4090, bf16, total p50 ms) | locale | v0.3 | v1 (K1-W1536-L22) | ratio |", "|---|---|---:|---:|---:|"]
    worst = {"max_typical_ms": 0.0, "max_typical_ratio": 0.0}
    for w in wl:
        for loc in ("en-US", "pt-PT"):
            v = rows.get((w, loc))
            if not v or "v1" not in v:
                continue
            ratio = v["v1"] / v["v03"] if v.get("v03") else None
            md.append(f"| {w} | {loc} | {v.get('v03', 0):.1f} | {v['v1']:.1f} | {ratio:.2f}× |")
            if w.startswith("typical"):
                worst["max_typical_ms"] = max(worst["max_typical_ms"], v["v1"])
                worst["max_typical_ratio"] = max(worst["max_typical_ratio"], ratio or 0)
    return md, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=str(ROOT / "results/v1/ft/eval_v1_ft.json"))
    ap.add_argument("--latency", default=str(ROOT / "results/v1/ft/latency_rtx4090.json"))
    ap.add_argument("--out", default=str(ROOT / "results/v1/ft/comparison_vs_v03"))
    args = ap.parse_args()
    E = json.loads(Path(args.eval).read_text())
    G = yaml.safe_load((ROOT / "configs/v1/ft_gates.yaml").read_text())
    a, b, D = E["metrics"], E["baseline"]["metrics"], E["paired_deltas"]
    md = ["# Krun One v1 candidate vs Krun One v0.3 — KrunBench v1 V (eval part)\n",
          "Both models: same V decisions, same signal/metric code (`kd1_evaluate.py`); temperatures and the margin threshold "
          "fitted on the FIT part for each model separately. v0.3 has no multi-label output (n/a).\n",
          "| metric | v0.3 | v1 | Δ |", "|---|---:|---:|---:|"]
    rows = [("Choice (intent) accuracy", "choice_accuracy_intent", True), ("Tool accuracy", "tool_accuracy", True),
            ("Hard-negative accuracy", "hard_negative_accuracy", True), ("Near-duplicate accuracy", "near_duplicate_accuracy", True),
            ("Cross-language accuracy", "cross_language_accuracy", True), ("Unseen-tool false acceptance ↓", "unseen_tool_false_acceptance", True),
            ("OOD AUROC (margin)", "ood_auroc_margin", False), ("OOD false acceptance ↓", "ood_false_acceptance", True),
            ("AURC ↓", "aurc", False), ("ECE ↓ (choice, after T)", "ece", False), ("NLL ↓ (choice)", "nll_choice", False),
            ("Coverage @ sel. acc 0.95", "coverage_at_0.95", True), ("Coverage @ sel. acc 0.98", "coverage_at_target", True),
            ("Coverage @ sel. acc 0.99", "coverage_at_0.99", True), ("Permutation consistency", "permutation_consistency", True),
            ("Cross-language consistency", "cross_language_consistency", True), ("Noul ROC-AUC", "noul_auroc", False),
            ("Noul accuracy", "noul_accuracy", True), ("Score MAE ↓", "score_mae", False), ("Score Spearman", "score_spearman", False),
            ("Score NLL ↓", "score_nll", False), ("Multi micro-F1", "multi_micro_f1", True), ("Multi macro-F1", "multi_macro_f1", True),
            ("Multi empty-set accuracy", "multi_empty_set_accuracy", True), ("Multi unseen-tool micro-F1", "multi_unseen_tool_micro_f1", True)]
    for name, k, pct in rows:
        va, vb = b.get(k), a.get(k)
        d = None if va is None or vb is None else vb - va
        md.append(f"| {name} | {f(va, pct)} | {f(vb, pct)} | {'—' if d is None else (f'{100 * d:+.1f}' if pct else f'{d:+.3f}')} |")
    md += ["\n## By language (choice accuracy; OOD false acceptance)\n",
           "| language | n | v0.3 acc | v1 acc | v0.3 intent | v1 intent | v0.3 tool | v1 tool | v0.3 OOD FA | v1 OOD FA |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for L in ("EN", "PT-BR", "PT-PT", "ES", "FR"):
        x, y = b["by_language"][L], a["by_language"][L]
        md.append(f"| {L} | {y['n']} | {f(x['choice_accuracy'])} | {f(y['choice_accuracy'])} | {f(x['intent_accuracy'])} | "
                  f"{f(y['intent_accuracy'])} | {f(x['tool_accuracy'])} | {f(y['tool_accuracy'])} | "
                  f"{f(x['ood_false_acceptance'])} | {f(y['ood_false_acceptance'])} |")
    md += ["\n## Paired deltas (v1 − v0.3, accuracy on shared answerable choice decisions; group bootstrap 95% CI)\n",
           "| slice | n | Δ (pp) | 95% CI |", "|---|---:|---:|---|"]
    for k, v in D.items():
        if v:
            md.append(f"| {k} | {v['n']} | {100 * v['delta']:+.2f} | [{100 * v['ci95'][0]:+.2f}, {100 * v['ci95'][1]:+.2f}] |")
    md += ["\n## By candidate count (choice accuracy)\n", "| k ≤ | n | v0.3 | v1 |", "|---:|---:|---:|---:|"]
    for k in ("2", "4", "8", "16", "32", "64", "128"):
        x, y = b["by_k"].get(k) or b["by_k"].get(int(k)), a["by_k"].get(k) or a["by_k"].get(int(k))
        md.append(f"| {k} | {y['n']} | {f(x['accuracy'])} | {f(y['accuracy'])} |")
    md += ["\n## OOD false acceptance by bucket (at each model's own threshold)\n", "| bucket | n | v0.3 | v1 |", "|---|---:|---:|---:|"]
    for k, v in sorted(a["ood_false_acceptance_by_bucket"].items()):
        x = b["ood_false_acceptance_by_bucket"].get(k, {})
        md.append(f"| {k} | {v['n']} | {f(x.get('fa'))} | {f(v['fa'])} |")
    lat_md, worst = ([], None)
    if Path(args.latency).exists():
        lat_md, worst = latency_table(json.loads(Path(args.latency).read_text()))
        md += ["\n## Latency (model part of the worker; same harness for both)\n"] + lat_md
    # ---------------- gates
    gates = {}
    for k in G["must_gain"]:
        v = D.get(k)
        gates[f"gain:{k}"] = bool(v and v["ci95"][0] > 0)
    ut = G["unseen_tool"]
    fa_a, fa_b = a.get("unseen_tool_false_acceptance"), b.get("unseen_tool_false_acceptance")
    gates["unseen_tool:false_acceptance"] = fa_a is not None and fa_b is not None and (fa_a - fa_b) * 100 <= ut["false_acceptance_rise_max_pp"]
    gates["unseen_tool:multi_f1"] = (a.get("multi_unseen_tool_micro_f1") or 0) >= ut["multi_micro_f1_min"]
    R = G["no_material_regression"]
    gates["ood_auroc"] = a["ood_auroc_margin"] >= b["ood_auroc_margin"] - R["ood_auroc_margin_drop_max"]
    gates["ood_false_acceptance"] = (a["ood_false_acceptance"] - b["ood_false_acceptance"]) * 100 <= R["ood_false_acceptance_rise_max_pp"]
    gates["calibration_ece"] = a["ece"] <= b["ece"] + R["ece_rise_max"]
    en_a, en_b = a["by_language"]["EN"]["intent_accuracy"], b["by_language"]["EN"]["intent_accuracy"]
    gates["choice_core_en_intent"] = (en_b - en_a) * 100 <= R["en_intent_accuracy_drop_max_pp"]
    if worst:
        gates["latency_typical_ms"] = worst["max_typical_ms"] <= R["latency_typical_max_ms"]
        gates["latency_typical_ratio"] = worst["max_typical_ratio"] <= R["latency_typical_max_ratio_vs_v03"]
    verdict = "CANDIDATE_ACCEPTED" if all(gates.values()) else "CANDIDATE_REJECTED"
    md += ["\n## Registered gates (configs/v1/ft_gates.yaml)\n", "| gate | pass |", "|---|---|"]
    md += [f"| {k} | {'✅' if v else '❌'} |" for k, v in gates.items()]
    md += [f"\n**Verdict: {verdict}**" + ("" if verdict == "CANDIDATE_ACCEPTED" else
                                          f" (failing: {', '.join(k for k, v in gates.items() if not v)})")]
    Path(args.out + ".md").write_text("\n".join(md) + "\n")
    Path(args.out + ".json").write_text(json.dumps({"gates": gates, "verdict": verdict, "latency_worst_typical": worst}, indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
