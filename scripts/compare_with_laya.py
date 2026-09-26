#!/usr/bin/env python
"""Laya vs Krun One report, generated only from result files (missing numbers -> N/A, never invented).

    .venv/bin/python scripts/compare_with_laya.py --laya eval_laya --krun eval_krun_one_A [--final]

Writes results/laya_vs_krun_one.md and results/laya_vs_krun_one.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def load(name: str) -> dict | None:
    p = RES / (name if name.endswith(".json") else f"{name}.json")
    return json.loads(p.read_text()) if p.exists() else None


def get(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def weighted(by_ds: dict, keep) -> float | None:
    rows = [(v["n"], v["accuracy"]) for k, v in (by_ds or {}).items() if keep(k) and v.get("n")]
    n = sum(r[0] for r in rows)
    return sum(a * b for a, b in rows) / n if n else None


def headline(rep: dict | None, score: str, target: str, final: bool) -> dict:
    """Metrics of one evaluation report (val_eval halves / final sets)."""
    if rep is None:
        return {}
    s = rep["sets"]
    out = {}
    kv, vt, vh = s.get("kb_val"), s.get("val_tools"), s.get("val_hard")
    out["intent_acc_kb_val"] = weighted(get(kv, "quality", "by_dataset"), lambda k: k != "hwu64")
    out["intent_macro_f1_kb_val"] = get(kv, "quality", "core", "macro_f1")
    out["hwu64_acc_kb_val"] = get(kv, "quality", "by_dataset", "hwu64", "accuracy")
    for lang in ("en-US", "pt-PT", "es-ES"):
        out[f"massive_{lang}_acc"] = get(kv, "quality", "by_dataset", f"massive/{lang}", "accuracy")
    out["banking77_acc"] = get(kv, "quality", "by_dataset", "banking77", "accuracy")
    out["intent_hard_acc"] = get(vh, "quality", "core", "accuracy")
    for k in ("4", "8", "16", "32"):
        out[f"intent_hard_k{k}_acc"] = get(vh, "quality", "by_option_count", k, "accuracy")
    out["tool_acc"] = get(vt, "quality", "core", "accuracy")
    out["tool_macro_f1"] = get(vt, "quality", "core", "macro_f1")
    for sl in ("seen_tool", "unseen_tool", "near_duplicate_distractor", "description_dependent", "toolset_large(k>=16)"):
        out[f"tool_{sl}_acc"] = get(vt, "quality", "tool_slices", sl, "accuracy")
    for name, r in (("kb_val", kv), ("val_hard", vh), ("val_tools", vt)):
        out[f"{name}_ece"] = get(r, "quality", "core", "ece")
        out[f"{name}_nll"] = get(r, "quality", "core", "nll")
        out[f"{name}_brier"] = get(r, "quality", "core", "brier")
        out[f"{name}_ece_raw"] = get(r, "quality_raw", "core", "ece")
        out[f"{name}_aurc"] = get(r, "abstention", "all", score, "aurc")
        out[f"{name}_ood_auroc"] = get(r, "abstention", "all", score, "ood_auroc")
        op = get(r, "abstention", "all", score, "operating_points", target) or {}
        out[f"{name}_far"] = op.get("ood_false_acceptance")
        out[f"{name}_coverage"] = op.get("coverage")
        out[f"{name}_selective_acc"] = op.get("selective_accuracy")
        out[f"{name}_perm_consistency"] = get(r, "permutation", "consistency")
    if final:
        ht, kt = s.get("holdout_tools"), s.get("kb_test")
        out["holdout_tool_acc"] = get(ht, "quality", "core", "accuracy")
        out["holdout_tool_ood_auroc"] = get(ht, "abstention", "all", score, "ood_auroc")
        out["holdout_tool_far"] = get(ht, "abstention", "all", score, "operating_points", target, "ood_false_acceptance")
        out["kb_test_intent_acc"] = weighted(get(kt, "quality", "by_dataset"), lambda k: k not in ("hwu64", "bfcl"))
        out["bfcl_acc"] = get(kt, "quality", "by_dataset", "bfcl", "accuracy")
        for sl in ("near_duplicate_distractor", "description_dependent", "toolset_large(k>=16)"):
            out[f"bfcl_{sl}_acc"] = None  # BFCL slices are in kb_test tool_slices (tool family = BFCL only)
            out[f"bfcl_{sl}_acc"] = get(kt, "quality", "tool_slices", sl, "accuracy")
        out["kb_test_ece"] = get(kt, "quality", "core", "ece")
        out["kb_test_ood_auroc"] = get(kt, "abstention", "all", score, "ood_auroc")
        out["kb_test_far"] = get(kt, "abstention", "all", score, "operating_points", target, "ood_false_acceptance")
        out["bfcl_irrelevance_auroc"] = get(kt, "abstention", "tool", score, "ood_auroc")
    return out


def fmt(v, pct=False):
    if v is None:
        return "N/A"
    if isinstance(v, str):
        return v
    return f"{100 * v:.2f}%" if pct else (f"{v:.4f}" if abs(v) < 100 else f"{v:,.0f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--laya", default="eval_laya")
    ap.add_argument("--laya-default", default="eval_laya_default")
    ap.add_argument("--krun", required=True)
    ap.add_argument("--krun-label", default="Krun One v0")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--bench-laya", default="bench_cuda_laya")
    ap.add_argument("--bench-krun", default="bench_cuda_krun_one_v0")
    ap.add_argument("--cpu-laya", default="bench_cpu_laya")
    ap.add_argument("--cpu-krun", default="bench_cpu_krun_one_v0")
    args = ap.parse_args()
    dec = load("krun_one_decisions") or load("krun_one_decisions_v0") or {}
    score, target = dec.get("abstention_score", "top1_probability"), dec.get("operating_point_target", "0.98")
    L, LD, K = (headline(load(x), score, target, args.final) for x in (args.laya, args.laya_default, args.krun))
    bl, bk, cl, ck = (load(x) for x in (args.bench_laya, args.bench_krun, args.cpu_laya, args.cpu_krun))

    def b(bench, bs, key):
        return get(bench, "batches", str(bs), key)

    rows = [
        ("Intent accuracy (KrunBench val: Banking77 + MASSIVE en/pt/es)", "intent_acc_kb_val", True),
        ("Intent hard accuracy (semantic hard negatives, k=4-32)", "intent_hard_acc", True),
        ("Tool-routing accuracy (val_tools)", "tool_acc", True),
        ("OOD AUROC (KrunBench val intents)", "kb_val_ood_auroc", False),
        ("OOD AUROC (val_tools)", "val_tools_ood_auroc", False),
        (f"False acceptance @ target {target} (KrunBench val)", "kb_val_far", True),
        (f"False acceptance @ target {target} (val_tools)", "val_tools_far", True),
        (f"Coverage @ target {target} (KrunBench val)", "kb_val_coverage", True),
        (f"Coverage @ target {target} (val_tools)", "val_tools_coverage", True),
        (f"Selective accuracy @ target {target} (KrunBench val)", "kb_val_selective_acc", True),
        ("ECE, T-scaled (KrunBench val)", "kb_val_ece", False),
        ("ECE, T-scaled (val_tools)", "val_tools_ece", False),
        ("AURC (KrunBench val)", "kb_val_aurc", False),
        ("AURC (val_tools)", "val_tools_aurc", False),
        ("Permutation consistency (KrunBench val)", "kb_val_perm_consistency", True),
        ("Permutation consistency (val_tools)", "val_tools_perm_consistency", True),
    ]
    if args.final:
        rows += [
            ("FINAL KrunBench test intent accuracy (Banking77 + MASSIVE)", "kb_test_intent_acc", True),
            ("FINAL BFCL accuracy (KrunBench test, never trained on)", "bfcl_acc", True),
            ("FINAL BFCL irrelevance OOD AUROC", "bfcl_irrelevance_auroc", False),
            ("FINAL holdout_tools accuracy", "holdout_tool_acc", True),
            ("FINAL holdout_tools OOD AUROC", "holdout_tool_ood_auroc", False),
            ("FINAL KrunBench test OOD AUROC (all)", "kb_test_ood_auroc", False),
            ("FINAL KrunBench test ECE", "kb_test_ece", False),
        ]
    lines = [f"# Laya vs {args.krun_label}", "",
             f"Protocol: identical decisions, options, serialization (Laya `build_sequence`, max_len 1024 / head 896) and metrics; "
             f"temperatures (per family) and thresholds fitted on val_fit, numbers on val_eval. Abstention score: `{score}`, "
             f"operating point: target selective accuracy {target}. Laya = upstream checkpoint, no training. "
             f"`Laya (512/192)` = Laya with its shipped budgets.", "",
             f"| Metric | Laya | Laya (512/192) | {args.krun_label} | Δ (Krun − Laya) |", "|---|---:|---:|---:|---:|"]
    for label, key, pct in rows:
        lv, kv = L.get(key), K.get(key)
        d = (kv - lv) if (lv is not None and kv is not None) else None
        lines.append(f"| {label} | {fmt(lv, pct)} | {fmt(LD.get(key), pct)} | {fmt(kv, pct)} | "
                     f"{('%+.2f pp' % (100 * d)) if (d is not None and pct) else fmt(d)} |")
    lines += [f"| CPU p50, batch 1 (ms) | {fmt(b(cl, 1, 'p50_ms'))} | — | {fmt(b(ck, 1, 'p50_ms'))} | |",
              f"| GPU p50, batch 1 (ms) | {fmt(b(bl, 1, 'p50_ms'))} | — | {fmt(b(bk, 1, 'p50_ms'))} | |",
              f"| Parameters | {fmt(get(bl, 'parameters'))} | — | {fmt(get(bk, 'parameters'))} | |",
              f"| RAM (CPU RSS, MiB) | {fmt(b(cl, 1, 'rss_mib'))} | — | {fmt(b(ck, 1, 'rss_mib'))} | |",
              f"| VRAM peak, batch 1 / 32 (MiB) | {fmt(b(bl, 1, 'peak_vram_allocated_mib'))} / {fmt(b(bl, 32, 'peak_vram_allocated_mib'))} | — | "
              f"{fmt(b(bk, 1, 'peak_vram_allocated_mib'))} / {fmt(b(bk, 32, 'peak_vram_allocated_mib'))} | |", ""]
    lines += ["## All headline metrics", "", f"| key | Laya | Laya (512/192) | {args.krun_label} |", "|---|---:|---:|---:|"]
    for key in K:
        lines.append(f"| {key} | {fmt(L.get(key))} | {fmt(LD.get(key))} | {fmt(K.get(key))} |")
    (RES / "laya_vs_krun_one.md").write_text("\n".join(lines) + "\n")
    (RES / "laya_vs_krun_one.json").write_text(json.dumps({"score": score, "target": target, "laya": L, "laya_default": LD,
                                                           "krun_one": K}, indent=2) + "\n")
    print("\n".join(lines[:22]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
