"""Krun One v1.1 — the v1 regression suite for a candidate, from eval dumps (offline; final_protected is never read).

    uv run python scripts/v11/regression_v11.py --models r1_s1 r1_s2 r2_s1 r2_s2 --out results/v11/regression

Baseline = the frozen v1 dumped on the SAME host with the same harness (results/v11/eval/v1). Metrics and paired
checks are the registered v1 ones (scripts/v1/kd1_evaluate.py, stage2_compare.block / preservation); limits come from
configs/v11/promotion_gates.yaml. Calibration (temperatures, thresholds) is refitted per model on the fit part only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S2 = _mod("stage2_compare", ROOT / "scripts/v1/stage2_compare.py")
C, K = S2.C, S2.K
EVAL = ROOT / "results/v11/eval"
G = yaml.safe_load((ROOT / "configs/v11/promotion_gates.yaml").read_text())["preservation"]


def block(name: str) -> dict:
    d = EVAL / name
    return S2.block(d, d / "regression_toolace_v0/signals.jsonl.gz")


def at(b, fam, cov, key):
    return C.at_cov(b, fam, cov, key)


def summary(b: dict) -> dict:
    m = b["metrics"]
    t = b.get("toolace", {})
    vt = b.get("val_tools_v2", {}).get("metrics", {})
    return {"choice_accuracy": float(np.mean([m["choice_accuracy_intent"] * m["n_intent"] + m["tool_accuracy"] * m["n_tool"]]) / (m["n_intent"] + m["n_tool"])),
            "intent_accuracy": m["choice_accuracy_intent"], "tool_accuracy": m["tool_accuracy"],
            "by_language": {k: v["choice_accuracy"] for k, v in m["by_language"].items()},
            "pt_br": m["by_language"]["PT-BR"]["choice_accuracy"], "cross_language": m["cross_language_accuracy"],
            "hard_negative": m["hard_negative_accuracy"], "near_duplicate": m.get("near_duplicate_accuracy"),
            "by_k": {str(k): v["accuracy"] for k, v in m["by_k"].items()}, "k128": {str(k): v for k, v in m["by_k"].items()}.get("128", {}).get("accuracy"),
            "k64": {str(k): v for k, v in m["by_k"].items()}.get("64", {}).get("accuracy"),
            "ece": m["ece"], "nll": m["nll_choice"], "ood_auroc": m["ood_auroc_margin"], "aurc": m["aurc"],
            "coverage_at_target": m["coverage_at_target"], "selective_accuracy_at_target": m["selective_accuracy_at_target"],
            "ood_fa_at_own_threshold": m["ood_false_acceptance"], "unseen_tool_fa_at_own_threshold": m.get("unseen_tool_false_acceptance"),
            "ood_fa_cov80": C.all_ood_fa(b, 0.8), "tool_ood_fa_cov70": at(b, "tool", 0.7, "ood_false_acceptance"),
            "tool_unseen_fa_cov70": at(b, "tool", 0.7, "unseen_tool_false_acceptance"),
            "permutation": m["permutation_consistency"], "score_mae": m["score_mae"], "score_spearman": m["score_spearman"],
            "noul_auroc": m["noul_auroc"], "multi_micro_f1": m["multi_micro_f1"],
            "toolace_holdout": t.get("holdout_tools", {}).get("full", {}).get("acc"), "toolace_val": t.get("val_tools", {}).get("full", {}).get("acc"),
            "toolace_holdout_small_k": t.get("holdout_tools", {}).get("small_k", {}).get("acc"),
            "val_tools_v2_tool_accuracy": vt.get("tool_accuracy"), "multi_unseen_tool_micro_f1": m.get("multi_unseen_tool_micro_f1"),
            "val_tools_v2_k128": {str(k): v for k, v in vt.get("by_k", {}).items()}.get("128", {}).get("accuracy"),
            "n_eval": m["n_eval"]}


def gates(A: dict, B: dict) -> dict:
    """Preservation gates of configs/v11/promotion_gates.yaml: candidate B against the same-host v1 baseline A."""
    a, b = summary(A), summary(B)
    nat = K.paired_delta(A["rows"], B["rows"], lambda r: r["file"] == "val_ml" and not r["id"].startswith("kd1val/mt/")
                         and not r["lang"].startswith("en"))
    pp = lambda k: (b[k] - a[k]) * 100 if a[k] is not None and b[k] is not None else None  # noqa: E731
    drop = lambda v, lim: None if v is None else bool(v >= -lim)  # noqa: E731
    rise = lambda v, lim: None if v is None else bool(v <= lim)  # noqa: E731
    g = {
        "krunbench_v1_choice_accuracy": (pp("choice_accuracy"), drop(pp("choice_accuracy"), G["krunbench_v1_choice_accuracy_drop_pp"])),
        "native_non_en": (nat["delta"] * 100, drop(nat["delta"] * 100, G["native_non_en_accuracy_drop_pp"])),
        "pt_br": (pp("pt_br"), drop(pp("pt_br"), G["pt_br_accuracy_drop_pp"])),
        "cross_language": (pp("cross_language"), drop(pp("cross_language"), G["cross_language_accuracy_drop_pp"])),
        "intent": (pp("intent_accuracy"), drop(pp("intent_accuracy"), G["intent_accuracy_drop_pp"])),
        "toolace_holdout": (pp("toolace_holdout"), drop(pp("toolace_holdout"), G["toolace_holdout_drop_pp"])),
        # V has no answerable unseen-tool items; "unseen tools" = ToolACE v0 holdout (above) + the in-distribution tool V below
        "tool_accuracy_val_tools_v2": (pp("val_tools_v2_tool_accuracy"), drop(pp("val_tools_v2_tool_accuracy"), G["unseen_tool_accuracy_drop_pp"])),
        "large_k_k128": (pp("k128"), drop(pp("k128"), G["large_k_k128_accuracy_drop_pp"])),
        "ood_auroc": (b["ood_auroc"] - a["ood_auroc"], drop(b["ood_auroc"] - a["ood_auroc"], G["ood_auroc_drop"])),
        "ood_fa_cov80": ((b["ood_fa_cov80"] - a["ood_fa_cov80"]) * 100, rise((b["ood_fa_cov80"] - a["ood_fa_cov80"]) * 100, G["ood_false_acceptance_at_cov80_rise_pp"])),
        "tool_unseen_fa_cov70": (b["tool_unseen_fa_cov70"], rise(b["tool_unseen_fa_cov70"] or 0, G["tool_unseen_fa_at_cov70_max"])),
        "selective_coverage": (pp("coverage_at_target"), drop(pp("coverage_at_target"), G["selective_coverage_at_target_drop_pp"])),
        "aurc": (b["aurc"] - a["aurc"], rise(b["aurc"] - a["aurc"], G["aurc_rise"])),
        "permutation": (pp("permutation"), drop(pp("permutation"), G["permutation_consistency_drop_pp"])),
        "score_mae": (b["score_mae"] - a["score_mae"], rise(b["score_mae"] - a["score_mae"], G["score_mae_rise"])),
        "score_spearman": (b["score_spearman"] - a["score_spearman"], drop(b["score_spearman"] - a["score_spearman"], G["score_spearman_drop"])),
        "noul_auroc": (b["noul_auroc"] - a["noul_auroc"], drop(b["noul_auroc"] - a["noul_auroc"], G["noul_auroc_drop"])),
        "multi_micro_f1": (pp("multi_micro_f1"), drop(pp("multi_micro_f1"), G["multi_micro_f1_drop_pp"])),
        "ece": (b["ece"] - a["ece"], rise(b["ece"] - a["ece"], G["ece_rise"])),
    }
    return {k: {"value": v, "pass": ok} for k, (v, ok) in g.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--baseline", default="v1")
    ap.add_argument("--out", default="results/v11/regression")
    args = ap.parse_args()
    A = block(args.baseline)
    res = {"baseline": args.baseline, "limits": G, "summary": {args.baseline: summary(A)}, "gates": {}, "latency": {}}
    for m in args.models:
        B = block(m)
        res["summary"][m] = summary(B)
        res["gates"][m] = gates(A, B)
        res["gates"][m]["ALL_PASS"] = all(v["pass"] is not False for v in res["gates"][m].values())
    for m in [args.baseline] + args.models:
        p = EVAL / m / "latency.json"
        if p.exists():
            d = json.loads(p.read_text())
            res["latency"][m] = {f"{r['dtype']}/{r['workload']}": {"p50": r["total_p50_ms"], "p95": r["total_p95_ms"], "vram_mib": r["peak_vram_mib"]}
                                 for r in d["results"]}
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "regression.json").write_text(json.dumps(res, indent=1, default=str))
    names = [args.baseline] + args.models
    keys = ["choice_accuracy", "intent_accuracy", "tool_accuracy", "pt_br", "cross_language", "hard_negative", "k128", "toolace_holdout",
            "toolace_val", "val_tools_v2_tool_accuracy", "val_tools_v2_k128", "multi_unseen_tool_micro_f1", "k64", "ood_auroc", "ood_fa_cov80", "tool_unseen_fa_cov70", "coverage_at_target",
            "aurc", "permutation", "score_mae", "score_spearman", "noul_auroc", "multi_micro_f1", "ece"]
    f = lambda v: "—" if v is None else (f"{v:.4f}" if abs(v) < 2 else f"{v:.2f}")  # noqa: E731
    md = ["# v1 regression suite (same host, same harness; V eval + ToolACE v0; no protected set opened)", "",
          "| metric | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for k in keys:
        md.append(f"| {k} | " + " | ".join(f(res["summary"][n][k]) for n in names) + " |")
    md += ["", "## Preservation gates vs the same-host v1 (value = candidate − v1; pp where the limit is in pp)", "",
           "| gate | " + " | ".join(args.models) + " |", "|---|" + "---|" * len(args.models)]
    for g in next(iter(res["gates"].values())):
        if g == "ALL_PASS":
            continue
        cells = []
        for m in args.models:
            x = res["gates"][m][g]
            cells.append("—" if x["value"] is None else f"{x['value']:+.3f} {'✅' if x['pass'] else '❌'}")
        md.append(f"| {g} | " + " | ".join(cells) + " |")
    md.append("| **all** | " + " | ".join("✅" if res["gates"][m]["ALL_PASS"] else "❌" for m in args.models) + " |")
    if res["latency"]:
        wl = sorted({k for v in res["latency"].values() for k in v})
        md += ["", "## Text latency (RTX 4090, total p50 / p95 ms; peak VRAM MiB)", "", "| workload | " + " | ".join(res["latency"]) + " |",
               "|---|" + "---|" * len(res["latency"])]
        for w in wl:
            md.append(f"| {w} | " + " | ".join(f"{v[w]['p50']:.1f} / {v[w]['p95']:.1f} ({v[w]['vram_mib']})" if w in v else "—"
                                               for v in res["latency"].values()) + " |")
    (out / "regression.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
