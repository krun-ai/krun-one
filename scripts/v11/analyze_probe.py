"""Krun One v1.1 R0 — metrics of a probe signal dump (offline, no GPU).

    uv run python scripts/v11/analyze_probe.py results/v11/capacity_probe/v1_zero_shot/signals.jsonl.gz

Writes metrics.json + metrics.md next to the dump. Temperature / abstention threshold = the frozen v1 ``tool``
family calibration (read-only), so "abstention" is what production would do with the same logits.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from krun_one.v11 import harness, probe  # noqa: E402


def kind(p: dict) -> str:
    if "pair_kind" in p:
        return "pair-" + p["pair_kind"]
    if p.get("flip"):
        return "flip"
    return "multiway" if p["answerable"] else "unanswerable"


def analyze(signals: Path, probe_path: Path, calib: dict) -> dict:
    problems = {p["id"]: p for p in harness.load_problems(probe_path)}
    T, thr = calib["temperature"]["tool"], calib["margin_threshold"]["tool"]
    rows = defaultdict(dict)
    with gzip.open(signals, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in problems:
                rows[r["cond"]][r["id"]] = probe.item_signals(problems[r["id"]], r["z"], T, thr)
    res: dict = {"signals": str(signals), "temperature_tool": T, "margin_threshold_tool": thr, "conditions": {}}
    for cond, by_id in rows.items():
        sel = lambda pred: [r for i, r in by_id.items() if pred(problems[i])]  # noqa: E731
        mw = sel(lambda p: kind(p) == "multiway")
        c = {"multiway": probe.group(mw, problems, ("slice", "k")),
             "multiway_track": probe.group(mw, problems, ("track", "k")),
             "multiway_difficulty": probe.group([r for r in mw if r["k"] == 8], problems, ("slice", "difficulty")),
             "pair_hardest": probe.group(sel(lambda p: kind(p) == "pair-hardest"), problems, ("slice",)),
             "pair_random": probe.group(sel(lambda p: kind(p) == "pair-random"), problems, ("slice",)),
             "flip": probe.flip_metrics(by_id, problems),
             "ood": probe.ood_metrics(list(by_id.values()), problems)}
        tech = [r for r in mw if problems[r["id"]]["slice"] == "techsel" and r["k"] == 8]
        c["techsel_k8"] = {
            "by_category": probe.group(tech, {i: {**p, "category": p["meta"]["category"]} for i, p in problems.items()
                                              if p["slice"] == "techsel"}, ("category",)),
            "ltrain_seen": probe.aggregate([r for r in tech if problems[r["id"]]["meta"]["ltrain_seen"]]),
            "ltrain_unseen": probe.aggregate([r for r in tech if not problems[r["id"]]["meta"]["ltrain_seen"]]),
            "unique_satisfier": probe.aggregate([r for r in tech if problems[r["id"]]["meta"]["n_satisfiers"] == 1]),
            "tie_break_needed": probe.aggregate([r for r in tech if problems[r["id"]]["meta"]["n_satisfiers"] > 1])}
        c["overall_k8"] = probe.aggregate([r for r in mw if r["k"] == 8])
        res["conditions"][cond] = c
    return res


def md(res: dict, title: str) -> str:
    out = [f"# {title}", "", f"signals: `{res['signals']}` · tool temperature {res['temperature_tool']:.3f} · "
           f"abstention threshold (margin) {res['margin_threshold_tool']:.4f}", ""]
    conds = list(res["conditions"])
    out += ["## Multi-way (answerable): top-1 / skill over chance", "",
            "| slice / k | n | chance | " + " | ".join(f"{c} top-1 | {c} skill" for c in conds) + " |",
            "|---|---:|---:|" + "---:|---:|" * len(conds)]
    cells = sorted({k for c in conds for k in res["conditions"][c]["multiway"]}, key=lambda s: (s.split("/")[0], float(s.split("/")[1])))
    for cell in cells:
        a = [res["conditions"][c]["multiway"].get(cell, {}) for c in conds]
        out.append(f"| {cell} | {a[0].get('n', 0)} | {a[0].get('chance', 0):.3f} | "
                   + " | ".join(f"{x.get('top1', float('nan')):.3f} | {x.get('skill', float('nan')):+.3f}" for x in a) + " |")
    for c in conds:
        out += ["", f"## {c}: ranking and distribution quality (multi-way)", "",
                "| slice / k | top-1 [95% CI] | MRR | NDCG | pairs ordered | gold > hardest | gold p | x uniform | top-1 p | "
                "entropy | gold-hardest logit | ECE | abstains | selective acc |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for cell in cells:
            x = res["conditions"][c]["multiway"].get(cell)
            if not x or not x.get("n"):
                continue
            sa = "—" if x["selective_acc"] is None else f"{x['selective_acc']:.3f}"
            out.append(f"| {cell} | {x['top1']:.3f} [{x['top1_ci95'][0]:.3f}, {x['top1_ci95'][1]:.3f}] | {x['mrr']:.3f} | "
                       f"{x['ndcg']:.3f} | {x['pair_all']:.3f} | {x['gold_vs_hardest']:.3f} | {x['gold_p']:.3f} | "
                       f"{x['gold_p_over_uniform']:.2f} | {x['top1_p']:.3f} | {x['entropy']:.3f} | {x['margin_logit']:+.2f} | "
                       f"{x['ece_T']:.3f} | {x['abstain_rate']:.3f} | {sa} |")
    out += ["", "## Pairwise (2 candidates, from the k=8 problems): accuracy", "",
            "| slice | n | " + " | ".join(f"{c} vs hardest | {c} vs random" for c in conds) + " |",
            "|---|---:|" + "---:|---:|" * len(conds)]
    for sl in sorted(res["conditions"][conds[0]]["pair_hardest"]):
        a = [(res["conditions"][c]["pair_hardest"][sl], res["conditions"][c]["pair_random"][sl]) for c in conds]
        out.append(f"| {sl} | {a[0][0]['n']} | " + " | ".join(f"{h['top1']:.3f} | {r['top1']:.3f}" for h, r in a) + " |")
    out += ["", "## Easy vs hard (k=8): top-1", "", "| slice / difficulty | n | " + " | ".join(conds) + " |",
            "|---|---:|" + "---:|" * len(conds)]
    for cell in sorted(res["conditions"][conds[0]]["multiway_difficulty"]):
        a = [res["conditions"][c]["multiway_difficulty"][cell] for c in conds]
        out.append(f"| {cell} | {a[0]['n']} | " + " | ".join(f"{x['top1']:.3f}" for x in a) + " |")
    out += ["", "## Constraint flips (same candidates; the flip adds / tightens ONE constraint)", "",
            "| slice | cond | pairs | base acc | flip acc | both | stuck on base answer | flip sensitivity |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for c in conds:
        for sl, x in res["conditions"][c]["flip"].items():
            out.append(f"| {sl} | {c} | {x['pairs']} | {x['base_acc']:.3f} | {x['flip_acc']:.3f} | {x['both_correct']:.3f} | "
                       f"{x['stuck_on_base_answer']:.3f} | {x['flip_sensitivity']:.3f} |")
    out += ["", "## None-of-the-above (k=8): abstention at the production threshold", "",
            "| slice | cond | n | false acceptance | answerable coverage | margin AUROC |", "|---|---|---:|---:|---:|---:|"]
    for c in conds:
        for sl, x in res["conditions"][c]["ood"].items():
            au = "—" if x["auroc_margin"] is None else f"{x['auroc_margin']:.3f}"
            out.append(f"| {sl} | {c} | {x['n_unanswerable']} | {x['false_acceptance']:.3f} | {x['answerable_coverage']:.3f} | {au} |")
    out += ["", "## Tech selection detail (k=8): top-1", "", "| subset | n | " + " | ".join(conds) + " |", "|---|---:|" + "---:|" * len(conds)]
    t0 = res["conditions"][conds[0]]["techsel_k8"]
    names = [("by_category", k) for k in t0["by_category"]] + [(k, None) for k in ("ltrain_seen", "ltrain_unseen", "unique_satisfier", "tie_break_needed")]
    for a, b in names:
        xs = [res["conditions"][c]["techsel_k8"][a] if b is None else res["conditions"][c]["techsel_k8"][a][b] for c in conds]
        if xs[0].get("n"):
            out.append(f"| {b or a} | {xs[0]['n']} | " + " | ".join(f"{x['top1']:.3f}" for x in xs) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("signals")
    ap.add_argument("--probe", default=str(ROOT / "data/v11/probe/probe.jsonl"))
    ap.add_argument("--calibration", default=str(ROOT / "artifacts/krun-one-v1-text-final/calibration.json"))
    ap.add_argument("--title", default="Technical Capacity Probe")
    args = ap.parse_args()
    sig = Path(args.signals)
    res = analyze(sig, Path(args.probe), json.loads(Path(args.calibration).read_text()))
    (sig.parent / "metrics.json").write_text(json.dumps(res, indent=1, default=lambda o: None if isinstance(o, float) and np.isnan(o) else o))
    (sig.parent / "metrics.md").write_text(md(res, args.title))
    print(md(res, args.title))


if __name__ == "__main__":
    main()
