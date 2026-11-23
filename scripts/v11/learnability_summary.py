"""Krun One v1.1 R0-L — one table over the learnability runs (held-out probe + train-fit sample), offline.

    uv run python scripts/v11/learnability_summary.py    # -> results/v11/capacity_probe/learnability/summary.{json,md}
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from krun_one.v11 import harness, probe  # noqa: E402

BASE = ROOT / "results/v11/capacity_probe"
SLICES = ("tetris", "scheduling", "routing", "sizing", "techsel")
ORDER = ["v1_zero_shot", "k1final_full", "k1final_full_ci2", "k1final_head", "k1final_head_L4", "k1final_head_ci2",
         "mmbert_small_full", "mmbert_base_full", "mmbert_base_full_s2", "k1_0b_full", "k1_0b_full_s2"]


def rows_of(signals: Path, problems: dict, T: float, thr: float) -> dict:
    out = defaultdict(dict)
    with gzip.open(signals, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in problems:
                out[r["cond"]][r["id"]] = probe.item_signals(problems[r["id"]], r["z"], T, thr)
    return out


def cell(rows: dict, problems: dict, pred) -> dict:
    sel = [r for i, r in rows.items() if pred(problems[i])]
    return probe.aggregate(sel)


def main() -> None:
    calib = json.loads((ROOT / "artifacts/krun-one-v1-text-final/calibration.json").read_text())
    T, thr = calib["temperature"]["tool"], calib["margin_threshold"]["tool"]
    test = {p["id"]: p for p in harness.load_problems(ROOT / "data/v11/probe/probe.jsonl")}
    train = {p["id"]: p for p in harness.load_problems(ROOT / "data/v11/probe/ltrain.jsonl")}
    mw = lambda p: p["answerable"] and not p.get("flip") and "pair_kind" not in p  # noqa: E731
    runs = {}
    dirs = {"v1_zero_shot": BASE / "v1_zero_shot"}
    dirs.update({d.name: d for d in sorted((BASE / "learnability").iterdir()) if (d / "run.json").exists()})
    for name in [n for n in ORDER if n in dirs] + [n for n in dirs if n not in ORDER]:
        d = dirs[name]
        info = json.loads((d / "run.json").read_text()) if (d / "run.json").exists() else {}
        sig = rows_of(d / "signals.jsonl.gz", test, T, thr)
        fit = rows_of(d / "signals_trainfit.jsonl.gz", train, T, thr) if (d / "signals_trainfit.jsonl.gz").exists() else {}
        r = {"params_total": info.get("params_total", 879356931), "params_non_embedding": info.get("params_non_embedding", 486139395),
             "params_trainable": info.get("params_trainable", 0), "train_seconds": info.get("train_seconds"),
             "final_train_loss": info["log"][-1]["loss"] if info else None, "test": {}, "trainfit": {}, "pair_hardest": {}, "flip": {}, "ood": {}}
        for cond in probe.CONDITIONS:
            for sl in SLICES:
                for k in (8, 16, 32):
                    c = cell(sig[cond], test, lambda p: mw(p) and p["slice"] == sl and p["k"] == k)
                    if c["n"]:
                        r["test"][f"{sl}/{cond}/k{k}"] = {x: c[x] for x in ("n", "top1", "skill", "top1_ci95", "margin_logit", "entropy", "gold_p", "pair_all", "ndcg", "abstain_rate", "ece_T")}
                if fit:
                    c = cell(fit[cond], train, lambda p: p["slice"] == sl and not p.get("flip"))
                    r["trainfit"][f"{sl}/{cond}"] = {"n": c["n"], "top1": c.get("top1")}
                c = cell(sig[cond], test, lambda p: p.get("pair_kind") == "hardest" and p["slice"] == sl)
                r["pair_hardest"][f"{sl}/{cond}"] = c["top1"]
            r["flip"][cond] = probe.flip_metrics(sig[cond], test)
            r["ood"][cond] = probe.ood_metrics(list(sig[cond].values()), test)
            tech = {i: x for i, x in sig[cond].items() if test[i]["slice"] == "techsel" and mw(test[i]) and test[i]["k"] == 8}
            r["test"][f"techsel_seen/{cond}/k8"] = probe.aggregate([x for i, x in tech.items() if test[i]["meta"]["ltrain_seen"]]).get("top1")
            r["test"][f"techsel_unseen_sets/{cond}/k8"] = probe.aggregate([x for i, x in tech.items() if not test[i]["meta"]["ltrain_seen"] and test[i]["meta"]["category"] != "messaging"]).get("top1")
            r["test"][f"techsel_unseen_category/{cond}/k8"] = probe.aggregate([x for i, x in tech.items() if test[i]["meta"]["category"] == "messaging"]).get("top1")
        runs[name] = r
    (BASE / "learnability").mkdir(parents=True, exist_ok=True)
    (BASE / "learnability/summary.json").write_text(json.dumps(runs, indent=1))
    f3 = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
    md = ["# R0-L learnability diagnostic — summary (held-out probe, k = 8 unless stated)", "",
          "Throwaway weights: 2,000 steps on `ltrain.jsonl` with v1's own objective; nothing saved, no replay.", ""]
    for cond in probe.CONDITIONS:
        md += [f"## {cond}: held-out top-1 (train-fit top-1 in brackets)", "",
               "| run | params | trainable | " + " | ".join(SLICES) + " |", "|---|---:|---:|" + "---:|" * len(SLICES)]
        for name, r in runs.items():
            cells = []
            for sl in SLICES:
                t = r["test"].get(f"{sl}/{cond}/k8", {}).get("top1")
                tf = r["trainfit"].get(f"{sl}/{cond}", {}).get("top1")
                cells.append(f3(t) + (f" [{tf:.3f}]" if tf is not None else ""))
            md.append(f"| {name} | {r['params_total'] / 1e6:.0f}M | {r['params_trainable'] / 1e6:.0f}M | " + " | ".join(cells) + " |")
        md.append("")
    md += ["## Large candidate sets: held-out top-1 at k = 16 / 32", "", "| run | cond | " + " | ".join(SLICES) + " |", "|---|---|" + "---|" * len(SLICES)]
    for name, r in runs.items():
        for cond in probe.CONDITIONS:
            md.append(f"| {name} | {cond} | " + " | ".join(
                f3(r["test"].get(f"{sl}/{cond}/k16", {}).get("top1")) + " / " + f3(r["test"].get(f"{sl}/{cond}/k32", {}).get("top1")) for sl in SLICES) + " |")
    md += ["", "## Gold vs hardest negative: pairwise accuracy and multi-way logit margin (k = 8)", "",
           "| run | cond | " + " | ".join(SLICES) + " |", "|---|---|" + "---|" * len(SLICES)]
    for name, r in runs.items():
        for cond in probe.CONDITIONS:
            md.append(f"| {name} | {cond} | " + " | ".join(
                f3(r["pair_hardest"].get(f"{sl}/{cond}")) + f" ({r['test'].get(f'{sl}/{cond}/k8', {}).get('margin_logit', float('nan')):+.2f})" for sl in SLICES) + " |")
    md += ["", "## Constraint flips: both correct / stuck on the base answer", "", "| run | cond | scheduling | routing | sizing | techsel |", "|---|---|---|---|---|---|"]
    for name, r in runs.items():
        for cond in probe.CONDITIONS:
            fl = r["flip"][cond]
            md.append(f"| {name} | {cond} | " + " | ".join(f"{fl[s]['both_correct']:.3f} / {fl[s]['stuck_on_base_answer']:.3f}" if s in fl else "—"
                                                             for s in ("scheduling", "routing", "sizing", "techsel")) + " |")
    md += ["", "## Tech selection generalization (k = 8 top-1): seen sets / unseen requirement sets / unseen category (messaging)", "",
           "| run | raw | feat |", "|---|---|---|"]
    for name, r in runs.items():
        md.append(f"| {name} | " + " | ".join(" / ".join(f3(r["test"].get(f"techsel_{s}/{cond}/k8")) for s in ("seen", "unseen_sets", "unseen_category"))
                                              for cond in probe.CONDITIONS) + " |")
    md += ["", "## None-of-the-above at the production threshold: false acceptance / answerable coverage / margin AUROC", "",
           "| run | cond | scheduling | sizing | techsel |", "|---|---|---|---|---|"]
    for name, r in runs.items():
        for cond in probe.CONDITIONS:
            o = r["ood"][cond]
            md.append(f"| {name} | {cond} | " + " | ".join(
                f"{o[s]['false_acceptance']:.3f} / {o[s]['answerable_coverage']:.3f} / {f3(o[s]['auroc_margin'])}" if s in o else "—"
                for s in ("scheduling", "sizing", "techsel")) + " |")
    (BASE / "learnability/summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
