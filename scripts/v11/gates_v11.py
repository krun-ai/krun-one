"""Krun One v1.1 — the registered TECHNICAL promotion gates (configs/v11/promotion_gates.yaml + the 2026-10-01 amendments
in configs/v11/r1_r2_protocol.yaml) on KrunTechBench v1, raw rendering, per seed and for the mean of the seeds.

    uv run python scripts/v11/gates_v11.py R1=r1_s1,r1_s2 R2_last4=r2l4_s1,r2l4_s2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/v11"))

import techbench_report as R  # noqa: E402
from krun_one.v11 import training as T  # noqa: E402

ex = {e["id"]: e for e in T.load_jsonl(ROOT / "data/v11/techbench/protected.jsonl")}
calib = json.loads((ROOT / "artifacts/krun-one-v1-text-final/calibration.json").read_text())
Tm, thr = calib["temperature"]["tool"], calib["margin_threshold"]["tool"]


def facts(model: str) -> dict:
    rows = R.load_rows(model, "protected", ex, Tm, thr)
    m = R.metrics(rows, ex)["raw"]
    hn = [r["margin_logit"] for i, r in rows.items() if R.cond(ex[i]) == "raw" and r.get("answerable") and "margin_logit" in r
          and "hard_negative" in ex[i]["tags"]]
    return {"macro_top1": m["macro_top1"], "by_track": {k: v["top1"] for k, v in m["by_track"].items()},
            "gold_vs_hardest": m["macro_gold_vs_hardest"], "margin_logit": m["macro_margin_logit"], "hn_median_margin": float(np.median(hn)),
            "flip_both": m["flip"]["both_correct"], "flip_stuck": m["flip"]["stuck_on_base_answer"],
            "unseen_combination": m["by_tag"]["unseen_combination"]["top1"], "seen_combination": m["by_tag"]["seen_combination"]["top1"],
            "unseen_technology": m["by_tag"]["unseen_technology"]["top1"], "unseen_family": m["by_tag"]["unseen_family"]["top1"],
            "ood_fa": m["ood"]["false_acceptance"], "rows": rows}


base = facts("v1")
paired = lambda rows: R.paired(base["rows"], rows, ex, "raw")  # noqa: E731


def gates(f: dict) -> dict:
    p = paired(f["rows"])
    worst = min((f["by_track"][t] - base["by_track"][t]) * 100 for t in base["by_track"])
    g = {"top1_macro_gain_pp": ((f["macro_top1"] - base["macro_top1"]) * 100, (f["macro_top1"] - base["macro_top1"]) * 100 >= 10 and p["ci95"][0] > 0),
         "pairwise_gold_vs_hardest_gain_pp": ((f["gold_vs_hardest"] - base["gold_vs_hardest"]) * 100, (f["gold_vs_hardest"] - base["gold_vs_hardest"]) * 100 >= 8),
         "hard_negative_margin_gain_logit": (f["margin_logit"] - base["margin_logit"], f["margin_logit"] - base["margin_logit"] >= 1.0 and f["hn_median_margin"] > 0),
         "constraint_flip_both_gain_pp": ((f["flip_both"] - base["flip_both"]) * 100, (f["flip_both"] - base["flip_both"]) * 100 >= 10),
         "flip_stuck_does_not_rise_pp": ((f["flip_stuck"] - base["flip_stuck"]) * 100, f["flip_stuck"] <= base["flip_stuck"]),
         "unseen_combination_gain_pp": ((f["unseen_combination"] - base["unseen_combination"]) * 100, (f["unseen_combination"] - base["unseen_combination"]) * 100 >= 10),
         "worst_track_delta_pp": (worst, worst >= -2.0),
         "ood_false_acceptance": (f["ood_fa"], f["ood_fa"] <= 0.08)}
    return {k: {"value": float(v), "pass": bool(ok)} for k, (v, ok) in g.items()}


out = {"baseline": {k: v for k, v in base.items() if k != "rows"}, "variants": {}}
md = ["# Technical promotion gates — KrunTechBench v1 (protected), raw rendering", "",
      f"Baseline v1: macro top-1 {100 * base['macro_top1']:.1f}, gold > hardest {100 * base['gold_vs_hardest']:.1f}, margin {base['margin_logit']:+.2f}, "
      f"flips both {100 * base['flip_both']:.1f} / stuck {100 * base['flip_stuck']:.1f}, unseen combination {100 * base['unseen_combination']:.1f}.", ""]
for arg in sys.argv[1:]:
    name, members = arg.split("=")
    seeds = {m: facts(m) for m in members.split(",")}
    per = {m: gates(f) for m, f in seeds.items()}
    keys = list(next(iter(per.values())))
    mean = {k: {"value": float(np.mean([per[m][k]["value"] for m in per])), "pass": all(per[m][k]["pass"] for m in per)} for k in keys}
    out["variants"][name] = {"seeds": {m: {"facts": {k: v for k, v in seeds[m].items() if k != "rows"}, "gates": per[m]} for m in per},
                             "mean": mean, "ALL_PASS": all(v["pass"] for v in mean.values())}
    md += [f"## {name}", "", "| gate | " + " | ".join(per) + " | mean | holds for every seed |", "|---|" + "---:|" * (len(per) + 1) + "---|"]
    for k in keys:
        md.append(f"| {k} | " + " | ".join(f"{per[m][k]['value']:+.2f} {'✅' if per[m][k]['pass'] else '❌'}" for m in per)
                  + f" | {mean[k]['value']:+.2f} | {'✅' if mean[k]['pass'] else '❌'} |")
    md += [f"| **all technical gates** | | | | {'✅' if out['variants'][name]['ALL_PASS'] else '❌'} |", ""]
(ROOT / "results/v11/techbench/gates.json").write_text(json.dumps(out, indent=1))
(ROOT / "results/v11/techbench/gates.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
