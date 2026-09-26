#!/usr/bin/env python
"""Apply the decision rules registered in configs/krun_one_v0.yaml (validation val_eval only) and freeze them.

    .venv/bin/python scripts/decide.py --a eval_krun_one_A --b eval_krun_one_B

Writes results/krun_one_decisions.json (refuses to overwrite). Only after this file exists can
scripts/evaluate.py --final touch holdout_tools / kb_test.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from compare_with_laya import weighted  # noqa: E402

SCORES = ("top1_probability", "top1_top2_margin", "neg_normalized_entropy")


def primary(rep: dict) -> dict:
    s = rep["sets"]
    intent = weighted(s["kb_val"]["quality"]["by_dataset"], lambda k: k != "hwu64")
    tool = s["val_tools"]["quality"]["core"]["accuracy"]
    return {"intent_core_acc_kb_val": intent, "tool_core_acc_val_tools": tool, "mean": (intent + tool) / 2}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="eval_krun_one_A")
    ap.add_argument("--b", default="eval_krun_one_B")
    ap.add_argument("--config", default=str(ROOT / "configs" / "krun_one_v0.yaml"))
    ap.add_argument("--out", default="krun_one_decisions_v0.json",
                    help="v0 choice (A vs B + abstention score); the final gate file krun_one_decisions.json is written "
                         "only after the error analysis / v0.1 decision")
    args = ap.parse_args()
    dst = ROOT / "results" / args.out
    if dst.exists():
        raise SystemExit(f"{dst} exists: decisions are frozen")
    cfg = yaml.safe_load(Path(args.config).read_text())
    A, B = (json.loads((ROOT / "results" / f"{n}.json").read_text()) for n in (args.a, args.b))
    pa, pb = primary(A), primary(B)
    chosen = "B" if pb["mean"] - pa["mean"] >= 0.005 else "A"
    rep = A if chosen == "A" else B
    sc = {}
    for name in SCORES:
        au = [rep["sets"][s]["abstention"]["all"][name]["ood_auroc"] for s in ("kb_val", "val_tools")]
        ar = [rep["sets"][s]["abstention"]["all"][name]["aurc"] for s in ("kb_val", "val_tools")]
        sc[name] = {"mean_ood_auroc": sum(au) / 2, "mean_aurc": sum(ar) / 2}
    best = max(sc, key=lambda n: sc[n]["mean_ood_auroc"])
    close = [n for n in sc if sc[best]["mean_ood_auroc"] - sc[n]["mean_ood_auroc"] <= 0.005]
    score = min(close, key=lambda n: sc[n]["mean_aurc"])
    out = {"frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rules": cfg["decision_rules"],
           "config_choice": {"A": pa, "B": pb, "delta_B_minus_A_pp": 100 * (pb["mean"] - pa["mean"]), "chosen": chosen},
           "chosen_config": chosen, "chosen_eval": args.a if chosen == "A" else args.b,
           "abstention_candidates": sc, "abstention_score": score,
           "operating_point_target": f"{cfg['evaluation']['target_selective_accuracy'][0]:.2f}",
           "note": "decided on validation val_eval only; final sets (holdout_tools, kb_test) not yet opened"}
    dst.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
