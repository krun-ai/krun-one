"""Krun One v1.1 — the two registered selections of configs/v11/r1_r2_protocol.yaml, computed on KrunTechVal only.

    uv run python scripts/v11/select_v11.py pilot                       # objective O0..O4
    uv run python scripts/v11/select_v11.py variants --groups R2=r2_s1,r2_s2 R2_noadapter=r2na_s1,r2na_s2 ...

score      = macro mean over tracks of raw top-1 on KrunTechVal (answerable multi-way choice items)
constraint = KrunBench v1 val_fit choice accuracy within 1.0 pp of the same-host v1
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/v11"))

import techbench_report as R  # noqa: E402
from krun_one.v11 import training as T  # noqa: E402

EVAL = ROOT / "results/v11/eval"


def val_fit_acc(model: str) -> float:
    ok = []
    with gzip.open(EVAL / model / "signals_val_fit.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["type"] == "choice" and r["answerable"] and "permutation" not in r["slices"]:
                ok.append(int(np.argmax(r["rank"])) in r["gold"])
    return float(np.mean(ok))


def tech(model: str, ex: dict, Tm: float, thr: float) -> dict:
    m = R.metrics(R.load_rows(model, "val", ex, Tm, thr), ex)
    return {"macro_raw": m["raw"]["macro_top1"], "macro_feat": m["feat"]["macro_top1"], "by_track_raw": {k: v["top1"] for k, v in m["raw"]["by_track"].items()},
            "flip_both_raw": m["raw"]["flip"].get("both_correct"), "gold_vs_hardest_raw": m["raw"]["macro_gold_vs_hardest"],
            "margin_raw": m["raw"]["macro_margin_logit"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pilot", "variants"])
    ap.add_argument("--groups", nargs="*", default=[])
    args = ap.parse_args()
    ex = {e["id"]: e for e in T.load_jsonl(ROOT / "data/v11/techbench/val.jsonl")}
    calib = json.loads((ROOT / "artifacts/krun-one-v1-text-final/calibration.json").read_text())
    Tm, thr = calib["temperature"]["tool"], calib["margin_threshold"]["tool"]
    base_fit = val_fit_acc("v1")
    out = {"v1": {**tech("v1", ex, Tm, thr), "val_fit": base_fit}}
    if args.cmd == "pilot":
        order = ["O0", "O1", "O2", "O3", "O4"]
        res = {}
        for o in order:
            if (EVAL / f"pilot_{o}" / "signals_val_fit.jsonl.gz").exists():
                fit = val_fit_acc(f"pilot_{o}")
                res[o] = {**tech(f"pilot_{o}", ex, Tm, thr), "val_fit": fit, "val_fit_drop_pp": (base_fit - fit) * 100,
                          "meets_constraint": (base_fit - fit) * 100 <= 1.0}
        elig = [o for o in res if res[o]["meets_constraint"]]
        choice = None
        if elig:
            best = max(res[o]["macro_raw"] for o in elig)
            tied = [o for o in elig if best - res[o]["macro_raw"] <= 0.005]
            choice = sorted(tied, key=lambda o: (round(res[o]["val_fit_drop_pp"], 2), order.index(o)))[0]
        out.update(objectives=res, eligible=elig, chosen=choice, rule="configs/v11/r1_r2_protocol.yaml training.pilot.rule")
        p = ROOT / "results/v11/r2/pilot_selection.json"
    else:
        res = {}
        for g in args.groups:
            name, members = g.split("=")
            ms = members.split(",")
            per = {m: {**tech(m, ex, Tm, thr), "val_fit": val_fit_acc(m)} for m in ms}
            res[name] = {"seeds": per, "macro_raw_mean": float(np.mean([v["macro_raw"] for v in per.values()])),
                         "macro_raw_std": float(np.std([v["macro_raw"] for v in per.values()])),
                         "macro_feat_mean": float(np.mean([v["macro_feat"] for v in per.values()])),
                         "val_fit_drop_pp_mean": float(np.mean([(base_fit - v["val_fit"]) * 100 for v in per.values()])),
                         "n_seeds": len(ms)}
            res[name]["eligible"] = res[name]["val_fit_drop_pp_mean"] <= 1.0
        out.update(variants=res)
        p = ROOT / "results/v11/r2/variant_selection.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
