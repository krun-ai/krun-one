"""Krun One v1.1 R3 — the registered batching selection (configs/v11/r3_protocol.yaml: batching_selection), validation only.

    uv run python scripts/v11/r3_select.py family=r3f_s1,r3f_s2 random=r3r_s1,r3r_s2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/v11"))

import r3_report as R  # noqa: E402
import select_v11 as S  # noqa: E402
from krun_one.v11 import training as T  # noqa: E402

ex = {e["id"]: e for f in ("val_v1_labeled.jsonl", "val_r3.jsonl") for e in T.load_jsonl(R.D / f)}
calib = json.loads((ROOT / "artifacts/krun-one-v1-text-final/calibration.json").read_text())
Tm, thr = calib["temperature"]["tool"], calib["margin_threshold"]["tool"]
STEMS = ["val", "val_v1_labeled", "val_r3"]


def facts(model: str) -> dict:
    c = R.metrics(ex, R.load(model, STEMS, ex, Tm, thr))["raw"]
    return {"v1_val_macro": c["v1_macro_top1"], "capability_macro": c["capability_macro"],
            "score": float(np.mean([c["v1_macro_top1"], c["capability_macro"]])) if c["capability_macro"] is not None else None,
            "stuck_rate_all_families": c["families_all"].get("stuck_rate"), "both_all_families": c["families_all"].get("both_correct"),
            "stuck_rate_v1_flips": c["flips_v1"].get("stuck_rate"), "unseen_catalog": c["r3_tables"]["catalog_unseen_combination"]["top1"],
            "capability": {k: v["top1"] for k, v in c["capability"].items()}, "v1_by_track": {k: v["top1"] for k, v in c["v1_by_track"].items()},
            "swe_involved": c["noul_swe_involved"]["top1"], "val_fit": S.val_fit_acc(model)}


v1 = facts("v1")
out = {"v1": v1, "modes": {}}
for arg in sys.argv[1:]:
    name, members = arg.split("=")
    seeds = {m: facts(m) for m in members.split(",")}
    mean = lambda k: float(np.mean([s[k] for s in seeds.values()]))  # noqa: E731
    out["modes"][name] = {"seeds": seeds, "score_mean": mean("score"), "score_std": float(np.std([s["score"] for s in seeds.values()])),
                          "stuck_mean": mean("stuck_rate_all_families"), "val_fit_drop_pp_mean": 100 * (v1["val_fit"] - mean("val_fit")),
                          "eligible": 100 * (v1["val_fit"] - mean("val_fit")) <= 1.0}
el = {k: v for k, v in out["modes"].items() if v["eligible"]}
if el:
    best = max(el, key=lambda k: el[k]["score_mean"])
    tied = [k for k in el if el[best]["score_mean"] - el[k]["score_mean"] <= 0.01]
    if len(tied) > 1:
        lo = min(tied, key=lambda k: el[k]["stuck_mean"])
        hi = max(tied, key=lambda k: el[k]["stuck_mean"])
        choice, why = (lo, "tie on score; stuck rate lower by >= 2 pp") if el[hi]["stuck_mean"] - el[lo]["stuck_mean"] >= 0.02 else \
            ("random" if "random" in tied else best, "tie on score and on stuck rate -> the simpler recipe")
    else:
        choice, why = best, "higher validation score (outside the 1.0 pp tie band)"
    seeds = out["modes"][choice]["seeds"]
    out["selected"] = {"mode": choice, "why": why, "candidate_seed": max(seeds, key=lambda m: seeds[m]["score"])}
(ROOT / "results/v11/r3").mkdir(parents=True, exist_ok=True)
(ROOT / "results/v11/r3/batching_selection.json").write_text(json.dumps(out, indent=1))
f = lambda v: "—" if v is None else f"{100 * v:.1f}"  # noqa: E731
print(f"v1: v1-val macro {f(v1['v1_val_macro'])}, capability {f(v1['capability_macro'])}, score {f(v1['score'])}, stuck {f(v1['stuck_rate_all_families'])}, val_fit {f(v1['val_fit'])}")
for k, v in out["modes"].items():
    print(f"{k}: score {f(v['score_mean'])} ± {f(v['score_std'])}, stuck {f(v['stuck_mean'])}, val_fit drop {v['val_fit_drop_pp_mean']:+.2f} pp, eligible {v['eligible']}")
    for m, s in v["seeds"].items():
        print(f"   {m}: v1-val {f(s['v1_val_macro'])} cap {f(s['capability_macro'])} score {f(s['score'])} stuck all {f(s['stuck_rate_all_families'])} / v1 flips {f(s['stuck_rate_v1_flips'])} "
              f"both {f(s['both_all_families'])} catalog {f(s['unseen_catalog'])} involved {f(s['swe_involved'])}")
print("SELECTED", json.dumps(out.get("selected")))
