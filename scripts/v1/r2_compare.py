"""R2 comparison: v0.3 / v1 (candidate 1, enc 2e-5) / v1-lr1e5 / v1-tool (candidate 2) under configs/v1/r2_protocol.yaml.

    python scripts/v1/r2_compare.py --stage lr     # LR selection (v1 vs v1-lr1e5)
    python scripts/v1/r2_compare.py --stage tool   # full comparison incl. the tool-data candidate + gates
    -> results/v1/r2/{lr_selection,comparison}.{json,md}

Coverage curves: per family (intent, tool) the margin threshold is the (1-coverage) quantile of the temperature-scaled
top1-top2 margin of ANSWERABLE fit decisions; metrics are reported on eval. Temperatures per family fitted on fit.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

import importlib.util

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("K", ROOT / "scripts/v1/kd1_evaluate.py")
K = importlib.util.module_from_spec(spec)
spec.loader.exec_module(K)
PROTO = yaml.safe_load((ROOT / "configs/v1/r2_protocol.yaml").read_text())
COVS = PROTO["ood_operating_point"]["coverages"]
RUNS = {"v0.3": ROOT / "runs/v1/ft/v03_baseline", "v1": ROOT / "runs/v1/ft/krun-one-v1-ft",
        "v1-lr1e5": ROOT / "runs/v1/ft/krun-one-v1-lr1e5", "v1-tool": ROOT / "runs/v1/ft/krun-one-v1-tool"}
TOOL_REG = {"v0.3": ROOT / "runs/v1/ft/regression_toolace_v0/signals_v03.jsonl.gz",
            "v1": ROOT / "runs/v1/ft/regression_toolace_v0/signals_v1.jsonl.gz",
            "v1-lr1e5": ROOT / "runs/v1/ft/krun-one-v1-lr1e5/regression_toolace_v0/signals_krun-one-v1-lr1e5.jsonl.gz",
            "v1-tool": ROOT / "runs/v1/ft/krun-one-v1-tool/regression_toolace_v0/signals_krun-one-v1-tool.jsonl.gz"}


def margin(rank, T):
    p = K.softmax(rank, T)
    s = np.sort(p)[::-1]
    return float(s[0] - (s[1] if len(s) > 1 else 0)), int(np.argmax(p))


def coverage_curves(rows, calib) -> dict:
    rows = [r for r in rows if r["type"] == "choice" and "permutation" not in r["slices"]]
    out = {}
    for fam in ("intent", "tool"):
        T = calib["temperature"].get(fam, 1.0)
        fit = [r for r in rows if r["part"] == "fit" and r["family"] == fam]
        ev = [r for r in rows if r["part"] == "eval" and r["family"] == fam]
        fm = np.array([margin(r["rank"], T)[0] for r in fit if r["answerable"]])
        if not len(fm) or not ev:
            continue
        E = [(r, *margin(r["rank"], T)) for r in ev]
        curve = []
        for cov in COVS:
            tau = float(np.quantile(fm, 1 - cov))
            acc = [x for x in E if x[1] >= tau]
            sel = [x[0]["answerable"] and x[2] in x[0]["gold"] for x in acc]
            un = [x[1] >= tau for x in E if not x[0]["answerable"]]
            ut = [x[1] >= tau for x in E if not x[0]["answerable"] and "unseen_tool" in x[0]["slices"]]
            ans_cov = [x[1] >= tau for x in E if x[0]["answerable"]]
            curve.append({"target_coverage": cov, "tau": round(tau, 4), "eval_coverage": K.mean(ans_cov),
                          "selective_accuracy": K.mean(sel), "ood_false_acceptance": K.mean(un),
                          "unseen_tool_false_acceptance": K.mean(ut), "n_ood": len(un), "n_unseen_ood": len(ut)})
        corr = np.array([float(x[0]["answerable"] and x[2] in x[0]["gold"]) for x in E])
        out[fam] = {"curve": curve, "aurc": K.aurc(np.array([x[1] for x in E]), corr)}
    return out


def serving_point(rows, calib) -> dict:
    """Registered rule: per family the largest coverage whose FIT selective accuracy >= .97 and OOD FA <= .05 / .08."""
    fitrows = [dict(r, part="eval") for r in rows if r["part"] == "fit"] + [dict(r, part="fit") for r in rows if r["part"] == "fit"]
    cc = coverage_curves(fitrows, calib)
    pick = {}
    for fam, v in cc.items():
        lim = 0.05 if fam == "intent" else 0.08
        ok = [c["target_coverage"] for c in v["curve"] if (c["selective_accuracy"] or 0) >= 0.97
              and (c["ood_false_acceptance"] if c["ood_false_acceptance"] is not None else 0) <= lim]
        pick[fam] = max(ok) if ok else 0.5
    return pick


def toolace(path: Path) -> dict:
    recs = {}
    for l in open(ROOT / "runs/v1/ft/regression_toolace_v0/records.jsonl"):
        r = json.loads(l)
        recs[r["id"]] = r
    rows = [json.loads(l) for l in gzip.open(path, "rt")]
    res = defaultdict(list)
    per = {}
    for s in rows:
        r = recs.get(s["id"])
        if r is None or not r["answerable"]:
            continue
        c = int(np.argmax(s["rank"])) in s["gold"]
        res[r["metadata"]["v0_file"]].append(c)
        res["all"].append(c)
        per[s["id"]] = (r["split_group_id"], float(c))
    return {"acc": {k: float(np.mean(v)) for k, v in res.items()}, "n": {k: len(v) for k, v in res.items()}, "per": per}


def paired(a: dict, b: dict, filt=lambda i: True, reps=1000, seed=0):
    ids = sorted(i for i in set(a) & set(b) if filt(i))
    g = defaultdict(list)
    for i in ids:
        g[a[i][0]].append(b[i][1] - a[i][1])
    keys = list(g)
    rng = np.random.default_rng(seed)
    bs = [np.mean([x for k in rng.integers(0, len(keys), len(keys)) for x in g[keys[k]]]) for _ in range(reps)]
    return {"delta": float(np.mean([b[i][1] - a[i][1] for i in ids])), "ci95": [float(np.percentile(bs, 2.5)),
            float(np.percentile(bs, 97.5))], "n": len(ids)}


def model_block(name, common_only=True):
    rows = K.load(RUNS[name])
    if common_only:                        # val_tools_v2 exists only for the tool-data candidate: never mix it in
        rows = [r for r in rows if r["file"] != "val_tools_v2"]
    res, calib = K.evaluate(rows)
    cc = coverage_curves(rows, calib)
    tr = toolace(TOOL_REG[name]) if TOOL_REG[name].exists() else None
    return {"rows": rows, "metrics": res, "calib": calib, "curves": cc, "serving": serving_point(rows, calib),
            "toolace": tr}


def at_cov(block, fam, cov, key):
    for c in block["curves"].get(fam, {}).get("curve", []):
        if c["target_coverage"] == cov:
            return c[key]
    return None


def all_ood_fa(block, cov):
    xs = [(at_cov(block, f, cov, "ood_false_acceptance"), at_cov(block, f, cov, "n_ood")) for f in ("intent", "tool")]
    xs = [(a, n) for a, n in xs if a is not None and n]
    return sum(a * n for a, n in xs) / max(1, sum(n for _, n in xs))


def lr_stage():
    A, B = model_block("v1"), model_block("v1-lr1e5")
    rule = PROTO["lr_selection"]
    ta = paired(A["toolace"]["per"], B["toolace"]["per"], lambda i: "holdout_tools" in i or True)
    hold = paired({k: v for k, v in A["toolace"]["per"].items()}, {k: v for k, v in B["toolace"]["per"].items()})
    recs = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "runs/v1/ft/regression_toolace_v0/records.jsonl")}
    hold = paired(A["toolace"]["per"], B["toolace"]["per"], lambda i: recs[i]["metadata"]["v0_file"] == "holdout_tools")
    ut_a, ut_b = at_cov(A, "tool", 0.7, "unseen_tool_false_acceptance"), at_cov(B, "tool", 0.7, "unseen_tool_false_acceptance")
    fa_a, fa_b = all_ood_fa(A, 0.7), all_ood_fa(B, 0.7)
    pref = {"holdout_tools_gain": hold["delta"] * 100 >= rule["prefer_1e-5_if_any"][0]["holdout_tools_gain_pp"] and hold["ci95"][0] > 0,
            "unseen_tool_fa_drop": ut_a is not None and ut_b is not None and (ut_a - ut_b) * 100 >= rule["prefer_1e-5_if_any"][1]["unseen_tool_fa_drop_pp_at_cov70"],
            "ood_fa_drop": (fa_a - fa_b) * 100 >= rule["prefer_1e-5_if_any"][2]["ood_fa_drop_pp_at_cov70"]}
    ma, mb = A["metrics"], B["metrics"]
    R = rule["and_no_material_regression"]
    native_a = K.paired_delta(A["rows"], B["rows"], lambda r: r["file"] == "val_ml" and not r["id"].startswith("kd1val/mt/") and not r["lang"].startswith("en"))
    k_ = lambda m: np.mean([m["by_k"][str(k)]["accuracy"] if str(k) in m["by_k"] else m["by_k"][k]["accuracy"] for k in (64, 128)])  # noqa: E731
    noreg = {"native_non_en": native_a["delta"] * 100 >= -R["native_non_en_accuracy_drop_pp"],
             "cross_language": (mb["cross_language_accuracy"] - ma["cross_language_accuracy"]) * 100 >= -R["cross_language_accuracy_drop_pp"],
             "k64_128": (k_(mb) - k_(ma)) * 100 >= -R["k64_128_accuracy_drop_pp"],
             "ece": mb["ece"] <= ma["ece"] + R["ece_rise"],
             "permutation": (mb["permutation_consistency"] - ma["permutation_consistency"]) * 100 >= -R["permutation_consistency_drop_pp"]}
    choice = "1e-5" if any(pref.values()) and all(noreg.values()) else "2e-5"
    out = {"prefer_1e-5_conditions": pref, "no_material_regression": noreg, "choice": choice,
           "evidence": {"toolace_holdout_delta_1e5_minus_2e5": hold, "toolace_all": ta,
                        "unseen_tool_fa_cov70": {"2e-5": ut_a, "1e-5": ut_b}, "ood_fa_cov70": {"2e-5": fa_a, "1e-5": fa_b},
                        "native_non_en_delta": native_a,
                        "toolace_acc": {"2e-5": A["toolace"]["acc"], "1e-5": B["toolace"]["acc"]}},
           "metrics": {"2e-5": ma, "1e-5": mb}, "curves": {"2e-5": A["curves"], "1e-5": B["curves"]},
           "serving_points": {"2e-5": A["serving"], "1e-5": B["serving"]}}
    p = ROOT / "results/v1/r2"
    p.mkdir(parents=True, exist_ok=True)
    (p / "lr_selection.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("prefer_1e-5_conditions", "no_material_regression", "choice")}, indent=1, default=str))
    print(json.dumps(out["evidence"], indent=1, default=str)[:3000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["lr", "tool"], required=True)
    a = ap.parse_args()
    if a.stage == "lr":
        lr_stage()
    else:
        tool_stage()


def tool_stage():
    names = [n for n in RUNS if (RUNS[n] / "signals_val_eval.jsonl.gz").exists()]
    blocks = {n: model_block(n) for n in names}
    out = {"models": names, "metrics": {n: b["metrics"] for n, b in blocks.items()},
           "curves": {n: b["curves"] for n, b in blocks.items()}, "serving": {n: b["serving"] for n, b in blocks.items()},
           "toolace": {n: (b["toolace"]["acc"] if b["toolace"] else None) for n, b in blocks.items()}}
    recs = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "runs/v1/ft/regression_toolace_v0/records.jsonl")}
    hold = lambda i: recs[i]["metadata"]["v0_file"] == "holdout_tools"  # noqa: E731
    out["toolace_paired_vs_v03"] = {n: paired(blocks["v0.3"]["toolace"]["per"], b["toolace"]["per"], hold)
                                    for n, b in blocks.items() if n != "v0.3" and b["toolace"]}
    out["v_paired_vs_v03"] = {n: {s: K.paired_delta(blocks["v0.3"]["rows"], b["rows"], f) for s, f in (
        ("native_non_en", lambda r: r["file"] == "val_ml" and not r["id"].startswith("kd1val/mt/") and not r["lang"].startswith("en")),
        ("cross_language", lambda r: "cross_language" in r["slices"]), ("hard", lambda r: r.get("difficulty") == "hard"),
        ("en_kbv1_val_eval", lambda r: r["file"] == "val_eval"))} for n, b in blocks.items() if n != "v0.3"}
    if (RUNS["v1-tool"] / "signals_val_tools_v2.jsonl.gz").exists():
        vt = [r for r in K.load(RUNS["v1-tool"]) if r["file"] == "val_tools_v2"]
        vres, vcal = K.evaluate(vt)
        out["v1_tool_val_tools_v2"] = {"metrics": vres, "curves": coverage_curves(vt, vcal)}
    p = ROOT / "results/v1/r2"
    (p / "comparison.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({"toolace": out["toolace"], "toolace_paired_vs_v03": out["toolace_paired_vs_v03"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
