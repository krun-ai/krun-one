"""Stage 2 evaluation under configs/v1/stage2_protocol.yaml (registered before training).

    python scripts/v1/stage2_compare.py  -> results/v1/stage2/stage2_comparison.json (+ printed summary)

Per checkpoint (25 / 50 / 75 / 100%): ToolACE breakdown on val_tools and holdout (gate harness bf16, and the fp32-scorer
diagnostic), V metrics + coverage curves, preservation checks vs v1-tool on V FIT (selection eligibility) and V EVAL
(gates). Selection: max ToolACE val_tools accuracy among eligible checkpoints (holdout never used). Gates on the selected
checkpoint only.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("C", ROOT / "scripts/v1/r2_compare.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
K = C.K
PROTO = yaml.safe_load((ROOT / "configs/v1/stage2_protocol.yaml").read_text())
S2 = ROOT / "runs/v1/ft/krun-one-v1-stage2"
RECS = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "runs/v1/ft/regression_toolace_v0/records.jsonl")}


def kbin(k):
    return "k2" if k <= 2 else "k4" if k <= 4 else "k8" if k <= 8 else "k16plus"


def toolace_breakdown(path: Path) -> dict:
    res = defaultdict(list)
    for l in gzip.open(path, "rt"):
        s = json.loads(l)
        r = RECS.get(s["id"])
        if r is None or not r["answerable"]:
            continue
        f = r["metadata"]["v0_file"]
        ok = float(int(np.argmax(s["rank"])) in s["gold"])
        st, k = str(r["candidate_set"]["strategy"]), len(r["candidates"])
        keys = ["full", kbin(k), "small_k" if k < 16 else "large_k",
                "clean" if st in ("random", "native+random") else "legacy_xlam_negative",
                "native" if st.startswith("native") else "non_native", f"strategy:{st}"]
        for key in keys:
            res[(f, key)].append(ok)
    out = defaultdict(dict)
    for (f, key), v in res.items():
        out[f][key] = {"acc": float(np.mean(v)), "n": len(v)}
    return dict(out)


def fit_as_eval(rows):
    return [dict(r, part="eval") for r in rows if r["part"] == "fit"] + [r for r in rows if r["part"] == "fit"]


def block(run: Path, reg: Path | None, part="eval") -> dict:
    rows = [r for r in K.load(run) if r["file"] != "val_tools_v2"]
    if part == "fit":
        rows = fit_as_eval(rows)
    res, calib = K.evaluate(rows)
    b = {"rows": rows, "metrics": res, "curves": C.coverage_curves(rows, calib)}
    if part == "eval":
        b["serving"] = C.serving_point(rows, calib)
        vt = [r for r in K.load(run) if r["file"] == "val_tools_v2"]
        if vt:
            vres, vcal = K.evaluate(vt)
            b["val_tools_v2"] = {"metrics": vres, "curves": C.coverage_curves(vt, vcal)}
    if reg is not None and reg.exists():
        b["toolace"] = toolace_breakdown(reg)
        fp = reg.with_name(reg.name.replace(".jsonl.gz", "_fp32.jsonl.gz"))
        if fp.exists():
            b["toolace_fp32_diagnostic"] = toolace_breakdown(fp)
    return b


def preservation(A: dict, B: dict) -> dict:
    P = PROTO["gates"]["preservation_vs_v1_tool"]
    ma, mb = A["metrics"], B["metrics"]
    nat = K.paired_delta(A["rows"], B["rows"], lambda r: r["file"] == "val_ml" and not r["id"].startswith("kd1val/mt/")
                         and not r["lang"].startswith("en"))
    pt = lambda m: m["by_language"]["PT-BR"]["choice_accuracy"]  # noqa: E731
    at = lambda b, key: C.at_cov(b, "tool", 0.7, key)  # noqa: E731
    v = {
        "native_non_en": (nat["delta"] * 100, nat["delta"] * 100 >= -P["native_non_en_accuracy_drop_pp"]),
        "pt_br": ((pt(mb) - pt(ma)) * 100, (pt(mb) - pt(ma)) * 100 >= -P["pt_br_accuracy_drop_pp"]),
        "cross_language": ((mb["cross_language_accuracy"] - ma["cross_language_accuracy"]) * 100,
                           (mb["cross_language_accuracy"] - ma["cross_language_accuracy"]) * 100 >= -P["cross_language_accuracy_drop_pp"]),
        "intent": ((mb["choice_accuracy_intent"] - ma["choice_accuracy_intent"]) * 100,
                   (mb["choice_accuracy_intent"] - ma["choice_accuracy_intent"]) * 100 >= -P["intent_accuracy_drop_pp"]),
        "permutation": ((mb["permutation_consistency"] - ma["permutation_consistency"]) * 100,
                        (mb["permutation_consistency"] - ma["permutation_consistency"]) * 100 >= -P["permutation_consistency_drop_pp"]),
        "ece": (mb["ece"], mb["ece"] <= ma["ece"] + P["ece_rise"] and mb["ece"] <= 0.05),
        "ood_auroc": (mb["ood_auroc_margin"] - ma["ood_auroc_margin"], mb["ood_auroc_margin"] >= ma["ood_auroc_margin"] - P["ood_auroc_drop"]),
        "aurc": (mb["aurc"] - ma["aurc"], mb["aurc"] <= ma["aurc"] + P["aurc_rise"]),
        "tool_unseen_fa_cov70": (at(B, "unseen_tool_false_acceptance"), (at(B, "unseen_tool_false_acceptance") or 0) <= P["tool_unseen_fa_at_cov70_max"]),
        "tool_ood_fa_cov70": (at(B, "ood_false_acceptance"), (at(B, "ood_false_acceptance") or 0) <= P["tool_ood_fa_at_cov70_max"]),
        "score_mae": (mb["score_mae"] - ma["score_mae"], mb["score_mae"] <= ma["score_mae"] + P["score_mae_rise"]),
        "score_spearman": (mb["score_spearman"] - ma["score_spearman"], mb["score_spearman"] >= ma["score_spearman"] - P["score_spearman_drop"]),
        "noul_auroc": (mb["noul_auroc"] - ma["noul_auroc"], mb["noul_auroc"] >= ma["noul_auroc"] - P["noul_auroc_drop"]),
        "multi_micro_f1": ((mb["multi_micro_f1"] - ma["multi_micro_f1"]) * 100,
                           (mb["multi_micro_f1"] - ma["multi_micro_f1"]) * 100 >= -P["multi_micro_f1_drop_pp"]),
    }
    return {k: {"value": x, "pass": bool(ok)} for k, (x, ok) in v.items()}


def latency_gate() -> dict:
    p = ROOT / "results/v1/stage2/latency_stage2.json"
    if not p.exists():
        return {"available": False, "pass": False}
    d = json.loads(p.read_text())
    w = {r["workload"]: r["total_p50_ms"] for r in d["results"]}
    P = PROTO["gates"]["preservation_vs_v1_tool"]
    return {"available": True, "gpu": d["gpu"], "typical_p50_ms": w.get("typical_1q_k4_ctx48"),
            "k128_p50_ms": w.get("k_128_tools_ctx64"), "all_p50_ms": w,
            "pass": w.get("typical_1q_k4_ctx48", 1e9) <= P["latency_typical_p50_ms_max"]
            and w.get("k_128_tools_ctx64", 1e9) <= P["latency_128_tools_p50_ms_max"]}


def main():
    ref = {n: (C.RUNS[n], C.TOOL_REG[n]) for n in ("v0.3", "v1", "v1-tool")}
    ckpts = {}
    for m in sorted((S2 / "milestones").glob("step_*")):
        if (m / "signals_val_eval.jsonl.gz").exists():
            ckpts[m.name] = (m, m / f"regression_toolace_v0/signals_s2_{m.name}.jsonl.gz")
    if (S2 / "signals_val_eval.jsonl.gz").exists():
        ckpts["final"] = (S2, S2 / "final/regression_toolace_v0/signals_s2_final.jsonl.gz")
    E = {n: block(*v) for n, v in {**ref, **ckpts}.items()}
    F = {n: block(*v, part="fit") for n, v in {"v1-tool": ref["v1-tool"], **ckpts}.items()}
    diag = ROOT / "runs/v1/ft/s2_v1tool_diag/regression_toolace_v0/signals_v1tool_fp32.jsonl.gz"
    if diag.exists():
        E["v1-tool"]["toolace_fp32_diagnostic"] = toolace_breakdown(diag)
    # ---- selection (registered): eligible = all preservation on FIT; max val_tools acc; tie 0.5 pp -> small-k, earlier
    sel_table = {}
    for n in ckpts:
        pres = preservation(F["v1-tool"], F[n])
        pres.pop("tool_unseen_fa_cov70")   # absolute OOD caps are gate-only (fit curves are the calibration set)
        pres.pop("tool_ood_fa_cov70")
        t = E[n]["toolace"]["val_tools"]
        sel_table[n] = {"eligible": all(v["pass"] for v in pres.values()), "fit_preservation": pres,
                        "val_tools_acc": t["full"]["acc"], "val_tools_small_k": t["small_k"]["acc"]}
    elig = [n for n in ckpts if sel_table[n]["eligible"]]
    selected = None
    if elig:
        best = max(sel_table[n]["val_tools_acc"] for n in elig)
        near = [n for n in elig if best - sel_table[n]["val_tools_acc"] <= 0.005]
        bs = max(sel_table[n]["val_tools_small_k"] for n in near)
        selected = [n for n in near if sel_table[n]["val_tools_small_k"] == bs][0]   # dict order = earlier first
    # ---- gates on the selected checkpoint (V eval + holdout)
    G = PROTO["gates"]
    gates = None
    if selected:
        h = E[selected]["toolace"]["holdout_tools"]
        v03 = E["v0.3"]
        pres = preservation(E["v1-tool"], E[selected])
        lat = latency_gate()
        ml_ok = all(pres[k]["pass"] for k in ("native_non_en", "pt_br", "cross_language"))
        g1 = h["full"]["acc"] >= 0.918 - 1e-9
        v03_tool_aurc = v03["curves"]["tool"]["aurc"]
        g1_alt = (h["full"]["acc"] >= 0.908 - 1e-9 and h["clean"]["acc"] > 0.963 and h["k16plus"]["acc"] > 0.871
                  and (C.at_cov(E[selected], "tool", 0.7, "unseen_tool_false_acceptance") or 0) <= 0.05
                  and E[selected]["curves"]["tool"]["aurc"] < v03_tool_aurc and ml_ok)
        g2 = h["small_k"]["acc"] >= 0.922 - 1e-9
        all_pres = all(v["pass"] for v in pres.values()) and lat["pass"]
        gates = {"G1_toolace_full": g1, "G1_alternative": (None if g1 else g1_alt), "G2_small_k": g2,
                 "preservation": pres, "latency": lat, "all_preservation": all_pres,
                 "FREEZE_TEXTUAL_BASE": bool((g1 or g1_alt) and g2 and all_pres)}
    strip = lambda b: {k: v for k, v in b.items() if k != "rows"}  # noqa: E731
    paired = {}
    for n in ckpts:
        paired[n] = {"holdout_vs_v0.3": C.paired(C.toolace(ref["v0.3"][1])["per"], C.toolace(ckpts[n][1])["per"],
                                                 lambda i: RECS[i]["metadata"]["v0_file"] == "holdout_tools"),
                     "holdout_vs_v1-tool": C.paired(C.toolace(ref["v1-tool"][1])["per"], C.toolace(ckpts[n][1])["per"],
                                                    lambda i: RECS[i]["metadata"]["v0_file"] == "holdout_tools")}
    out = {"protocol": "configs/v1/stage2_protocol.yaml", "checkpoints": list(ckpts), "selection": sel_table,
           "selected": selected, "gates": gates, "paired_toolace": paired, "models": {n: strip(b) for n, b in E.items()}}
    p = ROOT / "results/v1/stage2"
    p.mkdir(parents=True, exist_ok=True)
    (p / "stage2_comparison.json").write_text(json.dumps(out, indent=1, default=str))
    f = lambda x: "—" if x is None else f"{100 * x:.1f}"  # noqa: E731
    print("model | val_tools | holdout full | clean | legacy | k2 | k4 | k8 | k16+ | small-k | fp32diag holdout")
    for n, b in E.items():
        t = b.get("toolace")
        if not t:
            continue
        h, v = t["holdout_tools"], t["val_tools"]
        d = b.get("toolace_fp32_diagnostic", {}).get("holdout_tools", {}).get("full", {}).get("acc")
        print(n, "|", " | ".join(f(x) for x in [v["full"]["acc"], h["full"]["acc"], h["clean"]["acc"], h["legacy_xlam_negative"]["acc"],
                                                 h.get("k2", {}).get("acc"), h.get("k4", {}).get("acc"), h.get("k8", {}).get("acc"),
                                                 h["k16plus"]["acc"], h["small_k"]["acc"], d]))
    print(json.dumps({"selection": {n: {k: v for k, v in s.items() if k != "fit_preservation"} for n, s in sel_table.items()},
                      "selected": selected}, indent=1))
    if gates:
        print(json.dumps({k: v for k, v in gates.items() if k != "preservation"}, indent=1, default=str))
        print(json.dumps(gates["preservation"], indent=1, default=str))


if __name__ == "__main__":
    main()
