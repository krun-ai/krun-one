"""Krun One v1.1 R3 — the registered promotion gates (configs/v11/r3_protocol.yaml: promotion_gates), from the reports.

    uv run python scripts/v11/r3_gates.py --seeds r3f_s1 r3f_s2 --candidate r3f_s1 [--mm r3f_s1_m20260932_e1]

Inputs: results/v11/r3/techbench/protected.json, r2_reference.json, regression/regression.json, jevbench/jevbench_r3.json,
multimodal/multimodal_r3.json. Output: results/v11/r3/gates.{json,md}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R3 = ROOT / "results/v11/r3"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--mm", help="multimodal run tag of the candidate (re-aligned fusion)")
    ap.add_argument("--baseline", default="v1")
    a = ap.parse_args()
    tb = json.loads((R3 / "techbench/protected.json").read_text())["models"]
    r2 = json.loads((R3 / "r2_reference.json").read_text())["r2_no_adapter_mean"]
    reg = json.loads((R3 / "regression/regression.json").read_text())
    jv = json.loads((R3 / "jevbench/jevbench_r3.json").read_text())
    c = {m: tb[m]["raw"] for m in a.seeds}
    per = lambda f: [f(c[m]) for m in a.seeds]  # noqa: E731
    g = {}
    v = per(lambda x: x["v1_macro_top1"])
    g["kruntechbench_v1_macro"] = {"values": v, "mean": float(np.mean(v)), "need": ">= 0.617", "pass": float(np.mean(v)) >= 0.617}
    v = per(lambda x: x["v1_slices"]["unseen_combination"]["top1"])
    g["unseen_combination"] = {"values": v, "mean": float(np.mean(v)), "r2": r2["unseen_combination"], "need": "mean >= R2 + 10 pp, each seed >= R2 + 5 pp",
                               "pass": float(np.mean(v)) >= r2["unseen_combination"] + 0.10 and min(v) >= r2["unseen_combination"] + 0.05}
    v = per(lambda x: x["flips_v1"]["both_correct"])
    g["constraint_flips_both"] = {"values": v, "mean": float(np.mean(v)), "r2": r2["flips_both"], "need": "mean >= R2 + 5 pp", "pass": float(np.mean(v)) >= r2["flips_both"] + 0.05}
    v = per(lambda x: x["flips_v1"]["stuck_rate"])
    g["stuck_rate"] = {"values": v, "mean": float(np.mean(v)), "r2": r2["stuck_rate"], "need": "mean <= R2 - 5 pp", "pass": float(np.mean(v)) <= r2["stuck_rate"] - 0.05}
    v = per(lambda x: x["v1_macro_margin_logit"])
    g["hard_negative_margin"] = {"values": v, "mean": float(np.mean(v)), "r2": r2["margin_logit"], "need": "mean >= R2 - 0.3 logit", "pass": float(np.mean(v)) >= r2["margin_logit"] - 0.3}
    gg = {m: {k: v for k, v in reg["gates"][m].items() if isinstance(v, dict)} for m in a.seeds}
    n_ok = {m: sum(x["pass"] for x in gg[m].values()) for m in a.seeds}
    n_all = {m: len(gg[m]) for m in a.seeds}
    g["text_preservation"] = {"values": [f"{n_ok[m]}/{n_all[m]}" for m in a.seeds], "need": "19 / 19 on every seed",
                              "k128_pp": [reg["gates"][m]["large_k_k128"]["value"] for m in a.seeds], "pass": all(n_ok[m] == n_all[m] for m in a.seeds)}
    if a.mm and (R3 / "multimodal/multimodal_r3.json").exists():
        mm = json.loads((R3 / "multimodal/multimodal_r3.json").read_text())["rows"]
        row = next(v for k, v in mm.items() if k.startswith("R3 core + RE-ALIGNED") and f"[{a.mm}]" in k)
        old = next(v for k, v in mm.items() if k.startswith("R3 core + OLD") and f"[{a.mm}]" in k)
        g["multimodal"] = {"values": row["gates"], "passed": row["passed"], "old_fusion_passed": old["passed"], "need": "5 / 5", "pass": row["passed"] == 5}
    else:
        g["multimodal"] = {"values": None, "need": "5 / 5", "pass": False, "note": "not measured"}
    p = jv["paired_vs_baseline"]
    cand, others = p[a.candidate]["all 231"], [p[m]["all 231"] for m in a.seeds if m != a.candidate]
    g["jevbench"] = {"candidate": cand, "other_seeds_delta": [o["delta"] for o in others], "need": "candidate CI excludes 0 (above) AND the other seed above v1",
                     "pass": cand["ci95"][0] > 0 and all(o["delta"] > 0 for o in others)}
    lat = reg["latency"]
    d = {m: {"typical_p50_ms": lat[m]["bf16/typical_1q_tools8_ctx48"]["p50"] - lat[a.baseline]["bf16/typical_1q_tools8_ctx48"]["p50"],
             "k128_p50_ms": lat[m]["bf16/k_128_tools_ctx64"]["p50"] - lat[a.baseline]["bf16/k_128_tools_ctx64"]["p50"],
             "vram_mib": max(lat[m][k]["vram_mib"] - lat[a.baseline][k]["vram_mib"] for k in lat[m])} for m in a.seeds}
    g["latency_vram"] = {"values": d, "need": "typical within 1 ms, k128 within 2 ms, VRAM within 50 MiB of the same-host v1",
                         "pass": all(x["typical_p50_ms"] <= 1 and x["k128_p50_ms"] <= 2 and x["vram_mib"] <= 50 for x in d.values()),
                         "note": "one-sided (not slower); a candidate measured FASTER than v1 by more than the band is host noise, reported as such"}
    res = {"seeds": a.seeds, "candidate": a.candidate, "multimodal_run": a.mm, "gates": g, "ALL_PASS": all(x["pass"] for x in g.values())}
    (R3 / "gates.json").write_text(json.dumps(res, indent=1, default=str))
    pc = lambda x: f"{100 * x:.1f}"  # noqa: E731
    md = ["# R3 promotion gates (registered in configs/v11/r3_protocol.yaml before training)", "",
          f"Seeds: {', '.join(a.seeds)} · candidate seed: {a.candidate} · multimodal run: {a.mm or '—'}", "", "| gate | required | measured | result |", "|---|---|---|---|",
          f"| KrunTechBench v1 macro top-1 (raw) | mean ≥ 61.7 | {' / '.join(pc(x) for x in g['kruntechbench_v1_macro']['values'])} (mean {pc(g['kruntechbench_v1_macro']['mean'])}) | {'✅' if g['kruntechbench_v1_macro']['pass'] else '❌'} |",
          f"| unseen requirement combinations | mean ≥ {pc(r2['unseen_combination'] + 0.10)} (R2 {pc(r2['unseen_combination'])} + 10), each ≥ {pc(r2['unseen_combination'] + 0.05)} | {' / '.join(pc(x) for x in g['unseen_combination']['values'])} (mean {pc(g['unseen_combination']['mean'])}) | {'✅' if g['unseen_combination']['pass'] else '❌'} |",
          f"| constraint flips, both correct | mean ≥ {pc(r2['flips_both'] + 0.05)} (R2 {pc(r2['flips_both'])} + 5) | {' / '.join(pc(x) for x in g['constraint_flips_both']['values'])} (mean {pc(g['constraint_flips_both']['mean'])}) | {'✅' if g['constraint_flips_both']['pass'] else '❌'} |",
          f"| stuck rate | mean ≤ {pc(r2['stuck_rate'] - 0.05)} (R2 {pc(r2['stuck_rate'])} − 5) | {' / '.join(pc(x) for x in g['stuck_rate']['values'])} (mean {pc(g['stuck_rate']['mean'])}) | {'✅' if g['stuck_rate']['pass'] else '❌'} |",
          f"| hard-negative margin (logit) | mean ≥ {r2['margin_logit'] - 0.3:.2f} | {' / '.join(f'{x:+.2f}' for x in g['hard_negative_margin']['values'])} | {'✅' if g['hard_negative_margin']['pass'] else '❌'} |",
          f"| text preservation gates | 19 / 19 on every seed | {' · '.join(g['text_preservation']['values'])} (k = 128: {' / '.join(f'{x:+.2f}' for x in g['text_preservation']['k128_pp'])} pp) | {'✅' if g['text_preservation']['pass'] else '❌'} |",
          f"| multimodal gates, re-aligned fusion | 5 / 5 | {g['multimodal'].get('passed', 'not measured')} / 5 (old fusion: {g['multimodal'].get('old_fusion_passed', '—')} / 5) | {'✅' if g['multimodal']['pass'] else '❌'} |",
          f"| JevBench, all 231 (candidate − v1) | CI excludes 0, other seed above v1 | {100 * cand['delta']:+.1f} pp [{100 * cand['ci95'][0]:+.1f}, {100 * cand['ci95'][1]:+.1f}]; other seed {', '.join(f'{100 * o:+.1f}' for o in g['jevbench']['other_seeds_delta'])} pp | {'✅' if g['jevbench']['pass'] else '❌'} |",
          f"| latency / VRAM vs same-host v1 | ±1 ms typical, ±2 ms k128, +50 MiB | " + " · ".join(f"{m}: {x['typical_p50_ms']:+.2f} / {x['k128_p50_ms']:+.2f} ms, {x['vram_mib']:+d} MiB" for m, x in d.items()) + f" | {'✅' if g['latency_vram']['pass'] else '❌'} |",
          f"| **all gates** | | | {'✅' if res['ALL_PASS'] else '❌'} |"]
    (R3 / "gates.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
