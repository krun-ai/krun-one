"""JevBench public items — R3 external-transfer report (evaluation only; aggregates only).

    uv run python scripts/v11/jevbench_report_r3.py --baseline v1_r3host --models r2l4na_s1 r2l4na_s2 r3f_s1 r3f_s2

Reads results/v11/jevbench/<tag>.json (scripts/v11/jevbench_eval.py) and results/v11/jevbench/lengths.json
(scripts/v11/jevbench_lengths.py). Writes results/v11/r3/jevbench/jevbench_r3.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
J = ROOT / "results/v11/jevbench"
OUT = ROOT / "results/v11/r3/jevbench"
BUCKETS = ["0-512", "513-1024", "1025-2048", "2049-4096"]
# benchmark families mapped onto the R3 capability slices (by the benchmark's own category names; no item was read for this)
CAP = {"long_policy": ["long_policy"], "multi_hop": ["multi_hop"], "numeric + temporal": ["temporal_numeric"], "tradeoff": ["tradeoff"],
       "not targeted (judgment, traps, ambiguity, probability, routing)": ["judge_hard", "trap", "ambiguous", "adversarial", "probability", "routing_hard"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="v1_r3host")
    ap.add_argument("--models", nargs="+", required=True)
    a = ap.parse_args()
    L = json.loads((J / "lengths.json").read_text())["items"]
    pub = json.loads((ROOT / "data/v11/external/jevbench/per_task_v1.2.json").read_text())
    sysm = {}
    for name, key in (("Jev 1.13.0", "jev-1.13.0"), ("GPT-5.6 Luna", "gpt-5.6-luna"), ("Laya 421M", "laya")):
        sysm[name] = {i: x[0] == "c" for i, x in pub["systems"][key]["public_tasks"].items()}
    ours = [a.baseline] + [m for m in a.models if (J / f"{m}.json").exists()]
    for t in ours:
        sysm[f"Krun {t}"] = {r["id"]: bool(r["correct"]) for r in json.loads((J / f"{t}.json").read_text())["rows"]}
    cuts = {"all 231": lambda x: True, "easy": lambda x: x["tier"] == "easy", "standard": lambda x: x["tier"] == "standard", "hard": lambda x: x["tier"] == "hard",
            **{b: (lambda x, b=b: x["bucket"] == b) for b in BUCKETS}, "<= 1024": lambda x: not x["truncated"], "> 1024": lambda x: x["truncated"],
            "hard <= 1024": lambda x: x["tier"] == "hard" and not x["truncated"], "hard > 1024": lambda x: x["tier"] == "hard" and x["truncated"],
            **{f"type {t}": (lambda x, t=t: x["type"] == t) for t in ("choice", "noul", "score")},
            **{f"hard family {t}": (lambda x, t=t: x["tier"] == "hard" and x["topic"] == t) for t in sorted({x["topic"] for x in L.values() if x["tier"] == "hard"})},
            **{f"capability {c}": (lambda x, ts=ts: x["tier"] == "hard" and x["topic"] in ts) for c, ts in CAP.items()},
            "lost 1-1000 tokens": lambda x: 0 < x["lost_tokens"] <= 1000, "lost 1001-2000 tokens": lambda x: 1000 < x["lost_tokens"] <= 2000,
            "lost > 2000 tokens": lambda x: x["lost_tokens"] > 2000}

    def acc(o, f):
        v = [o[i] for i in L if f(L[i]) and i in o]
        return (float(np.mean(v)), len(v)) if v else (None, 0)

    res = {"n": len(L), "truncated_share": float(np.mean([x["truncated"] for x in L.values()])),
           "truncated_share_hard": float(np.mean([x["truncated"] for x in L.values() if x["tier"] == "hard"])),
           "median_lost_tokens_when_truncated": float(np.median([x["lost_tokens"] for x in L.values() if x["truncated"]])),
           "systems": {n: {c: acc(o, f)[0] for c, f in cuts.items()} for n, o in sysm.items()}, "n_per_cut": {c: acc(sysm["Jev 1.13.0"], f)[1] for c, f in cuts.items()},
           "paired_vs_baseline": {}}
    base = sysm[f"Krun {a.baseline}"]
    rng = np.random.default_rng(0)
    for t in ours[1:]:
        o = sysm[f"Krun {t}"]
        res["paired_vs_baseline"][t] = {}
        for c in ("all 231", "hard", "<= 1024", "> 1024", "hard <= 1024"):
            ids = [i for i in L if cuts[c](L[i])]
            d = np.array([float(o[i]) - float(base[i]) for i in ids])
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
            res["paired_vs_baseline"][t][c] = {"n": len(ids), "delta": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                              "gained": int((d > 0).sum()), "lost": int((d < 0).sum())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "jevbench_r3.json").write_text(json.dumps(res, indent=1))
    f = lambda v: "—" if v is None else f"{100 * v:.1f}"  # noqa: E731
    names = list(sysm)
    md = ["# JevBench public items (231) — R3 external-transfer report (evaluation only)", "",
          f"Sequence limit 1,024 tokens: {100 * res['truncated_share']:.0f}% of all items and {100 * res['truncated_share_hard']:.0f}% of the hard tier are truncated "
          f"(median {res['median_lost_tokens_when_truncated']:.0f} tokens lost). Length = tokens of the full request with Krun's tokenizer.", "",
          "| cut | n | " + " | ".join(names) + " |", "|---|---:|" + "---:|" * len(names)]
    for c in cuts:
        md.append(f"| {c} | {res['n_per_cut'][c]} | " + " | ".join(f(res["systems"][n][c]) for n in names) + " |")
    md += ["", f"## Paired difference vs Krun {a.baseline} (bootstrap 95% CI; items gained / lost)", "", "| model | all 231 | hard | <= 1024 | > 1024 | hard <= 1024 |", "|---|---|---|---|---|---|"]
    cell = lambda x: f"{100 * x['delta']:+.1f} pp [{100 * x['ci95'][0]:+.1f}, {100 * x['ci95'][1]:+.1f}] (+{x['gained']} / -{x['lost']})"  # noqa: E731
    for t, p in res["paired_vs_baseline"].items():
        md.append(f"| {t} | " + " | ".join(cell(p[c]) for c in ("all 231", "hard", "<= 1024", "> 1024", "hard <= 1024")) + " |")
    (OUT / "jevbench_r3.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
