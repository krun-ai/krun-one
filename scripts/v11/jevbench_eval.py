"""Krun One on the PUBLIC JevBench items (external, evaluation only — data/v11/external/jevbench/README.md).

    python scripts/v11/jevbench_eval.py --ckpt v1 --tag v1 [--device cpu]
    python scripts/v11/jevbench_eval.py --report v1 r1_s1 r2na_s1      # table vs the published per-task outcomes

Mapping = the benchmark's own typed request (state, question {type, instructions, criteria}), 1:1 onto Krun primitives:
  choice  options = labels, description = criteria[label]        -> argmax over labels
  noul    proposition = instructions, criteria {true, false}     -> P(true) vs expected yes / no
  score   levels = criteria (ordered)                            -> argmax level index (the benchmark's "correct")
A dict state is sent as compact JSON. States longer than the 1,024-token sequence are truncated by the serializer, as
a production request would be; the truncated share is reported. Probabilities use the frozen v1 temperatures.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/v11"))
D = ROOT / "data/v11/external/jevbench"
OUT = ROOT / "results/v11/jevbench"
TIER = {"easy": "easy", "original": "standard", "hard": "hard"}


def items() -> list[dict]:
    out = []
    for f in ("easy", "original", "hard"):
        for line in open(D / f"{f}.jsonl", encoding="utf-8").read().split("\n"):
            if line:
                out.append({**json.loads(line), "tier": TIER[f]})
    return out


def encode(ser, it):
    q = it["question"]
    ctx = it["state"] if isinstance(it["state"], str) else json.dumps(it["state"], ensure_ascii=False, separators=(",", ":"))
    crit = q.get("criteria")
    if q["type"] == "choice":
        opts = [{"id": lab, **({"description": crit[lab]} if isinstance(crit, dict) and crit.get(lab) else {})} for lab in it["labels"]]
        return ser.encode("choice", ctx, opts, q["instructions"]), ("tool" if any("description" in o for o in opts) else "intent")
    if q["type"] == "noul":
        return ser.encode("noul", ctx, [{"id": "false"}, {"id": "true"}], q["instructions"], crit if isinstance(crit, dict) else None), "noul"
    levels = crit if isinstance(crit, list) else it["labels"]
    return ser.encode("score", ctx, [{"id": str(i), "label": str(lv)} for i, lv in enumerate(levels)], q["instructions"]), "score"


def run(ckpt: str, tag: str, device: str) -> None:
    import torch

    import eval_v11
    from krun_one.v11 import harness, probe

    dev = torch.device(device)
    model, _ = eval_v11.load(ckpt)
    model.to(dev).eval()
    ser = harness.serializer(eval_v11.ARTIFACT)
    T = json.loads((eval_v11.ARTIFACT / "calibration.json").read_text())["temperature"]
    its = items()
    enc = [encode(ser, it) for it in its]
    zs = harness.rank_logits(model, [e for e, _ in enc], ser.tok.pad_token_id, dev, max_tokens=8192)
    rows = []
    for it, (q, fam), z in zip(its, enc, zs):
        p = probe.softmax(z, T[fam])
        if it["question"]["type"] == "noul":
            probs = {"yes": float(p[1]), "no": float(p[0])}
        else:
            probs = {lab: float(x) for lab, x in zip(it["labels"], p)}
        pred = max(sorted(probs), key=lambda k: probs[k])
        rows.append({"id": it["id"], "tier": it["tier"], "family": it["family"], "type": it["question"]["type"], "predicted": pred,
                     "correct": pred == str(it["expected"]), "confidence": probs[pred], "probs": {k: round(v, 5) for k, v in probs.items()},
                     "tokens": harness.length(q), "chunks": len(q.chunks), "truncated": any(c.context_truncated for c in q.chunks)})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{tag}.json").write_text(json.dumps({"ckpt": ckpt, "device": device, "rows": rows}, indent=0))
    print(json.dumps({"tag": tag, "n": len(rows), "accuracy": float(np.mean([r["correct"] for r in rows])),
                      "truncated": float(np.mean([r["truncated"] for r in rows]))}))


def report(tags: list[str]) -> None:
    from krun_one.v11 import probe

    pub = json.loads((D / "per_task_v1.2.json").read_text())
    ids = {t["id"]: t for t in pub["tasks"] if t["public"]}
    ours = {t: {r["id"]: r for r in json.loads((OUT / f"{t}.json").read_text())["rows"]} for t in tags if (OUT / f"{t}.json").exists()}
    ref = ["jev-1.13.0", "gpt-5.6-luna", "semif-qwen3.5-4b", "laya", "gliner2.5-multi", "open-jev-deberta-v3-large"]
    res = {"n_public": len(ids), "systems": {}}

    def acc(outcome: dict, pred) -> float | None:
        v = [outcome[i] for i in ids if pred(ids[i]) and i in outcome]
        return float(np.mean(v)) if v else None

    fams = sorted({t["topic"] for t in ids.values() if t["tier"] == "hard"})
    for name in ref:
        o = {i: x[0] == "c" for i, x in pub["systems"][name]["public_tasks"].items()}
        res["systems"][pub["systems"][name]["display"]] = {"outcome": o, "kind": "published"}
    for t, rows in ours.items():
        res["systems"][f"Krun {t}"] = {"outcome": {i: r["correct"] for i, r in rows.items()}, "kind": "krun",
                                       "truncated_hard": float(np.mean([r["truncated"] for r in rows.values() if r["tier"] == "hard"])),
                                       "ece": probe.ece(np.array([r["confidence"] for r in rows.values()]), np.array([float(r["correct"]) for r in rows.values()]))}
    md = ["# JevBench public items (231) — external check, evaluation only", "",
          "Published rows are the benchmark's own per-task outcomes on the same public items (v1.2 artifact). "
          "Krun rows were run here with the mapping in scripts/v11/jevbench_eval.py. Chance differs per item (2 to 9 labels).", "",
          "| system | all 231 | easy (48) | standard (72) | hard (111) | " + " | ".join(fams) + " |", "|---|---:|---:|---:|---:|" + "---:|" * len(fams)]
    f = lambda v: "—" if v is None else f"{100 * v:.1f}"  # noqa: E731
    for name, s in res["systems"].items():
        o = s["outcome"]
        s["accuracy"] = {"all": acc(o, lambda t: True), **{tr: acc(o, lambda t, tr=tr: t["tier"] == tr) for tr in ("easy", "standard", "hard")},
                         "hard_families": {fm: acc(o, lambda t, fm=fm: t["tier"] == "hard" and t["topic"] == fm) for fm in fams}}
        a = s["accuracy"]
        md.append(f"| {name} | {f(a['all'])} | {f(a['easy'])} | {f(a['standard'])} | {f(a['hard'])} | " + " | ".join(f(a["hard_families"][fm]) for fm in fams) + " |")
        del s["outcome"]
    k = [s for n, s in res["systems"].items() if s["kind"] == "krun"]
    if k:
        md += ["", "Krun rows: " + "; ".join(f"{n}: hard items with a truncated state {100 * s['truncated_hard']:.0f}%, ECE {s['ece']:.3f}"
                                              for n, s in res["systems"].items() if s["kind"] == "krun") + "."]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "jevbench_public.json").write_text(json.dumps(res, indent=1))
    (OUT / "jevbench_public.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--tag")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--report", nargs="*")
    a = ap.parse_args()
    if a.report is not None:
        report(a.report)
    else:
        run(a.ckpt, a.tag, a.device)
