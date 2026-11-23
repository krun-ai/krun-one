"""Krun One v0.3 (production) on the SAME V files as the v1 candidate, in the v1 signal format (training.dump_signals),
so scripts/v1/kd1_evaluate.py compares both with one code path.

v0.3 is the production artifact (artifacts/krun-one-v0.3: Laya-architecture DecisionModel, choice/noul/score). It has no
multi-label output: multi decisions are skipped (reported as n/a). Choice uses v0.3's own serving path (KrunOne.encode:
Laya even-shrinking for many options), noul/score use encode_question — exactly what production does.

    python scripts/v1/dump_signals_v03.py --out runs/v1/ft/v03_baseline [--limit N]
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from krun_one.model import KrunOne
from krun_one.primitives import encode_question

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=str(ROOT / "artifacts/krun-one-v0.3"))
    ap.add_argument("--config", default=str(ROOT / "configs/v1/finetune.yaml"))
    ap.add_argument("--out", default=str(ROOT / "runs/v1/ft/v03_baseline"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    k = KrunOne.from_pretrained(args.artifact)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for rel in cfg["data"]["validation"]:
        f = ROOT / rel
        if "final_protected" in f.name:
            raise SystemExit("refusing a protected file")
        recs = [json.loads(l) for l in open(f)]
        if args.limit:
            recs = recs[: args.limit]
        items, keep, skipped = [], [], 0
        for r in recs:
            t = r["task"]["type"]
            try:
                if t == "choice":
                    names = [c["id"] for c in r["candidates"]]
                    desc = {c["id"]: c.get("description") for c in r["candidates"]}
                    it = k.encode(r["context"]["text"], names, desc, r["task"].get("instructions"))
                elif t in ("noul", "score"):
                    q = {"type": t, "instructions": r["task"]["instructions"]}
                    if t == "noul" and r["task"].get("criteria"):
                        q["criteria"] = r["task"]["criteria"]
                    if t == "score":
                        q["levels"] = [c.get("label") or c["id"] for c in r["candidates"]]
                    it = encode_question(k.tok, r["context"]["text"], q, k.cfg["max_len"], k.cfg["head_max_len"])
                else:
                    skipped += 1
                    continue
            except ValueError:
                skipped += 1
                continue
            items.append(it)
            keep.append(r)
        order = np.argsort([len(it["ids"]) for it in items])
        z_all: list = [None] * len(items)
        for s in range(0, len(order), args.batch):
            idx = order[s:s + args.batch]
            for i, z in zip(idx, k.logits([items[i] for i in idx])):
                z_all[i] = z
        path = out / f"signals_{f.stem}.jsonl.gz"
        with gzip.open(path, "wt") as fh:
            for r, z in zip(keep, z_all):
                ids = [c["id"] for c in r["candidates"]]
                gen = (r.get("metadata") or {}).get("v0_generation") or {}
                row = {"id": r["id"], "type": r["task"]["type"], "family": r["task"].get("family"),
                       "lang": r["context"]["language"], "slices": r.get("slices") or [], "difficulty": r.get("difficulty"),
                       "answerable": r["answerable"], "k": len(ids),
                       "gold": [ids.index(l) for l in r["target"]["labels"]], "split_group_id": r["split_group_id"],
                       "consistency_group": gen.get("consistency_group"), "candidate_ids": ids,
                       "rank": [round(float(x), 5) for x in z[: len(ids)]]}
                fh.write(json.dumps(row) + "\n")
        print(json.dumps({"file": f.name, "n": len(keep), "skipped": skipped, "out": str(path)}), flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
