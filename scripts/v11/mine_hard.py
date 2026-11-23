"""Krun One v1.1 — hard-negative mining for the second mini-round (configs/v11/r1_r2_protocol.yaml: hard_negative_round).

    python scripts/v11/mine_hard.py --dump results/v11/eval/r2_s1/tech_train_tech.jsonl.gz --tag s1

Reads the R2 checkpoint's own logits on the technical TRAINING corpus (never on KrunTechBench / KrunTechVal) and keeps
the answerable choice examples that are (a) wrong with top-1 probability >= 0.5, (b) right or wrong with a
gold-vs-hardest-negative logit margin < 1.0, or (c) normalized entropy > 0.5. Writes the mined examples (bucket
hard_negative) and an equal-size seeded sample of the replay file.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from krun_one.v11 import probe, training as T  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cap", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=20261009)
    args = ap.parse_args()
    ex = {e["id"]: e for e in T.load_jsonl(ROOT / "data/v11/corpus/train_tech.jsonl")}
    why, picked = Counter(), []
    n = 0
    with gzip.open(ROOT / args.dump, "rt") as f:
        for line in f:
            r = json.loads(line)
            e = ex.get(r["id"])
            if e is None or e.get("qtype", "choice") != "choice" or not e["answerable"]:
                continue
            n += 1
            s = probe.item_signals(e, r["z"], 1.0, 0.0)
            reasons = [k for k, c in (("confident_wrong", (not s["correct"]) and s["top1_p"] >= 0.5), ("small_margin", s["margin_logit"] < 1.0),
                                      ("high_entropy", s["entropy"] > 0.5)) if c]
            if reasons:
                for k in reasons:
                    why[k] += 1
                picked.append((s["margin_logit"], e["id"], reasons))
    picked.sort(key=lambda x: x[0])                       # the worst margins first when the cap binds
    rng = random.Random(args.seed)
    chosen = picked[: args.cap]
    out = ROOT / "data/v11/corpus"
    with open(out / f"hn_{args.tag}_tech.jsonl", "w") as f:
        for _, i, reasons in chosen:
            f.write(json.dumps({**ex[i], "bucket": "hard_negative", "mined": reasons}, ensure_ascii=False) + "\n")
    rep = [x for x in open(out / "train_replay.jsonl", encoding="utf-8").read().split("\n") if x]
    with open(out / f"hn_{args.tag}_replay.jsonl", "w") as f:
        f.write("\n".join(rng.sample(rep, min(len(rep), len(chosen)))) + "\n")
    info = {"scored": n, "mined": len(picked), "kept": len(chosen), "reasons": dict(why),
            "by_slice": dict(Counter(ex[i]["slice"] for _, i, _ in chosen).most_common()),
            "by_render": dict(Counter(ex[i]["render"] for _, i, _ in chosen)),
            "train_accuracy": None}
    (out / f"hn_{args.tag}_manifest.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
