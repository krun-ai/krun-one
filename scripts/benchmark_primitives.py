#!/usr/bin/env python
"""KRUN-011 serving benchmark of a multi-primitive artifact: end-to-end KrunOne.decide_questions latency.

    .venv/bin/python scripts/benchmark_primitives.py --artifact artifacts/krun-one-v0.3-candidate

Scenarios: 1 choice question; 1 noul; 1 score; mixed 3 (choice + noul + score); batch 8 and batch 32 questions
(mixed types, one request = one forward). Contexts / questions are drawn from data/primitives_v2 val (fixed seed).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

import krun_one  # noqa: F401
from krun_one import KrunOne

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = KrunOne.from_pretrained(args.artifact, device=dev)
    rng = random.Random(7)
    ctx = "Customer says this is the third time exports failed and wants a human immediately."
    choice = {"type": "choice", "options": ["billing", "support", "sales", "technical", "account"]}
    noul = {"type": "noul", "instructions": "Is the customer asking for human assistance?"}
    score = {"type": "score", "instructions": "How severe is the reported issue?",
             "levels": ["Minor issue", "Feature degraded", "Blocking issue"]}
    pool = [choice, noul, score]
    scen = {"choice_1": {"q": choice}, "noul_1": {"q": noul}, "score_1": {"q": score},
            "mixed_3": {"c": choice, "n": noul, "s": score},
            "batch_8": {f"q{i}": pool[i % 3] for i in range(8)}, "batch_32": {f"q{i}": pool[i % 3] for i in range(32)}}
    out = {"artifact": args.artifact, "device": dev, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None,
           "scenarios": {}}
    for name, qs in scen.items():
        for _ in range(3):
            m.decide_questions(ctx, qs)
        if dev == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            m.decide_questions(ctx, qs)
            if dev == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
        ts = np.array(ts)
        out["scenarios"][name] = {"questions": len(qs), "p50_ms": round(float(np.median(ts)), 2),
                                  "p95_ms": round(float(np.quantile(ts, 0.95)), 2),
                                  "questions_per_s": round(len(qs) / (float(np.median(ts)) / 1000), 1),
                                  "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20) if dev == "cuda" else None}
        print(name, out["scenarios"][name], flush=True)
    dst = ROOT / "results" / f"bench_primitives_{dev}.json"
    dst.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
