#!/usr/bin/env python
"""Serving benchmark: end-to-end ``decide_batch`` latency (tokenization + forward + post-processing).

    .venv/bin/python scripts/benchmark_serving.py --model artifacts/krun-one-v0 --tag krun_one_v0 --device cuda
    .venv/bin/python scripts/benchmark_serving.py --model laya --tag laya --device cuda
    .venv/bin/python scripts/benchmark_serving.py --model artifacts/krun-one-v0 --tag krun_one_v0 --device cpu --threads 6 --batches 1,8

Requests: a fixed mix of real validation decisions (half intent, half tool; 4-16 options), identical for every model.
GPU: bf16 autocast on Ampere+ (fp16 otherwise); CPU: fp32. Reports p50/p95/mean latency per batch, decisions/s,
peak VRAM (allocated/reserved) and process RSS.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import krun_one  # noqa: F401
import psutil
import torch

from krun_one.model import KrunOne

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]


def requests(n: int = 64) -> list[dict]:
    from krunbench.schema import read_jsonl

    tools = [e for e in read_jsonl(PROJECT / "experiments/krun_one/data/v0/val_tools.jsonl")
             if e.suite == "core" and 4 <= len(e.options) <= 16][: n // 2]
    intents = [e for e in read_jsonl(PROJECT / "benchmarks/krunbench/data/v0/banking77/validation.jsonl")
               if e.suite == "core" and len(e.options) >= 4][: n // 2]
    out = []
    for a, b in zip(intents, tools):
        out.append({"context": a.context, "options": list(a.options), "instructions": a.instructions})
        out.append({"context": b.context, "options": [{"name": o, "description": b.option_descriptions[o]} for o in b.options],
                    "instructions": b.instructions})
    return out


def load(spec: str, device: str) -> KrunOne:
    if spec == "laya":
        from krun_one.laya_adapter import load_laya

        model, tok, cfg = load_laya(device=device)
        kcfg = {"max_len": 1024, "head_max_len": 896, "head_layers": cfg["head_layers"],
                "temperature": {"intent": 1.0, "tool": 1.0, "default": 1.0},
                "abstention": {"score": "top1_probability", "threshold": {"intent": 0.0, "tool": 0.0}}}
        return KrunOne(model.eval(), tok, kcfg, torch.device(device))
    return KrunOne.from_pretrained(spec if Path(spec).is_absolute() else ROOT / spec, device=device)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--dtype", default=None, choices=[None, "bf16", "fp16", "fp32"])
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    t0 = time.time()
    m = load(args.model, args.device)
    if args.dtype:
        m.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    if args.device == "cpu":
        m.dtype = torch.float32
    load_s = time.time() - t0
    reqs = requests()
    res = {"tag": args.tag, "model": args.model, "device": args.device, "dtype": str(m.dtype), "load_seconds": round(load_s, 2),
           "torch_threads": torch.get_num_threads(), "n_request_pool": len(reqs),
           "parameters": sum(p.numel() for p in m.model.parameters()),
           "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None, "batches": {}}
    if args.device == "cpu":
        from bakeoff.repro import cpu_name

        res["cpu"] = cpu_name()
    for bs in [int(x) for x in args.batches.split(",")]:
        if args.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        lat = []
        for r in range(args.warmup + args.runs):
            batch = [reqs[(r * bs + j) % len(reqs)] for j in range(bs)]
            if args.device == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            m.decide_batch(batch)
            if args.device == "cuda":
                torch.cuda.synchronize()
            if r >= args.warmup:
                lat.append((time.perf_counter() - t) * 1000)
        a = np.array(lat)
        e = {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)), "mean_ms": float(a.mean()),
             "decisions_per_s": float(bs * 1000 / a.mean()), "runs": args.runs}
        if args.device == "cuda":
            e["peak_vram_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
            e["peak_vram_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20)
        e["rss_mib"] = round(proc.memory_info().rss / 2**20)
        res["batches"][str(bs)] = e
        print(f"  bs={bs:3d} p50={e['p50_ms']:.1f} ms p95={e['p95_ms']:.1f} ms {e['decisions_per_s']:.1f} dec/s "
              f"vram={e.get('peak_vram_allocated_mib')} rss={e['rss_mib']}", flush=True)
    res["rss_mib_model_loaded_delta"] = round((proc.memory_info().rss - rss0) / 2**20)
    res["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dst = ROOT / "results" / f"bench_{args.device}_{args.tag}.json"
    dst.write_text(json.dumps(res, indent=2) + "\n")
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
