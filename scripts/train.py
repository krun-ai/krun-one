#!/usr/bin/env python
"""Fine-tune the Laya checkpoint into Krun One (config A or B).

    .venv/bin/python scripts/train.py --variant A                                   # official run -> runs/A/
    .venv/bin/python scripts/train.py --variant A --resume-from-checkpoint latest   # continue after a crash
    .venv/bin/python scripts/train.py --variant B --run-name smoke_B --max-steps 60 --save-every 20 --log-every 10

Outputs: runs/<run>/checkpoint-<step>/ (every save_every steps, keep_last), runs/<run>/final/model.safetensors,
results/[_scratch/]train_<run>.json (config, timings, GPU, log, trainable parameter counts).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import yaml

import krun_one  # noqa: F401  (env var before torch)
import torch
from safetensors.torch import save_model

from krun_one.checkpoint import resolve_resume
from krun_one.dataset import encode_for_model, load_examples
from krun_one.laya_adapter import laya_provenance, load_laya
from krun_one.training import train

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]


def encoded_items(train_file: Path, tok, max_len: int, head: int, limit: int = 0) -> tuple[list[dict], list[str]]:
    key = hashlib.sha256(f"{train_file.stat().st_size}:{train_file.stat().st_mtime_ns}:{max_len}:{head}".encode()).hexdigest()[:16]
    cache = train_file.with_name(f"{train_file.stem}.encoded-{key}.pt")
    if cache.exists():
        d = torch.load(cache, weights_only=False)
        items, dropped = d["items"], d["dropped"]
    else:
        t0 = time.time()
        items, dropped = encode_for_model(load_examples(train_file), tok, max_len, head)
        torch.save({"items": items, "dropped": dropped}, cache)
        print(f"encoded {len(items)} decisions in {time.time() - t0:.0f}s (dropped {len(dropped)})", flush=True)
    if limit:
        items = items[:: max(1, len(items) // limit)][:limit]
    return items, dropped


def gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"device": "cpu"}
    p = torch.cuda.get_device_properties(0)
    return {"device": "cuda", "gpu": p.name, "vram_mib": p.total_memory // 2**20, "capability": f"{p.major}.{p.minor}",
            "torch": torch.__version__, "cuda": torch.version.cuda}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "krun_one_v0.yaml"))
    ap.add_argument("--variant", required=True, choices=["A", "B"])
    ap.add_argument("--run-name", default=None, help="default: the variant name (official run)")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--train-file", default=None)
    ap.add_argument("--resume-from-checkpoint", default=None, help="a checkpoint dir, or 'latest'")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="smoke: train on N decisions only")
    ap.add_argument("--deterministic", action="store_true",
                    help="bitwise-reproducible GPU kernels (resume validation only; slower): deterministic algorithms, math SDPA")
    ap.add_argument("--gradient-checkpointing", action="store_true", help="memory only (same math)")
    args = ap.parse_args()
    if args.deterministic:
        import os

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.save_every:
        cfg = copy.deepcopy(cfg)
        cfg["checkpoint"]["save_every"] = args.save_every
    if args.gradient_checkpointing:
        cfg = copy.deepcopy(cfg)
        cfg["gradient_checkpointing"] = True
    run = args.run_name or args.variant
    official = args.run_name is None
    run_dir = ROOT / "runs" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir) if args.data_dir else PROJECT / cfg["data"]["dir"]
    train_file = Path(args.train_file) if args.train_file else data_dir / cfg["data"]["train_file"]

    model, tok, laya_cfg = load_laya(device="cpu")
    items, dropped = encoded_items(train_file, tok, cfg["sequence"]["max_len"], cfg["sequence"]["head_max_len"], args.limit)
    print(f"train decisions: {len(items)} (dropped, options did not fit: {len(dropped)})", flush=True)
    resume = resolve_resume(args.resume_from_checkpoint, run_dir)
    t0 = time.time()
    res = train(model, items, tok.pad_token_id, cfg, args.variant, run_dir, resume, args.max_steps, args.log_every)
    wall = time.time() - t0
    out = {"run": run, "variant": args.variant, "official": official, "config_file": str(Path(args.config).relative_to(PROJECT)),
           "config": cfg, "train_file": str(train_file.relative_to(PROJECT)) if train_file.is_relative_to(PROJECT) else str(train_file),
           "dropped_decisions": len(dropped), "deterministic": args.deterministic, "resumed_from": str(resume) if resume else None,
           "init": laya_provenance(with_hash=False), "hardware": gpu_info(), "wall_seconds_this_process": wall,
           "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **res}
    if res["completed"]:
        (run_dir / "final").mkdir(exist_ok=True)
        save_model(model, str(run_dir / "final" / "model.safetensors"))
        out["final_model"] = str((run_dir / "final" / "model.safetensors").relative_to(ROOT))
    res_dir = ROOT / "results" if official else ROOT / "results" / "_scratch"
    res_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if res["completed"] else "_partial"
    (res_dir / f"train_{run}{suffix}.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(f"done: step {res['global_step']}/{res['total_steps']} in {wall:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
