#!/usr/bin/env python
"""KRUN-011: continue fine-tuning Krun One v0 on a choice + noul + score mixture (configs/krun_one_v0_2.yaml).

    .venv/bin/python scripts/train_primitives.py --variant M50_CE                          # official -> runs/v02_M50_CE/
    .venv/bin/python scripts/train_primitives.py --variant M50_CE --resume-from-checkpoint latest
    .venv/bin/python scripts/train_primitives.py --variant M50_CE --run-name smoke --max-steps 40 --save-every 20

The mixture is a pure function of (config, variant, pools): seeded uniform samples of each pool, sizes =
round(total_decisions × fraction). Its id list is hashed into the run config (resume refuses another mixture).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

import krun_one  # noqa: F401
import torch
from safetensors.torch import load_file, save_model

from krun_one.checkpoint import resolve_resume
from krun_one.dataset import encode_for_model, load_examples
from krun_one.laya_adapter import load_laya, sha256_file
from krun_one.primitive_eval import encode_records, load_jsonl
from krun_one.training import train

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train import gpu_info  # noqa: E402


def sample(n_pool: int, n: int, seed: int, tag: str) -> list[int]:
    g = np.random.Generator(np.random.PCG64(int(hashlib.sha256(f"{seed}:{tag}".encode()).hexdigest()[:15], 16)))
    if n > n_pool:
        raise SystemExit(f"{tag}: mixture needs {n} but the pool has {n_pool}")
    return sorted(g.permutation(n_pool)[:n].tolist())


def build_items(cfg: dict, variant: str, tok, limit: int = 0) -> tuple[list[dict], dict]:
    mix = cfg["configs"][variant]["mixture"]
    total = cfg["total_decisions"]
    sizes = {k: int(round(total * v)) for k, v in mix.items()}
    ml, hl = cfg["sequence"]["max_len"], cfg["sequence"]["head_max_len"]
    key = hashlib.sha256(json.dumps([sizes, cfg["seed"], ml, hl, cfg["data"]], sort_keys=True).encode()).hexdigest()[:16]
    cache = PROJECT / cfg["data"]["primitives_dir"] / f"mixture-{key}.pt"
    if cache.exists():
        d = torch.load(cache, weights_only=False)
    else:
        t0 = time.time()
        choice = load_examples(PROJECT / cfg["data"]["choice_file"])
        keep = cfg["data"].get("choice_include_all")  # v0.3: e.g. every intent core hard_v1 decision is replayed
        if keep:
            forced = [i for i, e in enumerate(choice) if e.task == keep["task"] and e.suite == keep["suite"]
                      and (e.metadata or {}).get("negatives") == keep["negatives"]]
            fset = set(forced)
            rest = [i for i in range(len(choice)) if i not in fset]
            idx = sorted(forced + [rest[j] for j in sample(len(rest), sizes["choice"] - len(forced), cfg["seed"], "choice-rest")])
        else:
            idx = sample(len(choice), sizes["choice"], cfg["seed"], "choice")
        sel = [choice[i] for i in idx]
        items, dropped = encode_for_model(sel, tok, ml, hl)
        for it in items:
            it["qtype"] = 0
        stats = {"choice": {"n": len(items), "dropped": len(dropped),
                            "by_suite": dict(Counter(e.suite for e in sel)), "by_task": dict(Counter(e.task for e in sel))}}
        for q in ("noul", "score"):
            pool = load_jsonl(PROJECT / cfg["data"]["primitives_dir"] / f"{q}_train.jsonl")
            recs = [pool[i] for i in sample(len(pool), sizes[q], cfg["seed"], q)]
            enc, unfit = encode_records(tok, recs, ml, hl)
            by_id = {r["id"]: r for r in recs}
            for it in enc:
                r = by_id[it["id"]]
                k = len(it["markers"])
                it.update({"label": r["target"], "answerable": True, "task": q,
                           "target": [1.0 if i == r["target"] else 0.0 for i in range(k)]})
            items += enc
            stats[q] = {"n": len(enc), "dropped": len(unfit), "by_source": dict(Counter(r["source"] for r in recs)),
                        "by_suite": dict(Counter(f"{r['source']}:{r['suite']}" for r in recs)),
                        "by_target": dict(Counter(r["target"] for r in recs))}
        stats["tokens"] = int(sum(len(it["ids"]) for it in items))
        stats["tokens_by_primitive"] = {q: int(sum(len(it["ids"]) for it in items if it["task"] == q or
                                                   (q == "choice" and it["qtype"] == 0))) for q in ("choice", "noul", "score")}
        d = {"items": items, "stats": stats, "sizes": sizes}
        torch.save(d, cache)
        print(f"encoded mixture {sizes} in {time.time() - t0:.0f}s -> {cache.name}", flush=True)
    items = d["items"]
    if limit:
        items = items[:: max(1, len(items) // limit)][:limit]
    return items, {"sizes": d["sizes"], "stats": d["stats"], "cache": cache.name}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "krun_one_v0_2.yaml"))
    ap.add_argument("--variant", required=True)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume-from-checkpoint", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--build-only", action="store_true", help="encode + cache the mixture, no training")
    args = ap.parse_args()
    if args.deterministic:
        import os

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    cfg = copy.deepcopy(yaml.safe_load(Path(args.config).read_text()))
    if args.variant not in cfg["configs"]:
        raise SystemExit(f"unknown variant {args.variant}; have {list(cfg['configs'])}")
    cfg["objective"]["score_rps_weight"] = cfg["configs"][args.variant]["score_rps_weight"]
    for k in ("lr_encoder", "lr_head"):  # v0.3 variants may override the learning rates
        if k in cfg["configs"][args.variant]:
            cfg["optimizer"][k] = cfg["configs"][args.variant][k]
    if args.save_every:
        cfg["checkpoint"]["save_every"] = args.save_every
    if args.gradient_checkpointing:
        cfg["gradient_checkpointing"] = True
    run = args.run_name or f"{cfg.get('run_prefix', 'v02')}_{args.variant}"
    official = args.run_name is None
    run_dir = ROOT / "runs" / run
    run_dir.mkdir(parents=True, exist_ok=True)

    model, tok, _ = load_laya(device="cpu")
    init = PROJECT / cfg["init"]["checkpoint"]
    model.load_state_dict(load_file(str(init)), strict=True)
    init_state = {"checkpoint": cfg["init"]["checkpoint"], "sha256": sha256_file(init)}
    items, mix = build_items(cfg, args.variant, tok, args.limit)
    print(f"mixture: {mix['sizes']} -> {len(items)} items; {json.dumps(mix['stats'])[:600]}", flush=True)
    if args.build_only:
        return 0
    resume = resolve_resume(args.resume_from_checkpoint, run_dir)
    t0 = time.time()
    res = train(model, items, tok.pad_token_id, cfg, args.variant, run_dir, resume, args.max_steps, args.log_every,
                init_state=init_state)
    wall = time.time() - t0
    out = {"run": run, "variant": args.variant, "official": official, "config": cfg, "mixture": mix,
           "init": init_state, "deterministic": args.deterministic, "resumed_from": str(resume) if resume else None,
           "hardware": gpu_info(), "wall_seconds_this_process": wall,
           "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **res}
    if res["completed"]:
        (run_dir / "final").mkdir(exist_ok=True)
        save_model(model, str(run_dir / "final" / "model.safetensors"))
        out["final_model"] = str((run_dir / "final" / "model.safetensors").relative_to(ROOT))
    res_dir = ROOT / "results" / "primitives_candidates" if official else ROOT / "results" / "_scratch"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / f"train_{run}{'' if res['completed'] else '_partial'}.json").write_text(
        json.dumps(out, indent=2, default=str) + "\n")
    print(f"done: step {res['global_step']}/{res['total_steps']} in {wall:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
