"""Train / evaluate one Krun One v1 proxy variant.

    uv run python scripts/v1/train_proxy.py --variant C_ans_head [--max-steps N] [--resume-from-checkpoint latest]
    uv run python scripts/v1/train_proxy.py --variant C_ans_head --smoke     # CPU, tiny random encoder, 40 decisions
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import torch
import yaml

from krun_one.checkpoint import resolve_resume
from krun_one.v1 import arch
from krun_one.v1.arch import DecisionSpec, EncoderSpec, ModelSpec
from krun_one.v1.model import KrunOneV1Model
from krun_one.v1.serialization import Budget, Serializer
from krun_one.v1.training import (dump_signals, encode_records, evaluate, load_mmbert_encoder, load_records,
                                  records_digest, train)

ROOT = Path(__file__).resolve().parents[2]
HUB = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
SMOKE_SPEC = ModelSpec("smoke", EncoderSpec(hidden_size=32, num_layers=3, num_heads=2, intermediate_size=48,
                                            local_window=16), DecisionSpec(head_layers=1, head_ffn_mult=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/v1/proxy.yaml"))
    ap.add_argument("--variant", required=True)
    ap.add_argument("--run-name")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--resume-from-checkpoint")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-n", type=int, default=40)
    ap.add_argument("--runs-dir", default=str(ROOT / "runs/v1"))
    ap.add_argument("--save-every", type=int)
    ap.add_argument("--seed", type=int, help="overrides config seed (bakeoff replicates)")
    ap.add_argument("--spec", default="proxy", choices=sorted(arch.SPECS), help="decision model shape")
    ap.add_argument("--encoder-init", help="HF MLM directory (e.g. a scaling milestone) instead of mmBERT-small")
    ap.add_argument("--lr-encoder", type=float)
    ap.add_argument("--lr-head", type=float)
    ap.add_argument("--data-dir", help="directory with train.jsonl / validation.jsonl (default: config data paths)")
    ap.add_argument("--gradient-checkpointing", action="store_true", help="memory only (same math); needed for K1 on 24 GB")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["loss"] = {**cfg["loss"], **cfg["variants"][args.variant]}
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.gradient_checkpointing:
        cfg["gradient_checkpointing"] = True
    if args.lr_encoder:
        cfg["optimizer"]["lr_encoder"] = args.lr_encoder
    if args.lr_head:
        cfg["optimizer"]["lr_head"] = args.lr_head
    if args.save_every:
        cfg["checkpoint"]["save_every"] = args.save_every
    from transformers import AutoTokenizer

    snap = sorted(glob.glob(str(HUB / "models--jhu-clsp--mmBERT-small/snapshots/*/tokenizer.json")))[0]
    tok = AutoTokenizer.from_pretrained(str(Path(snap).parent))
    sq = cfg["sequence"]
    ser = Serializer(tok, Budget(sq["max_len"], sq["head_max_len"], max_candidates_per_chunk=sq["max_candidates_per_chunk"]))
    train_path, val_path = ROOT / cfg["data"]["train"], ROOT / cfg["data"]["validation"]
    if args.data_dir:
        train_path, val_path = Path(args.data_dir) / "train.jsonl", Path(args.data_dir) / "validation.jsonl"
    spec = SMOKE_SPEC if args.smoke else arch.SPECS[args.spec]
    torch.manual_seed(cfg["seed"])
    model = KrunOneV1Model.from_spec(spec, attn_implementation="sdpa")
    init = {"encoder": "random (smoke)"}
    if not args.smoke:
        if args.encoder_init:
            w = str(Path(args.encoder_init))
            init = {"encoder": args.encoder_init, "weights": w, **load_mmbert_encoder(model, Path(w))}
        else:
            if args.spec != "proxy":
                raise SystemExit("--encoder-init is required for non-proxy specs")
            w = sorted(glob.glob(str(HUB / "models--jhu-clsp--mmBERT-small/snapshots/*/model.safetensors")))[0]
            init = {"encoder": cfg["init"]["encoder"], "weights": w, **load_mmbert_encoder(model, Path(w))}
    limit = args.smoke_n if args.smoke else None
    items = encode_records(ser, load_records(train_path, limit))
    val_records = load_records(val_path, limit)
    val_items = encode_records(ser, val_records)
    run = Path(args.runs_dir) / (args.run_name or (args.variant
                                                 + (f"_s{args.seed}" if args.seed is not None else "")))
    run.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        cfg["batching"] = {"max_tokens": 4096, "max_questions": 8}
    extra = {"variant": args.variant, "spec": spec.to_dict(), "init": {k: v for k, v in init.items() if k != "weights"},
             "sequence": sq, "train_sha256": records_digest(train_path) if not args.smoke else "smoke",
             "n_train_limit": limit}
    res = train(model, items, tok.pad_token_id, cfg, run, resolve_resume(args.resume_from_checkpoint, run),
                args.max_steps, device="cpu" if args.smoke else None, run_config_extra=extra)
    (run / "train_result.json").write_text(json.dumps({k: v for k, v in res.items()}, indent=2, default=str))
    if res["completed"] or args.smoke:
        ev = evaluate(model, val_items, tok.pad_token_id, device="cpu" if args.smoke else None)
        (run / "eval_validation.json").write_text(json.dumps(ev, indent=2, default=str))
        print(json.dumps(ev, indent=2, default=str))
        n = dump_signals(model, val_items, val_records, tok.pad_token_id, run / "val_signals.jsonl.gz",
                         device="cpu" if args.smoke else None)
        print(f"dumped {n} validation decisions -> {run / 'val_signals.jsonl.gz'}")
        if res["completed"] and not args.smoke:
            from safetensors.torch import save_model
            (run / "final").mkdir(exist_ok=True)
            save_model(model, str(run / "final/model.safetensors"))


if __name__ == "__main__":
    main()
