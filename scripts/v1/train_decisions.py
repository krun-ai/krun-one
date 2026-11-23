"""Krun One v1 — full fine-tune of K1-W1536-L22 on Krun Decisions v1 (configs/v1/finetune.yaml).

    python scripts/v1/train_decisions.py smoke                         # CPU, tiny random encoder, 64 decisions
    python scripts/v1/train_decisions.py lr_smoke --init-dir <milestone-0 dir>
    python scripts/v1/train_decisions.py train --init-dir <dir> [--lr-encoder X] [--resume-from-checkpoint latest]
    python scripts/v1/train_decisions.py eval --run runs/v1/ft/<name>   # V only (never final_protected)

Checkpoints (atomic, every `save_every_minutes`): model, optimizer, scheduler, GradScaler, python/numpy/torch CPU+CUDA
RNG, curriculum shuffle state, trainer state. The saved run config holds the sha256 of the training file, of the init
weights and of this config; resume refuses any mismatch.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch
import yaml

from krun_one.checkpoint import resolve_resume
from krun_one.v1 import arch
from krun_one.v1.arch import DecisionSpec, EncoderSpec, ModelSpec
from krun_one.v1.model import KrunOneV1Model
from krun_one.v1.serialization import Budget, Serializer
from krun_one.v1.training import (dump_signals, encode_records, encode_records_parallel, evaluate, load_mmbert_encoder,
                                  load_records, records_digest, train)

ROOT = Path(__file__).resolve().parents[2]
HUB = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
SMOKE_SPEC = ModelSpec("smoke", EncoderSpec(hidden_size=32, num_layers=3, num_heads=2, intermediate_size=48,
                                            local_window=16), DecisionSpec(head_layers=1, head_ffn_mult=2))


def tokenizer():
    from transformers import AutoTokenizer

    snap = sorted(glob.glob(str(HUB / "models--jhu-clsp--mmBERT-small/snapshots/*/tokenizer.json")))
    if not snap:
        from huggingface_hub import snapshot_download

        return AutoTokenizer.from_pretrained(snapshot_download("jhu-clsp/mmBERT-small", allow_patterns=["tokenizer*", "*.json"]))
    return AutoTokenizer.from_pretrained(str(Path(snap[0]).parent))


def cfg_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def weights_sha(init_dir: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(Path(init_dir).glob("model*.safetensors")):
        with open(f, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 24), b""):
                h.update(b)
    return h.hexdigest()


def build_model(spec_name: str, init_dir: Path | None, smoke: bool):
    spec = SMOKE_SPEC if smoke else arch.SPECS[spec_name]
    model = KrunOneV1Model.from_spec(spec, attn_implementation="sdpa")
    init = {"encoder": "random (smoke)"}
    if not smoke:
        init = {"encoder": str(init_dir), **load_mmbert_encoder(model, init_dir)}
    return model, spec, init


def val_files(cfg) -> list[Path]:
    fs = [ROOT / p for p in cfg["data"]["validation"]]
    for f in fs:
        if "final_protected" in f.name:
            raise SystemExit("final_protected is never used for training / selection / tuning")
    return fs


def run_eval(model, ser, pad_id, cfg, run: Path, device=None, limit=None) -> dict:
    """Signals + summary on every V file (KrunBench v1 val_fit / val_eval, val_ml)."""
    out = {}
    for f in val_files(cfg):
        recs = load_records(f, limit)
        items = encode_records(ser, recs)
        ev = evaluate(model, items, pad_id, device=device)
        n = dump_signals(model, items, recs, pad_id, run / f"signals_{f.stem}.jsonl.gz", device=device)
        out[f.stem] = {**ev, "signals": n}
        print(f.stem, json.dumps({k: v for k, v in ev.items() if not isinstance(v, dict)}), flush=True)
    (run / "eval_validation.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def prepare(args, cfg, smoke=False):
    tok = tokenizer()
    sq = cfg["sequence"]
    ser = Serializer(tok, Budget(sq["max_len"], sq["head_max_len"], max_candidates_per_chunk=sq["max_candidates_per_chunk"]))
    return tok, ser


def cmd_train(args, cfg, smoke=False, subset: int | None = None, run_name: str | None = None, eval_limit=None) -> dict:
    tok, ser = prepare(args, cfg, smoke)
    train_path = ROOT / cfg["data"]["train"]
    run = ROOT / "runs/v1/ft" / (run_name or args.run_name or cfg["name"])
    run.mkdir(parents=True, exist_ok=True)
    lines = [l for l in open(train_path, encoding="utf-8").read().split("\n") if l]   # NOT splitlines(): JSON strings may hold U+2028 etc.
    if smoke:
        lines = lines[:64]
        cfg["batching"] = {"max_tokens": 4096, "max_questions": 8}
    if subset:
        lines = random.Random(cfg["seed"]).sample(lines, min(subset, len(lines)))
    train_sha = records_digest(train_path)
    key = f"{train_sha}|{len(lines)}|{sq_key(cfg)}"
    t0 = time.time()
    items = encode_records_parallel(ser, lines, cache=None if smoke else run.parent / f"encoded_{hashlib.sha1(key.encode()).hexdigest()[:12]}.pkl",
                                    cache_key=key)
    del lines
    print(json.dumps({"encoded": len(items), "seconds": round(time.time() - t0, 1)}), flush=True)
    torch.manual_seed(cfg["seed"])                  # model AFTER the fork-based encoding (no forked torch threads)
    full = getattr(args, "init_full", None)
    if full:                                         # probe: start from a fine-tuned Krun One v1 (encoder + decision stack)
        from safetensors.torch import load_model

        spec = arch.SPECS[cfg["spec"]]
        model = KrunOneV1Model.from_spec(spec, attn_implementation="sdpa")
        load_model(model, full)
        init = {"full": full, "sha256": hashlib.sha256(open(full, "rb").read()).hexdigest()}
    else:
        model, spec, init = build_model(cfg["spec"], None if smoke else Path(args.init_dir), smoke)
    extra = {"spec": spec.to_dict(), "init": init, "init_sha256": "smoke" if smoke else (init.get("sha256") or weights_sha(Path(args.init_dir))),
             "train_sha256": train_sha, "n_train": len(items), "config_hash": cfg_hash(cfg), "subset": subset}
    res = train(model, items, tok.pad_token_id, cfg, run, resolve_resume(args.resume_from_checkpoint, run) if not subset else None,
                args.max_steps, device="cpu" if smoke else None, run_config_extra=extra)
    (run / "train_result.json").write_text(json.dumps(res, indent=2, default=str))
    del items
    if res["completed"] or smoke:
        ev = run_eval(model, ser, tok.pad_token_id, cfg, run, device="cpu" if smoke else None,
                      limit=16 if smoke else eval_limit)
        if res["completed"] and not smoke and not subset:
            from safetensors.torch import save_model

            (run / "final").mkdir(exist_ok=True)
            save_model(model, str(run / "final/model.safetensors"))
            (run / "final/spec.json").write_text(json.dumps(spec.to_dict(), indent=2))
        return {"run": str(run), "train": {k: v for k, v in res.items() if k != "log"}, "eval": ev}
    return {"run": str(run), "train": {k: v for k, v in res.items() if k != "log"}}


def sq_key(cfg):
    return json.dumps(cfg["sequence"], sort_keys=True)


def smoke_score(ev: dict) -> float:
    """lr_smoke rule: mean of intent acc, tool acc, noul ROC-AUC, multi micro-F1 on the V fit subset."""
    parts = []
    for stem, e in ev.items():
        fam = e.get("choice_accuracy_by_family", {})
        parts += [(k, v) for k, v in (("intent", fam.get("intent")), ("tool", fam.get("tool")),
                                       ("noul", e.get("noul_auroc")), ("multi", e.get("multi_micro_f1"))) if v is not None]
    by = {}
    for k, v in parts:
        by.setdefault(k, []).append(v)
    return 100 * sum(sum(v) / len(v) for v in by.values()) / max(1, len(by))


def cmd_lr_smoke(args, cfg):
    ls = cfg["lr_smoke"]
    # fixed V "fit" subset for the smoke (written once, the same for every candidate)
    fit_path = ROOT / "runs/v1/ft/lr_smoke_eval_fit.jsonl"
    if not fit_path.exists():
        rows = [l for f in val_files(cfg) for l in open(f)
                if (json.loads(l).get("metadata") or {}).get("val_part", "fit" if "val_fit" in f.name else "eval") == "fit"
                and "val_eval" not in f.name]
        rows = random.Random(cfg["seed"]).sample(rows, min(ls["eval_subset"], len(rows)))
        fit_path.parent.mkdir(parents=True, exist_ok=True)
        fit_path.write_text("".join(r if r.endswith("\n") else r + "\n" for r in rows))
    results = []
    for cand in ls["candidates"]:
        c = json.loads(json.dumps(cfg))
        c["optimizer"].update(cand)
        c["data"]["validation"] = [str(fit_path.relative_to(ROOT))]
        c["checkpoint"]["save_every_minutes"] = 0
        name = f"lr_smoke_enc{cand['lr_encoder']:g}_head{cand['lr_head']:g}"
        t0 = time.time()
        out = cmd_train(args, c, subset=ls["subset"], run_name=name)
        losses = [e["loss"] for e in (json.loads((ROOT / "runs/v1/ft" / name / "train_result.json").read_text())["log"])]
        results.append({**cand, "score": smoke_score(out["eval"]), "final_loss_mean_last10": sum(losses[-10:]) / max(1, len(losses[-10:])),
                        "seconds": round(time.time() - t0, 1), "eval": out["eval"]})
        print(json.dumps({k: v for k, v in results[-1].items() if k != "eval"}), flush=True)
        for p in (ROOT / "runs/v1/ft" / name).glob("checkpoint-*"):
            import shutil

            shutil.rmtree(p, ignore_errors=True)
    best = max(results, key=lambda r: r["score"])
    ties = [r for r in results if best["score"] - r["score"] < 0.3]
    choice = min(ties, key=lambda r: r["lr_encoder"])
    rep = {"rule": ls["rule"], "results": results, "choice": {"lr_encoder": choice["lr_encoder"], "lr_head": choice["lr_head"]}}
    out = ROOT / "results/v1/ft"
    out.mkdir(parents=True, exist_ok=True)
    (out / "lr_smoke.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep["choice"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "lr_smoke", "train", "eval"])
    ap.add_argument("--config", default=str(ROOT / "configs/v1/finetune.yaml"))
    ap.add_argument("--init-dir")
    ap.add_argument("--run-name")
    ap.add_argument("--run")
    ap.add_argument("--lr-encoder", type=float)
    ap.add_argument("--lr-head", type=float)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--resume-from-checkpoint")
    ap.add_argument("--train-file", help="override data.train (smoke)")
    ap.add_argument("--init-full", help="full Krun One v1 model.safetensors to start from (xLAM ablation probe)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--weights", help="eval: model.safetensors to evaluate (default <run>/final); outputs go to --run")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.lr_encoder:
        cfg["optimizer"]["lr_encoder"] = args.lr_encoder
    if args.lr_head:
        cfg["optimizer"]["lr_head"] = args.lr_head
    if args.train_file:
        cfg["data"]["train"] = args.train_file
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.cmd == "smoke":
        cfg["gradient_checkpointing"] = False
        print(json.dumps(cmd_train(args, cfg, smoke=True, run_name="smoke"), default=str)[:2000])
    elif args.cmd == "lr_smoke":
        cmd_lr_smoke(args, cfg)
    elif args.cmd == "train":
        print(json.dumps(cmd_train(args, cfg), default=str)[:4000])
    else:
        from safetensors.torch import load_model

        run = Path(args.run)
        spec = ModelSpec.from_dict(json.loads((run / "final/spec.json").read_text())) if (run / "final/spec.json").exists() \
            else arch.SPECS[cfg["spec"]]
        model = KrunOneV1Model.from_spec(spec, attn_implementation="sdpa")
        load_model(model, args.weights or str(run / "final/model.safetensors"))   # --weights: a milestone snapshot
        tok, ser = prepare(args, cfg)
        run_eval(model, ser, tok.pad_token_id, cfg, run)


if __name__ == "__main__":
    main()
