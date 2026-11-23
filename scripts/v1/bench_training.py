"""Training-throughput benchmark for the K1-W1536-L22 model (cost study; random weights — speed does not depend on values).

Measures, on the GPU it runs on:
  mlm       continued-MLM steps of the 879M ModernBertForMaskedLM (seq 1024, 15% masks, sparse decoder, bf16 autocast,
            fused AdamW), for several micro-batch sizes, with/without gradient checkpointing, and with the token
            embedding (= tied decoder, 393M params) frozen or trainable;
  decision  Krun decision fine-tuning steps of the KrunOneV1Model (K1) on synthetic batches shaped like the proxy data
            (16,384 padded tokens / step), full fine-tuning.
Reports tokens/s, peak VRAM, and $ per 1B tokens for the prices passed on the command line.

    python scripts/v1/bench_training.py --out results/v1/bench_training_<gpu>.json --price-per-hour 0.34
    python scripts/v1/bench_training.py --spec tiny --device cpu --out /tmp/x.json        # local smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from krun_one.v1 import arch  # noqa: E402
from krun_one.v1.arch import DecisionSpec, EncoderSpec, ModelSpec  # noqa: E402
from krun_one.v1.losses import LossConfig, decision_loss  # noqa: E402
from krun_one.v1.model import KrunOneV1Model  # noqa: E402

TINY = ModelSpec("tiny", EncoderSpec(vocab_size=4096, hidden_size=64, num_layers=3, num_heads=2, intermediate_size=96,
                                     local_window=16), DecisionSpec(head_layers=1, head_ffn_mult=2))


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def bench_mlm(spec: ModelSpec, dev, micro: int, seq: int, ckpt: bool, freeze_emb: bool, steps: int, warmup: int) -> dict:
    from transformers import ModernBertForMaskedLM

    cfg = spec.encoder.hf_config()
    cfg._attn_implementation = "sdpa"
    cfg.sparse_prediction = True
    torch.manual_seed(0)
    model = ModernBertForMaskedLM(cfg).to(dev).train()
    if ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if freeze_emb:
        model.model.embeddings.tok_embeddings.weight.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4, fused=dev.type == "cuda")
    V = cfg.vocab_size
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    try:
        for i in range(warmup + steps):
            ids = torch.randint(5, V, (micro, seq), device=dev)
            labels = torch.full_like(ids, -100)
            m = torch.rand(ids.shape, device=dev) < 0.15
            labels[m] = ids[m]
            x = ids.masked_fill(m, 4)
            sync(dev)
            t0 = time.perf_counter()
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                loss = model(input_ids=x, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sync(dev)
            if i >= warmup:
                times.append(time.perf_counter() - t0)
    except torch.OutOfMemoryError:
        del model, opt
        torch.cuda.empty_cache()
        return {"oom": True}
    t = sorted(times)[len(times) // 2]
    res = {"oom": False, "step_s": t, "tokens_per_s": micro * seq / t,
           "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else None,
           "trainable_params": sum(p.numel() for p in params)}
    del model, opt
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return res


def bench_decision(spec: ModelSpec, dev, seqs: int, seq_len: int, k: int, ckpt: bool, steps: int, warmup: int) -> dict:
    torch.manual_seed(0)
    model = KrunOneV1Model.from_spec(spec, attn_implementation="sdpa").to(dev).train()
    if ckpt:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, fused=dev.type == "cuda")
    V = spec.encoder.vocab_size
    ids = torch.randint(5, V, (seqs, seq_len), device=dev)
    att = torch.ones_like(ids)
    lengths = torch.randint(seq_len // 3, seq_len + 1, (seqs,))      # realistic padding inside the bucket
    for i, ln in enumerate(lengths):
        att[i, ln:] = 0
    mpos = torch.stack([torch.arange(10, 10 + 3 * k, 3) for _ in range(seqs)]).to(dev)
    mmask = torch.ones(seqs, k, dtype=torch.bool, device=dev)
    qtype = torch.zeros(seqs, dtype=torch.long, device=dev)
    null_pos = torch.full((seqs,), 10 + 3 * k, device=dev)
    tgt = torch.zeros(seqs, k, device=dev)
    tgt[:, 0] = 1
    ans = torch.ones(seqs, dtype=torch.bool, device=dev)
    lcfg = LossConfig(unanswerable_mode="oe", w_ans=0.0, w_apply=0.0)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    try:
        for i in range(warmup + steps):
            sync(dev)
            t0 = time.perf_counter()
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                out = model(ids, att, mpos, mmask, qtype, null_pos)
            q = {"rank": out["rank"], "mask": mmask, "apply": out["apply"]}
            loss, _ = decision_loss(q, qtype, tgt, ans, lcfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sync(dev)
            if i >= warmup:
                times.append(time.perf_counter() - t0)
    except torch.OutOfMemoryError:
        del model, opt
        torch.cuda.empty_cache()
        return {"oom": True}
    t = sorted(times)[len(times) // 2]
    res = {"oom": False, "step_s": t, "padded_tokens_per_s": seqs * seq_len / t,
           "real_tokens_per_s": float(att.sum()) / t, "questions_per_s": seqs / t,
           "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else None}
    del model, opt
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="k1", choices=["k1", "tiny"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--price-per-hour", type=float, default=None, help="$/h of this pod (community or secure)")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="MLM full fine-tuning only (mb 4/8/16, no ckpt) + 1 decision config")
    args = ap.parse_args()
    dev = torch.device(args.device)
    spec = arch.CANDIDATES[0] if args.spec == "k1" else TINY
    seq = 1024 if args.spec == "k1" else 64
    out = {"spec": spec.name, "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
           "torch": torch.__version__, "cuda": torch.version.cuda, "price_per_hour": args.price_per_hour,
           "mlm": [], "decision": []}
    try:
        import transformers
        out["transformers"] = transformers.__version__
    except Exception:
        pass
    micro_sizes = ([4, 8, 16] if args.quick else [4, 8, 16, 32]) if args.spec == "k1" else [2]
    for freeze_emb in ((False,) if args.quick else (False, True)):
        for ckpt in ((False,) if args.quick else (False, True)):
            for mb in micro_sizes:
                r = bench_mlm(spec, dev, mb, seq, ckpt, freeze_emb, args.steps, args.warmup)
                r.update({"micro_batch": mb, "grad_ckpt": ckpt, "freeze_embeddings": freeze_emb})
                out["mlm"].append(r)
                print(json.dumps(r), flush=True)
                if r["oom"]:
                    break
                Path(args.out).write_text(json.dumps(out, indent=2))
    for ckpt in ((True,) if args.quick else (False, True)):
        for seqs, ln in (((64, 256),) if args.quick else ((64, 256), (32, 512), (16, 1024))) if args.spec == "k1" else ((4, 64),):
            r = bench_decision(spec, dev, seqs, ln, 8, ckpt, args.steps, args.warmup)
            r.update({"sequences": seqs, "seq_len": ln, "grad_ckpt": ckpt})
            out["decision"].append(r)
            print(json.dumps(r), flush=True)
            Path(args.out).write_text(json.dumps(out, indent=2))
    ok = [r for r in out["mlm"] if not r["oom"]]
    best_full = max((r for r in ok if not r["freeze_embeddings"]), key=lambda r: r["tokens_per_s"], default=None)
    best_frozen = max((r for r in ok if r["freeze_embeddings"]), key=lambda r: r["tokens_per_s"], default=None)
    dec = [r for r in out["decision"] if not r["oom"]]
    best_dec = max(dec, key=lambda r: r["real_tokens_per_s"], default=None)
    s = {"mlm_best_full": best_full, "mlm_best_frozen_emb": best_frozen, "decision_best": best_dec}
    if args.price_per_hour:
        p = args.price_per_hour
        for k, r, key in (("mlm_full", best_full, "tokens_per_s"), ("mlm_frozen_emb", best_frozen, "tokens_per_s"),
                          ("decision", best_dec, "real_tokens_per_s")):
            if r:
                s[f"usd_per_1B_tokens_{k}"] = p / 3600 / r[key] * 1e9
                s[f"hours_per_1B_tokens_{k}"] = 1e9 / r[key] / 3600
    out["summary"] = s
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
