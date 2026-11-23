"""G1: continued MLM pre-training of the grown K1 encoder with milestone checkpoints (HyperCloning scaling).

    python scripts/v1/continued_mlm.py --config configs/v1/scaling.yaml --stage 1 [--resume-from-checkpoint latest]
    python scripts/v1/continued_mlm.py --smoke            # CPU, tiny random encoder + synthetic corpus

* milestones (tokens seen) -> ``<run>/milestone-<tokens>/`` = HF ModernBertForMaskedLM weights + ``eval.json``
  (held-out MLM loss per language); ``milestone-0`` is written before the first step;
* resume checkpoints every ``save_every_tokens`` (atomic, full optimizer/scheduler/RNG/sampler state; config
  mismatch refused) via bakeoff.checkpoint, like every other Krun trainer.

Refuses to start (non-smoke) unless the tokenizer license gate file says CLEARED (see docs/KRUN_ONE_V1_TOKENIZER_LICENSE.md).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from krun_one.checkpoint import load_checkpoint, resolve_resume, restore_rng, rng_state, save_checkpoint
from krun_one.v1.mlm import MixtureSampler, enable_sparse_prediction, heldout_mlm_loss, mask_tokens, milestone_reached, wsd_lr

ROOT = Path(__file__).resolve().parents[2]
LICENSE_GATE = ROOT / "docs/KRUN_ONE_V1_TOKENIZER_LICENSE.md"


def license_cleared() -> bool:
    return LICENSE_GATE.exists() and "GATE: CLEARED" in LICENSE_GATE.read_text()


def smoke_setup(tmp: Path) -> tuple[dict, Path, object]:
    from transformers import ModernBertConfig, ModernBertForMaskedLM

    rng = np.random.default_rng(0)
    for lang in ("en", "pt"):
        d = tmp / "corpus" / lang
        d.mkdir(parents=True, exist_ok=True)
        pattern = rng.integers(5, 400, 50)                      # learnable: a repeated 50-token pattern
        np.save(d / "train-00000.npy", np.tile(pattern, 400).astype(np.uint32))
        np.save(d / "heldout.npy", np.tile(pattern, 60).astype(np.uint32))
    cfg = ModernBertConfig(vocab_size=400, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                           num_attention_heads=2, local_attention=16, pad_token_id=0, bos_token_id=2,
                           cls_token_id=2, eos_token_id=1, sep_token_id=1)
    torch.manual_seed(0)
    model = ModernBertForMaskedLM(cfg)
    init = tmp / "init"
    model.save_pretrained(init)
    c = {"name": "smoke", "seed": 7, "init": str(init), "corpus": str(tmp / "corpus"), "mixture": {"en": 0.5, "pt": 0.5},
         "sequence_length": 64, "mask_rate": 0.15, "batch_sequences": 8, "micro_batch": 4,
         "optimizer": {"lr": 1e-3, "weight_decay": 0.01, "betas": [0.9, 0.98], "eps": 1e-6, "max_grad_norm": 1.0},
         "schedule": {"warmup_tokens": 1000, "decay_fraction": 0.1},
         "milestones_stage1": [0, 2048, 5120], "checkpoint": {"save_every_tokens": 1536, "keep_last": 1},
         "eval": {"heldout_sequences_per_language": 8}}
    return c, tmp / "run", model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/v1/scaling.yaml"))
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--run-dir")
    ap.add_argument("--resume-from-checkpoint")
    ap.add_argument("--max-tokens", type=int, help="stop early (checkpointed) after this many tokens")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-dir")
    args = ap.parse_args()
    from transformers import ModernBertForMaskedLM

    if args.smoke:
        cfg, run, model = smoke_setup(Path(args.smoke_dir))
        dev = torch.device("cpu")
    else:
        if not license_cleared():
            raise SystemExit(f"blocked: tokenizer license gate not cleared ({LICENSE_GATE})")
        cfg = yaml.safe_load(Path(args.config).read_text())
        run = Path(args.run_dir or ROOT / "runs/v1/scaling" / f"{cfg['name']}")
        dev = torch.device("cuda")
        model = ModernBertForMaskedLM.from_pretrained(ROOT / cfg["init"], attn_implementation="sdpa")
    enable_sparse_prediction(model)
    model.to(dev).train()
    milestones = cfg["milestones_stage1"] + (cfg.get("milestones_stage2", []) if args.stage == 2 else [])
    total = max(milestones)
    seq, bs, mb = cfg["sequence_length"], cfg["batch_sequences"], cfg["micro_batch"]
    tokens_per_step = seq * bs
    corpus = ROOT / cfg["corpus"] if not args.smoke else Path(cfg["corpus"])
    sampler = MixtureSampler(corpus, cfg["mixture"], seq, cfg["seed"])
    oc = cfg["optimizer"]
    # fused on GPU, as benchmarked: the default foreach AdamW allocates param-sized temporaries at every step and went
    # OOM on a 22 GiB (Secure) RTX 4090 after the first micro-batches
    opt = torch.optim.AdamW(model.parameters(), lr=oc["lr"], weight_decay=oc["weight_decay"], betas=tuple(oc["betas"]),
                            eps=oc["eps"], fused=dev.type == "cuda")
    sc = cfg["schedule"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: wsd_lr(step * tokens_per_step, total, sc["warmup_tokens"], sc["decay_fraction"]))
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    mc = model.config
    special = torch.tensor([mc.pad_token_id, mc.bos_token_id, mc.eos_token_id])
    mask_id = 4 if not args.smoke else 3
    gen = torch.Generator().manual_seed(cfg["seed"])
    run_config = {k: cfg[k] for k in ("seed", "init", "corpus", "mixture", "sequence_length", "mask_rate",
                                      "batch_sequences", "optimizer", "schedule")} | {"milestones": milestones}
    step, tokens, log = 0, 0, []
    run.mkdir(parents=True, exist_ok=True)
    resume = resolve_resume(args.resume_from_checkpoint, run)
    if resume:
        st = load_checkpoint(resume, model, opt, sched, scaler, run_config)
        restore_rng(st["rng"])
        step, tokens, log = st["global_step"], st["tokens"], st["log"]
        sampler.load_state(st["sampler"])
        gen.set_state(torch.tensor(st["mask_gen"], dtype=torch.uint8))

    def evaluate_and_save(at: int) -> None:
        d = run / f"milestone-{at}"
        if (d / "eval.json").exists():
            return
        ev = heldout_mlm_loss(model, corpus, sorted(cfg["mixture"]), seq, cfg["eval"]["heldout_sequences_per_language"],
                              mc.bos_token_id, mc.eos_token_id, mask_id, mc.vocab_size, special, dev, cfg["mask_rate"])
        model.save_pretrained(d)
        (d / "eval.json").write_text(json.dumps({"tokens": at, "tokens_seen_by_lang": sampler.tokens_seen,
                                                 "heldout_mlm_loss": ev, "step": step}, indent=2))
        print(json.dumps({"milestone": at, "heldout_mlm_loss": ev}), flush=True)

    if tokens == 0 and 0 in milestones:
        evaluate_and_save(0)
    t0 = time.time()
    save_every = cfg["checkpoint"]["save_every_tokens"]
    stop_at = min(total, args.max_tokens) if args.max_tokens else total
    while tokens < stop_at:
        ids, _ = sampler.batch(bs, mc.bos_token_id, mc.eos_token_id)
        ids = torch.from_numpy(ids)
        loss_sum = 0.0
        for s in range(0, bs, mb):
            x, y = mask_tokens(ids[s:s + mb], mask_id, mc.vocab_size, cfg["mask_rate"], special, gen)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                out = model(input_ids=x.to(dev), labels=y.to(dev))   # no padding: no mask (fast SDPA path, as benchmarked)
            (out.loss * (x.size(0) / bs)).backward()
            loss_sum += float(out.loss) * x.size(0) / bs
        torch.nn.utils.clip_grad_norm_(model.parameters(), oc["max_grad_norm"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        prev, tokens, step = tokens, tokens + tokens_per_step, step + 1
        if step % 20 == 0:
            e = {"step": step, "tokens": tokens, "loss": loss_sum, "lr": sched.get_last_lr()[0],
                 "tok_per_s": round(tokens_per_step * 20 / max(1e-9, time.time() - t0))}
            t0 = time.time()
            log.append(e)
            print(json.dumps(e), flush=True)
        for m in milestone_reached(prev, tokens, milestones):
            evaluate_and_save(m)
        if prev // save_every != tokens // save_every or tokens >= stop_at:
            state = {"global_step": step, "tokens": tokens, "log": log[-200:], "sampler": sampler.state(),
                     "mask_gen": gen.get_state().tolist(), "epoch_index": 0, "next_batch_in_epoch": 0}
            save_checkpoint(run, step, model, opt, sched, scaler, rng_state(torch.tensor([0])), state, run_config,
                            cfg["checkpoint"]["keep_last"])
    print(json.dumps({"done": tokens >= total, "tokens": tokens}))


if __name__ == "__main__":
    main()
