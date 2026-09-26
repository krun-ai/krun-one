"""Supervised fine-tuning of the Laya checkpoint (Krun One v0).

Loss: soft-target cross entropy over each decision's options — one-hot for answerable decisions,
uniform for ``answerable = false`` (outlier exposure), weighted by ``unanswerable_weight``. No RL,
no REINFORCE/GRPO, no new heads. The upstream act head is frozen and unused.

Batches are length-bucketed under a padded-token budget; the batch plan of each epoch is a pure
function of a NumPy Generator seeded from (seed, epoch), whose state at the epoch start is
checkpointed, so a resume replays exactly the same plan and continues at the next batch.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from .checkpoint import load_checkpoint, restore_rng, rng_state, save_checkpoint
from .dataset import token_batches
from .model import DecisionModel, collate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_trainable(model: DecisionModel, top_layers: int | str) -> dict[str, int]:
    """Config A (int N): head + scorer + type_emb + top N encoder layers (+ final norm). Config B ('all'): everything.
    act_head is always frozen (unused)."""
    for p in model.parameters():
        p.requires_grad = False
    for mod in (model.head, model.scorer, model.type_emb):
        for p in mod.parameters():
            p.requires_grad = True
    enc = model.encoder
    if top_layers == "all":
        for p in enc.parameters():
            p.requires_grad = True
    else:
        n = int(top_layers)
        for layer in enc.layers[len(enc.layers) - n:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in enc.final_norm.parameters():
            p.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable}


def param_groups(model: DecisionModel, lr_encoder: float, lr_head: float, wd: float) -> list[dict]:
    head_ids = {id(p) for m in (model.head, model.scorer, model.type_emb) for p in m.parameters()}
    enc = [p for p in model.encoder.parameters() if p.requires_grad]
    head = [p for p in model.parameters() if p.requires_grad and id(p) in head_ids]
    groups = [{"params": head, "lr": lr_head, "weight_decay": wd}]
    if enc:
        groups.insert(0, {"params": enc, "lr": lr_encoder, "weight_decay": wd})
    return groups


def warmup_linear(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    return max(0.0, (total - step) / max(1, total - warmup))


def decision_loss(logits: torch.Tensor, target: torch.Tensor, marker_mask: torch.Tensor, answerable: torch.Tensor,
                  unanswerable_weight: float, qtype: torch.Tensor | None = None, rps_weight: float = 0.0) -> torch.Tensor:
    """Soft-target CE over each decision's options (all primitives: noul's 2-way CE == BCE on z_true - z_false).

    ``rps_weight > 0`` adds the ranked probability score to ``score`` questions (KRUN-011 ordinal-aware objective,
    the ordinal term of upstream Laya's proper reward): RPS = Σ_k (CDF_p − CDF_target)² / (K − 1). It is strictly
    proper and grows with the distance between predicted mass and the true level.
    """
    logp = torch.log_softmax(logits.float().masked_fill(~marker_mask, -1e4), -1)
    per = -(target * logp).sum(-1)
    if rps_weight and qtype is not None:
        is_score = qtype == 1
        if is_score.any():
            m = marker_mask.float()
            p = logp.exp() * m
            k = marker_mask.sum(-1).clamp(min=2).float()
            rps = (((torch.cumsum(p, -1) - torch.cumsum(target * m, -1)) ** 2) * m).sum(-1) / (k - 1)
            per = per + rps_weight * rps * is_score.float()
    w = torch.where(answerable, torch.ones_like(per), torch.full_like(per, unanswerable_weight))
    return (per * w).sum() / w.sum()


def items_digest(items: Sequence[dict]) -> str:
    h = hashlib.sha256()
    for it in items:
        h.update(it["id"].encode())
        h.update(bytes(str(len(it["ids"])), "ascii"))
    return h.hexdigest()


def epoch_plan(lengths: Sequence[int], cfg: dict, seed: int, epoch: int) -> tuple[list[list[int]], dict]:
    gen = np.random.Generator(np.random.PCG64(seed * 1000 + epoch))
    state = gen.bit_generator.state
    return token_batches(lengths, cfg["batching"]["max_tokens"], cfg["batching"]["max_seqs"], gen), state


def train(model: DecisionModel, items: Sequence[dict], pad_id: int, cfg: dict, variant: str, run_dir: Path,
          resume_from: Path | None = None, max_steps: int | None = None, log_every: int = 20,
          on_log=None, device: str | None = None, init_state: dict | None = None) -> dict[str, Any]:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed = cfg["seed"]
    set_seed(seed)
    vcfg = cfg["configs"][variant]
    counts = configure_trainable(model, vcfg["train_top_encoder_layers"])
    if cfg.get("gradient_checkpointing"):  # memory only (same math); not part of the resume config
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    model.to(dev).train()
    oc = cfg["optimizer"]
    opt = torch.optim.AdamW(param_groups(model, oc["lr_encoder"], oc["lr_head"], oc["weight_decay"]))
    lengths = [len(it["ids"]) for it in items]
    plans = [epoch_plan(lengths, cfg, seed, e)[0] for e in range(cfg["epochs"])]
    total = sum(len(p) for p in plans)
    warm = int(oc["warmup_ratio"] * total)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_linear(s, total, warm))
    use_amp = dev.type == "cuda"
    amp_dtype = torch.bfloat16 if cfg["optimizer"]["amp"] == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    run_config = {"variant": variant, "variant_config": vcfg, "seed": seed, "sequence": cfg["sequence"],
                  "objective": cfg["objective"], "batching": cfg["batching"], "optimizer": oc, "epochs": cfg["epochs"],
                  "n_train": len(items), "train_items_sha256": items_digest(items), "total_steps": total, **counts}
    log: list[dict] = []
    step, epoch0, b0, run_loss, run_n, elapsed, tokens = 0, 0, 0, 0.0, 0, 0.0, 0
    if init_state is not None:  # continued fine-tuning (KRUN-011: from Krun One v0); part of the resume config
        run_config["init"] = init_state
    if resume_from is not None:
        st = load_checkpoint(resume_from, model, opt, sched, scaler, run_config)
        restore_rng(st["rng"])
        log, step, epoch0, b0 = st["log"], st["global_step"], st["epoch_index"], st["next_batch_in_epoch"]
        run_loss, run_n, elapsed, tokens = st["run_loss"], st["run_n"], st["elapsed_s"], st["tokens"]
        print(f"  resumed from {resume_from}: step {step}/{total}, epoch {epoch0 + 1}, batch {b0}", flush=True)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.time() - elapsed
    t_int, tok_int = time.time(), 0
    trainable = [p for p in model.parameters() if p.requires_grad]
    obj = cfg["objective"]
    for epoch in range(epoch0, cfg["epochs"]):
        plan, shuffle_state = epoch_plan(lengths, cfg, seed, epoch)
        for bi in range(b0 if epoch == epoch0 else 0, len(plan)):
            batch = [items[i] for i in plan[bi]]
            b = collate(batch, pad_id)
            ans = torch.tensor([it["answerable"] for it in batch], dtype=torch.bool, device=dev)
            b = {k: v.to(dev) for k, v in b.items()}
            with torch.autocast(dev.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
            loss = decision_loss(logits, b["target"], b["marker_mask"], ans, obj["unanswerable_weight"], b["qtype"],
                                 obj.get("score_rps_weight", 0.0))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {loss.item()}")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = nn.utils.clip_grad_norm_(trainable, oc["max_grad_norm"])
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            ntok = int(b["attention_mask"].sum())
            tokens += ntok
            tok_int += ntok
            run_loss += loss.item()
            run_n += 1
            if step % log_every == 0 or step == total:
                now = time.time()
                entry = {"step": step, "epoch": epoch + 1, "train_loss": run_loss / run_n, "lr_head": sched.get_last_lr()[-1],
                         "grad_norm": float(gnorm), "seconds": round(now - t0, 1),
                         "steps_per_s": round(log_every / max(now - t_int, 1e-9), 3) if step % log_every == 0 else None,
                         "tokens_per_s": round(tok_int / max(now - t_int, 1e-9))}
                if dev.type == "cuda":
                    entry["peak_vram_mib"] = round(torch.cuda.max_memory_allocated(dev) / 2**20)
                log.append(entry)
                print(f"  step {step}/{total} loss {entry['train_loss']:.4f} gnorm {entry['grad_norm']:.2f} "
                      f"{entry['tokens_per_s']} tok/s vram {entry.get('peak_vram_mib')} MiB ({now - t0:.0f}s)", flush=True)
                if on_log:
                    on_log(entry)
                run_loss, run_n, t_int, tok_int = 0.0, 0, now, 0
            stop = max_steps is not None and step >= max_steps
            if (step % cfg["checkpoint"]["save_every"] == 0 or stop) and step < total:
                state = {"global_step": step, "total_steps": total, "epoch_index": epoch, "next_batch_in_epoch": bi + 1,
                         "run_loss": run_loss, "run_n": run_n, "elapsed_s": time.time() - t0, "tokens": tokens, "log": log}
                path = save_checkpoint(run_dir, step, model, opt, sched, scaler, rng_state(shuffle_state), state, run_config,
                                       cfg["checkpoint"]["keep_last"])
                print(f"  saved {path}", flush=True)
            if stop:
                return {"completed": False, "global_step": step, "total_steps": total, "log": log, "run_config": run_config,
                        "training_seconds_cumulative": time.time() - t0, "tokens": tokens}
    return {"completed": True, "global_step": step, "total_steps": total, "log": log, "run_config": run_config,
            "training_seconds_cumulative": time.time() - t0, "tokens": tokens,
            "peak_vram_mib": round(torch.cuda.max_memory_allocated(dev) / 2**20) if dev.type == "cuda" else None}
