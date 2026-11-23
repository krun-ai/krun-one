"""Krun One v1 trainer (proxy scale first; same code path is meant to scale to the 800M run).

Kept from v0.3 (src/krun_one/training.py, validated in KRUN-005/011):
* atomic checkpoints via bakeoff.checkpoint: model, optimizer, scheduler, GradScaler, python/numpy/torch/CUDA RNG,
  epoch-start shuffle-generator state, trainer state; resume refuses a different config (config includes the
  sha256 of the training file and of the model spec);
* length-bucketed batches under a padded-token budget, a pure function of (seed, epoch);
* bf16 autocast, AdamW with separate encoder / decision-stack learning rates, warmup + linear decay, grad clip.

New: krun.decision.v1 records, chunked questions (a batch unit is a question, whose chunks are all in the same
batch so the loss sees the merged distribution), v1 losses and answerability metrics.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from ..checkpoint import load_checkpoint, restore_rng, rng_state, save_checkpoint
from ..dataset import token_batches
from ..training import set_seed, warmup_linear
from .arch import QTYPES_V1, ModelSpec
from .losses import LossConfig, answerability_scores, decision_loss
from .model import KrunOneV1Model, chunk_index, collate_chunks, merge_chunks
from .serialization import Serializer


# ------------------------------------------------------------------------------------------------ data
def load_records(path: Path, limit: int | None = None) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def encode_records(ser: Serializer, records: Sequence[dict]) -> list[dict]:
    items = []
    for r in records:
        q = ser.encode_record(r)
        ids = [c["id"] for c in r["candidates"]]
        tgt = np.zeros(len(ids), dtype=np.float32)
        for lab in r["target"]["labels"]:
            tgt[ids.index(lab)] = 1.0
        items.append({"id": r["id"], "q": q, "target": tgt, "qtype": QTYPES_V1[r["task"]["type"]],
                      "answerable": bool(r["answerable"]), "lang": r["context"]["language"],
                      "slices": r.get("slices") or [], "type": r["task"]["type"], "family": r["task"].get("family"),
                      "length": sum(len(c.ids) for c in q.chunks), "stage": (r.get("metadata") or {}).get("stage", "D1")})
    return items


_SER: Serializer | None = None


def _encode_part(records: Sequence[dict | str]) -> list[dict]:
    items = encode_records(_SER, [json.loads(r) if isinstance(r, str) else r for r in records])
    for it in items:                               # compact storage: 1M decisions must fit in host RAM
        for c in it["q"].chunks:
            c.ids = np.asarray(c.ids, dtype=np.int32)
    return items


def encode_records_parallel(ser: Serializer, records: Sequence[dict], workers: int = 0, part: int = 4000,
                            cache: Path | None = None, cache_key: str | None = None) -> list[dict]:
    """encode_records over a fork pool (token ids stored as int32 arrays), optionally cached on disk keyed by
    ``cache_key`` (training-file sha256 + serializer budget), so a resumed / preempted run does not re-tokenize."""
    import os
    import pickle

    if cache is not None and cache.exists():
        with open(cache, "rb") as f:
            got = pickle.load(f)
        if got["key"] == cache_key and got["n"] == len(records):
            return got["items"]
    global _SER
    _SER = ser
    parts = [records[i:i + part] for i in range(0, len(records), part)]
    workers = workers or max(1, min(len(parts), 24, (os.cpu_count() or 2) - 1))
    if workers == 1 or len(parts) == 1:
        items = [it for p_ in parts for it in _encode_part(p_)]
    else:
        # fork BEFORE any model / CUDA / tokenizer thread exists (callers encode first); ProcessPoolExecutor raises
        # BrokenProcessPool if a worker dies instead of hanging forever like Pool.imap
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
            items = [it for got in ex.map(_encode_part, parts, chunksize=1) for it in got]
    if cache is not None:
        tmp = cache.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"key": cache_key, "n": len(records), "items": items}, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.rename(cache)
    return items


def make_batch(items: Sequence[dict], pad_id: int, dev) -> tuple[dict, list, list[int], dict[str, torch.Tensor]]:
    qs = [it["q"] for it in items]
    flat, where = chunk_index(qs)
    b = {k: v.to(dev, non_blocking=True) for k, v in collate_chunks(flat, pad_id).items()}
    kmax = max(q.n_candidates for q in qs)
    tgt = torch.zeros(len(items), kmax)
    for i, it in enumerate(items):
        tgt[i, :len(it["target"])] = torch.from_numpy(it["target"])
    y = {"target": tgt.to(dev), "qtype": torch.tensor([it["qtype"] for it in items], device=dev),
         "answerable": torch.tensor([it["answerable"] for it in items], device=dev)}
    return b, where, [q.n_candidates for q in qs], y


def curriculum_batches(lengths: Sequence[int], stages: Sequence[str], ranges: dict[str, Sequence[float]], max_tokens: int,
                       max_seqs: int, rng: np.random.Generator, chunk: int = 2048) -> list[list[int]]:
    """Progressive curriculum as ONE ordering of the epoch (every item exactly once).

    Each item gets a position key ~ U(lo, hi) of its stage's range (e.g. D1 [0, .55], D2 [.2, .9], D3 [.45, 1]); items
    are sorted by key, cut into windows of ``chunk`` items, length-bucketed into token batches inside each window, and
    the batches of a window are shuffled. So early training is dominated by core decisions, the middle by hard /
    cross-language ones and the end by OOD / minimal pairs / adversarial, while every stage appears throughout its
    range. A pure function of (lengths, stages, ranges, rng state)."""
    lo = np.array([ranges[s][0] for s in stages], dtype=np.float64)
    hi = np.array([ranges[s][1] for s in stages], dtype=np.float64)
    key = lo + (hi - lo) * rng.random(len(stages))
    order = np.argsort(key, kind="stable")
    ln = np.asarray(lengths)
    batches: list[list[int]] = []
    for s in range(0, len(order), chunk):
        part = order[s:s + chunk]
        part = part[np.argsort(ln[part], kind="stable")]
        win, cur, cur_max = [], [], 0
        for i in part:
            new_max = max(cur_max, int(ln[i]))
            if cur and (new_max * (len(cur) + 1) > max_tokens or len(cur) + 1 > max_seqs):
                win.append(cur)
                cur, new_max = [], int(ln[i])
            cur.append(int(i))
            cur_max = new_max
        if cur:
            win.append(cur)
        batches += [win[j] for j in rng.permutation(len(win))]
    return batches


def records_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ model init
def load_mmbert_encoder(model: KrunOneV1Model, weights: Path) -> dict[str, int]:
    """Copy an mmBERT (ModernBertForMaskedLM) checkpoint's encoder weights into model.encoder (strict on encoder)."""
    from safetensors.torch import load_file

    weights = Path(weights)
    if weights.is_dir():   # HF save_pretrained directory, possibly sharded
        files = sorted(weights.glob("model*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no safetensors in {weights}")
        sd = {}
        for f in files:
            sd.update(load_file(str(f)))
    else:
        sd = load_file(str(weights))
    enc = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    if "embeddings.tok_embeddings.weight" not in enc and "decoder.weight" in sd:  # tied weights are stored once
        enc["embeddings.tok_embeddings.weight"] = sd["decoder.weight"]
    missing, unexpected = model.encoder.load_state_dict(enc, strict=False)
    if missing or unexpected:
        raise ValueError(f"encoder init mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return {"loaded_tensors": len(enc)}


def param_groups(model: KrunOneV1Model, lr_encoder: float, lr_head: float, wd: float) -> list[dict]:
    enc = {id(p) for p in model.encoder.parameters()}
    return [{"params": [p for p in model.parameters() if id(p) in enc], "lr": lr_encoder, "weight_decay": wd},
            {"params": [p for p in model.parameters() if id(p) not in enc], "lr": lr_head, "weight_decay": wd}]


# ------------------------------------------------------------------------------------------------ training
def save_milestone(run_dir: Path, model: KrunOneV1Model, step: int, total: int, frac: float) -> Path:
    """Atomic weights-only snapshot at runs/.../milestones/step_XXXXXX/ (model.safetensors + meta.json)."""
    from safetensors.torch import save_model

    d = Path(run_dir) / "milestones" / f"step_{step:06d}"
    tmp = d.with_name(d.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    save_model(model, str(tmp / "model.safetensors"))
    (tmp / "meta.json").write_text(json.dumps({"global_step": step, "total_steps": total, "fraction": frac}))
    if d.exists():
        import shutil

        shutil.rmtree(d)
    tmp.rename(d)
    print(json.dumps({"milestone": str(d), "step": step, "fraction": frac}), flush=True)
    return d


def train(model: KrunOneV1Model, items: Sequence[dict], pad_id: int, cfg: dict, run_dir: Path,
          resume_from: Path | None = None, max_steps: int | None = None, device: str | None = None,
          log_every: int = 20, run_config_extra: dict | None = None) -> dict[str, Any]:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed = cfg["seed"]
    set_seed(seed)
    model.to(dev).train()
    if cfg.get("gradient_checkpointing"):
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    oc = cfg["optimizer"]
    opt = torch.optim.AdamW(param_groups(model, oc["lr_encoder"], oc["lr_head"], oc["weight_decay"]))
    lengths = [it["length"] for it in items]
    bc = cfg["batching"]

    cur = cfg.get("curriculum")

    def plan(epoch):
        gen = np.random.Generator(np.random.PCG64(seed * 1000 + epoch))
        state = gen.bit_generator.state
        if cur:
            return curriculum_batches(lengths, [it.get("stage", "D1") for it in items], cur["ranges"], bc["max_tokens"],
                                      bc["max_questions"], gen), state
        return token_batches(lengths, bc["max_tokens"], bc["max_questions"], gen), state

    plans = [plan(e)[0] for e in range(cfg["epochs"])]
    total = sum(len(p) for p in plans)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_linear(s, total, int(oc["warmup_ratio"] * total)))
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # bf16: no scaling
    lcfg = LossConfig(**cfg["loss"])
    run_config = {"seed": seed, "loss": cfg["loss"], "batching": bc, "optimizer": oc, "epochs": cfg["epochs"],
                  "n_items": len(items), "total_steps": total, **(run_config_extra or {})}
    log: list[dict] = []
    step, epoch0, b0, run_loss, run_n, elapsed = 0, 0, 0, 0.0, 0, 0.0
    if resume_from is not None:
        st = load_checkpoint(resume_from, model, opt, sched, scaler, run_config)
        restore_rng(st["rng"])
        log, step, epoch0, b0 = st["log"], st["global_step"], st["epoch_index"], st["next_batch_in_epoch"]
        run_loss, run_n, elapsed = st["run_loss"], st["run_n"], st["elapsed_s"]
    t0 = time.time() - elapsed
    last_save = time.time()
    # milestone weights (fractions of total, e.g. [.25, .5, .75]): weights-only safetensors that are never pruned by
    # keep_last; the 100% weights are the run's final/ (written by the driver)
    milestones = {max(1, int(round(f * total))): f for f in cfg["checkpoint"].get("milestones", []) if 0 < f < 1}
    params = [p for p in model.parameters() if p.requires_grad]
    for epoch in range(epoch0, cfg["epochs"]):
        p_, shuffle_state = plan(epoch)
        for bi in range(b0 if epoch == epoch0 else 0, len(p_)):
            batch = [items[i] for i in p_[bi]]
            b, where, ncand, y = make_batch(batch, pad_id, dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
                out = model(**b)
            q = merge_chunks(out, where, len(batch), ncand)
            loss, parts = decision_loss(q, y["qtype"], y["target"], y["answerable"], lcfg)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            loss.backward()
            gnorm = nn.utils.clip_grad_norm_(params, oc["max_grad_norm"])
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            run_loss += parts["loss"]
            run_n += 1
            if step % log_every == 0 or step == total:
                e = {"step": step, "loss": run_loss / run_n, **{k: v for k, v in parts.items() if k != "loss"},
                     "grad_norm": float(gnorm), "lr_head": sched.get_last_lr()[-1], "seconds": round(time.time() - t0, 1)}
                if dev.type == "cuda":
                    e["peak_vram_mib"] = round(torch.cuda.max_memory_allocated(dev) / 2**20)
                log.append(e)
                print(json.dumps(e), flush=True)
                run_loss, run_n = 0.0, 0
            if step in milestones:
                save_milestone(run_dir, model, step, total, milestones[step])
            stop = max_steps is not None and step >= max_steps
            due = step % cfg["checkpoint"]["save_every"] == 0
            if cfg["checkpoint"].get("save_every_minutes") and time.time() - last_save >= 60 * cfg["checkpoint"]["save_every_minutes"]:
                due = True
            if (due or stop) and step < total:
                last_save = time.time()
                state = {"global_step": step, "total_steps": total, "epoch_index": epoch, "next_batch_in_epoch": bi + 1,
                         "run_loss": run_loss, "run_n": run_n, "elapsed_s": time.time() - t0, "log": log}
                save_checkpoint(run_dir, step, model, opt, sched, scaler, rng_state(shuffle_state), state, run_config,
                                cfg["checkpoint"]["keep_last"])
            if stop:
                return {"completed": False, "global_step": step, "total_steps": total, "log": log}
    return {"completed": True, "global_step": step, "total_steps": total, "log": log,
            "seconds": time.time() - t0}


# ------------------------------------------------------------------------------------------------ evaluation
def _auroc(pos: np.ndarray, neg: np.ndarray) -> float | None:
    if len(pos) == 0 or len(neg) == 0:
        return None
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort().astype(np.float64) + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


@torch.no_grad()
def evaluate(model: KrunOneV1Model, items: Sequence[dict], pad_id: int, device=None, batch_tokens: int = 16384
             ) -> dict[str, Any]:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev).eval()
    order = sorted(range(len(items)), key=lambda i: items[i]["length"])
    rows: list[dict] = [None] * len(items)  # type: ignore
    i = 0
    while i < len(order):
        j, mx = i, 0
        while j < len(order) and max(mx, items[order[j]]["length"]) * (j - i + 1) <= batch_tokens and j - i < 128:
            mx = max(mx, items[order[j]]["length"])
            j += 1
        j = max(j, i + 1)
        batch = [items[k] for k in order[i:j]]
        b, where, ncand, y = make_batch(batch, pad_id, dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out = model(**b)
        q = {k: v.float() if v.is_floating_point() else v for k, v in merge_chunks(out, where, len(batch), ncand).items()}
        sc = {k: v.cpu().numpy() for k, v in answerability_scores(q).items()}
        p = torch.softmax(q["rank"].masked_fill(~q["mask"], -1e4), -1).cpu().numpy()
        ap = torch.sigmoid(q["apply"]).cpu().numpy() if "apply" in q else None
        for n, (k, it) in enumerate(zip(order[i:j], batch)):
            K = len(it["target"])
            rows[k] = {"p": p[n, :K], "apply": None if ap is None else ap[n, :K], **{s: float(v[n]) for s, v in sc.items()}}
        i = j
    res: dict[str, Any] = {"n": len(items)}
    # choice
    ch = [(it, r) for it, r in zip(items, rows) if it["type"] == "choice"]
    ans = [(it, r) for it, r in ch if it["answerable"]]
    if ans:
        corr = np.array([int(np.argmax(r["p"]) == int(np.argmax(it["target"]))) for it, r in ans])
        res["choice_accuracy"] = float(corr.mean())
        res["choice_ece"] = _ece(np.array([r["p"].max() for _, r in ans]), corr)
        by = {}
        for (it, _), c in zip(ans, corr):
            by.setdefault(it["lang"], []).append(c)
        res["choice_accuracy_by_lang"] = {k: round(float(np.mean(v)), 4) for k, v in sorted(by.items())}
        byf = {}
        for (it, _), c in zip(ans, corr):
            byf.setdefault(it.get("family") or "?", []).append(c)
        res["choice_accuracy_by_family"] = {k: round(float(np.mean(v)), 4) for k, v in sorted(byf.items())}
    una = [(it, r) for it, r in ch if not it["answerable"]]
    if ans and una:
        res["ood_auroc"] = {s: _auroc(np.array([r[s] for _, r in ans]), np.array([r[s] for _, r in una]))
                            for s in ("margin", "top1", "null_softmax", "ans_head", "max_apply") if s in rows[0]}
    for t in ("noul", "score"):
        sub = [(it, r) for it, r in zip(items, rows) if it["type"] == t]
        if not sub:
            continue
        if t == "noul":
            y = np.array([int(it["target"][1]) for it, _ in sub])
            pt = np.array([r["p"][1] for _, r in sub])
            res["noul_auroc"] = _auroc(pt[y == 1], pt[y == 0])
            res["noul_accuracy"] = float(((pt > 0.5) == y).mean())
        else:
            y = np.array([int(np.argmax(it["target"])) for it, _ in sub])
            e = np.array([float((np.arange(len(r["p"])) * r["p"]).sum()) for _, r in sub])
            res["score_mae"] = float(np.abs(e - y).mean())
            from scipy.stats import spearmanr  # noqa: PLC0415 (sklearn dependency brings scipy)
            res["score_spearman"] = float(spearmanr(e, y).correlation)
    mu = [(it, r) for it, r in zip(items, rows) if it["type"] == "multi"]
    if mu and mu[0][1]["apply"] is not None:
        tp = fp = fn = 0
        for it, r in mu:
            pred = r["apply"] > 0.5
            gold = it["target"] > 0.5
            tp += int((pred & gold).sum())
            fp += int((pred & ~gold).sum())
            fn += int((~pred & gold).sum())
        res["multi_micro_f1"] = 2 * tp / max(1, 2 * tp + fp + fn)
        res["multi_empty_set_rate_on_unanswerable"] = float(np.mean(
            [not (r["apply"] > 0.5).any() for it, r in mu if not it["answerable"]] or [math.nan]))
    return res


@torch.no_grad()
def dump_signals(model: KrunOneV1Model, items: Sequence[dict], records: Sequence[dict], pad_id: int, path: Path,
                 device=None, batch_tokens: int = 16384) -> int:
    """Raw per-decision outputs on a VALIDATION file (never a protected one) for offline calibration/analysis.

    One JSON line per decision: ids/groups/slices/language/family/K, gold index, raw rank / apply logits, null
    logit, answerability logit. Temperatures, thresholds and the fusion calibrator are fitted offline on val_fit.
    """
    import gzip

    if any(r.get("split") in ("test", "protected") for r in records):
        raise ValueError("dump_signals refuses test/protected records")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev).eval()
    order = sorted(range(len(items)), key=lambda i: items[i]["length"])
    n = 0
    with gzip.open(path, "wt") as f:
        i = 0
        while i < len(order):
            j, mx = i, 0
            while j < len(order) and max(mx, items[order[j]]["length"]) * (j - i + 1) <= batch_tokens and j - i < 128:
                mx = max(mx, items[order[j]]["length"])
                j += 1
            j = max(j, i + 1)
            batch = [items[k] for k in order[i:j]]
            b, where, ncand, _ = make_batch(batch, pad_id, dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                out = model(**b)
            q = {k: (v.float().cpu().numpy() if v.is_floating_point() else v.cpu().numpy())
                 for k, v in merge_chunks(out, where, len(batch), ncand).items()}
            for m, k in enumerate(order[i:j]):
                it, r = items[k], records[k]
                K = len(it["target"])
                gen = (r.get("metadata") or {}).get("v0_generation") or {}
                row = {"id": r["id"], "type": it["type"], "family": r["task"].get("family"), "lang": it["lang"],
                       "slices": it["slices"], "difficulty": r.get("difficulty"), "answerable": it["answerable"],
                       "k": K, "gold": [int(x) for x in np.flatnonzero(it["target"] > 0.5)],
                       "split_group_id": r["split_group_id"], "consistency_group": gen.get("consistency_group"),
                       "candidate_ids": [c["id"] for c in r["candidates"]],
                       "rank": [round(float(x), 5) for x in q["rank"][m, :K]]}
                if "apply" in q:
                    row["apply"] = [round(float(x), 5) for x in q["apply"][m, :K]]
                if "null_rank" in q:
                    row["null"] = round(float(q["null_rank"][m]), 5)
                if "ans" in q:
                    row["ans"] = round(float(q["ans"][m]), 5)
                f.write(json.dumps(row) + "\n")
                n += 1
            i = j
    return n
