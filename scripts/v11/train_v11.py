"""Krun One v1.1 — R1 / R2 trainer (configs/v11/r1_r2_protocol.yaml). The frozen v1 artifact is only READ.

    python scripts/v11/train_v11.py --tag r2_last2_s1 --mode last_n --last-n 2 --objective O3 --seed 20261001
    python scripts/v11/train_v11.py --tag r1_s1 --mode head --objective O3 --seed 20261001

Writes runs/v11/<track>/<tag>/{model.safetensors, spec.json, run.json}. ``--export-v1`` additionally writes
``model_v1.safetensors`` (the plain v1 architecture: aux heads dropped) when the model has no adapter / interaction block.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from krun_one.v11 import harness, training as T  # noqa: E402
from krun_one.v11.model import KrunOneV11Model, V11Spec  # noqa: E402

ARTIFACT = ROOT / "artifacts/krun-one-v1-text-final"
_CTX = {}


def _enc_part(part):
    ser, keys, stats, teacher = _CTX["ser"], _CTX["keys"], _CTX["stats"], _CTX["teacher"]
    out = []
    for kind, e in part:
        out.append(T.tech_item(ser, e, keys, stats) if kind == "tech" else T.replay_item(ser, e, teacher.get(e["id"])))
    return out


def encode_all(ser, tech, replay, keys, stats, teacher, cache: Path, key: str) -> list[dict]:
    if cache.exists():
        got = pickle.loads(cache.read_bytes())
        if got["key"] == key:
            return got["items"]
    _CTX.update(ser=ser, keys=keys, stats=stats, teacher=teacher)
    work = [("tech", e) for e in tech] + [("replay", r) for r in replay]
    parts = [work[i:i + 2000] for i in range(0, len(work), 2000)]
    workers = max(1, min(len(parts), 24, (os.cpu_count() or 2) - 1))
    if workers == 1:
        items = [it for p in parts for it in _enc_part(p)]
    else:   # fork BEFORE any CUDA / model object exists
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
            items = [it for got in ex.map(_enc_part, parts, chunksize=1) for it in got]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps({"key": key, "items": items}, protocol=pickle.HIGHEST_PROTOCOL))
    return items


def load_model(args, cfg) -> KrunOneV11Model:
    aux = cfg["objectives"][args.objective].get("l_value", 0) > 0
    if args.init_ckpt:
        d = Path(args.init_ckpt)
        from safetensors.torch import load_model as lm

        spec = V11Spec.from_dict(json.loads((d / "spec.json").read_text()))
        model = KrunOneV11Model.from_spec(spec)
        lm(model, str(d / "model.safetensors"))
        return model
    v1 = harness.load_v1(ARTIFACT)
    return KrunOneV11Model.from_v1(v1, cand_layers=args.cand_layers, adapter_dim=args.adapter_dim,
                                   aux_q=(1 + T.QD) if aux else 0, aux_v=T.VD if aux else 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--track", default="r2", choices=["r1", "r2", "pilot", "diag", "r3/text"])
    ap.add_argument("--mode", required=True, choices=["head", "last_n"])
    ap.add_argument("--last-n", type=int, default=0)
    ap.add_argument("--objective", required=True)
    ap.add_argument("--adapter-dim", type=int, default=256)
    ap.add_argument("--cand-layers", type=int, default=0)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", default=str(ROOT / "configs/v11/r1_r2_protocol.yaml"))
    ap.add_argument("--tech", default="data/v11/corpus/train_tech.jsonl")
    ap.add_argument("--replay", default="data/v11/corpus/train_replay.jsonl")
    ap.add_argument("--teacher", default="data/v11/corpus/teacher_replay.jsonl.gz")
    ap.add_argument("--subset-frac", type=float, default=1.0, help="pilot: train on a seeded fraction of the corpus")
    ap.add_argument("--init-ckpt", help="continue from a v1.1 checkpoint directory (hard-negative mini-round)")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--epochs", type=int, default=1, help="passes over the data (diagnostic runs only; R1 / R2 use 1)")
    ap.add_argument("--max-tokens", type=int, help="override batching.max_tokens (diagnostic full fine-tunes)")
    ap.add_argument("--grad-ckpt", action="store_true", help="encoder gradient checkpointing (many trainable blocks)")
    ap.add_argument("--lr-scale", type=float, default=1.0, help="multiplies every learning rate (hard-negative round: 0.5)")
    ap.add_argument("--export-v1", action="store_true")
    ap.add_argument("--batching", default="random", choices=["random", "family"], help="R3: keep counterfactual families in one batch")
    ap.add_argument("--aux-stats", default="data/v11/corpus/aux_stats.json")
    ap.add_argument("--device")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())["training"]
    obj = T.Objective(**cfg["objectives"][args.objective], **cfg["objective_constants"])
    out = ROOT / "runs/v11" / args.track / args.tag
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ser = harness.serializer(ARTIFACT)
    pad = ser.tok.pad_token_id

    tech = T.load_jsonl(ROOT / args.tech)
    replay = T.load_jsonl(ROOT / args.replay) if args.replay and args.replay != "none" and (ROOT / args.replay).exists() else []
    stats_path = ROOT / args.aux_stats
    if stats_path.exists():
        st = json.loads(stats_path.read_text())
        keys, stats = st["keys"], st["stats"]
    else:
        keys = T.cons_keys(tech)
        stats = T.cons_stats(tech, keys)
    teacher = {}
    if obj.l_kd and (ROOT / args.teacher).exists():
        with gzip.open(ROOT / args.teacher, "rt") as f:
            for line in f:
                r = json.loads(line)
                teacher[r["id"]] = r
    rng = random.Random(args.seed)
    if args.subset_frac < 1.0:      # the same subset for every pilot run (seeded by the config, not by the run seed)
        prng = random.Random(cfg["pilot_subset_seed"])
        tech = prng.sample(tech, int(len(tech) * args.subset_frac))
        replay = prng.sample(replay, int(len(replay) * args.subset_frac))
    sha = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()[:16]  # noqa: E731
    key = "|".join([sha(args.tech), (sha(args.replay) if replay else "-"), str(args.subset_frac), str(bool(teacher)), str(len(tech)), str(len(replay)),
                    sha(args.aux_stats) if stats_path.exists() else "-", "fam1"])
    cache = ROOT / "runs/v11/cache" / f"enc_{hashlib.sha1(key.encode()).hexdigest()[:12]}.pkl"
    t0 = time.time()
    items = encode_all(ser, tech, replay, keys, stats, teacher, cache, key)
    print(json.dumps({"encoded": len(items), "tech": len(tech), "replay": len(replay), "seconds": round(time.time() - t0, 1)}), flush=True)
    del tech, replay, teacher

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = load_model(args, cfg).to(dev)
    groups = T.set_trainable(model, args.mode, args.last_n)
    oc = cfg["optimizer"]
    pg = [{"params": groups["encoder"], "lr": oc["lr_encoder"]}, {"params": groups["head"], "lr": oc["lr_head"]},
          {"params": groups["new"], "lr": oc["lr_new"]}]
    opt = torch.optim.AdamW([g for g in pg if g["params"]], weight_decay=oc["weight_decay"])
    if args.grad_ckpt and dev.type == "cuda":
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lens, mt, mq = [it["length"] for it in items], args.max_tokens or cfg["batching"]["max_tokens"], cfg["batching"]["max_questions"]
    if args.batching == "family":
        fams = [it.get("family") for it in items]
        plan = [b for ep in range(args.epochs) for b in T.family_batches(lens, fams, mt, mq, np.random.default_rng(args.seed + ep))]
    else:
        plan = [b for ep in range(args.epochs) for b in T.batches(lens, mt, mq, np.random.default_rng(args.seed + ep))]
    fam_of = [it.get("family") for it in items]
    together = [len({fam_of[i] for i in b if fam_of[i]}) < sum(1 for i in b if fam_of[i]) for b in plan]
    total = min(len(plan), args.max_steps) if args.max_steps else len(plan)
    base_lr = [g["lr"] * args.lr_scale for g in opt.param_groups]
    params = [p for g in opt.param_groups for p in g["params"]]
    info = {"tag": args.tag, "args": vars(args), "objective": cfg["objectives"][args.objective], "spec": model.v11.to_dict(),
            "params_total": T.n_params(model.parameters()), "params_trainable": {k: T.n_params(v) for k, v in groups.items()},
            "items": len(items), "steps": total, "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
            "items_in_families": sum(f is not None for f in fam_of), "batches_with_siblings": float(np.mean(together)) if together else 0.0}
    print(json.dumps({k: v for k, v in info.items() if k not in ("spec",)}, default=str), flush=True)
    model.train()
    if args.mode == "head":
        model.encoder.eval()
    log, t0 = [], time.time()
    acc = {"tech": [0, 0], "replay": [0, 0]}
    agg: dict[str, list[float]] = {}
    for step in range(total):
        for g, b in zip(opt.param_groups, base_lr):
            g["lr"] = b * T.lr_at(step, total, int(oc["warmup_ratio"] * total))
        batch = [items[i] for i in plan[step]]
        out_, merged, rows = T.forward_batch(model, batch, pad, dev)
        loss, parts = T.loss_fn(out_, merged, rows, batch, obj, dev)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(params, oc["max_grad_norm"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        for k, v in parts.items():
            agg.setdefault(k, []).append(v)
        pred = merged["rank"].detach().argmax(-1).cpu().numpy()
        for it, pr in zip(batch, pred):
            if it["answerable"] and it["qtype"] in (0, 1, 2):
                a = acc[it["kind"]]
                a[0] += int(pr == int(it["target"].argmax()))
                a[1] += 1
        if (step + 1) % 100 == 0 or step + 1 == total:
            e = {"step": step + 1, **{k: round(float(np.mean(v)), 4) for k, v in agg.items()},
                 "acc_tech": round(acc["tech"][0] / max(1, acc["tech"][1]), 4), "acc_replay": round(acc["replay"][0] / max(1, acc["replay"][1]), 4),
                 "seconds": round(time.time() - t0, 1)}
            if dev.type == "cuda":
                e["peak_vram_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
            log.append(e)
            print(json.dumps(e), flush=True)
            agg, acc = {}, {"tech": [0, 0], "replay": [0, 0]}
    from safetensors.torch import save_file, save_model

    save_model(model, str(out / "model.safetensors"))
    (out / "spec.json").write_text(json.dumps(model.v11.to_dict(), indent=1))
    if args.export_v1 and model.v11.v1_compatible:
        save_file({k: v.contiguous() for k, v in model.v1_state_dict().items()}, str(out / "model_v1.safetensors"))
    info.update(train_seconds=round(time.time() - t0, 1), log=log, gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
                model_sha256=hashlib.sha256((out / "model.safetensors").read_bytes()).hexdigest())
    (out / "run.json").write_text(json.dumps(info, indent=1, default=str))
    print(json.dumps({"done": args.tag, "seconds": info["train_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
