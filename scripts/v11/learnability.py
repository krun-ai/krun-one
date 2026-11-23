"""Krun One v1.1 R0-L — learnability / capacity diagnostic (THROWAWAY weights; nothing is saved or promoted).

Question it answers: when the probe tasks are actually trained, is the limit the number of parameters, the decision
head, or neither? Every run trains on data/v11/probe/ltrain.jsonl (both conditions) with v1's own objective
(configs/v1/finetune.yaml `loss`) for a fixed number of steps and is evaluated on the held-out probe.

    --init k1-final      the frozen v1 text model (879M), copied into memory
    --init k1-0b         mmBERT-base HyperCloned x2 = v1's pre-decision init: SAME function as mmbert-base, 4.4x body params
    --init mmbert-base   307M (110M body)        --init mmbert-small  140M (42M body)
    --train full | head  (head = encoder frozen: decision head + scorer [+ new blocks] only — the R1 setting)
    --head-extra N       N extra identity-initialized decision-head layers          (deeper head)
    --head-ffn-mult M    wider decision-head FFN (function-preserving)               (wider head)
    --cand-layers N      N candidate-interaction layers over the candidate vectors   (krun_one.v11.model)

    python scripts/v11/learnability.py --init k1-final --train head --tag k1final_head
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from krun_one.v1.losses import LossConfig, decision_loss  # noqa: E402
from krun_one.v1.arch import QTYPES_V1  # noqa: E402
from krun_one.v11 import growth, harness, probe  # noqa: E402
from krun_one.v11.model import KrunOneV11Model  # noqa: E402

ARTIFACT = ROOT / "artifacts/krun-one-v1-text-final"
LOSS = LossConfig(unanswerable_mode="oe", w_oe=0.5, w_ans=0.0, w_apply=0.25)   # = configs/v1/finetune.yaml (L_current)


def encode_all(ser, problems, cache: Path) -> list:
    if cache.exists():
        got = pickle.loads(cache.read_bytes())
        if got["n"] == len(problems):
            return got["items"]
    items = []
    for p in problems:
        for cond in probe.CONDITIONS:
            q = harness.encode(ser, p, cond)
            for c in q.chunks:
                c.ids = np.asarray(c.ids, dtype=np.int32)
            tgt = np.zeros(p["k"], dtype=np.float32)
            if p["answerable"]:
                tgt[p["gold"]] = 1.0
            items.append({"q": q, "target": tgt, "answerable": p["answerable"], "slice": p["slice"], "cond": cond,
                          "k": p["k"], "length": harness.length(q)})
    cache.write_bytes(pickle.dumps({"n": len(problems), "items": items}, protocol=pickle.HIGHEST_PROTOCOL))
    return items


def build(args):
    if args.init == "k1-final":
        model = harness.load_v1(ARTIFACT)
    elif args.init == "k1-0b":
        model = harness.load_mmbert("mmbert-base", widen=True, seed=args.seed)
    else:
        model = harness.load_mmbert(args.init, seed=args.seed)
    if args.head_extra or args.head_ffn_mult:
        spec = growth.grown_spec(model.spec, model.spec.name + "+head", head_layers=model.spec.decision.head_layers + args.head_extra,
                                 head_ffn_mult=args.head_ffn_mult or None)
        model, _ = growth.grow_exact(model, spec, seed=args.seed)
    if args.cand_layers:
        model = KrunOneV11Model.from_v1(model, args.cand_layers)
    return model


def batches(items, max_tokens: int, max_q: int, rng: np.random.Generator):
    """Endless stream of length-bucketed batches (reshuffled every pass)."""
    lengths = np.array([it["length"] for it in items])
    while True:
        order = rng.permutation(len(items))
        for s in range(0, len(order), 4096):
            part = order[s:s + 4096]
            part = part[np.argsort(lengths[part], kind="stable")]
            win, cur, mx = [], [], 0
            for i in part:
                new = max(mx, int(lengths[i]))
                if cur and (new * (len(cur) + 1) > max_tokens or len(cur) >= max_q):
                    win.append(cur)
                    cur, new = [], int(lengths[i])
                cur.append(int(i))
                mx = new
            if cur:
                win.append(cur)
            for j in rng.permutation(len(win)):
                yield win[j]


def targets(batch, dev):
    kmax = max(len(it["target"]) for it in batch)
    tgt = torch.zeros(len(batch), kmax)
    for i, it in enumerate(batch):
        tgt[i, : len(it["target"])] = torch.from_numpy(it["target"])
    return (tgt.to(dev), torch.full((len(batch),), QTYPES_V1["choice"], device=dev),
            torch.tensor([it["answerable"] for it in batch], device=dev))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, choices=["k1-final", "k1-0b", "mmbert-base", "mmbert-small"])
    ap.add_argument("--train", required=True, choices=["full", "head"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=12288)
    ap.add_argument("--max-questions", type=int, default=32)
    ap.add_argument("--lr-encoder", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--head-extra", type=int, default=0)
    ap.add_argument("--head-ffn-mult", type=int, default=0)
    ap.add_argument("--cand-layers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--train-file", default="data/v11/probe/ltrain.jsonl")
    ap.add_argument("--only-cond", default="", help="train on one condition only (exploratory runs)")
    ap.add_argument("--smoke", action="store_true", help="CPU, tiny subset, 3 steps")
    args = ap.parse_args()

    out = ROOT / "results/v11/capacity_probe/learnability" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    ser = harness.serializer(ARTIFACT)
    pad = ser.tok.pad_token_id
    train_path = ROOT / args.train_file
    train_p = harness.load_problems(train_path)
    test_p = harness.load_problems(ROOT / "data/v11/probe/probe.jsonl")
    if args.smoke:
        train_p, test_p, args.steps = random.Random(0).sample(train_p, 24), random.Random(0).sample(test_p, 12), 3
        items = encode_all(ser, train_p, out / "enc_smoke.pkl")
    else:
        items = encode_all(ser, train_p, train_path.with_name(train_path.stem + "_encoded.pkl"))
    if args.only_cond:
        items = [it for it in items if it["cond"] == args.only_cond]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = build(args).to(dev)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    if args.train == "head":
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    elif model.encoder.config.hidden_size >= 1536 and dev.type == "cuda":
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    groups = [{"params": [p for p in model.parameters() if id(p) in enc_ids and p.requires_grad], "lr": args.lr_encoder},
              {"params": [p for p in model.parameters() if id(p) not in enc_ids], "lr": args.lr_head}]
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    warm = max(1, int(0.05 * args.steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (args.steps - s) / args.steps))
    n_params = sum(p.numel() for p in model.parameters())
    emb = model.encoder.embeddings.tok_embeddings.weight.numel()
    info = {"tag": args.tag, "args": vars(args), "spec": model.spec.to_dict(), "params_total": n_params,
            "params_non_embedding": n_params - emb, "params_trainable": sum(p.numel() for g in groups for p in g["params"]),
            "train_examples": len(items), "loss": "v1 finetune.yaml loss (oe, w_apply 0.25)"}
    print(json.dumps({k: v for k, v in info.items() if k != "spec"}), flush=True)

    stream = batches(items, args.max_tokens, args.max_questions, np.random.default_rng(args.seed))
    params = [p for g in groups for p in g["params"]]
    log, t0 = [], time.time()
    acc_n = acc_c = 0.0
    run_loss = 0.0
    by: dict[str, list[int]] = {}
    model.train()
    if args.train == "head":
        model.encoder.eval()
    for step in range(1, args.steps + 1):
        batch = [items[i] for i in next(stream)]
        m = harness.forward_questions(model, [it["q"] for it in batch], pad, dev)
        tgt, qtype, ans = targets(batch, dev)
        loss, _ = decision_loss(m, qtype, tgt, ans, LOSS)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        run_loss += loss.item()
        pred = m["rank"].detach().argmax(-1).cpu().numpy()
        for it, pr in zip(batch, pred):
            if it["answerable"]:
                ok = int(pr == int(it["target"].argmax()))
                acc_n += 1
                acc_c += ok
                by.setdefault(f"{it['slice']}/{it['cond']}", []).append(ok)
        if step % 50 == 0 or step == args.steps:
            e = {"step": step, "loss": run_loss / (50 if step % 50 == 0 else step % 50), "train_acc": acc_c / max(1, acc_n),
                 "seconds": round(time.time() - t0, 1),
                 "by": {k: round(float(np.mean(v)), 3) for k, v in sorted(by.items())}}
            if dev.type == "cuda":
                e["peak_vram_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
            log.append(e)
            print(json.dumps(e), flush=True)
            run_loss, acc_n, acc_c, by = 0.0, 0.0, 0.0, {}
    info["train_seconds"] = round(time.time() - t0, 1)
    info["log"] = log
    # held-out probe + a fixed train-fit sample (same harness as the zero-shot run)
    model.eval()
    fit_pool = [p for p in train_p if p["answerable"]]
    fit_p = random.Random(1).sample(fit_pool, min(1500, len(fit_pool)))
    for name, probs in (("signals", test_p), ("signals_trainfit", fit_p)):
        with gzip.open(out / f"{name}.jsonl.gz", "wt") as f:
            for cond in probe.CONDITIONS:
                qs = [harness.encode(ser, p, cond) for p in probs]
                for p, z in zip(probs, harness.rank_logits(model, qs, pad, dev)):
                    f.write(json.dumps({"id": p["id"], "cond": cond, "z": [round(float(x), 5) for x in z]}) + "\n")
    info["total_seconds"] = round(time.time() - t0, 1)
    info["gpu"] = torch.cuda.get_device_name(0) if dev.type == "cuda" else None
    (out / "run.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({"done": args.tag, "seconds": info["total_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
