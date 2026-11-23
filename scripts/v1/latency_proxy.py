"""Krun One v1 — latency proxy: random-init candidate architectures under production-shaped workloads.

Latency does not depend on weight values, so each candidate is built from its spec (krun_one.v1.arch) with
random weights and measured with the *real* serializer + tokenizer on real text (MASSIVE utterances as
contexts, official xLAM/ToolACE tool strings as candidates: results/v1/latency_corpus.json).

Each iteration is timed in stages, like the worker's timings:
    tokenize   serialization + tokenization (tokenizer cache cleared every iteration)
    collate    padding into tensors (CPU)
    h2d        pinned host -> device copy
    forward    model forward (CUDA-synchronized)
    post       chunk merge + softmax + device -> host
    total      sum of the above (= the model-specific part of the worker)

The reference (v0.3 shape, ModernBERT tokenizer, v0.3 serialization without [NULL]) runs through the same code,
so "x v0.3" ratios include the tokenizer effect (PT/ES/FR contexts need ~1/3 fewer Gemma tokens).

Weight modes: ``autocast`` = fp32 weights + bf16 autocast (what the production worker does today);
``bf16`` = weights cast to bf16 (a free serving optimization to evaluate).

    python scripts/v1/latency_proxy.py --specs all --gemma-tokenizer <dir> --ref-tokenizer artifacts/krun-one-v0.3/tokenizer \
        --out results/v1/latency/rtx4090.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from krun_one.v1 import arch  # noqa: E402
from krun_one.v1.model import KrunOneV1Model, chunk_index, collate_chunks, merge_chunks  # noqa: E402
from krun_one.v1.serialization import Budget, Serializer  # noqa: E402


def build_context(corpus, locale: str, target_tokens: int, tok, offset: int) -> str:
    utts = corpus["contexts"][locale]
    parts, i = [], offset
    while True:
        parts.append(utts[i % len(utts)])
        i += 1
        text = ". ".join(parts)
        if len(tok(text, add_special_tokens=False)["input_ids"]) >= target_tokens or len(parts) > 400:
            return text


def question(corpus, kind: str, k: int, offset: int) -> dict:
    tools = corpus["tools"]
    if kind == "tools":
        return {"type": "choice", "candidates": [{"id": t["name"], "label": t["name"], "description": t["description"]}
                                                 for t in (tools[(offset + j) % len(tools)] for j in range(k))],
                "instructions": "Choose the function that should be called to fulfil the user request."}
    if kind == "labels":
        return {"type": "choice", "candidates": [{"id": t["name"]} for t in (tools[(offset + j) % len(tools)] for j in range(k))],
                "instructions": "Choose the option that best describes the intent of the user message."}
    if kind == "noul":
        return {"type": "noul", "candidates": [{"id": "false"}, {"id": "true"}],
                "instructions": "Is the customer asking to cancel a subscription?"}
    if kind == "score":
        return {"type": "score", "candidates": [{"id": str(i), "label": x} for i, x in enumerate(
            ["not urgent", "low", "medium", "high", "critical"])], "instructions": "How urgent is the request?"}
    raise ValueError(kind)


def workloads() -> list[dict]:
    w = [
        # production-shaped (KRUN-014: prod requests average ~43 tokens/question, 1-16 questions)
        {"name": "typical_1q_k4_ctx48", "ctx": 48, "questions": [("labels", 4)], "typical": True},
        {"name": "typical_1q_tools8_ctx48", "ctx": 48, "questions": [("tools", 8)], "typical": True},
        {"name": "typical_4q_mixed_ctx96", "ctx": 96, "questions": [("tools", 8), ("noul", 2), ("score", 5), ("labels", 4)],
         "typical": True},
    ]
    for L in (128, 256, 512, 1024):
        w.append({"name": f"len_{L}_k4", "ctx": max(16, L - 40), "questions": [("labels", 4)]})
    for k in (2, 4, 8, 16, 32, 64, 128):
        w.append({"name": f"k_{k}_tools_ctx64", "ctx": 64, "questions": [("tools", k)]})
    for n in (1, 4, 16):
        w.append({"name": f"q_{n}_tools8_ctx256", "ctx": 256, "questions": [("tools", 8)] * n})
    w.append({"name": "stress_16q_tools32_ctx512", "ctx": 512, "questions": [("tools", 32)] * 16})
    w.append({"name": "throughput_32req_k4_ctx48", "ctx": 48, "questions": [("labels", 4)] * 32})
    return w


def pct(xs, p):
    return float(np.percentile(np.asarray(xs), p))


@torch.inference_mode()
def measure(model, ser: Serializer, tok, corpus, wl: dict, locale: str, dev, dtype, iters: int, warmup: int) -> dict:
    ctx = build_context(corpus, locale, wl["ctx"], tok, offset=7)
    qs_spec = [question(corpus, kind, k, offset=13 * j) for j, (kind, k) in enumerate(wl["questions"])]
    rec = []
    shapes = None
    for it in range(warmup + iters):
        ser._cache.clear()
        t0 = time.perf_counter()
        qs = [ser.encode(q["type"], ctx, q["candidates"], q["instructions"]) for q in qs_spec]
        t1 = time.perf_counter()
        flat, where = chunk_index(qs)
        b = collate_chunks(flat, tok.pad_token_id)
        t2 = time.perf_counter()
        b = {k: v.pin_memory().to(dev, non_blocking=True) for k, v in b.items()}
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == "autocast"):
            out = model(**b)
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        m = merge_chunks(out, where, len(qs), [q.n_candidates for q in qs])
        p = torch.softmax(m["rank"], -1).cpu().numpy()
        _ = p.sum()
        t5 = time.perf_counter()
        if it >= warmup:
            rec.append([t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t5 - t0])
        shapes = (b["input_ids"].shape[0], b["input_ids"].shape[1], int(b["attention_mask"].sum()))
    a = np.asarray(rec) * 1000
    names = ["tokenize", "collate", "h2d", "forward", "post", "total"]
    return {"sequences": shapes[0], "padded_len": shapes[1], "real_tokens": shapes[2],
            "questions": len(qs_spec),
            **{f"{n}_p50_ms": round(pct(a[:, i], 50), 3) for i, n in enumerate(names)},
            **{f"{n}_p95_ms": round(pct(a[:, i], 95), 3) for i, n in enumerate(names) if n in ("forward", "total")}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default="all")
    ap.add_argument("--gemma-tokenizer", required=True)
    ap.add_argument("--ref-tokenizer", default=str(ROOT / "artifacts/krun-one-v0.3/tokenizer"))
    ap.add_argument("--corpus", default=str(ROOT / "results/v1/latency_corpus.json"))
    ap.add_argument("--locales", default="pt-PT,en-US")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--dtypes", default="autocast,bf16")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--workloads", default="all")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    assert torch.cuda.is_available(), "latency proxy needs a CUDA GPU"
    dev = torch.device("cuda")
    corpus = json.loads(Path(args.corpus).read_text())
    specs = [arch.REFERENCE_V03] + arch.CANDIDATES + arch.LATENCY_EXTRA
    if args.specs != "all":
        keep = set(args.specs.split(","))
        specs = [s for s in specs if s.name in keep or s.name.split(" ")[0] in keep]
    wls = workloads() if args.workloads == "all" else [w for w in workloads() if w["name"] in args.workloads.split(",")]
    gemma = AutoTokenizer.from_pretrained(args.gemma_tokenizer)
    ref_tok = AutoTokenizer.from_pretrained(args.ref_tokenizer)
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
           "python": platform.python_version(), "attn": args.attn, "iters": args.iters, "results": []}
    try:
        import transformers
        out["transformers"] = transformers.__version__
    except Exception:
        pass
    for spec in specs:
        is_ref = spec is arch.REFERENCE_V03
        tok = ref_tok if is_ref else gemma
        ser = Serializer(tok, Budget(max_len=1024, head_max_len=896), use_null=not (is_ref or spec.name.endswith("-plain")))
        torch.manual_seed(0)
        model = KrunOneV1Model.from_spec(spec, attn_implementation=args.attn).to(dev).eval()
        for dtype in args.dtypes.split(","):
            if dtype == "bf16":
                model = model.to(torch.bfloat16)
            torch.cuda.reset_peak_memory_stats()
            for wl in wls:
                for loc in args.locales.split(","):
                    r = measure(model, ser, tok, corpus, wl, loc, dev, dtype, args.iters, args.warmup)
                    r.update({"spec": spec.name, "dtype": dtype, "workload": wl["name"], "locale": loc,
                              "typical": wl.get("typical", False)})
                    out["results"].append(r)
                    print(f"{spec.name:28s} {dtype:8s} {wl['name']:28s} {loc} seqs={r['sequences']:3d} "
                          f"len={r['padded_len']:4d} fwd p50 {r['forward_p50_ms']:7.2f} total p50 {r['total_p50_ms']:7.2f} "
                          f"p95 {r['total_p95_ms']:7.2f}", flush=True)
            out.setdefault("peak_vram_mib", {})[f"{spec.name}/{dtype}"] = round(torch.cuda.max_memory_allocated() / 2**20)
        del model
        torch.cuda.empty_cache()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
    Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
