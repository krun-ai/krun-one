"""Text-only latency: v1 BEFORE the multimodal scaffold vs AFTER (same weights, same GPU, interleaved measurement).

Variants (identical K1 weights):
  pre       results/v1/multimodal/model_pre_multimodal.py (frozen copy of v1 model.py before the scaffold)
  core      current KrunOneV1Model (has the optional extra_tokens hook; text-only call)
  wrapper   KrunOneV1Multimodal with ALL modalities enabled (mock towers configured) serving a TEXT-ONLY request
Workloads: typical 1 question, 4 mixed questions, 32 tools, 128 tools (real serializer + tokenizer + corpus).

    python scripts/v1/bench_text_regression.py --dtype fp16 --out results/v1/multimodal/text_regression_<gpu>.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/v1"))

from krun_one.v1 import arch  # noqa: E402
from krun_one.v1.model import KrunOneV1Model, chunk_index, collate_chunks  # noqa: E402
from krun_one.v1.multimodal.config import AudioConfig, DocumentConfig, MultimodalConfig, VisionConfig  # noqa: E402
from krun_one.v1.multimodal.model import KrunOneV1Multimodal  # noqa: E402
from krun_one.v1.serialization import Budget, Serializer  # noqa: E402
from latency_proxy import build_context, question  # noqa: E402

WORKLOADS = {"typical_1q_k4": (48, [("labels", 4)]),
             "typical_4q_mixed": (96, [("tools", 8), ("noul", 2), ("score", 5), ("labels", 4)]),
             "tools_32": (64, [("tools", 32)]), "tools_128": (64, [("tools", 128)])}


def load_pre():
    spec = importlib.util.spec_from_file_location("krun_one.v1._pre_mm_model", ROOT / "results/v1/multimodal/model_pre_multimodal.py")
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "krun_one.v1"
    spec.loader.exec_module(mod)
    return mod.KrunOneV1Model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--gemma-tokenizer", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    dev = torch.device("cuda")
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    tok = AutoTokenizer.from_pretrained(args.gemma_tokenizer)
    corpus = json.loads((ROOT / "results/v1/latency_corpus.json").read_text())
    spec = arch.CANDIDATES[0]
    torch.manual_seed(0)
    core = KrunOneV1Model.from_spec(spec, attn_implementation="sdpa").to(dev, dt).eval()
    pre = load_pre().from_spec(spec, attn_implementation="sdpa")       # built on CPU, then SHARES core's GPU tensors
    pre.load_state_dict(core.state_dict(), strict=True, assign=True)    # identical weights, no second copy in VRAM
    pre = pre.to(dev, dt).eval()
    mm_cfg = MultimodalConfig(vision=VisionConfig(enabled=True, model="mock"), document=DocumentConfig(enabled=True),
                              audio=AudioConfig(enabled=True, model="mock"))
    wrapper = KrunOneV1Multimodal(core, mm_cfg).eval()
    ser = Serializer(tok, Budget(max_len=1024, head_max_len=896))
    variants = {"pre": lambda b: pre(**b), "core": lambda b: core(**b), "wrapper": lambda b: wrapper(b)}
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "dtype": args.dtype, "results": {}}
    for name, (ctx_tokens, qspec) in WORKLOADS.items():
        ctx = build_context(corpus, "pt-PT", ctx_tokens, tok, offset=7)
        qs = [question(corpus, kind, k, offset=13 * j) for j, (kind, k) in enumerate(qspec)]
        enc = [ser.encode(q["type"], ctx, q["candidates"], q["instructions"]) for q in qs]
        flat, _ = chunk_index(enc)
        b = {k: v.to(dev) for k, v in collate_chunks(flat, tok.pad_token_id).items()}
        times = {v: [] for v in variants}
        outs = {}
        with torch.inference_mode():
            for it in range(args.warmup + args.iters):
                order = list(variants) if it % 2 == 0 else list(reversed(variants))   # interleave: cancel drift
                for v in order:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    o = variants[v](b)
                    torch.cuda.synchronize()
                    if it >= args.warmup:
                        times[v].append((time.perf_counter() - t0) * 1000)
                    outs[v] = o
        eq = all(torch.equal(outs["pre"][k], outs[v][k]) for v in ("core", "wrapper") for k in outs["pre"])
        r = {v: {"p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95))} for v, t in times.items()}
        r["delta_core_vs_pre_p50_pct"] = 100 * (r["core"]["p50_ms"] / r["pre"]["p50_ms"] - 1)
        r["delta_wrapper_vs_pre_p50_pct"] = 100 * (r["wrapper"]["p50_ms"] / r["pre"]["p50_ms"] - 1)
        r["outputs_bitwise_equal"] = bool(eq)
        r["sequences"], r["padded_len"] = int(b["input_ids"].shape[0]), int(b["input_ids"].shape[1])
        out["results"][name] = r
        print(name, json.dumps(r), flush=True)
    out["towers_loaded_after_text_traffic"] = wrapper.towers_loaded
    out["peak_vram_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
