"""Krun One v1.1 — latency / memory of the architecture candidates with RANDOM weights (latency does not depend on
weight values), measured BEFORE any training, with v1's own harness (scripts/v1/latency_proxy.py: real serializer,
real tokenizer, real text, staged timings) on the same GPU in the same process as the v1 shape.

    python scripts/v11/latency_v11.py --out results/v11/architecture_search/latency_rtx4090.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import resource
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("latency_proxy", ROOT / "scripts/v1/latency_proxy.py")
lp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lp)

from krun_one.v1.serialization import Budget, Serializer  # noqa: E402
from krun_one.v11 import candidates  # noqa: E402
from krun_one.v11.model import KrunOneV11Model, count_parameters_v11  # noqa: E402

WORKLOADS = ["typical_1q_k4_ctx48", "typical_1q_tools8_ctx48", "typical_4q_mixed_ctx96", "k_8_tools_ctx64",
             "k_32_tools_ctx64", "k_64_tools_ctx64", "k_128_tools_ctx64", "len_512_k4", "len_1024_k4",
             "q_16_tools8_ctx256", "stress_16q_tools32_ctx512", "throughput_32req_k4_ctx48"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=str(ROOT / "artifacts/krun-one-v1-text-final/tokenizer"))
    ap.add_argument("--corpus", default=str(ROOT / "results/v1/latency_corpus.json"))
    ap.add_argument("--locales", default="en-US,pt-PT")
    ap.add_argument("--dtypes", default="autocast,bf16")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--workloads", default=",".join(WORKLOADS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    assert torch.cuda.is_available()
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    corpus = json.loads(Path(args.corpus).read_text())
    wls = [w for w in lp.workloads() if w["name"] in args.workloads.split(",")]
    specs = candidates.ENCODER_CANDIDATES + candidates.HEAD_CANDIDATES[1:]
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
           "python": platform.python_version(), "iters": args.iters, "results": [], "memory": {}}
    for spec, route in specs:
        ser = Serializer(tok, Budget(max_len=1024, head_max_len=896))
        torch.manual_seed(0)
        model = KrunOneV11Model.from_spec(spec).to(dev).eval()
        for dtype in args.dtypes.split(","):
            if dtype == "bf16":
                model = model.to(torch.bfloat16)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            weights_mib = round(torch.cuda.memory_allocated() / 2**20)
            peaks = {}
            for wl in wls:
                torch.cuda.reset_peak_memory_stats()
                for loc in args.locales.split(","):
                    r = lp.measure(model, ser, tok, corpus, wl, loc, dev, dtype, args.iters, args.warmup)
                    r.update({"spec": spec.name, "dtype": dtype, "workload": wl["name"], "locale": loc, "growth": route})
                    out["results"].append(r)
                    print(f"{spec.name:26s} {dtype:8s} {wl['name']:28s} {loc} seqs={r['sequences']:3d} len={r['padded_len']:4d} "
                          f"fwd p50 {r['forward_p50_ms']:7.2f} total p50 {r['total_p50_ms']:7.2f} p95 {r['total_p95_ms']:7.2f}", flush=True)
                peaks[wl["name"]] = round(torch.cuda.max_memory_allocated() / 2**20)
            out["memory"][f"{spec.name}/{dtype}"] = {
                "params": count_parameters_v11(spec)["total"], "weights_vram_mib": weights_mib, "peak_vram_mib_by_workload": peaks,
                "peak_vram_mib": max(peaks.values()), "process_max_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)}
        del model
        torch.cuda.empty_cache()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
