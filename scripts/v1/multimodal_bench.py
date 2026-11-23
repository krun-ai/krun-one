"""Multimodal candidate: memory / load-time report and latency benchmark (GPU pod). No training.

    python scripts/v1/multimodal_bench.py memory  --config text|vision|audio|full  -> results/v1/multimodal_packaging/memory_<config>.json
    python scripts/v1/multimodal_bench.py latency                                   -> results/v1/multimodal_packaging/latency_rtx4090.json

memory: a FRESH process per configuration loads the artifact (SHA256SUMS + tower sha verification) and serves the
requests that pull in the components of that configuration; it reports load time per component, process RSS, torch
allocated / reserved VRAM and nvidia-smi used memory.
latency: one warm runtime, p50 / p95 of the runtime's own stage timings (metadata.timings_ms) + wall time per path.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/v1"))
ART = ROOT / "artifacts/krun-one-v1-multimodal-candidate"
FX = ROOT / "tests/fixtures/multimodal"
OUT = ROOT / "results/v1/multimodal_packaging"


def smi_used() -> int | None:
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True,
                                  text=True, timeout=10).stdout.split()[0])
    except Exception:  # noqa: BLE001
        return None


def rss_mib() -> int:
    import psutil

    return int(psutil.Process().memory_info().rss / 2**20)


def reqs():
    from krun_one.v1.multimodal.runtime import MediaPart, ResolvedMultimodalContext

    q_intent = {"q": {"type": "choice", "task_type": "intent", "options": [{"id": x} for x in ("cancel_subscription", "check_balance", "book_flight", "report_fraud")]}}
    q_obj = {"q": {"type": "choice", "instructions": "Which object is the main subject of the image?",
                   "options": [{"id": x} for x in ("elephant", "bus", "pizza", "train")]}}
    q_doc = {"q": {"type": "choice", "instructions": "What type of document is this?",
                   "options": [{"id": x} for x in ("invoice", "receipt", "payslip", "boarding_pass")]}}
    return {
        "text": (ResolvedMultimodalContext.from_text("quero cancelar minha assinatura, fui cobrado duas vezes"), q_intent),
        "image": (ResolvedMultimodalContext(image_inputs=[MediaPart("image/jpeg", path=str(FX / "image_elephant.jpg"))]), q_obj),
        "pdf_text_layer": (ResolvedMultimodalContext(document_inputs=[MediaPart("application/pdf", path=str(FX / "doc_invoice_textlayer.pdf"))]), q_doc),
        "scanned_page": (ResolvedMultimodalContext(document_inputs=[MediaPart("image/jpeg", path=str(FX / "doc_invoice_scan.jpg"))]), q_doc),
        "audio_5s": (ResolvedMultimodalContext(audio_inputs=[MediaPart("audio/wav", data=speech(5))]), q_intent),
    }


def speech(seconds: float) -> bytes:
    """Real TTS speech of the fixtures, concatenated (0.3 s gaps) and cut to ``seconds``; WAV bytes."""
    import soundfile as sf

    from krun_one.v1.multimodal.runtime import decode_audio

    clips = [decode_audio((FX / f).read_bytes()) for f in ("audio_en.wav", "audio_pt.mp3", "audio_es.flac", "audio_fr.ogg")]
    out, i = [], 0
    while sum(len(x) for x in out) < seconds * 16000:
        out += [clips[i % 4], np.zeros(4800, np.float32)]
        i += 1
    w = np.concatenate(out)[: int(seconds * 16000)]
    b = io.BytesIO()
    sf.write(b, w, 16000, format="WAV")
    return b.getvalue()


def memory(config: str) -> None:
    from krun_one.v1.multimodal.runtime import KrunOneMultimodalRuntime

    base_smi = smi_used()
    t0 = time.perf_counter()
    rt = KrunOneMultimodalRuntime.load(ART, device="cuda", dtype="bf16", verify=True)
    load_total = time.perf_counter() - t0
    R = reqs()
    use = {"text": ["text", "pdf_text_layer"], "vision": ["text", "image", "scanned_page"], "audio": ["text", "audio_5s"],
           "full": ["text", "image", "pdf_text_layer", "scanned_page", "audio_5s"]}[config]
    first = {}
    for k in use:
        a = time.perf_counter()
        rt.decide(*R[k])
        first[k] = (time.perf_counter() - a) * 1e3
    torch.cuda.synchronize()
    res = {"config": config, "gpu": torch.cuda.get_device_name(0), "loaded": rt.loaded, "load_seconds": rt.load_seconds,
           "artifact_load_total_s_incl_sha": load_total, "first_request_ms_incl_lazy_load": first, "rss_mib": rss_mib(),
           "torch_allocated_mib": int(torch.cuda.memory_allocated() / 2**20), "torch_reserved_mib": int(torch.cuda.memory_reserved() / 2**20),
           "torch_peak_allocated_mib": int(torch.cuda.max_memory_allocated() / 2**20), "nvidia_smi_used_mib": smi_used(),
           "nvidia_smi_used_before_mib": base_smi}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"memory_{config}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


def latency(iters: int, warmup: int) -> None:
    from latency_proxy import build_context, question

    from krun_one.v1.multimodal.runtime import KrunOneMultimodalRuntime, MediaPart, ResolvedMultimodalContext

    rt = KrunOneMultimodalRuntime.load(ART, device="cuda", dtype="bf16", verify=False)
    R = reqs()
    corpus = json.loads((ROOT / "results/v1/latency_corpus.json").read_text())
    tok = rt.text.tok
    ctx48 = build_context(corpus, "pt-PT", 48, tok, offset=7)
    ctx64 = build_context(corpus, "pt-PT", 64, tok, offset=7)
    q4 = question(corpus, "labels", 4, 0)
    q128 = question(corpus, "tools", 128, 0)
    shot = (FX / "screenshot_support_chat.png").read_bytes()      # rendered fixture (PNG)
    paths = {
        "text_typical": (ResolvedMultimodalContext.from_text(ctx48),
                         {"q": {"type": "choice", "task_type": "intent", "options": [{"id": c["id"]} for c in q4["candidates"]]}}),
        "text_128_tools": (ResolvedMultimodalContext.from_text(ctx64),
                           {"q": {"type": "choice", "task_type": "tool", "options": [{"id": c["id"], "description": c["description"]} for c in q128["candidates"]]}}),
        "image": R["image"],
        "screenshot": (ResolvedMultimodalContext(document_inputs=[MediaPart("image/png", data=shot)]), R["text"][1]),
        "pdf_text_layer": R["pdf_text_layer"],
        "scanned_page": R["scanned_page"],
        "audio_5s": (ResolvedMultimodalContext(audio_inputs=[MediaPart("audio/wav", data=speech(5))]), R["text"][1]),
        "audio_15s": (ResolvedMultimodalContext(audio_inputs=[MediaPart("audio/wav", data=speech(15))]), R["text"][1]),
        "audio_30s": (ResolvedMultimodalContext(audio_inputs=[MediaPart("audio/wav", data=speech(30))]), R["text"][1]),
    }
    out = {"gpu": torch.cuda.get_device_name(0), "dtype": "bf16", "iters": iters, "results": {}}
    for name, (ctx, q) in paths.items():
        wall, stages = [], {}
        for it in range(warmup + iters):
            torch.cuda.synchronize()
            a = time.perf_counter()
            r = rt.decide(ctx, q)
            torch.cuda.synchronize()
            if it >= warmup:
                wall.append((time.perf_counter() - a) * 1e3)
                for k, v in r["metadata"]["timings_ms"].items():
                    stages.setdefault(k, []).append(v)
        out["results"][name] = {"wall_p50_ms": float(np.percentile(wall, 50)), "wall_p95_ms": float(np.percentile(wall, 95)),
                                "stages_p50_ms": {k: float(np.percentile(v, 50)) for k, v in stages.items()},
                                "path": r["metadata"]["path"], "audio_duration_ms": r["metadata"]["audio_duration_ms"],
                                "asr_truncation_risk": r["metadata"]["asr_truncation_risk"]}
        print(name, json.dumps(out["results"][name]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latency_rtx4090.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["memory", "latency"])
    ap.add_argument("--config", default="full")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8)
    a = ap.parse_args()
    memory(a.config) if a.cmd == "memory" else latency(a.iters, a.warmup)
