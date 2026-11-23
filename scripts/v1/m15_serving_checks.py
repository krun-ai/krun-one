"""M1.5 serving checks on the GPU pod (the SERVING stack: freeze_core=False, inference_mode, no optimizer).

    python scripts/v1/m15_serving_checks.py textreg --run runs/v1/m15/<run> -> results/v1/m15/text_regression.json
    python scripts/v1/m15_serving_checks.py latency --run runs/v1/m15/<run> -> results/v1/m15/latency_rtx4090.json

All modalities are input-only: every path ends in the frozen text decision head + scorer and the v0.3 answer schema.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/v1"))
import m1_serving_checks as S1  # noqa: E402
import train_m15 as T  # noqa: E402
from krun_one.v1.model import chunk_index, collate_chunks, merge_chunks  # noqa: E402
from krun_one.v1.multimodal.features import ModalityFeatures  # noqa: E402
from krun_one.v1.multimodal.m1 import core_digest  # noqa: E402
from krun_one.v1.runtime import TextArtifact  # noqa: E402

ART = ROOT / "artifacts/krun-one-v1-text-final"
OUT = ROOT / "results/v1" / T.VARIANT
PARALLEL_OCR = T.IS_B          # M1.5b: OCR runs concurrently with SigLIP2 preprocessing + tower on scanned pages / screenshots


def stack(run: Path):
    from krun_one.v1.multimodal.m15_ocr import build_ocr, chosen_ocr, ocr_exec_options

    model, towers, tok, ser = T.build(freeze_core=False)
    tr = json.loads((run / "train_result.json").read_text())
    T.load_m15(model, run / tr["selected"] / "m15.safetensors")
    model.eval()
    ocr = build_ocr(chosen_ocr(ROOT), **(ocr_exec_options(ROOT) if T.IS_B else {}))
    return model, towers, tok, ser, ocr


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


REQ = {"task": {"type": "choice", "instructions": "What does the user want?"},
       "candidates": [{"id": x} for x in ("cancel_subscription", "check_balance", "report_fraud", "book_flight")]}


def decide(model, tok, ser, modality, context, feats):
    enc = [ser.encode_record(dict(REQ, context={"text": context, "language": "pt-BR"}))]
    flat, where = chunk_index(enc)
    b = {k: v.cuda() for k, v in collate_chunks(flat, tok.pad_token_id).items()}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        t0 = sync()
        tokm = model.fusion.media_tokens(feats)
        t1 = sync()
        o = model(b, tokm, [0], modality)
        q = merge_chunks(o, where, 1, [4])
        _ = q["rank"].float().cpu()
    t2 = sync()
    return (t1 - t0) * 1e3, (t2 - t1) * 1e3


def audio_once(model, towers, tok, ser, flac_bytes: bytes, sr_in: int):
    import soundfile as sf

    asr = towers.asr
    t = {}
    a = sync()
    w, sr = sf.read(io.BytesIO(flac_bytes), dtype="float32")
    b = sync(); t["decode_ms"] = (b - a) * 1e3
    if sr != 16000:
        n = int(round(len(w) * 16000 / sr))
        w = np.interp(np.linspace(0, len(w) - 1, n), np.arange(len(w)), w).astype(np.float32)
    c = sync(); t["resample_ms"] = (c - b) * 1e3
    wp = np.pad(w, (0, max(0, 480000 - len(w))))[:480000]
    d = sync(); t["padding_ms"] = (d - c) * 1e3
    feats = asr.proc.feature_extractor([wp], sampling_rate=16000, return_tensors="pt", device="cuda").input_features
    n = torch.tensor([int(np.ceil(len(w) / 160))])
    e = sync(); t["logmel_ms"] = (e - d) * 1e3
    enc = asr.encode(feats)
    f = sync(); t["asr_encoder_ms"] = (f - e) * 1e3
    text = asr.transcribe(enc)[0]
    g = sync(); t["asr_decoder_ms"] = (g - f) * 1e3
    h, m = asr.acoustic(enc, n)
    feats_m = ModalityFeatures("audio", h, m)
    t["audio_projector_ms"], t["fusion_decision_ms"] = decide(model, tok, ser, "audio", text, feats_m)
    t["total_ms"] = (sync() - a) * 1e3
    t["transcript_chars"] = len(text)
    return t


def cmd_textreg(args) -> None:
    run = Path(args.run)
    A = TextArtifact.load(ART, device="cuda", dtype="bf16", verify=True)
    dA = core_digest(A.model)
    sA = S1.dump_all(A.model, A.ser, A.tok.pad_token_id, run / "textreg/A")
    del A
    torch.cuda.empty_cache()
    model, towers, tok, ser, ocr = stack(run)
    lat_probe(model, towers, tok, ser, ocr, iters=3)          # serve image / document / audio traffic first
    dB = core_digest(model.core)
    sB = S1.dump_all(model.core, ser, tok.pad_token_id, run / "textreg/B")
    res = {"core_digest_A": dA, "core_digest_B": dB, "core_digest_equal": dA == dB, "signals_sha_A": sA, "signals_sha_B": sB,
           "signals_bitwise_identical": sA == sB, "requires_grad_core_params_in_B": all(p.requires_grad for p in model.core.parameters())}
    res["PASS"] = bool(res["core_digest_equal"] and res["signals_bitwise_identical"])
    (OUT / "text_regression.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def _audio_bytes(seconds: float) -> bytes:
    """Real speech of ~seconds: concatenated FLEURS val_eval clips (16 kHz FLAC in memory)."""
    import soundfile as sf

    rows = [json.loads(l) for l in open(T.D / "val_eval.jsonl") if '"source": "fleurs"' in l]
    ws, tot = [], 0.0
    for r in rows:
        w, _ = sf.read(str(ROOT / "data/m1" / r["media"]["path"]), dtype="float32")
        ws.append(w); tot += len(w) / 16000
        if tot >= seconds:
            break
    w = np.concatenate(ws)[: int(seconds * 16000)]
    bio = io.BytesIO()
    sf.write(bio, w, 16000, format="FLAC")
    return bio.getvalue()


def _shot():
    rows = [json.loads(l) for l in open(T.D / "val_eval.jsonl") if '"source": "kd1_screen"' in l]
    return str(ROOT / "data/m1" / rows[0]["media"]["path"])


_POOL = {}


def scanned(model, towers, tok, ser, ocr, ip, path, patches):
    """Page / screenshot image -> OCR text + SigLIP2 tokens -> decision. PARALLEL_OCR: OCR in a worker thread while the
    main thread preprocesses and encodes the page with SigLIP2 (ONNX Runtime and torch release the GIL)."""
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image

    t = {}
    a = sync()
    page = Image.open(path).convert("RGB")
    b = sync(); t["decode_ms"] = (b - a) * 1e3
    if PARALLEL_OCR:
        pool = _POOL.setdefault("p", ThreadPoolExecutor(1))
        fut = pool.submit(lambda: (time.perf_counter(), ocr.read([page])[0], time.perf_counter()))
        mv = dict(ip(images=[page], max_num_patches=patches, return_tensors="pt"))
        c = sync(); t["preprocess_ms"] = (c - b) * 1e3
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            h, m = towers.vision.encode(mv)
        d = sync(); t["tower_ms"] = (d - c) * 1e3
        o0, text, o1 = fut.result()
        e = sync(); t["ocr_ms"] = (o1 - o0) * 1e3; t["ocr_wait_after_tower_ms"] = (e - d) * 1e3
    else:
        text = ocr.read([page])[0]
        c = sync(); t["ocr_ms"] = (c - b) * 1e3
        mv = dict(ip(images=[page], max_num_patches=patches, return_tensors="pt"))
        d = sync(); t["preprocess_ms"] = (d - c) * 1e3
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            h, m = towers.vision.encode(mv)
        e = sync(); t["tower_ms"] = (e - d) * 1e3
    t["projector_ms"], t["fusion_decision_ms"] = decide(model, tok, ser, "document", text, ModalityFeatures("document", h, m))
    t["total_ms"] = (sync() - a) * 1e3
    return t


def _page():
    rows = [json.loads(l) for l in open(T.D / "val_eval.jsonl") if '"source": "synthdoc"' in l]
    return str(ROOT / "data/m1" / rows[0]["media"]["path"])


def _pdf_with_text_layer(text: str) -> bytes:
    """Minimal single-page PDF with a real text layer (Helvetica)."""
    lines = [l.replace("(", "[").replace(")", "]") for l in text.split("\n")][:40]
    content = "BT /F1 11 Tf 50 780 Td 14 TL " + " ".join(f"({l}) Tj T*" for l in lines) + " ET"
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>", f"<< /Length {len(content)} >>\nstream\n{content}\nendstream"]
    out, offs = "%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out.encode("latin-1", "replace")))
        out += f"{i} 0 obj\n{o}\nendobj\n"
    x = len(out.encode("latin-1", "replace"))
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n" + "".join(f"{o:010d} 00000 n \n" for o in offs)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{x}\n%%EOF"
    return out.encode("latin-1", "replace")


def lat_probe(model, towers, tok, ser, ocr, iters=50, warmup=10):
    from PIL import Image

    from krun_one.v1.multimodal.documents import DocumentInput, DocumentRouter
    from krun_one.v1.multimodal.config import DocumentConfig
    from krun_one.v1.multimodal.m1_towers import PATCHES, image_preprocessor

    ip = image_preprocessor(T.CFG["towers"]["vision"])
    router = DocumentRouter(DocumentConfig(enabled=True))
    img = Image.new("RGB", (640, 480), (200, 180, 160))
    page_path = _page()
    meta_text = json.loads(Path(page_path.replace(".jpg", ".json")).read_text())["text"]
    pdf = _pdf_with_text_layer(meta_text)
    audio = {s: _audio_bytes(s) for s in (5, 15, 30)}
    shot_path = _shot()
    res = {k: [] for k in ("image", "document_scanned", "screenshot", "document_text_layer", "audio_5s", "audio_15s", "audio_30s")}
    for it in range(warmup + iters):
        t = {}
        a = sync()
        mv = dict(ip(images=[img], max_num_patches=PATCHES["image"], return_tensors="pt"))
        b = sync(); t["preprocess_ms"] = (b - a) * 1e3
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            h, m = towers.vision.encode(mv)
        c = sync(); t["tower_ms"] = (c - b) * 1e3
        t["projector_ms"], t["fusion_decision_ms"] = decide(model, tok, ser, "image", "", ModalityFeatures("image", h, m))
        t["total_ms"] = (sync() - a) * 1e3
        if it >= warmup:
            res["image"].append(t)
        for kind, pth in (("document_scanned", page_path), ("screenshot", shot_path)):
            t = scanned(model, towers, tok, ser, ocr, ip, pth, PATCHES["document"])
            if it >= warmup:
                res[kind].append(t)
        # PDF with a text layer: extraction -> text fast path (no OCR, no tower)
        t = {}
        a = sync()
        r = router.route(DocumentInput(pdf, "application/pdf"))
        b = sync(); t["extract_ms"] = (b - a) * 1e3; t["route"] = r.route
        enc = [ser.encode_record(dict(REQ, context={"text": r.text or "", "language": "pt-BR"}))]
        flat, where = chunk_index(enc)
        bb = {k: v.cuda() for k, v in collate_chunks(flat, tok.pad_token_id).items()}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            _ = merge_chunks(model.core(**bb), where, 1, [4])["rank"].float().cpu()
        c = sync(); t["krun_text_decision_ms"] = (c - b) * 1e3; t["total_ms"] = (c - a) * 1e3
        if it >= warmup:
            res["document_text_layer"].append(t)
        for s, by in audio.items():
            t = audio_once(model, towers, tok, ser, by, 16000)
            if it >= warmup:
                res[f"audio_{s}s"].append(t)
    agg = {}
    for k, rows in res.items():
        if not rows:
            continue
        agg[k] = {f"{f}_p50": float(np.percentile([r[f] for r in rows], 50)) for f in rows[0] if isinstance(rows[0][f], (int, float))}
        agg[k]["total_ms_p95"] = float(np.percentile([r["total_ms"] for r in rows], 95))
        agg[k]["sum_of_component_p50"] = float(sum(v for f, v in agg[k].items() if f.endswith("_ms_p50") and not f.startswith("total")
                                                   and not (PARALLEL_OCR and f.startswith("ocr_ms"))))
        agg[k]["parallel_ocr"] = PARALLEL_OCR and k in ("document_scanned", "screenshot")
        if "route" in rows[0]:
            agg[k]["route"] = rows[0]["route"]
    return agg


def cmd_latency(args) -> None:
    run = Path(args.run)
    model, towers, tok, ser, ocr = stack(run)
    base = TextArtifact.load(ART, device="cuda", dtype="bf16", verify=False)
    from krun_one.v1.multimodal.m15_ocr import chosen_ocr, ocr_exec_options

    out = {"gpu": torch.cuda.get_device_name(0), "dtype": "bf16", "iters": args.iters, "config": T.CFG["name"], "ocr": chosen_ocr(ROOT),
           "ocr_exec_options": ocr_exec_options(ROOT) if T.IS_B else "rapidocr defaults (EXHAUSTIVE, rec 6)",
           "parallel_ocr": PARALLEL_OCR, "gate": T.GATE}
    out["multimodal"] = lat_probe(model, towers, tok, ser, ocr, iters=args.iters)
    # text-only: frozen artifact vs the M1.5 stack's text path (same core object semantics)
    from latency_proxy import build_context, question

    corpus = json.loads((ROOT / "results/v1/latency_corpus.json").read_text())
    stack_art = TextArtifact(ART, model.core, base.tok, base.ser, base.calib, base.dcfg, base.device, base.dtype)
    out["text_only"] = {}
    for name, (n_ctx, qs) in {"typical_1q_k4_ctx48": (48, [("labels", 4)]), "k_128_tools_ctx64": (64, [("tools", 128)])}.items():
        ctx = build_context(corpus, "pt-PT", n_ctx, base.tok, offset=7)
        req = {"context": ctx, "questions": {}}
        for j, (kind, k) in enumerate(qs):
            q = question(corpus, kind, k, offset=13 * j)
            req["questions"][f"q{j}"] = {"type": "choice", "task_type": "tool" if kind == "tools" else "intent",
                                         "options": [{"id": c["id"], **({"description": c["description"]} if c.get("description") else {})} for c in q["candidates"]]}
        tA, tB = [], []
        for it in range(10 + args.iters):
            for obj, acc in ((base, tA), (stack_art, tB)) if it % 2 == 0 else ((stack_art, tB), (base, tA)):
                a = sync(); obj.decide(req); b = sync()
                if it >= 10:
                    acc.append((b - a) * 1e3)
        same = json.dumps(base.decide(req)["answers"], sort_keys=True) == json.dumps(stack_art.decide(req)["answers"], sort_keys=True)
        out["text_only"][name] = {"frozen_artifact_p50_ms": float(np.percentile(tA, 50)), "m15_stack_p50_ms": float(np.percentile(tB, 50)),
                                  "delta_pct": float(100 * (np.percentile(tB, 50) / np.percentile(tA, 50) - 1)), "answers_identical": same}
    mm = out["multimodal"]
    out["GL_pass"] = bool(all(v["delta_pct"] <= 5 for v in out["text_only"].values()) and mm["image"]["total_ms_p50"] <= 150
                          and mm["document_scanned"]["total_ms_p50"] <= 150)
    out["audio_5s_soft_target_200ms"] = mm["audio_5s"]["total_ms_p50"] <= 200
    (OUT / ("latency_rtx4090" + (args.tag or "") + ".json")).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["textreg", "latency"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"textreg": cmd_textreg, "latency": cmd_latency}[a.cmd](a)
