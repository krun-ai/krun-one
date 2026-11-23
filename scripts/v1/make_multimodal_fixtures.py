"""Versioned reference fixtures for the multimodal runtime (license-safe, small).

    python scripts/v1/make_multimodal_fixtures.py media --tts-python <venv with kokoro>   -> tests/fixtures/multimodal/*
    python scripts/v1/make_multimodal_fixtures.py reference --device cpu|cuda               -> expected_<device>.json

Media:
* images: 2 COCO val2017 photos, Flickr license CC BY 2.0 (attribution in ATTRIBUTION.md). Both are in the M1 val
  splits (val_eval / val_fit): never trained on, not protected.
* documents: generated here (invoice with a text layer, as PDF; the same invoice as a scanned JPEG and as an image-only
  PDF; a DOCX letter). Faker + DejaVu / Noto fonts.
* audio: Kokoro-82M TTS (Apache-2.0), one per language and container: EN wav, PT-BR mp3, ES flac, FR ogg.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
FX = ROOT / "tests/fixtures/multimodal"
ART = ROOT / "artifacts/krun-one-v1-multimodal-candidate"
IMAGES = {"image_elephant.jpg": (133631, "http://images.cocodataset.org/val2017/000000133631.jpg",
                                 "http://farm9.staticflickr.com/8289/7822152024_2cea0d2cff_z.jpg", "val_eval"),
          "image_giraffe.jpg": (272364, "http://images.cocodataset.org/val2017/000000272364.jpg",
                                "http://farm4.staticflickr.com/3354/3262516437_2f4c70fa62_z.jpg", "val_fit")}
AUDIO = [("audio_en.wav", "a", "af_heart", "I want to cancel my subscription, it is too expensive.", "WAV"),
         ("audio_pt.mp3", "p", "pf_dora", "Quero cancelar minha assinatura, está muito cara.", "MP3"),
         ("audio_es.flac", "e", "ef_dora", "¿Cuál es el saldo de mi cuenta bancaria?", "FLAC"),
         ("audio_fr.ogg", "f", "ff_siwis", "Je voudrais réserver un vol pour Paris demain matin.", "OGG")]
TTS = r'''
import sys, json, numpy as np, soundfile as sf
from scipy.signal import resample_poly
from kokoro import KPipeline
jobs = json.loads(sys.argv[1])
for path, lc, voice, text, fmt in jobs:
    p = KPipeline(lang_code=lc, repo_id="hexgrad/Kokoro-82M")
    w = np.concatenate([np.asarray(a, dtype=np.float32) for _, _, a in p(text, voice=voice, speed=1.0) if a is not None])
    w = resample_poly(w, 2, 3).astype(np.float32)
    kw = {"subtype": "MPEG_LAYER_III"} if fmt == "MP3" else ({"subtype": "VORBIS"} if fmt == "OGG" else {})
    sf.write(path, w, 16000, format=fmt, **kw)
    print(path, len(w) / 16000)
'''

INTENTS = [{"id": "cancel_subscription"}, {"id": "check_balance"}, {"id": "book_flight"}, {"id": "report_fraud"},
           {"id": "play_music"}, {"id": "set_alarm"}]


def pdf_with_text_layer(text: str) -> bytes:
    lines = [l.replace("(", "[").replace(")", "]") for l in text.split("\n")][:40]
    content = "BT /F1 11 Tf 50 780 Td 14 TL " + " ".join(f"({l}) Tj T*" for l in lines) + " ET"
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
            f"<< /Length {len(content.encode('latin-1', 'replace'))} >>\nstream\n{content}\nendstream"]
    out, offs = "%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out.encode("latin-1", "replace")))
        out += f"{i} 0 obj\n{o}\nendobj\n"
    x = len(out.encode("latin-1", "replace"))
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n" + "".join(f"{o:010d} 00000 n \n" for o in offs)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{x}\n%%EOF"
    return out.encode("latin-1", "replace")


def media(tts_python: str) -> None:
    from krun_one.v1.multimodal.m1_render import render_document

    FX.mkdir(parents=True, exist_ok=True)
    attr = ["# Fixture attribution\n", "| file | source | license |", "|---|---|---|"]
    for f, (iid, url, flickr, split) in IMAGES.items():
        if not (FX / f).exists():
            req = urllib.request.Request(url, headers={"User-Agent": "krun-fixtures/1.0"})
            (FX / f).write_bytes(urllib.request.urlopen(req, timeout=60).read())
        attr.append(f"| {f} | COCO val2017 image {iid} ({url}); Flickr original {flickr}; M1 split {split} | CC BY 2.0 (http://creativecommons.org/licenses/by/2.0/) |")
    im, text, truth = render_document("invoice", "fatura", "pt", 20260929)
    im.save(FX / "doc_invoice_scan.jpg", quality=90)
    im.save(FX / "doc_invoice_scan_imageonly.pdf", "PDF", resolution=100.0)
    (FX / "doc_invoice_textlayer.pdf").write_bytes(pdf_with_text_layer(text.replace("R$", "BRL")))
    (FX / "doc_invoice_truth.json").write_text(json.dumps({"text": text, "truth": truth}, ensure_ascii=False, indent=1))
    import docx

    d = docx.Document()
    d.add_heading("Pedido de cancelamento", level=1)
    for para in ["Prezados,", "Venho por meio desta solicitar o cancelamento imediato da minha assinatura do plano Premium, "
                 "contrato número 48213. Fui cobrado duas vezes no mês de setembro e peço também o reembolso da cobrança "
                 "duplicada no valor de R$ 89,90.", "Atenciosamente,", "Maria Souza"]:
        d.add_paragraph(para)
    d.save(FX / "doc_cancel_letter.docx")
    for f in ("doc_invoice_scan.jpg", "doc_invoice_scan_imageonly.pdf", "doc_invoice_textlayer.pdf", "doc_cancel_letter.docx"):
        attr.append(f"| {f} | generated by scripts/v1/make_multimodal_fixtures.py (Faker + DejaVu / Noto fonts) | own work; fonts: Bitstream Vera / OFL-1.1 |")
    jobs = [(str(FX / f), lc, v, t, fmt) for f, lc, v, t, fmt in AUDIO]
    print(subprocess.run([tts_python, "-c", TTS, json.dumps(jobs)], check=True, capture_output=True, text=True).stdout)
    for f, lc, v, t, fmt in AUDIO:
        attr.append(f"| {f} | Kokoro-82M TTS, voice {v}, text written for the fixture | Apache-2.0 (model + voices); text: own |")
    (FX / "ATTRIBUTION.md").write_text("\n".join(attr) + "\n")


def cases() -> list[dict]:
    fields = [{"id": x} for x in ("signature", "due_date", "tax_id", "bank_details", "item_table", "barcode", "logo", "paid_stamp")]
    return [
        {"name": "text_intent_pt", "context": [{"type": "text", "text": "quero cancelar minha assinatura, fui cobrado duas vezes"}],
         "questions": {"intent": {"type": "choice", "task_type": "intent", "options": INTENTS},
                       "refund": {"type": "noul", "instructions": "The customer is asking for a refund."}}},
        {"name": "image_elephant", "context": [{"type": "image", "file": "image_elephant.jpg", "mime_type": "image/jpeg"}],
         "questions": {"main": {"type": "choice", "instructions": "Which object is the main subject of the image?",
                                "options": [{"id": x} for x in ("elephant", "bus", "pizza", "train", "laptop", "couch")]},
                       "has_elephant": {"type": "noul", "instructions": "The image contains this object: elephant."}}},
        {"name": "image_elephant_hard", "known_hard": "same-supercategory distractors (animals): hard visual negatives, M1.5c ~0.49",
         "context": [{"type": "image", "file": "image_elephant.jpg", "mime_type": "image/jpeg"}],
         "questions": {"main": {"type": "choice", "instructions": "Which object is the main subject of the image?",
                                "options": [{"id": x} for x in ("elephant", "giraffe", "dog", "bus", "pizza", "train")]}}},
        {"name": "image_giraffe_multi", "context": [{"type": "image", "file": "image_giraffe.jpg", "mime_type": "image/jpeg"}],
         "questions": {"objects": {"type": "multi", "instructions": "Select every object that appears in the image.",
                                   "options": [{"id": x} for x in ("giraffe", "elephant", "pizza", "laptop", "bus", "toilet")]}}},
        {"name": "doc_text_layer_pdf", "context": [{"type": "document", "file": "doc_invoice_textlayer.pdf", "mime_type": "application/pdf"}],
         "questions": {"doctype": {"type": "choice", "instructions": "What type of document is this?",
                                   "options": [{"id": x} for x in ("invoice", "receipt", "payslip", "boarding_pass", "rental_agreement")]}}},
        {"name": "doc_docx_letter", "context": [{"type": "document", "file": "doc_cancel_letter.docx",
                                                  "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}],
         "questions": {"intent": {"type": "choice", "task_type": "intent", "options": INTENTS},
                       "refund": {"type": "noul", "instructions": "The customer is asking for a refund."}}},
        {"name": "doc_scan_jpg", "context": [{"type": "document", "file": "doc_invoice_scan.jpg", "mime_type": "image/jpeg"}],
         "questions": {"doctype": {"type": "choice", "instructions": "What type of document is this?",
                                   "options": [{"id": x} for x in ("invoice", "receipt", "payslip", "boarding_pass", "rental_agreement")]},
                       "fields": {"type": "multi", "instructions": "Select every element present in the document.", "options": fields}}},
        {"name": "doc_scan_pdf", "context": [{"type": "document", "file": "doc_invoice_scan_imageonly.pdf", "mime_type": "application/pdf"}],
         "questions": {"doctype": {"type": "choice", "instructions": "What type of document is this?",
                                   "options": [{"id": x} for x in ("invoice", "receipt", "payslip", "boarding_pass", "rental_agreement")]}}},
    ] + [
        {"name": f"audio_{f.split('_')[1].split('.')[0]}", "context": [{"type": "audio", "file": f, "mime_type": m}],
         "questions": {"intent": {"type": "choice", "task_type": "intent", "options": INTENTS},
                       "urgency": {"type": "score", "instructions": "How urgent is the request?", "levels": ["low", "medium", "high"]}}}
        for f, m in (("audio_en.wav", "audio/wav"), ("audio_pt.mp3", "audio/mpeg"), ("audio_es.flac", "audio/flac"), ("audio_fr.ogg", "audio/ogg"))
    ]


def resolve(case: dict):
    from krun_one.v1.multimodal.runtime import MediaPart, ResolvedMultimodalContext, TextPart

    ctx = ResolvedMultimodalContext()
    for i, p in enumerate(case["context"]):
        if p["type"] == "text":
            ctx.text_parts.append(TextPart(p["text"], id=f"p{i}"))
        else:
            mp = MediaPart(mime_type=p["mime_type"], path=str(FX / p["file"]), id=f"p{i}")
            {"image": ctx.image_inputs, "document": ctx.document_inputs, "audio": ctx.audio_inputs}[p["type"]].append(mp)
    return ctx


def summarize(ans: dict) -> dict:
    keep = ("type", "decision", "top_option", "abstain", "abstention_status", "noul", "score", "selected", "probabilities", "scores")
    return {k: v for k, v in ans.items() if k in keep}


def reference(device: str, out: str | None) -> None:
    import torch

    from krun_one.v1.multimodal.runtime import KrunOneMultimodalRuntime

    if device == "cpu":
        torch.set_num_threads(11)
    rt = KrunOneMultimodalRuntime.load(ART, device=device, dtype="fp32" if device == "cpu" else "bf16")
    res = {"device": device, "torch": torch.__version__, "cases": {}}
    for c in cases():
        r = rt.decide(resolve(c), c["questions"])
        m = r["metadata"]
        res["cases"][c["name"]] = {"known_hard": c.get("known_hard"), "answers": {k: summarize(v) for k, v in r["answers"].items()},
                                   "metadata": {k: m[k] for k in ("modalities_used", "path", "ocr_used", "asr_used", "vision_used",
                                                                  "document_pages", "document_text_layer", "document_visual",
                                                                  "audio_duration_ms", "asr_truncation_risk")},
                                   "usage": r["usage"]}
        print(c["name"], {k: (v.get("decision"), v.get("noul"), v.get("score"), v.get("selected")) for k, v in r["answers"].items()}, m["path"], flush=True)
    p = Path(out or FX / f"expected_{device}.json")
    p.write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["media", "reference"])
    ap.add_argument("--tts-python")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out")
    a = ap.parse_args()
    media(a.tts_python) if a.cmd == "media" else reference(a.device, a.out)
