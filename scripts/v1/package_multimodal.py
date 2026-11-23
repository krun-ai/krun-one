"""Package the Krun One v1 multimodal CANDIDATE artifact (no training; nothing frozen is modified).

    python scripts/v1/package_multimodal.py build   -> artifacts/krun-one-v1-multimodal-candidate/
    python scripts/v1/package_multimodal.py sums    -> SHA256SUMS (after fixtures / reference outputs are added)

Contents:
* the M1.5c fusion weights (copied),
* the pinned OCR models (copied),
* references with sha256 for the frozen text artifact (relative path) and the HF towers (repo @ revision + file sha256,
  verified at load),
* runtime config, calibration, notices, provenance, limitations, parameter report.
Large checkpoints are never duplicated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "artifacts/krun-one-v1-multimodal-candidate"
TEXT = ROOT / "artifacts/krun-one-v1-text-final"
FUSION = ROOT / "runs/v1/m15c/krun-one-v1-m15c/step_001169/m15.safetensors"
FUSION_SHA = "8fb9ac49"
HF = Path.home() / ".cache/huggingface/hub"
OCR_DIR = ROOT / ".venv/lib/python3.12/site-packages/rapidocr/models"
TOWERS = {
    "vision": {"repo": "google/siglip2-base-patch16-naflex", "revision": "b53b807d3a2d5e2b3911292f2d69e5341cdc064c",
               "files": ["config.json", "model.safetensors", "preprocessor_config.json"], "license": "Apache-2.0"},
    "asr": {"repo": "openai/whisper-small", "revision": "973afd24965f72e36ca33b3055d56a652f456b4d",
            "files": ["config.json", "generation_config.json", "model.safetensors", "preprocessor_config.json", "added_tokens.json",
                      "merges.txt", "normalizer.json", "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json", "vocab.json"],
            "license": "MIT (openai/whisper) / Apache-2.0 (HF card)"},
}
PARAMS = {"text_core": 879_356_931, "siglip2_vision_tower": 92_930_304, "whisper_small_total": 241_734_912,
          "whisper_small_encoder": 88_154_112, "ocr_det_PP-OCRv5_mobile": 1_171_745, "ocr_rec_latin_PP-OCRv5_mobile": 1_949_655,
          "fusion_trainable": 26_681_856}
LIMITS = {"max_images": 4, "max_image_bytes": 10 * 2**20, "max_documents": 2, "max_document_bytes": 25 * 2**20,
          "max_document_pages": 20, "max_visual_pages": 4, "max_audio": 1, "max_audio_bytes": 10 * 2**20, "max_audio_seconds": 30.0}
MIME = {"image": ["image/png", "image/jpeg", "image/webp"],
        "document": ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                     "text/plain", "text/markdown", "text/html", "image/png", "image/jpeg", "image/webp"],
        "audio": ["audio/wav", "audio/x-wav", "audio/flac", "audio/mpeg", "audio/ogg"]}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fusion").mkdir(exist_ok=True)
    (OUT / "ocr").mkdir(exist_ok=True)
    if not sha(FUSION).startswith(FUSION_SHA):
        raise SystemExit("fusion weights do not match the M1.5c selected checkpoint")
    shutil.copy(FUSION, OUT / "fusion/m15c_step_001169.safetensors")
    for f in ("ch_PP-OCRv5_det_mobile.onnx", "latin_PP-OCRv5_rec_mobile.onnx"):
        shutil.copy(OCR_DIR / f, OUT / "ocr" / f)
    towers = {}
    for k, t in TOWERS.items():
        snap = HF / f"models--{t['repo'].replace('/', '--')}/snapshots/{t['revision']}"
        towers[k] = {**t, "sha256": {f: sha(snap / f) for f in t["files"]}}
    text_mcfg = json.loads((TEXT / "decision_config.json").read_text())
    m15c_eval = json.loads((ROOT / "results/v1/m15c/m15c_eval.json").read_text())
    th = m15c_eval["multi_thresholds_fit"]
    cfg = {
        "config_version": "krun_one.v1.multimodal_runtime.1", "model_id": "krun-one-v1-multimodal-candidate",
        "runtime": "krun_one.v1.multimodal.runtime.KrunOneMultimodalRuntime",
        "io_contract": "all modalities input-only; output = structured decision (choice / noul / score + multi); no generation; "
                       "OCR / ASR text is internal (never returned, never logged)",
        "text_artifact": {"path": "../krun-one-v1-text-final", "model_sha256": sha(TEXT / "model.safetensors"),
                          "decision_config": text_mcfg["config_version"], "frozen": True},
        "towers": towers,
        "vision": {"patches": {"image": 256, "document": 1024}},
        "fusion": {"weights": "fusion/m15c_step_001169.safetensors", "gate": False, "vision_dim": 768, "audio_dim": 768, "tokens": 32,
                   "heads": 8, "lora_r": 16, "lora_alpha": 32.0, "source": "runs/v1/m15c/krun-one-v1-m15c/step_001169 (M1_5C_PASS)"},
        "ocr": {"engine": "RapidOCR (ONNX Runtime GPU 1.22)", "variant": "v5-latin", "det_model": "ocr/ch_PP-OCRv5_det_mobile.onnx",
                "rec_model": "ocr/latin_PP-OCRv5_rec_mobile.onnx", "rec_batch_num": 32, "cudnn_conv_algo_search": "HEURISTIC",
                "parallel_with_vision": True, "license": "Apache-2.0 (RapidOCR code, PaddleOCR PP-OCRv5 weights)"},
        "asr": {"model": "openai/whisper-small", "decoding": "greedy, language auto-detected", "max_new_tokens": 96,
                "recommended_max_seconds": 20.0, "shared_encoder_pass": True},
        "documents": {"min_chars_per_page": 200, "text_layer_first": True},
        "limits": LIMITS, "mime_types": MIME,
        "calibration": {"text": "../krun-one-v1-text-final/calibration.json (unchanged, text-only decisions)",
                        "multi_threshold": {"image": th["image/C"], "document": th["document/C"], "audio": th["audio/C"]},
                        "media_choice_abstention_status": "advisory (text thresholds; not refit on media)"},
        "unsupported": ["video", "streaming audio", "any output modality"],
    }
    (OUT / "runtime_config.json").write_text(json.dumps(cfg, indent=1))
    p = PARAMS
    pkg = p["text_core"] + p["siglip2_vision_tower"] + p["whisper_small_total"] + p["ocr_det_PP-OCRv5_mobile"] + p["ocr_rec_latin_PP-OCRv5_mobile"] + p["fusion_trainable"]
    active = {
        "text_only": p["text_core"],
        "image": p["text_core"] + p["siglip2_vision_tower"] + p["fusion_trainable"],
        "document_text_layer": p["text_core"],
        "document_visual": p["text_core"] + p["siglip2_vision_tower"] + p["fusion_trainable"] + p["ocr_det_PP-OCRv5_mobile"] + p["ocr_rec_latin_PP-OCRv5_mobile"],
        "audio": p["text_core"] + p["whisper_small_total"] + p["fusion_trainable"],
    }
    (OUT / "params.json").write_text(json.dumps({"components": p, "package_total": pkg, "active_by_request": active,
                                                 "trainable_in_v1_multimodal": p["fusion_trainable"],
                                                 "notes": "fusion counts both adapters + LoRA + modality embeddings (the per-request active part is smaller: "
                                                          "one adapter); SigLIP2 = vision tower only (the text tower in the checkpoint is not loaded); "
                                                          "Whisper encoder is shared by ASR and the acoustic branch"}, indent=1))
    prov = {"text_core": "artifacts/krun-one-v1-text-final (FREEZE_TEXTUAL_BASE=true; results/v1/freeze/FREEZE_REPORT.md)",
            "chain": ["M1 configs/v1/m1.yaml -> runs/v1/m1/krun-one-v1-m1/step_009370 (fusion sha c744488c)",
                      "M1.5 configs/v1/m15.yaml -> runs/v1/m15/krun-one-v1-m15/step_006160 (gated; FAIL 2 gates)",
                      "M1.5b configs/v1/m15b.yaml (REMOVE_GATE, OCR execution optimization; FAIL 3 margins)",
                      "M1.5c configs/v1/m15c.yaml -> step_001169 (M1_5C_PASS = true)"],
            "configs_sha256": {c: sha(ROOT / c) for c in ("configs/v1/m1.yaml", "configs/v1/m15.yaml", "configs/v1/m15b.yaml", "configs/v1/m15c.yaml")},
            "data_manifests_sha256": {d: sha(ROOT / d) for d in ("data/m1/manifest.json", "data/m15/manifest.json", "data/m15c/manifest.json")},
            "fusion_sha256": sha(FUSION),
            "reports": ["results/v1/m1/KRUN_ONE_V1_M1_REPORT.md", "results/v1/m15/KRUN_ONE_V1_M15_REPORT.md",
                        "results/v1/m15b/KRUN_ONE_V1_M15B_REPORT.md", "results/v1/m15c/KRUN_ONE_V1_M15C_REPORT.md"],
            "leakage": "KrunBench v1 --check LEAKAGE = 0 for every M1 / M1.5 / M1.5b / M1.5c training set; final_protected never opened"}
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=1))
    g = m15c_eval["gates_quality"]
    (OUT / "metrics.json").write_text(json.dumps({"source": "results/v1/m15c/m15c_eval.json (val_eval)", "gates": g,
                                                  "by_modality_task_class": m15c_eval["by_modality_task_class"],
                                                  "by_speech": m15c_eval["by_speech"]}, indent=1, default=float))
    print(json.dumps({"out": str(OUT), "package_params": pkg, "active": active}, indent=1))


def sums() -> None:
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (OUT / "SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(OUT)}\n" for p in files))
    print(len(files), "files")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "sums"])
    {"build": build, "sums": sums}[ap.parse_args().cmd]()
