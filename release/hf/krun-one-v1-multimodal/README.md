---
license: apache-2.0
language: [en, pt, es, fr]
library_name: pytorch
tags: [decision-model, multimodal, document-understanding, speech, image-classification, abstention]
base_model: ferreslabs/krun-one-v1
---

# Krun One v1 — multimodal (M1.5c)

The frozen [Krun One v1](https://huggingface.co/ferreslabs/krun-one-v1) text model plus the **M1.5c fusion**: images,
documents (scans, screenshots, PDFs / DOCX with or without a text layer) and audio become evidence for the same typed
decisions (`choice`, `noul`, `score`, `multi`). Output is always a structured decision — no generation of any kind.

| component | parameters | |
|---|---:|---|
| Krun One v1 text core (frozen) | 879,356,931 | `ferreslabs/krun-one-v1` |
| SigLIP2 base NaFlex vision tower (frozen) | 92,930,304 | `google/siglip2-base-patch16-naflex`, pinned revision |
| Whisper-small (frozen; ASR + acoustic branch) | 241,734,912 | `openai/whisper-small`, pinned revision |
| RapidOCR PP-OCRv5 mobile det + latin rec (ONNX) | 3,121,400 | included in `ocr/` |
| **M1.5c fusion (trained)**: image adapter (shared by documents), audio adapter, modality embeddings, decision-head LoRA r16 | **26,681,856** | `fusion/` |

## Usage

```bash
git clone https://github.com/krun-ai/krun-one && cd krun-one && uv sync --group m1 --group m15
hf download ferreslabs/krun-one-v1 --local-dir models
hf download ferreslabs/krun-one-v1-multimodal --local-dir models
```

The package folder is `models/krun-one-v1-multimodal-candidate/`; its `runtime_config.json` references the text core at
`../krun-one-v1-text-final` and the towers by Hub repo + revision + per-file sha256 (all verified at load). Runtime:
`krun_one.v1.multimodal.runtime`.

## Results (M1.5c, val_eval)

| | |
|---|---|
| visual-only documents (mean) · DocLayNet layout | 68.0% · 38.7% |
| screenshots (text from OCR) | 99.2% |
| real speech (MInDS-14 + FLEURS) · MInDS-14 intent | 85.1% · 89.7% |
| images (visual-only) | 64.8% |
| hybrid documents: synthetic type · DocLayNet category · CORD type | 96.1% · 94.5% · 100% |
| latency, RTX 4090 (p50) | image 13.5 ms · scanned page with OCR ~79 ms · 5 s audio ~77 ms |

Limitations (full list in `KNOWN_LIMITATIONS.md`): speech over ~20 s may be truncated (30 s hard limit); weak layout
classes (footnote, caption); acted-emotion recognition is only a weak signal; media abstention is advisory; one media
page for vision tokens; no video.

## Licenses

Fusion weights: Apache-2.0. Towers keep their licenses (SigLIP2 Apache-2.0; Whisper MIT / Apache-2.0; PaddleOCR /
RapidOCR Apache-2.0). Fusion training data: COCO 2017 (annotations CC-BY-4.0; CC BY 2.0 / no-known-copyright images),
DocLayNet (CDLA-Permissive-1.0), CORD v2 (CC-BY-4.0), MInDS-14 (CC-BY-4.0), FLEURS (CC-BY-4.0), SIB-200 topic labels
(CC-BY-SA-4.0), CREMA-D (ODbL-1.0 + DbCL-1.0), Kokoro-82M synthetic speech (Apache-2.0), Krun Decisions v1 texts (see the
text model card). Details: `NOTICE.md`.
