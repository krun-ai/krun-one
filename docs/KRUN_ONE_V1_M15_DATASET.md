# Krun M1.5 v1 — hybrid multimodal dataset (2026-09-29)

* **Builder:** `scripts/v1/build_m15_data.py`. It runs on top of the deterministic M1 media builder,
  `build_m1_data.py`.
* **Manifest:** `data/m15/manifest.json`.
* **Licenses:**
  * `docs/KRUN_ONE_V1_M15_SOURCES_AUDIT.md` covers OCR, ASR and the new speech sets;
  * `docs/KRUN_ONE_V1_M1_SOURCES_AUDIT.md` covers the M1 sources.
* **Protocol:** `configs/v1/m15.yaml`, registered before the build.

## Splits

| split | decisions | sha256 |
|---|---:|---|
| train | 105,987 | `ac2d709f…` |
| val_fit | 11,762 | `fa58db66…` |
| val_eval | 17,299 | `9276fd71…` |
| protected | 17,005 | `db66fa66…` |

**Split rule.** It is the M1 split function (group hash), so M1 and M1.5 agree for every shared group. The groups are:
* COCO image;
* document;
* kd1 group;
* DocLayNet PDF;
* CORD image;
* MInDS clip;
* FLoRes sentence (FLEURS: the same sentence in all 4 languages is always in one split);
* CREMA-D actor (speaker-disjoint).

**protected.** It is written at selection time and never processed: no ASR, no OCR, no teacher. It is never read.

## Train mixture (decisions)

| source | modality | train | task class | notes |
|---|---|---:|---|---|
| COCO (CC BY / no-known-copyright) | image | 22,000 | visual_only | replay of M1 (sampled) |
| synthetic documents | document | 22,000 | hybrid / text_dependent / visual_only | OCR context |
| kd1 screenshots | document | 16,000 | text_dependent | OCR context |
| DocLayNet-base | document | 7,000 | hybrid / visual_only | OCR context |
| CORD v2 (real receipts): type + **new** "item listed" / "total > X" | document | 2,327 | hybrid / text_dependent | OCR context |
| **FLEURS × SIB-200 topics** (real speech, EN / PT-BR / ES-419 / FR) | audio | 12,758 | text_dependent | new; ASR context |
| **CREMA-D emotion / intensity** (real acted speech, EN) | audio | 6,582 | acoustic_only | new; ASR context (fixed sentences) |
| MInDS-14 (real phone speech, EN / FR / ES / PT-PT) | audio | 5,320 | text_dependent | M1 splits kept; val_eval / protected never trained |
| kd1 TTS (Kokoro, synthetic) | audio | 12,000 | text_dependent | capped |

**Speech mix.** Real speech is 24,660 decisions; synthetic is 12,000 (**33%**, below the registered cap of 40%).

**Eval speech.** Real 2,450 (MInDS 562, FLEURS 1,306, CREMA 582); synthetic 2,924.

**Task classes (all splits):**

| task class | decisions |
|---|---:|
| image visual_only | 29,577 |
| document hybrid | 24,695 |
| document text_dependent | 27,844 |
| document visual_only | 6,871 |
| audio text_dependent | 37,836 |
| audio acoustic_only | 8,225 |

## Context text (frozen extractors; the same models serve)

**ASR:** Whisper-small as released, greedy decoding, language auto-detected. 34,287 clips.

Approximate WER on val_eval (word-level):

| source | EN | ES | FR | PT |
|---|---:|---:|---:|---:|
| FLEURS | 9.7% | 15.5% | 9.1% | 9.6% |
| MInDS | 12.5% | 16.6% | 14.8% | **30.6% (pt-PT, 8 kHz phone)** |
| TTS | 6–15% | 6–15% | 6–15% | 6–15% |
| CREMA-D | 7.2% | — | — | — |

**OCR:** RapidOCR PP-OCRv5 (mobile det + latin rec, ONNX Runtime GPU). It covers 40,468 pages / screenshots.
* It was chosen by the registered benchmark rule (`results/v1/m15/ocr_benchmark.json`).
* Downstream accuracy on screenshots was 95.3% (ground-truth text 100%, empty context 42.4%).
* p50 was 116 ms / page in the benchmark.

**Ground truth (`gt_text`).** Every record keeps the ground-truth text: rendered text, transcript, caption, page text.
It is the teacher input and the oracle.

## Media rebuild (M1 media were not kept locally)

The deterministic builder regenerated the M1 media on the pod and compared each file with the sha256 stored in the M1
records:

| source | media | result |
|---|---:|---|
| COCO | 20,490 | identical |
| CORD | 998 | identical |
| DocLayNet | 8,054 | identical |
| MInDS-14 | 3,350 | identical |
| screenshots | 38,481 | identical |
| synthetic documents | 14,000 | identical |
| Kokoro TTS | 38,481 | re-synthesized, **different bytes** |

Kokoro on the GPU is not bit-reproducible. The M1.5 records carry the new sha256 of each clip.

## Leakage

* Groups are disjoint across splits (asserted).
* 0 media sha256 values are shared across splits.
* 22 train decisions were dropped because their ground-truth text equaled an eval ground-truth text. CREMA-D is exempt
  from this rule: its 12 fixed sentences are not its label, and its splits are speaker-disjoint.
* 5 ids were dropped: 4 duplicate CORD ids (identical images inside CORD) and 1 KrunBench hit.

**KrunBench v1 `--check`** ran on the laptop, so final_protected never left it. It covered 239,526 text views of
train + val_fit + val_eval, with both the ground-truth text **and** the ASR / OCR text.
* **First run: LEAKAGE = 9.**
  * 8 were duplicate-id text views from 2 identical CORD images.
  * 1 exact hit: a TTS clip's **ASR** transcript equals a SNIPS final_protected utterance. Its ground-truth text did not
    match. The fix was `--drop-ids m1/tts/e4c48d25dfb40153`.
* **Re-check: LEAKAGE = 0** (`results/v1/m15/krunbench_v1_leakage_check_m15.json`).
* Training was gated on a flag holding the train sha256.
