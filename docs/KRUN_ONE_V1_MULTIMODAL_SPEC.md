# Krun One v1 — multimodal-ready spec (scaffold; no multimodal training yet)

Date: 2026-09-26. The text core is frozen: **K1-W1536-L22** (22 layers, hidden 1536, 24×64 heads, GLU FFN 2304,
mmBERT/ModernBERT family, 879.4M parameters). This round adds interfaces, lazily loaded modality towers, a lightweight
fusion, schema fields, config, tests and a text-path regression benchmark. Nothing multimodal was trained, no tower
was downloaded, and no GPU pod was created for this work.

## 1. Architecture

```
                         KRUN ONE (shared decision space)

Text ──► tokenizer ─► K1 encoder (879M, unchanged) ─────────────────────────┐
                                                                            │  text tokens
Document ─► router ─┬─ extractable text ──► appended to the TEXT context ───┤  (fast path: no vision tower)
                    └─ visual page ───┐                                     │
Image ───────────────────────────────┴► vision tower (lazy) ─► adapter ─────┤  ≤ Q tokens (image / document)
Audio ─► speech encoder (lazy) ─► adapter ──────────────────────────────────┤  ≤ Q tokens (audio)
      └─ optional transcript (TranscriptProvider) ─► appended to TEXT ──────┤
                                                                            ▼
                     type embedding → 2-layer decision head (attends to text + media tokens)
                                            → option markers → Choice / Noul / Score / Multi (+ answerability)
```

* **Fusion point = the existing decision head.**
  * `KrunOneV1Model.forward(..., extra_tokens=None, extra_mask=None)` appends the media tokens after the text
    encoder's output; only the 2 head layers attend to them.
  * With `extra_tokens=None` the forward is the validated text computation, statement for statement.
* **Adapter per modality** (`src/krun_one/v1/multimodal/fusion.py`):
  * `Projector`: LN → Linear(D_src, 1536) → GELU → Linear(1536, 1536);
  * `LearnedQueryPooler`: Q learned queries (default 32) and one cross-attention, turning N tower tokens into Q
    tokens. This bounds the head's extra cost regardless of image resolution or audio length;
  * a modality embedding (image / document / audio / transcript).
  * Visual documents **share the image adapter** (`share_document_adapter: true`) and keep a distinct modality
    embedding: one perception block.
* Deliberately **not** included:
  * a cross-modal transformer stack;
  * MoE;
  * a generative decoder;
  * modality-specific heads.
  * Multi-label uses the existing per-candidate applicability output, so "image/document/audio → several applicable
    labels" needs no new head.

## 2. Code (`src/krun_one/v1/multimodal/`)

| file | content | status |
|---|---|---|
| `config.py` | `MultimodalConfig` (vision / document / audio / fusion); **everything disabled by default** | functional |
| `features.py` | `ModalityFeatures` + `TextFeatures` / `VisionFeatures` / `DocumentFeatures` / `AudioFeatures` ([B, N, D_src] + mask) | functional |
| `towers.py` | `ModalityEncoder` protocol; `LazyTower` (thread-safe, builds on first use); mock towers; generic `hf-vision:<repo>@rev` and `hf-speech:<repo>@rev` adapters (encoder half only) | mock: functional; HF adapters: **scaffold** (not exercised with real weights) |
| `fusion.py` | projector, learned-query pooler, modality embeddings, request→sequence expansion | functional (smoke-tested with gradients) |
| `documents.py` | `DocumentRouter`: txt/md/html/json built in, PDF (`pypdf`) and DOCX (`python-docx`) optional → **text fast path**; scans / receipts / forms / images / little text / `visual_required` → visual path (PDF pages rendered with optional `pypdfium2`) | text routing functional; PDF/DOCX extraction and rendering are **scaffold** (optional dependencies not installed) |
| `audio.py` | `AudioInput`, linear resampling, clipping to `max_seconds`, `TranscriptProvider` interface (no ASR model) | functional (mock); ASR: **interface only** |
| `model.py` | `KrunOneV1Multimodal`: `prepare()` (routes documents/transcripts into text), `encode_media()` (lazy towers → fusion), `forward()` (text-only → `core(...)` unchanged), separate multimodal checkpoint | functional with mocks |

Config: `configs/v1/multimodal.yaml` (all `enabled: false`). Schema: `krun.decision.v1` accepts optional
`context.images / documents / audio` and `modalities`. Existing text records are valid unchanged (tested).

## 3. Text path guarantees (tested / measured)

| guarantee | evidence |
|---|---|
| outputs unchanged | `test_text_only_outputs_bitwise_unchanged`: same weights → pre-scaffold model code == new core == multimodal wrapper serving a text request (**bitwise**) |
| zero extra parameters when disabled | `test_text_only_has_zero_extra_params_and_no_towers`: `fusion is None`, no towers, empty multimodal state dict |
| no tower loaded on text traffic, even with modalities enabled | `test_text_request_with_multimodal_enabled_loads_no_tower` |
| document text fast path never loads vision | `test_document_text_fast_path_does_not_load_vision` |
| text-only artifacts load without multimodal weights | `test_checkpoint_loads_without_multimodal_weights` |
| no latency regression | `scripts/v1/bench_text_regression.py` → `results/v1/multimodal/text_regression_*.json` (§6) |

## 4. Parameter budget

| component | target | scaffold / recommended candidate (measured) |
|---|---|---|
| text core | ~879M | 879,356,931 (unchanged) |
| vision + document tower | ~100–220M | SigLIP2-base NaFlex vision encoder **92.9M** (Florence-2-base DaViT 90.4M; granite-docling vision 86.4M) |
| audio tower | ~100–250M | Whisper-small encoder **88.2M** (xls-r-300m ~315M if prosody matters) |
| fusion / projectors | ~10–40M | **13.0M per adapter** (768-wide tower); image+document (shared) + audio = **26.1M** incl. modality embeddings |
| full model | ~1.2–1.4B | with the recommended towers: 879 + 93 + 88 + 26 ≈ **1.09B** (headroom up to ~300M for larger towers under the 1.5B ceiling) |

Parameters **active** per request type (recommended towers):

| request | active params |
|---|---|
| text-only | 879M |
| text + document (text fast path) | 879M |
| text + image / visual document | 879 + 93 + 13 ≈ **985M** |
| text + audio | 879 + 88 + 13 ≈ **980M** |
| text + image + audio | ≈ **1.09B** |

## 5. Candidate towers (survey 2026-09-26; parameter counts from safetensors headers, latency estimated)

**Vision / document**

| candidate | license | vision encoder | tokens | notes | verdict |
|---|---|---:|---|---|---|
| `google/siglip2-base-patch16-naflex` | Apache-2.0 | 92.9M | native aspect, ~256 (photo) to ~1024 (page) | contrastively aligned to multilingual text; NaFlex targets docs/OCR; ~3–10 ms (estimated) | **1st choice** |
| `ibm-granite/granite-docling-258M` vision tower | Apache-2.0 | 86.4M | 512px, 1024 → 64 | SigLIP2-base-512 tuned end-to-end for document conversion | **A/B init** against SigLIP2 for scans/forms |
| `microsoft/Florence-2-base` DaViT | MIT | 90.4M | 768px, 576 | OCR/region pretraining; `trust_remote_code`, harder to export | runner-up |
| Qwen3-VL-2B visual / InternViT-300M | Apache-2.0 / MIT | 407M / 304M | dynamic | stronger docs, above budget | stretch |
| excluded | — | — | — | MobileCLIP / AIMv2 (Apple research license), jina-clip-v2 (CC-BY-NC), LayoutLMv3 (CC-BY-NC-SA), Qwen2.5-VL-3B (research), PaliGemma / ColPali (Gemma terms) | — |

**Audio (multilingual EN/PT/ES/FR)**

| candidate | license | encoder | frames / 30 s | notes | verdict |
|---|---|---:|---|---|---|
| `openai/whisper-small` encoder | Apache-2.0 (HF) / MIT (GitHub) | 88.2M | 1500 | 99 languages; the recipe used by Qwen2-Audio / Voxtral; the same checkpoint's decoder gives the future ASR path; fixed 30 s input (short clips need padding or slicing) | **1st choice** |
| `facebook/wav2vec2-xls-r-300m` | Apache-2.0 | ~315M | 1499 | self-supervised; probably better prosody/paralinguistics (urgency, emotion) | runner-up (above budget) |
| whisper-medium encoder / w2v-BERT 2.0 | Apache / MIT | 307M / 580M | — | same families, larger | stretch |
| excluded | — | — | — | mms-300m, mHuBERT-147, SeamlessM4T (non-commercial); canary-180m (no PT); SenseVoice (no PT/ES/FR, FunASR license) | — |

**Licenses needing owner review before use:**
* FunASR MODEL_LICENSE (emotion2vec, SenseVoice): no explicit commercial grant.
* CC-BY-4.0 NVIDIA models: attribution required.
* DINOv3 custom license.
* Missing or inconsistent license fields on some HF cards: `openai/clip-vit-base-patch16`, `wavlm-base-plus`, Whisper (HF says Apache vs GitHub MIT).

## 6. Latency

* **Text fast path (SLO ~50 ms):** unchanged. Same K1 weights, interleaved measurement, 30 iterations, text-only
  requests; the wrapper has vision + document + audio **enabled** with mock towers
  (`results/v1/multimodal/text_regression_gtx1650.json`, local GTX 1650, fp16):

  | workload | pre-scaffold p50 | core p50 | multimodal wrapper p50 | Δ | outputs |
  |---|---:|---:|---:|---:|---|
  | 1 question (k=4) | 354.8 ms | 354.8 | 354.8 | −0.01% | bitwise equal |
  | 4 mixed questions | 3464.2 | 3464.1 | 3464.2 | 0.00% | bitwise equal |
  | 32 tools | 2451.2 | 2451.3 | 2451.3 | +0.00% | bitwise equal |
  | 128 tools (3 chunks) | 9244.6 | 9244.5 | 9244.6 | 0.00% | bitwise equal |

  * Towers loaded after the text traffic: none. Peak VRAM: 1988 MiB, the same model memory.
  * The GTX 1650 is slow in absolute terms. The same benchmark on an RTX 4090 (`text_regression_rtx4090.json`) runs
    as the last step of the continued-MLM pod job, to also cover the launch-bound regime. The added code is one
    `if` in `forward` and one wrapper call.
* **Multimodal SLO:** separate, defined later.
  * Expected extra per request: tower forward (vision ~3–10 ms, whisper-small encoder ~8–12 ms on a 4090,
    estimated) plus the head attending to +32/+64 tokens (small, since the head is 2 layers).
  * Towers load lazily, so the first image/audio request pays the tower load (weights ~0.2 GB each in bf16).
  * Text-only workers never load towers: no cold-start regression.

## 7. Serving / worker (not changed this round)

The Cloudflare edge, the public API, and the RunPod queue/LB were not modified. The public contract stays
Choice / Noul / Score.

Integration plan for the worker, when a tower is chosen and trained:
1. vendor `krun_one/v1/multimodal/`;
2. read `multimodal` from the artifact config (default disabled);
3. media arrive as references in an internal payload field (not public);
4. a text-only payload goes straight to `core(...)`;
5. towers live in the same process and load on first use, with an optional warmup flag for multimodal-only
   endpoints.

## 8. What is scaffold only

* Real towers were not loaded or validated. The HF adapters are generic and untested with real checkpoints.
* No real preprocessing: image processor / resize and audio log-mel are left to the tower's processor.
* PDF/DOCX extraction and page rendering depend on optional packages that are not installed.
* No ASR model; `TranscriptProvider` is an interface only.
* No multimodal data, alignment or training. Next steps would be:
  1. choose towers (A/B: SigLIP2 vs granite-docling vision);
  2. adapter alignment on licensed image/document/audio decision data;
  3. a multimodal KrunBench slice with its own SLO.
