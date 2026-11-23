# Krun One v1 multimodal — packaging report (2026-09-29)

**Stop condition reached:**

| condition | status |
|---|---|
| MODEL / RUNTIME | READY |
| CONTRACT | READY |
| ARTIFACT | READY |
| TESTS | READY |
| WORKER | NOT TOUCHED |

**No training ran:**
* no M2 / M3;
* the text core and towers are unchanged;
* no worker, API, asset service, storage, deploy or canary;
* final_protected not opened;
* video unsupported.

## 1. What is ready

| deliverable | where |
|---|---|
| runtime `KrunOneMultimodalRuntime` (transport-agnostic, lazy, typed errors, content-free metadata) | `src/krun_one/v1/multimodal/runtime.py` |
| internal contract `ResolvedMultimodalContext` (`TextPart`, `MediaPart` = path or bytes + MIME) | same file |
| candidate artifact | `artifacts/krun-one-v1-multimodal-candidate/` (114 MB; 10 files in SHA256SUMS) |
| worker integration contract | `docs/KRUN_ONE_V1_WORKER_INTEGRATION_CONTRACT.md` |
| OpenAPI draft (valid OpenAPI 3.1; production spec untouched) | `docs/openapi-multimodal-draft.yaml` (generator: `scripts/v1/make_openapi_multimodal_draft.py`) |
| SDK examples (draft, unpublished) | contract §9 |
| reference fixtures (license-safe, 892 KB) | `tests/fixtures/multimodal/` (+ `ATTRIBUTION.md`, `expected_cpu.json`, `expected_cuda.json`) |
| golden / reload / lazy-loading / privacy / error tests | `tests/v1/test_multimodal_runtime.py` |
| backward-compatibility / contract tests | `tests/v1/test_multimodal_contract.py` |
| memory, parameter and latency reports | this file + `results/v1/multimodal_packaging/*.json` |
| licenses / notices / limitations | artifact `NOTICE.md`, `KNOWN_LIMITATIONS.md` |

**Architecture (final v1):**
* Inputs are text, image, document and audio. Outputs are always the structured decision: `choice` / `noul` / `score`
  as today, plus the `multi` extension.
* No generation of any kind: no image / audio / document output, no TTS, no generative decoder.
* Routing is decided inside the runtime:

| input | path |
|---|---|
| text | frozen text path |
| document with a text layer (PDF / DOCX / text) | extraction → text path (no OCR, no tower) |
| scan / photo / screenshot | RapidOCR text ‖ SigLIP2 tokens → fusion |
| image | SigLIP2 → fusion |
| audio | Whisper encoder (one pass) → ASR transcript (text) + acoustic tokens → fusion |

## 2. What is frozen

| component | reference | sha / revision |
|---|---|---|
| text core (`TEXT_CORE_FROZEN = true`) | `artifacts/krun-one-v1-text-final` (weights, tokenizer, calibration, thresholds, decision heads, public abstention) | `781e5c7b…` |
| fusion (M1.5c, gate-less) | `fusion/m15c_step_001169.safetensors` | `8fb9ac49…` |
| SigLIP2 base NaFlex | `google/siglip2-base-patch16-naflex` | `b53b807d…` + file sha256 |
| Whisper small | `openai/whisper-small` | `973afd24…` + file sha256 |
| RapidOCR PP-OCRv5 det (mobile) + latin rec (mobile) | copied, pinned | `4d97c44a…` / `b20bd37c…` |
| OCR execution | rec batch 32, cuDNN HEURISTIC, OCR ‖ SigLIP2 (M1.5b / c) | — |

All of it is verified at load: SHA256SUMS, the tower file sha256 values, and the text-core sha against the manifest.

## 3. Exact worker boundary

**Worker (future):**
* auth, org / project;
* `asset_id` resolution, ownership, TTL, download / storage;
* transport limits, MIME validation;
* public error mapping (HTTP);
* billing;
* logging of the allowed fields.

**Runtime (ready):**
* decode;
* page count / duration checks;
* text-layer vs. visual routing, OCR, ASR, towers, fusion;
* decision and postprocessing;
* content-free metadata.

**What crosses the boundary.** `ResolvedMultimodalContext` in; `{model, answers (v0.3 answer schema + multi), usage,
metadata}` out.

The runtime never sees `asset_id`, URLs, HTTP, D1 / KV or RunPod. A legacy `context: string` maps to
`ResolvedMultimodalContext.from_text()`, which gives bitwise the frozen text path.

**Remaining for worker integration** (all out of scope here):
1. asset upload / storage / TTL (e.g. `POST /v1/assets`);
2. `ContentPart` parsing, and asset → `MediaPart` resolution with MIME sniffing;
3. mapping runtime errors to public `ErrorCode` / HTTP (contract §6);
4. answer mapping for `multi` (`selected` → `values`, `scores` → `probabilities`);
5. baking the candidate artifact + tower snapshots + OCR models into the serving image;
6. deps: pypdfium2, python-docx, soundfile, onnxruntime-gpu 1.22 (CUDA 12);
7. a preload vs. lazy-load policy;
8. observability fields;
9. OpenAPI promotion (the source `krun-api-app/openapi.json` also needs the missing `/health` / `/ready` response
   descriptions; see the draft's `x-krun-draft`);
10. SDK release.

## 4. Artifact and parameter report

| item | size on disk | params |
|---|---:|---:|
| candidate dir (fusion + OCR + configs + docs) | **114 MB** | fusion 26,681,856 · OCR 3,121,400 |
| text artifact (referenced, not duplicated) | 3.5 GB (fp32) | 879,356,931 |
| SigLIP2 checkpoint (HF; vision tower only loaded) | 1.50 GB file | 92,930,304 used |
| Whisper small (HF) | 0.97 GB | 241,734,912 (encoder 88,154,112) |
| **package total** | ≈ 6.1 GB of files | **1,243,825,403** |

Active parameters per request:

| request | active params |
|---|---:|
| text / text-layer document | 879.4M |
| image | 999.0M |
| visual document | 1,002.1M |
| audio | 1,147.8M |

**Trainable in v1 multimodal:** 26.68M (fusion only; nothing trainable remains in serving).

## 5. Memory and load times (RTX 4090, bf16 autocast, one fresh process per configuration)

| configuration | loaded components | torch allocated / nvidia-smi used | process RSS | lazy load of the component |
|---|---|---|---:|---|
| **text only** (+ text-layer PDF) | none multimodal | **3,419 / 3,938 MiB** | 1.7 GB | — |
| text + vision (image, scanned page) | SigLIP2, OCR, fusion | 3,718 / 4,848 MiB | 2.7 GB | vision 1.8 s, OCR 1.7 s, fusion 0.2 s |
| text + audio | Whisper, fusion | 4,011 / 4,540 MiB | 2.4 GB | ASR 1.6 s |
| **full** | all | **4,198 / 5,360 MiB** | 3.4 GB | — |

**Artifact load.** The artifact load measured ~93 s, dominated by sha256 verification of the 3.5 GB text weights,
which was **hashed twice**.
* **Fixed after the run:** the runtime now reuses the verified SHA256SUMS entry instead of re-hashing. It was
  re-tested locally; it was **not re-timed on the GPU**. The expected saving is one ~3.5 GB hash.
* `verify=False` skips hashing entirely, for trusted images.

**First request per modality.** It pays that component's lazy load (image ~2.1 s, scanned page ~2.4–2.6 s, audio
~1.7–2.3 s). Every later request is warm.

## 6. Latency (warm, batch 1, p50; runtime stage timings)

**Host caveat.** This pod (US, AMD EPYC 7K62, **load average 45**: a shared, busy host) is CPU-contended. The
CPU-bound OCR and the token-by-token Whisper decoding are slow and variable there. The clean-host reference is the
M1.5c run (EU host, same code path): scanned page 84 ms, audio 5 s 77 ms, image 13 ms, text typical 7 ms. The
text-core numbers below match the M1.5 US-host numbers (text typical 17–18 ms).

| path | wall p50 / p95 ms | stages (p50 ms) |
|---|---:|---|
| text typical | 17.4 / 18.7 | decision 17.4 |
| text 128 tools | 47.8 / 47.9 | decision 47.7 |
| image | 32.5 / 34.0 | decode 3.4 · vision 10.6 · fusion 0.9 · decision 17.1 |
| screenshot (PNG) | 79.4 / 86.3 | decode 4.6 · vision 21.0 ‖ OCR 46.7 · fusion 1.0 · decision 25.7 |
| PDF with a text layer (fast path) | 19.1 / 19.5 | extract 2.0 · decision 17.0 |
| scanned page | 172.0 / 184.6 | decode 7.4 · vision 39.6 ‖ OCR 129.8 · fusion 1.5 · decision 28.0 |
| audio 5 s | 83.3 / 87.4 | decode 0.5 · ASR 64.1 · fusion 1.0 · decision 17.0 |
| audio 15 s (4-language mix) | 787 / 805 | ASR 766 (mixed-language speech drives the decoder to the 96-token cap) |
| audio 30 s (4-language mix) | 239 / 243 | ASR 218 (`asr_truncation_risk = true`) |

**Same-host A/B** (`latency_ab_same_host.json`; scanned page p50):

| config | p50 |
|---|---:|
| M1.5 OCR config (EXHAUSTIVE, rec batch 6), sequential | 203.8 ms |
| optimized OCR (HEURISTIC, rec batch 32), sequential | 166.5 ms |
| **optimized OCR in parallel with SigLIP2 (runtime config)** | **146.1 ms** (−28%) |

OCR alone: 149.6 ms (default) → 113.4 ms (optimized).

**English-only speech on the same host:** 5 s 188 ms, 15 s 155 ms, 30 s 841 ms.
* **Decode time follows the number of generated tokens.** 30 s of speech hits the 96-token cap and is truncated
  (known limitation).
* **Short-clip variance** (5 s: 83 vs 188 ms across runs) is CPU contention on this host.

**Implication for serving.** The scanned-page and audio SLOs depend on host CPU quality:
* on a clean host: page ≈ 84 ms, 5 s audio ≈ 77 ms;
* on this host: page 146–172 ms, 5 s audio 83–188 ms;
* the text path is unaffected.

Do not co-locate the multimodal worker on CPU-contended hosts. GPU-side OCR preprocessing would be the next lever. It is
not done here, because the OCR stays frozen in this round.

## 7. Tests

| suite | where it ran | result |
|---|---|---|
| `test_multimodal_runtime.py`: text-only **bitwise** vs a fresh frozen text artifact (16 API cases), lazy loading, golden (13 cases), known-correct decisions, privacy, 7 typed errors | laptop CPU fp32 **and** pod GPU bf16 (fresh process = reload test) | 7 / 7 on both |
| `test_multimodal_contract.py`: legacy `context` string valid in the current spec and the draft; existing question / answer schemas byte-identical; ContentPart valid / invalid (video, `use_ocr`, URL, empty); multi; error codes; OpenAPI validity | both | 17 / 17 on both |
| existing M1 / M1.5 / scaffold tests | laptop | 24 / 24 |
| **total, laptop** | | **48 / 48** |
| **CPU fp32 vs GPU bf16 golden agreement** | 20 answers | top option, decision, abstention and multi set **100% identical**; max probability Δ 0.059 |

**Golden definition:**
* **Text:** bitwise.
* **Media:** identical decisions / abstentions / multi sets, and probabilities within tolerance (CPU 2e-3; GPU
  compared to its own reference 5e-2). Towers are not bit-reproducible across devices.

**Fixture outcomes.** The runtime reproduces the evaluation harness exactly on the same inputs. For example, the
val_eval elephant with the eval's distractors gives elephant 0.95.

| fixture | outcome |
|---|---|
| text, DOCX letter, EN / PT / FR audio | decided correctly |
| ES audio, invoice (text-layer / scan / image-only PDF), elephant | correct top option but **abstain**: the media margin is advisory under the strict 0.99 text threshold |
| `image_elephant_hard` (animal distractors) | answers "dog" |
| giraffe multi | includes a false "elephant" |

The last two are documented limitations (hard negatives), not runtime defects.

## 8. Known limitations (artifact `KNOWN_LIMITATIONS.md`)

* **Audio length.** Audio longer than ~20 s risks transcript truncation: the ASR is capped at 96 tokens, the hard
  limit is 30 s, and the runtime flags `asr_truncation_risk`.
* **pt-PT phone speech.** ASR WER is ~30%.
* **Emotion.** Emotion is ~0.35–0.40: a complementary signal, not an emotion model.
* **Numeric comparison.** Numeric comparison is at chance.
* **Layout classes.** Footnote / Caption layout is weak.
* **Hard negatives.** Same-supercategory visual negatives are hard.
* **Media abstention.** Media abstention is advisory (text thresholds not refit).
* **Multi-page documents.** For a visual multi-page document, OCR reads ≤ 4 pages and vision tokens come from page 1.
* **Several media items.** Multi-media and text + media inputs are functional but not quality-validated (training
  used one media item).
* **Unsupported:** streaming and video.
* **Latency.** It is host-CPU sensitive (§6).

## 9. Cost and pod

| pod | GPU | use | cost | status |
|---|---|---|---:|---|
| `vh8ftnaj5j4713` | RTX 4090 Secure (US; Community had no stock) | health (all components via the GPU fixture reference), fresh-process reload + golden + contract tests, CPU vs GPU, memory ×4, latency, same-host A/B | **≈ $0.45** (up 20:33 → ~21:05 UTC) | **TERMINATED**; `list-pods` empty |

No training and no new dataset.

## 10. Worker untouched

This round only **read** `krun-api-app/openapi.json`, which is unchanged since 2026-09-24. It wrote nothing in
krun-api-app, the Cloudflare worker, `services/krun-worker`, the SDKs or the Docker images.

**Observed changes from someone else.** While this round ran, files there were modified by the parallel refactor, not
by this round:
* `krun-api-app/crates/krun-api-cloudflare/src/{lib,native}.rs` at 17:56 BRT;
* `services/krun-worker/{README.md, observability.env.example, tests/test_telemetry.py}` at 17:51 BRT.

No conflict: the model side lives entirely under `experiments/krun_one/`.
