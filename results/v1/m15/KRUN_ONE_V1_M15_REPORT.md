# Krun One v1 — M1.5 hybrid multimodal (ASR + acoustic; OCR + visual), 2026-09-29

**Verdict: `M1_5_PASS = false`.** The protocol (`configs/v1/m15.yaml`, registered before the build) requires every hard
gate. Two gates miss narrowly:
* **GP document visual-only:** −5.6 pp vs M1, against a 2 pp tolerance. It comes from DocLayNet layout multi-label.
* **GL scanned page:** p50 153.7 ms, against 150 ms.

**The two M1 gaps are closed:**
* Real speech: 0.33 → **0.85**.
* Text inside screenshots: 0.52 → **0.99**.
* Text-dependent documents / screenshots: +23.7 pp.

**What is preserved:**
* **Text-only:** V dumps are bitwise identical, the core digest is unchanged, and the text latency Δ is under 1%.
* **Images:** 0.643 vs 0.647.

**Scope held:**
* the text base is frozen;
* the M1 weights are the starting point and are preserved;
* the towers, ASR and OCR are frozen;
* M2 / M3 were not started;
* final_protected was not opened.

**All modalities are input-only.** The output is always the existing structured decision response. OCR / ASR text is
never an output.

## 1. What ran

| item | value |
|---|---|
| base | `artifacts/krun-one-v1-text-final` (sha `781e5c7b…`), frozen; core digest identical before / after training |
| M1 init | `runs/v1/m1/krun-one-v1-m1/step_009370/fusion.safetensors` (sha `c744488c…`, verified) |
| ASR | Whisper-small as released (241.7M; the encoder pass is shared with the acoustic branch), greedy decoding, language auto-detected |
| OCR | RapidOCR PP-OCRv5 (mobile det + latin rec, ~3.2M params, ONNX Runtime GPU 1.22), chosen by the registered benchmark rule |
| vision | SigLIP2-base NaFlex (92.9M), frozen |
| fusion | late gated residual `z = z_text + α(z_fused − z_text)` (`src/krun_one/v1/multimodal/m15.py`) |
| trainable | **28.27M**: M1 image adapter 13.04M + M1 audio adapter 13.04M + modality embeddings 7.7k + head LoRA 0.59M + gates 1.59M (new) |
| objective | the text decision loss + 1.0 × KL to the frozen text teacher on the **ground-truth** text (unchanged recipe) |
| optimizer | lr 1e-4 adapters, 5e-5 LoRA, 1e-3 gates; 2 epochs, 6,160 steps; 54 min on an RTX 4090 |
| data | Krun M1.5 v1, 105,987 train decisions (`docs/KRUN_ONE_V1_M15_DATASET.md`); real speech 67% of audio |
| selection | registered rule; val_fit mean 0.581 (step 0 = M1 weights) → 0.682 → 0.690 → 0.689 → **0.695** (final) |

## 2. Gates (val_eval; means over task cells)

**Systems compared.**
* **C** = M1.5.
* **B** = the frozen text path on the ASR / OCR text.
* **A** = M1 (media only).
* **O** = oracle: the frozen text path on the ground-truth text.

| gate | required | measured | pass |
|---|---|---|---|
| GA1 real speech vs M1 | C ≥ A + 20 pp | C **0.851** vs A 0.330 | ✅ |
| GA2 real speech vs transcript only | C ≥ B − 1 pp | C 0.851 vs B 0.774 | ✅ |
| GS1 text visual vs vision only | C ≥ A + 20 pp | C **0.671** vs A 0.435 (+23.7) | ✅ |
| GS2 text visual gap | (C − A) ≥ 0.6 (O − A) | O 0.669; gap closed **≥ 100%** | ✅ |
| GP perception preserved | visual_only C ≥ M1 − 2 pp per modality | image 0.643 vs 0.647 ✅; **document 0.623 vs 0.679** ❌ | ❌ |
| GT text-only | core digest + full V bitwise | identical, 4 files / ~53k decisions, after multimodal traffic, `freeze_core=False` | ✅ |
| GL latency | text Δ ≤ 5%; image and scanned page p50 ≤ 150 ms | text −0.7% / +0.2%; image 31.9 ms; **scanned page 153.7 ms** | ❌ |
| G leakage | 0 | 0 after the fix-up (first run 9) | ✅ |

**Soft gates (reported):**

| soft gate | result | pass |
|---|---|---|
| acoustic value on CREMA-D | C 0.428 vs B 0.283 | ✅ |
| hybrid cells | C 0.866 ≥ max(A 0.814, B 0.537) | ✅ |
| audio 5 s | 181 ms ≤ 200 | ✅ |

## 3. Audio ablations (A = acoustic only / M1, B = transcript only, C = transcript + acoustic)

| slice | A | B | **C** | oracle |
|---|---:|---:|---:|---:|
| **MInDS-14** intent, real phone speech, k = 4–14 | 0.149 | 0.900 | **0.900** | 0.904 |
| MInDS-14 "asking about X" (noul) | 0.527 | 0.879 | **0.883** | 0.883 |
| **FLEURS** topic (real, 4 languages) | 0.136 | 0.730 | **0.842** | 0.760 |
| FLEURS topic noul | 0.508 | 0.587 | **0.779** | 0.588 |
| **CREMA-D emotion** (acoustic only; transcript uninformative) | 0.246 | 0.244 | **0.368** | 0.238 |
| CREMA-D intensity (score) | 0.400 | 0.322 | **0.489** | 0.333 |
| synthetic kd1 speech: choice / noul / multi / score | 0.52 / 0.60 / 0.15 / 0.40 | 0.98 / 0.96 / 0.89 / 0.58 | 0.98 / 0.96 / 0.89 / 0.57 | 0.997 / 0.975 / 0.96 / 0.52 |
| **real speech overall** | 0.313 | 0.616 | **0.729** | 0.624 |
| synthetic speech overall | 0.478 | 0.950 | 0.949 | 0.970 |

Audio by language (C / B / A):

| language | C | B | A |
|---|---:|---:|---:|
| en-US | 0.71 | 0.63 | 0.38 |
| en-AU | 0.95 | 0.90 | 0.35 |
| en-GB | 0.89 | 0.89 | 0.31 |
| pt-BR | 0.91 | 0.85 | 0.38 |
| pt-PT | 0.90 | 0.90 | 0.49 |
| es-419 | 0.90 | 0.84 | 0.38 |
| es-ES | 0.96 | 0.95 | 0.58 |
| fr-FR | 0.88 | 0.84 | 0.39 |

en-US includes CREMA.

**Reading the audio results:**
* **The transcript carries the semantics.**
  * On MInDS-14, B = C = oracle (0.90).
  * The M1 acoustic-only path is at chance (0.15), as expected.
* **The acoustic branch has measurable value where words carry no signal.** CREMA emotion goes 0.24 → 0.37, and
  intensity goes 0.32 → 0.49. It is well below a dedicated emotion model; it is complementary only.
* **FLEURS: C beats even the oracle (0.84 vs 0.76).** This is **task adaptation** of the fused head (LoRA) to the SIB
  topic taxonomy, which was trained on FLEURS train sentences. It is not an acoustic gain.
* **Synthetic speech:** C ≈ B. Where the transcript is clean, fusion neither helps nor hurts.

## 4. Screenshot / document ablations (A = vision only / M1, B = OCR only, C = OCR + vision)

| task (class) | A | B | **C** | oracle (GT text) |
|---|---:|---:|---:|---:|
| **kd1 screenshots choice** (text) | 0.517 | 0.958 | **0.989** | 0.997 |
| kd1 screenshots noul / multi / score | 0.58 / 0.16 / 0.35 | 0.85 / 0.81 / 0.42 | **0.91 / 0.85 / 0.44** | 0.98 / 0.96 / 0.52 |
| CORD "item listed" (real receipts, text) | 0.472 | 0.431 | **0.764** | 0.653 |
| CORD / synthetic "total > X" (text) | 0.50 / 0.56 | 0.51 / 0.53 | 0.51 / 0.53 | 0.51 / 0.53 |
| synthetic amount bucket (text, score) | 0.338 | 0.195 | **0.373** | 0.197 |
| synthetic document type (hybrid) | 0.915 | 0.620 | **0.963** | 0.638 |
| DocLayNet category (hybrid) | 0.916 | 0.709 | **0.945** | 0.717 |
| CORD type (hybrid) | 0.972 | 0.264 | **1.000** | — |
| synthetic fields (hybrid, multi) | 0.316 | 0.143 | **0.454** | 0.135 |
| paid (hybrid, noul) | 0.948 | 0.951 | **0.968** | 0.954 |
| signed (visual only) | 0.992 | 0.622 | 0.976 | 0.712 |
| **DocLayNet layout elements** (visual only, multi) | **0.366** | 0.175 | 0.270 | 0.206 |

**Reading the document results:**
* **OCR + vision beats both single branches on every hybrid task.**
* **The screenshot reading gap is closed:** 0.52 → 0.99.
* **Numeric comparisons ("total > X") stay at chance for every system, oracle included.** This is a limitation of the
  frozen text model on arithmetic-like propositions, not of OCR.
* **The GP miss is DocLayNet layout multi-label** (0.366 → 0.270). The fused head now weighs the OCR text heavily, and
  the multi threshold fitted on val_fit (0.30) did not restore the M1 layout recall.

## 5. Images (visual only; no OCR)

| task | M1 (A) | **M1.5 (C)** |
|---|---:|---:|
| main subject | 0.715 | **0.738** |
| contains X | 0.772 | **0.790** |
| people count | 0.811 | 0.808 |
| caption match | 0.637 | 0.628 |
| objects (multi) | 0.300 | 0.251 |
| **mean** | 0.647 | 0.643 |

Images are preserved within tolerance.

## 6. Cross-modal consistency (val_eval)

| measure | A (M1) | B | **C** |
|---|---:|---:|---:|
| kd1 speech ↔ text (same decision, argmax agreement) | 0.48 | 0.91 | 0.87 |
| kd1 screenshot ↔ text | 0.47 | 0.83 | **0.85** |
| kd1 screenshot ↔ speech | 0.78 | 0.82 | **0.87** |
| FLEURS same sentence, all languages correct (n = 79) | 0.03 | 0.43 | **0.51** |

## 7. The gate collapsed to α = 1 (finding)

**What happened.** The learned gates went to α ≈ 1.0 for every modality and task class, by the 25% checkpoint. So
C == F (the fused path) everywhere, and the "fallback to transcript-only" property is not exercised.

**Why it still works.** The fused path **also** receives the transcript / OCR text: the encoder output of the text is
shared. That is why C ≈ B on clean transcripts.

**Owner question:** keep the gate for a later round with a regularizer or a lower gate LR, or remove it. Its cost is
1.59M params and one extra head pass (~1 ms).

## 8. Text-only regression (GT)

The runs, on the same pod:
* **A** = the frozen artifact alone.
* **B** = a fresh load wrapped in the M1.5 serving stack: both towers, the ASR, the OCR and the trained fusion loaded,
  after image / scanned-page / audio requests.

Results:
* The core digests are equal.
* The full V dumps (val_fit, val_eval, val_ml, val_tools_v2) are **bitwise identical**.
* The core's `requires_grad` is left untouched in serving (`freeze_core=False`, the validated fix).
* Answers are identical in the latency harness.

## 9. Latency (RTX 4090, bf16, batch 1, p50; `latency_rtx4090.json`)

| path | stages | total p50 / p95 |
|---|---|---:|
| text typical | frozen artifact 18.1 vs M1.5 stack 18.0 (−0.7%) | 18.0 ms |
| text 128 tools | 46.8 vs 46.9 (+0.2%) | 46.9 ms |
| image | preprocess 4.2 · SigLIP2 7.9 · projector 0.9 · fusion + decision 18.4 | **31.9 / 34.9 ms** |
| PDF / DOCX with a text layer (fast path) | extraction 1.6 · Krun text decision 16.6; route = text, no OCR, no tower | **18.2 / 19.2 ms** |
| scanned page | decode 3.3 · **OCR 106.8** · preprocess 13.4 · SigLIP2 8.3 · projector 0.9 · fusion + decision 19.3 | **153.7 / 161.2 ms** ❌ (150) |
| audio 5 s | decode 1.4 · resample 0.0 · padding 0.2 · **log-mel (GPU) 1.5** · encoder 6.9 · **ASR decoder 151.4** · projector 0.9 · fusion + decision 18.2 | **181.4 / 184.0 ms** ✅ (200) |
| audio 15 s | decoder 228.9 | 261.1 / 265.7 ms |
| audio 30 s | decoder 228.5 (hits `max_new_tokens = 96`: the transcript is **truncated**) | 264.9 / 274.4 ms |

**The M1 audio frontend anomaly is resolved.** Components now sum to the total (5 s: 180.5 vs 181.4 ms). Log-mel on
the GPU costs 1.5 ms, down from the CPU 30 s-padded frontend of M1.

**Where the time goes:**
* audio: the ASR decoder, ~85%;
* scanned pages: OCR, ~70%.

**Levers, not applied (the protocol forbids post-hoc changes):**
* raise / chunk the ASR token budget, since 30 s audio needs it for quality;
* OCR detection at a lower resolution, or docTR on the GPU at batch > 1. The scanned page misses by 3.7 ms.

## 10. Parameters

| component | params | status |
|---|---:|---|
| Krun text core | 879.4M | frozen |
| SigLIP2-base vision | 92.9M | frozen |
| Whisper-small (encoder 88.2M + decoder 153.6M) | 241.7M | frozen |
| OCR RapidOCR PP-OCRv5 (det + latin rec, ONNX) | ~3.2M (from fp32 file sizes) | frozen |
| M1.5 fusion (M1 adapters + LoRA + embeddings + gates) | 28.3M | trainable |
| **total package** | **~1,245.5M** | |

Active parameters by request type:

| request | active params |
|---|---:|
| text | 879.4M |
| image | ~986.7M |
| scanned page | ~989.9M |
| audio | ~1,135.5M |

## 11. RunPod

| pod | GPU | $/h | use | cost | status |
|---|---|---:|---|---:|---|
| `6rduxsct9gsq4h` | RTX 4090 **Secure** (driver 570 / CUDA 12.8, 96 vCPU, 31 GB RAM cap) | 0.74 | health ×2, media rebuild, new sources, OCR benchmark, ASR / OCR passes, teacher, smoke, training, dumps, text regression, latency | **≈ $3.2** ($2.37 billed through 14:00 UTC; up 10:07 → ~14:20 UTC) | **TERMINATED**; `list-pods` empty |

**GPU choice.** Community 4090 / 5090 / 3090 had no stock, so 4090 Secure was taken, which the preference list allows.

**Incidents, all fixed and resumed from where they stopped:**
1. `onnxruntime-gpu` 1.30 needs CUDA 13 and fell back to CPU. Pinned 1.22 with `preload_dlls` from the torch CUDA 12
   libraries.
2. The DocLayNet / CORD / MInDS media extraction lived in the M1 decision stage. Added an `extract_media` stage.
3. The docTR orientation path failed with negative strides. Switched to straight pages.
4. The first decision build dropped CREMA-D train by the equal-text rule. Exempted CREMA-D (speaker-disjoint, fixed
   sentences).
5. Leakage fix-up (§ dataset).
6. The evaluator grouped kd1 items by hash id. Fixed to the task type before reporting.

## 12. Recommendation (owner decision; M2 not started)

1. **Both registered misses are small and specific.**
   * **GP-document:** handle DocLayNet layout recall first, for example with a small visual-only layout replay or a
     per-task multi threshold.
   * **GL:** the scanned page misses by 3.7 ms.

   A registered M1.5b could address both without touching any frozen part.
2. **Decide the gate:** regularize it, or drop it since it collapsed to α = 1.
3. **ASR:**
   * chunked decoding or a larger token budget for clips ≥ 20 s;
   * pt-PT phone WER is 30%;
   * a larger ASR checkpoint is an M1.5b / M2 question.
4. **Before M2 (unfreezing tower blocks):** the semantic gaps that motivated M1.5 are closed. M2 would target the
   remaining perceptual limits: layout multi-label, emotion, and hard visual negatives (0.49).

## Files

| topic | files |
|---|---|
| protocol | `configs/v1/m15.yaml` (io_contract, gates, OCR rule) |
| dataset | `docs/KRUN_ONE_V1_M15_DATASET.md`, `data/m15/{train,val_fit,val_eval,protected}.jsonl` + `manifest.json` + `asr.jsonl` / `ocr.*.jsonl` |
| audit | `docs/KRUN_ONE_V1_M15_SOURCES_AUDIT.md` (OCR / ASR / speech) |
| spec / streaming | `docs/KRUN_ONE_V1_M15_SPEC.md` |
| code | `src/krun_one/v1/multimodal/{m15,m15_asr,m15_ocr}.py`, `scripts/v1/{build_m15_data,train_m15,m15_ocr_bench,m15_evaluate,m15_serving_checks,runpod_m15,pod_upload_m15}.*`, `tests/v1/test_m15.py` (3 tests; 6 pass with `test_m1.py`) |
| results | `results/v1/m15/{m15_eval.json, ocr_benchmark.json, health*.json, teacher_stats.json, text_regression.json, latency_rtx4090.json, krunbench_v1_leakage_check_m15*.{json,log}, pod_logs/}` |
| run | `runs/v1/m15/krun-one-v1-m15/` (4 checkpoints `m15.safetensors` 28.3M, signals, `train_result.json`, SHA256SUMS verified) |
