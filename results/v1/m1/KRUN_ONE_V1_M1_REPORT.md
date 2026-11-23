# Krun One v1 — M1 multimodal alignment (2026-09-29)

**Verdict: M1 = FAIL on the registered gate.**

`configs/v1/m1.yaml` defines M1 PASS as G1..G5, all of which must hold. G2, G3, G4 and G5 pass. **G1 fails for
audio.** Image and visual documents pass G1 clearly.

So the M1 fusion is **functional and useful for images and visual documents**. **Speech is not yet aligned**, and
real speech is at chance. **The frozen text path is untouched:** V signals are bitwise identical and the core digest
is unchanged.

Scope held for this round:
* no M2 / M3;
* no text tuning;
* `final_protected` not opened;
* no public API change.

The text base was frozen first: see `results/v1/freeze/FREEZE_REPORT.md`.

## 1. What ran

| item | value |
|---|---|
| base | `artifacts/krun-one-v1-text-final` (sha `781e5c7b…`, verified); **K1-W1536-L22 frozen** |
| towers (frozen) | SigLIP2-base-patch16-naflex vision (92.9M; images 256 patches, documents / screenshots 1024), Whisper-small **encoder only** (88.2M, truncated to the clip length) |
| trainable | image adapter 13.04M (shared by images and visual documents, separate modality embedding) + audio adapter 13.04M (projector + 32-query pooler → 1536) + modality embeddings 7.7k + **decision-head LoRA r = 16, 0.59M** (active only on the multimodal forward) = **26.68M** |
| frozen | text core 879.4M, vision 92.9M, audio 88.2M |
| objective | the text decision loss (`stage2.yaml` loss, unchanged) + 1.0 × consistency |
| consistency | KL(frozen text teacher on the paired text ‖ multimodal student). For multi: BCE to the teacher sigmoid. For captions and documents, applied only where the teacher is right. |
| data | Krun M1 v1: 160,589 train decisions, 2 epochs, 9,370 steps, 43 min on an RTX 4090 (`docs/KRUN_ONE_V1_M1_DATASET.md`) |
| optimizer | AdamW; lr 3e-4 for adapters, 2e-4 for LoRA; warmup 3%, linear decay; bf16 |
| selection | the registered rule: max mean val_fit cell accuracy. It picked the final checkpoint `step_009370` (curve 0.433 / 0.474 / 0.502 / **0.526**) |
| checkpoints | `runs/v1/m1/krun-one-v1-m1/step_*/fusion.safetensors` (SHA256SUMS verified locally) |

## 2. Gates

| gate | result |
|---|---|
| **G1** above trivial | ❌. Per modality, the mean over primitives must be ≥ trivial + 10 pp and ≥ blind + 10 pp. Image **0.640** (trivial 0.442, blind 0.284) ✅. Document **0.539** (0.351 / 0.329) ✅. **Audio 0.391** (0.278 / **0.351**) ❌: only +4 pp over blind |
| **G2** text-only unchanged | ✅. The core digest is identical before and after training (`ee11d492…`). The full V dumps are **bitwise identical** between the frozen artifact alone and the M1 stack loaded after serving image / document / audio requests: val_fit, val_eval, val_ml and val_tools_v2, ~53k decisions (see §5). |
| **G3** projectors learn with frozen towers | ✅. Health check: gradients reach only fusion parameters; none reach the core or the towers. |
| **G4** latency | ✅. See §6. |
| **G5** no leakage | ✅. See `docs/KRUN_ONE_V1_M1_DATASET.md`. Groups are disjoint, 0 shared media, and KrunBench v1 `--check` gives LEAKAGE = 0 after removing 4 records (first run: 4). |

## 3. Results (val_eval; temperatures and multi thresholds fitted on val_fit)

**Columns.**
* **M1** = the multimodal model.
* **blind** = the frozen text model with the media removed.
* **no-LoRA** = the trained adapters with the head LoRA off.
* **teacher** = the frozen text model on the paired text (caption / transcript / document text). This is a reference,
  not a gate.
* **trivial** = the train-split majority.

**Metrics.** Multi uses exact-set accuracy; micro-F1 is given separately.

### Image (COCO, CC BY / no-known-copyright images only)

| task | M1 | blind | trivial | teacher (caption) | other |
|---|---:|---:|---:|---:|---|
| main subject (choice, k = 2–16) | **0.677** | 0.317 | 0.467 | 0.872 | hard negatives 0.450 (blind 0.124); cross-language schema 0.705 |
| contains X (noul) | **0.772** | 0.551 | 0.551 | 0.782 | AUROC **0.865** |
| people count (score, 5 levels) | **0.811** | 0.267 | 0.556 | 0.647 | MAE 0.28, Spearman **0.855** |
| objects present (multi) | 0.300 | 0.001 | 0.195 | 0.387 | micro-F1 **0.580** (blind 0.002) |

Unanswerable main-subject items: margin AUROC 0.62 (n = 64). Image-choice ECE is 0.033.

### Visual documents

| task | M1 | blind | trivial | teacher (text) |
|---|---:|---:|---:|---:|
| synthetic document type (12 types, 4 languages) | **0.915** | 0.154 | 0.187 | 0.640 |
| DocLayNet category (6) | **0.916** | 0.306 | 0.490 | 0.718 |
| CORD receipt (real scans; trivial = always "receipt") | 0.972 | 0.181 | 1.0 | — |
| synthetic doc fields (multi) | 0.316 | 0.055 | 0.035 | 0.136 |
| DocLayNet layout elements (multi) | 0.364 | 0.178 | 0.338 | 0.206 |
| paid / signed / total > X (noul) | **0.899** | 0.729 | 0.729 | 0.782 |
| amount bucket (score) | 0.338 | 0.215 | 0.321 | 0.197 |
| **app screenshots of a user message** (kd1 choice) | 0.517 | 0.370 | 0.195 | **0.997** |

**Layout / type / visual-mark decisions work.** The model beats the text teacher on category and type, because layout
carries information the extracted text does not.

**Reading the content of a screenshot does not work well yet.** Choice is +15 pp over blind but far from the text
(0.997). The document noul AUROC is 0.878.

### Audio

| slice | M1 | blind | teacher (transcript) |
|---|---:|---:|---:|
| synthetic speech (Kokoro, EN/PT/ES/FR), kd1 choice | 0.519 | 0.368 | 0.997 |
| synthetic speech, noul / score / multi | 0.597 / 0.386 / 0.145 | 0.566 / 0.341 / 0.183 | 0.975 / 0.523 / 0.959 |
| **real speech (MInDS-14 EN/FR/ES/PT-PT), intent choice k = 4–14** | **0.153** | 0.107 | 0.904 |
| real speech, noul | 0.527 | 0.527 | 0.883 |
| cross-language schema (audio ≠ schema language) | 0.455 | 0.365 | 0.982 |

Audio choice has an ECE of 0.030 and a margin AUROC of 0.53 on unanswerable items.

**Speech → decision was barely learned:**
* synthetic speech: +15 pp over blind on choice;
* real speech: no transfer.

The information is in the input: the transcript gives 0.90–0.99 through the frozen text model.

### Cross-modal consistency (2,585 kd1 decisions present as text, speech AND screenshot, val_eval)

| pair | argmax agreement |
|---|---:|
| speech ~ screenshot | 0.783 |
| speech ~ text | 0.481 |
| screenshot ~ text | 0.475 |
| all three agree | 0.410 |
| all three correct | 0.367 |

Speech and screenshot agree with each other far more than either agrees with the text. The likeliest reason is that
where neither modality carries enough signal, both fall back to the same question / candidate prior. So this is not
yet semantic consistency.

### The head LoRA is what makes fusion work

With LoRA off, every cell drops to the **blind** level: image 0.284 vs 0.640, document 0.331 vs 0.539. The frozen
2-layer head ignores the projected media tokens on its own.

## 4. Diagnosis: why audio fails and what M1 proves

1. **Where the design fuses.** Media tokens are appended **after** the frozen text encoder, so only the 2 decision-head
   layers (+ LoRA) mix them with the question and candidates.
   * That is enough for perception-level decisions: object, count, document type, layout, stamps.
   * It is not enough for **linguistic content** carried by pixels or waveforms: a spoken intent, or the text inside a
     screenshot. These need the ~22-layer language model to read them.
2. **Speech encoder features are acoustic.** The Whisper encoder has no ASR decoder on top, and 30k synthetic clips
   with 17 voices are too few to learn an intent mapping in a 13M adapter. Real speech has other speakers, 8 kHz
   phone audio and noise, and it does not transfer.
3. **The teacher numbers show the content is recoverable.** The transcript → frozen text model path gives
   0.90 (real) to 0.99 (synthetic).

Candidate M1b options (owner decision; nothing started, M2 not started). The first two follow the scaffold's
TranscriptProvider / OCR route:
* **(a) Transcript route for speech.** Whisper-small **ASR** (decoder) → transcript → the Krun text path.
  * This is the scaffold's `TranscriptProvider` hook.
  * It reuses the frozen text model's measured 0.90–0.99.
  * It costs a decoder pass in latency (~50–150 ms for 5 s).
* **(b) OCR / text-extraction route for screenshots,** mirroring the document text fast path.
* **(c) Deeper fusion.** Feed the media tokens into the text encoder input as prefix tokens, with the encoder still
  frozen, so all 22 layers read them.
  * This is a fusion-design change, so it needs a new registration.

Keep M1 as it is for images and visual documents.

## 5. Text-only regression (G2): one incident, fixed and re-verified

**The incident.** The first textreg run found the M1 stack's text signals **not** bitwise identical:
* the weights and the core digest were identical;
* 82–89% of rows were identical, with max |Δlogit| 0.0625 (bf16 resolution);
* argmax agreement was 99.8–99.99%.

**The controls, on the same pod** (`scripts/v1/m1_textreg_control.py`):

| run | result |
|---|---|
| two fresh artifact loads, 2 repetitions each | bitwise identical: the text path is deterministic |
| artifact + both towers loaded | identical |
| artifact + `M1Model(...)` built, no multimodal request | **differs** |
| artifact + only `requires_grad_(False)` on the core parameters | **reproduces the difference exactly** |

**The cause.** Freezing the core's parameters for training changes PyTorch's numeric path under bf16 autocast. The
exact op was not isolated.

**The fix:** `M1Model(freeze_core=False)` for serving. Training keeps `freeze_core=True`.

**Re-verified** (`text_regression.json`):
* the full V dumps are **bitwise identical**: 4 files, ~53k decisions;
* the core digests are equal;
* the multimodal eval was re-dumped with the serving stack, and every M1 number above is unchanged.

The pre-fix artifacts are kept as `text_regression_before_fix.json` and `latency_rtx4090_before_fix.json`.

**Text V metrics measured on this pod** are identical to the frozen artifact record:

| metric | value |
|---|---|
| intent accuracy | 97.5 |
| tool accuracy | 98.1 |
| cross-language accuracy | 98.2 |
| choice accuracy by language | EN 97.0, PT-BR 97.9, PT-PT 99.3, ES 98.1, FR 97.9 |
| ECE | 0.015 |
| OOD AUROC | 0.971 |
| AURC | 0.029 |
| permutation consistency | 99.3 |
| score MAE / Spearman | 0.600 / 0.836 |
| noul AUROC | 0.973 |
| multi micro-F1 | 0.917 |

## 6. Latency (RTX 4090, bf16, batch 1; `latency_rtx4090.json`)

**Text only: frozen artifact vs M1 loaded.** Answers are identical in every workload.

| workload | artifact p50 | with M1 p50 | Δ |
|---|---:|---:|---:|
| 1 typical question | 19.5 ms | 19.6 ms | +0.2% |
| 4 mixed questions | 22.0 ms | 22.1 ms | +0.7% |
| 32 tools | 18.5 ms | 18.5 ms | −0.1% |
| 128 tools | 48.1 ms | 48.1 ms | −0.05% |

Absolute text latency varied between runs on the same pod: 16.9 ms in the freeze check vs 19.5 ms here, measured
interleaved. The pairwise Δ is the regression measure.

**Multimodal SLO** (p50; separate from the text SLO; the M1 gate is ≤ 150 ms):

| input | preprocess (CPU) | tower | projection + fusion | Krun decision | total p50 | total p95 |
|---|---:|---:|---:|---:|---:|---:|
| image (256 patches) | 4.2 | 8.2 | 0.9 | 19.0 | **32.5** | 34.6 |
| document page (1024 patches) | 13.6 | 8.5 | 0.9 | 18.9 | **41.9** | 45.0 |
| audio 5 s | 5.2 | 8.5 | 1.0 | 19.1 | **72.1** | 96.5 |
| audio 15 s | 7.8 | 8.7 | 1.0 | 18.7 | **69.5** | 88.6 |

**The audio totals are unexplained.** They are far above the sum of the component p50s (5 s: 72 ms vs ~34 ms), and the
5 s clip is no faster than the 15 s one. The component spread was not measured. My unconfirmed hypothesis is the CPU
log-mel frontend, which always computes 30 s of padded mel and varies a lot between calls. Computing the mel only on
the real length would change the features the adapter was trained on, so it belongs with a retrain.

## 7. RunPod

| pod | GPU | $/h | use | cost | status |
|---|---|---:|---|---:|---|
| `zrp16wul62rwfv` | RTX 4090 **Secure** (US, driver 570, 96 vCPU, **31 GB container RAM limit**) | 0.74 | freeze GPU checks, M1 health, data build (fetch / render / 38.6k TTS clips), teacher, smoke, training, dumps, text regression, latency, G2 root-cause controls | **≈ $2.2** ($1.44 billed through 04:00 UTC; up 01:48 → ~04:45 UTC) | **TERMINATED**; `list-pods` empty |

**GPU choice.** Community 4090 / 5090 / 3090 had no stock, so 4090 Secure was taken, as the preference list allows.

**Incidents:**
* **OOM kill.** 94 render workers + 3 TTS processes hit the 31 GB cgroup limit and were killed. Render was capped at
  24 workers and the run resumed; everything is idempotent.
* **Leakage check.** The first run found 4 hits; they were filtered and re-checked to 0 before training.
* **The G2 root cause** in §5.

**Pod-only media.** The M1 media (13.7 GB) lived only on the pod. Every record keeps its media sha256, and the builder
regenerates everything from the fixed seeds and source URLs. Local copies: splits, manifests, the teacher stats and
the fusion checkpoints.

## 8. Files

| topic | files |
|---|---|
| dataset | `docs/KRUN_ONE_V1_M1_DATASET.md`, `data/m1/{train,val_fit,val_eval,protected}.jsonl` + `manifest.json` |
| licenses | `docs/KRUN_ONE_V1_M1_SOURCES_AUDIT.md` |
| protocol | `configs/v1/m1.yaml`, `configs/v1/m1_labels.yaml` |
| code | `src/krun_one/v1/multimodal/{m1,m1_towers,m1_render}.py`, `scripts/v1/{build_m1_data,m1_tts,m1_leakage_filter,train_m1,m1_evaluate,m1_serving_checks,m1_textreg_control,runpod_m1,pod_upload_m1}.*`, `tests/v1/test_m1.py` (3 tests; 20 pass with `test_multimodal.py`) |
| results | `results/v1/m1/{m1_eval.json, health.json, teacher_stats.json, text_regression.json, latency_rtx4090.json, krunbench_v1_leakage_check_m1.json, pod_logs/}` |
| run | `runs/v1/m1/krun-one-v1-m1/` (fusion checkpoints, signals, `train_result.json`, SHA256SUMS) |
