# Krun One v1 — M1.5b (close the two failed M1.5 gates), 2026-09-29

**Verdict: `M1_5B_PASS = false`.** The protocol (`configs/v1/m15b.yaml`) was registered before the data build, the OCR
optimization and training.

**Both targets were reached:**
* DocLayNet layout: **0.270 → 0.394** (M1 0.366; ideal ≥ 0.35).
* Scanned page: **−28% latency**, same host; 72.8 ms p50.

**What failed.** The layout-heavy replay shifted other document cells slightly. The registered rule fails on three
small margins, and no checkpoint was selection-eligible:
* **G1:** document visual-only mean **0.675 vs 0.679**.
* **G2:** document text-dependent mean **−1.1 pp vs −1.0**.
* **G4b:** synthetic document type **0.952 vs 0.953**.

**What passes:**
* G3 real speech;
* G4 image;
* G5 text-only (bitwise identical);
* G6 latency;
* G7 leakage.

**`REMOVE_GATE = true`:**
* the gate was useless (100% identical decisions);
* 1,586,178 params removed;
* one head pass removed.

**Scope held:**
* text core, SigLIP2, Whisper and RapidOCR are frozen;
* no API or output change (modalities are input-only);
* M2 not started;
* final_protected not opened.

## 1. Gate ablation (before training; no GPU)

The M1.5 selected checkpoint, gated (C) vs the fused path without the gate (F), on val_fit + val_eval (29,061
decisions):

| measure | value |
|---|---|
| α | min 0.996, mean 1.000 |
| max \|Δlogit\| | 0.008 (p99 0.0) |
| argmax agreement | 100% |
| task cells | 30 / 30 identical |

**Decision:** `REMOVE_GATE = true`. M1.5b is the fused path.

**Measured effect of the removal:**

| effect | value |
|---|---|
| removed parameters | 1,586,178 (gates) plus the z_text head pass |
| quality delta | 0 (identical decisions) |
| latency delta, same host | image fusion + decision 7.6 → 6.9 ms (−0.7 ms); scanned page 8.6 → 8.0 ms |

Trainable parameters: 28.27M → **26.68M**.

## 2. What ran

| item | value |
|---|---|
| start | M1.5 selected checkpoint `runs/v1/m15/krun-one-v1-m15/step_006160` (fusion weights; gate dropped) |
| data (`data/m15b`) | **56,008** train decisions (sha `3113db8d…`); val_fit / val_eval = the M1.5 files **verbatim** (same sha) |
| preservation replay | ~30k decisions sampled from M1.5 train, per-source caps (COCO 6k, synthetic docs 6k, screenshots 5k, DocLayNet category 2k, CORD 1.5k, FLEURS 3k, CREMA 2k, MInDS 2k, TTS 2.5k) |
| layout replay | all 4,593 M1 DocLayNet layout decisions, ×2 (M1.5 had sampled only 2,917: **the likely cause of the regression**) |
| new layout decisions | from TRAIN-split DocLayNet pages: full 8-class multi, small-region multi (< 1.5% of the page), confusable-pair noul (Caption / Footnote, Title / Section-header, Picture / Table, List-item / Formula) |
| layout total | 20,119 multi + 5,889 noul |
| loss | the M1.5 objective; **layout decisions weighted 2.0** (26,008 decisions); others 1.0 |
| optimizer | lr 5e-5 adapters, 3e-5 LoRA; 1 epoch, **1,980 steps**, 18.5 min on an RTX 4090 |
| leakage | KrunBench v1 `--check` on 105,361 train text views (ground truth + OCR / ASR): **LEAKAGE = 0** |
| hygiene | group-disjoint; 0 shared media; 3 train decisions dropped |
| core | digest identical before / after training |

**Checkpoint selection (registered, val_fit).** The rule requires every preservation group within 1 pp of step 0.
**No checkpoint was eligible**: document text_dependent fell 1.4–2.2 pp at every milestone. The rule then reports FAIL
with the least-regressing checkpoint, which is `step_001980`:

| group | step 0 → final (val_fit) |
|---|---|
| audio text_dependent | 0.817 → 0.815 |
| document text_dependent | 0.804 → 0.790 |
| document hybrid | 0.781 → 0.774 |
| image | 0.643 → 0.650 |
| **layout** | **0.299 → 0.437** |

## 3. Gates (val_eval; C = M1.5b, P = M1.5 on the same host / harness)

| gate | required | measured | pass |
|---|---|---|---|
| G1 document visual-only | mean ≥ 0.679 (M1) AND DocLayNet layout ≥ P + 5 pp | mean **0.675** (P 0.623); layout **0.394** (P 0.270, +12.5 pp; M1 0.366) | ❌ (mean −0.4 pp) |
| G2 screenshot text | screenshot choice ≥ 0.979 AND document text_dependent mean ≥ P − 1 pp | 0.990 (P 0.989); mean 0.660 vs P 0.671 (**−1.1 pp**) | ❌ |
| G3 real speech | ≥ 0.841 AND MInDS intent ≥ 0.890 | 0.849 (P 0.851); MInDS 0.897 (P 0.900) | ✅ |
| G4 image | ≥ 0.633 | 0.640 (P 0.643) | ✅ |
| G4b document hybrid | synthetic type ≥ 0.953, DocLayNet category ≥ 0.935, CORD ≥ 0.95 | **0.952** (P 0.963), 0.946, 1.000 | ❌ (−0.1 pp) |
| G5 text-only | core digest + full V bitwise with M1.5b loaded | identical (4 files, ~53k decisions) | ✅ |
| G6 scanned page | p50 ≤ 150 ms (target ≤ 145) | **72.8 ms** (M1.5 same host 100.7) | ✅ |
| G7 leakage | 0 | 0 | ✅ |

## 4. DocLayNet breakdown (val_eval, layout multi-label, n = 538 pages)

| class | M1.5b recall / precision | M1.5 (P) | M1 (A) |
|---|---|---|---|
| Caption | **0.26 / 0.32** | 0.09 / 0.27 | 0.17 / 0.67 |
| Formula | **0.79 / 0.71** | 0.74 / 0.43 | 0.74 / 0.58 |
| List-item | **0.45 / 0.30** | 0.27 / 0.23 | 0.42 / 0.33 |
| Picture | **0.43 / 0.51** | 0.19 / 0.32 | 0.30 / 0.52 |
| Section-header | **0.93 / 0.80** | 0.82 / 0.80 | 0.86 / 0.78 |
| Table | **0.65 / 0.78** | 0.59 / 0.74 | 0.57 / 0.62 |
| Title | 0.18 / 0.29 | 0.27 / 0.05 | 0.00 / 0.00 |
| Footnote | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| **micro-F1** | **0.670** | 0.558 | 0.625 |
| **exact-set accuracy** | **0.394** | 0.270 | 0.366 |

**Layout is recovered beyond M1** on every metric except Caption precision. Footnote is never recognized by any
system.

## 5. Where the preservation margins were lost (val_eval, C vs P)

| cell | n | M1.5b | M1.5 | Δ |
|---|---:|---:|---:|---:|
| signed (visual-only noul) | 368 | 0.957 | 0.976 | −1.9 |
| synthetic doc type | 870 | 0.952 | 0.963 | −1.1 |
| synthetic amount bucket (score) | 461 | 0.341 | 0.373 | −3.2 |
| screenshot multi | 339 | 0.829 | 0.853 | −2.4 |
| CORD item listed | 72 | 0.722 | 0.764 | −4.2 (3 items) |
| synthetic fields (multi) | 866 | 0.438 | 0.454 | −1.6 |
| screenshot choice / noul | 1,531 / 479 | 0.990 / 0.906 | 0.989 / 0.906 | ≈ |
| DocLayNet category / CORD type | 741 / 72 | 0.946 / 1.000 | 0.945 / 1.000 | ≈ |

**Why.** The layout replay was 46% of the M1.5b train mix. It pulled the shared image adapter and LoRA toward
DocLayNet. Each drop is small, but they all point the same way.

**Candidate next step (owner; nothing started).** An M1.5c with a smaller layout share (~20–25%), or per-task multi
thresholds, would likely keep the layout gain (0.39) without the drift. It would need a new registration; nothing can
be tuned post hoc here.

## 6. Audio and image regression (val_eval)

| slice | M1.5b | M1.5 |
|---|---:|---:|
| MInDS-14 intent / noul (real) | 0.897 / 0.886 | 0.900 / 0.883 |
| FLEURS topic / noul (real) | 0.847 / 0.767 | 0.842 / 0.779 |
| CREMA emotion / intensity (real) | 0.404 / 0.467 | 0.368 / 0.489 |
| real speech mean (gate cells) | 0.849 | 0.851 |
| synthetic speech | 0.948 | 0.949 |
| image: main / noul / people / caption / multi | 0.738 / 0.790 / 0.794 / 0.626 / 0.253 | 0.738 / 0.790 / 0.808 / 0.628 / 0.251 |

## 7. OCR latency profile and optimization (RapidOCR PP-OCRv5 latin kept)

**Stage profile** (80 real val_fit pages; the staged run equals the normal call on 100% of pages). p50 / p95, ms:

| stage | baseline (EXHAUSTIVE, rec batch 6) | **chosen (HEURISTIC, rec batch 32)** |
|---|---|---|
| decode image (JPEG → array) | 9.9 / 95.0 | 6.9 / 55.2 |
| load_img + bounds resize | 0.03 / 0.7 | 0.02 / 0.6 |
| det preprocess (resize + normalize, CPU) | 11.7 / 28.7 | 11.3 / 28.9 |
| det session (incl. input H2D ~1.5 ms of a 4.0 ms run) | 4.7 / 21.4 | 4.6 / 21.4 |
| det postprocess (DB boxes) | 2.5 / 4.0 | 2.5 / 3.6 |
| crop extraction | 0.5 / 0.9 | 0.5 / 0.8 |
| rec preprocess | 1.5 / 2.2 | 1.8 / 3.7 |
| **rec session** | **32.2 / 45.7** | **16.8 / 37.4** |
| rec postprocess | 0.6 / 0.9 | 0.6 / 1.0 |
| final output (map / filter / order) | 0.3 / 0.5 | 0.3 / 0.5 |
| **total** | **72.9 / 172.5** | **56.0 / 130.5** |

**CPU↔GPU transfer.** Measured on the det session only: numpy-input run vs IOBinding with the input already on the
GPU, 11.3 MB input. It is ~1.5 ms.

**Optimization sweep.** cuDNN algo search × rec batch; quality guard on 637 benchmark pages / screenshots:

| config | page p50 ms | downstream screenshot accuracy | CER synthetic / screenshot | guard |
|---|---:|---:|---|---|
| EXHAUSTIVE / 6 (M1.5) | 73.1 | 0.9535 | 0.0241 / 0.2639 | ✅ |
| EXHAUSTIVE / 32 | 48.2 | 0.9535 | 0.0243 / 0.2638 | ✅ |
| **HEURISTIC / 32 (chosen)** | **47.9** | 0.9535 | 0.0243 / 0.2638 | ✅ |
| DEFAULT / 32 | 110.3 | 0.9593 | 0.0242 / 0.2640 | ✅ (slower) |

**Pipeline change.** On scanned pages and screenshots, OCR now runs **concurrently** with SigLIP2 preprocessing and the
tower (a worker thread; ONNX Runtime and torch release the GIL).

**Optimization diff:**
* `m15_ocr.RapidOCRBackend(rec_batch_num=32, cudnn_conv_algo_search="HEURISTIC")`;
* `m15_serving_checks.scanned()` runs OCR in parallel;
* no resolution, model or quality change.

**Checklist items not needed:** OCR caching per request and OCR only when needed. One OCR per request is already
shared by all its questions, and documents with a text layer never reach OCR.

## 8. Full latency (same host, RTX 4090 EU-RO-1, bf16, batch 1, p50 ms)

| path | M1.5 (rerun on this host) | **M1.5b** |
|---|---:|---:|
| text typical (frozen artifact vs stack) | 7.2 vs 7.2 | 7.1 vs 7.1 (answers identical) |
| text 128 tools | 43.1 vs 43.1 | 43.0 vs 43.0 |
| image | 14.0 | **13.3** |
| screenshot | 75.0 | **61.5** |
| PDF with a text layer (fast path, no OCR) | 7.9 | 8.1 |
| **scanned page** | **100.7** (OCR 69.9 sequential) | **72.8** (OCR 55.2 overlapped; waits 41.5 after the tower) |
| audio 5 s | 77.8 | 77.6 |
| audio 15 s | 110.6 | 109.1 |
| audio 30 s | 112.0 | 110.8 |

**Host variance is large.** This host is ~2.4× faster than the M1.5 report's host (text typical 7.2 vs 18.1 ms), so
the absolute numbers are not comparable across reports. The same-host relative gain on the scanned page is **−28%**.
Applied to the M1.5 host's 153.7 ms that would be ≈ 111 ms. This is an estimate, not a measurement on that host.

## 9. RunPod

| pod | GPU | $/h | use | cost | status |
|---|---|---:|---|---:|---|
| `oht7rviecobxh9` | RTX 4090 Secure (EU-RO-1, CUDA 12.8, 32 vCPU / 61 GB) | 0.74 | health ×2, media rebuild (TTS limited to the 16,840 clips used), OCR profile + sweep, M1.5b data, smoke, training, dumps (C / P / A / B / O), text regression, latency M1.5b + M1.5 rerun | **≈ $1.95** ($1.15 billed so far; up 14:46 → ~17:20 UTC) | **TERMINATED**; `list-pods` empty |

**GPU choice.** 4090 Community had no stock, so 4090 Secure was taken. The 5090 was deliberately skipped: the pinned
onnxruntime-gpu 1.22 / CUDA 12 OCR stack is not validated on Blackwell.

## 10. Summary for the owner

**What M1.5b establishes:**
1. **Gate removed.** It had no effect; the design is simpler; parameters go 28.3M → 26.7M.
2. **Layout is fixable.** The M1.5 regression came from under-sampling layout decisions (2,917 vs 4,593 in M1). A
   targeted replay recovers and exceeds M1: layout micro-F1 0.670 vs M1 0.625 and M1.5 0.558.
3. **Scanned-page latency is fixed without quality loss:** rec batch 32 + HEURISTIC + OCR overlapped with SigLIP2
   gives −28% same-host latency. It can be adopted independently of the model weights; it is an OCR execution option.
4. **The model gate failure is a trade-off.** The replay share was too high and drifted other document cells by 1–3 pp.
   A re-registered M1.5c with a smaller layout share is the obvious next step, or you may accept M1.5 weights +
   M1.5b serving optimizations.

M2 is not started.

## Files

| topic | files |
|---|---|
| protocol | `configs/v1/m15b.yaml` |
| data | `data/m15b/{train,val_fit,val_eval}.jsonl` + `manifest.json` (val = M1.5 val, same sha) |
| code | `src/krun_one/v1/multimodal/{m15.py (gate optional), m15_ocr.py (exec options)}`, `src/krun_one/v1/losses.py` (optional per-decision weight; default unchanged), `scripts/v1/{build_m15b_data,m15b_ocr_opt,m15b_evaluate,runpod_m15b,pod_upload_m15b}.*`, `train_m15.py` / `m15_serving_checks.py` (config-driven), `tests/v1/test_m15.py` (4 pass) |
| results | `results/v1/m15b/{m15b_eval.json, ocr_optimization.json, text_regression.json, latency_rtx4090.json, health*.json, krunbench_v1_leakage_check_m15b.json, pod_logs/}`, `results/v1/m15/latency_rtx4090_same_host_as_m15b.json` |
| run | `runs/v1/m15b/krun-one-v1-m15b/` (4 checkpoints, signals, `train_result.json` with selection detail, SHA256SUMS verified) |
