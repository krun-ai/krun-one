# Krun One v1 — M1.5c (smaller layout share), 2026-09-29

**Verdict: `M1_5C_PASS = true`.** Every hard gate of the registered protocol (`configs/v1/m15c.yaml`, registered before
the data build and training; same gates, references and selection rule as M1.5b) passes. This closes the two gates M1.5
failed:
* **document visual-only / layout:** layout 0.270 → **0.387**, above M1 (0.366);
* **scanned-page latency:** −20% on the same host (84.1 ms).

It does so **without** the preservation drift that failed M1.5b.

Nothing frozen was touched: the text core (V bitwise identical), SigLIP2, Whisper and RapidOCR. The gate stays removed.
Modalities are input-only; the output is the existing decision response. M2 is not started, and final_protected was
not opened.

## 1. What changed vs M1.5b (owner option 1)

| | M1.5b | **M1.5c** |
|---|---|---|
| start | M1.5 selected checkpoint | same (not M1.5b) |
| layout share of the train decisions | 46% (M1 layout ×2 + every new decision) | **22%**: 10,095 = M1 layout ×1 (4,593) + 5,502 new |
| layout loss weight | 2.0 (~63% of the loss mass) | **1.5** (~30%) |
| preservation replay caps | synthetic docs 6k, screenshots 5k, CORD 1.5k | synthetic docs **9k**, screenshots **7k**, CORD **2.3k** (others unchanged) |
| train decisions | 56,008 | **45,895** (sha `d8cb7f66…`) |
| steps | 1,980 | **1,559** (15.9 min, RTX 4090) |
| OCR execution | swept + chosen (HEURISTIC, rec batch 32, overlap) | reused (not re-run) |

**Common to both:**
* val_fit / val_eval are the M1.5 files verbatim (same sha);
* KrunBench v1 `--check` **LEAKAGE = 0** (84,915 train text views, ground truth + OCR / ASR);
* the core digest is unchanged by training.

**Checkpoint selection** (registered, val_fit; eligible = every preservation group within 1 pp of step 0):

| checkpoint | eligible | layout | document visual-only | note |
|---|---|---:|---:|---|
| 25% | ❌ | 0.333 | 0.643 | document hybrid −1.4 pp |
| 50% | ✅ | 0.345 | 0.654 | |
| **75%** | ✅ | **0.356** | 0.672 | **selected** (max layout among eligible) |
| final | ✅ | 0.333 | 0.649 | |

**Trainable parameters:** 26,681,856 (no gate).

## 2. Gates (val_eval; C = M1.5c, P = M1.5 selected, same host / harness)

| gate | required | measured | pass |
|---|---|---|---|
| G1 document visual-only | mean ≥ 0.679 AND layout ≥ P + 5 pp | mean **0.680**; layout **0.387** (P 0.270, +11.7 pp; M1 0.366) | ✅ |
| G2 screenshot text | screenshot choice ≥ 0.979 AND document text_dependent ≥ P − 1 pp | **0.992** (P 0.989); **0.672** vs P 0.671 | ✅ |
| G3 real speech | ≥ 0.841 AND MInDS intent ≥ 0.890 | **0.851** (P 0.851); MInDS **0.897** | ✅ |
| G4 image | ≥ 0.633 | **0.648** (P 0.643) | ✅ |
| G4b document hybrid | synthetic type ≥ 0.953, DocLayNet category ≥ 0.935, CORD ≥ 0.95 | **0.961 / 0.945 / 1.000** | ✅ |
| G5 text-only | core digest + full V bitwise | identical (val_fit, val_eval, val_ml, val_tools_v2, ~53k decisions) with M1.5c loaded | ✅ |
| G6 scanned page | p50 ≤ 150 ms (target ≤ 145) | **84.1 ms** (M1.5 same host 105.1) | ✅ |
| G7 leakage | 0 | 0 | ✅ |

**Honest note on G1.** The visual-only mean passes by **+0.07 pp**, a thin margin. The layout component itself (+11.7 pp
over M1.5, +2.1 pp over M1) is the robust part of the result.

## 3. Cells (val_eval, C / P / A = M1.5c / M1.5 / M1)

| cell | n | M1.5c | M1.5 | M1 |
|---|---:|---:|---:|---:|
| **DocLayNet layout (multi)** | 538 | **0.387** | 0.270 | 0.366 |
| signed (visual-only) | 368 | 0.973 | 0.976 | 0.992 |
| synthetic doc type | 870 | 0.961 | 0.963 | 0.915 |
| DocLayNet category | 741 | 0.945 | 0.945 | 0.916 |
| CORD type | 72 | 1.000 | 1.000 | 0.972 |
| synthetic fields (multi) | 866 | 0.463 | 0.454 | 0.316 |
| paid (noul) | 349 | 0.966 | 0.968 | 0.948 |
| screenshot choice / noul / multi / score | 1,531 / 479 / 339 / 88 | 0.992 / 0.906 / 0.829 / 0.443 | 0.989 / 0.906 / 0.853 / 0.443 | 0.52 / 0.58 / 0.16 / 0.35 |
| CORD item listed / total > X | 72 / 70 | 0.792 / 0.543 | 0.764 / 0.514 | 0.47 / 0.50 |
| synthetic amount / total > X | 461 / 146 | 0.360 / 0.514 | 0.373 / 0.527 | 0.34 / 0.56 |
| image: main / noul / people / caption / multi | | 0.738 / 0.787 / 0.821 / 0.621 / 0.274 | 0.738 / 0.790 / 0.808 / 0.628 / 0.251 | 0.715 / 0.772 / 0.811 / 0.637 / 0.300 |
| MInDS-14 intent / noul (real speech) | 281 / 281 | 0.897 / 0.879 | 0.900 / 0.883 | 0.149 / 0.527 |
| FLEURS topic / noul (real speech) | 653 / 653 | 0.842 / 0.787 | 0.842 / 0.779 | 0.136 / 0.508 |
| CREMA emotion / intensity (acoustic) | 492 / 90 | 0.354 / 0.478 | 0.368 / 0.489 | 0.246 / 0.400 |
| synthetic speech (all) | 2,437 | 0.950 | 0.949 | 0.478 |

**Residual drifts** are within tolerance and mostly in small cells: screenshot multi −2.4 pp, CORD / synthetic
"total > X" ±3 pp (at chance for every system), CREMA emotion −1.4 pp.

**DocLayNet per class (M1.5c, recall / precision):**

| class | recall / precision |
|---|---|
| Section-header | 0.89 / 0.82 |
| Table | 0.60 / 0.80 |
| Formula | 0.65 / 0.59 |
| List-item | 0.33 / 0.31 |
| Picture | 0.30 / 0.54 |
| Title | 0.27 / 0.23 |
| Caption | 0.07 / 0.18 |
| Footnote | 0 / 0 |

Micro-F1 is **0.642** (M1.5 0.558, M1 0.625, M1.5b 0.670). Compared with M1.5b, M1.5c keeps less of the Caption /
Picture recall but no longer costs the rest.

## 4. Latency (same host: RTX 4090 EU-RO-1, bf16, batch 1, p50 / p95 ms)

| path | M1.5 (rerun) | **M1.5c** |
|---|---:|---:|
| text typical (artifact vs stack) | 7.25 vs 7.25 | 7.12 vs 7.12 (answers identical) |
| text 128 tools | 44.7 vs 44.7 | 44.8 vs 44.8 |
| image | 14.4 / 14.6 | **13.4 / 13.8** |
| screenshot | 75.5 / 78.4 | **69.3 / 91.3** |
| PDF with a text layer (no OCR) | 7.9 | 7.8 |
| **scanned page** | **105.1 / 116.3** (OCR 71.3 sequential) | **84.1 / 96.0** (OCR 55.5 overlapped with SigLIP2) |
| audio 5 s / 15 s / 30 s | 79.2 / 112.6 / 114.4 | 76.7 / 109.0 / 110.8 |

The M1.5b run on another host of the same type measured 72.8 ms. Host-to-host variance is ~±15%; only same-host
comparisons are used.

## 5. RunPod

| pod | GPU | $/h | cost | status |
|---|---|---:|---:|---|
| `772qh24rqipsak` | RTX 4090 Secure (EU-RO-1, CUDA 12.8, 32 vCPU / 61 GB); Community had no stock | 0.74 | **≈ $2.0** ($1.17 billed so far; up 17:24 → ~20:00 UTC) | **TERMINATED**; `list-pods` empty |

**Round totals so far:**
* M1.5: ≈ $3.2
* M1.5b: ≈ $1.95
* M1.5c: ≈ $2.0

## 6. Multimodal v1 state after M1.5c

The registered chain M1 → M1.5 → M1.5b → M1.5c is closed. The candidate multimodal package is:
* the frozen text artifact `artifacts/krun-one-v1-text-final`;
* the fusion weights `runs/v1/m15c/krun-one-v1-m15c/step_001169/m15.safetensors` (sha `8fb9ac49…`, 26.7M, gate-less);
* SigLIP2-base NaFlex;
* Whisper-small (ASR + acoustic, one encoder pass);
* RapidOCR PP-OCRv5 latin (rec batch 32, cuDNN HEURISTIC, overlapped with SigLIP2);
* the document text fast path.

Known limits carried forward:
* audio > ~20 s is truncated by the 96-token ASR budget;
* pt-PT phone WER is 30%;
* emotion is weak (0.35);
* numeric "total > X" is at chance;
* Footnote / Caption layout is weak.

**Owner decisions:**
* promote this package as the Krun One v1 multimodal candidate (packaging / serving work), or
* open M2 (tower-block unfreezing) for the remaining perceptual limits.

## Files

| topic | files |
|---|---|
| protocol | `configs/v1/m15c.yaml` |
| data | `data/m15c/{train,val_fit,val_eval}.jsonl` + `manifest.json` |
| results | `results/v1/m15c/{m15c_eval.json, text_regression.json, latency_rtx4090.json, health*.json, krunbench_v1_leakage_check_m15c.json, pod_logs/}`, `results/v1/m15/latency_rtx4090_same_host_as_m15c.json` |
| run | `runs/v1/m15c/krun-one-v1-m15c/` (4 checkpoints, signals, `train_result.json` with selection detail, SHA256SUMS verified) |
| code | the M1.5b code, config-driven (`VARIANT` from the config name; `build_m15b_data.py` layout_share), `scripts/v1/{runpod_m15c,pod_upload_m15c}.sh` |
