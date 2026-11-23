# Krun One v1 — text base FREEZE (2026-09-28/29)

**`FREEZE_TEXTUAL_BASE = true`**, by owner decision (product/roadmap), with the known gaps.

* **Frozen checkpoint:**
  * source: `runs/v1/ft/krun-one-v1-stage2/milestones/step_003299`, selected by the registered Stage 2 rule;
  * sha256: `781e5c7b7006da630d6b77693de26bd886839b0f8f96673774f9ace136dc18b6`.
* **Artifact:** `artifacts/krun-one-v1-text-final/`. It holds the weights (hard link), spec, tokenizer, calibration,
  decision config, provenance, metrics, README with the limitations, reference cases and SHA256SUMS.
* **Only authorized base for M1.** v0.3, v1-tool, k1_0b, the 1B MLM checkpoint and other milestones are not
  authorized.
* **Not rewritten.** The registered Stage 2 gate outcome (false) stands as the record.
* **Public API contract unchanged.** Production still serves krun-one-v0.3, and this artifact is not deployed.

## Calibration (fit on V fit only, registered protocols)

| family | temperature | margin threshold (serving point) | coverage |
|---|---:|---:|---:|
| intent | 1.161 | 0.99671 | 80% |
| tool | 1.544 | 0.99231 | 80% |
| noul | 1.043 | — | — |
| score | 1.081 | — | — |

The multi threshold is 0.35.

**Where the thresholds come from.** They are the serving-point rule of `r2_protocol.yaml`. They equal the τ of the
Stage 2 coverage curves at 80% (0.9967 / 0.9923).

**Reference only.** The 0.98-target thresholds are kept in `calibration.json` but are not used for serving.

## Reload verification

The artifact was loaded from scratch in a fresh process, with SHA256SUMS verified.

| check | CPU fp32 (laptop) | RTX 4090 bf16 (pod) |
|---|---|---|
| 16 API reference cases (choice / noul / score / multi, EN/PT/ES/FR, k = 2–128, OOD, 4-question mix) vs reference outputs | identical; max \|Δp\| **0.0** | same decisions / abstentions / multi sets; max \|Δp\| 0.006 (bf16 vs the fp32 reference) |
| repeat run identical | yes | yes |
| 384 V-eval records: artifact path vs original training/eval path (same tokens) | token ids identical; max \|Δlogit\| **0.0** | token ids identical; max \|Δlogit\| **0.0** |
| same 384 records vs the Stage 2 signal dumps (another 4090 host, bf16) | — | max \|Δlogit\| 0.125, argmax agreement 99.5% (2 near-ties), abstention agreement **100%** |
| **PASS** | ✅ | ✅ |

The 0.5% argmax flips against the stored dumps are the known bf16 drift between GPU hosts: the laptop fp32 path is
bitwise exact. Nothing was introduced by packaging.

Files:
* `reload_cpu_fp32.json`;
* `reload_cuda_bf16.json`.

## Latency regression (RTX 4090, bf16, `artifact.decide` end to end)

| workload | p50 ms | p95 ms | Stage 2 reference p50 | target |
|---|---:|---:|---:|---|
| 1 typical question (k = 4, ctx 48) | **16.9** | 21.0 | 16.9 | ~17 ms or better ✅ |
| 4 mixed questions (ctx 96) | 20.1 | 22.9 | 18.6 | — |
| 32 tools | 16.7 | 18.8 | 18.8 | — |
| 128 tools | **47.2** | 48.4 | 43.9 | < 50 ms ✅ |

The 128-tool and 4-question p50 run 2–3 ms above the Stage 2 table. The Stage 2 harness (`latency_proxy.py`) casts
the weights to bf16. The runtime keeps the weights in fp32 under bf16 autocast, which is the Stage 2 eval-harness
numerics, and the Stage 2 table was measured on a different 4090 host.

The two harnesses were not run side by side on the same host, so the host and harness contributions are not
separated. Packaging still did not cause this regression: the artifact path is bitwise identical to the original
forward.

**Freeze latency check: PASS.**

File: `latency_rtx4090.json`.

## Known limitations (documented in the artifact README)

* **ToolACE full holdout:** 90.1 vs 91.8 for v0.3.
* **Small-k:** 91.7, against a gate of 92.2. The gap is at k = 4 / 8.
* **Score Spearman:** down 1.9 pp.
* **Legacy / xLAM-negative subset:** still favors v0.3 (86.8 vs 89.7).
* **Tool OOD at 80% coverage:** false acceptance is 9.3%.

## Strengths

* **Multilingual:** choice accuracy EN 97.0, PT-BR 97.9, PT-PT 99.3, ES 98.1, FR 97.9.
* **Cross-language:** 98.2.
* **Large-k:** k = 128 at 90.6 vs 64.9 for v0.3.
* **OOD:** margin AUROC 0.971; tool AURC 0.031 vs 0.156.
* **Calibration:** ECE 0.015.
* **Permutation:** 99.3.
* **Latency:** as above.

These are all deferred to v2, not fixed in v1.
