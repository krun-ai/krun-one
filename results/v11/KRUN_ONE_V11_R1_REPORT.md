# Krun One v1.1 — R1 report (preservation baseline: encoder frozen)

Date: 2026-10-01 · Protocol: `configs/v11/r1_r2_protocol.yaml` (registered before any training step) · Gates:
`configs/v11/promotion_gates.yaml` · Frozen v1 (`artifacts/krun-one-v1-text-final`, sha 781e5c7b…) was only read.
Companion documents: `KRUN_ONE_V11_R2_REPORT.md`, `KRUN_ONE_V11_R1_R2_COMPARISON.md`.

## 1. Verdict

R1 **preserves v1's text behaviour** (19 / 19 text preservation gates on both seeds, latency and VRAM unchanged) and
**improves technical decisions substantially** (KrunTechBench v1 raw macro top-1 30.9 → 53.0, +22.1 pp), but:

* it is **clearly behind R2** (63.1);
* it **does not pass two of the registered technical gates** (unseen combinations, "stuck on the base answer");
* it **breaks the multimodal path**: with the M1.5c fusion weights frozen, all five registered multimodal gates fail on
  the R1 core (section 6), even though the encoder was not touched.

R1 is not a promotion candidate; it is the reference that shows how much of the gain needs the encoder (question 1 of
the comparison report).

## 2. What was trained

| | R1 |
|---|---|
| initial weights | frozen v1 (function-preserving: new modules zero-initialised, output bitwise identical at step 0) |
| trainable | type embedding, 2-layer decision head, scorer, answer head (44,883,459) + technical adapter dim 256 and training-only Q / V heads (805,129) |
| frozen | the entire encoder (0 trainable encoder parameters) |
| parameters served | 880,162,060 with the adapter (v1: 879,356,931); the Q / V heads are dropped at export |
| objective | O2 = v1's own loss on every item + 0.25 · value loss (Q(state, action), V(state)) — chosen by the registered pilot |
| data | 295,805 examples, one pass: 45.0% replay · 26.3% technical · 19.0% hard negatives · 9.8% OOD |
| optimiser | AdamW, lr head 5e-5 / new modules 2e-4, wd 0.01, warm-up 3%, clip 1.0, 16,384 tokens per step |
| steps / time | 6,845 and 6,842 steps; 28.7 min per seed on one RTX 4090; peak 7.3 GB VRAM |
| seeds | 20261001 (`r1_s1`, sha 94cad1ed3087…), 20261002 (`r1_s2`, sha 9db3a9b2cddb…) |

## 3. Technical results

### 3.1 KrunTechVal (selection set) — macro raw top-1

v1 22.8 → R1 **37.7 ± 0.95** (38.7 / 36.8). With the explicit consequence in the candidate text ("feat"): 38.0 → 74.7.
KrunBench v1 `val_fit` choice accuracy drop: 0.41 pp (limit 1.0).

### 3.2 KrunTechBench v1 (protected, 10,664 examples / 6,232 problems; raw rendering unless stated)

| metric | v1 | R1 seed 1 | R1 seed 2 | R1 mean ± std |
|---|---:|---:|---:|---:|
| macro top-1 over 10 tracks | 30.9 | 52.3 | 53.7 | 53.0 ± 0.7 |
| paired top-1 difference vs v1 (95% CI) | — | +17.6 [+16.2, +19.0] | +18.6 [+17.3, +20.1] | +18.1 |
| macro top-1, consequence rendering ("feat") | 45.3 | 77.9 | 76.4 | 77.2 ± 0.8 |
| gold > hardest negative (pairwise) | 66.9 | 75.2 | 76.3 | 75.8 ± 0.5 |
| hard-negative margin (logit, macro) | +1.12 | +2.70 | +2.80 | +2.75 |
| hard-negative slice top-1 | 39.2 | 57.3 | 56.7 | 57.0 |
| constraint flips: both members correct | 2.7 | 25.7 | 27.0 | 26.4 |
| constraint flips: stuck on the base answer | 12.5 | 21.3 | 23.3 | 22.3 |
| seen requirement combinations | 39.5 | 97.6 | 98.2 | 97.9 |
| **unseen requirement combinations** | 52.1 | 51.9 | 54.8 | 53.4 |
| unseen technologies | 39.0 | 63.0 | 68.5 | 65.8 |
| large candidate sets | 24.2 | 39.2 | 39.0 | 39.1 |
| unseen simulator families (eviction, placement) | 12.7 | 11.9 | 13.7 | 12.8 |
| OOD false acceptance | 0.4% | 0.4% | 0.0% | 0.2% |
| OOD AUROC (technical none-of-the-above) | 0.507 | 0.644 | 0.655 | 0.650 |

By track (raw top-1, v1 → R1 s1 / s2): api 26 → 41 / 43 · backend 26 → 59 / 63 · cloud 25 → 32 / 36 · database 28 → 56 / 57 ·
devops 4 → 63 / 66 · distributed 41 → 78 / 75 · mlinfra 41 → 50 / 48 · security 48 → 56 / 61 · software repair 60 → 74 / 74 ·
state / action 9 → 16 / 15. No track loses.

Full tables: `results/v11/techbench/protected.md`.

### 3.3 Registered technical gates (`results/v11/techbench/gates.md`)

| gate | required | R1 seed 1 | R1 seed 2 | holds |
|---|---|---:|---:|---|
| macro top-1 gain, CI > 0 | ≥ +10 pp | +21.4 | +22.8 | ✅ |
| pairwise gold vs hardest | ≥ +8 pp | +8.3 | +9.4 | ✅ (barely) |
| hard-negative margin (and median > 0) | ≥ +1.0 logit | +1.58 (median +2.16) | +1.68 (median +1.93) | ✅ |
| constraint flips, both correct | ≥ +10 pp | +23.0 | +24.3 | ✅ |
| stuck on the base answer does not rise | ≤ 0 pp | +8.8 | +10.8 | ❌ |
| unseen combinations | ≥ +10 pp | −0.2 | +2.7 | ❌ |
| worst track | ≥ −2 pp | +6.7 | +6.1 | ✅ |
| OOD false acceptance | ≤ 8% | 0.4% | 0.0% | ✅ |
| **all** | | | | **❌** |

Reading of the two failures:

* **Unseen combinations.** With the encoder frozen the head learns the requirement sets it has seen (97.9%) and gains
  nothing on requirement sets it has not seen (53.4 vs 52.1 for v1). This is the memorisation the brief asked to remove;
  independent variation of requirements and candidate subsets was not enough for R1.
* **Stuck on the base answer.** The registered metric is the share of flip pairs where the base member is right and the
  flipped member still gets the base answer. v1 is rarely right on the base member (14.9%), so it is rarely "stuck"
  (12.5%); R1 is right on 47–52% of the base members and stuck on 21–23%. Conditional on a correct base member the rate
  falls from 84% (v1) to 45–46% (R1), but the gate was registered on the unconditional number and R1 fails it.

## 4. v1 regression (same host, same harness; `results/v11/regression/regression.md`)

All **19 / 19 preservation gates pass for both seeds**. Largest movements (candidate − v1): permutation −0.98 pp (s1),
multi micro-F1 −0.39 pp (s1), intent −0.32 pp (s1), KrunBench v1 choice accuracy −0.24 / −0.04 pp; improvements:
ToolACE holdout +0.72 pp (s1), k = 128 +0.27 / +0.54 pp, selective coverage +0.5 / +0.8 pp, OOD false acceptance at 80%
coverage −0.7 / −0.9 pp. Multilingual (−0.07 pp), PT-BR (−0.26 / −0.17 pp), cross-language (−0.27 / −0.11 pp), Score
(MAE 0.600 → 0.599 / 0.602), Noul (AUROC 0.9728 → 0.9723 / 0.9721), Multi (0.917 → 0.913 / 0.916), ECE 0.0147 → 0.0150 /
0.0168: all inside their limits. `final_protected` was not opened.

Unseen tools: KrunBench v1 V has no answerable unseen-tool item; as registered before the results, the gate is applied
to `val_tools_v2` tool accuracy (90.68 → 90.62 / 90.68) and to the ToolACE v0 holdout (90.14 → 90.87 / 89.90).

## 5. Latency and memory (RTX 4090, same host as v1)

| workload (p50 / p95 ms) | v1 | R1 seed 1 | R1 seed 2 |
|---|---|---|---|
| typical, 1 question, 8 tools | 16.4 / 16.5 | 16.5 / 16.6 | 16.3 / 16.5 |
| 4 questions | 21.3 / 21.3 | 21.3 / 21.4 | 21.4 / 21.4 |
| k = 8 | 16.4 / 16.6 | 16.6 / 16.7 | 16.4 / 17.6 |
| k = 32 | 18.6 / 22.3 | 18.5 / 18.6 | 18.4 / 20.7 |
| k = 128 | 54.9 / 55.1 | 54.8 / 55.9 | 55.0 / 55.3 |
| k = 128, bf16 weights | 44.1 / 44.2 | 44.1 / 44.2 | 44.2 / 44.6 |
| peak VRAM (autocast / bf16) | 3,523 / 1,989 MiB | 3,526 / 1,990 MiB | 3,526 / 1,990 MiB |

The adapter costs +3 MiB and no measurable latency. Limits for a core that stays at 879M (typical ≤ 20 ms, k = 128
≤ 50 ms in bf16, VRAM ≤ 8 GB): met, identical to v1.

## 6. Multimodal regression (nothing multimodal retrained)

The registered M1.5c fusion weights (SigLIP2 / Whisper / OCR towers, fusion adapters, head LoRA — all frozen) were run on top
of the R1 text core, on the same host and harness as the v1 core (`results/v11/multimodal/multimodal_regression.md`).

| text core under the frozen M1.5c fusion | doc visual-only (G1) | layout (G1) | screenshot (G2) | text-dep. doc (G2) | real speech (G3) | MINDS intent (G3) | image (G4) | synthdoc (G4b) | DocLayNet (G4b) | CORD (G4b) | gates passed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| *registered threshold* | ≥ 67.9 | ≥ P + 5 | ≥ 97.9 | ≥ P − 1 | ≥ 84.1 | ≥ 89.0 | ≥ 63.3 | ≥ 95.3 | ≥ 93.5 | ≥ 95.0 | |
| M1.5c as registered (production host) | 68.0 | 38.7 | 99.2 | 67.2 | 85.1 | 89.7 | 64.8 | 96.1 | 94.5 | 100.0 | 5 / 5 |
| v1 core, this host | 68.7 | 40.1 | 99.2 | 67.1 | 85.1 | 89.7 | 64.8 | 96.2 | 94.5 | 100.0 | 5 / 5 |
| control: v1 weights in the v1.1 model class (untrained adapter) | 68.7 | 40.1 | 99.2 | 67.1 | 85.1 | 89.7 | 64.8 | 96.2 | 94.5 | 100.0 | 5 / 5 |
| R1 (seed 1) | 50.8 | 6.5 | 96.1 | 66.7 | 81.0 | 87.2 | 56.3 | 80.6 | 81.5 | 81.9 | 0 / 5 |

**All five registered multimodal gates fail with the R1 core**, although R1 never touched the encoder: document visual-only
−17.9 pp, layout −33.6 pp, image −8.6 pp, hybrid documents −13 to −18 pp, real speech −4.1 pp; only the text-dependent
document cell holds (−0.4 pp). A control run — the frozen v1 weights loaded through exactly the same v1.1 code path as the candidates (v1.1 model class, adapter wrapper, untrained zero-initialised adapter) — reproduces the v1 core to the last digit on all ten numbers, and the v1 core on this host reproduces the registered M1.5c result (differences ≤ 1.4 pp, all five gates pass). The loss is a property of the candidates, not of the harness.
The fusion adapters and the head LoRA were fitted to v1's decision head; R1 retrains that head (44.9M parameters) on text
only, and the head no longer reads the media tokens the way the fusion expects. R1 is therefore **not** a safe
"preservation baseline" for the multimodal product: it preserves text and breaks media.

## 7. Probes and external checks

* **Tetris views** (held-out states, top-1 at k = 8 / k = 16; chance 12.5 / 6.25): raw board A 18.8 / 12.0 and 18.0 / 11.5
  (v1 14.8 / 8.5); structured state B 17.5 / 11.5 and 19.5 / 10.5; explicit consequence C 82.2 / 94.0 and 84.0 / 96.0
  (v1 21.0 / 9.5). With the encoder frozen the head already uses stated consequences well, and cannot derive them.
* **JevBench public items (231, evaluation only):** v1 53.2 → R1 55.0 / 55.0 (hard items 30.6 → 37.8 / 36.0). Jev 1.13.0
  scores 86.6 and Laya 58.4 on the same items. 40% of the hard items do not fit in 1,024 tokens.

## 8. Limits and deviations that affect R1

* The 6,000 `swe_involved` (noul) training examples had no label (degenerate `[1, 1]` target) and the corresponding 120
  benchmark items are not scored — `data/v11/TECHNICAL_CORPUS_PLAN.md` § 8.1. v1's noul gate did not move.
* KrunTechBench v1 is English only and its rule tables were reviewed by a model, not by a person.
* Two seeds; standard deviations above are over two runs and only indicate the order of magnitude.

## 9. Cost

Shared with R2 — see the comparison report, section "Cost".
