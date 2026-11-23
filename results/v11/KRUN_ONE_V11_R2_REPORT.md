# Krun One v1.1 — R2 report (main candidate: last encoder blocks unfrozen)

Date: 2026-10-01 · Protocol: `configs/v11/r1_r2_protocol.yaml` (registered before any training step; deviations listed
at its end and in section 9 here) · Gates: `configs/v11/promotion_gates.yaml` · Frozen v1 was only read. Companion
documents: `KRUN_ONE_V11_R1_REPORT.md`, `KRUN_ONE_V11_R1_R2_COMPARISON.md`.

## 1. Verdict

Unfreezing the last encoder blocks is what makes the technical gain: KrunTechBench v1 raw macro top-1 **30.9 → 63.1**
(last 4 blocks + adapter) and **62.7** for the same recipe **without the adapter, i.e. the exact v1 architecture**, with
**19 / 19 text preservation gates on every seed** and **latency / VRAM identical to v1**.

**No R2 variant can be promoted as it stands**, for two independent reasons:

1. **Technical gates:** 6 of 8 pass with wide margins; *unseen combinations* (+6.7 pp and +9.0 pp against the +10 pp
   required) and *"stuck on the base answer" does not rise* (+2.4 / +2.7 pp) fail.
2. **Multimodal:** with the M1.5c fusion weights kept frozen, **every registered M1.5c gate fails** on an R2 text core
   (section 6). The fusion adapters were fitted to v1's head and last blocks; changing them breaks the image / document /
   audio path until the fusion is re-fitted — which this round was told not to do.

## 2. Variants and how the candidate was chosen (KrunTechVal only)

Objective pilot (25% of the corpus, R2 configuration, one seed; v1 = 22.8):

| objective | terms | macro raw | macro feat | `val_fit` drop (pp) | eligible (≤ 1.0) |
|---|---|---:|---:|---:|---|
| O0 | v1 loss only | 31.5 | 73.1 | 1.04 | no |
| O1 | + pairwise + margin | 28.9 | 70.8 | 0.87 | yes |
| **O2** | + value heads Q / V | **31.2** | 72.0 | 0.77 | yes → chosen by the registered rule |
| O3 | O1 + O2 | 30.5 | 71.4 | 0.55 | yes |
| O4 | O3 + self-distillation of v1 on replay | 29.3 | 71.7 | 0.38 | yes |

The spread (28.9–31.5) is of the size of the seed-to-seed spread seen later (~1 pp std, 2 pp between seeds): the pilot
does **not** show that any loss term helps. It shows that the pairwise / margin terms did not help at these weights, and
that distilling v1 on replay preserves v1 best. O2 was used for every run because the rule said so.

Variant selection (full corpus, one pass, macro raw top-1 on KrunTechVal; v1 = 22.8; `val_fit` drop limit 1.0 pp):

| variant | trainable encoder | adapter | seeds | macro raw (mean ± std) | per seed | macro feat | `val_fit` drop |
|---|---|---|---:|---:|---|---:|---:|
| R1 (reference) | none | yes | 2 | 37.7 ± 0.95 | 38.7 / 36.8 | 74.7 | +0.41 |
| R2 | last 2 blocks | yes | 2 | 47.6 ± 0.84 | 48.4 / 46.7 | 82.9 | +0.19 |
| R2 no adapter (v1 architecture) | last 2 | no | 2 | 49.2 ± 0.48 | 48.7 / 49.7 | 82.5 | −0.03 |
| R2 + 2 candidate-interaction layers | last 2 | yes | 1 | 47.3 | 47.3 | 83.3 | −0.22 |
| R2 + hard-negative mini-round | last 2 | yes | 2 | 46.5 ± 0.84 | 47.4 / 45.7 | 82.1 | +0.46 |
| **R2 last 4** | last 4 | yes | 2 | **52.0 ± 0.99** | 53.0 / 51.0 | 83.2 | −0.44 |
| R2 last 4, no adapter (v1 architecture) — *added after registration* | last 4 | no | 2 | 51.2 ± 0.63 | 51.8 / 50.5 | 83.3 | −0.68 |

* The last-4 run was conditional on R2 leaving ≥ 5 pp between the feat and raw renderings; R2 left 35 pp, so it ran.
* **Registered selection: R2 last 4** (2.8 pp above the next registered variant, outside the 1.0 pp tie band).
* *R2 last 4 without adapter* was added after the selection was seen (declared deviation). It is 0.8 pp below R2 last 4,
  inside the tie band, and the registered tie-break prefers the v1 architecture. Both were therefore taken to the final
  evaluation; they are statistically indistinguishable everywhere below.
* **The adapter does not help** (−1.6 pp at 2 blocks, +0.8 pp at 4 blocks: noise). **The candidate-interaction block
  does not help** (47.3 vs 47.6), confirming R0: the head is not the bottleneck.
* **The hard-negative mini-round did not help** (46.5 vs 47.6; software repair fell 66 → 51 in seed 1). 30,000 examples
  mined with the registered rule + 30,000 replay, half learning rate, ~1,450 steps. 4,014 of the mined examples were
  Tetris (deviation). More blocks helped more than more passes over hard examples.

## 3. What was trained (final-evaluation variants)

| | R2 last 4 (+ adapter) | R2 last 4, no adapter |
|---|---|---|
| trainable encoder | layers 18–21 + final norm: 80,229,888 (9.1% of the model) | same |
| trainable head | 44,883,459 | 44,883,459 |
| new modules | adapter 256 + training-only Q / V heads: 805,129 | training-only Q / V heads: 13,833 (dropped at export) |
| parameters served | 880,162,060 | **879,356,931 — the v1 architecture, loadable by the unchanged runtime** (`model_v1.safetensors`) |
| objective / data / optimiser | O2; 295,805 examples, one pass; lr encoder 2e-5, head 5e-5, new 2e-4 | same |
| steps / time / VRAM | 6,845 and 6,842 steps; 37.4 min per seed; peak 14.1 GB | 6,845 and 6,842 steps; 37.2 min per seed; peak 13.8 GB |
| weights (sha256) | `r2l4_s1` 5191e4cf0d84…, `r2l4_s2` 1c69d1feeebc… | `r2l4na_s1` 88e33af24b5a…, `r2l4na_s2` 788f10825452… |

No full fine-tune was run. The tokenizer, the multimodal towers, the public API and the worker were not touched.

## 4. KrunTechBench v1 (protected; raw rendering unless stated)

| metric | v1 | R2 last 4 (s1 / s2) | mean ± std | R2 last 4 no adapter (s1 / s2) | mean ± std |
|---|---:|---|---:|---|---:|
| macro top-1 over 10 tracks | 30.9 | 62.8 / 63.4 | 63.1 ± 0.3 | 61.7 / 63.7 | 62.7 ± 1.0 |
| paired difference vs v1 (95% CI) | — | +27.0 [+25.5, +28.6] / +27.5 [+25.9, +29.0] | | +26.5 [+24.8, +28.1] / +27.8 [+26.2, +29.3] | |
| macro top-1, consequence rendering | 45.3 | 84.5 / 84.0 | 84.3 | 83.9 / 84.1 | 84.0 |
| gold > hardest negative | 66.9 | 79.8 / 80.1 | 80.0 | 79.4 / 79.6 | 79.5 |
| hard-negative margin (logit) | +1.12 | +4.20 / +4.11 | +4.16 | +4.21 / +4.36 | +4.29 |
| hard-negative slice top-1 | 39.2 | 65.5 / 66.3 | 65.9 | 65.7 / 67.0 | 66.4 |
| constraint flips: both correct | 2.7 | 45.9 / 48.0 | 47.0 | 45.6 / 50.3 | 48.0 |
| constraint flips: stuck on base answer | 12.5 | 16.2 / 13.5 | 14.9 | 16.6 / 13.9 | 15.2 |
| seen requirement combinations | 39.5 | 100 / 100 | 100 | 99.8 / 100 | 99.9 |
| **unseen requirement combinations** | 52.1 | 58.6 / 59.1 | 58.9 | 59.9 / 62.4 | 61.2 |
| unseen technologies | 39.0 | 76.5 / 75.5 | 76.0 | 76.0 / 73.0 | 74.5 |
| large candidate sets | 24.2 | 50.8 / 50.7 | 50.8 | 51.1 / 52.3 | 51.7 |
| unseen simulator families | 12.7 | 9.0 / 9.6 | 9.3 | 9.8 / 11.0 | 10.4 |
| OOD false acceptance | 0.4% | 0.6% / 1.5% | 1.1% | 0.2% / 0.0% | 0.1% |
| OOD AUROC (technical) | 0.507 | 0.821 / 0.811 | 0.816 | 0.810 / 0.790 | 0.800 |
| ECE | 0.113 | 0.045 / 0.068 | | 0.047 / 0.058 | |

By track (raw top-1):

| model | api | backend | cloud | database | devops | distributed | ML infra | security | software repair | state / action |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 | 26.3 | 26.4 | 25.2 | 28.3 | 4.3 | 41.3 | 40.7 | 47.9 | 59.6 | 9.0 |
| R2 last 4 s1 | 51.0 | 73.3 | 50.6 | 68.0 | 88.2 | 94.7 | 54.7 | 53.9 | 74.5 | 19.0 |
| R2 last 4 s2 | 56.7 | 63.9 | 45.5 | 67.5 | 90.1 | 94.0 | 65.0 | 58.2 | 75.2 | 18.2 |
| R2 last 4 no adapter s1 | 55.3 | 73.0 | 46.5 | 69.7 | 87.0 | 94.8 | 47.4 | 50.3 | 74.9 | 18.1 |
| R2 last 4 no adapter s2 | 57.3 | 77.4 | 45.1 | 67.7 | 88.2 | 94.7 | 59.0 | 53.9 | 75.8 | 17.9 |

Full tables (feat rendering, slices, calibration): `results/v11/techbench/protected.md`.

What the numbers say:

* **Where the consequence can be computed from the text, R2 learns it:** quorum 4 → 97, rollout 2 → 83–86, retry
  20 → 63, sizing 18 → 63–69 (per-slice numbers in `protected.json`).
* **State / action stays low (18–19%) while the same items reach 65–66% once each candidate carries its consequence.**
  The Tetris diagnosis (section 7) isolates this.
* **Technology selection is still partly memorised.** Requirement sets seen in training: 100%. Requirement sets never
  seen: 59–61% (v1: 52%). In two tables the unseen-combination accuracy is *below* v1 (network component 65 → 15–17,
  security 37 → 15–23): the model learned requirement-set → answer shortcuts that transfer negatively. Independent
  variation of requirements and candidate subsets (as built here) is not sufficient.
* **New simulator families are not learned from the others** (eviction, placement: 9–11%, at or below chance of 11.5%).
* Software repair moves 59.6 → 75 with file / package / test-module / function decisions; multi-file F1 0.64 → 0.68–0.71.

## 5. Registered technical gates (`results/v11/techbench/gates.md`)

| gate | required | R2 last 4 (s1 / s2) | holds | R2 last 4 no adapter (s1 / s2) | holds |
|---|---|---|---|---|---|
| macro top-1 gain, CI > 0 | ≥ +10 pp | +31.9 / +32.5 | ✅ | +30.8 / +32.8 | ✅ |
| pairwise gold vs hardest | ≥ +8 pp | +12.9 / +13.2 | ✅ | +12.5 / +12.7 | ✅ |
| hard-negative margin, median > 0 | ≥ +1.0 logit | +3.08 / +3.00 (median +4.0 / +3.7) | ✅ | +3.09 / +3.24 (median +3.8 / +4.0) | ✅ |
| constraint flips, both correct | ≥ +10 pp | +43.2 / +45.3 | ✅ | +42.9 / +47.6 | ✅ |
| stuck on base answer does not rise | ≤ 0 pp | +3.7 / +1.0 | ❌ | +4.1 / +1.4 | ❌ |
| unseen combinations | ≥ +10 pp | +6.5 / +7.0 | ❌ | +7.8 / +10.3 | ❌ (one seed passes) |
| worst track | ≥ −2 pp | +6.1 / +9.2 | ✅ | +2.4 / +6.1 | ✅ |
| OOD false acceptance | ≤ 8% | 0.6% / 1.5% | ✅ | 0.2% / 0.0% | ✅ |
| **all** | | | **❌** | | **❌** |

On the "stuck" gate: it was registered on the unconditional rate (base member right and flipped member answered with
the base answer). v1 gets the base member right in 14.9% of the pairs, R2 in 63–67%, so R2 has four times more
opportunities to be stuck. Conditional on a correct base member the rate falls from 84% (v1) to 20–26% (R2). The gate
as registered fails and is reported as failed; the conditional number is given so the owner can judge whether the gate
measured what was intended. The unseen-combination failure is a real limitation, not a metric artefact.

## 6. v1 regression, latency, multimodal

**Text preservation (same host, same harness): 19 / 19 gates pass for all four seeds** (`results/v11/regression/regression.md`).

| gate (candidate − v1) | limit | R2 last 4 s1 / s2 | no adapter s1 / s2 |
|---|---|---|---|
| KrunBench v1 choice accuracy | −0.5 pp | −0.05 / −0.25 | −0.04 / −0.08 |
| native non-English | | −0.04 / −0.14 | −0.14 / −0.11 |
| PT-BR | | +0.26 / −0.09 | −0.09 / +0.17 |
| cross-language | | +0.05 / −0.27 | −0.07 / −0.14 |
| intent | | −0.09 / −0.34 | −0.08 / −0.12 |
| ToolACE v0 holdout | | +0.48 / +0.24 | +0.48 / −0.48 |
| tool accuracy (`val_tools_v2`) | | −0.07 / 0.00 | −0.08 / +0.02 |
| large-k, k = 128 | −2.0 pp | −0.80 / −1.07 | −0.27 / **−1.61** |
| OOD AUROC | | +0.002 / +0.001 | +0.002 / +0.003 |
| OOD false acceptance at 80% coverage | | −1.63 / −1.52 | −1.63 / −1.69 |
| selective coverage | | +1.91 / +2.06 | +1.88 / +2.62 |
| permutation | | −0.74 / −0.49 | −0.49 / −0.49 |
| Score MAE / Spearman | | −0.001 / −0.002 · 0.000 / −0.001 | +0.001 / −0.004 · 0.000 / 0.000 |
| Noul AUROC | | 0.000 / 0.000 | −0.001 / 0.000 |
| Multi micro-F1 | | +0.05 / −0.16 | +0.29 / −0.15 |
| ECE | | +0.002 / +0.001 | +0.002 / +0.002 |

The only movement worth watching is k = 128 (−0.3 to −1.6 pp; limit −2.0). Selectivity improves (coverage +1.9 to
+2.6 pp, OOD false acceptance −1.5 to −1.7 pp). `final_protected` was not opened.

**Latency / VRAM (RTX 4090, same host):** identical to v1 within measurement noise — typical 16.3–16.4 ms (v1 16.4),
4 questions 21.3 (21.3), k = 8 16.2–16.5 (16.4), k = 32 18.0–18.4 (18.6), k = 128 54.7–54.9 (54.9); bf16 weights:
typical 14.4–14.5 (14.6), k = 128 44.0–44.1 (44.1). Peak VRAM 3,523–3,526 MiB autocast, 1,989–1,990 MiB bf16 (v1 3,523 /
1,989). Limits for a 879M core (typical ≤ 20 ms, k = 128 ≤ 50 ms bf16, ≤ 8 GB): met.

**Multimodal M1.5c (fusion weights frozen, text core swapped; `results/v11/multimodal/multimodal_regression.md`):**

| text core under the frozen M1.5c fusion | doc visual-only (G1) | layout (G1) | screenshot (G2) | text-dep. doc (G2) | real speech (G3) | MINDS intent (G3) | image (G4) | synthdoc (G4b) | DocLayNet (G4b) | CORD (G4b) | gates passed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| *registered threshold* | ≥ 67.9 | ≥ P + 5 | ≥ 97.9 | ≥ P − 1 | ≥ 84.1 | ≥ 89.0 | ≥ 63.3 | ≥ 95.3 | ≥ 93.5 | ≥ 95.0 | |
| M1.5c as registered (production host) | 68.0 | 38.7 | 99.2 | 67.2 | 85.1 | 89.7 | 64.8 | 96.1 | 94.5 | 100.0 | 5 / 5 |
| v1 core, this host | 68.7 | 40.1 | 99.2 | 67.1 | 85.1 | 89.7 | 64.8 | 96.2 | 94.5 | 100.0 | 5 / 5 |
| control: v1 weights in the v1.1 model class (untrained adapter) | 68.7 | 40.1 | 99.2 | 67.1 | 85.1 | 89.7 | 64.8 | 96.2 | 94.5 | 100.0 | 5 / 5 |
| R2 last 4 + adapter (seed 1) | 56.0 | 18.6 | 97.6 | 67.3 | 82.4 | 89.3 | 47.3 | 80.3 | 88.8 | 98.6 | 0 / 5 |
| R2 last 4, no adapter (seed 1) | 57.5 | 17.1 | 97.4 | 66.6 | 81.5 | 87.9 | 46.5 | 78.7 | 88.0 | 93.1 | 0 / 5 |

**All five registered gates fail for both R2 variants** (candidate − same-host v1: document visual-only −11 to −13 pp,
layout −22 to −23 pp, image −17 to −18 pp, synthdoc −16 to −17 pp, real speech −2.7 to −3.6 pp; text-dependent cells
−1.8 to +0.2 pp). A control run — the frozen v1 weights loaded through exactly the same v1.1 code path as the candidates (v1.1 model class, adapter wrapper, untrained zero-initialised adapter) — reproduces the v1 core to the last digit on all ten numbers, and the v1 core on this host reproduces the registered M1.5c result (differences ≤ 1.4 pp, all five gates pass). The loss is a property of the candidates, not of the harness.
The variant without the adapter fails the same way, so the adapter is not the cause: the fusion adapters and the head
LoRA (26.7M parameters) were fitted to v1's head and last blocks, and those are exactly what R1 / R2 change. Under this
round's rule (multimodal is not retrained) the registered multimodal gate cannot be met by any candidate that changes
the head; re-fitting the fusion on the new core, or protecting the fused path during text training, is a prerequisite
for promotion and was not attempted here.

## 7. Tetris representation diagnosis (`results/v11/tetris_representation/tetris_views.md`)

Held-out states; top-1 at k = 8 / k = 16 (chance 12.5 / 6.25). A = raw board; B = A + structured board features
(heights, holes, bumpiness…); C = A + the explicit consequence of every placement.

| model | A raw | B structured state | C explicit consequence |
|---|---|---|---|
| v1 | 14.8 / 8.5 | 14.2 / 7.5 | 21.0 / 9.5 |
| R1 (s1) | 18.8 / 12.0 | 17.5 / 11.5 | 82.2 / 94.0 |
| R2 last 4 (s1 / s2) | 17.8 / 9.0 · 16.8 / 9.5 | 17.5 / 11.5 · 17.8 / 11.5 | **100 / 100** · 100 / 100 |
| R2 last 4 no adapter (s1 / s2) | 18.0 / 11.0 · 19.2 / 11.0 | 18.8 / 11.5 · 19.8 / 9.5 | **100 / 100** · 100 / 100 |
| dedicated full fine-tune on A, ~660 steps → ~2,650 steps | 20.5 / 10.5 → 30.5 / 18.0 | | |
| dedicated full fine-tune on B, ~820 steps → 3,000 steps | | 32.0 / 19.0 → 35.8 / 33.5 | |
| dedicated full fine-tune on C, 1,000 steps | | | 100 / 100 |

* **C ≫ B > A.** Given the consequence of each action the model chooses perfectly (and R2 learned that with Tetris at
  only 2% of the mixture). Given only the board — raw or with structured features of the *current* state — it stays
  near chance, and a dedicated full fine-tune moves it to 30–36%, slowly.
* So the missing piece is not the decision (comparison) and not the description of the state: it is **computing what
  each action does to the state** inside a single encoder pass. In the registered decision tree this is the branch
  "representation is the bottleneck → state encoding / consequence supervision", and specifically the consequence side.
* The auxiliary Q / V heads (the value loss) were this round's attempt at consequence supervision; they did not close
  the gap on raw states (state / action raw 18–19% vs 65–66% with consequences).

## 8. External check: JevBench public items (231, evaluation only)

v1 53.2 → R2 last 4 52.4 / 52.4, no adapter 52.4 / 52.4 (hard items: 30.6 → 31.5 / 29.7 and 32.4 / 29.7). Jev 1.13.0: 86.6;
Laya: 58.4. Technical training does not transfer to JevBench; 40% of its hard items do not fit in 1,024 tokens. Nothing
from JevBench was used for training (`results/v11/jevbench/jevbench_public.md`).

## 9. Deviations and limits

1. **R2 last 4 without adapter** was added after the variant selection was seen (≈ US$ 1.4). Reported as post-registration.
2. **Hard-negative round:** 13% of the mined technical examples were Tetris (cap spirit ≤ 2%). The variant lost anyway.
3. **`swe_involved` (noul) had no label:** 6,000 training examples (2.0% of the mixture) carried a degenerate `[1, 1]`
   target and the 120 protected items of that kind are not scored. v1's noul gate did not move. Fixed in code for the
   next build; this round's corpus and checkpoints were not rebuilt (`data/v11/TECHNICAL_CORPUS_PLAN.md` § 8.1).
4. **Tetris diagnostic** first ran one pass (661–1,000 steps) instead of 3,000; A and B were rerun to ~2,650 / 3,000
   steps; both points are in the table.
5. **Objective choice is within noise** (section 2).
6. **Hard-negative share 19%** of the mixture vs the 10% target (near-substitute negatives are a property of real
   items); replay kept at 45%.
7. **Multimodal media rebuild** on the pod skipped the synthetic-speech clips at first (the TTS environment failed to
   install and the script went on) and the pod volume filled up; the 4,840 clips needed by the validation splits were
   regenerated and a re-downloadable cache was deleted. No weight was lost; the v1 core reproduces the registered M1.5c
   numbers on this host (section 6).
8. KrunTechBench v1 is **English only**; its rule tables were reviewed cell by cell by a model, not yet by a person; two
   seeds per variant.

## 10. Artifacts

No `artifacts/krun-one-v1.1-r2-candidate/` (and no R1 artifact) was created: the protocol allows a final artifact only
for a candidate that passes the gates. Two checkpoints were downloaded and sha256-verified so a follow-up does not have
to retrain them — as run outputs, not as artifacts: `runs/v11/r2/r2l4na_s1/model_v1.safetensors` (v1 architecture,
3.5 GB) and `runs/v11/r2/r2l4_s1/model.safetensors` (with adapter, 3.5 GB). All other weights were left on the pods and
are gone (reproducible from the registered seeds, ~29–37 min each). Run records and hashes: `results/v11/runs_meta/`;
pod logs and result checksums: `results/v11/pod_logs/`.

## 11. Cost

See the comparison report, section "Cost".
