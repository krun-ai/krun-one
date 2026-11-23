# Krun One v1.1 — R3 text report

Date: 2026-10-01 / 02 · Protocol (registered before any training step): `configs/v11/r3_protocol.yaml` · Corpus manifest:
`data/v11/r3/manifest.json` · Frozen v1 (`artifacts/krun-one-v1-text-final`, sha 781e5c7b…) was only read.
Companion reports: `KRUN_ONE_V11_R3_GENERALIZATION_REPORT.md`, `KRUN_ONE_V11_R3_JEVBENCH_REPORT.md`,
`KRUN_ONE_V11_R3_MULTIMODAL_REPORT.md`, `KRUN_ONE_V11_R3_FINAL_REPORT.md`.

## 1. Verdict

The R3 text core **keeps R2's technical gain and v1's text behaviour** — KrunTechBench v1 62.2% (R2 62.7%), 19 / 19
preservation gates on both seeds, latency and VRAM unchanged, same architecture and parameter count as v1 — and it **repairs
the decision R2 had destroyed** ("is this file involved?": 49% → 82%). But R3 did **not** deliver what the round was for:

* unseen requirement combinations, constraint flips and the stuck rate are **where R2 left them**;
* the capability curriculum was learned **only when the consequence of each option is written next to it** (81%); from the
  raw state it stays near chance (41%, v1 37%);
* family-aware batching made no difference;
* JevBench did not move significantly (see the JevBench report).

## 2. What was trained

| | R3 text |
|---|---|
| initial weights | the frozen v1 (not the R2 checkpoint — decision documented in the protocol) |
| trainable | encoder blocks 18–21 + final norm (80,229,888) + decision head / scorer / type embedding / answer head (44,883,459) |
| training-only | Q(state, action) / V(state) heads, 13,833 parameters, dropped at export |
| served model | **879,356,931 parameters, the v1 architecture** (`model_v1.safetensors`), no adapter, same tokenizer, same 1,024-token sequence |
| objective / optimizer | O2 (v1's loss + 0.25 × value loss); lr encoder 2e-5, head 5e-5, new 2e-4; one pass; 16,384 tokens per step — all as in R2 |
| corpus | 431,715 examples: replay 42.0% · technical 24.4% · hard negatives 15.9% · capability 9.7% (10.7% with its none-of-the-above items) · OOD 8.1% |
| runs | `r3f_s1`, `r3f_s2` (family-aware batching) · `r3r_s1`, `r3r_s2` (random batching, the R2 batcher) · seeds 20261001 / 20261002 |
| steps / time | 9,495–9,511 steps, 47–49 min per run on one RTX 4090, peak 13.8–14.2 GB VRAM |
| exported weights (sha256) | `r3r_s1` c151c7b0d71f7aca… · `r3r_s2` 63b00067f51b371d… · `r3f_s1` 8f8d39496b2ed1f3… · `r3f_s2` fc510eca40bdd301… |

Corpus changes against R1 / R2 (details and hashes: `data/v11/TECHNICAL_CORPUS_PLAN.md` § 9):

* the 6,000 "is this file involved?" decisions carry their label (0 unlabeled examples; builder and trainer now fail on one);
* rule tables: fact probes (does X provide P / which of these provide P / which is lightest), requirement-set families, sets
  one property larger, invented-product catalogs with random capabilities stated in the context;
* capability curriculum, 16 generators with computed labels (numeric, temporal, multi-hop, long-policy, trade-off, mixed),
  choice + true / false + ordinal questions, four context formats; every item in a counterfactual family;
* 21,994 counterfactual families (77,474 examples).

## 3. The batching experiment (validation only — registered selection)

Score = mean of KrunTechVal v1 macro and the capability macro on `val_r3` (raw). v1: 22.8 / 37.0.

| mode | seed | KrunTechVal v1 macro | capability macro | score | stuck rate, all families | stuck rate, v1 flips | catalog (unseen sets) | `val_fit` drop |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| family-aware | s1 | 51.2 | 41.4 | 46.3 | 63.4 | 27.6 | 57.6 | |
| family-aware | s2 | 52.7 | 43.8 | 48.3 | 63.6 | 25.6 | 58.6 | |
| **family-aware** | mean ± std | | | **47.3 ± 1.0** | 63.5 | | | −0.03 pp |
| random | s1 | 55.7 | 42.3 | 49.0 | 63.8 | 22.5 | 57.9 | |
| random | s2 | 52.4 | 42.0 | 47.2 | 66.9 | 27.6 | 57.0 | |
| **random** | mean ± std | | | **48.1 ± 0.9** | 65.4 | | | −0.60 pp |

Within 1.0 pp on the score and within 2.0 pp on the stuck rate → by the registered rule the simpler recipe wins:
**random batching**, candidate seed **`r3r_s1`** (higher validation score). In 61% of the family-aware batches a
counterfactual sibling sat next to its base (2% with random batching), and it changed nothing measurable: putting the
contrast in the same batch is not what the model is missing.

## 4. KrunTechBench v1 (protected; labels restored; raw rendering unless stated)

| metric | v1 | R2 (kept seed / mean of 2) | R3 seed 1 | R3 seed 2 | R3 mean ± std |
|---|---:|---:|---:|---:|---:|
| macro top-1, 10 tracks | 30.9 | 61.7 / 62.7 | 62.9 | 61.6 | 62.2 ± 0.7 |
| paired difference vs v1 (95% CI) | — | +26.5 [+24.9, +28.1] | +27.7 [+26.1, +29.2] | +26.5 [+24.9, +28.1] | |
| macro, explicit consequences | 45.3 | 83.9 | 84.6 | 86.5 | 85.6 |
| gold > hardest negative | 66.9 | 79.4 | 79.5 | 79.5 | 79.5 |
| hard-negative margin (logit) | +1.12 | +4.21 / +4.29 | +4.08 | +4.19 | +4.14 |
| hard-negative slice | 39.2 | 65.7 | 65.8 | 65.5 | 65.7 |
| unseen requirement combinations | 52.1 | 59.9 / 61.2 | 58.3 | 59.8 | 59.0 |
| seen requirement combinations | 39.5 | 99.8 | 100.0 | 100.0 | 100.0 |
| unseen technologies | 39.0 | 76.0 | 75.5 | 74.5 | 75.0 |
| large candidate sets | 24.2 | 51.1 | 51.4 | 50.4 | 50.9 |
| unseen simulator families | 12.7 | 9.8 | 8.8 | 8.1 | 8.5 |
| constraint flips, both correct (286 pairs) | 2.8 | 46.5 / 48.3 | 49.7 | 44.4 | 47.0 |
| stuck rate (base right → variant keeps the base answer) | 65.1 | 20.8 / 18.7 | 20.1 | 21.7 | 20.9 |
| OOD false acceptance | 0.4% | 0.2% | 0.2% | 0.0% | 0.1% |
| **"is this file involved?" (true / false, 120 items)** | 60.8 | **49.2** | **83.3** | **80.8** | 82.1 |

By track (raw): api 60 / 55 · backend 54 / 64 · cloud 49 / 45 · database 68 / 67 · devops 91 / 89 · distributed 95 / 94 ·
ML infra 57 / 55 · security 60 / 52 · software repair 74 / 76 · state / action 20 / 19 (R2 kept seed: 55, 73, 47, 70, 87, 95,
47, 50, 75, 18). Track-level differences between R2 and R3 are within what two seeds of the same recipe show.

The technical benchmark is where R2 left it. The rebuilt corpus fixed what was broken (the unlabeled decision), and the
new table material neither helped nor hurt the old slices. Details on generalization: the generalization report.

## 5. R3 benchmark extension (protected; never trained on)

| metric (raw) | v1 | R2 kept seed | R3 seed 1 | R3 seed 2 |
|---|---:|---:|---:|---:|
| capability macro (6 tracks, all question types) | 37.2 | 37.0 | 41.2 | 41.0 |
| numeric | 40.1 | 39.2 | 43.2 | 44.3 |
| temporal | 34.8 | 34.1 | 42.0 | 41.2 |
| multi-hop | 33.4 | 33.1 | 32.8 | 32.8 |
| long-policy | 38.3 | 39.3 | 45.9 | 45.5 |
| trade-off | 33.5 | 32.7 | 37.7 | 34.6 |
| mixed | 43.1 | 43.8 | 45.7 | 47.8 |
| choice / true-false / ordinal | 27.5 / 54.5 / 42.0 | 27.7 / 53.5 / 42.0 | 32.0 / 56.2 / 55.5 | 31.6 / 56.5 / 54.6 |
| seen / unseen scenarios | 37.3 / 36.4 | 37.2 / 36.2 | 42.5 / 39.2 | 42.6 / 38.8 |
| paired difference vs v1 (95% CI, 2,821 items) | — | −0.2 [−1.5, +1.1] | +3.9 [+1.8, +6.0] | +3.8 [+1.6, +5.8] |
| **capability macro with explicit consequences** | 35.6 | 45.2 | **81.7** | **81.4** |
| invented-product catalogs, unseen sets | 31.7 | 27.4 | 55.6 | 54.7 |
| … with capabilities next to each option | 32.1 | 36.9 | 72.6 | 70.9 |

* **+4 pp from the raw state, +46 pp when the consequence is stated.** The model learned to *use* a computed outcome
  ("meets every limit; cost 1,800", "ends 22:20, 15 min to spare", "clause 5 is the first that matches") in every
  capability, and learned almost nothing about *computing* it: comparing numbers against limits, adding durations to
  clock times, walking two to four lookups, finding the first matching clause. True / false stays at 56% (chance 50%).
  This is the Tetris pattern (raw board ~20%, explicit consequence 100%) across six capabilities.
* Catalogs — the cleanest test of "read the facts in the context and apply the rule" — go from 32% to 55% (73% with the
  capabilities repeated next to each option): partially learned.
* Per generator (raw, R3 seeds): the only clear movers are `tmp_order` 32 → 53–55, `num_errbudget` 43 → 58–64,
  `pol_backlog` 32 → 48–49; `hop_lookup` (27), `num_limits` (39), `tmp_window` (32–33) do not move at all.

## 6. v1 regression, latency, memory (same host as v1)

**19 / 19 preservation gates on both seeds** (`results/v11/r3/regression/regression.md`). Candidate − v1: KrunBench v1
choice accuracy +0.06 / +0.03 pp · native non-English 0.00 / −0.07 · PT-BR +0.26 / +0.17 · cross-language −0.02 / 0.00 ·
intent +0.03 / +0.02 · ToolACE holdout +0.24 / −0.48 · tool accuracy +0.31 / +0.08 · **k = 128 −0.80 / −1.07 (limit −2.0)** ·
OOD AUROC +0.004 / +0.003 · OOD false acceptance at 80% coverage −1.58 / −1.63 · selective coverage +2.11 / +1.63 ·
permutation −0.25 / −0.98 · Score MAE 0.000 / −0.003 · Noul AUROC 0.000 / −0.001 · Multi F1 +0.22 / −0.02 · ECE +0.001 / 0.000.
`final_protected` was not opened.

| latency, p50 / p95 ms (RTX 4090) | v1 | R3 seed 1 | R3 seed 2 |
|---|---|---|---|
| typical, 1 question, 8 tools | 17.8 / 18.3 | 17.3 / 18.0 | 18.4 / 19.6 |
| 4 questions | 21.7 / 22.1 | 21.6 / 21.8 | 21.6 / 23.2 |
| k = 8 | 17.8 / 18.3 | 17.0 / 17.8 | 18.0 / 20.5 |
| k = 32 | 19.8 / 20.4 | 19.0 / 19.5 | 19.8 / 20.2 |
| k = 128 | 55.5 / 55.9 | 55.3 / 55.9 | 55.2 / 55.6 |
| k = 128, bf16 weights | 44.6 / 45.0 | 44.3 / 44.6 | 44.4 / 44.9 |
| typical, bf16 weights | 16.2 / 16.6 | 15.2 / 15.6 | 16.1 / 17.5 |
| peak VRAM autocast / bf16 | 3,523 / 1,989 MiB | 3,523 / 1,989 | 3,523 / 1,989 |
| process peak RSS | 8,837 MiB | 8,842 | 8,838 |

Identical model, identical cost. (This host was ~1.5 ms slower than the R1 / R2 host on every short workload, v1 included;
seed 1 measured 1 ms *faster* than v1 on the typical workload, which is host noise.)

## 7. Tetris (diagnostic only)

Held-out states, top-1 at k = 8 / k = 16: raw board 20.8 / 9.0 and 22.5 / 11.0 (v1 14.8 / 8.5; R2 18.0 / 11.0); structured
state 22.0 / 8.5 and 21.8 / 14.0; explicit consequence 100 / 100. Unchanged: the model compares consequences and does not
compute them.

## 8. Post-registration diagnostic: can the capability curriculum be learned at all?

**Not registered. Throwaway runs, not candidates; no gate or selection uses them.** After the first validation numbers
showed the capability curriculum near chance, two dedicated runs asked whether it can be learned at all from the raw state.
Both start from the frozen v1, train ONLY on the 46,068 capability examples of the R3 corpus (no replay), 3,000 steps,
same objective and learning rates, and are evaluated on `val_r3` (validation, never the protected extension):

| raw rendering, `val_r3` | v1 | R3 (2 seeds) | **last 4 blocks, capability only** | **all 22 encoder blocks, capability only** |
|---|---:|---:|---:|---:|
| trainable encoder parameters | — | 80.2M | 80.2M | 441.3M (embeddings frozen) |
| capability macro | 37.0 | 42.3 / 42.0 | 43.9 | **81.6** |
| numeric | 34.6 | 46.0 / 46.3 | 47.8 | 66.5 |
| temporal | 38.5 | 42.7 / 38.5 | 45.2 | 74.1 |
| multi-hop (2 / 3 / 4 hops) | 34.0 | 36.2 / 36.9 | 35.6 | **100.0** (100 / 100 / 100) |
| long-policy | 36.4 | 43.8 / 44.2 | 50.5 | 84.1 |
| trade-off | 36.3 | 39.3 / 41.1 | 42.3 | **99.4** |
| mixed | 42.3 | 45.8 / 45.1 | 42.3 | 65.5 |
| choice / true-false / ordinal | 27.8 / 52.9 / 41.9 | 33 / 57 / 58 | 34.9 / 59.9 / 59.7 | **82.4 / 83.7 / 74.2** |
| capability families: both correct | 0.9 | 3.7 | 6.8 | **64.8** |
| capability families: stuck rate | 97.2 | 86.9 | 78.2 | **18.2** |
| with explicit consequences (macro) | 36.0 | 84.8 | 93.1 | 98.9 |
| time (RTX 4090) | — | 47–49 min | 13.8 min | 32.1 min |

Per generator, all 22 blocks: lookups 100, rule chains 100, ordering 97, trade-off priorities 99, policy document 95,
limits 81, error budget 78, horizon 72, dates 66, backlog 62, windows 60, budget 52, capacity 38 (the two that need
multiplication / division stay low).

**Reading.** (1) It is not an exposure problem: three passes over the capability data alone, with the R3 trainable set,
give the same near-chance result as R3 (43.9 vs 42). (2) It is not a capacity or architecture problem either: the same
879M model learns multi-hop lookups, policy application, ordering and priority trade-offs from the raw state almost
perfectly, and stops being stuck (87% → 18%), **when all 22 blocks adapt**. (3) The limit R3 hit is the *depth of
adaptation* fixed for this line — "last 4 blocks". Reading numbers against limits, following links and matching clauses
has to be set up in the lower and middle layers; four top blocks receive representations in which that work was never
done.

What this diagnostic does **not** show: that a full-encoder run preserves v1 (no replay was used), that it keeps the
multimodal path, or that it generalises beyond these generators.

**Second diagnostic (also not registered): does the learned curriculum transfer?** The all-blocks run was repeated
(same recipe and seed; capability macro 80.7%, stuck rate 17.2% — reproduced) and evaluated outside its own generators:

| | v1 | R3 seed 1 | all 22 blocks, capability only, no replay |
|---|---:|---:|---:|
| capability macro, raw (`val_r3`) | 37.0 | 42.3 | 80.7 |
| JevBench, all 231 | 53.2 | 53.7 | **43.3** |
| JevBench, items that fit (≤ 1,024 tokens) | 59.4 | 58.8 | 48.7 |
| JevBench, hard items that fit | 32.8 | 35.8 | 29.9 |
| JevBench hard families: temporal_numeric / tradeoff / multi_hop / long_policy | 20 / 33 / 17 / 32 | 33 / 67 / 17 / 37 | 13 / 50 / 17 / 21 |
| KrunBench v1 `val_fit` choice accuracy | 95.5 | 96.0 | **84.4** |
| KrunTechVal v1 macro | 22.8 | 55.7 | 22.1 |

Two things at once: (1) without replay, adapting the whole encoder **forgets** — 11 pp on v1's own validation set after
3,000 steps — so the depth that makes the curriculum learnable is exactly what R1 / R2 / R3 avoided for preservation; (2)
having learned these generators to 81% bought **nothing on the external benchmark**, not even on its numeric / temporal
family or on short items. The two effects are confounded in this run (forgetting alone can explain the drop), so it does
not prove that the curriculum cannot transfer; it does show that "learn these generators" is not sufficient evidence of a
general capability, and that the generators are narrower than the benchmark's items.

## 9. Registered gates (text part)

| gate | required | measured | |
|---|---|---|---|
| KrunTechBench v1 macro | mean ≥ 61.7 | 62.9 / 61.6 (mean 62.2) | ✅ |
| unseen combinations | mean ≥ 71.2, each ≥ 66.2 | 58.3 / 59.8 | ❌ |
| constraint flips, both correct | mean ≥ 53.3 | 49.7 / 44.4 | ❌ |
| stuck rate | mean ≤ 13.7% | 20.1 / 21.7 | ❌ |
| hard-negative margin | mean ≥ 3.99 | 4.08 / 4.19 | ✅ |
| text preservation | 19 / 19 every seed | 19 / 19 · 19 / 19 | ✅ |
| latency / VRAM | not slower than v1 by > 1 ms (typical) / 2 ms (k = 128); VRAM +50 MiB | −1.08 / −0.32 ms, −0.16 / −0.21 ms; +0 MiB | ✅ |

## 10. Deviations and limits

* The capability share is 9.7% of the examples counting answerable items only (10.7% with its none-of-the-above items),
  just under the 10–15% band; technical + hard negatives are 40.3% (band 30–35%) because the table additions were added
  on top of the R2 technical corpus.
* Technical and capability data are English only; rule tables were reviewed by a model, not by a person.
* The capability benchmark tracks come from the same generators as the training data (other seeds, plus benchmark-only
  scenarios). They measure whether the skill was learned, not whether it transfers; JevBench is the external check.
* The diagnostic in § 8 was not registered; it is reported as a diagnostic and no gate or selection uses it.
* The latency gate was registered as "within 1 ms"; it is applied one-sided (not slower), see § 6.
* The R2 reference on the new files uses the one R2 seed whose weights were kept (`r2l4na_s1`).
