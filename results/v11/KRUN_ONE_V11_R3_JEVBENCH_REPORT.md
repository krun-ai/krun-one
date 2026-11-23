# Krun One v1.1 — R3 JevBench report (external transfer; evaluation only)

Date: 2026-10-02 · 231 public items in `data/v11/external/jevbench/` (EVAL_ONLY) · run only after training ·
mapping unchanged (`scripts/v11/jevbench_eval.py`) · tables: `results/v11/r3/jevbench/jevbench_r3.md` · taxonomy:
`data/v11/JEVBENCH_CAPABILITY_GAPS.md`. No item was used, paraphrased or mutated for training; the corpus manifest records
0 training examples sharing an 8-word sequence with any public item.

## 1. Verdict

**No measurable external transfer.** R3 scores 53.7% and 56.3% on the 231 items against 53.2% for v1 (same host). The
paired difference of the candidate seed is +0.4 pp with a 95% interval of [−4.3, +5.6]; the other seed is +3.0 pp
[−1.7, +7.8]. Neither interval excludes zero, so the registered gate fails. Registered reading: **case C — internal
benchmarks moved, the external one did not: domain specialisation, not a general decision improvement.** (Internally the
capability tracks themselves barely moved from the raw state — text report § 5 — so this is the expected outcome, not a
surprise.)

## 2. Full table

| cut | n | Jev 1.13.0 | GPT-5.6 Luna | Laya 421M | Krun v1 | R2 s1 | R2 s2 | **R3 s1** | **R3 s2** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all 231 | 231 | 86.6 | 97.4 | 58.4 | 53.2 | 52.4 | 52.4 | 53.7 | 56.3 |
| easy | 48 | 100.0 | 100.0 | 95.8 | 91.7 | 91.7 | 89.6 | 89.6 | 91.7 |
| standard | 72 | 98.6 | 97.2 | 69.4 | 62.5 | 56.9 | 62.5 | 59.7 | 63.9 |
| hard | 111 | 73.0 | 96.4 | 35.1 | 30.6 | 32.4 | 29.7 | 34.2 | 36.0 |
| ≤ 1,024 tokens (fits) | 187 | 90.9 | 97.3 | 63.6 | 59.4 | 57.8 | 58.8 | 58.8 | 61.0 |
| > 1,024 tokens (truncated) | 44 | 68.2 | 97.7 | 36.4 | 27.3 | 29.5 | 25.0 | 31.8 | 36.4 |
| hard, ≤ 1,024 | 67 | 76.1 | 95.5 | 34.3 | 32.8 | 34.3 | 32.8 | 35.8 | 35.8 |
| hard, > 1,024 | 44 | 68.2 | 97.7 | 36.4 | 27.3 | 29.5 | 25.0 | 31.8 | 36.4 |
| length 0-512 | 168 | 93.5 | 97.6 | 66.1 | 62.5 | 60.1 | 60.1 | 60.7 | 63.7 |
| length 513-1024 | 19 | 68.4 | 94.7 | 42.1 | 31.6 | 36.8 | 47.4 | 42.1 | 36.8 |
| length 1025-2048 | 7 | 42.9 | 100.0 | 71.4 | 28.6 | 28.6 | 28.6 | 42.9 | 57.1 |
| length 2049-4096 | 37 | 73.0 | 97.3 | 29.7 | 27.0 | 29.7 | 24.3 | 29.7 | 32.4 |
| choice | 139 | 88.5 | 98.6 | 56.8 | 55.4 | 54.0 | 54.0 | 55.4 | 58.3 |
| true / false | 74 | 85.1 | 94.6 | 58.1 | 51.4 | 48.6 | 48.6 | 48.6 | 51.4 |
| ordinal | 18 | 77.8 | 100.0 | 72.2 | 44.4 | 55.6 | 55.6 | 61.1 | 61.1 |

Paired difference to v1 on the same host (bootstrap 95% CI; items gained / lost):

| model | all 231 | hard | ≤ 1,024 | > 1,024 | hard ≤ 1,024 |
|---|---|---|---|---|---|
| R2 s1 | -0.9 pp [-5.2, +3.5] (+11 / −13) | +1.8 pp [-4.5, +9.0] (+9 / −7) | -1.6 pp [-5.9, +2.7] (+7 / −10) | +2.3 pp [-9.1, +13.6] (+4 / −3) | +1.5 pp [-7.5, +10.4] (+5 / −4) |
| R2 s2 | -0.9 pp [-4.8, +3.0] (+9 / −11) | -0.9 pp [-7.2, +6.3] (+7 / −8) | -0.5 pp [-4.3, +3.2] (+6 / −7) | -2.3 pp [-13.6, +9.1] (+3 / −4) | +0.0 pp [-9.0, +9.0] (+4 / −4) |
| R3 s1 (candidate) | +0.4 pp [-4.3, +5.6] (+18 / −17) | +3.6 pp [-4.5, +11.7] (+13 / −9) | -0.5 pp [-5.9, +4.8] (+13 / −14) | +4.5 pp [-9.1, +18.2] (+5 / −3) | +3.0 pp [-7.5, +13.4] (+8 / −6) |
| R3 s2 | +3.0 pp [-1.7, +7.8] (+20 / −13) | +5.4 pp [-3.6, +13.5] (+15 / −9) | +1.6 pp [-3.2, +6.4] (+11 / −8) | +9.1 pp [-6.8, +25.0] (+9 / −5) | +3.0 pp [-6.0, +11.9] (+6 / −4) |

The two R3 seeds differ from v1 on 33–35 items each, gaining about as many as they lose: the predictions changed, the
accuracy did not.

## 3. Hard tier by family and by capability slice

| cut | n | Jev 1.13.0 | GPT-5.6 Luna | Laya 421M | Krun v1 | R2 s1 | R2 s2 | **R3 s1** | **R3 s2** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adversarial | 6 | 100.0 | 100.0 | 66.7 | 50.0 | 50.0 | 33.3 | 50.0 | 33.3 |
| ambiguous | 7 | 85.7 | 100.0 | 14.3 | 28.6 | 42.9 | 42.9 | 14.3 | 28.6 |
| judge_hard | 17 | 76.5 | 94.1 | 41.2 | 47.1 | 41.2 | 41.2 | 41.2 | 41.2 |
| long_policy | 19 | 63.2 | 100.0 | 21.1 | 31.6 | 42.1 | 31.6 | 36.8 | 36.8 |
| multi_hop | 18 | 83.3 | 94.4 | 33.3 | 16.7 | 16.7 | 16.7 | 16.7 | 27.8 |
| probability | 10 | 70.0 | 90.0 | 50.0 | 40.0 | 30.0 | 30.0 | 50.0 | 50.0 |
| routing_hard | 5 | 100.0 | 100.0 | 20.0 | 20.0 | 60.0 | 40.0 | 60.0 | 80.0 |
| temporal_numeric | 15 | 26.7 | 93.3 | 33.3 | 20.0 | 20.0 | 20.0 | 33.3 | 33.3 |
| tradeoff | 6 | 83.3 | 100.0 | 100.0 | 33.3 | 50.0 | 66.7 | 66.7 | 50.0 |
| trap | 8 | 100.0 | 100.0 | 0.0 | 25.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Mapped onto the R3 curriculum (benchmark families grouped by their own names; no item was read for the mapping):

| cut | n | Jev 1.13.0 | GPT-5.6 Luna | Laya 421M | Krun v1 | R2 s1 | R2 s2 | **R3 s1** | **R3 s2** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| long_policy | 19 | 63.2 | 100.0 | 21.1 | 31.6 | 42.1 | 31.6 | 36.8 | 36.8 |
| multi_hop | 18 | 83.3 | 94.4 | 33.3 | 16.7 | 16.7 | 16.7 | 16.7 | 27.8 |
| numeric + temporal | 15 | 26.7 | 93.3 | 33.3 | 20.0 | 20.0 | 20.0 | 33.3 | 33.3 |
| tradeoff | 6 | 83.3 | 100.0 | 100.0 | 33.3 | 50.0 | 66.7 | 66.7 | 50.0 |
| not targeted (judgment, traps, ambiguity, probability, routing) | 53 | 84.9 | 96.2 | 34.0 | 37.7 | 35.8 | 32.1 | 35.8 | 37.7 |

* **numeric + temporal** (15 items): 20.0% → 33.3% on both seeds, i.e. 3 → 5 items. In the right direction, far too
  few items to claim.
* **trade-off** (6 items): 33% → 67% / 50% (2 → 4 / 3 items). Same caveat. (R2, without any trade-off training, was at
  50% / 67%.)
* **long-policy** (19 items, all truncated): 31.6% → 36.8% (6 → 7 items). **multi-hop** (18 items, 17 truncated):
  16.7% → 16.7% / 27.8%. These cannot be read as capability results while 1,300–2,300 tokens of each state are cut.
* **Families R3 did not target** (53 items): 37.7% → 35.8% / 37.7% — flat, as expected. Traps: 0 of 8 for every R2 / R3
  seed (v1: 2 of 8).
* **True / false** stays at chance (48.6% / 51.4%; v1 51.4%; Jev 85.1%). **Ordinal** rises 44.4% → 61.1% on both seeds
  (8 → 11 of 18 items; R2 was already at 55.6%).

## 4. Context length: how much of the gap is truncation?

Sequence limit 1,024 tokens (unchanged in R3). 19% of the items — all in the hard tier, 40% of it — lose
part of their state; the median loss is 1468 tokens.

| cut | n | Jev 1.13.0 | GPT-5.6 Luna | Laya 421M | Krun v1 | R2 s1 | R2 s2 | **R3 s1** | **R3 s2** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lost 1-1000 tokens | 7 | 42.9 | 100.0 | 71.4 | 28.6 | 28.6 | 28.6 | 42.9 | 57.1 |
| lost 1001-2000 tokens | 26 | 84.6 | 96.2 | 34.6 | 30.8 | 30.8 | 23.1 | 30.8 | 30.8 |
| lost > 2000 tokens | 11 | 45.5 | 100.0 | 18.2 | 18.2 | 27.3 | 27.3 | 27.3 | 36.4 |

* On the **187 items that fit**, Krun is at 59–61% against 90.9% for Jev: **a 30-point gap with no truncation involved.**
  On the 67 hard items that fit: 36% against 76%.
* On the **44 truncated items**: v1 27%, R3 32–36%, Jev 68%. Even a perfect model on the truncated items would add at most
  44 × (1 − 0.3) ≈ 31 items ≈ 13 pp to the total; matching Jev there would add ≈ 7 pp.
* So truncation explains roughly **a fifth to a quarter** of the distance to Jev on this set (≈ 7 of 33 pp); the rest is
  reasoning on inputs that fit. R3 did not improve the items that fit (−0.5 / +1.6 pp), so the registered trigger for
  case E ("≤ 1,024 improves strongly, > 1,024 does not → context extension next") is **not** met: a longer context is
  necessary for two families (long_policy, multi_hop) but it is not the main blocker.

## 5. What the external check says about R3

1. The curriculum that was supposed to transfer was not learned from the raw state (text report § 5), so there was little
   to transfer. Where it was partly learned (ordering in time, simple budget arithmetic) the corresponding benchmark
   family moved by two items.
2. The largest single deficit against the systems above Krun is **true / false decisions that require reasoning** — at
   chance before and after.
3. Jev (86.6%) and a small language model (97.4%) solve these items by generating intermediate steps or by a far larger
   computation budget per decision; Laya, the other single-pass encoder, sits where Krun sits (58.4%). That is a statement
   about the decision procedure (one encoder pass, no intermediate result), not about parameter count.
