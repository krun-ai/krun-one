# Krun One v1.1 — R3 final report

Date: 2026-10-02 · Protocol and gates registered before training: `configs/v11/r3_protocol.yaml` · Details:
`KRUN_ONE_V11_R3_TEXT_REPORT.md`, `…_GENERALIZATION_REPORT.md`, `…_JEVBENCH_REPORT.md`, `…_MULTIMODAL_REPORT.md` · Gate table:
`results/v11/r3/gates.md` · Taxonomy: `data/v11/JEVBENCH_CAPABILITY_GAPS.md`.

State at the end of the round: **R3 measured** (4 text runs, 2 seeds of the selected recipe) · **external transfer
measured** (JevBench, by tier / length / family / type) · **multimodal recovered in large part and diagnosed** (2 / 5 gates)
· full text regression done · latency / VRAM measured · cost reported · both pods terminated (`list-pods` empty) ·
no 1.5B, no new architecture / tokenizer, no RL, no context extension, **no deploy, no artifact created** · the frozen v1
(sha 781e5c7b…) untouched · JevBench never used for training (8-gram overlap check: 0).

## 1. Outcome in one table

| registered gate | required | measured (seed 1 / seed 2) | |
|---|---|---|---|
| KrunTechBench v1 macro top-1 | ≥ 61.7 (R2 62.7) | 62.9 / 61.6 | ✅ |
| unseen requirement combinations | ≥ 71.2 (R2 61.2 + 10) | 58.3 / 59.8 | ❌ |
| constraint flips, both correct | ≥ 53.3 (R2 48.3 + 5) | 49.7 / 44.4 | ❌ |
| stuck rate | ≤ 13.7% (R2 18.7 − 5) | 20.1 / 21.7 | ❌ |
| hard-negative margin | ≥ 3.99 logit | 4.08 / 4.19 | ✅ |
| text preservation | 19 / 19, every seed | 19 / 19 · 19 / 19 | ✅ |
| multimodal, re-aligned fusion | 5 / 5 | 2 / 5 (old fusion 0 / 5) | ❌ |
| JevBench, all 231, candidate − v1 | 95% CI above 0 | +0.4 pp [−4.3, +5.6] (other seed +3.0 [−1.7, +7.8]) | ❌ |
| latency / VRAM | not slower than v1 | same (−1.1 / −0.2 ms typical; +0 MiB) | ✅ |
| **candidate** | all | | **❌ — no Krun One v1.1 candidate** |

Registered decision tree: **C** (internal benchmark held, JevBench flat → domain specialisation) + **D** (unseen stays
~60%, stuck stays ~20% → dataset / generalization not solved; do not grow the model) + **B** for the multimodal part
(not 5 / 5 → no promotion). **E** (context length next) is *not* triggered: the items that fit did not improve.

## 2. The twenty questions

**1. Did the corrected rebuild improve generalization?** No. It repaired the decision the missing labels had destroyed
("is this file involved?": v1 60.8%, R2 49.2%, R3 83.3 / 80.8%), and everything else on the R1 / R2 benchmark is where R2
left it (macro 62.2 vs 62.7).

**2. Unseen combinations?** 59.0% (58.3 / 59.8) against 61.2% for R2: −2 pp, inside the seed spread; gate (+10 pp) failed.
By set size: 1 requirement 39.6% (R2 40.6%, v1 46.6%), 2 requirements 67.8% (R2 70.6%), 3 requirements 96.5% (R2 97.7%).
On invented-product catalogs, where nothing can be memorised: 55% (v1 32%, R2 27%).

**3. Constraint flips?** Both members right: 47.0% (49.7 / 44.4) vs 48.3% for R2. No change. With the consequence
stated: 95.8%.

**4. Stuck rate?** 20.9% (20.1 / 21.7) vs 18.7% for R2 (its seeds: 20.8 / 16.7). No drop. On the new capability families
it is 85% from the raw state and 5–7% with consequences stated.

**5. Did the long-policy curriculum help?** Internally a little: 38.3 → 45.7% raw (policy documents 40 → 46–47, backlog
32 → 48–49, horizon 38 → 41–42), 88% with consequences. Externally not measurably: the benchmark's long_policy family
(19 items, all truncated) 6 → 7 correct.

**6. Multi-hop?** No. 33.4 → 32.8% raw, flat at 2, 3 and 4 hops; 87–89% with the chain written next to each option.
Benchmark family (18 items, 17 truncated): 3 → 3 / 5 correct.

**7. Numeric?** Marginally: 40.1 → 43.8% raw (error budget 43 → 58–64; limits 39 → 39, budget 42 → 39–42). Benchmark
temporal_numeric family: 3 → 5 of 15.

**8. Temporal?** Partly: 34.8 → 41.6% raw, almost all from ordering (32 → 53–55%); clock windows 34 → 32–33 and dates
37 → 38–39 unchanged.

**9. Trade-off?** Barely: 33.5 → 36.2% raw; 77–80% with the priorities resolved next to each option. Benchmark family:
2 → 4 / 3 of 6.

**10. Hard-negative separation preserved?** Yes. Gold above the hardest negative 79.5% (R2 79.4%), margin +4.14 logit
(R2 +4.29), hard-negative slice 65.7%, OOD false acceptance 0.1%.

**11. 19 / 19 text gates?** Yes, on both seeds. Largest negative: k = 128 −0.80 / −1.07 pp (limit −2.0). Selective
coverage +1.6 to +2.1 pp, OOD false acceptance −1.6 pp.

**12. Did the re-aligned fusion recover 5 / 5?** No — **2 / 5**. From 0 / 5 and −17 to −27 pp with the old fusion to within
3.0 pp of the v1 stack on every gated cell after the registered 3-epoch fallback (1 / 5 after one epoch). G1 and G2 pass;
G3 fails on MInDS-14 intent by 0.4 pp, G4b on DocLayNet category by 0.4 pp, G4 (image) by 1.5 pp.

**13. Did image / document / audio regress?** Documents: no (visual-only +1.7 pp, layout +2.6 pp, screenshots −0.4 pp).
Audio: real speech +0.1 pp, MInDS-14 −1.1 pp, acoustic-only +5.5 pp. **Image: yes, −3.0 pp**, and it plateaus
(−5.0 → −3.8 → −4.0 pp over the checkpoints on `val_fit`).

**14. JevBench?**

| | v1 | R2 (2 seeds) | R3 seed 1 | R3 seed 2 | Jev 1.13.0 |
|---|---:|---:|---:|---:|---:|
| all 231 | 53.2 | 52.4 / 52.4 | 53.7 | 56.3 | 86.6 |
| hard (111) | 30.6 | 32.4 / 29.7 | 34.2 | 36.0 | 73.0 |
| ≤ 1,024 tokens (187) | 59.4 | 57.8 / 58.8 | 58.8 | 61.0 | 90.9 |
| > 1,024 tokens (44) | 27.3 | 29.5 / 25.0 | 31.8 | 36.4 | 68.2 |
| hard ≤ 1,024 (67) | 32.8 | 34.3 / 32.8 | 35.8 | 35.8 | 76.1 |
| 0–512 / 513–1,024 / 1,025–2,048 / 2,049–4,096 | 62.5 / 31.6 / 28.6 / 27.0 | | 60.7 / 42.1 / 42.9 / 29.7 | 63.7 / 36.8 / 57.1 / 32.4 | 93.5 / 68.4 / 42.9 / 73.0 |
| long_policy / multi_hop / temporal_numeric / tradeoff | 31.6 / 16.7 / 20.0 / 33.3 | | 36.8 / 16.7 / 33.3 / 66.7 | 36.8 / 27.8 / 33.3 / 50.0 | 63.2 / 83.3 / 26.7 / 83.3 |
| choice / true-false / ordinal | 55.4 / 51.4 / 44.4 | | 55.4 / 48.6 / 61.1 | 58.3 / 51.4 / 61.1 | 88.5 / 85.1 / 77.8 |

No paired interval excludes zero (candidate +0.4 pp [−4.3, +5.6]; ≤ 1,024: −0.5 pp [−5.9, +4.8]).

**15. How much of the gap looks like context truncation?** Roughly a fifth to a quarter. 44 items (19%; 40% of the hard
tier) lose a median of ~1,470 tokens; on them Krun is at 27–36% against Jev's 68%, which is ≈ 7 of the ≈ 32 points
between the two systems. On the 187 items that fit, the gap is 30 points with no truncation involved.

**16. Latency?** Unchanged: typical 17.3 / 18.4 ms (v1 17.8 on the same host), 4 questions 21.6, k = 128 55.3 / 55.2 ms
(v1 55.5); bf16 weights 15.2 / 16.1 and 44.3 / 44.4 ms. Multimodal: image 13.5 ms, scanned document 80.3 ms, audio 5 s
77.6 ms — equal to the v1 stack.

**17. VRAM?** Unchanged: 3,523 MiB autocast, 1,989 MiB bf16; process RSS 8.84 GB (v1 8.84 GB).

**18. Is there a promotable Krun One v1.1 candidate?** **No.** Five of nine gates pass. The R3 core is a safe drop-in
for text (same architecture and cost, v1 preserved, +31 pp on KrunTechBench) but it is not better than R2 on what R3 was
for, it is not an external improvement, and it does not carry the multimodal path at 5 / 5.

**19. What is the main blocker now?** **The model does not compute consequences from the raw state within the adaptation
this line allows.** Every R3 failure is that one thing: stuck (85% from the raw state, 5% with consequences), unseen
single-property decisions, the capability curriculum (41% vs 81%), Tetris (21% vs 100%), JevBench's short hard items.
The two post-registration diagnostics locate it:

| trained only on the capability curriculum, 3,000 steps | capability, raw | stuck rate | v1 `val_fit` | JevBench |
|---|---:|---:|---:|---:|
| v1 (reference) | 37.0 | 97 | 95.5 | 53.2 |
| last 4 blocks (the R3 trainable set) | 43.9 | 78 | — | — |
| **all 22 encoder blocks** | **81.6** (multi-hop 100, trade-off 99, policy 95, ordering 97) | **18** | **84.4** (no replay) | **43.3** |

* It is **not exposure** (the last-4 run on the curriculum alone is as bad as R3) and **not capacity** (the same 879M
  model learns it when all blocks adapt). It is the **depth of adaptation**: "last 4 blocks" keeps v1 and the latency,
  and cannot build what these decisions need.
* Deeper adaptation is not free: without replay it loses 11 pp on v1's own validation set in 3,000 steps, and — on this
  evidence — learning these generators does not move JevBench at all. So the next question is not "unfreeze everything"
  but **how much depth can be opened with replay (and with the fused path protected) before preservation breaks, and
  whether what is learned then transfers** — measured on JevBench during selection, not at the end.
* Second blocker, independent: the multimodal path (image −3 pp after re-alignment).

**20. Any new evidence that justifies more than 879M?** **No — the opposite.** The one experiment in this round that
changed the picture did it with the same 879M parameters and zero new ones. R0 (143M / 318M / 879M reach the same
ceiling), R1 / R2 (the head is not the bottleneck) and R3 (depth of adaptation, not size) all point away from growth.

## 3. What R3 established (to keep) and what it retired

Keep:
* the R3 text core as the reference "safe" recipe (v1 architecture, v1 preserved, technical +31 pp);
* the corrected corpus builder, the label assertions, counterfactual families, the stuck-rate definition, the split of
  "unseen" by set size, the catalog and capability generators as **diagnostic instruments**;
* fusion re-alignment as a working repair for documents and speech (≥ 3 epochs);
* JevBench length / family / type breakdown as the external yardstick.

Retired (do not spend another round on them without new evidence):
* family-aware batching (no effect, two seeds);
* fact probes in true / false or multi-select form as a fix for unseen single-requirement sets (no transfer);
* "more curriculum in the same recipe" — the last-4 run on the curriculum alone shows where that ends;
* context extension as the *first* lever for JevBench (the items that fit are where most of the gap is).

## 4. Options for the owner (nothing started)

1. **Depth-of-adaptation study at 879M** — the direct follow-up: 8 / 12 / 22 blocks (or low-rank updates on all blocks)
   with the R3 mixture, layer-wise learning rates and v1 self-distillation on replay; gates from the first run:
   19 text gates, multimodal with re-aligned fusion, JevBench. Cost of the same order as R3 per configuration.
2. **Protect the fused path during text training** (replay of fused media inputs / distillation of the v1 stack) so that
   the image gap never opens; alternatively accept serving multimodal requests on the v1 core.
3. **Broader, less templated capability data** (the diagnostic says 81% on these generators is worth nothing outside
   them): more surface forms, verified LLM-written scenarios with the reviewer loop already in place (99.4% agreement on
   computed labels), true / false propositions that need reasoning.
4. Context extension as its own round, for the two JevBench families that are cut (long_policy, multi_hop) — after 1.
5. Use the R3 core for text only where the technical gain matters and multimodal is not in the path. Not recommended
   as "v1.1": it does not meet the gates that were set for that name.

## 5. Deviations from the registered protocol

Post-registration diagnostics (three throwaway runs; no gate or selection uses them) · second fusion seed not run (the
first failed its gates; the registered fallback ran instead) · no multimodal checkpoint was eligible on `val_fit` (least
regressing taken, as registered) · multimodal dump wrapper fixed after a first failed dump · speech synthesis re-run after
sharing the GPU with training (caught by the media check) · capability share 9.7% answerable-only (band 10–15%), technical
40.3% (band 30–35%) · latency gate applied one-sided · R2 reference on the new files from one seed · technical and
capability data English only; rule tables model-reviewed only · one Ollama-found generator bug fixed before the final
build (`results/v11/r3/quality_review/README.md`).

## 6. Cost

| | estimate (registered before training) | actual |
|---|---|---|
| pod A `krun-v11-r3-a` (RTX 4090 Secure, EU-RO-1, US$ 0.74/h) | | 4.85 h ≈ US$ 3.7 |
| pod B `krun-v11-r3-b` (RTX 4090 Secure, EUR-NO-1, US$ 0.74/h) | | 5.0 h ≈ US$ 3.8 |
| **RunPod total** | 8.5–10 GPU-h, US$ 6.5–7.5 (upper bound US$ 12) | **9.9 GPU-h, ≈ US$ 7.5** (billing for the last hours not fully posted) |
| of which (approx.): text training / evaluation / multimodal / diagnostics / setup | 4.0 / 1.6 / 3.0 (+1.2 fallback) / — / — | ≈ 3.7 / 1.1 / 2.7 (incl. fallback) / 2.0 (not in the estimate, ≈ US$ 1.5) / 0.4 h |
| Ollama cloud credits | < US$ 1 | < US$ 1 (539 reviewed items) |

Both pods terminated; `list-pods` returns no pods; no network volume. Results, logs, run records and the two kept weight
files were downloaded and checksum-verified before termination (`results/v11/r3/pod_logs/`).

Kept locally (run outputs, **not** artifacts): `runs/v11/r3/text/r3r_s1/model_v1.safetensors` (candidate text core, sha
c151c7b0…) and `runs/v11/r3/multimodal/r3r_s1_m20260932_e3/step_003507/m15.safetensors` (re-aligned fusion, sha db8c607a…).
`artifacts/krun-one-v1.1-candidate/` was **not** created.
