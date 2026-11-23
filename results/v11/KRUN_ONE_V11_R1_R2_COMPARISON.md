# Krun One v1.1 — R1 vs R2: comparison and decision report

Date: 2026-10-01 · Protocol and gates registered before training: `configs/v11/r1_r2_protocol.yaml`,
`configs/v11/promotion_gates.yaml` · Details: `KRUN_ONE_V11_R1_REPORT.md`, `KRUN_ONE_V11_R2_REPORT.md`.

State at the end of the round: R1 measured (2 seeds) · R2 measured (6 variants, 11 runs) · KrunTechBench v1 built,
frozen and measured · Tetris diagnosis complete · full v1 regression (text + multimodal) complete · cost reported ·
both pods terminated · **no 1.5B training · no production deploy · no artifact created · frozen v1 untouched.**

## Summary

| | v1 | R1 (encoder frozen) | R2 last 4 + adapter | R2 last 4, v1 architecture |
|---|---:|---:|---:|---:|
| KrunTechBench v1, raw macro top-1 | 30.9 | 53.0 ± 0.7 | 63.1 ± 0.3 | 62.7 ± 1.0 |
| … with explicit consequences ("feat") | 45.3 | 77.2 | 84.3 | 84.0 |
| gold > hardest negative | 66.9 | 75.8 | 80.0 | 79.5 |
| hard-negative margin (logit) | +1.12 | +2.75 | +4.16 | +4.29 |
| constraint flips, both correct | 2.7 | 26.4 | 47.0 | 48.0 |
| unseen requirement combinations | 52.1 | 53.4 | 58.9 | 61.2 |
| unseen technologies | 39.0 | 65.8 | 76.0 | 74.5 |
| technical gates passed (of 8) | — | 6 | 6 | 6 |
| text preservation gates (of 19), every seed | — | 19 | 19 | 19 |
| multimodal M1.5c gates with frozen fusion (of 5) | 5 | 0 | 0 | 0 |
| latency / VRAM vs v1 | — | same | same | same |
| served parameters | 879,356,931 | 880,162,060 | 880,162,060 | 879,356,931 |

**Nothing is promotable today.** The direction is clear, though: adapting the encoder is what works, at 879M, with no
cost in latency and no loss in v1's text behaviour. Two things stand between R2 and a promotion: the multimodal path
(needs the fusion re-fitted to the new core, or a different way of protecting it) and generalisation to unseen
requirement combinations.

## The ten questions

**1. R1 vs R2 — how much does unfreezing the encoder help?**
A lot, and consistently. Same corpus, same splits, same objective, same seeds:

| | R1 | R2 (last 2) | R2 (last 4) | R2 − R1 |
|---|---:|---:|---:|---|
| KrunTechVal raw macro | 37.7 ± 0.95 | 47.6 ± 0.84 | 52.0 ± 0.99 | +9.9 pp (2 blocks), +14.3 pp (4 blocks) |
| KrunTechBench raw macro (protected) | 53.0 | — | 63.1 | **+10.1 pp** (paired vs v1: +18.1 → +27.3) |
| constraint flips, both correct | 26.4 | — | 47.0 | +20.6 pp |
| hard-negative margin | +2.75 | — | +4.16 | +1.4 logit |
| large candidate sets | 39.1 | — | 50.8 | +11.7 pp |
| technical OOD AUROC | 0.650 | — | 0.816 | +0.17 |

Of the +32 pp that R2 gains over v1 on the protected benchmark, about two thirds (+22 pp) are reachable with the encoder
frozen and one third (+10 pp) needs the encoder; on the hardest properties (flips, separation, large k, OOD) the
encoder's share is about half. Going from 2 to 4 blocks added +4.4 pp on validation, with no preservation cost.
In the registered decision tree this is **"R2 ≫ R1 → encoder adaptation is key → continue v1.1 around 879M"**.

**2. Did technical decision accuracy improve?** Yes: 30.9 → 53.0 (R1) → 63.1 / 62.7 (R2), every track up, paired CI
far from zero (+27 pp [+25.5, +29.0]). Strongest: distributed systems 41 → 94, devops 4 → 88–90, backend 26 → 64–77,
database 28 → 68–70. Weakest: state / action 9 → 18–19, cloud 25 → 45–51, security 48 → 50–58.

**3. Unseen combinations?** Only partly: 52.1 → 53.4 (R1), 58.9 (R2 + adapter), 61.2 (R2, v1 architecture). The
registered gate (+10 pp) **fails** (one of four R2 seeds reaches +10.3). Seen requirement sets are at 100%, so the gap
between seen and unseen is 40 pp: the model still memorises requirement-set → technology mappings. In two rule tables
(network component, security) it is worse than v1 on unseen sets. Unseen *technologies* generalise better (39 → 75–76),
unseen simulator *families* not at all (≈ chance).

**4. Constraint flips?** Large improvement: both members of a pair correct 2.7% → 26.4% (R1) → 47–48% (R2); with
explicit consequences 11% → 94–95% (R2). The companion gate "stuck on the base answer does not rise" **fails as
registered** (12.5% → 22.3% R1, 14.9–15.2% R2) because the rate was registered unconditionally; conditional on getting
the base member right it falls from 84% to 45% (R1) and 20–26% (R2).

**5. Hard-negative separation?** Yes: gold above the hardest negative 66.9% → 75.8% (R1) → 79.5–80.0% (R2); logit
margin +1.12 → +2.75 → +4.2–4.3; median margin on the hard-negative slice +0.8 → +2.0 → +3.7–4.0. The pairwise /
margin loss terms were not what produced this (the pilot found no benefit from them); the data did. The hard-negative
mini-round did not add to it.

**6. Tetris: raw vs structured vs consequence?** Top-1 at k = 8 on held-out states (chance 12.5): raw board ~17–19
after R2 and 30.5 after a dedicated full fine-tune; structured state features 17–20 and 35.8 dedicated; explicit
consequence of each placement **100** (R2, and a dedicated run in 1,000 steps). **C ≫ B > A**: the model compares
consequences perfectly and cannot compute them from the board in one pass; describing the current state better barely
helps. → "representation is the bottleneck", on the action-consequence side. Not "all similarly poor", so the tree
does not reopen growth.

**7. Did v1 regress?**
*Text: no.* 19 / 19 preservation gates on all six evaluated seeds (multilingual, PT-BR, cross-language, intent, tools,
unseen tools as registered, large-k, OOD AUROC / false acceptance, selective coverage / AURC, permutation, Score, Noul,
Multi, ECE). The largest negative movement is k = 128: −0.3 to −1.6 pp (limit −2.0). Selective behaviour improves.
*Multimodal: yes, severely, when the candidate core is put under the frozen M1.5c fusion weights:*

| text core under the frozen M1.5c fusion | doc visual-only (G1) | layout (G1) | screenshot (G2) | text-dep. doc (G2) | real speech (G3) | MINDS intent (G3) | image (G4) | synthdoc (G4b) | DocLayNet (G4b) | CORD (G4b) | gates passed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| *registered threshold* | ≥ 67.9 | ≥ P + 5 | ≥ 97.9 | ≥ P − 1 | ≥ 84.1 | ≥ 89.0 | ≥ 63.3 | ≥ 95.3 | ≥ 93.5 | ≥ 95.0 | |
| M1.5c as registered (production host) | 68.0 | 38.7 | 99.2 | 67.2 | 85.1 | 89.7 | 64.8 | 96.1 | 94.5 | 100.0 | 5 / 5 |
| v1 core, this host | 68.7 | 40.1 | 99.2 | 67.1 | 85.1 | 89.7 | 64.8 | 96.2 | 94.5 | 100.0 | 5 / 5 |
| control: v1 weights in the v1.1 model class (untrained adapter) | 68.7 | 40.1 | 99.2 | 67.1 | 85.1 | 89.7 | 64.8 | 96.2 | 94.5 | 100.0 | 5 / 5 |
| R1 (seed 1) | 50.8 | 6.5 | 96.1 | 66.7 | 81.0 | 87.2 | 56.3 | 80.6 | 81.5 | 81.9 | 0 / 5 |
| R2 last 4 + adapter (seed 1) | 56.0 | 18.6 | 97.6 | 67.3 | 82.4 | 89.3 | 47.3 | 80.3 | 88.8 | 98.6 | 0 / 5 |
| R2 last 4, no adapter (seed 1) | 57.5 | 17.1 | 97.4 | 66.6 | 81.5 | 87.9 | 46.5 | 78.7 | 88.0 | 93.1 | 0 / 5 |

Text-dependent media (screenshots, speech with usable ASR) lose 0–4 pp; media that must be read through the fusion
adapters (visual-only documents, layout, images, hybrid documents) lose up to 34 pp (typically 6–23 pp). The fusion adapters and the head LoRA
(26.7M parameters) were trained against v1's head and last blocks, which every candidate changes — including R1, where
the encoder is untouched and only the head moved. A control run — the frozen v1 weights loaded through exactly the same v1.1 code path as the candidates (v1.1 model class, adapter wrapper, untrained zero-initialised adapter) — reproduces the v1 core to the last digit on all ten numbers, and the v1 core on this host reproduces the registered M1.5c result (differences ≤ 1.4 pp, all five gates pass). The loss is a property of the candidates, not of the harness.

**8. Latency / VRAM?** No impact. All candidates within ±0.2 ms of v1 on every workload (typical 16.2–16.5 ms, 4 questions
21.3, k = 32 18.0–18.5, k = 128 54.7–55.0; bf16 14.4–14.6 and 44.0–44.2), VRAM 3,523–3,526 MiB (bf16 1,989–1,990). The
adapter adds 3 MiB.

**9. Which candidate to promote?** **None now.** Ranking for the owner's decision:

1. *R2 last 4, no adapter* — the line to continue. Same accuracy as with the adapter (62.7 ± 1.0 vs 63.1 ± 0.3; the
   validation tie-break registered in the protocol prefers it), identical served architecture and parameter count to
   v1, loadable by the unchanged runtime, best unseen-combination numbers. Added after registration, so its standing is
   that of a confirmed control, not of the registered selection.
2. *R2 last 4 + adapter* — the registered selection; no advantage over (1) and a runtime change.
3. *R1* — safe for text but 10 pp weaker, and it breaks multimodal just the same; no reason to prefer it.

What blocks promotion of (1): the multimodal gates (hard block for any product that serves images / documents / audio
through the same core) and the unseen-combination gate.

**10. Any evidence that more than 879M is needed?** No. Everything measured in this round points away from capacity:
the gain came from unfreezing 9% of the weights for a single pass; the two remaining failures are a data / objective
problem (memorised requirement sets) and a representation problem (consequences not computed) that the *same* model
solves perfectly when the consequence is in the text; Tetris with consequences is 100%; R0 had already shown that
143M, 318M and 879M reach the same accuracy on the environment probe. No larger model was trained or evaluated in this
round (by instruction), so this is absence of evidence for growth, consistent with R0 — not a new direct test.

## Decision tree outcome

* R1 ≈ R2? **No** → not a data-only problem.
* R2 ≫ R1? **Yes** → encoder adaptation is key; continue v1.1 around 879M.
* Tetris B / C ≫ A? **C ≫ A (B slightly > A)** → representation bottleneck on action consequences.
* Tetris A / B / C similarly poor? **No** → architecture / growth is not reopened.

## What a next round would need (not started; the owner decides)

1. **Protect the multimodal path.** Options to evaluate: re-fit the M1.5c fusion adapters on the new core (they are
   26.7M parameters; the owner excluded multimodal retraining from *this* round); or constrain text training so the
   function the fusion relies on is preserved (e.g. replaying fused media inputs / distilling v1's outputs on them while
   training); or keep serving multimodal requests on the v1 core. The multimodal gate must be part of variant selection
   from the start, not only of the final regression.
2. **Unseen combinations.** More requirement sets per table and larger held-out sets, property facts stated in the
   context rather than memorisable from names, more tables with overlapping technologies; the gate as registered is a
   good target.
3. **Consequences.** Supervision that makes the encoder compute action outcomes (per-candidate consequence prediction
   as a real training target rather than an auxiliary scalar, intermediate "what happens if" decisions, multi-step
   states), measured by the raw-vs-feat gap, which is the cleanest indicator this round produced (63 vs 84).
4. Corpus fixes found here: label the `swe_involved` items; multilingual technical examples; human review of the rule
   tables; reconsider the "stuck" gate definition (conditional rate).
5. Optional (owner's Ollama resource, bake-off in `results/v11/teacher_bakeoff/report.md`): gpt-oss:120b + glm-5.3
   agreement (97% correct where they agree, 77% of items) as a *disagreement miner / reviewer* for LLM-written
   scenarios; no teacher was reliable on state / action or none-of-the-above, so simulators remain the ground truth
   there. Not used for training data in this round.

## Other measurements

* **JevBench public items** (231, evaluation only): Jev 1.13.0 86.6 · Laya 58.4 · Krun v1 53.2 · R1 55.0 · R2 52.4.
  No transfer from technical training; 40% of the hard items exceed 1,024 tokens. Useful as an external yardstick for
  long-policy / multi-hop / temporal items, not as training data.
* **Loss terms:** O0–O4 within noise on the pilot; O2 used by the registered rule. No evidence yet that pairwise,
  margin or value terms help at the registered weights; self-distillation on replay gave the smallest v1 drift.
* **Head variants:** adapter and candidate-interaction block give nothing (as R0 predicted).

## Deviations (all detailed in the R2 report, section 9)

Post-registration variant (last 4 without adapter) · hard-negative round over-sampled Tetris · `swe_involved` items
unlabeled (2% of training, 120 benchmark items unscored) · Tetris diagnostic step count · hard-negative share 19% vs
10% · multimodal media rebuild incident on the pod · benchmark English-only and model-reviewed · cost above the estimate.

## Cost

| item | estimate (registered before training) | actual |
|---|---|---|
| pod A `krun-v11-r12-a` (RTX 4090 Secure, US-IL-1, US$ 0.74/h) | | 10.03 h → ≈ US$ 7.6 (GPU 7.42 + disk ≈ 0.2; RunPod billing for the last hour not yet posted) |
| pod B `krun-v11-r12-b` (RTX 4090 Secure, EUR-NO-1, US$ 0.74/h) | | 9.36 h → US$ 6.91 billed (GPU 6.72 + disk 0.19) |
| **RunPod total, R1 / R2 round** | ~12.5 GPU-h, **US$ 9–10** (upper bound US$ 14) | **19.4 GPU-h, ≈ US$ 14.5** |
| Ollama cloud credits (teacher bake-off, not used for training data) | — | ≈ US$ 3.8 |

The round cost about 50% more than the estimate and slightly more than the stated upper bound. Where the extra ~7 GPU-hours
went: pods idle or lightly used while the other one finished (two pods were kept for the whole day instead of being
released between phases); the conditional last-4 runs plus the post-registration last-4-no-adapter runs (≈ 3 h); the
Tetris diagnostic rerun to 3,000 steps (≈ 1.2 h); the multimodal regression having to be run twice after the media
rebuild incident, plus the control run (≈ 1.5 h); slow laptop → US-IL uploads at the start.

Both pods are terminated (`list-pods` returns no pods). No network volume was created. Results, logs and run records
were downloaded and checksum-verified before termination (`results/v11/pod_logs/*/…sha256.txt`).

