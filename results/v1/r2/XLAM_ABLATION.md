# xLAM ablation probe (R2)

Two identical short fine-tunes (1 epoch over 210k decisions: 150k new tool decisions + the same 60k base replay)
from the v1-lr1e5 checkpoint, encoder LR 1e-5 / head 5e-5, same seed and order. Only difference: the 150k tool
decisions are sampled from ALL Tool Corpus v2 decisions (64k xLAM-sourced) vs from the non-xLAM ones (0 xLAM).
A probe, not two full trainings: it answers *direction and size* of the xLAM effect under a fixed budget.

| slice (with − without) | n | Δ accuracy (pp) | 95% CI |
|---|---:|---:|---|
| toolace_holdout | 416 | +0.00 | [-2.25, +2.36] |
| toolace_holdout_k>=16 | 140 | -0.71 | [-5.71, +4.29] |
| toolace_holdout_native+hard | 272 | -1.10 | [-4.35, +2.47] |
| V_tool | 1461 | -0.21 | [-0.63, +0.21] |
| V_near_duplicate | 2016 | +0.50 | [-0.51, +1.59] |
| V_k>=32 | 1809 | +1.22 | [+0.17, +2.32] |
| V_cross_language | 4453 | +0.13 | [-0.29, +0.54] |

ToolACE accuracy without / with xLAM: holdout 90.1 / 90.1; val_tools 93.0 / 90.8

| tool coverage | OOD FA without / with | unseen-tool FA without / with |
|---:|---|---|
| 50% | 0.0 / 0.0 | 0.0 / 0.0 |
| 60% | 0.0 / 0.0 | 0.0 / 0.0 |
| 70% | 0.0 / 0.0 | 0.0 / 0.0 |
| 80% | 0.0 / 0.0 | 0.0 / 0.0 |
| 90% | 6.0 / 5.1 | 7.2 / 6.4 |

Tool AURC without / with: 0.0261 / 0.0257

## Reading

1. **Given the rest of Tool Corpus v2, xLAM adds ~nothing measurable.** The ToolACE holdout is identical (90.1 / 90.1,
   CI ±2.3 pp). Tool accuracy on V, near-duplicates, cross-language and OOD at every coverage point are within noise.
   The only significant effect is **+1.2 pp on V decisions with k ≥ 32**, CI [+0.2, +2.3]. val_tools is 2.2 pp lower
   with xLAM; that slice is not a gate and has no CI here. This matches the corpus analysis: after protection, xLAM
   contributes 84 exclusive tool names. Its 122k decisions add query volume, not API diversity.
2. **Tool Corpus v2 itself works.** Both short probes (~6k steps from v1-lr1e5) reach **90.1% on the ToolACE holdout**,
   against 86.1% for v1-lr1e5 and 91.8% for v0.3. Tool unseen false acceptance is 0 up to 80% coverage.
3. Practical consequence for the owner: if the xLAM license risk ever needs to be removed, a Tool Corpus v2 without xLAM
   costs ~1 pp on large candidate sets and nothing on the clean unseen-tool benchmark (probe-level evidence).

Cost: pod `y15ix0fk8jduny` (RTX 4090 Secure, $0.74/h), 12:32–13:00 BRT … ≈ 3.5 h ≈ $2.6 (billing in the final report).
