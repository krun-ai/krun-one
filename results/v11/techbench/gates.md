# Technical promotion gates — KrunTechBench v1 (protected), raw rendering

Baseline v1: macro top-1 30.9, gold > hardest 66.9, margin +1.12, flips both 2.7 / stuck 12.5, unseen combination 52.1.

## R1

| gate | r1_s1 | r1_s2 | mean | holds for every seed |
|---|---:|---:|---:|---|
| top1_macro_gain_pp | +21.44 ✅ | +22.83 ✅ | +22.13 | ✅ |
| pairwise_gold_vs_hardest_gain_pp | +8.31 ✅ | +9.38 ✅ | +8.85 | ✅ |
| hard_negative_margin_gain_logit | +1.58 ✅ | +1.68 ✅ | +1.63 | ✅ |
| constraint_flip_both_gain_pp | +22.97 ✅ | +24.32 ✅ | +23.65 | ✅ |
| flip_stuck_does_not_rise_pp | +8.78 ❌ | +10.81 ❌ | +9.80 | ❌ |
| unseen_combination_gain_pp | -0.20 ❌ | +2.73 ❌ | +1.26 | ❌ |
| worst_track_delta_pp | +6.70 ✅ | +6.07 ✅ | +6.39 | ✅ |
| ood_false_acceptance | +0.00 ✅ | +0.00 ✅ | +0.00 | ✅ |
| **all technical gates** | | | | ❌ |

## R2_last4

| gate | r2l4_s1 | r2l4_s2 | mean | holds for every seed |
|---|---:|---:|---:|---|
| top1_macro_gain_pp | +31.88 ✅ | +32.51 ✅ | +32.20 | ✅ |
| pairwise_gold_vs_hardest_gain_pp | +12.89 ✅ | +13.24 ✅ | +13.07 | ✅ |
| hard_negative_margin_gain_logit | +3.08 ✅ | +3.00 ✅ | +3.04 | ✅ |
| constraint_flip_both_gain_pp | +43.24 ✅ | +45.27 ✅ | +44.26 | ✅ |
| flip_stuck_does_not_rise_pp | +3.72 ❌ | +1.01 ❌ | +2.36 | ❌ |
| unseen_combination_gain_pp | +6.46 ❌ | +6.97 ❌ | +6.72 | ❌ |
| worst_track_delta_pp | +6.06 ✅ | +9.19 ✅ | +7.62 | ✅ |
| ood_false_acceptance | +0.01 ✅ | +0.02 ✅ | +0.01 | ✅ |
| **all technical gates** | | | | ❌ |

## R2_last4_noadapter

| gate | r2l4na_s1 | r2l4na_s2 | mean | holds for every seed |
|---|---:|---:|---:|---|
| top1_macro_gain_pp | +30.79 ✅ | +32.80 ✅ | +31.80 | ✅ |
| pairwise_gold_vs_hardest_gain_pp | +12.52 ✅ | +12.72 ✅ | +12.62 | ✅ |
| hard_negative_margin_gain_logit | +3.09 ✅ | +3.24 ✅ | +3.17 | ✅ |
| constraint_flip_both_gain_pp | +42.91 ✅ | +47.64 ✅ | +45.27 | ✅ |
| flip_stuck_does_not_rise_pp | +4.05 ❌ | +1.35 ❌ | +2.70 | ❌ |
| unseen_combination_gain_pp | +7.78 ❌ | +10.30 ✅ | +9.04 | ❌ |
| worst_track_delta_pp | +2.42 ✅ | +6.06 ✅ | +4.24 | ✅ |
| ood_false_acceptance | +0.00 ✅ | +0.00 ✅ | +0.00 | ✅ |
| **all technical gates** | | | | ❌ |

