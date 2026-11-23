# Continued MLM gate: 0B vs 1B (rule: configs/v1/mlm_gate.yaml, registered before training)

| language | MLM loss 0B | MLM loss 1B | relative gain | perplexity 0B → 1B |
|---|---|---|---|---|
| en | 1.391 | 1.365 | +1.8% | 4.0 → 3.9 |
| pt | 1.106 | 1.029 | +7.0% | 3.0 → 2.8 |
| es | 1.085 | 1.027 | +5.4% | 3.0 → 2.8 |
| fr | 0.983 | 0.924 | +6.0% | 2.7 → 2.5 |

| decision probe (val_eval, mean of 2 seeds) | 0B | 1B | Δ |
|---|---|---|---|
| **aggregate score** | 88.53 | 89.20 | +0.67 |
| choice intent acc | 92.0 | 92.4 | +0.3 |
| tool acc | 91.3 | 93.0 | +1.7 |
| noul ROC-AUC | 0.927 | 0.927 | +0.000 |
| score Spearman | 0.734 | 0.736 | +0.002 |
| score MAE | 0.669 | 0.668 | -0.002 |
| multi micro-F1 | 83.9 | 84.4 | +0.5 |
| cross-language acc | 91.7 | 92.2 | +0.5 |
| unseen-tool acc | 91.4 | 93.0 | +1.6 |
| hard-negative acc | 91.9 | 92.4 | +0.5 |
| EN acc | 92.3 | 92.9 | +0.5 |
| PT acc | 89.0 | 91.9 | +2.9 |
| ES acc | 93.7 | 92.6 | -1.1 |
| FR acc | 92.2 | 91.9 | -0.3 |
| OOD AUROC (margin) | 0.951 | 0.953 | +0.002 |
| macro OOD FA @0.98 | 3.8 | 7.4 | +3.5 |
| coverage @0.98 | 70.0 | 71.4 | +1.4 |
| ECE | 0.016 | 0.019 | +0.003 |
| NLL | 0.277 | 0.283 | +0.006 |
| permutation consistency | 95.1 | 97.0 | +1.8 |

| gate | pass |
|---|---|
| G1_mlm | ❌ |
| G2_decision | ❌ |
| G3_multilingual | ❌ |
| G4_calibration_ood | ❌ |
| G5_tool_crosslang_permutation | ✅ |

**Decision: STOP_AT_1B** (aggregate Δ per seed: +1.26, +0.08)
