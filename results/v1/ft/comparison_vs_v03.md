# Krun One v1 candidate vs Krun One v0.3 — KrunBench v1 V (eval part)

Both models: same V decisions, same signal/metric code (`kd1_evaluate.py`); temperatures and the margin threshold fitted on the FIT part for each model separately. v0.3 has no multi-label output (n/a).

| metric | v0.3 | v1 | Δ |
|---|---:|---:|---:|
| Choice (intent) accuracy | 89.1 | 97.6 | +8.4 |
| Tool accuracy | 81.2 | 98.5 | +17.2 |
| Hard-negative accuracy | 87.7 | 97.8 | +10.1 |
| Near-duplicate accuracy | 86.5 | 96.2 | +9.7 |
| Cross-language accuracy | 82.7 | 98.3 | +15.6 |
| Unseen-tool false acceptance ↓ | 0.0 | 21.6 | +21.6 |
| OOD AUROC (margin) | 0.875 | 0.975 | +0.100 |
| OOD false acceptance ↓ | 3.3 | 6.8 | +3.4 |
| AURC ↓ | 0.073 | 0.027 | -0.046 |
| ECE ↓ (choice, after T) | 0.034 | 0.031 | -0.003 |
| NLL ↓ (choice) | 0.501 | 0.140 | -0.361 |
| Coverage @ sel. acc 0.95 | 60.9 | 97.8 | +36.9 |
| Coverage @ sel. acc 0.98 | 51.7 | 90.5 | +38.9 |
| Coverage @ sel. acc 0.99 | 42.6 | 78.2 | +35.6 |
| Permutation consistency | 86.0 | 97.8 | +11.8 |
| Cross-language consistency | 74.8 | 97.9 | +23.1 |
| Noul ROC-AUC | 0.973 | 0.960 | -0.013 |
| Noul accuracy | 89.0 | 85.1 | -3.9 |
| Score MAE ↓ | 0.810 | 0.631 | -0.179 |
| Score Spearman | 0.744 | 0.843 | +0.100 |
| Score NLL ↓ | 1.307 | 1.092 | -0.215 |
| Multi micro-F1 | — | 90.8 | — |
| Multi macro-F1 | — | 82.2 | — |
| Multi empty-set accuracy | — | — | — |
| Multi unseen-tool micro-F1 | — | 90.7 | — |

## By language (choice accuracy; OOD false acceptance)

| language | n | v0.3 acc | v1 acc | v0.3 intent | v1 intent | v0.3 tool | v1 tool | v0.3 OOD FA | v1 OOD FA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EN | 3566 | 93.9 | 97.1 | 95.8 | 96.8 | 85.2 | 98.3 | 5.8 | 7.2 |
| PT-BR | 1170 | 74.8 | 98.4 | 75.9 | 98.4 | 68.7 | 98.3 | 1.8 | 7.9 |
| PT-PT | 676 | 94.1 | 99.4 | 96.9 | 99.8 | 87.2 | 98.5 | 0.0 | 0.0 |
| ES | 1345 | 85.4 | 98.2 | 85.8 | 98.0 | 83.3 | 99.1 | 0.6 | 6.4 |
| FR | 1262 | 81.2 | 97.7 | 83.1 | 97.5 | 72.0 | 98.6 | 2.4 | 7.5 |

## Paired deltas (v1 − v0.3, accuracy on shared answerable choice decisions; group bootstrap 95% CI)

| slice | n | Δ (pp) | 95% CI |
|---|---:|---:|---|
| all_choice | 8019 | +10.05 | [+9.13, +10.98] |
| intent | 6558 | +8.45 | [+7.39, +9.42] |
| tool | 1461 | +17.25 | [+15.39, +19.35] |
| cross_language | 4453 | +15.56 | [+14.27, +16.99] |
| hard | 4887 | +10.11 | [+8.99, +11.31] |
| non_en | 4453 | +15.56 | [+14.27, +16.99] |
| PT-BR | 1170 | +23.59 | [+21.20, +26.07] |
| PT-PT | 676 | +5.33 | [+3.55, +6.95] |
| ES | 1345 | +12.86 | [+10.86, +14.80] |
| FR | 1262 | +16.48 | [+14.34, +18.54] |
| EN | 3566 | +3.17 | [+2.32, +4.06] |

## By candidate count (choice accuracy)

| k ≤ | n | v0.3 | v1 |
|---:|---:|---:|---:|
| 2 | 1107 | 96.7 | 98.3 |
| 4 | 1545 | 93.0 | 97.9 |
| 8 | 1822 | 90.5 | 98.0 |
| 16 | 1726 | 86.1 | 97.8 |
| 32 | 949 | 82.4 | 98.9 |
| 64 | 497 | 73.6 | 95.4 |
| 128 | 373 | 64.9 | 94.1 |

## OOD false acceptance by bucket (at each model's own threshold)

| bucket | n | v0.3 | v1 |
|---|---:|---:|---:|
| intent/gold_removed | 1356 | 4.1 | 4.9 |
| intent/native_irrelevance | 89 | 4.5 | 0.0 |
| tool/gold_removed | 33 | 0.0 | 0.0 |
| tool/native_irrelevance | 299 | 0.0 | 17.7 |

## Latency (model part of the worker; same harness for both)

| workload (RTX 4090, bf16, total p50 ms) | locale | v0.3 | v1 (K1-W1536-L22) | ratio |
|---|---|---:|---:|---:|
| typical_1q_k4_ctx48 | en-US | 17.5 | 14.9 | 0.85× |
| typical_1q_k4_ctx48 | pt-PT | 17.3 | 14.9 | 0.86× |
| typical_1q_tools8_ctx48 | en-US | 17.6 | 14.9 | 0.85× |
| typical_1q_tools8_ctx48 | pt-PT | 18.2 | 14.9 | 0.82× |
| typical_4q_mixed_ctx96 | en-US | 20.0 | 16.6 | 0.83× |
| typical_4q_mixed_ctx96 | pt-PT | 19.9 | 16.6 | 0.84× |
| k_32_tools_ctx64 | en-US | 19.1 | 16.6 | 0.87× |
| k_32_tools_ctx64 | pt-PT | 19.5 | 16.3 | 0.84× |
| k_64_tools_ctx64 | en-US | 24.2 | 27.3 | 1.13× |
| k_64_tools_ctx64 | pt-PT | 23.9 | 27.3 | 1.14× |
| k_128_tools_ctx64 | en-US | 46.1 | 41.6 | 0.90× |
| k_128_tools_ctx64 | pt-PT | 46.1 | 41.6 | 0.90× |
| stress_16q_tools32_ctx512 | en-US | 194.5 | 230.4 | 1.18× |
| stress_16q_tools32_ctx512 | pt-PT | 194.3 | 230.3 | 1.18× |

## Registered gates (configs/v1/ft_gates.yaml)

| gate | pass |
|---|---|
| gain:non_en | ✅ |
| gain:tool | ✅ |
| gain:cross_language | ✅ |
| gain:hard | ✅ |
| unseen_tool:false_acceptance | ❌ |
| unseen_tool:multi_f1 | ✅ |
| ood_auroc | ✅ |
| ood_false_acceptance | ❌ |
| calibration_ece | ✅ |
| choice_core_en_intent | ✅ |
| latency_typical_ms | ✅ |
| latency_typical_ratio | ✅ |

**Verdict: CANDIDATE_REJECTED** (failing: unseen_tool:false_acceptance, ood_false_acceptance)
