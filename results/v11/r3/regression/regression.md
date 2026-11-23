# v1 regression suite (same host, same harness; V eval + ToolACE v0; no protected set opened)

| metric | v1 | r3r_s1 | r3r_s2 |
|---|---:|---:|---:|
| choice_accuracy | 0.9763 | 0.9769 | 0.9766 |
| intent_accuracy | 0.9753 | 0.9756 | 0.9754 |
| tool_accuracy | 0.9808 | 0.9829 | 0.9815 |
| pt_br | 0.9795 | 0.9821 | 0.9812 |
| cross_language | 0.9816 | 0.9814 | 0.9816 |
| hard_negative | 0.9750 | 0.9769 | 0.9759 |
| k128 | 0.9062 | 0.8981 | 0.8954 |
| toolace_holdout | 0.9014 | 0.9038 | 0.8966 |
| toolace_val | 0.9483 | 0.9438 | 0.9438 |
| val_tools_v2_tool_accuracy | 0.9068 | 0.9099 | 0.9076 |
| val_tools_v2_k128 | 0.8446 | 0.8378 | 0.8514 |
| multi_unseen_tool_micro_f1 | 0.9161 | 0.9184 | 0.9159 |
| k64 | 0.9577 | 0.9557 | 0.9598 |
| ood_auroc | 0.9710 | 0.9745 | 0.9744 |
| ood_fa_cov80 | 0.0450 | 0.0293 | 0.0287 |
| tool_unseen_fa_cov70 | 0.0381 | 0.0127 | 0.0169 |
| coverage_at_target | 0.8939 | 0.9150 | 0.9102 |
| aurc | 0.0288 | 0.0274 | 0.0273 |
| permutation | 0.9926 | 0.9902 | 0.9828 |
| score_mae | 0.6000 | 0.6000 | 0.5966 |
| score_spearman | 0.8355 | 0.8352 | 0.8348 |
| noul_auroc | 0.9728 | 0.9729 | 0.9716 |
| multi_micro_f1 | 0.9173 | 0.9196 | 0.9171 |
| ece | 0.0147 | 0.0162 | 0.0150 |

## Preservation gates vs the same-host v1 (value = candidate − v1; pp where the limit is in pp)

| gate | r3r_s1 | r3r_s2 |
|---|---|---|
| krunbench_v1_choice_accuracy | +0.062 ✅ | +0.025 ✅ |
| native_non_en | +0.000 ✅ | -0.070 ✅ |
| pt_br | +0.256 ✅ | +0.171 ✅ |
| cross_language | -0.022 ✅ | +0.000 ✅ |
| intent | +0.030 ✅ | +0.015 ✅ |
| toolace_holdout | +0.240 ✅ | -0.481 ✅ |
| tool_accuracy_val_tools_v2 | +0.309 ✅ | +0.077 ✅ |
| large_k_k128 | -0.804 ✅ | -1.072 ✅ |
| ood_auroc | +0.004 ✅ | +0.003 ✅ |
| ood_fa_cov80 | -1.576 ✅ | -1.632 ✅ |
| tool_unseen_fa_cov70 | +0.013 ✅ | +0.017 ✅ |
| selective_coverage | +2.107 ✅ | +1.634 ✅ |
| aurc | -0.001 ✅ | -0.001 ✅ |
| permutation | -0.245 ✅ | -0.980 ✅ |
| score_mae | -0.000 ✅ | -0.003 ✅ |
| score_spearman | -0.000 ✅ | -0.001 ✅ |
| noul_auroc | +0.000 ✅ | -0.001 ✅ |
| multi_micro_f1 | +0.224 ✅ | -0.020 ✅ |
| ece | +0.001 ✅ | +0.000 ✅ |
| **all** | ✅ | ✅ |

## Text latency (RTX 4090, total p50 / p95 ms; peak VRAM MiB)

| workload | v1 | r3r_s1 | r3r_s2 |
|---|---|---|---|
| autocast/k_128_tools_ctx64 | 55.5 / 55.9 (3523) | 55.3 / 55.9 (3523) | 55.2 / 55.6 (3523) |
| autocast/k_32_tools_ctx64 | 19.8 / 20.4 (3424) | 19.0 / 19.5 (3424) | 19.8 / 20.2 (3424) |
| autocast/k_8_tools_ctx64 | 17.8 / 18.3 (3424) | 17.0 / 17.8 (3424) | 18.0 / 20.5 (3424) |
| autocast/typical_1q_k4_ctx48 | 17.8 / 18.5 (3380) | 17.2 / 17.5 (3380) | 17.8 / 19.2 (3380) |
| autocast/typical_1q_tools8_ctx48 | 17.8 / 18.3 (3386) | 17.3 / 18.0 (3386) | 18.4 / 19.6 (3386) |
| autocast/typical_4q_mixed_ctx96 | 21.7 / 22.1 (3424) | 21.6 / 21.8 (3424) | 21.6 / 23.2 (3424) |
| bf16/k_128_tools_ctx64 | 44.6 / 45.0 (1989) | 44.3 / 44.6 (1989) | 44.4 / 44.9 (1989) |
| bf16/k_32_tools_ctx64 | 18.0 / 18.5 (1738) | 17.3 / 17.8 (1738) | 17.8 / 19.3 (1738) |
| bf16/k_8_tools_ctx64 | 16.3 / 16.7 (1737) | 15.4 / 15.8 (1737) | 15.8 / 17.2 (1737) |
| bf16/typical_1q_k4_ctx48 | 15.7 / 16.7 (1689) | 15.0 / 15.8 (1689) | 15.6 / 16.0 (1689) |
| bf16/typical_1q_tools8_ctx48 | 16.2 / 16.6 (1696) | 15.2 / 15.6 (1696) | 16.1 / 17.5 (1696) |
| bf16/typical_4q_mixed_ctx96 | 18.1 / 18.6 (1737) | 16.9 / 17.5 (1737) | 17.3 / 18.6 (1737) |
