# v1 regression suite (same host, same harness; V eval + ToolACE v0; no protected set opened)

| metric | v1 | r1_s1 | r1_s2 |
|---|---:|---:|---:|
| choice_accuracy | 0.9763 | 0.9739 | 0.9759 |
| intent_accuracy | 0.9753 | 0.9721 | 0.9745 |
| tool_accuracy | 0.9808 | 0.9822 | 0.9822 |
| pt_br | 0.9795 | 0.9769 | 0.9778 |
| cross_language | 0.9816 | 0.9789 | 0.9805 |
| hard_negative | 0.9750 | 0.9726 | 0.9754 |
| k128 | 0.9062 | 0.9088 | 0.9115 |
| toolace_holdout | 0.9014 | 0.9087 | 0.8990 |
| toolace_val | 0.9483 | 0.9461 | 0.9438 |
| val_tools_v2_tool_accuracy | 0.9068 | 0.9062 | 0.9068 |
| val_tools_v2_k128 | 0.8446 | 0.8446 | 0.8446 |
| multi_unseen_tool_micro_f1 | 0.9161 | 0.9122 | 0.9143 |
| k64 | 0.9577 | 0.9618 | 0.9618 |
| ood_auroc | 0.9710 | 0.9711 | 0.9711 |
| ood_fa_cov80 | 0.0450 | 0.0377 | 0.0360 |
| tool_unseen_fa_cov70 | 0.0381 | 0.0297 | 0.0339 |
| coverage_at_target | 0.8939 | 0.8992 | 0.9021 |
| aurc | 0.0288 | 0.0293 | 0.0287 |
| permutation | 0.9926 | 0.9828 | 0.9902 |
| score_mae | 0.6000 | 0.5994 | 0.6020 |
| score_spearman | 0.8355 | 0.8353 | 0.8344 |
| noul_auroc | 0.9728 | 0.9723 | 0.9721 |
| multi_micro_f1 | 0.9173 | 0.9134 | 0.9155 |
| ece | 0.0147 | 0.0150 | 0.0168 |

## Preservation gates vs the same-host v1 (value = candidate − v1; pp where the limit is in pp)

| gate | r1_s1 | r1_s2 |
|---|---|---|
| krunbench_v1_choice_accuracy | -0.237 ✅ | -0.037 ✅ |
| native_non_en | -0.070 ✅ | -0.070 ✅ |
| pt_br | -0.256 ✅ | -0.171 ✅ |
| cross_language | -0.269 ✅ | -0.112 ✅ |
| intent | -0.320 ✅ | -0.076 ✅ |
| toolace_holdout | +0.721 ✅ | -0.240 ✅ |
| tool_accuracy_val_tools_v2 | -0.066 ✅ | +0.000 ✅ |
| large_k_k128 | +0.268 ✅ | +0.536 ✅ |
| ood_auroc | +0.000 ✅ | +0.000 ✅ |
| ood_fa_cov80 | -0.732 ✅ | -0.900 ✅ |
| tool_unseen_fa_cov70 | +0.030 ✅ | +0.034 ✅ |
| selective_coverage | +0.536 ✅ | +0.823 ✅ |
| aurc | +0.001 ✅ | -0.000 ✅ |
| permutation | -0.980 ✅ | -0.245 ✅ |
| score_mae | -0.001 ✅ | +0.002 ✅ |
| score_spearman | -0.000 ✅ | -0.001 ✅ |
| noul_auroc | -0.000 ✅ | -0.001 ✅ |
| multi_micro_f1 | -0.390 ✅ | -0.179 ✅ |
| ece | +0.000 ✅ | +0.002 ✅ |
| **all** | ✅ | ✅ |

## Text latency (RTX 4090, total p50 / p95 ms; peak VRAM MiB)

| workload | v1 | r1_s1 | r1_s2 |
|---|---|---|---|
| autocast/k_128_tools_ctx64 | 54.9 / 55.1 (3523) | 54.8 / 55.9 (3526) | 55.0 / 55.3 (3526) |
| autocast/k_32_tools_ctx64 | 18.6 / 22.3 (3424) | 18.5 / 18.6 (3427) | 18.4 / 20.7 (3427) |
| autocast/k_8_tools_ctx64 | 16.4 / 16.6 (3424) | 16.6 / 16.7 (3427) | 16.4 / 17.6 (3427) |
| autocast/typical_1q_k4_ctx48 | 16.4 / 16.5 (3380) | 16.5 / 16.6 (3383) | 16.1 / 18.0 (3383) |
| autocast/typical_1q_tools8_ctx48 | 16.4 / 16.5 (3386) | 16.5 / 16.6 (3389) | 16.3 / 16.5 (3389) |
| autocast/typical_4q_mixed_ctx96 | 21.3 / 21.3 (3424) | 21.3 / 21.4 (3427) | 21.4 / 21.4 (3427) |
| bf16/k_128_tools_ctx64 | 44.1 / 44.2 (1989) | 44.1 / 44.2 (1990) | 44.2 / 44.6 (1990) |
| bf16/k_32_tools_ctx64 | 16.5 / 16.6 (1738) | 16.5 / 18.5 (1739) | 16.5 / 16.6 (1739) |
| bf16/k_8_tools_ctx64 | 14.6 / 14.8 (1737) | 14.7 / 14.7 (1738) | 14.7 / 14.8 (1738) |
| bf16/typical_1q_k4_ctx48 | 14.2 / 16.1 (1689) | 14.3 / 14.4 (1691) | 14.3 / 14.3 (1691) |
| bf16/typical_1q_tools8_ctx48 | 14.6 / 14.6 (1696) | 14.5 / 15.7 (1698) | 14.5 / 14.6 (1698) |
| bf16/typical_4q_mixed_ctx96 | 16.1 / 16.1 (1737) | 16.1 / 16.2 (1738) | 16.2 / 16.3 (1738) |
