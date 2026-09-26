# Laya vs Krun One v0

Protocol: identical decisions, options, serialization (Laya `build_sequence`, max_len 1024 / head 896) and metrics; temperatures (per family) and thresholds fitted on val_fit, numbers on val_eval. Abstention score: `top1_top2_margin`, operating point: target selective accuracy 0.98. Laya = upstream checkpoint, no training. `Laya (512/192)` = Laya with its shipped budgets.

| Metric | Laya | Laya (512/192) | Krun One v0 | Δ (Krun − Laya) |
|---|---:|---:|---:|---:|
| Intent accuracy (KrunBench val: Banking77 + MASSIVE en/pt/es) | 64.04% | 64.04% | 96.01% | +31.97 pp |
| Intent hard accuracy (semantic hard negatives, k=4-32) | 49.45% | 49.45% | 91.29% | +41.85 pp |
| Tool-routing accuracy (val_tools) | 66.61% | 68.05% | 94.86% | +28.25 pp |
| OOD AUROC (KrunBench val intents) | 0.6614 | 0.6614 | 0.9536 | 0.2922 |
| OOD AUROC (val_tools) | 0.6737 | 0.7366 | 0.9901 | 0.3164 |
| False acceptance @ target 0.98 (KrunBench val) | 0.00% | 0.00% | 8.07% | +8.07 pp |
| False acceptance @ target 0.98 (val_tools) | 4.32% | 2.73% | 3.41% | -0.91 pp |
| Coverage @ target 0.98 (KrunBench val) | 2.40% | 2.40% | 87.87% | +85.47 pp |
| Coverage @ target 0.98 (val_tools) | 31.41% | 27.17% | 91.34% | +59.93 pp |
| Selective accuracy @ target 0.98 (KrunBench val) | 97.75% | 97.75% | 99.17% | +1.42 pp |
| ECE, T-scaled (KrunBench val) | 0.0401 | 0.0401 | 0.0087 | -0.0313 |
| ECE, T-scaled (val_tools) | 0.0391 | 0.0378 | 0.0170 | -0.0221 |
| AURC (KrunBench val) | 0.1564 | 0.1564 | 0.0035 | -0.1529 |
| AURC (val_tools) | 0.1055 | 0.1037 | 0.0068 | -0.0987 |
| Permutation consistency (KrunBench val) | 67.76% | 67.76% | 98.91% | +31.15 pp |
| Permutation consistency (val_tools) | 64.58% | 64.58% | 91.67% | +27.08 pp |
| FINAL KrunBench test intent accuracy (Banking77 + MASSIVE) | 66.98% | 66.98% | 96.09% | +29.11 pp |
| FINAL BFCL accuracy (KrunBench test, never trained on) | 82.83% | 83.39% | 92.33% | +9.50 pp |
| FINAL BFCL irrelevance OOD AUROC | 0.7007 | 0.7114 | 0.8043 | 0.1035 |
| FINAL holdout_tools accuracy | 69.53% | 69.74% | 96.42% | +26.89 pp |
| FINAL holdout_tools OOD AUROC | 0.7235 | 0.7811 | 0.9927 | 0.2693 |
| FINAL KrunBench test OOD AUROC (all) | 0.6691 | 0.6748 | 0.9472 | 0.2780 |
| FINAL KrunBench test ECE | 0.0327 | 0.0338 | 0.0072 | -0.0255 |
| CPU p50, batch 1 (ms) | 351 | — | 356 | |
| GPU p50, batch 1 (ms) | 10.4300 | — | 10.9257 | |
| Parameters | 421,293,827 | — | 421,293,827 | |
| RAM (CPU RSS, MiB) | 2,565 | — | 2,565 | |
| VRAM peak, batch 1 / 32 (MiB) | 2,338 / 3,002 | — | 2,338 / 3,002 | |

## All headline metrics

| key | Laya | Laya (512/192) | Krun One v0 |
|---|---:|---:|---:|
| intent_acc_kb_val | 0.6404 | 0.6404 | 0.9601 |
| intent_macro_f1_kb_val | 0.7025 | 0.7025 | 0.9453 |
| hwu64_acc_kb_val | 0.7656 | 0.7656 | 0.9241 |
| massive_en-US_acc | 0.7341 | 0.7341 | 0.9759 |
| massive_pt-PT_acc | 0.5514 | 0.5514 | 0.9497 |
| massive_es-ES_acc | 0.5569 | 0.5569 | 0.9508 |
| banking77_acc | 0.7788 | 0.7788 | 0.9673 |
| intent_hard_acc | 0.4945 | 0.4945 | 0.9129 |
| intent_hard_k4_acc | 0.5620 | 0.5620 | 0.9392 |
| intent_hard_k8_acc | 0.5355 | 0.5355 | 0.9290 |
| intent_hard_k16_acc | 0.4547 | 0.4547 | 0.9140 |
| intent_hard_k32_acc | 0.4279 | 0.4279 | 0.8694 |
| tool_acc | 0.6661 | 0.6805 | 0.9486 |
| tool_macro_f1 | 0.7114 | 0.7168 | 0.9512 |
| tool_seen_tool_acc | 0.7116 | 0.7249 | 0.9696 |
| tool_unseen_tool_acc | 0.6248 | 0.6403 | 0.9294 |
| tool_near_duplicate_distractor_acc | 0.3692 | 0.3943 | 0.9176 |
| tool_description_dependent_acc | 0.5223 | 0.5567 | 0.9347 |
| tool_toolset_large(k>=16)_acc | 0.4671 | 0.5070 | 0.9155 |
| kb_val_ece | 0.0401 | 0.0401 | 0.0087 |
| kb_val_nll | 0.9632 | 0.9632 | 0.1478 |
| kb_val_brier | 0.4585 | 0.4585 | 0.0670 |
| kb_val_ece_raw | 0.0640 | 0.0640 | 0.0079 |
| kb_val_aurc | 0.1564 | 0.1564 | 0.0035 |
| kb_val_ood_auroc | 0.6614 | 0.6614 | 0.9536 |
| kb_val_far | 0.0000 | 0.0000 | 0.0807 |
| kb_val_coverage | 0.0240 | 0.0240 | 0.8787 |
| kb_val_selective_acc | 0.9775 | 0.9775 | 0.9917 |
| kb_val_perm_consistency | 0.6776 | 0.6776 | 0.9891 |
| val_hard_ece | 0.0653 | 0.0653 | 0.0212 |
| val_hard_nll | 1.5956 | 1.5956 | 0.3054 |
| val_hard_brier | 0.6551 | 0.6551 | 0.1342 |
| val_hard_ece_raw | 0.1132 | 0.1132 | 0.0330 |
| val_hard_aurc | 0.3144 | 0.3144 | 0.0157 |
| val_hard_ood_auroc | 0.5356 | 0.5635 | 0.9408 |
| val_hard_far | 0.0000 | 0.0000 | 0.0784 |
| val_hard_coverage | 0.0116 | 0.0116 | 0.8096 |
| val_hard_selective_acc | 0.9737 | 0.9737 | 0.9769 |
| val_hard_perm_consistency | 0.4756 | 0.4756 | 0.9756 |
| val_tools_ece | 0.0391 | 0.0378 | 0.0170 |
| val_tools_nll | 1.0594 | 1.0442 | 0.1778 |
| val_tools_brier | 0.4219 | 0.4146 | 0.0800 |
| val_tools_ece_raw | 0.0413 | 0.0623 | 0.0285 |
| val_tools_aurc | 0.1055 | 0.1037 | 0.0068 |
| val_tools_ood_auroc | 0.6737 | 0.7366 | 0.9901 |
| val_tools_far | 0.0432 | 0.0273 | 0.0341 |
| val_tools_coverage | 0.3141 | 0.2717 | 0.9134 |
| val_tools_selective_acc | 0.9885 | 0.9900 | 0.9812 |
| val_tools_perm_consistency | 0.6458 | 0.6458 | 0.9167 |
| holdout_tool_acc | 0.6953 | 0.6974 | 0.9642 |
| holdout_tool_ood_auroc | 0.7235 | 0.7811 | 0.9927 |
| holdout_tool_far | 0.0298 | 0.0132 | 0.0254 |
| kb_test_intent_acc | 0.6698 | 0.6698 | 0.9609 |
| bfcl_acc | 0.8283 | 0.8339 | 0.9233 |
| bfcl_near_duplicate_distractor_acc | 0.8227 | 0.8177 | 0.8621 |
| bfcl_description_dependent_acc | 0.7222 | 0.7222 | 0.8642 |
| bfcl_toolset_large(k>=16)_acc | 1.0000 | 0.5000 | 0.5000 |
| kb_test_ece | 0.0327 | 0.0338 | 0.0072 |
| kb_test_ood_auroc | 0.6691 | 0.6748 | 0.9472 |
| kb_test_far | 0.0154 | 0.0094 | 0.1144 |
| bfcl_irrelevance_auroc | 0.7007 | 0.7114 | 0.8043 |
