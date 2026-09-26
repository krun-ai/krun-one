| Primitive | Dataset/Suite | Metric | Laya (1024/896) | Laya (512/192) | Krun One v0 | v03_B |
|---|---|---|---:|---:|---:|---:|
| Choice | KrunBench test intents (B77+MASSIVE) | accuracy | 66.98% | 66.98% | 96.09% | 96.14% |
| Choice | BFCL | accuracy | 82.83% | 83.39% | 92.33% | 92.25% |
| Choice | holdout_tools | accuracy | 69.53% | 69.74% | 96.42% | 96.50% |
| Choice | KrunBench val intents | accuracy | 64.04% | 64.04% | 96.01% | 95.95% |
| Choice | val_tools | accuracy | 66.61% | 68.05% | 94.86% | 95.22% |
| Choice | KrunBench test | OOD AUROC | 0.6691 | 0.6748 | 0.9472 | 0.9513 |
| Choice | holdout_tools | OOD AUROC | 0.7235 | 0.7811 | 0.9927 | 0.9926 |
| Choice | BFCL irrelevance | OOD AUROC | 0.7007 | 0.7114 | 0.8043 | 0.7948 |
| Choice | KrunBench test | ECE | 0.0327 | 0.0338 | 0.0072 | 0.0082 |
| Noul | test in-domain (intents, tools, Civil, MHS) | roc_auc | 0.8263 | 0.8245 | 0.8655 | 0.9677 |
| Noul | test in-domain (intents, tools, Civil, MHS) | pr_auc | 0.7687 | 0.7664 | 0.8152 | 0.9593 |
| Noul | test in-domain (intents, tools, Civil, MHS) | accuracy@0.5 | 0.7500 | 0.7478 | 0.7875 | 0.9078 |
| Noul | test in-domain (intents, tools, Civil, MHS) | nll | 0.5147 | 0.5168 | 0.4638 | 0.2291 |
| Noul | test in-domain (intents, tools, Civil, MHS) | brier | 0.1689 | 0.1698 | 0.1504 | 0.0688 |
| Noul | test in-domain (intents, tools, Civil, MHS) | ece | 0.0244 | 0.0241 | 0.0490 | 0.0116 |
| Noul | BoolQ (external) | roc_auc | 0.8134 | 0.8129 | 0.8140 | 0.8116 |
| Noul | BoolQ (external) | accuracy@0.5 | 0.7563 | 0.7557 | 0.7639 | 0.7560 |
| Noul | BoolQ (external) | brier | 0.1668 | 0.1670 | 0.1693 | 0.1873 |
| Noul | BoolQ (external) | ece | 0.0411 | 0.0412 | 0.0481 | 0.1319 |
| Noul | positives vs hard negatives | roc_auc | 0.8290 | 0.8274 | 0.8831 | 0.9663 |
| Noul | positives vs hard negatives | accuracy@0.5 | 0.7288 | 0.7266 | 0.7700 | 0.8867 |
| Noul | positives vs hard negatives | brier | 0.1875 | 0.1887 | 0.1712 | 0.0835 |
| Noul | positives vs hard negatives | ece | 0.1624 | 0.1639 | 0.1886 | 0.0866 |
| Noul | positives vs cross-domain negatives | roc_auc | 0.9529 | 0.9526 | 0.9698 | 0.9914 |
| Noul | positives vs cross-domain negatives | accuracy@0.5 | 0.7214 | 0.7185 | 0.7499 | 0.8803 |
| Noul | positives vs cross-domain negatives | brier | 0.1913 | 0.1929 | 0.1893 | 0.0882 |
| Noul | positives vs cross-domain negatives | ece | 0.3264 | 0.3284 | 0.3326 | 0.1570 |
| Noul | unseen propositions (never trained) | roc_auc | 0.8589 | 0.8584 | 0.8768 | 0.9069 |
| Noul | unseen propositions (never trained) | accuracy@0.5 | 0.7296 | 0.7296 | 0.7296 | 0.7750 |
| Noul | unseen propositions (never trained) | brier | 0.1880 | 0.1890 | 0.1912 | 0.1542 |
| Noul | unseen propositions (never trained) | ece | 0.1810 | 0.1827 | 0.2093 | 0.1509 |
| Score | test in-domain (HelpSteer2 + MHS) | mae | 0.9412 | 0.9476 | 0.8879 | 0.6200 |
| Score | test in-domain (HelpSteer2 + MHS) | rmse | 1.1947 | 1.2006 | 1.1268 | 0.8224 |
| Score | test in-domain (HelpSteer2 + MHS) | accuracy | 0.4085 | 0.4035 | 0.4228 | 0.5535 |
| Score | test in-domain (HelpSteer2 + MHS) | spearman | 0.4587 | 0.4496 | 0.5455 | 0.7668 |
| Score | test in-domain (HelpSteer2 + MHS) | kendall_tau_b | 0.3550 | 0.3474 | 0.4197 | 0.6275 |
| Score | test in-domain (HelpSteer2 + MHS) | nll | 1.3889 | 1.3943 | 1.3495 | 1.0265 |
| Score | test in-domain (HelpSteer2 + MHS) | brier | 0.7131 | 0.7154 | 0.6965 | 0.5596 |
| Score | test in-domain (HelpSteer2 + MHS) | ece | 0.0293 | 0.0269 | 0.0409 | 0.0142 |
| Score | test in-domain (HelpSteer2 + MHS) | rps | 0.1571 | 0.1581 | 0.1467 | 0.0971 |
| Score | test in-domain (HelpSteer2 + MHS) | distance_weighted_error | 1.2549 | 1.2597 | 1.1855 | 0.7565 |
| Score | HelpSteer2 | mae | 0.7517 | 0.7644 | 0.7265 | 0.6425 |
| Score | HelpSteer2 | accuracy | 0.5103 | 0.5003 | 0.5007 | 0.5617 |
| Score | HelpSteer2 | spearman | 0.6542 | 0.6363 | 0.6613 | 0.6972 |
| Score | HelpSteer2 | nll | 1.1718 | 1.1829 | 1.1744 | 1.0343 |
| Score | HelpSteer2 | ece | 0.0736 | 0.0659 | 0.0195 | 0.0458 |
| Score | HelpSteer2 | distance_weighted_error | 1.0224 | 1.0317 | 0.9732 | 0.7444 |
| Score | MHS | mae | 1.1308 | 1.1308 | 1.0492 | 0.5974 |
| Score | MHS | accuracy | 0.3067 | 0.3067 | 0.3450 | 0.5453 |
| Score | MHS | spearman | 0.2247 | 0.2247 | 0.3819 | 0.8126 |
| Score | MHS | nll | 1.6060 | 1.6057 | 1.5246 | 1.0187 |
| Score | MHS | ece | 0.1023 | 0.1019 | 0.0888 | 0.0318 |
| Score | MHS | distance_weighted_error | 1.4875 | 1.4876 | 1.3979 | 0.7686 |
| Score | SST-5 (external) | mae | 0.9622 | 0.9621 | 0.7632 | 0.6840 |
| Score | SST-5 (external) | accuracy | 0.3566 | 0.3566 | 0.4226 | 0.4113 |
| Score | SST-5 (external) | spearman | 0.7601 | 0.7602 | 0.8140 | 0.8233 |
| Score | SST-5 (external) | nll | 1.6995 | 1.6986 | 1.3363 | 1.2823 |
| Score | SST-5 (external) | ece | 0.2335 | 0.2332 | 0.1320 | 0.1857 |
| Score | SST-5 (external) | distance_weighted_error | 1.0588 | 1.0590 | 0.9450 | 0.7965 |
| Mixed | MHS mixed requests (choice) | accuracy | 0.7200 | 0.7200 | 0.7600 | 0.7670 |
| Mixed | MHS mixed requests (noul) | roc_auc | 0.8646 | 0.8646 | 0.8469 | 0.9270 |
| Mixed | MHS mixed requests (score) | mae | 1.1755 | 1.1753 | 1.0150 | 0.5995 |
| Mixed | MHS mixed requests (score) | spearman | 0.2611 | 0.2612 | 0.3884 | 0.7733 |
| Mixed | all questions correct | rate | 0.0530 | 0.0530 | 0.0760 | 0.2240 |
| Mixed | invariance alone vs mixed (choice) | max |Δp| | 0.0589 | 0.0589 | 0.0437 | 0.0638 |
| Mixed | invariance alone vs mixed (noul) | max |Δp| | 0.0266 | 0.0266 | 0.0301 | 0.0216 |
| Mixed | invariance alone vs mixed (score) | max |Δp| | 0.0321 | 0.0320 | 0.0439 | 0.0346 |
