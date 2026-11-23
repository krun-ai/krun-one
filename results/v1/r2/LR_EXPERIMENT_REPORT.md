# R2: encoder LR 2e-5 vs 1e-5 (controlled full fine-tune)

**Result: the registered rule picks 1e-5** (`configs/v1/r2_protocol.yaml → lr_selection`). Raw data:
`results/v1/r2/lr_selection.json`.

## Setup (only the encoder LR differs)

| | candidate 1 (`krun-one-v1-ft`) | LR experiment (`krun-one-v1-lr1e5`) |
|---|---|---|
| encoder LR / head LR | 2e-5 / 5e-5 | **1e-5** / 5e-5 |
| init | k1_0b.tar (sha `061160e3…`) | same |
| data | `data/kd1/train.jsonl` (sha `ddcf6d1b…`, 1,042,946) | same file (checksum-verified on the pod) |
| seed, curriculum, batching, steps, schedule, evaluation | identical (21,757 steps) | identical |
| GPU | RTX 4090 Secure | RTX 4090 Secure (`sij0m65kp9k8nr`), 4.1 h |

## Rule outcome

| condition (any one prefers 1e-5) | measured | met |
|---|---|---|
| ToolACE holdout gain ≥ 2 pp with the CI above 0 | +1.9 pp, 95% CI [−0.2, +4.3] | no |
| unseen-tool FA drop ≥ 3 pp at 70% tool coverage | 8.5% → **3.8%** (−4.7 pp) | **yes** |
| OOD FA drop ≥ 2 pp at 70% coverage | 2.5% → 2.0% | no |

| no-material-regression check | measured | ok |
|---|---|---|
| native non-EN accuracy (paired) | Δ 0.00 pp, CI [−0.31, +0.28] | ✅ |
| cross-language accuracy | 98.29 → 98.36 | ✅ |
| k = 64 / 128 accuracy | 95.4 / 94.1 → 97.0 / 93.6 | ✅ |
| ECE | 0.031 → 0.039 (limit +0.010) | ✅ |
| permutation consistency | 97.8 → 98.8 | ✅ |

## Other metrics (V eval)

| metric | 2e-5 | 1e-5 |
|---|---:|---:|
| intent accuracy | 97.6 | 97.4 |
| tool accuracy (V) | 98.5 | 98.6 |
| hard-negative accuracy | 97.8 | 97.8 |
| OOD AUROC (margin) | 0.975 | 0.971 |
| AURC | 0.027 | 0.028 |
| NLL (choice) | 0.140 | 0.163 |
| noul ROC-AUC / accuracy | 0.960 / 85.1 | 0.965 / 86.8 |
| score MAE / Spearman | 0.631 / 0.843 | 0.647 / 0.836 |
| multi micro / macro F1 | 90.8 / 82.2 | 91.1 / 81.6 |
| accuracy EN / PT-BR / PT-PT / ES / FR | 97.1 / 98.4 / 99.4 / 98.2 / 97.7 | 96.8 / 98.2 / 99.4 / 98.4 / 97.9 |
| ToolACE val_tools / holdout_tools | 92.6 / 84.1 | 92.8 / **86.1** |

## Tool coverage curves (V eval; thresholds from fit)

| coverage | selective acc 2e-5 / 1e-5 | OOD FA 2e-5 / 1e-5 | unseen-tool FA 2e-5 / 1e-5 |
|---:|---|---|---|
| 50% | 99.7 / 99.9 | 0.6 / 0.3 | 0.8 / 0.4 |
| 60% | 98.6 / 99.4 | 3.6 / 1.5 | 5.1 / 2.1 |
| 70% | 98.0 / 99.1 | 6.0 / **2.7** | 8.5 / **3.8** |
| 80% | 97.6 / 98.6 | 8.7 / **4.2** | 12.3 / **5.9** |
| 90% | 95.8 / 96.5 | 16.9 / 13.3 | 22.9 / 17.4 |

Tool AURC: 0.033 → 0.030. The intent curves are practically identical. The serving points chosen on fit are
80% / 80% for both models.

## Reading

Lower encoder LR generalizes better to unseen tools and irrelevance, with no measurable multilingual cost, at a small
calibration cost (ECE +0.008, NLL +0.02; within the limits). It does **not** by itself recover the ToolACE holdout
(86.1% vs v0.3 91.8%). That is the job of the tool-data experiment, which uses 1e-5.
