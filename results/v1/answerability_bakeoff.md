# Answerability Bakeoff — proxy (mmBERT-small + v1 stack), validation only

Operating point: selective accuracy 0.98 on val_fit per family; numbers on val_eval; mean ± sd over seeds. FA = false acceptance (%) of unanswerable decisions; risk = accepted-and-wrong / accepted (%).

| mechanism | macro FA | tool irrelevance FA | tool gold-removed FA | intent near-OOD FA | cross-task FA | coverage | AUROC | AURC | unseen-tool risk | hard risk |
|---|---|---|---|---|---|---|---|---|---|---|
| A margin (baseline) | 2.8 ±0.2 | 1.8 ±0.5 | 1.6 ±0.2 | 5.0 ±0.8 | 0.0 ±0.0 | 64.0 ±2.2 | 0.959 ±0.002 | 0.052 ±0.001 | 1.0 ±0.2 | 0.9 ±0.2 |
| E[A] softmax-only fusion | 2.9 ±0.4 | 2.0 ±0.3 | 1.6 ±0.2 | 5.0 ±1.6 | 0.3 ±0.4 | 60.4 ±5.9 | 0.968 ±0.002 | 0.052 ±0.001 | 1.3 ±0.1 | 1.1 ±0.2 |
| B [NULL] in softmax | 0.1 ±0.1 | 0.0 ±0.0 | 0.2 ±0.2 | 0.0 ±0.0 | 0.0 ±0.0 | 6.5 ±4.3 | 0.937 ±0.027 | 0.098 ±0.014 | 1.8 ±2.5 | 4.3 ±0.7 |
| C answerability head | 0.9 ±0.4 | 0.0 ±0.0 | 0.0 ±0.0 | 2.6 ±1.1 | 0.0 ±0.0 | 28.9 ±5.9 | 0.979 ±0.002 | 0.076 ±0.003 | 5.6 ±1.9 | 2.5 ±0.1 |
| D max applicability | 1.7 ±0.8 | 1.6 ±0.8 | 0.8 ±0.8 | 2.8 ±1.1 | 0.0 ±0.0 | 49.2 ±5.3 | 0.962 ±0.003 | 0.055 ±0.001 | 1.1 ±0.3 | 0.9 ±0.3 |
| J margin | 2.1 ±0.5 | 0.8 ±0.3 | 0.6 ±0.2 | 5.0 ±1.7 | 0.0 ±0.0 | 39.7 ±4.7 | 0.895 ±0.004 | 0.064 ±0.002 | 0.2 ±0.2 | 0.7 ±0.1 |
| J p_null | 0.1 ±0.1 | 0.0 ±0.0 | 0.0 ±0.0 | 0.4 ±0.3 | 0.0 ±0.0 | 17.4 ±11.0 | 0.949 ±0.040 | 0.087 ±0.016 | 4.0 ±1.0 | 2.6 ±0.8 |
| J ans head | 0.5 ±0.2 | 0.0 ±0.0 | 0.2 ±0.2 | 1.3 ±0.5 | 0.0 ±0.0 | 32.6 ±8.4 | 0.979 ±0.001 | 0.070 ±0.003 | 3.2 ±0.8 | 2.6 ±0.3 |
| J max apply | 1.6 ±0.1 | 1.2 ±0.5 | 0.8 ±0.5 | 2.8 ±0.6 | 0.0 ±0.0 | 48.2 ±5.3 | 0.965 ±0.002 | 0.057 ±0.001 | 1.6 ±0.3 | 1.3 ±0.4 |
| E fusion (on J) | 1.9 ±0.4 | 1.6 ±0.3 | 0.6 ±0.2 | 3.5 ±1.6 | 0.0 ±0.0 | 58.2 ±4.6 | 0.973 ±0.001 | 0.055 ±0.002 | 2.1 ±0.4 | 1.6 ±0.0 |

## Model-level gates (per trained variant)

| variant | intent acc | tool acc | ECE | NLL | permutation consistency |
|---|---|---|---|---|---|
| A_margin | 90.0 ±0.2 | 93.8 ±0.2 | 0.014 ±0.001 | 0.273 ±0.000 | 96.6 ±1.3 |
| B_null | 89.9 ±0.5 | 93.0 ±0.7 | 0.011 ±0.001 | 0.263 ±0.006 | 94.6 ±2.2 |
| C_ans_head | 90.0 ±0.3 | 93.2 ±0.5 | 0.009 ±0.003 | 0.267 ±0.003 | 97.2 ±1.3 |
| D_apply | 90.0 ±0.3 | 93.3 ±0.1 | 0.011 ±0.002 | 0.267 ±0.005 | 95.1 ±1.0 |
| J_joint | 90.1 ±0.0 | 92.8 ±0.4 | 0.011 ±0.003 | 0.270 ±0.008 | 95.3 ±0.6 |

## Decision (rule registered in configs/v1/proxy.yaml before training)

| mechanism | Δ macro FA vs A (pp) | relative | every seed | 95% CI (A−X) | gates | clear win |
|---|---|---|---|---|---|---|
| E[A] softmax-only fusion | -0.1 | -2% | False | [-0.4, +0.3] | FAIL: coverage | no |
| B [NULL] in softmax | +2.7 | +98% | True | [+1.4, +4.3] | FAIL: tool_acc,perm,coverage | no |
| C answerability head | +1.9 | +69% | True | [+1.0, +3.0] | FAIL: tool_acc,coverage | no |
| D max applicability | +1.1 | +38% | True | [+0.2, +2.0] | FAIL: perm,coverage | no |
| J margin | +0.7 | +24% | True | [-0.2, +1.6] | FAIL: tool_acc,perm,coverage | no |
| J p_null | +2.7 | +95% | True | [+1.4, +4.1] | FAIL: tool_acc,perm,coverage | no |
| J ans head | +2.3 | +83% | True | [+1.2, +3.6] | FAIL: tool_acc,perm,coverage | no |
| J max apply | +1.2 | +43% | True | [+0.5, +2.0] | FAIL: tool_acc,perm,coverage | no |
| E fusion (on J) | +0.9 | +32% | True | [+0.2, +1.7] | FAIL: tool_acc,perm,coverage | no |

**Winner:** none -> keep top1-top2 margin as the public abstention mechanism

## Post-hoc diagnostic (NOT part of the registered rule): macro OOD FA at A's answerable coverage

Each mechanism's threshold is moved (on val_eval) until it accepts the same fraction of answerable decisions as A-margin does at its operating point. Lower = better OOD discrimination at equal coverage.

| mechanism | macro FA @ A coverage (%) |
|---|---|
| J_joint:ans_head | 1.9 ±0.3 |
| J_joint:E | 2.3 ±0.0 |
| C_ans_head:ans_head | 2.3 ±0.5 |
| A_margin:margin | 2.8 ±0.2 |
| A_margin:E_softmax | 3.3 ±0.1 |
| J_joint:max_apply | 3.3 ±0.1 |
| D_apply:max_apply | 3.4 ±0.2 |
| B_null:null_softmax | 5.9 ±1.5 |
| J_joint:null_softmax | 6.4 ±5.7 |
| J_joint:margin | 7.4 ±0.3 |
| B_null:E_softmax | 8.0 ±1.1 |
| C_ans_head:E_softmax | 8.3 ±0.7 |
| D_apply:E_softmax | 8.9 ±1.1 |
