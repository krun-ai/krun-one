# Technical metrics — `data/v11/techbench/val.jsonl` (4054 examples)

## raw: top-1 by track (macro = mean over tracks)

| model | macro | overall | api | backend | cloud | database | devops | distributed | mlinfra | security | software_repair | state_action |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 | 22.8 | 29.5 | 19.7 | 17.1 | 15.9 | 23.6 | 9.3 | 16.2 | 22.3 | 32.9 | 63.0 | 8.4 |

## raw: slices (top-1), separation and flips

| model | easy | hard | hard_negative | constraint_flip | unseen_combination | seen_combination | unseen_technology | large_k | unseen_family | gold > hardest | margin (logit) | flips both | flips stuck | OOD FA | OOD AUROC | entropy | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 | 20.3 | 34.5 | 35.8 | 17.5 | — | — | — | 20.5 | — | 60.6 | +0.63 | 2.1 | 10.7 | 0.0 | 0.489 | 0.618 | 0.156 |

## feat: top-1 by track (macro = mean over tracks)

| model | macro | overall | api | backend | cloud | database | devops | distributed | mlinfra | security | state_action |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 | 38.0 | 35.6 | 40.3 | 69.4 | 18.9 | 33.9 | 6.5 | 44.9 | 41.9 | 75.0 | 11.5 |

## feat: slices (top-1), separation and flips

| model | easy | hard | hard_negative | constraint_flip | unseen_combination | seen_combination | unseen_technology | large_k | unseen_family | gold > hardest | margin (logit) | flips both | flips stuck | OOD FA | OOD AUROC | entropy | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 | 43.6 | 29.1 | 27.6 | 24.6 | — | — | — | 29.5 | — | 66.8 | +1.51 | 7.1 | 10.0 | 0.0 | 0.530 | 0.615 | 0.094 |

