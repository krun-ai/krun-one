# JevBench public items (231) — external check, evaluation only

Published rows are the benchmark's own per-task outcomes on the same public items (v1.2 artifact). Krun rows were run here with the mapping in scripts/v11/jevbench_eval.py. Chance differs per item (2 to 9 labels).

| system | all 231 | easy (48) | standard (72) | hard (111) | adversarial | ambiguous | judge_hard | long_policy | multi_hop | probability | routing_hard | temporal_numeric | tradeoff | trap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Jev 1.13.0 (TypeSafe AI) | 86.6 | 100.0 | 98.6 | 73.0 | 100.0 | 85.7 | 76.5 | 63.2 | 83.3 | 70.0 | 100.0 | 26.7 | 83.3 | 100.0 |
| GPT-5.6 Luna (low reasoning effort) | 97.4 | 100.0 | 97.2 | 96.4 | 100.0 | 100.0 | 94.1 | 100.0 | 94.4 | 90.0 | 100.0 | 93.3 | 100.0 | 100.0 |
| SemIf, formerly OpenJev (Qwen3.5-4B, TheoLeeCJ) | 81.0 | 100.0 | 98.6 | 61.3 | 66.7 | 71.4 | 76.5 | 52.6 | 66.7 | 40.0 | 100.0 | 20.0 | 66.7 | 100.0 |
| Laya (Convai Innovations, ModernBERT-large 421M) | 58.4 | 95.8 | 69.4 | 35.1 | 66.7 | 14.3 | 41.2 | 21.1 | 33.3 | 50.0 | 20.0 | 33.3 | 100.0 | 0.0 |
| GLiNER2.5 multi (Fastino, 287M) | 48.9 | 91.7 | 44.4 | 33.3 | 66.7 | 42.9 | 41.2 | 36.8 | 27.8 | 40.0 | 0.0 | 40.0 | 16.7 | 0.0 |
| open-jev-deberta-v3-large (local CPU) | 52.4 | 100.0 | 43.1 | 37.8 | 66.7 | 57.1 | 41.2 | 31.6 | 44.4 | 30.0 | 20.0 | 20.0 | 50.0 | 37.5 |
| Krun v1_gpu | 53.2 | 91.7 | 62.5 | 30.6 | 50.0 | 28.6 | 47.1 | 31.6 | 16.7 | 40.0 | 20.0 | 20.0 | 33.3 | 25.0 |
| Krun r1_s1 | 55.0 | 91.7 | 56.9 | 37.8 | 33.3 | 57.1 | 41.2 | 47.4 | 27.8 | 60.0 | 20.0 | 26.7 | 50.0 | 12.5 |
| Krun r1_s2 | 55.0 | 91.7 | 59.7 | 36.0 | 33.3 | 57.1 | 41.2 | 36.8 | 22.2 | 60.0 | 20.0 | 26.7 | 66.7 | 12.5 |
| Krun r2l4_s1 | 52.4 | 91.7 | 58.3 | 31.5 | 50.0 | 42.9 | 41.2 | 42.1 | 16.7 | 30.0 | 40.0 | 20.0 | 50.0 | 0.0 |
| Krun r2l4_s2 | 52.4 | 91.7 | 61.1 | 29.7 | 33.3 | 28.6 | 41.2 | 36.8 | 22.2 | 40.0 | 20.0 | 20.0 | 50.0 | 0.0 |
| Krun r2l4na_s1 | 52.4 | 91.7 | 56.9 | 32.4 | 50.0 | 42.9 | 41.2 | 42.1 | 16.7 | 30.0 | 60.0 | 20.0 | 50.0 | 0.0 |
| Krun r2l4na_s2 | 52.4 | 89.6 | 62.5 | 29.7 | 33.3 | 42.9 | 41.2 | 31.6 | 16.7 | 30.0 | 40.0 | 20.0 | 66.7 | 0.0 |

Krun rows: Krun v1_gpu: hard items with a truncated state 40%, ECE 0.172; Krun r1_s1: hard items with a truncated state 40%, ECE 0.137; Krun r1_s2: hard items with a truncated state 40%, ECE 0.106; Krun r2l4_s1: hard items with a truncated state 40%, ECE 0.178; Krun r2l4_s2: hard items with a truncated state 40%, ECE 0.167; Krun r2l4na_s1: hard items with a truncated state 40%, ECE 0.172; Krun r2l4na_s2: hard items with a truncated state 40%, ECE 0.170.
