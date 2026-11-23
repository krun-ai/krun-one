### Full sample (240 examples): headline

| model / variant | n | acc % | 95% CI | answerable | unanswerable (says none) | says none overall | false none on answerable | flip both-correct | valid JSON | truncated |
|---|---|---|---|---|---|---|---|---|---|---|
| glm-5.3_cloud__think_plus16k_retry | 240 | 87.9 | 83.8-92.1 | 87.0 | 92.5 | 15.4 | 0.0 | 86.7 (30 pairs) | 93.3 | 6.7 |
| gpt-oss_120b-cloud | 240 | 82.5 | 77.5-87.1 | 84.5 | 72.5 | 17.1 | 6.0 | 83.3 (30 pairs) | 99.2 | 0.8 |
| glm-5.3_cloud__think | 240 | 80.4 | 75.4-85.4 | 82.5 | 70.0 | 11.7 | 0.0 | 80.0 (30 pairs) | 83.8 | 15.4 |
| kimi-k3_cloud | 240 | 74.2 | 68.3-79.6 | 74.5 | 72.5 | 13.3 | 1.5 | 46.7 (30 pairs) | 100.0 | 0.0 |
| deepseek-v4.1-flash_cloud | 240 | 70.8 | 65.0-76.7 | 70.0 | 75.0 | 13.3 | 1.0 | 26.7 (30 pairs) | 95.0 | 1.7 |
| deepseek-v4-pro_cloud | 240 | 64.2 | 58.3-70.4 | 68.5 | 42.5 | 7.9 | 1.0 | 36.7 (30 pairs) | 100.0 | 0.0 |
| qwen3_8b | 239 | 52.7 | 46.4-59.4 | 58.8 | 22.5 | 4.6 | 1.0 | 16.7 (30 pairs) | 100.0 | 0.0 |
| qwen3_4b | 240 | 30.8 | 25.0-36.7 | 19.0 | 90.0 | 79.6 | 77.5 | 3.3 (30 pairs) | 100.0 | 0.0 |

### Full sample (240 examples): accuracy % per track

| model / variant | api | backend | cloud | database | devops | distributed | mlinfra | security | software_repair | state_action |
|---|---|---|---|---|---|---|---|---|---|---|
| glm-5.3_cloud__think_plus16k_retry | 100.0 | 91.7 | 100.0 | 100.0 | 95.8 | 95.8 | 62.5 | 100.0 | 70.8 | 62.5 |
| gpt-oss_120b-cloud | 91.7 | 70.8 | 100.0 | 87.5 | 95.8 | 95.8 | 62.5 | 95.8 | 70.8 | 54.2 |
| glm-5.3_cloud__think | 87.5 | 75.0 | 100.0 | 95.8 | 95.8 | 87.5 | 50.0 | 100.0 | 62.5 | 50.0 |
| kimi-k3_cloud | 70.8 | 95.8 | 91.7 | 45.8 | 87.5 | 54.2 | 62.5 | 100.0 | 83.3 | 50.0 |
| deepseek-v4.1-flash_cloud | 62.5 | 95.8 | 79.2 | 66.7 | 70.8 | 62.5 | 37.5 | 100.0 | 83.3 | 50.0 |
| deepseek-v4-pro_cloud | 62.5 | 79.2 | 91.7 | 50.0 | 70.8 | 45.8 | 37.5 | 95.8 | 70.8 | 37.5 |
| qwen3_8b | 45.8 | 83.3 | 62.5 | 33.3 | 66.7 | 33.3 | 33.3 | 87.5 | 62.5 | 17.4 |
| qwen3_4b | 25.0 | 66.7 | 25.0 | 29.2 | 12.5 | 16.7 | 16.7 | 54.2 | 45.8 | 16.7 |
| n per track | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 24 |

### Full sample (240 examples): accuracy % per slice family

| model / variant | rule_table | simulator | swe | tetris |
|---|---|---|---|---|
| glm-5.3_cloud__think_plus16k_retry | 91.6 | 97.3 | 70.8 | 25.0 |
| gpt-oss_120b-cloud | 85.5 | 93.2 | 70.8 | 8.3 |
| glm-5.3_cloud__think | 84.0 | 93.2 | 62.5 | 0.0 |
| kimi-k3_cloud | 95.4 | 42.5 | 83.3 | 16.7 |
| deepseek-v4.1-flash_cloud | 83.2 | 53.4 | 83.3 | 16.7 |
| deepseek-v4-pro_cloud | 80.2 | 41.1 | 70.8 | 16.7 |
| qwen3_8b | 70.2 | 24.7 | 62.5 | 9.1 |
| qwen3_4b | 38.9 | 16.4 | 45.8 | 0.0 |
| n | 131 | 73 | 24 | 12 |

### Full sample (240 examples): latency, tokens, cost

| model / variant | median s | p95 s | clean sequential median s | prompt tok (mean) | completion tok (mean, incl. thinking) | USD / 10k examples | hours / 10k (sequential) | hours / 10k (3 concurrent cloud slots) |
|---|---|---|---|---|---|---|---|---|
| glm-5.3_cloud__think_plus16k_retry | 2.5285 | 85.433 | - | 330 | 2313 | 106.4059 | 37.9628 | 12.6543 |
| gpt-oss_120b-cloud | 4.054 | 8.961 | 0.9975 | 345 | 202 | 1.7291 | 3.4447 | 1.1482 |
| glm-5.3_cloud__think | 2.5285 | 27.195 | - | 285 | 1101 | 52.45 | 20.635 | 6.8783 |
| kimi-k3_cloud | 4.0695 | 8.903 | 1.125 | 356 | 13 | 12.6449 | 4.4417 | 1.4806 |
| deepseek-v4.1-flash_cloud | 3.5795 | 8.511 | 0.4255 | 286 | 12 | 1.0029 | 1.2456 | 0.4152 |
| deepseek-v4-pro_cloud | 3.6 | 8.511 | 0.616 | 285 | 7 | 4.0344 | 1.8464 | 0.6155 |
| qwen3_8b | 3.401 | 4.598 | - | 300 | 7 | 0.0 | 9.5747 | - |
| qwen3_4b | 1.5835 | 2.368 | - | 295 | 10 | 0.0 | 4.3262 | - |

### Fixed 60-example subset: headline

| model / variant | n | acc % | 95% CI | answerable | unanswerable (says none) | says none overall | false none on answerable | flip both-correct | valid JSON | truncated |
|---|---|---|---|---|---|---|---|---|---|---|
| kimi-k3_cloud__think | 60 | 90.0 | 81.7-96.7 | 88.2 | 100.0 | 15.0 | 0.0 | 85.7 (7 pairs) | 93.3 | 6.7 |
| glm-5.3_cloud__think_plus16k_retry | 60 | 90.0 | 81.7-96.7 | 88.2 | 100.0 | 15.0 | 0.0 | 85.7 (7 pairs) | 96.7 | 3.3 |
| gpt-oss_120b-cloud | 60 | 83.3 | 73.3-91.7 | 84.3 | 77.8 | 16.7 | 5.9 | 57.1 (7 pairs) | 100.0 | 0.0 |
| gpt-oss_120b-cloud__think_high | 60 | 83.3 | 73.3-91.7 | 84.3 | 77.8 | 13.3 | 2.0 | 100.0 (7 pairs) | 88.3 | 11.7 |
| glm-5.3_cloud__think | 60 | 81.7 | 71.7-90.0 | 82.3 | 77.8 | 11.7 | 0.0 | 71.4 (7 pairs) | 86.7 | 13.3 |
| gpt-oss_20b | 59 | 78.0 | 67.8-88.1 | 76.0 | 88.9 | 17.0 | 4.0 | 57.1 (7 pairs) | 96.6 | 3.4 |
| kimi-k3_cloud | 60 | 75.0 | 63.3-85.0 | 74.5 | 77.8 | 11.7 | 0.0 | 57.1 (7 pairs) | 100.0 | 0.0 |
| deepseek-v4-pro_cloud | 60 | 68.3 | 56.7-80.0 | 70.6 | 55.6 | 8.3 | 0.0 | 42.9 (7 pairs) | 100.0 | 0.0 |
| deepseek-v4.1-flash_cloud | 60 | 63.3 | 51.7-75.0 | 58.8 | 88.9 | 13.3 | 0.0 | 14.3 (7 pairs) | 90.0 | 3.3 |
| qwen3_8b | 60 | 51.7 | 40.0-65.0 | 54.9 | 33.3 | 6.7 | 2.0 | 28.6 (7 pairs) | 100.0 | 0.0 |
| qwen3_4b | 60 | 31.7 | 20.0-43.3 | 19.6 | 100.0 | 78.3 | 74.5 | 0.0 (7 pairs) | 100.0 | 0.0 |

### Fixed 60-example subset: accuracy % per track

| model / variant | api | backend | cloud | database | devops | distributed | mlinfra | security | software_repair | state_action |
|---|---|---|---|---|---|---|---|---|---|---|
| kimi-k3_cloud__think | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 83.3 | 66.7 | 100.0 | 100.0 | 50.0 |
| glm-5.3_cloud__think_plus16k_retry | 100.0 | 83.3 | 100.0 | 100.0 | 83.3 | 100.0 | 66.7 | 100.0 | 83.3 | 83.3 |
| gpt-oss_120b-cloud | 83.3 | 83.3 | 100.0 | 83.3 | 100.0 | 83.3 | 83.3 | 100.0 | 66.7 | 50.0 |
| gpt-oss_120b-cloud__think_high | 83.3 | 83.3 | 100.0 | 66.7 | 100.0 | 100.0 | 100.0 | 83.3 | 66.7 | 50.0 |
| glm-5.3_cloud__think | 100.0 | 66.7 | 100.0 | 100.0 | 83.3 | 83.3 | 50.0 | 100.0 | 83.3 | 50.0 |
| gpt-oss_20b | 100.0 | 83.3 | 83.3 | 50.0 | 50.0 | 100.0 | 66.7 | 100.0 | 83.3 | 60.0 |
| kimi-k3_cloud | 66.7 | 100.0 | 83.3 | 50.0 | 83.3 | 66.7 | 66.7 | 100.0 | 83.3 | 50.0 |
| deepseek-v4-pro_cloud | 66.7 | 83.3 | 100.0 | 50.0 | 66.7 | 50.0 | 33.3 | 100.0 | 83.3 | 50.0 |
| deepseek-v4.1-flash_cloud | 66.7 | 100.0 | 66.7 | 50.0 | 50.0 | 66.7 | 16.7 | 100.0 | 83.3 | 33.3 |
| qwen3_8b | 33.3 | 100.0 | 50.0 | 33.3 | 50.0 | 33.3 | 16.7 | 100.0 | 66.7 | 33.3 |
| qwen3_4b | 33.3 | 66.7 | 33.3 | 16.7 | 16.7 | 16.7 | 16.7 | 66.7 | 33.3 | 16.7 |
| n per track | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 |

### Fixed 60-example subset: accuracy % per slice family

| model / variant | rule_table | simulator | swe | tetris |
|---|---|---|---|---|
| kimi-k3_cloud__think | 93.9 | 94.4 | 100.0 | 0.0 |
| glm-5.3_cloud__think_plus16k_retry | 87.9 | 100.0 | 83.3 | 66.7 |
| gpt-oss_120b-cloud | 93.9 | 83.3 | 66.7 | 0.0 |
| gpt-oss_120b-cloud__think_high | 93.9 | 83.3 | 66.7 | 0.0 |
| glm-5.3_cloud__think | 84.9 | 88.9 | 83.3 | 0.0 |
| gpt-oss_20b | 81.8 | 77.8 | 83.3 | 0.0 |
| kimi-k3_cloud | 93.9 | 50.0 | 83.3 | 0.0 |
| deepseek-v4-pro_cloud | 84.9 | 38.9 | 83.3 | 33.3 |
| deepseek-v4.1-flash_cloud | 81.8 | 33.3 | 83.3 | 0.0 |
| qwen3_8b | 69.7 | 22.2 | 66.7 | 0.0 |
| qwen3_4b | 42.4 | 16.7 | 33.3 | 0.0 |
| n | 33 | 18 | 6 | 3 |

### Fixed 60-example subset: latency, tokens, cost

| model / variant | median s | p95 s | clean sequential median s | prompt tok (mean) | completion tok (mean, incl. thinking) | USD / 10k examples | hours / 10k (sequential) | hours / 10k (3 concurrent cloud slots) |
|---|---|---|---|---|---|---|---|---|
| kimi-k3_cloud__think | 3.4785 | 43.155 | - | 423 | 640 | 108.682 | 23.1817 | 7.7272 |
| glm-5.3_cloud__think_plus16k_retry | 2.6725 | 81.102 | - | 327 | 2182 | 100.5916 | 37.1759 | 12.3919 |
| gpt-oss_120b-cloud | 4.352 | 9.011 | 0.9975 | 344 | 196 | 1.6952 | 3.4447 | 1.1482 |
| gpt-oss_120b-cloud__think_high | 4.0985 | 21.953 | - | 344 | 1981 | 12.4014 | 20.465 | 6.8217 |
| glm-5.3_cloud__think | 2.6725 | 27.447 | - | 285 | 1035 | 49.5308 | 20.0436 | 6.6812 |
| gpt-oss_20b | 9.266 | 67.643 | - | 342 | 188 | 0.0 | 60.5588 | - |
| kimi-k3_cloud | 4.3465 | 9.022 | 1.125 | 356 | 13 | 12.617 | 4.4417 | 1.4806 |
| deepseek-v4-pro_cloud | 3.433 | 8.272 | 0.616 | 284 | 7 | 4.024 | 1.8464 | 0.6155 |
| deepseek-v4.1-flash_cloud | 3.9705 | 11.404 | 0.4255 | 285 | 19 | 1.087 | 1.2456 | 0.4152 |
| qwen3_8b | 3.42 | 4.573 | - | 300 | 7 | 0.0 | 10.2773 | - |
| qwen3_4b | 1.5845 | 2.35 | - | 294 | 10 | 0.0 | 4.4611 | - |

### Full sample: accuracy % per slice

| slice (n) | glm-5.3_cloud__think_plus16k_retry | gpt-oss_120b-cloud | glm-5.3_cloud__think | kimi-k3_cloud | deepseek-v4.1-flash_cloud | deepseek-v4-pro_cloud | qwen3_8b | qwen3_4b |
|---|---|---|---|---|---|---|---|---|
| apistyle (6) | 100.0 | 66.7 | 83.3 | 100.0 | 83.3 | 83.3 | 66.7 | 50.0 |
| authn (12) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 83.3 | 50.0 |
| batching (11) | 90.9 | 81.8 | 81.8 | 18.2 | 18.2 | 0.0 | 0.0 | 27.3 |
| datastore (12) | 100.0 | 91.7 | 91.7 | 91.7 | 66.7 | 58.3 | 50.0 | 58.3 |
| deploystrategy (12) | 91.7 | 91.7 | 91.7 | 83.3 | 66.7 | 50.0 | 58.3 | 25.0 |
| indexing (12) | 100.0 | 83.3 | 100.0 | 0.0 | 66.7 | 41.7 | 16.7 | 0.0 |
| messaging (10) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 90.0 | 70.0 | 0.0 |
| mlserving (2) | 50.0 | 50.0 | 50.0 | 100.0 | 50.0 | 50.0 | 50.0 | 50.0 |
| netcomponent (6) | 100.0 | 100.0 | 66.7 | 100.0 | 83.3 | 83.3 | 66.7 | 33.3 |
| quorum (14) | 92.9 | 92.9 | 78.6 | 21.4 | 35.7 | 14.3 | 7.1 | 28.6 |
| retry (7) | 100.0 | 100.0 | 100.0 | 28.6 | 28.6 | 28.6 | 28.6 | 14.3 |
| rollout (12) | 100.0 | 100.0 | 100.0 | 91.7 | 75.0 | 91.7 | 75.0 | 0.0 |
| routing (5) | 100.0 | 100.0 | 100.0 | 60.0 | 60.0 | 60.0 | 20.0 | 0.0 |
| scaling (12) | 100.0 | 58.3 | 83.3 | 100.0 | 100.0 | 83.3 | 83.3 | 75.0 |
| scheduling (12) | 100.0 | 100.0 | 100.0 | 83.3 | 83.3 | 58.3 | 25.0 | 33.3 |
| security (12) | 100.0 | 91.7 | 100.0 | 100.0 | 100.0 | 91.7 | 91.7 | 58.3 |
| sizing (24) | 100.0 | 100.0 | 100.0 | 91.7 | 79.2 | 91.7 | 62.5 | 25.0 |
| structure (12) | 83.3 | 83.3 | 66.7 | 91.7 | 91.7 | 75.0 | 83.3 | 58.3 |
| swe_file (5) | 80.0 | 80.0 | 80.0 | 100.0 | 100.0 | 80.0 | 60.0 | 20.0 |
| swe_file_ood (5) | 60.0 | 0.0 | 40.0 | 40.0 | 40.0 | 20.0 | 0.0 | 80.0 |
| swe_function (5) | 60.0 | 100.0 | 40.0 | 100.0 | 100.0 | 80.0 | 100.0 | 60.0 |
| swe_subsystem (5) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 60.0 |
| swe_tests (4) | 50.0 | 75.0 | 50.0 | 75.0 | 75.0 | 75.0 | 50.0 | 0.0 |
| tetris (12) | 25.0 | 8.3 | 0.0 | 16.7 | 16.7 | 16.7 | 9.1 | 0.0 |
| vectorindex (11) | 36.4 | 45.5 | 18.2 | 100.0 | 54.5 | 72.7 | 63.6 | 0.0 |

### Pairwise agreement (full sample)

| pair | agreement % | n agree | acc when agree % | 95% CI | n disagree | at least one right when they disagree % |
|---|---|---|---|---|---|---|
| gpt-oss_120b-cloud + glm-5.3_cloud__think | 76.7 | 184 | 97.3 | 94.6-99.5 | 56 | 58.9 |
| gpt-oss_120b-cloud + kimi-k3_cloud | 66.7 | 160 | 97.5 | 95.0-99.4 | 80 | 80.0 |
| gpt-oss_120b-cloud + deepseek-v4.1-flash_cloud | 67.9 | 163 | 93.9 | 90.2-97.5 | 77 | 80.5 |
| gpt-oss_120b-cloud + deepseek-v4-pro_cloud | 64.2 | 154 | 90.9 | 86.4-95.5 | 86 | 83.7 |
| gpt-oss_120b-cloud + qwen3_8b | 53.1 | 127 | 90.5 | 85.0-95.3 | 112 | 83.9 |
| gpt-oss_120b-cloud + qwen3_4b | 32.1 | 77 | 83.1 | 74.0-90.9 | 163 | 88.3 |
| glm-5.3_cloud__think + kimi-k3_cloud | 63.7 | 153 | 98.7 | 96.7-100.0 | 87 | 79.3 |
| glm-5.3_cloud__think + deepseek-v4.1-flash_cloud | 64.2 | 154 | 97.4 | 94.8-99.4 | 86 | 73.3 |
| glm-5.3_cloud__think + deepseek-v4-pro_cloud | 60.0 | 144 | 97.2 | 94.4-99.3 | 96 | 69.8 |
| glm-5.3_cloud__think + qwen3_8b | 48.5 | 116 | 95.7 | 91.4-99.1 | 123 | 78.9 |
| glm-5.3_cloud__think + qwen3_4b | 25.8 | 62 | 98.4 | 95.2-100.0 | 178 | 81.5 |
| kimi-k3_cloud + deepseek-v4.1-flash_cloud | 70.0 | 168 | 89.9 | 85.1-94.0 | 72 | 63.9 |
| kimi-k3_cloud + deepseek-v4-pro_cloud | 68.8 | 165 | 85.5 | 80.0-90.3 | 75 | 66.7 |
| kimi-k3_cloud + qwen3_8b | 59.0 | 141 | 84.4 | 78.7-90.1 | 98 | 67.3 |
| kimi-k3_cloud + qwen3_4b | 29.2 | 70 | 91.4 | 84.3-97.1 | 170 | 72.9 |
| deepseek-v4.1-flash_cloud + deepseek-v4-pro_cloud | 68.8 | 165 | 84.9 | 79.4-90.3 | 75 | 58.7 |
| deepseek-v4.1-flash_cloud + qwen3_8b | 60.7 | 145 | 80.7 | 74.5-86.9 | 94 | 66.0 |
| deepseek-v4.1-flash_cloud + qwen3_4b | 29.2 | 70 | 92.9 | 85.7-98.6 | 170 | 67.1 |
| deepseek-v4-pro_cloud + qwen3_8b | 66.5 | 159 | 73.6 | 66.7-80.5 | 80 | 57.5 |
| deepseek-v4-pro_cloud + qwen3_4b | 24.6 | 59 | 89.8 | 81.4-96.6 | 181 | 67.4 |
| qwen3_8b + qwen3_4b | 22.6 | 54 | 85.2 | 75.9-94.4 | 185 | 58.4 |

### Top-2 agreement per track (gpt-oss_120b-cloud + glm-5.3_cloud__think)

| track | n | n agree | acc when agree % |
|---|---|---|---|
| api | 24 | 20 | 100.0 |
| backend | 24 | 14 | 100.0 |
| cloud | 24 | 24 | 100.0 |
| database | 24 | 21 | 100.0 |
| devops | 24 | 22 | 100.0 |
| distributed | 24 | 20 | 100.0 |
| mlinfra | 24 | 14 | 71.4 |
| security | 24 | 23 | 100.0 |
| software_repair | 24 | 14 | 92.9 |
| state_action | 24 | 12 | 100.0 |
