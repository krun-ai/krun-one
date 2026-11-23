LLM cost per 100k accepted: $3.89 (pilot); training on RTX 4090 community ($0.34/h, 13,381 real tok/s measured).

## S1M: target 1,000,000 → supplied 1,000,000 (short 0)

| category | demand | REAL | DERIVED | MT | RULES | LLM | short | hard negatives |
|---|---|---|---|---|---|---|---|---|
| choice_intent | 240,000 | 107,507 | 132,493 | 0 | 0 | 0 | 0 | 156,000 |
| tool_routing | 220,000 | 6,332 | 120,308 | 93,360 | 0 | 0 | 0 | 154,000 |
| noul | 160,000 | 160,000 | 0 | 0 | 0 | 0 | 0 | 0 |
| score | 100,000 | 70,000 | 30,000 | 0 | 0 | 0 | 0 | 0 |
| multi_label | 80,000 | 0 | 56,000 | 0 | 4,800 | 19,200 | 0 | 0 |
| ood | 120,000 | 0 | 84,000 | 0 | 7,200 | 28,800 | 0 | 72,000 |
| minimal_pairs | 80,000 | 0 | 0 | 0 | 0 | 80,000 | 0 | 0 |
| **total** | 1,000,000 | 343,839 | 422,801 | 93,360 | 12,000 | 128,000 | 0 | 382,000 |

Storage 2.33 GB jsonl · padded training tokens 0.40B · train 6.6 h ≈ $2.24 · LLM $4.98 · MT $0.14

## S2M: target 2,000,000 → supplied 2,000,000 (short 0)

| category | demand | REAL | DERIVED | MT | RULES | LLM | short | hard negatives |
|---|---|---|---|---|---|---|---|---|
| choice_intent | 480,000 | 107,507 | 372,493 | 0 | 0 | 0 | 0 | 312,000 |
| tool_routing | 440,000 | 6,332 | 120,308 | 227,952 | 26,400 | 59,008 | 0 | 308,000 |
| noul | 320,000 | 250,000 | 70,000 | 0 | 0 | 0 | 0 | 0 |
| score | 200,000 | 70,000 | 130,000 | 0 | 0 | 0 | 0 | 0 |
| multi_label | 160,000 | 0 | 112,000 | 0 | 9,600 | 38,400 | 0 | 0 |
| ood | 240,000 | 0 | 168,000 | 0 | 14,400 | 57,600 | 0 | 144,000 |
| minimal_pairs | 160,000 | 0 | 0 | 0 | 0 | 160,000 | 0 | 0 |
| **total** | 2,000,000 | 433,839 | 972,801 | 227,952 | 50,400 | 315,008 | 0 | 764,000 |

Storage 4.66 GB jsonl · padded training tokens 0.59B · train 9.9 h ≈ $3.36 · LLM $12.25 · MT $0.34

## S3M: target 3,000,000 → supplied 2,923,191 (short 76,809)

| category | demand | REAL | DERIVED | MT | RULES | LLM | short | hard negatives |
|---|---|---|---|---|---|---|---|---|
| choice_intent | 720,000 | 107,507 | 537,535 | 74,958 | 0 | 0 | 0 | 468,000 |
| tool_routing | 660,000 | 6,332 | 120,308 | 227,952 | 39,600 | 230,999 | 34,809 | 437,633 |
| noul | 480,000 | 250,000 | 230,000 | 0 | 0 | 0 | 0 | 0 |
| score | 300,000 | 70,000 | 140,000 | 0 | 18,000 | 30,000 | 42,000 | 0 |
| multi_label | 240,000 | 0 | 168,000 | 0 | 14,400 | 57,600 | 0 | 0 |
| ood | 360,000 | 0 | 251,999 | 0 | 21,600 | 86,401 | 0 | 216,000 |
| minimal_pairs | 240,000 | 0 | 0 | 0 | 0 | 240,000 | 0 | 0 |
| **total** | 3,000,000 | 433,839 | 1,447,842 | 302,910 | 93,600 | 645,000 | 76,809 | 1,121,633 |

Storage 6.81 GB jsonl · padded training tokens 0.57B · train 9.4 h ≈ $3.2 · LLM $25.07 · MT $0.45

## Context → schema language pairs (S3M supplied)

| pair | decisions |
|---|---|
| en->en | 876,957 |
| pt-BR->en (cross-language) | 210,469 |
| pt-BR->pt-BR (translated schema) | 70,156 |
| pt-BR->en-labels (humanized EN labels) | 187,084 |
| pt-PT->en (cross-language) | 105,234 |
| pt-PT->pt-PT (translated schema) | 35,078 |
| pt-PT->en-labels (humanized EN labels) | 93,542 |
| es-ES->en (cross-language) | 118,389 |
| es-ES->es-ES (translated schema) | 39,463 |
| es-ES->en-labels (humanized EN labels) | 105,234 |
| es-419->en (cross-language) | 157,852 |
| es-419->es-419 (translated schema) | 52,617 |
| es-419->en-labels (humanized EN labels) | 140,313 |
| fr-FR->en (cross-language) | 263,087 |
| fr-FR->fr-FR (translated schema) | 87,695 |
| fr-FR->en-labels (humanized EN labels) | 233,855 |
| fr-CA->en (cross-language) | 65,771 |
| fr-CA->fr-CA (translated schema) | 21,923 |
| fr-CA->en-labels (humanized EN labels) | 58,463 |
