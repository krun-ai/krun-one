# Krun Decisions v1 — stats (1,042,946 training decisions)


## Task bucket

| value | decisions | share |
|---|---:|---:|
| tool_routing | 316,901 | 30.4% |
| choice_intent | 282,980 | 27.1% |
| ood | 197,339 | 18.9% |
| noul | 99,854 | 9.6% |
| score | 79,998 | 7.7% |
| multi_label | 65,874 | 6.3% |

## Origin

| value | decisions | share |
|---|---:|---:|
| mt | 435,701 | 41.8% |
| derived | 337,458 | 32.4% |
| real | 236,123 | 22.6% |
| rules | 33,043 | 3.2% |
| llm | 621 | 0.1% |

## Context language

| value | decisions | share |
|---|---:|---:|
| EN | 452,006 | 43.3% |
| PT-BR | 196,640 | 18.9% |
| ES | 181,991 | 17.4% |
| FR | 176,597 | 16.9% |
| PT-PT | 35,712 | 3.4% |

## Locale

| value | decisions | share |
|---|---:|---:|
| en-US | 447,014 | 42.9% |
| pt-BR | 196,640 | 18.9% |
| fr-FR | 176,553 | 16.9% |
| es-419 | 144,989 | 13.9% |
| es-ES | 36,965 | 3.5% |
| pt-PT | 35,712 | 3.4% |
| en | 4,992 | 0.5% |
| fr | 44 | 0.0% |
| es | 37 | 0.0% |

## Curriculum stage

| value | decisions | share |
|---|---:|---:|
| D2 | 424,330 | 40.7% |
| D3 | 313,730 | 30.1% |
| D1 | 304,886 | 29.2% |

## Question type

| value | decisions | share |
|---|---:|---:|
| choice | 791,236 | 75.9% |
| noul | 99,854 | 9.6% |
| score | 79,998 | 7.7% |
| multi | 71,858 | 6.9% |

## Candidate count (choice/multi)

| value | decisions | share |
|---|---:|---:|
| <=8 | 264,852 | 30.7% |
| <=4 | 196,872 | 22.8% |
| <=16 | 184,125 | 21.3% |
| <=2 | 93,041 | 10.8% |
| <=32 | 80,749 | 9.4% |
| <=64 | 36,437 | 4.2% |
| <=128 | 7,018 | 0.8% |

## OOD type

| value | decisions | share |
|---|---:|---:|
| gold_removed | 127,738 | 64.7% |
| cross_domain | 34,988 | 17.7% |
| cross_task | 19,964 | 10.1% |
| native_irrelevance | 8,500 | 4.3% |
| gold_removed_multi | 5,984 | 3.0% |
| near_ood_synthetic | 165 | 0.1% |

## Difficulty

| value | decisions | share |
|---|---:|---:|
| hard | 665,824 | 63.8% |
| easy | 229,103 | 22.0% |
| medium | 148,019 | 14.2% |

## Source dataset

| value | decisions | share |
|---|---:|---:|
| toolace | 195,281 | 18.7% |
| clinc150 | 181,802 | 17.4% |
| massive | 167,464 | 16.1% |
| bitext | 119,542 | 11.5% |
| banking77 | 90,047 | 8.6% |
| snips | 65,305 | 6.3% |
| civil_comments | 55,712 | 5.3% |
| mhs | 44,165 | 4.2% |
| multiwoz | 38,737 | 3.7% |
| helpsteer2 | 38,339 | 3.7% |
| banking77_pt_br | 35,427 | 3.4% |
| mhs_items | 11,125 | 1.1% |

## Cross-language

Non-EN decisions: 590,940. Of these, **98.3% are cross-language** and **92.2% use an EN schema**.

| context → schema | decisions |
|---|---:|
| EN->en | 447,092 |
| PT-BR->en | 178,557 |
| ES->en | 169,048 |
| FR->en | 163,371 |
| PT-PT->en | 33,602 |
| PT-BR->mul | 14,398 |
| ES->mul | 10,157 |
| FR->mul | 9,917 |
| EN->mul | 4,602 |
| PT-BR->pt | 3,394 |
| FR->fr | 3,309 |
| ES->es | 2,786 |
| PT-PT->mul | 1,350 |
| PT-PT->pt | 697 |
| EN->pt | 312 |
| PT-BR->es | 291 |
| PT-PT->es | 63 |

## Category × origin

| category | origin | decisions |
|---|---|---:|
| choice_intent | derived | 18,603 |
| choice_intent | llm | 340 |
| choice_intent | mt | 140,082 |
| choice_intent | real | 108,082 |
| choice_intent | rules | 15,873 |
| multi_label | derived | 39,828 |
| multi_label | mt | 25,409 |
| multi_label | real | 637 |
| noul | derived | 36,638 |
| noul | mt | 21,873 |
| noul | real | 41,343 |
| ood | derived | 103,309 |
| ood | llm | 165 |
| ood | mt | 92,358 |
| ood | real | 1,507 |
| score | derived | 19,998 |
| score | real | 60,000 |
| tool_routing | derived | 119,082 |
| tool_routing | llm | 116 |
| tool_routing | mt | 155,979 |
| tool_routing | real | 24,554 |
| tool_routing | rules | 17,170 |

## Category × language

| category | language | decisions |
|---|---|---:|
| choice_intent | EN | 80,412 |
| choice_intent | ES | 62,118 |
| choice_intent | FR | 62,084 |
| choice_intent | PT-BR | 65,605 |
| choice_intent | PT-PT | 12,761 |
| multi_label | EN | 37,606 |
| multi_label | ES | 9,424 |
| multi_label | FR | 9,416 |
| multi_label | PT-BR | 8,903 |
| multi_label | PT-PT | 525 |
| noul | EN | 59,363 |
| noul | ES | 14,247 |
| noul | FR | 8,963 |
| noul | PT-BR | 9,967 |
| noul | PT-PT | 7,314 |
| ood | EN | 73,561 |
| ood | ES | 37,945 |
| ood | FR | 37,935 |
| ood | PT-BR | 40,628 |
| ood | PT-PT | 7,270 |
| score | EN | 79,998 |
| tool_routing | EN | 121,066 |
| tool_routing | ES | 58,257 |
| tool_routing | FR | 58,199 |
| tool_routing | PT-BR | 71,537 |
| tool_routing | PT-PT | 7,842 |

## Slices

| slice | decisions |
|---|---:|
| cross_language | 585,668 |
| same_language | 457,278 |
| ood_near | 133,887 |
| gold_removed | 133,722 |
| near_duplicate | 126,364 |
| ood_far | 58,484 |
| typo | 33,043 |
| cross_task | 19,964 |
| minimal_pair | 18,684 |
| native_irrelevance | 8,500 |
| counterfactual | 116 |
| slang | 113 |

## Tool routing block

Tool decisions (family tool, choice/multi): 410,404. ToolACE unique query groups 7,063, unique gold tools 6,763, unique candidate tools 21,424.

| tool source | origin | decisions |
|---|---|---:|
| ? | derived | 8,884 |
| ? | llm | 196 |
| ? | mt | 10,959 |
| ? | real | 100 |
| intent_as_function | derived | 52,724 |
| intent_as_function | mt | 75,979 |
| intent_as_function | real | 912 |
| intent_as_function | rules | 7,804 |
| multiwoz | derived | 18,938 |
| multiwoz | mt | 3,355 |
| multiwoz | real | 17,549 |
| multiwoz | rules | 569 |
| toolace_pool | derived | 11,091 |
| toolace_pool | mt | 8,757 |
| toolace_pool | real | 116 |
| toolace | derived | 76,147 |
| toolace | mt | 100,431 |
| toolace | real | 7,096 |
| toolace | rules | 8,797 |

| flag | decisions | share of tool decisions |
|---|---:|---:|
| description_dependent | 132,269 | 32.2% |
| near_duplicate | 111,495 | 27.2% |
| ood | 89,629 | 21.8% |
| translated_schema | 46,573 | 11.3% |

| language | tool decisions |
|---|---:|
| EN | 162,151 |
| PT-BR | 89,194 |
| FR | 74,729 |
| ES | 74,673 |
| PT-PT | 9,657 |

Minimal-pair sets: 4,376 ({'natural_contrast_mined': 18457, 'label_flip_minimal_edit': 227}).
Synthetic (LLM) transformations: {'counterfactual': 116, 'minimal_pair': 227, 'near_ood': 165, 'slang': 113}.
Approx. text tokens (chars/3.6, context + candidates): 195,865,473.

## Validation (val_ml)

| part / language / family | decisions |
|---|---:|
| dev|ES|intent | 166 |
| dev|ES|tool | 47 |
| dev|FR|intent | 167 |
| dev|FR|tool | 45 |
| dev|PT-BR|intent | 163 |
| dev|PT-BR|tool | 47 |
| eval|EN|intent | 1,565 |
| eval|EN|tool | 827 |
| eval|ES|intent | 1,589 |
| eval|ES|tool | 449 |
| eval|FR|intent | 1,515 |
| eval|FR|tool | 417 |
| eval|PT-BR|intent | 1,344 |
| eval|PT-BR|tool | 389 |
| eval|PT-PT|intent | 678 |
| eval|PT-PT|tool | 229 |
| fit|EN|intent | 1,541 |
| fit|EN|tool | 799 |
| fit|ES|intent | 1,608 |
| fit|ES|tool | 430 |
| fit|FR|intent | 1,456 |
| fit|FR|tool | 417 |
| fit|PT-BR|intent | 1,418 |
| fit|PT-BR|tool | 355 |
| fit|PT-PT|intent | 659 |
| fit|PT-PT|tool | 229 |
