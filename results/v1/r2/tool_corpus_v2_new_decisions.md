# Krun Decisions v1 — stats (286,516 training decisions)


## Task bucket

| value | decisions | share |
|---|---:|---:|
| tool_routing | 207,849 | 72.5% |
| ood | 46,153 | 16.1% |
| noul | 18,922 | 6.6% |
| multi_label | 13,592 | 4.7% |

## Origin

| value | decisions | share |
|---|---:|---:|
| derived | 109,221 | 38.1% |
| mt | 106,104 | 37.0% |
| real | 71,191 | 24.8% |

## Context language

| value | decisions | share |
|---|---:|---:|
| EN | 180,412 | 63.0% |
| PT-BR | 41,092 | 14.3% |
| FR | 32,513 | 11.3% |
| ES | 32,499 | 11.3% |

## Locale

| value | decisions | share |
|---|---:|---:|
| en-US | 180,412 | 63.0% |
| pt-BR | 41,092 | 14.3% |
| fr-FR | 32,513 | 11.3% |
| es-419 | 32,499 | 11.3% |

## Curriculum stage

| value | decisions | share |
|---|---:|---:|
| D3 | 125,022 | 43.6% |
| D1 | 91,521 | 31.9% |
| D2 | 69,973 | 24.4% |

## Question type

| value | decisions | share |
|---|---:|---:|
| choice | 254,002 | 88.7% |
| noul | 18,922 | 6.6% |
| multi | 13,592 | 4.7% |

## Candidate count (choice/multi)

| value | decisions | share |
|---|---:|---:|
| <=4 | 77,648 | 29.0% |
| <=8 | 62,387 | 23.3% |
| <=16 | 51,210 | 19.1% |
| <=2 | 29,012 | 10.8% |
| <=32 | 28,113 | 10.5% |
| <=64 | 15,924 | 6.0% |
| <=128 | 3,300 | 1.2% |

## OOD type

| value | decisions | share |
|---|---:|---:|
| gold_removed | 38,140 | 82.6% |
| no_available_tool | 5,949 | 12.9% |
| native_irrelevance | 2,064 | 4.5% |

## Difficulty

| value | decisions | share |
|---|---:|---:|
| hard | 266,071 | 92.9% |
| medium | 20,445 | 7.1% |

## Source dataset

| value | decisions | share |
|---|---:|---:|
| xlam | 120,143 | 41.9% |
| toolmind | 66,518 | 23.2% |
| apibench | 54,000 | 18.8% |
| when2call | 38,999 | 13.6% |
| johin | 3,000 | 1.0% |
| xlam_irrelevance | 2,064 | 0.7% |
| hermes | 1,792 | 0.6% |

## Cross-language

Non-EN decisions: 106,104. Of these, **100.0% are cross-language** and **100.0% use an EN schema**.

| context → schema | decisions |
|---|---:|
| EN->en | 180,412 |
| PT-BR->en | 41,092 |
| FR->en | 32,513 |
| ES->en | 32,499 |

## Category × origin

| category | origin | decisions |
|---|---|---:|
| multi_label | derived | 4,460 |
| multi_label | mt | 3,500 |
| multi_label | real | 5,632 |
| noul | derived | 11,380 |
| noul | mt | 7,542 |
| ood | derived | 25,337 |
| ood | mt | 19,700 |
| ood | real | 1,116 |
| tool_routing | derived | 68,044 |
| tool_routing | mt | 75,362 |
| tool_routing | real | 64,443 |

## Category × language

| category | language | decisions |
|---|---|---:|
| multi_label | EN | 10,092 |
| multi_label | ES | 1,000 |
| multi_label | FR | 1,000 |
| multi_label | PT-BR | 1,500 |
| noul | EN | 11,380 |
| noul | ES | 2,282 |
| noul | FR | 2,285 |
| noul | PT-BR | 2,975 |
| ood | EN | 26,453 |
| ood | ES | 6,143 |
| ood | FR | 6,142 |
| ood | PT-BR | 7,415 |
| tool_routing | EN | 132,487 |
| tool_routing | ES | 23,074 |
| tool_routing | FR | 23,086 |
| tool_routing | PT-BR | 29,202 |

## Slices

| slice | decisions |
|---|---:|
| same_language | 180,412 |
| near_duplicate | 145,986 |
| cross_language | 106,104 |
| gold_removed | 38,140 |
| ood_near | 38,140 |
| native_irrelevance | 12,333 |

## Tool routing block

Tool decisions (family tool, choice/multi): 242,829. ToolACE unique query groups 0, unique gold tools 0, unique candidate tools 0.

| tool source | origin | decisions |
|---|---|---:|
| apibench | derived | 21,232 |
| apibench | mt | 19,000 |
| apibench | real | 13,768 |
| hermes | derived | 1,344 |
| hermes | real | 448 |
| johin | real | 3,000 |
| toolmind | derived | 22,711 |
| toolmind | mt | 20,269 |
| toolmind | real | 23,538 |
| when2call | derived | 4,731 |
| when2call | mt | 6,492 |
| xlam_irrelevance | derived | 1,020 |
| xlam_irrelevance | real | 1,044 |
| xlam | derived | 40,896 |
| xlam | mt | 41,393 |
| xlam | real | 21,943 |

| flag | decisions | share of tool decisions |
|---|---:|---:|
| near_duplicate | 145,986 | 60.1% |
| description_dependent | 75,520 | 31.1% |
| ood | 46,153 | 19.0% |

| language | tool decisions |
|---|---:|
| EN | 155,675 |
| PT-BR | 33,662 |
| FR | 26,749 |
| ES | 26,743 |

Minimal-pair sets: 0 ({}).
Synthetic (LLM) transformations: {}.
Approx. text tokens (chars/3.6, context + candidates): 126,509,648.

## Validation (val_ml)

| part / language / family | decisions |
|---|---:|
| eval|EN|proposition | 567 |
| eval|EN|tool | 7,701 |
| eval|EN|tool_action | 668 |
| eval|ES|proposition | 117 |
| eval|ES|tool | 1,347 |
| eval|ES|tool_action | 191 |
| eval|FR|proposition | 130 |
| eval|FR|tool | 1,328 |
| eval|FR|tool_action | 195 |
| eval|PT-BR|proposition | 146 |
| eval|PT-BR|tool | 1,696 |
| eval|PT-BR|tool_action | 239 |
| fit|EN|proposition | 554 |
| fit|EN|tool | 7,879 |
| fit|EN|tool_action | 653 |
| fit|ES|proposition | 113 |
| fit|ES|tool | 1,353 |
| fit|ES|tool_action | 159 |
| fit|FR|proposition | 100 |
| fit|FR|tool | 1,372 |
| fit|FR|tool_action | 155 |
| fit|PT-BR|proposition | 154 |
| fit|PT-BR|tool | 1,704 |
| fit|PT-BR|tool_action | 211 |
