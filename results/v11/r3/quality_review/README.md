# Quality review of the R3 generators with an Ollama model (reviewer only — never a label source)

Script: `scripts/v11/ollama_review.py` · model `gpt-oss:120b-cloud` (thinking: high) · prompt version `r3-review-1` ·
sample: 539 raw-rendered items of `val_r3.jsonl` as first built (up to 14 per generator and question type) · files
`review_r3a.jsonl` (per-item provenance), `summary_r3a.json`, `disagreements_r3a.json`.

Agreement with the computed labels: **516 / 539 (95.7%)**. The 23 disagreements, looked at one by one:

| group | n | finding | action |
|---|---:|---|---|
| `num_capacity` choice | 9 | the reviewer answered with the NUMBER of units instead of the candidate index (its number was the labelled one) | none — artefact of the review prompt |
| `num_limits` ordinal | 11 | same artefact: answered the count instead of the level index (the counts match the labels) | none |
| `mix_sla_lookup` | 2 | **generator bug**: a reply more than 24 h later was printed as "the next day" | fixed in `capability._hm` ("N days later"); benchmark extension and training corpus rebuilt after the fix |
| `datastore` (rule table) | 1 | the reviewer preferred a different store; the label follows the reviewed table | none |

Excluding the two prompt artefacts: 516 / 519 = 99.4%. No reviewer answer was used as a label.
