# Ollama "technical-decision teacher" bake-off (side study)

Date: 2026-10-01 (BRT). Everything ran on the laptop through the local Ollama server 0.35.0 (`http://localhost:11434`); cloud models were reached through that same server. No pod, no `protected.jsonl`, nothing written outside this folder.

**Scope warning.** Every number here is accuracy against the dataset's own `gold` / `answerable` on rule-table, simulator, test-execution and Tetris items. It measures whether a model can reproduce a verifiable answer from candidate *names only* (no `feat`, no `value`). It may not transfer to open-ended technical judgement, and no model output was used as truth anywhere.

## 1. What responded

| model | where | status |
|---|---|---|
| qwen3:4b | local, 100% GPU | ran, full 240 |
| qwen3:8b | local, 48% CPU / 52% GPU | ran, 239 of 240 (10-minute cap hit one item short) |
| gpt-oss:20b | local, 78% CPU / 22% GPU | ran on the 60-subset only, 59 of 60 (about 22 s per example) |
| qwen3-coder:30b | local | not run (18 GB, would be slower than gpt-oss:20b; dropped for time) |
| gpt-oss:120b-cloud | cloud | ran, full 240 (thinking cannot be turned off; used `think: "low"`) + `think: "high"` on the 60-subset |
| glm-5.3:cloud | cloud | ran, full 240 **with thinking on only** (see failures) |
| kimi-k3:cloud | cloud | ran, full 240 without thinking + thinking on the 60-subset |
| deepseek-v4-pro:cloud | cloud | ran, full 240 without thinking |
| deepseek-v4.1-flash:cloud | cloud | ran, full 240 without thinking |
| mistral-large-3:cloud | cloud | 404 "model not found" on the 1-token probe (it is listed on ollama.com) |
| minimax-m3, nemotron-3-ultra, glm-5.3-flash, gemma4:31b, kimi-k2.7-code | cloud | answered the 1-token probe; not evaluated (request budget) |

Cloud access: `POST /api/me` reports the account signed in with plan `pro`. Pro allows 3 concurrent cloud requests (ollama.com/pricing); I initially ran 15 workers and got some `429 too many concurrent requests`, all retried successfully.

## 2. Method

- Sample: `sample_ids.json`, seed 20261010, 240 choice-type examples from `val.jsonl`, 24 per track, 40 unanswerable, 30 constraint-flip pairs (both members). `software_repair` uses render `feat` shown as "name: path"; all other tracks use render `raw`, names only. `devops` has no unanswerable items and `devops`/`security`/`software_repair` have no flip pairs in the raw pool. A fixed 60-example subset (6 per track, 9 unanswerable, 7 flip pairs) is in the same file.
- Slice families: `tetris`; `swe_*`; `simulator` = `evidence.type == "simulation"` (batching, indexing, quorum, retry, rollout, routing, scheduling); `rule_table` = the rest (includes `sizing`).
- Prompt: system instruction + task + context + numbered candidates + "0. none of the candidates is acceptable"; JSON schema `{"choice": int}` via `format`, temperature 0, seed fixed. `num_predict` 64 (local qwen3), 256 (cloud no-thinking), 1500 (gpt-oss low), 4000 (glm / kimi thinking), 6000 (gpt-oss high).
- Scoring: an answer is correct only if it is valid JSON with an in-range index equal to gold (0 for unanswerable). Invalid or truncated output counts as wrong.
- CIs: 95% percentile bootstrap, 5,000 resamples.
- `glm-5.3_cloud__think_plus16k_retry` is not a separate model: it is the glm-5.3 thinking run where 33 of the 39 items that produced no valid JSON at the 4,000-token cap were retried once with a 16,000-token cap (the other 6, all Tetris, were not retried — time cap). Its tokens and latency include both attempts. Treat it as "glm-5.3 with a retry policy", not as a like-for-like row.

## 3. Findings

1. **Best single runs on the full sample:** gpt-oss:120b-cloud (low thinking) 82.5% [77.5-87.1] and glm-5.3 with thinking 80.4% [75.4-85.4] (87.9% [83.8-92.1] with the 16k retry). The intervals overlap; the two are not separable on 240 items.
2. **Thinking is what matters on simulator items.** Without thinking, kimi-k3 / deepseek-v4-pro / deepseek-v4.1-flash score 42.5% / 41.1% / 53.4% on simulators, against 93.2% for both thinking runs. On rule tables the picture reverses: kimi-k3 without thinking is the best at 95.4%.
3. **Thinking variant, same 60 items:** kimi-k3 75.0% -> 90.0% (12 items fixed, 3 broken, 4 truncated at 4,000 tokens; 96.4% on the 56 valid answers). gpt-oss:120b low -> high: 83.3% -> 83.3% (4 fixed, 4 broken, 7 truncated at 6,000 tokens) at about 7x the cost. n=60, so these intervals are wide.
4. **Constraint-flip pairs, both correct:** 83.3% gpt-oss:120b, 80.0% glm-5.3 thinking, 46.7% kimi-k3 without thinking, 36.7% deepseek-v4-pro, 26.7% deepseek-v4.1-flash, 16.7% qwen3:8b, 3.3% qwen3:4b. The non-thinking models mostly do not react to the changed constraint.
5. **Unanswerable items:** best single run 72.5-75.0% (92.5% for glm with retry). qwen3:4b's 90.0% is an artifact: it answers "none" on 79.6% of all items, including 77.5% of the answerable ones.
6. **Agreement of the two best (gpt-oss:120b-cloud + glm-5.3 thinking):** they agree on 184 of 240 (76.7%); when they agree they are right 97.3% of the time [94.6-99.5]. When they disagree (56 items), at least one is right in only 58.9%. Per track the agreed answers were 100% correct on 8 tracks, 92.9% on software_repair (14 agreed) and 71.4% on mlinfra (14 agreed). All three of gpt-oss:120b + glm-5.3 thinking + kimi-k3 agree on 142 (59.2%), 99.3% correct [97.9-100].
7. **Local models are not usable as judges here:** qwen3:8b 52.7% [46.4-59.4], qwen3:4b 30.8% [25.0-36.7]. gpt-oss:20b reached 78.0% [67.8-88.1] on 59 subset items but at a mean of 21.8 s per example on this laptop (about 60 hours per 10,000).

## 4. Where no model is reliable enough to be a judge

Per-track cells have n=24 (roughly +-15-20 points), so read these as "clearly bad" rather than as precise rates.

- **state_action: best 62.5%** (glm with retry; 54.2% best single run). The track is half Tetris, where every model is at 0-25% (chance at K=8 is 12.5%); scheduling alone is 100% for the thinking runs.
- **mlinfra: best 62.5%.** Driven by `vectorindex`, where the thinking models score 18-45% while kimi-k3 without thinking scores 100% (11 items), and by `batching` for the non-thinking cloud models (0-18%). Agreement of the top two is also unreliable here (71.4%).
- **software_repair: best 83.3%** (kimi-k3 and deepseek-v4.1-flash, no thinking), 70.8% for the top two overall. `swe_file_ood` (the correct answer is "none") is 0-60% for every model except qwen3:4b (80%, because it says "none" to almost everything): models pick a plausible file instead of rejecting.
- **Unanswerable detection in general** stays at or below 75% for every single run (qwen3:4b's 90% is the say-none artifact above).

Tracks where the best run was at or above 95.8%: api, cloud, database, devops, distributed, security, and backend (backend only via kimi-k3 / deepseek-v4.1-flash without thinking; the thinking runs were 70.8-91.7%).

## 5. Recommendation

- **Cheap local transformations** (reformatting, paraphrase, field extraction — not judging): qwen3:8b. It gave 100% valid JSON at a 3.4 s median. Do not use it, or qwen3:4b, for any decision: 52.7% and 30.8%. gpt-oss:20b is the only local model with usable accuracy but is too slow on this hardware.
- **Hard decisions / default cloud judge:** gpt-oss:120b-cloud with low thinking — best single-run accuracy, 99.2% valid JSON, about $1.7 and about 1.1 h per 10,000 examples at 3 concurrent slots.
- **Disagreement mining / quality review:** pair gpt-oss:120b-cloud with glm-5.3 thinking (or kimi-k3 thinking; untested on the full sample). Agreement is a usable filter on 8 of 10 tracks (97.3% overall), and the 23% they disagree on is the set worth mining: neither model is right in 41% of it. glm-5.3 needs a retry-on-no-JSON policy and costs about $52-106 per 10,000.
- **Do not use any of these as a judge** for Tetris / state_action, mlinfra (vectorindex), software_repair out-of-distribution rejection, or "is none of these acceptable?" decisions. Keep the simulators and rule tables as truth there.
- Not established: kimi-k3 with thinking looks strongest on the 60-subset (90.0%, 96.4% on valid answers) but was not run on the full 240 because it would have exceeded the request budget; it is also the most expensive (about $109 per 10,000).

## 6. Failures and things not verified

- **glm-5.3:cloud ignores `format` when `think: false`:** it writes its reasoning into `content` and never emits the JSON (45 requests, 0 valid; saved as `logs/aborted_glm-5.3_cloud_schema_nothink.jsonl`). With `think: true` it returns valid JSON, so it was only evaluated as a thinking model.
- **glm-5.3 thinking ran out of tokens on 37 of 240** at the 4,000 cap (all 12 Tetris items among them); 10 of the 33 retried still did not finish at 16,000. Its thinking length is not stable between attempts at temperature 0.
- **deepseek-v4.1-flash returned invalid output on 5.0%** (prose instead of JSON, or an out-of-range index such as 32 for 8 candidates). The schema is not strictly enforced for cloud models.
- **gpt-oss cannot disable thinking;** "low" was used as its cheap setting. Truncation: 2 of 240 at low, 7 of 60 at high.
- **Latency:** the four no-thinking cloud runs shared 3 slots across 12-15 workers, so their recorded median/p95 (about 4 s / 9 s) include queueing. The "clean sequential" column is a separate 8-request probe per model and is what the time-per-10k estimates use for those models. Eight requests is a thin basis.
- **Cost** is tokens x the list prices on ollama.com/pricing as fetched today, assuming thinking tokens are billed as output (the page does not say). I did not check the account's usage page, so actual charges are unverified. Estimated spend of this study: about $3.8, about 1,490 cloud requests including retries and probes.
- qwen3:8b is missing 1 of 240 items and gpt-oss:20b 1 of 60 (time caps). The session restarted during the gpt-oss:20b run; completed items were kept, nothing was re-run.
- Local runs used Ollama's default 4,096-token context; the longest local prompt was 992 tokens, so nothing was truncated on input.
- Sample sizes: 240 overall, 24 per track, 12 Tetris, 24 SWE. Per-slice numbers (2-24 items) are indicative only.

## 7. Tables (generated by `analyze.py`; same numbers in `summary.json`)

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
