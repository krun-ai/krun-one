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

