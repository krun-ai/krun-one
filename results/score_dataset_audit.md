# KRUN-011 — Score dataset audit

Score = a distribution over **ordered** levels; the public value is the expected level `Σ i·p_i`. Only genuinely
ordinal targets are used (ratings / Likert intensities); no nominal classes were turned into levels. Level texts
are semantic sentences, never bare `"0","1","2"`. Build: `scripts/prepare_primitives_data.py`.
**No synthetic or LLM-labelled examples.**

## 1. Candidates

| dataset | HF id @ revision | license | size | ordinal target | levels | domain / lang | commercial use | decision |
|---|---|---|---|---|---|---|---|---|
| **HelpSteer2** (NVIDIA × Scale AI) | `nvidia/HelpSteer2` @ `990b2711` | CC-BY-4.0 | 20,324 train + 1,038 val responses × 5 attributes | human Likert 0–4: helpfulness, correctness, coherence, complexity, verbosity | 5 | response quality, EN | ✅ | **train / val / test** |
| **Measuring Hate Speech** (UC Berkeley D-Lab) | `ucberkeley-dlab/measuring-hate-speech` @ `5468f6e1` | CC-BY-4.0 | 39,565 comments, 135,556 annotations | 9 five-point items (sentiment, respect, insult, humiliate, status, dehumanize, violence, genocide, attack/defend) | 5 | moderation / severity, EN | ✅ | **train / val / test** |
| SST-5 | `SetFit/sst5` @ `e51bdcd8` | not stated (SST: unknown) | 2,210 test sentences | sentiment 0–4 | 5 | movie reviews, EN | ⚠️ unclear | **test only (external)**, never trained on |
| HelpSteer (v1) | `nvidia/HelpSteer` | CC-BY-4.0 | 37k | same attributes | 5 | EN | ✅ | not used (prompt overlap risk with v2; v2 enough) |
| Feedback-Collection (Prometheus) | `prometheus-eval/Feedback-Collection` | CC-BY-4.0 | 100k | 1–5 rubric scores | 5 | GPT-4 generated | ⚠️ synthetic + OpenAI terms | ❌ (synthetic; priority 4 not needed) |
| UltraFeedback | `openbmb/UltraFeedback` | MIT | 64k | 1–5 ratings | 5 | GPT-4 annotated | ⚠️ synthetic labels | ❌ |
| customer-support-tickets | `Tobi-Bueck/customer-support-tickets` | CC-BY-NC-4.0 | 20k | priority | 3 | support | ❌ non-commercial | ❌ |
| STS-B | `mteb/stsbenchmark-sts` | unknown | 8.6k | similarity 0–5 | cont. | pairs | ⚠️ | ❌ (pairs + unclear license) |
| Yelp / Amazon reviews | `Yelp/yelp_review_full`, `McAuley-Lab/Amazon-Reviews-2023` | "other" / none | large | stars | 5 | reviews | ⚠️ restrictive ToS | ❌ |
| SICK relatedness | — | CC-BY-NC-SA | — | 1–5 | — | — | ❌ NC | ❌ |

## 2. Construction

**HelpSteer2.** One score question per (response, attribute). Context = `"Prompt:\n{prompt tail ≤1,500 chars}\n\n
Response:\n{response head ≤2,500 chars}"` so the response is always inside the 1,024-token budget. Instructions and
5 level texts per attribute written by us from the dataset card's attribute definitions (e.g. correctness:
"Completely incorrect or missing the pertinent facts" … "Fully correct and complete"). Split: official `validation`
→ **test**; official `train` split **by prompt** (sha256 group) into train (90%) / val (10%); no prompt crosses
splits (verified: 0 overlaps).

**Measuring Hate Speech.** One score question per (comment, item); 9 items (the 10th, `hatespeech`, is 0/1/2 with
"unsure" in the middle → used as **noul** instead). Target = **median of annotators, rounded half-up**; raw
annotations kept in `metadata.annotations`. Items are coded "higher = more hateful" (Kennedy et al. 2020); the exact
survey wording is not in the dataset card or the paper text, so the level texts are **ours** and follow that
direction (e.g. insult: "Not insulting at all" … "Extremely insulting"). Split by `comment_id` 80/10/10.

**SST-5 (external).** Test split only: "What is the sentiment of this movie review sentence?", levels Very negative …
Very positive. Laya's README reports SST-5 (0.372) and lists "sentiment and rating" among its training families, so
SST-5 may be in-distribution for Laya.

## 3. Sizes (`score_{train,val,test}.jsonl`)

| split | total | HelpSteer2 | MHS | SST-5 | label distribution (0/1/2/3/4) |
|---|---:|---:|---:|---:|---|
| train pool | 60,000 | 30,000 (≈6k per attribute) | 30,000 (≈3.3k per item) | — | 6,986 / 8,583 / 15,092 / 13,946 / 15,393 |
| val | 6,000 | 3,000 | 3,000 | — | 700 / 841 / 1,509 / 1,393 / 1,557 |
| test | 8,210 | 3,000 | 3,000 | 2,210 | 984 / 1,449 / 1,917 / 1,886 / 1,974 |

Mixtures use 34,000 (M50) or 51,000 (M40) of the pool. All levels in this sprint are 5-level scales; score requests
with 2–16 levels are serialized identically, but only K = 5 is trained/validated here (known limitation).

## 4. Limitations

* HelpSteer2 labels are skewed (coherence: 72% level 4); MAE/Spearman are reported per attribute.
* MHS median of 1–3 annotators is noisy; items are highly correlated (violence ↔ genocide 0.93).
* No customer-support "severity/urgency" dataset with a usable license exists → the product example (ticket
  severity) is not directly trained; transfer relies on the semantic level texts.
