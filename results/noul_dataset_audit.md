# KRUN-011 — Noul dataset audit

Noul = a yes/no proposition about the context, answered with `p_true ∈ [0,1]`. Supervised targets are only
`1` (true) and `0` (false); **no 0.5 targets**: ambiguous items are left out, never labelled "uncertain".
Build: `scripts/prepare_primitives_data.py` + `configs/primitives_data.yaml` (seeded, pinned revisions);
counts and sha256 in `results/primitives_dataset_stats.json`. **No synthetic or LLM-generated examples.**

## 1. Sources audited

| source | HF id @ revision | license | used for | decision |
|---|---|---|---|---|
| BANKING77 (via KrunBench v0 / Krun One v0 splits) | `legacy-datasets/banking77` @ `f5412156` | CC-BY-4.0 | train / val / test | ✅ deterministic transformation (propositions) |
| MASSIVE en-US / pt-PT / es-ES (same) | `AmazonScience/massive` (KrunBench v0 pin) | CC-BY-4.0 | train / val / test | ✅ same |
| xLAM function-calling-60k (via `argilla/apigen-function-calling`) | @ `170a8a7e` | CC-BY-4.0 | train / val / test | ✅ "should function X be called?" |
| ToolACE | `Team-ACE/ToolACE` @ `6bda777c` | Apache-2.0 | train / val / test | ✅ same |
| xlam-irrelevance-7.5k | `MadeAgents/xlam-irrelevance-7.5k` @ `34323bf0` | CC-BY-4.0 | train / val / test | ✅ natively irrelevant tools → target 0 |
| Civil Comments (Jigsaw) | `google/civil_comments` @ `f2970eb3` | CC0-1.0 | train / val / test (official splits) | ✅ real crowd-rated binary properties |
| Measuring Hate Speech (item `hatespeech`) | `ucberkeley-dlab/measuring-hate-speech` @ `5468f6e1` | CC-BY-4.0 | train / val / test (by comment_id) | ✅ |
| BoolQ | `google/boolq` @ `35b264d0` (validation) | CC-BY-SA-3.0 | **test only (external)** | ⚠️ eval only — share-alike; never in training/artifact |
| PAWS | `google-research-datasets/paws` | "other" (free use, attribution appreciated) | — | not used (paraphrase pairs don't fit the single-context API) |
| SNLI / MultiNLI | `stanfordnlp/snli`, `nyu-mll/multi_nli` | CC-BY-SA-4.0 / mixed (incl. "other") | — | ❌ share-alike / mixed-license genres |
| Amazon polarity, IMDB, Yelp | various | card says Apache-2.0 / "other" but upstream ToS unclear | — | ❌ ambiguous upstream terms |
| TypeSafe/Laya "typed-decisions" | `convaiinnovations/typed-decisions` | not accessible (401) | — | ❌ |

HWU64 and BFCL stay excluded from training exactly as in v0.

## 2. Transformations

**Intents (BANKING77, MASSIVE).** Every label got a hand-written plain-language **proposition**
(`configs/noul_propositions.yaml`, e.g. `card_arrival` → *"Is the customer waiting for a card that has not arrived
yet?"*) — never "Is label X?". Four labels have no precise proposition and are skipped (`general_quirky`,
`qa_factoid`, `cooking_query`, `audio_volume_other`). Per context (reusing the v0 split of that context):

* **positive**: the proposition of its own label → 1;
* **hard negative**: the proposition of a *confusable* label → 0. Confusable = top-8 labels that v0's hard-negative
  miner (RRF of label / example / kNN-confusion similarity) placed next to the gold in v0 training. A hand-checked
  **overlap list** (`overlaps:` in the same file, 47 entries: e.g. `card_arrival`↔`card_delivery_estimate`,
  `extra_charge_on_statement` vs any fee label, `play_music`↔`play_radio`) forbids pairs where the proposition could
  legitimately be true;
* **cross-domain** (50% of contexts): a proposition from the other dataset (banking ↔ assistant) → 0.

**Tools (xLAM, ToolACE, irrelevance).** Instruction *"Should the function `{name}` be called to fulfil this request?
Function description: {official description}"*. Positive = gold tool; hard negative = another tool of the **same v0
decision** (native tools of the item first — genuinely wrong, same domain); irrelevance items → one of their own
tools → 0. Splits = v0 (train / val_tools / holdout_tools), so unseen tools stay unseen.

**Civil Comments.** Six property propositions (toxicity, insult, obscene, threat, identity_attack,
sexual_explicit), label = rater fraction ≥ 0.5 (the dataset's majority convention). Per proposition: positives;
hard negatives = toxic comments (toxicity ≥ 0.5) whose property fraction is < 0.3 (for `toxicity` itself:
0 < toxicity ≤ 0.3); cross-domain / easy = comments no rater found toxic. **`identity_attack` is never trained on**
→ unseen-proposition slice in val/test.

**MHS `hatespeech`.** Annotators answer 0 no / 1 unsure / 2 yes: mean ≥ 1.5 → 1, mean ≤ 0.5 → 0, in between
(ambiguous) → dropped.

**Criteria path.** 30% of items (seeded) carry explicit `criteria {true, false}` texts (generic per source) so the
optional request field is exercised; the rest use Laya's default texts.

## 3. Sizes (`noul_{train,val,test}.jsonl`)

| split | total | positive (1) | negative (0) | by source |
|---|---:|---:|---:|---|
| train pool | 85,000 | 34,867 (41%) | 50,133 | intents 30,000 (MASSIVE 22,829 · BANKING77 7,171) · tools 24,000 · Civil 25,000 · MHS 6,000 |
| val | 9,000 | 3,485 | 5,515 | intents 3,000 · tools 2,000 · Civil 3,000 · MHS 1,000 |
| test | 12,270 | 5,588 | 6,682 | same + **BoolQ 3,270 (external)** |

Train composition by kind: positive 34,867 (41%) · hard negative 34,370 (40%) · cross-domain / easy 11,172 (13%) ·
MHS plain negatives 4,591 (5%)
(close to the 40/40/20 target; tool items have no cross-domain kind — tool cross-task is covered by intent
cross-domain). One training mixture uses 51,000 of the pool (uniform seeded sample).

## 4. Known limitations

* MHS single-annotator comments are noisy (≈50% of comments have 1 annotator).
* Civil Comments hard negatives for `toxicity` (0 < fraction ≤ 0.3) are near the label boundary by construction.
* Intent hard negatives can still hold residual overlaps not caught by the hand list.
* All contexts are English except MASSIVE pt-PT / es-ES (propositions are English).
* BoolQ is possibly in Laya's upstream training mix (reading comprehension family) — Laya's BoolQ number may be
  in-distribution for Laya; it is never in-distribution for Krun.

## 5. Data v2 (round 2, `configs/primitives_data_v2.yaml` → `data/primitives_v2/`, used by v0.3)

Same sources and licenses. Additions, both deterministic transformations of real human labels:

| addition | construction | train pool |
|---|---|---:|
| MHS item propositions (CC-BY-4.0) | 9 propositions ("Does this comment insult the group it mentions?" …); median annotator rating ≥ 3 → 1, ≤ 1 → 0, neutral 2 dropped | 16,000 |
| HelpSteer2 quality propositions (CC-BY-4.0) | 5 propositions ("Is the response factually correct and complete?" …); top rating(s) → 1, low ratings → 0, middle ratings dropped | 12,000 |

**Held-out ("unseen") propositions — never trained, evaluated separately and excluded from `in_domain`:**

* Civil `identity_attack`;
* MHS `dehumanize` and `status`;
* HelpSteer2 `coherence`;
* 15 intent propositions (10% of labels by stable hash): `calendar_query`, `card_arrival`, `card_delivery_estimate`,
  `declined_transfer`, `disposable_card_limits`, `extra_charge_on_statement`, `fiat_currency_support`,
  `lost_or_stolen_phone`, `pending_cash_withdrawal`, `pending_transfer`, `qa_definition`, `qa_stock`,
  `recommendation_locations`, `terminate_account`, `top_up_reverted`.

Sizes:

| split | total | unseen-proposition items |
|---|---:|---:|
| train pool | 113,000 (50k true / 63k false) | — |
| val | 13,000 | 1,735 |
| test | 16,270 | 1,738 |

Score data v2 is byte-identical to v1.
