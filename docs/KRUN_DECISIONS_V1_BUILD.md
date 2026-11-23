# Krun Decisions v1: how the ~1M training decisions were built (2026-09-27)

Code: the scripts and modules below. Numbers: `results/v1/kd1_stats.md`. Build manifest (quotas, pools, MT filter,
leakage, hashes): `data/kd1/build_manifest.json`.

| stage | code | output |
|---|---|---|
| 1. context pool + split assignment + GPU job lists | `scripts/v1/kd1_contexts.py`, `src/krun_one/v1/corpus/sources.py` | `data/kd1/contexts.jsonl`, `data/kd1/gpu_jobs/` |
| 2. GPU text jobs (RunPod) | `scripts/v1/kd1_gpu_jobs.py`, `scripts/v1/runpod_kd1_gpu.sh` | `data/kd1/gpu_out/` |
| 3. synthetic pilot v2 | `scripts/v1/synthetic_pilot_v2.py` | `results/v1/synthetic_pilot_v2/` |
| 4. decisions + val_ml + leakage | `scripts/v1/build_decisions_v1.py`, `src/krun_one/v1/corpus/{mining,decisions}.py` | `data/kd1/{train,val_ml}.jsonl` |
| 5. stats | `scripts/v1/kd1_stats.py` | `results/v1/kd1_stats.{json,md}` |

Record format: `krun.decision.v1` (`src/krun_one/v1/schema.py`). Every record passes `validate()`.

## 1. Sources and splits

Licenses and audit: `docs/KRUN_DECISIONS_V1_SOURCES_AUDIT.md`. The registry is `configs/v1/sources.yaml`; loaders
refuse `excluded*` sources.

The split unit is `split_group`: a MASSIVE `id` covers its 4 locales; a BANKING77 row covers EN + PT-BR; a ToolACE
normalized query; a MultiWOZ dialogue.

| source | train | val_ml (multilingual V, new) | never used |
|---|---|---|---|
| MASSIVE en-US / pt-PT / es-ES / fr-FR | train (non-EN rows need 3/3 judges `intent_score == 1`) | validation | test (KrunBench v0 regression) |
| BANKING77 EN + PT-BR (row-aligned) | train | 1,500 test rows × 2 languages | — |
| CLINC150 plus / SNIPS / Bitext (minus 3 held-out intents) | train | MT of KrunBench v1 validation items | KrunBench v1 validation/test |
| ToolACE single-call (minus v0 val/holdout queries) + KrunBench v1 unused train pool | train | MT of KrunBench v1 val ToolACE items | KrunBench v1 val/final |
| MultiWOZ 2.2 | train | dev | test |
| Civil Comments / MHS / HelpSteer2 | train rows (noul/score via `data/primitives_v2`, xLAM-derived rows dropped; new multi-label and toxicity-score decisions from train rows) | — | — |

Stage 1 drops train contexts that are exact or near duplicates (char 5-gram Jaccard ≥ 0.9) of any KrunBench v1 item or
val_ml item. The whole split group is dropped. The exception is MultiWOZ, where only the offending turn is dropped:
those turns are generic ("thank you, goodbye").

## 2. GPU text jobs (RunPod RTX 5090, `vz93igvcqa7fs9`, terminated)

* **Machine translation**: `Qwen/Qwen3-8B` (Apache-2.0), temperature 0.2. Train contexts go EN → pt-BR / es-419 /
  fr-FR. KrunBench v1 **validation** items are translated too; their outputs stay validation and never become training
  data. Every translation is then:
  * judged by `ibm-granite/granite-3.3-8b-instruct` (adequacy ≥ 4, fluency ≥ 3);
  * checked by rules: lingua language ID, length ratio 0.4–2.6, numbers kept, `{{placeholders}}` kept.

  Measured acceptance: see `build_manifest.json → mt_filter`. The regional variants (es-419, pt-BR vs pt-PT) are the
  **requested** variants. No model verified them.
* **Intent descriptions**: one sentence per intent label, written by Qwen3-8B from train examples. The author reviewed
  all 318; 56 were rewritten (`configs/v1/intent_desc_overrides.yaml`), and translations of the rewritten ones were
  discarded. MultiWOZ uses its own human-written schema descriptions.
* **Description translation**: a sample of 3,000 ToolACE tool descriptions + the intent descriptions, into 3 locales,
  for the translated-schema slice. Tool names are never translated.

## 3. Decisions

Categories are the sampler's task buckets (`metadata.category`): choice_intent, tool_routing, noul, score, multi_label,
ood. Quotas per category × language are ceilings (`QUOTA` in the builder). Caps set the maximum number of decisions per
context per generator. A generator that cannot fill its quota from capped pools is reported SHORT.

| generator | what it builds |
|---|---|
| intent choice | k ∈ {2,…,128} (never above the catalog size); hard negatives v2 65% / random 35%; the schema is the EN label ids → every PT/ES/FR decision is cross-language |
| tool routing: ToolACE | native tools + pool negatives mined by query↔doc and gold↔doc e5 + name TF-IDF (RRF). Never used as a negative: same humanized name, same description, or doc cos ≥ 0.95 as the gold. Doc cos 0.85–0.95 marks `near_duplicate`. ~20% of non-EN decisions use a translated schema (plus some EN→PT and PT→ES) |
| tool routing: MultiWOZ | dialogue context (last 4 turns + user turn) → the intent that became active at this turn *with evidence* in the state; candidates = v2.2 schema intents with human descriptions |
| tool routing: intent-as-function | intent utterances vs intent labels rendered as functions with descriptions (description-dependent routing, multilingual) |
| OOD | gold removed (intent, tool, MultiWOZ, intent-as-function); native irrelevance (ToolACE first turns with no call, CLINC out-of-scope); cross-domain (catalog pairs with disjoint domains, **and** an e5 filter: no candidate may be more similar to the context than 0.85 or than its own gold − 0.05); cross-task (intent utterance vs tool catalog, same filter); synthetic near-OOD (pilot) |
| multi-label | intent concatenations (same catalog, connector in the context language, labels {a, b}); partial applicability (the second utterance comes from a disjoint catalog, so only a applies); empty set (gold removed); ToolACE multi-call; MultiWOZ multi-domain; Civil Comments properties (5 of the 6; `identity_attack` stays the unseen proposition); MHS target groups |
| noul | primitives v2 (Civil Comments, MHS, HelpSteer2, BANKING77/MASSIVE/ToolACE propositions; xLAM rows dropped); cross-language propositions (PT/ES/FR context, EN proposition built from the intent description, or the translated one); MultiWOZ requested-slot propositions |
| score | primitives v2 (MHS 9 items, HelpSteer2 5 attributes); Civil Comments toxicity levels (5 levels, train rows) |
| minimal pairs (mined) | real train utterances of the same catalog, word-Jaccard ≥ 0.6, different labels ("I am still waiting on my card?" ↔ "…my transfer?"), plus every language version of both members; `minimal_pair = {set_id, member, contrast, relation: natural_contrast_mined, difficulty: hard, transformation}` |
| typos (rules) | 6% of choice/tool decisions get a rule-typo copy (`origin = rules`) |
| synthetic (LLM) | the 695 audited accepted items of pilot v2 (minimal_pair, near_ood, counterfactual, slang); the pilot failed its gate, so nothing was scaled |

**Origins** (`metadata.origin`, never mixed):
* `real`: the first decision on an unchanged real context;
* `derived`: any further decision on it;
* `mt`: any decision on a machine-translated context;
* `rules`;
* `llm`.

**Curriculum stage** (`metadata.stage`, `decisions.stage_of`):
* **D3**: unanswerable, minimal pair / counterfactual / ambiguous / near-OOD, LLM origin, k ≥ 128, ≥ 2 near-duplicate
  tools, partial applicability;
* **D2**: cross-language choice/multi, typos, hard negatives with k ≥ 8, dialogue context;
* **D1**: everything else.

The trainer orders one epoch progressively (`training.curriculum_batches`; D1 keys ~U(0, .55), D2 ~U(.2, .9),
D3 ~U(.45, 1)).

## 4. Leakage

The builder removes violations and re-checks until zero:
* exact / normalized text, near duplicates (Jaccard 0.9), split and semantic groups, and ids;
* compared against KrunBench v1 (dev, val_fit, val_eval, final_protected) and val_ml.

`check_split_groups` covers train + val_ml. Protected unseen tools (KrunBench v1 manifest, 1,695) are removed from every
native tool list before mining. A context whose gold is protected is dropped. After the build, the independent
`build_krunbench_v1.py --check` validates the training file against the protected manifest, including unseen tool
id / name / description / equivalent description (`results/v1/krunbench_v1_leakage_check.json`).

## 5. Known limitations (honest list)

* **Tool routing rests on ~7k ToolACE queries.** They are LLM-synthetic at the source and have ~9–14 decisions each,
  plus MT. The rest of the tool block is intent-as-function and MultiWOZ. See `docs/KRUN_V1_TOOL_DATA_REPORT.md`.
* **LLM-synthetic share is ~0.07%** (target 10–15%), because pilot v2 failed its gate. MT and derived data fill the gap.
* **Regional variants are unverified.** es-419 and fr-FR MT are the requested variants; there is no fr-CA and no
  native es-419 source.
* **Noul / score are EN-only**, apart from the cross-language intent propositions. The overall EN share is therefore
  above the 35% target.
