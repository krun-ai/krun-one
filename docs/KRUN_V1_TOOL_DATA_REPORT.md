# Krun One v1 — tool/function routing data report

## 0. Krun Decisions v1 as built (2026-09-27) — the numbers that matter

Source: `results/v1/kd1_stats.{md,json}` on `data/kd1/train.jsonl` (sha256 `ddcf6d1b…40ba`, 1,042,946 decisions).
Tool decisions = task family `tool`, choice + multi: **410,404** (39% of the corpus; the `tool_routing` sampler bucket
alone is 316,944 = 30.4%, the rest are tool OOD decisions in the `ood` bucket).

| quantity | value | comment |
|---|---:|---|
| **real tool queries** (ToolACE query groups) | **7,063** | ToolACE is itself LLM-synthetic at the source (Apache-2.0, generator undisclosed). There is **no other license-clean tool source**: xLAM is excluded, and SGD / MTOP are out |
| real dialogue routing contexts (MultiWOZ 2.2 turns with an evidenced intent change) | 17,355 | 11 schema intents across 7 services, with human descriptions |
| unique real APIs / functions used as gold | 6,763 | ToolACE official names + descriptions (never rewritten) |
| unique tools shown as candidates | 21,424 | train-side pool; KrunBench v1 protected tools removed (4,951 native occurrences) |
| intents rendered as functions (description-dependent) | 312 labels (5 catalogs) | Qwen3-8B descriptions, all reviewed by the author, 56 rewritten |
| domains | ToolACE: not annotated (~30 per the paper, unverified); MultiWOZ 7; intent catalogs: banking, e-commerce support, general assistant, voice assistant | — |
| schemas | one JSON schema per ToolACE tool; MultiWOZ v2.2 schema; intent-as-function = name + one-sentence description | — |
| **description-dependent** decisions (gold and another candidate share ≥ 50% of name tokens) | 132,269 (32.2%) | — |
| **near-duplicate** decisions (≥ 1 candidate with doc cos 0.85–0.95 to the gold) | 111,495 (27.2%) | candidates with cos ≥ 0.95, the same humanized name or the same description as the gold are never negatives |
| **OOD** tool decisions (gold removed, native irrelevance, cross-task) | 89,629 (21.8%) | — |
| translated schema (PT/ES/FR descriptions) | 46,573 (11.3%) | tool names never translated |
| **unseen-tool split** | 1,695 protected tools (KrunBench v1 `final_protected`) | absent from training by id, name, description and equivalent description. Independent `--check`: **LEAKAGE = 0**. The val slice `unseen_tool` has 175 decisions in val_eval |
| language (tool decisions) | EN 162,151 · PT-BR 89,194 · FR 74,729 · ES 74,673 · PT-PT 9,657 | — |

**Origins inside the tool family** (never mixed):

| source | real | derived | MT | rules | LLM |
|---|---:|---:|---:|---:|---:|
| ToolACE (native + pool negatives) | 7,096 | 76,147 | 100,431 | 8,797 | 116 (counterfactual, pilot) |
| MultiWOZ | 17,549 | 18,938 | 3,355 (cross-domain MT contexts) | 569 | — |
| intent-as-function | 912 | 52,724 | 75,979 | 7,804 | — |
| cross-task pool (OOD) | 116 | 11,091 | 8,757 | — | — |
| other (gold-removed intent-as-function, synthetic) | 100 | 8,884 | 10,959 | — | 196 |

**Synthetic share: 0.08%** of tool decisions (LLM pilot items only; pilot v2 failed its gate). Derived share: 41%.
MT share: 49%.

### Reading (no hiding behind permutations)

1. **Every ToolACE decision traces back to 7,063 queries.** Each has ~26 decisions (EN + 3 MT locales, different k and
   negatives, gold removed, typos). Candidate-set permutations add **variety of negatives, not new user needs**.
2. The **diversity of real tool queries is the binding limit** of tool routing in v1. intent-as-function (312 labels,
   ~130k decisions) and MultiWOZ add description-dependent and contextual routing, but they are not API calls.
3. Before a 2M/3M round, the tool block needs **new license-clean queries**, not more decisions per query. Options,
   all needing an owner decision:
   * SGD/SGD-X (CC-BY-SA);
   * MTOP (license to confirm);
   * a synthetic pilot v3 with a stricter hard-negative verifier (§ synthetic pilot v2 report).

---

## History: candidate audit before the build (2026-09-26)


Date: 2026-09-26. **xLAM, and everything derived from it (xlam-irrelevance, When2Call, Nemotron-Agentic
`tool_calling`, Synth-APIGen, …), stays out of Krun One v1:** `excluded_pending_license_clarification`.

Source counts were measured by `scripts/v1/tool_data_audit.py` (→ `results/v1/tool_audit.json`). The candidate survey
is in `docs/KRUN_V1_TOOL_SOURCES_CANDIDATES.md`. Nothing here is legal advice.

## 1. Candidate audit

| dataset | revision | license | examples | tools / intents | domains | single / multi | none-applicable? | languages | origin | leakage vs our benchmarks | recommended use |
|---|---|---|---:|---:|---:|---|---|---|---|---|---|
| **ToolACE** `Team-ACE/ToolACE` | 6bda777c | Apache-2.0 | 11,300 dialogues. Single-call first turns: 7,036. Multi-call: 2,040. No call in first turn: 1,993 | **13,403 unique tools** (JSON schemas) | ~30 (paper; not verified) | 7,036 single / 2,040 multi | **yes** (1,993) | EN (23 ZH) | **LLM-synthetic** (generator not disclosed) | 704 single-call contexts overlap v0 validation/holdout (already excluded from the train pool); 6 tools equal to BFCL tools | **train** (single-call remainder + 40% of the unused items) · **val/final** (remaining unused items) |
| **CLINC150** `clinc/clinc_oos` plus | 155b9c7104 | CC-BY-3.0 | 23,850 (train 15,250 / val 3,100 / test 5,500) | 150 intents | 10 | single | **yes**: 1,350 human out-of-scope | EN | **REAL** (crowdsourced) | 67 exact / 2 near vs protected contexts (short common phrases) → removed | **train** (train split) · **KrunBench v1** (val → dev/val, test → final) |
| **SNIPS** `benayas/snips` | 16915b8957 | Apache-2.0 (mirror) / CC0 (upstream) | 14,484 | 7 intents | 1 (assistant) | single | no | EN | **REAL** | 1 exact | **train** (train) · **final** (test) |
| **Bitext support** | 430d1a89bd | CDLA-Sharing-1.0 | 26,872 | 27 intents | 11 categories | single | no | EN | **HYBRID** (templates + linguist curation; flags for colloquial/typos) | 1 exact | **train** (24 intents) · **final/val** (3 intents held out: unseen labels) |
| **SGD / SGD-X** (GitHub) | master | **CC-BY-SA-4.0** | ~20k dialogues (README) | 45 services / 88 intents, all with human descriptions | 20 | single | implicit | EN | **REAL** (schemas human-written) | not measured (not ingested) | **REVIEW**: share-alike (same reason FineWiki was excluded). eval/validation only until the owner decides |
| **API-Bank** level-1/2 | 12e8158b76 | MIT | 534 human-annotated items | 73 runnable APIs | many | single | few | EN | **REAL** (eval set) | — | **eval-only**. Its train set is GPT-generated → REVIEW |
| **MTOP** `mteb/mtop_intent` | 9a200bf94b | "unknown" on HF (believed CC-BY-SA-4.0, not verified) | ~100k | 117 intents | 11 | single | no | EN/**ES/FR**/DE/HI/TH | **REAL** (human-translated) | — | **REVIEW** (confirm license). Best ES/FR candidate |
| **Leyzer** | ed035b0695 | CC-BY-4.0 | ~20–50k (not verified) | ~186 (not verified) | ~20 | single | no | EN/PL/**ES** | REAL (templates?) | — | **REVIEW** (loaded via script; size check pending) |
| ToolBench, APIGen-MT, NexusRaven, jina, … | — | research-only / NC | — | — | — | — | — | — | — | — | **EXCLUDE** |
| glaive-fc-v2, hermes-fc-v1, Gorilla APIBench, Seal-Tools | — | Apache/MIT, but the generator is undisclosed or was OpenAI | — | thousands (synthetic) | — | — | some | EN | synthetic | — | **REVIEW** (not used) |

**No licensed Portuguese tool-routing dataset was found.** PT comes from MASSIVE (pt-PT; intents), local MT (opus-mt)
of EN queries, and pt-BR adaptation by a local LLM with a blind verifier.

## 2. Proposed tool-routing block for Krun Decisions v1

The **REAL** base is small, and the plan is explicit about that.

| quantity | value | notes |
|---|---:|---|
| real tool-routing contexts (train) | **6,332** ToolACE single-call + **813** ToolACE unused items in the train pool = **7,145** | ToolACE is the only license-clean tool source with thousands of schemas. Its **queries and APIs are LLM-synthetic** (Apache-2.0). They are called "real" here only in the sense of *source data, not generated by us*. |
| real routing contexts (intents, human) | CLINC train 15,250 · SNIPS 13,784 · Bitext 23,600 · BANKING77 8,993 · MASSIVE 45,880 (4 locales) | intent routing ≈ function routing with written descriptions ("tool = intent + description") |
| unique tools (train pool) | ~13.4k ToolACE schemas, minus 1,695 reserved as **protected unseen** | plus ~350 intents rendered as tools |
| domains | ToolACE ~30 (not verified) + CLINC 10 + Bitext 11 + SNIPS 1 + MASSIVE 18 | |
| distinct schemas | ≈ number of unique tools (each ToolACE tool has its own JSON schema) | |
| description-dependent decisions | **287** ToolACE decisions where the gold and a native distractor share ≥ 50% of name tokens | amplified in the derived set: hard negatives chosen by description similarity |
| near-duplicate tools | **256** (same or near-identical description under another name) | never used as negatives of each other (the v0 rule is kept) |
| none-applicable (train) | real: 40% of ToolACE's 1,993 no-call items + CLINC train out-of-scope; derived: gold removed | the rest of the native no-call items are in KrunBench v1 |
| unseen tools reserved | **1,695** (KrunBench v1 `final_protected`; name **and** description absent from training) | 3,071 before the stricter name rule |

**Origins inside the tool block** (from the capacity-aware plan, `results/v1/decisions_plan.md`):

| scale | tool decisions | REAL | DERIVED | MACHINE-TRANSLATED | RULE | LLM-SYNTHETIC | short |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1M | 220,000 | 6,332 | 120,308 | 93,360 | 0 | 0 | 0 |
| 2M | 440,000 | 6,332 | 120,308 | 227,952 | 26,400 | 59,008 | 0 |
| 3M | 660,000 | 6,332 | 120,308 | 227,952 | 39,600 | 230,999 (35%) | 34,809 |

Reading:
1. Without xLAM, every derived and translated tool decision traces back to only ~7k tool queries (ToolACE is itself
   synthetic).
2. At 3M, 35% of the tool block would be LLM-synthetic, and it still falls short. **Recommendation: cap the tool block
   at ~220–300k decisions (1M–2M scales)** until another license-clean real source is added.
3. The candidates that would change this picture all need an owner decision:
   * SGD/SGD-X: share-alike;
   * MTOP: license to confirm;
   * APIs.guru specs: checked per spec, plus queries generated locally.

## 3. Rules kept

* Tool names are never translated; descriptions may be (the translated-schema slice is ~15% of non-EN).
* A tool is "seen" if it appears on the training side as gold **or** distractor, by exact key, same description,
  near-duplicate description (Jaccard ≥ 0.8) **or the same name** (for the protected unseen set).
* Every tool file goes through `check_leakage` + the tool index against KrunBench v1 before use. The build aborts on any
  violation. Result today: **LEAKAGE = 0** (`results/v1/krunbench_v1_leakage_check.json`).
