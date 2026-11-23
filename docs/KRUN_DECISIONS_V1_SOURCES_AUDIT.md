# Krun Decisions v1 — source audit (2026-09-27)

Scope: the sources the fine-tuning round asks to add or re-check. Registry of record: `configs/v1/sources.yaml`
(builders skip every source whose status starts with `excluded`). Nothing here is legal advice; the commercial-use
column is an engineering assessment for the owner.

| source | revision | data license | provenance | commercial-use assessment | status |
|---|---|---|---|---|---|
| **MASSIVE** `AmazonScience/massive` | `ed58ac42` (HF, script repo; data from the Amazon tarball in the HF cache) | **CC-BY-4.0**: `NOTICE.md` of github.com/alexa/massive states "The MASSIVE dataset is licensed under CC BY 4.0". The repo's `LICENSE.txt` (Apache-2.0) covers the **code**. That is why GitHub shows NOASSERTION. | Localized/translated by professional workers from SLURP (CC-BY-4.0). Every row carries 3 judge ratings (`intent_score`, `grammar_score`, `language_identification`). | OK with attribution. The open question noted in round 1 is **resolved**. | `approved_with_notice` |
| **Banking77 PT-BR** `c2d-usp/banking77-pt-br` | `85f51c4a36c69482fae32ec31afec784f9722a7e` | **CC-BY-4.0** (card) | Translation of BANKING77 (CC-BY-4.0) by **gpt-3.5-turbo (5,400 train rows) / gpt-4o-mini (the rest)**, followed by human review. Column `alterado`: 727 train + 183 test rows edited by reviewers. | Dataset license allows it. OpenAI's terms bind whoever generated the text (c2d-usp), not downstream users of a CC-BY-4.0 dataset. **Residual risk: low**, but it is the same class of concern that put glaive/hermes in REVIEW (OpenAI-generated text). **Owner can veto**; removing it touches only the pt-BR intent slice. | `approved_with_notice` (flagged) |
| **MultiWOZ 2.2** `budzianowski/multiwoz` | git `fe0c8e65cfcd8462bd33c86e35f21addc84ca82b` (raw JSON, `data/raw/multiwoz22/`). The `pfb30/multi_woz_v22` HF script (`fd8c4a94`) downloads the same files. | **MIT** (repo LICENSE, © 2019 P. Budzianowski). The HF card says Apache-2.0. | Wizard-of-Oz dialogues collected by Cambridge. v2.1 annotations by Amazon, v2.2 by Google (schema with human-written service/intent/slot descriptions). | OK with notice. | `approved_with_notice` |
| **XNLI** `facebook/xnli` | `b8dd5d7a` | **CC-BY-NC-4.0** (github.com/facebookresearch/XNLI `LICENSE`). The HF card says "More Information Needed". Train = machine-translated MultiNLI (mixed OANC/other licenses). | Meta; dev/test human-translated from MNLI | **Non-commercial.** Material doubt → not usable for a commercial model. | **`excluded_noncommercial`** |
| **PAWS-X** `google-research-datasets/paws-x` | `4cd8187c` | Google `LICENSE`: "may be freely used for any purpose" | Sentences come from **Wikipedia** (PAWS-Wiki, CC-BY-SA-3.0/4.0 text) via word scrambling and back-translation. Train = machine translation; dev/test = human translation. **No Portuguese.** | Google's grant is permissive, but the underlying Wikipedia text is share-alike. That is the same concern the owner used to exclude FineWiki. Material doubt, and low value (no PT; only FR/ES pairs). | **`excluded_pending_review`** |
| Banking77 (EN) | `f5412156` | CC-BY-4.0 | PolyAI | unchanged | `approved_with_notice` |
| ToolACE | `6bda777c` | Apache-2.0 | LLM-synthetic (generator not disclosed) | unchanged; see `docs/KRUN_V1_TOOL_DATA_REPORT.md` | `approved_with_notice` |
| Civil Comments / HelpSteer2 / MHS | unchanged | CC0 / CC-BY-4.0 / CC-BY-4.0 | — | unchanged | approved |

Still out, and never re-added automatically: xLAM (+ xlam-irrelevance), SGD, MTOP, HWU64, BFCL (protected), BoolQ,
SST-5, FineWiki, XNLI, PAWS-X.

## Banking77 PT-BR ↔ Banking77 mapping

* **Row alignment is exact.** Train: 10,003 PT rows ↔ 10,003 EN rows. Test: 3,080 ↔ 3,080, plus 2 empty trailing
  CSV rows that are dropped.
* **Label mapping is one-to-one.** Each of the 77 Portuguese labels (`chegada_do_cartão`, …) maps to exactly one
  English label, row by row.
* Therefore each PT-BR row gets the `semantic_group_id` / `split_group_id` of its English row (`banking77:<split>-<row>`).
  The two languages always land in the same split.
* Portuguese label strings are **not** used as candidate ids. Candidates stay the official English ids. A PT-BR
  schema, when needed, uses the translated description, not the Portuguese label.

## MASSIVE use

Locales: en-US, pt-PT, es-ES, fr-FR. `id` is parallel across locales, so `semantic_group_id = split_group_id =
massive:<id>` for the four rows. MASSIVE test and validation stay out of training (test = regression, validation =
multilingual V set). Train rows whose three judges did not all give `intent_score == 1` are dropped as unreliable.

## MultiWOZ use

Only decisions that are well defined by the v2.2 annotations are derived (never millions):
* **Contextual intent / tool routing**: user turn + previous turns → the active intent of the turn's service
  (`find_restaurant`, `book_hotel`, …). Candidates are the schema intents with their human descriptions, rendered as
  functions. The decision is taken only when the frame's active intent changed at that turn.
* **Multi-domain routing**: multi-label over services active in the turn.
* **State-dependent noul**: "The user has asked for the <slot>" from `requested_slots`.

Dialogue splits are the official ones: train → training, dev → validation, test → unused.
