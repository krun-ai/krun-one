---
license: other
license_name: mixed-per-source
language: [en]
task_categories: [text-classification]
tags: [decision, technical-decisions, reasoning, counterfactual, benchmark, software-engineering]
configs:
- config_name: v1
  data_files:
  - {split: protected, path: protected_v1_labeled.jsonl}
  - {split: validation, path: val_v1_labeled.jsonl}
- config_name: r3_extension
  data_files:
  - {split: protected, path: protected_r3.jsonl}
  - {split: validation, path: val_r3.jsonl}
- config_name: tetris_views
  data_files:
  - {split: test, path: tetris_representation_test.jsonl}
---

# KrunTechBench

Technical and state / action decisions built for the Krun One v1.1 research rounds
(https://github.com/krun-ai/krun-one, `results/v11/`). Every label is computed — rule tables, simulators, arithmetic,
calendar, graph walks, a policy interpreter, or execution evidence from SWE-smith — never written by a model.

| config | split | examples | what |
|---|---|---:|---|
| `v1` | protected / validation | 10,664 / 4,054 | 10 tracks (api, backend, cloud, database, devops, distributed, ML infra, security, software repair, state / action); slices: hard negatives, constraint flips, unseen requirement combinations, unseen technologies, large candidate sets, unseen simulator families, OOD |
| `r3_extension` | protected / validation | 7,641 / 3,496 | capability tracks (numeric, temporal, multi-hop, long-policy, trade-off, mixed) in choice / true-false / ordinal form, counterfactual families, invented-product catalogs; benchmark-only scenarios tagged `unseen_scenario` |
| `tetris_views` | test | 1,800 | the same 600 held-out boards as raw board / structured state / explicit consequence |

Each problem usually appears in two renderings: `raw` (candidate names only; facts in the context) and `feat` (each
candidate followed by its computed consequence) — the gap between the two is the measurement the reports rely on.
Families (`family.id`, `variant`) let you compute both-correct and stuck rates (`scripts/v11/r3_report.py`).

`protected` splits were the held-out sets of the research; they are public now and no longer blind for Krun models.
`protected_v1_labeled` = the original file with the 120 `swe_involved` labels restored (record-by-record identical otherwise).

## Licenses

Rule tables, simulators, capability generators and the decisions they produce: CC-BY-4.0 (Krun). Software-repair
items are derived from SWE-bench/SWE-smith (MIT) problem statements of allowlisted, permissively licensed repositories
(`swe_smith_allowlist.json`); those texts keep their repositories' licenses.
