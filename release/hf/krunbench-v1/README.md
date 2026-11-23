---
license: other
license_name: mixed-per-source
language: [en]
task_categories: [text-classification]
tags: [decision, intent, tool-routing, abstention, ood, benchmark]
configs:
- config_name: default
  data_files:
  - {split: dev, path: dev.jsonl}
  - {split: val_fit, path: val_fit.jsonl}
  - {split: val_eval, path: val_eval.jsonl}
  - {split: final_protected, path: final_protected.jsonl}
---

# KrunBench v1

The decision benchmark used to build, calibrate and gate [Krun One v1](https://huggingface.co/krun-ai/krun-one-v1).
Records follow the `krun.decision.v1` schema (context, typed question, candidates, gold labels, answerability, slices).

| split | records | role during v1 development |
|---|---:|---|
| `dev` | 695 | debugging |
| `val_fit` | 2,844 | temperatures, thresholds, variant selection |
| `val_eval` | 2,905 | reported validation numbers |
| `final_protected` | 11,715 | held out (now public: it can no longer serve as a blind final test for Krun models) |

Question types: choice, multi, noul, score. Slices: candidate count k = 2 … 128, near-duplicate hard negatives, unseen
labels, 1,695 unseen tools, gold removed / near and far OOD, native irrelevance, typos. `protected_manifest.json`
holds the text hashes used for leakage checks. Specification: `KRUNBENCH_V1_SPEC.md` (Portuguese).

## Sources and licenses

| source | license | splits |
|---|---|---|
| CLINC150 | CC-BY-3.0 | dev / val (validation part), final (test part) |
| SNIPS (test) | Apache-2.0 | final |
| Bitext customer support (3 held-out intents) | **CDLA-Sharing-1.0** — records derived from it are shared under the same terms | val / final |
| ToolACE (items unused by Krun One v0) | Apache-2.0 | val / final |
| Civil Comments (test rows) | CC0-1.0 | val / final |

The added structure (questions, candidate sets, slices, abstention labels) is released under CC-BY-4.0 by Krun.
