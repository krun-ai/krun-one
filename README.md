# Krun One

Krun One is a family of **encoder-only decision models**: given a context and a typed question — `choice`
(pick one option, or none), `noul` (true / false), `score` (an ordinal level) and `multi` (select every option that
applies) — it returns a calibrated decision with an abstention signal in one forward pass. No text generation.

This repository holds everything behind it: model and serialization code, data builders, training, calibration,
evaluation, export, the research rounds and their reports. Weights and datasets are on the Hugging Face Hub.

| release | what | where |
|---|---|---|
| **Krun One v1** (text) | 879,356,931 parameters, K1-W1536-L22 (ModernBERT / mmBERT architecture, 22 layers, hidden 1536), 256k Gemma-2 vocabulary, 1,024-token sequence | `ferreslabs/krun-one-v1` |
| Krun One v1 multimodal | the frozen v1 text core + M1.5c fusion (26.7M trained parameters) over SigLIP2 (images, documents), Whisper-small (speech) and RapidOCR | `ferreslabs/krun-one-v1-multimodal` |
| KrunBench v1 | the decision benchmark used to build and gate v1 (incl. the former protected split) | `ferreslabs/krunbench-v1` |
| KrunTechBench v1 (+ R3 extension) | technical / state-action decisions, counterfactual families, capability tracks | `ferreslabs/kruntechbench` |
| Krun One v0 | Laya-based predecessor (421M) | `docs/KRUN_ONE_V0_README.md` |

License: **Apache-2.0** for code and weights (see `LICENSE`, `NOTICE` and the data notices below).

## Quick start

```bash
uv sync                                   # Python 3.12; torch 2.14 cu126; add --group m1 --group m15 for multimodal
hf download ferreslabs/krun-one-v1 --local-dir artifacts/krun-one-v1-text-final
```

```python
from krun_one.v1.runtime import TextArtifact

a = TextArtifact.load("artifacts/krun-one-v1-text-final", device="cuda", dtype="bf16")   # verifies SHA256SUMS
a.decide({"context": "quero cancelar minha assinatura",
          "questions": {"q": {"type": "choice", "task_type": "intent",
                              "options": [{"id": "cancel_subscription"}, {"id": "check_balance"}]}}})
```

Multimodal: `krun_one.v1.multimodal.runtime` (see the model card of `ferreslabs/krun-one-v1-multimodal`).

## Krun One v1 at a glance (RTX 4090)

| | |
|---|---|
| KrunBench v1 choice accuracy (V eval) | 97.6% · tools 98.1% · k = 128 candidates 90.6% · OOD AUROC 0.971 |
| ToolACE v0 holdout | 90.1% |
| latency, bf16 weights | typical request ~14–17 ms p50 · k = 128 ~44 ms · 4 questions ~16–21 ms |
| memory | ~1.7 GB VRAM (bf16 weights) |
| multimodal (M1.5c, val_eval) | visual-only documents 68.0% · screenshots 99.2% · real speech 85.1% · images 64.8% |
| JevBench public items (external, 231) | 53.2% (hard tier 30.6%) |

Known limitations are documented in the model cards and in `results/`; the most important: one encoder pass does not
compute multi-step consequences from a raw state (see the v1.1 reports), and inputs are cut at 1,024 tokens.

## Repository map

| path | contents |
|---|---|
| `src/krun_one/v1/` | architecture (`arch.py`, `model.py`), serialization, losses, calibration, runtime, multimodal (`multimodal/`) |
| `src/krun_one/v11/` | v1.1 research: environments, rule tables, simulators, capability curriculum, SWE-smith decisions, growth |
| `src/krun_one/*.py` | v0 (Laya-based) |
| `scripts/v1/`, `scripts/v11/` | data builders, trainers, evaluation, reports, pod runners |
| `configs/v1/`, `configs/v11/` | registered protocols, gates and source registries (`sources.yaml`) |
| `docs/` | specs, corpus provenance, source / license audits |
| `results/` | reports and metrics of every round (v0, v1 stages, multimodal M1 → M1.5c, v1.1 R0 → R3) |
| `tests/` | unit and equivalence tests (`uv run pytest`) |

Research reports worth starting from: `results/v11/KRUN_ONE_V11_R0_REPORT.md` (capacity), `KRUN_ONE_V11_R1_R2_COMPARISON.md`
(encoder adaptation), `KRUN_ONE_V11_R3_FINAL_REPORT.md` (generalization, external transfer, multimodal re-alignment).

Large inputs (`data/`, `runs/`, `artifacts/`) are not versioned: datasets are rebuilt by `scripts/` from the public
sources in `configs/*/sources.yaml` and KrunBench; weights come from the Hub. Several scripts expect the original
workspace layout (`<project>/experiments/krun_one`, `<project>/benchmarks/krunbench`); set `KRUN_PROJECT_ROOT`.

## Data and license notices

The code and the weights are released under Apache-2.0 by the project owner. The training data is a mix of
third-party datasets, each under its own terms; the complete registry with revisions is in `configs/v1/sources.yaml`,
`configs/v11/sources.yaml` and the audits in `docs/`. Points users should know about:

* **xLAM function-calling 60k** (Salesforce) and data derived from it (MadeAgents xlam-irrelevance, NVIDIA When2Call)
  were used for v1 training. The official xLAM card states "this release is for research purposes only in support of
  an academic paper". The owner decided to use and redistribute the resulting weights (`docs/KRUN_XLAM_OFFICIAL_LICENSE_REAUDIT.md`);
  downstream users should assess this for their own use.
* Some sources contain text generated by commercial models (banking77 PT-BR translations by OpenAI models; APIBench
  instructions by GPT-4; ToolACE, Hermes and ToolMind with undisclosed generators; xLAM by DeepSeek / Mixtral).
* Attribution / share-alike data: CC-BY-4.0 sources (MASSIVE, banking77, Taskmaster, HelpSteer2, MInDS-14, FLEURS, CORD,
  COCO annotations…), CDLA-Sharing-1.0 (Bitext), CDLA-Permissive-1.0 (DocLayNet), ODbL / DbCL (CREMA-D), CC-BY-SA-4.0
  (SIB-200 topic labels), ODC-By + Common Crawl terms (FineWeb family). See `NOTICE`.
* Tokenizer: the mmBERT (MIT) / Gemma-2 vocabulary; analysis in `docs/KRUN_ONE_V1_TOKENIZER_LICENSE.md`.
* JevBench (MIT) was used **only for evaluation**; nothing derived from it was trained on.

## Citation

```bibtex
@software{krun_one_2026,
  title  = {Krun One: calibrated encoder-only decision models},
  author = {Ferres, Rafael},
  year   = {2026},
  url    = {https://github.com/krun-ai/krun-one}
}
```
