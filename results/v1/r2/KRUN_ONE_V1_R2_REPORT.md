# Krun One v1: correction round R2 (tool routing, LR, OOD operating point), 2026-09-28

## Verdict

**The text base is NOT frozen.** Protocol: `configs/v1/r2_protocol.yaml`, registered before any result. Gates:
`tool_candidate_gates.json`. The tool-data candidate passes **7 of 8 gates**. It fails the one this round was about: the
clean ToolACE unseen holdout, **87.3% vs v0.3 91.8%** (paired Δ −4.6 pp, 95% CI [−8.6, −0.5]; required ≥ 91.8% or a
CI lower bound > −1 pp).

What R2 did solve:
* **LR:** 1e-5 is chosen (registered rule).
* **OOD / unseen tools** under the prospective coverage protocol: tool false acceptance at 70% coverage goes from
  35.8% (v0.3) and 6.0% (v1) to **1.2%**.
* All the v1 multilingual / large-k / consistency gains are **preserved**.
* Latency stays at 42 ms for 128 tools.

`final_protected` was **not opened**. Multimodal was **not started**. All pods are **TERMINATED** (`list-pods` empty).

## 1. Candidates

| id | what | artifact |
|---|---|---|
| v0.3 | production (Laya-based, 421M) | `artifacts/krun-one-v0.3` |
| v1 (candidate 1) | K1-W1536-L22, Krun Decisions v1 (1.04M), encoder LR 2e-5 | `artifacts/krun-one-v1-candidate` (`7362b07d…`) |
| v1-lr1e5 | same data / seed / steps, **encoder LR 1e-5** | `runs/v1/ft/krun-one-v1-lr1e5/final` (`sha256` in its SHA256SUMS) |
| **v1-tool (candidate 2)** | LR 1e-5, Krun Decisions v1 **+ Krun Tool Corpus v2** (1,329,462 decisions, 30,757 steps) | `artifacts/krun-one-v1-tool-candidate` (`model.safetensors c52475ab…`) |

## 2. Comparison on the common V

Common V = KrunBench v1 val_eval + val_ml eval. Same code for every model; thresholds and temperatures fitted on fit.

| metric | v0.3 | v1 | v1-lr1e5 | **v1-tool** |
|---|---:|---:|---:|---:|
| intent accuracy | 89.1 | 97.6 | 97.4 | **97.7** |
| tool accuracy (V: MultiWOZ / intent-as-function) | 81.2 | 98.5 | 98.6 | 98.4 |
| hard-negative / near-duplicate | 87.7 / 86.5 | 97.8 / 96.2 | 97.8 / 96.1 | 97.8 / 96.2 |
| cross-language accuracy | 82.7 | 98.3 | 98.4 | 98.3 |
| EN / PT-BR / PT-PT / ES / FR | 93.9 / 74.8 / 94.1 / 85.4 / 81.2 | 97.1 / 98.4 / 99.4 / 98.2 / 97.7 | 96.8 / 98.2 / 99.4 / 98.4 / 97.9 | 97.3 / 98.4 / 99.0 / 98.4 / 97.6 |
| k = 64 / 128 | 73.6 / 64.9 | 95.4 / 94.1 | 97.0 / 93.6 | 96.2 / 91.4 |
| OOD AUROC (margin) / AURC | 0.875 / 0.073 | 0.975 / 0.027 | 0.971 / 0.028 | 0.974 / 0.027 |
| ECE / NLL (choice) | 0.034 / 0.50 | 0.031 / 0.14 | 0.039 / 0.16 | 0.032 / 0.16 |
| permutation / cross-language consistency | 86.0 / 74.8 | 97.8 / 97.9 | 98.8 / 97.6 | 97.6 / 97.2 |
| noul ROC-AUC / accuracy | 0.973 / 89.0 | 0.960 / 85.1 | 0.965 / 86.8 | 0.962 / 85.1 |
| score MAE / Spearman | 0.81 / 0.74 | 0.63 / 0.84 | 0.65 / 0.84 | **0.61 / 0.85** |
| multi micro / macro F1 | n/a | 90.8 / 82.2 | 91.1 / 81.6 | 90.6 / 81.7 |
| **ToolACE holdout (unseen tools, 416)** | **91.8** | 84.1 | 86.1 | **87.3** |
| ToolACE val_tools (445) | 91.2 | 92.6 | 92.8 | 92.1 |

**Paired Δ vs v0.3** (group bootstrap, 95% CI):

| slice | v1-tool − v0.3 |
|---|---|
| native non-EN | **+12.5 pp** [+11.3, +13.8] |
| cross-language | **+15.5 pp** [+14.2, +17.0] |
| hard | **+10.1 pp** [+9.1, +11.3] |
| EN, never-trained KrunBench sources | +1.0 pp [−0.4, +2.2] |
| ToolACE holdout | **−4.6 pp** [−8.6, −0.5] |

**v1-tool vs v1**:
* native non-EN −0.14 pp, CI [−0.46, +0.17];
* cross-language −0.02 pp;
* multilingual gains fully preserved.

**In-distribution tool V** (`val_tools_v2`, 14,325 eval decisions from the new sources; v1-tool only):

| metric | value |
|---|---:|
| tool accuracy | 90.2 |
| near-duplicate | 91.9 |
| cross-language | 89.4 |
| OOD AUROC | 0.918 |
| noul (DO_NOT_CALL) ROC-AUC | 0.976 |
| multi micro-F1 | 97.6 |

## 3. OOD operating point (prospective protocol, tool family, V eval)

| coverage | v0.3 sel.acc / OOD FA / unseen FA | v1 | v1-lr1e5 | **v1-tool** |
|---:|---|---|---|---|
| 50% | 87.3 / 22.6 / 25.4 | 99.7 / 0.6 / 0.8 | 99.9 / 0.3 / 0.4 | **100.0 / 0.0 / 0.0** |
| 60% | 84.9 / 27.7 / 30.5 | 98.6 / 3.6 / 5.1 | 99.4 / 1.5 / 2.1 | **100.0 / 0.0 / 0.0** |
| 70% | 82.2 / 35.8 / 40.3 | 98.0 / 6.0 / 8.5 | 99.1 / 2.7 / 3.8 | **99.6 / 1.2 / 1.7** |
| 80% | 78.8 / 44.9 / 50.0 | 97.6 / 8.7 / 12.3 | 98.6 / 4.2 / 5.9 | **98.6 / 4.8 / 6.8** |
| 90% | 74.2 / 61.7 / 65.7 | 95.8 / 16.9 / 22.9 | 96.5 / 13.3 / 17.4 | **97.0 / 11.4 / 15.3** |

**Tool AURC:** 0.156 (v0.3), 0.033 (v1), 0.030 (v1-lr1e5), **0.028 (v1-tool)**.

**Serving points** chosen on fit by the registered rule:
* v1-tool: intent 80%, tool 90%;
* v1: intent 80%, tool 80%.

The intent curves are practically identical across the v1 models: FA 1.5–1.8% at 70% coverage, 2.4–2.8% at 80%.

## 4. Why the ToolACE holdout still misses

Holdout accuracy by slice (v0 negative strategy):

| slice | n | v0.3 | v1-tool | probe (v1-lr1e5 + tool stage) |
|---|---:|---:|---:|---:|
| k ≥ 16 | 140 | 87.1 | **88.6** | 89.3–90.0 |
| k < 16 | 276 | 94.2 | 86.6 | 90.2–90.6 |
| native tools + hard negatives from the v0 pool | 176 | 90.3 | 81.8 | 86.4–87.5 |
| native + random | 96 | 95.8 | 93.8 | 93.8–94.8 |

1. v1-tool already **beats v0.3 on large candidate sets**. The gap is in small sets whose negatives were mined from
   the v0 tool pool. That pool is mostly xLAM, which v0.3 trained on, including many of these exact distractors. The R2
   protocol forbids them on the v1 training side (benchmark distractors are protected). The benchmark therefore carries
   a structural advantage for v0.3 that the cleaner protocol refuses to copy. This explains part of the gap, not all of it.
2. **The tool stage matters more than the tool data volume.** The two short probes continued v1-lr1e5 on a
   tool-heavy mix (~6k steps, 71% tool decisions) and reach **90.1%**. The full run dilutes the same corpus to 21.5% of
   1.33M decisions spread over the curriculum, and gets 87.3%. The probe weights were **not kept**: they were analysis
   runs, and the pod volume was deleted.
3. **xLAM is not the missing piece.** With vs without xLAM, the probe holdout is identical (90.1 / 90.1, CI ±2.3 pp);
   the only gain is +1.2 pp on k ≥ 32 (`XLAM_ABLATION.md`).

## 5. What this round delivered (files)

| deliverable | where |
|---|---|
| xLAM official re-audit + decision (engineering OWNER_REVIEW_REQUIRED → owner CLEARED; the official card also says "research purposes only") | `docs/KRUN_XLAM_OFFICIAL_LICENSE_REAUDIT.md`, `configs/v1/sources.yaml` |
| source audit update (When2Call train = xLAM-derived and test = BFCL; APIBench = GPT-4 instructions; ToolMind tools = xLAM + Glaive + ToolACE; Glaive pending; Nemotron catalogued) | same + `docs/KRUN_TOOL_CORPUS_V2_REPORT.md` |
| Krun Tool Corpus v2 report: per-source counts, unique queries / tools / descriptions, near-duplicate / description-dependent / OOD / cross-language shares, strict unseen protection, xLAM contribution | `docs/KRUN_TOOL_CORPUS_V2_REPORT.md` |
| When2Call integration (tool action call / ask / none + DO_NOT_CALL noul + irrelevance) | same, §4 |
| LR experiment | `results/v1/r2/LR_EXPERIMENT_REPORT.md`, `lr_selection.json` |
| xLAM ablation | `results/v1/r2/XLAM_ABLATION.{md,json}` |
| comparison v0.3 / v1 / v1-lr1e5 / v1-tool + coverage curves | `results/v1/r2/comparison.json`, this report |
| leakage | builder 0 + independent `--check` **LEAKAGE = 0** on 1,329,462 decisions (`krunbench_v1_leakage_check_r2_tool.json`) |
| latency (RTX 4090, bf16, total p50, trained v1-tool) | 13.8–14.1 ms typical 1q, 15.5 ms 4q, 15.4–15.8 ms 32 tools, 27.7 ms 64, **42.0–42.2 ms 128 tools**, stress 229.5 ms (`latency_rtx4090_r2_tool.json`) |
| translation review pack (12,000 PROTECTED machine drafts, 60 stratified batches, 632 auto-flagged, local HTML tool). **No human review done**. All jobs are CLINC150, a limitation of the KrunBench v1 job list | `data/krunbench_v1/translation_review/`, `scripts/v1/translation_review.html` |

## 6. RunPod (RunPod billing, 2026-09-28)

| pod | GPU | $/h | use | cost | status |
|---|---|---:|---|---:|---|
| `i3o4h0zpetobql` | RTX 3090 Community (CUDA 12.5) | 0.22 | container never started (the cu128 image needs CUDA ≥ 12.8) | 0.02 | TERMINATED |
| `sij0m65kp9k8nr` | RTX 4090 Secure (US-NC-1) | 0.74 | health check; LR 1e-5 run (4.1 h); v1-tool run (6.6 h); evaluations; latency | 7.87 | TERMINATED |
| `peoesxa5c76mlj` | RTX 4090 Secure | 0.74 | 120k MT + judge; 12k protected drafts (vLLM pinned to 0.11 for CUDA 12.8) | 1.98 | TERMINATED |
| `y15ix0fk8jduny` | RTX 4090 Secure | 0.74 | xLAM ablation (2 probes) | 2.72 | TERMINATED |
| `mnuxqgv51nyy07` | RTX PRO 6000 Secure | 2.09 | **mistake**: chosen for speed, against the GPU preference list; venv failed (torch cu130 not installed); terminated without use | 1.20 | TERMINATED |
| **R2 total** | | | | **$13.79** | `list-pods`: empty |

**4090 Community was never available with CUDA ≥ 12.8 during the round.** Round 1's final billed cost was $6.70, not
the $5.96 reported earlier: the billing for the last hour of `zbo9w6afimsu0d` posted after that report.

## 7. Recommended next step (owner decision; nothing started)

1. **A tool-focused stage 2 on top of v1-tool, registered before running.** It would reuse the probe recipe:
   * ~6k steps;
   * ~70% Tool Corpus v2 + 30% replay;
   * LR 1e-5;
   * ~$2–3 on a 4090.

   Evaluate it on the same gates. The probe evidence (86.1 → 90.1 from v1-lr1e5) suggests it can close most of the gap.
   Keep the full-run candidate as the rollback.
2. Decide whether the ToolACE-holdout gate should be scored on the **clean slices** (random / native + random / k ≥ 16).
   The v0-pool-negative slice structurally favours a model trained on xLAM's distractor pool. **This decision belongs to
   the owner before the next result**; it is not applied retroactively here.
3. Human review of the translation pack (first pass: batches 000–005). `final_protected` stays closed until then.
4. Multimodal (M1) stays blocked until the text base is frozen.
