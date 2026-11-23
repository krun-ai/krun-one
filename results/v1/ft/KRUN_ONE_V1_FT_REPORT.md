# Krun One v1: fine-tune of K1-W1536-L22 on Krun Decisions v1 (2026-09-27/28)

## Verdict

**Registered gate verdict: CANDIDATE_REJECTED.** Rule: `configs/v1/ft_gates.yaml`, registered before any result.
Output: `comparison_vs_v03.md`.

Two gates fail. Both are OOD false acceptance measured at each model's own threshold:
* overall: +3.4 pp, limit 3.0;
* the unseen-tool slice: 0.0% → 21.6%.

A post-hoc analysis at **equal coverage** favours v1 on OOD by a wide margin (§3). An extra ToolACE regression check shows a
**real weakness in API/tool routing** (§4). Recommendation: **do not promote to production**. Keep the artifact as the
first real v1 candidate. Items that need an owner decision are in §8.

| | |
|---|---|
| artifact | `artifacts/krun-one-v1-candidate/` (`model.safetensors` sha256 `7362b07d…7018`, `spec.json`, `calibration.json`, `provenance.json`, `SHA256SUMS`) |
| init | `runs/v1/mlm1b_import/k1_0b.tar` (sha256 `061160e3…7565`), as decided (STOP_AT_1B / FINE-TUNE_FROM_0B) |
| architecture | K1-W1536-L22, 879M, unchanged. Multimodal scaffold off |
| data | Krun Decisions v1: `data/kd1/train.jsonl`, 1,042,946 decisions, sha256 `ddcf6d1b…40ba` |
| RunPod | final status **TERMINATED**. `list-pods` = empty |

## 1. Training

| item | value |
|---|---|
| objective | as validated (A_margin): choice softmax CE + outlier exposure on unanswerable; noul 2-way CE; score CE; multi sigmoid/BCE over applicability (+0.25 aux on choice); public abstention = top1 − top2 margin |
| optimizer | AdamW, bf16 autocast, gradient checkpointing, dynamic token batching (16,384 padded tokens / 64 questions), grad clip 1.0, 3% warmup + linear decay |
| LR | **encoder 2e-5 / heads 5e-5**, chosen by the registered LR smoke (`lr_smoke.json`): 3 candidates × 36k decisions, score = mean(intent acc, tool acc, noul AUROC, multi F1) on V fit, 79.0 / 84.6 / 88.6 for 5e-6 / 1e-5 / 2e-5 |
| curriculum | one epoch, progressive ordering: D1 core [0, .55], D2 hard/cross-language [.2, .9], D3 OOD/minimal pairs/adversarial [.45, 1] |
| steps | 21,757 (1 epoch), 4.0 h on an RTX 4090 (0.57–0.72 s/step), peak VRAM 16.7 GB |
| loss | 1.52 → 0.27 by step 7k; up to ~0.6 when D3 started (step ~8–10k, harder data), then down; grad-norm spikes clipped |
| checkpoint / resume | atomic every 30 min (model, optimizer, scheduler, scaler, all RNGs, curriculum state, config / data / init hashes; resume refuses a mismatch). Bit-identical resume tested (`tests/v1/test_finetune.py`) |

**Caveat.** 2e-5 is above the suggested 5e-6–1e-5 range. The registered rule chose it. A ~600-step smoke tends to favour
larger LRs, so a longer-horizon check of 1e-5 is a cheap follow-up (§8).

## 2. KrunBench v1 V: v1 vs v0.3

Same code path for both models. Calibration is fitted on the V FIT part; everything below is on the EVAL part. Full
tables: `comparison_vs_v03.md` / `eval_v1_ft.json`.

| metric | v0.3 | v1 |
|---|---:|---:|
| choice (intent) accuracy | 89.1 | **97.6** |
| tool accuracy (V: MultiWOZ + intent-as-function, see §4) | 81.2 | **98.5** |
| hard-negative accuracy | 87.7 | **97.8** |
| cross-language accuracy | 82.7 | **98.3** |
| accuracy at k = 64 / 128 | 73.6 / 64.9 | **95.4 / 94.1** |
| OOD AUROC (margin) | 0.875 | **0.975** |
| AURC ↓ | 0.073 | **0.027** |
| ECE ↓ / NLL ↓ (choice) | 0.034 / 0.501 | **0.031 / 0.140** |
| coverage @ selective accuracy 0.95 / 0.98 / 0.99 | 60.9 / 51.7 / 42.6 | **97.8 / 90.5 / 78.2** |
| permutation consistency | 86.0 | **97.8** |
| cross-language consistency (same group, different language) | 74.8 | **97.9** |
| noul ROC-AUC / accuracy | **0.973 / 89.0** | 0.960 / 85.1 |
| score MAE ↓ / Spearman / NLL ↓ | 0.810 / 0.744 / 1.307 | **0.631 / 0.843 / 1.092** |
| multi micro / macro F1 | n/a (no multi output) | 90.8 / 82.2 |

**Per language** (choice accuracy, v0.3 → v1):

| language | v0.3 | v1 |
|---|---:|---:|
| EN | 93.9 | 97.1 |
| PT-BR | 74.8 | **98.4** |
| PT-PT | 94.1 | 99.4 |
| ES | 85.4 | 98.2 |
| FR | 81.2 | 97.7 |

**Where the gains are real, and where they are inflated** (paired group bootstrap, 95% CI):

| slice | Δ acc (pp) | CI | reading |
|---|---:|---|---|
| native multilingual V (MASSIVE val, BANKING77 test EN + PT-BR, MultiWOZ dev) | **+10.2** | [+9.2, +11.1] | real |
| native non-EN only | **+12.6** | [+11.5, +13.9] | real multilingual gain |
| machine-translated V (KrunBench val → PT/ES/FR by the same Qwen3-8B) | +20.9 | [+18.0, +23.9] | **inflated**: v1 trained on the same MT system's output |
| KrunBench v1 val_eval (EN, CLINC / Bitext unseen labels, ToolACE) | +0.5 | [−0.8, +1.9] | **no EN gain** on never-trained sources (and no regression) |

## 3. OOD / abstention

At each model's own 0.98-selective-accuracy threshold (the registered gate), v1 accepts more unanswerable decisions:
* all buckets: 6.8% vs 3.3%;
* ToolACE native irrelevance: 17.7% vs 0.0%;
* the unseen-tool slice: 21.6% vs 0.0%.

**But v0.3 reaches its 0% by answering only 52% of the answerable decisions; v1 answers 90%.** At equal coverage
(post-hoc, not part of the registered rule):

| family / coverage | v0.3 OOD FA | v1 OOD FA | v0.3 unseen-tool FA | v1 unseen-tool FA | v0.3 selective acc | v1 selective acc |
|---|---:|---:|---:|---:|---:|---:|
| tool / 50% | 20.8 | **0.3** | 23.3 | **0.4** | 87.8 | **99.9** |
| tool / 70% | 34.3 | **6.0** | 38.1 | **8.5** | 82.0 | **98.1** |
| tool / 90% | 61.4 | **16.9** | 64.8 | **22.9** | 73.8 | **95.8** |
| intent / 90% | 31.6 | **4.2** | — | — | 87.1 | **98.4** |

Tool native-irrelevance AUROC: 0.752 → **0.936**.

**Reading:** the gate compares false acceptance at two very different operating points. The registered gate was
threshold-dependent, and in hindsight it should have been defined at matched coverage. That is a gate-design lesson,
not a reason to override the verdict. For serving, the tool threshold of v1 should target ~70% coverage (~6% FA)
instead of 0.98 selective accuracy. That is an owner/product choice (§8).

## 4. Tool routing: the honest part

V contains **no answerable ToolACE-style decisions**: the KrunBench v1 val ToolACE items are no-call or multi-call
items. The "tool +17 pp" of §2 therefore measures MultiWOZ and intent-as-function routing, which are formats v1 trained on.
Extra regression check (`regression_toolace_v0.json`) on the ToolACE items of the v0 regression sets. Neither model
trained on them. v0 = regression only.

| set | n | v0.3 | v1 |
|---|---:|---:|---:|
| val_tools (answerable) | 445 | 91.2 | **92.6** |
| **holdout_tools** (answerable; tools v0.3 never saw) | 416 | **91.8** | **84.1** |
| of which: native tools + hard negatives from the v0 pool | 176 | 90.3 | **75.0** |
| answerable-vs-irrelevant AUROC (margin, 171 unanswerable) | — | 0.971 | 0.978 |

The drop concentrates where negatives were mined from the v0 tool pool, which contains xLAM functions. v0.3 trained
on those functions (xLAM, 60k queries); v1 never saw them. Part of the gap is that distribution advantage. Part is
the expected weakness of the v1 tool block: 7,063 ToolACE queries only (`docs/KRUN_V1_TOOL_DATA_REPORT.md`). These
data cannot separate the two. **Conclusion: v1 does not yet beat v0.3 on API/tool routing, one of the required gain
areas.**

## 5. Latency (RTX 4090, same harness, model part of the worker, bf16, total p50)

| workload | v0.3 | v1 |
|---|---:|---:|
| 1 question, typical (k4 / 8 tools) | 17.3–18.2 ms | **14.9 ms** |
| 4 mixed questions | 19.9–20.0 ms | **16.6 ms** |
| 32 tools | 19.1–19.5 ms | **16.3–16.6 ms** |
| 64 tools (2 chunks) | 23.9–24.2 ms | 27.3 ms (1.13×) |
| 128 tools | 46.1 ms | **41.6 ms** |
| stress 16q × 32 tools × 512 (separate) | 194 ms | 230 ms (1.18×) |

Typical ≤ 50 ms: **met** (0.83–0.86× v0.3). Raw: `latency_rtx4090.json`. The text path is unchanged by the disabled
multimodal scaffold.

## 6. Dataset (Krun Decisions v1): summary

Details: `results/v1/kd1_stats.md`, `docs/KRUN_DECISIONS_V1_BUILD.md`, `docs/KRUN_DECISIONS_V1_SOURCES_AUDIT.md`,
`docs/KRUN_V1_TOOL_DATA_REPORT.md`, `results/v1/synthetic_pilot_v2/REPORT.md`.

**Mix**

| task | share |
|---|---:|
| tool | 30.4% |
| choice | 27.1% |
| OOD | 18.9% |
| noul | 9.6% |
| score | 7.7% |
| multi | 6.3% |

**Languages**

| language | share |
|---|---:|
| EN | 43.3% |
| PT | 22.3% (PT-BR 18.9 / PT-PT 3.4) |
| ES | 17.4% |
| FR | 16.9% |

**Cross-language and candidate counts**
* 98.3% of non-EN decisions are cross-language; 92.2% use an EN schema.
* Option counts go from 2 to 128; k ≥ 128 is 0.8%.

**Origins**

| origin | share | target |
|---|---:|---:|
| real | 22.6% | 20–30% |
| derived | 32.4% | 35–40% |
| MT | 41.8% | 20–25% |
| rules | 3.2% | 5–10% |
| LLM | 0.07% | 10–15% |

**Synthetic.** Pilot v2 reached 46.2% acceptance and **failed the gate**, so nothing was scaled. 695 audited items are used.

**Minimal pairs.** 18,684 decisions in 4,376 sets, mostly mined natural contrasts.

**Leakage.** 0 in the builder, and 0 in the independent `build_krunbench_v1.py --check`. The first build had 9,635
protected-tool name hits (Bitext intent `review`). The gate blocked training until they were fixed.

**Sources.**
* Approved: MASSIVE (CC-BY-4.0 via NOTICE.md), Banking77 PT-BR (approved with notice: GPT translation + human review),
  MultiWOZ 2.2 (MIT).
* Out: XNLI (CC-BY-NC), PAWS-X (share-alike origin).
* Still out: xLAM, SGD, MTOP, HWU64, BFCL, BoolQ, SST-5.

## 7. RunPod cost report (this phase; RunPod pod billing)

| pod | GPU | region | $/h | use | runtime | cost | final status |
|---|---|---|---:|---|---|---:|---|
| `6yxequ2m641eg8` | RTX 4090 Community | — | 0.34 | none: known broken host `-64411a5a`, terminated before setup | < 1 min | 0.00 | TERMINATED |
| `vz93igvcqa7fs9` | RTX 5090 Community | — | 0.69 | health check 6/6; MT 142k + describe + translate_desc 10k (Qwen3-8B); judge 152k + pilot verify (Granite 3.3 8B) | 1 h 57 min | 1.38 | TERMINATED |
| `q0l33aarj72aop` | RTX 3090 Community | — | 0.22 | none: health check FAILED (CUDA unknown error, CUDA 13.0 host) | ~1 min | 0.01 | TERMINATED |
| `zbo9w6afimsu0d` | RTX 4090 Secure | US-IL-1 | 0.74 | health check 6/6; LR smoke; full fine-tune; V eval; v0.3 signals; latency | 6 h 55 min | 4.57 | TERMINATED |
| **total** | | | | | | **5.96** | `list-pods`: empty |

About the 4090 pod:
* It was idle or light for ~2 h: upload, the dataset rebuild after the leakage finding, and an encoding deadlock that was
  found and fixed.
* During the rebuild it ran the v0.3 baseline and the latency benchmark.
* Community 4090s were unavailable or only on broken CUDA 13.0 hosts, so Secure was used, as the brief allows.

**Processed.** 1,042,946 decisions (≈ 196M text tokens) in 21,757 steps, plus 3 × 36k decisions (LR smoke).
**Throughput.** 0.57–0.72 s/step. **Peak VRAM.** 16.7 GB.

## 8. Next steps (owner decisions first)

1. **Tool data** is the blocker for a production v1. Options:
   * reintroduce xLAM after a license decision (owner said: not for now; incremental, ~$4 + ~½ day);
   * SGD / MTOP license review;
   * synthetic pilot v3 with a stricter hard-negative verifier.

   Do not scale 2M/3M by adding more decisions per ToolACE query. That is the data bottleneck the 1M run exposed.
2. **Abstention operating point.** Re-register the OOD gate at matched coverage (or pick the tool coverage target:
   ~70% → ~6% FA). Then re-evaluate this same checkpoint; no retraining is needed.
3. **LR check.** A 1e-5 run for the full horizon (~$3) would show whether 2e-5 cost tool generalisation (holdout_tools).
4. **Before any final claim:**
   * human-review the 12,000 KrunBench v1 translation jobs (PT/ES/FR final slices do not exist yet);
   * open `final_protected` once, for a frozen candidate.
5. Noul is slightly below v0.3 (AUROC 0.973 → 0.960). Worth a look when the mix is revised; it is not a gate.

## 9. Files

| what | where |
|---|---|
| candidate artifact + checksums | `artifacts/krun-one-v1-candidate/` |
| run (signals, eval, calibration, train_result) | `runs/v1/ft/krun-one-v1-ft/` |
| comparison vs v0.3 + gates | `results/v1/ft/comparison_vs_v03.{md,json}`, `results/v1/ft/eval_v1_ft.json` |
| ToolACE regression | `results/v1/ft/regression_toolace_v0.json`, `runs/v1/ft/regression_toolace_v0/` |
| LR smoke | `results/v1/ft/lr_smoke.json` |
| latency | `results/v1/ft/latency_rtx4090.json` |
| pod logs | `results/v1/ft/pod_logs/` |
| training config / gates | `configs/v1/finetune.yaml`, `configs/v1/ft_gates.yaml` |
| code | `scripts/v1/{kd1_contexts,kd1_gpu_jobs,synthetic_pilot_v2,build_decisions_v1,kd1_stats,train_decisions,dump_signals_v03,kd1_evaluate,kd1_report,regression_toolace_v0}.py`, `scripts/v1/runpod_{kd1_gpu,finetune}.sh`, `src/krun_one/v1/corpus/` |
| tests | `tests/v1/test_finetune.py`, `tests/v1/test_kd1_corpus.py` |
