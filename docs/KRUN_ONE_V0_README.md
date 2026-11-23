# KRUN-005 — Krun One v0

Krun One = the public **Laya** checkpoint (ModernBERT-large, 421,293,827 parameters), **architecture unchanged**,
fine-tuned with supervised cross-entropy on Krun decision data (intent routing, tool routing, OOD/abstention,
hard negatives, multilingual). Goal: a closed-beta release candidate that beats upstream Laya on the product tasks.

> Results: [`results/laya_vs_krun_one.md`](results/laya_vs_krun_one.md) · error analysis:
> [`results/error_analysis_v0.md`](results/error_analysis_v0.md) · final report: section **Final report** below.

<!-- METHODS -->

## 1. Architecture and base checkpoint

`src/krun_one/model.py` is a self-contained port of upstream `rl_common.py` (`DecisionModel`, option rendering,
`build_sequence`). `tests/test_laya_equivalence.py` checks it is **bit-identical** to upstream (serialization at both
budgets, logits and act logits of the real checkpoint). Nothing under `./laya` is imported by Krun One code or modified;
`laya_adapter.py` only reads the weight file, encoder config, tokenizer and `rl_agent_config.json`.

| | |
|---|---|
| base | `convaiinnovations/laya` repo root @ `1c5edc17a7acd8701df6fc341c0d179f1c62c982`, `model.safetensors` sha256 `891102d3…8d86c` |
| encoder | ModernBERT-large, 28 layers, hidden 1024, 16 heads |
| head | type embedding + 2-layer transformer decision head + option-marker scorer (softmax over the decision's options) |
| act head | kept (strict state-dict compatibility), **frozen and unused** — upstream `act_probability` saturates at ~1.0 |
| serialization | Laya's: `[CLS] "choice question: {instructions}" [SEP] ([MASK] option)* [SEP] context [SEP]`, 48 tokens max per option |
| budgets | `max_len` 1024 / `head_max_len` 896 (Laya ships 512 / 192: 16 described tools need ~784 option tokens) |
| option text | intent label → humanized (`card_arrival` → `card arrival`); tool → `name: official description` |

## 2. Data (`scripts/prepare_training_data.py`, `configs/datasets.yaml`)

Every decision uses the KrunBench schema (`krunbench.decision.v0`) and passes KrunBench's validator (0 errors).
`data/` is gitignored and reproducible; stats + sha256 in `results/dataset_stats.json`.

**Sources**

| task | source | license | use |
|---|---|---|---|
| intent | KrunBench v0 **train** items: BANKING77 (8,993), MASSIVE en-US/pt-PT/es-ES (29,910 = 9,970 × 3) | CC-BY-4.0 | train |
| intent | KrunBench v0 **validation** items | CC-BY-4.0 | `val_hard` suite (+ `kb_val` verbatim) |
| intent | HWU64 | mirror license unconfirmed | **never trained on**; secondary slice of `kb_val`/`kb_test` only |
| tool | xLAM function-calling-60k (Salesforce) via the verbatim ungated re-host `argilla/apigen-function-calling` @ `170a8a7e`, rows `origin == xLAM` with exactly one distinct called function (39,391) | CC-BY-4.0 | train / val / holdout |
| tool | ToolACE (Team-ACE) @ `6bda777c`: first user turn whose first assistant turn calls exactly one available function (7,036) | Apache-2.0 | train / val / holdout |
| OOD tool | xlam-irrelevance-7.5k (MadeAgents) @ `34323bf0`: none of the given tools applies (6,034) | CC-BY-4.0 | train / val / holdout |
| tool | BFCL v3 (KrunBench test subset) | Apache-2.0 | **evaluation only** (final report); never trained or tuned on |

Tool names and descriptions are always the official strings of the source. The synthetic `distilabel` half of the
argilla dataset was not used. **No LLM-generated (teacher) data was added in v0** — the public sources covered the
gaps (tool routing, no-tool/irrelevance), so no provenance for synthetic data is needed.

**Leakage control.** Tool queries are deduplicated by normalized text; a query that occurs anywhere in KrunBench v0
(all splits, incl. BFCL) is dropped (3). The split is decided per query group (an xLAM query and its
xlam-irrelevance variant share a group and a split). 6% of gold tools (570) are **held out entirely** (never in the
train pool, not even as distractors) → `unseen_tool` slices of `val_tools`/`holdout_tools`. The build aborts if a
group appears in both train and an eval file.

**Option-set construction (explicit distributions, exact counts):**

| | answerable option counts | negatives |
|---|---|---|
| intents (BANKING77 ×4 decisions/item, MASSIVE ×2/item/locale) | 2: 15%, 4: 25%, 8: 30%, 16: 25%, 32: 5% | hard_v1 65% / random 35% |
| tools (1 decision/item, +1 for 25% of items) | 2: 10%, 4: 25%, 8: 30%, 16: 25%, 32: 10% | native tools first, then pool hard_v1 65% / random 35% |
| unanswerable (all kinds) | 2: 15%, 4: 30%, 8: 30%, 16: 25% | see OOD |

**Hard-negative mining v1** (`src/krun_one/mining.py`): reciprocal-rank fusion (k=10) of
(1) label similarity (KrunBench v0 name + prototype TF-IDF), (2) **example-level** similarity between *this* context and
each label centroid (multilingual-e5-small, so pt/es contexts get their own neighbours), and (3) **confusion data**:
a leave-one-group-out kNN classifier over train embeddings. Tools: RRF of query↔tool, gold↔tool (e5 embeddings of
`name: description`) and tool-name char-TF-IDF similarity, over a 200-tool shortlist. Tools that may be
*interchangeable* with the gold (same humanized name, or e5 doc cosine ≥ 0.95 — threshold set by inspecting pairs) are
never negatives. Negatives are sampled (seeded) from the top 2(k−1) of the fused ranking, so examples of one label get
different — still hard — sets. Every decision records `generation.negative_strategy` and `metadata.negatives`.

**OOD / `answerable = false`** (no artificial "other" option is ever injected):

| kind | construction |
|---|---|
| intent label removed (near-OOD) | gold removed, options = hard_v1 neighbours of the gold |
| intent cross-dataset (far-OOD) | BANKING77 ↔ MASSIVE with KrunBench's exclusion lists |
| cross-task | intent context vs tools, or tool query vs intent labels, drawn only from the least-similar half (e5) |
| tool removed | gold removed; the item's *native* other tools (genuinely wrong, same domain) + random pool tools below the 90th percentile of query similarity and < 0.90 cosine to the removed gold |
| native irrelevance | xlam-irrelevance tool list (+ same safe fill) |

**Permutation robustness.** Every decision's order is a seeded shuffle; 6% of answerable train decisions get one
rotated copy (`suite = permutation`, shared `consistency_group`); eval suites have 4-rotation Latin-square groups.

**Resulting sizes** (`results/dataset_stats.json`)

| file | decisions | answerable | unanswerable |
|---|---:|---:|---:|
| `train.jsonl` | 172,430 (intent 108,812 · tool 63,618) | 154,065 | 18,365 (10.6%) |
| `val_hard.jsonl` | 9,339 | 8,051 | 1,288 |
| `val_tools.jsonl` | 3,466 | 2,602 | 864 |
| `holdout_tools.jsonl` | 3,828 | 2,921 | 907 |

Train by source: MASSIVE 68,935 (en-US 22,990 · pt-PT 23,004 · es-ES 22,941), BANKING77 39,852, xLAM 49,220,
ToolACE 9,270, xlam-irrelevance 5,153. Answerable by option count: 2: 20,463 · 4: 38,514 · 8: 46,188 · 16: 38,557 ·
32: 10,343. Tokens (Laya tokenizer, 1024/896): 25.3M; intents mean 69 tokens, tools mean 280 (p95 738); 0 decisions
dropped.

## 3. Training (`configs/krun_one_v0.yaml`, `src/krun_one/training.py`)

* Loss: soft-target **cross entropy** over each decision's options — one-hot for answerable, **uniform** for
  `answerable = false` (outlier exposure), weight 0.5. No RLCD / REINFORCE / GRPO, no new heads.
* AdamW (wd 0.01), lr encoder 2e-5 / head+scorer+type-emb 1e-4, 5% warmup + linear decay, grad clip 1.0, bf16 autocast,
  1 epoch, length-bucketed batches ≤ 8,192 padded tokens / ≤ 64 decisions (4,106 steps).
* **Config A**: decision head + scorer + type embeddings + top 8 of 28 encoder layers (+ final norm).
  **Config B**: full fine-tuning. Same data, seed, schedule. Choice rule registered before training (config file).

**Checkpoint / resume** (`src/krun_one/checkpoint.py` = KRUN-004's validated `bakeoff.checkpoint`): every 100 steps,
atomic (`.tmp` + rename, `trainer_state.json` last); model, optimizer, scheduler, GradScaler, Python/NumPy/Torch
CPU/CUDA RNG, shuffle-generator state, global step, next batch, running loss, log, training config (resume refuses a
different config). `--resume-from-checkpoint latest|<dir>`. Validation: see §6.

## 4. Evaluation protocol (`src/krun_one/evaluation.py`, `scripts/evaluate.py`)

Laya and Krun One go through **the same code path**: same decisions, options, `build_sequence`, budgets, dtype
(bf16 on the RTX 4090) and metrics (`krunbench.metrics`). Laya is additionally reported at its shipped 512/192 budget.

| set | content | role |
|---|---|---|
| `kb_val` | KrunBench v0 validation, verbatim (13,848) | validation |
| `val_hard` | KrunBench validation items with hard_v1 option sets, k = 4/8/16/32, + OOD + permutations | validation |
| `val_tools` | held-out tool queries (seen + unseen tools), tool-removed / irrelevance / cross-task OOD, permutations | validation |
| `holdout_tools` | same construction, disjoint | final (once) |
| `kb_test` | KrunBench v0 test, verbatim, incl. BFCL | final (once) |

Validation sets are split by group into **val_fit** (temperatures, thresholds) and **val_eval** (all reported
validation numbers), with KRUN-004's seed, so `kb_val`'s split equals KRUN-004's. The KrunBench test split — the blind
test opened in KRUN-004 — and `holdout_tools` are used **once**, after `results/krun_one_decisions.json` froze every
choice (`evaluate.py --final` refuses to run before). Calibration: one temperature per task family (intent / tool),
LBFGS on val_fit NLL. Abstention: top-1 probability, top1−top2 margin, 1 − normalized entropy; thresholds per family
chosen on val_fit for a target selective accuracy (primary 0.98; 0.99/0.95 also reported).

<!-- RESULTS -->

## 5. Results (all numbers from `results/*.json`; val = val_eval halves, final = opened once)

### 5.1 Headline — Laya vs Krun One v0 ([`results/laya_vs_krun_one.md`](results/laya_vs_krun_one.md))

| Metric | Laya | Krun One v0 | Δ |
|---|---:|---:|---:|
| Intent accuracy, KrunBench val (Banking77 + MASSIVE en/pt/es) | 64.04% | **96.01%** | +31.97 pp |
| Intent accuracy, hard negatives k=4–32 (val_hard) | 49.45% | **91.29%** | +41.85 pp |
| Tool-routing accuracy (val_tools) | 66.61% (68.05% at 512/192) | **94.86%** | +26.81 pp vs best Laya |
| OOD AUROC, KrunBench val / val_tools (`top1_top2_margin`) | 0.661 / 0.674 (0.737 at 512/192) | **0.954 / 0.990** | |
| False acceptance @ 0.98, KrunBench val / val_tools | 0.00% / 4.32% — *at 2.4% / 31.4% coverage* | 8.07% / 3.41% — *at 87.9% / 91.3% coverage* | |
| ECE (T-scaled), KrunBench val / val_tools | 0.0401 / 0.0391 | **0.0087 / 0.0170** | |
| AURC, KrunBench val / val_tools | 0.156 / 0.106 | **0.0035 / 0.0068** | |
| Permutation consistency, KrunBench val / val_tools | 67.8% / 64.6% | **98.9% / 91.7%** | |
| **FINAL** KrunBench test intents (Banking77 + MASSIVE) | 66.98% | **96.09%** | +29.11 pp |
| **FINAL** BFCL (KrunBench test, never trained on) | 82.83% (83.39% at 512/192) | **92.33%** | +8.94 pp vs best Laya |
| **FINAL** holdout_tools | 69.53% | **96.42%** | +26.89 pp |
| **FINAL** OOD AUROC KrunBench test / holdout_tools / BFCL irrelevance | 0.669 / 0.724 / 0.701 | **0.947 / 0.993 / 0.804** | |
| **FINAL** ECE KrunBench test | 0.0327 | **0.0072** | |
| CPU p50, batch 1 (i7-10750H, 6 threads, fp32) | 351 ms | 356 ms | same architecture |
| GPU p50, batch 1 (RTX 4090, bf16) | 10.4 ms | 10.9 ms | |
| Parameters | 421,293,827 | 421,293,827 | |
| RAM (CPU RSS) | 2,565 MiB | 2,565 MiB | |
| VRAM peak, batch 1 / 32 | 2,338 / 3,002 MiB | 2,338 / 3,002 MiB | |

Laya's 0% false acceptance on KrunBench val is not a strength: to reach 98% selective accuracy its threshold accepts only
2.4% of in-distribution decisions. At its shipped 512/192 budget Laya is slightly better on tools (68.05%), so the
table always compares against the better Laya setting.

### 5.2 Config A vs B (validation; rule registered before training)

| | trainable params | train time (RTX 4090) | peak VRAM | intent (kb_val) | tools | mean | hard | OOD AUROC kb_val / tools |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A (head + top 8/28 layers) | 124.3M | 610 s | 6.7 GB | 86.76% | 87.45% | 87.11% | 78.02% | 0.802 / 0.934 |
| **B (full)** | 421.0M | 1,155 s | 16.8 GB | **96.01%** | **94.86%** | **95.44%** | **91.29%** | **0.954 / 0.990** |

B − A = +8.33 pp → **B** (`results/krun_one_decisions_v0.json`). Freezing 20 of 28 layers costs ~9 pp: the lower
layers must adapt to the Krun input distribution (short labels, long tool lists, pt/es).

### 5.3 Intent routing and multilingual (accuracy, core)

| slice | Laya val | Krun One val | Laya **test** | Krun One **test** |
|---|---:|---:|---:|---:|
| BANKING77 | 77.88% | 96.73% | 80.13% | **97.60%** |
| MASSIVE en-US | 73.41% | 97.59% | 73.50% | **96.97%** |
| MASSIVE pt-PT | 55.14% | 94.97% | 56.69% | **94.62%** |
| MASSIVE es-ES | 55.69% | 95.08% | 57.13% | **95.12%** |
| HWU64 (secondary, never trained on) | 76.56% | 92.41% | 76.02% | 93.40% |
| val_hard k=4 / 8 / 16 / 32 | 56.2 / 53.6 / 45.5 / 42.8% | 93.9 / 92.9 / 91.4 / **86.9%** | | |

Macro-F1 (kb_val core): Laya 0.7025 → 0.9453; KrunBench test 0.8118 → 0.9203.

### 5.4 Tool routing (accuracy, core)

| slice | Laya val_tools | Krun One val_tools | Laya holdout | Krun One holdout | Laya BFCL (test) | Krun One BFCL (test) |
|---|---:|---:|---:|---:|---:|---:|
| all | 66.61% | 94.86% | 69.53% | 96.42% | 82.83% | 92.33% |
| seen tools / **unseen tools** | 71.2 / 62.5% | 97.0 / **92.9%** | 67.8 / 71.0% | 97.2 / **95.8%** | — | — |
| near-duplicate distractor | 36.9% | 91.8% | 40.0% | 92.0% | 82.3% | 86.2% |
| description-dependent (gold name ∉ query) | 52.2% | 93.5% | 56.9% | 92.9% | 72.2% | 86.4% |
| large tool sets (k ≥ 16) | 46.7% | 91.6% | 52.2% | 95.6% | n=2 | n=2 |
| k = 32 | 39.7% | 90.8% | 42.7% | 95.4% | — | — |

BFCL sets are small (2–37 functions, median 3), so BFCL "large tool set" has n = 2 and is not interpretable; the
large-set evidence comes from val_tools/holdout_tools.

### 5.5 OOD / abstention

OOD AUROC by kind (`top1_top2_margin`, T-scaled): near-OOD label-removed 0.629 → **0.946** (val) / 0.653 → **0.951**
(test); far-OOD cross-dataset 0.780 → **0.983** / 0.761 → **0.989**; tool removed 0.669 → **0.990**; xlam irrelevance
0.655 → **0.988**; **BFCL live-irrelevance 0.701 → 0.804** (test).

Operating point (target selective accuracy 0.98, per-family thresholds from val_fit: intent margin ≥ 0.831, tool
margin ≥ 0.730):

| set | coverage | selective acc. | OOD false acceptance |
|---|---:|---:|---:|
| KrunBench val (intents) | 87.9% | 99.2% | 8.1% (near 9.9%, far 1.3%) |
| KrunBench **test** (intents) | 88.3% | 99.1% | 7.6% (near 9.6%, far 0.5%) |
| val_tools | 91.3% | 98.1% | 3.8% |
| holdout_tools | 91.9% | 98.8% | 3.0% |
| **BFCL (test)** | 82.3% | 97.7% | **39.1%** (live irrelevance) |

**Abstention protocol chosen** (rule: best mean OOD AUROC, ties within 0.005 → lower AURC): candidates
top1 0.9728 / 0.00519, **margin 0.9719 / 0.00516**, entropy 0.9761 / 0.00690 → `top1_top2_margin` (entropy wins AUROC
but has 33% higher AURC; top1 and margin are practically tied). Abstain when margin < family threshold.

### 5.6 Calibration

Temperatures (val_fit): intent T = 1.140, tool T = 1.451 (Laya: 1.355 / 1.010). Krun One is already well calibrated
raw (kb_val ECE 0.0079 raw vs 0.0087 T-scaled; NLL 0.146; Brier 0.067); temperature helps on tools (ECE 0.0285 →
0.0170, NLL 0.204 → 0.178). Laya: ECE 0.064 raw → 0.040, NLL 0.98, Brier 0.46. Reliability tables in each
`results/eval_*.json` (`reliability`).

### 5.7 Permutation robustness (same option set, 4 rotations)

| | Laya | Krun One |
|---|---:|---:|
| KrunBench val / test | 67.8% / 63.6% | **98.9% / 97.2%** |
| val_hard | 47.6% | **97.6%** |
| val_tools / holdout_tools | 64.6% / 70.8% | **91.7% / 95.6%** |

### 5.8 Error analysis ([`results/error_analysis_v0.md`](results/error_analysis_v0.md), val_eval only)

| category | Krun One | Laya |
|---|---:|---:|
| option count (intent kb_val k2 / k4 / k8 / k16) | 2.3 / 2.9 / 5.0 / 7.0% err | 25.0 / 29.3 / 36.8 / 43.8% |
| option count (val_hard k32 / tools k32) | 13.1% / 9.2% | 57.2% / 60.3% |
| multilingual (val_hard en-US / pt-PT / es-ES) | 8.5 / 9.3 / 9.5% | 38.8 / 60.9 / 58.8% |
| tool description-dependent / near-duplicate | 6.5% / 8.2% | 47.8% / 63.1% |
| unseen vs seen tools | 7.1% vs 3.0% | 37.5% vs 28.8% |
| OOD false accept @0.98: intent near / far; tool removed / irrelevance | 9.9% / 1.3%; 4.0% / 3.6% | (Laya accepts ~nothing) |
| permutation flips (kb_val / tools) | 1.1% / 8.3% of groups | 32.2% / 35.4% |

Dominant residual errors are **semantic confusions between genuinely overlapping labels**: MASSIVE
`general_quirky` (catch-all), `calendar_set`↔`calendar_query`, `takeaway_order`↔`takeaway_query`, `play_radio`↔`play_music`;
BANKING77 `card_arrival`↔`card_delivery_estimate`, `failed_transfer`↔`declined_transfer`. The same pairs dominate
near-OOD false acceptance (removing the gold leaves a legitimately acceptable neighbour). **v0.1 was not run**
(`results/krun_one_decisions.json` → `v0_1`): no single clear, data-fixable failure; a second round would have been
a tuning loop.

## 6. Checkpoint / resume validation (RunPod RTX 4090, real data, config B)

`scripts/runpod/resume_test.sh`, logs in `runs/runpod_logs/`:

| test | result |
|---|---|
| fast kernels: two *fresh* identical 60-step runs | differ from step 10 (bf16 SDPA / scatter-add nondeterminism) → bitwise comparison impossible in this mode |
| `--deterministic` (deterministic algorithms, math SDPA, gradient checkpointing = memory only): continuous 60 steps vs **kill -9** right after `checkpoint-40` + `--resume-from-checkpoint latest` | **206/206 tensors bit-identical** (max \|Δ\| = 0), optimizer, scheduler, GradScaler, RNG (Python/NumPy/Torch CPU/CUDA/shuffle) identical, trainer state identical except timing telemetry, losses at steps 50/60 identical (1.303079 / 1.020914) |
| unit tests (CPU) | `tests/test_training_resume.py`: stop at step 5 + resume = continuous run bit-for-bit across the epoch boundary; config mismatch refused; checkpoint contents |

Smoke (60 steps, config B, 8,192 tokens/step): loss 1.31 → 1.00, no NaN, no OOM, 16.8 GB peak VRAM, ~26k tokens/s.

## 7. Export, reload, serving

`artifacts/krun-one-v0/` (gitignored, 1.6 GB): `model.safetensors` (fp32, sha256 `21e821ee…4a26e12` = `runs/B/final`),
`config.json`, `tokenizer/`, `krun_config.json` (budgets, temperatures, `top1_top2_margin` thresholds intent 0.831 /
tool 0.730), `metrics.json`, `provenance.json` (Laya sha256, data file hashes, sources/licenses, training summary,
decisions), `README.md`. Loads with `KrunOne.from_pretrained(path)` — only torch + transformers + safetensors +
`krun_one/model.py`; no `./laya`, no training code (asserted in `tests/test_export_reload.py`).

**Reload test** (`scripts/reload_test.py`, new process, `laya` scrubbed from `sys.path`): intent ✓
(`transaction_charged_twice`, 0.999), tool ✓ (`calendar_search`, 0.983), OOD ✓ (abstained, margin 0.15) — on GPU
(bf16) and on local CPU (fp32). Fidelity vs the evaluated logits (64 val_tools decisions): GPU bf16 argmax agreement
100% (max |Δlogit| 0.10, batch-composition effects in bf16); CPU fp32 97% (2/64 near-ties flip, max |Δlogit| 0.14).

**Serving benchmark** (end-to-end `decide_batch`, half intent / half tool requests with 4–16 options):

| | batch 1 p50 / p95 | batch 8 p50 / p95 | batch 32 p50 / p95 | throughput (dec/s) b1 / b8 / b32 | VRAM peak b1 / b8 / b32 |
|---|---|---|---|---|---|
| RTX 4090, bf16 | 10.9 / 11.8 ms | 33.5 / 47.7 ms | 202 / 229 ms | 92 / 233 / 159 | 2,338 / 2,466 / 3,002 MiB |
| CPU i7-10750H, 6 threads, fp32 | 356 / 1,148 ms | 8.4 / 12.6 s | — | 2.1 / 0.9 | RSS 2,565 MiB |

Batch 32 is slower per decision than batch 8 because mixed requests pad to the longest tool list (~700 tokens);
serving should bucket by length. Laya measures the same (same architecture): GPU 10.4 ms, CPU 351 ms.

## 8. Cost

| | |
|---|---|
| pod | `krun-one-train` (`8kvd3anfhb0j1m`), RTX 4090 24 GB, Secure Cloud EU-RO-1, **$0.74/h**, network volume `krun-research` |
| why a new pod | the old `krun-training` pod could not start ("not enough free GPUs on the host"); same image/volume/config, 4090 was the cheapest validated GPU with stock in the volume's DC |
| active time | 02:09:26 → ~04:05 UTC, uptime 6,958 s = **1.93 GPU-hours** |
| cost | ≈ **$1.43** (uptime × price; RunPod billing showed $0.82 at the time of writing — it lags) |
| of which training | A 610 s + B 1,155 s ≈ 0.49 GPU-h ≈ $0.36 |
| storage | volume 100 GB (~$7/month, unchanged); 59 GB used (runs/A, runs/B with 2 checkpoints each, artifact, KRUN-004) |

Both pods are stopped (EXITED); the volume is preserved.

## 9. Tests

`.venv/bin/python -m pytest` → **15 passed**: Laya equivalence (4, real checkpoint), training/resume (4), export +
reload in a new process (1), data/serialization/mining/slices (6).

## 10. Limitations

1. **BFCL irrelevance does not transfer**: AUROC 0.804 and **39% false acceptance** at the tool threshold calibrated on
   xLAM-style validation. Real user "no tool applies" queries (BFCL live) are harder than the constructed ones
   (tool removed / xlam-irrelevance, AUROC 0.99). Tool OOD in production needs real irrelevance data before relying on
   the threshold.
2. Near-OOD (label removed) false acceptance ~8–10%, dominated by overlapping labels: the benchmark's own ambiguity.
3. One seed per configuration; no variance estimate (differences here are 8–40 pp, far above seed noise, but small
   deltas like ECE should not be over-read).
4. Evaluation ran in bf16 on GPU; CPU fp32 inference flips ~3% of near-tie decisions vs the evaluated logits.
5. Constructed tool option sets can contain genuinely interchangeable tools (near-duplicate filter at e5 cosine 0.95 is
   imperfect); the "native" xLAM tool lists are the most reliable negatives.
6. Unanswerable decisions are 10.6% of training; OOD kinds are constructed, not sampled from real traffic.
7. Multilingual = MASSIVE pt-PT / es-ES only (no pt-BR in MASSIVE; no non-Latin scripts; Laya's English tokenizer).
8. HWU64 overlaps MASSIVE en-US textually (KRUN-003) — secondary only.
9. Serving cost is Laya's: 421M parameters, ~356 ms/decision on a laptop CPU → GPU serving (or Krun Tiny) for production.
10. Tool descriptions are truncated to 48 tokens per option (Laya serialization); 32-tool requests get ~27 tokens each.

<!-- FINAL -->

# KRUN-005 FINAL REPORT

1. **Architecture**: unchanged Laya `DecisionModel` — ModernBERT-large (28 layers, d 1024, 16 heads) + type embedding +
   2-layer decision head + option-marker scorer; act head frozen/unused. Budgets raised to 1024 / 896.
2. **Checkpoint**: `convaiinnovations/laya` repo root @ `1c5edc17…`, sha256 `891102d3…8d86c` (421,293,827 params).
3. **Dataset**: KrunBench v0 train (BANKING77, MASSIVE en/pt/es) + xLAM-60k + ToolACE + xlam-irrelevance; no HWU64, no
   BFCL, no synthetic/LLM data (§2).
4. **Examples**: 172,430 training decisions (154,065 answerable, 18,365 unanswerable); eval: kb_val 13,848, val_hard 9,339,
   val_tools 3,466, holdout_tools 3,828, kb_test 21,891.
5. **By task**: intent 108,812 (MASSIVE 68,935, BANKING77 39,852), tool 63,618 (xLAM 49,220, ToolACE 9,270, irrelevance
   5,153); suites core 145,344, permutation 8,721, near-OOD 9,434, far/cross-task 3,778, native OOD 5,153.
6. **By option count** (answerable): 2: 20,463 · 4: 38,514 · 8: 46,188 · 16: 38,557 · 32: 10,343 (explicit, exact).
7. **Hard negatives v1**: RRF of label similarity + example-level e5 similarity + kNN confusion (intents); query/gold/name
   similarity with interchangeable-tool exclusion (tools); 65% hard / 35% random, recorded per decision.
8. **Training**: SFT, soft-target CE (uniform target for unanswerable, weight 0.5), AdamW 2e-5 / 1e-4, 1 epoch, 4,106
   steps, bf16, ≤ 8,192 tokens/step.
9. **Layers trained**: A = head/scorer/type-emb + top 8 layers (124.3M); **released = B, full fine-tuning (421.0M)**.
10. **Duration**: B 1,155 s (19.3 min), A 610 s; whole pod session 1.93 h.
11. **RunPod cost**: ≈ $1.43 (1.93 GPU-h × $0.74, RTX 4090 Secure).
12. **Laya validation**: intent 64.04%, hard 49.45%, tools 66.61% (68.05% @512/192), OOD AUROC 0.661 / 0.674, ECE 0.040.
13. **Krun One validation**: intent 96.01%, hard 91.29%, tools 94.86%, OOD AUROC 0.954 / 0.990, ECE 0.0087.
14. **Delta**: +31.97 pp intent, +41.85 pp hard, +26.8 pp tools (vs best Laya), +0.29 / +0.25 OOD AUROC, ECE ÷4.6.
15. **Intent (final test)**: 66.98% → 96.09% (BANKING77 97.60%, MASSIVE en 96.97% / pt 94.62% / es 95.12%).
16. **Tools (final)**: holdout 69.53% → 96.42% (unseen tools 95.8%); BFCL 82.83% → 92.33%.
17. **OOD (final)**: AUROC KrunBench test 0.669 → 0.947, holdout tools 0.724 → 0.993, BFCL irrelevance 0.701 → 0.804.
18. **Calibration**: ECE test 0.0327 → 0.0072; T intent 1.140 / tool 1.451; well calibrated even raw.
19. **Abstention**: `top1_top2_margin`, per-family thresholds (0.831 / 0.730) for 98% selective accuracy → coverage
   88% (intents) / 92% (tools) at 99% / 98.8% selective accuracy; OOD false acceptance 7.6% / 3.0% (but 39% on BFCL).
20. **Permutation**: consistency 67.8% → 98.9% (kb_val), 63.6% → 97.2% (test), 64.6% → 91.7% (tools).
21. **Error analysis**: residual errors = overlapping-label confusions; 32-option decisions; unseen tools; BFCL-style
   irrelevance. No v0.1 (§5.8).
22. **Serving**: GPU 4090 bf16 p50 10.9 ms (b1), 33.5 ms (b8), 202 ms (b32); CPU laptop 356 ms (b1).
23. **VRAM/RAM**: 2.3–3.0 GB VRAM (b1–b32), 2.6 GB CPU RSS; training peak 16.8 GB (B).
24. **Artifact**: `artifacts/krun-one-v0/` (self-contained, `KrunOne.from_pretrained`).
25. **Reload test**: pass on GPU and CPU in a new process without `./laya`.
26. **Automated tests**: 15 passed.
27. **Limitations**: §10 (BFCL irrelevance transfer is the important one).
28. **Decision**: **READY for KRUN-006 Serverless** (closed beta), with one condition: tool-OOD abstention must be treated
   as advisory until validated on real "no tool applies" traffic (BFCL-style false acceptance 39%). Intent routing,
   tool selection, calibration and permutation robustness clearly beat Laya on every validation and final set.
29. **Next step**: KRUN-006 — package `KrunOne` into a RunPod Serverless worker (GPU, bf16, length-bucketed batching,
   artifact baked into the image or on the volume), log decisions + confidences in the closed beta, and collect real
   irrelevance / abstention cases; use them to recalibrate the tool threshold (and, if needed, a targeted v0.1 on
   real irrelevance data) before relying on abstention for tools.
