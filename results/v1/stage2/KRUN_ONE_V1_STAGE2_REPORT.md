# Krun One v1 — Stage 2 (tool-focused continuation of v1-tool), 2026-09-28

**Verdict: `FREEZE_TEXTUAL_BASE = false`.**

The protocol was registered before training (`configs/v1/stage2_protocol.yaml`). Stage 2 closes most of the ToolACE gap:
* holdout: 87.3 → **90.1**, vs v0.3 91.8;
* small-k: 86.6 → **91.7**;
* k = 2: 80.0 → **96.0**.

It still misses three registered gates:
* G1 (holdout ≥ 91.8);
* G2 (small-k ≥ 92.2, missed by 0.5 pp);
* one preservation gate (score Spearman −1.9 pp against a 1.0 pp limit).

Multilingual, PT-BR, cross-language, permutation, calibration and latency are preserved. `final_protected` was not opened. Multimodal was not started.

Raw data: `stage2_comparison.json`. Build: `stage2_build_manifest.json`. Latency: `latency_stage2.json`. Pod logs:
`pod_logs/`. Checksums: `SHA256SUMS`.

## 1. What ran

| item | value |
|---|---|
| start point | `artifacts/krun-one-v1-tool-candidate/model.safetensors` (sha `c52475ab…`, verified on the pod); architecture unchanged (K1-W1536-L22, 879M) |
| LR | encoder 1e-5, head 5e-5 (no sweep); warmup 3%, linear decay, AdamW, bf16 |
| data | `data/kd1_s2/train.jsonl`, 199,475 decisions (sha `9aad2018…`) |
| steps | **4,399** (one pass, never extended), 54 min on an RTX 4090 |
| checkpoints kept | 25% (step 1100), 50% (2200), 75% (3299), final (4399): all 4 weights are local with sha verified |

**Why 4,399 steps and not ~6k.** The mixture has the probe's decision count (~200k). The derived small-k decisions are
short, so more fit per 16k-token batch. The registered rule was "one pass, never extended", so the run was not padded
to 6k.

**Mixture** (`scripts/v1/build_stage2.py`):
* tool block 69.9% / replay 30.1%, with explicit quotas per curriculum phase:
  * S2P1 [0, .40] tool core;
  * S2P2 [.40, .75] hard / near-duplicate / description-dependent / cross-language;
  * S2P3 [.75, 1] small-k hard, tool OOD, none-applicable, adversarial;
  * replay spread uniformly over [0, 1].
* 27,620 **new** small-k decisions, derived by candidate subsetting (gold + the hardest negatives by gold or query
  similarity), plus 4,286 small-k none-applicable decisions.
* Tool-block sources:

  | source | decisions |
  |---|---:|
  | ToolACE | 49,047 |
  | ToolMind | 18,569 |
  | intent-as-function | 15,539 |
  | APIBench | 15,324 |
  | xLAM | 14,718 |
  | When2Call | 13,650 |
  | MultiWOZ | 5,040 |
  | Johin | 1,810 |
  | Hermes | 1,495 |

* **xLAM (low weight):**
  * 15,449 decisions: 7.7% of the total, 11.1% of the tool block (R2: 42.7% of the new data);
  * source Salesforce/xlam-function-calling-60k (argilla re-host) + xlam-irrelevance, license note CC-BY-4.0 + "research
    purposes only";
  * owner decision `approved_by_owner`;
  * 0 tool names exclusive to xLAM;
  * every protected tool excluded, so no influence on the unseen split.
* Replay:

  | kind | decisions |
  |---|---:|
  | intent choice | 30k |
  | intent OOD | 10k |
  | noul | 8k |
  | score | 8k |
  | multi | 7k |

  All languages are present.
* Languages:

  | language | decisions |
  |---|---:|
  | EN | 99k |
  | PT-BR | 35k |
  | FR | 31k |
  | ES | 31k |
  | PT-PT | 3.3k |

  100k decisions are cross-language.

**Leakage = 0:**
* the builder's schema check;
* MinHash near-duplicate / group / id against KrunBench v1 (incl. the `final_protected` manifest), val_ml and
  val_tools_v2;
* the strict protected-tool registry (BFCL v3, KrunBench v1 unseen tools, every ToolACE v0 val/holdout gold +
  distractor): name, alias, description, equivalent description, query;
* the independent `build_krunbench_v1.py --check`: LEAKAGE = 0 on 199,475 decisions.

The flag was added to `leakage_ok.flag`.

### A finding that corrects the R2 report

Applying the strict registry to the Stage 2 mixture hit **~400k protected tool candidates**. The R2 builder only
applied the registry to its *new* records. The round-1 base (ToolACE train, intents-as-functions) contains many of the
tools that appear as golds and distractors in the ToolACE regression sets, because ToolACE repeats APIs across
conversations. What this means:
* **194 of the 416 holdout golds appear by name in v1's training data.** The holdout is therefore *not* a pure
  unseen-tool test for any v1 model. The R2 report's wording ("clean ToolACE unseen holdout", "benchmark distractors
  are protected on the v1 training side") was true only for the R2 additions.
* The exposure did not help. v1-tool scores 83.5% on holdout items with a seen gold and 90.5% with an unseen gold;
  v0.3 scores 88.7 / 94.6.
* Stage 2 handled this correctly:
  * 61,128 decisions with a protected gold or query were dropped;
  * 516,474 protected negatives were removed from 60,932 decisions;
  * no protected tool reaches the Stage 2 data.

## 2. Checkpoint selection (registered rule; the holdout is never used)

| checkpoint | eligible (preservation on V fit) | ToolACE val_tools | val_tools small-k |
|---|---|---:|---:|
| 25% (1100) | yes | 91.7 | 93.0 |
| 50% (2200) | yes | 94.2 | 95.6 |
| **75% (3299)** | yes | **94.8** | 95.3 |
| final (4399) | yes | 93.5 | 94.6 |

**Selected: step 3299 (75%).** For transparency: on the holdout, the 50% checkpoint (90.6) is slightly above the
selected one (90.1). Selecting on the holdout would not change any gate outcome (90.6 < 90.8 < 91.8).

## 3. ToolACE breakdown (holdout 416 answerable; gate harness bf16)

| model | val_tools | **holdout full** | clean | legacy / xLAM-negative | k=2 | k=4 | k=8 | k≥16 | **small-k** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v0.3 production | 91.2 | **91.8** | 96.3 | 89.7 | 97.3 | 90.0 | 98.6 | 87.1 | **94.2** |
| v1 (candidate 1) | 92.6 | 84.1 | 94.8 | 79.0 | 84.0 | 86.9 | 87.3 | 80.0 | 86.2 |
| v1-tool (R2) | 92.1 | 87.3 | 94.8 | 83.6 | 80.0 | 89.2 | 88.7 | 88.6 | 86.6 |
| S2 25% | 91.7 | 87.3 | 97.0 | 82.6 | 93.3 | 88.5 | 88.7 | 82.1 | 89.9 |
| S2 50% | 94.2 | 90.6 | 96.3 | 87.9 | 94.7 | 91.5 | 90.1 | 87.9 | 92.0 |
| **S2 75% (selected)** | **94.8** | **90.1** | **97.0** | 86.8 | **96.0** | 89.2 | 91.5 | 87.1 | **91.7** |
| S2 final | 93.5 | 89.7 | 96.3 | 86.5 | 94.7 | 90.8 | 91.5 | 85.0 | 92.0 |

Subset sizes: clean n = 135, legacy n = 281; k=2 n = 75, k=4 n = 130, k=8 n = 71, k≥16 n = 140.

Native vs random negatives, selected checkpoint:

| strategy | S2 selected | v0.3 |
|---|---:|---:|
| random (n = 39) | 100.0 | 97.4 |
| native + random (n = 96) | 95.8 | 95.8 |
| hard_v1, legacy (n = 105) | 86.7 | 88.6 |
| native + hard_v1, legacy (n = 176) | 86.9 | 90.3 |

Per-strategy data is in `stage2_comparison.json → models.*.toolace`.

Paired group bootstrap on the holdout, selected checkpoint:
* **vs v0.3: −1.7 pp, 95% CI [−5.0, +1.9]**. No longer significantly worse (v1-tool was −4.6, CI [−8.6, −0.5]).
* vs v1-tool: +2.9 pp, CI [0.0, +5.8].

What remains is concentrated in the **legacy (v0-pool, mostly xLAM) distractors** (86.8 vs 89.7). Stage 2 now
**beats** v0.3 on the clean subset (97.0 vs 96.3) and matches it on k ≥ 16 (87.1 vs 87.1). k≥16 dipped at the end of
training (final 85.0), which is why the 75% checkpoint was the better one.

**fp32-scorer diagnostic (not a gate).**
* v1-tool: running the scorer in fp32 lifts its holdout from 87.3 to 88.2. Its floor ties were partly a bf16
  resolution artifact.
* Stage 2 models: fp32 changes no argmax. Stage 2 fixed the cause (answerable small-k items pinned at the rejection
  floor), not just the symptom.

## 4. Gates (registered before training; selected checkpoint, V eval)

| gate | required | measured | pass |
|---|---|---|---|
| G1 ToolACE holdout full | ≥ 91.8 | 90.1 | ❌ |
| G1-alternative | holdout ≥ 90.8 AND clean > 96.3 AND k≥16 > 87.1 AND unseen FA@70 ≤ 5% AND tool AURC < v0.3 AND multilingual ok | 90.1 / 97.0 / 87.1 / 3.8% / 0.031 < 0.156 / ok | ❌ (holdout 90.1 < 90.8; k≥16 = 87.1, not >) |
| G2 small-k | ≥ 92.2 | 91.7 | ❌ |
| native non-EN (paired vs v1-tool) | drop ≤ 1.0 pp | −0.03 pp | ✅ |
| PT-BR | drop ≤ 1.0 pp | 98.4 → 98.0 (−0.43) | ✅ |
| cross-language | drop ≤ 1.0 pp | 98.27 → 98.16 (−0.11) | ✅ |
| intent accuracy | drop ≤ 1.0 pp | 97.68 → 97.53 (−0.15) | ✅ |
| permutation consistency | drop ≤ 1.0 pp | 97.6 → **99.3** | ✅ |
| ECE | ≤ v1-tool + 0.01 and ≤ 0.05 | 0.032 → **0.015** | ✅ |
| OOD AUROC (margin) | drop ≤ 0.01 | 0.974 → 0.971 | ✅ |
| AURC | rise ≤ 0.005 | 0.0274 → 0.0288 | ✅ |
| tool unseen-tool FA @70% | ≤ 5% | 3.8% (v1-tool 1.7) | ✅ |
| tool OOD FA @70% | ≤ 5% | 2.7% (v1-tool 1.2) | ✅ |
| score MAE | rise ≤ 0.03 | 0.607 → 0.600 | ✅ |
| **score Spearman** | drop ≤ 0.01 | 0.854 → 0.836 (**−0.019**) | ❌ |
| noul AUROC | drop ≤ 0.01 | 0.962 → 0.973 | ✅ |
| multi micro-F1 | drop ≤ 1.0 pp | 90.6 → 91.7 | ✅ |
| latency typical p50 | ≤ 20 ms | 16.9 ms | ✅ |
| latency 128 tools p50 | ≤ 50 ms | 43.9 ms | ✅ |

`FREEZE_TEXTUAL_BASE = (G1 or G1-alt) and G2 and all preservation` → **false**.

## 5. Other metrics (V eval)

| metric | v0.3 | v1-tool | S2 selected |
|---|---:|---:|---:|
| intent accuracy | 89.1 | 97.7 | 97.5 |
| tool accuracy (V) | 81.2 | 98.4 | 98.1 |
| hard-negative / near-duplicate accuracy | 87.7 / 86.5 | 97.8 / 96.2 | 97.5 / 95.7 |
| k = 2 / 4 / 8 / 16 / 32 / 64 / 128 (V) | 96.8 / 93.0 / 90.5 / 86.1 / 82.4 / 73.6 / 64.9 | 98.6 / 98.5 / 98.3 / 97.6 / 98.6 / 96.2 / 91.4 | 98.7 / 98.6 / 97.8 / 97.3 / 98.7 / 95.8 / 90.6 |
| NLL (choice) | 0.501 | 0.157 | **0.103** |
| noul accuracy | 89.0 | 85.1 | **90.4** |
| val_tools_v2 tool accuracy / ECE | — | 90.2 / 0.073 | 90.7 / **0.024** |

Tool coverage curves (thresholds from fit, reported on eval):

| coverage | v0.3 sel. acc / OOD FA / unseen FA | v1-tool | S2 selected |
|---:|---|---|---|
| 50% | 87.3 / 22.6 / 25.4 | 100.0 / 0.0 / 0.0 | 99.7 / 0.6 / 0.8 |
| 60% | 84.9 / 27.7 / 30.5 | 100.0 / 0.0 / 0.0 | 99.7 / 0.9 / 1.3 |
| 70% | 82.2 / 35.8 / 40.3 | 99.6 / 1.2 / 1.7 | 98.9 / 2.7 / 3.8 |
| 80% | 78.8 / 44.9 / 50.0 | 98.6 / 4.8 / 6.8 | 97.3 / 9.3 / 11.9 |
| 90% | 74.2 / 61.7 / 65.7 | 97.0 / 11.4 / 15.3 | 96.0 / 16.0 / 19.1 |

Tool AURC: v0.3 0.156 / v1-tool 0.028 / S2 0.031.

Intent coverage curves: at 70%, OOD FA is 7.6 / 1.5 / 1.9 (v0.3 / v1-tool / S2); at 80%, 14.1 / 2.4 / 3.4.

Serving points (fit rule): S2 intent 80%, tool 80% (v1-tool: 80% / 90%).

Stage 2 trades a little tool OOD rejection (FA @80%: 4.8 → 9.3) for recall on weakly matching answerable tools. That
is the expected direction. It is still far better than v0.3 at every coverage.

## 6. Focused diagnosis (why not frozen)

1. **The remaining holdout gap is the legacy-distractor slice**:
   * 86.8 vs v0.3 89.7, n = 281;
   * these are small sets whose negatives come from the v0 tool pool, which v0.3 trained on;
   * the clean subset and k ≥ 16 are at or above v0.3.

   The owner question from R2 still stands: should the gate be scored on the clean slices? It was **not** applied
   here, and it cannot be applied retroactively to this result.
2. **Small-k missed by 0.5 pp** (91.7 vs 92.2):
   * k = 2 is solved (96.0);
   * the residue is k = 4 (89.2) and k = 8 (91.5) with legacy hard negatives.
3. **Score Spearman −1.9 pp**:
   * score replay was only 8k decisions (4%);
   * MAE improved, so this is a ranking-only regression;
   * a larger score replay share is the obvious fix.
4. **The late checkpoint degrades k≥16** (87.9 at 50% → 85.0 final), while the final phase emphasized small-k.

   The S2P3 phase is over-weighted toward small sets; a large-k slice in S2P3 would likely hold it.

Candidate next step (owner decision, nothing started): a Stage 2b from this selected checkpoint with
* replay score share ×2;
* a k ≥ 16 slice kept in S2P3;
* more native + hard small-k (k = 4 / 8) from ToolACE/ToolMind.

Its gates would stay identical, and it would be registered before running.

## 7. RunPod

| pod | GPU | $/h | use | cost | status |
|---|---|---:|---|---:|---|
| `s6xu3zsw37zqsc` | RTX 4090 **Secure** (EUR-IS-1, CUDA 13.0, driver 580) | 0.74 | preflight incl. VRAM smoke (18.5 GiB); hashes; 20-step smoke; training 54 min; eval of 4 checkpoints; latency | ~2.6 (1.91 billed through 00:00 UTC, remainder posting) | **TERMINATED**; `list-pods` empty |

GPU choice followed the preference list:
* 4090 Community had no stock at all;
* 5090 Community had only CUDA 13.0 stock and create-pod failed;
* 3090 had no stock;
* 4090 Secure is the listed fallback.

Upload lesson: a single ssh stream to EUR-IS-1 ran at ~0.5 MB/s. 10 parallel streams reached ~7 MB/s (3.5 GB in 8 min).
