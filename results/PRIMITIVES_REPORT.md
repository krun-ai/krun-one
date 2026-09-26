# KRUN-011 — Decision primitives (choice · noul · score): model phase report

**Status: MODEL_PRIMITIVES_READY = true.** Selected candidate: **`krun-one-v0.3-candidate`**, round 2 variant B
(`configs/krun_one_v0_3.yaml`). It passed every gate registered before its training and the one-time final gate on
the test sets. The artifact is `artifacts/krun-one-v0.3-candidate/` (sha256 `7432c2c0…`); it is **not promoted to
production**.

* **Round 1 (v0.2, sections 4–5):** none of the 3 variants was eligible (small choice regressions, and an
  unseen-proposition regression). No round-1 test set was opened.
* **Round 2 (v0.3, section 10):** the fixes were registered before training. Variant B is eligible and was selected;
  the test sets were opened once for it (`results/primitives_decisions.json`).

Numbers come from `results/primitives_*/eval_*.json`, `results/eval_*.json` and `results/primitives_final_table.md`.

## 1. Architecture audit (`primitives_architecture_audit.md`)

No architecture change is needed. The same encoder, head and option-marker scorer serve all three primitives. The
primitive enters the model in two ways:

* the header text `"{type} question: …"`;
* the `type_emb` row that is added to every token before the decision head.

Noul is a 2-way softmax over `[false, true]` (so 2-way CE = BCE and temperature = Platt scaling without bias). Score
is a K-way softmax over `level i: …` markers that are never permuted. Upstream score confidence is `1 − H/ln K`. One
sequence per question, all questions of a request in one forward pass.

Two Krun One v0 details: v0 hard-coded `qtype = choice`, and only `type_emb[0]` received gradient during its
training. Krun's serialization is now byte-identical to upstream for noul and score, checked by
`tests/test_primitives.py`.

## 2. Datasets (`noul_dataset_audit.md`, `score_dataset_audit.md`)

All training data is real or a deterministic transformation of real data. No synthetic or LLM-labelled examples.

| primitive | train sources (license) | pool | eval-only (external) |
|---|---|---:|---|
| noul | intent propositions from BANKING77 / MASSIVE (CC-BY-4.0, hand-written propositions + overlap list); tool propositions from xLAM / ToolACE / xlam-irrelevance (CC-BY-4.0 / Apache-2.0); Civil Comments properties (CC0); MHS hate-speech (CC-BY-4.0) | 85,000 (41% pos · 40% hard neg · 13% cross-domain · 5% plain neg) | BoolQ (CC-BY-SA, eval only) |
| score | HelpSteer2 5 attributes (CC-BY-4.0); Measuring Hate Speech 9 Likert items (CC-BY-4.0) | 60,000 | SST-5 (license unknown, eval only) |

Rejected sources:

* SNLI and MultiNLI: share-alike or mixed licensing.
* Amazon, Yelp and IMDB: unclear terms of use.
* customer-support-tickets: non-commercial license.
* Prometheus and UltraFeedback: synthetic labels.

Val / test: noul 9,000 / 12,270; score 6,000 / 8,210; mixed requests 1,000 (MHS test comments, choice + noul +
2 scores on one context).

## 3. Baselines — does Laya still do noul/score, did v0 forget? (test sets, in-domain, calibrated)

| primitive | metric | Laya 1024/896 | Laya 512/192 | Krun One v0 |
|---|---|---:|---:|---:|
| noul | ROC-AUC | 0.856 | 0.856 | **0.911** |
| noul | Brier | 0.149 | 0.149 | **0.119** |
| noul | ECE | **0.022** | **0.022** | 0.048 |
| noul | BoolQ ROC-AUC (external) | 0.813 | 0.813 | 0.814 |
| score | MAE (expected score) | 0.941 | 0.948 | **0.888** |
| score | Spearman | 0.459 | 0.450 | **0.546** |
| score | ECE | **0.029** | **0.027** | 0.041 |
| score | SST-5 Spearman (external) | 0.760 | 0.760 | **0.814** |

**No catastrophic forgetting.** v0's choice-only fine-tuning *improved* noul discrimination and score ranking over
Laya. It did cost calibration: noul ECE went 0.022 → 0.048 and score ECE 0.029 → 0.041. Laya is usable but weak on
score (Spearman 0.46).

## 4. Training (continued SFT from Krun One v0; no RLCD)

| | M50_CE | M40_CE | M50_RPS |
|---|---|---|---|
| mixture choice / noul / score | 85k / 51k / 34k | 68k / 51k / 51k | 85k / 51k / 34k |
| score loss | CE | CE | CE + RPS (λ = 1) |
| steps / tokens | 4,277 / 26.7M | 4,561 / 29.3M | 4,277 / 26.7M |
| train time (RTX 4090) | 1,213 s | 1,345 s | 1,222 s |
| peak VRAM | 16.9 GB | 16.9 GB | 16.9 GB |

Other settings, shared by all three:

* optimizer: same as v0 (AdamW, lr 2e-5 for the encoder / 1e-4 for the head, bf16, 1 epoch);
* batches of at most 8,192 tokens;
* initialized from `krun-one-v0`, sha256 `21e821ee…`.

**Checkpoint/resume** reuses the KRUN-005 mechanism, run in deterministic mode (M50_RPS, 60 steps). Killing the
process with `kill -9` after checkpoint-40 and resuming gives **206/206 tensors bit-identical** to the continuous run
(max |Δ| = 0). Optimizer, scheduler, scaler, RNG, trainer state and losses are all identical.

An unplanned real-world test followed: the volume quota was hit during M40_CE. The atomic write left an incomplete
`checkpoint-200.tmp`, which was ignored, and auto-resume continued from `checkpoint-100`.

## 5. Candidates on validation (gates registered before training — `primitives_candidates/selection_v02.md`)

| gate (val) | v0 / Laya ref | M50_CE | M40_CE | M50_RPS |
|---|---:|---:|---:|---:|
| intent acc kb_val (drop ≤ 0.5 pp) | 96.01% | ✅ 95.68% | ✅ 95.74% | ✅ 95.98% |
| **val_hard acc (drop ≤ 0.5 pp)** | 91.29% | ❌ **90.50%** (−0.79) | ✅ 90.99% (−0.30) | ❌ **90.47%** (−0.82) |
| val_tools acc | 94.86% | ✅ 95.31% | ✅ 94.68% | ✅ 94.68% |
| OOD AUROC kb_val / val_tools | 0.954 / 0.990 | ✅ 0.959 / 0.990 | ✅ 0.959 / 0.988 | ✅ 0.959 / 0.989 |
| **choice ECE kb_val (rise ≤ 0.005)** | 0.0087 | ✅ 0.0128 | ❌ **0.0138** (+0.0051) | ❌ **0.0144** |
| choice ECE val_tools | 0.0170 | ✅ 0.0197 | ✅ 0.0129 | ❌ **0.0277** |
| noul ROC-AUC (Laya 0.850 / v0 0.904) | | ✅ **0.974** | ✅ 0.972 | ✅ 0.972 |
| noul Brier (0.152 / 0.122) | | ✅ **0.058** | ✅ 0.062 | ✅ 0.062 |
| noul ECE (0.027 / 0.049) | | ✅ **0.016** | ✅ 0.021 | ✅ 0.019 |
| score MAE (0.891 / 0.847) | | ✅ 0.589 | ✅ **0.579** | ✅ 0.587 |
| score Spearman (0.535 / 0.602) | | ✅ 0.803 | ✅ **0.810** | ✅ 0.804 |
| **score ECE (0.031 / 0.0196)** | | ✅ 0.0195 | ❌ **0.0273** vs v0 | ✅ **0.0167** |
| score NLL (1.332 / 1.301) | | ✅ 0.971 | ✅ **0.956** | ✅ 0.971 |
| **eligible** | | **no** | **no** | **no** |

In short:

* **New primitives:** every candidate crushes Laya and v0. Noul AUROC is 0.85 → 0.97, noul Brier is ÷2.6, score MAE
  is 0.89 → 0.58 and score Spearman is 0.54 → 0.81, with better calibration.
* **Choice:** it degrades slightly and never beyond ~0.8 pp. The failures are all on the strict pre-registered
  tolerances:
  * M50_CE and M50_RPS each lose ~0.8 pp on the hard-negative intent suite (spread over k and languages; pt-PT
    −1.6 pp).
  * M40_CE passes all accuracy gates but misses the kb_val choice-ECE tolerance by 0.0001, and its score ECE is
    worse than v0's (0.027 vs 0.020, though better than Laya's 0.031).

### Further findings

| finding | numbers (val) |
|---|---|
| **Unseen-proposition regression** (Civil `identity_attack`, never trained) | noul ROC-AUC Laya 0.815 · v0 0.815 · **M50_CE 0.751 · M40_CE 0.710 · M50_RPS 0.744**. Trained propositions reach 0.95–0.99 but **generalization to a new proposition got worse**. This matters for a runtime-schema product. |
| Positive vs hard negatives | noul AUROC 0.834 (Laya) → 0.899 (v0) → **0.973** (M50_CE) |
| HelpSteer2 helpfulness / correctness | Spearman Laya 0.605 / 0.572 → M50_CE 0.524 / 0.539. **Below Laya** on these two attributes; possible reason: HelpSteer-like data in Laya's upstream mix. MAE is still better. |
| MHS items | Spearman 0.26 (Laya) → 0.82; sentiment MAE 1.74 → 0.52 |
| CE vs CE + RPS (M50) | MAE 0.589 vs 0.587 · Spearman 0.803 vs 0.804 · RPS 0.0895 vs 0.0895 · score ECE 0.0195 vs **0.0167**. The ordinal term brings **no ranking/MAE gain**, only a small calibration gain, and it came with worse choice ECE on tools. |
| Permutation consistency (choice) | kb_val 98.9% (v0) → 99.5% all; val_tools 91.7% → 95.8% / 97.9% / 93.8% |
| Score confidence | normalized entropy vs ordinal dispersion vs top-1 within ~0.02 on trained models (`primitives_baseline/score_confidence.md`); not frozen |

## 6. Mixed requests and batching (baselines, MHS mixed test, 1,000 requests × 4 questions, bf16 RTX 4090)

| | Laya | Krun One v0 |
|---|---:|---:|
| choice acc (target group) / noul AUROC (hate speech) / score Spearman (insult, sentiment pooled) | 0.720 / 0.865 / 0.261 | 0.760 / 0.847 / 0.388 |
| alone vs mixed: max \|Δp\| choice / noul / score | 0.059 / 0.030 / 0.032 | 0.044 / 0.039 / 0.044 |
| alone vs mixed: argmax flips choice / noul / score | 0.6% / 0.1% / 0.65% | 0.3% / 0.2% / 0.95% |
| one forward per request vs one per question | 10.9 s vs 40.9 s (3.8× faster) | 10.7 s vs 39.8 s |

Question answers are independent by construction: there is no cross-question attention. The residual Δp comes from
bf16 padding numerics (different batch shapes); in fp32 it is < 1e-4 (`tests/test_primitives.py`). The worker
should keep one forward per request. Candidates were not run on the mixed set because it is part of the locked
test protocol.

## 7. Cost

| | |
|---|---|
| pod | new pod `krun-011-train` (`18ad7go5lcyd5l`): RTX 4090, Secure Cloud, EU-RO-1, $0.74/h, volume `krun-research` |
| why a new pod | the host of `krun-one-train` had no free GPU |
| uptime | 8,520 s = 2.37 GPU-h ≈ **$1.75**, covering tests, 3 baselines, resume test, 3 trainings and 6 evaluations |
| status | pod **stopped** (EXITED), volume kept; runs `v02_*` keep `final/` only (1.6 GB each) |

Volume note: the 100 GB quota was hit during this sprint. Cleaned: this sprint's resume-test runs and intermediate
checkpoints. KRUN-005's `runs/A` and `runs/B` were left untouched (6.6 + 11 GB).

## 8. Definition of Done after round 1 (historical)

| item | status |
|---|---|
| Laya baseline choice/noul/score | ✅ (choice from KRUN-005; noul/score at both budgets) |
| Krun One v0 baseline | ✅ |
| Noul / score dataset audits; licensed datasets | ✅ |
| training mixture; candidates trained | ✅ 3 variants |
| Noul calibrated / score calibrated | ✅ on val (ECE 0.016–0.021 / 0.017–0.027) |
| Krun > Laya on noul / score | ✅ on val, by a large margin, except for the unseen-proposition slice and HelpSteer2 helpfulness/correctness ranking |
| **Choice regression within gate** | ❌ **no variant passes every choice gate** |
| mixed primitive tests | ✅ unit tests + baseline invariance; candidate mixed eval not run (test locked) |
| artifact exported / reload | ⏸ not done: nothing to promote (`export_primitives.py` and `reload_primitives_test.py` are written and ready) |

## 9. Round-1 recommendation (historical — option 1 was chosen → section 10)

The failures are small and all on the choice side, but the rule was registered before training and is kept. Options:

1. **(Recommended) One more registered round (v0.3), ~1 GPU-h (≈ $1).** Register it before training:
   * keep M40's primitive share, but raise choice replay to ≥ 50% by making the mixture bigger instead of shrinking
     choice;
   * over-sample hard_v1 intent decisions in the replay;
   * **add proposition diversity to fix the unseen-proposition regression**, e.g. MHS items as noul propositions and
     more Civil sub-properties, holding out several propositions for evaluation rather than one;
   * keep plain CE (RPS brought no gain);
   * register ordinal dispersion or entropy for score confidence.
2. **Accept M40_CE by explicit decision.** It passes every accuracy gate. Its misses are +0.0001 over the choice-ECE
   tolerance, and score ECE worse than v0 though better than Laya. This is a relaxation of the registered rule, so it
   is the user's call, not the model's; the unseen-proposition regression would ship with it.
3. Stop here. Keep choice-only in production and revisit later.

Nothing public changed. No SDK or package was published, no pricing changed, no production change, no commits.


---

## 10. Round 2 — Krun One v0.3 (the selected candidate)

### 10.1 What changed (registered before training in `configs/krun_one_v0_3.yaml`)

| v0.2 failure | v0.3 change |
|---|---|
| val_hard regressed about 0.8 pp (M50 variants) | choice replay kept at 50% of a **bigger** mixture (220k = 110k choice + 60k noul + 50k score); **all 62,265 hard intent decisions** replayed |
| choice ECE and score ECE regressions | variant B halves the learning rates (encoder 1e-5, head 5e-5) |
| unseen-proposition regression (0.815 → 0.71–0.75) | noul data v2 (`primitives_data_v2.yaml`) adds 9 MHS item propositions and 5 HelpSteer2 quality propositions, both deterministic from real labels (middle ratings dropped); **19 held-out propositions** over 4 sources; new gate: unseen AUROC ≥ best(Laya, v0) |
| RPS brought no ranking gain | plain CE |
| score confidence not frozen | **ordinal dispersion** registered in advance (`primitives_baseline/score_confidence.md`) |

Gates are otherwise **identical** to v0.2; nothing was relaxed. Baselines were re-measured on data v2.

### 10.2 Validation gates (`primitives_candidates/selection_v03.md`)

| gate (val) | reference | A (v0 LR) | **B (½ LR)** |
|---|---:|---:|---:|
| intent kb_val / val_hard / val_tools accuracy | 96.01 / 91.29 / 94.86% | ✅ 95.55 / 91.17 / 94.86 | ✅ 95.95 / 91.45 / 95.22 |
| OOD AUROC kb_val / val_tools | 0.954 / 0.990 | ✅ | ✅ 0.958 / 0.988 |
| choice ECE kb_val / val_hard / val_tools | 0.0087 / 0.0212 / 0.0170 | ✅ | ✅ 0.0136 / 0.0214 / 0.0131 |
| noul in-domain AUROC / Brier / ECE | Laya 0.835 / 0.163 / 0.035 | ✅ 0.972 / 0.063 / 0.008 | ✅ 0.971 / 0.065 / 0.012 |
| score MAE / Spearman / NLL | Laya 0.891 / 0.535 / 1.33 | ✅ | ✅ 0.575 / 0.810 / 0.959 |
| score ECE (≤ v0 0.0196) | | ❌ 0.0218 | ✅ **0.0186** |
| **unseen propositions AUROC (≥ 0.8685)** | v0 0.869, Laya 0.848 | ✅ 0.907 | ✅ **0.893** |
| **eligible** | | no | **yes → selected** |

Training B: 5,677 steps, 35.5M tokens, 1,498 s on the RTX 4090, peak VRAM 17.0 GB, initialized from krun-one-v0.
Checkpoint/resume: same mechanism as the round-1 bit-exact test. The only code changes are the mixture sampling and
the per-variant learning rate, which enter the resume-checked run config.

### 10.3 Final evaluation (test sets, opened once; `results/primitives_final_table.md`)

| Primitive | Dataset/Suite | Metric | Laya (1024/896) | Laya (512/192) | Krun One v0 | v03_B |
|---|---|---|---:|---:|---:|---:|
| Choice | KrunBench test intents (B77+MASSIVE) | accuracy | 66.98% | 66.98% | 96.09% | 96.14% |
| Choice | BFCL | accuracy | 82.83% | 83.39% | 92.33% | 92.25% |
| Choice | holdout_tools | accuracy | 69.53% | 69.74% | 96.42% | 96.50% |
| Choice | KrunBench val intents | accuracy | 64.04% | 64.04% | 96.01% | 95.95% |
| Choice | val_tools | accuracy | 66.61% | 68.05% | 94.86% | 95.22% |
| Choice | KrunBench test | OOD AUROC | 0.6691 | 0.6748 | 0.9472 | 0.9513 |
| Choice | holdout_tools | OOD AUROC | 0.7235 | 0.7811 | 0.9927 | 0.9926 |
| Choice | BFCL irrelevance | OOD AUROC | 0.7007 | 0.7114 | 0.8043 | 0.7948 |
| Choice | KrunBench test | ECE | 0.0327 | 0.0338 | 0.0072 | 0.0082 |
| Noul | test in-domain (intents, tools, Civil, MHS) | roc_auc | 0.8263 | 0.8245 | 0.8655 | 0.9677 |
| Noul | test in-domain (intents, tools, Civil, MHS) | pr_auc | 0.7687 | 0.7664 | 0.8152 | 0.9593 |
| Noul | test in-domain (intents, tools, Civil, MHS) | accuracy@0.5 | 0.7500 | 0.7478 | 0.7875 | 0.9078 |
| Noul | test in-domain (intents, tools, Civil, MHS) | nll | 0.5147 | 0.5168 | 0.4638 | 0.2291 |
| Noul | test in-domain (intents, tools, Civil, MHS) | brier | 0.1689 | 0.1698 | 0.1504 | 0.0688 |
| Noul | test in-domain (intents, tools, Civil, MHS) | ece | 0.0244 | 0.0241 | 0.0490 | 0.0116 |
| Noul | BoolQ (external) | roc_auc | 0.8134 | 0.8129 | 0.8140 | 0.8116 |
| Noul | BoolQ (external) | accuracy@0.5 | 0.7563 | 0.7557 | 0.7639 | 0.7560 |
| Noul | BoolQ (external) | brier | 0.1668 | 0.1670 | 0.1693 | 0.1873 |
| Noul | BoolQ (external) | ece | 0.0411 | 0.0412 | 0.0481 | 0.1319 |
| Noul | positives vs hard negatives | roc_auc | 0.8290 | 0.8274 | 0.8831 | 0.9663 |
| Noul | positives vs hard negatives | accuracy@0.5 | 0.7288 | 0.7266 | 0.7700 | 0.8867 |
| Noul | positives vs hard negatives | brier | 0.1875 | 0.1887 | 0.1712 | 0.0835 |
| Noul | positives vs hard negatives | ece | 0.1624 | 0.1639 | 0.1886 | 0.0866 |
| Noul | positives vs cross-domain negatives | roc_auc | 0.9529 | 0.9526 | 0.9698 | 0.9914 |
| Noul | positives vs cross-domain negatives | accuracy@0.5 | 0.7214 | 0.7185 | 0.7499 | 0.8803 |
| Noul | positives vs cross-domain negatives | brier | 0.1913 | 0.1929 | 0.1893 | 0.0882 |
| Noul | positives vs cross-domain negatives | ece | 0.3264 | 0.3284 | 0.3326 | 0.1570 |
| Noul | unseen propositions (never trained) | roc_auc | 0.8589 | 0.8584 | 0.8768 | 0.9069 |
| Noul | unseen propositions (never trained) | accuracy@0.5 | 0.7296 | 0.7296 | 0.7296 | 0.7750 |
| Noul | unseen propositions (never trained) | brier | 0.1880 | 0.1890 | 0.1912 | 0.1542 |
| Noul | unseen propositions (never trained) | ece | 0.1810 | 0.1827 | 0.2093 | 0.1509 |
| Score | test in-domain (HelpSteer2 + MHS) | mae | 0.9412 | 0.9476 | 0.8879 | 0.6200 |
| Score | test in-domain (HelpSteer2 + MHS) | rmse | 1.1947 | 1.2006 | 1.1268 | 0.8224 |
| Score | test in-domain (HelpSteer2 + MHS) | accuracy | 0.4085 | 0.4035 | 0.4228 | 0.5535 |
| Score | test in-domain (HelpSteer2 + MHS) | spearman | 0.4587 | 0.4496 | 0.5455 | 0.7668 |
| Score | test in-domain (HelpSteer2 + MHS) | kendall_tau_b | 0.3550 | 0.3474 | 0.4197 | 0.6275 |
| Score | test in-domain (HelpSteer2 + MHS) | nll | 1.3889 | 1.3943 | 1.3495 | 1.0265 |
| Score | test in-domain (HelpSteer2 + MHS) | brier | 0.7131 | 0.7154 | 0.6965 | 0.5596 |
| Score | test in-domain (HelpSteer2 + MHS) | ece | 0.0293 | 0.0269 | 0.0409 | 0.0142 |
| Score | test in-domain (HelpSteer2 + MHS) | rps | 0.1571 | 0.1581 | 0.1467 | 0.0971 |
| Score | test in-domain (HelpSteer2 + MHS) | distance_weighted_error | 1.2549 | 1.2597 | 1.1855 | 0.7565 |
| Score | HelpSteer2 | mae | 0.7517 | 0.7644 | 0.7265 | 0.6425 |
| Score | HelpSteer2 | accuracy | 0.5103 | 0.5003 | 0.5007 | 0.5617 |
| Score | HelpSteer2 | spearman | 0.6542 | 0.6363 | 0.6613 | 0.6972 |
| Score | HelpSteer2 | nll | 1.1718 | 1.1829 | 1.1744 | 1.0343 |
| Score | HelpSteer2 | ece | 0.0736 | 0.0659 | 0.0195 | 0.0458 |
| Score | HelpSteer2 | distance_weighted_error | 1.0224 | 1.0317 | 0.9732 | 0.7444 |
| Score | MHS | mae | 1.1308 | 1.1308 | 1.0492 | 0.5974 |
| Score | MHS | accuracy | 0.3067 | 0.3067 | 0.3450 | 0.5453 |
| Score | MHS | spearman | 0.2247 | 0.2247 | 0.3819 | 0.8126 |
| Score | MHS | nll | 1.6060 | 1.6057 | 1.5246 | 1.0187 |
| Score | MHS | ece | 0.1023 | 0.1019 | 0.0888 | 0.0318 |
| Score | MHS | distance_weighted_error | 1.4875 | 1.4876 | 1.3979 | 0.7686 |
| Score | SST-5 (external) | mae | 0.9622 | 0.9621 | 0.7632 | 0.6840 |
| Score | SST-5 (external) | accuracy | 0.3566 | 0.3566 | 0.4226 | 0.4113 |
| Score | SST-5 (external) | spearman | 0.7601 | 0.7602 | 0.8140 | 0.8233 |
| Score | SST-5 (external) | nll | 1.6995 | 1.6986 | 1.3363 | 1.2823 |
| Score | SST-5 (external) | ece | 0.2335 | 0.2332 | 0.1320 | 0.1857 |
| Score | SST-5 (external) | distance_weighted_error | 1.0588 | 1.0590 | 0.9450 | 0.7965 |
| Mixed | MHS mixed requests (choice) | accuracy | 0.7200 | 0.7200 | 0.7600 | 0.7670 |
| Mixed | MHS mixed requests (noul) | roc_auc | 0.8646 | 0.8646 | 0.8469 | 0.9270 |
| Mixed | MHS mixed requests (score) | mae | 1.1755 | 1.1753 | 1.0150 | 0.5995 |
| Mixed | MHS mixed requests (score) | spearman | 0.2611 | 0.2612 | 0.3884 | 0.7733 |
| Mixed | all questions correct | rate | 0.0530 | 0.0530 | 0.0760 | 0.2240 |
| Mixed | invariance alone vs mixed (choice) | max |Δp| | 0.0589 | 0.0589 | 0.0437 | 0.0638 |
| Mixed | invariance alone vs mixed (noul) | max |Δp| | 0.0266 | 0.0266 | 0.0301 | 0.0216 |
| Mixed | invariance alone vs mixed (score) | max |Δp| | 0.0321 | 0.0320 | 0.0439 | 0.0346 |


**Final gate.**

* Choice vs v0: intents +0.05 pp, BFCL −0.08 pp, holdout_tools +0.08 pp. ✅
* Noul in-domain vs Laya: AUROC 0.968 vs 0.826, Brier 0.069 vs 0.169, ECE 0.012 vs 0.024. ✅
* Score in-domain vs Laya: MAE 0.620 vs 0.941, Spearman 0.767 vs 0.459, ECE 0.014 vs 0.029. ✅
* **→ MODEL_PRIMITIVES_READY = true.**

### 10.4 Honest limitations of v0.3

1. **Calibration is not universal.** On data far from training, calibration is worse than Laya's:
   * BoolQ: AUROC is the same (0.812), but ECE is 0.132 vs 0.041;
   * SST-5: score ECE 0.186, vs 0.132 for v0.

   Discrimination holds; probabilities are over-confident on these sets. The docs must not promise universal
   calibration.
2. **Unseen propositions** as a pool: 0.907 vs Laya 0.859. Per source:
   * held-out intents: 0.98–0.99;
   * MHS items: 0.84 vs 0.70;
   * HelpSteer2 coherence: 0.64, roughly equal to Laya;
   * **Civil `identity_attack`: 0.827 vs Laya 0.865 (a regression on that one proposition).**
3. HelpSteer2 score ECE is 0.046, vs 0.020 for v0 (in-domain pooled 0.014). Coherence is skewed (72% at level 4).
4. Choice: BFCL irrelevance AUROC 0.804 → 0.795; KrunBench test ECE 0.0072 → 0.0082. Both are within the gates but
   slightly worse.
5. Only 5-level score scales were trained and validated. 2–16 levels serialize identically but are untested.
6. **Mixed batches (bf16):** the answer to a question alone vs inside a 4-question request differs by mean |Δp| 0.001–0.003
   and max 0.064 (choice). Decisions flip in 0.1% (noul), 0.6% (score) and 0.3% (choice) of cases. This is padding
   numerics, not semantics (fp32: < 1e-4).
7. Score confidence (ordinal dispersion) vs error Spearman on test: 0.408 (entropy 0.380). It is a useful but weak
   error signal.

### 10.5 Artifact, reload, serving, cost

* **Artifact** `artifacts/krun-one-v0.3-candidate/` (1.6 GB):
  * files: `model.safetensors` (sha256 `7432c2c0…` = `runs/v03_B/final`), `config.json`, `tokenizer/`,
    `metrics.json`, `provenance.json`, `README.md`;
  * `krun_config.json` (`config_version` `krun_one.config.v1`) contains `primitives: [choice, noul, score]`,
    temperatures intent 1.405 / tool 1.535 / noul 1.222 / score 1.169, choice thresholds intent 0.819 / tool 0.705
    (top1_top2_margin @ 0.98), and `score_confidence: ordinal_dispersion`.
  * `krun-one-v0` is untouched.
* **Reload** (`reload_primitives_test.py`, new process, `./laya` removed from `sys.path`) passes on GPU bf16 and on
  CPU fp32:
  * choice, noul true/false/criteria, score high/low, mixed request and legacy `decide()` all pass;
  * fidelity vs the evaluated logits: 100% argmax agreement, max |Δlogit| 0.11.
* **Serving** (`bench_primitives_cuda.json`, RTX 4090 bf16, `decide_questions`):

  | request | p50 | throughput |
  |---|---:|---:|
  | 1 question (choice / noul / score) | 9.3–9.6 ms | |
  | mixed 3 | 11.0 ms | 272 q/s |
  | batch 8 | 11.3 ms | 710 q/s |
  | batch 32 | 20.6 ms | 1,552 q/s |

  VRAM is 2.3–2.4 GB, the same as v0. On the MHS mixed set, one forward per request took 10.6 s vs 39.8 s with one
  forward per question.
* **Cost:**

  | pod | GPU-h | cost |
  |---|---:|---:|
  | round 1 (`18ad7go5lcyd5l`) | 2.37 | $1.75 |
  | round 2 (`bt1u0icrkudkpx`, a new pod: the others' hosts had no free GPU) | 1.85 | $1.37 |
  | **total** | | **≈ $3.12** |

  Both pods are stopped. Artifact size and parameters (421,293,827) are unchanged.

### 10.6 Definition of Done — model phase

All done:

* Laya and v0 baselines;
* noul and score dataset audits, all datasets licensed;
* mixture, candidate trained;
* noul and score calibrated in-domain;
* choice regression within the gate;
* Krun beats Laya on noul and on score;
* mixed primitive tests;
* artifact exported, reload passes.

Caveats: section 10.4.

---

## 11. Product phase (implemented and tested locally + staging; NOT deployed to production)

Order followed: model → worker → API → SDKs → docs → console. Nothing is pushed, published or deployed to production;
every repository has uncommitted changes awaiting review.

| component | change | tests |
|---|---|---|
| **worker** (`services/krun-worker`) | decisions carry `type` (default `choice`, so every pre-KRUN-011 payload is unchanged); noul/score validation (instructions required; criteria ≤ 500; 2–16 levels ≤ 500 chars); typed results; one job = N heterogeneous decisions in shared length-bucketed forward passes; `UNSUPPORTED_QUESTION_TYPE` when the artifact lacks a primitive (krun-one-v0 = choice only); vendored `krun_one/{model,primitives}.py` (drift-tested); per-version integrity manifests; Dockerfile `--build-arg KRUN_MODEL_VERSION`; logs = type counts + numeric noul/score only (never instructions/criteria/levels) | 55 passed (CPU; v0 + v0.3 artifacts) |
| **Rust API** (`krun-api-app`) | `Question` = serde-tagged union `ChoiceQuestion \| NoulQuestion \| ScoreQuestion`; `Answer` = `ChoiceAnswer \| NoulAnswer \| ScoreAnswer`; choice wire format unchanged; `KRUN_PRIMITIVES` feature gate (default `choice`: noul/score → 400 before any GPU job); `KRUN_MODEL_ID` + `KRUN_MODEL_ALIASES` (clients pinning `krun-one-v0` keep working after promotion); 1 question = 1 decision for quota/billing, real token counts, no `output_tokens`; typed feedback `expected: {type, value}` with `expected_decision` still accepted; migration **0006** (additive, nullable): `usage_events.noul_decisions/score_decisions`, `feedback.expected_type/expected_value`; OpenAPI unions flattened to `oneOf` + `discriminator` (utoipa's `allOf` + `additionalProperties:false` rejected every valid body) with choice/noul/score/mixed examples | 46 unit + 69 integration (7 new) passed; fmt + clippy clean |
| **Python SDK** (`krun-python` 0.2.0) | `NoulQuestion`, `ScoreQuestion`, `NoulAnswer`, `ScoreAnswer`, typed decoding by `type`, `result.choice/noul/score(id)`, `feedback(expected=...)`, error classes for 402/403/409 codes; OpenAPI snapshot refreshed | 119 passed; ruff + mypy clean |
| **TypeScript SDK** (`krun-typescript` 0.2.0) | discriminated unions; `AnswerFor<Q>` types each inline answer by its question (`result.answers.severity.score`); typed feedback; same error classes | 117 passed; biome + tsc clean |
| **SDK parity** | byte-identical fixture `decision_primitives_parity.json` in both repos; both SDKs must serialize exactly its `wire` body + feedback bodies | ✅ both |
| **Docs** (`krun-docs`) | Concepts → Decision primitives (overview + table, Choice, Noul, Score with formula and confidence), guide *Multiple decision primitives*, updated Q&A / limitations / feedback / SDK pages / API overlay (`ChoiceAnswer` target). No changelog entry and no GA wording until deployed | `mint validate` + `mint broken-links` pass against the new spec |
| **Console** (`krun-console-app`) | per-question type selector; Noul builder (instructions, criteria true/false); Score builder (ordered levels, add/remove, move up/down, warning "Changing level order changes the meaning of the score."); lossless builder ↔ JSON; Noul card (YES/NO %, `noul = …`, bar); Score card (`score / max`, level bars, confidence); typed feedback (choice keeps `expected_decision`); preset "Choice + noul + score" | 55 passed; lint + production build pass |
| **Landing** | not changed (only after GA, per sprint rule) | — |

### End-to-end validation

**Local**, real model: the candidate artifact on the GTX 1650 behind the real worker handler, the real API and
Postgres. On the same mixed request, **raw HTTP == Python SDK == TypeScript SDK == Console** (answers and usage
identical, compared programmatically; the Console run was driven by Playwright in JSON mode). Screenshot:
builder + typed cards.

**Backward compatibility** — all unchanged, all passing against the new API:
- the **published `krun-ai` 0.1.0** from PyPI (choice + tool, `model="krun-one-v0"` via the alias, legacy
  `expected_decision` feedback);
- **TypeScript 0.1.0** built from git HEAD;
- a raw old choice payload.

**Staging**:
- image `ferresdb/krun-worker:krun-one-v0.3-candidate-r1` (digest `sha256:0ffb9872…`) pushed to the private repo;
- **separate** RunPod endpoint `krun-one-v0.3-candidate` (`bd5hk0ei04ufzm`, same GPU pools as production, min 0 /
  max 1, idle 30 s); production endpoint `xftso8bbxl7v1b` untouched;
- cold start 232 s (first image pull), then on an RTX A4500 **0.84–1.5 s** per mixed request through the API;
- same answers as local (noul 0.990, score 1.406); SDKs and the old 0.1.0 client pass.

### Promotion checklist (needs authorization — production / publishing)

1. Re-export the same weights as `krun-one-v0.3`: `export_primitives.py --name krun-one-v0.3`, then `make_manifest.py`,
   then build and push `krun-worker:krun-one-v0.3-r1`.
2. Production endpoint: point `xftso8bbxl7v1b` at the new image, or create a new production endpoint and switch
   `RUNPOD_ENDPOINT_ID`. The v0 image stays available for rollback.
3. Billing: set a billing rate for `krun-one-v0.3` equal to v0's current rate with `krun-api set-billing-rate`. Pricing
   is unchanged; without a rate row, requests would be free.
4. API: push `krun-api-app` to `main` (CI + Fly deploy, migration 0006 runs in the release command) with Fly env
   `KRUN_MODEL_ID=krun-one-v0.3`, `KRUN_MODEL_ALIASES=krun-one-v0`, `KRUN_PRIMITIVES=choice,noul,score`. Rollback =
   `KRUN_PRIMITIVES=choice`.
5. Production smoke: choice (the old 0.1.0 client), noul, score, mixed.
6. Publish `krun-ai` 0.2.0 (PyPI) and `@krun-ai/sdk` 0.2.0 (npm, first release).
7. Docs: push `krun-docs`. The live spec must already have `ChoiceAnswer`, otherwise the overlay target fails. Then add
   the changelog entry.
8. Console deploy. Then the landing section ("Three primitives. One decision model.") if the feature is GA.

### Known limitations (product)

- Calibration outside the evaluated domains is weaker (§10.4); the docs state it explicitly.
- Only 5-level scores are validated.
- In bf16, a question inside a mixed batch can differ from the same question alone by up to 0.06 in probability
  (0.1–1% decision flips).
- The RunPod queue dominates latency (≈0.8–1.5 s warm per request), not the model (≈11 ms for 3 questions).
- TypeScript users who build `Questions` at runtime must narrow `Answer` with `answer.type`. Python users reading
  `result.answers[...]` under mypy need `result.choice(...)`: a typing-only change, documented in both changelogs.
- `usage_events.task_type` is `noul` / `score` / `mixed` for new requests, and the console usage page groups only
  intent/tool. Per-primitive analytics are stored (0006) but not yet shown.

### Regression suite

`scripts/regression_suite.sh <model> <tag>`: unit tests, the choice sets, noul/score val (data v2) and mixed
requests. Then apply the registered gates with `compare_primitives.py select v03`. Every future release must pass it.

### Cost (whole sprint)

- Training/eval pods: $3.12 (4.22 GPU-h).
- Staging: RunPod serverless per-second only; one ≈4-minute cold start plus a few jobs, well under $0.10. The endpoint
  is idle at min 0 workers.
- Pods: both sprint pods terminated. The volume is kept.

## 12. Recommendation

**Model: READY. Product: implemented and validated locally and on staging, but NOT deployed.**

Production promotion needs your authorization for the checklist above: production endpoint, API push/deploy, billing
rate row, SDK publishing, docs push, console deploy.

## 13. Production promotion (2026-09-25, authorized by the user)

| step | status |
|---|---|
| promoted artifact `artifacts/krun-one-v0.3` | ✅ same weights as the candidate (sha256 `7432c2c0…`); only the name/status metadata changed |
| image | ✅ `ferresdb/krun-worker:krun-one-v0.3-r1`, digest `sha256:fe726638…` |
| billing | ✅ rate `krun-one-v0.3` = **$0.50 / M input tokens**, identical to `krun-one-v0` (no pricing change); verified in `billing_ledger` (138 tokens → $0.000069) |
| production endpoint `xftso8bbxl7v1b` (renamed `krun-one`) | ✅ image v0.3; stale v0 workers recycled (max 0 → 2); health job reports `krun-one-v0.3`, primitives `[choice, noul, score]`. Rollback image `krun-one-v0-r3` stays in the registry |
| API | ✅ `krun-api-app` `fda1e42` pushed; CI + Fly deploy (release v10, migration 0006); `KRUN_MODEL_ID=krun-one-v0.3`, `KRUN_MODEL_ALIASES=krun-one-v0`, `KRUN_PRIMITIVES=choice,noul,score` |
| production smoke | ✅ `/v1/models` = `krun-one-v0.3`; mixed request (1.4–1.9 s), noul-only, score-only; typed feedback; **`krun-ai` 0.1.0 (PyPI) unchanged**, pinning `krun-one-v0`, plus legacy feedback; usage rows carry the per-primitive counts |
| docs | ✅ `krun-docs` `e6fbd7c` live on docs.krun.ai (Decision primitives, Noul, Score, guide, changelog entry, models page) |
| SDKs | ⏳ `krun-python` `ca3b5b1` and `krun-typescript` `c4c3e85` pushed and tagged `v0.2.0`. **PyPI** publishing waits for a GitHub Release `v0.2.0` (trusted publishing). **npm** publishing needs interactive web/2FA login |
| console | ⏳ `d1a98b2` pushed to main; not confirmed live (Vercel) |
| landing | ⏳ `f9c0b5a` pushed to main; still the old version on krun.ai after ~10 minutes, and the previous landing commit (`/blog`) also looks not deployed: Vercel's git integration probably isn't connected |
| staging endpoint `bd5hk0ei04ufzm` | kept at min 0 (no idle cost) for future candidates |

**Rollback:**
- API: set `KRUN_PRIMITIVES=choice` (noul/score then answer 400 before any GPU job).
- Model: point `xftso8bbxl7v1b` back to `krun-one-v0-r3`, together with `KRUN_MODEL_ID=krun-one-v0`.
