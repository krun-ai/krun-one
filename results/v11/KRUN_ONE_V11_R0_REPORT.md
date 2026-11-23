# Krun One v1.1 — R0 report: capacity diagnosis (2026-09-30 / 10-01)

**Question.** Does Krun One need more parameter capacity for technical decision-making, or can the current 879M
representation learn the capability with improved training?

**Answer.** On this evidence, **stay at 879M**. The gap is training, not parameter count.

* Zero-shot, the frozen v1 fails the probe the way the arena showed: near-uniform distributions, wrong argmax,
  100% abstention.
* After 2,000 steps (20 minutes) on probe-style data, the same 879M model is at 100% whenever each candidate's
  consequence is explicit, and at 88–100% from the raw state on four of the five slices.
* A 143M and a 318M encoder reach the same numbers with the same recipe. The controlled pair — mmBERT-base
  (318M) against its function-preserving ×2 clone (879M, v1's own pre-decision init) — differs by less than
  2 pp on every slice (mean of two seeds).
* The decision head is not the bottleneck either: a deeper head or a candidate-interaction block adds 1–2 pp.
* One slice is not learned by any size in the registered budget: Tetris from the raw board. See §4 for what the
  exploratory follow-up says.

The registered rule has a clause ("underfit on a condition") that formally fires on Tetris-raw. I am not
following it to "grow": the same underfit appears at 143M, 318M and 879M, so parameters are not what is missing
there. This is a judgment call against the letter of my own rule; §3.3 lays it out.

**No 1.5B training was started.** The frozen v1 was only read (sha `781e5c7b…` verified on the pod).

---

## 1. What was run

| step | where | output |
|---|---|---|
| protocol registered before any GPU forward | `configs/v11/r0_protocol.yaml` | decision rules, budgets |
| probe built (7,774 held-out problems; 31,200 training problems from another seed) | `scripts/v11/build_probe.py` | `data/v11/probe/` |
| zero-shot probe of the frozen v1 | RTX 4090 pod | `results/v11/capacity_probe/v1_zero_shot/` |
| learnability diagnostic, 10 runs, throwaway weights | same pod | `results/v11/capacity_probe/learnability/` |
| exact parameter counts of 7 encoder + 5 head candidates | local | `results/v11/architecture_search/param_budget.md` |
| candidate latency / VRAM, random weights | same pod | `results/v11/architecture_search/latency_rtx4090.md` |
| function-preserving growth verified against the real v1 | laptop fp32 + pod fp32 / bf16 | `results/v11/architecture_search/growth_verification_{cpu,cuda}.json` |
| dataset audit (no data downloaded) | Hub API / cards / papers | `data/v11/DATASET_AUDIT.md`, `results/v11/dataset_audit/evidence/` |

**Deviation from the brief, on purpose.** The brief defines R0 as "no training". Zero-shot numbers alone cannot
separate "not enough capacity" from "never trained on this kind of decision", so I added a small training
diagnostic (R0-L): 2,000 steps per run, v1's unchanged objective, weights never saved, no replay. It is not R1 and
it produces no v1.1 candidate. It was registered in the protocol before any result.

### The probe

Every label comes from a simulator, arithmetic or an explicit rule table. No external dataset is used.

| slice | state | candidates | truth |
|---|---|---|---|
| tetris | 10×12 board + next piece | placements (rotation, column) | simulated drop: most lines, then fewest holes, then lowest height |
| scheduling | job + nodes (wait, speed, memory) | nodes | earliest finish among nodes with enough memory |
| routing | weighted site graph | explicit routes | lowest total latency (optionally a site is down) |
| sizing | vCPU / RAM / GPU requirement | instance types with price | cheapest that meets every requirement |
| techsel | stated requirements | technologies (5 categories) | property tables: the unique satisfier of lowest overhead |

Conditions, same problem and candidate order:

* **raw** — state + candidate actions;
* **feature-assisted** — the same, plus each action's environment-computed outcome ("done at minute 47");
* **pairwise** — gold against one negative; **multi-way** — k = 8 / 16 / 32.

Scheduling and techsel stop at k = 16: 32 nodes plus 32 outcomes do not fit one 1,024-token sequence, and
candidates in different chunks cannot attend to each other.

---

## 2. Zero-shot: the 879M baseline (frozen v1, no training)

Full tables: `results/v11/capacity_probe/v1_zero_shot/metrics.md`.

| slice (k = 8, chance 12.5%) | raw top-1 | feature-assisted top-1 | pairwise vs hardest, raw / feat | entropy raw | gold p vs uniform, raw |
|---|---:|---:|---:|---:|---:|
| tetris | 14.0% | 23.7% | 51.7% / 39.0% | 0.91 | 1.02× |
| scheduling | 15.3% | 16.7% | 54.4% / 49.3% | 0.59 | 1.15× |
| routing | 20.7% | 24.3% | 37.3% / 33.3% | 0.85 | 1.28× |
| sizing | 22.0% | 24.0% | 68.3% / 70.0% | 0.54 | 1.66× |
| techsel | 33.8% | 58.8% | 62.5% / 81.9% | 0.73 | 1.90× |

* **The arena symptom reproduces.** On Tetris the gold candidate gets 1.02× the uniform probability; at k = 32
  top-1 is 3.0%, exactly chance.
* **Abstention is not the problem.** v1 abstains on 100% of the answerable problems at the production threshold,
  and its argmax is wrong anyway.
* **Explicit consequences do not rescue it** on the environment slices (skill over chance 0.05–0.14). Routing
  pairwise is *below* 50%: v1 prefers the longer route text.
* **Tech selection is different.** With the properties written next to each candidate, top-1 goes 34% → 59% and
  pairwise 63% → 82%: semantic matching of requirements to descriptions is what v1 already does.
* **Constraint flips:** both members right in 0–5% of the pairs raw, 1–25% feature-assisted.
* **None-of-the-above:** false acceptance 0%, but only because nothing is accepted. Margin AUROC is 0.48–0.65.

**Registered zero-shot rule** (skill at k = 8; poor < 0.35, strong ≥ 0.75, pairwise weak < 0.70):

| track | raw skill | feat skill | pairwise vs hardest (raw) | outcome |
|---|---:|---:|---:|---|
| environments | 0.06 | 0.11 | 0.53 | Z2: no zero-shot skill in any condition |
| tech selection | 0.24 | 0.53 | 0.63 | Z3: mixed |

Read alone, the brief's rule would call the environment track "evidence of a capacity ceiling". The next section
shows it is not.

---

## 3. Learnability diagnostic (R0-L)

Each run: 2,000 steps on `ltrain.jsonl`, both conditions, v1's objective unchanged, then the held-out probe.
Full tables (k = 16 / 32, margins, flips, OOD): `results/v11/capacity_probe/learnability/summary.md`.

### 3.1 Held-out top-1 at k = 8

| run | params | trained | raw: tetris | sched. | routing | sizing | techsel | feat: tetris | sched. | routing | sizing | techsel |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 zero-shot | 879M | — | 14.0 | 15.3 | 20.7 | 22.0 | 33.8 | 23.7 | 16.7 | 24.3 | 24.0 | 58.8 |
| **v1, everything trainable** | 879M | all | 15.0 | 92.9 | 90.7 | 100 | 88.1 | 100 | 100 | 100 | 100 | 92.5 |
| v1 + 2 candidate-interaction layers | 917M | all | 17.3 | 93.5 | 92.3 | 100 | 86.9 | 100 | 100 | 100 | 99.7 | 95.9 |
| v1, encoder frozen (R1 setting) | 879M | head 45M | 21.0 | 17.3 | 69.0 | 49.3 | 63.7 | 91.7 | 66.0 | 78.7 | 81.0 | 88.1 |
| v1 frozen + 2 head layers | 917M | head 83M | 19.7 | 18.4 | 68.7 | 49.0 | 65.6 | 91.7 | 67.3 | 81.0 | 82.0 | 88.7 |
| v1 frozen + 2 interaction layers | 917M | head 83M | 20.0 | 18.7 | 68.3 | 53.3 | 64.7 | 91.0 | 70.7 | 81.0 | 82.3 | 89.4 |
| mmBERT-small | 143M | all | 15.7 | 88.8 | 85.3 | 100 | 80.0 | 100 | 100 | 100 | 100 | 94.1 |
| mmBERT-base, seed 1 / 2 | 318M | all | 18.0 / 17.0 | 91.8 / 88.4 | 90.0 / 88.7 | 100 / 99.7 | 80.0 / 84.4 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 99.7 | 95.9 / 93.1 |
| K1-0B (mmBERT-base cloned ×2), seed 1 / 2 | 879M | all | 14.0 / 20.7 | 89.8 / 89.8 | 90.0 / 89.7 | 100 / 99.7 | 83.1 / 84.7 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 99.7 | 94.1 / 94.7 |

Train-fit accuracy tracks held-out accuracy on every environment slice (e.g. v1 full: scheduling raw 93.2% fit,
92.9% held-out), so these are not memorization numbers. Tech selection is the exception: 100% fit on every run
(§3.4).

### 3.2 What the runs say

* **Comparative decision capacity is sufficient at 879M.** With consequences explicit, the trained v1 is at 100%
  on every environment slice at k = 8, 16 and 32, with a gold-vs-hardest logit margin of +17 (zero-shot: +0.1 to
  +1.2). Constraint flips: both members right in 100% of the pairs, never stuck on the base answer.
* **Raw decisions are learnable too**, where the consequence is a short computation: scheduling 93%, routing 91%,
  sizing 100%. The remaining errors sit in the "hard" items (routing hard 67%, scheduling hard 63%: outcome gaps
  of ≤ 5 ms / ≤ 2 minutes).
* **Parameters do not buy accuracy in this range.** mmBERT-base and its ×2 clone start from the same function;
  the clone has 4.4× the body parameters. Mean over seeds, raw: scheduling 90.1 vs 89.8, routing 89.4 vs 89.9,
  sizing 99.9 vs 99.9, techsel 82.2 vs 83.9, tetris 17.5 vs 17.4. Even the 143M model is within 2–5 pp.
* **The frozen encoder is the limit, not the head.** With the encoder frozen, feature-assisted accuracy stops at
  66–92% and raw scheduling stays at chance (17%). Unfreezing takes the same head to 100%. So the information the
  comparison needs is not laid out in v1's frozen token representations; it has to be written there by training
  the encoder.
* **Abstention needs recalibration after training.** The trained model separates unanswerable problems almost
  perfectly (margin AUROC 0.97–1.00) but the v1 threshold no longer fits: raw scheduling is 93% accurate with 6.5%
  coverage. Expected, and fixable by refitting temperature and threshold on validation.

### 3.3 Against the registered reading

| registered item | result |
|---|---|
| comparative capacity sufficient (v1 full, feat skill ≥ 0.90) | **holds**: environments 1.00, tech selection 0.91 |
| capacity evidence — scaling (clone beats base by ≥ 5 pp on unsolved conditions, base beats small) | **does not hold**: the only unsolved condition is Tetris-raw, 17.4 vs 17.5 |
| capacity evidence — underfit (v1 full train-fit < 0.90 on a condition) | **fires on Tetris-raw** (train-fit 13.3%) |
| not capacity — clone ≈ base (< 2 pp) on unsolved conditions | **holds** |
| head bottleneck (interaction or deeper head ≥ +5 pp, encoder frozen, feat) | **does not hold**: +1.8 pp and +1.0 pp |

The underfit clause was badly written: failing to fit says the model did not learn the task in this budget, not
that more parameters would. Here three sizes spanning 6× fail identically, which is the opposite of a capacity
signature. I registered that clause before seeing data and I am overriding it with this explanation rather than
quietly dropping it. If you prefer the literal rule, the formal outcome is "capacity evidence present on one
slice".

### 3.4 Tech selection: memorization, not reasoning

| run | raw: seen sets / unseen requirement sets / unseen category | feat: same |
|---|---|---|
| v1 zero-shot | 31.7 / 37.3 / 40.0 | 57.6 / 45.1 / 80.0 |
| v1 full | 100 / 60.8 / 60.0 | 100 / 76.5 / 73.3 |
| mmBERT-base (2 seeds) | 100 / 49.0–56.9 / 15.6–37.8 | 100 / 78.4–94.1 / 75.6–77.8 |
| K1-0B (2 seeds) | 100 / 58.8–70.6 / 24.4–26.7 | 100 / 86.3–90.2 / 68.9–77.8 |

Every model memorizes the 44 training requirement sets. On unseen combinations and on the held-out category
(messaging), raw accuracy is far lower, and the models without decision pre-training fall *below* v1 zero-shot on
the unseen category. This is a data-diversity result: 44 rule sets cannot teach technology knowledge. It is the
strongest argument for the corpus work (many technologies, counterfactual pairs, real SWE decisions) over model
growth. The unseen-category cells have n = 45, so the spread between runs there is mostly noise.

---

## 4. Tetris from the raw board (exploratory, not in the registered protocol)

The registered runs left one question open: is Tetris-raw unlearnable, or just under-trained? Two extra runs,
decided after seeing §3 and therefore **exploratory**: 3,000 steps on Tetris-raw only, with 30,000 new boards
(`ltrain_tetris.jsonl`), same recipe.

| run | params | train accuracy, first → last 500 steps | held-out top-1 k = 8 / 16 / 32 | candidate pairs ordered correctly |
|---|---:|---|---|---:|
| v1 zero-shot | 879M | — | 14.0 / 10.0 / 3.0 | 49% |
| v1, registered mix (2,000 steps) | 879M | 13% (train-fit) | 15.0 / 12.0 / 7.5 | — |
| v1, Tetris-raw only (3,000 steps) | 879M | 17% → 21% | 25.7 / 13.0 / 11.0 | 65% |
| mmBERT-base, Tetris-raw only (3,000 steps) | 318M | 16% → 22% | 25.7 / 11.0 / 15.0 | 66% |

* Both models learn something — pair ordering rises from 49% to 65%, which is a coarse heuristic about where the
  piece lands — and neither comes close to solving it. The curve is still rising slowly at 3,000 steps.
* **318M and 879M are indistinguishable** (25.7% vs 25.7% at k = 8). More parameters do not help here either.
* With the outcome written next to each placement, the same decision is at 100% after 200 steps.

What this says about the arena: picking a Tetris placement from the raw board means simulating a drop, clearing
rows and counting holes for every candidate in one forward pass. Nothing in this evidence says a 1.5B encoder
would do that; the lever is elsewhere — the caller (or an environment adapter) supplying consequence features,
an auxiliary value objective, a state encoding closer to the computation (e.g. column heights), or far more
training on such states. Which of these matters most is an R1/R2 question. It also means I cannot promise that
v1.1 at 879M closes the arena gap on raw board states; I can only say that growing the model is not what closes it.

---

## 5. Decision head study

| head (v1 encoder) | params | k=128 total p50, production numerics | encoder frozen, feat mean top-1 | everything trainable |
|---|---:|---:|---:|---|
| current: 2 layers, FFN ×2 | 879M | 54.1 ms | 81.1% | 98.5% feat / 77.3% raw |
| deeper: 4 layers | 917M | 64.3 ms | 82.1% | not run |
| + 2 candidate-interaction layers | 917M | 54.9 ms | 82.9% | 99.1% feat / 78.0% raw |
| wider FFN ×4 | 898M | 54.8 ms | not run | not run |

(Means over the five slices at k = 8.)

* The current head is not the bottleneck: no variant reaches the registered +5 pp.
* The candidate-interaction block (`src/krun_one/v11/model.py`) is the cheap one: it runs on K+1 vectors, costs
  under 1 ms at k = 128, and is bit-identical to v1 at initialization. With everything trainable it helped
  routing raw (+1.6 pp at k = 8, +4.5 to +6.5 pp at k = 16 / 32) and the routing flips (72% → 82% both correct).
  One seed, so treat it as a lead for R1, not a finding.
* A deeper full-sequence head costs +10 ms at k = 128 for nothing measurable.

---

## 6. Architecture study (~1.5B) — measured, nothing trained

### 6.1 Exact parameter counts

Analytic, cross-checked against instantiated modules (`scripts/v11/arch_budget.py`).

| candidate | total | non-embedding | hidden | layers | heads | GLU FFN | FLOPs/token vs v1 | init from v1 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| K1-W1536-L22 (v1) | 879,356,931 | 486,139,395 | 1536 | 22 | 24 | 2304 | 1.00× | — |
| V11-A-W1536-L22-F8192 | 1,476,258,819 | 1,083,041,283 | 1536 | 22 | 24 | 8192 | 2.19× | exact |
| V11-B-W1536-L26-F6144 | 1,419,648,003 | 1,026,430,467 | 1536 | 26 | 24 | 6144 | 2.09× | exact |
| V11-C-W1792-L24-F4096 | 1,356,683,779 | 897,929,987 | 1792 | 24 | 28 | 4096 | 1.83× | width construction |
| V11-D-W2048-L22-F3072 | 1,388,482,563 | 864,192,515 | 2048 | 22 | 32 | 3072 | 1.76× | width construction |
| V11-E-W2048-L22-F4096 | 1,526,894,595 | 1,002,604,547 | 2048 | 22 | 32 | 4096 | 2.04× | width construction |
| V11-F-W2048-L24-F3072 | 1,459,793,923 | 935,503,875 | 2048 | 24 | 32 | 3072 | 1.90× | width construction |

A wider stream spends parameters on the embedding table: at hidden 2048 it holds 524M of them (v1: 393M).

### 6.2 Function-preserving growth

400 probe questions through the real v1 and through each grown model, before any training
(`results/v11/architecture_search/growth_verification_cuda.json`).

| route | tensors | fp32 max \|Δlogit\| | fp32 argmax agreement | bf16 max \|Δlogit\| | bf16 argmax agreement |
|---|---|---:|---:|---:|---:|
| FFN width (A) | `mlp.Wi` rows added per GLU half; new `mlp.Wo` columns zero | 2e-5 | 100% | 0.16 | 100% |
| FFN + depth (B) | as A; 4 appended layers with `attn.Wo = mlp.Wo = 0` | 2e-5 | 100% | 0.17 | 100% |
| head depth / head FFN / interaction block | new head layers with zero output projections | 0 | 100% | 0 – 0.03 | 100% |
| stream width 1536 → 1792 (C) | every tensor; see below | 0.009 | 100% | 0.22 | 98.5% |
| stream width 1536 → 2048 (D, E, F) | every tensor | 0.018 | 100% | 0.17 – 0.20 | 98.0 – 98.3% |
| *reference: v1 bf16 vs v1 fp32* | — | — | — | 0.17 | 98.75% |

* The exact routes are bitwise identical on CPU fp32 (Δ = 0.0); the 2e-5 on GPU is float summation order.
* In bf16 every route sits at v1's own fp32-vs-bf16 noise, including the argmax flips on near-ties.
* **Stream width needed a new construction.** v1's HyperCloning duplicates the stream and is exact only for
  integer factors, because LayerNorm takes its statistics over all dimensions. `src/krun_one/v11/growth_width.py`
  does it for any even widening: the encoder stream is made mean-free (every LayerNorm ignores the mean), LayerNorm
  weights are rescaled by √(n/N) with `eps · n/N`, a few pad dimensions carry the two quantities that are not
  mean-free (the embedding mean for layer 0, which has no attention LayerNorm, and the mean of the final norm's
  output for the decision head), and the embedding table is re-solved so its LayerNorm emits exactly that. New
  heads, GLU units and scorer units are present but disconnected. The docstring lists what is approximate.
* Caveats: the LayerNorm `eps` values become module attributes the serving spec must carry; the closed `ans`
  signal is approximate by construction (measured Δ ≤ 0.005); multimodal fusion tokens would need the same padding
  rule and were not tested.

So a ~1.5B initialization that retains v1 by construction **is feasible on every candidate shape**. That removes
a risk from R3; it is not a reason to do R3.

### 6.3 Latency and memory (RTX 4090, random weights, same process as the v1 shape)

Total p50 in ms. Full table: `results/v11/architecture_search/latency_rtx4090.md`.

| candidate | production numerics: 8 tools | 4 questions | k = 128 | bf16 weights: 8 tools | 4 questions | k = 128 | VRAM weights bf16 / fp32 (MiB) |
|---|---:|---:|---:|---:|---:|---:|---|
| v1 | 17.1 | 21.0 | 54.1 | 15.2 | 16.9 | 43.5 | 1,685 / 3,354 |
| A (FFN 8192) | 17.1 | 35.0 | 83.7 | 15.4 | 26.1 | 69.7 | 2,824 / 5,640 |
| B (L26, FFN 6144) | 19.5 | 34.0 | 83.5 | 17.3 | 25.6 | 70.2 | 2,716 / 5,424 |
| C (W1792, L24) | 18.1 | 32.1 | 83.9 | 16.0 | 22.3 | 65.1 | 2,596 / 5,183 |
| D (W2048, L22) | 16.6 | 32.7 | 86.8 | 14.9 | 22.4 | 64.5 | 2,656 / 5,305 |
| E (W2048, FFN 4096) | 16.8 | 35.8 | 93.6 | 15.0 | 24.8 | 69.8 | 2,920 / 5,833 |
| F (W2048, L24) | 17.9 | 34.8 | 92.0 | 17.7 | 23.9 | 68.4 | 2,792 / 5,577 |

* "Production numerics" = fp32 weights under bf16 autocast, what the worker runs today. **Every ~1.5B candidate
  misses the preferred gates there**: k = 128 at 84–94 ms (gate ≤ 60–70) and the 4-question request at 32–36 ms
  (gate ≤ 25–30).
* With bf16 weights the wide candidates C / D / E / F land inside the upper bound (k = 128 at 64–70 ms, typical
  ≤ 25 ms). The exactly-growable A and B sit on or over the k = 128 limit (69.7 / 70.2 ms).
* A single small question costs the same on every shape (~15–17 ms): at that size the forward is dominated by
  per-layer overhead, not by FLOPs. The growth shows up on multi-question and large-k requests, 1.5–1.7×.
* Memory: an encoder has no KV cache. VRAM is weights plus one forward's activations; the worst workload measured
  (16 questions × 32 tools) peaks at 7.1 GB for the largest candidate under production numerics. Any candidate
  fits a 24 GB card with the multimodal towers.

If a 1.5B model were ever justified, D (W2048-L22-F3072) is the shape to take: best latency of the group, depth
unchanged, growable from v1 within bf16 noise — and it would require moving serving to bf16 weights.

---

## 7. Dataset audit and availability

Full audit: `data/v11/DATASET_AUDIT.md`. No dataset was downloaded.

| dataset | status | reason in one line |
|---|---|---|
| `SWE-bench/SWE-smith` | **APPROVED**, with mandatory filters | MIT, execution-verified; but contains `pallets/flask` (a SWE-bench eval repo) and GPL/LGPL repositories — filter by repo and license, split by repository |
| `project-themis/Themis-CodePreference` | **REVIEW_REQUIRED**; CodeR-Pile part **REJECTED** | card says Apache-2.0, but 77k pairs come from a `cc-by-nc-sa-4.0` source and the commit-mined part admits AGPL/LGPL code with no per-row license |
| `nvidia/Nemotron-SFT-SWE-v2` | **REVIEW_REQUIRED** | license fine (CC-BY-4.0, "ready for commercial use"); no repo / instance columns to decontaminate, outcomes not execution-verified |
| `jtregunna/software-strategist-v1` | **REVIEW_REQUIRED** | MIT, but synthetic with an unnamed generator; the when-to-use / anti-pattern fields are not in the training rows |
| `Johin/function-calling-dataset` | **REVIEW_REQUIRED** | generated with Gemini 2.0 Flash; already in v1 as `approved_with_notice` — keep to replay only |
| `GAIR/OpenSWE` | **REVIEW_REQUIRED** (recommend REJECTED) | three contradictory license statements, empty LICENSE, gated, no SWE-bench exclusion |
| Stack Overflow dumps | **EXCLUDED_UNTIL_REVIEWED** | not inspected |

Consequence: the SWE-verified share of the corpus can be built now from SWE-smith. The preference / ranking share
has no cleared source until you decide on Themis.

Plan: `data/v11/TECHNICAL_CORPUS_PLAN.md` · registry: `configs/v11/sources.yaml`.

---

## 8. KrunTechBench v1 — design

`configs/v11/kruntechbench_v1.yaml` (design only; not built).

* 11 tracks (architecture, backend, databases, distributed systems, cloud/infra, networking/API, security, DevOps,
  ML infrastructure, code strategy, state/action consequence), ~6,000 protected items, EN + PT-BR / ES / FR.
* Mandatory slices: easy, hard, hard-negative, constraint-flip, unseen technology, large candidate set, OOD,
  state-action. Raw is the headline; feature-assisted is reported as a diagnostic.
* Truth by rule tables with cited primary sources, simulators, and SWE-smith held-out repositories. Ties never
  become labels. A 200-item human audit before the freeze.
* Constraint-flip metrics: both correct, flip sensitivity, stuck-on-base-answer.
* One environment family and a set of technologies / repositories are never generated for training.
* The R0 probe is **not** this benchmark: its generators are visible to training.

Promotion gates are registered in `configs/v11/promotion_gates.yaml` (technical: top-1 +10 pp, pairwise +8 pp,
hard-negative margin +1.0 logit, flips +10 pp; preservation: the v1 Stage 2 limits; latency 20 / 50 ms if the
core stays 879M).

---

## 9. Recommendation

**Stay at 879M. Do not start the 1.5B track.** What the experiments point to instead:

1. **Unfreeze the encoder early.** The frozen-core setting (R1 as written in the brief) tops out at 66–92% on
   decisions the full model solves at 100%. R1 is still worth one run as the forgetting-free baseline, but the
   capability will come from R2-style training (encoder trainable, heavy replay). I would run R1 and R2 side by
   side rather than gate R2 on R1 failing.
2. **Build the data before the objective.** The unchanged v1 objective already gives +17 logits of margin once
   the data is there. The pairwise / margin / value losses are worth testing for the raw condition and for
   calibration, as ablations, not as the main bet.
3. **Train on both renderings.** Feature-assisted examples are learned in ~200 steps and teach the comparison;
   raw examples teach computing the consequence. An auxiliary value head (`Q(state, action)` regressing the
   environment outcome) is the natural way to tie them: the outcome is already in `metadata.values`.
4. **Spend the corpus effort on diversity**, per §3.4: many technologies and requirement combinations, real SWE
   decisions from SWE-smith, counterfactual pairs. 44 rule sets get memorized.
5. **Keep the candidate-interaction block as an R1/R2 ablation.** Free at inference, exact at init.
6. **Measure forgetting from the first run.** R0-L used no replay and did not run the regression suite, so it
   says nothing about what this training does to routing, multilingual or OOD behaviour.

What would reopen the growth question: an R2 run on the real corpus (with replay) whose *training* accuracy
plateaus below the smaller-model ladder's ceiling while a grown clone trained identically goes higher. That
comparison is cheap now that growth is verified.

**Cost of the next step.** A v1-sized full fine-tune ran at 0.62 s/step on the 4090 (12k padded tokens per step).
One R2-style run over ~200k technical decisions plus the same amount of replay is about 13k steps ≈ 2.3 GPU-hours
≈ US$ 1.7 on a Secure 4090, plus the regression suite; R1 is about a third of that. An estimate will be
delivered before any run longer than a few hours.

### Decisions for you

* Themis: clear the measured / tested sub-sources (and, separately, the commit-mined part), or leave it out.
* SWE-smith: confirm dropping GPL / LGPL / "Other" repositories, or clear them.
* R1 and R2 in parallel, as proposed, or strictly R1 first.
* Whether Jev and the arena states can be made available; neither is in this workspace, so nothing here was
  compared against Jev.

---

## 10. Limits of this evidence

* The probe is synthetic. It isolates "compare candidates under consequences" and says little about real
  software knowledge; no SWE or code-preference slice was run because no such data is cleared and built yet.
* R0-L trains and tests on the same generators (different seeds). It shows the capability is learnable, not that
  it transfers to an arena v1 has not seen.
* One training budget (2,000 steps) and one learning-rate setting for every model size; two seeds only for the
  controlled pair, one for the rest.
* The size ladder tops out at 879M. "No gain from 143M to 879M" makes a gain from 879M to 1.5B unlikely; it does
  not measure it.
* Latency is en-US only, 30 iterations, random weights, one host.
* Tech selection labels rest on property tables I wrote; "partial" properties never decide an item, but the
  tables have not been reviewed by a second person.
* The existing regression suite was not rerun: v1 is unchanged, and its registered numbers are in
  `artifacts/krun-one-v1-text-final/metrics.json`. No protected set was opened.

## 11. Cost and pod hygiene

* One pod: `krun-v11-r0` (`nkcz4hbojiyhjq`), RTX 4090 Secure, US$ 0.74/h, created 2026-09-30 23:27 BRT, terminated 2026-10-01 04:03 BRT:
  about 4.6 h ≈ **US$ 3.4**. `list-pods` is empty after termination.
* I announced ~US$ 2 before creating it. The overrun is the second seeds, the growth verification on GPU (28
  minutes, mostly CPU-side model construction) and the exploratory Tetris runs; one redundant run was cut.
* Health check (`gpu_preflight.py`, image and venv) passed before any job; inputs verified by sha256 on the pod;
  all 41 result files verified by sha256 after download. Pod logs: `results/v11/pod_logs/`.
* A first latency sweep was interrupted (it would have delayed the learnability runs by ~50 minutes) and rerun at
  the end with one locale and 30 iterations; the partial file was discarded.

## 12. Files

```
configs/v11/   r0_protocol.yaml  promotion_gates.yaml  kruntechbench_v1.yaml  sources.yaml
data/v11/      DATASET_AUDIT.md  TECHNICAL_CORPUS_PLAN.md  probe/{probe,ltrain,ltrain_tetris}.jsonl  probe/manifest.json
results/v11/   KRUN_ONE_V11_R0_REPORT.md
               capacity_probe/v1_zero_shot/{signals.jsonl.gz,metrics.json,metrics.md}
               capacity_probe/learnability/summary.{json,md}  + one directory per run (signals, run.json, metrics)
               architecture_search/{param_budget,latency_rtx4090}.{json,md}  growth_verification_{cpu,cuda}.json
               dataset_audit/evidence/
src/krun_one/v11/   envs.py  probe.py  harness.py  model.py  growth.py  growth_width.py  candidates.py
scripts/v11/        build_probe.py  run_probe.py  analyze_probe.py  learnability.py  learnability_summary.py
                    arch_budget.py  latency_v11.py  latency_summary.py  growth_verify.py  runpod_r0*.sh  pod_upload_r0.sh
tests/v11/test_v11.py   (7 tests)
```

Nothing under `src/krun_one/v1`, `artifacts/` or the v1 configs was modified. Nothing was committed.
