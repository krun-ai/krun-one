# KRUN-011 — Decision primitives: architecture audit

Read in full: `laya/rl_common.py` (408 lines), `laya/rl_agent_api.py` (78), `laya/rl_agent_config.json` (32), and
the Krun One port `experiments/krun_one/src/krun_one/{model,training,evaluation,calibration}.py`. Line references are
to those files as checked out on 2026-09-24 (Laya repo root @ `1c5edc17…`). Nothing was reconstructed from memory.

## 1. How `qtype` enters the model

```python
QTYPES = {"choice": 0, "score": 1, "noul": 2}            # rl_common.py:17
self.type_emb = nn.Embedding(3, d)                        # DecisionModel.__init__, d = 1024
h = encoder(input_ids, attention_mask).last_hidden_state  # forward
h = h + self.type_emb(qtype)[:, None, :]                  # one vector per sequence, added to EVERY position
h = head(h)                                               # 2 pre-norm transformer layers (16 heads, 4d FFN)
```

The question type enters in **two** places:

1. **Text**: the header of the sequence is `"{type} question: {instructions}"` (`build_sequence`), so the encoder
   itself reads `choice` / `score` / `noul`.
2. **Type embedding**: after the encoder and before the 2-layer decision head, `type_emb[qtype]` is broadcast-added
   to every token state. The encoder never sees it; only the head and scorer do.

`qtype` is per **sequence**, and each question is its own sequence (§4) — so a batch can freely mix types.

**Krun One v0 status.** `krun_one/model.py:collate` hard-codes `qtype = zeros` ("every Krun decision is a choice")
and `build_sequence(..., qtype="choice")`. During v0 training only row 0 of `type_emb` received gradient. Rows 1
(score) and 2 (noul) received zero gradient but AdamW's decoupled weight decay (0.01 × lr 1e-4 × 4,106 steps) still
shrank them by ≈0.4%. The encoder, head and scorer — shared by all types — were fully fine-tuned on choice only. So
v0 *can* technically answer noul/score, but whether it still does so well is an empirical question (§8 → baseline).

## 2. Serialization (`render_options` + `build_sequence`)

```
[CLS] "{type} question: {instructions}" [SEP] [MASK] opt_0 [MASK] opt_1 … [MASK] opt_{K-1} [SEP] state [SEP]
```

`render_options(q)` (option texts in label-index order):

| type | options (index order) | notes |
|---|---|---|
| choice | `"k: v"` for each criterion `k → v` (or `"k"` if `v` empty) | order = request order; **shuffled during Laya training** (`encode_record`, `train and q["t"] != "score"`) |
| score | `"level {i}: {criterion_i}"`, i = 0…K−1 | **never shuffled** (ordinal); the index is written into the text |
| noul | `["false: " + (criteria.false or "no, the statement does not hold"), "true: " + (criteria.true or "yes, the statement holds")]` | **always 2 options, false first**, so `p[1]` is the noul value |

Budgets: every option is `[MASK] + ≤ 48 tokens`; options share `head_max_len` (Laya 192; Krun One 896). If
options exceed the budget minus 16, every option is truncated evenly to `max(4, (head_max_len−16)//K)`. The header is
cut to `max(8, head_max_len − option tokens)`. The state fills the rest of `max_len` (Laya 512; Krun One 1024). If a
marker falls beyond `max_len`, the API raises (`rl_agent_api.py`: "options do not fit"). The upstream API renders a
non-string state with `json.dumps`, and for `instructions` that are not a string, also `json.dumps`.

`_to_internal` (API): `choice` criteria may be a list (→ `{c: None}`); `score` criteria is the **list of levels**;
`noul` criteria is an optional `{"true": …, "false": …}` dict.

## 3. Logits, temperature, probabilities

```python
m = gather(h, marker_pos)                        # hidden state at each option's [MASK]
logits = scorer(m).squeeze(-1).float()            # LayerNorm → Linear(d,d) → GELU → Linear(d,1): one scalar per option
logits = logits.masked_fill(~marker_mask, -1e4)   # padding options
```

Scoring is **identical for all three types**: one scalar per option marker, softmax over the question's options.
There is no type-specific head; the type changes the computation only through the header text and `type_emb`.

Inference (`RLAgent.system_one`): `z = logits[:K] / T` with
`T = temperature_by_options.get(temp_bucket(qtype, K), temperature[qtype])`; `p = softmax(z)`.
`temp_bucket` = `"{type}:{2 | 3-5 | 6-10 | 11+}"`. Shipped Laya values: `temperature = [choice 1.637, score 1.251,
noul 1.983]`, `noul:2 = 1.983`, `score:3-5 = 1.251`, `choice:11+ = 0.10` (sic — fitted on few examples). The model's
`temperature` buffer is all ones and unused by the API.

Note: for `noul`, a 2-way softmax is exactly a sigmoid of the logit difference:
`p_true = softmax([z_f, z_t])[1] = σ(z_t − z_f)`, so 2-way cross entropy **is** binary cross entropy on
`z_t − z_f`, and temperature scaling is Platt scaling without bias.

## 4. Batching / mixed requests

`system_one` builds **one sequence per question** and runs **all questions of a call in one forward pass**
(`collate_items` pads to the longest sequence and the widest option set; `qtype` is a per-row tensor). There is no
cross-attention between questions: each sequence contains only its own question + the state. Therefore:

* mixing types in one request is supported natively (one forward, N heterogeneous rows);
* an answer can only depend on the other questions through padding numerics (padding length changes kernel
  shapes; in bf16 this moves logits by up to ~0.1 — KRUN-005 reload test). Mixed-batch invariance must be measured,
  but there is no semantic coupling to guard against;
* the state is re-encoded once **per question** (cost ∝ number of questions, not amortized).

## 5. Output per type (upstream API)

```python
choice: {"choice": keys[argmax p], "probabilities": {k: p}, "confidence": 1 − H(p)/ln K}
score:  {"score": Σ_i i·p_i, "legend": {str(i): level_i}, "probabilities": {str(i): p_i}, "confidence": 1 − H(p)/ln K}
noul:   {"noul": p[1]}
```

(+ `rl_agent.act_probability` from the act head, which KRUN-002 showed saturates at ≈1.0 and Krun One does not use;
`usage.output_tokens: 0`, which the Krun API never exposes.)

* **Expected score** = `Σ index × probability` over *temperature-scaled* probabilities, rounded to 4 decimals. Not
  argmax.
* **Score confidence** (`confidence_from_probs`) = `1 − H(p)/ln K` ("Jev-style"), computed on the
  temperature-scaled distribution. Same formula as choice confidence in upstream Laya. Krun One v0 exposes
  `top1_top2_margin` as choice confidence instead (KRUN-005 abstention rule). See §7.
* **Noul** returns only `p_true`; no confidence field.

## 6. Training objective upstream (for reference)

`proper_reward`: log score + 0.5 × spherical score for all types, **minus ranked probability score (RPS) for score
questions** (`rps = Σ_k (CDF_q − CDF_target)² / (K−1)`). Targets may be one-hot or soft (`q["soft"]`). Laya was
trained with RLCD (REINFORCE with group baseline over noisy logits against that reward). Krun One uses supervised
soft-target CE instead (KRUN-005); for score we keep Laya's ordinal term as the ordinal-aware candidate
(**CE + RPS**), which is strictly proper and penalises distance (0→4 costs more than 0→1).

Laya's upstream eval (`laya/eval/results.md`) reports the ordinal family as the weakest ("sentiment and rating"
in-task acc 0.442, ECE 0.438; README: "SST-5 0.372"). Laya's training mix is not published in detail; the task
families listed there (sentiment and rating, response quality scoring, moderation, inference/fact checking, reading
comprehension …) mean **public benchmarks like SST-5 or BoolQ may be in Laya's training data** — Laya numbers on such
sets are possibly in-distribution for Laya, never for Krun.

## 7. Score `confidence` — options compared (decision in `results/primitives_baseline/score_confidence.md`)

Let `p` be the (temperature-scaled) distribution over K ordered levels, `s = Σ i p_i`.

| candidate | definition | range | properties |
|---|---|---|---|
| **A. normalized entropy** (upstream) | `1 − H(p)/ln K` | [0,1] | Laya/Jev-compatible; order-blind: `[.5,0,.5]` and `[.5,.5,0]` get the same value although the first is far more uncertain *about the score* |
| B. ordinal dispersion | `1 − Var_p(i) / ((K−1)/2)²` | [0,1] | order-aware: measures the uncertainty of the expected score itself; 0 exactly for the maximal-variance split `[.5,0,…,0,.5]` |
| C. top-1 probability | `max p` | [1/K,1] | order-blind, depends on K |

Choice rule (registered before looking at results): on score **val_eval**, pick the candidate with the highest
Spearman correlation between confidence and −|score − y| (lower error ↔ higher confidence), pooled over sources;
if A is within 0.02 of the best, keep A (compatibility with upstream semantics and the public example).

## 8. Consequences for KRUN-011

1. **No architecture change is needed.** Noul and score use the same encoder/head/scorer; only the header text,
   option rendering and `type_emb` row differ. The artifact size stays 421,293,827 parameters.
2. Krun One's `collate` / `build_sequence` must carry the real `qtype` (currently hard-coded to choice).
3. Serialization must follow upstream exactly for noul (false first, default criteria texts) and score
   (`level i: …`, never permuted) so Laya and Krun are evaluated through the same code path.
4. Per-type temperatures are required (Laya's shipped temperatures differ strongly per type: 1.25–1.98).
5. The worker can keep "one forward per request" for heterogeneous questions.
