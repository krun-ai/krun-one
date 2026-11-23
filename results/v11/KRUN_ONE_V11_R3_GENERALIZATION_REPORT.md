# Krun One v1.1 — R3 generalization report (unseen combinations, counterfactual families, stuck rate)

Date: 2026-10-02 · Protocol: `configs/v11/r3_protocol.yaml` · Tables: `results/v11/r3/techbench/protected.md`,
`val.md` · R2 reference re-scored with the R3 definitions before any R3 result: `results/v11/r3/r2_reference.json`.
All numbers are on protected sets, raw rendering, unless stated. R3 = the selected recipe (random batching), two seeds.

## 1. Verdict

**Generalization did not improve.** Unseen requirement combinations 59.0% (R2 61.2%), constraint flips with both members
right 47.0% (R2 48.3%), stuck rate 20.9% (R2 18.7%). The three registered gates of this block fail. In the registered
decision tree this is **case D: the dataset / generalization problem is not solved — and it is not a reason to grow the
model.** What R3 adds is a sharper description of the problem.

## 2. Definitions

* **Unseen combination**: a requirement set (of a rule table) whose exact set never appears in training. Seen sets: 100%.
* **Pair**: a base problem and a variant with the same candidates, order and wording in which one constraint changed and
  the correct decision is different (R1 / R2 constraint flips; R3 families).
* **Both correct**: share of pairs with both members right.
* **Stuck rate**: P(the variant is answered with the base's correct answer | the base was answered correctly).

## 3. Unseen requirement combinations

| | n | v1 | R2 (2 seeds) | R3 seed 1 | R3 seed 2 |
|---|---:|---:|---:|---:|---:|
| all unseen sets | 990 | 52.1 | 59.9 / 62.4 | 58.3 | 59.8 |
| 1 requirement | 350 | 46.6 | 37.1 / 44.0 | 40.3 | 38.9 |
| 2 requirements | 597 | 55.1 | 70.5 / 70.7 | 65.8 | 69.7 |
| 3 requirements | 43 | 55.8 | 97.7 / 97.7 | 100.0 | 93.0 |
| seen sets (control) | 552 | 39.5 | 99.8 / 100 | 100 | 100 |
| with capabilities written next to each option | 990 | 75.3 | 74.3 | 79.3 | 79.7 |
| invented-product catalogs, unseen sets (R3 extension) | 726 | 31.7 | 27.4 | 55.6 | 54.7 |
| … with capabilities next to each option | 726 | 32.1 | 36.9 | 72.6 | 70.9 |

Per table (R3 seeds; v1 → R2 kept seed in brackets): messaging 100 / 100 [78 → 100] · replication 98 / 97 [95 → 97] ·
compute 81 / 78 [55 → 71] · datastore 77 / 74 [40 → 79] · API style 62 / 59 [23 → 44] · ML serving 42 / 50 [74 → 43] ·
scaling 41 / 41 [41 → 37] · structure 23 / 47 [5 → 77] · security 25 / 13 [37 → 8] · network component 19 / 20 [65 → 21].

What this shows:

1. **The failure is not "combinatorial".** The more requirements an unseen set has, the *better* the model does (98% at
   three). It fails on unseen sets with **one** requirement (39–40%, below v1's 46.6%). A one-requirement decision is
   "which of these options has property P, and which of those is the lightest": a knowledge question about a property
   the model never saw alone, asked with every other signal removed.
2. **The R3 fact probes did not fix it.** 300 true / false facts, 150 "select every option that has P" and 60 "which is
   lightest" items per table taught the facts in a different question format; the knowledge did not carry over to the
   choice decision (1 requirement: R2 40.6% → R3 39.6%). Three tables are *below v1* in both rounds (network component,
   security, ML serving): there the model answers an unseen single requirement with the winner it memorised for the
   sets it did see.
3. **Reading the rule from the context is learnable but not learned well**: on catalogs of invented products, where
   nothing can be memorised, R3 reaches 55% (v1 32%, R2 27%) and 71–73% when each option repeats its capabilities.
   The rule "all requirements met, then lowest overhead" is applied far from reliably even when every fact is on the
   page — the same computation gap as in the capability tracks (text report § 5).
4. With capabilities written next to each option, unseen sets reach only 79%: on the option names it knows, the model
   still leans on what it memorised.

## 4. Constraint flips and counterfactual families

| pairs | n | v1 both / stuck | R2 kept seed both / stuck | R3 seed 1 | R3 seed 2 |
|---|---:|---|---|---|---|
| KrunTechBench v1 constraint flips | 286 | 2.8 / 65.1 | 46.5 / 20.8 (other seed 50.0 / 16.7) | 49.7 / 20.1 | 44.4 / 21.7 |
| … with explicit consequences | 286 | 8.7 / 39.4 | 94.4 / 0.0 | 95.8 / 0.0 | 95.8 / 0.0 |
| table families on unseen sets (R3 extension) | 472 | 9.5 / 51.0 | 10.6 / 33.3 | 16.5 / 32.7 | 15.7 / 32.9 |
| … with explicit consequences | 472 | 31.6 / 19.8 | 24.8 / 22.0 | 44.7 / 17.6 | 41.5 / 20.4 |
| capability families (R3 extension) | 1,171 | 1.3 / 95.0 | 1.7 / 91.9 | 3.8 / 85.4 | 3.8 / 85.0 |
| … with explicit consequences | 754 | 10.2 / 58.5 | 19.1 / 35.8 | 71.0 / 7.3 | 72.1 / 5.2 |

* On the R1 / R2 flips nothing moved: both-correct 47.0% (R2 48.3%), stuck 20.9% (R2 18.7%) — differences inside the
  seed spread (R2's two seeds: 20.8 and 16.7).
* **When the consequence is stated, the stuck rate is 0–7%.** The model is not insensitive to a changed constraint as
  such; it is insensitive to a changed constraint **whose effect it would have to compute**. On capability families
  from the raw state, when the base is right the variant keeps the base answer in 85% of the cases: the base was right
  by prior, not by reading the limit, the deadline or the link that changed.
* So "stuck" and "does not compute consequences" are the same problem seen from two sides.

## 5. Family-aware batching

Tested against random batching with everything else equal, two seeds each (validation, text report § 3): score 47.3 ± 1.0
vs 48.1 ± 0.9, stuck rate over all validation families 63.5% vs 65.4%, stuck rate on the v1 flips 26.6% vs 25.1%.
No effect. Showing the model both members of a pair in the same batch does not teach it to recompute.

## 6. Hard negatives and OOD (preserved)

Gold above the hardest negative 79.5% (R2 79.4%); margin +4.14 logit (R2 +4.29; gate ≥ 3.99 ✅); hard-negative slice
65.7%; OOD false acceptance 0.1% on the v1 benchmark and 0.0–0.1% over all 866 none-of-the-above items of the two
protected files.

## 7. What this means for the next data build (not started)

1. The "unseen combination" slice mixes two things; keep them apart from now on: *unseen single properties* (knowledge)
   and *unseen conjunctions* (composition — already ≥ 93% at three requirements, 66–70% at two).
2. Facts taught as true / false or multi-select did not transfer to the choice format. If single-property decisions
   matter, they have to be trained in the decision format, with the property varied over many phrasings.
3. Where the answer must come from the context (catalogs, policies, numbers, lookups), more examples of the same kind
   are unlikely to be enough: the dedicated diagnostic (text report § 8) trained only on such data.
