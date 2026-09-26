# Score `confidence` — comparison (rule: `primitives_architecture_audit.md` §7, registered before results)

Metric: Spearman ρ between the confidence of a score answer and −|score − y| (higher = confidence tracks the error
of the expected score), score **val_eval**, in-domain (HelpSteer2 + MHS), calibrated probabilities.
Source: `results/primitives_*/eval_*.json` → `score.val_eval.calibrated.in_domain.confidence_vs_error_spearman`.

| model | A. normalized entropy `1 − H(p)/ln K` | B. ordinal dispersion `1 − Var/((K−1)/2)²` | C. top-1 probability |
|---|---:|---:|---:|
| Laya (1024/896) | 0.113 | **0.311** | 0.070 |
| Laya (512/192) | 0.125 | **0.318** | 0.079 |
| Krun One v0 | 0.220 | **0.346** | 0.178 |
| v0.2 M50_CE | 0.378 | 0.393 | **0.399** |
| v0.2 M40_CE | 0.398 | **0.416** | 0.415 |
| v0.2 M50_RPS | 0.378 | 0.396 | **0.398** |

**Decision (rule: keep A unless another candidate beats it by more than 0.02):** applied per candidate it gives
M50_CE → C (top-1 beats A by 0.021), M40_CE → A (best alternative +0.018), M50_RPS → A (+0.020, not more than
0.02). No candidate was promoted this round, so **no definition is frozen**; the public contract is not affected.
Observations to carry into the next round: on untrained-for-score models the order-aware ordinal dispersion is far
better (0.31–0.35 vs 0.11–0.22); once trained on score data all three converge (0.38–0.42); ordinal dispersion is
never worse than entropy and is the only candidate that treats a bimodal split (`[.5, 0, .5]` → 0) as maximal
uncertainty about the score. Recommendation for the next registered rule: **B (ordinal dispersion)**, or keep A
for upstream/Jev compatibility — a product decision, since the gap to A is ~0.02 (0.018–0.021) on trained models.

Mathematical definitions (p = temperature-scaled distribution over K ordered levels, s = Σ i·p_i):

* A: `c = 1 + Σ p_i ln p_i / ln K` ∈ [0, 1]; 1 = one-hot, 0 = uniform. Order-blind.
* B: `c = 1 − Σ p_i (i − s)² / ((K − 1)/2)²` ∈ [0, 1]; 1 = one-hot, 0 = mass split between the two extremes.
* C: `c = max_i p_i` ∈ [1/K, 1].
