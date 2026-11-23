"""Krun One v1.1 architecture candidates (~1.35B - 1.6B) and decision-head variants. Plain data: counted exactly by
scripts/v11/arch_budget.py and benchmarked with random weights by scripts/v11/latency_v11.py BEFORE any training.

Same family as v1 in every candidate: ModernBERT/mmBERT encoder (pre-norm, GLU, RoPE, global attention every 3rd
layer, local window 128), head_dim 64, the ~256k Gemma/mmBERT vocabulary, the v1 serialization and decision stack.

``growth`` says how the candidate can be initialized from the frozen v1 weights:
  exact        function-preserving by construction with standard modules (krun_one.v11.growth.grow_exact)
  width        needs non-integer width growth of the residual stream (krun_one.v11.growth_width; LayerNorm makes this
               exact only up to a small, measured perturbation)
"""

from __future__ import annotations

from ..v1.arch import CANDIDATES, DecisionSpec, EncoderSpec, ModelSpec
from .model import V11Spec

V1 = CANDIDATES[0]                                   # K1-W1536-L22 (879M) — the frozen baseline shape
D2 = DecisionSpec(head_layers=2, head_ffn_mult=2)    # v1 decision stack
D4 = DecisionSpec(head_layers=4, head_ffn_mult=2)


def _m(name, h, layers, heads, ffn, dec=D2, notes=""):
    return ModelSpec(name, EncoderSpec(hidden_size=h, num_layers=layers, num_heads=heads, intermediate_size=ffn), dec, notes)


# (spec, growth route from v1)
ENCODER_CANDIDATES: list[tuple[V11Spec, str]] = [
    (V11Spec(V1), "baseline"),
    (V11Spec(_m("V11-A-W1536-L22-F8192", 1536, 22, 24, 8192, notes="FFN only: stream, heads and depth unchanged")), "exact"),
    (V11Spec(_m("V11-B-W1536-L26-F6144", 1536, 26, 24, 6144, notes="FFN + 4 appended layers")), "exact"),
    (V11Spec(_m("V11-C-W1792-L24-F4096", 1792, 24, 28, 4096, notes="wider stream (x1.17), 28 heads, +2 layers")), "width"),
    (V11Spec(_m("V11-D-W2048-L22-F3072", 2048, 22, 32, 3072, notes="wider stream (x1.33), 32 heads, v1 depth, v1 FFN ratio")), "width"),
    (V11Spec(_m("V11-E-W2048-L22-F4096", 2048, 22, 32, 4096, notes="as D with a larger GLU FFN")), "width"),
    (V11Spec(_m("V11-F-W2048-L24-F3072", 2048, 24, 32, 3072, notes="as D with +2 layers")), "width"),
]

# decision-head study on the v1 encoder (all function-preserving from v1)
HEAD_CANDIDATES: list[tuple[V11Spec, str]] = [
    (V11Spec(V1), "baseline"),
    (V11Spec(ModelSpec("V1+head-ffn4", V1.encoder, DecisionSpec(head_layers=2, head_ffn_mult=4), "wider head FFN")), "exact"),
    (V11Spec(ModelSpec("V1+head-L4", V1.encoder, D4, "deeper head (4 full-sequence layers)")), "exact"),
    (V11Spec(V1, cand_layers=2), "exact"),
    (V11Spec(V1, cand_layers=4), "exact"),
]
