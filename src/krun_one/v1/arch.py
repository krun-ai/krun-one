"""Krun One v1 architecture specs and parameter accounting.

A spec is plain data (JSON-serializable) so candidates can be compared, counted and benchmarked without
weights. ``count_parameters`` is analytic; ``instantiate_counts`` builds the real modules on the ``meta``
device and counts them, and the tests assert the two agree exactly — the numbers in
docs/KRUN_ONE_V1_ARCHITECTURE_SPEC.md come from ``scripts/v1/param_budget.py``, not from mental math.

Encoder = HF ``ModernBertModel`` (the mmBERT architecture: pre-norm, GLU MLP, no biases, RoPE,
alternating global/local attention). Decision stack on top = Laya's (type embedding + transformer head +
per-marker scorer), extended with two extra outputs (applicability, answerability) — see model.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

QTYPES_V1 = {"choice": 0, "score": 1, "noul": 2, "multi": 3}  # rows 0-2 keep Laya's order


@dataclass(frozen=True)
class EncoderSpec:
    vocab_size: int = 256_000
    hidden_size: int = 1536
    num_layers: int = 22
    num_heads: int = 24
    intermediate_size: int = 2304  # GLU: Wi is hidden -> 2 * intermediate
    global_every: int = 3          # ModernBERT/mmBERT: layer i is global iff i % global_every == 0
    local_window: int = 128
    rope_theta_global: float = 160_000.0
    rope_theta_local: float = 160_000.0  # mmBERT uses 160k for both (ModernBERT-large: 10k local)
    max_position: int = 8192
    pad_token_id: int = 0          # Gemma-2 / mmBERT ids
    bos_token_id: int = 2
    eos_token_id: int = 1
    mask_token_id: int = 4

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def layer_types(self) -> list[str]:
        return ["full_attention" if i % self.global_every == 0 else "sliding_attention" for i in range(self.num_layers)]

    def hf_config(self):
        from transformers import ModernBertConfig

        return ModernBertConfig(
            vocab_size=self.vocab_size, hidden_size=self.hidden_size, intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_layers, num_attention_heads=self.num_heads,
            max_position_embeddings=self.max_position, local_attention=self.local_window,
            layer_types=self.layer_types(),
            rope_parameters={"full_attention": {"rope_type": "default", "rope_theta": self.rope_theta_global},
                             "sliding_attention": {"rope_type": "default", "rope_theta": self.rope_theta_local}},
            pad_token_id=self.pad_token_id, bos_token_id=self.bos_token_id, cls_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id, sep_token_id=self.eos_token_id, tie_word_embeddings=True)


@dataclass(frozen=True)
class DecisionSpec:
    head_layers: int = 2
    head_ffn_mult: int = 4       # Laya: nn.TransformerEncoderLayer(d, d // 64, 4 * d)
    n_qtypes: int = len(QTYPES_V1)
    scorer_outputs: int = 2      # [rank logit, applicability logit] per option marker
    answerability_head: bool = True


@dataclass(frozen=True)
class ModelSpec:
    name: str
    encoder: EncoderSpec = field(default_factory=EncoderSpec)
    decision: DecisionSpec = field(default_factory=DecisionSpec)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelSpec":
        return cls(name=d["name"], encoder=EncoderSpec(**d["encoder"]), decision=DecisionSpec(**d["decision"]),
                   notes=d.get("notes", ""))

    def with_encoder(self, **kw) -> "ModelSpec":
        return replace(self, encoder=replace(self.encoder, **kw))

    def with_decision(self, **kw) -> "ModelSpec":
        return replace(self, decision=replace(self.decision, **kw))


# ---------------------------------------------------------------------------------------------- analytic counts
def encoder_counts(e: EncoderSpec) -> dict[str, int]:
    h, i = e.hidden_size, e.intermediate_size
    per_layer_linear = 3 * h * h + h * h + h * 2 * i + i * h   # Wqkv, attn Wo, Wi (GLU), mlp Wo; no biases
    norms = (2 * e.num_layers - 1) * h                          # layer 0 has no attn_norm
    embeddings = e.vocab_size * h + h                           # tok_embeddings + embedding LayerNorm (no bias)
    body = e.num_layers * per_layer_linear + norms + h          # + final_norm
    return {"embeddings": embeddings, "body": body, "total": embeddings + body,
            "per_layer": per_layer_linear + 2 * h}


def decision_counts(d: DecisionSpec, h: int) -> dict[str, int]:
    ff = d.head_ffn_mult * h
    # nn.TransformerEncoderLayer: in_proj (3h*h + 3h), out_proj (h*h + h), linear1 (ff*h + ff), linear2 (h*ff + h),
    # norm1/norm2 (2h each, with bias)
    head_layer = (3 * h * h + 3 * h) + (h * h + h) + (ff * h + ff) + (h * ff + h) + 4 * h
    head = d.head_layers * head_layer
    type_emb = d.n_qtypes * h
    scorer = 2 * h + (h * h + h) + (h * d.scorer_outputs + d.scorer_outputs)  # LN, Linear(h,h), Linear(h,outs)
    answer = (2 * (2 * h) + (2 * h * h + h) + (h + 1)) if d.answerability_head else 0  # LN(2h), Linear(2h,h), Linear(h,1)
    return {"head": head, "type_emb": type_emb, "scorer": scorer, "answerability": answer,
            "total": head + type_emb + scorer + answer}


def count_parameters(spec: ModelSpec) -> dict[str, int]:
    enc = encoder_counts(spec.encoder)
    dec = decision_counts(spec.decision, spec.encoder.hidden_size)
    return {"encoder_embeddings": enc["embeddings"], "encoder_body": enc["body"], "encoder_total": enc["total"],
            "decision_head": dec["head"], "decision_other": dec["type_emb"] + dec["scorer"] + dec["answerability"],
            "decision_total": dec["total"], "total": enc["total"] + dec["total"],
            "non_embedding": enc["body"] + dec["total"]}


def instantiate_counts(spec: ModelSpec) -> dict[str, int]:
    """Real module count on the meta device (no memory is allocated)."""
    import torch

    from .model import KrunOneV1Model

    with torch.device("meta"):
        m = KrunOneV1Model.from_spec(spec)
    enc = sum(p.numel() for p in m.encoder.parameters())
    emb = m.encoder.embeddings.tok_embeddings.weight.numel() + sum(
        p.numel() for p in m.encoder.embeddings.norm.parameters())
    total = sum(p.numel() for p in m.parameters())
    return {"encoder_embeddings": emb, "encoder_total": enc, "total": total, "non_embedding": total - emb}


def forward_flops_per_token(spec: ModelSpec, seq_len: int) -> float:
    """Dense forward FLOPs per token (2 * MACs): linear layers + attention score/value products.

    Local layers attend to min(seq_len, local_window + 1) keys. Decision head attends globally.
    Embedding lookup is ~free and excluded; the scorer runs on K markers only (negligible).
    """
    e, d = spec.encoder, spec.decision
    h, i = e.hidden_size, e.intermediate_size
    lin = 2 * (4 * h * h + 3 * h * i) * e.num_layers
    attn = 0.0
    for t in e.layer_types():
        keys = seq_len if t == "full_attention" else min(seq_len, e.local_window + 1)
        attn += 2 * 2 * keys * h
    head = d.head_layers * (2 * (4 * h * h + 2 * d.head_ffn_mult * h * h) + 2 * 2 * seq_len * h)
    return lin + attn + head


# ---------------------------------------------------------------------------------------------- candidates
LAYA_HEAD = DecisionSpec(head_layers=2, head_ffn_mult=4, n_qtypes=3, scorer_outputs=1, answerability_head=False)

REFERENCE_V03 = ModelSpec(
    name="krun-one-v0.3 (reference)",
    encoder=EncoderSpec(vocab_size=50_368, hidden_size=1024, num_layers=28, num_heads=16, intermediate_size=2624,
                        rope_theta_local=10_000.0, pad_token_id=50283, bos_token_id=50281, eos_token_id=50282,
                        mask_token_id=50284),
    decision=LAYA_HEAD,
    notes="ModernBERT-large + Laya head; includes the unused act head in the real artifact (+263,170 params)")

CANDIDATES: list[ModelSpec] = [
    ModelSpec("K1-W1536-L22", EncoderSpec(hidden_size=1536, num_layers=22, num_heads=24, intermediate_size=2304),
              DecisionSpec(head_layers=2, head_ffn_mult=2),
              notes="mmBERT-base x2 width (HyperCloning-compatible: every weight is an exact 2x tiling of mmBERT-base)"),
    ModelSpec("K1-W1536-L22-H1", EncoderSpec(hidden_size=1536, num_layers=22, num_heads=24, intermediate_size=2304),
              DecisionSpec(head_layers=1, head_ffn_mult=4),
              notes="same encoder, 1-layer Laya head"),
    ModelSpec("K1-W1280-L28", EncoderSpec(hidden_size=1280, num_layers=28, num_heads=20, intermediate_size=2560),
              DecisionSpec(head_layers=2, head_ffn_mult=2),
              notes="ModernBERT-large depth, 1.25x width; no pretrained source of this shape (scratch or depth/width growth)"),
    ModelSpec("K1-W1792-L18", EncoderSpec(hidden_size=1792, num_layers=18, num_heads=28, intermediate_size=2688),
              DecisionSpec(head_layers=2, head_ffn_mult=2),
              notes="wide & shallow (18 layers): fewer sequential kernels at batch 1"),
    ModelSpec("K1-W1536-L22-hd128", EncoderSpec(hidden_size=1536, num_layers=22, num_heads=12, intermediate_size=2304),
              DecisionSpec(head_layers=2, head_ffn_mult=2),
              notes="same shape as K1-W1536-L22 with head_dim 128 (fewer, larger heads)"),
]

PROXY = ModelSpec(
    name="K1-proxy (mmBERT-small)",
    encoder=EncoderSpec(vocab_size=256_000, hidden_size=384, num_layers=22, num_heads=6, intermediate_size=1152),
    decision=DecisionSpec(head_layers=2, head_ffn_mult=4),
    notes="initialized from jhu-clsp/mmBERT-small; ~42M non-embedding encoder params")

# Latency-only variants (not parameter candidates): the chosen encoder without the v1 answerability additions
# ([NULL] token, 2-output scorer, answer head) — measures what answerability costs at serving time.
LATENCY_EXTRA: list[ModelSpec] = [
    ModelSpec("K1-W1536-L22-plain", CANDIDATES[0].encoder,
              DecisionSpec(head_layers=2, head_ffn_mult=2, scorer_outputs=1, answerability_head=False),
              notes="K1-W1536-L22 without [NULL] / applicability / answer head (latency reference)"),
]

# Reference for the HyperCloning scaling experiment: mmBERT-base (not grown) + the same v1 decision stack.
MMBERT_BASE_REF = ModelSpec(
    "mmbert-base-ref", EncoderSpec(hidden_size=768, num_layers=22, num_heads=12, intermediate_size=1152),
    DecisionSpec(head_layers=2, head_ffn_mult=2), notes="jhu-clsp/mmBERT-base + v1 stack (scaling reference)")
SPECS = {"proxy": PROXY, "k1": CANDIDATES[0], "mmbert-base-ref": MMBERT_BASE_REF}
