"""Function-preserving growth of Krun One v1 along the axes that are EXACT with standard modules.

``grow_exact`` builds a larger KrunOneV1Model that computes the same function as the source (bitwise in fp32 up to
the order of float additions), by construction:

* FFN width (``intermediate_size``): ModernBERT's MLP is ``Wo(act(a) * g)`` with ``[a | g] = Wi(x)``. The old rows of
  ``Wi`` are copied into the first ``i`` slots of each half; the new GLU units get random ``Wi`` rows and ZERO ``Wo``
  columns, so they contribute nothing until trained.
* encoder depth: new layers are appended after the last layer with ``attn.Wo = 0`` and ``mlp.Wo = 0`` — a pre-norm
  residual block with a zero output projection is the identity. Appending keeps the global/local attention pattern
  of the existing layers (layer i is global iff i % 3 == 0).
* decision-head depth / FFN: same two ideas on nn.TransformerEncoderLayer (``self_attn.out_proj`` and ``linear2``
  zeroed for new layers; zero ``linear2`` columns for new FFN units).

Width of the residual stream (hidden size) is NOT handled here: with LayerNorm it is exact only for integer
multiples (v1's HyperCloning, x2). The non-integer case is krun_one.v11.growth_width.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from ..v1.arch import ModelSpec
from ..v1.model import KrunOneV1Model


def grown_spec(spec: ModelSpec, name: str, intermediate_size: int | None = None, num_layers: int | None = None,
               head_layers: int | None = None, head_ffn_mult: int | None = None) -> ModelSpec:
    enc = replace(spec.encoder, intermediate_size=intermediate_size or spec.encoder.intermediate_size,
                  num_layers=num_layers or spec.encoder.num_layers)
    dec = replace(spec.decision, head_layers=head_layers or spec.decision.head_layers,
                  head_ffn_mult=head_ffn_mult or spec.decision.head_ffn_mult)
    if (enc.intermediate_size < spec.encoder.intermediate_size or enc.num_layers < spec.encoder.num_layers
            or dec.head_layers < spec.decision.head_layers or dec.head_ffn_mult < spec.decision.head_ffn_mult):
        raise ValueError("growth only")
    return ModelSpec(name, enc, dec, notes=f"function-preserving growth of {spec.name}")


@torch.no_grad()
def grow_exact(src: KrunOneV1Model, spec: ModelSpec, seed: int = 0) -> tuple[KrunOneV1Model, dict]:
    """Return (grown model, report of expanded tensors). ``src`` is not modified."""
    torch.manual_seed(seed)
    dst = KrunOneV1Model.from_spec(spec, attn_implementation=src.encoder.config._attn_implementation)
    se, de = src.encoder, dst.encoder
    i_old, L_old = se.config.intermediate_size, se.config.num_hidden_layers
    rep = {"copied": [], "expanded": [], "new_identity_blocks": []}

    def copy(a, b, name):
        b.copy_(a)
        rep["copied"].append(name)

    copy(se.embeddings.tok_embeddings.weight, de.embeddings.tok_embeddings.weight, "embeddings.tok_embeddings")
    copy(se.embeddings.norm.weight, de.embeddings.norm.weight, "embeddings.norm")
    copy(se.final_norm.weight, de.final_norm.weight, "final_norm")
    for li, dl in enumerate(de.layers):
        if li < L_old:
            sl = se.layers[li]
            if hasattr(sl.attn_norm, "weight"):
                dl.attn_norm.weight.copy_(sl.attn_norm.weight)
            dl.mlp_norm.weight.copy_(sl.mlp_norm.weight)
            dl.attn.Wqkv.weight.copy_(sl.attn.Wqkv.weight)
            dl.attn.Wo.weight.copy_(sl.attn.Wo.weight)
            i_new = dl.mlp.Wo.weight.shape[1]
            if i_new == i_old:
                dl.mlp.Wi.weight.copy_(sl.mlp.Wi.weight)
                dl.mlp.Wo.weight.copy_(sl.mlp.Wo.weight)
            else:
                dl.mlp.Wi.weight[:i_old].copy_(sl.mlp.Wi.weight[:i_old])                       # activation half
                dl.mlp.Wi.weight[i_new:i_new + i_old].copy_(sl.mlp.Wi.weight[i_old:])          # gate half
                dl.mlp.Wo.weight.zero_()
                dl.mlp.Wo.weight[:, :i_old].copy_(sl.mlp.Wo.weight)
                if li == 0:
                    rep["expanded"].append(f"layers.*.mlp.Wi {tuple(sl.mlp.Wi.weight.shape)} -> {tuple(dl.mlp.Wi.weight.shape)} "
                                           "(old rows kept per GLU half, new rows random)")
                    rep["expanded"].append(f"layers.*.mlp.Wo {tuple(sl.mlp.Wo.weight.shape)} -> {tuple(dl.mlp.Wo.weight.shape)} "
                                           "(new columns zero)")
        else:
            dl.attn.Wo.weight.zero_()
            dl.mlp.Wo.weight.zero_()
            rep["new_identity_blocks"].append(f"encoder.layers.{li} ({de.config.layer_types[li]})")
    # decision stack
    dst.type_emb.load_state_dict(src.type_emb.state_dict())
    dst.scorer.load_state_dict(src.scorer.state_dict())
    if src.answer_head is not None:
        dst.answer_head.load_state_dict(src.answer_head.state_dict())
    n_old = len(src.head.layers) if src.head is not None else 0
    for li, dl in enumerate(dst.head.layers if dst.head is not None else []):
        if li < n_old:
            sl = src.head.layers[li]
            ff_old = sl.linear1.weight.shape[0]
            for k in ("self_attn", "norm1", "norm2"):
                getattr(dl, k).load_state_dict(getattr(sl, k).state_dict())
            if dl.linear1.weight.shape[0] == ff_old:
                dl.linear1.load_state_dict(sl.linear1.state_dict())
                dl.linear2.load_state_dict(sl.linear2.state_dict())
            else:
                dl.linear1.weight[:ff_old].copy_(sl.linear1.weight)
                dl.linear1.bias[:ff_old].copy_(sl.linear1.bias)
                dl.linear2.weight.zero_()
                dl.linear2.weight[:, :ff_old].copy_(sl.linear2.weight)
                dl.linear2.bias.copy_(sl.linear2.bias)
                if li == 0:
                    rep["expanded"].append(f"head.layers.*.linear1/linear2 FFN {ff_old} -> {dl.linear1.weight.shape[0]} (new columns zero)")
        else:
            dl.self_attn.out_proj.weight.zero_()
            dl.self_attn.out_proj.bias.zero_()
            dl.linear2.weight.zero_()
            dl.linear2.bias.zero_()
            rep["new_identity_blocks"].append(f"head.layers.{li}")
    rep["copied"] += ["layers.*.attn.{Wqkv,Wo}", "layers.*.{attn_norm,mlp_norm}", "type_emb", "scorer", "answer_head",
                      "head.layers.*.{self_attn,norm1,norm2}"]
    return dst, rep
