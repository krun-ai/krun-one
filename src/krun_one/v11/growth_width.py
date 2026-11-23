"""Function-preserving growth of the residual-stream WIDTH by a non-integer factor (e.g. 1536 -> 1792 / 2048).

Why this needs a construction: v1's HyperCloning duplicates the stream ([x, x]), which LayerNorm tolerates only for
integer factors. Padding the stream with zeros is NOT function-preserving in general, because LayerNorm's mean and
variance are taken over all N = n + m dimensions.

The construction (n old dims, m new "pad" dims):

1. **Centered stream.** Every LayerNorm subtracts the mean, so every writer into the encoder stream (attention
   ``Wo``, MLP ``Wo``, the embedding) can be replaced by its mean-free version ``C W`` (C = I - 11^T/n) without
   changing what any LayerNorm reader sees. With a mean-free stream and pads that sum to ~0, LN over N dims has the
   same mean (0) and a variance scaled by exactly n/N, which is undone by ``g' = g * sqrt(n/N)`` and
   ``eps' = eps * n/N``. Pad dims get LN weight 0, so no reader sees them.
2. **Layer 0 has no attention LayerNorm** (ModernBERT): its ``Wqkv`` reads the embedding output e directly, mean
   included. The per-token mean c_v is carried by two pad dims (+k c_v, -k c_v) and ``Wqkv`` of layer 0 gets two
   extra columns that add ``c_v * Wqkv 1`` back. Exact (linear).
3. **The decision head reads final_norm's output, which is not mean-free.** The head is widened with the
   "mean in the pads" rule (pads = mean of the old dims makes LN exact again), so final_norm must output that mean
   on its pads. The mean of final_norm's output is the linear functional f(x) = g_f^T C x / sigma of the stream, so
   the remaining pad dims carry (+/- lambda f(x)) through the encoder (every writer gets extra rows
   ``+/- lambda g_f^T C W``) and final_norm's pad weights turn it into the mean.
4. **Embedding.** The table is a lookup, so its rows are set such that the widened embedding LayerNorm outputs
   exactly [C e_v | carriers]; two "absorber" pad dims with LN weight 0 equalize the per-token variance.

What is NOT exact, by construction, and is measured by scripts/v11/growth_verify.py:
* carriers add ``(2 k^2 c^2 + (m-4) lambda^2 f^2) / N`` to every encoder LN variance (tiny for small k, lambda);
* 4 of the m head pads hold 0 instead of the mean (variance error ~ 4 mu^2 / N);
* the answerability head's LayerNorm spans 2 vectors (closed signal ``ans``; not a public output);
* new attention heads / FFN units / scorer units are present but disconnected (zero output weights).

Attention heads: N // 64 heads of dim 64; the old heads keep their positions, new heads have zero ``Wo`` columns.
FFN: may be widened in the same call (zero ``Wo`` columns for new GLU units), as in growth.grow_exact.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..v1.arch import ModelSpec
from ..v1.model import KrunOneV1Model


def _set_ln(ln: nn.LayerNorm, weight: torch.Tensor, n: int, scale: float, eps: float, bias: torch.Tensor | None = None,
            pad_weight: torch.Tensor | None = None) -> None:
    ln.weight.zero_()
    ln.weight[:n].copy_(weight.double().mul(scale))
    if pad_weight is not None:
        ln.weight[n:].copy_(pad_weight)
    if ln.bias is not None:
        ln.bias.zero_()
        if bias is not None:
            ln.bias[:n].copy_(bias)
    ln.eps = eps


@torch.no_grad()
def widen_stream(src: KrunOneV1Model, spec: ModelSpec, lam: float = 1e-3, kappa: float = 1e-2, seed: int = 0
                 ) -> tuple[KrunOneV1Model, dict]:
    """Grow ``src`` to ``spec`` (same depth; hidden N > n, heads N/64, FFN >= old). Returns (model, report)."""
    se = src.encoder
    n, N = se.config.hidden_size, spec.encoder.hidden_size
    i_old, i_new = se.config.intermediate_size, spec.encoder.intermediate_size
    m = N - n
    if m < 8 or m % 2 or spec.encoder.num_layers != se.config.num_hidden_layers or i_new < i_old:
        raise ValueError("need an even widening of >= 8 dims, the same depth and FFN >= old")
    if spec.encoder.head_dim != se.config.hidden_size // se.config.num_attention_heads:
        raise ValueError("head_dim must not change")
    torch.manual_seed(seed)
    dst = KrunOneV1Model.from_spec(spec, attn_implementation=se.config._attn_implementation)
    de = dst.encoder
    dd = torch.float64
    r = math.sqrt(n / N)
    C = torch.eye(n, dtype=dd) - torch.full((n, n), 1.0 / n, dtype=dd)
    P1, P3, P2 = [n, n + 1], [n + 2, n + 3], list(range(n + 4, N))
    m2 = len(P2)
    sgn = torch.tensor([1.0 if j % 2 == 0 else -1.0 for j in range(m2)], dtype=dd)
    eps = se.final_norm.eps
    eps_new = eps * n / N
    g_f = se.final_norm.weight.double()
    w_f = C @ g_f                                         # f(x) = w_f . x  (= g_f^T C x)

    def writer(W_old: torch.Tensor, dst_w: torch.Tensor, cols: int) -> None:
        """Encoder writer (out n x in): centered rows + carrier rows; everything else zero."""
        Wd = W_old.double()
        dst_w.zero_()
        dst_w[:n, :cols].copy_(C @ Wd)
        dst_w[P2, :cols] = (lam * sgn[:, None] * (w_f @ Wd)[None, :]).to(dst_w.dtype)

    # ---- embedding: rows such that LN'(row) = [C e_v | +k c_v, -k c_v | 0, 0 | +/- lam f(e_v)]
    g_e = se.embeddings.norm.weight.double()
    tok_w = se.embeddings.tok_embeddings.weight

    def targets(lo: int, hi: int) -> torch.Tensor:        # chunked: the full table in float64 would not fit in RAM
        z = tok_w[lo:hi].double()
        z = z - z.mean(1, keepdim=True)
        e = g_e * z / torch.sqrt(z.pow(2).mean(1, keepdim=True) + se.embeddings.norm.eps)   # original stream_0 per token
        c = e.mean(1)
        T = torch.zeros(hi - lo, N, dtype=dd)
        T[:, :n] = e - c[:, None]
        T[:, P1[0]], T[:, P1[1]] = kappa * c, -kappa * c
        T[:, P2] = lam * sgn[None, :] * (e @ w_f)[:, None]
        return T

    V, step = tok_w.shape[0], 16384
    G = 1.05 * math.sqrt(max(float(targets(lo, min(V, lo + step)).pow(2).sum(1).max()) for lo in range(0, V, step))
                         / (N * (1 - eps_new)))
    for lo in range(0, V, step):
        rows = targets(lo, min(V, lo + step)) / G
        b = torch.sqrt(((N * (1 - eps_new)) - rows.pow(2).sum(1)) / 2)                  # absorbers: var(row) + eps' = 1
        rows[:, P3[0]], rows[:, P3[1]] = b, -b
        de.embeddings.tok_embeddings.weight[lo:lo + rows.shape[0]].copy_(rows)
    pad_w = torch.full((m,), G, dtype=dd)
    pad_w[2:4] = 0.0
    de.embeddings.norm.weight[:n] = G
    de.embeddings.norm.weight[n:].copy_(pad_w)
    de.embeddings.norm.eps = eps_new

    # ---- encoder layers
    ones = torch.ones(n, dtype=dd)
    for li, (sl, dl) in enumerate(zip(se.layers, de.layers)):
        if isinstance(dl.attn_norm, nn.LayerNorm):
            _set_ln(dl.attn_norm, sl.attn_norm.weight, n, r, eps_new)
        _set_ln(dl.mlp_norm, sl.mlp_norm.weight, n, r, eps_new)
        Wq = sl.attn.Wqkv.weight.double()                 # (3n, n), blocks q | k | v, old heads first in each block
        dWq = dl.attn.Wqkv.weight
        keep = dWq.clone()                                # random rows for the NEW heads (disconnected: Wo cols are zero)
        dWq.zero_()
        for blk in range(3):
            dWq[blk * N + n:(blk + 1) * N].copy_(keep[blk * N + n:(blk + 1) * N])
            dWq[blk * N + n:(blk + 1) * N, n:] = 0        # new heads do not read the carriers
            dWq[blk * N:blk * N + n, :n].copy_(Wq[blk * n:(blk + 1) * n])
            if li == 0:                                   # no attn LayerNorm: add c_v * Wqkv 1 back from the P1 carriers
                col = (Wq[blk * n:(blk + 1) * n] @ ones) / (2 * kappa)
                dWq[blk * N:blk * N + n, P1[0]] = col.to(dWq.dtype)
                dWq[blk * N:blk * N + n, P1[1]] = (-col).to(dWq.dtype)
        writer(sl.attn.Wo.weight, dl.attn.Wo.weight, n)
        Wi = sl.mlp.Wi.weight
        dWi = dl.mlp.Wi.weight
        dWi[:, n:] = 0                                     # no GLU unit reads the pads
        dWi[:i_old, :n].copy_(Wi[:i_old])
        dWi[i_new:i_new + i_old, :n].copy_(Wi[i_old:])
        writer(sl.mlp.Wo.weight, dl.mlp.Wo.weight, i_old)
    pad_f = torch.zeros(m, dtype=dd)
    pad_f[4:] = sgn * r / (n * lam) * (m / m2)             # final_norm pads -> mean(h) (x m/m2: 4 pads stay 0)
    _set_ln(de.final_norm, se.final_norm.weight, n, r, eps_new, pad_weight=pad_f)

    # ---- decision stack: "mean in the pads" (P2 pads = m/m2 * mean of the old dims)
    def mean_rows(W: torch.Tensor) -> torch.Tensor:        # (n, in) -> (in,)
        return W.double().mean(0) * (m / m2)

    dst.type_emb.weight.zero_()
    dst.type_emb.weight[:, :n].copy_(src.type_emb.weight)
    dst.type_emb.weight[:, P2] = (src.type_emb.weight.double().mean(1, keepdim=True) * (m / m2)).to(dst.type_emb.weight.dtype)

    def head_writer(lin_s: nn.Linear, lin_d: nn.Linear) -> None:
        cols = lin_s.weight.shape[1]
        lin_d.weight.zero_()
        lin_d.bias.zero_()
        lin_d.weight[:n, :cols].copy_(lin_s.weight)
        lin_d.bias[:n].copy_(lin_s.bias)
        lin_d.weight[P2, :cols] = mean_rows(lin_s.weight)[None, :].to(lin_d.weight.dtype)
        lin_d.bias[P2] = float(lin_s.bias.double().mean() * (m / m2))

    for sl, dl in zip(src.head.layers if src.head is not None else [], dst.head.layers):
        e_h = sl.norm1.eps * n / N
        _set_ln(dl.norm1, sl.norm1.weight, n, r, e_h, bias=sl.norm1.bias)
        _set_ln(dl.norm2, sl.norm2.weight, n, r, e_h, bias=sl.norm2.bias)
        Wp, bp = sl.self_attn.in_proj_weight, sl.self_attn.in_proj_bias
        dWp, dbp = dl.self_attn.in_proj_weight, dl.self_attn.in_proj_bias
        dWp[:, n:] = 0
        for blk in range(3):
            dWp[blk * N:blk * N + n].zero_()
            dWp[blk * N:blk * N + n, :n].copy_(Wp[blk * n:(blk + 1) * n])
            dbp[blk * N:blk * N + n].copy_(bp[blk * n:(blk + 1) * n])
        head_writer(sl.self_attn.out_proj, dl.self_attn.out_proj)
        ff = sl.linear1.weight.shape[0]
        dl.linear1.weight[:, n:] = 0
        dl.linear1.weight[:ff, :n].copy_(sl.linear1.weight)
        dl.linear1.bias[:ff].copy_(sl.linear1.bias)
        head_writer(sl.linear2, dl.linear2)
    s_ln, s_l1, _, s_l2 = src.scorer
    d_ln, d_l1, _, d_l2 = dst.scorer
    _set_ln(d_ln, s_ln.weight, n, r, s_ln.eps * n / N, bias=s_ln.bias)
    d_l1.weight[:, n:] = 0
    d_l1.weight[:n].zero_()
    d_l1.weight[:n, :n].copy_(s_l1.weight)
    d_l1.bias[:n].copy_(s_l1.bias)
    d_l2.weight.zero_()
    d_l2.weight[:, :n].copy_(s_l2.weight)
    d_l2.bias.copy_(s_l2.bias)
    if src.answer_head is not None:
        a_ln, a_l1, _, a_l2 = src.answer_head
        b_ln, b_l1, _, b_l2 = dst.answer_head
        b_ln.weight.zero_()
        b_ln.bias.zero_()
        for half in range(2):
            b_ln.weight[half * N:half * N + n].copy_(a_ln.weight[half * n:(half + 1) * n].double() * r)
            b_ln.bias[half * N:half * N + n].copy_(a_ln.bias[half * n:(half + 1) * n])
        b_ln.eps = a_ln.eps * n / N
        b_l1.weight[:n].zero_()
        for half in range(2):
            b_l1.weight[:, half * N + n:(half + 1) * N] = 0
            b_l1.weight[:n, half * N:half * N + n].copy_(a_l1.weight[:, half * n:(half + 1) * n])
        b_l1.bias[:n].copy_(a_l1.bias)
        b_l2.weight.zero_()
        b_l2.weight[:, :n].copy_(a_l2.weight)
        b_l2.bias.copy_(a_l2.bias)
    rep = {"n": n, "N": N, "pads": m, "carriers_layer0_mean": 2, "absorbers": 2, "carriers_final_norm_mean": m2,
           "lambda": lam, "kappa": kappa, "norm_eps_encoder": eps_new, "embedding_ln_gain": G,
           "old_heads": se.config.num_attention_heads, "new_heads": spec.encoder.num_heads, "ffn": [i_old, i_new],
           "note": "LayerNorm eps values are module attributes set by this function (eps * n/N); a serving spec for a "
                   "width-grown model must carry them"}
    return dst, rep
