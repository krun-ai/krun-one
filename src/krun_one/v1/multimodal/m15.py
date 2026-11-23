"""M1.5 hybrid fusion: the frozen text decision on the transcript / OCR text, corrected by the M1 media fusion through
a learned gate (configs/v1/m15.yaml).

    h        = frozen text encoder([question + candidates + context=transcript|OCR text]) + type_emb   (one pass)
    z_text   = frozen head(h)                         -> scorer                        (the text representation)
    z_fused  = frozen head + M1 LoRA([h ; media])     -> scorer                        (M1 adapters: acoustic / visual)
    alpha    = sigmoid(MLP_g([h_CLS ; mean(media)]))  per sequence, g = audio | visual
    z        = z_text + alpha * (z_fused - z_text)    for rank / apply / null

alpha -> 0 recovers the transcript/OCR-only decision exactly; the media branch contributes only where it helps.
Text-only requests never enter this module (core(**batch), the frozen artifact's own forward).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from ..model import NEG
from .m1 import M1Fusion, M1Model, head_layer_forward

GATE_GROUP = {"audio": "audio", "image": "visual", "document": "visual"}


class Gate(nn.Module):
    def __init__(self, d: int, hidden: int = 256, bias: float = -1.0):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, bias)

    def forward(self, cls: torch.Tensor, media: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.to(media.dtype)[..., None]
        pooled = (media * m).sum(1) / m.sum(1).clamp(min=1)
        return torch.sigmoid(self.net(torch.cat([cls, pooled.to(cls.dtype)], -1).float()))[:, 0]


class M15Model(nn.Module):
    def __init__(self, m1: M1Model, gate_bias: float = -1.0, gate: bool = True):
        """gate=False (M1.5b, REMOVE_GATE): the decision is z_fused directly; no gate parameters, no z_text head pass."""
        super().__init__()
        self.m1 = m1
        d = m1.core.encoder.config.hidden_size
        self.gates = nn.ModuleDict({"audio": Gate(d, bias=gate_bias), "visual": Gate(d, bias=gate_bias)} if gate else {})

    @property
    def core(self):
        return self.m1.core

    @property
    def fusion(self) -> M1Fusion:
        return self.m1.fusion

    def trainable_state_dict(self) -> dict:
        sd = {f"fusion.{k}": v for k, v in self.fusion.state_dict().items()}
        sd.update({f"gates.{k}": v for k, v in self.gates.state_dict().items()})
        return sd

    def _scores(self, h, pad, batch, lora_layers):
        c = self.core
        for i, layer in enumerate(c.head.layers):
            h = head_layer_forward(layer, lora_layers[i] if lora_layers is not None else None, h, pad)
        d = h.size(-1)
        mk = torch.gather(h, 1, batch["marker_pos"].clamp(min=0)[:, :, None].expand(-1, -1, d))
        s = c.scorer(mk).float()
        out = {"rank": s[..., 0].masked_fill(~batch["marker_mask"], NEG), "apply": s[..., 1].masked_fill(~batch["marker_mask"], NEG)}
        null_pos = batch["null_pos"]
        has_null = null_pos >= 0
        hn = torch.gather(h, 1, null_pos.clamp(min=0)[:, None, None].expand(-1, 1, d))[:, 0]
        out["null_rank"] = c.scorer(hn).float()[:, 0].masked_fill(~has_null, NEG)
        out["has_null"] = has_null
        return out

    def forward(self, batch: dict, media: tuple[torch.Tensor, torch.Tensor], seq_request: Sequence[int], modality: str,
                mode: str = "gated") -> dict[str, torch.Tensor]:
        """mode: gated (M1.5) | fused (alpha = 1, diagnostic) | text (alpha = 0: transcript/OCR only, functional head)."""
        c = self.core
        with torch.no_grad():
            h = c.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
            h = h + c.type_emb(batch["qtype"])[:, None, :].to(h.dtype)
        att = batch["attention_mask"]
        if not self.gates and mode == "gated":
            mode = "fused"                                   # REMOVE_GATE: the fused path is the decision
        if mode != "fused":
            with torch.no_grad():
                zt = self._scores(h, ~att.bool(), batch, None)
        if mode == "text":
            zt["alpha"] = torch.zeros(h.size(0), device=h.device)
            return zt
        idx = torch.as_tensor(list(seq_request), device=h.device)
        t, m = media[0].index_select(0, idx), media[1].index_select(0, idx)
        hf = torch.cat([h, t.to(h.dtype)], 1)
        attf = torch.cat([att, m.to(att.dtype)], 1)
        lora = self.fusion.lora.layers if self.fusion.lora is not None else None
        zf = self._scores(hf, ~attf.bool(), batch, lora)
        if mode == "fused":
            zf["alpha"] = torch.ones(h.size(0), device=h.device)
            return zf
        a = self.gates[GATE_GROUP[modality]](h[:, 0], t, m)
        out = {"has_null": zf["has_null"], "alpha": a}
        for k in ("rank", "apply"):
            mask = batch["marker_mask"]
            out[k] = (zt[k] + a[:, None] * (zf[k] - zt[k])).masked_fill(~mask, NEG)
        out["null_rank"] = (zt["null_rank"] + a * (zf["null_rank"] - zt["null_rank"])).masked_fill(~zf["has_null"], NEG)
        return out
