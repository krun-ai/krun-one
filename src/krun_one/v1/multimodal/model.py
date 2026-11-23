"""KrunOneV1Multimodal: the text decision model + optional, lazily loaded modality towers and a lightweight fusion.

    text-only request      -> self.core(...)                                  (exactly the validated text path)
    document, fast path    -> extracted text joins the context -> self.core   (no vision tower)
    image / visual document-> vision tower (lazy) -> adapter -> tokens --\\
    audio                  -> speech tower (lazy) -> adapter -> tokens ---+-> self.core(..., extra_tokens, extra_mask)
    (optional) transcript  -> appended to the context text               --/

Checkpoints: the text core keeps its own state dict (a text-only artifact loads unchanged). Fusion adapters are
saved separately (``multimodal.safetensors``) and tower weights come from their own model ids — nothing multimodal
is required to load or serve a text-only artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch import nn

from ..model import KrunOneV1Model
from .audio import AudioInput, TranscriptProvider, batch_waveforms, transcripts_for
from .config import MultimodalConfig
from .documents import DocumentInput, DocumentRouter
from .features import ModalityFeatures
from .fusion import LightweightFusion, expand_to_sequences
from .towers import LazyTower


@dataclass
class RequestMedia:
    """Media attached to ONE request (all of its questions/chunks see the same media)."""
    images: list[torch.Tensor] = field(default_factory=list)          # preprocessed [3, H, W]
    documents: list[DocumentInput] = field(default_factory=list)
    audio: list[AudioInput] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.images or self.documents or self.audio)


def group_by_request(emb: torch.Tensor, mask: torch.Tensor, owner: Sequence[int], n_requests: int
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Item features [K, N, D] owned by requests -> [R, max_i sum N, D] (a request's items concatenated) + mask."""
    per = [[] for _ in range(n_requests)]
    for k, r in enumerate(owner):
        per[r].append(emb[k][mask[k]])
    lens = [sum(x.size(0) for x in p) for p in per]
    L = max(1, max(lens))
    out = emb.new_zeros(n_requests, L, emb.size(-1))
    m = torch.zeros(n_requests, L, dtype=torch.bool, device=emb.device)
    for r, p in enumerate(per):
        if p:
            x = torch.cat(p, 0)
            out[r, : x.size(0)] = x
            m[r, : x.size(0)] = True
    return out, m


class KrunOneV1Multimodal(nn.Module):
    def __init__(self, core: KrunOneV1Model, mm: MultimodalConfig | None = None,
                 transcript_provider: TranscriptProvider | None = None, tower_kwargs: dict | None = None):
        super().__init__()
        self.core = core
        self.mm = mm or MultimodalConfig()
        d = core.encoder.config.hidden_size
        for sub in (self.mm.vision, self.mm.audio):
            if sub.projector_dim != d:
                raise ValueError(f"projector_dim {sub.projector_dim} != Krun hidden size {d}")
        # fusion parameters exist only if some modality is enabled: a text-only model has zero extra parameters
        self.fusion = (LightweightFusion(d, self.mm.vision.tokens, self.mm.fusion.pooler_heads, self.mm.fusion.share_document_adapter)
                       if self.mm.any_enabled else None)
        kw = tower_kwargs or {}
        # towers are NOT nn submodules: not in state_dict, not moved with .to(), built only on first use
        self._vision = LazyTower(self.mm.vision.model, "image", **kw) if self.mm.needs_vision_tower else None
        self._audio = LazyTower(self.mm.audio.model, "audio", **kw) if self.mm.audio.enabled else None
        self.router = DocumentRouter(self.mm.document)
        self.transcripts = transcript_provider

    # ------------------------------------------------------------------ status
    @property
    def towers_loaded(self) -> dict[str, bool]:
        return {"vision": bool(self._vision and self._vision.loaded), "audio": bool(self._audio and self._audio.loaded)}

    # ------------------------------------------------------------------ pre-processing (CPU side, per request)
    def prepare(self, contexts: list[str], media: Sequence[RequestMedia | None]) -> tuple[list[str], list[dict]]:
        """Route documents and transcripts into the TEXT context when possible. Returns new contexts and, per request,
        the leftover visual/audio items that need towers ({"visual": [(tensor, modality)], "audio": [AudioInput]})."""
        out_ctx, leftovers = [], []
        for ctx, m in zip(contexts, media):
            left = {"visual": [], "audio": []}
            if m is None or m.empty:
                out_ctx.append(ctx)
                leftovers.append(left)
                continue
            extra_text = []
            for doc in m.documents:
                if not self.mm.document.enabled:
                    raise ValueError("document received but multimodal.document.enabled is false")
                r = self.router.route(doc)
                if r.route == "text":
                    extra_text.append(r.text)
                else:
                    left["visual"] += [(p, "document") for p in r.pages]
            for img in m.images:
                if not self.mm.vision.enabled:
                    raise ValueError("image received but multimodal.vision.enabled is false")
                left["visual"].append((img, "image"))
            if m.audio:
                if not self.mm.audio.enabled:
                    raise ValueError("audio received but multimodal.audio.enabled is false")
                left["audio"] += list(m.audio)
                extra_text += [f"[transcript] {t}" for t in transcripts_for(list(m.audio), self.transcripts) if t]
            out_ctx.append("\n".join([ctx] + extra_text) if extra_text else ctx)
            leftovers.append(left)
        return out_ctx, leftovers

    # ------------------------------------------------------------------ towers + fusion (per batch of requests)
    def encode_media(self, leftovers: list[dict]) -> tuple[torch.Tensor, torch.Tensor] | None:
        R = len(leftovers)
        feats: list[ModalityFeatures] = []
        for modality in ("image", "document"):
            items = [(r, x) for r, lo in enumerate(leftovers) for x, mod in lo["visual"] if mod == modality]
            if items:
                tower = self._vision.get()
                f = tower.encode(torch.stack([x for _, x in items]))
                emb, m = group_by_request(f.embeddings, f.mask, [r for r, _ in items], R)
                self.fusion.ensure_adapter(modality, f.dim, self.mm.vision.tokens)
                feats.append(ModalityFeatures(modality, emb, m))
        au = [(r, a) for r, lo in enumerate(leftovers) for a in lo["audio"]]
        if au:
            tower = self._audio.get()
            wav, lengths, _ = batch_waveforms([a for _, a in au], self.mm.audio)
            f = tower.encode((wav, lengths))
            emb, m = group_by_request(f.embeddings, f.mask, [r for r, _ in au], R)
            self.fusion.ensure_adapter("audio", f.dim, self.mm.audio.tokens)
            feats.append(ModalityFeatures("audio", emb, m))
        if not feats:
            return None
        dev = next(self.core.parameters()).device
        self.fusion.to(dev)
        return self.fusion([ModalityFeatures(f.modality, f.embeddings.to(dev), f.mask.to(dev)) for f in feats])

    def forward(self, batch: dict[str, torch.Tensor], media_tokens: tuple[torch.Tensor, torch.Tensor] | None = None,
                seq_request: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if media_tokens is None:
            return self.core(**batch)                                  # text-only: the validated path, untouched
        t, m = expand_to_sequences(*media_tokens, seq_request)
        return self.core(**batch, extra_tokens=t, extra_mask=m)

    # ------------------------------------------------------------------ checkpoints
    def multimodal_state_dict(self) -> dict[str, torch.Tensor]:
        return {} if self.fusion is None else {f"fusion.{k}": v for k, v in self.fusion.state_dict().items()}

    def load_text_core(self, state: dict[str, torch.Tensor]) -> None:
        self.core.load_state_dict(state, strict=True)
