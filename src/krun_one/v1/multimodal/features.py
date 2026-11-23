"""Modality feature containers: what every modality encoder returns, before projection into the Krun space."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

MODALITY_IDS = {"text": 0, "image": 1, "document": 2, "audio": 3, "transcript": 4}


@dataclass
class ModalityFeatures:
    """``embeddings`` [B, N, D_src] in the tower's own width; ``mask`` [B, N] (True = real)."""
    modality: str
    embeddings: torch.Tensor
    mask: torch.Tensor
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.modality not in MODALITY_IDS:
            raise ValueError(f"unknown modality {self.modality!r}")
        if self.embeddings.dim() != 3 or self.mask.shape != self.embeddings.shape[:2]:
            raise ValueError(f"bad shapes {tuple(self.embeddings.shape)} / {tuple(self.mask.shape)}")

    @property
    def dim(self) -> int:
        return self.embeddings.size(-1)


class TextFeatures(ModalityFeatures):
    def __init__(self, embeddings, mask, meta=None):
        super().__init__("text", embeddings, mask, meta or {})


class VisionFeatures(ModalityFeatures):
    def __init__(self, embeddings, mask, meta=None):
        super().__init__("image", embeddings, mask, meta or {})


class DocumentFeatures(ModalityFeatures):
    """Visual document pages (the text fast path never produces features: it becomes context text)."""
    def __init__(self, embeddings, mask, meta=None):
        super().__init__("document", embeddings, mask, meta or {})


class AudioFeatures(ModalityFeatures):
    def __init__(self, embeddings, mask, meta=None):
        super().__init__("audio", embeddings, mask, meta or {})
