"""Modality encoders ("towers"): a small protocol, a registry, lazy loading, and a mock for tests.

Nothing here is imported or built on the text path. A tower is created on the first request that needs it
(``LazyTower.get()``), so a text-only worker has no extra weights in memory and no extra cold start.

Tower ids (config ``model``):
  "mock"                     tiny deterministic random encoder (unit/smoke tests, CPU)
  "hf-vision:<repo>[@rev]"   HF vision encoder returning patch tokens (SigLIP/SigLIP2-like, CLIP-like, ...)
  "hf-speech:<repo>[@rev]"   HF speech encoder returning frame features (Whisper-encoder-like, w2v-BERT-like, ...)
The HF adapters are thin and generic on purpose: no Krun code depends on a specific vision/audio model.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Protocol

import torch
from torch import nn

from .features import AudioFeatures, DocumentFeatures, ModalityFeatures, VisionFeatures


class ModalityEncoder(Protocol):
    modality: str
    output_dim: int

    def encode(self, inputs: Any) -> ModalityFeatures: ...


# ------------------------------------------------------------------------------------------------ mock
class MockVisionEncoder(nn.Module):
    """images [B, 3, H, W] float -> patch tokens [B, (H/16)*(W/16), 64]. Deterministic (seeded), tiny."""
    modality = "image"
    output_dim = 64

    def __init__(self, patch: int = 16, modality: str = "image"):
        super().__init__()
        torch.manual_seed(0)
        self.patch = nn.Conv2d(3, self.output_dim, patch, patch)
        self.modality = modality

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> ModalityFeatures:
        x = self.patch(images.float()).flatten(2).transpose(1, 2)
        m = torch.ones(x.shape[:2], dtype=torch.bool)
        cls = DocumentFeatures if self.modality == "document" else VisionFeatures
        return cls(x, m, {"encoder": "mock"})


class MockSpeechEncoder(nn.Module):
    """waveforms [B, T] (+ lengths) -> frame features [B, T/320, 48] (20 ms frames at 16 kHz)."""
    modality = "audio"
    output_dim = 48

    def __init__(self, hop: int = 320):
        super().__init__()
        torch.manual_seed(0)
        self.hop = hop
        self.proj = nn.Linear(hop, self.output_dim)

    @torch.no_grad()
    def encode(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> ModalityFeatures:
        wav, lengths = inputs
        B, T = wav.shape
        n = T // self.hop
        x = self.proj(wav[:, : n * self.hop].reshape(B, n, self.hop).float())
        m = torch.arange(n)[None, :] < (lengths // self.hop)[:, None]
        return AudioFeatures(x, m, {"encoder": "mock"})


# ------------------------------------------------------------------------------------------------ HF adapters
class HFVisionEncoder(nn.Module):
    """Generic HF vision tower: last_hidden_state patch tokens. Frozen by default (inference mode)."""
    modality = "image"

    def __init__(self, repo: str, revision: str | None = None, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        from transformers import AutoModel

        m = AutoModel.from_pretrained(repo, revision=revision, dtype=dtype)
        self.model = getattr(m, "vision_model", m).to(device).eval()   # dual encoders (SigLIP/CLIP): keep vision half
        self.output_dim = self.model.config.hidden_size
        self.device, self.dtype = device, dtype

    @torch.no_grad()
    def encode(self, pixel_values: torch.Tensor) -> ModalityFeatures:
        h = self.model(pixel_values=pixel_values.to(self.device, self.dtype)).last_hidden_state
        return VisionFeatures(h, torch.ones(h.shape[:2], dtype=torch.bool, device=h.device), {"encoder": "hf"})


class HFSpeechEncoder(nn.Module):
    """Generic HF speech encoder (encoder half only; never a generative decoder)."""
    modality = "audio"

    def __init__(self, repo: str, revision: str | None = None, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        from transformers import AutoModel

        m = AutoModel.from_pretrained(repo, revision=revision, dtype=dtype)
        self.model = (m.get_encoder() if hasattr(m, "get_encoder") else m).to(device).eval()
        self.output_dim = self.model.config.hidden_size if hasattr(self.model.config, "hidden_size") else self.model.config.d_model
        self.device, self.dtype = device, dtype

    @torch.no_grad()
    def encode(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> ModalityFeatures:
        feats, mask = inputs    # model-specific input features (e.g. log-mel) prepared by the model's processor
        h = self.model(feats.to(self.device, self.dtype)).last_hidden_state
        return AudioFeatures(h, mask.to(h.device)[:, : h.size(1)] if mask.size(1) >= h.size(1) else
                             torch.ones(h.shape[:2], dtype=torch.bool, device=h.device), {"encoder": "hf"})


def _parse(model_id: str) -> tuple[str, str, str | None]:
    kind, _, rest = model_id.partition(":")
    repo, _, rev = rest.partition("@")
    return kind, repo, rev or None


REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "mock-vision": lambda **kw: MockVisionEncoder(modality=kw.get("modality", "image")),
    "mock-speech": lambda **kw: MockSpeechEncoder(),
}


def build_tower(model_id: str, modality: str, **kw) -> nn.Module:
    if model_id == "mock":
        return REGISTRY["mock-speech" if modality == "audio" else "mock-vision"](modality=modality)
    kind, repo, rev = _parse(model_id)
    if kind == "hf-vision":
        return HFVisionEncoder(repo, rev, **kw)
    if kind == "hf-speech":
        return HFSpeechEncoder(repo, rev, **kw)
    if model_id in REGISTRY:
        return REGISTRY[model_id](modality=modality)
    raise ValueError(f"unknown tower id {model_id!r}")


class LazyTower:
    """Builds/loads the tower on first use (thread-safe). ``loaded`` stays False on text-only traffic."""

    def __init__(self, model_id: str | None, modality: str, **kw):
        self.model_id, self.modality, self.kw = model_id, modality, kw
        self._tower: nn.Module | None = None
        self._lock = threading.Lock()
        self.load_seconds: float | None = None

    @property
    def loaded(self) -> bool:
        return self._tower is not None

    def get(self) -> nn.Module:
        if self._tower is None:
            with self._lock:
                if self._tower is None:
                    if not self.model_id:
                        raise RuntimeError(f"{self.modality} tower requested but no model configured")
                    t0 = time.perf_counter()
                    self._tower = build_tower(self.model_id, self.modality, **self.kw)
                    self.load_seconds = time.perf_counter() - t0
        return self._tower
