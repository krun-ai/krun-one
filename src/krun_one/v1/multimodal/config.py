"""Multimodal configuration (all modalities DISABLED by default: a text-only deployment never builds or loads a tower).

YAML shape (configs/v1/multimodal.yaml)::

    multimodal:
      vision:   {enabled: false, model: null, projector_dim: 1536, tokens: 32, image_size: 384}
      document: {enabled: false, text_fast_path: true, visual_fallback: true, min_chars: 200, max_pages: 4}
      audio:    {enabled: false, model: null, projector_dim: 1536, tokens: 32, sample_rate: 16000,
                 max_seconds: 30, asr: null}
      fusion:   {type: lightweight, pooler_heads: 8}

``model`` is a tower id resolved by towers.py: ``"mock"`` (tiny random encoder, tests), or ``"hf:<repo>@<rev>"``
(HF vision / speech encoder, e.g. a SigLIP-like vision model or a multilingual speech encoder). Documents share the
vision tower (a visual page is an image); their fast path never touches it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VisionConfig:
    enabled: bool = False
    model: str | None = None
    projector_dim: int = 1536
    tokens: int = 32              # learned-query pooling: patches -> this many tokens (latency control)
    image_size: int = 384


@dataclass
class DocumentConfig:
    enabled: bool = False
    text_fast_path: bool = True   # extractable text -> the Krun text path (no vision tower)
    visual_fallback: bool = True  # scans / receipts / forms / layout-sensitive pages -> vision tower
    min_chars: int = 200          # below this much extracted text a page is treated as visual
    max_pages: int = 4


@dataclass
class AudioConfig:
    enabled: bool = False
    model: str | None = None
    projector_dim: int = 1536
    tokens: int = 32
    sample_rate: int = 16_000
    max_seconds: float = 30.0
    asr: str | None = None        # optional transcript provider (future); transcript joins the text path


@dataclass
class FusionConfig:
    type: str = "lightweight"     # projector + learned-query pooler + modality embedding -> decision head
    pooler_heads: int = 8
    share_document_adapter: bool = True   # visual documents reuse the image projector/pooler (distinct modality emb)


@dataclass
class MultimodalConfig:
    vision: VisionConfig = field(default_factory=VisionConfig)
    document: DocumentConfig = field(default_factory=DocumentConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)

    @property
    def any_enabled(self) -> bool:
        return self.vision.enabled or self.audio.enabled or (self.document.enabled and self.document.visual_fallback)

    @property
    def needs_vision_tower(self) -> bool:
        return self.vision.enabled or (self.document.enabled and self.document.visual_fallback)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "MultimodalConfig":
        d = (d or {}).get("multimodal", d or {})
        return cls(vision=VisionConfig(**(d.get("vision") or {})), document=DocumentConfig(**(d.get("document") or {})),
                   audio=AudioConfig(**(d.get("audio") or {})), fusion=FusionConfig(**(d.get("fusion") or {})))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MultimodalConfig":
        import yaml

        return cls.from_dict(yaml.safe_load(Path(path).read_text()))
