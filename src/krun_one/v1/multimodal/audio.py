"""Audio: waveform -> speech ENCODER features -> projector -> decision (no generative speech model).

Two complementary routes, combinable without touching the Krun core:
  * direct speech features  (tone, emotion, language, intent cues)   -> modality "audio" tokens
  * optional transcript     (TranscriptProvider, e.g. an ASR model)   -> appended to the text context; the transcript
                                                                         can also be fed as "transcript" tokens later
No ASR model is integrated or trained in this round: ``TranscriptProvider`` is an interface only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .config import AudioConfig


@dataclass
class AudioInput:
    waveform: torch.Tensor           # [T] mono float32
    sample_rate: int
    language: str | None = None      # BCP-47 hint (en, pt-BR, es-419, fr-CA, ...)
    transcript: str | None = None    # caller-provided transcript (skips ASR)


class TranscriptProvider(Protocol):
    def transcribe(self, audio: AudioInput) -> str: ...


def resample_linear(w: torch.Tensor, sr: int, target: int) -> torch.Tensor:
    """Dependency-free resampling (good enough for the scaffold/tests; a real front-end should use torchaudio)."""
    if sr == target:
        return w
    n = int(round(w.numel() * target / sr))
    return torch.nn.functional.interpolate(w[None, None].float(), size=n, mode="linear", align_corners=False)[0, 0]


def batch_waveforms(items: list[AudioInput], cfg: AudioConfig) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """-> padded [K, T_max] at cfg.sample_rate, lengths [K], per-item meta (duration, language, clipped)."""
    ws, meta = [], []
    max_len = int(cfg.max_seconds * cfg.sample_rate)
    for a in items:
        w = resample_linear(a.waveform.float().flatten(), a.sample_rate, cfg.sample_rate)
        meta.append({"duration_s": w.numel() / cfg.sample_rate, "language": a.language, "clipped": w.numel() > max_len})
        ws.append(w[:max_len])
    T = max(w.numel() for w in ws)
    out = torch.zeros(len(ws), T)
    for i, w in enumerate(ws):
        out[i, : w.numel()] = w
    return out, torch.tensor([w.numel() for w in ws]), meta


def transcripts_for(items: list[AudioInput], provider: TranscriptProvider | None) -> list[str | None]:
    return [a.transcript if a.transcript is not None else (provider.transcribe(a) if provider else None) for a in items]
