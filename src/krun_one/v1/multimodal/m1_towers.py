"""M1 towers (frozen): SigLIP2-base NaFlex (images + visual documents) and the Whisper-small ENCODER (audio).

Both run under no_grad in bf16 and return (features [B, N, D], mask [B, N]).

* SigLIP2 NaFlex keeps the aspect ratio; the token budget is ``max_num_patches`` per item (images 256, document
  pages / screenshots 1024: text needs resolution). Output = last_hidden_state (post-layernorm patch tokens).
* Whisper encoder, truncated: the stock HF encoder insists on 30 s (3000 mel frames). We run conv1/conv2 + the first
  T/2 positional embeddings + the layers on the real length only (rounded up to 1 s), like whisper.cpp's audio_ctx.
  A 5 s clip costs 250 encoder frames instead of 1500. Training and serving use the same truncation.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

SIGLIP2 = "google/siglip2-base-patch16-naflex"
WHISPER = "openai/whisper-small"
PATCHES = {"image": 256, "document": 1024}


class SigLIP2Tower(nn.Module):
    def __init__(self, repo: str = SIGLIP2, revision: str | None = None, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        from transformers import AutoModel

        m = AutoModel.from_pretrained(repo, revision=revision, dtype=dtype)
        self.model = m.vision_model.to(device).eval()
        self.model.requires_grad_(False)
        self.proc = image_preprocessor(repo, revision)
        self.output_dim = self.model.config.hidden_size
        self.device, self.dtype = device, dtype

    def preprocess(self, images, max_num_patches: int) -> dict:
        """PIL images -> CPU tensors (DataLoader workers call this)."""
        return dict(self.proc(images=images, max_num_patches=max_num_patches, return_tensors="pt"))

    @torch.no_grad()
    def encode(self, inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        pv = inputs["pixel_values"].to(self.device, self.dtype, non_blocking=True)
        pm = inputs["pixel_attention_mask"].to(self.device, non_blocking=True)
        ss = inputs["spatial_shapes"].to(self.device, non_blocking=True)
        h = self.model(pixel_values=pv, pixel_attention_mask=pm, spatial_shapes=ss).last_hidden_state
        return h, pm.bool()


def image_preprocessor(repo: str = SIGLIP2, revision: str | None = None):
    from transformers.models.siglip2.image_processing_pil_siglip2 import Siglip2ImageProcessorPil   # no torchvision dep

    return Siglip2ImageProcessorPil.from_pretrained(repo, revision=revision)


class WhisperEncoderTower(nn.Module):
    def __init__(self, repo: str = WHISPER, revision: str | None = None, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        from transformers import WhisperModel

        self.enc = WhisperModel.from_pretrained(repo, revision=revision, dtype=dtype).encoder.to(device).eval()
        self.enc.requires_grad_(False)
        self.output_dim = self.enc.config.d_model
        self.device, self.dtype = device, dtype

    @torch.no_grad()
    def encode(self, feats: torch.Tensor, n_frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """feats [B, 80, T] log-mel (T <= 3000, even); n_frames [B] valid mel frames -> ([B, T/2, D], mask)."""
        e = self.enc
        x = feats.to(self.device, self.dtype, non_blocking=True)
        x = F.gelu(e.conv1(x))
        x = F.gelu(e.conv2(x)).permute(0, 2, 1)
        L = x.size(1)
        x = x + e.embed_positions.weight[:L].to(x.dtype)
        for layer in e.layers:
            x = layer(x, None)
            x = x[0] if isinstance(x, tuple) else x
        x = e.layer_norm(x)
        mask = torch.arange(L, device=x.device)[None, :] < ((n_frames.to(x.device) + 1) // 2)[:, None]
        return x, mask


class AudioFrontend:
    """waveform (16 kHz mono float32) -> Whisper log-mel, truncated to the clip length rounded up to 1 s."""

    def __init__(self, repo: str = WHISPER, max_seconds: float = 30.0):
        from transformers import WhisperFeatureExtractor

        self.fe = WhisperFeatureExtractor.from_pretrained(repo)
        self.max_seconds = max_seconds

    def __call__(self, wavs: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        sr = 16000
        wavs = [w[: int(self.max_seconds * sr)] for w in wavs]
        frames = [max(1, math.ceil(len(w) / 160)) for w in wavs]
        T = min(3000, int(math.ceil(max(frames) / 100.0) * 100))              # round up to whole seconds
        f = self.fe(wavs, sampling_rate=sr, return_tensors="pt").input_features[:, :, :T]   # padded to 30 s, cut
        return f, torch.tensor([min(n, T) for n in frames])


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    import soundfile as sf

    w, sr = sf.read(path, dtype="float32", always_2d=False)
    if w.ndim > 1:
        w = w.mean(1)
    if sr != target_sr:
        n = int(round(len(w) * target_sr / sr))
        w = np.interp(np.linspace(0, len(w) - 1, n), np.arange(len(w)), w).astype(np.float32)
    return w
