"""M1.5 audio front: Whisper-small (frozen, as released) for BOTH the transcript and the acoustic features.

    wav (16 kHz) -> log-mel 30 s padded (GPU)  -> Whisper encoder (ONE pass, [B, 1500, 768])
                                                  |-> decoder, greedy, language auto-detected -> transcript -> Krun text path
                                                  '-> valid frames [:ceil(n/2)] -> M1 audio adapter -> acoustic tokens

Every stage is separately timeable (m15_serving_checks.py). Streaming (not implemented): the same encoder / decoder
would run on growing chunks with partial transcripts feeding incremental Krun decisions; the request schema carries a
caller transcript already, so a streaming client can send partial transcripts without API changes.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutput

WHISPER = "openai/whisper-small"


class WhisperASR:
    def __init__(self, repo: str = WHISPER, device="cuda", dtype=torch.bfloat16, max_new_tokens: int = 96):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.proc = WhisperProcessor.from_pretrained(repo)
        self.model = WhisperForConditionalGeneration.from_pretrained(repo, dtype=dtype).to(device).eval()
        self.model.requires_grad_(False)
        self.device, self.dtype, self.max_new_tokens = device, dtype, max_new_tokens
        self.output_dim = self.model.config.d_model
        self.lang_tokens = {self.proc.tokenizer.convert_tokens_to_ids(f"<|{l}|>"): l for l in ("en", "pt", "es", "fr")}

    def logmel(self, wavs: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        """-> [B, 80, 3000] log-mel (30 s padded, computed on the GPU) and valid mel frames per clip."""
        wavs = [w[: 30 * 16000] for w in wavs]
        f = self.proc.feature_extractor(wavs, sampling_rate=16000, return_tensors="pt",
                                        device=self.device if self.device != "cpu" else "cpu").input_features
        n = torch.tensor([max(1, math.ceil(len(w) / 160)) for w in wavs])
        return f, n

    @torch.inference_mode()
    def encode(self, feats: torch.Tensor) -> torch.Tensor:
        return self.model.model.encoder(feats.to(self.device, self.dtype)).last_hidden_state

    @torch.inference_mode()
    def transcribe(self, enc: torch.Tensor, language: str | None = None) -> list[str]:
        kw = {"task": "transcribe", "max_new_tokens": self.max_new_tokens, "do_sample": False, "num_beams": 1}
        if language:
            kw["language"] = language
        ids = self.model.generate(encoder_outputs=BaseModelOutput(last_hidden_state=enc), **kw)
        return [t.strip() for t in self.proc.batch_decode(ids, skip_special_tokens=True)]

    @staticmethod
    def acoustic(enc: torch.Tensor, n_frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        L = int(min(enc.size(1), math.ceil(int(n_frames.max()) / 2)))
        x = enc[:, :L]
        mask = torch.arange(L, device=enc.device)[None, :] < ((n_frames.to(enc.device) + 1) // 2)[:, None]
        return x, mask
