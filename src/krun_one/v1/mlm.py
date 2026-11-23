"""Continued MLM pre-training pieces for the HyperCloning scaling experiment (G1).

Corpus layout (built by scripts/v1/prepare_mlm_corpus.py), one directory per language::

    <corpus>/<lang>/train-00000.npy ...   uint32 token ids, documents concatenated, each ending with <eos>
    <corpus>/<lang>/heldout.npy           held-out tokens (never trained on) for per-language MLM loss
    <corpus>/manifest.json                sources, revisions, licenses, token counts, sha256 per shard

Sampling is a pure function of (seed, step): a language is drawn by mixture weight and the next window of that
language's stream is taken, so the state to checkpoint is just {cursor per language, generator state}.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch


class LanguageStream:
    def __init__(self, files: list[Path]):
        self.arrays = [np.load(f, mmap_mode="r") for f in files]
        self.sizes = [len(a) for a in self.arrays]
        self.total = sum(self.sizes)
        self.offsets = np.cumsum([0] + self.sizes)

    def window(self, cursor: int, n: int) -> np.ndarray:
        cursor %= max(1, self.total - n)
        i = int(np.searchsorted(self.offsets, cursor, side="right") - 1)
        a, start = self.arrays[i], cursor - int(self.offsets[i])
        if start + n <= len(a):
            return np.asarray(a[start:start + n])
        return np.concatenate([np.asarray(a[start:]), self.window(int(self.offsets[i + 1]) if i + 1 < len(self.arrays) else 0,
                                                                  n - (len(a) - start))])


class MixtureSampler:
    def __init__(self, corpus: Path, weights: dict[str, float], seq_len: int, seed: int):
        self.langs = sorted(weights)
        w = np.array([weights[k] for k in self.langs], dtype=np.float64)
        self.p = w / w.sum()
        self.streams = {k: LanguageStream(sorted((corpus / k).glob("train-*.npy"))) for k in self.langs}
        self.seq_len = seq_len
        self.gen = np.random.Generator(np.random.PCG64(seed))
        self.cursor = {k: 0 for k in self.langs}
        self.tokens_seen = {k: 0 for k in self.langs}

    def state(self) -> dict:
        return {"gen": self.gen.bit_generator.state, "cursor": dict(self.cursor), "tokens_seen": dict(self.tokens_seen)}

    def load_state(self, st: dict) -> None:
        self.gen.bit_generator.state = st["gen"]
        self.cursor, self.tokens_seen = dict(st["cursor"]), dict(st["tokens_seen"])

    def batch(self, n: int, bos: int, eos: int) -> tuple[np.ndarray, list[str]]:
        body = self.seq_len - 2
        out = np.empty((n, self.seq_len), dtype=np.int64)
        langs = []
        for i in range(n):
            k = self.langs[int(self.gen.choice(len(self.langs), p=self.p))]
            out[i, 0], out[i, -1] = bos, eos
            out[i, 1:-1] = self.streams[k].window(self.cursor[k], body)
            self.cursor[k] += body
            self.tokens_seen[k] += self.seq_len
            langs.append(k)
        return out, langs


def mask_tokens(ids: torch.Tensor, mask_id: int, vocab_size: int, rate: float, special: torch.Tensor,
                gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """BERT masking: `rate` of non-special positions are targets; 80% -> <mask>, 10% random token, 10% unchanged."""
    labels = ids.clone()
    prob = torch.rand(ids.shape, generator=gen)
    target = (prob < rate) & ~torch.isin(ids, special)
    labels[~target] = -100
    r = torch.rand(ids.shape, generator=gen)
    x = ids.clone()
    x[target & (r < 0.8)] = mask_id
    rnd = target & (r >= 0.8) & (r < 0.9)
    x[rnd] = torch.randint(0, vocab_size, (int(rnd.sum()),), generator=gen)
    return x, labels


def enable_sparse_prediction(model) -> None:
    """Run the MLM head (256k-vocab decoder) on masked positions only.

    ModernBertForMaskedLM copies ``config.sparse_prediction`` into ``self.sparse_prediction`` in ``__init__``, so
    flipping the config after ``from_pretrained`` alone does nothing: full logits (16 x 1024 x 256k fp32 = 15.6 GiB in
    the held-out eval) went OOM on the first MLM run. Set both.
    """
    model.config.sparse_prediction = True
    model.sparse_prediction = True


@torch.no_grad()
def heldout_mlm_loss(model, corpus: Path, langs: list[str], seq_len: int, n_seqs: int, bos: int, eos: int,
                     mask_id: int, vocab_size: int, special: torch.Tensor, device, rate: float = 0.15,
                     seed: int = 12345, batch: int = 16) -> dict[str, float]:
    """Mean MLM cross entropy per language on fixed held-out windows with fixed masks (comparable across ckpts)."""
    model.eval()
    res = {}
    for k in langs:
        a = np.load(corpus / k / "heldout.npy", mmap_mode="r")
        body = seq_len - 2
        n = min(n_seqs, max(1, (len(a) - 1) // body))
        g = torch.Generator().manual_seed(seed)
        tot, cnt = 0.0, 0
        for s in range(0, n, batch):
            rows = []
            for j in range(s, min(n, s + batch)):
                w = np.asarray(a[j * body:(j + 1) * body])
                rows.append(np.concatenate([[bos], w, [eos]]))
            ids = torch.tensor(np.stack(rows), dtype=torch.long)
            x, y = mask_tokens(ids, mask_id, vocab_size, rate, special, g)
            n_t = int((y != -100).sum())
            if n_t == 0:
                continue
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                # labels -> sparse decoder on masked positions only (full 256k-vocab logits would not fit in 24 GB)
                loss = model(input_ids=x.to(device), labels=y.to(device)).loss
            tot += float(loss) * n_t
            cnt += n_t
        res[k] = tot / max(1, cnt)
    model.train()
    return res


def milestone_reached(prev_tokens: int, tokens: int, milestones: list[int]) -> list[int]:
    return [m for m in milestones if prev_tokens < m <= tokens]


def wsd_lr(tokens: int, total: int, warmup: int, decay_frac: float) -> float:
    """Warmup-stable-decay multiplier on the token axis (decay only at the end of the *planned* run)."""
    if tokens < warmup:
        return (tokens + 1) / max(1, warmup)
    d0 = total * (1 - decay_frac)
    if tokens < d0:
        return 1.0
    return max(0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, (tokens - d0) / max(1, total - d0)))))


def write_manifest(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2))
