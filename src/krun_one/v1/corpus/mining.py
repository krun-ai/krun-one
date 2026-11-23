"""Hard-negative mining v2 for Krun Decisions v1 (train-side pools only; see docs/KRUN_DECISIONS_V1_SPEC.md §4).

Intent labels (per catalog), fused by RRF (k=10):
  1. label-name similarity      char-TF-IDF of humanized label names
  2. label-centroid similarity  multilingual-e5-small centroids of EN train examples (gold ↔ candidate)
  3. confusion history          kNN (k=10) leave-one-group-out over EN train embeddings: share of g's neighbours with label c
  4. example level              cos(context, centroid(c)), cross-lingual (e5 is multilingual: PT context vs EN centroid)
Tools (pool = every train tool, official name + description), fused by RRF:
  1. query ↔ tool doc (e5)       2. gold doc ↔ tool doc (e5)       3. name char-TF-IDF (on the shortlist)
Safety: candidates that may be correct are never negatives — interchangeable labels (centroid cos >= 0.95), tools with
the same humanized name / same description / doc cos >= 0.95 as the gold. Tools with doc cos in [0.85, 0.95) are
kept and mark the decision ``near_duplicate`` (description-dependent routing).
Online confusion mining from a model checkpoint is supported by ``add_confusions`` (confusion history table), used only
with train-side predictions.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

RRF_K = 10


def humanize(x: str) -> str:
    return " ".join(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", x).replace("_", " ").replace("-", " ").replace("/", " ").split()).lower()


def rng_for(*parts) -> np.random.Generator:
    h = hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(h[:8], "little")))


def ranks(scores: np.ndarray) -> np.ndarray:
    """0-based rank per column along the last axis, highest score first."""
    return np.argsort(np.argsort(-scores, axis=-1, kind="stable"), axis=-1, kind="stable")


class EmbeddingCache:
    """e5 embeddings keyed by exact text (npz on disk). Missing texts are embedded in one batch."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.index: dict[str, int] = {}
        self.mat = np.zeros((0, 384), dtype=np.float16)
        if self.path.exists():
            z = np.load(self.path, allow_pickle=False)
            self.mat = z["emb"]
            self.index = {t: i for i, t in enumerate(z["texts"].tolist())}

    def ensure(self, texts: Sequence[str], batch_size: int = 256) -> None:
        from krun_one.mining import embed_texts

        todo = sorted({t for t in texts if t not in self.index})
        if not todo:
            return
        e = embed_texts(todo, batch_size=batch_size).astype(np.float16)
        base = len(self.index)
        for i, t in enumerate(todo):
            self.index[t] = base + i
        self.mat = np.concatenate([self.mat, e]) if len(self.mat) else e
        texts_arr = np.array(sorted(self.index, key=self.index.get), dtype=object)
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez(tmp, emb=self.mat, texts=texts_arr.astype(str))
        tmp.rename(self.path)

    def get(self, texts: Sequence[str]) -> np.ndarray:
        return self.mat[[self.index[t] for t in texts]].astype(np.float32)


class LabelSpace:
    def __init__(self, catalog: str, labels: Sequence[str], examples: dict[str, list[tuple[str, str]]], emb: EmbeddingCache):
        """examples: label -> [(text, split_group)] (EN train only)."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.catalog, self.labels = catalog, sorted(labels)
        self.pos = {l: i for i, l in enumerate(self.labels)}
        n = len(self.labels)
        a = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform([humanize(l) for l in self.labels])
        s_name = (a @ a.T).toarray()
        cents = np.zeros((n, 384), dtype=np.float32)
        all_e, all_l, all_g = [], [], []
        for l in self.labels:
            ex = examples.get(l) or []
            if not ex:
                continue
            e = emb.get([t for t, _ in ex])
            c = e.mean(0)
            cents[self.pos[l]] = c / max(np.linalg.norm(c), 1e-9)
            all_e.append(e)
            all_l += [self.pos[l]] * len(ex)
            all_g += [g for _, g in ex]
        self.centroids = cents
        s_cent = cents @ cents.T
        # confusion history: leave-one-group-out kNN over train examples (subsampled to keep it O(n^2)-cheap)
        conf = np.zeros((n, n))
        if all_e:
            E = np.concatenate(all_e)
            L, G = np.array(all_l), np.array(all_g)
            rng = np.random.default_rng(0)
            take = rng.permutation(len(E))[: min(len(E), 20000)]
            E, L, G = E[take], L[take], G[take]
            for s in range(0, len(E), 2048):
                sim = E[s:s + 2048] @ E.T
                same = G[s:s + 2048, None] == G[None, :]
                sim[same] = -9
                nn = np.argpartition(-sim, 10, axis=1)[:, :10]
                for i, row in enumerate(nn):
                    for j in row:
                        conf[L[s + i], L[j]] += 0.1
            conf /= np.maximum(conf.sum(1, keepdims=True), 1e-9)
        self.interchangeable = (s_cent >= 0.95) & ~np.eye(n, dtype=bool)
        self.label_rrf = sum(1.0 / (RRF_K + ranks(m)) for m in (s_name, s_cent, conf))
        self.s_cent, self.conf = s_cent, conf

    def negatives(self, gold: str | None, n: int, ctx_emb: np.ndarray | None, rng: np.random.Generator,
                  hard: bool = True, exclude: Sequence[str] = ()) -> list[str]:
        """n negatives for a decision; gold None -> neighbours of the context (for gold-removed near-OOD the gold is
        passed and excluded)."""
        n = min(n, len(self.labels) - (1 if gold else 0) - len(exclude))
        ban = set(exclude) | ({gold} if gold else set())
        if gold:
            ban |= {self.labels[j] for j in np.flatnonzero(self.interchangeable[self.pos[gold]])}
        pool = [l for l in self.labels if l not in ban]
        if not hard or n <= 0:
            return list(rng.choice(pool, size=max(0, min(n, len(pool))), replace=False))
        idx = np.array([self.pos[l] for l in pool])
        rl = []
        if gold:
            rl.append(ranks(self.label_rrf[self.pos[gold]][idx][None])[0])
        if ctx_emb is not None:
            rl.append(ranks((self.centroids[idx] @ ctx_emb)[None])[0])
        fused = sum(1.0 / (RRF_K + r) for r in rl)
        return sample_top(pool, fused, n, rng)

    def nearest(self, ctx_emb: np.ndarray, k: int = 1) -> list[tuple[str, float]]:
        s = self.centroids @ ctx_emb
        o = np.argsort(-s)[:k]
        return [(self.labels[i], float(s[i])) for i in o]


def sample_top(pool: Sequence[str], fused: np.ndarray, n: int, rng: np.random.Generator) -> list[str]:
    """Sample n from the top 2n of a fused ranking, weights ∝ 1/rank (seeded per decision)."""
    order = np.argsort(-fused, kind="stable")[: max(2 * n, n + 2)]
    w = 1.0 / (1 + np.arange(len(order)))
    pick = rng.choice(len(order), size=min(n, len(order)), replace=False, p=w / w.sum())
    return [pool[order[i]] for i in sorted(pick)]


class ToolPool:
    """All train-side tools (official name + description), e5 doc embeddings, and hard negatives for a query."""

    def __init__(self, tools: Sequence[tuple[str, str]], emb: EmbeddingCache):
        self.tools = sorted(set(tools))
        self.docs = [f"{n}: {d}" for n, d in self.tools]
        self.E = emb.get(self.docs)
        self.pos = {t: i for i, t in enumerate(self.tools)}
        self.hname = [humanize(n) for n, _ in self.tools]
        self.by_desc: dict[str, list[int]] = defaultdict(list)
        for i, (_, d) in enumerate(self.tools):
            self.by_desc[" ".join(d.lower().split())].append(i)

    def doc_emb(self, tool: tuple[str, str]) -> np.ndarray:
        return self.E[self.pos[tool]]

    def negatives(self, gold: tuple[str, str] | None, q_emb: np.ndarray, n: int, rng: np.random.Generator,
                  exclude_names: set[str], hard: bool = True, shortlist: int = 400) -> tuple[list[tuple[str, str]], int]:
        """Returns (negatives, n_near_duplicates). Never returns a tool equivalent to the gold or a name already used."""
        if n <= 0:
            return [], 0
        if not hard:
            idx = rng.permutation(len(self.tools))[: 4 * n + 16]
            out, seen = [], set(exclude_names)
            for i in idx:
                if self.tools[i][0] not in seen:
                    seen.add(self.tools[i][0])
                    out.append(self.tools[i])
                if len(out) == n:
                    break
            return out, 0
        s_q = self.E @ q_emb
        s_g = self.E @ self.doc_emb(gold) if gold and gold in self.pos else None
        base = s_q if s_g is None else 0.5 * (s_q + s_g)
        short = np.argpartition(-base, min(shortlist, len(base) - 1))[:shortlist]
        gname = humanize(gold[0]) if gold else None
        gdesc = " ".join(gold[1].lower().split()) if gold else None
        keep, near = [], []
        for i in short:
            name, desc = self.tools[i]
            if name in exclude_names or (gold and (self.hname[i] == gname or " ".join(desc.lower().split()) == gdesc)):
                continue
            if s_g is not None and s_g[i] >= 0.95:
                continue
            keep.append(i)
            near.append(s_g is not None and s_g[i] >= 0.85)
        if not keep:
            return [], 0
        keep_a = np.array(keep)
        rl = [ranks(s_q[keep_a][None])[0]]
        if s_g is not None:
            rl.append(ranks(s_g[keep_a][None])[0])
        fused = sum(1.0 / (RRF_K + r) for r in rl)
        names_seen: set[str] = set()
        cand = [self.tools[i] for i in keep_a]
        picked = sample_top(cand, fused, min(len(cand), n + 8), rng)
        out = []
        for t in picked:
            if t[0] in names_seen:
                continue
            names_seen.add(t[0])
            out.append(t)
            if len(out) == n:
                break
        nd = sum(1 for t in out if near[cand.index(t)])
        return out, nd
