"""Hard-negative mining v1.

Intent routing — for an example x with gold label g in taxonomy L, the candidate negatives c != g are
ranked by reciprocal-rank fusion (RRF, k=10) of three signals:

  1. label similarity (label level): KrunBench v0 ``LabelSimilarity`` = char-n-gram TF-IDF of the
     humanized label names + word TF-IDF of the label's train-utterance prototypes;
  2. example-level text similarity: cosine between the embedding of *this* context and the centroid
     embedding of each label's train utterances (multilingual-e5-small, so pt-PT/es-ES contexts work);
  3. confusion data: leave-one-group-out kNN classifier (k=10) over train embeddings; C[g][c] = mean
     share of an example of g's neighbours that carry label c. After Krun One v0 is trained, its own
     confusions on train can be added as a 4th ranking (v0.1 error mining).

Tool routing — candidates come from the train tool pool; ranked by RRF of
  1. example-level: cosine(embedding(query), embedding("name: description")),
  2. tool-level: cosine(embedding(gold doc), embedding(candidate doc)),
  3. tool-name similarity: char-n-gram TF-IDF of humanized names (only over the pre-selected shortlist).
  Possible false negatives are *excluded*: same humanized name as the gold, or doc-embedding cosine
  >= ``near_duplicate_cosine`` with the gold (interchangeable tools from different sources).

Embeddings only use train text to build prototypes/centroids/pools, so evaluation text never shapes a
training option set. Sampling from the fused ranking is stochastic but seeded per decision (the top
``2*(k-1)`` candidates, weights ∝ 1/rank), so different examples of one label get different — still hard —
option sets.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Mapping, Sequence

import numpy as np

E5_ID = "intfloat/multilingual-e5-small"
E5_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"  # pinned in KRUN-004 configs/candidates.yaml
RRF_K = 10


def seeded_rng(*parts: object) -> np.random.Generator:
    h = hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(h[:8], "little")))


def embed_texts(texts: Sequence[str], batch_size: int = 256, max_length: int = 128, device: str | None = None) -> np.ndarray:
    """L2-normalised multilingual-e5-small embeddings ("query: " prefix, mean pooling), float32 [N, 384]."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(E5_ID, revision=E5_REVISION)
    model = AutoModel.from_pretrained(E5_ID, revision=E5_REVISION).to(dev).eval()
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            idx = order[s:s + batch_size]
            b = tok(["query: " + texts[i] for i in idx], padding=True, truncation=True, max_length=max_length,
                    return_tensors="pt").to(dev)
            with torch.autocast(dev if dev != "cpu" else "cpu", dtype=torch.float16, enabled=dev == "cuda"):
                h = model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
            e = (h * m).sum(1) / m.sum(1)
            e = torch.nn.functional.normalize(e.float(), dim=-1)
            out[idx] = e.cpu().numpy()
    return out


def rank_of(scores: np.ndarray) -> np.ndarray:
    """1-based rank per position, highest score first (stable)."""
    order = np.argsort(-scores, kind="stable")
    r = np.empty(len(scores), dtype=np.int64)
    r[order] = np.arange(1, len(scores) + 1)
    return r


def rrf(rank_lists: Iterable[np.ndarray], k: int = RRF_K) -> np.ndarray:
    return sum(1.0 / (k + r) for r in rank_lists)


def sample_from_ranking(candidates: Sequence[str], fused: np.ndarray, n: int, rng: np.random.Generator,
                        pool_factor: int = 2) -> list[str]:
    """Pick n of the top ``pool_factor*n`` candidates, weights ∝ 1/rank (seeded)."""
    if n <= 0:
        return []
    order = np.argsort(-fused, kind="stable")
    top = order[:max(n, min(len(order), pool_factor * n))]
    if len(top) <= n:
        return [candidates[i] for i in top]
    w = 1.0 / np.arange(1, len(top) + 1)
    pick = rng.choice(len(top), size=n, replace=False, p=w / w.sum())
    return [candidates[top[i]] for i in sorted(pick)]


class IntentMiner:
    """Hard negatives over one intent taxonomy (fit on train items only)."""

    def __init__(self, labels: Sequence[str], train_contexts: Sequence[str], train_labels: Sequence[str],
                 train_groups: Sequence[str], train_emb: np.ndarray, label_prototypes: Mapping[str, Sequence[str]],
                 knn: int = 10):
        from krunbench.options import LabelSimilarity

        self.labels = sorted(set(labels))
        self.index = {l: i for i, l in enumerate(self.labels)}
        self.label_sim = LabelSimilarity(self.labels, label_prototypes).matrix  # [L, L], same order (sorted)
        y = np.array([self.index[l] for l in train_labels])
        cent = np.zeros((len(self.labels), train_emb.shape[1]), dtype=np.float32)
        np.add.at(cent, y, train_emb)
        cent /= np.linalg.norm(cent, axis=1, keepdims=True).clip(1e-9)
        self.centroids = cent
        # leave-one-group-out kNN confusion
        conf = np.zeros((len(self.labels), len(self.labels)))
        cnt = np.zeros(len(self.labels))
        groups = np.asarray(train_groups)
        for s in range(0, len(y), 2048):
            sims = train_emb[s:s + 2048] @ train_emb.T
            for j in range(sims.shape[0]):
                sims[j, groups == groups[s + j]] = -np.inf
            nn = np.argpartition(-sims, knn, axis=1)[:, :knn]
            for j in range(sims.shape[0]):
                g = y[s + j]
                np.add.at(conf[g], y[nn[j]], 1.0 / knn)
                cnt[g] += 1
        self.confusion = conf / cnt[:, None].clip(1)

    def ranked(self, gold: str, emb: np.ndarray, exclude: Iterable[str] = ()) -> tuple[list[str], np.ndarray]:
        """Candidates (all labels but gold/excluded) and their fused RRF scores."""
        ex = set(exclude) | {gold}
        cand = [l for l in self.labels if l not in ex]
        ci = np.array([self.index[l] for l in cand])
        g = self.index[gold]
        r1 = rank_of(self.label_sim[g, ci])
        r2 = rank_of(self.centroids[ci] @ emb)
        r3 = rank_of(self.confusion[g, ci] + 1e-6 * self.label_sim[g, ci])  # ties (0 confusion) -> label sim
        return cand, rrf([r1, r2, r3])

    def hard_negatives(self, gold: str, emb: np.ndarray, n: int, rng: np.random.Generator,
                       exclude: Iterable[str] = ()) -> list[str]:
        cand, fused = self.ranked(gold, emb, exclude)
        return sample_from_ranking(cand, fused, n, rng)

    def random_negatives(self, gold: str | None, n: int, rng: np.random.Generator, exclude: Iterable[str] = ()) -> list[str]:
        ex = set(exclude) | ({gold} if gold else set())
        cand = [l for l in self.labels if l not in ex]
        return [cand[i] for i in rng.choice(len(cand), size=min(n, len(cand)), replace=False)]

    def top_confusions(self, gold: str, n: int = 5) -> list[tuple[str, float]]:
        g = self.index[gold]
        row = self.confusion[g].copy()
        row[g] = -1
        return [(self.labels[i], float(row[i])) for i in np.argsort(-row)[:n]]


class ToolPool:
    """A pool of tools (one split's tools) with e5 doc embeddings, for hard / random / safe distractors."""

    def __init__(self, tools: Sequence, emb: np.ndarray, near_duplicate_cosine: float = 0.95, shortlist: int = 200):
        from krun_one.model import humanize_label

        self.tools = list(tools)
        self.emb = emb
        self.hname = np.array([humanize_label(t.name) for t in self.tools], dtype=object)
        self.names = np.array([t.name for t in self.tools], dtype=object)
        self.near_dup = near_duplicate_cosine
        self.shortlist = shortlist
        self.name_index: dict[str, list[int]] = {}
        for i, t in enumerate(self.tools):
            self.name_index.setdefault(t.name, []).append(i)
        self.hname_index: dict[str, list[int]] = {}
        for i, h in enumerate(self.hname):
            self.hname_index.setdefault(h, []).append(i)
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.name_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(list(self.hname))
        self.name_mat = self.name_vec.transform(list(self.hname)).tocsr()

    def interchangeable(self, gold_emb: np.ndarray, gold_hname: str) -> np.ndarray:
        """Pool tools that could be interchangeable with the gold (never used as negatives)."""
        bad = (self.emb @ gold_emb) >= self.near_dup
        bad[self.hname_index.get(gold_hname, [])] = True
        return bad

    def _forbid(self, forbid_names: set[str]) -> np.ndarray:
        bad = np.zeros(len(self.tools), bool)
        for n in forbid_names:
            bad[self.name_index.get(n, [])] = True
        return bad

    def _take(self, order: Iterable[int], n: int, forbid_names: set[str]) -> list[int]:
        picked, taken = [], set(forbid_names)
        if n <= 0:
            return picked
        for i in order:
            name = self.tools[i].name
            if name not in taken:  # option names must be unique within a decision
                picked.append(int(i))
                taken.add(name)
                if len(picked) == n:
                    break
        return picked

    def hard(self, query_emb: np.ndarray, gold_emb: np.ndarray, gold_hname: str, n: int, rng: np.random.Generator,
             forbid_names: set[str]) -> list[int]:
        """RRF(query similarity, gold-doc similarity, name char-TF-IDF) over a shortlist; interchangeable excluded."""
        if n <= 0:
            return []
        s_query, s_gold = self.emb @ query_emb, self.emb @ gold_emb
        bad = self._forbid(forbid_names) | self.interchangeable(gold_emb, gold_hname)
        score = np.where(bad, -np.inf, s_query + s_gold)
        short = np.argsort(-score, kind="stable")[: self.shortlist]
        short = short[np.isfinite(score[short])]
        if len(short) == 0:
            return []
        nm = (self.name_mat[short] @ self.name_vec.transform([gold_hname]).T).toarray().ravel()
        fused = rrf([rank_of(s_query[short]), rank_of(s_gold[short]), rank_of(nm)])
        cand = sample_from_ranking(list(range(len(short))), fused, min(len(short), 2 * n), rng, pool_factor=1)
        return self._take([short[i] for i in cand], n, forbid_names)

    def random(self, n: int, rng: np.random.Generator, forbid_names: set[str], gold_emb: np.ndarray | None = None,
               gold_hname: str | None = None) -> list[int]:
        bad = self._forbid(forbid_names)
        if gold_emb is not None:
            bad |= self.interchangeable(gold_emb, gold_hname or "")
        order = rng.permutation(len(self.tools))
        return self._take(order[~bad[order]], n, forbid_names)

    def safe_random(self, query_emb: np.ndarray, n: int, rng: np.random.Generator, forbid_names: set[str],
                    quantile: float, gold_emb: np.ndarray | None = None, gold_cos: float = 0.90) -> list[int]:
        """Random tools that are unlikely to answer the query: query similarity below ``quantile`` of the pool
        (and, if a removed gold is given, doc similarity to it < ``gold_cos``). Used for UNANSWERABLE sets."""
        s = self.emb @ query_emb
        bad = self._forbid(forbid_names) | (s >= np.quantile(s, quantile))
        if gold_emb is not None:
            bad |= (self.emb @ gold_emb) >= gold_cos
        order = rng.permutation(len(self.tools))
        return self._take(order[~bad[order]], n, forbid_names)
