"""Automated leakage checks for Krun Decisions v1 / KrunBench v1.

Checks (all must return zero violations before a training corpus, a mining pool or a teacher-input file is used):

1. ``group``      — a ``split_group_id`` or ``semantic_group_id`` present in two splits;
2. ``exact``      — the same normalized context text in a training-side file and a protected file;
3. ``near``       — MinHash/LSH near duplicates (char 5-gram Jaccard >= threshold) between training-side and
                    protected texts (catches paraphrase-light copies, casing/punctuation/typo variants);
4. ``candidate``  — protected *tool* strings ("name: description") reused as training candidates, for benchmarks
                    whose tools must stay unseen (e.g. BFCL);
5. ``id``         — duplicate record ids.

"Training-side" = everything that can influence weights: training files, hard-negative mining pools,
teacher-generation inputs, translation inputs, similarity-search indexes. "Protected" = KrunBench v1 test /
final sets and external benchmarks registered in configs/v1/protected_sets.yaml.

Normalization matches krunbench.splits (NFKC, casefold, punctuation stripped, whitespace collapsed).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text)).casefold()
    t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ MinHash / LSH
_PRIME = 4294967291  # largest prime < 2**32: (a*x + b) with a, b, x < p fits in uint64


class MinHasher:
    def __init__(self, num_perm: int = 64, ngram: int = 5, seed: int = 20260926):
        rng = np.random.Generator(np.random.PCG64(seed))
        self.a = rng.integers(1, _PRIME, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, _PRIME, size=num_perm, dtype=np.uint64)
        self.num_perm, self.ngram = num_perm, ngram

    def shingles(self, text: str) -> np.ndarray:
        t = normalize(text)
        n = self.ngram
        grams = {t[i:i + n] for i in range(max(1, len(t) - n + 1))} or {t}
        return np.array([int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "little") % _PRIME
                         for g in grams], dtype=np.uint64)

    def signature(self, text: str) -> np.ndarray:
        x = self.shingles(text)
        h = (self.a[:, None] * x[None, :] + self.b[:, None]) % np.uint64(_PRIME)
        return h.min(axis=1)


def jaccard(a: str, b: str, ngram: int = 5) -> float:
    na, nb = normalize(a), normalize(b)
    sa = {na[i:i + ngram] for i in range(max(1, len(na) - ngram + 1))}
    sb = {nb[i:i + ngram] for i in range(max(1, len(nb) - ngram + 1))}
    return len(sa & sb) / max(1, len(sa | sb))


class NearDuplicateIndex:
    """LSH index over protected texts; ``query`` returns protected ids whose exact char-5-gram Jaccard >= threshold."""

    def __init__(self, threshold: float = 0.8, num_perm: int = 64, bands: int = 16):
        assert num_perm % bands == 0
        self.threshold, self.bands, self.rows = threshold, bands, num_perm // bands
        self.mh = MinHasher(num_perm)
        self.buckets: list[dict[bytes, list[int]]] = [defaultdict(list) for _ in range(bands)]
        self.texts: list[str] = []
        self.ids: list[str] = []

    def add(self, rid: str, text: str) -> None:
        sig = self.mh.signature(text)
        i = len(self.texts)
        self.texts.append(text)
        self.ids.append(rid)
        for b in range(self.bands):
            self.buckets[b][sig[b * self.rows:(b + 1) * self.rows].tobytes()].append(i)

    def query(self, text: str) -> list[tuple[str, float]]:
        sig = self.mh.signature(text)
        cand: set[int] = set()
        for b in range(self.bands):
            cand.update(self.buckets[b].get(sig[b * self.rows:(b + 1) * self.rows].tobytes(), ()))
        hits = []
        for i in cand:
            j = jaccard(text, self.texts[i])
            if j >= self.threshold:
                hits.append((self.ids[i], j))
        return hits


# ------------------------------------------------------------------------------------------------ checks
@dataclass
class LeakageReport:
    violations: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(self.violations.values())

    def summary(self) -> dict[str, Any]:
        return {"ok": self.ok, "counts": self.counts,
                "violations": {k: len(v) for k, v in self.violations.items()},
                "examples": {k: v[:5] for k, v in self.violations.items() if v}}


def _cand_strings(rec: Mapping[str, Any]) -> list[str]:
    return [normalize(f"{c.get('label') or c['id']}: {c['description']}") for c in rec.get("candidates") or []
            if c.get("description")]


def check_leakage(training_side: Iterable[Mapping[str, Any]], protected: Iterable[Mapping[str, Any]],
                  near_threshold: float = 0.8, protect_candidates: bool = False,
                  min_near_chars: int = 20) -> LeakageReport:
    """Compare every training-side record against every protected record (krun.decision.v1 dicts)."""
    rep = LeakageReport()
    prot = list(protected)
    p_groups: dict[str, str] = {}
    p_hash: dict[str, str] = {}
    p_cands: set[str] = set()
    idx = NearDuplicateIndex(threshold=near_threshold)
    seen_ids: set[str] = set()
    for r in prot:
        for g in (r.get("split_group_id"), r.get("semantic_group_id")):
            if g:
                p_groups[g] = r["id"]
        p_hash[text_hash(r["context"]["text"])] = r["id"]
        if len(normalize(r["context"]["text"])) >= min_near_chars:
            idx.add(r["id"], r["context"]["text"])
        if protect_candidates:
            p_cands.update(_cand_strings(r))
        if r["id"] in seen_ids:
            rep.violations["id"].append({"id": r["id"]})
        seen_ids.add(r["id"])
    n = 0
    near_cache: dict[str, list] = {}
    for r in training_side:
        n += 1
        if r["id"] in seen_ids:
            rep.violations["id"].append({"id": r["id"]})
        seen_ids.add(r["id"])
        for g in (r.get("split_group_id"), r.get("semantic_group_id")):
            if g and g in p_groups:
                rep.violations["group"].append({"id": r["id"], "group": g, "protected": p_groups[g]})
                break
        text = r["context"]["text"]
        h = text_hash(text)
        if h in p_hash:
            rep.violations["exact"].append({"id": r["id"], "protected": p_hash[h]})
        elif len(normalize(text)) >= min_near_chars:
            if h not in near_cache:
                near_cache[h] = idx.query(text)
            for pid, j in near_cache[h]:
                rep.violations["near"].append({"id": r["id"], "protected": pid, "jaccard": round(j, 3)})
        if protect_candidates:
            for c in _cand_strings(r):
                if c in p_cands:
                    rep.violations["candidate"].append({"id": r["id"], "candidate": c[:80]})
    rep.counts = {"training_side": n, "protected": len(prot)}
    return rep


def check_split_groups(records: Sequence[Mapping[str, Any]]) -> LeakageReport:
    """Within one corpus: no split_group_id / semantic_group_id may occur in two splits."""
    rep = LeakageReport()
    where: dict[tuple[str, str], str] = {}
    for r in records:
        for kind in ("split_group_id", "semantic_group_id"):
            g = r.get(kind)
            if not g:
                continue
            prev = where.setdefault((kind, g), r["split"])
            if prev != r["split"]:
                rep.violations["group"].append({"id": r["id"], kind: g, "splits": sorted({prev, r["split"]})})
    rep.counts = {"records": len(records)}
    return rep
