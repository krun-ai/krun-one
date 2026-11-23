"""Krun One v1 serialization: Laya's ``build_sequence`` layout + an answerability marker + candidate chunking.

Layout of one sequence (one question, one chunk of its candidates)::

    [CLS] "{type} question: {instructions}" [SEP] [MASK] cand_0 [MASK] cand_1 ... [NULL] [SEP] context [SEP]

What is kept from Laya / Krun One v0.3 (validated in production):

* one marker per candidate, and the candidate is scored at its marker; the marker is the tokenizer's MLM
  ``[MASK]`` token (its pretrained role — "what belongs here?" — is the reason it works as a slot);
* the ``"{type} question: {instructions}"`` header, candidate rendering (``"name: description"``, humanized bare
  labels, ``"level i: ..."``, noul ``[false, true]``), the 48-token per-candidate cap, and ``max_len`` / ``head_max_len``;
* candidates before the context, so candidate tokens are never truncated by a long context.

What is new (each item is an ablation in the proxy, see docs/KRUN_ONE_V1_ARCHITECTURE_SPEC.md):

* ``[NULL]``: one extra marker without text after the candidates of ``choice`` / ``multi`` questions. It is the
  slot the answerability output is read from. It is an *existing* reserved token of the Gemma-2 vocabulary
  (``<unused0>``), so the vocabulary and the mmBERT embedding matrix are unchanged;
* **chunking** instead of Laya's even shrinking: when the candidates do not fit in ``head_max_len`` they are split
  into several sequences (each with the full header, context and its own ``[NULL]``). Logits of all chunks are
  merged into one distribution per question (model.merge_chunks). Laya shrinks every candidate to
  ``(head_max_len - 16) // K`` tokens instead, which destroys tool descriptions at k >= 32.
* ``multi`` question type (independent per-candidate decisions).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .arch import QTYPES_V1

OPTION_TOKEN_CAP = 48
NULL_TOKEN = "<unused0>"
NOUL_DEFAULT_FALSE = "no, the statement does not hold"
NOUL_DEFAULT_TRUE = "yes, the statement holds"
DEFAULT_INSTRUCTIONS = {
    "intent": "Choose the option that best describes the intent of the user message.",
    "tool": "Choose the function that should be called to fulfil the user request.",
    "multi": "Select every option that applies to the user message.",
}
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def humanize_label(label: str) -> str:
    t = _CAMEL_RE.sub(" ", label)
    t = re.sub(r"[_\-./:]+", " ", t)
    return " ".join(t.lower().split())


def render_candidates(qtype: str, candidates: Sequence[Mapping[str, Any]], criteria: Mapping[str, str] | None = None
                      ) -> list[str]:
    """Candidate texts in candidate order (the same renderings as v0.3, so the API semantics do not move)."""
    if qtype == "noul":
        c = criteria or {}
        return ["false: " + (c.get("false") or NOUL_DEFAULT_FALSE), "true: " + (c.get("true") or NOUL_DEFAULT_TRUE)]
    if qtype == "score":
        return ["level %d: %s" % (i, c.get("label") or c["id"]) for i, c in enumerate(candidates)]
    names = [c.get("label") or c["id"] for c in candidates]
    if any(c.get("description") for c in candidates):
        return [f"{n}: {c['description']}" if c.get("description") else n for n, c in zip(names, candidates)]
    shown = [humanize_label(n) for n in names]
    return shown if len(set(shown)) == len(shown) else list(names)


@dataclass
class Budget:
    max_len: int = 1024
    head_max_len: int = 896
    option_token_cap: int = OPTION_TOKEN_CAP
    max_candidates_per_chunk: int = 64


@dataclass
class EncodedChunk:
    ids: list[int]
    markers: list[int]              # positions of candidate markers, in chunk order
    candidate_index: list[int]      # index of each marker's candidate in the question's candidate list
    null_pos: int                   # position of [NULL] (-1 when the question type has none)
    qtype: int
    context_truncated: bool = False


@dataclass
class EncodedQuestion:
    chunks: list[EncodedChunk]
    n_candidates: int
    qtype: str
    meta: dict[str, Any] = field(default_factory=dict)


class Serializer:
    """Tokenizer-bound serializer. ``tok`` is a HF tokenizer with mask / cls / sep tokens."""

    def __init__(self, tok, budget: Budget | None = None, null_token: str = NULL_TOKEN, use_null: bool = True):
        self.tok, self.budget, self.use_null = tok, budget or Budget(), use_null
        self.null_id = tok.convert_tokens_to_ids(null_token) if use_null else None
        if use_null and (self.null_id is None or self.null_id == tok.unk_token_id):
            raise ValueError(f"null token {null_token!r} is not in the vocabulary")
        self._cache: dict[str, list[int]] = {}

    def _ids(self, text: str) -> list[int]:
        got = self._cache.get(text)
        if got is None:
            clean = str(text).replace(self.tok.mask_token, " ")
            got = self.tok(clean, add_special_tokens=False)["input_ids"]
            if len(self._cache) < 100_000:
                self._cache[text] = got
        return got

    def plan_chunks(self, cand_ids: Sequence[Sequence[int]], has_null: bool) -> list[list[int]]:
        """Greedy, order-preserving packing of candidates into chunks that fit ``head_max_len``."""
        b = self.budget
        room = b.head_max_len - (1 if has_null else 0)
        chunks, cur, used = [], [], 0
        for i, c in enumerate(cand_ids):
            need = len(c)
            if cur and (used + need > room or len(cur) >= b.max_candidates_per_chunk):
                chunks.append(cur)
                cur, used = [], 0
            cur.append(i)
            used += need
        if cur:
            chunks.append(cur)
        return chunks

    def encode(self, qtype: str, context: str, candidates: Sequence[Mapping[str, Any]], instructions: str,
               criteria: Mapping[str, str] | None = None) -> EncodedQuestion:
        if qtype not in QTYPES_V1:
            raise ValueError(f"unknown question type {qtype!r}")
        b, tok = self.budget, self.tok
        texts = render_candidates(qtype, candidates, criteria)
        cand_ids = [[tok.mask_token_id] + self._ids(" " + t)[: b.option_token_cap] for t in texts]
        has_null = self.use_null and qtype in ("choice", "multi")
        if qtype in ("noul", "score"):
            plan = [list(range(len(texts)))]      # never split: the distribution over levels / {false,true} is joint
            if sum(len(c) for c in cand_ids) > b.head_max_len:
                raise ValueError(f"{len(texts)} {qtype} candidates do not fit in head_max_len={b.head_max_len}")
        else:
            plan = self.plan_chunks(cand_ids, has_null)
        header = self._ids("%s question: %s" % (qtype, instructions))
        ctx = self._ids(context)
        chunks = []
        for idx in plan:
            opt_len = sum(len(cand_ids[i]) for i in idx) + (1 if has_null else 0)
            head = header[: max(8, b.head_max_len - opt_len)]
            ids = [tok.cls_token_id] + head + [tok.sep_token_id]
            markers = []
            for i in idx:
                markers.append(len(ids))
                ids.extend(cand_ids[i])
            null_pos = -1
            if has_null:
                null_pos = len(ids)
                ids.append(self.null_id)
            ids.append(tok.sep_token_id)
            room = max(0, b.max_len - len(ids) - 1)
            if len(ids) + 1 > b.max_len:
                raise ValueError("header + candidates exceed max_len")
            chunks.append(EncodedChunk(ids=ids + ctx[:room] + [tok.sep_token_id], markers=markers, candidate_index=list(idx),
                                       null_pos=null_pos, qtype=QTYPES_V1[qtype], context_truncated=len(ctx) > room))
        return EncodedQuestion(chunks=chunks, n_candidates=len(texts), qtype=qtype)

    def encode_record(self, rec: Mapping[str, Any]) -> EncodedQuestion:
        """Encode a ``krun.decision.v1`` record (see schema.py)."""
        task = rec["task"]
        qtype = task["type"]
        cands = rec["candidates"]
        ins = task.get("instructions") or ""
        if not ins and qtype in ("choice", "multi"):
            ins = DEFAULT_INSTRUCTIONS["multi" if qtype == "multi" else
                                       ("tool" if any(c.get("description") for c in cands) else "intent")]
        q = self.encode(qtype, rec["context"]["text"], cands, ins, task.get("criteria"))
        q.meta["id"] = rec.get("id")
        return q
