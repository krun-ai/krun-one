"""Seen/unseen tool bookkeeping shared by the data builders (v0.3 primitives and v1).

A tool is **seen** if it appears anywhere on the training side — as gold *or* as a distractor/negative — either
exactly (same normalized ``name: description``) or **equivalently**: the same normalized description under another
name, or a near-duplicate description (char-5-gram Jaccard >= ``near``), or the same name with a near-duplicate
description. Only tools that are none of these are ``unseen``.

History: the v0.3 noul builder read ``metadata.unseen_tool`` from choice records that write ``tool_seen``, so the
noul "unseen tool" slice was always empty (scripts/prepare_primitives_data.py). It also keyed the flag on the
*decision's gold*, while a noul proposition is about one specific tool (often a negative). Both are fixed by asking
this index about the proposition's own tool.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from .leakage import NearDuplicateIndex, normalize

SEEN_EXACT, SEEN_EQUIVALENT, UNSEEN = "seen_exact", "seen_equivalent", "unseen"


def tool_key(name: str, description: str | None) -> str:
    return normalize(name) + "::" + normalize(description or "")


class TrainToolIndex:
    def __init__(self, near: float = 0.8, min_desc_chars: int = 25):
        self.keys: set[str] = set()
        self.descs: set[str] = set()
        self.near_idx = NearDuplicateIndex(threshold=near)
        self.min_desc_chars = min_desc_chars
        self._n = 0

    def add(self, name: str, description: str | None) -> None:
        k = tool_key(name, description)
        if k in self.keys:
            return
        self.keys.add(k)
        d = normalize(description or "")
        if d:
            self.descs.add(d)
            if len(d) >= self.min_desc_chars:
                self.near_idx.add(f"t{self._n}", description or "")
                self._n += 1

    def add_v0_decisions(self, records: Iterable[Mapping]) -> "TrainToolIndex":
        """krunbench.decision.v0 records (options + option_descriptions): every option, gold or not."""
        for d in records:
            desc = d.get("option_descriptions") or {}
            if not desc:
                continue
            for o in d["options"]:
                self.add(o, desc.get(o))
        return self

    def add_v1_records(self, records: Iterable[Mapping]) -> "TrainToolIndex":
        for r in records:
            for c in r.get("candidates") or []:
                if c.get("description"):
                    self.add(c.get("label") or c["id"], c["description"])
        return self

    def status(self, name: str, description: str | None) -> str:
        if tool_key(name, description) in self.keys:
            return SEEN_EXACT
        d = normalize(description or "")
        if d and d in self.descs:
            return SEEN_EQUIVALENT
        if d and len(d) >= self.min_desc_chars and self.near_idx.query(description or ""):
            return SEEN_EQUIVALENT
        return UNSEEN

    def is_unseen(self, name: str, description: str | None) -> bool:
        return self.status(name, description) == UNSEEN

    def __len__(self) -> int:
        return len(self.keys)
