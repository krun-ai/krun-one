"""Source items for Krun One data (one record per original item, before option-set generation).

Intent routing
    KrunBench v0 items (already decontaminated across all datasets/splits by KRUN-003):
    train items -> training; validation items -> the Krun One hard validation suite.
    HWU64 is *never* used for training (mirror license unconfirmed); it stays a secondary
    evaluation slice (KrunBench validation/test, verbatim).

Tool routing (training + tool validation/holdout; BFCL stays evaluation-only)
    * xLAM function-calling 60k (Salesforce, CC-BY-4.0), via the ungated verbatim re-host
      ``argilla/apigen-function-calling`` (rows with ``origin == "xLAM"`` only; the synthetic
      distilabel half is not used). Kept: queries whose answer calls exactly ONE distinct function.
    * ToolACE (Team-ACE, Apache-2.0): first user turn whose first assistant turn is a call list
      naming exactly one distinct available function.
    * xlam-irrelevance-7.5k (MadeAgents, CC-BY-4.0): queries where none of the given tools applies
      (native ``answerable = false``).
    Tool names and descriptions are the official strings of each source (never rewritten).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from krunbench.schema import iter_jsonl_dicts
from krunbench.splits import normalize_text

TOOL_SOURCES = {
    "xlam": {"hf_id": "argilla/apigen-function-calling", "revision": "170a8a7e0832101f2495a00c9ad9ba1821d3a9b0",
             "file": "data/train-00000-of-00001.parquet", "license": "CC-BY-4.0",
             "upstream": "Salesforce/xlam-function-calling-60k"},
    "toolace": {"hf_id": "Team-ACE/ToolACE", "revision": "6bda777c88d21e5a204703c1ee45597a8fa4f734",
                "file": "data.json", "license": "Apache-2.0", "upstream": "Team-ACE/ToolACE"},
    "xlam_irrelevance": {"hf_id": "MadeAgents/xlam-irrelevance-7.5k", "revision": "34323bf09efc7e4a394998a0fa91ff997617c369",
                         "file": "xlam-7.5k-irrelevancek.json", "license": "CC-BY-4.0",
                         "upstream": "MadeAgents/xlam-irrelevance-7.5k"},
}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str

    @property
    def key(self) -> str:  # identity of a tool across items: exact official name + description
        return f"{self.name}\x1f{self.description}"


@dataclass
class IntentItem:
    dataset: str
    split: str
    group_id: str
    original_id: str
    config: str | None
    context: str
    label: str
    language: str | None
    hf_id: str
    revision: str
    original_split: str
    transform_version: str


@dataclass
class ToolItem:
    source: str
    item_id: str
    query: str
    tools: list[Tool]  # native tool list of the item (official order)
    gold: str | None  # gold tool name; None = no tool applies (irrelevance)
    split: str = "train"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def group_id(self) -> str:
        return f"tool:{normalize_text(self.query)}"

    def tool(self, name: str) -> Tool:
        return next(t for t in self.tools if t.name == name)


# ----------------------------------------------------------------------------------------- intents
def load_intent_items(krunbench_dir: Path, split: str, datasets: Iterable[str]) -> list[IntentItem]:
    """Unique original items of a KrunBench split (from all its examples, answerable or not)."""
    seen: dict[tuple, IntentItem] = {}
    for ds in datasets:
        path = krunbench_dir / ds / f"{split}.jsonl"
        for d in iter_jsonl_dicts(path):
            p = d["provenance"]
            if p["original_label"] is None:
                continue
            key = (p["dataset"], p["config"], p["original_id"])
            if key not in seen:
                seen[key] = IntentItem(p["dataset"], split, p["group_id"], p["original_id"], p["config"], d["context"],
                                       p["original_label"], d["language"], p["hf_id"], p["revision"],
                                       p["original_split"], p["transform_version"])
    return sorted(seen.values(), key=lambda it: (it.dataset, it.config or "", it.original_id))


# ------------------------------------------------------------------------------------------- tools
def _fetch(source: str) -> str:
    from huggingface_hub import hf_hub_download

    s = TOOL_SOURCES[source]
    return hf_hub_download(s["hf_id"], s["file"], repo_type="dataset", revision=s["revision"])


def _clean(s: Any) -> str:
    return " ".join(str(s or "").split())


def _tools_from_json(raw: list[dict]) -> list[Tool] | None:
    tools = [Tool(_clean(t.get("name")), _clean(t.get("description"))) for t in raw if isinstance(t, dict)]
    names = [t.name for t in tools]
    if not tools or any(not n for n in names) or len(set(names)) != len(names):
        return None
    return tools


def load_xlam(path: str | None = None) -> tuple[list[ToolItem], dict[str, int]]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path or _fetch("xlam")).to_pylist()
    out, skip = [], {}
    for r in rows:
        if r["origin"] != "xLAM":
            continue
        try:
            calls, raw = json.loads(r["answers"]), json.loads(r["tools"])
        except (json.JSONDecodeError, TypeError):
            skip["unparseable"] = skip.get("unparseable", 0) + 1
            continue
        tools = _tools_from_json(raw)
        names = {c.get("name") for c in calls if isinstance(c, dict)}
        if tools is None:
            skip["bad tool list"] = skip.get("bad tool list", 0) + 1
        elif len(names) != 1:
            skip["not exactly one distinct function"] = skip.get("not exactly one distinct function", 0) + 1
        elif next(iter(names)) not in {t.name for t in tools}:
            skip["gold not among tools"] = skip.get("gold not among tools", 0) + 1
        elif not _clean(r["query"]):
            skip["empty query"] = skip.get("empty query", 0) + 1
        else:
            out.append(ToolItem("xlam", f"xlam-{r['id']}", _clean(r["query"]), tools, next(iter(names)),
                                metadata={"n_calls": len(calls)}))
    return out, skip


_TOOLACE_TOOLS = re.compile(r"Here is a list of functions in JSON format that you can invoke:\s*(\[.*\])\s*\.\s*\nShould you",
                            re.S)
_CALL_NAME = re.compile(r"(?:^\[|,\s*)([^\(\)\[\],]+?)\(")


def load_toolace(path: str | None = None) -> tuple[list[ToolItem], dict[str, int]]:
    data = json.loads(Path(path or _fetch("toolace")).read_text())
    out, skip = [], {}

    def s(reason):
        skip[reason] = skip.get(reason, 0) + 1

    for i, r in enumerate(data):
        m = _TOOLACE_TOOLS.search(r["system"])
        conv = r["conversations"]
        if not m:
            s("tool list not found")
            continue
        try:
            tools = _tools_from_json(json.loads(m.group(1)))
        except json.JSONDecodeError:
            s("unparseable tool list")
            continue
        if tools is None:
            s("bad tool list")
            continue
        if len(conv) < 2 or conv[0]["from"] != "user" or conv[1]["from"] != "assistant":
            s("first turns are not user -> assistant")
            continue
        a = conv[1]["value"].strip()
        if not (a.startswith("[") and a.endswith("]")):
            s("first assistant turn is not a call list")
            continue
        avail = {t.name for t in tools}
        called = {n.strip() for n in _CALL_NAME.findall(a)}
        if not called or not called <= avail:
            s("called names not parseable / not available")
            continue
        if len(called) != 1:
            s("not exactly one distinct function")
            continue
        q = _clean(conv[0]["value"])
        if not q:
            s("empty query")
            continue
        out.append(ToolItem("toolace", f"toolace-{i}", q, tools, next(iter(called))))
    return out, skip


def load_xlam_irrelevance(path: str | None = None) -> tuple[list[ToolItem], dict[str, int]]:
    data = json.loads(Path(path or _fetch("xlam_irrelevance")).read_text())
    out, skip = [], {}
    for i, r in enumerate(data):
        try:
            tools = _tools_from_json(json.loads(r["tools"]) if isinstance(r["tools"], str) else r["tools"])
        except json.JSONDecodeError:
            tools = None
        if tools is None or len(tools) < 1 or not _clean(r["query"]):
            skip["bad record"] = skip.get("bad record", 0) + 1
            continue
        out.append(ToolItem("xlam_irrelevance", f"xlam_irrelevance-{i}", _clean(r["query"]), tools, None))
    return out, skip
