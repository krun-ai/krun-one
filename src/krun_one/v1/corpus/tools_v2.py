"""Krun Tool Corpus v2 — adapters for the round-2 tool sources + the strict protected-tool registry.

Every adapter yields *tool context items* (same shape as corpus.sources items, plus ``action``):

    {"source", "oid", "orig_split", "text", "lang", "tools": [(name, description)], "gold_tool" | None,
     "calls": [names] | None, "action": "call" | "ask" | "none", "history": [(speaker, text)], "group",
     "split_group", "catalog", "meta"}

``action``: call = a listed function should be called now (gold_tool / calls); ask = a listed function fits but required
information is missing (clarification); none = no listed function can fulfil the request.

Sources and statuses: configs/v1/sources.yaml (loaders refuse ``excluded*``). Only TRAIN splits are read; When2Call test,
APIBench eval, and ToolMind open_datasets are never touched.
"""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from ..leakage import NearDuplicateIndex, normalize
from .sources import HUB, _item, _snap, source_info

_WS = re.compile(r"\s+")


def clean(s) -> str:
    return _WS.sub(" ", str(s or "")).strip()


def alias(name: str) -> str:
    """Alias key of a tool name: camelCase / snake / kebab / dotted / spaced variants collapse to sorted tokens."""
    t = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name))
    t = re.sub(r"[^A-Za-z0-9]+", " ", t).lower().split()
    return " ".join(t)


def _tools_from_dicts(raw: Iterable) -> list[tuple[str, str]]:
    out, seen = [], set()
    for t in raw or []:
        if isinstance(t, str):
            try:
                t = json.loads(t)
            except json.JSONDecodeError:
                continue
        if not isinstance(t, dict):
            continue
        if "function" in t and isinstance(t["function"], dict):
            t = t["function"]
        n, d = clean(t.get("name")), clean(t.get("description"))
        if n and n not in seen:
            seen.add(n)
            out.append((n, d))
    return out


def _group(prefix: str, text: str) -> str:
    return f"tool:{normalize(text)}" if text else prefix


# ------------------------------------------------------------------------------------------------ protected registry
class ProtectedTools:
    """Tools and queries that must never appear on the training side (BFCL v3 all categories, KrunBench v1 protected
    unseen tools, the v0 tool regression sets used as the R2 tool benchmark). Match: exact normalized name, alias,
    normalized description, near-duplicate description (char 5-gram Jaccard >= 0.8)."""

    def __init__(self):
        self.names, self.aliases, self.descs = set(), set(), set()
        self.near = NearDuplicateIndex(threshold=0.8)
        self.queries: set[str] = set()
        self.n = 0
        self.by_origin = defaultdict(int)

    def add_tool(self, name: str, desc: str, origin: str):
        name, desc = clean(name), clean(desc)
        if not name:
            return
        self.names.add(normalize(name))
        self.aliases.add(alias(name))
        d = normalize(desc)
        if d:
            self.descs.add(d)
            if len(d) >= 25:
                self.near.add(f"p{self.n}", desc)
                self.n += 1
        self.by_origin[origin] += 1

    def add_query(self, text: str):
        if text:
            self.queries.add(normalize(text))

    def tool_hit(self, name: str, desc: str | None) -> str | None:
        if normalize(name) in self.names:
            return "name"
        if alias(name) in self.aliases:
            return "alias"
        d = normalize(desc or "")
        if d and d in self.descs:
            return "description"
        if d and len(d) >= 25 and self.near.query(desc or ""):
            return "equivalent_description"
        return None

    def query_hit(self, text: str) -> bool:
        return normalize(text) in self.queries


def build_protected(root: Path) -> ProtectedTools:
    P = ProtectedTools()
    # BFCL v3: every category, every function and every question (protected external evaluation)
    bfcl = next((HUB / "datasets--gorilla-llm--Berkeley-Function-Calling-Leaderboard/snapshots").iterdir())
    for f in sorted(bfcl.glob("BFCL_v3_*.json")):
        for l in open(f):
            try:
                d = json.loads(l)
            except json.JSONDecodeError:
                continue
            for fn in d.get("function") or []:
                if isinstance(fn, dict):
                    P.add_tool(fn.get("name", ""), fn.get("description", ""), "bfcl")
            q = d.get("question")
            for turn in (q if isinstance(q, list) else []):
                for m in (turn if isinstance(turn, list) else [turn]):
                    if isinstance(m, dict) and m.get("role") == "user":
                        P.add_query(m.get("content", ""))
    # KrunBench v1 protected unseen tools
    man = json.loads((root / "data/krunbench_v1/protected_manifest.json").read_text())
    for t in man["unseen_tools"]:
        P.add_tool(t["name"], t["description"], "krunbench_v1_unseen")
    # v0 tool regression sets (R2 tool benchmark): every tool shown and every query
    for f in ("val_tools", "holdout_tools"):          # the ToolACE items = the R2 tool benchmark (gold + distractors)
        for l in open(root / f"data/v0/{f}.jsonl"):
            d = json.loads(l)
            if d["provenance"].get("dataset") != "toolace":
                continue
            P.add_query(d["context"])
            for o in d["options"]:
                P.add_tool(o, (d.get("option_descriptions") or {}).get(o, ""), f"v0_{f}")
    return P


# ------------------------------------------------------------------------------------------------ adapters
def load_xlam() -> list[dict]:
    """xLAM 60k (owner-approved 2026-09-28). Local copy: argilla re-host rows with origin == xLAM."""
    import pyarrow.parquet as pq

    source_info("xlam")
    snap = next((HUB / "datasets--argilla--apigen-function-calling/snapshots").iterdir())
    rows = pq.read_table(next(snap.rglob("*.parquet"))).to_pylist()
    out = []
    for r in rows:
        if r.get("origin") != "xLAM":
            continue
        try:
            calls, tools = json.loads(r["answers"]), _tools_from_dicts(json.loads(r["tools"]))
        except (json.JSONDecodeError, TypeError):
            continue
        q = clean(r["query"])
        names = sorted({c.get("name") for c in calls if isinstance(c, dict)})
        if not q or not tools or not names or not set(names) <= {t[0] for t in tools}:
            continue
        it = _item("xlam", f"xlam-{r['id']}", "train", q, "en-US", tools=tools, calls=names,
                   gold_tool=names[0] if len(names) == 1 else None, group=_group("xlam", q), catalog="xlam")
        it["action"] = "call"
        out.append(it)
    return out


def load_xlam_irrelevance() -> list[dict]:
    source_info("xlam_irrelevance")
    snap = next((HUB / "datasets--MadeAgents--xlam-irrelevance-7.5k/snapshots").iterdir())
    data = json.loads(next(snap.glob("*.json")).read_text())
    out = []
    for i, r in enumerate(data):
        raw = r["tools"]
        tools = _tools_from_dicts(json.loads(raw) if isinstance(raw, str) else raw)
        q = clean(r.get("query"))
        if tools and q:
            it = _item("xlam_irrelevance", f"xirr-{i}", "train", q, "en-US", tools=tools, calls=[],
                       group=_group("xirr", q), catalog="xlam")
            it["action"] = "none"
            out.append(it)
    return out


_TOOLCALL = re.compile(r"<TOOLCALL>(.*?)</TOOLCALL>", re.S)


def _w2c_action(content: str) -> tuple[str, list[str]]:
    m = _TOOLCALL.search(content or "")
    if m:
        try:
            calls = json.loads(m.group(1))
            return "call", sorted({c["name"] for c in calls if isinstance(c, dict) and c.get("name")})
        except (json.JSONDecodeError, TypeError):
            return "bad", []
    c = (content or "").strip()
    return ("ask" if c.endswith("?") else "none"), []


def load_when2call() -> list[dict]:
    """When2Call TRAIN only (sft + pref chosen). test = BFCL-derived -> never read."""
    info = source_info("when2call_train")
    snap = _snap("nvidia/When2Call", info["revision"][:8])
    out = []
    for fname, key in (("train/when2call_train_sft.jsonl", "sft"), ("train/when2call_train_pref.jsonl", "pref")):
        for i, l in enumerate(open(snap / fname)):
            r = json.loads(l)
            tools = _tools_from_dicts(r["tools"])
            user = [m for m in r["messages"] if m["role"] == "user"]
            if not user:
                continue
            q = clean(user[-1]["content"])
            resp = r["chosen_response"] if key == "pref" else r["messages"][-1]
            if resp.get("role") != "assistant":
                continue
            action, names = _w2c_action(resp.get("content"))
            if action == "bad" or (action == "call" and not set(names) <= {t[0] for t in tools}):
                continue
            it = _item("when2call", f"w2c-{key}-{i}", "train", q, "en-US", tools=tools, calls=names,
                       gold_tool=names[0] if action == "call" and len(names) == 1 else None,
                       group=_group("w2c", q), catalog="when2call", meta={"file": key})
            it["action"] = action
            out.append(it)
    return out


_APIB_INS = re.compile(r"###Instruction:\s*(.*?)\s*###Output", re.S)


def load_apibench() -> tuple[list[dict], dict[str, list[tuple[str, str, str]]]]:
    """APIBench TRAIN splits (HF / TorchHub / TF Hub). Returns (items, api catalog per hub: [(name, desc, functionality)])."""
    info = source_info("apibench")
    snap = _snap("gorilla-llm/APIBench", info["revision"][:8])
    cat: dict[str, list] = {}
    for hub, f in (("huggingface", "huggingface_api.jsonl"), ("torchhub", "torchhub_api.jsonl"),
                   ("tensorflowhub", "tensorflowhub_api.jsonl")):
        apis = []
        for l in open(snap / f):
            try:
                d = json.loads(l)
            except json.JSONDecodeError:
                continue
            name = clean(d.get("api_name"))
            desc = clean(d.get("description"))[:300]
            func = clean(d.get("functionality") or d.get("domain"))
            if name and desc:
                apis.append((name, desc, func))
        cat[hub] = apis
    out = []
    for hub, f in (("huggingface", "huggingface_train.json"), ("torchhub", "torchhub_train.json"),
                   ("tensorflowhub", "tensorflow_train.json")):
        for i, l in enumerate(open(snap / f)):
            try:
                d = json.loads(l)
            except json.JSONDecodeError:
                continue
            code = d.get("code") or ""
            m = _APIB_INS.search(code)
            ins = m.group(1) if m else None
            if ins is None:
                try:
                    ins = ast.literal_eval(code.split(", 'Output'")[0] + "}").get("Instruction") if code.startswith("{") else None
                except Exception:  # noqa: BLE001
                    mm = re.search(r"'Instruction':\s*(.*?),\s*'Output'", code, re.S)
                    ins = mm.group(1) if mm else None
            api = d.get("api_data") or {}
            name = clean(api.get("api_name"))
            if not ins or not name:
                continue
            it = _item("apibench", f"{hub}-{i}", "train", clean(ins), "en-US",
                       tools=None, gold_tool=name, group=_group("apib", clean(ins)), catalog=f"apibench_{hub}",
                       meta={"hub": hub, "functionality": clean(api.get("functionality") or api.get("domain")),
                             "gold_desc": clean(api.get("description"))[:300]})
            it["action"] = "call"
            out.append(it)
    return out, cat


def _hermes_tools(r) -> list[tuple[str, str]]:
    raw = r.get("tools")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = None
    if not raw:
        sysmsg = next((c["value"] for c in r["conversations"] if c["from"] == "system"), "")
        m = re.search(r"<tools>\s*(.*?)\s*</tools>", sysmsg, re.S)
        try:
            raw = json.loads(m.group(1)) if m else []
        except json.JSONDecodeError:
            raw = []
    return _tools_from_dicts(raw)


_HERMES_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def load_hermes() -> list[dict]:
    out = []
    for key, f in (("hermes_fc_singleturn", "func-calling-singleturn.json"), ("hermes_fc_multiturn", "func-calling.json")):
        info = source_info(key)
        snap = _snap("NousResearch/hermes-function-calling-v1", info["revision"][:8])
        for r in json.loads((snap / f).read_text()):
            tools = _hermes_tools(r)
            conv = [c for c in r["conversations"] if c["from"] != "system"]
            if len(conv) < 2 or conv[0]["from"] != "human" or conv[1]["from"] != "gpt":
                continue
            names = []
            for m in _HERMES_CALL.finditer(conv[1]["value"]):
                try:
                    names.append(ast.literal_eval(m.group(1)).get("name"))
                except Exception:  # noqa: BLE001
                    try:
                        names.append(json.loads(m.group(1)).get("name"))
                    except Exception:  # noqa: BLE001
                        pass
            names = sorted({n for n in names if n})
            q = clean(conv[0]["value"])
            if not names or not tools or not set(names) <= {t[0] for t in tools}:
                continue
            it = _item("hermes", f"{key}-{r['id']}", "train", q, "en-US", tools=tools, calls=names,
                       gold_tool=names[0] if len(names) == 1 else None, group=_group("hermes", q), catalog="hermes",
                       meta={"category": r.get("category"), "subcategory": r.get("subcategory"), "file": key})
            it["action"] = "call"
            out.append(it)
    return out


def glaive_function_names() -> set[str]:
    """Filter list only (Glaive v2 stays PENDING_LICENSE_HISTORY_REVIEW; never training data)."""
    snap = next((HUB / "datasets--glaiveai--glaive-function-calling-v2/snapshots").iterdir())
    names = set()
    for r in json.loads((snap / "glaive-function-calling-v2.json").read_text()):
        for m in re.finditer(r'"name"\s*:\s*"([^"]+)"', r.get("system", "")):
            names.add(normalize(m.group(1)))
    return names


def load_toolmind(max_conversations: int = 60000, seed: int = 20260928) -> tuple[list[dict], dict]:
    """ToolMind graph_syn: turn-level decisions (user turn -> the assistant's next tool call(s)). Conversations that use
    ANY Glaive-v2 function name are dropped."""
    import random

    info = source_info("toolmind_graph_syn")
    snap = _snap("Nanbeige/ToolMind", info["revision"][:8])
    glaive = glaive_function_names()
    lines = open(snap / "graph_syn_datasets/graphsyn.jsonl").read().split("\n")
    lines = [l for l in lines if l]
    random.Random(seed).shuffle(lines)
    out, stats = [], defaultdict(int)
    for ci, l in enumerate(lines[:max_conversations]):
        r = json.loads(l)
        tools = _tools_from_dicts(r.get("tools") or [])
        if not tools:
            stats["no_tools"] += 1
            continue
        if any(normalize(n) in glaive for n, _ in tools):
            stats["dropped_glaive_function"] += 1
            continue
        conv = r["conversations"]
        hist: list[tuple[str, str]] = []
        for ti, m in enumerate(conv):
            if m["role"] == "user":
                nxt = conv[ti + 1] if ti + 1 < len(conv) else None
                calls = sorted({c["function"]["name"] for c in (nxt or {}).get("tool_calls") or []
                                if isinstance(c, dict) and c.get("function", {}).get("name")})
                q = clean(m["content"])
                if calls and set(calls) <= {t[0] for t in tools} and q:
                    it = _item("toolmind", f"tm-{ci}/t{ti}", "train", q, "en-US", tools=tools, calls=calls,
                               gold_tool=calls[0] if len(calls) == 1 else None, group=f"toolmind:{ci}",
                               history=list(hist[-4:]), catalog="toolmind")
                    it["action"] = "call"
                    out.append(it)
                hist.append(("user", clean(m["content"])))
            elif m["role"] == "assistant" and clean(m.get("content")):
                hist.append(("system", re.sub(r"<think>.*?</think>", "", m["content"], flags=re.S).strip()[:400]))
    stats["conversations_read"] = min(len(lines), max_conversations)
    stats["conversations_total"] = len(lines)
    return out, dict(stats)


def load_johin() -> list[dict]:
    info = source_info("johin_fc")
    snap = _snap("Johin/function-calling-dataset", info["revision"][:8])
    out = []
    for i, l in enumerate(open(snap / "data/train.jsonl")):
        r = json.loads(l)
        msgs = r["messages"]
        sysm = next((m["content"] for m in msgs if m["role"] == "system"), "")
        tools = [(clean(a), clean(b)) for a, b in re.findall(r"^- ([\w\.]+): (.+)$", sysm, re.M)]
        users = [j for j, m in enumerate(msgs) if m["role"] == "user"]
        if not tools or not users:
            continue
        j = users[0]
        nxt = msgs[j + 1] if j + 1 < len(msgs) else {}
        fc = nxt.get("function_call") or []
        fcs = fc if isinstance(fc, list) else [fc]
        names = sorted({c.get("name") for c in fcs if isinstance(c, dict) and c.get("name")})
        q = clean(msgs[j]["content"])
        if names and set(names) <= {t[0] for t in tools}:
            it = _item("johin", f"johin-{i}", "train", q, "en-US", tools=tools, calls=names,
                       gold_tool=names[0] if len(names) == 1 else None,
                       group=_group("johin", q), catalog="johin", meta=r.get("metadata") or {})
            it["action"] = "call"
            out.append(it)
    return out
