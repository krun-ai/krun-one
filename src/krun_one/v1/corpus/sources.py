"""Krun Decisions v1 — raw source loaders (one *context item* per original example, before any decision is built).

Every loader returns plain dicts ("context items"):

    {"source", "oid", "orig_split", "text", "lang", "label" | None, "labels" | None, "group", "split_group",
     "tools" [(name, description)] | None, "gold_tool" | None, "calls" [names] | None, "history" [(speaker, text)],
     "catalog", "meta"}

``group`` is the semantic group (same meaning across languages: MASSIVE ``id``, BANKING77 row); ``split_group`` is the
unit that never crosses splits (⊇ group). Licenses / revisions come from configs/v1/sources.yaml (single source of
truth) and every loader refuses a source whose status is ``excluded*``.
"""

from __future__ import annotations

import csv
import glob
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[4]
HUB = Path.home() / ".cache/huggingface/hub"
RAW = ROOT / "data/raw"
BITEXT_HELDOUT_INTENTS = ("check_refund_policy", "delivery_period", "newsletter_subscription")  # KrunBench v1 unseen labels
MASSIVE_LOCALES = {"en-US": "en-US", "pt-PT": "pt-PT", "es-ES": "es-ES", "fr-FR": "fr-FR"}


@lru_cache(maxsize=1)
def registry() -> dict[str, dict]:
    return yaml.safe_load((ROOT / "configs/v1/sources.yaml").read_text())["decision_sources"]


def source_info(key: str) -> dict:
    s = registry()[key]
    if str(s.get("status", "")).startswith("excluded"):
        raise PermissionError(f"source {key} is {s['status']} (configs/v1/sources.yaml)")
    return s


def _snap(repo: str, rev: str | None = None) -> Path:
    base = HUB / ("datasets--" + repo.replace("/", "--")) / "snapshots"
    if rev:
        cands = [p for p in base.iterdir() if p.name.startswith(rev)]
        if cands:
            return cands[0]
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo, repo_type="dataset", revision=rev))
    return sorted(base.iterdir())[0]


def _item(source, oid, orig_split, text, lang, *, label=None, labels=None, group=None, split_group=None, tools=None,
          gold_tool=None, calls=None, history=None, catalog=None, meta=None) -> dict[str, Any]:
    g = group or f"{source}:{oid}"
    return {"source": source, "oid": str(oid), "orig_split": orig_split, "text": " ".join(str(text).split()),
            "lang": lang, "label": label, "labels": labels, "group": g, "split_group": split_group or g,
            "tools": tools, "gold_tool": gold_tool, "calls": calls, "history": history or [],
            "catalog": catalog or source, "meta": meta or {}}


# ------------------------------------------------------------------------------------------------ intents
def load_banking77(split: str) -> list[dict]:
    """EN rows + the row-aligned PT-BR translation (same semantic / split group)."""
    import pandas as pd
    import pyarrow.parquet as pq

    en_info, pt_info = source_info("banking77"), source_info("banking77_pt_br")
    en_dir = _snap("legacy-datasets/banking77", en_info["revision"][:8]) / "data"
    t = pq.read_table(en_dir / f"{split}-00000-of-00001.parquet")
    names = json.loads(t.schema.metadata[b"huggingface"])["info"]["features"]["label"]["names"]
    en = t.to_pylist()
    pt = pd.read_csv(_snap("c2d-usp/banking77-pt-br", pt_info["revision"][:8]) / f"{split}.csv").dropna(subset=["frase"])
    if len(pt) != len(en):
        raise ValueError(f"banking77-pt-br {split}: {len(pt)} rows vs {len(en)} EN rows")
    out = []
    for i, (r, (_, p)) in enumerate(zip(en, pt.iterrows())):
        g = f"banking77:{split}-{i}"
        lab = names[r["label"]]
        out.append(_item("banking77", f"{split}-{i}", split, r["text"], "en-US", label=lab, group=g))
        out.append(_item("banking77_pt_br", f"{split}-{i}", split, p["frase"], "pt-BR", label=lab, group=g,
                         catalog="banking77", meta={"translator": p["modelo_tradutor"], "human_edited": bool(p["alterado"])}))
    return out


def load_massive(split: str, locales=tuple(MASSIVE_LOCALES)) -> list[dict]:
    """MASSIVE rows; non-EN rows whose 3 judges did not all give intent_score == 1 are dropped (unreliable)."""
    import pyarrow.parquet as pq

    info = source_info("massive")
    snap = _snap("AmazonScience/massive", info["revision"][:8])
    out = []
    for loc in locales:
        t = pq.read_table(next((snap / loc / split).glob("*.parquet")))
        names = json.loads(t.schema.metadata[b"huggingface"])["info"]["features"]["intent"]["names"]
        for r in t.to_pylist():
            js = r["judgments"]["intent_score"] or []
            if loc != "en-US" and (not js or any(x != 1 for x in js)):
                continue
            out.append(_item("massive", f"{r['id']}/{loc}", split, r["utt"], MASSIVE_LOCALES[loc], label=names[r["intent"]],
                             group=f"massive:{r['id']}", meta={"scenario": names[r["intent"]].split("_")[0]}))
    return out


def load_clinc_train() -> list[dict]:
    import pyarrow.parquet as pq

    info = source_info("clinc150")
    t = pq.read_table(_snap("clinc/clinc_oos", info["revision"][:8]) / "plus/train-00000-of-00001.parquet")
    names = json.loads(t.schema.metadata[b"huggingface"])["info"]["features"]["intent"]["names"]
    return [_item("clinc150", f"train-{i}", "train", x, "en-US", label=None if names[y] == "oos" else names[y],
                  meta={"oos": names[y] == "oos"})
            for i, (x, y) in enumerate(zip(t.column("text").to_pylist(), t.column("intent").to_pylist()))]


def load_snips_train() -> list[dict]:
    import pyarrow.parquet as pq

    info = source_info("snips")
    t = pq.read_table(_snap("benayas/snips", info["revision"][:8]) / "data/train-00000-of-00001.parquet")
    return [_item("snips", f"train-{i}", "train", x, "en-US", label=y)
            for i, (x, y) in enumerate(zip(t.column("text").to_pylist(), t.column("category").to_pylist()))]


def load_bitext_train() -> list[dict]:
    """All rows except the 3 intents held out entirely for KrunBench v1 (unseen labels)."""
    info = source_info("bitext_support")
    path = next(_snap("bitext/Bitext-customer-support-llm-chatbot-training-dataset", info["revision"][:8]).glob("*.csv"))
    rows = list(csv.DictReader(open(path, newline="")))
    return [_item("bitext", f"row-{i}", "train", r["instruction"], "en-US", label=r["intent"],
                  meta={"category": r["category"], "flags": r["flags"]})
            for i, r in enumerate(rows) if r["intent"] not in BITEXT_HELDOUT_INTENTS]


# ------------------------------------------------------------------------------------------------ tools
def load_toolace_single(exclude_queries: set[str] = frozenset()) -> list[dict]:
    """ToolACE first user turns whose first assistant turn calls exactly one available function (official names and
    descriptions). ``exclude_queries``: normalized queries reserved elsewhere (v0 val_tools / holdout_tools)."""
    from krun_one.sources import load_toolace

    from ..leakage import normalize

    info = source_info("toolace")
    items, _ = load_toolace(str(_snap("Team-ACE/ToolACE", info["revision"][:8]) / "data.json"))
    out = []
    for it in items:
        if normalize(it.query) in exclude_queries or any("一" <= ch <= "鿿" for ch in it.query):
            continue
        out.append(_item("toolace", it.item_id, "train", it.query, "en-US", tools=[(t.name, t.description) for t in it.tools],
                         gold_tool=it.gold, group=f"tool:{normalize(it.query)}"))
    return out


def load_toolace_pool() -> list[dict]:
    """KrunBench v1 `train_pool_toolace_unused.jsonl`: ToolACE dialogues v0 never used, 40% share assigned to training
    (first turn with no call = native irrelevance, several calls = multi-label)."""
    from ..leakage import normalize

    out = []
    for l in open(ROOT / "data/krunbench_v1/train_pool_toolace_unused.jsonl"):
        d = json.loads(l)
        out.append(_item("toolace", d["oid"], "train", d["text"], "en-US", tools=[tuple(t) for t in d["tools"]],
                         calls=d["calls"], group=f"tool:{normalize(d['text'])}", meta={"pool": "krunbench_v1_unused"}))
    return out


# ------------------------------------------------------------------------------------------------ MultiWOZ 2.2
def multiwoz_schema() -> dict[str, dict]:
    s = json.loads((RAW / "multiwoz22/schema.json").read_text())
    return {sv["service_name"]: {"description": sv["description"],
                                 "intents": {i["name"]: i["description"] for i in sv["intents"]},
                                 "slots": {x["name"]: x["description"] for x in sv["slots"]}} for sv in s}


def load_multiwoz(split: str, history_turns: int = 4) -> list[dict]:
    """Turn-level contexts. For each USER turn: services whose active intent CHANGED at this turn and whose state
    (slot values or requested slots) also changed there — the second condition removes the annotation noise of
    frames that switch intent without any evidence in the turn. ``labels`` = those intents; ``meta.requested`` = slots
    newly requested. Contexts = the last ``history_turns`` turns + the user turn (speaker-tagged)."""
    source_info("multiwoz")
    sub = {"train": "train", "validation": "dev", "test": "test"}[split]
    out = []
    for f in sorted((RAW / "multiwoz22" / sub).glob("dialogues_*.json")):
        for dl in json.loads(f.read_text()):
            prev: dict[str, dict] = {}
            hist: list[tuple[str, str]] = []
            for ti, t in enumerate(dl["turns"]):
                if t["speaker"] != "USER":
                    hist.append(("system", t["utterance"]))
                    continue
                changed, requested, active = [], [], []
                for fr in t.get("frames", []):
                    st = fr.get("state")
                    if not st:
                        continue
                    svc, ai = fr["service"], st["active_intent"]
                    p = prev.get(svc, {"active_intent": "NONE", "slot_values": {}, "requested_slots": []})
                    evidence = st["slot_values"] != p["slot_values"] or set(st["requested_slots"]) - set(p["requested_slots"])
                    if ai != "NONE":
                        active.append(ai)
                    if ai != "NONE" and ai != p["active_intent"] and evidence:
                        changed.append(ai)
                    requested += [s for s in st["requested_slots"] if s not in p["requested_slots"]]
                    prev[svc] = st
                oid = f"{dl['dialogue_id']}/t{ti}"
                out.append(_item("multiwoz", oid, split, t["utterance"], "en-US", labels=sorted(set(changed)),
                                 group=f"multiwoz:{dl['dialogue_id']}", history=list(hist[-history_turns:]),
                                 meta={"requested": requested, "active": sorted(set(active)), "turn": ti,
                                       "services": dl["services"]}))
                hist.append(("user", t["utterance"]))
    return out


def render_dialogue(item: dict) -> str:
    """Dialogue context as one text: previous turns speaker-tagged, then the current user turn."""
    if not item["history"]:
        return item["text"]
    lines = [f"{'User' if s == 'user' else 'Assistant'}: {x}" for s, x in item["history"]]
    return "\n".join(lines + [f"User: {item['text']}"])


def glob_one(pattern: str) -> str:
    got = sorted(glob.glob(pattern))
    if not got:
        raise FileNotFoundError(pattern)
    return got[0]
