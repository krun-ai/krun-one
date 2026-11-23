"""Audit of REAL, license-clean tool/function/intent routing sources after the xLAM exclusion (counts measured).

Sources loaded (small downloads, pinned revisions):
  ToolACE (Apache-2.0)            Team-ACE/ToolACE @6bda777c         synthetic APIs/queries (generator undisclosed)
  CLINC150 plus (CC-BY-3.0)       clinc/clinc_oos @155b9c7104         human queries, 150 intents + out-of-scope
  SNIPS (Apache-2.0 mirror)       benayas/snips @16915b8957           human queries, 7 intents
  Bitext support (CDLA-Sharing)   bitext/... @430d1a89bd              hybrid/synthetic support queries, 27 intents
  API-Bank level-1/2 test (MIT)   liminghao1630/API-Bank @12e8158b76  human-annotated, runnable APIs (EVAL ONLY)
  SGD schemas (CC-BY-SA-4.0)      github google-research-datasets/dstc8-schema-guided-dialogue (schemas only)
Not loaded (listed in the report): Leyzer (loading script), MTOP (license unconfirmed), xLAM & derived (excluded).

Leakage: contexts vs protected sets (KrunBench v0 test incl. BFCL/HWU64, holdout_tools, primitives test, probe
validation) and ToolACE tools vs BFCL tools (exact + equivalent description).

    uv run python scripts/v1/tool_data_audit.py        # -> results/v1/tool_audit.{json,md}
"""

from __future__ import annotations

import csv
import itertools
import json
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from krun_one.sources import load_toolace
from krun_one.v1 import leakage
from krun_one.v1.tools import TrainToolIndex, tool_key

ROOT = Path(__file__).resolve().parents[2]
SGD_RAW = "https://raw.githubusercontent.com/google-research-datasets/dstc8-schema-guided-dialogue/master/{split}/schema.json"


def hf(repo, f, rev):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, f, repo_type="dataset", revision=rev)


def humanize_tokens(name: str) -> set[str]:
    t = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return set(re.sub(r"[_\-./:]+", " ", t).lower().split())


def lang_guess(text: str) -> str:
    if re.search(r"[一-鿿]", text):
        return "zh"
    return "en"   # all loaded sources are English except ToolACE's Chinese part


def protected_context_index():
    import importlib.util

    spec = importlib.util.spec_from_file_location("bpd", ROOT / "scripts/v1/build_proxy_data.py")
    bpd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bpd)
    prot = bpd.protected_records()
    val = [json.loads(l) for l in open(ROOT / "data/v1_probe/validation.jsonl")]
    return prot + val, [p for p in prot if p["provenance"]["source_dataset"] == "bfcl"]


def toolace_audit(prot_bfcl) -> dict:
    path = hf("Team-ACE/ToolACE", "data.json", "6bda777c88d21e5a204703c1ee45597a8fa4f734")
    raw = json.loads(Path(path).read_text())
    items, skip = load_toolace(path)
    langs = Counter(lang_guess(r["conversations"][0]["value"]) for r in raw if r.get("conversations"))
    no_call = sum(1 for r in raw if len(r.get("conversations", [])) > 1 and r["conversations"][0]["from"] == "user"
                  and not r["conversations"][1]["value"].strip().startswith("["))
    tools = {}
    for it in items:
        for t in it.tools:
            tools[tool_key(t.name, t.description)] = t
    # near-duplicate tools: different key but equivalent description (same normalized desc or Jaccard >= 0.8)
    idx = TrainToolIndex()
    near_dup = 0
    for k, t in tools.items():
        if not idx.is_unseen(t.name, t.description):
            near_dup += 1
        idx.add(t.name, t.description)
    # description-dependent decisions: gold and some native distractor share >= 50% of humanized name tokens
    dd = 0
    for it in items:
        g = humanize_tokens(it.gold)
        for t in it.tools:
            if t.name != it.gold:
                o = humanize_tokens(t.name)
                if g and o and len(g & o) / len(g | o) >= 0.5:
                    dd += 1
                    break
    # BFCL overlap (tools)
    bidx = TrainToolIndex()
    for p in prot_bfcl:
        for c in p["candidates"]:
            if c.get("description"):
                bidx.add(c.get("label") or c["id"], c["description"])
    bfcl_overlap = sum(1 for t in tools.values() if not bidx.is_unseen(t.name, t.description))
    k_native = Counter(len(it.tools) for it in items)
    return {"license": "Apache-2.0", "origin": "LLM-synthetic (generator undisclosed)", "raw_dialogues": len(raw),
            "single_call_queries": len(items), "skipped": skip, "no_call_first_turn": no_call,
            "unique_tools": len(tools), "near_duplicate_tools": near_dup, "description_dependent_decisions": dd,
            "tools_overlapping_bfcl": bfcl_overlap, "languages": dict(langs),
            "native_toolset_size": dict(sorted(k_native.items())), "queries": [it.query for it in items]}


def clinc_audit() -> dict:
    out, labels, oos = {}, set(), 0
    queries = []
    for split in ("train", "validation", "test"):
        t = pq.read_table(hf("clinc/clinc_oos", f"plus/{split}-00000-of-00001.parquet", "155b9c7104"))
        txt, lab = t.column("text").to_pylist(), t.column("intent").to_pylist()
        out[split] = len(txt)
        labels |= set(lab)
        oos += sum(1 for x in lab if x == 42)
        queries += txt
    return {"license": "CC-BY-3.0", "origin": "REAL (crowdsourced)", "queries": queries, "by_split": out,
            "intents": len(labels) - 1, "out_of_scope_queries": oos, "domains": 10, "languages": {"en": len(queries)}}


def snips_audit() -> dict:
    q, labs = [], set()
    for split in ("train", "test"):
        t = pq.read_table(hf("benayas/snips", f"data/{split}-00000-of-00001.parquet", "16915b8957"))
        cols = t.column_names
        q += t.column(cols[0]).to_pylist()
        labs |= set(t.column(cols[1]).to_pylist())
    return {"license": "Apache-2.0 (mirror) / CC0 (upstream sonos)", "origin": "REAL", "queries": q, "intents": len(labs),
            "domains": 1, "languages": {"en": len(q)}}


def bitext_audit() -> dict:
    p = hf("bitext/Bitext-customer-support-llm-chatbot-training-dataset",
           "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv", "430d1a89bd")
    rows = list(csv.DictReader(open(p, newline="")))
    return {"license": "CDLA-Sharing-1.0", "origin": "HYBRID (template/synthetic + linguist curation)",
            "queries": [r["instruction"] for r in rows], "intents": len({r["intent"] for r in rows}),
            "domains": len({r["category"] for r in rows}), "languages": {"en": len(rows)},
            "flags": dict(Counter(f for r in rows for f in (r.get("flags") or ""))) }


def apibank_audit() -> dict:
    apis, dialogs = set(), 0
    for f in ("test-data/level-1-api.json", "test-data/level-2-api.json"):
        data = json.loads(Path(hf("liminghao1630/API-Bank", f, "12e8158b76")).read_text())
        dialogs += len(data)
        for d in data:
            for k in ("api_name", "apis"):
                v = d.get(k) if isinstance(d, dict) else None
                if isinstance(v, str):
                    apis.add(v)
    return {"license": "MIT", "origin": "REAL (human-annotated dialogues, runnable APIs)", "items": dialogs,
            "distinct_apis_seen": len(apis), "use": "eval-only"}


def sgd_audit() -> dict:
    services, intents, domains = set(), set(), set()
    described = 0
    for split in ("train", "dev", "test"):
        try:
            data = json.loads(urllib.request.urlopen(SGD_RAW.format(split=split), timeout=60).read())
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}
        for s in data:
            services.add(s["service_name"])
            domains.add(s["service_name"].split("_")[0])
            for it in s["intents"]:
                intents.add((s["service_name"], it["name"]))
                described += bool(it.get("description"))
    return {"license": "CC-BY-SA-4.0", "origin": "REAL (crowd paraphrased dialogues; human-written schemas)",
            "services": len(services), "service_intents": len(intents), "intents_with_description": described,
            "domains": len(domains), "queries": "~20k dialogues (README; user turns not counted here)",
            "use": "REVIEW (share-alike) — eval/validation only until the owner decides"}


def main() -> None:
    prot, prot_bfcl = protected_context_index()
    res = {"toolace": toolace_audit(prot_bfcl), "clinc150": clinc_audit(), "snips": snips_audit(),
           "bitext": bitext_audit(), "api_bank_eval": apibank_audit(), "sgd": sgd_audit()}
    # context leakage vs protected, per source (exact + near-duplicate, same checker as the corpus builds)
    for name in ("toolace", "clinc150", "snips", "bitext"):
        qs = res[name].pop("queries")
        recs = [{"id": f"{name}/{i}", "context": {"text": q}, "split_group_id": f"{name}:{i}",
                 "semantic_group_id": f"{name}:{i}"} for i, q in enumerate(qs) if q]
        rep = leakage.check_leakage(recs, prot, near_threshold=0.9)
        res[name]["contexts"] = len(recs)
        res[name]["unique_contexts"] = len({leakage.normalize(r["context"]["text"]) for r in recs})
        res[name]["leakage_vs_protected"] = rep.summary()["violations"]
    out = ROOT / "results/v1"
    (out / "tool_audit.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str)[:6000])


if __name__ == "__main__":
    main()
