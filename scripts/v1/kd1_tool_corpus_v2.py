"""Krun Tool Corpus v2 — context pool of the round-2 tool sources, strict protection, per-source stats, MT job list.

    uv run python scripts/v1/kd1_tool_corpus_v2.py [--toolmind-conversations 40000]
    -> data/kd1_tools_v2/{contexts.jsonl, gpu_jobs/translate.jsonl, manifest.json}, results/v1/r2/tool_corpus_v2_stats.{json,md}

Protection (build aborts on any leftover violation): every item whose query is a protected query is dropped; every
protected tool (BFCL v3 all categories, KrunBench v1 unseen tools, ToolACE items of the v0 tool regression sets = the R2
tool benchmark) is removed from native tool lists by name / alias / description / equivalent description; an item
whose gold or any call is protected is dropped. Queries duplicated across sources are kept once (first source wins in
the order ToolACE-v1 > xLAM > When2Call > Hermes > APIBench > ToolMind > Johin).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from krun_one.v1.corpus import tools_v2 as T
from krun_one.v1.leakage import normalize

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260928
MT_LOCALES = ("pt-BR", "es-419", "fr-FR")
MT_PLAN = {"xlam": 20000, "when2call": 9000, "apibench": 5000, "hermes": 2180, "toolmind": 6000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--toolmind-conversations", type=int, default=40000)
    ap.add_argument("--out", default=str(ROOT / "data/kd1_tools_v2"))
    args = ap.parse_args()
    out = Path(args.out)
    (out / "gpu_jobs").mkdir(parents=True, exist_ok=True)
    P = T.build_protected(ROOT)
    tm, tm_stats = T.load_toolmind(args.toolmind_conversations, SEED)
    ab, api_catalog = T.load_apibench()
    by_src = {"xlam": T.load_xlam(), "xlam_irrelevance": T.load_xlam_irrelevance(), "when2call": T.load_when2call(),
              "hermes": T.load_hermes(), "apibench": ab, "toolmind": tm, "johin": T.load_johin()}
    # queries already used by the v1 corpus (ToolACE train side) win over the new sources
    seen_q = set()
    for l in open(ROOT / "data/kd1/contexts.jsonl"):
        c = json.loads(l)
        if c["source"] == "toolace" and c["split"] == "train":
            seen_q.add(normalize(c["text"]))
    drops = defaultdict(Counter)
    kept: list[dict] = []
    for src in ("xlam", "xlam_irrelevance", "when2call", "hermes", "apibench", "toolmind", "johin"):
        for it in by_src[src]:
            q = normalize(it["text"])
            if P.query_hit(it["text"]):
                drops[src]["protected_query"] += 1
                continue
            if q in seen_q and src != "toolmind":            # ToolMind turns are dialogue-dependent (history differs)
                drops[src]["duplicate_query"] += 1
                continue
            gold = set(it.get("calls") or ([it["gold_tool"]] if it.get("gold_tool") else []))
            if it.get("tools"):
                bad = {n for n, d in it["tools"] if P.tool_hit(n, d)}
                if bad & gold:
                    drops[src]["gold_protected"] += 1
                    continue
                if bad:
                    drops[src]["native_tools_removed"] += len(bad)
                    it["tools"] = [t for t in it["tools"] if t[0] not in bad]
                if len(it["tools"]) < 1:
                    drops[src]["no_tools_left"] += 1
                    continue
            elif src == "apibench":
                if P.tool_hit(it["gold_tool"], it["meta"]["gold_desc"]):
                    drops[src]["gold_protected"] += 1
                    continue
            seen_q.add(q)
            it["split"] = "train"
            kept.append(it)
    api_catalog = {h: [a for a in apis if not P.tool_hit(a[0], a[1])] for h, apis in api_catalog.items()}
    # final guard: no protected tool / query anywhere on the training side
    viol = Counter()
    for it in kept:
        if P.query_hit(it["text"]):
            viol["query"] += 1
        for n, d in it.get("tools") or []:
            h = P.tool_hit(n, d)
            if h:
                viol[f"tool_{h}"] += 1
    for h, apis in api_catalog.items():
        for a in apis:
            if P.tool_hit(a[0], a[1]):
                viol["api"] += 1
    if viol:
        raise SystemExit(f"protected tools/queries left on the training side: {dict(viol)}")
    with open(out / "contexts.jsonl", "w") as f:
        for it in kept:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    (out / "api_catalog.json").write_text(json.dumps(api_catalog))
    # MT jobs (train only)
    jobs = []
    for src, cap in MT_PLAN.items():
        pool = [it for it in kept if it["source"] == src and len(it["text"]) <= 600]
        pool = random.Random(f"{SEED}{src}").sample(pool, min(cap, len(pool)))
        for it in pool:
            for loc in MT_LOCALES:
                jobs.append({"job": "translate", "key": f"{it['source']}:{it['oid']}>{loc}", "split": "train",
                             "source": src, "oid": it["oid"], "text": it["text"], "target_locale": loc})
    with open(out / "gpu_jobs/translate.jsonl", "w") as f:
        for j in jobs:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    # ---------------- stats per source
    def desc_norm(d):
        return normalize(d or "")
    stats = {}
    for src in sorted({it["source"] for it in kept}):
        items = [it for it in kept if it["source"] == src]
        tools = {(n, d) for it in items for n, d in (it.get("tools") or [])}
        if src == "apibench":
            tools = {(a[0], a[1]) for h in api_catalog.values() for a in h}
        golds = {g for it in items for g in (it.get("calls") or ([it["gold_tool"]] if it.get("gold_tool") else []))}
        stats[src] = {"items": len(items), "unique_queries": len({normalize(it["text"]) for it in items}),
                      "actions": dict(Counter(it["action"] for it in items)),
                      "single_call": sum(1 for it in items if it.get("gold_tool")),
                      "multi_call": sum(1 for it in items if len(it.get("calls") or []) >= 2),
                      "unique_tools": len({n for n, _ in tools}), "unique_tool_descriptions": len({desc_norm(d) for _, d in tools}),
                      "unique_gold_tools": len(golds),
                      "native_candidates_per_item": round(sum(len(it.get("tools") or []) for it in items) / max(1, len(items)), 2),
                      "dropped": dict(drops[src]),
                      "mt_jobs": sum(1 for j in jobs if j["source"] == src)}
    stats["toolmind_loader"] = tm_stats
    stats["protected_registry"] = {"tools_by_origin": dict(P.by_origin), "unique_names": len(P.names),
                                   "unique_aliases": len(P.aliases), "queries": len(P.queries)}
    stats["sha256_contexts"] = hashlib.sha256((out / "contexts.jsonl").read_bytes()).hexdigest()
    stats["mt_jobs_total"] = len(jobs)
    r = ROOT / "results/v1/r2"
    r.mkdir(parents=True, exist_ok=True)
    (r / "tool_corpus_v2_contexts.json").write_text(json.dumps(stats, indent=1))
    (out / "manifest.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items()}, indent=1)[:6000])


if __name__ == "__main__":
    main()
