"""Krun Decisions v1-R2 (tool-data experiment): candidate-1 training set UNCHANGED + Krun Tool Corpus v2 decisions.

    uv run python scripts/v1/build_decisions_r2_tools.py [--gpu-out data/kd1_tools_v2/gpu_out] [--scale 1.0]
    -> data/kd1_r2/{train.jsonl, val_tools_v2.jsonl, build_manifest.json}

Only tool routing / hard tool negatives / tool OOD / cross-language tools / unseen-schema exposure grow; Choice / Noul /
Score / Multi keep the candidate-1 volume (the base file is copied verbatim).

New decision kinds from When2Call (public API unchanged; they are ordinary choice / noul questions):
  tool action (choice, 3 described candidates): call_tool | ask_clarification | no_available_tool
  DO_NOT_CALL (noul): "One of the available functions can be called right now with the information given."
A held-out 8% of the new sources' query groups (by hash) forms val_tools_v2 (in-distribution tool V; never trained).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from krun_one.v1 import leakage, schema
from krun_one.v1.corpus import tools_v2 as T
from krun_one.v1.corpus.decisions import Builder, pick_k, prim, stage_of
from krun_one.v1.corpus.mining import EmbeddingCache, ToolPool, rng_for
from krun_one.v1.corpus.sources import render_dialogue

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260928
spec = importlib.util.spec_from_file_location("b1", ROOT / "scripts/v1/build_decisions_v1.py")
b1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b1)

QUOTA = {   # (source, kind) -> {lang: n}; ceilings
    ("xlam", "route"): {"en": 45000, "pt": 12000, "es": 9000, "fr": 9000},
    ("xlam", "multi"): {"en": 8000, "pt": 1500, "es": 1000, "fr": 1000},
    ("xlam", "gold_removed"): {"en": 10000, "pt": 3000, "es": 2500, "fr": 2500},
    ("toolmind", "route"): {"en": 40000, "pt": 6000, "es": 5000, "fr": 5000},
    ("toolmind", "multi"): {"en": 4000},
    ("toolmind", "gold_removed"): {"en": 8000, "pt": 2000, "es": 1500, "fr": 1500},
    ("apibench", "route"): {"en": 30000, "pt": 6000, "es": 5000, "fr": 5000},
    ("apibench", "gold_removed"): {"en": 5000, "pt": 1000, "es": 1000, "fr": 1000},
    ("hermes", "route"): {"en": 3000},
    ("johin", "route"): {"en": 3000},
    ("when2call", "action"): {"en": 12000, "pt": 3000, "es": 2500, "fr": 2500},
    ("xlam", "action_call"): {"en": 6000, "pt": 1500, "es": 1000, "fr": 1000},
    ("when2call", "noul_now"): {"en": 8000, "pt": 2000, "es": 1500, "fr": 1500},
    ("xlam", "noul_now"): {"en": 4000, "pt": 1000, "es": 800, "fr": 800},
    ("when2call", "irrelevance"): {"en": 6000, "pt": 1500, "es": 1200, "fr": 1200},
    ("when2call", "ask_route"): {"en": 5000, "pt": 1000, "es": 800, "fr": 800},
    ("xlam_irrelevance", "irrelevance"): {"en": 6000},
}
CAPS = {"route": 4, "gold_removed": 1, "multi": 2, "action": 1, "action_call": 1, "noul_now": 1, "irrelevance": 2,
        "ask_route": 1}
ACTIONS = [
    {"id": "call_tool", "label": "call_tool", "language": "en",
     "description": "One of the listed functions can fulfil the request now, with the information given."},
    {"id": "ask_clarification", "label": "ask_clarification", "language": "en",
     "description": "A listed function fits, but required information is missing: ask the user first."},
    {"id": "no_available_tool", "label": "no_available_tool", "language": "en",
     "description": "None of the listed functions can fulfil this request."},
]
ACTION_OF = {"call": "call_tool", "ask": "ask_clarification", "none": "no_available_tool"}
NOW_PROP = "One of the available functions can be called right now with the information given."


def with_functions(c: dict, max_tools: int = 16) -> str:
    fns = "\n".join(f"- {n}: {d[:160]}" for n, d in (c.get("tools") or [])[:max_tools])
    base = render_dialogue(c) if c.get("history") else c["text"]
    return f"{base}\n\nAvailable functions:\n{fns}"


def is_val(group: str) -> bool:
    return int(hashlib.sha256(f"r2val|{group}".encode()).hexdigest()[:8], 16) / 16**8 < 0.08


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-out", default=str(ROOT / "data/kd1_tools_v2/gpu_out"))
    ap.add_argument("--base", default=str(ROOT / "data/kd1/train.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data/kd1_r2"))
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    man: dict = {"seed": SEED, "quota": {f"{a}/{b}": v for (a, b), v in QUOTA.items()}, "caps": CAPS}
    ctx = [json.loads(l) for l in open(ROOT / "data/kd1_tools_v2/contexts.jsonl")]
    api_catalog = json.loads((ROOT / "data/kd1_tools_v2/api_catalog.json").read_text())
    tr, _, _, mt_stats = b1.load_gpu(Path(args.gpu_out)) if Path(args.gpu_out).exists() else ({}, {}, {}, Counter())
    man["mt_filter"] = dict(mt_stats)
    for c in ctx:
        c["origin"] = "real"
        c["val"] = is_val(c["group"]) and c["source"] != "johin"
    by_key = {f"{c['source']}:{c['oid']}": c for c in ctx}
    mt = []
    for key, text in tr.items():
        base, loc = key.rsplit(">", 1)
        c = by_key.get(base)
        if c is None:
            continue
        mt.append(dict(c, text=text, lang=loc, oid=f"{c['oid']}>{loc}", origin="mt", mining_text=c["text"],
                       transforms=[b1.MT_TAG], history=[]))
    allc = ctx + mt
    # ---- pools / mining
    emb = EmbeddingCache(ROOT / "data/kd1/emb_e5.npz")
    P = T.build_protected(ROOT)
    tool_set = {tuple(t) for c in ctx if not c["val"] for t in (c.get("tools") or []) if t[1]}
    base_ctx = [json.loads(l) for l in open(ROOT / "data/kd1/contexts.jsonl")]
    for c in base_ctx:                      # candidate-1 ToolACE train tools stay in the pool (already filtered there too)
        if c["split"] == "train" and c.get("tools"):
            tool_set |= {tuple(t) for t in c["tools"] if t[1] and not P.tool_hit(t[0], t[1])}
    tool_set = sorted(tool_set)
    emb.ensure([f"{n}: {d}" for n, d in tool_set] + [c["text"] for c in ctx])
    tools = ToolPool(tool_set, emb)
    B = Builder(emb, {}, tools, {}, {}, SEED)
    api_by_hub = {h: [(a[0], a[1], a[2]) for a in apis] for h, apis in api_catalog.items()}
    emb.ensure([f"{a[0]}: {a[1]}" for apis in api_by_hub.values() for a in apis])
    apib_pools = {h: ToolPool([(a[0], a[1]) for a in apis], emb) for h, apis in api_by_hub.items()}
    api_func = {(h, a[0]): a[2] for h, apis in api_by_hub.items() for a in apis}

    def pools(src, pred, split):
        d = defaultdict(list)
        for c in allc:
            if c["source"] == src and pred(c) and (c["val"] == (split == "val")):
                d[prim(c["lang"])].append(c)
        return d

    records, val, fill_report = [], [], {}

    def fill(key, pool_by_lang, make, dest):
        src, kind = key
        q = QUOTA[key] if dest is records else {L: max(1, n // 10) for L, n in QUOTA[key].items()}
        rep = {}
        for L, n in q.items():
            n = int(round(n * args.scale))
            pool = list(pool_by_lang.get(L, []))
            random.Random(f"{SEED}|{src}|{kind}|{L}").shuffle(pool)
            got = []
            for cyc in range(CAPS[kind]):
                for c in pool:
                    if len(got) >= n:
                        break
                    r = make(c, rng_for(SEED, src, kind, c["oid"], cyc))
                    if r:
                        got.append(r)
                if len(got) >= n:
                    break
            dest.extend(got[:n])
            rep[L] = {"quota": n, "made": len(got[:n]), "pool": len(pool)}
        fill_report[f"{'train' if dest is records else 'val'}:{src}/{kind}"] = rep
        print(key, "train" if dest is records else "val", {L: v["made"] for L, v in rep.items()}, flush=True)

    def route(c, r):
        if c.get("history"):
            c2 = dict(c, text=render_dialogue(c), history=[], mining_text=c.get("mining_text") or c["text"])
            return tool_choice_ctx(c2, r)
        return tool_choice_ctx(c, r)

    def tool_choice_ctx(c, r, gold_removed=False):
        rec = B.tool_choice(c, r, gold_removed=gold_removed, tag="r2" if not gold_removed else "r2_gold_removed")
        if rec:
            rec["metadata"]["tool_source"] = c["source"]
        return rec

    def gold_removed(c, r):
        c2 = dict(c, text=render_dialogue(c), history=[], mining_text=c.get("mining_text") or c["text"]) if c.get("history") else c
        return tool_choice_ctx(c2, r, gold_removed=True)

    def apib(c, r, removed=False):
        hub = c["meta"]["hub"]
        pool = apib_pools[hub]
        gold = next(((n, d) for n, d, _ in api_by_hub[hub] if n == c["gold_tool"]), None)
        if gold is None:
            return None
        func = c["meta"]["functionality"]
        k = pick_k(r, 64)
        e = B.ctx_emb(c.get("mining_text") or c["text"])
        # never a negative with the same functionality as the gold (it could also be a correct answer)
        excl = {n for (h, n), f in api_func.items() if h == hub and f == func}
        negs, nd = pool.negatives(gold, e, k - (0 if removed else 1), r, exclude_names=excl, hard=r.random() < 0.7)
        cands = negs + ([] if removed else [gold])
        if len(cands) < 2:
            return None
        r.shuffle(cands)
        rec = B.rec(c, qtype="choice", family="tool", instr_family="tool",
                    candidates=[{"id": n, "label": n, "description": d, "language": "en"} for n, d in cands],
                    labels=[] if removed else [gold[0]], slices=(["gold_removed", "ood_near"] if removed else [])
                    + (["near_duplicate"] if nd else []), strategy="hard_v2", category="ood" if removed else "tool_routing",
                    tag="apib_removed" if removed else "apib", rng=r,
                    extra_md={"tool_source": "apibench", "hub": hub, **({"ood_type": "gold_removed"} if removed else {})})
        return rec

    def multi(c, r):
        if len(c.get("calls") or []) < 2:
            return None
        rec = B.tool_multi(dict(c, text=render_dialogue(c), history=[], mining_text=c.get("mining_text") or c["text"]) if c.get("history") else c, r)
        if rec:
            rec["metadata"]["tool_source"] = c["source"]
        return rec

    def action(c, r):
        if not c.get("tools"):
            return None
        return B.rec(c, qtype="choice", family="tool_action", instr_family="tool",
                     instructions="Given the user request and the available functions, what should the assistant do?",
                     ins_lang="en", candidates=[dict(a) for a in ACTIONS], labels=[ACTION_OF[c["action"]]],
                     slices=["native_irrelevance"] if c["action"] == "none" else [], strategy="fixed",
                     category="tool_routing", tag=f"action_{c['action']}", rng=r, ctx_text=with_functions(c),
                     extra_md={"tool_source": c["source"], "tool_action": ACTION_OF[c["action"]]})

    def noul_now(c, r):
        if not c.get("tools"):
            return None
        return B.rec(c, qtype="noul", family="proposition", instr_family="tool", instructions=NOW_PROP, ins_lang="en",
                     candidates=[{"id": "false"}, {"id": "true"}], labels=["true" if c["action"] == "call" else "false"],
                     slices=[], strategy="native", category="noul", tag=f"noul_now_{c['action']}", rng=r,
                     ctx_text=with_functions(c), extra_md={"tool_source": c["source"], "tool_action": ACTION_OF[c["action"]]})

    def irrelevance(c, r):
        if c["action"] != "none" or not c.get("tools"):
            return None
        rec = B.tool_native_irrelevance(c, r, tag="r2_irrelevance")
        if rec:
            rec["metadata"]["tool_source"] = c["source"]
            rec["metadata"]["ood_type"] = "no_available_tool" if c["source"] == "when2call" else "native_irrelevance"
        return rec

    def ask_route(c, r):
        if c["action"] != "ask" or len(c.get("tools") or []) != 1:
            return None
        c2 = dict(c, gold_tool=c["tools"][0][0])
        rec = B.tool_choice(c2, r, tag="r2_ask_route")
        if rec:
            rec["metadata"].update({"tool_source": "when2call", "tool_action": "ask_clarification"})
        return rec

    call = lambda c: c.get("gold_tool") is not None  # noqa: E731
    for dest, split in ((records, "train"), (val, "val")):
        fill(("xlam", "route"), pools("xlam", call, split), route, dest)
        fill(("xlam", "multi"), pools("xlam", lambda c: len(c.get("calls") or []) >= 2, split), multi, dest)
        fill(("xlam", "gold_removed"), pools("xlam", call, split), gold_removed, dest)
        fill(("toolmind", "route"), pools("toolmind", call, split), route, dest)
        fill(("toolmind", "multi"), pools("toolmind", lambda c: len(c.get("calls") or []) >= 2, split), multi, dest)
        fill(("toolmind", "gold_removed"), pools("toolmind", call, split), gold_removed, dest)
        fill(("apibench", "route"), pools("apibench", call, split), lambda c, r: apib(c, r), dest)
        fill(("apibench", "gold_removed"), pools("apibench", call, split), lambda c, r: apib(c, r, removed=True), dest)
        fill(("hermes", "route"), pools("hermes", call, split), route, dest)
        if split == "train":
            fill(("johin", "route"), pools("johin", call, split), route, dest)
        fill(("when2call", "action"), pools("when2call", lambda c: True, split), action, dest)
        fill(("xlam", "action_call"), pools("xlam", call, split), action, dest)
        fill(("when2call", "noul_now"), pools("when2call", lambda c: True, split), noul_now, dest)
        fill(("xlam", "noul_now"), pools("xlam", call, split), noul_now, dest)
        fill(("when2call", "irrelevance"), pools("when2call", lambda c: c["action"] == "none", split), irrelevance, dest)
        fill(("when2call", "ask_route"), pools("when2call", lambda c: c["action"] == "ask", split), ask_route, dest)
        fill(("xlam_irrelevance", "irrelevance"), pools("xlam_irrelevance", lambda c: True, split), irrelevance, dest)
    records = [r for r in records if r]
    for v in val:
        v["split"] = "validation"
        v["id"] = v["id"].replace("kd1/", "kd1r2val/", 1)
        v["metadata"]["val_part"] = "fit" if int(hashlib.sha256(("fit|" + v["split_group_id"]).encode()).hexdigest()[:8], 16) % 2 else "eval"
    # ---- validation + leakage (new records vs everything protected / validation)
    errs = Counter(e for r in records + val for e in schema.validate(r))
    if errs:
        raise SystemExit(f"schema errors: {dict(errs.most_common(8))}")
    kb = [json.loads(l) for s in ("dev", "val_fit", "val_eval", "final_protected") for l in open(ROOT / f"data/krunbench_v1/{s}.jsonl")]
    val_ml = [json.loads(l) for l in open(ROOT / "data/kd1/val_ml.jsonl")]
    rep = leakage.check_leakage(records, kb + val_ml + val, near_threshold=0.9)
    drop = {v["id"] for k in ("exact", "near", "group", "id") for v in rep.violations.get(k, [])}
    man["leakage_dropped"] = {k: len(v) for k, v in rep.violations.items()}
    records = [r for r in records if r["id"] not in drop]
    rep2 = leakage.check_leakage(records, kb + val_ml + val, near_threshold=0.9)
    tool_viol = Counter()
    for r in records:
        if P.query_hit(r["context"]["text"]):
            tool_viol["query"] += 1
        for c in r["candidates"]:
            if c.get("description"):
                h = P.tool_hit(c.get("label") or c["id"], c["description"])
                if h and r["task"]["family"] != "tool_action":
                    tool_viol[h] += 1
    man["leakage_pass2"] = rep2.summary()["violations"]
    man["protected_tool_violations"] = dict(tool_viol)
    if not rep2.ok or tool_viol:
        raise SystemExit(f"LEAKAGE != 0: {rep2.summary()['violations']} {dict(tool_viol)}")
    random.Random(SEED).shuffle(records)
    shutil.copyfile(args.base, out / "train.jsonl")
    with open(out / "train.jsonl", "a") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "val_tools_v2.jsonl", "w") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    man.update({"fill": fill_report, "n_new_train": len(records), "n_val_tools_v2": len(val),
                "new_by_source_origin": dict(Counter(f"{r['metadata'].get('tool_source')}|{r['metadata']['origin']}" for r in records)),
                "new_by_category": dict(Counter(r["metadata"]["category"] for r in records)),
                "sha256": {n: hashlib.sha256((out / n).read_bytes()).hexdigest() for n in ("train.jsonl", "val_tools_v2.jsonl")}})
    (out / "build_manifest.json").write_text(json.dumps(man, indent=1, default=str))
    print(json.dumps({k: man[k] for k in ("n_new_train", "n_val_tools_v2", "leakage_pass2", "protected_tool_violations",
                                          "new_by_category")}, indent=1))


if __name__ == "__main__":
    main()
