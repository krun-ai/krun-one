"""Krun Decisions v1 — stage 1: context pool, split assignment, and the GPU job lists (MT / descriptions).

    uv run python scripts/v1/kd1_contexts.py            # -> data/kd1/{contexts.jsonl, catalogs.json, gpu_jobs/*.jsonl, manifest.json}

Split rule (unit = split_group; nothing below is ever training-side if it is validation / regression / protected):

| source | train | val_ml (multilingual V set, new) | excluded (regression / protected) |
|---|---|---|---|
| MASSIVE en/pt/es/fr | train | validation | test (KrunBench v0 regression) |
| BANKING77 EN + PT-BR | train | test rows (already opened by v0 -> fine for V, never T) | — |
| CLINC150 / SNIPS / Bitext | train (Bitext minus 3 held-out intents) | — | validation/test = KrunBench v1 |
| ToolACE | single-call (minus v0 val_tools/holdout_tools queries) + KrunBench v1 unused train pool | MT of KrunBench v1 val ToolACE items | KrunBench v1 val/final, v0 regression |
| MultiWOZ 2.2 | train | dev | test |

Train items that exactly / near duplicate (Jaccard >= 0.9) a val_ml or KrunBench v1 item are dropped here already
(BANKING77 has train/test near duplicates).

GPU jobs (run on RunPod by scripts/v1/kd1_gpu_jobs.py; nothing here needs a GPU):
* ``translate``: train contexts -> pt-BR / es-419 / fr-FR (+ KrunBench v1 **validation** ToolACE/CLINC items -> val_ml;
  their output inherits the validation split and never becomes training data);
* ``describe``: one-sentence description per intent label (EN), from train examples only;
* ``translate_desc``: ToolACE tool descriptions + intent descriptions -> pt-BR / es-419 / fr-FR (translated-schema slice;
  tool names are never translated).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from krun_one.v1 import leakage
from krun_one.v1.corpus import sources as S

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260927
MT_LOCALES = ("pt-BR", "es-419", "fr-FR")
MT_PLAN = {  # source -> (max contexts, locales); None = all
    "toolace": (None, MT_LOCALES),
    "clinc150": (None, MT_LOCALES),
    "snips": (5000, MT_LOCALES),
    "bitext": (8000, MT_LOCALES),
    "banking77": (None, ("es-419", "fr-FR")),          # pt-BR exists (human-reviewed)
    "massive": (None, ("pt-BR",)),                     # pt-PT exists natively; pt-BR gives the PT-BR/PT-PT pair
}
VAL_MT = {"toolace": None, "clinc150": 1200, "bitext": 300}


def h01(*parts) -> float:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16) / 16**12


def kbv1_items() -> list[dict]:
    out = []
    for split in ("dev", "val_fit", "val_eval", "final_protected"):
        for l in open(ROOT / f"data/krunbench_v1/{split}.jsonl"):
            r = json.loads(l)
            r["_kbv1_split"] = split
            out.append(r)
    return out


def v0_regression_tool_queries() -> set[str]:
    qs = set()
    for f in ("val_tools", "holdout_tools"):
        for l in open(ROOT / f"data/v0/{f}.jsonl"):
            d = json.loads(l)
            if d["provenance"].get("dataset") == "toolace":
                qs.add(leakage.normalize(d["context"]))
    return qs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/kd1"))
    args = ap.parse_args()
    out = Path(args.out)
    (out / "gpu_jobs").mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    items: list[dict] = []
    for split in ("train", "test"):
        for it in S.load_banking77(split):
            it["split"] = "train" if split == "train" else "val_ml"
            items.append(it)
    for split, dest in (("train", "train"), ("validation", "val_ml")):
        for it in S.load_massive(split):
            it["split"] = dest
            items.append(it)
    for f in (S.load_clinc_train, S.load_snips_train, S.load_bitext_train):
        for it in f():
            it["split"] = "train"
            items.append(it)
    for it in S.load_toolace_single(v0_regression_tool_queries()) + S.load_toolace_pool():
        it["split"] = "train"
        items.append(it)
    for split, dest in (("train", "train"), ("validation", "val_ml")):
        for it in S.load_multiwoz(split):
            it["split"] = dest
            items.append(it)

    # ---- leakage vs KrunBench v1 (all its splits) and vs val_ml, on the training side
    kb = kbv1_items()
    val_side = [{"id": f"{i['source']}:{i['oid']}", "context": {"text": i["text"]}, "split_group_id": i["split_group"],
                 "semantic_group_id": i["group"]} for i in items if i["split"] == "val_ml"]
    prot = [{"id": r["id"], "context": r["context"], "split_group_id": r["split_group_id"],
             "semantic_group_id": r["semantic_group_id"]} for r in kb] + val_side
    train_side = [{"id": f"{i['source']}:{i['oid']}:{n}", "context": {"text": i["text"]}, "split_group_id": i["split_group"],
                   "semantic_group_id": i["group"]} for n, i in enumerate(items) if i["split"] == "train"]
    rep = leakage.check_leakage(train_side, prot, near_threshold=0.9)
    bad = {v["id"] for k in ("exact", "near", "group") for v in rep.violations.get(k, [])}
    # the whole split group goes (translations / parallel rows of a leaked text), except MultiWOZ where the group is a
    # dialogue and the leaked turns are generic ("thank you, goodbye"): only those turns are dropped
    bad_items = {n for n, i in enumerate(items) if i["split"] == "train" and f"{i['source']}:{i['oid']}:{n}" in bad}
    bad_groups = {items[n]["split_group"] for n in bad_items if items[n]["source"] != "multiwoz"}
    dropped = Counter()
    for n, i in enumerate(items):
        if i["split"] == "train" and (i["split_group"] in bad_groups or n in bad_items):
            i["split"] = "dropped_leakage"
            dropped[i["source"]] += 1
    # group isolation inside the pool
    where: dict[str, set] = defaultdict(set)
    for i in items:
        where[i["split_group"]].add(i["split"])
    crossing = [g for g, s in where.items() if len(s - {"dropped_leakage"}) > 1]  # a group lives in ONE split
    if crossing:
        raise SystemExit(f"{len(crossing)} split groups cross splits, e.g. {crossing[:3]}")

    with open(out / "contexts.jsonl", "w") as f:
        for i in items:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")

    # ---- label catalogs (train examples only) for the description job
    cat_ex: dict[tuple, list] = defaultdict(list)
    for i in items:
        if i["split"] == "train" and i["label"] and i["lang"] == "en-US":
            cat_ex[(i["catalog"], i["label"])].append(i["text"])
    catalogs: dict[str, dict] = defaultdict(dict)
    for (cat, lab), ex in sorted(cat_ex.items()):
        catalogs[cat][lab] = {"examples": random.Random(f"{SEED}{cat}{lab}").sample(ex, min(6, len(ex)))}
    for svc, sv in S.multiwoz_schema().items():                      # human-written descriptions (MultiWOZ 2.2)
        for name, desc in sv["intents"].items():
            catalogs["multiwoz"][name] = {"description": desc, "service": svc, "service_description": sv["description"]}
    (out / "catalogs.json").write_text(json.dumps(catalogs, indent=1, ensure_ascii=False))
    describe = [{"job": "describe", "key": f"{cat}::{lab}", "catalog": cat, "label": lab, "examples": v["examples"]}
                for cat, labs in catalogs.items() for lab, v in labs.items() if "description" not in v]

    # ---- translation jobs
    jobs = []
    by_src: dict[str, list] = defaultdict(list)
    for i in items:
        if i["split"] == "train" and i["lang"] == "en-US" and i["source"] in MT_PLAN:
            by_src[i["source"]].append(i)
    for src, (cap, locs) in MT_PLAN.items():
        pool = by_src[src]
        if cap and len(pool) > cap:
            pool = random.Random(f"{SEED}mt{src}").sample(pool, cap)
        for i in pool:
            for loc in locs:
                jobs.append({"job": "translate", "key": f"{i['source']}:{i['oid']}>{loc}", "split": "train",
                             "source": i["source"], "oid": i["oid"], "text": i["text"], "target_locale": loc})
    val_by: dict[str, list] = defaultdict(list)
    for r in kb:
        if r["_kbv1_split"] in ("val_fit", "val_eval", "dev") and "typo" not in r["slices"]:
            val_by[r["provenance"]["source_dataset"]].append(r)
    seen_txt = set()
    for src, cap in VAL_MT.items():
        pool = val_by[src]
        if cap and len(pool) > cap:
            pool = random.Random(f"{SEED}vmt{src}").sample(pool, cap)
        for r in pool:
            if r["context"]["text"] in seen_txt:
                continue
            seen_txt.add(r["context"]["text"])
            for loc in MT_LOCALES:
                jobs.append({"job": "translate", "key": f"kbv1:{r['id']}>{loc}", "split": "validation",
                             "source": src, "oid": r["id"], "text": r["context"]["text"], "target_locale": loc})
    # ---- description translation (translated-schema slice): sample of ToolACE tools
    tools = {}
    for i in items:
        if i["split"] == "train" and i["tools"]:
            for n, d in i["tools"]:
                if d:
                    tools.setdefault((n, d), None)
    tool_list = sorted(tools)
    tool_sample = random.Random(f"{SEED}tooldesc").sample(tool_list, min(3000, len(tool_list)))
    tdesc = [{"job": "translate_desc", "key": f"tool::{n}::{hashlib.sha1(d.encode()).hexdigest()[:10]}>{loc}", "name": n,
              "text": d, "target_locale": loc} for n, d in tool_sample for loc in MT_LOCALES]
    for name, rows in (("describe", describe), ("translate", jobs), ("translate_desc", tdesc)):
        with open(out / "gpu_jobs" / f"{name}.jsonl", "w") as f:
            for j in rows:
                f.write(json.dumps(j, ensure_ascii=False) + "\n")
    man = {"seed": SEED, "n_items": len(items),
           "by_split_source_lang": {f"{s}/{src}/{lang}": n for (s, src, lang), n in
                                    sorted(Counter((i["split"], i["source"], i["lang"]) for i in items).items())},
           "dropped_leakage": dict(dropped), "leakage_violations": rep.summary()["violations"],
           "gpu_jobs": {"describe": len(describe), "translate": len(jobs), "translate_desc": len(tdesc)},
           "translate_by_source_split": dict(Counter(f"{j['split']}/{j['source']}" for j in jobs)),
           "catalog_sizes": {c: len(v) for c, v in catalogs.items()}, "unique_train_tools": len(tool_list)}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
