"""KrunBench v1 builder: dev / val_fit / val_eval / final_protected + protected manifest + zero-leakage validation.

Rule for final_protected: only semantic groups that NO previous Krun evaluation or training has opened.
  * MASSIVE / BANKING77 / HWU64 / BFCL / v0 holdout_tools / primitives v2 test were opened by v0/v0.3 -> KrunBench v0,
    REGRESSION ONLY (never final). MASSIVE fr-FR is a translation of opened groups -> not final either.
  * Never-opened sources used here:
      CLINC150 plus (CC-BY-3.0)   validation -> dev/val_fit/val_eval, test -> final (incl. out-of-scope = far/near-OOD)
      SNIPS (Apache-2.0 mirror)   test -> final
      Bitext (CDLA-Sharing-1.0)   3 intents held out entirely -> unseen-label slices (val / final by group)
      ToolACE (Apache-2.0)        dialogues v0 never used: first turn with NO call (native irrelevance) and with
                                  SEVERAL calls (multi-label tool selection); tools absent from every training pool
                                  -> unseen_tool slice
PT/ES/FR (pt-BR, pt-PT, es-ES, es-419, fr-FR, fr-CA) versions of final items are emitted as TRANSLATION JOBS
(`pending_translation.jsonl`, protected, human review required) — they are not part of any split until reviewed.

Protection: every final_protected record is registered in `protected_manifest.json` (normalized-text hashes, semantic /
split group ids, tool ids/names/descriptions of unseen tools). `--check` validates LEAKAGE = 0 of any training-side
file against it (exact, normalized, group ids, near-duplicate, tool id / name / description / equivalent description)
and aborts on any violation. The build itself aborts if its own training-pool references leak.

    uv run python scripts/v1/build_krunbench_v1.py --out data/krunbench_v1
    uv run python scripts/v1/build_krunbench_v1.py --check data/v1_probe/train.jsonl [more training-side files]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from krun_one.sources import _CALL_NAME, _TOOLACE_TOOLS, _clean, _tools_from_json
from krun_one.v1 import leakage, schema
from krun_one.v1.tools import TrainToolIndex, tool_key

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260926
KS = (2, 4, 8, 16, 32, 64, 128)
BITEXT_HELDOUT_INTENTS = ("check_refund_policy", "delivery_period", "newsletter_subscription")
REV = {"google/civil_comments": "f2970eb3a55777454c94069077cc8d9b5866312d", "clinc/clinc_oos": "155b9c7104", "benayas/snips": "16915b8957", "Team-ACE/ToolACE": "6bda777c88d21e5a204703c1ee45597a8fa4f734",
       "bitext/Bitext-customer-support-llm-chatbot-training-dataset": "430d1a89bd"}
LICENSE = {"clinc150": "CC-BY-3.0", "snips": "Apache-2.0", "toolace": "Apache-2.0", "bitext": "CDLA-Sharing-1.0",
           "civil_comments": "CC0-1.0"}
TARGET_LOCALES = ("pt-BR", "pt-PT", "es-ES", "es-419", "fr-FR", "fr-CA")
# every training-side file known at build time; protected items overlapping them (text / near-dup) are dropped
TRAINING_SIDE = [ROOT / "data/v1_probe/train.jsonl"]


def hf(repo, f):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, f, repo_type="dataset", revision=REV[repo])


def gid(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16]


def split_of(group: str, source_split: str) -> str:
    """test -> final_protected; validation -> dev (10%) / val_fit (45%) / val_eval (45%) by group hash."""
    if source_split == "test":
        return "final_protected"
    h = int(hashlib.sha256(("kbv1|" + group).encode()).hexdigest()[:8], 16) / 16**8
    return "dev" if h < 0.10 else ("val_fit" if h < 0.55 else "val_eval")


def humanize(x: str) -> str:
    return " ".join(x.replace("_", " ").split())


# ------------------------------------------------------------------------------------------------ sources
def load_clinc():
    names = None
    items = []
    for split in ("validation", "test"):
        t = pq.read_table(hf("clinc/clinc_oos", f"plus/{split}-00000-of-00001.parquet"))
        if names is None:
            meta = json.loads(t.schema.metadata[b"huggingface"])
            names = meta["info"]["features"]["intent"]["names"]
        for i, (txt, lab) in enumerate(zip(t.column("text").to_pylist(), t.column("intent").to_pylist())):
            items.append({"source": "clinc150", "src_split": split, "oid": f"{split}-{i}", "text": txt,
                          "label": names[lab], "oos": names[lab] == "oos"})
    return items, [n for n in names if n != "oos"]


def load_snips():
    t = pq.read_table(hf("benayas/snips", "data/test-00000-of-00001.parquet"))
    c = t.column_names
    return [{"source": "snips", "src_split": "test", "oid": f"test-{i}", "text": x, "label": y, "oos": False}
            for i, (x, y) in enumerate(zip(t.column(c[0]).to_pylist(), t.column(c[1]).to_pylist()))]


def load_bitext_heldout():
    import csv

    rows = list(csv.DictReader(open(hf("bitext/Bitext-customer-support-llm-chatbot-training-dataset",
                                        "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"), newline="")))
    intents = sorted({r["intent"] for r in rows})
    held = [{"source": "bitext", "src_split": "test" if int(gid(r["instruction"]), 16) % 2 else "validation",
             "oid": f"row-{i}", "text": r["instruction"], "label": r["intent"], "oos": False}
            for i, r in enumerate(rows) if r["intent"] in BITEXT_HELDOUT_INTENTS]
    return held, intents


def load_toolace_unused():
    data = json.loads(Path(hf("Team-ACE/ToolACE", "data.json")).read_text())
    none_items, multi_items = [], []
    for i, r in enumerate(data):
        m = _TOOLACE_TOOLS.search(r.get("system", ""))
        conv = r.get("conversations", [])
        if not m or len(conv) < 2 or conv[0]["from"] != "user" or conv[1]["from"] != "assistant":
            continue
        try:
            tools = _tools_from_json(json.loads(m.group(1)))
        except json.JSONDecodeError:
            continue
        if not tools or len(tools) < 2:
            continue
        q, a = _clean(conv[0]["value"]), conv[1]["value"].strip()
        if not q or any("一" <= ch <= "鿿" for ch in q):
            continue
        base = {"source": "toolace", "oid": f"toolace-{i}", "text": q, "tools": [(t.name, t.description) for t in tools]}
        if not (a.startswith("[") and a.endswith("]")):
            none_items.append(dict(base, calls=[]))
            continue
        called = {n.strip() for n in _CALL_NAME.findall(a)}
        if len(called) >= 2 and called <= {t.name for t in tools} and len(called) < len(tools):
            multi_items.append(dict(base, calls=sorted(called)))
    return none_items, multi_items


# ------------------------------------------------------------------------------------------------ decisions
def base_record(item, split, task, candidates, labels, slices, strategy, extra_meta=None) -> dict:
    g = f"kbv1:{item['source']}:{item['oid']}"
    rec = {"schema_version": schema.SCHEMA_VERSION, "id": f"kbv1/{split}/{item['source']}/{item['oid']}/k{len(candidates)}",
           "split": "protected" if split == "final_protected" else "validation", "task": task,
           "context": {"text": item["text"], "language": "en-US"}, "candidates": candidates,
           "target": {"labels": labels, "distribution": None}, "answerable": bool(labels),
           "semantic_group_id": g, "split_group_id": g,
           "candidate_set": {"strategy": strategy, "k": len(candidates)}, "slices": sorted(set(slices)), "minimal_pair": None,
           "provenance": {"source_dataset": item["source"], "revision": REV.get({"clinc150": "clinc/clinc_oos", "snips": "benayas/snips",
                          "toolace": "Team-ACE/ToolACE", "bitext": "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
                          "civil_comments": "google/civil_comments"}[item["source"]]),
                          "license": LICENSE[item["source"]], "original_split": item.get("src_split"), "original_id": item["oid"],
                          "transforms": ["krunbench_v1"], "synthetic": None},
           "metadata": {"kbv1_split": split, **(extra_meta or {})}}
    return schema.derive_fields(rec)


class Neighbours:
    """Hard negatives by char-TF-IDF similarity of candidate texts (eval-side only; protected data never feeds a
    training-side mining index)."""

    def __init__(self, labels: list[str], texts: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.labels = labels
        m = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(texts)
        self.sim = (m @ m.T).toarray()
        self.pos = {l: i for i, l in enumerate(labels)}

    def hard(self, label: str, n: int, rng: random.Random, exclude=()) -> list[str]:
        order = np.argsort(-self.sim[self.pos[label]])
        pool = [self.labels[j] for j in order if self.labels[j] != label and self.labels[j] not in exclude][: max(2 * n, n + 4)]
        rng.shuffle(pool)
        return pool[:n]


def intent_decisions(items, labels, rng, family_name, instructions):
    nb = Neighbours(labels, [humanize(l) for l in labels])
    out = []
    for it in items:
        split = split_of(f"{it['source']}:{it['oid']}", it["src_split"])
        k = rng.choice([k for k in KS if k <= len(labels)])
        if it["oos"]:
            negs = rng.sample(labels, k)                                   # far-OOD: nothing applies
            cands, lab, sl, strat = negs, [], ["ood_far", "native_irrelevance"], "random"
        else:
            gold_removed = rng.random() < 0.15
            hard = rng.random() < 0.6
            negs = nb.hard(it["label"], k - (0 if gold_removed else 1), rng) if hard else \
                rng.sample([l for l in labels if l != it["label"]], k - (0 if gold_removed else 1))
            cands = negs if gold_removed else negs + [it["label"]]
            rng.shuffle(cands)
            lab = [] if gold_removed else [it["label"]]
            sl = (["gold_removed", "ood_near"] if gold_removed else []) + (["near_duplicate"] if hard else [])
            strat = "hard_v2" if hard else "random"
        extra = {"unseen_label": it["source"] == "bitext"}
        if it["source"] == "bitext":
            sl = sl + ["unseen_label"]
        c = [{"id": l, "label": l, "description": None, "language": "en"} for l in cands]
        out.append(base_record(it, split, {"type": "choice", "instructions": instructions, "instructions_language": "en",
                                           "family": family_name}, c, lab, sl, strat, extra))
    return out


def tool_decisions(none_items, multi_items, train_tools: TrainToolIndex, train_names: set, rng):
    out, unseen_tools = [], {}
    instr = "Choose the function that should be called to fulfil the user request."
    train_pool = []
    for it in none_items + multi_items:
        # ToolACE dialogues never used by v0: 40% -> TRAINING pool (real none-applicable / multi-call examples for
        # Krun Decisions v1; NOT protected), 20% -> validation, 40% -> final_protected (by item hash)
        h = int(gid("toolace-alloc", it["oid"]), 16) / 16**16
        if h < 0.40:
            train_pool.append(it)
            continue
        split = split_of(f"toolace:{it['oid']}", "validation" if h < 0.60 else "test")
        cands = [{"id": n, "label": n, "description": d, "language": "en"} for n, d in it["tools"]]
        unseen = [c for c in cands if train_tools.is_unseen(c["label"], c["description"])
                  and leakage.normalize(c["label"]) not in train_names]
        sl = []
        if unseen:
            sl.append("unseen_tool")
            if split == "final_protected":
                for c in unseen:
                    unseen_tools[tool_key(c["label"], c["description"])] = c
        if it["calls"]:
            out.append(base_record(it, split, {"type": "multi", "instructions": "Select every function that should be called "
                                               "to fulfil the user request.", "instructions_language": "en", "family": "tool"},
                                   cands, list(it["calls"]), sl, "native"))
        else:
            out.append(base_record(it, split, {"type": "choice", "instructions": instr, "instructions_language": "en",
                                               "family": "tool"}, cands, [], sl + ["native_irrelevance"], "native"))
    return out, unseen_tools, train_pool


CIVIL_REV = "f2970eb3a55777454c94069077cc8d9b5866312d"
TOX_LEVELS = ["not toxic at all", "slightly toxic", "moderately toxic", "very toxic", "extremely toxic"]


def civil_noul_score(rng, n_noul=1500, n_score=1000) -> list[dict]:
    """Noul + Score suites from Civil Comments TEST rows never used by any v0/v0.3 file (CC0-1.0)."""
    import importlib.util

    from huggingface_hub import hf_hub_download

    spec = importlib.util.spec_from_file_location("ppd", ROOT / "scripts/prepare_primitives_data.py")
    ppd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ppd)
    used = set()
    for f in (ROOT / "data/primitives_v2").glob("*.jsonl"):
        for l in open(f):
            used.add(leakage.normalize(json.loads(l)["context"]))
    t = pq.read_table(hf_hub_download("google/civil_comments", "data/test-00000-of-00001.parquet", repo_type="dataset",
                                      revision=CIVIL_REV)).to_pandas()
    t["row"] = range(len(t))
    t = t[t.text.str.len().between(30, 600)]
    t = t[~t.text.map(lambda x: leakage.normalize(x) in used)]
    out = []
    props = list(ppd.CIVIL_PROPS.items())
    rows = t.sample(n=min(len(t), 60000), random_state=SEED)
    # noul: balance positives/negatives per proposition
    per = n_noul // len(props)
    for attr, question in props:
        pos = rows[rows[attr] >= 0.5].head(per // 2)
        neg = rows[(rows[attr] < 0.1)].sample(n=per - len(pos), random_state=SEED)
        for _, r in list(pos.iterrows()) + list(neg.iterrows()):
            item = {"source": "civil_comments", "oid": f"test-{r.row}/{attr}", "text": r.text}
            split = split_of(f"civil:{r.row}", "test" if int(gid("civil", r.row), 16) % 5 >= 2 else "validation")
            rec = base_record(item, split, {"type": "noul", "instructions": question, "instructions_language": "en",
                                            "criteria": ppd.CIVIL_CRITERIA, "family": "proposition"},
                              [{"id": "false"}, {"id": "true"}], ["true" if r[attr] >= 0.5 else "false"], [], "fixed",
                              {"attribute": attr, "score": float(r[attr])})
            rec["semantic_group_id"] = rec["split_group_id"] = f"kbv1:civil:{r.row}"
            out.append(rec)
    used_rows = {int(x["provenance"]["original_id"].split("/")[0].split("-")[1]) for x in out}
    sc = rows[~rows.row.isin(used_rows)]
    bins = [0.1, 0.3, 0.5, 0.7]
    by_level = [sc[(sc.toxicity >= lo) & (sc.toxicity < hi)] for lo, hi in zip([0] + bins, bins + [1.01])]
    for lvl, dfl in enumerate(by_level):
        for _, r in dfl.head(n_score // 5).iterrows():
            item = {"source": "civil_comments", "oid": f"test-{r.row}/toxicity_level", "text": r.text}
            split = split_of(f"civil:{r.row}", "test" if int(gid("civil", r.row), 16) % 5 >= 2 else "validation")
            rec = base_record(item, split, {"type": "score", "instructions": "How toxic is this comment?", "instructions_language": "en",
                                            "family": "rating"},
                              [{"id": str(i), "label": x} for i, x in enumerate(TOX_LEVELS)], [str(lvl)], [], "fixed",
                              {"toxicity": float(r.toxicity)})
            rec["semantic_group_id"] = rec["split_group_id"] = f"kbv1:civil:{r.row}"
            out.append(rec)
    return out


def typo_variants(records, rng):
    """Deterministic rule typos of final/val items (same split & group; the clean item's semantic group)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("sp", ROOT / "scripts/v1/synthetic_pilot.py")
    sp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sp)
    out = []
    for r in rng.sample(records, min(len(records), 1500)):
        t = json.loads(json.dumps(r))
        t["id"] = r["id"] + "/typo"
        t["context"] = dict(t["context"], text=sp.rule_typos(r["context"]["text"], rng), noise=["typo"])
        t["slices"] = sorted(set(t["slices"]) | {"typo"})
        t["provenance"]["transforms"] = t["provenance"]["transforms"] + ["rule:typos-v1"]
        out.append(schema.derive_fields(t))
    return out


# ------------------------------------------------------------------------------------------------ protection
def protected_manifest(final: list[dict], unseen_tools: dict) -> dict:
    return {"version": "krunbench.v1", "n_records": len(final),
            "text_hashes": sorted({leakage.text_hash(r["context"]["text"]) for r in final}),
            "groups": sorted({g for r in final for g in (r["split_group_id"], r["semantic_group_id"])}),
            "unseen_tools": [{"id": c["id"], "name": c["label"], "description": c["description"]} for c in unseen_tools.values()],
            "never_use_for": ["training", "hard-negative mining", "embedding retrieval", "translation generation",
                              "synthetic generation", "teacher generation", "similarity lookup", "corpus selection"]}


def check_training_side(paths: list[Path], bench_dir: Path) -> dict:
    """LEAKAGE must be 0: exact/normalized text, groups, near-duplicate, unseen-tool id/name/description/equivalent."""
    final = [json.loads(l) for l in open(bench_dir / "final_protected.jsonl")]
    man = json.loads((bench_dir / "protected_manifest.json").read_text())
    train = [json.loads(l) for p in paths for l in open(p)]
    rep = leakage.check_leakage(train, final, near_threshold=0.9)
    viol = {k: len(v) for k, v in rep.violations.items() if v}
    idx = TrainToolIndex()
    names = set()
    for t in man["unseen_tools"]:
        idx.add(t["name"], t["description"])
        names.add(leakage.normalize(t["name"]))
    tool_hits = defaultdict(int)
    for r in train:
        for c in r.get("candidates") or []:
            if not c.get("description"):
                continue
            nm = c.get("label") or c["id"]
            st = idx.status(nm, c["description"])
            if st != "unseen":
                tool_hits[f"tool_{st}"] += 1
            elif leakage.normalize(nm) in names:
                tool_hits["tool_name"] += 1
    viol.update(tool_hits)
    return {"training_side_files": [str(p) for p in paths], "n_training_side": len(train), "n_protected": len(final),
            "violations": viol, "LEAKAGE": sum(viol.values())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/krunbench_v1"))
    ap.add_argument("--check", nargs="*", help="training-side jsonl files to validate against the protected manifest")
    args = ap.parse_args()
    out = Path(args.out)
    if args.check:
        res = check_training_side([Path(p) for p in args.check], out)
        (ROOT / "results/v1/krunbench_v1_leakage_check.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        raise SystemExit(0 if res["LEAKAGE"] == 0 else 1)
    rng = random.Random(SEED)
    out.mkdir(parents=True, exist_ok=True)
    clinc, clinc_labels = load_clinc()
    snips = load_snips()
    bitext, bitext_intents = load_bitext_heldout()
    none_items, multi_items = load_toolace_unused()
    # training pools (everything the model may train on) define "seen" tools
    train_tools = TrainToolIndex()
    train_side = [json.loads(l) for p in TRAINING_SIDE for l in open(p)]
    train_tools.add_v1_records(train_side)
    train_names = {leakage.normalize(c.get("label") or c["id"]) for r in train_side for c in r["candidates"] if c.get("description")}
    for it in none_items + multi_items:        # tools of the ToolACE train-pool share become "seen" (conservative)
        if int(gid("toolace-alloc", it["oid"]), 16) / 16**16 < 0.40:
            for n, d in it["tools"]:
                train_tools.add(n, d)
                train_names.add(leakage.normalize(n))
    recs = []
    recs += intent_decisions(clinc, clinc_labels, rng, "intent", "Choose the option that best describes the intent of the user message.")
    recs += intent_decisions(snips, sorted({s["label"] for s in snips}), rng, "intent",
                             "Choose the option that best describes the intent of the user message.")
    recs += intent_decisions(bitext, bitext_intents, rng, "intent",
                             "Choose the customer-support intent of the message.")
    tool_recs, unseen_tools, toolace_train_pool = tool_decisions(none_items, multi_items, train_tools, train_names, rng)
    with open(out / "train_pool_toolace_unused.jsonl", "w") as f:     # NOT protected: Krun Decisions v1 may use it
        for it in toolace_train_pool:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    recs += tool_recs
    recs += civil_noul_score(rng)
    recs += typo_variants([r for r in recs if r["task"]["type"] in ("choice", "multi")], rng)
    overlap = leakage.check_leakage(recs, train_side, near_threshold=0.9)
    drop = {v["id"] for k in ("exact", "near", "group") for v in overlap.violations.get(k, [])}
    dropped_overlap = len(drop)
    recs = [r for r in recs if r["id"] not in drop]
    errs = Counter(e for r in recs for e in schema.validate(r))
    if errs:
        raise SystemExit(f"schema errors: {dict(errs)}")
    by = defaultdict(list)
    for r in recs:
        by[r["metadata"]["kbv1_split"]].append(r)
    out.mkdir(parents=True, exist_ok=True)
    for split, rs in by.items():
        with open(out / f"{split}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # translation jobs for the multilingual/cross-language final slices (protected; human review before use)
    jobs = [{"id": r["id"] + f"/{loc}", "source_id": r["id"], "target_locale": loc, "text": r["context"]["text"],
             "keep_schema_language": "en", "status": "pending_translation_and_human_review"}
            for r in by["final_protected"] if r["task"]["family"] in ("intent", "proposition") and "typo" not in r["slices"]
            for loc in TARGET_LOCALES][:12000]
    with open(out / "pending_translation.jsonl", "w") as f:
        for j in jobs:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    man = protected_manifest(by["final_protected"], unseen_tools)
    (out / "protected_manifest.json").write_text(json.dumps(man, indent=1))
    # within-benchmark group isolation (a group never crosses dev/val_fit/val_eval/final)
    where = {}
    for r in recs:
        prev = where.setdefault(r["split_group_id"], r["metadata"]["kbv1_split"])
        if prev != r["metadata"]["kbv1_split"]:
            raise SystemExit(f"group {r['split_group_id']} crosses splits")
    stats = {s: {"n": len(rs), "by_type": dict(Counter(r["task"]["type"] for r in rs)),
                 "by_source": dict(Counter(r["provenance"]["source_dataset"] for r in rs)),
                 "slices": dict(Counter(x for r in rs for x in r["slices"])),
                 "k": dict(sorted(Counter(len(r["candidates"]) for r in rs).items())),
                 "answerable": sum(r["answerable"] for r in rs)} for s, rs in by.items()}
    stats["pending_translation_jobs"] = len(jobs)
    stats["dropped_overlapping_training_side"] = dropped_overlap
    stats["toolace_train_pool_items"] = len(toolace_train_pool)
    stats["unseen_tools_protected"] = len(unseen_tools)
    stats["sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob("*.jsonl"))}
    (ROOT / "results/v1/krunbench_v1_manifest.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({s: v["n"] if isinstance(v, dict) and "n" in v else v for s, v in stats.items() if s != "sha256"}, indent=1))


if __name__ == "__main__":
    main()
