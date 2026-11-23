"""Krun One v1 proxy data: the v0.3 corpus converted to krun.decision.v1 (+ fr-FR, + multi-label mechanics).

This is NOT Krun Decisions v1 (docs/KRUN_DECISIONS_V1_SPEC.md). It only exercises the proxy's mechanics
(serialization, heads, losses, answerability, multi-label, French and cross-language inputs) with licensed data
the project already uses:

* choice   : data/v0/train.jsonl (subsample) and val_tools / val_hard (validation), converted verbatim;
* fr-FR    : for every converted MASSIVE en-US decision, a copy whose context is the *parallel* fr-FR utterance
             (same MASSIVE id -> same group -> same split; options and gold unchanged). MASSIVE fr-FR @ ed58ac42,
             CC-BY-4.0. Split source: MASSIVE train for train decisions, MASSIVE validation for validation ones;
* noul/score: data/primitives_v2 train / val;
* multi    : deterministic proxy-only construction (no LLM): two MASSIVE utterances of different intents joined
             ("A. B"), labels = both intents, candidates = both + the union of their v0 hard negatives.
             Tagged transforms=["proxy:concat2"]; never a Krun Decisions v1 source.

Leakage: every output record is checked against the protected sets (KrunBench v0 test incl. BFCL, holdout_tools,
primitives test + mixed test) with krun_one.v1.leakage (group, exact, near-duplicate, protected tool strings for
BFCL). Violating records are dropped and reported (the v0 corpus has some: see leakage_report.json);
the build aborts if anything remains.

    uv run python scripts/v1/build_proxy_data.py --train-choice 60000
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from krun_one.v1 import leakage, schema
from krun_one.v1.tools import TrainToolIndex

ROOT = Path(__file__).resolve().parents[2]
KB = ROOT.parents[1] / "benchmarks/krunbench/data/v0"
HUB = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
MASSIVE_REV = "ed58ac423a2f4121720918bf5301577edce4ffd3"
OUT = ROOT / "data/v1_proxy"


def jl(path: Path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def massive_fr(split: str) -> dict[str, str]:
    f = glob.glob(str(HUB / f"datasets--AmazonScience--massive/snapshots/{MASSIVE_REV}/fr-FR/{split}/*.parquet"))[0]
    t = pq.read_table(f)
    return dict(zip(map(str, t.column("id").to_pylist()), t.column("utt").to_pylist()))


def fr_copy(r: dict, fr: dict[str, str]) -> dict | None:
    oid = str(r["provenance"]["original_id"]).split("-")[-1] if r["provenance"].get("original_id") else None
    text = fr.get(oid) if oid else None
    if not text:
        return None
    c = json.loads(json.dumps(r))
    c["id"] = r["id"] + "/fr-FR"
    c["context"] = {"text": text, "language": "fr-FR"}
    c["provenance"]["transforms"] = c["provenance"]["transforms"] + ["massive:parallel_locale:fr-FR"]
    return schema.derive_fields(c)


def multi_records(choice: list[dict], n: int, rng: random.Random, split: str) -> list[dict]:
    pool = [r for r in choice if r["provenance"]["source_dataset"] == "massive" and r["answerable"]
            and r["context"]["language"] == "en-US" and "permutation" not in r["slices"]]
    out = []
    for i in range(n * 3):
        if len(out) >= n:
            break
        a, b = rng.sample(pool, 2)
        la, lb = a["target"]["labels"][0], b["target"]["labels"][0]
        if la == lb:
            continue
        negs = [c for c in a["candidates"] + b["candidates"] if c["id"] not in (la, lb)]
        seen, cands = set(), []
        for c in [x for x in a["candidates"] if x["id"] == la] + [x for x in b["candidates"] if x["id"] == lb] + negs:
            if c["id"] not in seen:
                seen.add(c["id"])
                cands.append({"id": c["id"], "label": c["id"], "description": None, "language": "en"})
        cands = cands[: rng.choice([4, 8, 16])]
        rng.shuffle(cands)
        empty = rng.random() < 0.1   # some all-negative multi decisions: gold removed
        if empty:
            cands = [c for c in cands if c["id"] not in (la, lb)]
            if len(cands) < 2:
                continue
        labels = [] if empty else [x for x in (la, lb) if x in {c["id"] for c in cands}]
        g = "multi:" + hashlib.sha256((a["split_group_id"] + "|" + b["split_group_id"]).encode()).hexdigest()[:16]
        out.append(schema.derive_fields({
            "schema_version": schema.SCHEMA_VERSION, "id": f"v1proxy/{split}/multi/{i}", "split": split,
            "task": {"type": "multi", "instructions": "Select every option that applies to the user message.",
                     "instructions_language": "en", "family": "intent"},
            "context": {"text": a["context"]["text"].rstrip(".?! ") + ". " + b["context"]["text"], "language": "en-US"},
            "candidates": cands, "target": {"labels": labels, "distribution": None}, "answerable": bool(labels),
            "semantic_group_id": g, "split_group_id": g,
            # the two source groups must stay in this split: record them for the group check
            "candidate_set": {"strategy": "hard_v1", "k": len(cands)},
            "slices": ["gold_removed"] if empty else [], "minimal_pair": None,
            "provenance": {"source_dataset": "massive", "hf_id": "AmazonScience/massive", "revision": MASSIVE_REV,
                           "license": "CC-BY-4.0", "original_split": split, "original_id": None,
                           "transforms": ["proxy:concat2"], "synthetic": None},
            "metadata": {"source_groups": [a["split_group_id"], b["split_group_id"]]},
        }))
    return out


def _tool_of(r: dict) -> tuple[str, str | None] | None:
    """The tool a validation record is *about*: choice -> gold tool; noul tool proposition -> its tool."""
    if r["task"]["type"] == "choice" and r["answerable"] and r["task"].get("family") == "tool":
        gold = r["target"]["labels"][0]
        c = next(c for c in r["candidates"] if c["id"] == gold)
        return c.get("label") or c["id"], c.get("description")
    tool = (r["provenance"].get("tool") if isinstance(r.get("provenance"), dict) else None) or \
        (r.get("metadata") or {}).get("tool")
    if r["task"]["type"] == "noul" and tool:
        return tool, (r.get("metadata") or {}).get("tool_description")
    return None


def mark_unseen_tools(train: list[dict], val: list[dict]) -> dict:
    """Recompute the unseen_tool slice against THIS training set (gold + distractors, exact or equivalent)."""
    idx = TrainToolIndex().add_v1_records(train)
    stats = defaultdict(int)
    for r in val:
        t = _tool_of(r)
        sl = set(r.get("slices") or []) - {"unseen_tool"}
        if t is not None:
            st = idx.status(*t)
            r.setdefault("metadata", {})["tool_status"] = st
            stats[f"{r['task']['type']}/{st}"] += 1
            if st == "unseen":
                sl.add("unseen_tool")
        r["slices"] = sorted(sl)
    return dict(stats)


def check_unseen_tool_invariants(train: list[dict], val: list[dict]) -> dict:
    """Invariants (build aborts otherwise): the unseen-tool slice is non-empty for choice AND noul, and no unseen
    tool (by exact key or equivalent description) is present on the training side as gold or distractor."""
    idx = TrainToolIndex().add_v1_records(train)
    n = defaultdict(int)
    for r in val:
        if "unseen_tool" in r["slices"]:
            n[r["task"]["type"]] += 1
            t = _tool_of(r)
            if t is None or not idx.is_unseen(*t):
                raise SystemExit(f"unseen-tool invariant violated: {r['id']}")
    if n["choice"] == 0 or n["noul"] == 0:
        raise SystemExit(f"unseen-tool slice empty: {dict(n)}")
    return {"unseen_tool_slice": dict(n)}


def excluded_sources(registry: str) -> set[str]:
    import yaml

    reg = yaml.safe_load(Path(registry).read_text())
    return {k for k, v in (reg.get("decision_sources") or {}).items() if str(v.get("status", "")).startswith("excluded")}


def apply_exclusions(train: list[dict], val: list[dict], registry: str) -> tuple[list[dict], list[dict], dict]:
    """Remove excluded sources everywhere: records whose context comes from them, and their TOOLS used as
    candidates in other sources' decisions (v0 drew distractors from a mixed pool). A tool counts as belonging to an
    excluded source if it is a candidate in any excluded-source decision and is never the gold of a kept decision.
    Decisions that lose their gold or fall below 2 candidates are dropped."""
    from krun_one.v1.tools import tool_key

    excl = excluded_sources(registry)
    allr = train + val
    ex_keys, kept_gold = set(), set()
    for r in allr:
        src = r["provenance"]["source_dataset"]
        cands = {c["id"]: c for c in r["candidates"]}
        if src in excl:
            ex_keys |= {tool_key(c.get("label") or c["id"], c.get("description")) for c in r["candidates"] if c.get("description")}
        elif r["task"]["type"] == "choice" and r["answerable"]:
            g = cands[r["target"]["labels"][0]]
            if g.get("description"):
                kept_gold.add(tool_key(g.get("label") or g["id"], g.get("description")))
    ex_tools = ex_keys - kept_gold
    stats = {"excluded_sources": sorted(excl), "excluded_tool_keys": len(ex_tools),
             "dropped_context": 0, "dropped_after_candidate_removal": 0, "candidates_removed": 0}

    def keep(r):
        if r["provenance"]["source_dataset"] in excl:
            stats["dropped_context"] += 1
            return None
        if r["task"]["type"] not in ("choice", "multi"):
            tool = (r.get("metadata") or {}).get("tool")
            if tool and tool_key(tool, (r.get("metadata") or {}).get("tool_description")) in ex_tools:
                stats["dropped_context"] += 1
                return None
            return r
        bad = {c["id"] for c in r["candidates"] if c.get("description")
               and tool_key(c.get("label") or c["id"], c["description"]) in ex_tools}
        if not bad:
            return r
        labels = r["target"]["labels"]
        if any(l in bad for l in labels) or len(r["candidates"]) - len(bad) < 2:
            stats["dropped_after_candidate_removal"] += 1
            return None
        r = json.loads(json.dumps(r))
        r["candidates"] = [c for c in r["candidates"] if c["id"] not in bad]
        r["candidate_set"] = dict(r.get("candidate_set") or {}, k=len(r["candidates"]))
        r.setdefault("provenance", {}).setdefault("transforms", []).append("v1:excluded_source_candidates_removed")
        stats["candidates_removed"] += len(bad)
        return schema.derive_fields(r)

    tr = [x for x in (keep(r) for r in train) if x is not None]
    va = [x for x in (keep(r) for r in val) if x is not None]
    stats.update({"train_before": len(train), "train_after": len(tr), "val_before": len(val), "val_after": len(va)})
    return tr, va, stats


def protected_records() -> list[dict]:
    prot = []
    for f in [KB / "banking77/test.jsonl", KB / "massive/test.jsonl", KB / "bfcl/test.jsonl", KB / "hwu64/test.jsonl"]:
        if f.exists():
            prot += [schema.from_krunbench_v0(d, split="protected") for d in jl(f)]
    prot += [schema.from_krunbench_v0(d, split="protected") for d in jl(ROOT / "data/v0/holdout_tools.jsonl")]
    for f in ("noul_test.jsonl", "score_test.jsonl"):
        prot += [schema.from_primitive_v0(d, split="protected") for d in jl(ROOT / "data/primitives_v2" / f)]
    return prot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-choice", type=int, default=60000)
    ap.add_argument("--train-noul", type=int, default=20000)
    ap.add_argument("--train-score", type=int, default=15000)
    ap.add_argument("--train-multi", type=int, default=8000)
    ap.add_argument("--val-per-kind", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--sources", default=str(ROOT / "configs/v1/sources.yaml"),
                    help="registry; sources with status excluded_* are removed (as context, gold AND distractor)")
    args = ap.parse_args()
    out_dir = Path(args.out)
    rng = random.Random(args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_all = [schema.from_krunbench_v0(d, split="train") for d in jl(ROOT / "data/v0/train.jsonl")]
    rng.shuffle(train_all)
    train = train_all[: args.train_choice]
    val = [schema.from_krunbench_v0(d, split="validation") for d in jl(ROOT / "data/v0/val_tools.jsonl")]
    val_hard = [schema.from_krunbench_v0(d, split="validation") for d in jl(ROOT / "data/v0/val_hard.jsonl")]
    rng.shuffle(val_hard)
    val += val_hard[: args.val_per_kind]
    fr = {sp: massive_fr(sp) for sp in ("train", "validation")}  # keyed by the MASSIVE split the item came from

    def add_fr(recs):
        return [c for c in (fr_copy(r, fr[r["provenance"]["original_split"]]) for r in recs
                            if r["provenance"]["source_dataset"] == "massive" and r["context"]["language"] == "en-US"
                            and r["provenance"].get("original_split") in fr) if c]
    train += add_fr(train)
    val += add_fr(val)
    for kind, n in (("noul", args.train_noul), ("score", args.train_score)):
        tr = [schema.from_primitive_v0(d, split="train") for d in jl(ROOT / f"data/primitives_v2/{kind}_train.jsonl")]
        va = [schema.from_primitive_v0(d, split="validation") for d in jl(ROOT / f"data/primitives_v2/{kind}_val.jsonl")]
        rng.shuffle(tr)
        rng.shuffle(va)
        train += tr[:n]
        val += va[: args.val_per_kind]
    train, val, excl_stats = apply_exclusions(train, val, args.sources)
    print("excluded sources:", json.dumps(excl_stats), flush=True)
    train += multi_records(train, args.train_multi, rng, "train")
    val += multi_records(val, args.val_per_kind // 2, rng, "validation")

    tool_stats = mark_unseen_tools(train, val)

    errors = defaultdict(int)
    for r in train + val:
        for e in schema.validate(r):
            errors[e] += 1
    if errors:
        raise SystemExit(f"schema errors: {dict(errors)}")
    # within-corpus: no group in two splits (multi records also carry their source groups)
    expanded = train + val + [{"id": r["id"] + "#src" + str(i), "split": r["split"], "split_group_id": g}
                              for r in train + val for i, g in enumerate(r["metadata"].get("source_groups", []))]
    rep_groups = leakage.check_split_groups(expanded)
    prot = protected_records()
    bfcl = [p for p in prot if p["provenance"]["source_dataset"] == "bfcl"]

    def audit(recs):
        rep = leakage.check_leakage(recs, prot, near_threshold=0.9)
        # BFCL tools must stay unseen: protected tool strings may not appear as candidates anywhere
        rep_tools = leakage.check_leakage(recs, bfcl, near_threshold=1.01, protect_candidates=True)
        return rep, rep_tools

    rep, rep_tools = audit(train + val)
    drop = {v["id"] for r in (rep, rep_tools) for k in ("group", "exact", "near", "candidate") for v in r.violations.get(k, [])}
    before = {"train": len(train), "validation": len(val)}
    train = [r for r in train if r["id"] not in drop]
    val = [r for r in val if r["id"] not in drop]
    rep2, rep_tools2 = audit(train + val)   # must be clean now
    report = {"groups": rep_groups.summary(), "v0_corpus_findings": {"protected": rep.summary(), "bfcl_tools": rep_tools.summary()},
              "dropped": len(drop), "before": before, "after_drop": {"protected": rep2.summary(), "bfcl_tools": rep_tools2.summary()}}
    (out_dir / "leakage_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"found": {"protected": rep.summary()["violations"], "bfcl_tools": rep_tools.summary()["violations"]},
                      "dropped": len(drop), "after": {"protected": rep2.summary()["violations"],
                                                      "bfcl_tools": rep_tools2.summary()["violations"]}}))
    if not (rep_groups.ok and rep2.ok and rep_tools2.ok):
        raise SystemExit("LEAKAGE remains after dropping: see data/v1_proxy/leakage_report.json")

    tool_stats.update(check_unseen_tool_invariants(train, val))
    manifest = {"seed": args.seed, "args": vars(args), "files": {}, "unseen_tools": tool_stats}
    for name, recs in (("train", train), ("validation", val)):
        rng.shuffle(recs)
        path = out_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts = defaultdict(int)
        for r in recs:
            counts[f"{r['task']['type']}/{r['context']['language']}/{'ans' if r['answerable'] else 'unans'}"] += 1
        manifest["files"][name] = {"n": len(recs), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                   "counts": dict(sorted(counts.items()))}
    manifest["excluded_sources"] = excl_stats
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v["n"] for k, v in manifest["files"].items()}))


if __name__ == "__main__":
    main()
