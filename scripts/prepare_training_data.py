#!/usr/bin/env python
"""Build Krun One v0 training data + Krun One validation/holdout suites.

    .venv/bin/python scripts/prepare_training_data.py                  # full build (~15 min on a GTX 1650: e5 embeddings)
    .venv/bin/python scripts/prepare_training_data.py --max-items 300  # tiny smoke build into data/smoke

Outputs (data/ is gitignored and fully reproducible from configs/datasets.yaml):
    data/v0/train.jsonl           training decisions (KrunBench train intents + xLAM/ToolACE/irrelevance train tools)
    data/v0/val_hard.jsonl        KrunBench *validation* intent items with hard_v1 option sets (k=4..32) + OOD + permutations
    data/v0/val_tools.jsonl       tool validation (seen + unseen tools, removed-gold OOD, irrelevance, cross-task, permutations)
    data/v0/holdout_tools.jsonl   same construction on a disjoint holdout (reported once, never tuned on)
    data/v0/manifest.json         config, sources (revisions, licenses), skip counts, split stats, sha256 of every file
    results/dataset_stats.json    same stats (versioned)
KrunBench v0 validation/test are used verbatim by the evaluation (never rewritten here).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

import krun_one  # noqa: F401  (env var before torch)
from krun_one.dataset import (EmbeddingIndex, IntentBuilder, ToolBuilder, describe, exact_allocation, sha256_path,
                              to_example, tool_doc, with_permutations, write_json)
from krun_one.mining import IntentMiner, ToolPool, embed_texts, seeded_rng
from krun_one.sources import (TOOL_SOURCES, load_intent_items, load_toolace, load_xlam, load_xlam_irrelevance)
from krunbench.schema import iter_jsonl_dicts, write_jsonl
from krunbench.splits import normalize_text, stable_fraction
from krunbench.validation import validate_examples

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cached_embeddings(name: str, texts: list[str], cache_dir: Path) -> EmbeddingIndex:
    """Embeddings are cached by the exact text list (sha256)."""
    import hashlib

    h = hashlib.sha256("\n".join(texts).encode()).hexdigest()[:16]
    f = cache_dir / f"{name}-{h}.npy"
    if f.exists():
        emb = np.load(f)
    else:
        t0 = time.time()
        emb = embed_texts(texts)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(f, emb)
        log(f"  embedded {len(texts)} {name} texts in {time.time() - t0:.0f}s")
    return EmbeddingIndex(texts, emb)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "datasets.yaml"))
    ap.add_argument("--max-items", type=int, default=0, help="smoke: cap items per source")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = cfg["seed"]
    kb = PROJECT / cfg["krunbench_dir"]
    out = PROJECT / cfg["out_dir"] if not args.max_items else ROOT / "data" / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "data" / "emb_cache"
    manifest: dict = {"config": cfg, "seed": seed, "sources": {"tools": TOOL_SOURCES}, "skipped": {}, "notes": []}

    # ------------------------------------------------------------------ intents (KrunBench items)
    it_train = load_intent_items(kb, "train", cfg["intents"]["train_datasets"])
    it_val = load_intent_items(kb, "validation", cfg["intents"]["hard_eval_datasets"])
    if args.max_items:
        it_train = [it for i, it in enumerate(it_train) if i % max(1, len(it_train) // args.max_items) == 0]
        it_val = [it for i, it in enumerate(it_val) if i % max(1, len(it_val) // args.max_items) == 0]
    log(f"intent items: train {len(it_train)}, validation {len(it_val)} ({Counter(i.dataset for i in it_train)})")

    # ------------------------------------------------------------------ tools
    tool_items, skipped = [], {}
    for name, fn in (("xlam", load_xlam), ("toolace", load_toolace), ("xlam_irrelevance", load_xlam_irrelevance)):
        items, sk = fn()
        skipped[name] = {"kept": len(items), **sk}
        tool_items += items
    # decontamination: any query whose normalized text occurs anywhere in KrunBench v0 (all splits, incl. BFCL)
    kb_ctx = {normalize_text(d["context"]) for f in kb.glob("*/*.jsonl") for d in iter_jsonl_dicts(f)}
    by_q = defaultdict(list)
    for it in tool_items:
        by_q[normalize_text(it.query)].append(it)
    kept, dropped = [], Counter()
    for q, its in by_q.items():
        if q in kb_ctx:
            dropped["query occurs in KrunBench v0 (incl. BFCL)"] += len(its)
            continue
        # xlam-irrelevance reuses xLAM queries with a different tool set: one answerable + one irrelevance item may
        # share a query (and therefore a group / split). Several answerables must agree on the gold.
        ans_its = [it for it in its if it.gold]
        irr_its = [it for it in its if it.gold is None]
        if len({it.gold for it in ans_its}) > 1:
            dropped["same query, conflicting gold"] += len(ans_its)
        elif ans_its:
            kept.append(ans_its[0])
            dropped["duplicate query (first kept)"] += len(ans_its) - 1
        if irr_its:
            kept.append(irr_its[0])
            dropped["duplicate irrelevance query (first kept)"] += len(irr_its) - 1
    skipped["tool_dedup_decontamination"] = dict(dropped)
    tool_items = sorted(kept, key=lambda it: it.item_id)
    if args.max_items:
        tool_items = [it for i, it in enumerate(tool_items) if i % max(1, len(tool_items) // (3 * args.max_items)) == 0]
    manifest["skipped"] = skipped
    log(f"tool items after dedup/decontamination: {len(tool_items)} ({skipped['tool_dedup_decontamination']})")

    # tool splits: unseen gold tools (held out entirely), then seen items by query hash
    tc = cfg["tools"]
    gold_keys = sorted({it.tool(it.gold).key for it in tool_items if it.gold})
    unseen = {k for k in gold_keys if stable_fraction(seed, "unseen-tool", k) < tc["unseen_tool_fraction"]}
    groups = defaultdict(list)
    for it in tool_items:
        groups[it.group_id].append(it)
    for gid, its in groups.items():  # the split is decided per group (query), never per item
        gold_it = next((it for it in its if it.gold), None)
        if gold_it is None:
            f = stable_fraction(seed, "irrelevance-split", gid)
            fr = tc["irrelevance_eval_fraction"]
            split = "validation" if f < fr["validation"] else "holdout" if f < fr["validation"] + fr["holdout"] else "train"
        elif gold_it.tool(gold_it.gold).key in unseen:
            split = "validation" if stable_fraction(seed, "unseen-side", gold_it.tool(gold_it.gold).key) < 0.5 else "holdout"
            gold_it.metadata["tool_seen"] = False
        else:
            f = stable_fraction(seed, "seen-split", gid)
            fr = tc["seen_eval_fraction"]
            split = "validation" if f < fr["validation"] else "holdout" if f < fr["validation"] + fr["holdout"] else "train"
            gold_it.metadata["tool_seen"] = True
        for it in its:
            it.split = split
    # held-out tools never appear in training, not even as native distractors
    train_tools = [it for it in tool_items if it.split == "train"]
    for it in train_tools:
        it.tools = [t for t in it.tools if t.key not in unseen]
    train_tools = [it for it in train_tools if (it.gold is None and len(it.tools) >= 1) or (it.gold and len(it.tools) >= 1)]
    split_count = Counter((it.split, it.source, it.gold is not None) for it in tool_items)
    manifest["tool_splits"] = {"|".join(map(str, k)): v for k, v in sorted(split_count.items())}
    manifest["unseen_gold_tools"] = len(unseen)
    log(f"tool splits: {dict(Counter(it.split for it in tool_items))}; unseen gold tools {len(unseen)}")

    # ------------------------------------------------------------------ embeddings (e5, cached)
    def uniq(xs):
        return sorted(set(xs))

    ctx_emb = cached_embeddings("contexts", uniq([i.context for i in it_train + it_val] + [t.query for t in tool_items]), cache)
    all_tools = {}
    for it in tool_items:
        for t in it.tools:
            all_tools.setdefault(t.key, t)
    for it in tool_items:  # gold tools removed from train natives are still needed for eval items
        pass
    doc_emb = cached_embeddings("tool_docs", uniq(tool_doc(t) for t in all_tools.values()), cache)
    emb = EmbeddingIndex(list(ctx_emb.row) + list(doc_emb.row), np.vstack([ctx_emb.emb, doc_emb.emb]))

    train_pool_tools = list({t.key: t for it in train_tools for t in it.tools}.values())
    train_pool_tools.sort(key=lambda t: t.key)
    full_pool_tools = sorted(all_tools.values(), key=lambda t: t.key)
    near = tc["near_duplicate_cosine"]
    train_pool = ToolPool(train_pool_tools, np.stack([emb(tool_doc(t)) for t in train_pool_tools]), near)
    full_pool = ToolPool(full_pool_tools, np.stack([emb(tool_doc(t)) for t in full_pool_tools]), near)
    log(f"tool pools: train {len(train_pool_tools)}, full {len(full_pool_tools)}")

    # ------------------------------------------------------------------ intent miners (train only)
    miners = {}
    for ds in cfg["intents"]["train_datasets"]:
        its = [i for i in it_train if i.dataset == ds]
        protos = defaultdict(list)
        for i in its:
            if i.language in (None, "en", "en-US"):
                protos[i.label].append(i.context)
        miners[ds] = IntentMiner(sorted({i.label for i in its}), [i.context for i in its], [i.label for i in its],
                                 [i.group_id for i in its], np.stack([emb(i.context) for i in its]), protos)
        log(f"miner {ds}: {len(miners[ds].labels)} labels")
    manifest["confusion_examples"] = {ds: {g: m.top_confusions(g, 3) for g in m.labels[:10]} for ds, m in miners.items()}

    ib_train = IntentBuilder(miners, emb, train_pool, cfg["ood_cross_dataset_pairs"], cfg["cross_task_similarity_quantile"])
    ib_eval = IntentBuilder(miners, emb, full_pool, cfg["ood_cross_dataset_pairs"], cfg["cross_task_similarity_quantile"])
    tb_train = ToolBuilder(train_pool, emb, miners, cfg["cross_task_similarity_quantile"])
    tb_eval = ToolBuilder(full_pool, emb, miners, cfg["cross_task_similarity_quantile"])
    rng = seeded_rng(seed, "allocation")
    ic = cfg["intents"]
    specs, failed = [], Counter()

    # ------------------------------------------------------------------ TRAIN: intents
    jobs = [(it, v) for it in it_train for v in range(ic["decisions_per_item"][it.dataset])]
    ks = exact_allocation(len(jobs), ic["option_counts"], rng)
    st = exact_allocation(len(jobs), ic["negative_strategies"], rng)
    for (it, v), k, s in zip(jobs, ks, st):
        specs.append(ib_train.answerable(it, "train", k, s, v))
    n_items = len(it_train)
    for kind, frac in ic["ood"].items():
        chosen = [it_train[i] for i in sorted(rng.choice(n_items, size=int(round(frac * n_items)), replace=False))]
        oks = exact_allocation(len(chosen), ic["ood_option_counts"], rng)
        for it, k in zip(chosen, oks):
            fn = {"label_removed": ib_train.label_removed, "cross_dataset": ib_train.cross_dataset,
                  "cross_task": ib_train.cross_task}[kind]
            sp = fn(it, "train", k, 90)
            if sp is None:
                failed[f"intent {kind}"] += 1
            else:
                specs.append(sp)

    # ------------------------------------------------------------------ TRAIN: tools
    ans = [it for it in train_tools if it.gold]
    irr = [it for it in train_tools if it.gold is None]
    second = {ans[i].item_id for i in rng.choice(len(ans), size=int(round(tc["second_decision_fraction"] * len(ans))), replace=False)}
    jobs = [(it, 0) for it in ans] + [(it, 1) for it in ans if it.item_id in second]
    ks = exact_allocation(len(jobs), tc["option_counts"], rng)
    st = exact_allocation(len(jobs), tc["negative_strategies"], rng)
    for (it, v), k, s in zip(jobs, ks, st):
        sp = tb_train.answerable(it, "train", k, s, v)
        if sp is None:
            failed["tool answerable (pool too small)"] += 1
        else:
            specs.append(sp)
    chosen = [ans[i] for i in sorted(rng.choice(len(ans), size=int(round(tc["ood"]["tool_removed"] * len(ans))), replace=False))]
    for it, k in zip(chosen, exact_allocation(len(chosen), tc["ood_option_counts"], rng)):
        sp = tb_train.tool_removed(it, "train", k, 90)
        (specs.append(sp) if sp else failed.update(["tool removed"]))
    for it, k in zip(irr, exact_allocation(len(irr), tc["ood_option_counts"], rng)):
        sp = tb_train.irrelevance(it, "train", max(k, 2), 90)
        (specs.append(sp) if sp else failed.update(["tool irrelevance"]))
    chosen = [ans[i] for i in sorted(rng.choice(len(ans), size=int(round(tc["ood"]["cross_task"] * len(ans))), replace=False))]
    for j, (it, k) in enumerate(zip(chosen, exact_allocation(len(chosen), tc["ood_option_counts"], rng))):
        sp = tb_train.cross_task(it, "train", k, 91, "banking77" if j % 2 else "massive")
        (specs.append(sp) if sp else failed.update(["tool cross_task"]))

    bad_k = [s for s in specs if len(s.options) > 32 or len(s.options) < 2]
    if bad_k:
        raise SystemExit(f"{len(bad_k)} decisions with an invalid option count, e.g. {bad_k[0].variant}: {len(bad_k[0].options)}")
    train = [to_example(s) for s in specs]
    core = [s for s in specs if s.suite == "core"]
    copies = [core[i] for i in sorted(rng.choice(len(core), size=int(round(cfg["permutation"]["train_copy_fraction"] * len(core))),
                                                 replace=False))]
    for s in copies:
        group = f"{s.group_id}/{s.variant}/train-perm"
        train.append(to_example(type(s)(**{**s.__dict__, "suite": "permutation"}), 1 + int(seeded_rng(group).integers(len(s.options) - 1)),
                                group, "-perm"))
    manifest["train_generation_failures"] = dict(failed)
    log(f"train decisions: {len(train)} (failures {dict(failed)})")

    # ------------------------------------------------------------------ EVAL suites
    ec = cfg["eval"]
    pf, pk = cfg["permutation"]["eval_fraction"], cfg["permutation"]["eval_option_count"]

    def eval_intents(items, split_name):
        r = seeded_rng(seed, split_name)
        exs = []
        ks = exact_allocation(len(items), {k: 1 for k in ec["intent_hard_option_counts"]}, r)
        for it, k in zip(items, ks):
            exs.append(to_example(ib_eval.answerable(it, split_name, k, "hard_v1", 0)))
            f = stable_fraction(seed, split_name, "ood", it.group_id, it.config)
            if f < ec["intent_label_removed_fraction"]:
                exs.append(to_example(ib_eval.label_removed(it, split_name, [4, 8, 16][int(f * 1000) % 3], 0)))
            elif f < ec["intent_label_removed_fraction"] + ec["cross_task_fraction"]:
                sp = ib_eval.cross_task(it, split_name, [4, 8, 16][int(f * 1000) % 3], 0)
                if sp:
                    exs.append(to_example(sp))
            if stable_fraction(seed, split_name, "perm", it.group_id, it.config) < pf:
                exs += with_permutations(ib_eval.answerable(it, split_name, pk, "hard_v1", 7), pk, "rot")
        return exs

    def eval_tools(items, split_name):
        r = seeded_rng(seed, split_name, "tools")
        exs, fails = [], Counter()
        a = [it for it in items if it.gold]
        ks = exact_allocation(len(a), {k: 1 for k in ec["tool_option_counts"]}, r)
        stt = exact_allocation(len(a), tc["negative_strategies"], r)
        for it, k, s in zip(a, ks, stt):
            meta = {"tool_seen": it.metadata.get("tool_seen", True)}
            sp = tb_eval.answerable(it, split_name, k, s, 0, meta)
            (exs.append(to_example(sp)) if sp else fails.update(["answerable"]))
            f = stable_fraction(seed, split_name, "ood", it.group_id)
            kk = [4, 8, 16][int(f * 1000) % 3]
            if f < ec["tool_removed_fraction"]:
                sp = tb_eval.tool_removed(it, split_name, kk, 0, meta)
                (exs.append(to_example(sp)) if sp else fails.update(["removed"]))
            elif f < ec["tool_removed_fraction"] + ec["cross_task_fraction"]:
                sp = tb_eval.cross_task(it, split_name, kk, 0, "banking77" if int(f * 1e6) % 2 else "massive")
                (exs.append(to_example(sp)) if sp else fails.update(["cross_task"]))
            if stable_fraction(seed, split_name, "perm", it.group_id) < pf:
                sp = tb_eval.answerable(it, split_name, pk, "hard_v1", 7, meta)
                if sp:
                    exs += with_permutations(sp, pk, "rot")
        for it in items:
            if it.gold is None:
                sp = tb_eval.irrelevance(it, split_name, 8, 0)
                (exs.append(to_example(sp)) if sp else fails.update(["irrelevance"]))
        return exs, dict(fails)

    known = {(it.dataset, it.label) for it in it_train}
    manifest["val_hard_items_with_label_unseen_in_train"] = sum((it.dataset, it.label) not in known for it in it_val)
    val_hard = eval_intents([it for it in it_val if (it.dataset, it.label) in known], "val_hard")
    val_tools, f1 = eval_tools([it for it in tool_items if it.split == "validation"], "val_tools")
    holdout_tools, f2 = eval_tools([it for it in tool_items if it.split == "holdout"], "holdout_tools")
    manifest["eval_generation_failures"] = {"val_tools": f1, "holdout_tools": f2}

    # ------------------------------------------------------------------ write + validate
    files = {"train": train, "val_hard": val_hard, "val_tools": val_tools, "holdout_tools": holdout_tools}
    stats, checks = {}, {}
    group_split = {}
    for name, exs in files.items():
        for e in exs:  # leakage unit: a group (normalized query / KrunBench group) lives in one Krun split family
            fam = "train" if name == "train" else "eval"
            if group_split.setdefault(e.provenance.group_id, fam) != fam:
                raise SystemExit(f"leakage: group {e.provenance.group_id} in train and eval")
        ids = [e.id for e in exs]
        if len(ids) != len(set(ids)):
            dup = [k for k, v in Counter(ids).items() if v > 1][:5]
            raise SystemExit(f"duplicate ids in {name}: {dup}")
        rep = validate_examples(e.to_dict() for e in exs)
        checks[name] = rep.summary()
        write_jsonl(out / f"{name}.jsonl", exs)
        stats[name] = describe(exs)
        log(f"{name}: {len(exs)} decisions -> {out / (name + '.jsonl')}; validation errors={rep.summary().get('errors')}")
    manifest["stats"] = stats
    manifest["krunbench_validation"] = checks
    manifest["files"] = {n: {"path": str((out / f"{n}.jsonl").relative_to(PROJECT)), "sha256": sha256_path(out / f"{n}.jsonl")}
                         for n in files}
    write_json(out / "manifest.json", manifest)
    if not args.max_items:
        write_json(ROOT / "results" / "dataset_stats.json", manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
