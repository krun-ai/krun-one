"""Krun One decision data: construction (source items -> DecisionExample) and training encoding.

Every decision uses the canonical KrunBench schema (``krunbench.decision.v0``) so the same metrics,
validation and slicing apply to KrunBench and to Krun One's own data. Schema enums are respected:
``generation.negative_strategy`` is ``random`` / ``hard`` / ``native``; the finer strategy
(``hard_v1``, ``native+hard_v1``, ``safe_random`` ...) is recorded in ``metadata.negatives``.

Suites used here:
  core                answerable decision
  permutation         rotated copies sharing ``generation.consistency_group``
  ood_label_removed   near-OOD: intent gold removed (hard neighbours) / tool gold removed (native others + safe fill)
  ood_cross_dataset   far-OOD: banking77 <-> massive, and cross-task (intent context vs tools, tool query vs intents)
  ood_native          the source says no tool applies (xlam-irrelevance)
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from krunbench.options import permute_options
from krunbench.schema import DecisionExample, Generation, Provenance, read_jsonl
from krunbench.splits import stable_int

from .mining import IntentMiner, ToolPool, seeded_rng
from .model import DEFAULT_INSTRUCTIONS, humanize_label
from .sources import TOOL_SOURCES, IntentItem, Tool, ToolItem

TRANSFORM = "krun_one/0.1.0"
SCHEMA_SPLIT = {"val_hard": "validation", "val_tools": "validation", "holdout_tools": "test"}  # file name keeps the suite
INSTR = DEFAULT_INSTRUCTIONS


def exact_allocation(n: int, dist: Mapping[Any, float], rng: np.random.Generator) -> list:
    """n values distributed exactly per ``dist`` (largest remainder), in a seeded random order."""
    keys = list(dist)
    w = np.array([dist[k] for k in keys], dtype=float)
    w = w / w.sum()
    raw = w * n
    cnt = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - cnt), kind="stable")[: n - cnt.sum()]:
        cnt[i] += 1
    vals = [k for k, c in zip(keys, cnt) for _ in range(c)]
    return [vals[i] for i in rng.permutation(n)]


def perm_seed(*parts: object) -> int:
    return stable_int("krun_one-permutation", *parts)


class EmbeddingIndex:
    """text -> row in a precomputed embedding matrix."""

    def __init__(self, texts: Sequence[str], emb: np.ndarray):
        self.row = {t: i for i, t in enumerate(texts)}
        self.emb = emb

    def __call__(self, text: str) -> np.ndarray:
        return self.emb[self.row[text]]


def tool_doc(t: Tool) -> str:
    return f"{t.name}: {t.description}"


# ----------------------------------------------------------------------------------------- examples
@dataclass
class Spec:
    """Everything that defines one decision before permutation."""
    split: str
    task: str
    suite: str
    context: str
    options: list[str]
    answer: str | None
    descriptions: dict[str, str] | None
    language: str | None
    dataset: str
    hf_id: str
    revision: str
    config: str | None
    original_split: str
    original_id: str
    original_label: str | None
    group_id: str
    strategy: str  # random | hard | native (schema enum)
    negatives: str  # fine-grained strategy
    option_source: str
    variant: str
    metadata: dict[str, Any]


def to_example(s: Spec, rotation: int = 0, consistency_group: str | None = None, variant_suffix: str = "") -> DecisionExample:
    seed = perm_seed(s.group_id, s.variant, s.original_id)  # suite-independent: copies are rotations of the core order
    opts = permute_options(s.options, seed, rotation)
    variant = s.variant + variant_suffix
    ex = DecisionExample(
        id=f"krun1/{s.split}/{s.dataset}/{s.config or '-'}/{s.original_id}/{s.suite}/{variant}",
        split=SCHEMA_SPLIT.get(s.split, s.split), task=s.task, suite=s.suite, decision_type="choice", context=s.context, options=opts,
        answer=s.answer, answerable=s.answer is not None,
        provenance=Provenance(s.dataset, s.hf_id, s.revision, s.config, s.original_split, s.original_id,
                              s.original_label, s.group_id, TRANSFORM),
        generation=Generation(s.strategy, len(opts), seed, s.option_source, variant, consistency_group, rotation),
        instructions=INSTR["tool" if s.descriptions else "intent"],
        option_descriptions={o: s.descriptions[o] for o in opts} if s.descriptions else None,
        language=s.language, metadata={"negatives": s.negatives, **s.metadata},
    )
    return ex


def with_permutations(s: Spec, n_rot: int, tag: str) -> list[DecisionExample]:
    group = f"{s.group_id}/{s.suite}/{s.variant}/{tag}"
    return [to_example(Spec(**{**s.__dict__, "suite": "permutation"}), r, group, f"-{tag}{r}") for r in range(n_rot)]


# ------------------------------------------------------------------------------------------ intents
class IntentBuilder:
    def __init__(self, miners: Mapping[str, IntentMiner], emb: EmbeddingIndex, tool_pool: ToolPool | None,
                 cross_pairs: Sequence[Mapping[str, Any]], cross_quantile: float):
        self.miners, self.emb, self.pool = miners, emb, tool_pool
        self.cross_pairs = {p["source"]: p for p in cross_pairs}
        self.q = cross_quantile

    @staticmethod
    def _base(it: IntentItem, split: str, suite: str, options, answer, variant, strategy, negatives, option_source,
              meta=None, descriptions=None, task="intent_routing") -> Spec:
        return Spec(split, task, suite, it.context, list(options), answer, descriptions, it.language, it.dataset,
                    it.hf_id, it.revision, it.config, it.original_split, it.original_id, it.label, it.group_id,
                    strategy, negatives, option_source, variant, meta or {})

    def answerable(self, it: IntentItem, split: str, k: int, strat: str, v: int) -> Spec:
        m = self.miners[it.dataset]
        k = min(k, len(m.labels))
        rng = seeded_rng("intent-neg", it.group_id, it.config, v)
        if strat == "hard_v1":
            neg = m.hard_negatives(it.label, self.emb(it.context), k - 1, rng)
        else:
            neg = m.random_negatives(it.label, k - 1, rng)
        return self._base(it, split, "core", [it.label] + neg, it.label, f"v{v}-k{k}-{strat}",
                          "hard" if strat == "hard_v1" else "random", strat, it.dataset)

    def label_removed(self, it: IntentItem, split: str, k: int, v: int) -> Spec:
        m = self.miners[it.dataset]
        rng = seeded_rng("intent-ood-near", it.group_id, it.config, v)
        neg = m.hard_negatives(it.label, self.emb(it.context), k, rng)
        return self._base(it, split, "ood_label_removed", neg, None, f"v{v}-k{k}-near", "hard", "hard_v1(gold removed)",
                          it.dataset, {"ood_kind": "label_removed", "removed_label": it.label})

    def cross_dataset(self, it: IntentItem, split: str, k: int, v: int) -> Spec | None:
        pair = self.cross_pairs.get(it.dataset)
        if pair is None or it.label in pair.get("exclude_source_labels", []):
            return None
        tm = self.miners[pair["target"]]
        rng = seeded_rng("intent-ood-far", it.group_id, it.config, v)
        opts = tm.random_negatives(None, k, rng, exclude=pair.get("exclude_target_labels", []))
        return self._base(it, split, "ood_cross_dataset", opts, None, f"v{v}-k{k}-far-{pair['target']}", "random",
                          "random(other taxonomy)", pair["target"], {"ood_kind": "cross_dataset"})

    def cross_task(self, it: IntentItem, split: str, k: int, v: int) -> Spec | None:
        rng = seeded_rng("intent-ood-xtask", it.group_id, it.config, v)
        idx = self.pool.safe_random(self.emb(it.context), k, rng, set(), self.q)
        if len(idx) < k:
            return None
        tools = [self.pool.tools[i] for i in idx]
        return self._base(it, split, "ood_cross_dataset", [t.name for t in tools], None, f"v{v}-k{k}-xtask", "random",
                          "safe_random(tool pool)", "tool_pool", {"ood_kind": "cross_task"},
                          descriptions={t.name: t.description for t in tools}, task="tool_routing")


# -------------------------------------------------------------------------------------------- tools
class ToolBuilder:
    def __init__(self, pool: ToolPool, emb: EmbeddingIndex, intent_miners: Mapping[str, IntentMiner], cross_quantile: float):
        self.pool, self.emb, self.miners, self.q = pool, emb, intent_miners, cross_quantile

    @staticmethod
    def _base(it: ToolItem, split: str, suite: str, tools: Sequence[Tool], answer, variant, strategy, negatives,
              meta) -> Spec:
        s = TOOL_SOURCES[it.source]
        return Spec(split, "tool_routing", suite, it.query, [t.name for t in tools], answer,
                    {t.name: t.description for t in tools}, None, it.source, s["hf_id"], s["revision"], None, "train",
                    it.item_id, it.gold, it.group_id, strategy, negatives, f"{it.source}:native+pool", variant, meta)

    def _native_subset(self, natives: list[Tool], n: int, strat: str, qemb: np.ndarray, rng) -> list[Tool]:
        if len(natives) <= n:
            return natives
        if strat == "hard_v1":  # the most query-similar native tools
            sims = np.array([self.emb(tool_doc(t)) @ qemb for t in natives])
            return [natives[i] for i in sorted(np.argsort(-sims, kind="stable")[:n])]
        return [natives[i] for i in sorted(rng.choice(len(natives), size=n, replace=False))]

    def answerable(self, it: ToolItem, split: str, k: int, strat: str, v: int, meta=None) -> Spec | None:
        gold = it.tool(it.gold)
        qemb, gemb = self.emb(it.query), self.emb(tool_doc(gold))
        rng = seeded_rng("tool-neg", it.item_id, v)
        natives = self._native_subset([t for t in it.tools if t.name != it.gold], k - 1, strat, qemb, rng)
        forbid = {gold.name} | {t.name for t in natives}
        need = k - 1 - len(natives)
        hname = humanize_label(gold.name)
        idx = (self.pool.hard(qemb, gemb, hname, need, rng, forbid) if strat == "hard_v1"
               else self.pool.random(need, rng, forbid, gemb, hname))
        if len(idx) < need:
            return None
        tools = [gold] + natives + [self.pool.tools[i] for i in idx]
        strategy = "native" if need == 0 and len(natives) == len(it.tools) - 1 else ("hard" if strat == "hard_v1" else "random")
        return self._base(it, split, "core", tools, gold.name, f"v{v}-k{k}-{strat}", strategy,
                          f"native+{strat}" if natives else strat,
                          {"native_count": len(natives) + 1, "pool_count": len(idx), **(meta or {})})

    def tool_removed(self, it: ToolItem, split: str, k: int, v: int, meta=None) -> Spec | None:
        gold = it.tool(it.gold)
        qemb, gemb = self.emb(it.query), self.emb(tool_doc(gold))
        rng = seeded_rng("tool-ood-removed", it.item_id, v)
        hname = humanize_label(gold.name)
        natives = [t for t in it.tools if t.name != gold.name and humanize_label(t.name) != hname
                   and self.emb(tool_doc(t)) @ gemb < self.pool.near_dup]
        natives = self._native_subset(natives, k, "hard_v1", qemb, rng)
        forbid = {gold.name} | {t.name for t in natives}
        idx = self.pool.safe_random(qemb, k - len(natives), rng, forbid, 0.9, gold_emb=gemb)
        if len(natives) + len(idx) < k:
            return None
        tools = natives + [self.pool.tools[i] for i in idx]
        return self._base(it, split, "ood_label_removed", tools, None, f"v{v}-k{k}-removed", "hard" if natives else "random",
                          "native_others+safe_random(gold removed)",
                          {"ood_kind": "tool_removed", "removed_tool": gold.name, "native_count": len(natives), **(meta or {})})

    def irrelevance(self, it: ToolItem, split: str, k: int, v: int) -> Spec | None:
        qemb = self.emb(it.query)
        rng = seeded_rng("tool-ood-native", it.item_id, v)
        natives = list(it.tools)
        if len(natives) > k:
            natives = self._native_subset(natives, k, "hard_v1", qemb, rng)
        idx = self.pool.safe_random(qemb, k - len(natives), rng, {t.name for t in natives}, 0.9)
        if len(natives) + len(idx) < max(2, k):
            return None
        tools = natives + [self.pool.tools[i] for i in idx]
        return self._base(it, split, "ood_native", tools, None, f"v{v}-k{len(tools)}-irrelevance",
                          "native" if not idx else "random", "native+safe_random",
                          {"ood_kind": "native_irrelevance", "native_count": len(natives)})

    def cross_task(self, it: ToolItem, split: str, k: int, v: int, taxonomy: str) -> Spec | None:
        m = self.miners[taxonomy]
        sims = m.centroids @ self.emb(it.query)
        ok = [m.labels[i] for i in np.nonzero(sims < np.quantile(sims, self.q))[0]]
        if len(ok) < k:
            return None
        rng = seeded_rng("tool-ood-xtask", it.item_id, v)
        labels = [ok[i] for i in sorted(rng.choice(len(ok), size=k, replace=False))]
        s = TOOL_SOURCES[it.source]
        return Spec(split, "intent_routing", "ood_cross_dataset", it.query, labels, None, None, None, it.source, s["hf_id"],
                    s["revision"], None, "train", it.item_id, it.gold, it.group_id, "random", "safe_random(intent labels)",
                    taxonomy, f"v{v}-k{k}-xtask-{taxonomy}", {"ood_kind": "cross_task"})


# --------------------------------------------------------------------------------------- statistics
def describe(examples: Sequence[DecisionExample]) -> dict[str, Any]:
    def c(fn: Callable[[DecisionExample], Any]) -> dict[str, int]:
        return dict(sorted(Counter(str(fn(e)) for e in examples).items()))

    return {
        "n": len(examples),
        "answerable": sum(e.answerable for e in examples),
        "unanswerable": sum(not e.answerable for e in examples),
        "by_task": c(lambda e: e.task),
        "by_suite": c(lambda e: e.suite),
        "by_dataset": c(lambda e: e.provenance.dataset),
        "by_language": c(lambda e: e.language),
        "by_option_count": c(lambda e: e.generation.option_count),
        "by_negative_strategy": c(lambda e: e.generation.negative_strategy),
        "by_negatives_detail": c(lambda e: e.metadata.get("negatives")),
        "by_ood_kind": c(lambda e: e.metadata.get("ood_kind", "-")),
        "answerable_by_option_count": c(lambda e: e.generation.option_count if e.answerable else "ood"),
    }


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ----------------------------------------------------------------------------------- training input
def load_examples(path: Path) -> list[DecisionExample]:
    return read_jsonl(path)


def encode_for_model(examples: Sequence[DecisionExample], tok, max_len: int, head_max_len: int) -> tuple[list[dict], list[str]]:
    """DecisionExample -> {"ids", "markers", "label", "target"} (soft target: one-hot, or uniform if unanswerable).
    Returns (items, ids of examples dropped because their options did not fit)."""
    from .model import build_sequence, option_texts

    items, dropped = [], []
    for e in examples:
        ids, markers = build_sequence(tok, e.context, option_texts(e.options, e.option_descriptions), e.instructions or "",
                                      max_len, head_max_len)
        k = len(e.options)
        if len(markers) != k:
            dropped.append(e.id)
            continue
        y = e.answer_index
        target = [1.0 if i == y else 0.0 for i in range(k)] if y is not None else [1.0 / k] * k
        items.append({"id": e.id, "ids": ids, "markers": markers, "label": -1 if y is None else y, "target": target,
                      "task": e.task, "answerable": y is not None})
    return items, dropped


def token_batches(lengths: Sequence[int], max_tokens: int, max_seqs: int, rng: np.random.Generator,
                  chunk: int = 2048) -> list[list[int]]:
    """Length-bucketed batches under a padded-token budget (Laya's make_token_batches idea), seeded order."""
    order = rng.permutation(len(lengths))
    batches = []
    for s in range(0, len(order), chunk):
        part = order[s:s + chunk]
        part = part[np.argsort(np.asarray(lengths)[part], kind="stable")]
        cur, cur_max = [], 0
        for i in part:
            ln = int(lengths[i])
            new_max = max(cur_max, ln)
            if cur and (new_max * (len(cur) + 1) > max_tokens or len(cur) + 1 > max_seqs):
                batches.append(cur)
                cur, new_max = [], ln
            cur.append(int(i))
            cur_max = new_max
        if cur:
            batches.append(cur)
    return [batches[i] for i in rng.permutation(len(batches))]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")
