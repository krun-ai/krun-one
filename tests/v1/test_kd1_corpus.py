"""Krun Decisions v1 corpus: mining safety rules and generator invariants (no network, no real embeddings)."""

import numpy as np

from krun_one.v1 import schema
from krun_one.v1.corpus.decisions import Builder, stage_of
from krun_one.v1.corpus.mining import LabelSpace, ToolPool, rng_for


class FakeEmb:
    def __init__(self):
        self.cache = {}

    def get(self, texts):
        out = []
        for t in texts:
            if t not in self.cache:
                v = np.random.default_rng(abs(hash(t)) % 2**32).normal(size=384)
                self.cache[t] = v / np.linalg.norm(v)
            out.append(self.cache[t])
        return np.array(out, dtype=np.float32)


def ctx(text, label, lang="en-US", cat="bank", oid="1"):
    return {"source": cat, "oid": oid, "orig_split": "train", "text": text, "lang": lang, "label": label, "labels": None,
            "group": f"{cat}:{oid}", "split_group": f"{cat}:{oid}", "tools": None, "gold_tool": None, "calls": None,
            "history": [], "catalog": cat, "meta": {}, "origin": "real"}


def space(emb):
    labels = [f"label_{i}" for i in range(20)]
    ex = {l: [(f"example {j} for {l}", f"g{l}{j}") for j in range(3)] for l in labels}
    return LabelSpace("bank", labels, ex, emb)


def test_label_negatives_never_gold_and_seeded():
    emb = FakeEmb()
    sp = space(emb)
    e = emb.get(["hello"])[0]
    for i in range(20):
        n1 = sp.negatives("label_3", 7, e, rng_for("t", i))
        n2 = sp.negatives("label_3", 7, e, rng_for("t", i))
        assert n1 == n2 and "label_3" not in n1 and len(set(n1)) == 7


def test_tool_pool_excludes_equivalents():
    emb = FakeEmb()
    tools = [("getWeather", "Get the weather for a city."), ("get_weather", "Weather lookup."),
             ("fetchWeather", "Get the weather for a city."), ("sendEmail", "Send an email.")] + \
            [(f"tool{i}", f"Does thing number {i}.") for i in range(40)]
    pool = ToolPool(tools, emb)
    gold = ("getWeather", "Get the weather for a city.")
    for i in range(20):
        negs, _ = pool.negatives(gold, emb.get(["q"])[0], 10, rng_for("x", i), exclude_names={"getWeather"})
        names = [n for n, _ in negs]
        assert "get_weather" not in names            # same humanized name as the gold
        assert "fetchWeather" not in names           # same description as the gold
        assert len(set(names)) == len(names)
        negs, _ = pool.negatives(gold, emb.get(["q"])[0], 10, rng_for("x", i), exclude_names={"getWeather"}, hard=False)
        assert len({n for n, _ in negs}) == len(negs)


def test_builder_records_are_valid_and_staged():
    emb = FakeEmb()
    sp = space(emb)
    B = Builder(emb, {"bank": sp}, ToolPool([("a", "A."), ("b", "B.")], emb), {}, {}, 1)
    c = ctx("quero cancelar", "label_5", lang="pt-BR")
    r = B.intent_choice(c, rng_for("a"), k=8)
    assert not schema.validate(r) and r["target"]["labels"] == ["label_5"] and r["cross_language"]
    assert r["metadata"]["origin"] == "real"
    r2 = B.intent_choice(c, rng_for("b"), gold_removed=True, k=8)
    assert not schema.validate(r2) and r2["answerable"] is False and "label_5" not in [x["id"] for x in r2["candidates"]]
    assert r2["metadata"]["origin"] == "derived" and r2["metadata"]["stage"] == "D3"
    m = B.intent_concat(ctx("block my card", "label_1", oid="2"), ctx("what is my balance", "label_7", oid="3"), rng_for("c"))
    assert not schema.validate(m) and sorted(m["target"]["labels"]) == ["label_1", "label_7"]
    assert B.rec(c, qtype="choice", family="intent", instr_family="intent", candidates=[{"id": "x"}], labels=["x"],
                 slices=[], strategy="random", category="choice_intent", tag="t", rng=rng_for("d")) is None


def test_stage_rules():
    base = {"slices": [], "answerable": True, "candidates": [{}] * 4, "cross_language": False,
            "candidate_set": {"strategy": "random"}, "task": {"type": "choice"}, "metadata": {"origin": "real"}}
    assert stage_of(base) == "D1"
    assert stage_of({**base, "cross_language": True}) == "D2"
    assert stage_of({**base, "answerable": False}) == "D3"
    assert stage_of({**base, "metadata": {"origin": "llm"}}) == "D3"
    assert stage_of({**base, "candidates": [{}] * 128}) == "D3"
