"""R3: capability generators, table probes / families / catalogs, family-aware batching (no GPU, no model)."""

import datetime as dt
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from krun_one.v11 import capability as C, tech, training as T  # noqa: E402
from krun_one.v11.tables import TABLES  # noqa: E402


def _gold(v):
    return v["candidates"][v["gold"]]["name"] if v["answerable"] else None


def test_capability_families_are_labelled_and_counterfactual():
    rng = random.Random(11)
    for name, gen in C.GENERATORS.items():
        got = 0
        for _ in range(300):
            fam = gen(rng, rng.choice(["train", "bench"]))
            if not fam:
                continue
            got += 1
            assert all(v["slice"] == name for v in fam)
            for v in fam:
                assert v["qtype"] in ("choice", "noul", "score")
                assert (v["gold"] is not None) == v["answerable"]
                names = [c["name"] for c in v["candidates"]]
                assert len(set(names)) == len(names)
                if v["qtype"] == "choice" and v["answerable"]:      # the label is the unique best outcome
                    vals = [c["value"] for c in v["candidates"]]
                    best = max(x for x in vals if x is not None)
                    assert vals.count(best) == 1 and vals.index(best) == v["gold"]
            if len(fam) > 1:                                         # same candidates, different decision
                assert [c["name"] for c in fam[0]["candidates"]] == [c["name"] for c in fam[1]["candidates"]]
                assert _gold(fam[0]) != _gold(fam[1]) and all(v["answerable"] for v in fam)
        assert got > 100, name


def test_training_scenarios_never_use_benchmark_scenarios():
    rng = random.Random(5)
    for name, gen in C.GENERATORS.items():
        for _ in range(150):
            fam = gen(rng, "train")
            if fam:
                assert fam[0]["meta"].get("scenario") not in C.BENCH_SCENARIOS, name


def test_unanswerable_capability_problems_have_no_feasible_candidate():
    rng = random.Random(3)
    for name in C.UNANSWERABLE:
        got = 0
        for _ in range(3000):
            fam = C.GENERATORS[name](rng, "train", form="choice", want="unanswerable")
            if fam:
                got += 1
                assert len(fam) == 1 and not fam[0]["answerable"] and all(c["value"] is None for c in fam[0]["candidates"])
            if got >= 5:
                break
        assert got >= 1, name


def test_clock_and_calendar_helpers():
    assert C._hm(10 * 60 + 5) == "10:05" and C._hm(1440 + 30) == "00:30 the next day" and C._hm(2 * 1440 + 625) == "10:25 2 days later"
    rng = random.Random(1)
    for _ in range(200):
        fam = C.gen_tmp_dates(rng, "train", form="choice")
        m = fam[0]["meta"]
        want = 0 if m["gap_days"] <= m["w1"] else 1 if m["gap_days"] <= m["w2"] else 2
        sc = next(s for s in C.DATE_SCEN if s["id"] == m["scenario"])
        assert _gold(fam[0]) == sc["out"][want]
    assert (dt.date(2024, 2, 28) + dt.timedelta(days=2)).isoformat() == "2024-03-01"


def test_policy_interpreter_first_match():
    rules = [([("num", "amount", "gt", 100)], "a"), ([("bool", "receipt", False)], "b"), ([("cat", "kind", "x"), ("num", "amount", "le", 50)], "c")]
    assert C._policy_eval(rules, "d", {"amount": 150, "receipt": False, "kind": "x"}) == (0, "a")
    assert C._policy_eval(rules, "d", {"amount": 100, "receipt": False, "kind": "x"}) == (1, "b")
    assert C._policy_eval(rules, "d", {"amount": 40, "receipt": True, "kind": "x"}) == (2, "c")
    assert C._policy_eval(rules, "d", {"amount": 60, "receipt": True, "kind": "x"}) == (3, "d")


def test_table_families_and_catalogs_follow_the_rule():
    rng = random.Random(2)
    for tb in TABLES:
        for _ in range(40):
            fam = tech.family(rng, tb, "train", extra_req=1)
            if fam:
                golds = [_gold(v) for v in fam]
                assert len(set(golds)) == len(golds) >= 2
                for v in fam:
                    assert tech.split_of(tb, tuple(v["meta"]["requirements"])) == "train"
                    assert not v["meta"]["has_bench_only"]
            cat = tech.catalog(rng, tb, "bench")
            if cat:
                for v in cat:
                    assert tech.split_of(tb, tuple(v["meta"]["requirements"])) == "bench"
                    best = max(c["value"] for c in v["candidates"])
                    assert v["candidates"][v["gold"]]["value"] == best
                    g = v["candidates"][v["gold"]]
                    assert g["cons"]["met"] == len(v["meta"]["requirements"])
                    assert all(c["cons"]["overhead"] > g["cons"]["overhead"] for c in v["candidates"]
                               if c is not g and c["cons"]["met"] == len(v["meta"]["requirements"]))
        pr = tech.probe_fact(rng, tb)
        if pr:
            assert sorted(v["gold"] for v in pr) == [0, 1]
            for v in pr:
                o = TABLES[tb]["options"][v["meta"]["option"]]
                assert not o["bench_only"] and o["props"][v["meta"]["property"]] == ("Y" if v["gold"] else "N")


def test_family_batches_keep_siblings_together_and_cover_everything():
    rng = np.random.default_rng(0)
    lens = list(np.random.default_rng(1).integers(60, 1000, 3000))
    fam = [f"f{i // 3}" if i < 1500 else None for i in range(3000)]
    bs = T.family_batches(lens, fam, 16384, 64, rng)
    assert sorted(i for b in bs for i in b) == list(range(3000))
    where = {i: n for n, b in enumerate(bs) for i in b}
    assert all(where[i] == where[i - i % 3] for i in range(1500))
    assert max(len(b) for b in bs) <= 64
