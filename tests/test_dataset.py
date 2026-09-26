from collections import Counter

import numpy as np
import pytest

from krun_one.dataset import exact_allocation, token_batches
from krun_one.evaluation import tool_slices
from krun_one.mining import rank_of, rrf, sample_from_ranking, seeded_rng
from krun_one.model import build_sequence, option_texts


def test_exact_allocation_counts():
    vals = exact_allocation(1000, {2: 0.15, 4: 0.25, 8: 0.30, 16: 0.25, 32: 0.05}, np.random.default_rng(0))
    assert Counter(vals) == {2: 150, 4: 250, 8: 300, 16: 250, 32: 50}
    vals = exact_allocation(7, {"a": 0.65, "b": 0.35}, np.random.default_rng(0))
    assert len(vals) == 7 and set(Counter(vals).values()) == {5, 2}


def test_token_batches_budget_and_coverage():
    lens = np.random.default_rng(0).integers(10, 900, size=500).tolist()
    b = token_batches(lens, 4096, 64, np.random.default_rng(1))
    assert sorted(i for x in b for i in x) == list(range(500))
    assert all(max(lens[i] for i in x) * len(x) <= 4096 or len(x) == 1 for x in b)
    assert b == token_batches(lens, 4096, 64, np.random.default_rng(1))  # deterministic


def test_option_texts():
    assert option_texts(["card_arrival", "Refund_not_showing_up"]) == ["card arrival", "refund not showing up"]
    assert option_texts(["a_b", "a-b"]) == ["a_b", "a-b"]  # humanizing would collide -> raw
    assert option_texts(["f", "g"], {"f": "does f", "g": None}) == ["f: does f", "g"]


def test_markers_point_at_mask_tokens(tok):
    texts = option_texts([f"tool_{i}" for i in range(20)], {f"tool_{i}": "a description " * 10 for i in range(20)})
    ids, markers = build_sequence(tok, "hello " * 50, texts, "Choose.", 1024, 896)
    assert len(markers) == 20 and all(ids[m] == tok.mask_token_id for m in markers)
    ids, markers = build_sequence(tok, "a [MASK] b", ["x", "y"], "Choose [MASK].", 64, 32)
    assert sum(i == tok.mask_token_id for i in ids) == 2  # user-supplied [MASK] strings never become markers


def test_hard_sampling_prefers_top_ranks():
    cand = [f"c{i}" for i in range(50)]
    fused = rrf([rank_of(-np.arange(50.0))])
    picks = Counter(x for s in range(200) for x in sample_from_ranking(cand, fused, 3, seeded_rng("t", s)))
    assert set(picks) <= set(cand[:6]) and picks["c0"] > picks["c5"]


def test_tool_slices():
    from krunbench.schema import DecisionExample, Generation, Provenance

    e = DecisionExample(id="x", split="validation", task="tool_routing", suite="core", decision_type="choice",
                        context="what meetings do I have", options=["calendar_search", "calendar_searcher", "send_email"],
                        answer="calendar_search", answerable=True,
                        provenance=Provenance("d", "h", "r", None, "s", "o", "calendar_search", "g", "t"),
                        generation=Generation("hard", 3, 1, "x", "v"),
                        option_descriptions={"calendar_search": "Search calendar events",
                                             "calendar_searcher": "Search calendar events quickly", "send_email": "Send"},
                        metadata={"tool_seen": False})
    s = tool_slices(e)
    assert "near_duplicate_distractor" in s and "description_dependent" in s and "unseen_tool" in s
