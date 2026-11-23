"""v1.1 research code: generators give verified unique answers; growth and the interaction block preserve v1."""

import random

import torch

from krun_one.v1.arch import DecisionSpec, EncoderSpec, ModelSpec, count_parameters
from krun_one.v1.model import KrunOneV1Model
from krun_one.v11 import envs, growth, probe
from krun_one.v11.model import KrunOneV11Model, V11Spec, count_parameters_v11

TINY = ModelSpec("tiny", EncoderSpec(vocab_size=300, hidden_size=128, num_layers=4, num_heads=2, intermediate_size=96,
                                     local_window=8), DecisionSpec(head_layers=1, head_ffn_mult=2))


def _batch():
    torch.manual_seed(1)
    ids = torch.randint(5, 300, (3, 40))
    att = torch.ones_like(ids)
    att[1, 30:] = 0
    return dict(input_ids=ids, attention_mask=att, marker_pos=torch.tensor([[3, 9, 15, 20], [2, 5, 0, 0], [4, 8, 12, 0]]),
                marker_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool),
                qtype=torch.tensor([0, 0, 3]), null_pos=torch.tensor([22, 7, 14]))


def _model(spec=TINY):
    torch.manual_seed(0)
    m = KrunOneV1Model.from_spec(spec).eval()
    with torch.no_grad():   # non-trivial norms / biases so the test would catch a missed tensor
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    return m


def test_generators_unique_gold_and_feasible():
    rng = random.Random(0)
    for name, gen in envs.GENERATORS.items():
        got = [p for p in (gen(rng, 8) for _ in range(200)) if p]
        assert len(got) > 20, name
        for p in got:
            if not p["answerable"]:
                continue
            vals = [c["value"] for c in p["candidates"]]
            best = max(v for v in vals if v is not None)
            assert vals.count(best) == 1 and vals[p["gold"]] == best
            assert len({c["name"] for c in p["candidates"]}) == p["k"]


def test_tetris_simulation():
    board = [[0] * envs.W for _ in range(envs.H)]
    board[-1] = [1] * (envs.W - 4) + [0] * 4
    nb, lines = envs.tetris_drop(board, envs.PIECES["I"][0], envs.W - 4)
    assert lines == 1 and not any(any(r) for r in nb)
    nb, lines = envs.tetris_drop(board, envs.PIECES["O"][0], 0)
    assert lines == 0 and envs.tetris_stats(nb) == (0, 2 * 3 + 4 * 1, 3)   # O on the bottom row: cols 0-1 height 3, cols 2-5 height 1


def test_tech_rules():
    for cat in envs.TECH:
        for rs in envs.tech_requirement_sets(cat):
            opts = envs.TECH[cat]["options"]
            assert all(opts[o][1][r] != "P" for o in rs["kept"] for r in rs["req"])
            if rs["gold"]:
                sat = [o for o in rs["kept"] if all(opts[o][1][r] == "Y" for r in rs["req"])]
                assert rs["gold"] in sat and all(opts[rs["gold"]][0] < opts[o][0] for o in sat if o != rs["gold"])
        for base, flip in envs.tech_flip_pairs(cat):
            assert base["gold"] != flip["gold"] and base["gold"] in flip["kept"] and len(flip["req"]) == len(base["req"]) + 1
    rng = random.Random(0)
    b, f = envs.tech_flip_pairs("datastore")[0]
    pb, pf = envs.tech_pair(rng, b, f, 8)
    assert [c["name"] for c in pb["candidates"]] == [c["name"] for c in pf["candidates"]]
    assert pb["candidates"][pb["gold"]]["name"] == b["gold"] and pf["candidates"][pf["gold"]]["name"] == f["gold"]


def test_item_signals():
    p = {"id": "x", "answerable": True, "gold": 1, "k": 3,
         "candidates": [{"name": "a", "value": 0.0}, {"name": "b", "value": 2.0}, {"name": "c", "value": None}]}
    r = probe.item_signals(p, [0.0, 2.0, 1.0], 1.0, 0.5)
    assert r["correct"] and r["rank"] == 1 and r["gold_vs_hardest"] and r["pair_all"] == 2 / 3 and abs(r["margin_logit"] - 2.0) < 1e-9
    r = probe.item_signals(p, [3.0, 2.0, 1.0], 1.0, 0.5)
    assert not r["correct"] and r["rank"] == 2 and not r["gold_vs_hardest"]


def test_grow_exact_preserves_function():
    src, b = _model(), _batch()
    spec = growth.grown_spec(TINY, "grown", intermediate_size=256, num_layers=6, head_layers=3, head_ffn_mult=4)
    dst, rep = growth.grow_exact(src, spec)
    dst.eval()
    assert sum(p.numel() for p in dst.parameters()) == count_parameters(spec)["total"]
    with torch.no_grad():
        a, c = src(**b), dst(**b)
    for k in ("rank", "apply", "null_rank", "ans"):
        assert torch.allclose(a[k], c[k], atol=2e-5), (k, float((a[k] - c[k]).abs().max()))
    assert len(rep["new_identity_blocks"]) == 4   # 2 encoder layers + 2 head layers


def test_candidate_interaction_is_identity_at_init():
    src, b = _model(), _batch()
    m = KrunOneV11Model.from_v1(src, cand_layers=2).eval()
    assert sum(p.numel() for p in m.parameters()) == count_parameters_v11(V11Spec(TINY, 2))["total"]
    with torch.no_grad():
        a, c = src(**b), m(**b)
    for k in ("rank", "apply", "null_rank", "ans"):
        assert torch.equal(a[k], c[k]), k
    # and it does change the output once its weights move
    with torch.no_grad():
        for lyr in m.cand.layers:
            lyr.linear2.weight.normal_(std=0.05)
        assert not torch.allclose(m(**b)["rank"], a["rank"])


def test_widen_stream_preserves_function():
    from krun_one.v11 import growth_width

    spec = ModelSpec("tiny", EncoderSpec(vocab_size=300, hidden_size=192, num_layers=4, num_heads=3, intermediate_size=96,
                                         local_window=8), DecisionSpec(head_layers=2, head_ffn_mult=2))
    src, b = _model(spec).double(), _batch()
    wide = ModelSpec("wide", EncoderSpec(vocab_size=300, hidden_size=256, num_layers=4, num_heads=4, intermediate_size=160,
                                         local_window=8), DecisionSpec(head_layers=2, head_ffn_mult=2))
    dst, rep = growth_width.widen_stream(src, wide, lam=1e-4, kappa=1e-3)
    dst = dst.double().eval()
    with torch.no_grad():
        a, c = src(**b), dst(**b)
    for k in ("rank", "apply", "null_rank"):
        assert torch.allclose(a[k], c[k], atol=1e-4), (k, float((a[k] - c[k]).abs().max()))
    assert float((a["ans"] - c["ans"]).abs().max()) < 0.05      # closed signal: approximate by construction
