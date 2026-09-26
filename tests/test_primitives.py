"""KRUN-011 primitives: serialization identical to upstream Laya, real qtype, typed answers, RPS loss, batch invariance."""

import importlib.util
import math
import sys

import numpy as np
import pytest
import torch

from krun_one.laya_adapter import LAYA_DIR
from krun_one.model import QTYPES, collate
from krun_one.primitives import (encode_question, noul_answer, ordinal_dispersion_confidence, render_options, score_answer,
                                 validate_question)
from krun_one.training import decision_loss

HAS_LAYA = (LAYA_DIR / "model.safetensors").exists()

NOUL = {"type": "noul", "instructions": "Is the customer asking to speak with a human?"}
NOUL_CRIT = {**NOUL, "criteria": {"true": "Explicitly requests a person", "false": "Does not request human help"}}
SCORE = {"type": "score", "instructions": "How severe is the reported issue?",
         "levels": ["Cosmetic issue; no functionality affected", "Feature degraded; workaround available",
                    "Critical functionality blocked; no workaround"]}
CHOICE = {"type": "choice", "instructions": "Which department?", "options": ["billing", "support", "sales"]}
CTX = "Customer says this is the third time exports failed and wants a human immediately."


@pytest.fixture(scope="module")
def upstream():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("laya_rl_common_upstream_prim", LAYA_DIR / "rl_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def to_upstream(q):
    if q["type"] == "noul":
        return {"t": "noul", "ins": q["instructions"], "crit": q.get("criteria")}
    if q["type"] == "score":
        return {"t": "score", "ins": q["instructions"], "crit": q["levels"]}
    return {"t": "choice", "ins": q["instructions"], "crit": {o: None for o in q["options"]}}


def test_render_options_matches_upstream_semantics():
    assert render_options(NOUL) == ["false: no, the statement does not hold", "true: yes, the statement holds"]
    assert render_options(NOUL_CRIT) == ["false: Does not request human help", "true: Explicitly requests a person"]
    assert render_options(SCORE)[2] == "level 2: Critical functionality blocked; no workaround"


@pytest.mark.skipif(not HAS_LAYA, reason="./laya checkout not present")
@pytest.mark.parametrize("q", [NOUL, NOUL_CRIT, SCORE, CHOICE])
def test_sequence_identical_to_upstream(upstream, tok, q):
    for max_len, head in ((512, 192), (1024, 896)):
        ours = encode_question(tok, CTX, q, max_len, head)
        ids, markers = upstream.build_sequence(tok, CTX, to_upstream(q), max_len, head)
        assert (ours["ids"], ours["markers"]) == (list(ids), list(markers))
        assert ours["qtype"] == upstream.QTYPES[q["type"]]
        assert tok.decode(ours["ids"][1:4]).strip().startswith(q["type"])


def test_qtype_reaches_the_model(tiny_model):
    model, _ = tiny_model
    model.eval()
    it = {"ids": [50281] + list(range(100, 130)) + [50282], "markers": [3, 7]}
    b = collate([{**it, "qtype": t} for t in (0, 1, 2)], 50283)
    assert b["qtype"].tolist() == [0, 1, 2]
    with torch.no_grad():
        z = model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
    assert not torch.allclose(z[0], z[1]) and not torch.allclose(z[1], z[2])  # different type_emb rows
    assert collate([it], 50283)["qtype"].tolist() == [0]  # default stays choice (v0 behaviour)


def test_mixed_batch_invariance_fp32(tiny_model):
    model, _ = tiny_model
    model.eval()
    g = torch.Generator().manual_seed(3)
    items = []
    for t, k in ((0, 4), (2, 2), (1, 5)):
        L = int(torch.randint(20, 60, (1,), generator=g))
        ids = [50281] + torch.randint(100, 5000, (L,), generator=g).tolist() + [50282]
        items.append({"ids": ids, "markers": list(range(2, 2 + 2 * k, 2)), "qtype": t})
    with torch.no_grad():
        b = collate(items, 50283)
        zm = model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
        for i, it in enumerate(items):
            b1 = collate([it], 50283)
            z1 = model(b1["input_ids"], b1["attention_mask"], b1["marker_pos"], b1["marker_mask"], b1["qtype"])
            k = len(it["markers"])
            assert torch.allclose(z1[0, :k], zm[i, :k], atol=1e-4)


def test_typed_answers():
    p = np.array([0.0, 0.57, 0.43])
    a = score_answer(p, SCORE["levels"])
    assert a["score"] == pytest.approx(1.43)
    assert a["legend"]["1"] == SCORE["levels"][1] and a["probabilities"]["1"] == pytest.approx(0.57)
    h = -(0.57 * math.log(0.57) + 0.43 * math.log(0.43)) / math.log(3)
    assert a["confidence"] == pytest.approx(1 - h, abs=1e-6)
    assert ordinal_dispersion_confidence(np.array([0.5, 0.0, 0.5])) == pytest.approx(0.0)
    assert ordinal_dispersion_confidence(np.array([0.0, 1.0, 0.0])) == pytest.approx(1.0)
    assert noul_answer(np.array([0.027, 0.973])) == {"type": "noul", "noul": 0.973}


def test_validation():
    validate_question(NOUL)
    validate_question(NOUL_CRIT)
    validate_question(SCORE)
    for bad in ({"type": "noul", "instructions": " "}, {"type": "score", "instructions": "x", "levels": ["a"]},
                {"type": "score", "instructions": "x", "levels": ["a", ""]},
                {"type": "score", "instructions": "x", "levels": [str(i) for i in range(17)]},
                {"type": "noul", "instructions": "x", "criteria": {"maybe": "y"}}, {"type": "rank"}):
        with pytest.raises(ValueError):
            validate_question(bad)


def test_rps_loss_is_ordinal():
    mask = torch.ones(2, 5, dtype=torch.bool)
    target = torch.zeros(2, 5)
    target[:, 0] = 1
    near, far = torch.tensor([0.0, 3.0, 0, 0, 0]), torch.tensor([0.0, 0, 0, 0, 3.0])  # same CE, different distance
    logits = torch.stack([near, far])
    q = torch.tensor([QTYPES["score"]] * 2)
    ans = torch.ones(2, dtype=torch.bool)
    ce = [decision_loss(l[None], target[:1], mask[:1], ans[:1], 0.5) for l in logits]
    assert torch.isclose(ce[0], ce[1])
    rps = [decision_loss(l[None], target[:1], mask[:1], ans[:1], 0.5, q[:1], 1.0) for l in logits]
    assert rps[1] > rps[0] > ce[0]
    # RPS never applies to non-score questions
    assert torch.isclose(decision_loss(logits, target, mask, ans, 0.5, torch.tensor([2, 0]), 1.0),
                         decision_loss(logits, target, mask, ans, 0.5))


ARTIFACT = __import__("pathlib").Path(__file__).resolve().parents[1] / "artifacts" / "krun-one-v0"


@pytest.mark.skipif(not (ARTIFACT / "model.safetensors").exists(), reason="v0 artifact not present")
def test_decide_questions_mixed_one_forward():
    from krun_one import KrunOne

    m = KrunOne.from_pretrained(ARTIFACT, device="cpu", dtype="fp32")
    with pytest.raises(ValueError, match="not supported"):  # v0 declares only choice
        m.decide_questions(CTX, {"n": NOUL})
    m.cfg.update({"primitives": ["choice", "noul", "score"], "score_confidence": "normalized_entropy"})
    m.cfg["temperature"].update({"noul": 1.0, "score": 1.0})
    calls = []
    orig = m.logits
    m.logits = lambda items: calls.append(len(items)) or orig(items)
    a = m.decide_questions(CTX, {"department": {"type": "choice", "options": ["billing", "support", "sales"]},
                                 "needs_human": NOUL, "severity": SCORE})
    assert calls == [3]  # one forward for three heterogeneous questions
    assert a["department"]["type"] == "choice" and set(a["department"]["probabilities"]) == {"billing", "support", "sales"}
    assert 0.0 <= a["needs_human"]["noul"] <= 1.0 and set(a["needs_human"]) == {"type", "noul"}
    s = a["severity"]
    assert s["type"] == "score" and 0.0 <= s["score"] <= 2.0 and list(s["legend"]) == ["0", "1", "2"]
    assert s["score"] == pytest.approx(sum(int(k) * v for k, v in s["probabilities"].items()), abs=1e-5)
    # legacy API unchanged
    assert m.decide(CTX, ["billing", "support", "sales"])["top_option"] == a["department"]["top_option"]
