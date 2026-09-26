"""Krun One's port of the Laya architecture must be numerically identical to upstream.

Upstream ``./laya/rl_common.py`` is imported read-only (no bytecode written). Skipped when the
checkout is absent (e.g. on a serving box).
"""

import importlib.util
import sys

import pytest
import torch

from krun_one.laya_adapter import LAYA_DIR, load_laya
from krun_one.model import build_sequence, collate, option_texts

pytestmark = pytest.mark.skipif(not (LAYA_DIR / "model.safetensors").exists(), reason="./laya checkout not present")


@pytest.fixture(scope="module")
def upstream():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("laya_rl_common_upstream", LAYA_DIR / "rl_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def laya():
    model, tok, cfg = load_laya(device="cpu")
    return model.eval(), tok, cfg


CASES = [
    ("my card was charged twice", ["transaction_charged_twice", "card_payment_not_recognised", "Refund_not_showing_up"], None),
    ("Find my meetings tomorrow", ["calendar_search", "send_email", "web_search"],
     {"calendar_search": "Search calendar events", "send_email": "Send an email", "web_search": "Search the public web"}),
    ("x " * 600, [f"label_{i}_with_a_rather_long_name_that_takes_tokens" for i in range(40)], None),  # shrink + truncation
]


@pytest.mark.parametrize("context,names,desc", CASES)
def test_build_sequence_identical(upstream, laya, context, names, desc):
    _, tok, cfg = laya
    ins = "Choose the option that best describes the intent of the user message."
    texts = option_texts(names, desc)
    crit = {t: None for t in texts} if not desc else {n: desc[n] for n in names}
    q = {"t": "choice", "ins": ins, "crit": crit}
    for max_len, head in ((cfg["max_len"], cfg["head_max_len"]), (1024, 768)):
        ours = build_sequence(tok, context, texts, ins, max_len, head)
        theirs = upstream.build_sequence(tok, context, q, max_len, head)
        assert ours == (list(theirs[0]), list(theirs[1]))


def test_forward_identical(upstream, laya):
    model, tok, cfg = laya
    ref = upstream.build_model(cfg, encoder_dir=str(LAYA_DIR / "encoder"))
    ref.load_state_dict(model.state_dict(), strict=True)
    ref.eval()
    items = []
    for context, names, desc in CASES[:2]:
        ids, markers = build_sequence(tok, context, option_texts(names, desc), "Choose.", cfg["max_len"], cfg["head_max_len"])
        items.append({"ids": ids, "markers": markers})
    b = collate(items, tok.pad_token_id)
    with torch.no_grad():
        ours = model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"], with_act=True)
        theirs = ref(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
    assert torch.equal(ours[0], theirs[0])
    assert torch.equal(ours[1], theirs[1])
