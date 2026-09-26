"""Export -> load in a NEW Python process (no ./laya on the path, no training modules) -> same decisions."""

import json
import subprocess
import sys

import pytest

from krun_one.export import build_krun_config, export_artifact
from krun_one.laya_adapter import LAYA_DIR
from krun_one.model import KrunOne

REQS = [
    {"context": "my card was charged twice", "options": ["transaction_charged_twice", "card_payment_not_recognised",
                                                         "pending_card_payment", "reverted_card_payment?"]},
    {"context": "Find my meetings tomorrow", "options": [{"name": "calendar_search", "description": "Search calendar events"},
                                                         {"name": "send_email", "description": "Send an email"},
                                                         {"name": "web_search", "description": "Search the public web"}]},
    {"context": "What is the weather tomorrow?", "options": [{"name": "send_email", "description": "Send an email"},
                                                             {"name": "create_invoice", "description": "Create an invoice"}]},
]

CHILD = r"""
import json, sys
sys.path = [p for p in sys.path if 'laya' not in p]
from krun_one.model import KrunOne
m = KrunOne.from_pretrained(sys.argv[1], device="cpu")
reqs = json.loads(sys.argv[2])
bad = [k for k in sys.modules if k.startswith(("bakeoff", "krunbench", "krun_one.training", "krun_one.dataset", "rl_common"))]
print(json.dumps({"out": m.decide_batch(reqs), "bad_modules": bad}))
"""


@pytest.mark.skipif(not (LAYA_DIR / "tokenizer").is_dir(), reason="Laya checkpoint not found (KRUN_PROJECT_ROOT)")
def test_export_and_reload_in_new_process(tiny_model, tmp_path):
    model, enc_dir = tiny_model
    kcfg = build_krun_config(max_len=256, head_max_len=128, head_layers=1, n_act=2, temperatures={"intent": 1.3, "tool": 0.9},
                             score="top1_probability", thresholds={"intent": 0.3, "tool": 0.4}, target="0.98")
    art = export_artifact(model.eval(), LAYA_DIR / "tokenizer", enc_dir / "config.json", tmp_path / "art", kcfg,
                          {"m": 1}, {"p": 1}, "# test\n")
    assert {p.name for p in art.iterdir()} == {"model.safetensors", "config.json", "tokenizer", "krun_config.json",
                                               "metrics.json", "provenance.json", "README.md"}
    here = KrunOne.from_pretrained(art, device="cpu").decide_batch(REQS)
    r = subprocess.run([sys.executable, "-c", CHILD, str(art), json.dumps(REQS)], capture_output=True, text=True, check=True)
    child = json.loads(r.stdout.strip().splitlines()[-1])
    assert child["bad_modules"] == []
    for a, b in zip(here, child["out"]):
        assert a["top_option"] == b["top_option"]
        for k in a["probabilities"]:
            assert abs(a["probabilities"][k] - b["probabilities"][k]) < 1e-5
    assert here[1]["task"] == "tool" and here[0]["task"] == "intent"
    assert all(set(o) >= {"choice", "abstain", "confidence", "probabilities"} for o in here)
