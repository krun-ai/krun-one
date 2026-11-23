"""Krun One v1.1 — multimodal M1.5c regression with a different TEXT CORE (nothing multimodal is retrained).

    KRUN_M15_CONFIG=configs/v1/m15c.yaml python scripts/v11/mm_eval.py --ckpt v1 --tag v1
    KRUN_M15_CONFIG=configs/v1/m15c.yaml python scripts/v11/mm_eval.py --ckpt runs/v11/r2/r2_s1 --tag r2_s1

Loads the registered M1.5c fusion weights (frozen SigLIP2 / Whisper / OCR / fusion) on top of the given text core and
writes the same val_fit / val_eval dumps as scripts/v1/train_m15.py dump; the registered gates are then computed by
scripts/v1/m15b_evaluate.py. ``--ckpt v1`` is the same-host reference. A core with the technical adapter is supported
(the adapter is folded into the encoder call); a core with the candidate-interaction block is not.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/v1"))
sys.path.insert(0, str(ROOT / "scripts/v11"))


class AdaptedEncoder(nn.Module):
    """encoder + technical adapter behind the encoder interface the multimodal code calls."""

    def __init__(self, encoder, adapter):
        super().__init__()
        self.encoder, self.adapter = encoder, adapter
        self.config = encoder.config

    def forward(self, input_ids=None, attention_mask=None, **kw):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kw)
        out.last_hidden_state = out.last_hidden_state + self.adapter(out.last_hidden_state)
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    import eval_v11
    import train_m15 as M
    T1 = M.T1

    orig = T1.load_core
    if args.ckpt != "v1":
        def load_core(device="cuda"):
            _, tok, ser = orig(device)                       # tokenizer / serializer of the frozen artifact (sha-checked)
            core, _ = eval_v11.load(args.ckpt)
            if core.cand is not None:
                raise SystemExit("the multimodal path does not know the candidate-interaction block")
            if core.adapter is not None:
                core.encoder = AdaptedEncoder(core.encoder, core.adapter)
                core.adapter = None
            return core.to(device).eval(), tok, ser

        T1.load_core = load_core
    run = ROOT / "runs/v11/mm" / args.tag
    src = ROOT / "runs/v1/m15c/krun-one-v1-m15c"
    (run / "step_001169").mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "train_result.json", run / "train_result.json")
    if not (run / "step_001169/m15.safetensors").exists():
        shutil.copy(src / "step_001169/m15.safetensors", run / "step_001169/m15.safetensors")
    assert json.loads((run / "train_result.json").read_text())["selected"] == "step_001169"
    M.cmd_dump(argparse.Namespace(run=str(run), limit=0, max_steps=0, run_name=None))
    spec = importlib.util.spec_from_file_location("m15b_evaluate", ROOT / "scripts/v1/m15b_evaluate.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    out = ROOT / "results/v11/multimodal" / f"{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    sys.argv = ["m15b_evaluate.py", "--run", str(run), "--out", str(out)]
    ev.main()


if __name__ == "__main__":
    main()
