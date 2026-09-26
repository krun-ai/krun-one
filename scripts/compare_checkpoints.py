#!/usr/bin/env python
"""Bitwise comparison of two checkpoints (resume validation): weights, optimizer, scheduler, scaler, RNG, trainer state.

    .venv/bin/python scripts/compare_checkpoints.py runs/smoke_ref/checkpoint-60 runs/smoke_res/checkpoint-60
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

TELEMETRY = {"seconds", "elapsed_s", "peak_vram_mib", "steps_per_s", "tokens_per_s"}


def eq(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(eq(x, y) for x, y in zip(a, b))
    try:
        import numpy as np

        if isinstance(a, np.ndarray):
            return bool((a == b).all())
    except ImportError:
        pass
    return a == b


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in TELEMETRY}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def main(a: str, b: str) -> int:
    a, b = Path(a), Path(b)
    wa, wb = load_file(str(a / "model.safetensors")), load_file(str(b / "model.safetensors"))
    diffs = [k for k in wa if not torch.equal(wa[k], wb[k])]
    maxd = max((float((wa[k].float() - wb[k].float()).abs().max()) for k in diffs), default=0.0)
    res = {"tensors": len(wa), "weights_identical": not diffs, "differing_tensors": len(diffs), "max_abs_diff": maxd}
    for f in ("optimizer.pt", "scheduler.pt", "scaler.pt", "rng.pt"):
        res[f] = eq(torch.load(a / f, weights_only=False, map_location="cpu"), torch.load(b / f, weights_only=False, map_location="cpu"))
    sa, sb = (json.loads((p / "trainer_state.json").read_text()) for p in (a, b))
    res["trainer_state_identical_except_telemetry"] = strip(sa) == strip(sb)
    res["losses"] = {"a": [round(e["train_loss"], 6) for e in sa["log"]][-5:], "b": [round(e["train_loss"], 6) for e in sb["log"]][-5:]}
    print(json.dumps(res, indent=2))
    return 0 if res["weights_identical"] else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
