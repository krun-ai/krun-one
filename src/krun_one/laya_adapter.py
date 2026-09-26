"""Load the public upstream Laya checkpoint (./laya, read-only) into Krun One's DecisionModel.

Nothing under ./laya is imported or modified here: only the weight file, the encoder config,
the tokenizer and ``rl_agent_config.json`` are *read*. The upstream checkpoint is both
(a) the Laya baseline and (b) the initialization of Krun One.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from .model import DecisionModel, build_encoder

# Workspace layout: <project>/experiments/krun_one/src/krun_one/ → <project> holds laya/ and benchmarks/. A standalone
# checkout sets KRUN_PROJECT_ROOT to the directory that holds laya/ (and benchmarks/krunbench/ for data preparation).
PROJECT_ROOT = Path(os.environ.get("KRUN_PROJECT_ROOT") or Path(__file__).resolve().parents[4])
LAYA_DIR = PROJECT_ROOT / "laya"


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def laya_config(laya_dir: Path = LAYA_DIR) -> dict[str, Any]:
    return json.loads((laya_dir / "rl_agent_config.json").read_text())


def load_laya(laya_dir: Path = LAYA_DIR, device: str | torch.device = "cpu") -> tuple[DecisionModel, Any, dict[str, Any]]:
    """Returns (model, tokenizer, upstream rl_agent_config). Loaded with strict=True."""
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    cfg = laya_config(laya_dir)
    tok = AutoTokenizer.from_pretrained(str(laya_dir / "tokenizer"))
    model = DecisionModel(build_encoder(laya_dir / "encoder"), cfg["head_layers"], len(cfg["act_costs"]) + 1)
    model.load_state_dict(load_file(str(laya_dir / "model.safetensors")), strict=True)
    return model.to(device), tok, cfg


def laya_provenance(laya_dir: Path = LAYA_DIR, with_hash: bool = True) -> dict[str, Any]:
    import subprocess

    try:
        commit = subprocess.run(["git", "--no-optional-locks", "-C", str(laya_dir), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - on the pod ./laya is an rsync copy without .git
        commit = None
    out = {"source": "convaiinnovations/laya (repo root checkpoint, ModernBERT-large, 421M)", "git_commit": commit,
           "rl_agent_config": laya_config(laya_dir)}
    if with_hash:
        out["model_safetensors_sha256"] = sha256_file(laya_dir / "model.safetensors")
    return out
