"""Krun One v1 fine-tune mechanics: curriculum ordering, and bit-identical resume through the train_decisions driver."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from krun_one.v1.training import curriculum_batches

ROOT = Path(__file__).resolve().parents[2]


def test_curriculum_every_item_once_and_progressive():
    rng = np.random.default_rng(0)
    n = 6000
    stages = list(rng.choice(["D1", "D2", "D3"], size=n, p=[0.5, 0.3, 0.2]))
    lengths = list(rng.integers(20, 300, size=n))
    ranges = {"D1": [0.0, 0.55], "D2": [0.2, 0.9], "D3": [0.45, 1.0]}
    b = curriculum_batches(lengths, stages, ranges, 4096, 64, np.random.Generator(np.random.PCG64(7)), chunk=512)
    flat = [i for x in b for i in x]
    assert sorted(flat) == list(range(n))
    pos = np.empty(n)
    pos[flat] = np.arange(n) / n
    st = np.array(stages)
    assert pos[st == "D1"].mean() < pos[st == "D2"].mean() < pos[st == "D3"].mean()
    assert (pos[st == "D3"] >= 0.4).mean() > 0.85          # OOD / adversarial live in the last part
    assert (pos[st == "D1"] <= 0.6).mean() > 0.85
    b2 = curriculum_batches(lengths, stages, ranges, 4096, 64, np.random.Generator(np.random.PCG64(7)), chunk=512)
    assert b == b2                                          # pure function of the generator state


def _run(args, tmp):
    return subprocess.run([sys.executable, str(ROOT / "scripts/v1/train_decisions.py"), *args], cwd=ROOT,
                          capture_output=True, text=True, timeout=600)


@pytest.mark.skipif(not (ROOT / "data/v1_probe/train.jsonl").exists(), reason="needs data/v1_probe")
def test_resume_is_identical(tmp_path):
    cfg = (ROOT / "configs/v1/finetune.yaml").read_text().replace("    - data/kd1/val_ml.jsonl\n", "")
    cfg = cfg.replace("save_every_minutes: 30", "save_every_minutes: 0").replace("save_every: 1000000000", "save_every: 4")
    c = tmp_path / "ft.yaml"
    c.write_text(cfg)
    base = ["smoke", "--config", str(c), "--train-file", "data/v1_probe/train.jsonl"]
    r = _run(base, tmp_path)
    assert r.returncode == 0, r.stderr[-2000:]
    run = ROOT / "runs/v1/ft/smoke"
    full = json.loads((run / "train_result.json").read_text())["log"]
    import shutil

    shutil.rmtree(run)
    r = _run(base + ["--max-steps", "4"], tmp_path)
    assert r.returncode == 0, r.stderr[-2000:]
    r = _run(base + ["--resume-from-checkpoint", "latest"], tmp_path)
    assert r.returncode == 0, r.stderr[-2000:]
    resumed = json.loads((run / "train_result.json").read_text())["log"]
    assert [round(e["loss"], 6) for e in full] == [round(e["loss"], 6) for e in resumed]
