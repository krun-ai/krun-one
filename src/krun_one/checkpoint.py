"""Checkpoint / resume — the mechanism validated in KRUN-004 (bit-identical resume on the RTX 4090),
reused as-is from ``bakeoff.checkpoint``:

* written every ``save_every`` optimizer steps into ``checkpoint-<step>.tmp/`` and atomically renamed;
  ``trainer_state.json`` is written last (completeness marker);
* model (safetensors), optimizer, scheduler, GradScaler, RNG (python / numpy / torch CPU / torch CUDA /
  shuffle generator), trainer state (global step, epoch, next batch, running loss, log), training config;
* resume refuses a checkpoint whose saved config differs from the current one;
* ``latest`` = highest complete checkpoint in the run directory.

Krun One's shuffle generator is a NumPy ``Generator``; its ``bit_generator.state`` (at the start of the
current epoch) is what goes into the ``shuffle_generator_epoch_start`` RNG slot.
"""

from __future__ import annotations

from pathlib import Path

from bakeoff.checkpoint import (COMPLETE, CheckpointConfig, list_checkpoints, load_checkpoint,  # noqa: F401
                                restore_rng, rng_state, save_checkpoint)


def resolve_resume(arg: str | None, run_dir: Path) -> Path | None:
    if not arg:
        return None
    if arg == "latest":
        cks = list_checkpoints(run_dir)
        if not cks:
            raise FileNotFoundError(f"--resume-from-checkpoint latest: no complete checkpoint in {run_dir}")
        return cks[-1]
    p = Path(arg)
    if not (p / COMPLETE).exists():
        raise FileNotFoundError(f"{p} is not a complete checkpoint")
    return p
