"""Every media file referenced by the M1.5c train / validation splits must exist and (audio) be readable.
Exit code 1 and a one-line summary otherwise. Written after the R1 / R2 incident (speech clips silently missing)."""

import collections
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("KRUN_M15_CONFIG", "configs/v1/m15c.yaml")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/v1"))
import train_m15 as M  # noqa: E402

bad = collections.Counter()
n = 0
for sp in ("train", "val_fit", "val_eval"):
    for r in M.read_split(sp):
        n += 1
        p = M.MEDIA_ROOT / r["media"]["path"]
        if not p.exists() or p.stat().st_size == 0:
            bad[(sp, r["modality"], r.get("source"), "missing")] += 1
        elif r["modality"] == "audio":
            import soundfile as sf

            try:
                sf.info(str(p))
            except Exception:  # noqa: BLE001
                bad[(sp, r["modality"], r.get("source"), "unreadable")] += 1
for k, v in bad.items():
    print(k, v)
print(f"checked {n} records, BAD {sum(bad.values())}")
sys.exit(1 if bad else 0)
