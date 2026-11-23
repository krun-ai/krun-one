"""M1 leakage fix-up after the KrunBench v1 --check (run locally, final_protected never leaves the laptop).

    python scripts/v1/m1_leakage_filter.py --drop-groups G1,G2  -> rewrites data/m1/{split}.jsonl + textviews + manifest

Drops every record of the listed split groups (all splits, incl. protected) and de-duplicates record ids (identical
source images inside CORD produce the same id). Recomputes file hashes in manifest.json and records what was removed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data/m1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop-groups", default="")
    a = ap.parse_args()
    drop = {g for g in a.drop_groups.split(",") if g}
    man = json.loads((D / "manifest.json").read_text())
    seen, removed = set(), {"groups": sorted(drop), "by_group": 0, "duplicate_ids": 0}
    import importlib.util

    spec = importlib.util.spec_from_file_location("B", ROOT / "scripts/v1/build_m1_data.py")
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)
    for sp in ("train", "val_fit", "val_eval", "protected"):
        p = D / f"{sp}.jsonl"
        keep = []
        for l in open(p, encoding="utf-8").read().split("\n"):
            if not l:
                continue
            r = json.loads(l)
            if r["split_group_id"] in drop:
                removed["by_group"] += 1
                continue
            if r["id"] in seen:
                removed["duplicate_ids"] += 1
                continue
            seen.add(r["id"])
            keep.append(l)
        p.write_text("\n".join(keep) + "\n", encoding="utf-8")
        man["files"][sp] = {"n": len(keep), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        if sp != "protected":
            with open(D / f"textview_{sp}.jsonl", "w", encoding="utf-8") as f:
                for l in keep:
                    f.write(json.dumps(B.text_view(json.loads(l)), ensure_ascii=False) + "\n")
    man["leakage_filter"] = removed
    (D / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({"removed": removed, "files": man["files"]}, indent=1))


if __name__ == "__main__":
    main()
