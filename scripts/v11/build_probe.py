"""Krun One v1.1 R0 — build the Technical Capacity Probe (configs/v11/r0_protocol.yaml).

    uv run python scripts/v11/build_probe.py            # -> data/v11/probe/{probe,ltrain}.jsonl + manifest.json

``probe.jsonl``  held-out problems (never trained on): multi-way k = 8 / 16 / 32, pairwise (derived from k = 8),
                 constraint-flip pairs, none-of-the-above. Truth = simulation / arithmetic / rule tables.
``ltrain.jsonl`` problems from a different seed for the R0-L learnability diagnostic (throwaway weights only).
                 Tech selection: the whole ``messaging`` category and 25% of the requirement sets of the other
                 categories never appear in it (``meta.ltrain_seen = false`` on the probe side).

Every problem must fit ONE chunk with an untruncated context in both conditions with the v1 serializer, otherwise
it is dropped (counts in the manifest) — candidates that fall into different chunks cannot attend to each other.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from krun_one.v1.serialization import Budget, Serializer, humanize_label  # noqa: E402
from krun_one.v11 import envs, probe  # noqa: E402

OUT = ROOT / "data/v11/probe"
PROBE_SEED, LTRAIN_SEED = 20260930, 20261001
# scheduling lists every node in the state: 32 nodes + 32 explicit outcomes do not fit one 1024-token sequence
# techsel: only two unrelated categories are safe fillers for each category (<= 24 candidates)
MAX_K = {"scheduling": 16, "techsel": 16}
HELD_OUT_CATEGORY = "messaging"


def set_key(rs: dict) -> str:
    return rs["cat"] + ":" + "+".join(sorted(rs["req"]))


def set_seen(rs: dict) -> bool:
    """Deterministic 75/25 split of requirement sets (by hash); messaging is held out entirely."""
    if rs["cat"] == HELD_OUT_CATEGORY:
        return False
    return int(hashlib.sha256(set_key(rs).encode()).hexdigest(), 16) % 4 != 0


class Builder:
    def __init__(self, ser: Serializer, seed: int):
        self.ser, self.rng, self.items, self.dropped, self.seen = ser, random.Random(seed), [], Counter(), set()

    def fits(self, p: dict) -> bool:
        for cond in probe.CONDITIONS:
            try:
                q = self.ser.encode("choice", p["context"], probe.options(p, cond), p["instructions"])
            except ValueError:
                return False
            if len(q.chunks) != 1 or q.chunks[0].context_truncated:
                return False
        return True

    def add(self, p: dict | None, pid: str, **extra) -> bool:
        if p is None:
            return False
        if any(humanize_label(c["name"]) != c["name"] for c in p["candidates"]):
            raise AssertionError("candidate name is not humanization-stable: %r" % [c["name"] for c in p["candidates"]])
        h = hashlib.sha256((p["context"] + "|" + "|".join(c["name"] for c in p["candidates"])).encode()).hexdigest()
        if h in self.seen:
            self.dropped["duplicate"] += 1
            return False
        if not self.fits(p):
            self.dropped["budget:%s:k%d" % (p["slice"], p["k"])] += 1
            return False
        self.seen.add(h)
        self.items.append({"id": pid, **p, **extra, "hash": h})
        return True

    def fill(self, gen, n: int, k: int, prefix: str, max_tries: int = 200_000, **extra) -> None:
        done = tries = 0
        while done < n and tries < max_tries:
            tries += 1
            if self.add(gen(self.rng, k), f"{prefix}-{done:05d}", **extra):
                done += 1
            if tries == 400 and done == 0:
                raise RuntimeError(f"{prefix}: nothing fits ({dict(self.dropped)})")
        if done < n:
            raise RuntimeError(f"{prefix}: only {done}/{n}")

    def fill_pairs(self, gen, n: int, k: int, prefix: str) -> None:
        done = tries = 0
        while done < n and tries < 200_000:
            tries += 1
            got = gen(self.rng, k)
            if got is None:
                continue
            b, f = got
            pair = f"{prefix}-{done:05d}"
            base_gold = b["candidates"][b["gold"]]["name"]
            if not self.fits(b) or not self.fits(f):
                self.dropped["budget:%s:flip" % b["slice"]] += 1
                continue
            if self.add(b, pair + "-base", flip={"pair": pair, "role": "base", "base_gold": base_gold}):
                self.add(f, pair + "-flip", flip={"pair": pair, "role": "flip", "base_gold": base_gold})
                done += 1


def tech_sets(answerable: bool, seen: bool | None) -> list[dict]:
    out = [s for c in envs.TECH for s in envs.tech_requirement_sets(c) if (s["gold"] is not None) == answerable]
    return out if seen is None else [s for s in out if set_seen(s) == seen]


def tech_fill(b: Builder, sets: list[dict], n: int, k: int, prefix: str) -> None:
    done = tries = 0
    while done < n and tries < 100_000:
        rs = sets[tries % len(sets)]
        tries += 1
        p = envs.tech_problem(b.rng, rs, k)
        if p is not None:
            p["meta"].update(ltrain_seen=set_seen(rs), n_satisfiers=len(rs["sat"]))
        if b.add(p, f"{prefix}-{done:05d}"):
            done += 1
    if done < n:
        raise RuntimeError(f"{prefix}: only {done}/{n}")


def tech_fill_pairs(b: Builder, pairs: list[tuple[dict, dict]], n: int, k: int, prefix: str) -> None:
    done = tries = 0
    while done < n and tries < 100_000:
        base, flip = pairs[tries % len(pairs)]
        tries += 1
        got = envs.tech_pair(b.rng, base, flip, k)
        if got is None:
            continue
        pb, pf = got
        for p, rs in ((pb, base), (pf, flip)):
            p["meta"].update(ltrain_seen=set_seen(base) and set_seen(flip), n_satisfiers=len(rs["sat"]))
        pair = f"{prefix}-{done:05d}"
        base_gold = pb["candidates"][pb["gold"]]["name"]
        if not b.fits(pb) or not b.fits(pf):
            continue
        if b.add(pb, pair + "-base", flip={"pair": pair, "role": "base", "base_gold": base_gold}):
            b.add(pf, pair + "-flip", flip={"pair": pair, "role": "flip", "base_gold": base_gold})
            done += 1


def build_probe(ser) -> Builder:
    b = Builder(ser, PROBE_SEED)
    for name, gen in envs.GENERATORS.items():
        for k, n in ((8, 300), (16, 200), (32, 200)):
            if k <= MAX_K.get(name, 32):
                b.fill(gen, n, k, f"{name}-k{k}")
    for name, gen in envs.FLIPS.items():
        b.fill_pairs(gen, 150, 8, f"{name}-flip")
    for name, gen in envs.UNANSWERABLE.items():
        b.fill(gen, 150, 8, f"{name}-una")
    for k, n in ((8, 320), (16, 200)):
        tech_fill(b, tech_sets(True, None), n, k, f"techsel-k{k}")
    tech_fill_pairs(b, [p for c in envs.TECH for p in envs.tech_flip_pairs(c)], 138, 8, "techsel-flip")
    tech_fill(b, tech_sets(False, None), 150, 8, "techsel-una")
    prng = random.Random(PROBE_SEED + 1)
    for p in [p for p in b.items if p["k"] == 8 and p["answerable"] and not p["flip"]]:
        b.items.extend(probe.pairwise(p, prng))
    return b


def build_ltrain(ser, exclude: set[str]) -> Builder:
    b = Builder(ser, LTRAIN_SEED)
    b.seen = set(exclude)                     # nothing identical to a probe problem
    for name, gen in envs.GENERATORS.items():
        for k, n in ((8, 3000), (16, 1800), (32, 1200)):
            if k <= MAX_K.get(name, 32):
                b.fill(gen, n, k, f"lt-{name}-k{k}")
    for name, gen in envs.FLIPS.items():
        b.fill_pairs(gen, 600, 8, f"lt-{name}-flip")
    for name, gen in envs.UNANSWERABLE.items():
        b.fill(gen, 400, 8, f"lt-{name}-una")
    seen_ans = tech_sets(True, True)
    for k, n in ((8, 2000), (16, 1000)):
        tech_fill(b, seen_ans, n, k, f"lt-techsel-k{k}")
    seen_pairs = [(x, y) for c in envs.TECH for x, y in envs.tech_flip_pairs(c) if set_seen(x) and set_seen(y)]
    tech_fill_pairs(b, seen_pairs, 300, 8, "lt-techsel-flip")
    tech_fill(b, tech_sets(False, True), 400, 8, "lt-techsel-una")
    return b


def main() -> None:
    from transformers import AutoTokenizer

    art = ROOT / "artifacts/krun-one-v1-text-final"
    sq = json.loads((art / "decision_config.json").read_text())["sequence"]
    ser = Serializer(AutoTokenizer.from_pretrained(str(art / "tokenizer")),
                     Budget(sq["max_len"], sq["head_max_len"], max_candidates_per_chunk=sq["max_candidates_per_chunk"]))
    OUT.mkdir(parents=True, exist_ok=True)
    pb = build_probe(ser)
    lt = build_ltrain(ser, pb.seen)
    man = {"probe_seed": PROBE_SEED, "ltrain_seed": LTRAIN_SEED, "sequence": sq, "files": {}}
    for name, b in (("probe", pb), ("ltrain", lt)):
        path = OUT / f"{name}.jsonl"
        path.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in b.items))
        cells = Counter()
        for p in b.items:
            kind = ("pair-" + p["pair_kind"]) if "pair_kind" in p else ("flip-" + p["flip"]["role"]) if p["flip"] \
                else "unanswerable" if not p["answerable"] else "multiway"
            cells[f"{p['slice']}/{kind}/k{p['k']}"] += 1
        man["files"][name] = {"path": str(path.relative_to(ROOT)), "n": len(b.items),
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "cells": dict(sorted(cells.items())),
                              "dropped": dict(b.dropped), "difficulty": dict(Counter(p["difficulty"] for p in b.items))}
    seen = [set_key(s) for s in tech_sets(True, None) if set_seen(s)]
    held = [set_key(s) for s in tech_sets(True, None) if not set_seen(s)]
    man["techsel"] = {"held_out_category": HELD_OUT_CATEGORY, "requirement_sets_seen": seen, "requirement_sets_held_out": held}
    (OUT / "manifest.json").write_text(json.dumps(man, indent=2))
    print(json.dumps({k: {"n": v["n"], "dropped": v["dropped"]} for k, v in man["files"].items()}, indent=1))


if __name__ == "__main__":
    main()
