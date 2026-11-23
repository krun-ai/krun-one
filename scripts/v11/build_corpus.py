"""Krun One v1.1 — build the technical corpus, KrunTechBench v1 (protected), KrunTechVal, the Tetris representation
sets and the v1 replay sample (configs/v11/r1_r2_protocol.yaml).

    uv run python scripts/v11/build_corpus.py bench      # data/v11/techbench/{protected,val}.jsonl  (built FIRST, then frozen)
    uv run python scripts/v11/build_corpus.py tetris     # data/v11/tetris_representation/{train,test}.jsonl
    uv run python scripts/v11/build_corpus.py train      # data/v11/corpus/train_tech.jsonl (+ aux_stats.json)
    uv run python scripts/v11/build_corpus.py replay     # data/v11/corpus/train_replay.jsonl
    uv run python scripts/v11/build_corpus.py manifest   # data/v11/corpus/manifest.json (counts, shares, sha256, leakage check)

An *example* is one problem in one rendering (raw = candidate names only; feat = name + explicit outcome / path).
Separation between training and evaluation, by construction:
  rule tables   requirement sets are split train / val / bench by hash; ``bench_only`` technologies never train
  simulators    different seeds + exact-duplicate rejection; the eviction and placement families are benchmark-only
  SWE-smith     split by REPOSITORY (allowlisted repositories only)
Every problem must fit one 1,024-token sequence with an untruncated context and <= 48 tokens per candidate.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from krun_one.v1.serialization import OPTION_TOKEN_CAP  # noqa: E402
from krun_one.v11 import harness, sims, swe, tech, training as T  # noqa: E402
from krun_one.v11.tables import TABLES  # noqa: E402

D = ROOT / "data/v11"
SEEDS = {"bench": 20261101, "val": 20261102, "train": 20261103, "tetris_test": 20261104, "tetris_train": 20261105, "replay": 20261106}
TIER_WEIGHT = {"A": 1.0, "rule": 0.7, "B": 0.7}
SWE_SPLIT = {"bench": 0.2, "val": 0.1}                     # by repository; the rest trains


def h_of(p: dict) -> str:
    return hashlib.sha256((p["instructions"] + "|" + p["context"] + "|" + "|".join(sorted(c["name"] for c in p["candidates"]))).encode()).hexdigest()


class Out:
    """Collects examples; enforces the sequence budget and exact-duplicate rejection against ``seen``."""

    def __init__(self, ser, seed: int, seen: set[str] | None = None, prefix: str = ""):
        self.ser, self.rng, self.items, self.dropped = ser, random.Random(seed), [], Counter()
        self.seen = set(seen or ())
        self.prefix, self.n = prefix, 0

    def fits(self, e: dict) -> bool:
        try:
            q = T.encode_tech(self.ser, e)
        except ValueError:
            return False
        if len(q.chunks) != 1 or q.chunks[0].context_truncated:
            return False
        if e.get("qtype", "choice") != "noul" and e["render"] == "feat":
            if any(len(self.ser._ids(" " + c["name"] + ": " + c["feat"])) > OPTION_TOKEN_CAP for c in e["candidates"]):
                return False
        return True

    def add(self, p: dict | None, renders=("raw", "feat"), bucket: str = "technical", tags: tuple = (), tier: str = "A",
            flip: dict | None = None, check_dup: bool = True) -> bool:
        if p is None:
            return False
        h = h_of(p)
        if check_dup and h in self.seen:
            self.dropped["duplicate"] += 1
            return False
        clean = {k: v for k, v in p.items() if not k.startswith("_")}
        exs = []
        for r in renders:
            e = {**clean, "render": r, "bucket": bucket, "tier": tier, "weight": TIER_WEIGHT[tier],
                 "tags": sorted(set(tags) | {clean["difficulty"]} | ({"ood"} if not clean["answerable"] else set())),
                 "flip": flip, "hash": h, "group": clean.get("group") or clean["slice"]}
            if not self.fits(e):
                self.dropped[f"budget:{clean['slice']}:{r}"] += 1
                return False
            exs.append(e)
        self.seen.add(h)
        for e in exs:
            e["id"] = f"{self.prefix}{self.n:07d}-{e['render']}"
            e["problem"] = f"{self.prefix}{self.n:07d}"
            self.items.append(e)
        self.n += 1
        return True

    def pair(self, got, renders=("raw", "feat"), bucket="hard_negative", tags=(), tier="A") -> bool:
        if got is None:
            return False
        b, f = got
        pid = f"{self.prefix}flip{self.n:07d}"
        base_gold = b["candidates"][b["gold"]]["name"]
        mark = len(self.items), self.n, set(self.seen)
        ok = (self.add(b, renders, bucket, tuple(tags) + ("constraint_flip",), tier, {"pair": pid, "role": "base", "base_gold": base_gold})
              and self.add(f, renders, bucket, tuple(tags) + ("constraint_flip",), tier, {"pair": pid, "role": "flip", "base_gold": base_gold}))
        if not ok:      # roll back a half-added pair
            del self.items[mark[0]:]
            self.n, self.seen = mark[1], mark[2]
        return ok

    def fill(self, n: int, make, max_tries: int | None = None, label: str = "") -> int:
        done = tries = 0
        while done < n and tries < (max_tries or 60 * n + 2000):
            tries += 1
            if make():
                done += 1
        if done < n:
            self.dropped[f"short:{label}"] += n - done
        return done

    def write(self, path: Path) -> dict:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in self.items))
        return {"path": str(path.relative_to(ROOT)), "examples": len(self.items), "problems": self.n,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "dropped": dict(self.dropped)}


# ------------------------------------------------------------------------------------------------ rule tables
def table_sets(split: str, extra: int = 0) -> dict[str, list[tuple]]:
    return {tb: [r for r in tech.req_sets_ext(tb, extra) if tech.split_of(tb, r) == split] for tb in TABLES}


def add_tech(o: Out, split: str, per_table: dict[str, int], bench: bool, extra: int = 0) -> None:
    """per_table: counts for answerable / hard_negative / flip / ood / large_k / unseen_technology (bench only)."""
    sets = table_sets(split, extra)
    for tb in TABLES:
        rs = sets[tb]
        if not rs:
            continue
        pick = lambda: o.rng.choice(rs)  # noqa: E731
        tags = ("unseen_combination",) if split == "bench" else ()
        o.fill(per_table["answerable"], lambda: o.add(tech.sample(o.rng, tb, pick(), k=o.rng.choice([None, None, 8])), tags=tags, tier="rule"),
               label=f"{tb}:answerable")
        o.fill(per_table["hard_negative"], lambda: o.add(tech.sample(o.rng, tb, pick(), hard_negative=True), bucket="hard_negative",
                                                         tags=tags + ("hard_negative",), tier="rule"), label=f"{tb}:hard_negative")
        o.fill(per_table["flip"], lambda: o.pair(tech.flip_pair(o.rng, tb, pick()), tags=tags + ("hard_negative",), tier="rule"),
               max_tries=40 * per_table["flip"] + 200, label=f"{tb}:flip")
        o.fill(per_table["ood"], lambda: o.add(tech.sample(o.rng, tb, pick(), want="unanswerable"), bucket="ood", tags=tags, tier="rule"),
               label=f"{tb}:ood")
        o.fill(per_table["large_k"], lambda: o.add(tech.sample(o.rng, tb, pick(), k=16), tags=tags + ("large_k",), tier="rule"),
               label=f"{tb}:large_k")
        if bench and any(s["bench_only"] for s in TABLES[tb]["options"].values()):
            allr = [r for r in tech.req_sets(tb)]

            def unseen():
                p = tech.sample(o.rng, tb, o.rng.choice(allr), allow_bench_only=True)
                if p is None or not p["meta"]["has_bench_only"]:
                    return False
                return o.add(p, tags=("unseen_technology",), tier="rule")

            o.fill(per_table["unseen_technology"], unseen, max_tries=80 * per_table["unseen_technology"] + 500, label=f"{tb}:unseen_tech")


# ------------------------------------------------------------------------------------------------ simulators
def add_sims(o: Out, per_family: dict[str, int], families: dict, flips: dict, unans: dict, extra_tags: tuple = ()) -> None:
    for name, gen in families.items():
        kmax = sims.MAX_K.get(name, 32)
        for k, key in ((8, "k8"), (16, "k16"), (32, "k32")):
            if k > kmax or not per_family.get(key):
                continue
            tags = extra_tags + (("large_k",) if k >= 16 else ())

            def make(k=k, tags=tags):
                p = gen(o.rng, k)
                if p is None or not p["answerable"]:
                    return False
                p["track"] = sims.TRACK.get(name, p["track"])
                hard = p["difficulty"] == "hard"
                return o.add(p, bucket="hard_negative" if hard else "technical", tags=tags + (("hard_negative",) if hard else ()))

            o.fill(per_family[key], make, label=f"{name}:k{k}")
        if name in flips and per_family.get("flip"):
            def mk():
                got = flips[name](o.rng, 8)
                if got is None:
                    return False
                for p in got:
                    p["track"] = sims.TRACK.get(name, p["track"])
                return o.pair(got, tags=extra_tags)

            o.fill(per_family["flip"], mk, label=f"{name}:flip")
        if name in unans and per_family.get("ood"):
            def mu():
                p = unans[name](o.rng, 8)
                if p is None or p["answerable"]:
                    return False
                p["track"] = sims.TRACK.get(name, p["track"])
                return o.add(p, bucket="ood", tags=extra_tags)

            o.fill(per_family["ood"], mu, label=f"{name}:ood")


def add_tetris(o: Out, n8: int, n16: int, views: str = "ABC", bucket: str = "technical") -> None:
    """The same state in the requested views; the three examples of one state share ``meta.state_id``."""
    for k, n in ((8, n8), (16, n16)):
        done = tries = 0
        while done < n and tries < 60 * n + 500:
            tries += 1
            v = sims.tetris_views(o.rng, k)
            if v is None:
                continue
            sid = f"{o.prefix}tetris{o.n:07d}"
            mark = len(o.items), o.n, set(o.seen)
            ok = True
            for a in views:
                p = v[a]
                p["meta"]["state_id"] = sid
                ok = ok and o.add(p, renders=("feat",) if a == "C" else ("raw",), bucket=bucket,
                                  tags=("tetris_" + a,) + (("large_k",) if k >= 16 else ()), check_dup=(a == views[0]))
            if not ok:
                del o.items[mark[0]:]
                o.n, o.seen = mark[1], mark[2]
                continue
            done += 1


# ------------------------------------------------------------------------------------------------ SWE-smith
def swe_rows() -> tuple[dict[str, list[dict]], dict[str, list[str]], dict]:
    allow = json.loads((D / "swe_smith_allowlist.json").read_text())
    mirrors = {r["mirror"]: r for r in allow["repos"]}
    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(Path.home() / ".cache/huggingface/hub/datasets--SWE-bench--SWE-smith/snapshots/*/data/*.parquet")))
    rows = defaultdict(list)
    for f in files:
        for r in pq.read_table(f, columns=["instance_id", "patch", "FAIL_TO_PASS", "PASS_TO_PASS", "repo", "problem_statement"]).to_pylist():
            if r["repo"] in mirrors or r["repo"].split("/", 1)[-1] in mirrors:
                rows[r["repo"]].append(r)
    trees = {}
    for m in rows:
        p = D / "swe_smith/trees" / (m.split("/", 1)[-1] + ".json")
        if p.exists():
            t = json.loads(p.read_text())
            trees[m] = t.get("paths") or t.get("py_paths") or t.get("files") or []
    return rows, trees, mirrors


def swe_split(mirror: str) -> str:
    x = int(hashlib.sha256(mirror.encode()).hexdigest(), 16) % 1000 / 1000
    return "bench" if x < SWE_SPLIT["bench"] else "val" if x < SWE_SPLIT["bench"] + SWE_SPLIT["val"] else "train"


def cut_statement(ser, text: str, max_tokens: int) -> str:
    ids = ser.tok(text, add_special_tokens=False)["input_ids"]
    return text if len(ids) <= max_tokens else ser.tok.decode(ids[:max_tokens]).rstrip() + " ..."


def add_swe(o: Out, split: str, quota: dict[str, int]) -> None:
    """quota: kind -> number of decisions; rows are visited in a seeded order, at most one decision kind per visit."""
    rows, trees, _ = swe_rows()
    repos = sorted(m for m in rows if swe_split(m) == split and trees.get(m))
    pool = [r for m in repos for r in rows[m]]
    fpool = swe.function_pool(pool)
    budget = {8: 520, 4: 560, 16: 400, 32: 120}
    for kind, n in quota.items():       # one decision of this kind per row, rows in a fresh seeded order, until the quota
        base = kind.split("@")[0]
        k = int(kind.split("@")[1]) if "@" in kind else 8
        order = pool[:]
        o.rng.shuffle(order)
        done = 0
        for r in order:
            if done >= n:
                break
            row = dict(r, problem_statement=cut_statement(o.ser, (r["problem_statement"] or "").strip(), budget.get(k, 400)))
            for p in swe.decisions(row, trees[r["repo"]], o.rng, [base], k, fpool):
                if p["slice"] != "swe_" + base:
                    continue
                tags = tuple(t for t, c in (("large_k", p["k"] >= 16),
                                            ("hard_negative", p["meta"].get("near_negatives", 0) >= p["k"] - 1 and p["k"] >= 4)) if c)
                bucket = "ood" if base == "file_ood" else "hard_negative" if "hard_negative" in tags else "technical"
                if o.add(p, renders=("feat",), bucket=bucket, tags=tags, tier=p["tier"]):
                    done += 1
        if done < n:
            o.dropped[f"short:swe:{kind}"] += n - done


# ------------------------------------------------------------------------------------------------ commands
def cmd_bench(ser) -> dict:
    out = {}
    for name, split, scale in (("protected", "bench", 1.0), ("val", "val", 0.5)):
        o = Out(ser, SEEDS[split], prefix=f"ktb-{name}-" if name == "protected" else "ktv-")
        s = lambda n: max(1, int(round(n * scale)))  # noqa: E731
        add_tech(o, split, {"answerable": s(60), "hard_negative": s(30), "flip": s(20), "ood": s(20), "large_k": s(15),
                            "unseen_technology": s(40)}, bench=(name == "protected"))
        if name == "protected":   # "seen combination" control: requirement sets that DO train, with new subsets / wording
            add_tech(o, "train", {"answerable": 25, "hard_negative": 10, "flip": 8, "ood": 0, "large_k": 0, "unseen_technology": 0}, bench=False)
            for e in o.items:
                if e["slice"] in TABLES and e["meta"]["combo_split"] == "train" and "unseen_technology" not in e["tags"]:
                    e["tags"] = sorted(set(e["tags"]) | {"seen_combination"})
        add_sims(o, {"k8": s(110), "k16": s(45), "k32": s(30), "flip": s(40), "ood": s(40)}, sims.GENERATORS, sims.FLIPS, sims.UNANSWERABLE)
        if name == "protected":
            add_sims(o, {"k8": 220, "k16": 80, "k32": 0, "flip": 0, "ood": 0}, sims.BENCH_ONLY, {}, {}, extra_tags=("unseen_family",))
        add_tetris(o, s(160), s(60), views="AC")
        if (D / "swe_smith_allowlist.json").exists():
            add_swe(o, split, {"file@8": s(320), "file@16": s(160), "file@32": s(110), "tests@8": s(220), "subsystem@8": s(150),
                               "function@8": s(110), "involved": s(120), "files@8": s(60), "file_ood@8": s(110)})
        out[name] = o
    return out


def cmd_train(ser, exclude: set[str]) -> Out:
    o = Out(ser, SEEDS["train"], seen=exclude, prefix="kt-")
    add_tech(o, "train", {"answerable": 900, "hard_negative": 320, "flip": 90, "ood": 500, "large_k": 180, "unseen_technology": 0}, bench=False)
    add_sims(o, {"k8": 1000, "k16": 500, "k32": 300, "flip": 250, "ood": 500}, sims.GENERATORS, sims.FLIPS, sims.UNANSWERABLE)
    add_tetris(o, 1300, 700, views="ABC")
    if (D / "swe_smith_allowlist.json").exists():
        add_swe(o, "train", {"file@8": 14000, "file@16": 9000, "file@4": 6000, "file@32": 3000, "tests@8": 12000, "subsystem@8": 8000,
                             "function@8": 6000, "involved": 6000, "files@8": 4000, "file_ood@8": 10000})
    return o


def cmd_replay(n_total: int, path: Path | None = None, seed: int | None = None) -> dict:
    """Stratified sample of the data v1 was trained on (data/kd1_r2/train.jsonl): sqrt-balanced over
    (category, language, k bucket) so rare strata (score, multi, large-k, OOD, PT-BR ...) keep a real share."""
    src = ROOT / "data/kd1_r2/train.jsonl"
    rng = random.Random(SEEDS["replay"] if seed is None else seed)

    def stratum(r):
        k = len(r["candidates"])
        kb = "k<=4" if k <= 4 else "k<=16" if k <= 16 else "k<=32" if k <= 32 else "k>=64"
        cat = (r.get("metadata") or {}).get("category") or r["task"]["type"]
        return f"{cat}|{r['context']['language'].split('-')[0]}|{kb}|{'ans' if r['answerable'] else 'ood'}"

    counts = Counter()
    with open(src, encoding="utf-8") as f:
        for line in f:
            counts[stratum(json.loads(line))] += 1
    z = sum(c ** 0.5 for c in counts.values())
    quota = {s: min(c, int(round(n_total * c ** 0.5 / z))) for s, c in counts.items()}
    keep_p = {s: quota[s] / counts[s] for s in counts}
    out, got = [], Counter()
    with open(src, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            s = stratum(r)
            if got[s] < quota[s] and rng.random() < keep_p[s] * 1.15:
                got[s] += 1
                out.append(line if line.endswith("\n") else line + "\n")
    path = path or D / "corpus/train_replay.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rng.shuffle(out)
    path.write_text("".join(out))
    recs = [json.loads(x) for x in out]
    sl = Counter(t for r in recs for t in (r.get("slices") or []))
    return {"path": str(path.relative_to(ROOT)), "n": len(out), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source": str(src.relative_to(ROOT)), "source_sha256_prefix": "d7668d3b",
            "by_type": dict(Counter(r["task"]["type"] for r in recs)), "by_language": dict(Counter(r["context"]["language"] for r in recs)),
            "by_family": dict(Counter(str(r["task"].get("family")) for r in recs)), "answerable": dict(Counter(str(r["answerable"]) for r in recs)),
            "k>=64": sum(len(r["candidates"]) >= 64 for r in recs), "k>=128": sum(len(r["candidates"]) >= 128 for r in recs),
            "slices": dict(sl.most_common()), "by_category": dict(Counter((r.get("metadata") or {}).get("category") for r in recs)),
            "by_source": dict(Counter(r["provenance"]["source_dataset"] for r in recs).most_common())}


def summarize(items: list[dict]) -> dict:
    return {"by_bucket": dict(Counter(e["bucket"] for e in items)), "by_track": dict(Counter(e["track"] for e in items)),
            "by_slice": dict(Counter(e["slice"] for e in items)), "by_render": dict(Counter(e["render"] for e in items)),
            "by_tier": dict(Counter(e["tier"] for e in items)), "by_qtype": dict(Counter(e.get("qtype", "choice") for e in items)),
            "tags": dict(Counter(t for e in items for t in e["tags"])), "by_k": dict(Counter(str(e["k"]) for e in items)),
            "groups": len({e["group"] for e in items})}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bench", "tetris", "train", "replay", "manifest"])
    ap.add_argument("--replay-n", type=int, default=162000)
    args = ap.parse_args()
    ser = harness.serializer(ROOT / "artifacts/krun-one-v1-text-final")
    man_path = D / "corpus/manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {"seeds": SEEDS, "files": {}}
    if args.cmd == "bench":
        for name, o in cmd_bench(ser).items():
            f = o.write(D / f"techbench/{name}.jsonl")
            man["files"][f"techbench_{name}"] = {**f, **summarize(o.items)}
            print(name, json.dumps({k: f[k] for k in ("examples", "problems", "dropped")}))
    elif args.cmd == "tetris":
        seen = {e["hash"] for f in ("protected", "val") for e in T.load_jsonl(D / f"techbench/{f}.jsonl")}
        for name, seed, n8, n16 in (("test", SEEDS["tetris_test"], 400, 200), ("train", SEEDS["tetris_train"], 14000, 6000)):
            o = Out(ser, seed, seen=seen, prefix=f"tr-{name}-")
            add_tetris(o, n8, n16, views="ABC")
            seen |= o.seen
            f = o.write(D / f"tetris_representation/{name}.jsonl")
            for v in "ABC":      # one file per view for the dedicated diagnostic runs
                (D / f"tetris_representation/{name}_{v}.jsonl").write_text(
                    "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in o.items if "tetris_" + v in e["tags"]))
            man["files"][f"tetris_{name}"] = {**f, **summarize(o.items)}
            print(name, json.dumps({k: f[k] for k in ("examples", "problems", "dropped")}))
    elif args.cmd == "train":
        ex = set()
        for f in ("techbench/protected.jsonl", "techbench/val.jsonl", "tetris_representation/test.jsonl"):
            ex |= {e["hash"] for e in T.load_jsonl(D / f)}
        o = cmd_train(ser, ex)
        f = o.write(D / "corpus/train_tech.jsonl")
        keys = T.cons_keys(o.items)
        (D / "corpus/aux_stats.json").write_text(json.dumps({"keys": keys, "stats": T.cons_stats(o.items, keys)}, indent=1))
        man["files"]["train_tech"] = {**f, **summarize(o.items)}
        print(json.dumps({k: f[k] for k in ("examples", "problems", "dropped")}), json.dumps(summarize(o.items)["by_bucket"]))
    elif args.cmd == "replay":
        man["files"]["train_replay"] = cmd_replay(args.replay_n)
        print(json.dumps({k: v for k, v in man["files"]["train_replay"].items() if k in ("n", "by_type", "answerable", "k>=64", "k>=128")}))
    else:
        tr = T.load_jsonl(D / "corpus/train_tech.jsonl")
        ev = {f: T.load_jsonl(D / f) for f in ("techbench/protected.jsonl", "techbench/val.jsonl", "tetris_representation/test.jsonl")}
        trh, trg = {e["hash"] for e in tr}, {e["group"] for e in tr if e["track"] == "software_repair"}
        leak = {f: {"exact_overlap": len(trh & {e["hash"] for e in v}),
                    "swe_repo_overlap": len(trg & {e["group"] for e in v if e["track"] == "software_repair"}),
                    "bench_only_tech_in_train": 0} for f, v in ev.items()}
        bo = {o_ for t in TABLES.values() for o_, s in t["options"].items() if s["bench_only"]}
        leak["bench_only_technology_names_in_train"] = sum(1 for e in tr for c in e["candidates"] if c["name"] in bo)
        tr_sets = {(e["slice"], tuple(sorted(e["meta"]["requirements"]))) for e in tr if e["slice"] in TABLES}
        leak["unseen_combination_sets_in_train"] = len({(e["slice"], tuple(sorted(e["meta"]["requirements"]))) for e in ev["techbench/protected.jsonl"]
                                                         if "unseen_combination" in e["tags"]} & tr_sets)
        n_rep = man["files"].get("train_replay", {}).get("n", 0)
        b = Counter(e["bucket"] for e in tr)
        tot = len(tr) + n_rep
        man["mixture"] = {"total_examples": tot, "replay": n_rep / tot, "technical": b["technical"] / tot,
                          "hard_negative": b["hard_negative"] / tot, "ood": b["ood"] / tot}
        man["leakage"] = leak
        print(json.dumps(man["mixture"]), json.dumps(leak))
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
