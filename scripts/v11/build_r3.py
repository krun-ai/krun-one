"""Krun One v1.1 R3 — corpus and benchmark extension (configs/v11/r3_protocol.yaml).

    uv run python scripts/v11/build_r3.py relabel    # KrunTechBench v1 / KrunTechVal with the swe_involved labels restored
    uv run python scripts/v11/build_r3.py bench      # data/v11/techbench/{protected_r3,val_r3}.jsonl (built FIRST, then frozen)
    uv run python scripts/v11/build_r3.py train      # data/v11/r3/train_tech.jsonl (+ aux_stats.json)
    uv run python scripts/v11/build_r3.py replay     # data/v11/r3/train_replay.jsonl
    uv run python scripts/v11/build_r3.py manifest   # data/v11/r3/manifest.json: hashes, counts, shares, separation checks

What changes against the R1 / R2 corpus (data/v11/corpus, kept untouched and marked as defective in its manifest):
  * swe_involved (noul) examples carry their label (R1 / R2: 6,000 unlabeled training examples, 120 unscored items);
  * rule tables: fact probes, counterfactual families, invented-product catalogs, requirement sets one property larger;
  * capability curriculum (src/krun_one/v11/capability.py): numeric, temporal, multi-hop, long-policy, trade-off, mixed —
    every item belongs to a counterfactual family; choice, true / false and ordinal questions;
  * every example has ``family`` = {"id", "variant", "changed"} when it has siblings (used by family-aware batching and by
    the stuck-rate metric).
The benchmark extension uses requirement sets of the BENCH split, benchmark-only scenarios and its own seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/v11"))

import build_corpus as B  # noqa: E402
from krun_one.v11 import capability as C, harness, sims, tech, training as T  # noqa: E402
from krun_one.v11.tables import TABLES  # noqa: E402

D = ROOT / "data/v11"
R3 = D / "r3"
SEEDS = {"bench": 20261201, "val": 20261202, "train": 20261203, "replay": 20261204}
B.TIER_WEIGHT.update({"A": 1.0, "rule": 0.7, "B": 0.7})
# share of the capability curriculum per capability (brief: 20 / 20 / 20 / 15 / 15 / 10) -> families per generator
CAP_FAMILIES = {"pol_document": 1700, "pol_horizon": 650, "pol_backlog": 650, "hop_lookup": 1500, "hop_rules": 1500,
                "num_limits": 1200, "num_budget": 600, "num_capacity": 600, "num_errbudget": 600,
                "tmp_order": 750, "tmp_window": 750, "tmp_dates": 750, "trd_priorities": 2250,
                "mix_window_budget": 500, "mix_sla_lookup": 500, "mix_policy_totals": 500}
CAP_OOD = 260                      # none-of-the-above problems per generator that supports them


def renders_of(p: dict) -> tuple[str, ...]:
    return ("raw", "feat") if p.get("qtype", "choice") == "choice" else ("raw",)


def add_family(o: B.Out, fam: list[dict] | None, bucket: str, tags: tuple = (), tier: str = "A", renders=None) -> bool:
    """All variants or none; every example of the family gets the same family id."""
    if not fam:
        return False
    fid = f"{o.prefix}fam{o.n:07d}"
    mark = len(o.items), o.n, set(o.seen)
    base_gold = fam[0]["candidates"][fam[0]["gold"]]["name"] if fam[0].get("gold") is not None else None
    for p in fam:
        f = p.get("family") or {"variant": 0, "changed": "base"}
        p["family"] = {**f, "id": fid, "size": len(fam), "base_gold": base_gold}
        t = tuple(tags) + (("family",) if len(fam) > 1 else ()) + (p.get("qtype", "choice"),)
        b = "ood" if not p["answerable"] else bucket
        if not o.add(p, renders or renders_of(p), b, t, tier):
            del o.items[mark[0]:]
            o.n, o.seen = mark[1], mark[2]
            return False
    return True


def cap_tags(fam: list[dict]) -> tuple:
    m = fam[0]["meta"]
    out = ["capability", "unseen_scenario" if m.get("scenario") in C.BENCH_SCENARIOS else "seen_scenario"]
    if "hops" in m:
        out.append(f"hops_{m['hops']}")
    return tuple(out)


def add_capability(o: B.Out, split: str, families: dict[str, int], n_ood: int) -> None:
    for name, n in families.items():
        gen = C.GENERATORS[name]

        def make():
            fam = gen(o.rng, split)
            return bool(fam) and add_family(o, fam, "capability", cap_tags(fam))

        o.fill(n, make, label=f"{name}:family")
        if name in C.UNANSWERABLE and n_ood:
            def mu():
                fam = gen(o.rng, split, form="choice", want="unanswerable")
                return bool(fam) and add_family(o, fam, "ood", cap_tags(fam))

            o.fill(n_ood, mu, max_tries=400 * n_ood + 2000, label=f"{name}:ood")


def add_tables_r3(o: B.Out, split: str, n: dict[str, int], probes: bool, extra_req: int) -> None:
    """Fact probes (train only), requirement-set families and invented-product catalogs for every rule table."""
    combo = ("unseen_combination",) if split == "bench" else ()
    for tb in TABLES:
        if probes:
            o.fill(n["fact"], lambda: add_family(o, tech.probe_fact(o.rng, tb), "technical", ("probe",), "rule"),
                   max_tries=30 * n["fact"], label=f"{tb}:fact")
            o.fill(n["which"], lambda: add_family(o, [x] if (x := tech.probe_which(o.rng, tb)) else None, "technical", ("probe",), "rule"),
                   max_tries=30 * n["which"], label=f"{tb}:which")
            o.fill(n["overhead"], lambda: add_family(o, [x] if (x := tech.probe_overhead(o.rng, tb)) else None, "technical", ("probe",), "rule",
                                                     renders=("raw",)), max_tries=30 * n["overhead"], label=f"{tb}:overhead")
        o.fill(n["family"], lambda: add_family(o, tech.family(o.rng, tb, split, extra_req=extra_req), "hard_negative",
                                               combo + ("hard_negative", "table_family"), "rule"), max_tries=60 * n["family"], label=f"{tb}:family")
        o.fill(n["catalog"], lambda: add_family(o, tech.catalog(o.rng, tb, split), "technical", combo + ("catalog",), "rule"),
               max_tries=40 * n["catalog"], label=f"{tb}:catalog")
        o.fill(n["catalog_ood"], lambda: add_family(o, tech.catalog(o.rng, tb, split, want="unanswerable"), "ood", combo + ("catalog",), "rule"),
               max_tries=40 * n["catalog_ood"] + 50, label=f"{tb}:catalog_ood")


def families_from_flips(items: list[dict]) -> None:
    """Constraint-flip pairs of the R1 / R2 generators become families of two (same id for both members and renderings)."""
    for e in items:
        if e.get("flip") and not e.get("family"):
            e["family"] = {"id": e["flip"]["pair"], "variant": 0 if e["flip"]["role"] == "base" else 1, "size": 2,
                           "changed": "base" if e["flip"]["role"] == "base" else "one constraint", "base_gold": e["flip"]["base_gold"]}


def summarize(items: list[dict]) -> dict:
    s = B.summarize(items)
    s["families"] = len({e["family"]["id"] for e in items if e.get("family") and e["family"].get("size", 1) > 1})
    s["examples_in_families"] = sum(1 for e in items if e.get("family") and e["family"].get("size", 1) > 1)
    s["by_language"] = {"en": len(items)}
    s["label_counts"] = {"choice_with_gold": sum(e.get("qtype", "choice") == "choice" and e["gold"] is not None for e in items),
                         "choice_none_of_the_above": sum(e.get("qtype", "choice") == "choice" and not e["answerable"] for e in items),
                         "noul_true": sum(e.get("qtype") == "noul" and e["gold"] == 1 for e in items),
                         "noul_false": sum(e.get("qtype") == "noul" and e["gold"] == 0 for e in items),
                         "noul_unlabeled": sum(e.get("qtype") == "noul" and e["gold"] is None for e in items),
                         "score": sum(e.get("qtype") == "score" for e in items), "multi": sum(e.get("qtype") == "multi" for e in items)}
    s["by_evidence"] = dict(Counter(e["evidence"]["type"] for e in items))
    return s


def check_labels(items: list[dict]) -> None:
    for e in items:
        qt = e.get("qtype", "choice")
        if qt == "multi":
            assert e["gold_multi"], e["id"]
        elif e["answerable"]:
            assert e["gold"] is not None and 0 <= e["gold"] < len(e["candidates"]), e["id"]
        else:
            assert qt == "choice" and e["gold"] is None, e["id"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ commands
def cmd_relabel(ser) -> dict:
    """Rebuild KrunTechBench v1 / KrunTechVal with the fixed builder and check that NOTHING but the missing labels differs."""
    out = {}
    built = B.cmd_bench(ser)
    for name, o in built.items():
        old = {e["id"]: e for e in T.load_jsonl(D / f"techbench/{name}.jsonl")}
        new = {e["id"]: e for e in o.items}
        assert set(old) == set(new), (name, len(old), len(new))
        changed = 0
        for i, e in new.items():
            a, b = dict(old[i]), dict(e)
            if a.get("qtype") == "noul":
                assert a["gold"] is None and b["gold"] in (0, 1), i
                a["gold"] = b["gold"]
                changed += 1
            assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True), ("rebuild differs", i)
        path = D / f"techbench/{name}_v1_labeled.jsonl"
        path.write_text("".join(json.dumps(new[i], ensure_ascii=False) + "\n" for i in old))
        check_labels(list(new.values()))
        out[name] = {"path": str(path.relative_to(ROOT)), "examples": len(new), "labels_restored": changed, "sha256": sha(path),
                     "source_sha256": sha(D / f"techbench/{name}.jsonl"), "identical_except_restored_labels": True}
        print(name, json.dumps(out[name]))
    return out


def cmd_bench(ser) -> dict:
    out = {}
    seen = {e["hash"] for f in ("protected", "val") for e in T.load_jsonl(D / f"techbench/{f}.jsonl")}
    seen |= {e["hash"] for e in T.load_jsonl(D / "tetris_representation/test.jsonl")}
    for name, split, scale in (("protected_r3", "bench", 1.0), ("val_r3", "val", 0.5)):
        o = B.Out(ser, SEEDS[split], seen=seen, prefix="ktb3-" if split == "bench" else "ktv3-")
        add_capability(o, split, {g: max(20, int(round(f * scale * 0.11))) for g, f in CAP_FAMILIES.items()}, int(30 * scale))
        add_tables_r3(o, split, {"family": int(40 * scale), "catalog": int(40 * scale), "catalog_ood": int(8 * scale), "fact": 0, "which": 0, "overhead": 0},
                      probes=False, extra_req=0)
        seen |= o.seen
        check_labels(o.items)
        f = o.write(D / f"techbench/{name}.jsonl")
        out[name] = {**f, **summarize(o.items)}
        print(name, json.dumps({k: f[k] for k in ("examples", "problems")}), json.dumps(summarize(o.items)["by_track"]))
    return out


def cmd_train(ser) -> tuple[dict, list[dict]]:
    ex = set()
    for f in ("techbench/protected.jsonl", "techbench/val.jsonl", "techbench/protected_r3.jsonl", "techbench/val_r3.jsonl",
              "tetris_representation/test.jsonl"):
        ex |= {e["hash"] for e in T.load_jsonl(D / f)}
    o = B.Out(ser, SEEDS["train"], seen=ex, prefix="k3-")
    B.add_tech(o, "train", {"answerable": 900, "hard_negative": 320, "flip": 90, "ood": 500, "large_k": 180, "unseen_technology": 0}, bench=False, extra=1)
    add_tables_r3(o, "train", {"fact": 300, "which": 150, "overhead": 60, "family": 250, "catalog": 320, "catalog_ood": 60}, probes=True, extra_req=1)
    B.add_sims(o, {"k8": 1000, "k16": 500, "k32": 300, "flip": 250, "ood": 500}, sims.GENERATORS, sims.FLIPS, sims.UNANSWERABLE)
    B.add_tetris(o, 1300, 700, views="ABC")
    B.add_swe(o, "train", {"file@8": 14000, "file@16": 9000, "file@4": 6000, "file@32": 3000, "tests@8": 12000, "subsystem@8": 8000,
                           "function@8": 6000, "involved": 6000, "files@8": 4000, "file_ood@8": 10000})
    add_capability(o, "train", CAP_FAMILIES, CAP_OOD)
    families_from_flips(o.items)
    check_labels(o.items)
    f = o.write(R3 / "train_tech.jsonl")
    keys = T.cons_keys(o.items)
    (R3 / "aux_stats.json").write_text(json.dumps({"keys": keys, "stats": T.cons_stats(o.items, keys)}, indent=1))
    return {**f, **summarize(o.items)}, o.items


def ngrams(text: str, n: int = 8) -> set[int]:
    w = text.lower().split()
    return {hash(" ".join(w[i:i + n])) for i in range(len(w) - n + 1)}


def cmd_manifest(man: dict) -> dict:
    tr = T.load_jsonl(R3 / "train_tech.jsonl")
    ev = {f: T.load_jsonl(D / f) for f in ("techbench/protected.jsonl", "techbench/val.jsonl", "techbench/protected_r3.jsonl",
                                           "techbench/val_r3.jsonl", "tetris_representation/test.jsonl")}
    trh = {e["hash"] for e in tr}
    trg = {e["group"] for e in tr if e["track"] == "software_repair"}
    leak = {f: {"exact_overlap": len(trh & {e["hash"] for e in v}),
                "swe_repo_overlap": len(trg & {e["group"] for e in v if e["track"] == "software_repair"})} for f, v in ev.items()}
    bo = {o_ for t in TABLES.values() for o_, s in t["options"].items() if s["bench_only"]}
    leak["bench_only_technology_names_in_train"] = sum(1 for e in tr for c in e["candidates"] if c["name"] in bo) + \
        sum(1 for e in tr if e["meta"].get("option") in bo)
    rs = lambda e: (e["meta"].get("table"), tuple(sorted(e["meta"]["requirements"])))  # noqa: E731
    tr_sets = {rs(e) for e in tr if "requirements" in e["meta"]}
    bench_sets = {rs(e) for f in ("techbench/protected.jsonl", "techbench/protected_r3.jsonl") for e in ev[f]
                  if "unseen_combination" in e["tags"]}
    leak["unseen_combination_sets_in_train"] = len(bench_sets & tr_sets)
    leak["train_sets_not_in_train_split"] = sum(1 for t, r in tr_sets if tech.split_of(t, r) != "train")
    leak["benchmark_only_scenarios_in_train"] = sum(1 for e in tr if e["meta"].get("scenario") in C.BENCH_SCENARIOS)
    leak["benchmark_only_simulator_families_in_train"] = sum(1 for e in tr if e["slice"] in sims.BENCH_ONLY)
    # JevBench (EVAL_ONLY): no 8-word sequence of any public item may appear in a training context or instruction
    jv = set()
    for f in ("easy", "original", "hard"):
        for line in open(D / f"external/jevbench/{f}.jsonl", encoding="utf-8").read().split("\n"):
            if line:
                it = json.loads(line)
                st = it["state"] if isinstance(it["state"], str) else json.dumps(it["state"], ensure_ascii=False)
                jv |= ngrams(st + " \n " + it["question"]["instructions"])
    hit = sum(1 for e in tr if e["bucket"] in ("capability", "ood", "technical", "hard_negative") and e["track"] != "software_repair"
              and ngrams(e["context"] + " \n " + e["instructions"]) & jv)
    leak["jevbench_8gram_overlap_examples"] = hit
    n_rep = man["files"].get("train_replay", {}).get("n", 0)
    b = Counter(e["bucket"] for e in tr)
    tot = len(tr) + n_rep
    man["mixture"] = {"total_examples": tot, "replay": n_rep / tot, **{k: b[k] / tot for k in ("technical", "hard_negative", "capability", "ood")}}
    cap = [e for e in tr if "capability" in e["tags"]]
    man["capability_shares"] = {k: v / len(cap) for k, v in Counter(e["track"] for e in cap).items()}
    man["tetris_share"] = sum(e["slice"] == "tetris" for e in tr) / tot
    man["leakage"] = leak
    man["source_hashes"] = {"swe_smith_allowlist": sha(D / "swe_smith_allowlist.json"), "swe_smith_denylist": sha(D / "swe_smith_denylist.json"),
                            "tables.py": sha(ROOT / "src/krun_one/v11/tables.py"), "tech.py": sha(ROOT / "src/krun_one/v11/tech.py"),
                            "sims.py": sha(ROOT / "src/krun_one/v11/sims.py"), "envs.py": sha(ROOT / "src/krun_one/v11/envs.py"),
                            "swe.py": sha(ROOT / "src/krun_one/v11/swe.py"), "capability.py": sha(ROOT / "src/krun_one/v11/capability.py"),
                            "replay_source data/kd1_r2/train.jsonl": sha(ROOT / "data/kd1_r2/train.jsonl"),
                            "swe_smith_dataset": "SWE-bench/SWE-smith @ ea6d7173 (allowlisted repositories only)"}
    print(json.dumps(man["mixture"]), json.dumps(leak), json.dumps(man["capability_shares"]))
    return man


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["relabel", "bench", "train", "replay", "manifest"])
    ap.add_argument("--replay-n", type=int, default=0)
    args = ap.parse_args()
    ser = harness.serializer(ROOT / "artifacts/krun-one-v1-text-final")
    R3.mkdir(parents=True, exist_ok=True)
    mp = R3 / "manifest.json"
    man = json.loads(mp.read_text()) if mp.exists() else {"seeds": SEEDS, "files": {}, "supersedes": "data/v11/corpus (R1 / R2; swe_involved unlabeled)"}
    if args.cmd == "relabel":
        man["files"].update({f"techbench_{k}_v1_labeled": v for k, v in cmd_relabel(ser).items()})
    elif args.cmd == "bench":
        man["files"].update({f"techbench_{k}": v for k, v in cmd_bench(ser).items()})
    elif args.cmd == "train":
        f, items = cmd_train(ser)
        man["files"]["train_tech"] = f
        print(json.dumps({k: f[k] for k in ("examples", "problems", "dropped", "by_bucket", "by_track", "families", "label_counts")}))
    elif args.cmd == "replay":
        n = args.replay_n or int(round(man["files"]["train_tech"]["examples"] * 0.42 / 0.58))
        man["files"]["train_replay"] = B.cmd_replay(n, R3 / "train_replay.jsonl", SEEDS["replay"])
        print(json.dumps({k: v for k, v in man["files"]["train_replay"].items() if k in ("n", "by_type", "answerable", "k>=64", "k>=128")}))
    else:
        man = cmd_manifest(man)
    mp.write_text(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
