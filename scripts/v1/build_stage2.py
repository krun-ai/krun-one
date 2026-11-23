"""Krun One v1 Stage 2 mixture (tool-focused continuation of v1-tool). Protocol: configs/v1/stage2_protocol.yaml.

    uv run python scripts/v1/build_stage2.py [--n 200000]
    -> data/kd1_s2/{train.jsonl, build_manifest.json}

Every decision is either resampled from the R2 training file (data/kd1_r2/train.jsonl, LEAKAGE = 0) or DERIVED from one
of its decisions by keeping a subset of its candidates (same context, same gold, candidates re-ordered):
  * small-k hard (k = 2 / 4): gold + the negatives closest to the gold (name + description char-3-gram Jaccard) or,
    for half of them, closest to the query (adversarial confusion);
  * small-k none-applicable (k = 2 / 4): from unanswerable tool decisions, the candidates closest to the query.
A subset of an answerable set with its gold stays answerable with the same gold; a subset of an unanswerable set stays
unanswerable. No new text, tool or label is introduced.

Explicit quotas (fractions of --n): 70% tool block / 30% replay, per curriculum phase (metadata.stage):
  S2P1 [0, .40]   tool core + replay
  S2P2 [.40, .75] hard / near-duplicate / description-dependent / cross-language / unseen-schema sources
  S2P3 [.75, 1]   small-k hard, tool OOD, none-applicable, adversarial confusion
  S2R  [0, 1]     replay (uniform, so every phase keeps the 70 / 30 ratio)
xLAM is supplementary (low weight). A shortfall in a (phase, bucket) is topped up from ToolACE of the same kind.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from krun_one.v1 import leakage, schema
from krun_one.v1.corpus import tools_v2 as T

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260929
SRC = ROOT / "data/kd1_r2/train.jsonl"
TOOL_SOURCES = {"toolace", "apibench", "toolmind", "hermes", "johin", "when2call", "xlam", "xlam_irrelevance",
                "intent_as_function", "multiwoz", "toolace_pool"}

# (phase, bucket) -> share of --n. Buckets: route_core/<src>, route_hard/<src>, action/<src>, multi/<src>, noul/<src>,
# ood/<src|any>, der_small/<src>, der_small4/<src>, der_ood/any, adv_k2/any, typo/any, replay/<kind>
TOOL_P1 = {"route_core/toolace": 14.0, "route_core/apibench": 5.0, "route_core/toolmind": 6.0, "route_core/hermes": .6,
           "route_core/johin": .8, "route_core/when2call": 1.5, "route_core/xlam": 3.0,
           "route_core/intent_as_function": 5.0, "route_core/multiwoz": 3.0,
           "action/when2call": 4.0, "action/xlam": 1.0,
           "multi/toolace": 1.5, "multi/xlam": 1.0, "multi/toolmind": 1.0, "multi/multiwoz": .1,
           "noul/when2call": 3.0, "noul/xlam": 1.0, "noul/toolace": 1.5, "noul/multiwoz": 1.0,
           "ood/toolace": 2.0, "ood/any": 2.8}
TOOL_P2 = {"route_hard/toolace": 15.0, "route_hard/apibench": 6.0, "route_hard/toolmind": 7.0, "route_hard/hermes": .8,
           "route_hard/johin": .8, "route_hard/when2call": 1.5, "route_hard/xlam": 3.5,
           "route_hard/intent_as_function": 6.0, "route_hard/multiwoz": 2.0,
           "der_small4/any_xl": 6.0, "ood/any": 2.9}
TOOL_P3 = {"der_small/toolace": 8.0, "der_small/apibench": 2.5, "der_small/toolmind": 3.0, "der_small/hermes": .4,
           "der_small/johin": .3, "der_small/when2call": .8, "der_small/xlam": 1.5, "der_small/intent_as_function": 2.0,
           "ood_irr/when2call": 2.0, "ood_native/any": 3.0, "ood_removed/any": 4.0, "der_ood/any": 4.5,
           "adv_k2/any": 2.7, "typo/any": 2.0}
REPLAY = {"replay/choice_intent": 30.0, "replay/intent_ood": 10.0, "replay/noul": 8.0, "replay/score": 8.0,
          "replay/multi": 7.0}   # quotas are thousands of a 210k reference: tool 147 (70%) + replay 63 (30%)

TOK = re.compile(r"[a-z0-9]+")


def grams(s: str) -> set:
    s = re.sub(r"\s+", " ", s.lower())
    return {s[i:i + 3] for i in range(max(1, len(s) - 2))}


def jac(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def name_tokens(c: dict) -> set:
    n = re.sub(r"([a-z])([A-Z])", r"\1 \2", c.get("label") or c["id"])
    return set(TOK.findall(n.lower()))


def src_of(r: dict) -> str | None:
    m = r["metadata"]
    return m.get("tool_source") or m.get("catalog")


def desc_dependent(r: dict) -> bool:
    gold = set(r["target"]["labels"])
    g = [c for c in r["candidates"] if c["id"] in gold]
    if not g:
        return False
    gt = name_tokens(g[0])
    if not gt:
        return False
    return any(len(gt & name_tokens(c)) / len(gt) >= 0.5 for c in r["candidates"] if c["id"] not in gold)


def classify(r: dict) -> list[str]:
    """Buckets this ORIGINAL decision can fill (a decision is used at most once, as original or as derived)."""
    fam, typ, m = r["task"].get("family"), r["task"]["type"], r["metadata"]
    cat, src, sl = m.get("category"), src_of(r), set(r["slices"])
    k = len(r["candidates"] or [])
    if fam in ("tool", "tool_action"):
        if src not in TOOL_SOURCES:
            src = "other"
        if cat == "ood":
            out = [f"ood/{src}", "ood/any"]
            if src == "when2call" or src == "xlam_irrelevance":
                out.append("ood_irr/when2call" if src == "when2call" else "ood_irr/xlam_irrelevance")
            if "native_irrelevance" in sl:
                out.append("ood_native/any")
            if "gold_removed" in sl:
                out.append("ood_removed/any")
            if typ == "choice" and k >= 4:
                out.append("der_ood/any")
            return out
        if cat == "multi_label":
            return [f"multi/{src}"]
        if fam == "tool_action":
            return [f"action/{src}"]
        if typ != "choice" or not r["answerable"]:
            return []
        hard = "near_duplicate" in sl or desc_dependent(r)
        out = [f"route_hard/{src}" if hard else f"route_core/{src}"]
        if hard and k <= 2:
            out.append("adv_k2/any")
        if "typo" in sl:
            out.append("typo/any")
        if k >= 8:
            out.append(f"der_small/{src}")
            if r.get("cross_language"):
                out.append("der_small4/any_xl")
        return out
    if typ == "noul" and (src in TOOL_SOURCES or (m.get("tool_source") in TOOL_SOURCES)):
        return [f"noul/{src}"]
    if typ == "noul":
        src2 = r["id"].split("/")
        if any(s in ("when2call", "xlam", "toolace", "multiwoz") for s in src2[:6]):
            s = next(s for s in src2[:6] if s in ("when2call", "xlam", "toolace", "multiwoz"))
            return [f"noul/{s}"]
        return ["replay/noul"]
    if typ == "score":
        return ["replay/score"]
    if typ == "multi":
        return ["replay/multi"] if cat != "ood" else ["replay/intent_ood"]
    if typ == "choice" and fam == "intent":
        return ["replay/intent_ood"] if cat == "ood" else ["replay/choice_intent"]
    return []


def derive_small(r: dict, k: int, rng: random.Random, mode: str) -> dict | None:
    gold = set(r["target"]["labels"])
    g = [c for c in r["candidates"] if c["id"] in gold]
    negs = [c for c in r["candidates"] if c["id"] not in gold]
    if len(g) != 1 or len(negs) < k - 1:
        return None
    gg = grams(f"{g[0].get('label') or g[0]['id']} {g[0].get('description') or ''}")
    q = grams(r["context"]["text"][-600:])
    ref = gg if mode == "gold" else q
    # never keep a negative that is (near) equivalent to the gold: identical name tokens or description Jaccard >= .9
    gt = name_tokens(g[0])
    ok = [c for c in negs if name_tokens(c) != gt and jac(gg, grams(f"{c.get('label') or c['id']} {c.get('description') or ''}")) < 0.9]
    if len(ok) < k - 1:
        return None
    ok.sort(key=lambda c: -jac(ref, grams(f"{c.get('label') or c['id']} {c.get('description') or ''}")))
    cands = g + ok[:k - 1]
    rng.shuffle(cands)
    return _mk(r, cands, f"s2_small_{mode}", k)


def derive_ood(r: dict, k: int, rng: random.Random) -> dict | None:
    if r["answerable"] or len(r["candidates"]) < k:
        return None
    q = grams(r["context"]["text"][-600:])
    c = sorted(r["candidates"], key=lambda c: -jac(q, grams(f"{c.get('label') or c['id']} {c.get('description') or ''}")))[:k]
    rng.shuffle(c)
    return _mk(r, c, "s2_none_small", k)


def sanitize(d: dict, P) -> dict | None:
    """Strict protected-tool rule on tool-family decisions (same as the R2 builder): a protected query or a protected gold
    drops the decision; protected negatives are removed from the candidate set (dropped if < 2 candidates remain)."""
    if P.query_hit(d["context"]["text"]):             # protected queries: every family
        return None
    if d["task"].get("family") != "tool" or not d.get("candidates"):
        return d
    gold = set(d["target"]["labels"])
    keep, removed = [], 0
    for c in d["candidates"]:
        hit = P.tool_hit(c.get("label") or c["id"], c.get("description")) if c.get("description") else None
        if hit and c["id"] in gold:
            return None
        if hit:
            removed += 1
            continue
        keep.append(c)
    if len(keep) < 2 or (d["answerable"] and not gold <= {c["id"] for c in keep}):
        return None
    if removed:
        d = json.loads(json.dumps(d))
        d["candidates"] = keep
        d["candidate_set"] = {**d["candidate_set"], "k": len(keep)}
        d["metadata"]["k"] = len(keep)
        d["metadata"]["s2_protected_removed"] = removed
    return d


def _mk(r: dict, cands: list, tag: str, k: int) -> dict:
    d = json.loads(json.dumps(r))
    d["id"] = r["id"].replace("kd1/", "kd1s2/", 1) + f"/{tag}{k}"
    d["candidates"] = cands
    d["candidate_set"] = {"strategy": f"{tag}<{r['candidate_set']['strategy']}", "k": k}
    d["metadata"]["k"] = k
    d["metadata"]["origin"] = "derived"
    d["metadata"]["s2_derived_from"] = r["id"]
    d["metadata"]["s2_small_k"] = True
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--out", default=str(ROOT / "data/kd1_s2"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    phases = {"S2P1": TOOL_P1, "S2P2": TOOL_P2, "S2P3": TOOL_P3, "S2R": REPLAY}
    assert abs(sum(TOOL_P1.values()) + sum(TOOL_P2.values()) + sum(TOOL_P3.values()) - 147.0) < 1e-6
    assert abs(sum(REPLAY.values()) - 63.0) < 1e-6
    scale = args.n / 210000.0
    want = {(ph, b): int(round(v * 1000 * scale)) for ph, qs in phases.items() for b, v in qs.items()}
    # ---- pass 1: reservoir per bucket (line offsets), each decision eligible for several buckets
    cap = {b: 0 for b in {b for _, b in want}}
    for (ph, b), n in want.items():
        cap[b] += n
    res: dict[str, list] = defaultdict(list)
    seen: Counter = Counter()
    avail: Counter = Counter()
    with open(SRC, "rb") as f:
        while True:
            off = f.tell()
            line = f.readline()
            if not line:
                break
            r = json.loads(line)
            for b in classify(r):
                avail[b] += 1
                if b not in cap:
                    continue
                c = max(cap.get(b, 0) * 3, 2000)   # oversample: derived may fail, buckets overlap
                seen[b] += 1
                if len(res[b]) < c:
                    res[b].append(off)
                else:
                    j = rng.randrange(seen[b])
                    if j < c:
                        res[b][j] = off
    # ---- fill in a fixed order: derived / specific buckets first, generic ones last
    used: set[int] = set()
    P = T.build_protected(ROOT)
    san = Counter()
    fh = open(SRC, "rb")

    def load(off):
        fh.seek(off)
        return json.loads(fh.readline())

    order = sorted(want, key=lambda pb: (pb[1].startswith(("replay/", "ood/any", "route_core/")), pb[0], pb[1]))
    records, fill = [], {}
    for ph, b in order:
        n = want[(ph, b)]
        pool = list(res.get(b, []))
        rng.shuffle(pool)
        got = 0
        for off in pool:
            if got >= n:
                break
            if off in used:
                continue
            r = load(off)
            r = sanitize(r, P)
            if r is None:
                san["dropped"] += 1
                used.add(off)
                continue
            if b.startswith("der_small4/"):
                d = derive_small(r, 4, rng, "gold")
            elif b.startswith("der_small/"):
                d = derive_small(r, 2 if rng.random() < .5 else 4, rng, "gold" if rng.random() < .5 else "query")
            elif b.startswith("der_ood/"):
                d = derive_ood(r, 2 if rng.random() < .5 else 4, rng)
            else:
                d = r
            if d is None:
                continue
            used.add(off)
            d["metadata"]["stage_v1"] = d["metadata"].get("stage")
            d["metadata"]["stage"] = ph
            d["metadata"]["s2_bucket"] = b
            records.append(d)
            got += 1
        fill[f"{ph}|{b}"] = {"want": n, "got": got, "available": avail.get(b, 0)}
    # shortfalls -> ToolACE of the same kind in the same phase (replay shortfalls are reported, never topped up)
    for key, v in list(fill.items()):
        ph, b = key.split("|")
        short = v["want"] - v["got"]
        if short <= 0 or b.startswith("replay/"):
            continue
        kind = b.split("/")[0]
        alt = {"route_core": "route_core/toolace", "route_hard": "route_hard/toolace", "der_small": "der_small/toolace",
               "multi": "multi/toolace", "noul": "noul/toolace", "action": "action/when2call"}.get(kind, "ood/toolace")
        got = 0
        pool = list(res.get(alt, []))
        rng.shuffle(pool)
        for off in pool:
            if got >= short:
                break
            if off in used:
                continue
            r = sanitize(load(off), P)
            if r is None:
                used.add(off)
                continue
            d = derive_small(r, 2 if rng.random() < .5 else 4, rng, "gold") if kind == "der_small" else r
            if d is None:
                continue
            used.add(off)
            d["metadata"]["stage_v1"] = d["metadata"].get("stage")
            d["metadata"]["stage"] = ph
            d["metadata"]["s2_bucket"] = alt + "(topup)"
            records.append(d)
            got += 1
        v["topup_from"], v["topup_got"] = alt, got
    # ---- validation + leakage (all Stage 2 records vs KrunBench v1 incl. final_protected manifest, val_ml, val_tools_v2)
    errs = Counter(e for r in records for e in schema.validate(r))
    if errs:
        raise SystemExit(f"schema errors: {dict(errs.most_common(8))}")
    ids = Counter(r["id"] for r in records)
    if max(ids.values()) > 1:
        raise SystemExit("duplicate ids")
    kb = [json.loads(l) for s in ("dev", "val_fit", "val_eval", "final_protected") for l in open(ROOT / f"data/krunbench_v1/{s}.jsonl")]
    val_ml = [json.loads(l) for l in open(ROOT / "data/kd1/val_ml.jsonl")]
    val2 = [json.loads(l) for l in open(ROOT / "data/kd1_r2/val_tools_v2.jsonl")]
    rep = leakage.check_leakage(records, kb + val_ml + val2, near_threshold=0.9)
    tool_viol = Counter()
    for r in records:
        if P.query_hit(r["context"]["text"]):
            tool_viol["query"] += 1
        for c in r["candidates"] or []:
            if c.get("description") and r["task"].get("family") in ("tool", "intent") and r["task"]["type"] in ("choice", "multi"):
                h = P.tool_hit(c.get("label") or c["id"], c["description"])
                if h and r["task"]["family"] == "tool":
                    tool_viol[h] += 1
    if not rep.ok or tool_viol:
        raise SystemExit(f"LEAKAGE != 0: {rep.summary()['violations']} {dict(tool_viol)}")
    rng.shuffle(records)
    with open(out / "train.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # ---- manifest: mixture, phases, xLAM share, unique tools
    tool = [r for r in records if not r["metadata"]["s2_bucket"].startswith("replay/")]
    xl = [r for r in tool if src_of(r) in ("xlam", "xlam_irrelevance")]
    names = lambda rs: {(c.get("label") or c["id"]).lower() for r in rs if r["task"].get("family") == "tool" for c in r["candidates"] or []}  # noqa: E731
    other_names = names([r for r in records if src_of(r) not in ("xlam", "xlam_irrelevance")])
    xl_names = names(xl)
    man = {"seed": SEED, "n": len(records), "source_file": str(SRC.relative_to(ROOT)),
           "source_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
           "tool_share": len(tool) / len(records), "replay_share": 1 - len(tool) / len(records),
           "by_phase": dict(Counter(r["metadata"]["stage"] for r in records)),
           "by_phase_tool": dict(Counter(r["metadata"]["stage"] for r in tool)),
           "by_source": dict(Counter(str(src_of(r)) for r in tool)),
           "by_k_tool_choice": dict(sorted(Counter(len(r["candidates"]) for r in tool if r["task"]["type"] == "choice").items())),
           "derived": dict(Counter(r["candidate_set"]["strategy"].split("<")[0] for r in records if r["metadata"]["origin"] == "derived")),
           "by_language": dict(Counter(r["context"]["language"] for r in records)),
           "cross_language": sum(bool(r.get("cross_language")) for r in records),
           "xlam": {"source": "Salesforce/xlam-function-calling-60k (argilla re-host, origin xLAM) + xlam-irrelevance-7.5k",
                    "revision": "as pinned in data/kd1_tools_v2/manifest.json", "license_note":
                    "CC-BY-4.0 dataset card + 'research purposes only' note; owner decision approved_by_owner "
                    "(docs/KRUN_XLAM_OFFICIAL_LICENSE_REAUDIT.md)",
                    "decisions": len(xl), "share_of_total": len(xl) / len(records), "share_of_tool_block": len(xl) / len(tool),
                    "unique_tool_names": len(xl_names), "unique_tool_names_only_in_xlam": len(xl_names - other_names),
                    "unseen_split_influence": "none: every xLAM tool matching the protected registry (incl. the ToolACE "
                                              "v0 val/holdout distractors and KrunBench v1 unseen tools) is excluded"},
           "protected_sanitize": {"dropped_decisions": san["dropped"],
                                  "decisions_with_protected_negatives_removed": sum(1 for r in records if r["metadata"].get("s2_protected_removed")),
                                  "protected_negatives_removed": sum(r["metadata"].get("s2_protected_removed", 0) for r in records)},
           "fill": fill, "leakage": rep.summary()["violations"], "protected_tool_violations": dict(tool_viol),
           "sha256": hashlib.sha256((out / "train.jsonl").read_bytes()).hexdigest()}
    (out / "build_manifest.json").write_text(json.dumps(man, indent=1, default=str))
    print(json.dumps({k: v for k, v in man.items() if k != "fill"}, indent=1, default=str))


if __name__ == "__main__":
    main()
