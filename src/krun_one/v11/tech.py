"""Technology-selection decisions from the rule tables (tables.py), with independent variation of the two things a
model could otherwise memorize: the requirement set AND the candidate subset.

R0 finding this fixes: 44 fixed (requirement set -> answer) pairs were memorized (100% seen, 49-71% unseen). Here the
same requirement set gives different answers depending on which options are offered, constraint flips change one
requirement with everything else fixed, and whole requirement sets / technologies are reserved for evaluation.

Splits (deterministic, by sha256 of "table:req1+req2"):  bench 25% ("unseen combination"), val 15%, train 60%.
``bench_only`` options never appear in train / val problems ("unseen technology").
"""

from __future__ import annotations

import hashlib
import itertools
import random
from typing import Any

from .envs import _finish
from .tables import INTROS, OVERHEAD, TABLES

TEMPLATES = ["{intro} They need a {noun}. Requirements: {reqs}.",
             "{intro} The {noun} they pick has to provide: {reqs}.",
             "{intro} They are choosing a {noun} and what matters is {reqs}."]


def req_sets(table: str) -> list[tuple[str, ...]]:
    t = TABLES[table]
    return [r for n in range(1, t["max_req"] + 1) for r in itertools.combinations(t["props"], n)]


def split_of(table: str, req: tuple[str, ...]) -> str:
    h = int(hashlib.sha256((table + ":" + "+".join(sorted(req))).encode()).hexdigest(), 16) % 20
    return "bench" if h < 5 else "val" if h < 8 else "train"


def usable(table: str, req: tuple[str, ...], allow_bench_only: bool) -> list[str]:
    """Options with no P on any stated requirement (and, for training, not reserved for the benchmark)."""
    return [o for o, s in TABLES[table]["options"].items()
            if all(s["props"][r] != "P" for r in req) and (allow_bench_only or not s["bench_only"])]


def outcome(table: str, req: tuple[str, ...], subset: list[str]) -> dict[str, Any]:
    opts = TABLES[table]["options"]
    sat = [o for o in subset if all(opts[o]["props"][r] == "Y" for r in req)]
    if not sat:
        return {"gold": None, "sat": [], "tie": False}
    low = min(opts[o]["overhead"] for o in sat)
    best = [o for o in sat if opts[o]["overhead"] == low]
    return {"gold": best[0] if len(best) == 1 else None, "sat": sat, "tie": len(best) != 1}


def feat(table: str, opt: str) -> str:
    s = TABLES[table]["options"][opt]
    tags = [TABLES[table]["props"][p][1] for p, v in s["props"].items() if v == "Y"]
    return "%s; overhead %s" % (", ".join(tags) if tags else "none of the listed capabilities", OVERHEAD[s["overhead"]])


def _fillers(rng: random.Random, table: str, n: int, allow_bench_only: bool) -> list[tuple[str, str]]:
    pool = [(ft, o) for ft in TABLES[table]["fillers"] for o, s in TABLES[ft]["options"].items()
            if (allow_bench_only or not s["bench_only"]) and o not in TABLES[table]["options"]]
    rng.shuffle(pool)
    seen, out = set(), []
    for ft, o in pool:
        if o not in seen:
            seen.add(o)
            out.append((ft, o))
    return out[:n]


def problem(rng: random.Random, table: str, req: tuple[str, ...], subset: list[str], k: int | None = None,
            allow_bench_only: bool = False, order: list[tuple[str, str]] | None = None, intro: str | None = None,
            variants: dict[str, int] | None = None, template: int | None = None, req_order: list[str] | None = None
            ) -> dict | None:
    """One problem for (table, requirement set, candidate subset). ``k`` > len(subset) adds options of unrelated tables."""
    t = TABLES[table]
    res = outcome(table, req, subset)
    if res["tie"]:
        return None
    cands = order or ([(table, o) for o in subset] + _fillers(rng, table, max(0, (k or 0) - len(subset)), allow_bench_only))
    if order is None:
        rng.shuffle(cands)
    keys = []
    for tb, o in cands:
        if tb != table:
            keys.append((0, 0, 0))
            continue
        s = t["options"][o]
        met = sum(s["props"][r] == "Y" for r in req)
        keys.append((1, 0, -s["overhead"]) if met == len(req) else (0, met, 0))
    if res["gold"] is None:
        keys = [None] * len(cands)
    variants = variants or {r: rng.randrange(2) for r in req}
    ro = req_order or rng.sample(list(req), len(req))
    reqs = [t["props"][r][0][variants[r]] for r in ro]
    template = rng.randrange(len(TEMPLATES)) if template is None else template
    intro = intro or rng.choice(INTROS)
    ctx = TEMPLATES[template].format(intro=intro, noun=t["noun"], reqs="; ".join(reqs))
    ins = (f"Choose the {t['noun']} that satisfies every requirement. If several do, choose the one with the lowest "
           "overhead (the simplest to run or implement).")
    p = _finish(table, t["track"], ins, ctx, [o for _, o in cands], [feat(tb, o) for tb, o in cands], keys,
                "hard" if len(res["sat"]) > 1 or len(req) >= 2 else "easy", "rule",
                {"table": table, "requirements": list(req), "subset": sorted(subset), "n_satisfiers": len(res["sat"]),
                 "n_same_table": len(subset), "combo_split": split_of(table, req),
                 "has_bench_only": any(TABLES[tb]["options"][o]["bench_only"] for tb, o in cands)})
    if p is None:
        return None
    for c, (tb, o) in zip(p["candidates"], cands):
        s = TABLES[tb]["options"][o]
        c["cons"] = {"met": float(sum(s["props"].get(r) == "Y" for r in req)) if tb == table else 0.0,
                     "overhead": float(s["overhead"])}
    p["_cands"] = cands
    p["_fmt"] = {"intro": intro, "template": template, "variants": variants, "req_order": ro}
    return p


def sample(rng: random.Random, table: str, req: tuple[str, ...], allow_bench_only: bool = False, k: int | None = None,
           want: str = "answerable", hard_negative: bool = False) -> dict | None:
    """Random candidate subset for a requirement set. want: answerable | unanswerable."""
    pool = usable(table, req, allow_bench_only)
    if len(pool) < 3:
        return None
    for _ in range(20):
        sub = rng.sample(pool, rng.randint(3, len(pool)))
        res = outcome(table, req, sub)
        if res["tie"] or (want == "answerable") != (res["gold"] is not None):
            continue
        if hard_negative and len(res["sat"]) < 2 and len(req) < 2:
            continue
        kk = None if hard_negative else k
        return problem(rng, table, req, sub, kk, allow_bench_only)
    return None


def flip_pair(rng: random.Random, table: str, req: tuple[str, ...], allow_bench_only: bool = False) -> tuple[dict, dict] | None:
    """(context A -> X, context A + ONE added requirement -> Y): same candidates, order, intro and wording.
    Both requirement sets must belong to the SAME split, so a flip never carries an evaluation set into training."""
    t = TABLES[table]
    extras = [p for p in t["props"] if p not in req]
    rng.shuffle(extras)
    for extra in extras:
        req2 = tuple(p for p in t["props"] if p in req or p == extra)
        if split_of(table, req2) != split_of(table, req):
            continue
        pool = usable(table, req2, allow_bench_only)
        if len(pool) < 3:
            continue
        for _ in range(10):
            sub = rng.sample(pool, rng.randint(3, len(pool)))
            a, b = outcome(table, req, sub), outcome(table, req2, sub)
            if a["tie"] or b["tie"] or not a["gold"] or not b["gold"] or a["gold"] == b["gold"]:
                continue
            base = problem(rng, table, req, sub, None, allow_bench_only)
            if base is None:
                continue
            f = base["_fmt"]
            flip = problem(rng, table, req2, sub, None, allow_bench_only, order=base["_cands"], intro=f["intro"],
                           variants={**f["variants"], extra: rng.randrange(2)}, template=f["template"],
                           req_order=f["req_order"] + [extra])
            if flip is not None:
                return base, flip
    return None


# ================================================================================================ R3 additions
# What R1 / R2 showed: requirement sets seen in training reach 100%, unseen ones ~60% (and ~75% even when every option's
# capabilities are written next to it). The model learned "requirement set -> preferred options", not the rule. The
# generators below attack that from three sides, all on TRAIN-split requirement sets and non-benchmark options only:
#   probes     the single facts the rule composes (does X provide P? which of these provide P? which is lightest?)
#   families   one fixed candidate list and wording, several requirement sets, different winners
#   catalog    invented products whose capabilities are assigned at random and stated in the context: nothing can be
#              memorized, only the rule (every requirement met, then lowest overhead) gives the answer
CODENAMES = ["atlas", "birch", "cedar", "delta", "ember", "flint", "garnet", "harbor", "iris", "jade", "kestrel", "lumen", "maple", "nova",
             "onyx", "pine", "quartz", "raven", "sable", "tundra", "umber", "vale", "willow", "zephyr"]


def req_sets_ext(table: str, extra: int = 0) -> list[tuple[str, ...]]:
    """Requirement sets up to ``max_req + extra`` properties (the benchmark only uses up to ``max_req``)."""
    t = TABLES[table]
    return [r for n in range(1, t["max_req"] + extra + 1) for r in itertools.combinations(t["props"], n)]


def _noul_problem(table: str, proposition: str, ctx: str, truth: bool, meta: dict, kind: str) -> dict:
    return {"slice": table, "track": TABLES[table]["track"], "k": 2, "instructions": proposition, "context": ctx,
            "candidates": [{"name": n, "feat": n, "value": float((n == "true") == truth), "cons": {}} for n in ("false", "true")],
            "gold": int(truth), "answerable": True, "difficulty": "easy", "flip": None, "qtype": "noul",
            "evidence": {"type": "rule", "verified": True}, "state": {}, "meta": {"table": table, "probe": kind, **meta}}


def probe_fact(rng: random.Random, table: str) -> list[dict] | None:
    """A family of two true / false propositions about ONE property: an option that has it and one that does not."""
    t = TABLES[table]
    prop = rng.choice(list(t["props"]))
    yes = [o for o, s in t["options"].items() if s["props"][prop] == "Y" and not s["bench_only"]]
    no = [o for o, s in t["options"].items() if s["props"][prop] == "N" and not s["bench_only"]]
    if not yes or not no:
        return None
    phrase = t["props"][prop][0][rng.randrange(2)]
    intro = rng.choice(INTROS)
    pair = [(rng.choice(yes), True), (rng.choice(no), False)]
    rng.shuffle(pair)
    out = []
    for i, (o, truth) in enumerate(pair):
        p = _noul_problem(table, f"As a {t['noun']}, {o} provides {phrase}.", intro, truth, {"property": prop, "option": o}, "fact")
        p["family"] = {"variant": i, "changed": "base" if i == 0 else "the option"}
        out.append(p)
    return out


def probe_which(rng: random.Random, table: str) -> dict | None:
    """multi: every option of the list that has one property."""
    t = TABLES[table]
    prop = rng.choice(list(t["props"]))
    pool = [o for o, s in t["options"].items() if s["props"][prop] in "YN" and not s["bench_only"]]
    if len(pool) < 3:
        return None
    sub = rng.sample(pool, rng.randint(3, min(8, len(pool))))
    gold = [i for i, o in enumerate(sub) if t["options"][o]["props"][prop] == "Y"]
    if not gold or len(gold) == len(sub):
        return None
    phrase = t["props"][prop][0][rng.randrange(2)]
    return {"slice": table, "track": t["track"], "k": len(sub), "instructions": f"Select every {t['noun']} that provides {phrase}.",
            "context": rng.choice(INTROS),
            "candidates": [{"name": o, "feat": feat(table, o), "value": float(i in gold), "cons": {"met": float(i in gold)}} for i, o in enumerate(sub)],
            "gold": None, "gold_multi": gold, "answerable": True, "difficulty": "easy", "flip": None, "qtype": "multi",
            "evidence": {"type": "rule", "verified": True}, "state": {}, "meta": {"table": table, "probe": "which", "property": prop}}


def probe_overhead(rng: random.Random, table: str) -> dict | None:
    """choice: the option with the lowest overhead of a list (unique minimum)."""
    t = TABLES[table]
    pool = [o for o, s in t["options"].items() if not s["bench_only"]]
    sub = rng.sample(pool, rng.randint(3, min(6, len(pool))))
    ov = [t["options"][o]["overhead"] for o in sub]
    if ov.count(min(ov)) != 1:
        return None
    p = _finish(table, t["track"], f"Choose the {t['noun']} with the lowest overhead (the simplest to run or implement).", rng.choice(INTROS),
                sub, ["overhead " + OVERHEAD[x] for x in ov], [(-x,) for x in ov], "easy", "rule", {"table": table, "probe": "overhead"})
    if p is None:
        return None
    for c, x in zip(p["candidates"], ov):
        c["cons"] = {"overhead": float(x)}
    p["state"] = {}
    return p


def family(rng: random.Random, table: str, split: str, n_max: int = 4, extra_req: int = 0, allow_bench_only: bool = False) -> list[dict] | None:
    """One candidate list, order, intro and wording; 2..n_max requirement sets of ``split`` with DIFFERENT winners."""
    t = TABLES[table]
    pool = [o for o, s in t["options"].items() if allow_bench_only or not s["bench_only"]]
    if len(pool) < 3:
        return None
    sub = rng.sample(pool, rng.randint(3, min(7, len(pool))))
    sets = [r for r in req_sets_ext(table, extra_req) if split_of(table, r) == split
            and all(t["options"][o]["props"][p] != "P" for o in sub for p in r)]
    rng.shuffle(sets)
    by_gold: dict[str, tuple] = {}
    for r in sets:
        res = outcome(table, r, sub)
        if res["gold"] and not res["tie"] and res["gold"] not in by_gold:
            by_gold[res["gold"]] = r
        if len(by_gold) >= n_max:
            break
    if len(by_gold) < 2:
        return None
    order = [(table, o) for o in sub]
    rng.shuffle(order)
    intro, template = rng.choice(INTROS), rng.randrange(len(TEMPLATES))
    variants = {p: rng.randrange(2) for p in t["props"]}
    out = []
    for i, r in enumerate(by_gold.values()):
        p = problem(rng, table, r, sub, None, allow_bench_only, order=order, intro=intro, variants={q: variants[q] for q in r}, template=template,
                    req_order=list(r))
        if p is None:
            return None
        p["family"] = {"variant": i, "changed": "base" if i == 0 else "the requirement set"}
        out.append(p)
    return out


def catalog(rng: random.Random, table: str, split: str, n_max: int = 3, k: int | None = None, want: str = "answerable") -> list[dict] | None:
    """Invented products with RANDOM capabilities stated in the context; a family = the same catalog under several
    requirement sets of ``split``. Requirement wording is the table's own; capability tags are the table's short tags."""
    t = TABLES[table]
    props = list(t["props"])
    k = k or rng.choice([3, 4, 4, 5, 6])
    names = rng.sample(CODENAMES, k)
    shown = rng.sample(props, min(len(props), rng.choice([4, 5, 6])))
    has = {n: {p for p in shown if rng.random() < 0.5} for n in names}
    ov = {n: rng.randrange(3) for n in names}
    sets = [r for r in req_sets(table) if split_of(table, r) == split and all(p in shown for p in r)]
    rng.shuffle(sets)

    def win(r):
        sat = [n for n in names if all(p in has[n] for p in r)]
        if not sat:
            return None, sat
        low = min(ov[n] for n in sat)
        best = [n for n in sat if ov[n] == low]
        return (best[0] if len(best) == 1 else "tie"), sat

    picked: dict = {}
    for r in sets:
        g, _ = win(r)
        if want == "unanswerable":
            if g is None:
                picked[None] = r
                break
        elif g not in (None, "tie") and g not in picked:
            picked[g] = r
            if len(picked) >= n_max:
                break
    if not picked:
        return None
    intro, template = rng.choice(INTROS), rng.randrange(len(TEMPLATES))
    variants = {p: rng.randrange(2) for p in props}
    cat_txt = " ".join("%s: %s; overhead %s." % (n, ", ".join(t["props"][p][1] for p in shown if p in has[n]) or "none of these capabilities",
                                                 OVERHEAD[ov[n]]) for n in names)
    out = []
    for i, r in enumerate(picked.values()):
        reqs = [t["props"][p][0][variants[p]] for p in r]
        ctx = TEMPLATES[template].format(intro=intro, noun=t["noun"], reqs="; ".join(reqs)) + " The products on offer: " + cat_txt
        keys = []
        for n in names:
            met = sum(p in has[n] for p in r)
            keys.append((1, 0, -ov[n]) if met == len(r) else (0, met, 0))
        g, sat = win(r)
        if g is None:
            keys = [None] * k
        ins = (f"Choose the {t['noun']} that satisfies every requirement. If several do, choose the one with the lowest overhead "
               "(the simplest to run or implement).")
        feats = ["%s; overhead %s" % (", ".join(t["props"][p][1] for p in shown if p in has[n]) or "none of these capabilities", OVERHEAD[ov[n]])
                 for n in names]
        p = _finish("catalog", t["track"], ins, ctx, names, feats, keys, "hard" if len(sat) > 1 or len(r) >= 2 else "easy", "rule",
                    {"table": table, "requirements": list(r), "combo_split": split, "catalog": True})
        if p is None:
            return None
        for c, n in zip(p["candidates"], names):
            c["cons"] = {"met": float(sum(q in has[n] for q in r)), "overhead": float(ov[n])}
        p["state"] = {}
        p["family"] = {"variant": i, "changed": "base" if i == 0 else "the requirement set"}
        out.append(p)
    return out
