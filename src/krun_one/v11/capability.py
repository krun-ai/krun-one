"""Capability curriculum for Krun One v1.1 R3: numeric, temporal, multi-hop, long-policy, trade-off and mixed decisions.

Why it exists: the external check of R1 / R2 (JevBench public items, evaluation only) showed that Krun is weak on
decisions that need arithmetic comparison, time reasoning, short chains of lookups / implications, applying a policy
text and weighing stated priorities — also on short inputs, and on true / false propositions as much as on choices
(data/v11/JEVBENCH_CAPABILITY_GAPS.md, aggregate taxonomy only). Nothing here is derived from that benchmark: every
scenario, entity, number, rule and wording below was written for this module, and every label is computed by the code
next to it (arithmetic, calendar, graph walk, rule interpreter) — never by a model.

Every generator returns a counterfactual FAMILY: a list of 1..3 variants that share the base context, candidates and
candidate order and differ in the stated constraint(s) — and therefore in the correct decision::

    variant["family"] = {"variant": i, "changed": "<what differs from variant 0>"}      (the builder adds the id)

Question types: ``choice`` (also none-of-the-above when no option qualifies), ``noul`` (a proposition about the same
state) and ``score`` (an ordinal level). A choice problem has two renderings downstream: raw (names only; the facts are
in the context) and feat (name + its computed consequence).

``split``: "train" and "val" draw from the training scenarios; "bench" also draws from scenarios that never train
(tag ``unseen_scenario``).
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
import math
import random
from typing import Callable

from .envs import _finish

Problem = dict
NAMES = ["atlas", "birch", "cedar", "delta", "ember", "flint", "garnet", "harbor", "iris", "jade", "kestrel", "lumen", "maple",
         "nova", "onyx", "pine", "quartz", "raven", "sable", "tundra", "umber", "vale", "willow", "zephyr", "aspen", "basalt",
         "cobalt", "dune", "echo", "fjord", "grove", "heron", "indigo", "juniper", "koa", "larch", "mesa", "nimbus", "opal", "piper"]


# ================================================================================================ helpers
def num(x: float, dec: int = 0) -> str:
    return f"{x:,.{dec}f}"


def _scen(rng: random.Random, pool: list[dict], split: str) -> dict:
    ok = [s for s in pool if split == "bench" or not s.get("bench")]
    if split == "bench" and rng.random() < 0.5:
        ok = [s for s in pool if s.get("bench")] or ok
    return rng.choice(ok)


def _table(rng: random.Random, rows: list[tuple[str, list[tuple[str, str]]]], style: str | None = None) -> str:
    """The same facts as prose, a bullet list, key-value lines or compact JSON."""
    style = style or rng.choice(["prose", "bullets", "kv", "json"])
    if style == "prose":
        return " ".join("%s: %s." % (n, ", ".join(f"{a} {b}" for a, b in cols)) for n, cols in rows)
    if style == "bullets":
        return "\n".join("- %s | %s" % (n, " | ".join(f"{a} {b}" for a, b in cols)) for n, cols in rows)
    if style == "kv":
        return "\n".join("%s\n%s" % (n, "\n".join(f"  {a}: {b}" for a, b in cols)) for n, cols in rows)
    return json.dumps({n: {a: b for a, b in cols} for n, cols in rows}, separators=(",", ":"))


def _choice(slice_: str, track: str, ins: str, ctx: str, names: list[str], feats: list[str], keys: list, cons: list[dict],
            difficulty: str, evidence: str, meta: dict, state: dict | None = None) -> Problem | None:
    p = _finish(slice_, track, ins, ctx, names, feats, keys, difficulty, evidence, meta)
    if p is None:
        return None
    for c, x in zip(p["candidates"], cons):
        c["cons"] = x
    p["state"], p["qtype"] = state or {}, "choice"
    return p


def _noul(slice_: str, track: str, proposition: str, ctx: str, truth: bool, difficulty: str, evidence: str, meta: dict) -> Problem:
    return {"slice": slice_, "track": track, "k": 2, "instructions": proposition, "context": ctx,
            "candidates": [{"name": n, "feat": n, "value": float((n == "true") == truth), "cons": {}} for n in ("false", "true")],
            "gold": int(truth), "answerable": True, "difficulty": difficulty, "flip": None, "qtype": "noul",
            "evidence": {"type": evidence, "verified": True}, "state": {}, "meta": meta}


def _score(slice_: str, track: str, ins: str, ctx: str, levels: list[str], gold: int, difficulty: str, evidence: str, meta: dict) -> Problem:
    return {"slice": slice_, "track": track, "k": len(levels), "instructions": ins, "context": ctx,
            "candidates": [{"name": lv, "feat": lv, "value": float(len(levels) - abs(i - gold)), "cons": {}} for i, lv in enumerate(levels)],
            "gold": gold, "answerable": True, "difficulty": difficulty, "flip": None, "qtype": "score",
            "evidence": {"type": evidence, "verified": True}, "state": {}, "meta": meta}


def _family(variants: list[Problem | None], changed: list[str]) -> list[Problem] | None:
    """Keep a family only when every variant exists and consecutive variants have DIFFERENT correct decisions."""
    if any(v is None for v in variants) or (len(variants) > 1 and not all(v["answerable"] for v in variants)):
        return None      # "none qualifies" problems are generated on their own (want="unanswerable"), never as a variant
    gold = [(v["answerable"], v["candidates"][v["gold"]]["name"] if v["answerable"] else None) for v in variants]
    if len(variants) > 1 and len(set(gold)) != len(gold):
        return None
    for i, (v, ch) in enumerate(zip(variants, ["base"] + changed)):
        v["family"] = {"variant": i, "changed": ch}
    return variants


# ================================================================================================ numeric: limits
# metric: (key, label, lo, hi, step, better, unit_fmt, alt)   alt = (factor, unit_fmt) to state a limit in another unit
def _u(fmt: str, dec: int = 0) -> Callable[[float], str]:
    return lambda v: fmt.format(num(v, dec))


NUM_SCEN = [
    dict(id="gateway", intro=["A platform team is comparing api gateway setups.", "Several vendors pitched an api gateway and the team benchmarked each one."],
         metrics=[("lat", "p95 latency", 40, 400, 5, "low", _u("{} ms"), (1000, _u("{} s", 2))), ("cost", "monthly cost", 400, 6000, 50, "low", _u("${}"), None),
                  ("err", "error rate", 1, 60, 1, "low", lambda v: f"{v / 10:.1f}%", None), ("rps", "peak throughput", 500, 9000, 100, "high", _u("{} requests per second"), None)]),
    dict(id="gpu", intro=["An ml team is picking an instance type for inference.", "A research group needs a gpu server for a new model."],
         metrics=[("mem", "gpu memory", 8, 96, 4, "high", _u("{} GB"), None), ("price", "hourly price", 30, 600, 5, "low", lambda v: f"${v / 100:.2f}", None),
                  ("tps", "throughput", 20, 400, 10, "high", _u("{} tokens per second"), None), ("boot", "cold start", 10, 300, 5, "low", _u("{} s"), None)]),
    dict(id="dbplan", intro=["A startup is choosing a managed database plan.", "The data team reviews hosted database tiers."],
         metrics=[("stor", "storage", 50, 2000, 50, "high", _u("{} GB"), None), ("iops", "provisioned iops", 1000, 40000, 1000, "high", _u("{}"), None),
                  ("cost", "monthly price", 90, 4000, 10, "low", _u("${}"), None), ("fail", "failover time", 10, 240, 5, "low", _u("{} s"), (60, _u("{} min", 1)))]),
    dict(id="carrier", intro=["An online shop compares shipping carriers for a new route.", "The logistics lead is choosing a carrier for fragile goods."],
         metrics=[("days", "delivery time", 1, 14, 1, "low", _u("{} days"), None), ("fee", "cost per parcel", 300, 2500, 10, "low", lambda v: f"${v / 100:.2f}", None),
                  ("dmg", "damage rate", 1, 50, 1, "low", lambda v: f"{v / 10:.1f}%", None), ("cap", "weekly capacity", 200, 5000, 100, "high", _u("{} parcels"), None)]),
    dict(id="laptop", intro=["The it desk is choosing a laptop model for field staff.", "A design studio is buying laptops for new hires."],
         metrics=[("batt", "battery life", 5, 22, 1, "high", _u("{} hours"), None), ("wt", "weight", 90, 260, 5, "low", lambda v: f"{v / 100:.2f} kg", None),
                  ("price", "price", 600, 3200, 50, "low", _u("${}"), None), ("ram", "memory", 8, 64, 8, "high", _u("{} GB"), None)]),
    dict(id="venue", intro=["A company is booking a venue for its yearly meeting.", "The events team shortlists venues for a product launch."],
         metrics=[("seats", "capacity", 80, 900, 10, "high", _u("{} seats"), None), ("rent", "day rate", 1500, 20000, 100, "low", _u("${}"), None),
                  ("dist", "distance from the office", 1, 60, 1, "low", _u("{} km"), None), ("rooms", "breakout rooms", 0, 12, 1, "high", _u("{}"), None)]),
    dict(id="supplier", intro=["A factory is selecting a supplier for a machined part.", "Procurement compares quotes for a packaging component."],
         metrics=[("unit", "unit price", 120, 990, 5, "low", lambda v: f"${v / 100:.2f}", None), ("lead", "lead time", 3, 60, 1, "low", _u("{} days"), None),
                  ("defect", "defect rate", 1, 40, 1, "low", lambda v: f"{v / 10:.1f}%", None), ("moq", "minimum order", 100, 5000, 100, "low", _u("{} units"), None)]),
    dict(id="cdn", bench=True, intro=["A media site is choosing a content delivery network.", "The web team trials several cdn contracts."],
         metrics=[("ttfb", "median time to first byte", 20, 300, 5, "low", _u("{} ms"), (1000, _u("{} s", 2))), ("cost", "cost per terabyte", 5, 90, 1, "low", _u("${}"), None),
                  ("pops", "points of presence", 20, 400, 10, "high", _u("{}"), None), ("hit", "cache hit ratio", 60, 99, 1, "high", _u("{}%"), None)]),
    dict(id="loan", bench=True, intro=["A small business compares loan offers.", "A bakery owner received several financing offers."],
         metrics=[("rate", "yearly interest", 30, 180, 1, "low", lambda v: f"{v / 10:.1f}%", None), ("pay", "monthly payment", 300, 4000, 20, "low", _u("${}"), None),
                  ("term", "term", 12, 96, 6, "low", _u("{} months"), None), ("fee", "setup fee", 0, 2000, 50, "low", _u("${}"), None)]),
    dict(id="contractor", bench=True, intro=["A studio is hiring a contractor for a three month project.", "The team lead compares freelance offers."],
         metrics=[("rate", "hourly rate", 30, 220, 5, "low", _u("${}"), None), ("avail", "availability", 10, 40, 2, "high", _u("{} hours per week"), None),
                  ("start", "earliest start", 1, 60, 1, "low", _u("in {} days"), None), ("exp", "experience", 1, 20, 1, "high", _u("{} years"), None)]),
]
_LIMIT = {("low", True): ["{m} must be at most {t}", "{m} may not exceed {t}"], ("low", False): ["{m} must stay under {t}", "{m} has to be below {t}"],
          ("high", True): ["{m} must be at least {t}", "{m} of {t} or more is required"], ("high", False): ["{m} must be above {t}", "{m} has to exceed {t}"]}


def _meets(v: float, th: float, better: str, incl: bool) -> bool:
    if better == "low":
        return v <= th if incl else v < th
    return v >= th if incl else v > th


def _threshold(rng: random.Random, vals: list[float], step: float, better: str) -> tuple[float, bool] | None:
    """A limit that splits the candidates; sometimes exactly on a candidate's value (then inclusiveness decides)."""
    s = sorted(set(vals))
    if len(s) < 2:
        return None
    i = rng.randrange(len(s) - 1)
    if rng.random() < 0.3:
        return s[i if better == "low" else i + 1], rng.random() < 0.5
    cut = [x for x in (s[i] + step * j for j in range(1, 400)) if x < s[i + 1]]
    return (rng.choice(cut[:3] + cut[-3:]) if cut else s[i if better == "low" else i + 1]), rng.random() < 0.5


def _limits_setup(rng: random.Random, split: str, k: int):
    sc = _scen(rng, NUM_SCEN, split)
    ms = rng.sample(sc["metrics"], rng.choice([2, 3, 3, 4]))
    names = rng.sample(NAMES, k)
    vals = {m[0]: [rng.randrange(m[2], m[3] + 1, m[4]) for _ in names] for m in ms}
    obj = rng.choice(ms)
    cons_m = rng.sample([m for m in ms if m is not obj] or ms, min(len(ms) - 1 or 1, rng.choice([1, 2, 2, 3])))
    lims = []
    for m in cons_m:
        t = _threshold(rng, vals[m[0]], m[4], m[5])
        if t is None:
            return None
        lims.append([m, t[0], t[1], rng.randrange(2), m[7] is not None and rng.random() < 0.35])
    return sc, ms, names, vals, obj, lims, rng.choice(sc["intro"]), rng.choice(["prose", "bullets", "kv", "json"])


def _limit_text(m, th, incl, phr, alt) -> str:
    t = m[7][1](th / m[7][0]) if alt else m[6](th)
    return _LIMIT[(m[5], incl)][phr].format(m=m[1], t=t)


def _limits_eval(ms, names, vals, obj, lims):
    viol = [[m[1] for m, th, incl, _, _ in lims if not _meets(vals[m[0]][i], th, m[5], incl)] for i in range(len(names))]
    sign = -1 if obj[5] == "low" else 1
    keys = [None if viol[i] else (sign * vals[obj[0]][i],) for i in range(len(names))]
    return viol, keys


def _limits_ctx(rng, sc, ms, names, vals, lims, intro, style) -> str:
    rows = [(n, [(m[1], m[6](vals[m[0]][i])) for m in ms]) for i, n in enumerate(names)]
    return "%s\n%s\nLimits: %s." % (intro, _table(rng, rows, style), "; ".join(_limit_text(*x) for x in lims))


def _limits_problem(rng, setup, lims, form: str, focus: int | None = None) -> Problem | None:
    sc, ms, names, vals, obj, _, intro, style = setup
    viol, keys = _limits_eval(ms, names, vals, obj, lims)
    ctx = _limits_ctx(random.Random(0), sc, ms, names, vals, lims, intro, style)
    near = any(abs(vals[m[0]][i] - th) <= 2 * m[4] for m, th, *_ in lims for i in range(len(names)))
    diff = "hard" if near or len(lims) >= 2 or any(x[4] for x in lims) else "easy"
    meta = {"scenario": sc["id"], "objective": obj[0], "limits": [[m[0], th, incl] for m, th, incl, *_ in lims], "unit_twist": any(x[4] for x in lims)}
    if form == "choice":
        ins = "Choose the option that meets every limit and has the %s %s." % ("lowest" if obj[5] == "low" else "highest", obj[1])
        feats = [("violates the limit on " + " and ".join(v)) if v else "meets every limit; %s %s" % (obj[1], obj[6](vals[obj[0]][i]))
                 for i, v in enumerate(viol)]
        cons = [{"feasible": float(not v), "violations": float(len(v)), "objective": float(vals[obj[0]][i])} for i, v in enumerate(viol)]
        return _choice("num_limits", "numeric", ins, ctx, names, feats, keys, cons, diff, "arithmetic", meta,
                       {"feasible_frac": sum(not v for v in viol) / len(names), "n_limits": float(len(lims))})
    if form == "noul":
        return _noul("num_limits", "numeric", f"{names[focus]} meets every stated limit.", ctx, not viol[focus], diff, "arithmetic", meta)
    levels = ["none", "one", "two", "three"][: len(lims) + 1]
    return _score("num_limits", "numeric", f"How many of the stated limits does {names[focus]} violate?", ctx, levels, len(viol[focus]), diff,
                  "arithmetic", meta)


def gen_num_limits(rng: random.Random, split: str = "train", form: str | None = None, k: int | None = None, want: str = "answerable"):
    form = form or rng.choice(["choice", "choice", "choice", "noul", "noul", "score"])
    setup = _limits_setup(rng, split, k or rng.choice([3, 4, 4, 5, 6, 8]))
    if setup is None:
        return None
    sc, ms, names, vals, obj, lims, *_ = setup
    viol, keys = _limits_eval(ms, names, vals, obj, lims)
    feas = [i for i, x in enumerate(keys) if x is not None]
    if want == "unanswerable":
        if feas or form != "choice":
            return None
        return _family([_limits_problem(rng, setup, lims, "choice")], [])
    if not feas:
        return None
    gold = max(feas, key=lambda i: keys[i])
    focus = gold if rng.random() < 0.5 else rng.randrange(len(names))
    base = _limits_problem(rng, setup, lims, form, focus)
    # counterfactual: move ONE limit so that the base winner (or the proposition's subject) no longer qualifies
    j = rng.randrange(len(lims))
    m, th, incl, phr, alt = lims[j]
    v = vals[m[0]][focus if form != "choice" else gold]
    new_th = v - m[4] * rng.choice([0, 1, 2]) if m[5] == "low" else v + m[4] * rng.choice([0, 1, 2])
    new_incl = False if new_th == v else incl
    if not _meets(v, th, m[5], incl):          # the subject already violates this limit: relax it instead
        new_th, new_incl = v, True
    lims2 = [list(x) for x in lims]
    lims2[j] = [m, new_th, new_incl, phr, alt]
    var = _limits_problem(rng, setup, lims2, form, focus)
    fam = _family([base, var], [f"limit on {m[1]}"])
    return fam or (_family([base], []) if base is not None else None)


# ================================================================================================ numeric: budget
BUD_SCEN = [
    dict(id="office", intro="An office manager compares proposals to equip a new floor.", what="proposal",
         items=[("desks", 180, 520), ("chairs", 90, 380), ("monitors", 140, 460), ("docking stations", 80, 240), ("lamps", 25, 90)], value="desks"),
    dict(id="cloud", intro="An engineering lead compares cloud proposals for the next quarter.", what="proposal",
         items=[("api servers", 60, 310), ("database nodes", 220, 900), ("cache nodes", 70, 260), ("load balancers", 20, 60), ("build runners", 40, 180)], value="api servers"),
    dict(id="event", intro="A coordinator compares catering quotes for a company event.", what="quote",
         items=[("lunch boxes", 9, 24), ("coffee stations", 110, 280), ("dessert trays", 35, 90), ("waiters", 120, 260), ("water crates", 8, 20)], value="lunch boxes"),
    dict(id="lab", intro="A school compares bids to furnish a science lab.", what="bid",
         items=[("microscopes", 210, 780), ("workbenches", 300, 950), ("safety kits", 25, 80), ("sinks", 180, 540), ("stools", 30, 95)], value="microscopes"),
    dict(id="fleet", bench=True, intro="A delivery firm compares offers to extend its fleet.", what="offer",
         items=[("vans", 21000, 38000), ("cargo bikes", 2400, 5200), ("roof racks", 180, 420), ("trackers", 60, 150), ("chargers", 700, 1900)], value="vans"),
]


def gen_num_budget(rng: random.Random, split: str = "train", form: str | None = None, k: int | None = None, want: str = "answerable"):
    form = form or rng.choice(["choice", "choice", "noul"])
    sc = _scen(rng, BUD_SCEN, split)
    k = k or rng.choice([3, 4, 5])
    names = rng.sample(NAMES, k)
    items = rng.sample(sc["items"], rng.choice([2, 3, 3]))
    if sc["value"] not in [i[0] for i in items]:
        items[0] = next(i for i in sc["items"] if i[0] == sc["value"])
    lines, totals, value = [], [], []
    for _ in names:
        row = [(it[0], rng.randint(2, 24), rng.randrange(it[1], it[2] + 1, max(1, (it[2] - it[1]) // 40 or 1))) for it in items]
        lines.append(row)
        totals.append(sum(q * p for _, q, p in row))
        value.append(next(q for n, q, _ in row if n == sc["value"]))
    style = rng.choice(["prose", "bullets", "kv"])

    def build(budget: int) -> Problem | None:
        rows = [(n, [(it, f"{q} at ${num(p)} each") for it, q, p in row]) for n, row in zip(names, lines)]
        ctx = f"{sc['intro']}\n{_table(random.Random(0), rows, style)}\nThe budget is ${num(budget)}."
        ok = [t <= budget for t in totals]
        meta = {"scenario": sc["id"], "budget": budget, "totals": totals}
        diff = "hard" if any(abs(t - budget) <= 0.03 * budget for t in totals) else "easy"
        if form == "noul":
            return _noul("num_budget", "numeric", f"The {sc['what']} from {names[0]} fits within the budget.", ctx, ok[0], diff, "arithmetic", meta)
        ins = f"Choose the {sc['what']} that stays within the budget and delivers the most {sc['value']}; if tied, the cheaper one."
        keys = [(value[i], -totals[i]) if ok[i] else None for i in range(k)]
        feats = [f"total ${num(totals[i])}, " + (f"within budget; {value[i]} {sc['value']}" if ok[i] else f"over budget by ${num(totals[i] - budget)}")
                 for i in range(k)]
        cons = [{"feasible": float(ok[i]), "total": float(totals[i]), "value": float(value[i])} for i in range(k)]
        return _choice("num_budget", "numeric", ins, ctx, names, feats, keys, cons, diff, "arithmetic", meta,
                       {"feasible_frac": sum(ok) / k, "budget": float(budget)})

    s = sorted(totals)
    if want == "unanswerable":
        return _family([build(int(s[0] * rng.uniform(0.7, 0.97)))], []) if form == "choice" else None
    i = rng.randrange(1, k) if k > 1 else 0
    b1 = int(s[i] * rng.choice([1.0, 1.01, 1.04, 1.1])) if form == "choice" else int(totals[0] * rng.choice([1.0, 1.02, 1.1]))
    b2 = int(s[i - 1] * rng.choice([1.0, 1.02])) if form == "choice" else int(totals[0] * rng.choice([0.9, 0.98, 0.995])) - 1
    base = build(b1)
    fam = _family([base, build(b2)], ["budget"])
    return fam or (_family([base], []) if base is not None else None)


# ================================================================================================ numeric: capacity
CAP_SCEN = [
    dict(id="replicas", unit="replicas", one="replica", load="requests per second", intro="A service is being sized for its peak."),
    dict(id="agents", unit="support agents", one="agent", load="tickets per hour", intro="A support desk is staffing the evening shift."),
    dict(id="trucks", unit="trucks", one="truck", load="pallets per day", intro="A warehouse plans its outbound fleet."),
    dict(id="workers", unit="queue workers", one="worker", load="jobs per minute", intro="A batch system is sizing its worker pool."),
    dict(id="tills", bench=True, unit="checkout lanes", one="lane", load="customers per hour", intro="A supermarket plans how many lanes to open."),
]


def gen_num_capacity(rng: random.Random, split: str = "train", form: str | None = None, k: int | None = None, want: str = "answerable"):
    if want != "answerable":
        return None
    form = form or rng.choice(["choice", "noul", "score"])
    sc = _scen(rng, CAP_SCEN, split)
    per = rng.choice([40, 50, 60, 75, 80, 120, 150, 200, 250])
    util = rng.choice([50, 60, 70, 75, 80])
    style = rng.randrange(2)
    need_of = lambda load: math.ceil(load * 100 / (per * util))  # noqa: E731
    load_for = lambda need: int(need * per * util / 100 * rng.uniform(0.9, 1.0)) // 10 * 10  # noqa: E731
    load = load_for(rng.randint(3, 30))
    need0 = need_of(load)
    kk = k or rng.choice([4, 5, 6, 8])
    lo = max(1, need0 - rng.randint(1, kk - 2))
    ns = list(range(lo, lo + kk))
    rng.shuffle(ns)
    n_noul = max(1, need0 + rng.choice([-1, -1, 0, 0, 1]))
    n_score = max(1, need0 + rng.choice([-3, -2, -1, 0, 1, 2]))

    def build(load: int) -> Problem | None:
        need = need_of(load)
        ctx = ([f"{sc['intro']} Peak load is {num(load)} {sc['load']}. Each {sc['one']} handles {per} {sc['load']} at full load, and "
                f"utilization must stay at or below {util}% at peak.",
                f"{sc['intro']}\npeak load: {num(load)} {sc['load']}\ncapacity per {sc['one']}: {per} {sc['load']}\nmaximum utilization at peak: {util}%"][style])
        meta = {"scenario": sc["id"], "load": load, "per": per, "util": util, "need": need}
        if form == "noul":
            return _noul("num_capacity", "numeric", f"{n_noul} {sc['unit']} are enough to keep utilization within the limit.", ctx, n_noul >= need, "hard",
                         "arithmetic", {**meta, "n": n_noul})
        if form == "score":
            u = load * 100 / (per * n_score)
            if abs(u - util) < 1e-9 or abs(u - 100) < 1e-9:
                return None
            lv = 0 if u <= util else 1 if u <= 100 else 2
            return _score("num_capacity", "numeric", f"With {n_score} {sc['unit']}, how loaded is the system at peak?", ctx,
                          ["within the utilization limit", "over the limit but under full capacity", "over full capacity"], lv, "hard", "arithmetic",
                          {**meta, "n": n_score})
        if need not in ns:
            return None
        keys = [(-n,) if n >= need else None for n in ns]
        feats = ["%.0f%% utilization at peak" % (load * 100 / (per * n)) for n in ns]
        cons = [{"feasible": float(n >= need), "utilization": load * 100 / (per * n), "n": float(n)} for n in ns]
        return _choice("num_capacity", "numeric", f"Choose the smallest number of {sc['unit']} that keeps utilization within the limit.", ctx,
                       [f"{n} {sc['unit']}" for n in ns], feats, keys, cons, "hard", "arithmetic", meta, {"need": float(need)})

    base = build(load)
    need2 = rng.choice([n for n in ns if n != need0]) if form == "choice" else max(1, need0 + rng.choice([-4, -2, 2, 4]))
    load2 = load_for(need2)
    var = build(load2) if load2 != load and load2 > 0 else None
    fam = _family([base, var], ["peak load"])
    return fam or (_family([base], []) if base is not None else None)


# ================================================================================================ numeric: error budget
def gen_num_errbudget(rng: random.Random, split: str = "train", form: str | None = None, k: int | None = None, want: str = "answerable"):
    if want != "answerable":
        return None
    form = form or rng.choice(["choice", "noul", "score"])
    svc = rng.choice(["checkout api", "search service", "login service", "billing worker", "notification service", "upload service"]
                     + (["tracking api"] if split == "bench" else []))
    days = rng.choice([7, 28, 30])
    slo = rng.choice([990, 995, 999])
    allowed = days * 1440 * (1000 - slo) / 1000
    inc = [rng.randint(2, max(3, int(allowed * 0.5))) for _ in range(rng.randint(1, 4))]
    order = rng.sample(range(3), 3)

    def build(incidents: list[int]) -> Problem | None:
        used = sum(incidents)
        left = allowed - used
        frac = used / allowed
        ctx = (f"The {svc} has an availability objective of {slo / 10:.1f}% over a {days} day window. In the current window it was down "
               f"during {len(incidents)} incident{'s' if len(incidents) > 1 else ''} lasting {', '.join(str(x) for x in incidents)} minutes. "
               "Release policy: if the error budget is exhausted, freeze feature releases; if less than a quarter of the budget remains, "
               "ship only fixes; otherwise release normally.")
        meta = {"scenario": svc, "allowed_min": allowed, "used_min": used, "slo": slo, "days": days}
        diff = "hard" if abs(frac - 1) < 0.15 or abs(frac - 0.75) < 0.1 else "easy"
        act = 0 if left < 0 else 1 if left < 0.25 * allowed else 2
        if abs(left) < 1e-9 or abs(left - 0.25 * allowed) < 1e-9:
            return None
        if form == "noul":
            return _noul("num_errbudget", "numeric", "The error budget of the current window is exhausted.", ctx, left < 0, diff, "arithmetic", meta)
        if form == "score":
            lv = 0 if frac < 0.5 else 1 if frac <= 1 else 2
            if abs(frac - 0.5) < 1e-9:
                return None
            return _score("num_errbudget", "numeric", "How much of the error budget has been consumed?", ctx,
                          ["less than half", "half or more but not all", "more than all of it"], lv, diff, "arithmetic", meta)
        names = ["freeze feature releases", "ship only fixes", "release normally"]
        keys = [(1,) if i == act else (0,) for i in order]
        feats = [f"budget {allowed:.1f} min, used {used} min" for _ in order]
        return _choice("num_errbudget", "numeric", "Choose the action the release policy requires.", ctx, [names[i] for i in order], feats, keys,
                       [{"match": float(i == act)} for i in order], diff, "arithmetic", meta, {"used_frac": frac})

    base = build(inc)
    extra = inc + [max(2, int(allowed * rng.choice([0.3, 0.6, 0.9])))]
    fam = _family([base, build(extra)], ["one more incident"])
    return fam or (_family([base], []) if base is not None else None)


# ================================================================================================ temporal: ordering
ORD_SCEN = [
    dict(id="schema", intro="A team plans a database change.", steps=["add the new column", "backfill the data", "deploy the api change",
                                                                    "switch reads over", "remove the old column", "update the dashboards"]),
    dict(id="release", intro="A release manager plans this week's rollout.", steps=["freeze the branch", "run the load test", "publish the notes",
                                                                                 "deploy to canary", "deploy everywhere", "brief support"]),
    dict(id="office", intro="A company plans its move to a new office.", steps=["sign the lease", "install the network", "move the furniture",
                                                                              "move the servers", "hand back the old keys", "update the address"]),
    dict(id="event", intro="An organizer plans a conference.", steps=["book the venue", "confirm the speakers", "open registration",
                                                                     "print the badges", "send the schedule", "order catering"]),
    dict(id="pipeline", intro="A data engineer plans the nightly pipeline.", steps=["export the orders", "clean the records", "join the customers",
                                                                                  "compute the totals", "refresh the report", "archive the raw files"]),
    dict(id="incident", intro="An on-call engineer plans the response to an outage.", steps=["page the owner", "roll back the release", "restore the data",
                                                                                           "verify the checks", "reopen traffic", "write the summary"]),
    dict(id="kitchen", bench=True, intro="A contractor plans a kitchen renovation.", steps=["remove the cabinets", "fix the wiring", "paint the walls",
                                                                                          "lay the floor", "install the counters", "connect the sink"]),
    dict(id="onboard", bench=True, intro="A manager plans a new hire's first week.", steps=["issue the laptop", "create the accounts", "run the safety briefing",
                                                                                         "assign a mentor", "review the codebase", "ship a first fix"]),
]
_ORD_PHR = ["{a} must happen before {b}", "{b} can only start after {a}", "{a} comes before {b}"]


def _viol(seq: tuple[int, ...], cons: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pos = {s: i for i, s in enumerate(seq)}
    return [c for c in cons if pos[c[0]] > pos[c[1]]]


def gen_tmp_order(rng: random.Random, split: str = "train", form: str | None = None, k: int | None = None, want: str = "answerable"):
    form = form or rng.choice(["choice", "choice", "noul"])
    sc = _scen(rng, ORD_SCEN, split)
    n = rng.choice([4, 4, 5])
    steps = rng.sample(sc["steps"], n)
    perms = list(itertools.permutations(range(n)))
    gold = rng.choice(perms)
    pairs = [(gold[i], gold[j]) for i in range(n) for j in range(i + 1, n)]
    cons = rng.sample(pairs, rng.randint(n - 1, min(len(pairs), n + 2)))
    phr = [rng.randrange(len(_ORD_PHR)) for _ in cons] + [rng.randrange(len(_ORD_PHR))]
    one = [p for p in perms if len(_viol(p, cons)) == 1]
    more = [p for p in perms if len(_viol(p, cons)) >= 2]
    valid = [p for p in perms if not _viol(p, cons)]
    k = k or rng.choice([3, 4, 4, 5, 6])
    if not one or len(one) + len(more) < k - 1:
        return None
    alt = rng.choice(one)
    broken = _viol(alt, cons)[0]
    rest = [p for p in one + more if p != alt]
    cands = [gold, alt] + rng.sample(rest, k - 2)
    rng.shuffle(cands)
    if want == "unanswerable":
        cands = [p for p in cands if p != gold] + rng.sample([p for p in rest if p not in cands] or rest, 1)
        if any(p in valid for p in cands) or len(set(cands)) != len(cands):
            return None
    elif sum(p in valid for p in cands) != 1:
        return None
    text = lambda p: "; ".join(steps[i] for i in p)  # noqa: E731

    def build(cs: list[tuple[int, int]], focus: tuple[int, ...]) -> Problem | None:
        ctx = "%s Constraints: %s." % (sc["intro"], "; ".join(_ORD_PHR[ph].format(a=steps[a], b=steps[b]) for (a, b), ph in zip(cs, phr)))
        v = [_viol(p, cs) for p in cands]
        meta = {"scenario": sc["id"], "n_steps": n, "n_constraints": len(cs)}
        diff = "hard" if n >= 5 or len(cs) >= 5 else "easy"
        if form == "noul":
            return _noul("tmp_order", "temporal", f"This order satisfies every constraint: {text(focus)}.", ctx, not _viol(focus, cs), diff, "rule", meta)
        keys = [None if x else (1,) for x in v]
        feats = ["valid order" if not x else "breaks: %s before %s" % (steps[x[0][0]], steps[x[0][1]]) for x in v]
        return _choice("tmp_order", "temporal", "Choose the order that satisfies every constraint.", ctx, [text(p) for p in cands], feats, keys,
                       [{"violations": float(len(x))} for x in v], diff, "rule", meta, {"n_constraints": float(len(cs))})

    base = build(cons, gold)
    if want == "unanswerable":
        return _family([base], []) if form == "choice" and base is not None else None
    cons2 = [c if c != broken else (broken[1], broken[0]) for c in cons]
    if sum(not _viol(p, cons2) for p in cands) != 1:
        return _family([base], []) if base is not None else None
    fam = _family([base, build(cons2, gold)], ["one constraint reversed"])
    return fam or (_family([base], []) if base is not None else None)


# ================================================================================================ temporal: windows
def _hm(t: int) -> str:
    d, m = divmod(t, 1440)
    return "%02d:%02d%s" % (m // 60, m % 60, "" if d == 0 else " the next day" if d == 1 else f" {d} days later")


def _dur(m: int, style: int) -> str:
    return f"{m} minutes" if style == 0 or m < 60 else ("%d h %02d min" % divmod(m, 60))


WIN_SCEN = [
    dict(id="maint", intro="A maintenance window is open.", close="the window closes", noun="job",
         names=["index rebuild", "disk swap", "kernel patch", "cache warmup", "schema change", "log rotation", "backup check", "router update"]),
    dict(id="studio", intro="A recording studio is booked for the day.", close="the studio must be empty", noun="session",
         names=["drum session", "vocal session", "mixing pass", "string overdubs", "podcast taping", "piano tuning", "guitar tracking", "choir take"]),
    dict(id="dock", intro="A warehouse dock closes for the day.", close="the dock closes", noun="unloading",
         names=["truck red", "truck blue", "truck green", "truck grey", "truck white", "truck black", "truck amber", "truck teal"]),
    dict(id="clinic", bench=True, intro="A clinic schedules procedures before closing.", close="the clinic closes", noun="procedure",
         names=["scan a", "scan b", "minor surgery", "dental cleaning", "eye exam", "blood panel", "cast removal", "allergy test"]),
]


def gen_tmp_window(rng: random.Random, split: str = "train", form: str | None = None, k: int | None = None, want: str = "answerable"):
    form = form or rng.choice(["choice", "choice", "noul"])
    sc = _scen(rng, WIN_SCEN, split)
    k = k or rng.choice([3, 4, 5, 6])
    names = rng.sample(sc["names"], k)
    late = rng.random() < 0.35
    t0 = rng.randrange(20 * 60, 23 * 60, 5) if late else rng.randrange(7 * 60, 15 * 60, 5)
    starts = [t0 + rng.randrange(0, 240, 5) for _ in names]
    durs = [rng.randrange(20, 200, 5) for _ in names]
    ends = [s + d for s, d in zip(starts, durs)]
    ds = rng.randrange(2)
    rows = [(n, [("starts at", _hm(s)), ("takes", _dur(d, ds))]) for n, s, d in zip(names, starts, durs)]
    style = rng.choice(["prose", "bullets", "kv"])

    def build(close: int) -> Problem | None:
        ctx = f"{sc['intro']} {sc['close'].capitalize()} at {_hm(close)}.\n{_table(random.Random(0), rows, style)}"
        ok = [e <= close for e in ends]
        meta = {"scenario": sc["id"], "close": close, "ends": ends}
        diff = "hard" if any(abs(e - close) <= 10 for e in ends) or late else "easy"
        if form == "noul":
            return _noul("tmp_window", "temporal", f"The {sc['noun']} '{names[0]}' is finished by the time {sc['close']}.", ctx, ends[0] <= close, diff,
                         "arithmetic", meta)
        keys = [(ends[i],) if ok[i] else None for i in range(k)]
        feats = [f"ends at {_hm(ends[i])}, " + (f"{close - ends[i]} min to spare" if ok[i] else f"{ends[i] - close} min too late") for i in range(k)]
        return _choice("tmp_window", "temporal", f"Choose the {sc['noun']} that ends last while still ending by the time {sc['close']}.", ctx,
                       names, feats, keys, [{"feasible": float(ok[i]), "end": float(ends[i]), "slack": float(close - ends[i])} for i in range(k)],
                       diff, "arithmetic", meta, {"feasible_frac": sum(ok) / k})

    s = sorted(set(ends))
    if want == "unanswerable":
        return _family([build(s[0] - rng.choice([5, 10, 30]))], []) if form == "choice" else None
    if len(s) < 3:
        return None
    i = rng.randrange(1, len(s))
    c1 = s[i] + rng.choice([0, 0, 5, 15]) if form == "choice" else ends[0] + rng.choice([0, 5, 20])
    c2 = s[i - 1] + rng.choice([0, 5]) if form == "choice" else ends[0] - rng.choice([5, 10, 25])
    if form == "choice" and not (s[i - 1] <= c2 < s[i] <= c1 and (i == len(s) - 1 or c1 < s[i + 1])):
        return None
    base = build(c1)
    fam = _family([base, build(c2)], ["closing time"])
    return fam or (_family([base], []) if base is not None else None)


# ================================================================================================ temporal: dates
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
DATE_SCEN = [
    dict(id="returns", a="The parcel was delivered on {d0}.", b="The customer asked to return it on {d1}.",
         rule="Returns are refunded in full within {w1} days of delivery, get store credit from day {w1p} to day {w2}, and are refused after that.",
         out=["full refund", "store credit", "refuse the return"], prop="The customer is entitled to a full refund."),
    dict(id="warranty", a="The device was purchased on {d0}.", b="The fault was reported on {d1}.",
         rule="Faults reported within {w1} days of purchase are repaired free of charge, from day {w1p} to day {w2} at half price, and later at full price.",
         out=["free repair", "half price repair", "full price repair"], prop="The repair is free of charge."),
    dict(id="invoice", a="The invoice was issued on {d0}.", b="The payment arrived on {d1}.",
         rule="Payments within {w1} days of the invoice get the early discount, from day {w1p} to day {w2} the list amount is due, and later a late fee is added.",
         out=["apply the early discount", "charge the list amount", "add the late fee"], prop="The early discount applies."),
    dict(id="trial", a="The trial started on {d0}.", b="The cancellation request arrived on {d1}.",
         rule="A trial cancelled within {w1} days is not charged, from day {w1p} to day {w2} the first month is charged, and later the full year is charged.",
         out=["charge nothing", "charge the first month", "charge the full year"], prop="Nothing is charged."),
    dict(id="permit", bench=True, a="The permit was issued on {d0}.", b="The inspection took place on {d1}.",
         rule="An inspection within {w1} days of issue is accepted as is, from day {w1p} to day {w2} a renewal form is required, and later a new permit is required.",
         out=["accept as is", "require the renewal form", "require a new permit"], prop="The inspection is accepted as is."),
]


def _date(d: dt.date, style: int) -> str:
    return [f"{d.day} {MONTHS[d.month - 1]} {d.year}", d.isoformat(), f"{MONTHS[d.month - 1]} {d.day}, {d.year}"][style]


def gen_tmp_dates(rng: random.Random, split: str = "train", form: str | None = None, k: int | None = None, want: str = "answerable"):
    if want != "answerable":
        return None
    form = form or rng.choice(["choice", "choice", "noul"])
    sc = _scen(rng, DATE_SCEN, split)
    w1 = rng.choice([7, 10, 14, 15, 30, 45, 60])
    w2 = w1 + rng.choice([7, 15, 30, 60])
    d0 = dt.date(rng.randint(2023, 2027), rng.randint(1, 12), rng.randint(1, 28))
    style = rng.randrange(3)
    order = rng.sample(range(3), 3)

    def build(gap: int) -> Problem | None:
        d1 = d0 + dt.timedelta(days=gap)
        ctx = " ".join([sc["rule"].format(w1=w1, w1p=w1 + 1, w2=w2), sc["a"].format(d0=_date(d0, style)), sc["b"].format(d1=_date(d1, style))])
        act = 0 if gap <= w1 else 1 if gap <= w2 else 2
        meta = {"scenario": sc["id"], "gap_days": gap, "w1": w1, "w2": w2}
        diff = "hard" if min(abs(gap - w1), abs(gap - w2)) <= 2 or d0.month != d1.month else "easy"
        if form == "noul":
            return _noul("tmp_dates", "temporal", sc["prop"], ctx, act == 0, diff, "calendar", meta)
        keys = [(1,) if i == act else (0,) for i in order]
        return _choice("tmp_dates", "temporal", "Choose the outcome the rule gives for this case.", ctx, [sc["out"][i] for i in order],
                       [f"day {gap} after the start date" for _ in order], keys, [{"match": float(i == act), "gap": float(gap)} for i in order],
                       diff, "calendar", meta, {"gap": float(gap)})

    g1 = rng.choice([rng.randint(1, w1), w1 - rng.randint(0, 2), w1 + rng.randint(1, 3), rng.randint(w1 + 1, w2), w2 + rng.randint(1, 40)])
    g1 = max(1, g1)
    a1 = 0 if g1 <= w1 else 1 if g1 <= w2 else 2
    g2 = rng.choice([w1 + rng.randint(1, 3), w2 + rng.randint(1, 5)]) if a1 == 0 else rng.choice([w1 - rng.randint(0, 2), w2 + rng.randint(1, 5)]) if a1 == 1 \
        else rng.choice([w2 - rng.randint(0, 3), w1 - rng.randint(0, 2)])
    base = build(g1)
    fam = _family([base, build(max(1, g2))], ["second date"])
    return fam or (_family([base], []) if base is not None else None)


# ================================================================================================ multi-hop: lookup
PEOPLE = ["ana", "bruno", "carla", "diego", "elena", "farid", "greta", "hugo", "ines", "jonas", "kira", "leo", "mara", "nuno", "olga", "paulo", "quinn", "rosa"]
CITIES = ["lisbon", "porto", "madrid", "lyon", "turin", "krakow", "gdansk", "malmo", "bergen", "cork", "leeds", "ghent", "graz", "split", "tartu"]
HOP_SCEN = [
    dict(id="org", layers=[("service", ["ledger", "inbox", "catalog", "router", "vault", "tracker", "notifier", "planner", "gallery"]),
                           ("team", ["storage", "payments", "growth", "platform", "mobile", "identity", "search", "insights"]),
                           ("lead", PEOPLE), ("office", CITIES),
                           ("building", ["tower one", "tower two", "the annex", "the mill", "the depot", "the foundry", "the atrium"])],
         rel=[["{a} is owned by the {b} team.", "The {b} team owns {a}."], ["The {a} team is led by {b}.", "{b} leads the {a} team."],
              ["{a} works from the {b} office.", "{a} is based in {b}."], ["The {a} office is in {b}.", "In {a} the company rents {b}."]],
         q=["Which team owns {x}?", "Who leads the team that owns {x}?", "From which office does the lead of the team that owns {x} work?",
            "In which building is the office of the lead of the team that owns {x}?"]),
    dict(id="supply", layers=[("product", ["the kettle", "the lamp", "the backpack", "the drill", "the scooter", "the tent", "the blender", "the speaker"]),
                              ("supplier", ["norte", "calder", "vintra", "belmar", "ostrand", "quillon", "tamsin", "harrow"]),
                              ("port", ["santos", "valencia", "rotterdam", "piraeus", "gdynia", "antwerp", "genoa", "leixoes"]),
                              ("broker", PEOPLE), ("day", ["monday", "tuesday", "wednesday", "thursday", "friday"])],
         rel=[["{a} is made by {b}.", "{b} manufactures {a}."], ["{a} ships everything through {b}.", "All goods of {a} leave from {b}."],
              ["Cargo in {a} is cleared by {b}.", "{b} is the customs broker for {a}."], ["{a} only files paperwork on {b}.", "On {b}, {a} files paperwork."]],
         q=["Who makes {x}?", "Through which port does the maker of {x} ship?", "Who clears the cargo at the port used by the maker of {x}?",
            "On which day does the broker for the port used by the maker of {x} file paperwork?"]),
    dict(id="campus", layers=[("course", ["statistics", "optics", "ethics", "compilers", "ceramics", "genetics", "acoustics", "topology"]),
                              ("teacher", PEOPLE), ("department", ["physics", "philosophy", "design", "biology", "mathematics", "music", "computing"]),
                              ("hall", ["north hall", "river hall", "stone hall", "garden hall", "east hall", "glass hall", "old hall"]),
                              ("campus", ["the hill campus", "the harbor campus", "the city campus", "the forest campus", "the lake campus"])],
         rel=[["{a} is taught by {b}.", "{b} teaches {a}."], ["{a} belongs to the {b} department.", "The {b} department employs {a}."],
              ["The {a} department sits in {b}.", "{b} houses the {a} department."], ["{a} is on {b}.", "{b} includes {a}."]],
         q=["Who teaches {x}?", "To which department does the teacher of {x} belong?", "In which hall is the department of the teacher of {x}?",
            "On which campus is the hall of the department of the teacher of {x}?"]),
    dict(id="assets", bench=True, layers=[("device", ["laptop k1", "laptop k2", "tablet m4", "phone z9", "printer p3", "scanner s7", "camera c5"]),
                                          ("user", PEOPLE), ("manager", ["tessa", "viktor", "wanda", "xavier", "yara", "zeno", "ravi"]),
                                          ("cost center", ["center 410", "center 520", "center 630", "center 740", "center 850", "center 960"]),
                                          ("controller", ["alba", "boris", "celia", "dario", "edith", "felix"])],
         rel=[["{a} is assigned to {b}.", "{b} uses {a}."], ["{a} reports to {b}.", "{b} manages {a}."],
              ["{a} charges expenses to {b}.", "Expenses approved by {a} go to {b}."], ["{a} is controlled by {b}.", "{b} signs off {a}."]],
         q=["Who uses {x}?", "Who manages the user of {x}?", "Which cost center does the manager of the user of {x} charge?",
            "Who controls the cost center charged by the manager of the user of {x}?"]),
]


def gen_hop_lookup(rng: random.Random, split: str = "train", form: str | None = None, k: int | None = None, want: str = "answerable"):
    if want != "answerable":
        return None
    form = form or rng.choice(["choice", "choice", "noul"])
    sc = _scen(rng, HOP_SCEN, split)
    hops = rng.choice([2, 2, 3, 3, 4])
    m = k or rng.choice([3, 4, 4, 5, 6])
    if any(len(pool) < m for _, pool in sc["layers"][: hops + 1]):
        m = min(len(pool) for _, pool in sc["layers"][: hops + 1])
    ents = [rng.sample(pool, m) for _, pool in sc["layers"][: hops + 1]]
    maps = [rng.sample(range(m), m) for _ in range(hops)]              # layer i entity j -> layer i+1 entity maps[i][j]
    ph = [[rng.randrange(2) for _ in range(m)] for _ in range(hops)]
    order = [(i, j) for i in range(hops) for j in range(m)]
    rng.shuffle(order)
    x = rng.randrange(m)
    cand_order = rng.sample(range(m), m)

    def build(first: list[int]) -> Problem | None:
        mp = [first] + maps[1:]
        facts = [sc["rel"][i][ph[i][j]].format(a=ents[i][j], b=ents[i + 1][mp[i][j]]) for i, j in order]
        ctx = " ".join(f[0].upper() + f[1:] for f in facts)
        cur = x
        for i in range(hops):
            cur = mp[i][cur]
        meta = {"scenario": sc["id"], "hops": hops, "chains": m}
        q = sc["q"][hops - 1].format(x=ents[0][x])
        diff = "hard" if hops >= 3 else "easy"
        if form == "noul":
            claim = cand_order[0]
            return _noul("hop_lookup", "multi_hop", f"{q} The answer is {ents[hops][claim]}.", ctx, claim == cur, diff, "graph", meta)

        def src(j):      # the layer-0 entity whose chain ends in candidate j
            inv = j
            for i in reversed(range(hops)):
                inv = mp[i].index(inv)
            return ents[0][inv]

        keys = [(1,) if j == cur else (0,) for j in cand_order]
        return _choice("hop_lookup", "multi_hop", q, ctx, [ents[hops][j] for j in cand_order], [f"reached from {src(j)}" for j in cand_order], keys,
                       [{"match": float(j == cur)} for j in cand_order], diff, "graph", meta, {"hops": float(hops), "chains": float(m)})

    first2 = maps[0][:]
    y = rng.choice([j for j in range(m) if j != x])
    first2[x], first2[y] = first2[y], first2[x]
    if form == "noul" and rng.random() < 0.5:
        cur = x
        for i in range(hops):
            cur = maps[i][cur]
        cand_order = [cur] + [j for j in cand_order if j != cur]
    base = build(maps[0])
    fam = _family([base, build(first2)], ["first link"])
    return fam or (_family([base], []) if base is not None else None)


# ================================================================================================ multi-hop: rule chains
RULE_SCEN = [
    dict(id="data", subject=["the survey tool", "the payroll export", "the wiki", "the booking app", "the audit archive", "the loyalty app"],
         l0=("stores", ["card numbers", "health records", "public content only", "employee records", "location history"], "{s} stores {v}."),
         l1=(["class a", "class b", "class c", "class d"], "Systems that store {a} are {b}."),
         l2=(["a security review", "a privacy review", "a peer review", "a self check"], "{a} systems need {b} before launch."),
         l3=(["ten working days ahead", "five working days ahead", "two working days ahead", "one working day ahead"], "Requests for {a} must be filed {b}."),
         q=["Which class is {s}?", "Which review does {s} need before launch?", "How far ahead must the launch request for {s} be filed?"]),
    dict(id="travel", subject=["the trip to the summit", "the client visit", "the training week", "the site inspection", "the trade fair trip"],
         l0=("lasts", ["one day", "two to four days", "a full week", "more than a week"], "{s} lasts {v}."),
         l1=(["tier one", "tier two", "tier three", "tier four"], "Trips that last {a} are {b}."),
         l2=(["the team lead", "the department head", "the finance office", "the director"], "{a} trips are approved by {b}."),
         l3=(["within a day", "within three days", "within a week", "within two weeks"], "Approvals by {a} are answered {b}."),
         q=["Which tier is {s}?", "Who approves {s}?", "How quickly is the approval for {s} answered?"]),
    dict(id="support", subject=["the ticket from the clinic", "the ticket from the retailer", "the ticket from the school", "the ticket from the airline"],
         l0=("plan", ["the basic plan", "the team plan", "the business plan", "the enterprise plan"], "{s} comes from a customer on {v}."),
         l1=(["queue green", "queue yellow", "queue orange", "queue red"], "Customers on {a} are served from {b}."),
         l2=(["the day shift", "the swing shift", "the night shift", "the follow the sun crew"], "{a} is staffed by {b}."),
         l3=(["within eight hours", "within four hours", "within one hour", "within fifteen minutes"], "Tickets handled by {a} get a first reply {b}."),
         q=["Which queue serves {s}?", "Who staffs the queue that serves {s}?", "How fast is the first reply for {s}?"]),
    dict(id="waste", bench=True, subject=["the solvent drum", "the battery crate", "the paper bale", "the glass bin", "the oil filter box"],
         l0=("contains", ["flammable liquids", "heavy metals", "clean fibre", "inert material"], "{s} contains {v}."),
         l1=(["category one", "category two", "category three", "category four"], "Loads that contain {a} are {b}."),
         l2=(["a sealed truck", "a licensed carrier", "the regular truck", "the recycling van"], "{a} loads travel in {b}."),
         l3=(["a signed manifest", "a weight ticket", "a pickup note", "no paperwork"], "Shipments in {a} need {b}."),
         q=["Which category is {s}?", "How does {s} travel?", "Which paperwork does the shipment of {s} need?"]),
]


def gen_hop_rules(rng: random.Random, split: str = "train", form: str | None = None, k: int | None = None, want: str = "answerable"):
    if want != "answerable":
        return None
    form = form or rng.choice(["choice", "choice", "noul"])
    sc = _scen(rng, RULE_SCEN, split)
    hops = rng.choice([2, 3, 3])
    n = rng.choice([3, 4])
    l0 = rng.sample(sc["l0"][1], n)
    layers = [l0] + [rng.sample(sc[f"l{i}"][0], n) for i in (1, 2, 3)][:hops]
    maps = [rng.sample(range(n), n) for _ in range(hops)]
    rules = [(i, j) for i in range(hops) for j in range(n)]
    rng.shuffle(rules)
    s = rng.choice(sc["subject"])
    others = rng.sample([x for x in sc["subject"] if x != s], min(2, len(sc["subject"]) - 1))
    cand_order = rng.sample(range(n), n)
    v0 = rng.randrange(n)
    ov = [rng.randrange(n) for _ in others]

    def build(v: int) -> Problem | None:
        rule_txt = [sc[f"l{i + 1}"][1].format(a=layers[i][j], b=layers[i + 1][maps[i][j]]) for i, j in rules]
        facts = [sc["l0"][2].format(s=o, v=l0[x]) for o, x in zip(others, ov)] + [sc["l0"][2].format(s=s, v=l0[v])]
        ctx = " ".join(t[0].upper() + t[1:] for t in rule_txt + facts)
        cur = v
        for i in range(hops):
            cur = maps[i][cur]
        meta = {"scenario": sc["id"], "hops": hops + 0, "n": n}
        q = sc["q"][hops - 1].format(s=s)
        if form == "noul":
            return _noul("hop_rules", "multi_hop", f"{q} The answer is {layers[hops][cand_order[0]]}.", ctx, cand_order[0] == cur, "hard", "rule", meta)
        keys = [(1,) if j == cur else (0,) for j in cand_order]
        return _choice("hop_rules", "multi_hop", q, ctx, [layers[hops][j] for j in cand_order],
                       ["applies to cases with " + layers[0][next(a for a in range(n) if _walk(maps, a, hops) == j)] for j in cand_order], keys,
                       [{"match": float(j == cur)} for j in cand_order], "hard", "rule", meta, {"hops": float(hops)})

    base = build(v0)
    fam = _family([base, build(rng.choice([x for x in range(n) if x != v0]))], ["the case fact"])
    return fam or (_family([base], []) if base is not None else None)


def _walk(maps, a, hops):
    for i in range(hops):
        a = maps[i][a]
    return a


# ================================================================================================ long policy: documents
# attribute kinds: ("num", key, lo, hi, step, gt_phrase, le_phrase, case_phrase)   ("cat", key, values, cond_phrase, case_phrase)
#                  ("bool", key, true_cond, false_cond, true_case, false_case)
POL_SCEN = [
    dict(id="expense", title="Expense policy", subject="claim",
         attrs=[("num", "amount", 20, 3000, 10, "the amount is over ${x}", "the amount is ${x} or less", "The claim is for ${v}."),
                ("bool", "receipt", "a receipt is attached", "no receipt is attached", "A receipt is attached.", "No receipt is attached."),
                ("cat", "category", ["travel", "meals", "equipment", "training"], "the category is {c}", "The category is {v}."),
                ("num", "age", 1, 120, 1, "it was filed more than {x} days after the purchase", "it was filed within {x} days of the purchase",
                 "It was filed {v} days after the purchase."),
                ("cat", "level", ["intern", "staff", "manager", "director"], "the claimant's level is {c}", "The claimant's level is {v}.")],
         outcomes=["pay automatically", "send to the manager", "send to finance", "reject the claim", "ask for more documents"]),
    dict(id="access", title="Access request policy", subject="request",
         attrs=[("cat", "system", ["the wiki", "the billing console", "the production database", "the hr archive"], "the request is for {c}", "The request is for {v}."),
                ("bool", "training", "the security training is complete", "the security training is missing", "The requester completed the security training.",
                 "The requester has not completed the security training."),
                ("num", "tenure", 0, 60, 1, "the requester has been employed for more than {x} months", "the requester has been employed for {x} months or less",
                 "The requester has been employed for {v} months."),
                ("cat", "role", ["contractor", "engineer", "analyst", "team lead"], "the requester's role is {c}", "The requester's role is {v}."),
                ("num", "days", 1, 90, 1, "access is requested for more than {x} days", "access is requested for {x} days or less", "Access is requested for {v} days.")],
         outcomes=["grant immediately", "grant after the owner approves", "grant read only access", "deny the request", "escalate to security"]),
    dict(id="leave", title="Leave policy", subject="request",
         attrs=[("num", "length", 1, 40, 1, "the leave is longer than {x} days", "the leave is {x} days or shorter", "The leave is {v} days long."),
                ("num", "notice", 0, 60, 1, "it was requested more than {x} days in advance", "it was requested {x} days in advance or less",
                 "It was requested {v} days in advance."),
                ("cat", "season", ["the quiet season", "the regular season", "the peak season"], "it falls in {c}", "It falls in {v}."),
                ("bool", "cover", "a colleague covers the shift", "nobody covers the shift", "A colleague covers the shift.", "Nobody covers the shift."),
                ("cat", "kind", ["vacation", "study leave", "family leave", "unpaid leave"], "it is {c}", "It is {v}.")],
         outcomes=["approve directly", "approve after the manager signs", "approve half of the days", "postpone the request", "refuse the request"]),
    dict(id="refund", title="Refund policy", subject="order",
         attrs=[("num", "value", 5, 900, 5, "the order value is over ${x}", "the order value is ${x} or less", "The order value is ${v}."),
                ("cat", "state", ["unopened", "opened", "damaged in transit", "used"], "the item is {c}", "The item is {v}."),
                ("num", "since", 1, 90, 1, "more than {x} days passed since delivery", "{x} days or fewer passed since delivery", "{v} days passed since delivery."),
                ("bool", "member", "the buyer is a member", "the buyer is not a member", "The buyer is a member.", "The buyer is not a member."),
                ("cat", "channel", ["the web shop", "a retail store", "a marketplace"], "it was bought through {c}", "It was bought through {v}.")],
         outcomes=["refund in full", "refund as store credit", "offer a replacement", "refuse the refund", "hand over to a supervisor"]),
    dict(id="incident", title="Incident classification guide", subject="incident",
         attrs=[("num", "users", 0, 5000, 50, "more than {x} users are affected", "{x} users or fewer are affected", "{v} users are affected."),
                ("bool", "dataloss", "data was lost", "no data was lost", "Data was lost.", "No data was lost."),
                ("cat", "area", ["checkout", "search", "reports", "the admin panel"], "the affected area is {c}", "The affected area is {v}."),
                ("num", "minutes", 1, 240, 1, "it has lasted more than {x} minutes", "it has lasted {x} minutes or less", "It has lasted {v} minutes."),
                ("bool", "workaround", "a workaround exists", "there is no workaround", "A workaround exists.", "There is no workaround.")],
         outcomes=["severity one", "severity two", "severity three", "severity four", "not an incident"]),
    dict(id="procure", bench=True, title="Purchasing policy", subject="purchase",
         attrs=[("num", "price", 50, 20000, 50, "the price is over ${x}", "the price is ${x} or less", "The price is ${v}."),
                ("cat", "vendor", ["an approved vendor", "a new vendor", "a foreign vendor"], "the seller is {c}", "The seller is {v}."),
                ("bool", "budgeted", "the purchase is in the budget", "the purchase is not in the budget", "The purchase is in the budget.",
                 "The purchase is not in the budget."),
                ("num", "quotes", 0, 5, 1, "more than {x} quotes were collected", "{x} quotes or fewer were collected", "{v} quotes were collected."),
                ("cat", "kind", ["software", "hardware", "services", "furniture"], "it is a purchase of {c}", "It is a purchase of {v}.")],
         outcomes=["order directly", "order after the manager signs", "order after procurement reviews", "collect more quotes", "decline the purchase"]),
    dict(id="moderation", bench=True, title="Forum moderation guide", subject="post",
         attrs=[("num", "reports", 0, 40, 1, "it has more than {x} reports", "it has {x} reports or fewer", "It has {v} reports."),
                ("cat", "content", ["an advertisement", "an off topic remark", "a personal attack", "a duplicate question"], "the post is {c}", "The post is {v}."),
                ("bool", "repeat", "the author was warned before", "the author was never warned", "The author was warned before.", "The author was never warned."),
                ("num", "age", 0, 400, 1, "the account is older than {x} days", "the account is {x} days old or younger", "The account is {v} days old."),
                ("bool", "staff", "the author is staff", "the author is not staff", "The author is staff.", "The author is not staff.")],
         outcomes=["leave it up", "hide the post", "warn the author", "suspend the author", "send to a senior moderator"]),
]


def _cond(rng: random.Random, a) -> tuple:
    if a[0] == "num":
        x = rng.randrange(a[2] + a[4], a[3], a[4])
        return ("num", a[1], rng.choice(["gt", "le"]), x)
    if a[0] == "cat":
        return ("cat", a[1], rng.choice(a[2]))
    return ("bool", a[1], rng.random() < 0.5)


def _cond_ok(c, case: dict) -> bool:
    v = case[c[1]]
    if c[0] == "num":
        return v > c[3] if c[2] == "gt" else v <= c[3]
    return v == c[2]


def _cond_txt(c, attrs: dict) -> str:
    a = attrs[c[1]]
    if c[0] == "num":
        return (a[5] if c[2] == "gt" else a[6]).format(x=num(c[3]))
    if c[0] == "cat":
        return a[3].format(c=c[2])
    return a[2] if c[2] else a[3]


def _case_txt(case: dict, attrs: dict, keys: list[str]) -> str:
    out = []
    for k in keys:
        a, v = attrs[k], case[k]
        out.append(a[7].format(v=num(v)) if a[0] == "num" else a[4].format(v=v) if a[0] == "cat" else (a[4] if v else a[5]))
    return " ".join(out)


def _policy(rng: random.Random, sc: dict, n_rules: int, extra: tuple | None = None):
    attrs = {a[1]: a for a in sc["attrs"]}
    if extra:
        attrs[extra[1]] = extra
    outs = rng.sample(sc["outcomes"], rng.choice([3, 4, 4, 5]))
    rules = []
    for _ in range(n_rules):
        ks = rng.sample(list(attrs), rng.choice([1, 1, 2, 2, 3]))
        if extra and rng.random() < 0.5 and extra[1] not in ks:
            ks[0] = extra[1]
        rules.append(([_cond(rng, attrs[k]) for k in ks], rng.choice(outs)))
    default = rng.choice(outs)
    return attrs, outs, rules, default


def _policy_eval(rules, default, case) -> tuple[int, str]:
    for i, (cs, out) in enumerate(rules):
        if all(_cond_ok(c, case) for c in cs):
            return i, out
    return len(rules), default


def _policy_text(sc, attrs, rules, default) -> str:
    lines = [f"{sc['title']}. Apply the first clause that matches the {sc['subject']}."]
    for i, (cs, out) in enumerate(rules):
        lines.append(f"{i + 1}. If {' and '.join(_cond_txt(c, attrs) for c in cs)}: {out}.")
    lines.append(f"{len(rules) + 1}. In every other case: {default}.")
    return "\n".join(lines)


def _random_case(rng: random.Random, attrs: dict) -> dict:
    return {k: (rng.randrange(a[2], a[3] + 1, a[4]) if a[0] == "num" else rng.choice(a[2]) if a[0] == "cat" else rng.random() < 0.5)
            for k, a in attrs.items()}


def gen_pol_document(rng: random.Random, split: str = "train", form: str | None = None, k: int | None = None, want: str = "answerable",
                     n_rules: int | None = None):
    if want != "answerable":
        return None
    form = form or rng.choice(["choice", "choice", "noul"])
    sc = _scen(rng, POL_SCEN, split)
    attrs, outs, rules, default = _policy(rng, sc, n_rules or rng.choice([4, 5, 6, 8, 10, 12, 14]))
    doc = _policy_text(sc, attrs, rules, default)
    keys = rng.sample(list(attrs), len(attrs))
    order = rng.sample(range(len(outs)), len(outs))
    case_first = rng.random() < 0.3
    claim = rng.choice(outs)

    def build(case: dict) -> tuple[Problem | None, int, str]:
        i, out = _policy_eval(rules, default, case)
        c = f"The {sc['subject']}: {_case_txt(case, attrs, keys)}"
        ctx = f"{c}\n{doc}" if case_first else f"{doc}\n{c}"
        meta = {"scenario": sc["id"], "n_rules": len(rules), "fired": i + 1}
        diff = "hard" if i >= 3 or len(rules) >= 8 else "easy"
        if form == "noul":
            return _noul("pol_document", "long_policy", f"Under this policy the outcome for the {sc['subject']} is: {claim}.", ctx, out == claim,
                         diff, "rule", meta), i, out
        ks = [(1,) if outs[j] == out else (0,) for j in order]
        return _choice("pol_document", "long_policy", f"Choose the outcome the policy gives for this {sc['subject']}.", ctx, [outs[j] for j in order],
                       [f"clause {i + 1} is the first that matches" if outs[j] == out else "not the outcome of the first matching clause" for j in order],
                       ks, [{"match": float(outs[j] == out)} for j in order], diff, "rule", meta, {"n_rules": float(len(rules)), "fired": float(i + 1)}), i, out

    for _ in range(30):
        case = _random_case(rng, attrs)
        base, i0, o0 = build(case)
        if i0 == len(rules) and rng.random() < 0.7:        # prefer cases that a real clause decides
            continue
        for _ in range(40):                                # counterfactual: change ONE attribute of the case
            kk = rng.choice(list(attrs))
            case2 = dict(case)
            case2[kk] = _random_case(rng, attrs)[kk]
            if case2[kk] == case[kk]:
                continue
            i1, o1 = _policy_eval(rules, default, case2)
            if o1 != o0 and (form != "noul" or (o0 == claim) != (o1 == claim)):
                fam = _family([base, build(case2)[0]], [f"the case attribute {kk}"])
                if fam:
                    return fam
        return _family([base], []) if base is not None else None
    return None


# ================================================================================================ long policy: horizon
HOR_SCEN = [
    dict(id="storage", intro="A team must choose a storage plan.", unit="TB", what="stored data"),
    dict(id="seats", intro="A company must choose a software licence.", unit="seats", what="active users"),
    dict(id="lines", intro="A call centre must choose a telephony contract.", unit="lines", what="concurrent calls"),
    dict(id="lease", intro="A courier must choose a vehicle lease.", unit="vans", what="daily routes"),
    dict(id="power", bench=True, intro="A workshop must choose a power contract.", unit="kW", what="peak demand"),
]


def gen_pol_horizon(rng: random.Random, split: str = "train", form: str | None = None, k: int | None = None, want: str = "answerable"):
    form = form or rng.choice(["choice", "choice", "noul"])
    sc = _scen(rng, HOR_SCEN, split)
    k = k or rng.choice([3, 4, 5])
    names = rng.sample(NAMES, k)
    now, grow = rng.randrange(40, 400, 10), rng.randrange(2, 40, 2)
    caps = [now + rng.randrange(0, grow * 40, 10) for _ in names]
    month = [rng.randrange(100, 3000, 50) for _ in names]
    up = [rng.choice([0, 0, rng.randrange(500, 20000, 500)]) for _ in names]
    style = rng.choice(["prose", "bullets", "kv"])
    rows = [(n, [("capacity", f"{num(c)} {sc['unit']}"), ("monthly fee", f"${num(m)}"), ("one time fee", f"${num(u)}")])
            for n, c, m, u in zip(names, caps, month, up)]

    def build(h: int) -> Problem | None:
        need = now + grow * h
        ctx = (f"{sc['intro']} Today {sc['what']} stands at {num(now)} {sc['unit']} and it grows by {grow} {sc['unit']} every month. "
               f"The plan cannot be changed for {h} months.\n{_table(random.Random(0), rows, style)}")
        ok = [c >= need for c in caps]
        total = [u + m * h for u, m in zip(up, month)]
        last = [min(h, (c - now) // grow) for c in caps]
        meta = {"scenario": sc["id"], "horizon": h, "need": need}
        cheap_now = min(range(k), key=lambda i: month[i])
        diff = "hard" if not ok[cheap_now] or len({t for t, o in zip(total, ok) if o}) > 1 else "easy"
        if form == "noul":
            return _noul("pol_horizon", "long_policy", f"Plan {names[0]} has enough capacity for the whole {h} months.", ctx, ok[0], diff, "simulation", meta)
        keys = [(-total[i],) if ok[i] else None for i in range(k)]
        feats = [(f"enough for {h} months; total cost ${num(total[i])}" if ok[i] else f"runs out of capacity after month {last[i]}") for i in range(k)]
        return _choice("pol_horizon", "long_policy", f"Choose the plan that has enough capacity for the whole period at the lowest total cost over the {h} months.",
                       ctx, names, feats, keys, [{"feasible": float(ok[i]), "total": float(total[i]), "last_month": float(last[i])} for i in range(k)],
                       diff, "simulation", meta, {"need": float(need), "feasible_frac": sum(ok) / k})

    h1 = rng.choice([3, 6, 6, 12, 12, 18, 24, 36])
    if want == "unanswerable":
        if form != "choice":
            return None
        h = next((h for h in (12, 24, 36, 48, 60) if all(c < now + grow * h for c in caps)), None)
        return _family([build(h)], []) if h else None
    base = build(h1)
    if base is None or not base["answerable"]:
        return None
    for h2 in rng.sample([x for x in (3, 6, 12, 18, 24, 36, 48) if x != h1], 6):
        fam = _family([base, build(h2)], ["the horizon"])
        if fam:
            return fam
    return _family([base], [])


# ================================================================================================ long policy: backlog
def gen_pol_backlog(rng: random.Random, split: str = "train", form: str | None = None, k: int | None = None, want: str = "answerable"):
    form = form or rng.choice(["choice", "choice", "noul"])
    what = rng.choice([("messages", "consumers"), ("orders", "packers"), ("tickets", "agents"), ("images", "render nodes")]
                      + ([("claims", "adjusters")] if split == "bench" else []))
    k = k or rng.choice([3, 4, 5])
    names = rng.sample(NAMES, k)
    backlog, arr = rng.randrange(500, 20000, 100), rng.randrange(20, 400, 10)
    rate = [arr + rng.randrange(-40, 600, 10) for _ in names]
    delay = [rng.choice([0, 5, 10, 20, 30, 45, 60]) for _ in names]
    cost = [rng.randrange(20, 900, 10) for _ in names]
    rows = [(n, [("clears", f"{num(r)} {what[0]} per minute"), ("ready after", f"{d} minutes"), ("cost", f"${num(c)}")])
            for n, r, d, c in zip(names, rate, delay, cost)]
    style = rng.choice(["prose", "bullets", "kv"])

    def done(i: int) -> float | None:
        if rate[i] <= arr:
            return None
        return delay[i] + (backlog + arr * delay[i]) / (rate[i] - arr)

    t = [done(i) for i in range(k)]

    def build(deadline: int) -> Problem | None:
        ctx = (f"A queue holds {num(backlog)} {what[0]} and {arr} more arrive every minute. Until an option is ready nothing is processed. "
               f"The queue must be empty within {deadline} minutes.\n{_table(random.Random(0), rows, style)}")
        ok = [x is not None and x <= deadline for x in t]
        if any(x is not None and abs(x - deadline) < 0.5 for x in t):
            return None
        meta = {"scenario": what[0], "deadline": deadline, "arrival": arr, "backlog": backlog}
        if form == "noul":
            return _noul("pol_backlog", "long_policy", f"Option {names[0]} empties the queue in time.", ctx, ok[0], "hard", "simulation", meta)
        keys = [(-cost[i],) if ok[i] else None for i in range(k)]
        feats = ["never catches up with arrivals" if t[i] is None else f"queue empty after {t[i]:.0f} minutes" + ("" if ok[i] else ", too late") for i in range(k)]
        return _choice("pol_backlog", "long_policy", "Choose the cheapest option that empties the queue in time.", ctx, names, feats, keys,
                       [{"feasible": float(ok[i]), "minutes": float(t[i] if t[i] is not None else 1e4), "cost": float(cost[i])} for i in range(k)],
                       "hard", "simulation", meta, {"feasible_frac": sum(ok) / k})

    fin = sorted(x for x in t if x is not None)
    if want == "unanswerable":
        if form != "choice":
            return None
        return _family([build(max(5, int(fin[0] * 0.8)) if fin else 60)], [])
    if len(fin) < 2:
        return None
    i = rng.randrange(1, len(fin))
    base = build(int(fin[i] * 1.1) + 2)
    fam = _family([base, build(int((fin[i - 1] + fin[i]) / 2))], ["the deadline"])
    return fam or (_family([base], []) if base is not None else None)


# ================================================================================================ trade-off
LEVEL = ["low", "medium", "high"]
TRD_SCEN = [
    dict(id="queue", intro="A team is choosing how to move events between services; these are its own assessments.",
         opts=["kafka", "sqs", "rabbitmq", "nats", "redis streams", "pubsub"]),
    dict(id="store", intro="A team is choosing where to keep its main records; these are its own assessments for this workload.",
         opts=["postgresql", "dynamodb", "cockroachdb", "redis", "mongodb", "cassandra"]),
    dict(id="api", intro="Two services need to talk to each other; the team assessed each style for this case.",
         opts=["rest", "grpc", "graphql", "message queue", "webhooks"]),
    dict(id="arch", intro="A product is being restructured; the team assessed each direction.",
         opts=["keep the monolith", "split into services", "modular monolith", "serverless functions"]),
    dict(id="infer", intro="A model has to be served; the team assessed each way of running it.",
         opts=["cpu instances", "gpu instances", "managed endpoint", "batch jobs", "edge devices"]),
    dict(id="cache", intro="A slow report needs to get faster; the team assessed each approach.",
         opts=["redis cache", "materialized view", "read replica", "precomputed files", "query tuning only"]),
    dict(id="buy", intro="A company needs a billing system; the team assessed each route.",
         opts=["build in house", "buy from a vendor", "adopt open source", "outsource to an agency"]),
    dict(id="hire", bench=True, intro="A team needs more capacity for a project; the manager assessed each route.",
         opts=["hire full time", "hire contractors", "train the current team", "use an agency", "delay the project"]),
    dict(id="commute", bench=True, intro="A firm is choosing how staff reach a new site; facilities assessed each way.",
         opts=["shuttle bus", "parking passes", "transit subsidy", "bike scheme", "remote days"]),
]
TRD_ATTR = [("cost", "low"), ("latency", "low"), ("operational effort", "low"), ("complexity", "low"), ("consistency", "high"), ("availability", "high"),
            ("scalability", "high"), ("team familiarity", "high"), ("failure tolerance", "high"), ("time to deliver", "low")]


def gen_trd_priorities(rng: random.Random, split: str = "train", form: str | None = None, k: int | None = None, want: str = "answerable"):
    form = form or rng.choice(["choice", "choice", "choice", "noul"])
    sc = _scen(rng, TRD_SCEN, split)
    k = min(k or rng.choice([3, 4, 4, 5]), len(sc["opts"]))
    names = rng.sample(sc["opts"], k)
    attrs = rng.sample(TRD_ATTR, rng.choice([3, 4, 4, 5]))
    lv = {a: [rng.randrange(3) for _ in names] for a, _ in attrs}
    good = lambda a, d, v: v if d == "high" else 2 - v  # noqa: E731     0 worst .. 2 best
    style = rng.choice(["prose", "bullets", "kv", "json"])
    rows = [(n, [(a, LEVEL[lv[a][i]]) for a, _ in attrs]) for i, n in enumerate(names)]
    tab = _table(rng, rows, style)

    def build(must: tuple[str, str] | None, prio: list[tuple[str, str]]) -> Problem | None:
        def fails(i):
            return must is not None and good(must[0], must[1], lv[must[0]][i]) == 0
        must_txt = "" if must is None else " They cannot accept %s %s." % (("high" if must[1] == "low" else "low"), must[0])
        words = ["above all", "then", "then", "then"]
        pr_txt = ", ".join("%s %s %s" % (w, "low" if d == "low" else "high", a) for w, (a, d) in zip(words, prio))
        ctx = f"{sc['intro']}\n{tab}\nPriorities: {pr_txt}.{must_txt}"
        keys = [None if fails(i) else tuple(good(a, d, lv[a][i]) for a, d in prio) for i in range(k)]
        meta = {"scenario": sc["id"], "priorities": [a for a, _ in prio], "must": must[0] if must else None}
        feas = [x for x in keys if x is not None]
        if not feas:
            gold = None
        else:
            best = max(feas)
            if feas.count(best) != 1:
                return None
            gold = keys.index(best)
        top = max((good(prio[0][0], prio[0][1], lv[prio[0][0]][i]) for i in range(k)))
        tie_first = sum(good(prio[0][0], prio[0][1], lv[prio[0][0]][i]) == top for i in range(k)) > 1
        diff = "hard" if tie_first or must is not None else "easy"
        if form == "noul":
            if gold is None:
                return None
            return _noul("trd_priorities", "tradeoff", f"Given these priorities, {names[0]} is the best choice.", ctx, gold == 0, diff, "rule", meta)
        feats = [("ruled out by the hard requirement" if fails(i) else ", ".join(f"{a} {LEVEL[lv[a][i]]}" for a, _ in prio)) for i in range(k)]
        return _choice("trd_priorities", "tradeoff", "Choose the option that fits the stated priorities best: compare on the first priority, and use "
                       "the next one only to break ties. Options ruled out by a hard requirement cannot be chosen.", ctx, names, feats, keys,
                       [{"feasible": float(not fails(i)), "first": float(good(prio[0][0], prio[0][1], lv[prio[0][0]][i]))} for i in range(k)],
                       diff, "rule", meta, {"n_priorities": float(len(prio))})

    prio = rng.sample(attrs, rng.choice([2, 3, 3]))
    rest = [a for a in attrs if a not in prio]
    must = rng.choice(rest) if rest and rng.random() < 0.5 else None
    if want == "unanswerable":
        if form != "choice":
            return None
        for a in attrs:
            if all(good(a[0], a[1], lv[a[0]][i]) == 0 for i in range(k)):
                return _family([build(a, [x for x in attrs if x != a][:2])], [])
        return None
    base = build(must, prio)
    alts = [(must, rng.sample(prio, len(prio))), (must, prio[::-1]), (rng.choice(attrs), prio)]
    rng.shuffle(alts)
    for m2, p2 in alts:
        if (m2, p2) == (must, prio) or (m2 is not None and m2 in p2):
            continue
        fam = _family([base, build(m2, p2)], ["the priorities" if m2 == must else "the hard requirement"])
        if fam:
            return fam
    return _family([base], []) if base is not None else None


# ================================================================================================ mixed: window + budget
def gen_mix_window_budget(rng: random.Random, split: str = "train", form: str | None = None, k: int | None = None, want: str = "answerable"):
    form = form or rng.choice(["choice", "choice", "noul"])
    job = rng.choice(["the database migration", "the stage build", "the network upgrade", "the stock count", "the kitchen refit"]
                     + (["the lab relocation"] if split == "bench" else []))
    k = k or rng.choice([3, 4, 5])
    names = rng.sample(NAMES, k)
    start = [rng.randrange(18 * 60, 23 * 60, 10) for _ in names]
    dur = [rng.randrange(40, 420, 10) for _ in names]
    rate = [rng.randrange(40, 300, 5) for _ in names]
    crew = [rng.randint(1, 6) for _ in names]
    end = [s + d for s, d in zip(start, dur)]
    cost = [math.ceil(d / 60 * r * c) for d, r, c in zip(dur, rate, crew)]
    rows = [(n, [("can start at", _hm(s)), ("needs", _dur(d, 1)), ("crew", str(c)), ("rate per person per hour", f"${r}")])
            for n, s, d, c, r in zip(names, start, dur, crew, rate)]
    style = rng.choice(["prose", "bullets", "kv"])

    def build(close: int, budget: int) -> Problem | None:
        ctx = (f"Contractors bid for {job}. The work must be finished by {_hm(close)} and may cost at most ${num(budget)} in total "
               f"(crew x hours x rate).\n{_table(random.Random(0), rows, style)}")
        ok_t = [e <= close for e in end]
        ok_c = [c <= budget for c in cost]
        ok = [a and b for a, b in zip(ok_t, ok_c)]
        meta = {"scenario": job, "close": close, "budget": budget, "skills": ["numeric", "temporal"]}
        if form == "noul":
            return _noul("mix_window_budget", "mixed", f"The bid from {names[0]} meets both the deadline and the budget.", ctx, ok[0], "hard", "arithmetic", meta)
        keys = [(-end[i],) if ok[i] else None for i in range(k)]
        feats = [f"ends {_hm(end[i])}, costs ${num(cost[i])}" + ("" if ok[i] else "; " + " and ".join(x for x, o in (("too late", ok_t[i]), ("over budget", ok_c[i])) if not o))
                 for i in range(k)]
        return _choice("mix_window_budget", "mixed", "Choose the bid that meets the deadline and the budget and finishes earliest.", ctx, names, feats, keys,
                       [{"feasible": float(ok[i]), "end": float(end[i]), "cost": float(cost[i])} for i in range(k)], "hard", "arithmetic", meta,
                       {"feasible_frac": sum(ok) / k})

    close = sorted(end)[rng.randrange(k // 2, k)] + rng.choice([0, 10, 30])
    budget = int(sorted(cost)[rng.randrange(k // 2, k)] * rng.choice([1.0, 1.05, 1.2]))
    if want == "unanswerable":
        return _family([build(min(end) - 10, budget)], []) if form == "choice" else None
    base = build(close, budget)
    if base is None:
        return None
    g = base["gold"] if form == "choice" else 0
    alts = [(close, cost[g] - 1), (end[g] - 10, budget)] if (form == "choice" or base["gold"] == 1) else [(max(close, end[0]), max(budget, cost[0]))]
    rng.shuffle(alts)
    for c2, b2 in alts:
        fam = _family([base, build(c2, b2)], ["the budget" if c2 == close else "the deadline"])
        if fam:
            return fam
    return _family([base], [])


# ================================================================================================ mixed: lookup + time
def gen_mix_sla_lookup(rng: random.Random, split: str = "train", form: str | None = None, k: int | None = None, want: str = "answerable"):
    if want != "answerable":
        return None
    form = form or rng.choice(["choice", "noul"])
    custs = rng.sample(["the clinic", "the bakery", "the airline", "the school", "the museum", "the brewery", "the law firm"]
                       + (["the observatory"] if split == "bench" else []), 3)
    plans = rng.sample(["the basic plan", "the plus plan", "the pro plan", "the premier plan"], 3)
    hours = rng.sample([1, 2, 4, 8, 12, 24, 48], 3)
    pm = rng.sample(range(3), 3)
    facts = [f"{c[0].upper() + c[1:]} is on {plans[pm[i]]}." for i, c in enumerate(custs)] + \
            [f"Tickets on {p} must get a first reply within {h} hour{'s' if h > 1 else ''}." for p, h in zip(plans, hours)]
    rng.shuffle(facts)
    x = rng.randrange(3)
    opened = rng.randrange(6 * 60, 20 * 60, 5)
    order = rng.sample(range(3), 3)
    acts = ["the target was met", "the target was missed by less than an hour", "the target was missed by an hour or more"]

    def build(wait: int) -> Problem | None:
        limit = hours[pm[x]] * 60
        if wait == limit + 60:
            return None
        ctx = " ".join(facts) + f" A ticket from {custs[x]} was opened at {_hm(opened)} and got its first reply at {_hm(opened + wait)}."
        act = 0 if wait <= limit else 1 if wait < limit + 60 else 2
        meta = {"scenario": custs[x], "wait_min": wait, "limit_min": limit, "skills": ["multi_hop", "temporal"]}
        if form == "noul":
            return _noul("mix_sla_lookup", "mixed", "The first reply met the target for this customer.", ctx, act == 0, "hard", "arithmetic", meta)
        return _choice("mix_sla_lookup", "mixed", "Choose the statement that is true for this ticket.", ctx, [acts[i] for i in order],
                       [f"reply after {wait} min, target {limit} min" for _ in order], [(1,) if i == act else (0,) for i in order],
                       [{"match": float(i == act)} for i in order], "hard", "arithmetic", meta, {"wait": float(wait), "limit": float(limit)})

    lim = hours[pm[x]] * 60
    w1 = rng.choice([lim - rng.choice([0, 5, 20, 45]), lim + rng.choice([5, 20, 55]), lim + rng.choice([65, 90, 200])])
    w1 = max(5, w1)
    w2 = lim + rng.choice([10, 40]) if w1 <= lim else max(5, lim - rng.choice([0, 10, 30]))
    base = build(w1)
    fam = _family([base, build(w2)], ["the reply time"])
    return fam or (_family([base], []) if base is not None else None)


# ================================================================================================ mixed: policy + totals
def gen_mix_policy_totals(rng: random.Random, split: str = "train", form: str | None = None, k: int | None = None, want: str = "answerable"):
    """A policy document whose clauses depend on a TOTAL the case only gives as line items (quantity x unit price)."""
    if want != "answerable":
        return None
    form = form or rng.choice(["choice", "choice", "noul"])
    sc = _scen(rng, [s for s in POL_SCEN if s["id"] in ("expense", "refund", "procure")], split)
    tot = ("num", "total", 50, 4000, 50, "the total is over ${x}", "the total is ${x} or less", "")
    base_attrs = [a for a in sc["attrs"] if a[1] not in ("amount", "value", "price")]
    sc2 = dict(sc, attrs=rng.sample(base_attrs, 2))
    attrs, outs, rules, default = _policy(rng, sc2, rng.choice([4, 5, 6, 8]), extra=tot)
    doc = _policy_text(sc2, attrs, rules, default)
    keys = [a[1] for a in sc2["attrs"]]
    order = rng.sample(range(len(outs)), len(outs))
    goods = rng.sample(["cables", "adapters", "manuals", "filters", "badges", "cases", "sensors", "toner packs"], rng.choice([2, 3]))
    claim = rng.choice(outs)

    def items(scale: float):
        return [(g, rng.randint(1, 12), max(5, int(rng.randrange(10, 300, 5) * scale))) for g in goods]

    def build(case: dict, lines) -> tuple[Problem | None, str]:
        total = sum(q * p for _, q, p in lines)
        full = dict(case, total=total)
        i, out = _policy_eval(rules, default, full)
        if any(c[1] == "total" and c[3] == total for cs, _ in rules for c in cs):
            return None, out
        c = (f"The {sc['subject']}: " + "; ".join(f"{q} {g} at ${p} each" for g, q, p in lines) + ". " + _case_txt(case, attrs, keys))
        ctx = f"{doc}\n{c}"
        meta = {"scenario": sc["id"], "total": total, "fired": i + 1, "skills": ["long_policy", "numeric"]}
        if form == "noul":
            return _noul("mix_policy_totals", "mixed", f"Under this policy the outcome is: {claim}.", ctx, out == claim, "hard", "rule", meta), out
        return _choice("mix_policy_totals", "mixed", f"Choose the outcome the policy gives for this {sc['subject']}.", ctx, [outs[j] for j in order],
                       [f"total ${num(total)}; clause {i + 1} applies" if outs[j] == out else f"total ${num(total)}; another clause applies" for j in order],
                       [(1,) if outs[j] == out else (0,) for j in order], [{"match": float(outs[j] == out), "total": float(total)} for j in order],
                       "hard", "rule", meta, {"total": float(total)}), out

    for _ in range(20):
        case = {k_: v for k_, v in _random_case(rng, attrs).items() if k_ != "total"}
        l1 = items(1.0)
        b, o0 = build(case, l1)
        if b is None:
            continue
        for _ in range(30):
            j = rng.randrange(len(l1))
            l2 = list(l1)
            l2[j] = (l1[j][0], max(1, l1[j][1] + rng.choice([-6, -3, 3, 6, 12])), l1[j][2])
            v, o1 = build(case, l2)
            if v is not None and o1 != o0 and (form != "noul" or (o0 == claim) != (o1 == claim)):
                fam = _family([b, v], ["one quantity"])
                if fam:
                    return fam
        return _family([b], [])
    return None


# ================================================================================================ registry
GENERATORS: dict[str, Callable] = {
    "num_limits": gen_num_limits, "num_budget": gen_num_budget, "num_capacity": gen_num_capacity, "num_errbudget": gen_num_errbudget,
    "tmp_order": gen_tmp_order, "tmp_window": gen_tmp_window, "tmp_dates": gen_tmp_dates,
    "hop_lookup": gen_hop_lookup, "hop_rules": gen_hop_rules,
    "pol_document": gen_pol_document, "pol_horizon": gen_pol_horizon, "pol_backlog": gen_pol_backlog,
    "trd_priorities": gen_trd_priorities,
    "mix_window_budget": gen_mix_window_budget, "mix_sla_lookup": gen_mix_sla_lookup, "mix_policy_totals": gen_mix_policy_totals,
}
CAPABILITY = {"numeric": ["num_limits", "num_budget", "num_capacity", "num_errbudget"], "temporal": ["tmp_order", "tmp_window", "tmp_dates"],
              "multi_hop": ["hop_lookup", "hop_rules"], "long_policy": ["pol_document", "pol_horizon", "pol_backlog"],
              "tradeoff": ["trd_priorities"], "mixed": ["mix_window_budget", "mix_sla_lookup", "mix_policy_totals"]}
UNANSWERABLE = ["num_limits", "num_budget", "tmp_order", "tmp_window", "pol_horizon", "pol_backlog", "trd_priorities", "mix_window_budget"]
BENCH_SCENARIOS = {s["id"] for pool in (NUM_SCEN, BUD_SCEN, CAP_SCEN, ORD_SCEN, WIN_SCEN, DATE_SCEN, HOP_SCEN, RULE_SCEN, POL_SCEN, HOR_SCEN, TRD_SCEN)
                   for s in pool if s.get("bench")} | {"tracking api", "claims", "the lab relocation", "the observatory"}
