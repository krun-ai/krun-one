"""State / action / consequence simulators for the v1.1 corpus and KrunTechBench (truth = arithmetic or simulation).

Adds to the R0 environments (envs.py: tetris, scheduling, routing, sizing):

  quorum      distributed   read/write quorum sizes that stay consistent and available under f failures
  retry       api           timeout / attempts / backoff that fit a deadline
  rollout     devops        deployment order that respects service dependencies
  batching    mlinfra       inference batch size under a latency SLA, a memory limit and an arrival rate
  indexing    database      composite index for a query mix under a stated cost model
  eviction    state_action  (benchmark only) which cache key to evict given the future request trace
  placement   cloud         (benchmark only) tightest-fit host for a virtual machine

Every problem carries, per candidate, ``cons`` (the numeric consequence of the action) next to the textual ``feat``,
and ``state`` (numeric facts about the state). They supervise the auxiliary Q(state, action) / V(state) heads.

Tetris is produced in three views of the SAME state (``tetris_views``):
  A raw           board + candidate placements
  B structured    A + computed board features (column heights, holes, bumpiness, wells, nearly-full rows)
  C consequence   A + per-candidate outcome (lines, holes / height / bumpiness deltas, survival, heights after)
"""

from __future__ import annotations

import itertools
import math
import random
import re

from . import envs
from .envs import H, PIECES, W, _finish, _pick

Problem = dict


def _attach(p: Problem | None, cons: list[dict], state: dict | None = None) -> Problem | None:
    if p is None:
        return None
    for c, x in zip(p["candidates"], cons):
        c["cons"] = x
    p["state"] = state or {}
    return p


def _choose(rng: random.Random, keys: list, k: int) -> list[int] | None:
    feas = [x for x in keys if x is not None]
    if not feas or feas.count(max(feas)) != 1:
        return None
    return _pick(rng, list(range(len(keys))), keys.index(max(feas)), keys, k)


# ------------------------------------------------------------------------------------------------ R0 environments + cons
def _nums(s: str) -> list[int]:
    return [int(x) for x in re.findall(r"-?\d+", s)]


def with_cons(p: Problem | None) -> Problem | None:
    """Numeric consequences for the R0 generators, read back from their own outcome strings."""
    if p is None:
        return None
    sl = p["slice"]
    cons = []
    for c in p["candidates"]:
        feas = float(c["value"] is not None)
        if sl == "scheduling":
            cons.append({"feasible": feas, **({"finish": float(_nums(c["feat"])[0])} if feas else {})})
        elif sl == "routing":
            cons.append({"feasible": feas, **({"cost": float(_nums(c["feat"])[0])} if feas else {})})
        elif sl == "sizing":
            cons.append({"feasible": feas, "price": float(_nums(c["name"])[-1])})
        else:
            cons.append({})
    feas = [c for c in cons if c.get("feasible")]
    return _attach(p, cons, {"feasible_frac": len(feas) / len(cons)})


# ------------------------------------------------------------------------------------------------ quorum
def gen_quorum(rng: random.Random, k: int, n: int | None = None, f: int | None = None, heavy: str | None = None,
               pairs: list | None = None) -> Problem | None:
    n = n or rng.choice([3, 5, 5, 7, 7] if k <= 8 else [5, 7, 7])
    f = f or rng.randint(1, (n - 1) // 2)
    heavy = heavy or rng.choice(["read", "write"])
    allp = [(r, w) for r in range(1, n + 1) for w in range(1, n + 1)]

    def key(r, w):
        if r + w <= n or r > n - f or w > n - f:
            return None
        return (-r, -w) if heavy == "read" else (-w, -r)

    if pairs is None:
        keys = [key(r, w) for r, w in allp]
        idx = _choose(rng, keys, min(k, len(allp)))
        if idx is None or len(idx) < k:
            return None
        pairs = [allp[i] for i in idx]
    keys = [key(r, w) for r, w in pairs]
    feats, cons = [], []
    for (r, w), ky in zip(pairs, keys):
        why = ("stale reads possible" if r + w <= n else "writes blocked with %d replicas down" % f if w > n - f
               else "reads blocked with %d replicas down" % f if r > n - f else "consistent, survives %d failures" % f)
        feats.append(why)
        cons.append({"feasible": float(ky is not None), "r": float(r), "w": float(w)})
    other = "write" if heavy == "read" else "read"
    ctx = (f"A replicated store keeps {n} copies of every record. A read contacts r replicas and a write waits for w "
           f"replicas. It must never return stale data, and it must keep serving reads and writes with {f} "
           f"replica{'s' if f > 1 else ''} down. The workload is {heavy}-heavy.")
    ins = (f"Choose the quorum configuration that meets every requirement and makes {heavy}s cheapest: the smallest "
           f"{heavy} quorum; if tied, the smallest {other} quorum.")
    p = _finish("quorum", "distributed", ins, ctx, ["read quorum %d, write quorum %d" % x for x in pairs], feats, keys,
                "hard" if sum(x is not None for x in keys) >= 3 else "easy", "simulation", {"n": n, "f": f, "heavy": heavy})
    p = _attach(p, cons, {"feasible_frac": sum(x is not None for x in keys) / len(keys), "n": float(n), "f": float(f)})
    if p is not None:
        p["_pairs"] = pairs
    return p


def gen_quorum_flip(rng: random.Random, k: int):
    base = gen_quorum(rng, k)
    if base is None or base["gold"] is None:
        return None
    m = base["meta"]
    flip = gen_quorum(rng, k, m["n"], m["f"], "write" if m["heavy"] == "read" else "read", base["_pairs"])
    if flip is None or flip["gold"] is None or flip["gold"] == base["gold"]:
        return None
    return base, flip


def gen_quorum_unanswerable(rng: random.Random, k: int) -> Problem | None:
    n = rng.choice([5, 7])
    f = rng.randint(1, (n - 1) // 2)
    bad = [(r, w) for r in range(1, n + 1) for w in range(1, n + 1) if r + w <= n or r > n - f or w > n - f]
    if len(bad) < k:
        return None
    return gen_quorum(rng, k, n, f, rng.choice(["read", "write"]), rng.sample(bad, k))


# ------------------------------------------------------------------------------------------------ retry budget
def gen_retry(rng: random.Random, k: int, deadline: int | None = None, p99: int | None = None, cands: list | None = None
              ) -> Problem | None:
    p99 = p99 or rng.choice([80, 120, 150, 200, 250, 300, 400])
    deadline = deadline or rng.choice([600, 800, 1000, 1500, 2000, 3000])
    if cands is None:
        pool = list({(t, a, b) for t in (50, 100, 150, 200, 250, 300, 400, 500, 800) for a in (1, 2, 3, 4, 5)
                     for b in (0, 50, 100, 200)})
        cands = rng.sample(pool, k)
    keys, feats, cons = [], [], []
    for t, a, b in cands:
        worst = a * t + (a - 1) * b
        if t < p99:
            keys.append(None)
            feats.append("timeout below the p99 latency")
        elif worst > deadline:
            keys.append(None)
            feats.append("worst case %d ms, over the deadline" % worst)
        else:
            keys.append((a, -worst))
            feats.append("worst case %d ms, %d attempts" % (worst, a))
        cons.append({"feasible": float(keys[-1] is not None), "worst": float(worst), "attempts": float(a)})
    feas = [x for x in keys if x is not None]
    ctx = (f"A service calls a dependency whose p99 latency is {p99} ms. The caller must answer its own client within "
           f"{deadline} ms in the worst case. A retry policy is a per-attempt timeout, a number of attempts and a fixed "
           "backoff between attempts; its worst case is attempts times timeout plus the backoffs in between.")
    ins = ("Choose the retry policy with the most attempts among those whose timeout is at least the p99 latency and "
           "whose worst case fits the deadline; if tied, the smallest worst case.")
    p = _finish("retry", "api", ins, ctx, ["timeout %d ms, %d attempts, %d ms backoff" % c for c in cands], feats, keys,
                "hard" if len(feas) >= 3 else "easy", "simulation", {"p99": p99, "deadline": deadline})
    p = _attach(p, cons, {"feasible_frac": len(feas) / len(keys), "deadline": float(deadline), "p99": float(p99)})
    if p is not None:
        p["_cands"] = cands
    return p


def gen_retry_flip(rng: random.Random, k: int):
    base = gen_retry(rng, k)
    if base is None or base["gold"] is None:
        return None
    m = base["meta"]
    tighter = [d for d in (400, 600, 800, 1000, 1500, 2000) if d < m["deadline"]]
    rng.shuffle(tighter)
    for d in tighter:
        flip = gen_retry(rng, k, d, m["p99"], base["_cands"])
        if flip is not None and flip["gold"] is not None and flip["gold"] != base["gold"]:
            return base, flip
    return None


def gen_retry_unanswerable(rng: random.Random, k: int) -> Problem | None:
    p = gen_retry(rng, k, deadline=rng.choice([100, 150]), p99=rng.choice([200, 300, 400]))
    return p if p is not None and not p["answerable"] else None


# ------------------------------------------------------------------------------------------------ rollout order
SERVICES = ["database", "cache", "auth", "api", "web", "worker", "search", "billing", "gateway", "queue"]


def gen_rollout(rng: random.Random, k: int) -> Problem | None:
    n = 5 if k <= 8 else 6
    svc = rng.sample(SERVICES, n)
    order = svc[:]
    rng.shuffle(order)                                    # a hidden valid order; edges only go forward in it
    edges = set()
    while len(edges) < n + 1:
        i, j = sorted(rng.sample(range(n), 2))
        edges.add((order[i], order[j]))                   # order[j] depends on order[i]

    def violations(seq):
        pos = {s: i for i, s in enumerate(seq)}
        return [(a, b) for a, b in sorted(edges) if pos[a] > pos[b]]

    valid = [list(p) for p in itertools.permutations(svc) if not violations(p)]
    if len(valid) != 1:                                   # the dependencies must pin down exactly one order
        return None
    gold = valid[0]
    cands, seen = [gold], {tuple(gold)}
    tries = 0
    while len(cands) < k and tries < 2000:
        tries += 1
        s = gold[:]
        if len(cands) <= k // 2:                          # hard negatives: one adjacent swap
            i = rng.randrange(n - 1)
            s[i], s[i + 1] = s[i + 1], s[i]
        else:
            rng.shuffle(s)
        if tuple(s) not in seen:
            seen.add(tuple(s))
            cands.append(s)
    if len(cands) < k:
        return None
    rng.shuffle(cands)
    keys, feats, cons = [], [], []
    for s in cands:
        v = violations(s)
        keys.append((-len(v),))
        feats.append("valid" if not v else "breaks %d: %s before %s" % (len(v), v[0][1], v[0][0]))
        cons.append({"violations": float(len(v))})
    dep = "; ".join("%s needs %s" % (b, a) for a, b in sorted(edges))
    ctx = (f"Services to deploy: {', '.join(sorted(svc))}. Dependencies (a service must be deployed after everything it "
           f"needs): {dep}.")
    ins = "Choose the deployment order in which every service is deployed after the services it needs."
    p = _finish("rollout", "devops", ins, ctx, [" > ".join(s) for s in cands], feats, keys, "hard", "simulation", {"n": n})
    return _attach(p, cons, {"edges": float(len(edges))})


# ------------------------------------------------------------------------------------------------ batching
def gen_batching(rng: random.Random, k: int, sla: int | None = None, setup: dict | None = None) -> Problem | None:
    s = setup or {"a": rng.choice([4, 6, 8, 10, 15, 20]), "b": rng.choice([1, 2, 3, 4, 5]), "rate": rng.choice([50, 100, 200, 400, 800]),
                  "m0": rng.choice([1000, 2000, 4000]), "m": rng.choice([50, 100, 200, 400]), "M": rng.choice([8000, 12000, 16000, 24000]),
                  "sizes": sorted(rng.sample([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128], k))}
    sla = sla or rng.choice([30, 50, 80, 120, 200])
    keys, feats, cons = [], [], []
    for B in s["sizes"]:
        lat = math.ceil((B - 1) * 1000 / s["rate"]) + s["a"] + s["b"] * B
        cap = int(B * 1000 / (s["a"] + s["b"] * B))
        mem = s["m0"] + s["m"] * B
        why = ("needs %d mb, over the memory limit" % mem if mem > s["M"] else "latency %d ms, over the limit" % lat if lat > sla
               else "capacity %d per second, below the arrival rate" % cap if cap < s["rate"] else None)
        keys.append(None if why else (cap,))
        feats.append(why or "latency %d ms, capacity %d per second" % (lat, cap))
        cons.append({"feasible": float(why is None), "latency": float(lat), "capacity": float(cap)})
    feas = [x for x in keys if x is not None]
    ctx = (f"An inference server takes {s['a']} ms of fixed overhead plus {s['b']} ms per item for each batch. Requests "
           f"arrive at {s['rate']} per second, so a batch of size b waits for b - 1 more arrivals before it runs. Memory "
           f"use is {s['m0']} mb plus {s['m']} mb per item in the batch, and the gpu has {s['M']} mb. The latency limit "
           f"is {sla} ms, counted as the wait to fill the batch plus the batch time.")
    ins = ("Choose the batch size with the highest capacity (items per second) among those that fit in memory, stay "
           "within the latency limit and keep up with the arrival rate.")
    rng2 = random.Random(sum(s["sizes"]) + sla)
    order = list(range(len(keys)))
    rng2.shuffle(order)
    p = _finish("batching", "mlinfra", ins, ctx, ["batch size %d" % s["sizes"][i] for i in order], [feats[i] for i in order],
                [keys[i] for i in order], "hard" if len(feas) >= 3 else "easy", "simulation", {"sla": sla})
    p = _attach(p, [cons[i] for i in order], {"feasible_frac": len(feas) / len(keys), "rate": float(s["rate"]), "sla": float(sla)})
    if p is not None:
        p["_setup"] = s
    return p


def gen_batching_flip(rng: random.Random, k: int):
    base = gen_batching(rng, k)
    if base is None or base["gold"] is None:
        return None
    tighter = [x for x in (20, 30, 50, 80, 120) if x < base["meta"]["sla"]]
    rng.shuffle(tighter)
    for sla in tighter:
        flip = gen_batching(rng, k, sla, base["_setup"])
        if flip is not None and flip["gold"] is not None and flip["gold"] != base["gold"]:
            return base, flip
    return None


def gen_batching_unanswerable(rng: random.Random, k: int) -> Problem | None:
    p = gen_batching(rng, k, sla=rng.choice([5, 8]))
    return p if p is not None and not p["answerable"] else None


# ------------------------------------------------------------------------------------------------ index choice
COLUMNS = ["user id", "status", "created at", "country", "plan", "email", "team id", "kind"]


def gen_indexing(rng: random.Random, k: int) -> Problem | None:
    cols = rng.sample(COLUMNS, rng.choice([3, 4]))
    queries = []
    for _ in range(rng.choice([2, 3, 4])):
        queries.append((frozenset(rng.sample(cols, rng.randint(1, min(3, len(cols))))), rng.choice([1, 2, 5, 10, 20, 50])))
    pool = [t for n in (1, 2, 3) for t in itertools.permutations(cols, n)]

    def total(ix):
        tot = 0
        for filt, freq in queries:
            p = 0
            for c in ix:
                if c in filt:
                    p += 1
                else:
                    break
            tot += freq * (1000 // 10 ** p)
        return tot

    keys = [(-total(ix), -len(ix)) for ix in pool]
    idx = _choose(rng, keys, min(k, len(pool)))
    if idx is None or len(idx) < k:
        return None
    sel = [pool[i] for i in idx]
    q = "; ".join("%d calls filter on %s" % (fr, " and ".join(sorted(f))) for f, fr in queries)
    ctx = (f"A table is queried with equality filters. Per minute: {q}. A query can use an index only through the "
           "index's leading columns that it filters on: with 0 such columns a call costs 1000, with 1 it costs 100, with "
           "2 it costs 10 and with 3 it costs 1.")
    ins = "Choose the single index with the lowest total cost per minute; if tied, the one with fewer columns."
    tots = [total(ix) for ix in sel]
    srt = sorted(tots)
    p = _finish("indexing", "database", ins, ctx, ["index on " + ", ".join(ix) for ix in sel], ["total cost %d" % t for t in tots],
                [keys[i] for i in idx], "hard" if srt[1] - srt[0] <= 0.15 * srt[0] else "easy", "simulation", {})
    return _attach(p, [{"cost": float(t), "columns": float(len(ix))} for t, ix in zip(tots, sel)], {"queries": float(len(queries))})


# ------------------------------------------------------------------------------------------------ eviction (benchmark only)
def gen_eviction(rng: random.Random, k: int) -> Problem | None:
    keys_ = ["k%02d" % i for i in rng.sample(range(10, 100), k + 4)]
    cache, others = keys_[:k], keys_[k:]
    trace = [rng.choice(cache + others) for _ in range(rng.randint(14, 22))]
    nxt = [trace.index(c) + 1 if c in trace else None for c in cache]
    dist = [10_000 if x is None else x for x in nxt]
    if dist.count(max(dist)) != 1:
        return None
    srt = sorted(dist, reverse=True)
    ctx = ("A cache is full and holds: " + " ".join(cache) + ". A new key must be stored, so one key has to be evicted. "
           "The upcoming requests, in order, are: " + " ".join(trace) + ".")
    ins = "Choose the key to evict: the one whose next use is farthest in the future (a key never used again is best)."
    p = _finish("eviction", "state_action", ins, ctx, ["evict " + c for c in cache],
                ["never used again" if x is None else "next used at request %d" % x for x in nxt], [(d,) for d in dist],
                "hard" if srt[0] != 10_000 and srt[0] - srt[1] <= 2 else "easy", "simulation", {})
    return _attach(p, [{"next_use": float(min(d, 40))} for d in dist], {"trace_len": float(len(trace))})


# ------------------------------------------------------------------------------------------------ placement (benchmark only)
def gen_placement(rng: random.Random, k: int) -> Problem | None:
    cpu, mem = rng.choice([2, 4, 8, 16]), rng.choice([4, 8, 16, 32, 64])
    hosts = [{"id": i, "cpu": rng.choice([1, 2, 4, 6, 8, 12, 16, 24, 32]), "mem": rng.choice([2, 4, 8, 16, 24, 32, 64, 96, 128])}
             for i in rng.sample(range(10, 100), k)]
    keys, feats, cons = [], [], []
    for h in hosts:
        ok = h["cpu"] >= cpu and h["mem"] >= mem
        keys.append((-(h["cpu"] - cpu), -(h["mem"] - mem)) if ok else None)
        feats.append("leaves %d cpu and %d gb" % (h["cpu"] - cpu, h["mem"] - mem) if ok else "does not fit")
        cons.append({"feasible": float(ok), "left_cpu": float(h["cpu"] - cpu), "left_mem": float(h["mem"] - mem)})
    feas = [x for x in keys if x is not None]
    ctx = (f"A virtual machine needs {cpu} cpu and {mem} gb of memory. Hosts and their free capacity:\n"
           + "\n".join("host %02d: %d cpu free, %d gb free" % (h["id"], h["cpu"], h["mem"]) for h in hosts))
    ins = ("Choose the host where the machine fits most tightly: the smallest leftover cpu; if tied, the smallest "
           "leftover memory. A host that lacks cpu or memory cannot be used.")
    p = _finish("placement", "cloud", ins, ctx, ["host %02d" % h["id"] for h in hosts], feats, keys,
                "hard" if len(feas) >= 4 else "easy", "simulation", {})
    return _attach(p, cons, {"feasible_frac": len(feas) / len(keys)})


# ------------------------------------------------------------------------------------------------ tetris, three views
def board_features(board) -> dict:
    heights = [H - next((y for y in range(H) if board[y][x]), H) for x in range(W)]
    holes = [sum(1 for y in range(H - heights[x], H) if not board[y][x]) for x in range(W)]
    bump = sum(abs(heights[i] - heights[i + 1]) for i in range(W - 1))
    wells = []
    for x in range(W):
        left = heights[x - 1] if x > 0 else H
        right = heights[x + 1] if x < W - 1 else H
        wells.append(max(0, min(left, right) - heights[x]))
    near = []
    for y in range(H):
        gaps = [x for x in range(W) if not board[y][x]]
        if 0 < len(gaps) <= 2:
            near.append((H - y, gaps))                       # row counted from the bottom, 1 = floor
    return {"heights": heights, "holes": holes, "total": sum(heights), "max": max(heights), "bump": bump, "wells": wells,
            "near": near}


def tetris_views(rng: random.Random, k: int) -> dict[str, Problem] | None:
    """The same Tetris state and candidates as views A (raw), B (structured state) and C (explicit consequence)."""
    board = envs.tetris_board(rng)
    piece = rng.choice("IOTSZJL")
    ps = envs.tetris_placements(board, piece)
    keys = [(p["lines"], -p["holes"], -p["height"]) for p in ps]
    idx = _choose(rng, keys, k) if keys else None
    if idx is None:
        return None
    ps, keys = [ps[i] for i in idx], [keys[i] for i in idx]
    best = max(keys)
    second = max(x for x in keys if x != best)
    sep = next(i for i in range(3) if best[i] != second[i])
    f0 = board_features(board)
    rows = "\n".join(" ".join("#" if c else "." for c in row) for row in board)
    shapes = "\n".join("rot %d cells (column offset, row offset): %s" % (r, " ".join("(%d,%d)" % c for c in s))
                       for r, s in enumerate(PIECES[piece]))
    base = (f"Tetris board, {W} columns numbered 0 to {W - 1} from the left, {H} rows, top row first "
            f"(# filled, . empty):\n{rows}\nNext piece: {piece}.\n{shapes}\n"
            "A placement is a rotation plus the column of the piece's leftmost cell; the piece drops straight down "
            "and every full row is cleared.")
    deep = max(range(W), key=lambda x: f0["wells"][x])
    near = "; ".join("row %d from the floor is missing columns %s" % (r, " ".join(map(str, g))) for r, g in f0["near"]) or "none"
    struct = (f"\nBoard features: column heights {' '.join(map(str, f0['heights']))}; total height {f0['total']}; "
              f"max height {f0['max']}; holes {sum(f0['holes'])} (per column {' '.join(map(str, f0['holes']))}); "
              f"bumpiness {f0['bump']}; deepest well column {deep} depth {f0['wells'][deep]}; nearly full rows: {near}.")
    ins = ("Choose the best placement for the next piece: the one that clears the most lines; if tied, the one that "
           "leaves the fewest holes; if still tied, the lowest total column height.")
    names = ["rot %d col %d" % (p["rot"], p["col"]) for p in ps]
    short = ["lines %d, holes %d, height %d" % (p["lines"], p["holes"], p["height"]) for p in ps]
    cons, full = [], []
    for p in ps:
        f1 = board_features(p["board"])
        dh, dt, db = p["holes"] - sum(f0["holes"]), p["height"] - f0["total"], f1["bump"] - f0["bump"]
        alive = f1["max"] <= H - 2
        cons.append({"lines": float(p["lines"]), "holes": float(p["holes"]), "height": float(p["height"]), "bump": float(f1["bump"])})
        txt = "lines %d holes %+d height %+d bump %+d %s" % (p["lines"], dh, dt, db, "safe" if alive else "near top")
        if k <= 8:                                           # the resulting surface only fits the 48-token candidate cap at k <= 8
            txt += " after " + " ".join(map(str, f1["heights"]))
        full.append(txt)
    state = {"total": float(f0["total"]), "max": float(f0["max"]), "holes": float(sum(f0["holes"])), "bump": float(f0["bump"])}
    out = {}
    for view, ctx, feats in (("A", base, short), ("B", base + struct, short), ("C", base, full)):
        p = _finish("tetris", "state_action", ins, ctx, names, feats, keys, "easy" if sep == 0 else "hard", "simulation",
                    {"piece": piece, "separating_key": sep, "view": view})
        if p is None:
            return None
        out[view] = _attach(p, cons, state)
    return out


GENERATORS = {"quorum": gen_quorum, "retry": gen_retry, "rollout": gen_rollout, "batching": gen_batching, "indexing": gen_indexing,
              "scheduling": lambda r, k: with_cons(envs.gen_scheduling(r, k)), "routing": lambda r, k: with_cons(envs.gen_routing(r, k)),
              "sizing": lambda r, k: with_cons(envs.gen_sizing(r, k))}
BENCH_ONLY = {"eviction": gen_eviction, "placement": gen_placement}
FLIPS = {"quorum": gen_quorum_flip, "retry": gen_retry_flip, "batching": gen_batching_flip,
         "scheduling": lambda r, k: _both(envs.gen_scheduling_flip(r, k)), "routing": lambda r, k: _both(envs.gen_routing_flip(r, k)),
         "sizing": lambda r, k: _both(envs.gen_sizing_flip(r, k))}
UNANSWERABLE = {"quorum": gen_quorum_unanswerable, "retry": gen_retry_unanswerable, "batching": gen_batching_unanswerable,
                "scheduling": lambda r, k: with_cons(envs.gen_scheduling_unanswerable(r, k)),
                "sizing": lambda r, k: with_cons(envs.gen_sizing_unanswerable(r, k))}
TRACK = {"scheduling": "state_action", "routing": "api", "sizing": "cloud"}     # benchmark tracks of the R0 environments
MAX_K = {"scheduling": 16, "rollout": 8, "batching": 12, "retry": 16, "quorum": 32, "indexing": 32, "routing": 32, "sizing": 32,
         "eviction": 8, "placement": 16, "tetris": 16}


def _both(pair):
    return None if pair is None else (with_cons(pair[0]), with_cons(pair[1]))
