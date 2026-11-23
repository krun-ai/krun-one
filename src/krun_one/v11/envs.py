"""Problem generators for the v1.1 Technical Capacity Probe (R0) — the environment / the rules decide the truth.

Every generator returns *problems*: a state, K candidate actions and a verified outcome per candidate::

    {"slice", "track", "k", "instructions", "context",
     "candidates": [{"name", "feat", "value"}],     # value: higher = better, None = infeasible
     "gold": index | None, "answerable": bool, "difficulty": "easy" | "hard",
     "flip": None | {"pair", "role": "base" | "flip", "base_gold"},
     "evidence": {"type": "simulation" | "rule", "verified": True}, "meta": {...}}

``name`` is the raw action as an API caller would send it (it survives v1's bare-label humanization: lowercase,
none of ``_ - . / :``); ``feat`` is the explicit consequence of the action, computed by the environment. The probe
renders the same problem RAW (names only) and FEATURE-ASSISTED (``name: feat``) — see probe.py.

Tracks: ``env`` (state/action/consequence: tetris, scheduling, routing, sizing — truth by simulation / arithmetic)
and ``tech`` (rule-verified technology selection: explicit property tables, exactly one candidate satisfies the
stated requirements; "partial" properties are never allowed to decide an item).
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Any

Problem = dict[str, Any]


def _rank_values(keys: list[tuple | None]) -> list[float | None]:
    """Sortable outcome keys (higher = better, None = infeasible) -> dense rank values (0 = worst feasible)."""
    levels = sorted({k for k in keys if k is not None})
    pos = {k: float(i) for i, k in enumerate(levels)}
    return [None if k is None else pos[k] for k in keys]


def _pick(rng: random.Random, pool: list[int], gold: int, keys: list, k: int) -> list[int] | None:
    """gold + (k-1) negatives: half the hardest (best outcomes), half random; shuffled. None if the pool is too small."""
    neg = [i for i in pool if i != gold]
    if len(neg) < k - 1:
        return None
    neg.sort(key=lambda i: (keys[i] is not None, keys[i] if keys[i] is not None else ()), reverse=True)
    n_hard = (k - 1 + 1) // 2
    chosen = neg[:n_hard] + rng.sample(neg[n_hard:], k - 1 - n_hard)
    out = [gold] + chosen
    rng.shuffle(out)
    return out


def _finish(slice_: str, track: str, instructions: str, context: str, names: list[str], feats: list[str],
            keys: list[tuple | None], difficulty: str, evidence: str, meta: dict | None = None) -> Problem | None:
    """Assemble a problem; None when the best outcome is not unique (ties never become labels)."""
    feas = [k for k in keys if k is not None]
    if len(set(names)) != len(names):
        return None
    if feas:
        best = max(feas)
        if sum(1 for k in feas if k == best) != 1:
            return None
        gold = keys.index(best)
    else:
        gold = None
    vals = _rank_values(keys)
    return {"slice": slice_, "track": track, "k": len(names), "instructions": instructions, "context": context,
            "candidates": [{"name": n, "feat": f, "value": v} for n, f, v in zip(names, feats, vals)],
            "gold": gold, "answerable": gold is not None, "difficulty": difficulty, "flip": None,
            "evidence": {"type": evidence, "verified": True}, "meta": meta or {}}


# ================================================================================================ tetris
W, H = 10, 12
PIECES = {
    "I": [[(0, 0), (1, 0), (2, 0), (3, 0)], [(0, 0), (0, 1), (0, 2), (0, 3)]],
    "O": [[(0, 0), (1, 0), (0, 1), (1, 1)]],
    "T": [[(0, 0), (1, 0), (2, 0), (1, 1)], [(0, 0), (0, 1), (0, 2), (1, 1)], [(1, 0), (0, 1), (1, 1), (2, 1)],
          [(1, 0), (1, 1), (1, 2), (0, 1)]],
    "S": [[(1, 0), (2, 0), (0, 1), (1, 1)], [(0, 0), (0, 1), (1, 1), (1, 2)]],
    "Z": [[(0, 0), (1, 0), (1, 1), (2, 1)], [(1, 0), (1, 1), (0, 1), (0, 2)]],
    "J": [[(0, 0), (0, 1), (1, 1), (2, 1)], [(0, 0), (1, 0), (0, 1), (0, 2)], [(0, 0), (1, 0), (2, 0), (2, 1)],
          [(1, 0), (1, 1), (1, 2), (0, 2)]],
    "L": [[(2, 0), (0, 1), (1, 1), (2, 1)], [(0, 0), (0, 1), (0, 2), (1, 2)], [(0, 0), (1, 0), (2, 0), (0, 1)],
          [(0, 0), (1, 0), (1, 1), (1, 2)]],
}


def tetris_drop(board: list[list[int]], shape: list[tuple[int, int]], x0: int):
    """Drop ``shape`` with its leftmost cell at column x0. Returns (new_board, lines_cleared) or None (does not fit)."""
    if x0 < 0 or x0 + max(dx for dx, _ in shape) >= W:
        return None

    def ok(y0):
        return all(y0 + dy < H and not board[y0 + dy][x0 + dx] for dx, dy in shape)

    if not ok(0):
        return None
    y0 = 0
    while ok(y0 + 1):
        y0 += 1
    nb = [row[:] for row in board]
    for dx, dy in shape:
        nb[y0 + dy][x0 + dx] = 1
    kept = [row for row in nb if not all(row)]
    lines = H - len(kept)
    return [[0] * W for _ in range(lines)] + kept, lines


def tetris_stats(board: list[list[int]]) -> tuple[int, int, int]:
    """(holes, total column height, max column height)."""
    holes = total = mx = 0
    for x in range(W):
        top = next((y for y in range(H) if board[y][x]), H)
        h = H - top
        total += h
        mx = max(mx, h)
        holes += sum(1 for y in range(top, H) if not board[y][x])
    return holes, total, mx


def tetris_placements(board, piece: str) -> list[dict]:
    out = []
    for r, shape in enumerate(PIECES[piece]):
        for x0 in range(W):
            got = tetris_drop(board, shape, x0)
            if got is None:
                continue
            nb, lines = got
            holes, total, _ = tetris_stats(nb)
            out.append({"rot": r, "col": x0, "lines": lines, "holes": holes, "height": total, "board": nb})
    return out


def tetris_board(rng: random.Random) -> list[list[int]]:
    """A board reached by actual play (a noisy greedy policy), so stacks, gaps and near-full rows look real."""
    board = [[0] * W for _ in range(H)]
    for _ in range(rng.randint(6, 16)):
        ps = tetris_placements(board, rng.choice("IOTSZJL"))
        if not ps:
            break
        p = rng.choice(ps) if rng.random() < 0.3 else min(ps, key=lambda p: (p["holes"], p["height"], rng.random()))
        if tetris_stats(p["board"])[2] > H - 4:
            break
        board = p["board"]
    return board


def gen_tetris(rng: random.Random, k: int) -> Problem | None:
    board = tetris_board(rng)
    piece = rng.choice("TJL" if k > 16 else "IOTSZJL")
    ps = tetris_placements(board, piece)
    keys = [(p["lines"], -p["holes"], -p["height"]) for p in ps]
    if not keys:
        return None
    best = max(keys)
    if keys.count(best) != 1:
        return None
    idx = _pick(rng, list(range(len(ps))), keys.index(best), keys, k)
    if idx is None:
        return None
    ps, keys = [ps[i] for i in idx], [keys[i] for i in idx]
    second = max(x for x in keys if x != best)
    sep = next(i for i in range(3) if best[i] != second[i])
    rows = "\n".join(" ".join("#" if c else "." for c in row) for row in board)
    shapes = "\n".join("rot %d cells (column offset, row offset): %s" % (r, " ".join("(%d,%d)" % c for c in s))
                       for r, s in enumerate(PIECES[piece]))
    ctx = (f"Tetris board, {W} columns numbered 0 to {W - 1} from the left, {H} rows, top row first "
           f"(# filled, . empty):\n{rows}\nNext piece: {piece}.\n{shapes}\n"
           "A placement is a rotation plus the column of the piece's leftmost cell; the piece drops straight down "
           "and every full row is cleared.")
    ins = ("Choose the best placement for the next piece: the one that clears the most lines; if tied, the one that "
           "leaves the fewest holes; if still tied, the lowest total column height.")
    return _finish("tetris", "env", ins, ctx, ["rot %d col %d" % (p["rot"], p["col"]) for p in ps],
                   ["lines %d, holes %d, height %d" % (p["lines"], p["holes"], p["height"]) for p in ps], keys,
                   "easy" if sep == 0 else "hard", "simulation", {"piece": piece, "separating_key": sep})


# ================================================================================================ scheduling
def _sched_problem(nodes: list[dict], work: int, mem: int, extra: dict | None = None) -> Problem | None:
    keys, feats = [], []
    for n in nodes:
        if n["mem"] < mem:
            keys.append(None)
            feats.append("cannot run, only %d gb" % n["mem"])
        else:
            fin = n["wait"] + math.ceil(work / n["speed"])
            keys.append((-fin,))
            feats.append("done at minute %d" % fin)
    feas = sorted((k for k in keys if k is not None), reverse=True)
    hard = len(feas) > 1 and feas[0][0] - feas[1][0] <= 2
    ctx = (f"Job: {work} work units, needs {mem} gb of memory.\n"
           "Nodes (wait = minutes until the node is free, speed = work units per minute, mem = gb of memory):\n"
           + "\n".join("node %02d: wait %d, speed %d, mem %d" % (n["id"], n["wait"], n["speed"], n["mem"]) for n in nodes))
    ins = "Choose the node that finishes the job earliest. A node can only run the job if it has enough memory."
    return _finish("scheduling", "env", ins, ctx, ["node %02d" % n["id"] for n in nodes], feats, keys,
                   "hard" if hard else "easy", "simulation", {"work": work, "mem": mem, **(extra or {})})


def _sched_nodes(rng: random.Random, k: int) -> list[dict]:
    return [{"id": i, "wait": rng.randint(0, 60), "speed": rng.choice([1, 2, 3, 4, 5, 6, 8]),
             "mem": rng.choice([4, 8, 8, 16, 16, 32, 64])} for i in rng.sample(range(10, 100), k)]


def gen_scheduling(rng: random.Random, k: int) -> Problem | None:
    return _sched_problem(_sched_nodes(rng, k), rng.randint(24, 240), rng.choice([4, 8, 8, 16, 16, 32]))


def gen_scheduling_unanswerable(rng: random.Random, k: int) -> Problem | None:
    return _sched_problem(_sched_nodes(rng, k), rng.randint(24, 240), 128)


def gen_scheduling_flip(rng: random.Random, k: int) -> tuple[Problem, Problem] | None:
    """Same nodes and job; the flip raises the memory requirement so the base winner can no longer run it."""
    nodes, work = _sched_nodes(rng, k), rng.randint(24, 240)
    base = _sched_problem(nodes, work, rng.choice([4, 8]))
    if base is None or base["gold"] is None:
        return None
    gmem = nodes[base["gold"]]["mem"]
    bigger = [m for m in (8, 16, 32, 64) if m > gmem]
    if not bigger:
        return None
    flip = _sched_problem(nodes, work, bigger[0])
    if flip is None or flip["gold"] is None:
        return None
    return base, flip


# ================================================================================================ routing
SITES = ["gru", "iad", "fra", "nrt", "syd", "lhr", "sin", "jnb", "bom", "scl", "yyz", "cdg", "dfw", "sea", "ams", "hkg"]


def _routing_graph(rng: random.Random, n: int, n_edges: int):
    sites = rng.sample(SITES, n)
    edges: dict[frozenset, int] = {}
    order = sites[:]
    rng.shuffle(order)
    for i in range(1, n):   # random spanning tree, then extra links
        edges[frozenset((order[i], rng.choice(order[:i])))] = rng.randint(5, 60)
    while len(edges) < n_edges:
        a, b = rng.sample(sites, 2)
        edges.setdefault(frozenset((a, b)), rng.randint(5, 60))
    return sites, edges


def _routing_paths(sites, edges, s, t, max_hops: int) -> list[tuple[list[str], int]]:
    adj: dict[str, list[tuple[str, int]]] = {x: [] for x in sites}
    for e, w in edges.items():
        a, b = tuple(e)
        adj[a].append((b, w))
        adj[b].append((a, w))
    out = []

    def walk(path, cost):
        if path[-1] == t:
            out.append((path[:], cost))
            return
        if len(path) > max_hops:
            return
        for nxt, w in adj[path[-1]]:
            if nxt not in path:
                path.append(nxt)
                walk(path, cost + w)
                path.pop()

    walk([s], 0)
    return out


def _routing_problem(sites, edges, s, t, paths, down: str | None) -> Problem | None:
    keys, feats = [], []
    for p, c in paths:
        if down and down in p:
            keys.append(None)
            feats.append("unusable, passes through %s" % down)
        else:
            keys.append((-c,))
            feats.append("total %d ms" % c)
    feas = sorted((k for k in keys if k is not None), reverse=True)
    hard = len(feas) > 1 and feas[0][0] - feas[1][0] <= 5
    links = sorted((sorted(e), w) for e, w in edges.items())
    ctx = ("Network links between sites (latency in ms, usable in both directions):\n"
           + "\n".join("%s - %s %d" % (a, b, w) for (a, b), w in links)
           + f"\nTraffic must go from {s} to {t}." + (f" Site {down} is down." if down else ""))
    ins = (f"Choose the route from {s} to {t} with the lowest total latency."
           + (" A route through a site that is down cannot be used." if down else ""))
    return _finish("routing", "env", ins, ctx, [" > ".join(p) for p, _ in paths], feats, keys,
                   "hard" if hard else "easy", "simulation", {"down": down})


def _routing_setup(rng: random.Random, k: int):
    n = 7 if k <= 8 else (8 if k <= 16 else 9)
    sites, edges = _routing_graph(rng, n, {7: 12, 8: 15, 9: 19}[n])
    s, t = rng.sample(sites, 2)
    paths = _routing_paths(sites, edges, s, t, max_hops=4 if k <= 16 else 5)
    costs = [(-c,) for _, c in paths]
    if not costs or costs.count(max(costs)) != 1:
        return None
    idx = _pick(rng, list(range(len(paths))), costs.index(max(costs)), costs, k)
    if idx is None:
        return None
    return sites, edges, s, t, [paths[i] for i in idx]


def gen_routing(rng: random.Random, k: int) -> Problem | None:
    got = _routing_setup(rng, k)
    return None if got is None else _routing_problem(*got, down=None)


def gen_routing_flip(rng: random.Random, k: int) -> tuple[Problem, Problem] | None:
    got = _routing_setup(rng, k)
    if got is None:
        return None
    sites, edges, s, t, paths = got
    base = _routing_problem(sites, edges, s, t, paths, None)
    if base is None:
        return None
    mid = paths[base["gold"]][0][1:-1]
    if not mid:
        return None
    flip = _routing_problem(sites, edges, s, t, paths, rng.choice(mid))
    if flip is None or flip["gold"] is None:
        return None
    return base, flip


# ================================================================================================ sizing
def _sizing_catalog(rng: random.Random, k: int) -> list[dict] | None:
    seen, cat = set(), []
    for _ in range(k * 20):
        v, r, g = rng.choice([2, 4, 8, 16, 32, 64]), rng.choice([4, 8, 16, 32, 64, 128, 256]), rng.random() < 0.25
        price = int(round((v * 9 + r * 2.2 + (260 if g else 0)) * rng.uniform(0.7, 1.4)))
        if (v, r, g) in seen or any(c["price"] == price for c in cat):
            continue
        seen.add((v, r, g))
        cat.append({"vcpu": v, "ram": r, "gpu": g, "price": price})
        if len(cat) == k:
            return cat
    return None


def _sizing_problem(cat: list[dict], need_v: int, need_r: int, need_g: bool, workload: str) -> Problem | None:
    keys, feats = [], []
    for c in cat:
        why = ("too few vcpus" if c["vcpu"] < need_v else "too little ram" if c["ram"] < need_r
               else "no gpu" if need_g and not c["gpu"] else None)
        keys.append(None if why else (-c["price"],))
        feats.append(why or "meets every requirement")
    feas = sorted((k for k in keys if k is not None), reverse=True)
    hard = len(feas) >= 4 and (feas[0][0] - feas[1][0]) <= 0.1 * -feas[0][0]
    ctx = (f"{workload} It needs at least {need_v} vcpus and {need_r} gb of ram"
           + (", and it needs a gpu." if need_g else "; a gpu is not needed."))
    ins = "Choose the cheapest instance type that meets every stated requirement."
    names = ["%d vcpu, %d gb ram, %s, %d usd per month" % (c["vcpu"], c["ram"], "gpu" if c["gpu"] else "no gpu", c["price"])
             for c in cat]
    return _finish("sizing", "env", ins, ctx, names, feats, keys, "hard" if hard else "easy", "rule",
                   {"need": [need_v, need_r, need_g]})


WORKLOADS = ["A team is sizing the server for a reporting service.", "An image-processing worker is being deployed.",
             "A new search indexer needs a machine.", "The staging copy of an api server needs an instance.",
             "A nightly batch job needs a host.", "A video transcoding service is being provisioned.",
             "A recommendation model server is being set up.", "A build agent for continuous integration is needed."]


def gen_sizing(rng: random.Random, k: int) -> Problem | None:
    cat = _sizing_catalog(rng, k)
    if cat is None:
        return None
    p = _sizing_problem(cat, rng.choice([2, 3, 4, 6, 8, 12, 16]), rng.choice([4, 6, 8, 12, 16, 24, 32, 48, 64]),
                        rng.random() < 0.2, rng.choice(WORKLOADS))
    return p if p and p["gold"] is not None else None


def gen_sizing_unanswerable(rng: random.Random, k: int) -> Problem | None:
    cat = _sizing_catalog(rng, k)
    return None if cat is None else _sizing_problem(cat, 96, rng.choice([64, 128]), False, rng.choice(WORKLOADS))


def gen_sizing_flip(rng: random.Random, k: int) -> tuple[Problem, Problem] | None:
    """One added / tightened requirement (more RAM, or a GPU) that the base winner does not meet."""
    cat = _sizing_catalog(rng, k)
    if cat is None:
        return None
    v, r, w = rng.choice([2, 3, 4, 6, 8]), rng.choice([4, 6, 8, 12, 16]), rng.choice(WORKLOADS)
    base = _sizing_problem(cat, v, r, False, w)
    if base is None or base["gold"] is None:
        return None
    g = cat[base["gold"]]
    if not g["gpu"] and rng.random() < 0.5:
        flip = _sizing_problem(cat, v, r, True, w)
    else:
        bigger = [x for x in (8, 12, 16, 24, 32, 48, 64, 128) if x > g["ram"]]
        flip = _sizing_problem(cat, v, bigger[0], False, w) if bigger else None
    if flip is None or flip["gold"] is None:
        return None
    return base, flip


# ================================================================================================ tech selection
# Property tables: Y = clearly yes, N = clearly no, P = partial / debatable. A candidate with P on a stated
# requirement is REMOVED from that problem's candidate list, so no label ever depends on a debatable fact.
# overhead: 0 = nothing to operate / trivial to implement, 1 = one server / moderate, 2 = a cluster / complex.
OVERHEAD = {0: "minimal", 1: "moderate", 2: "high"}
TECH: dict[str, dict[str, Any]] = {
    "datastore": {
        "noun": "data store",
        "props": {
            "joins": (["ad-hoc sql joins across many tables", "reports that join several tables with sql"], "sql joins"),
            "acid": (["multi-row acid transactions", "all-or-nothing transactions that span several rows"], "acid transactions"),
            "network": (["many application servers connecting to it over the network",
                         "concurrent access from dozens of separate backend hosts"], "network clients"),
            "horiz": (["write throughput that keeps growing by adding nodes",
                       "horizontal scaling of writes across machines"], "horizontal write scaling"),
            "inmem": (["sub-millisecond reads served from memory", "in-memory access with microsecond latency"], "in-memory speed"),
            "fulltext": (["relevance-ranked full-text search", "typo-tolerant keyword search ranked by relevance"], "full-text search"),
            "graph": (["multi-hop relationship traversals such as friends of friends as first-class queries",
                       "queries that walk many hops across connected entities"], "graph traversal"),
            "columnar": (["aggregations over billions of rows in seconds", "columnar analytics over very large fact tables"],
                         "columnar analytics"),
        },
        "options": {
            "sqlite": (0, dict(joins="Y", acid="Y", network="N", horiz="N", inmem="N", fulltext="P", graph="N", columnar="N")),
            "postgresql": (1, dict(joins="Y", acid="Y", network="Y", horiz="N", inmem="N", fulltext="P", graph="P", columnar="N")),
            "cockroachdb": (2, dict(joins="Y", acid="Y", network="Y", horiz="Y", inmem="N", fulltext="N", graph="N", columnar="N")),
            "dynamodb": (0, dict(joins="N", acid="P", network="Y", horiz="Y", inmem="N", fulltext="N", graph="N", columnar="N")),
            "cassandra": (2, dict(joins="N", acid="N", network="Y", horiz="Y", inmem="N", fulltext="N", graph="N", columnar="N")),
            "redis": (1, dict(joins="N", acid="P", network="Y", horiz="P", inmem="Y", fulltext="P", graph="N", columnar="N")),
            "elasticsearch": (2, dict(joins="N", acid="N", network="Y", horiz="Y", inmem="N", fulltext="Y", graph="N", columnar="P")),
            "clickhouse": (1, dict(joins="P", acid="N", network="Y", horiz="P", inmem="N", fulltext="P", graph="N", columnar="Y")),
            "neo4j": (1, dict(joins="N", acid="Y", network="Y", horiz="N", inmem="N", fulltext="P", graph="Y", columnar="N")),
        },
        "fillers": ["security", "scaling"], "max_req": 3,
    },
    "messaging": {
        "noun": "messaging system",
        "props": {
            "durable": (["messages that survive a broker restart until they are consumed",
                         "no message loss if the broker crashes before delivery"], "durable messages"),
            "replay": (["consumers that can re-read messages from days ago", "replaying the full event history for a new consumer"],
                       "replayable log"),
            "routing": (["per-message routing rules by topic pattern and headers",
                         "flexible exchange-style routing of each message to different queues"], "flexible routing"),
            "ordered": (["strict per-key ordering at very high throughput", "ordered partitions handling millions of events per second"],
                        "high-throughput ordered partitions"),
            "iot": (["a lightweight protocol for battery-powered devices on unreliable networks",
                     "tiny sensors that publish over flaky mobile connections"], "lightweight iot protocol"),
        },
        "options": {
            "kafka": (2, dict(durable="Y", replay="Y", routing="N", ordered="Y", iot="N")),
            "rabbitmq": (1, dict(durable="Y", replay="P", routing="Y", ordered="P", iot="P")),
            "amazon sqs": (0, dict(durable="Y", replay="N", routing="N", ordered="P", iot="N")),
            "nats core": (1, dict(durable="N", replay="N", routing="P", ordered="N", iot="P")),
            "redis pub sub": (1, dict(durable="N", replay="N", routing="P", ordered="N", iot="N")),
            "mqtt broker": (1, dict(durable="P", replay="N", routing="P", ordered="N", iot="Y")),
        },
        "fillers": ["security", "structure"], "max_req": 2,
    },
    "structure": {
        "noun": "data structure",
        "props": {
            "lookup": (["constant-time lookup by exact key", "o(1) average access by exact key"], "o(1) exact lookup"),
            "range": (["in-order iteration and range queries", "listing all keys between two bounds in sorted order"], "ordered range queries"),
            "insert": (["frequent inserts and deletes without rebuilding", "cheap insertion and removal as the data changes constantly"],
                       "fast inserts and deletes"),
            "minpop": (["repeatedly removing the smallest element", "always extracting the lowest-priority item next"], "extract minimum"),
            "prefix": (["autocomplete by prefix", "finding every key that starts with a given prefix"], "prefix search"),
            "approx": (["a membership test in a few bits per item where rare false positives are fine",
                        "checking whether an item was probably seen, using almost no memory"], "approximate membership"),
            "evict": (["bounded memory with automatic eviction of the least recently used entries",
                       "a fixed-size store that drops whatever was used longest ago"], "lru eviction"),
        },
        "options": {
            "hash table": (0, dict(lookup="Y", range="N", insert="Y", minpop="N", prefix="N", approx="N", evict="N")),
            "sorted array": (0, dict(lookup="N", range="Y", insert="N", minpop="P", prefix="P", approx="N", evict="N")),
            "binary heap": (0, dict(lookup="N", range="N", insert="Y", minpop="Y", prefix="N", approx="N", evict="N")),
            "balanced search tree": (1, dict(lookup="N", range="Y", insert="Y", minpop="Y", prefix="P", approx="N", evict="N")),
            "trie": (1, dict(lookup="P", range="P", insert="Y", minpop="N", prefix="Y", approx="N", evict="N")),
            "bloom filter": (0, dict(lookup="N", range="N", insert="P", minpop="N", prefix="N", approx="Y", evict="N")),
            "lru cache": (1, dict(lookup="Y", range="N", insert="Y", minpop="N", prefix="N", approx="N", evict="Y")),
            "fifo queue": (0, dict(lookup="N", range="N", insert="P", minpop="N", prefix="N", approx="N", evict="N")),
        },
        "fillers": ["security", "messaging"], "max_req": 3,
    },
    "security": {
        "noun": "security mechanism",
        "props": {
            "pw": (["storing user passwords so that a leaked database resists offline guessing",
                    "a slow salted way to keep login passwords"], "slow salted password hashing"),
            "fingerprint": (["a fast fixed-size fingerprint of large files to spot duplicates, with no secret involved",
                             "a quick checksum-like identifier for file contents"], "fast content fingerprint"),
            "mac": (["two services that share a secret key checking that a message was not modified",
                     "tamper detection on webhooks using a shared secret"], "shared-key message authentication"),
            "pubverify": (["anyone verifying who published a release using only a public key",
                           "proof of origin that others can check but cannot forge"], "public-key signatures"),
            "atrest": (["stored records that are unreadable without a key but that the key holder can read back",
                        "reversible protection of data at rest"], "encryption of stored data"),
            "transit": (["protecting traffic between browser and server from eavesdropping",
                         "an encrypted channel for a network connection"], "encrypted transport"),
            "textsafe": (["representing binary bytes as plain text inside a json field, with no secrecy needed",
                          "a text-safe representation of raw bytes"], "binary-to-text encoding"),
        },
        "options": {
            "argon2id password hashing": (1, dict(pw="Y", fingerprint="N", mac="N", pubverify="N", atrest="N", transit="N", textsafe="N")),
            "sha 256 digest": (0, dict(pw="N", fingerprint="Y", mac="N", pubverify="N", atrest="N", transit="N", textsafe="N")),
            "hmac sha256": (0, dict(pw="N", fingerprint="P", mac="Y", pubverify="N", atrest="N", transit="N", textsafe="N")),
            "ed25519 signature": (1, dict(pw="N", fingerprint="N", mac="P", pubverify="Y", atrest="N", transit="N", textsafe="N")),
            "aes 256 gcm encryption": (1, dict(pw="N", fingerprint="N", mac="P", pubverify="N", atrest="Y", transit="P", textsafe="N")),
            "tls": (1, dict(pw="N", fingerprint="N", mac="N", pubverify="N", atrest="N", transit="Y", textsafe="N")),
            "base64 encoding": (0, dict(pw="N", fingerprint="N", mac="N", pubverify="N", atrest="N", transit="N", textsafe="Y")),
        },
        "fillers": ["structure", "messaging"], "max_req": 1,
    },
    "scaling": {
        "noun": "scaling pattern",
        "props": {
            "repeat": (["the same expensive queries repeated constantly, where results a minute old are acceptable",
                        "identical read requests recomputed over and over although slightly stale answers are fine"], "reuses repeated results"),
            "static": (["static assets that load slowly for users on other continents",
                        "images and scripts served too slowly to distant visitors"], "serves static files near users"),
            "writes": (["write volume beyond what a single primary database can accept",
                        "more inserts per second than one database primary can handle"], "splits writes across databases"),
            "burst": (["bursts of background jobs that overwhelm workers, although the work may finish minutes later",
                       "spiky asynchronous work that does not need an immediate result"], "buffers work for later"),
            "abuse": (["a few clients flooding the api and starving everyone else",
                       "individual callers sending far more requests than their fair share"], "caps requests per client"),
            "cascade": (["a slow downstream dependency that makes callers hang until failures cascade",
                         "calls to a failing service piling up and taking the caller down with it"], "stops calling a failing dependency"),
            "cpu": (["stateless web servers that hit cpu limits at peak hours and sit idle at night",
                     "request load on stateless servers that swings widely during the day"], "adds and removes servers with load"),
            "fresh": (["arbitrary read queries that overload the primary although results must be nearly real time",
                       "heavy varied read traffic that needs fresh data"], "extra database copies for reads"),
        },
        "options": {
            "cache with ttl": (1, dict(repeat="Y", static="P", writes="N", burst="N", abuse="N", cascade="N", cpu="N", fresh="P")),
            "cdn": (1, dict(repeat="P", static="Y", writes="N", burst="N", abuse="N", cascade="N", cpu="N", fresh="N")),
            "database sharding": (1, dict(repeat="N", static="N", writes="Y", burst="N", abuse="N", cascade="N", cpu="N", fresh="P")),
            "message queue": (1, dict(repeat="N", static="N", writes="P", burst="Y", abuse="N", cascade="P", cpu="N", fresh="N")),
            "rate limiting": (1, dict(repeat="N", static="N", writes="N", burst="P", abuse="Y", cascade="N", cpu="N", fresh="N")),
            "circuit breaker": (1, dict(repeat="N", static="N", writes="N", burst="N", abuse="N", cascade="Y", cpu="N", fresh="N")),
            "autoscaling": (1, dict(repeat="N", static="N", writes="N", burst="P", abuse="N", cascade="N", cpu="Y", fresh="N")),
            "read replicas": (1, dict(repeat="P", static="N", writes="N", burst="N", abuse="N", cascade="N", cpu="N", fresh="Y")),
        },
        "fillers": ["security", "structure"], "max_req": 1,
    },
}
INTROS = ["A team is building the backend of a food delivery app.", "A startup is designing an internal analytics tool.",
          "An engineer is reworking part of a payments platform.", "A logistics company is modernizing its tracking system.",
          "A small team maintains a multiplayer game backend.", "A hospital software group is planning a new service.",
          "A developer is adding a feature to an e-commerce site.", "A fintech team is reviewing a design proposal.",
          "A media company is rebuilding its content pipeline.", "An open-source project is choosing its building blocks.",
          "A telecom operator is replacing a legacy component.", "A research lab is setting up shared infrastructure."]


def tech_requirement_sets(cat: str) -> list[dict]:
    """Every requirement set (1..max_req properties) of a category with its rule-derived outcome.

    kept = options without P on any required property; satisfiers = kept options with Y on all of them;
    gold = the unique satisfier of lowest overhead (sets with a tie are not usable); no satisfier -> unanswerable."""
    c = TECH[cat]
    out = []
    for n in range(1, c["max_req"] + 1):
        for req in itertools.combinations(c["props"], n):
            kept = [o for o, (_, pr) in c["options"].items() if all(pr[r] != "P" for r in req)]
            sat = [o for o in kept if all(c["options"][o][1][r] == "Y" for r in req)]
            if len(kept) < 3:
                continue
            if sat:
                low = min(c["options"][o][0] for o in sat)
                best = [o for o in sat if c["options"][o][0] == low]
                if len(best) != 1:
                    continue
                gold = best[0]
            else:
                gold = None
            out.append({"cat": cat, "req": req, "kept": kept, "sat": sat, "gold": gold})
    return out


def tech_feat(cat: str, opt: str) -> str:
    ov, pr = TECH[cat]["options"][opt]
    tags = [TECH[cat]["props"][p][1] for p, v in pr.items() if v == "Y"]
    return "%s; overhead %s" % (", ".join(tags) if tags else "none of the listed capabilities", OVERHEAD[ov])


def _tech_options(rng: random.Random, rs: dict, k: int, kept: list[str] | None = None) -> list[tuple[str, str]] | None:
    """k candidates: the category's usable options (always the gold) + options of unrelated categories as fillers."""
    cat, c = rs["cat"], TECH[rs["cat"]]
    opts = [(cat, o) for o in (kept or rs["kept"])]
    fill = [(fc, o) for fc in c["fillers"] for o in TECH[fc]["options"]]
    rng.shuffle(fill)
    if len(opts) > k:
        return None
    opts = opts + fill[:k - len(opts)]
    if len(opts) < k:
        return None
    rng.shuffle(opts)
    return opts


def tech_problem(rng: random.Random, rs: dict, k: int, opts: list[tuple[str, str]] | None = None,
                 intro: str | None = None, variants: dict[str, int] | None = None) -> Problem | None:
    cat, c = rs["cat"], TECH[rs["cat"]]
    opts = opts or _tech_options(rng, rs, k)
    if opts is None:
        return None
    keys = []
    for oc, o in opts:
        if oc != cat:
            keys.append((0, 0, 0))                      # another kind of thing: satisfies nothing
            continue
        ov, pr = c["options"][o]
        met = sum(pr[r] == "Y" for r in rs["req"])
        full = met == len(rs["req"])
        keys.append((1, 0, -ov) if full else (0, met, 0))
    if rs["gold"] is None:
        keys = [None] * len(opts)
    variants = variants or {r: rng.randrange(2) for r in rs["req"]}
    reqs = [c["props"][r][0][variants[r]] for r in rs["req"]]
    ctx = "%s They need a %s. Requirements: %s." % (intro or rng.choice(INTROS), c["noun"], "; ".join(reqs))
    ins = (f"Choose the {c['noun']} that satisfies every requirement. If several do, choose the one with the lowest "
           "overhead (the simplest to run or implement).")
    hard = len(rs["sat"]) > 1 or len(rs["req"]) >= 2
    return _finish("techsel", "tech", ins, ctx, [o for _, o in opts], [tech_feat(oc, o) for oc, o in opts], keys,
                   "hard" if hard else "easy", "rule",
                   {"category": cat, "requirements": list(rs["req"]), "gold_name": rs["gold"],
                    "n_same_category": sum(oc == cat for oc, _ in opts)})


def tech_pair(rng: random.Random, base: dict, flip: dict, k: int) -> tuple[Problem, Problem] | None:
    """A constraint-flip pair: identical intro, wording, candidates and order; the flip adds ONE requirement."""
    opts = _tech_options(rng, flip, k)             # flip's usable options are a subset of the base's
    if opts is None:
        return None
    intro = rng.choice(INTROS)
    variants = {r: rng.randrange(2) for r in flip["req"]}
    extra = next(r for r in flip["req"] if r not in base["req"])
    b = dict(base, req=tuple(base["req"]))
    f = dict(flip, req=tuple(base["req"]) + (extra,))     # the added requirement comes last in the text
    pb = tech_problem(rng, b, k, opts, intro, variants)
    pf = tech_problem(rng, f, k, opts, intro, variants)
    if pb is None or pf is None or pb["gold"] is None or pf["gold"] is None:
        return None
    return pb, pf


def tech_flip_pairs(cat: str) -> list[tuple[dict, dict]]:
    """(base set, base + one requirement) where the rule-derived answer changes and the base answer stays offered."""
    sets = {s["req"]: s for s in tech_requirement_sets(cat)}
    out = []
    for req, base in sets.items():
        if base["gold"] is None:
            continue
        for extra in TECH[cat]["props"]:
            if extra in req:
                continue
            flip = sets.get(tuple(p for p in TECH[cat]["props"] if p in req or p == extra))
            if flip and flip["gold"] and flip["gold"] != base["gold"] and base["gold"] in flip["kept"]:
                out.append((base, flip))
    return out


GENERATORS = {"tetris": gen_tetris, "scheduling": gen_scheduling, "routing": gen_routing, "sizing": gen_sizing}
UNANSWERABLE = {"scheduling": gen_scheduling_unanswerable, "sizing": gen_sizing_unanswerable}
FLIPS = {"scheduling": gen_scheduling_flip, "routing": gen_routing_flip, "sizing": gen_sizing_flip}
