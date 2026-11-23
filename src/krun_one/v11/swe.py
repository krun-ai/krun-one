"""SWE-smith rows -> intermediate engineering decisions (never "write the patch").

A SWE-smith row is (mirror repo, bug-introducing ``patch``, LLM-written ``problem_statement``, ``FAIL_TO_PASS``,
``PASS_TO_PASS``). The reverse of the patch is the fix and the failing tests verify it by execution. From that:

  file        which source file has to change        gold = the file the patch touches            tier A
  subsystem   which package                          gold = the package of that file              tier A
  tests       which test module fails because of it  gold = the module of FAIL_TO_PASS            tier A
  function    which function in the file             gold = enclosing ``def`` of the first change  tier B (parsed from the diff)
  involved    noul: "fixing this needs a change in X"                                              tier A
  files       multi: every file that has to change (multi-file patches)                            tier A
  file_ood    the right file is not offered          none-of-the-above                             tier A

Not derivable without a judge model, so NOT built: repair strategy, API / algorithm choice, responsible dependency.

Candidates are other real files / packages / test modules / functions of the SAME repository at the mirror commit
(file tree from the mirror repo). Hard negatives are the nearest ones: same directory, then same top-level package.
Candidate text: a short readable handle + the real path as description, because bare ids are humanized by the v1
serializer ("a/b_c.py" would become "a b c py").
"""

from __future__ import annotations

import ast
import random
import re
from collections import defaultdict
from typing import Any

FILE_RE = re.compile(r"^diff --git a/(\S+) b/", re.M)
DEF_RE = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(|^\s*class\s+([A-Za-z_]\w*)")
TEST_RE = re.compile(r"(^|/)(tests?|testing|test_utils)/|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$")
STRIP_RE = re.compile(r"[_\-./:]+")


def touched_files(patch: str) -> list[str]:
    return list(dict.fromkeys(FILE_RE.findall(patch)))


def is_test_path(path: str) -> bool:
    return bool(TEST_RE.search(path))


def parse_list(s: Any) -> list[str]:
    if isinstance(s, (list, tuple)):
        return [str(x) for x in s]
    try:
        v = ast.literal_eval(s)
        return [str(x) for x in v] if isinstance(v, (list, tuple)) else []
    except (ValueError, SyntaxError):
        return []


def test_module(test_id: str) -> str:
    return test_id.split("::")[0]


def changed_function(patch: str) -> str | None:
    """Enclosing ``def`` of the first changed line of a single-file patch (hunk header or context lines above it)."""
    cur, found = None, []
    for line in patch.split("\n"):
        if line.startswith("@@"):
            m = DEF_RE.match(line.split("@@", 2)[-1])
            cur = (m.group(1) if m and m.group(1) else None)
            first = True
            continue
        if line.startswith(("+++", "---", "diff --git", "index ")):
            continue
        body = line[1:] if line[:1] in "+- " else line
        m = DEF_RE.match(body)
        if line.startswith((" ", "-")) and m and m.group(1):
            cur = m.group(1)
        if line.startswith(("+", "-")) and cur and (not found or found[-1] != cur):
            found.append(cur)
    names = list(dict.fromkeys(found))
    return names[0] if len(names) == 1 else None


def handle(path: str, taken: set[str]) -> str:
    """Short readable id for a path: 'web application', made unique with parent directories when needed."""
    parts = [STRIP_RE.sub(" ", p).strip().lower() for p in path.removesuffix(".py").split("/") if p]
    for n in range(1, len(parts) + 1):
        h = " ".join(" ".join(parts[-n:]).split())
        if h and h not in taken:
            taken.add(h)
            return h
    h = STRIP_RE.sub(" ", path).lower() + " %d" % len(taken)
    taken.add(h)
    return h


def near(gold: str, pool: list[str], rng: random.Random, n: int) -> tuple[list[str], int]:
    """n negatives for ``gold``: same directory first, then same top-level package, then the rest. Returns the list
    and how many of them are 'near' (same directory or package)."""
    d, top = gold.rsplit("/", 1)[0] if "/" in gold else "", gold.split("/")[0]
    same_dir = [p for p in pool if p != gold and (p.rsplit("/", 1)[0] if "/" in p else "") == d]
    same_top = [p for p in pool if p != gold and p.split("/")[0] == top and p not in same_dir]
    rest = [p for p in pool if p != gold and p not in same_dir and p not in same_top]
    for x in (same_dir, same_top, rest):
        rng.shuffle(x)
    out = (same_dir + same_top + rest)[:n]
    return out, sum(1 for p in out if p in set(same_dir) | set(same_top))


def package_of(path: str, depth: int = 2) -> str:
    parts = path.split("/")[:-1]
    return "/".join(parts[:depth]) if parts else "(repository root)"


def _problem(kind: str, repo: str, row_id: str, instructions: str, context: str, paths: list[str], gold: list[int],
             values: list[float], tier: str, evidence: dict, meta: dict, qtype: str = "choice") -> dict:
    taken: set[str] = set()
    cands = [{"name": handle(p, taken), "feat": p, "value": v, "cons": {"relevance": float(v)}} for p, v in zip(paths, values)]
    return {"slice": "swe_" + kind, "track": "software_repair", "k": len(paths), "instructions": instructions, "context": context,
            "candidates": cands, "gold": gold[0] if (qtype in ("choice", "noul") and gold) else None, "gold_multi": gold if qtype == "multi" else None,
            "answerable": bool(gold), "difficulty": meta.pop("difficulty"), "flip": None, "qtype": qtype,
            "evidence": {"type": "execution" if tier == "A" else "parsed_diff", "verified": True, **evidence},
            "state": {}, "group": repo, "tier": tier, "meta": {"instance_id": row_id, "repo": repo, **meta}}


def decisions(row: dict, tree: list[str], rng: random.Random, kinds: list[str], k: int, func_pool: dict[str, set[str]]
              ) -> list[dict]:
    """Decisions of the requested kinds for one row (those that can be built for it)."""
    files = touched_files(row["patch"])
    src = [f for f in files if f.endswith(".py") and not is_test_path(f)]
    stmt = (row.get("problem_statement") or "").strip()
    if not stmt or not src:
        return []
    repo = row["repo"].split("/", 1)[-1]
    f2p = parse_list(row["FAIL_TO_PASS"])
    ev = {"tests_fail_without_fix": len(f2p), "patch_verified": True}
    pool = [p for p in tree if p.endswith(".py") and not is_test_path(p)]
    out = []
    if len(src) == 1 and src[0] in set(pool):
        g = src[0]
        if "file" in kinds or "file_ood" in kinds:
            neg, n_near = near(g, pool, rng, k - 1 if "file" in kinds else k)
            if len(neg) >= 3:
                top, d = g.split("/")[0], g.rsplit("/", 1)[0] if "/" in g else ""
                val = lambda p: 2.0 if p == g else 1.0 if (p.rsplit("/", 1)[0] if "/" in p else "") == d else 0.5 if p.split("/")[0] == top else 0.0  # noqa: E731
                if "file" in kinds:
                    paths = [g] + neg
                    rng.shuffle(paths)
                    out.append(_problem("file", repo, row["instance_id"], "Choose the source file that has to change to fix this issue.",
                                        stmt, paths, [paths.index(g)], [val(p) for p in paths], "A", ev,
                                        {"difficulty": "hard" if n_near >= (len(neg) + 1) // 2 else "easy", "near_negatives": n_near}))
                else:
                    rng.shuffle(neg)
                    out.append(_problem("file_ood", repo, row["instance_id"], "Choose the source file that has to change to fix this issue.",
                                        stmt, neg, [], [0.0] * len(neg), "A", ev, {"difficulty": "hard", "near_negatives": n_near, "removed": g}))
        if "subsystem" in kinds:
            pk = sorted({package_of(p) for p in pool})
            gp = package_of(g)
            if len(pk) >= 4 and gp in pk:
                neg, n_near = near(gp + "/x", [x + "/x" for x in pk], rng, min(k, len(pk)) - 1)
                paths = [gp] + [x[:-2] for x in neg]
                rng.shuffle(paths)
                out.append(_problem("subsystem", repo, row["instance_id"], "Choose the package where the fix for this issue belongs.",
                                    stmt, paths, [paths.index(gp)], [2.0 if p == gp else 1.0 if p.split("/")[0] == gp.split("/")[0] else 0.0 for p in paths],
                                    "A", ev, {"difficulty": "hard" if n_near >= len(neg) // 2 else "easy"}))
        if "involved" in kinds:
            neg, _ = near(g, pool, rng, 1)
            if neg:
                yes = rng.random() < 0.5
                target = g if yes else neg[0]
                p = _problem("involved", repo, row["instance_id"], f"Fixing this issue requires a change in {target}.", stmt,
                             ["false", "true"], [1 if yes else 0], [0.0, 0.0], "A", ev, {"difficulty": "hard", "target": target}, qtype="noul")
                out.append(p)
        if "function" in kinds:
            fn = changed_function(row["patch"])
            others = sorted(func_pool.get(row["repo"] + ":" + g, set()) - {fn}) if fn else []
            if fn and len(others) >= 3:
                rng.shuffle(others)
                names = [fn] + others[: k - 1]
                rng.shuffle(names)
                p = _problem("function", repo, row["instance_id"], f"Choose the function in {g} that has to change to fix this issue.",
                             stmt, names, [names.index(fn)], [2.0 if n == fn else 1.0 for n in names], "B", ev, {"difficulty": "hard", "file": g})
                out.append(p)
    if "files" in kinds and len(src) >= 2 and all(s in set(pool) for s in src) and len(src) <= 4:
        neg, n_near = near(src[0], [p for p in pool if p not in src], rng, max(3, k - len(src)))
        if len(neg) >= 3:
            paths = src + neg
            rng.shuffle(paths)
            out.append(_problem("files", repo, row["instance_id"], "Select every source file that has to change to fix this issue.",
                                stmt, paths, sorted(paths.index(s) for s in src), [2.0 if p in src else 0.0 for p in paths], "A", ev,
                                {"difficulty": "hard", "near_negatives": n_near}, qtype="multi"))
    if "tests" in kinds:
        mods = sorted({test_module(t) for t in f2p})
        allm = sorted({test_module(t) for t in f2p + parse_list(row["PASS_TO_PASS"])})
        if len(mods) == 1 and len(allm) >= 5:
            neg, n_near = near(mods[0], allm, rng, min(k, len(allm)) - 1)
            paths = [mods[0]] + neg
            rng.shuffle(paths)
            out.append(_problem("tests", repo, row["instance_id"], "Choose the test module whose tests fail because of this bug.",
                                stmt, paths, [paths.index(mods[0])], [2.0 if p == mods[0] else 0.0 for p in paths], "A", ev,
                                {"difficulty": "hard" if n_near >= len(neg) // 2 else "easy", "near_negatives": n_near}))
    return out


def function_pool(rows: list[dict]) -> dict[str, set[str]]:
    """repo:file -> function names seen as the changed function of any bug in that file (the candidate pool)."""
    pool: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        files = touched_files(r["patch"])
        if len(files) == 1:
            fn = changed_function(r["patch"])
            if fn:
                pool[r["repo"] + ":" + files[0]].add(fn)
    return pool
