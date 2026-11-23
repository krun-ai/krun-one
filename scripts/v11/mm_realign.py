"""Krun One v1.1 R3 — re-align the M1.5c multimodal fusion to a new (frozen) text core. Nothing else is trained.

    export KRUN_M15_CONFIG=configs/v11/r3_m15c_realign.yaml
    python scripts/v11/mm_realign.py teacher --core runs/v11/r3/text/r3f_s1
    python scripts/v11/mm_realign.py train   --core runs/v11/r3/text/r3f_s1 --tag r3f_s1_a [--seed 20260932] [--epochs 1]
    python scripts/v11/mm_realign.py dump    --core runs/v11/r3/text/r3f_s1 --tag r3f_s1_a      # + registered gate table
    python scripts/v11/mm_realign.py latency --core runs/v11/r3/text/r3f_s1 --tag r3f_s1_a
    python scripts/v11/mm_realign.py latency --core v1 --tag v1_m15c --run runs/v1/m15c/krun-one-v1-m15c   # same-host reference

The registered M1.5c trainer (scripts/v1/train_m15.py) is reused unchanged; this wrapper only swaps what it loads:
  text core   the candidate's v1-architecture weights (frozen; its digest is checked before / after by the trainer)
  init        the registered M1.5c fusion (config m15_init) -> in the dumps, system P = "new core + old fusion"
  teacher     the candidate core's own answers on the ground-truth text (data/v11/r3/mm/teacher_<core>.jsonl.gz)
  selection   val_fit only, against the M1.5c checkpoint ON THE v1 CORE (the numbers the product has today):
              eligible = every preservation group >= that reference - 1.0 pp; among eligible the best DocLayNet layout;
              if none is eligible, the checkpoint with the smallest worst shortfall (reported as not eligible).
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/v1"))
sys.path.insert(0, str(ROOT / "scripts/v11"))
os.environ.setdefault("KRUN_M15_CONFIG", "configs/v11/r3_m15c_realign.yaml")

OUT = ROOT / "results/v11/r3/multimodal"
RUNS = ROOT / "runs/v11/r3/multimodal"
REF = ROOT / "runs/v1/m15c/krun-one-v1-m15c"
GROUPS = ("audio/text_dependent", "document/text_dependent", "document/hybrid", "image/visual_only")


def grp(cells: dict, prefix: str) -> float | None:
    v = [x for k, x in cells.items() if k.startswith(prefix)]
    return float(np.mean(v)) if v else None


def select_r3(evals: dict) -> tuple[str, dict]:
    ref = json.loads((REF / "train_result.json").read_text())
    base = ref["evals"][ref["selected"]]["cells"]
    detail = {}
    for k, ev in evals.items():
        c = ev["cells"]
        short = {g: (grp(c, g) or 0) - (grp(base, g) or 0) for g in GROUPS}
        detail[k] = {"eligible": all(v >= -0.01 for v in short.values()) and k != "step_000000", "delta_vs_v1_core_m15c": short,
                     "layout": c.get("document/visual_only/multi"), "doc_visual_only": grp(c, "document/visual_only"), "worst": min(short.values())}
    cand = [k for k in detail if k != "step_000000"]
    el = [k for k in cand if detail[k]["eligible"]]
    if el:
        best = max(detail[k]["layout"] or 0 for k in el)
        tied = [k for k in el if (detail[k]["layout"] or 0) >= best - 0.005]
        return max(tied, key=lambda k: (detail[k]["doc_visual_only"] or 0, k)), detail
    return max(cand, key=lambda k: (detail[k]["worst"], k)), detail


def setup(core: str, seed: int | None = None, epochs: int | None = None):
    import eval_v11
    import train_m15 as M

    T1 = M.T1
    orig = T1.load_core
    if core != "v1":
        def load_core(device="cuda"):
            _, tok, ser = orig(device)                      # tokenizer / serializer of the frozen artifact (sha-checked)
            m, _ = eval_v11.load(core)
            if getattr(m, "cand", None) is not None or getattr(m, "adapter", None) is not None:
                raise SystemExit("R3 multimodal re-alignment expects a v1-architecture core (no adapter, no interaction block)")
            if hasattr(m, "v1_state_dict"):                 # a v1.1 training checkpoint without adapter: export to the plain v1 module
                from krun_one.v1.model import KrunOneV1Model

                v1m = KrunOneV1Model.from_spec(m.v11.base)
                v1m.load_state_dict(m.v1_state_dict())
                m = v1m
            return m.to(device).eval(), tok, ser

        T1.load_core = load_core
    M.RUNS = RUNS
    tag = Path(core).name if core != "v1" else "v1"
    tp = ROOT / "data/v11/r3/mm" / f"teacher_{tag}.jsonl.gz"
    M.load_teacher = lambda: ({json.loads(x)["id"]: json.loads(x) for x in gzip.open(tp, "rt")} if tp.exists() else {})
    M.select_b = select_r3
    if seed is not None:
        M.CFG["seed"] = seed
    if epochs is not None:
        M.CFG["optimizer"]["epochs"] = epochs
    return M, tp


@torch.no_grad()
def cmd_teacher(M, tp: Path) -> None:
    """scripts/v1/train_m15.py cmd_teacher with the candidate core, written next to the R3 data (never over the v1 file)."""
    T1 = M.T1
    core, tok, ser = T1.load_core()
    tp.parent.mkdir(parents=True, exist_ok=True)
    stats = {}
    with gzip.open(tp, "wt") as f:
        for sp in ("train", "val_fit", "val_eval"):
            recs = [r for r in M.read_split(sp) if r.get("gt_text") and r.get("teacher_filter") != "never"]
            order = sorted(range(len(recs)), key=lambda i: len(recs[i]["gt_text"]))
            for s in range(0, len(order), 64):
                rs = [recs[i] for i in order[s:s + 64]]
                enc = [ser.encode_record(M.with_context(r, r["gt_text"])) for r in rs]
                flat, where = M.chunk_index(enc)
                b = {k: v.cuda() for k, v in M.collate_chunks(flat, tok.pad_token_id).items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = core(**b)
                q = M.merge_chunks(o, where, len(rs), [e.n_candidates for e in enc])
                for j, r in enumerate(rs):
                    K = len(r["candidates"])
                    rank = q["rank"][j, :K].float().cpu().numpy()
                    app = q["apply"][j, :K].float().cpu().numpy()
                    gold = T1.target_of(r)
                    t = r["task"]["type"]
                    ok = bool(((1 / (1 + np.exp(-app)) > 0.35) == (gold > 0.5)).all()) if t == "multi" else \
                        bool(r["answerable"] and gold[int(np.argmax(rank))] > 0.5)
                    use = r.get("teacher_filter") == "always" or ok
                    st = stats.setdefault(f"{sp}/{r['source']}/{t}", [0, 0, 0])
                    st[0] += 1; st[1] += ok; st[2] += use
                    f.write(json.dumps({"id": r["id"], "rank": [round(float(x), 4) for x in rank],
                                        "apply": [round(float(x), 4) for x in app], "correct": ok, "use": use}) + "\n")
    res = {k: {"n": v[0], "teacher_acc": v[1] / v[0], "used": v[2]} for k, v in sorted(stats.items())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"teacher_stats_{tp.stem.split('.')[0]}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"teacher": str(tp), "n": sum(v["n"] for v in res.values()), "used": sum(v["used"] for v in res.values())}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["teacher", "train", "dump", "latency"])
    ap.add_argument("--core", required=True)
    ap.add_argument("--tag")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--run", help="latency: an existing run directory (default runs/v11/r3/multimodal/<tag>)")
    ap.add_argument("--max-steps", type=int, default=0)
    a = ap.parse_args()
    M, tp = setup(a.core, a.seed, a.epochs)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "teacher":
        return cmd_teacher(M, tp)
    run = Path(a.run) if a.run else RUNS / a.tag
    if a.cmd == "train":
        if not tp.exists():
            raise SystemExit(f"missing teacher {tp}")
        M.cmd_train(argparse.Namespace(limit=0, max_steps=a.max_steps, run_name=a.tag))
        tr = json.loads((run / "train_result.json").read_text())
        tr["r3"] = {"core": a.core, "seed": M.CFG["seed"], "epochs": M.CFG["optimizer"]["epochs"], "init": M.CFG["m15_init"], "teacher": str(tp)}
        (run / "train_result.json").write_text(json.dumps(tr, indent=1))
        print(json.dumps({"selected": tr["selected"], "detail": tr["selection_detail"]}, indent=1))
    elif a.cmd == "dump":
        # system P of the dump = the init checkpoint on the same core. The registered dump builds P as a GATED model (M1.5);
        # here P is the gate-less M1.5c fusion, so P must be built without gates (same forward path as C).
        orig_m15 = M.M15Model
        M.M15Model = lambda inner, gate=None: orig_m15(inner, gate=False)
        M.cmd_dump(argparse.Namespace(run=str(run), limit=0, max_steps=0, run_name=None))
        spec = importlib.util.spec_from_file_location("m15b_evaluate", ROOT / "scripts/v1/m15b_evaluate.py")
        ev = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ev)
        sys.argv = ["m15b_evaluate.py", "--run", str(run), "--out", str(OUT / f"{a.tag}.json")]
        ev.main()
    else:
        import m15_serving_checks as SC

        SC.OUT = OUT
        SC.cmd_latency(argparse.Namespace(run=str(run), iters=40, tag="_" + a.tag))


if __name__ == "__main__":
    main()
