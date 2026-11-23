"""M1.5 OCR benchmark + registered selection rule (configs/v1/m15.yaml ocr_selection). Runs on the pod BEFORE the
dataset OCR pass; uses val_fit renders only (ground truth known), never val_eval / protected.

    python scripts/v1/m15_ocr_bench.py --candidates doctr:fast_base+crnn_vgg16_bn,tesseract,... -> results/v1/m15/ocr_benchmark.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/v1"))
from krun_one.v1.leakage import normalize  # noqa: E402

D1 = ROOT / "data/m1"


def samples(n_doc=300, n_screen=300, n_cord=100):
    rng = random.Random(7)
    fit = [json.loads(l) for l in open(D1 / "val_fit.jsonl", encoding="utf-8")]
    docs = {r["media"]["path"]: r for r in fit if r["source"] == "synthdoc"}
    screens = [r for r in fit if r["source"] == "kd1_screen"]
    cord = [r for r in fit if r["source"] == "cord"]
    out = []
    for p in rng.sample(sorted(docs), min(n_doc, len(docs))):
        meta = json.loads((D1 / p.replace(".jpg", ".json")).read_text())
        out.append({"kind": "synthdoc", "path": p, "gt": meta["text"]})
    for r in rng.sample(screens, min(n_screen, len(screens))):
        out.append({"kind": "screenshot", "path": r["media"]["path"], "gt": r["paired_text"], "rec": r})
    for r in rng.sample(cord, min(n_cord, len(cord))):
        out.append({"kind": "cord", "path": r["media"]["path"], "gt": None})
    return out


def cer(gt: str, hyp: str) -> float:
    a, b = normalize(gt), normalize(hyp)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return 1 - sm.ratio()


def word_recall(gt: str, hyp: str) -> float:
    g = normalize(gt).split()
    h = set(normalize(hyp).split())
    return float(np.mean([w in h for w in g])) if g else 1.0


def downstream(texts: dict[str, str], S) -> float:
    """Frozen Krun accuracy on the kd1 screenshot decisions with context = OCR text (answerable choice only)."""
    import train_m1 as T1
    from krun_one.v1.model import chunk_index, collate_chunks, merge_chunks

    core, tok, ser = S["core"]
    rs = [s["rec"] for s in S["items"] if s["kind"] == "screenshot" and s["rec"]["task"]["type"] == "choice" and s["rec"]["answerable"]]
    ok = []
    for i in range(0, len(rs), 32):
        chunk = rs[i:i + 32]
        enc = [ser.encode_record(dict(r, context={"text": texts[r["media"]["path"]], "language": r["context"]["language"]})) for r in chunk]
        flat, where = chunk_index(enc)
        b = {k: v.cuda() for k, v in collate_chunks(flat, tok.pad_token_id).items()}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            q = merge_chunks(core(**b), where, len(chunk), [e.n_candidates for e in enc])
        for j, r in enumerate(chunk):
            g = T1.target_of(r)
            ok.append(float(g[int(q["rank"][j, :len(g)].argmax())] > 0.5))
    return float(np.mean(ok)), len(ok)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default=str(ROOT / "results/v1/m15/ocr_benchmark.json"))
    a = ap.parse_args()
    import train_m1 as T1
    from krun_one.v1.multimodal.m15_ocr import build_ocr

    items = samples()
    S = {"items": items, "core": T1.load_core()}
    gt_texts = {s["path"]: s["gt"] for s in items if s["kind"] == "screenshot"}
    res = {"n": {k: sum(1 for s in items if s["kind"] == k) for k in ("synthdoc", "screenshot", "cord")}, "candidates": {}}
    res["oracle_downstream_gt_text"] = downstream(gt_texts, S)[0]
    res["empty_context_downstream"] = downstream({p: "" for p in gt_texts}, S)[0]
    for spec in a.candidates.split(","):
        try:
            torch.cuda.reset_peak_memory_stats()
            vr0 = torch.cuda.memory_allocated()
            t0 = time.time()
            ocr = build_ocr(spec)
            load_s = time.time() - t0
            ocr.read([str(D1 / items[0]["path"])])                  # warm-up
            texts, lat = {}, []
            for s in items:
                torch.cuda.synchronize()
                t = time.perf_counter()
                texts[s["path"]] = ocr.read([str(D1 / s["path"])])[0]
                torch.cuda.synchronize()
                lat.append((time.perf_counter() - t) * 1e3)
            by = {}
            for k in ("synthdoc", "screenshot"):
                ss = [s for s in items if s["kind"] == k]
                by[k] = {"cer": float(np.mean([cer(s["gt"], texts[s["path"]]) for s in ss])),
                         "word_recall": float(np.mean([word_recall(s["gt"], texts[s["path"]]) for s in ss])),
                         "p50_ms": float(np.percentile([l for s, l in zip(items, lat) if s["kind"] == k], 50))}
            by["cord"] = {"p50_ms": float(np.percentile([l for s, l in zip(items, lat) if s["kind"] == "cord"], 50)),
                          "chars_mean": float(np.mean([len(texts[s["path"]]) for s in items if s["kind"] == "cord"]))}
            dacc, dn = downstream(texts, S)
            res["candidates"][spec] = {"name": ocr.name, "params": ocr.n_params, "load_s": load_s, "by_kind": by,
                                       "page_p50_ms": float(np.percentile(lat, 50)), "page_p95_ms": float(np.percentile(lat, 95)),
                                       "downstream_screenshot_acc": dacc, "n_downstream": dn,
                                       "peak_vram_mib": round((torch.cuda.max_memory_allocated() - vr0) / 2**20),
                                       "max_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
                                       "examples": {s["path"]: texts[s["path"]][:300] for s in items[:2] + items[300:302]}}
            print(spec, json.dumps({k: v for k, v in res["candidates"][spec].items() if k != "examples"}), flush=True)
            del ocr
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            res["candidates"][spec] = {"error": f"{type(e).__name__}: {e}"}
            print(spec, "ERROR", e, flush=True)
    ok = {k: v for k, v in res["candidates"].items() if "error" not in v}
    fast = {k: v for k, v in ok.items() if v["page_p50_ms"] <= 150}
    if fast:
        best = max(v["downstream_screenshot_acc"] for v in fast.values())
        tied = [k for k, v in fast.items() if v["downstream_screenshot_acc"] >= best - 0.01]
        chosen = min(tied, key=lambda k: (np.mean([fast[k]["by_kind"][x]["cer"] for x in ("synthdoc", "screenshot")]),
                                          fast[k]["page_p50_ms"], fast[k]["params"] or 0))
        rule = "fast (<=150 ms) -> best downstream (1 pp tie) -> lower CER -> latency -> params"
    else:
        good = {k: v for k, v in ok.items() if v["downstream_screenshot_acc"] >= 0.9} or ok
        chosen = min(good, key=lambda k: good[k]["page_p50_ms"])
        rule = "none <= 150 ms -> fastest with downstream >= 0.9"
    res["chosen"], res["rule_applied"] = chosen, rule
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("CHOSEN", chosen, rule)


if __name__ == "__main__":
    main()
