"""M1.5b evaluation (configs/v1/m15b.yaml gates) from runs/v1/m15b/<run>/signals_{val_fit,val_eval}.jsonl.gz.

Systems: C = M1.5b, P = M1.5 selected (same host / harness), A = M1, B = frozen text on OCR / ASR, O = oracle.
val_eval decisions are the M1.5 val_eval file verbatim.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import m15_evaluate as E

ROOT = Path(__file__).resolve().parents[2]
E.SYS = ("C", "P", "B", "A", "O")


def fit_multi(rows):
    th = {}
    by = defaultdict(list)
    for r in rows:
        if r["type"] == "multi":
            by[r["modality"]].append(r)
    for mod, rs in by.items():
        for s in ("C", "P", "A"):
            best = (-1, 0.5)
            for t in np.linspace(0.2, 0.8, 25):
                tp = fp = fn = 0
                for r in rs:
                    pr = 1 / (1 + np.exp(-np.asarray(r[f"{s}_apply"]))) > t
                    g = np.zeros(r["k"], bool); g[r["gold"]] = True
                    tp += int((pr & g).sum()); fp += int((pr & ~g).sum()); fn += int((~pr & g).sum())
                best = max(best, (2 * tp / max(1, 2 * tp + fp + fn), float(t)))
            th[(mod, s)] = best[1]
    return th


def correct(r, s, th):
    g = np.zeros(r["k"], bool); g[r["gold"]] = True
    if r["type"] == "multi":
        t = th.get((r["modality"], s), 0.35) if s in ("C", "P", "A") else 0.35
        return float(((1 / (1 + np.exp(-np.asarray(r[f"{s}_apply"]))) > t) == g).all())
    if r["type"] == "choice" and not r["answerable"]:
        return None
    return float(g[int(np.argmax(r[f"{s}_rank"]))])


def table(rows, key, th):
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for s in E.SYS:
            if s == "O" and not r["has_gt"]:
                continue
            c = correct(r, s, th)
            if c is not None:
                agg[key(r)][s].append(c)
    return {k: {**{s: float(np.mean(v)) for s, v in d.items()}, "n": len(d["C"])} for k, d in sorted(agg.items())}


def layout_breakdown(ev, th):
    out = {}
    rows = [r for r in ev if r["source"] == "doclaynet" and r["type"] == "multi"]
    for s in ("C", "P", "A"):
        t = th.get(("document", s), 0.35)
        cnt = defaultdict(lambda: [0, 0, 0])
        for r in rows:
            pr = 1 / (1 + np.exp(-np.asarray(r[f"{s}_apply"]))) > t
            for i, cid in enumerate(r["candidate_ids"]):
                gi = i in r["gold"]
                cnt[cid][0] += int(pr[i] and gi); cnt[cid][1] += int(pr[i] and not gi); cnt[cid][2] += int(gi and not pr[i])
        out[s] = {c: {"precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn), "f1": 2 * tp / max(1, 2 * tp + fp + fn), "support": tp + fn}
                  for c, (tp, fp, fn) in sorted(cnt.items())}
        tp, fp, fn = (sum(v[i] for v in cnt.values()) for i in range(3))
        out[s]["_micro_f1"] = 2 * tp / max(1, 2 * tp + fp + fn)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=str(ROOT / "results/v1/m15b/m15b_eval.json"))
    a = ap.parse_args()
    run = Path(a.run)
    fit, ev = E.load(run, "val_fit"), E.load(run, "val_eval")
    th = fit_multi(fit)
    task = lambda r: "kd1" if r["source"] in ("kd1_tts", "kd1_screen") else r["id"].rsplit("/", 1)[-1]  # noqa: E731
    cells = table(ev, lambda r: f"{r['modality']}|{r['source']}|{r['task_class']}|{task(r)}|{r['type']}", th)
    out = {"multi_thresholds_fit": {f"{k[0]}/{k[1]}": v for k, v in th.items()}, "cells": cells,
           "by_modality_task_class": table(ev, lambda r: f"{r['modality']}/{r['task_class']}", th),
           "by_speech": table([r for r in ev if r["modality"] == "audio"], lambda r: f"{r['speech']}/{r['source']}", th),
           "doclaynet_layout_breakdown": layout_breakdown(ev, th)}
    cm = lambda s, pred: E.cell_mean(cells, s, pred)  # noqa: E731
    vis_doc = lambda k: k.startswith("document|") and "|visual_only|" in k  # noqa: E731
    lay = "document|doclaynet|visual_only|layout|multi"
    txt_doc = lambda k: k.startswith("document|") and "|text_dependent|" in k  # noqa: E731
    real = lambda k: k.startswith("audio|minds14|") or k.startswith("audio|fleurs|")  # noqa: E731
    img = lambda k: k.startswith("image|")  # noqa: E731
    g = {}
    g["G1_document_visual_only"] = {"visual_only_mean_C": cm("C", vis_doc), "M1_ref": 0.679, "P": cm("P", vis_doc),
                                    "layout_C": cells[lay]["C"], "layout_P": cells[lay]["P"], "layout_A_M1": cells[lay]["A"]}
    g["G1_document_visual_only"]["pass"] = g["G1_document_visual_only"]["visual_only_mean_C"] >= 0.679 and cells[lay]["C"] >= cells[lay]["P"] + 0.05
    shot = cells["document|kd1_screen|text_dependent|kd1|choice"]
    g["G2_screenshot_text"] = {"screenshot_choice_C": shot["C"], "P": shot["P"], "text_dep_doc_C": cm("C", txt_doc), "text_dep_doc_P": cm("P", txt_doc)}
    g["G2_screenshot_text"]["pass"] = shot["C"] >= 0.979 and cm("C", txt_doc) >= cm("P", txt_doc) - 0.01
    mi = cells["audio|minds14|text_dependent|intent|choice"]
    g["G3_real_speech"] = {"real_C": cm("C", real), "real_P": cm("P", real), "minds_intent_C": mi["C"], "minds_intent_P": mi["P"]}
    g["G3_real_speech"]["pass"] = cm("C", real) >= 0.841 and mi["C"] >= 0.890
    g["G4_image"] = {"C": cm("C", img), "P": cm("P", img), "pass": cm("C", img) >= 0.633}
    hy = {"synthdoc_type": cells["document|synthdoc|hybrid|type|choice"], "doclaynet_category": cells["document|doclaynet|hybrid|category|choice"],
          "cord_type": cells["document|cord|hybrid|type|choice"]}
    g["G4b_document_hybrid"] = {k: {"C": v["C"], "P": v["P"]} for k, v in hy.items()}
    g["G4b_document_hybrid"]["pass"] = hy["synthdoc_type"]["C"] >= 0.953 and hy["doclaynet_category"]["C"] >= 0.935 and hy["cord_type"]["C"] >= 0.95
    out["gates_quality"] = g
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"gates": g, "by_modality_task_class": out["by_modality_task_class"],
                      "layout_micro_f1": {s: v["_micro_f1"] for s, v in out["doclaynet_layout_breakdown"].items()}}, indent=1))


if __name__ == "__main__":
    main()
