"""M1.5 evaluation (configs/v1/m15.yaml) from runs/v1/m15/<run>/signals_{val_fit,val_eval}.jsonl.gz.

Systems per decision: C = M1.5 (gated), F = fused (alpha = 1, diagnostic), B = frozen text path on ASR / OCR text,
A = M1 (media only, empty context), O = oracle (frozen text path on ground-truth text). Multi thresholds for C / F / A
are fitted on val_fit per (modality, primitive); B / O use the frozen text calibration (0.35). Accuracy = argmax in
gold (choice answerable / noul / score), exact set (multi). Every number is on val_eval.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SYS = ("C", "F", "B", "A", "O")


def load(run: Path, sp: str):
    return [json.loads(l) for l in gzip.open(run / f"signals_{sp}.jsonl.gz", "rt")]


def fit_multi(rows):
    th = {}
    by = defaultdict(list)
    for r in rows:
        if r["type"] == "multi":
            by[r["modality"]].append(r)
    for mod, rs in by.items():
        for s in ("C", "F", "A"):
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
        t = th.get((r["modality"], s), 0.35) if s in ("C", "F", "A") else 0.35
        return float(((1 / (1 + np.exp(-np.asarray(r[f"{s}_apply"]))) > t) == g).all())
    if r["type"] == "choice" and not r["answerable"]:
        return None
    return float(g[int(np.argmax(r[f"{s}_rank"]))])


def table(rows, key, th):
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for s in SYS:
            if s == "O" and not r["has_gt"]:
                continue
            c = correct(r, s, th)
            if c is not None:
                agg[key(r)][s].append(c)
    return {k: {**{s: float(np.mean(v)) for s, v in d.items()}, "n": len(d["C"])} for k, d in sorted(agg.items())}


def cell_mean(cells, s, pred):
    v = [c[s] for k, c in cells.items() if pred(k) and s in c]
    return float(np.mean(v)) if v else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=str(ROOT / "results/v1/m15/m15_eval.json"))
    a = ap.parse_args()
    run = Path(a.run)
    fit, ev = load(run, "val_fit"), load(run, "val_eval")
    th = fit_multi(fit)
    out = {"multi_thresholds_fit": {f"{k[0]}/{k[1]}": v for k, v in th.items()}}
    task = lambda r: "kd1" if r["source"] in ("kd1_tts", "kd1_screen") else r["id"].rsplit("/", 1)[-1]  # noqa: E731  (kd1 ids end in a hash)
    src_cells = table(ev, lambda r: f"{r['modality']}|{r['source']}|{r['task_class']}|{task(r)}|{r['type']}", th)
    out["cells"] = src_cells
    out["by_modality_task_class"] = table(ev, lambda r: f"{r['modality']}/{r['task_class']}", th)
    out["by_speech"] = table([r for r in ev if r["modality"] == "audio"], lambda r: f"{r['speech']}/{r['source']}", th)
    out["by_language"] = table(ev, lambda r: f"{r['modality']}/{r['lang_media']}", th)
    out["by_slice"] = table([dict(r, _s=s) for r in ev for s in r["slices"]], lambda r: f"{r['modality']}/{r['_s']}", th)
    out["alpha_mean"] = {k: float(np.mean([r["alpha"] for r in ev if f"{r['modality']}/{r['task_class']}" == k]))
                         for k in sorted({f"{r['modality']}/{r['task_class']}" for r in ev})}
    # gates: means over task cells (modality|source|class|task|type)
    real = lambda k: k.startswith("audio|minds14|") or k.startswith("audio|fleurs|")  # noqa: E731
    textvis = lambda k: k.startswith("document|") and "|text_dependent|" in k  # noqa: E731
    g = {}
    cA, cC, cB = cell_mean(src_cells, "A", real), cell_mean(src_cells, "C", real), cell_mean(src_cells, "B", real)
    g["GA1_real_speech_vs_m1"] = {"C": cC, "A": cA, "pass": cC is not None and cA is not None and cC >= cA + 0.20}
    g["GA2_real_speech_vs_transcript"] = {"C": cC, "B": cB, "pass": cC is not None and cB is not None and cC >= cB - 0.01}
    sA, sC, sB, sO = (cell_mean(src_cells, s, textvis) for s in ("A", "C", "B", "O"))
    g["GS1_text_visual_vs_vision"] = {"C": sC, "A": sA, "B": sB, "pass": sC >= sA + 0.20}
    g["GS2_text_visual_gap"] = {"C": sC, "A": sA, "oracle": sO, "closed": (sC - sA) / max(1e-9, sO - sA), "pass": (sC - sA) >= 0.6 * (sO - sA)}
    gp = {}
    for mod in ("image", "document"):
        pv = lambda k, mod=mod: k.startswith(f"{mod}|") and "|visual_only|" in k  # noqa: E731
        gp[mod] = {"C": cell_mean(src_cells, "C", pv), "A_m1": cell_mean(src_cells, "A", pv)}
        gp[mod]["pass"] = gp[mod]["C"] >= gp[mod]["A_m1"] - 0.02
    g["GP_preserve_perception"] = {**gp, "pass": all(v["pass"] for v in gp.values())}
    ac = lambda k: "|acoustic_only|" in k  # noqa: E731
    hy = lambda k: "|hybrid|" in k  # noqa: E731
    g["soft_acoustic_value"] = {"C": cell_mean(src_cells, "C", ac), "B": cell_mean(src_cells, "B", ac), "A": cell_mean(src_cells, "A", ac)}
    g["soft_acoustic_value"]["pass"] = (g["soft_acoustic_value"]["C"] or 0) > (g["soft_acoustic_value"]["B"] or 0)
    hyC, hyA, hyB = (cell_mean(src_cells, s, hy) for s in ("C", "A", "B"))
    g["soft_hybrid"] = {"C": hyC, "A": hyA, "B": hyB, "pass": hyC >= max(hyA, hyB)}
    out["gates_quality"] = g
    # cross-modal consistency: same semantic group (kd1 decision as screenshot + speech + text; FLoRes sentence in 4 languages)
    grp = defaultdict(list)
    for r in ev:
        if r.get("semantic_group_id") and r["type"] != "multi":
            grp[(r["semantic_group_id"], r["id"].rsplit("/", 1)[-1] if r["source"] == "fleurs" else "kd1")].append(r)
    cons = defaultdict(list)
    for (gid, task), rs in grp.items():
        if len(rs) < 2:
            continue
        for s in ("C", "A", "B"):
            if task == "kd1":
                am = {r["modality"]: int(np.argmax(r[f"{s}_rank"])) for r in rs}
                if len(am) == 2:
                    cons[f"kd1 screenshot~speech/{s}"].append(am["document"] == am["audio"])
                o = [int(np.argmax(r["O_rank"])) for r in rs if r["has_gt"]]
                if o:
                    for r in rs:
                        cons[f"kd1 {r['modality']}~text/{s}"].append(int(np.argmax(r[f"{s}_rank"])) == o[0])
            elif task == "topic":
                cids = [r for r in rs]
                if len({r["lang_media"] for r in cids}) >= 2:
                    corr = [bool(np.argmax(r[f"{s}_rank"]) in r["gold"]) for r in cids]
                    cons[f"fleurs cross-language all-correct/{s}"].append(all(corr))
    out["cross_modal_consistency"] = {k: {"rate": float(np.mean(v)), "n": len(v)} for k, v in sorted(cons.items())}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"gates_quality": g, "by_modality_task_class": out["by_modality_task_class"], "by_speech": out["by_speech"],
                      "alpha": out["alpha_mean"]}, indent=1))


if __name__ == "__main__":
    main()
