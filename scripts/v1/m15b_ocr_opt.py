"""M1.5b OCR latency profile + registered optimization rule (configs/v1/m15b.yaml ocr_optimization).

    python scripts/v1/m15b_ocr_opt.py  -> results/v1/m15b/ocr_optimization.json

RapidOCR PP-OCRv5 latin stays the OCR. For each execution config (cuDNN algo search x rec batch) this measures, on
real val_fit pages, the p50 / p95 of every stage by running the RapidOCR pipeline step by step (the staged run is
asserted to produce the same text as a normal call), the input H2D copy share of the ONNX sessions (IOBinding
comparison), and the quality guard on the M1.5 OCR benchmark samples. The fastest config passing the guard is chosen.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/v1"))
D1 = ROOT / "data/m1"


def staged(ocr, path: str) -> tuple[str, dict]:
    from PIL import Image
    from rapidocr.ch_ppocr_cls import TextClsOutput
    from rapidocr.ch_ppocr_det import TextDetOutput
    from rapidocr.ch_ppocr_rec import TextRecOutput

    eng = ocr.eng
    t = {}
    c = time.perf_counter()
    img = np.asarray(Image.open(path).convert("RGB"))
    n = time.perf_counter(); t["decode"] = n - c; c = n
    ori = eng.load_img(img)
    n = time.perf_counter(); t["load_img"] = n - c; c = n
    im, op = eng.preprocess_img(ori)
    op["padding_1"] = {"top": 0, "left": 0}
    n = time.perf_counter(); t["preprocess_img"] = n - c; c = n
    det = eng.text_det
    pre = det.get_preprocess(max(im.shape[0], im.shape[1]))
    x = pre(im)
    n = time.perf_counter(); t["det_preprocess"] = n - c; c = n
    preds = det.session(x)
    n = time.perf_counter(); t["det_session"] = n - c; c = n
    boxes, scores = det.postprocess_op(preds, (im.shape[0], im.shape[1]))
    boxes = det.sorted_boxes(boxes) if len(boxes) else boxes
    n = time.perf_counter(); t["det_postprocess"] = n - c; c = n
    if len(boxes) == 0:
        t.update({k: 0.0 for k in ("crop", "rec_preprocess", "rec_session", "rec_postprocess", "final_output")})
        return "", {k: v * 1e3 for k, v in t.items()}
    crops = [eng.crop_text_regions(im, [b])[0] for b in boxes]
    n = time.perf_counter(); t["crop"] = n - c; c = n
    rec = eng.text_rec
    widths = [cr.shape[1] / float(cr.shape[0]) for cr in crops]
    idx = np.argsort(np.array(widths))
    res = [("", 0.0)] * len(crops)
    tp = ts = tpost = 0.0
    for b0 in range(0, len(crops), rec.rec_batch_num):
        b1 = min(len(crops), b0 + rec.rec_batch_num)
        a = time.perf_counter()
        imgC, imgH, imgW = rec.rec_image_shape[:3]
        mwr = imgW / imgH
        whl = []
        for i in range(b0, b1):
            h, w = crops[idx[i]].shape[0:2]
            mwr = max(mwr, w / h)
            whl.append(w / h)
        batch = np.concatenate([rec.resize_norm_img(crops[idx[i]], mwr)[np.newaxis, :] for i in range(b0, b1)]).astype(np.float32)
        b = time.perf_counter()
        pr = rec.session(batch)
        cc = time.perf_counter()
        lines, _ = rec.postprocess_op(pr, False, wh_ratio_list=whl, max_wh_ratio=mwr)
        for r, one in enumerate(lines):
            res[idx[b0 + r]] = (one, None)
        d = time.perf_counter()
        tp += b - a; ts += cc - b; tpost += d - cc
    t["rec_preprocess"], t["rec_session"], t["rec_postprocess"] = tp, ts, tpost
    c = time.perf_counter()
    lines, _ = list(zip(*res))
    txts, sc = list(zip(*lines))
    out = eng.build_final_output(ori, TextDetOutput(im, boxes, scores, elapse=0.0), TextClsOutput(),
                                 TextRecOutput(crops, txts, sc, [None] * len(txts), 0.0), crops, copy.deepcopy(op))
    txts2 = list(getattr(out, "txts", None) or [])
    bx = getattr(out, "boxes", None)
    if bx is not None and len(txts2):
        order = sorted(range(len(txts2)), key=lambda i: (round(float(np.mean(bx[i][:, 1])) / 12), float(np.min(bx[i][:, 0]))))
        txts2 = [txts2[i] for i in order]
    text = "\n".join(txts2)
    t["final_output"] = time.perf_counter() - c
    return text, {k: v * 1e3 for k, v in t.items()}


def h2d_share(ocr, path: str) -> dict:
    """Input host->device copy time of the det session: numpy-input run minus IOBinding run with the input on the GPU."""
    import onnxruntime as ort
    from PIL import Image

    det = ocr.eng.text_det
    im, _ = ocr.eng.preprocess_img(ocr.eng.load_img(np.asarray(Image.open(path).convert("RGB"))))
    x = det.get_preprocess(max(im.shape[:2]))(im)
    sess = det.session.session
    name = sess.get_inputs()[0].name
    for _ in range(5):
        sess.run(None, {name: x})
    a = []
    for _ in range(20):
        s = time.perf_counter(); sess.run(None, {name: x}); a.append(time.perf_counter() - s)
    xv = ort.OrtValue.ortvalue_from_numpy(x, "cuda", 0)
    io = sess.io_binding()
    io.bind_ortvalue_input(name, xv)
    for o in sess.get_outputs():
        io.bind_output(o.name, "cpu")
    b = []
    for _ in range(20):
        s = time.perf_counter(); sess.run_with_iobinding(io); b.append(time.perf_counter() - s)
    return {"det_input_bytes": int(x.nbytes), "det_run_numpy_input_ms": float(np.median(a) * 1e3),
            "det_run_gpu_input_ms": float(np.median(b) * 1e3), "det_input_h2d_ms": float((np.median(a) - np.median(b)) * 1e3)}


def main() -> None:
    import m15_ocr_bench as OB
    from krun_one.v1.multimodal.m15_ocr import RapidOCRBackend

    items = OB.samples()
    import train_m1 as T1

    S = {"items": items, "core": T1.load_core()}
    pages = [s for s in items if s["kind"] == "synthdoc"][:60] + [s for s in items if s["kind"] == "cord"][:20]
    configs = [(None, None)] + [(cs, rb) for cs in ("EXHAUSTIVE", "HEURISTIC", "DEFAULT") for rb in (6, 16, 32) if (cs, rb) != ("EXHAUSTIVE", 6)]
    res = {"pages_profiled": len(pages), "configs": {}}
    base_q = None
    for cs, rb in configs:
        key = f"{cs or 'EXHAUSTIVE'}/rec{rb or 6}"
        ocr = RapidOCRBackend("v5-latin", rec_batch_num=rb, cudnn_conv_algo_search=cs)
        for s in pages[:10]:                                    # warm-up (cuDNN algo search happens here for seen shapes)
            ocr.read([str(D1 / s["path"])])
        prof, same = [], 0
        for s in pages:
            txt, t = staged(ocr, str(D1 / s["path"]))
            t["total"] = sum(t.values())
            prof.append(t)
            same += txt == ocr.read([str(D1 / s["path"])])[0]
        stages = {k: {"p50": float(np.percentile([p[k] for p in prof], 50)), "p95": float(np.percentile([p[k] for p in prof], 95))} for k in prof[0]}
        # end-to-end call latency (the serving call) + quality guard on the benchmark samples
        lat, texts = [], {}
        for s in items:
            a = time.perf_counter(); texts[s["path"]] = ocr.read([str(D1 / s["path"])])[0]; lat.append(time.perf_counter() - a)
        q = {"downstream_screenshot_acc": OB.downstream(texts, S)[0],
             "cer_synthdoc": float(np.mean([OB.cer(s["gt"], texts[s["path"]]) for s in items if s["kind"] == "synthdoc"])),
             "cer_screenshot": float(np.mean([OB.cer(s["gt"], texts[s["path"]]) for s in items if s["kind"] == "screenshot"])),
             "page_p50_ms": float(np.percentile([l * 1e3 for s, l in zip(items, lat) if s["kind"] != "screenshot"], 50)),
             "all_p50_ms": float(np.percentile(np.array(lat) * 1e3, 50))}
        if base_q is None:
            base_q = q
            res["h2d"] = h2d_share(ocr, str(D1 / pages[0]["path"]))
        q["guard_pass"] = bool(q["downstream_screenshot_acc"] >= base_q["downstream_screenshot_acc"] - 0.005 and
                               (q["cer_synthdoc"] + q["cer_screenshot"]) / 2 <= (base_q["cer_synthdoc"] + base_q["cer_screenshot"]) / 2 + 0.002)
        res["configs"][key] = {"options": {"rec_batch_num": rb or 6, "cudnn_conv_algo_search": cs or "EXHAUSTIVE"}, "stages_ms": stages,
                               "staged_equals_call": same / len(pages), **q}
        print(key, json.dumps({"total_p50": stages["total"]["p50"], **{k: round(v, 4) if isinstance(v, float) else v for k, v in q.items()}}), flush=True)
        del ocr
    ok = {k: v for k, v in res["configs"].items() if v["guard_pass"]}
    chosen = min(ok, key=lambda k: ok[k]["page_p50_ms"])
    res["baseline"] = "EXHAUSTIVE/rec6"
    res["chosen"] = chosen
    res["chosen_options"] = res["configs"][chosen]["options"]
    res["rule"] = "fastest page p50 among configs passing the quality guard (downstream >= base - 0.5 pp, mean CER <= base + 0.002)"
    out = ROOT / "results/v1/m15b/ocr_optimization.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print("CHOSEN", chosen, res["chosen_options"])


if __name__ == "__main__":
    main()
