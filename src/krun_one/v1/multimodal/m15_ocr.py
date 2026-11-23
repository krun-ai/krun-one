"""M1.5 OCR backends (frozen; chosen by the registered benchmark rule in configs/v1/m15.yaml).

OCR text is an INTERNAL input to the Krun text path (like an extracted text layer), never a model output.
Every backend: ``read(list of image paths or PIL images) -> list[str]`` (reading order, lines joined by newlines),
``n_params`` (None when not applicable, e.g. Tesseract), ``name``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _pil(x):
    from PIL import Image

    return x.convert("RGB") if hasattr(x, "convert") else Image.open(x).convert("RGB")


class DocTR:
    """mindee/doctr, torch backend. ``reco`` may be a doctr arch name or 'hub:<repo>' (e.g. a multilingual parseq)."""

    def __init__(self, det: str = "fast_base", reco: str = "crnn_vgg16_bn", device="cuda"):
        import torch
        from doctr.models import ocr_predictor

        r: Any = reco
        if reco.startswith("hub:"):
            from doctr.models import from_hub

            r = from_hub(reco[4:])
        self.p = ocr_predictor(det_arch=det, reco_arch=r, pretrained=True, assume_straight_pages=True,
                               detect_orientation=False, straighten_pages=False)
        if device == "cuda" and torch.cuda.is_available():
            self.p = self.p.cuda()
        self.name = f"doctr:{det}+{reco}"
        self.n_params = sum(p.numel() for p in self.p.det_predictor.parameters()) + sum(p.numel() for p in self.p.reco_predictor.parameters())

    def read(self, items) -> list[str]:
        import torch

        imgs = [np.asarray(_pil(x)) for x in items]
        with torch.inference_mode():
            res = self.p(imgs)
        out = []
        for page in res.pages:
            lines = []
            for block in page.blocks:
                for line in block.lines:
                    lines.append(" ".join(w.value for w in line.words))
            out.append("\n".join(lines))
        return out


class RapidOCRBackend:
    """RapidOCR 3.x (ONNX Runtime) with PaddleOCR weights. spec: 'v5-latin' (PP-OCRv5 det + latin rec) | 'v6' (PP-OCRv6,
    multilingual) | 'v6-tiny'."""

    def __init__(self, variant: str = "v5-latin", device="cuda", rec_batch_num: int | None = None,
                 cudnn_conv_algo_search: str | None = None, det_model_path: str | None = None, rec_model_path: str | None = None):
        import onnxruntime as ort
        import torch  # noqa: F401  (loads the CUDA 12 / cuDNN 9 libraries onnxruntime-gpu 1.22 links against)

        if device == "cuda" and hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        from rapidocr import EngineType, LangDet, LangRec, OCRVersion, RapidOCR

        ver, _, lang = variant.partition("-")
        v = {"v5": "PPOCRV5", "v6": "PPOCRV6"}[ver]
        if not hasattr(OCRVersion, v):
            raise RuntimeError(f"rapidocr has no {v}")
        params = {"Det.engine_type": EngineType.ONNXRUNTIME, "Rec.engine_type": EngineType.ONNXRUNTIME,
                  "Cls.engine_type": EngineType.ONNXRUNTIME, "Det.ocr_version": getattr(OCRVersion, v),
                  "Rec.ocr_version": getattr(OCRVersion, v), "Global.use_cls": False}
        if lang == "latin":
            params["Rec.lang_type"] = LangRec.LATIN
            if v == "PPOCRV6":
                params["Det.lang_type"] = LangDet.MULTI
            else:                                   # the PP-OCRv5 latin recognizer only exists as a mobile model
                from rapidocr import ModelType

                params["Rec.model_type"] = ModelType.MOBILE
                params["Det.model_type"] = ModelType.MOBILE
        if lang == "tiny" and hasattr(__import__("rapidocr"), "ModelType"):
            from rapidocr import ModelType

            params["Det.model_type"] = ModelType.MOBILE
            params["Rec.model_type"] = ModelType.MOBILE
        if device == "cuda":
            params["EngineConfig.onnxruntime.use_cuda"] = True
            if cudnn_conv_algo_search:
                params["EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search"] = cudnn_conv_algo_search
        if rec_batch_num:
            params["Rec.rec_batch_num"] = rec_batch_num
        if det_model_path:                          # pinned model files (artifact) instead of the package download cache
            params["Det.model_path"] = det_model_path
        if rec_model_path:
            params["Rec.model_path"] = rec_model_path
        self.eng = RapidOCR(params=params)
        self.config = {"variant": variant, "rec_batch_num": rec_batch_num or 6, "cudnn_conv_algo_search": cudnn_conv_algo_search or "EXHAUSTIVE"}
        self.name = f"rapidocr:{variant}"
        self.n_params = None

    def read(self, items) -> list[str]:
        out = []
        for x in items:
            r = self.eng(np.asarray(_pil(x)))
            txts = list(getattr(r, "txts", None) or [])
            boxes = getattr(r, "boxes", None)
            if boxes is not None and len(txts):
                order = sorted(range(len(txts)), key=lambda i: (round(float(np.mean(boxes[i][:, 1])) / 12), float(np.min(boxes[i][:, 0]))))
                txts = [txts[i] for i in order]
            out.append("\n".join(txts))
        return out


class Tesseract:
    def __init__(self, langs: str = "eng+por+spa+fra"):
        import pytesseract

        self.t, self.langs = pytesseract, langs
        self.name = f"tesseract:{langs}"
        self.n_params = None

    def read(self, items) -> list[str]:
        return [self.t.image_to_string(_pil(x), lang=self.langs).strip() for x in items]


def build_ocr(spec: str, **kw):
    """'doctr' | 'doctr:<det>+<reco>' | 'rapidocr[:variant]' | 'tesseract'; kw -> RapidOCR execution options."""
    kind, _, rest = spec.partition(":")
    if kind == "doctr":
        det, _, reco = (rest or "fast_base+crnn_vgg16_bn").partition("+")
        return DocTR(det, reco or "crnn_vgg16_bn")
    if kind == "rapidocr":
        return RapidOCRBackend(rest or "v5-latin", **kw)
    if kind == "tesseract":
        return Tesseract()
    raise ValueError(spec)


def chosen_ocr(root) -> str:
    """The OCR picked by the registered benchmark rule (results/v1/m15/ocr_benchmark.json)."""
    import json
    from pathlib import Path

    return json.loads((Path(root) / "results/v1/m15/ocr_benchmark.json").read_text())["chosen"]


def ocr_exec_options(root) -> dict:
    """The execution options picked by the M1.5b OCR optimization rule (results/v1/m15b/ocr_optimization.json), or {}."""
    import json
    from pathlib import Path

    p = Path(root) / "results/v1/m15b/ocr_optimization.json"
    return json.loads(p.read_text())["chosen_options"] if p.exists() else {}
