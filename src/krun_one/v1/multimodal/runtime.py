"""Krun One v1 multimodal runtime (candidate): text core + lazily loaded perception + M1.5c fusion -> decisions.

The runtime is transport-agnostic. It never sees HTTP, asset ids, signed URLs, storage providers, D1 / KV or RunPod:
the caller (a future worker) authenticates, resolves assets, enforces transport limits and hands over a
``ResolvedMultimodalContext`` of local files or bytes. See docs/KRUN_ONE_V1_WORKER_INTEGRATION_CONTRACT.md.

I/O contract (configs/v1/m15.yaml io_contract): every modality is INPUT-ONLY; the output is always the structured
decision response (choice / noul / score, plus multi). No generation of any kind; OCR / ASR text never leaves the
runtime (not in answers, not in metadata, not in logs).

Paths
    text only                 -> the frozen text artifact's own decide()             (bitwise identical to it)
    image                     -> decode -> SigLIP2 (256 patches) -> adapter -> fusion -> decision
    document, text layer      -> extraction (pypdf / python-docx / text) -> text path; no OCR, no vision tower
    document, visual          -> page images (pypdfium2 for PDFs) -> OCR text (context) || SigLIP2 (1024 patches) -> fusion
    audio                     -> decode + resample -> Whisper encoder (one pass) -> ASR transcript (context)
                                                                                 + acoustic tokens -> fusion
Every tower, the OCR engine, the ASR decoder and the fusion weights load on first use only.
"""

from __future__ import annotations

import hashlib
import io
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

# ------------------------------------------------------------------------------------------------ internal contract


@dataclass
class TextPart:
    text: str
    id: str | None = None


@dataclass
class MediaPart:
    """A resolved media input: exactly one of ``path`` (local file) or ``data`` (bytes), plus its MIME type."""
    mime_type: str
    path: str | None = None
    data: bytes | None = None
    id: str | None = None

    def read(self) -> bytes:
        if self.data is not None:
            return self.data
        if self.path is None:
            raise RuntimeInputError("decode_failed", "media input has neither path nor data", self.id)
        return Path(self.path).read_bytes()

    def size(self) -> int:
        return len(self.data) if self.data is not None else (Path(self.path).stat().st_size if self.path else 0)


ImagePart = DocumentPart = AudioPart = MediaPart


@dataclass
class ResolvedMultimodalContext:
    """What the runtime consumes. Order inside each list is preserved; text parts are joined with newlines."""
    text_parts: list[TextPart] = field(default_factory=list)
    image_inputs: list[MediaPart] = field(default_factory=list)
    document_inputs: list[MediaPart] = field(default_factory=list)
    audio_inputs: list[MediaPart] = field(default_factory=list)

    @classmethod
    def from_text(cls, text: str) -> "ResolvedMultimodalContext":
        return cls(text_parts=[TextPart(text)])

    @property
    def is_text_only(self) -> bool:
        return not (self.image_inputs or self.document_inputs or self.audio_inputs)

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.text_parts if p.text)


# ------------------------------------------------------------------------------------------------ errors

ERRORS = {  # code -> category
    "unsupported_modality": "client", "unsupported_mime_type": "client", "asset_too_large": "client",
    "too_many_images": "client", "too_many_documents": "client", "too_many_audio": "client",
    "document_too_many_pages": "client", "audio_too_long": "client", "empty_context": "client",
    "decode_failed": "client",
    "ocr_failed": "runtime", "asr_failed": "runtime", "vision_failed": "runtime", "artifact_integrity": "runtime",
    "multimodal_inference_failed": "model",
}


class RuntimeInputError(Exception):
    def __init__(self, code: str, message: str, part_id: str | None = None):
        super().__init__(message)
        self.code, self.message, self.part_id = code, message, part_id
        self.category = ERRORS[code]

    def to_dict(self) -> dict:
        return {"code": self.code, "category": self.category, "message": self.message, "part_id": self.part_id}


# ------------------------------------------------------------------------------------------------ helpers
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


class Timer:
    def __init__(self, cuda: bool):
        self.cuda, self.t = cuda, {}

    def __call__(self, key: str):
        timer = self

        class _Ctx:
            def __enter__(self_):
                if timer.cuda:
                    torch.cuda.synchronize()
                self_.t0 = time.perf_counter()

            def __exit__(self_, *a):
                if timer.cuda:
                    torch.cuda.synchronize()
                timer.t[key] = timer.t.get(key, 0.0) + (time.perf_counter() - self_.t0) * 1e3

        return _Ctx()


def decode_audio(data: bytes, target_sr: int = 16000) -> np.ndarray:
    import soundfile as sf

    w, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if w.ndim > 1:
        w = w.mean(1)
    if sr != target_sr:                                   # the same linear resampling the training media used
        n = int(round(len(w) * target_sr / sr))
        w = np.interp(np.linspace(0, len(w) - 1, n), np.arange(len(w)), w).astype(np.float32)
    return np.ascontiguousarray(w, dtype=np.float32)


# ------------------------------------------------------------------------------------------------ runtime
class KrunOneMultimodalRuntime:
    def __init__(self, root: Path, cfg: dict, text, device: str, verify: bool):
        self.root, self.cfg, self.text, self.device, self.verify = root, cfg, text, device, verify
        self.cuda = device == "cuda"
        self._lock = threading.Lock()
        self._vision = self._ip = self._asr = self._ocr = self._fusion = None
        self.load_seconds: dict[str, float] = {}

    # ---- load
    @classmethod
    def load(cls, root: str | Path, device: str = "cuda", dtype: str = "bf16", verify: bool = True) -> "KrunOneMultimodalRuntime":
        from ..runtime import TextArtifact

        root = Path(root)
        cfg = json.loads((root / "runtime_config.json").read_text())
        if verify:
            bad = [l for l in (root / "SHA256SUMS").read_text().splitlines()
                   if l and sha256_file(root / l.split(None, 1)[1].strip()) != l.split()[0]]
            if bad:
                raise RuntimeInputError("artifact_integrity", f"checksum mismatch: {bad[:3]}")
        t0 = time.perf_counter()
        text = TextArtifact.load(root / cfg["text_artifact"]["path"], device=device, dtype=dtype, verify=verify)
        if verify:  # TextArtifact.load just hashed every file against its SHA256SUMS; compare that entry (no second 3.5 GB hash)
            sums = dict(reversed(l.split(None, 1)) for l in (root / cfg["text_artifact"]["path"] / "SHA256SUMS").read_text().splitlines() if l)
            if sums.get("model.safetensors", "").strip() != cfg["text_artifact"]["model_sha256"]:
                raise RuntimeInputError("artifact_integrity", "text core weights do not match the manifest")
        rt = cls(root, cfg, text, device, verify)
        rt.load_seconds["text_core"] = time.perf_counter() - t0
        return rt

    @property
    def loaded(self) -> dict[str, bool]:
        return {"vision": self._vision is not None, "asr": self._asr is not None, "ocr": self._ocr is not None,
                "fusion": self._fusion is not None}

    def _hf_dir(self, key: str) -> str:
        from huggingface_hub import snapshot_download

        t = self.cfg["towers"][key]
        d = snapshot_download(t["repo"], revision=t["revision"], allow_patterns=t["files"])
        if self.verify:
            for f, sha in t["sha256"].items():
                if sha256_file(Path(d) / f) != sha:
                    raise RuntimeInputError("artifact_integrity", f"{t['repo']}/{f} does not match the manifest")
        return d

    def _get_vision(self):
        with self._lock:
            if self._vision is None:
                from .m1_towers import SigLIP2Tower

                t0 = time.perf_counter()
                d = self._hf_dir("vision")
                self._vision = SigLIP2Tower(d, device=self.device, dtype=torch.bfloat16 if self.cuda else torch.float32)
                self._ip = self._vision.proc
                self.load_seconds["vision"] = time.perf_counter() - t0
        return self._vision

    def _get_asr(self):
        with self._lock:
            if self._asr is None:
                from .m15_asr import WhisperASR

                t0 = time.perf_counter()
                d = self._hf_dir("asr")
                self._asr = WhisperASR(d, device=self.device, dtype=torch.bfloat16 if self.cuda else torch.float32,
                                       max_new_tokens=self.cfg["asr"]["max_new_tokens"])
                self.load_seconds["asr"] = time.perf_counter() - t0
        return self._asr

    def _get_ocr(self):
        with self._lock:
            if self._ocr is None:
                from .m15_ocr import RapidOCRBackend

                t0 = time.perf_counter()
                o = self.cfg["ocr"]
                self._ocr = RapidOCRBackend(o["variant"], device=self.device, rec_batch_num=o["rec_batch_num"],
                                            cudnn_conv_algo_search=o["cudnn_conv_algo_search"] if self.cuda else None,
                                            det_model_path=str(self.root / o["det_model"]), rec_model_path=str(self.root / o["rec_model"]))
                self.load_seconds["ocr"] = time.perf_counter() - t0
        return self._ocr

    def _get_fusion(self):
        with self._lock:
            if self._fusion is None:
                from safetensors.torch import load_file

                from .m1 import M1Fusion, M1Model
                from .m15 import M15Model

                t0 = time.perf_counter()
                f = self.cfg["fusion"]
                core = self.text.model
                fus = M1Fusion(core, f["vision_dim"], f["audio_dim"], tokens=f["tokens"], heads=f["heads"],
                               lora_r=f["lora_r"], lora_alpha=f["lora_alpha"]).to(self.device)
                sd = load_file(str(self.root / f["weights"]))
                fus.load_state_dict({k[len("fusion."):]: v for k, v in sd.items() if k.startswith("fusion.")})
                # serving: the core's parameters are left exactly as loaded (freeze_core=False); no optimizer, no backward
                self._fusion = M15Model(M1Model(core, fus, freeze_core=False), gate=False).to(self.device).eval()
                self.load_seconds["fusion"] = time.perf_counter() - t0
        return self._fusion

    # ---- validation
    def validate(self, ctx: ResolvedMultimodalContext) -> None:
        L, M = self.cfg["limits"], self.cfg["mime_types"]
        if not ctx.text and ctx.is_text_only:
            raise RuntimeInputError("empty_context", "context has no text and no media")
        for kind, items, lim in (("image", ctx.image_inputs, "max_images"), ("document", ctx.document_inputs, "max_documents"),
                                 ("audio", ctx.audio_inputs, "max_audio")):
            if len(items) > L[lim]:
                raise RuntimeInputError({"image": "too_many_images", "document": "too_many_documents", "audio": "too_many_audio"}[kind],
                                        f"{len(items)} {kind} inputs > {L[lim]}")
            for it in items:
                if it.mime_type not in M[kind]:
                    raise RuntimeInputError("unsupported_mime_type", f"{it.mime_type} is not a supported {kind} type", it.id)
                if it.size() > L[f"max_{kind}_bytes"]:
                    raise RuntimeInputError("asset_too_large", f"{kind} larger than {L[f'max_{kind}_bytes']} bytes", it.id)

    # ---- media -> (context text, media tokens)
    def _images(self, parts, T, meta):
        from PIL import Image

        from .features import ModalityFeatures

        out = []
        for p in parts:
            with T("decode_ms"):
                try:
                    im = Image.open(io.BytesIO(p.read())).convert("RGB")
                except Exception as e:  # noqa: BLE001
                    raise RuntimeInputError("decode_failed", f"image decode failed: {type(e).__name__}", p.id) from e
            v = self._get_vision()
            with T("vision_ms"):
                try:
                    mv = dict(self._ip(images=[im], max_num_patches=self.cfg["vision"]["patches"]["image"], return_tensors="pt"))
                    h, m = v.encode(mv)
                except Exception as e:  # noqa: BLE001
                    raise RuntimeInputError("vision_failed", f"vision tower failed: {type(e).__name__}", p.id) from e
            out.append(ModalityFeatures("image", h, m))
        meta["vision_used"] = meta["vision_used"] or bool(parts)
        return out

    def _pages(self, p) -> tuple[str, list[Any], str | None]:
        """-> (route, page images, extracted text). Text layer -> ("text", [], text); visual -> ("visual", pages, None)."""
        from PIL import Image

        from .config import DocumentConfig
        from .documents import DocumentInput, DocumentRouter

        data = p.read()
        L = self.cfg["limits"]
        if p.mime_type.startswith("image/"):
            try:
                return "visual", [Image.open(io.BytesIO(data)).convert("RGB")], None
            except Exception as e:  # noqa: BLE001
                raise RuntimeInputError("decode_failed", f"page image decode failed: {type(e).__name__}", p.id) from e
        if p.mime_type == "application/pdf":
            try:
                import pypdfium2 as pdfium

                n = len(pdfium.PdfDocument(data))
            except Exception as e:  # noqa: BLE001
                raise RuntimeInputError("decode_failed", f"pdf decode failed: {type(e).__name__}", p.id) from e
            if n > L["max_document_pages"]:
                raise RuntimeInputError("document_too_many_pages", f"{n} pages > {L['max_document_pages']}", p.id)
        router = DocumentRouter(DocumentConfig(enabled=True, min_chars=self.cfg["documents"]["min_chars_per_page"],
                                               max_pages=L["max_visual_pages"]))
        try:
            r = router.route(DocumentInput(data, p.mime_type))
        except Exception as e:  # noqa: BLE001
            raise RuntimeInputError("decode_failed", f"document decode failed: {type(e).__name__}", p.id) from e
        if r.route == "text":
            return "text", [], r.text
        if not r.pages:
            if r.text:                                      # e.g. a short DOCX: not renderable, but it has text
                return "text", [], r.text
            raise RuntimeInputError("decode_failed", "document has no text layer and no renderable page", p.id)
        return "visual", [pg.convert("RGB") for pg in r.pages[: L["max_visual_pages"]]], None

    def _documents(self, parts, T, meta, ctx_texts):
        from concurrent.futures import ThreadPoolExecutor

        from .features import ModalityFeatures

        feats = []
        for p in parts:
            with T("decode_ms"):
                route, pages, text = self._pages(p)
            meta["document_pages"] += max(1, len(pages))
            if route == "text":
                meta["document_text_layer"] += 1
                ctx_texts.append(text)
                continue
            meta["document_visual"] += 1
            ocr, v = self._get_ocr(), self._get_vision()
            with ThreadPoolExecutor(1) as pool:                  # OCR || SigLIP2 (validated in M1.5b / M1.5c)
                t_o = time.perf_counter()
                fut = pool.submit(lambda: [ocr.read([pg])[0] for pg in pages])
                with T("vision_ms"):
                    try:                                         # the fusion saw one page per decision: tokens of page 1
                        mv = dict(self._ip(images=[pages[0]], max_num_patches=self.cfg["vision"]["patches"]["document"], return_tensors="pt"))
                        h, m = v.encode(mv)
                    except Exception as e:  # noqa: BLE001
                        raise RuntimeInputError("vision_failed", f"vision tower failed: {type(e).__name__}", p.id) from e
                try:
                    texts = fut.result()
                except Exception as e:  # noqa: BLE001
                    raise RuntimeInputError("ocr_failed", f"ocr failed: {type(e).__name__}", p.id) from e
                T.t["ocr_ms"] = T.t.get("ocr_ms", 0.0) + (time.perf_counter() - t_o) * 1e3
            ctx_texts.append("\n".join(t for t in texts if t))
            feats.append(ModalityFeatures("document", h, m))
            meta["ocr_used"] = meta["vision_used"] = True
        return feats

    def _audio(self, parts, T, meta, ctx_texts):
        from .features import ModalityFeatures

        feats = []
        L = self.cfg["limits"]
        for p in parts:
            with T("decode_ms"):
                try:
                    w = decode_audio(p.read())
                except Exception as e:  # noqa: BLE001
                    raise RuntimeInputError("decode_failed", f"audio decode failed: {type(e).__name__}", p.id) from e
            dur = len(w) / 16000
            if dur > L["max_audio_seconds"]:
                raise RuntimeInputError("audio_too_long", f"{dur:.1f} s > {L['max_audio_seconds']} s", p.id)
            meta["audio_duration_ms"] += int(dur * 1000)
            meta["asr_truncation_risk"] = meta["asr_truncation_risk"] or dur > self.cfg["asr"]["recommended_max_seconds"]
            asr = self._get_asr()
            try:
                with T("asr_ms"):
                    feats_mel, n = asr.logmel([w])
                    enc = asr.encode(feats_mel)
                    text = asr.transcribe(enc)[0]
            except Exception as e:  # noqa: BLE001
                raise RuntimeInputError("asr_failed", f"asr failed: {type(e).__name__}", p.id) from e
            h, m = asr.acoustic(enc, n)
            ctx_texts.append(text)
            feats.append(ModalityFeatures("audio", h, m))
            meta["asr_used"] = True
        return feats

    # ---- decide
    def decide(self, ctx: ResolvedMultimodalContext, questions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        """-> {"model", "answers" (krun-worker v0.3 answer schema + multi), "usage", "metadata" (no content)}."""
        from ..model import chunk_index, collate_chunks, merge_chunks

        self.validate(ctx)
        T = Timer(self.cuda)
        meta = {"modalities_used": [], "image_count": len(ctx.image_inputs), "document_count": len(ctx.document_inputs),
                "audio_count": len(ctx.audio_inputs), "document_pages": 0, "document_text_layer": 0, "document_visual": 0,
                "audio_duration_ms": 0, "ocr_used": False, "asr_used": False, "vision_used": False, "asr_truncation_risk": False}
        t_all = time.perf_counter()
        if ctx.is_text_only:
            with T("decision_ms"):
                res = self.text.decide({"context": ctx.text, "questions": questions})
            meta["modalities_used"] = ["text"]
            meta["path"] = "text"
            meta["timings_ms"] = {**T.t, "total_inference_ms": (time.perf_counter() - t_all) * 1e3}
            return {**res, "metadata": meta}
        ctx_texts = [ctx.text] if ctx.text else []
        feats = []
        feats += self._images(ctx.image_inputs, T, meta)
        feats += self._documents(ctx.document_inputs, T, meta, ctx_texts)
        feats += self._audio(ctx.audio_inputs, T, meta, ctx_texts)
        meta["modalities_used"] = [m for m, on in (("text", bool(ctx.text)), ("image", bool(ctx.image_inputs)),
                                                    ("document", bool(ctx.document_inputs)), ("audio", bool(ctx.audio_inputs))) if on]
        context = "\n".join(t for t in ctx_texts if t)
        if not feats:                                        # only text-layer documents: the text path, fast
            meta["path"] = "text_fast_path"
            with T("decision_ms"):
                res = self.text.decide({"context": context, "questions": questions})
            meta["timings_ms"] = {**T.t, "total_inference_ms": (time.perf_counter() - t_all) * 1e3}
            return {**res, "metadata": meta}
        meta["path"] = "multimodal"
        model = self._get_fusion()
        try:
            with T("fusion_ms"), torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.cuda):
                toks, masks = zip(*[model.fusion.media_tokens(f) for f in feats])
                media = (torch.cat(toks, 1), torch.cat(masks, 1))
            qids = list(questions)
            enc = [self.text.encode(context, questions[k]) for k in qids]
            with T("decision_ms"), torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.cuda):
                flat, where = chunk_index([e for e, _ in enc])
                b = {k: v.to(self.device) for k, v in collate_chunks(flat, self.text.tok.pad_token_id).items()}
                o = model(b, media, [0] * len(flat), feats[0].modality)
                q = merge_chunks(o, where, len(enc), [e.n_candidates for e, _ in enc])
                rank = q["rank"].float().cpu().numpy()
                apply = q["apply"].float().cpu().numpy()
        except RuntimeInputError:
            raise
        except Exception as e:  # noqa: BLE001
            raise RuntimeInputError("multimodal_inference_failed", f"fusion / decision failed: {type(e).__name__}") from e
        answers = {}
        primary = feats[-1].modality if ctx.audio_inputs else feats[0].modality
        for i, k in enumerate(qids):
            n = enc[i][0].n_candidates
            a = self.text.answer(questions[k], enc[i][1], rank[i, :n], apply[i, :n])
            if a["type"] == "multi":
                th = self.cfg["calibration"]["multi_threshold"][primary]
                s = 1 / (1 + np.exp(-apply[i, :n].astype(np.float64)))
                ids = [o_["id"] for o_ in questions[k]["options"]]
                a.update({"selected": [x for x, v in zip(ids, s) if v > th], "threshold": th})
            if a["type"] == "choice":                         # media decisions: the text calibration was not refit on media
                a["abstention_status"] = "advisory"
            a["model"] = self.cfg["model_id"]
            answers[k] = a
        tokens = int(sum(len(c.ids) for e, _ in enc for c in e.chunks))
        meta["media_tokens"] = int(media[1].sum())
        meta["timings_ms"] = {**T.t, "total_inference_ms": (time.perf_counter() - t_all) * 1e3}
        return {"model": self.cfg["model_id"], "answers": answers, "usage": {"input_tokens": tokens}, "metadata": meta}


def load_runtime(root: str | Path, **kw) -> KrunOneMultimodalRuntime:
    return KrunOneMultimodalRuntime.load(root, **kw)

