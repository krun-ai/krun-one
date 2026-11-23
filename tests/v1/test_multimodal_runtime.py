"""KrunOneMultimodalRuntime against the packaged candidate artifact and the versioned fixtures.

Heavy (loads the 879M text core + towers). Runs on CPU fp32 by default; KRUN_MM_DEVICE=cuda runs the same tests
against expected_cuda.json (bf16). Skipped when the artifact or fixtures are absent.

* text-only answers are bitwise identical to a freshly loaded frozen text artifact (the text path is untouched);
* lazy loading: nothing multimodal is loaded by text traffic or by text-layer documents;
* golden: decisions / abstentions / multi sets identical to the reference, probabilities within tolerance;
* privacy: metadata carries no content (no OCR / ASR text, no user text);
* errors: typed, with the client / runtime / model category.
"""

import io
import json
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/krun-one-v1-multimodal-candidate"
FX = ROOT / "tests/fixtures/multimodal"
DEVICE = os.environ.get("KRUN_MM_DEVICE", "cpu")
EXPECTED = FX / f"expected_{DEVICE}.json"
TOL = {"cpu": 2e-3, "cuda": 5e-2}[DEVICE]

pytestmark = pytest.mark.skipif(not (ART / "runtime_config.json").exists() or not EXPECTED.exists(),
                                reason="multimodal artifact / fixtures not present")


@pytest.fixture(scope="module")
def rt():
    import torch

    from krun_one.v1.multimodal.runtime import KrunOneMultimodalRuntime

    if DEVICE == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    return KrunOneMultimodalRuntime.load(ART, device=DEVICE, dtype="fp32" if DEVICE == "cpu" else "bf16", verify=True)


def fixtures():
    import sys

    sys.path.insert(0, str(ROOT / "scripts/v1"))
    import make_multimodal_fixtures as F

    return F


def test_lazy_loading_text_and_text_layer(rt):
    F = fixtures()
    assert rt.loaded == {"vision": False, "asr": False, "ocr": False, "fusion": False}
    for name in ("text_intent_pt", "doc_text_layer_pdf", "doc_docx_letter"):
        c = next(c for c in F.cases() if c["name"] == name)
        rt.decide(F.resolve(c), c["questions"])
    assert rt.loaded == {"vision": False, "asr": False, "ocr": False, "fusion": False}


def test_text_only_bitwise_vs_frozen_text_artifact(rt):
    from krun_one.v1.multimodal.runtime import ResolvedMultimodalContext
    from krun_one.v1.runtime import TextArtifact

    ref = TextArtifact.load(ROOT / "artifacts/krun-one-v1-text-final", device=DEVICE, dtype="fp32" if DEVICE == "cpu" else "bf16")
    cases = json.loads((ROOT / "artifacts/krun-one-v1-text-final/reference/api_cases.json").read_text())
    for name, req in cases.items():
        got = rt.decide(ResolvedMultimodalContext.from_text(req["context"]), req["questions"])
        exp = ref.decide(req)
        assert json.dumps(got["answers"], sort_keys=True) == json.dumps(exp["answers"], sort_keys=True), name
        assert got["usage"] == exp["usage"]
        assert got["metadata"]["path"] == "text"


def test_golden(rt):
    F = fixtures()
    exp = json.loads(EXPECTED.read_text())["cases"]
    for c in F.cases():
        got = rt.decide(F.resolve(c), c["questions"])
        e = exp[c["name"]]
        for k, a in got["answers"].items():
            x = e["answers"][k]
            for f in ("type", "decision", "top_option", "abstain", "abstention_status", "selected"):
                assert a.get(f) == x.get(f), (c["name"], k, f, a.get(f), x.get(f))
            for f in ("noul", "score"):
                if f in x:
                    assert abs(a[f] - x[f]) <= TOL, (c["name"], k, f)
            for f in ("probabilities", "scores"):
                if f in x:
                    assert max(abs(a[f][i] - x[f][i]) for i in x[f]) <= TOL, (c["name"], k, f)
        for f, v in e["metadata"].items():
            assert got["metadata"][f] == v, (c["name"], f)


def test_known_correct_decisions(rt):
    """Beyond stability: the fixture cases whose answer is unambiguous must be RIGHT (top option)."""
    exp = json.loads(EXPECTED.read_text())["cases"]
    want = {("text_intent_pt", "intent"): "cancel_subscription", ("doc_docx_letter", "intent"): "cancel_subscription",
            ("image_elephant", "main"): "elephant", ("doc_text_layer_pdf", "doctype"): "invoice",
            ("doc_scan_jpg", "doctype"): "invoice", ("doc_scan_pdf", "doctype"): "invoice",
            ("audio_en", "intent"): "cancel_subscription", ("audio_pt", "intent"): "cancel_subscription",
            ("audio_es", "intent"): "check_balance", ("audio_fr", "intent"): "book_flight"}
    for (case, q), top in want.items():
        assert exp[case]["answers"][q]["top_option"] == top, (case, q)
    assert "giraffe" in exp["image_giraffe_multi"]["answers"]["objects"]["selected"]


def test_lazy_loading_after_media(rt):
    assert rt.loaded["vision"] and rt.loaded["fusion"] and rt.loaded["ocr"] and rt.loaded["asr"]


def test_metadata_has_no_content(rt):
    F = fixtures()
    truth = json.loads((FX / "doc_invoice_truth.json").read_text())["text"]
    secrets = [w for w in truth.split() if len(w) > 6] + ["cancelar", "assinatura", "subscription", "saldo", "Paris", "reembolso"]
    allowed = {"modalities_used", "image_count", "document_count", "audio_count", "document_pages", "document_text_layer",
               "document_visual", "audio_duration_ms", "ocr_used", "asr_used", "vision_used", "asr_truncation_risk", "path",
               "timings_ms", "media_tokens"}
    for c in F.cases():
        m = rt.decide(F.resolve(c), c["questions"])["metadata"]
        assert set(m) <= allowed, set(m) - allowed
        blob = json.dumps(m)
        assert not any(s in blob for s in secrets), c["name"]


def _wav(seconds, sr=16000):
    import soundfile as sf

    b = io.BytesIO()
    sf.write(b, np.zeros(int(seconds * sr), np.float32), sr, format="WAV")
    return b.getvalue()


def test_errors(rt):
    from PIL import Image

    from krun_one.v1.multimodal.runtime import MediaPart, ResolvedMultimodalContext, RuntimeInputError

    q = {"q": {"type": "noul", "instructions": "Is it urgent?"}}
    img = (FX / "image_giraffe.jpg").read_bytes()
    pages = io.BytesIO()
    blank = Image.new("RGB", (200, 200), (255, 255, 255))
    blank.save(pages, "PDF", save_all=True, append_images=[blank] * 20)
    cases = [
        (ResolvedMultimodalContext(), "empty_context", "client"),
        (ResolvedMultimodalContext(image_inputs=[MediaPart("image/png", data=img)] * 5), "too_many_images", "client"),
        (ResolvedMultimodalContext(image_inputs=[MediaPart("video/mp4", data=b"x")]), "unsupported_mime_type", "client"),
        (ResolvedMultimodalContext(image_inputs=[MediaPart("image/png", data=b"\0" * (11 * 2**20))]), "asset_too_large", "client"),
        (ResolvedMultimodalContext(image_inputs=[MediaPart("image/png", data=b"not an image")]), "decode_failed", "client"),
        (ResolvedMultimodalContext(audio_inputs=[MediaPart("audio/wav", data=_wav(31))]), "audio_too_long", "client"),
        (ResolvedMultimodalContext(document_inputs=[MediaPart("application/pdf", data=pages.getvalue())]), "document_too_many_pages", "client"),
    ]
    for ctx, code, cat in cases:
        with pytest.raises(RuntimeInputError) as ei:
            rt.decide(ctx, q)
        assert ei.value.code == code and ei.value.category == cat, (code, ei.value.code)
