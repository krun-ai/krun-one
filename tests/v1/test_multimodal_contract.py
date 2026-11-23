"""Proposed public contract (docs/openapi-multimodal-draft.yaml): backward compatibility + the ContentPart extension.

The draft is generated from krun-api-app/openapi.json (read-only). Every legacy request must stay valid, every existing
question / answer schema must be byte-identical, and the new parts must be strict (no public OCR / ASR switches,
no URLs, no video).
"""

import copy
import json
from pathlib import Path

import pytest
import yaml

jsonschema = pytest.importorskip("jsonschema")

ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / "docs/openapi-multimodal-draft.yaml"
CURRENT = ROOT.parents[1] / "krun-api-app/openapi.json"


def validator(doc, name):
    schema = copy.deepcopy(doc)
    schema["$ref"] = f"#/components/schemas/{name}"
    return jsonschema.Draft202012Validator(schema)


@pytest.fixture(scope="module")
def draft():
    return yaml.safe_load(DRAFT.read_text())


@pytest.fixture(scope="module")
def current():
    if not CURRENT.exists():
        pytest.skip("krun-api-app/openapi.json not present")
    return json.loads(CURRENT.read_text())


LEGACY = {"context": "Customer says their card was charged twice.",
          "questions": {"intent": {"type": "choice", "options": {"billing": "Billing and payment issues", "fraud": None}},
                        "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
                        "severity": {"type": "score", "instructions": "How severe is it?", "levels": ["low", "medium", "high"]}}}


def test_legacy_request_valid_in_current_and_draft(draft, current):
    assert not list(validator(current, "DecideRequest").iter_errors(LEGACY))
    assert not list(validator(draft, "DecideRequest").iter_errors(LEGACY))


def test_existing_schemas_unchanged(draft, current):
    for name in ("ChoiceQuestion", "NoulQuestion", "ScoreQuestion", "ChoiceAnswer", "NoulAnswer", "ScoreAnswer", "DecideResponse",
                 "Usage", "FeedbackRequest", "AbstentionStatus", "TaskType", "NoulCriteria"):
        assert draft["components"]["schemas"][name] == current["components"]["schemas"][name], name
    assert set(current["paths"]) == set(draft["paths"])            # no new endpoint (asset upload is future work)


@pytest.mark.parametrize("context", [
    [{"type": "text", "text": "quero cancelar"}],
    [{"type": "text", "text": "o que diz a fatura?"}, {"type": "document", "asset_id": "asset_0123456789ab"}],
    [{"id": "img1", "type": "image", "asset_id": "asset_abcdefgh"}, {"type": "image", "asset_id": "asset_abcdefgi"}],
    [{"type": "audio", "asset_id": "asset_audio_0001"}],
])
def test_content_parts_valid(draft, context):
    req = dict(LEGACY, context=context)
    assert not list(validator(draft, "DecideRequest").iter_errors(req))


@pytest.mark.parametrize("context", [
    [],                                                                           # empty
    [{"type": "video", "asset_id": "asset_abcdefgh"}],                            # video unsupported
    [{"type": "image", "asset_id": "asset_abcdefgh", "use_ocr": True}],           # no public OCR / ASR / vision switches
    [{"type": "image", "url": "https://example.com/x.png"}],                      # asset_id is canonical; no URLs
    [{"type": "image"}],                                                          # asset_id required
    [{"type": "image", "asset_id": "../etc/passwd"}],                             # opaque id pattern
    [{"type": "text", "text": ""}],                                               # empty text part
    "",                                                                           # empty legacy string
])
def test_content_parts_invalid(draft, context):
    req = dict(LEGACY, context=context)
    assert list(validator(draft, "DecideRequest").iter_errors(req))


def test_multi_extension(draft):
    q = {"type": "multi", "options": {"billing": None, "refund": "Refund request"}, "instructions": "Select every topic."}
    req = dict(LEGACY, questions={"tags": q})
    assert not list(validator(draft, "DecideRequest").iter_errors(req))
    ans = {"type": "multi", "values": ["billing"], "probabilities": {"billing": 0.97, "refund": 0.2}}
    assert not list(validator(draft, "Answer").iter_errors(ans))
    assert list(validator(draft, "Question").iter_errors({"type": "image_choice", "options": {"a": None}}))   # no per-modality primitives


def test_error_codes_extend_existing(draft, current):
    old, new = current["components"]["schemas"]["ErrorCode"]["enum"], draft["components"]["schemas"]["ErrorCode"]["enum"]
    assert new[: len(old)] == old
    assert {"UNSUPPORTED_MIME_TYPE", "AUDIO_TOO_LONG", "DOCUMENT_TOO_MANY_PAGES", "OCR_FAILED", "ASR_FAILED"} <= set(new)


def test_draft_is_valid_openapi(draft):
    osv = pytest.importorskip("openapi_spec_validator")
    osv.validate(draft)
