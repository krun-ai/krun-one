"""Generate docs/openapi-multimodal-draft.yaml from the CURRENT public OpenAPI (krun-api-app/openapi.json, read-only).

A DRAFT only: nothing in krun-api-app, the Cloudflare worker, the SDKs or production uses it. Every existing schema is
copied unchanged except:
* DecideRequest.context: string (unchanged semantics) | ContentPart[]  (TextPart / ImagePart / DocumentPart / AudioPart)
* Question / Answer: + MultiQuestion / MultiAnswer (compatible extension; existing primitives untouched)
* ErrorCode: + proposed multimodal codes
No public switches (use_ocr / use_asr / use_vision), no per-modality primitives, no video.

    python scripts/v1/make_openapi_multimodal_draft.py [--check]
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT.parents[1] / "krun-api-app/openapi.json"
OUT = ROOT / "docs/openapi-multimodal-draft.yaml"
ASSET = {"type": "string", "pattern": "^asset_[A-Za-z0-9_-]{8,64}$",
         "description": "Opaque id of an uploaded asset (future asset mechanism, e.g. `POST /v1/assets`). The asset carries its MIME "
                        "type and size; the API resolves it before inference. Remote URLs are NOT accepted in this contract "
                        "(asset_id is canonical; URL ingestion is a possible future extension)."}
PART_ID = {"type": ["string", "null"], "maxLength": 100, "description": "Optional caller-chosen id of the part (echoed only in errors)."}
NEW_ERRORS = ["UNSUPPORTED_MODALITY", "UNSUPPORTED_MIME_TYPE", "ASSET_NOT_FOUND", "ASSET_TOO_LARGE", "TOO_MANY_IMAGES",
              "TOO_MANY_DOCUMENTS", "TOO_MANY_AUDIO", "DOCUMENT_TOO_MANY_PAGES", "AUDIO_TOO_LONG", "DECODE_FAILED",
              "OCR_FAILED", "ASR_FAILED", "VISION_FAILED", "MULTIMODAL_INFERENCE_FAILED"]


def build() -> dict:
    d = json.loads(SRC.read_text())
    d = copy.deepcopy(d)
    normalized = []
    for path, ops in d["paths"].items():            # the source spec lacks some response descriptions (known upstream issue)
        for op, spec in ops.items():
            for code, r in (spec.get("responses") or {}).items():
                if isinstance(r, dict) and "description" not in r and "$ref" not in r:
                    r["description"] = f"{code} response."
                    normalized.append(f"{op.upper()} {path} {code}")
    d["info"]["title"] = "Krun API — multimodal DRAFT"
    d["info"]["version"] = "1.1.0-multimodal-draft"
    d["info"]["description"] = (d["info"]["description"] + "\n\n**DRAFT — not implemented.** Proposed backward-compatible extension "
                                "for Krun One v1 multimodal input (docs/KRUN_ONE_V1_WORKER_INTEGRATION_CONTRACT.md). All modalities "
                                "are input-only: every answer is the existing structured decision. `context` keeps accepting a "
                                "string. Media is referenced by `asset_id` (the upload mechanism is future work and is not part of "
                                "this draft). Video is unsupported.")
    s = d["components"]["schemas"]
    legacy_ctx = s["DecideRequest"]["properties"]["context"]
    s["ContextString"] = {**legacy_ctx, "minLength": 1, "maxLength": 8000}
    s["TextPart"] = {"type": "object", "required": ["type", "text"], "additionalProperties": False,
                     "properties": {"id": PART_ID, "type": {"type": "string", "enum": ["text"]},
                                    "text": {"type": "string", "minLength": 1, "maxLength": 8000}}}
    for kind, desc in (("Image", "PNG / JPEG / WebP, max 10 MB. Up to 4 images per request."),
                       ("Document", "PDF / DOCX / plain text / Markdown / HTML, or a page image (PNG / JPEG / WebP); max 25 MB, max 20 "
                                    "pages. Documents with a text layer are read as text (no OCR); scans / photos / screenshots go "
                                    "through OCR + vision (at most 4 rendered pages). Up to 2 documents per request."),
                       ("Audio", "WAV / FLAC / MP3 / OGG, max 10 MB, max 30 s (<= 20 s recommended: longer speech may be "
                                 "transcribed only partially). One audio part per request.")):
        s[f"{kind}Part"] = {"type": "object", "required": ["type", "asset_id"], "additionalProperties": False, "description": desc,
                            "properties": {"id": PART_ID, "type": {"type": "string", "enum": [kind.lower()]}, "asset_id": ASSET}}
    s["ContentPart"] = {"oneOf": [{"$ref": f"#/components/schemas/{k}Part"} for k in ("Text", "Image", "Document", "Audio")],
                        "description": "One piece of evidence. The modality is evidence; the question type is the decision.",
                        "discriminator": {"propertyName": "type", "mapping": {k.lower(): f"#/components/schemas/{k}Part"
                                                                              for k in ("Text", "Image", "Document", "Audio")}}}
    s["DecideRequest"]["properties"]["context"] = {
        "oneOf": [{"$ref": "#/components/schemas/ContextString"},
                  {"type": "array", "minItems": 1, "maxItems": 16, "items": {"$ref": "#/components/schemas/ContentPart"}}],
        "description": "Either the legacy text (unchanged) or an ordered list of content parts. Text parts are concatenated; "
                       "at least one part is required."}
    s["MultiQuestion"] = {"type": "object", "required": ["type", "options"], "additionalProperties": False,
                          "description": "Select every option that applies (independent per-option decisions). Extension.",
                          "properties": {"type": {"type": "string", "enum": ["multi"]},
                                         "options": copy.deepcopy(s["ChoiceQuestion"]["properties"]["options"]),
                                         "instructions": {"type": ["string", "null"], "maxLength": 1000}}}
    s["MultiAnswer"] = {"type": "object", "required": ["type", "values", "probabilities"],
                        "description": "Answer to a `multi` question.",
                        "properties": {"type": {"type": "string", "enum": ["multi"]},
                                       "values": {"type": "array", "items": {"type": "string"},
                                                  "description": "Selected option ids, in request order (may be empty)."},
                                       "probabilities": {"type": "object", "additionalProperties": {"type": "number", "format": "double"},
                                                         "propertyNames": {"type": "string"},
                                                         "description": "Per-option applicability in [0, 1] (independent, not a "
                                                                        "distribution)."}}}
    for name, extra in (("Question", "MultiQuestion"), ("Answer", "MultiAnswer")):
        s[name]["oneOf"].append({"$ref": f"#/components/schemas/{extra}"})
        s[name]["discriminator"]["mapping"]["multi"] = f"#/components/schemas/{extra}"
    if "ErrorCode" in s and "enum" in s["ErrorCode"]:
        s["ErrorCode"]["enum"] = list(s["ErrorCode"]["enum"]) + [e for e in NEW_ERRORS if e not in s["ErrorCode"]["enum"]]
        s["ErrorCode"]["description"] = (s["ErrorCode"].get("description", "") + " DRAFT additions: " + ", ".join(NEW_ERRORS)).strip()
    d["x-krun-draft"] = {"status": "draft", "implemented": False, "source": "krun-api-app/openapi.json (unchanged, read-only)",
                         "normalized_in_draft_only": {"missing_response_descriptions_added": normalized,
                                                      "note": "the source spec is not valid OpenAPI 3.1 because of these; fix in krun-api-app"},
                         "future": ["asset upload (e.g. POST /v1/assets) — not specified here", "optional remote URL parts"]}
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    d = build()
    OUT.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True, width=120))
    if a.check:
        from openapi_spec_validator import validate

        validate(yaml.safe_load(OUT.read_text()))
        print("openapi draft valid")


if __name__ == "__main__":
    main()
