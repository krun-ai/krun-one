# Krun One v1 multimodal — worker integration contract (2026-09-29)

**Status: PROPOSAL.** Nothing here is implemented in the worker, the API (krun-api-app / Cloudflare), the RunPod
serving worker, the SDKs or production. This document defines the boundary so the worker refactor can proceed in
parallel. The model side is ready: `artifacts/krun-one-v1-multimodal-candidate` +
`krun_one.v1.multimodal.runtime.KrunOneMultimodalRuntime`.

**Principle.** Every modality is input-only: text, image, document, audio. The response is always the existing
structured decision. A modality is evidence; a question type is the decision. There are no per-modality primitives and
no public OCR / ASR / vision switches.

## 1. Public API proposal (draft: `docs/openapi-multimodal-draft.yaml`, valid OpenAPI 3.1)

`POST /v1/decide`, same path, same auth, same response. The only request change is:

```text
context: string                      # unchanged (1–8,000 chars)
       | ContentPart[]               # 1–16 parts, in order
ContentPart = TextPart | ImagePart | DocumentPart | AudioPart     (discriminator: type)
```

| part | fields | notes |
|---|---|---|
| `TextPart` | `type: "text"`, `text` (1–8,000), `id?` | text parts are concatenated with newlines |
| `ImagePart` | `type: "image"`, `asset_id`, `id?` | PNG / JPEG / WebP |
| `DocumentPart` | `type: "document"`, `asset_id`, `id?` | PDF / DOCX / text / Markdown / HTML, or a page image |
| `AudioPart` | `type: "audio"`, `asset_id`, `id?` | WAV / FLAC / MP3 / OGG |

**Schema rules:**
* `asset_id` matches `^asset_[A-Za-z0-9_-]{8,64}$`.
* `additionalProperties: false` on every part: no `use_ocr`, no `url`, no inline bytes.
* `video` is rejected by the schema.

**Questions.** `choice`, `noul` and `score` are byte-identical to today's schemas. `multi` is an optional compatible
extension:
* request: `MultiQuestion {type: "multi", options, instructions?}`;
* answer: `MultiAnswer {type: "multi", values: [ids], probabilities: {id: p}}` (independent applicabilities).

**Error codes.** The draft adds UPPER_SNAKE codes (section 6).

**Backward compatibility.** The legacy string `context` and every current primitive are unchanged. This is tested in
`tests/v1/test_multimodal_contract.py`: legacy requests validate against both the current spec and the draft, and the
existing question / answer schemas are identical dictionaries.

Examples:

```json
{"context": "Customer says their card was charged twice.",
 "questions": {"route": {"type": "choice", "options": {"billing": null, "fraud": null}}}}
```
```json
{"context": [{"type": "text", "text": "Is this invoice paid?"},
             {"type": "document", "asset_id": "asset_7fQ2mZkP0aLx"}],
 "questions": {"paid": {"type": "noul", "instructions": "Is the document marked as paid?"},
               "tags": {"type": "multi", "options": {"invoice": null, "receipt": null, "overdue": null}}}}
```

**Asset ids.** `asset_id` is the canonical media reference. The upload mechanism (for example `POST /v1/assets`
returning `asset_id`, MIME type, size and TTL) is **future work** and is not specified in the draft.

**Remote URLs.** Remote URL parts are **not** accepted (owner preference), for security (SSRF, unbounded downloads),
predictability and download control. URL ingestion can be added later as a separate part type without breaking this
contract.

## 2. Internal resolved contract (what the runtime consumes)

```python
from krun_one.v1.multimodal.runtime import ResolvedMultimodalContext, TextPart, MediaPart

ResolvedMultimodalContext(
    text_parts=[TextPart(text="Is this invoice paid?", id="p0")],
    image_inputs=[],                                    # MediaPart(mime_type, path=... | data=bytes, id=...)
    document_inputs=[MediaPart("application/pdf", path="/tmp/req123/p1.pdf", id="p1")],
    audio_inputs=[],
)
```

**What the runtime receives from the worker:**
* bytes or a local path, plus the **validated** MIME type (from the asset record, not from sniffing user input);
* the part `id`, which is only used in error messages.

**What the runtime never sees:** `asset_id`, URLs, auth, org / project ids, storage providers, D1 / KV, RunPod
job ids.

The worker maps `context: string` to `ResolvedMultimodalContext.from_text(context)`. That call runs the frozen text
path and is bitwise identical to the text artifact.

## 3. Runtime entrypoint

```python
rt = KrunOneMultimodalRuntime.load(artifact_dir, device="cuda", dtype="bf16", verify=True)   # once per process
out = rt.decide(ctx: ResolvedMultimodalContext, questions: dict[str, dict]) -> dict
```

`questions` uses the krun-worker internal shape:
* `choice` / `multi`: `{"type", "options": [{"id", "description"?}], "task_type"?, "instructions"?}`;
* `noul`: `{"type": "noul", "instructions", "criteria"?}`;
* `score`: `{"type": "score", "instructions", "levels": [...]}`.

The worker already converts the public `options` object into this list.

`out`:

```json
{"model": "krun-one-v1-multimodal-candidate",
 "answers": {"q": {"type": "choice", "decision": "invoice", "top_option": "invoice", "probabilities": {"invoice": 0.97, "receipt": 0.02},
                   "confidence": 0.95, "margin": 0.95, "abstain": false, "abstention_status": "advisory", "...": "..."}},
 "usage": {"input_tokens": 312},
 "metadata": {"path": "multimodal", "modalities_used": ["text", "document"], "document_pages": 1, "ocr_used": true, "...": "...",
              "timings_ms": {"decode_ms": 3.1, "ocr_ms": 55.2, "vision_ms": 12.0, "fusion_ms": 0.9, "decision_ms": 8.1, "total_inference_ms": 80.4}}}
```

**Answer mapping.** Answers use the krun-worker v0.3 answer fields, so the existing public mapping applies (`decision`
→ `choice`, etc.). For `multi` the worker maps:
* `selected` → `values`;
* `scores` → `probabilities`.

**Abstention on media.** Media choices abstain with the frozen text thresholds and are marked `advisory`.

**Routing** (decided by the runtime, never by the caller):

| path | when |
|---|---|
| `text` | text-only request |
| `text_fast_path` | only text-layer documents |
| `multimodal` | any image, any visual document, any audio |

**Serving requirements (from M1 / M1.5):**
* Serve with the core **untouched**: `M1Model(..., freeze_core=False)`; flipping `requires_grad` changes bf16 numerics.
* Run `inference_mode`; there is never a backward pass.
* Use onnxruntime-gpu 1.22 on CUDA-12 hosts (preloading the torch CUDA / cuDNN libraries). Blackwell (CUDA 13) is not
  validated.

## 4. Responsibilities

| concern | worker / API (future) | runtime (ready) |
|---|---|---|
| auth, org / project, billing, rate limits | ✅ | — |
| `asset_id` → object (existence, ownership, TTL) | ✅ | — |
| download / storage, temp files, cleanup | ✅ | — |
| transport limits (bytes per part, parts per request, total) | ✅ (first line) | ✅ (re-checked: defense in depth) |
| MIME validation (declared vs. magic bytes) | ✅ | ✅ (allow-list per modality) |
| decode (image / PDF / DOCX / audio), page count, duration | — | ✅ |
| text-layer vs. visual routing, OCR, ASR, towers, fusion, decision | — | ✅ |
| error mapping to HTTP / public `ErrorCode` | ✅ | typed errors with category |
| logging / metrics | ✅ (allowed fields only) | returns content-free metadata |

## 5. Limits (runtime config; proposed public limits)

| limit | value | note |
|---|---|---|
| context parts | ≤ 16 | draft schema |
| text part | 1–8,000 chars | unchanged |
| images | ≤ 4 per request, ≤ 10 MB each | quality validated for 1 image per decision |
| documents | ≤ 2 per request, ≤ 25 MB, ≤ 20 pages | a visual document: OCR on ≤ 4 rendered pages; vision tokens from page 1 |
| audio | 1 per request, ≤ 10 MB, **≤ 30 s hard** | **≤ 20 s recommended**: longer speech may be transcribed partially |
| video | unsupported | rejected |

**MIME types, all tested with real fixtures:**

| modality | MIME types |
|---|---|
| image | `image/png`, `image/jpeg`, `image/webp` |
| document | `application/pdf`, `application/vnd.openxmlformats-officedocument.wordprocessingml.document`, `text/plain`, `text/markdown`, `text/html`, and page images (`image/png`, `image/jpeg`, `image/webp`) |
| audio | `audio/wav` (`audio/x-wav`), `audio/flac`, `audio/mpeg`, `audio/ogg` (Vorbis) |

Exercised by the fixtures: JPEG, text-layer PDF, image-only PDF, DOCX, WAV, MP3, FLAC and OGG. PNG and WebP go
through the same PIL path; WebP is not yet exercised by a fixture.

## 6. Errors

| runtime code | category | proposed public `ErrorCode` | suggested HTTP (not implemented) |
|---|---|---|---|
| `empty_context` | client | `INVALID_REQUEST` | 400 |
| `unsupported_modality` | client | `UNSUPPORTED_MODALITY` | 400 |
| `unsupported_mime_type` | client | `UNSUPPORTED_MIME_TYPE` | 415 |
| — (worker) | client | `ASSET_NOT_FOUND` | 404 |
| `asset_too_large` | client | `ASSET_TOO_LARGE` | 413 |
| `too_many_images` / `too_many_documents` / `too_many_audio` | client | `TOO_MANY_IMAGES` / `TOO_MANY_DOCUMENTS` / `TOO_MANY_AUDIO` | 400 |
| `document_too_many_pages` | client | `DOCUMENT_TOO_MANY_PAGES` | 400 |
| `audio_too_long` | client | `AUDIO_TOO_LONG` | 400 |
| `decode_failed` | client | `DECODE_FAILED` | 422 |
| `ocr_failed` | runtime | `OCR_FAILED` | 502 |
| `asr_failed` | runtime | `ASR_FAILED` | 502 |
| `vision_failed` | runtime | `VISION_FAILED` | 502 |
| `artifact_integrity` | runtime | `INTERNAL_ERROR` | 500 (worker must not start) |
| `multimodal_inference_failed` | model | `MULTIMODAL_INFERENCE_FAILED` | 500 |

`RuntimeInputError.to_dict()` gives `{code, category, message, part_id}`. Messages never contain content, only types
and sizes.

## 7. Observability (allowed fields)

**Allowed, taken from `metadata`:**

| kind | fields |
|---|---|
| request shape | `modality_count` (derived), `image_count`, `document_count`, `audio_count` |
| document | `document_pages`, `document_text_layer`, `document_visual` |
| audio | `audio_duration_ms`, `asr_truncation_risk` |
| components used | `ocr_used`, `asr_used`, `vision_used`, `path`, `media_tokens` |
| timings | `decode_ms` (preprocess), `ocr_ms`, `asr_ms`, `vision_ms`, `fusion_ms`, `decision_ms`, `total_inference_ms` |
| usage | `usage.input_tokens` |

**Privacy: never log, store or return:**
* raw context text;
* OCR text or ASR transcripts;
* images, audio or raw documents, or their bytes;
* file names or page renders;
* per-option probabilities in logs, unless the existing policy already allows them.

**The runtime enforces its side:**
* OCR / ASR text never leaves it: not in answers, not in metadata, not in errors;
* `tests/v1/test_multimodal_runtime.py::test_metadata_has_no_content` asserts this.

## 8. Resource profile (RTX 4090, bf16)

The measured numbers are in `results/v1/multimodal_packaging/KRUN_ONE_V1_MULTIMODAL_PACKAGING_REPORT.md`:
* memory and load time per configuration: text-only / +vision / +audio / full;
* latency per path.

Summary:
* Text-only memory equals the text worker's: no multimodal weights are loaded until the first media request.
* The first request of each modality pays that component's lazy load. The worker can preload on start if cold-start
  latency matters more than memory.

## 9. SDK examples (DRAFT — nothing published)

Python:

```python
client.decide(
    context=[
        {"type": "text", "text": "Is this invoice paid?"},
        {"type": "document", "asset_id": asset.id},          # asset upload API: future
    ],
    questions={"paid": {"type": "noul", "instructions": "Is the document marked as paid?"}},
)
client.decide(context="legacy text still works", questions={...})
```

TypeScript:

```ts
await krun.decide({
  context: [
    { type: "text", text: "Qual a intenção do cliente?" },
    { type: "audio", asset_id: audio.id },
  ],
  questions: { intent: { type: "choice", options: { cancel: null, billing: null } } },
});
```

## 10. Out of scope (not done in this round)

The following were not built:
* Cloudflare worker changes;
* API handlers;
* asset endpoint / storage / signed URLs;
* RunPod worker integration or Docker changes;
* canary and deployment;
* streaming;
* video;
* M2.

**Open owner decisions:**
* asset TTL and retention;
* whether and how media consumes usage / billing (the fusion adds 32 tokens per media item plus OCR / ASR text
  tokens, which already appear in `input_tokens`);
* URL ingestion;
* preload vs. lazy-load policy on workers.
