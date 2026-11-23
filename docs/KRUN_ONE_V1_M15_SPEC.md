# Krun One v1 — M1.5 hybrid multimodal: routing, I/O contract, streaming interface

## I/O contract

Protocol: `configs/v1/m15.yaml` `io_contract`.

**All modalities are input-only:** audio, image, visual document and screenshot.

**Output.** The output is always the existing structured textual decision response:
* the primitives are `choice`, `noul` and `score`, plus `multi` (internal);
* the schema is the krun-worker v0.3 answer schema;
* the calibration semantics are the frozen text semantics.

**What is never output:**
* No audio, images or free text are generated.
* ASR transcripts and OCR text are internal inputs to the text path, never outputs. At most they appear as a debug /
  provenance field, off by default.

## Routing (the frozen text core is the semantic centre)

```
text only                     -> core(**batch)                                  (frozen artifact, bitwise identical)
document with a text layer    -> extraction (pypdf / docx / html)  -> text fast path (no OCR, no tower)
scan / photo / screenshot     -> OCR (RapidOCR PP-OCRv5 latin)     -> context text  ┐
                              -> SigLIP2 (1024 patches) -> M1 image adapter -> 32 tokens ┴-> M1.5 fusion -> decision
natural image                 -> SigLIP2 (256 patches)  -> M1 image adapter -> 32 tokens -> M1.5 fusion -> decision
audio                         -> log-mel (GPU) -> Whisper-small encoder (ONE pass)
                                   ├-> Whisper decoder (greedy, language auto) -> transcript -> context text ┐
                                   └-> valid frames -> M1 audio adapter -> 32 tokens ---------------------┴-> M1.5 fusion
```

## M1.5 fusion (`src/krun_one/v1/multimodal/m15.py`)

The fusion computes:

```
z = z_text + α·(z_fused − z_text)
```

* `z_text` is the frozen head on [question + candidates + transcript / OCR text].
* `z_fused` is the same encoder output plus the media tokens, run through the head with the M1 LoRA.
* `α` is a small MLP gate, one per group (audio / visual), on [h_CLS ; mean(media)].

Measured: the trained gates **saturated at α ≈ 1**, so the deployed decision is `z_fused`. See the report.

## Streaming (NOT implemented; interface kept possible)

```
client audio chunks (e.g. 1 s) -> incremental log-mel -> Whisper encoder on the growing window (≤ 30 s)
      -> partial transcript (decoder, previous tokens as prompt) -> request with context = partial transcript
      -> incremental Krun decision (same answer schema; flag "partial": true) -> final decision on end-of-audio
```

**No API change is needed.**
* The request can already carry a caller transcript (`AudioInput.transcript`).
* Decisions are stateless per request.
* A streaming client re-issues the decision on each partial transcript.

**To add later:**
* chunked encoder caching;
* a `partial` flag in the response;
* debouncing.

**Known limit today.** Greedy decoding caps at `max_new_tokens = 96`, which truncates transcripts of ~30 s speech.
Streaming or chunking removes this cap.
