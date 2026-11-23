"""Krun One v1 multimodal scaffold: text path unchanged, towers lazy, shapes/masks, schema compatibility, smoke fusion."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from krun_one.v1 import schema
from krun_one.v1.losses import LossConfig, decision_loss
from krun_one.v1.model import KrunOneV1Model, chunk_index, collate_chunks, merge_chunks
from krun_one.v1.multimodal.audio import AudioInput
from krun_one.v1.multimodal.config import AudioConfig, DocumentConfig, MultimodalConfig, VisionConfig
from krun_one.v1.multimodal.documents import DocumentInput, DocumentRouter
from krun_one.v1.multimodal.features import VisionFeatures
from krun_one.v1.multimodal.fusion import LightweightFusion, expand_to_sequences
from krun_one.v1.multimodal.model import KrunOneV1Multimodal, RequestMedia
from krun_one.v1.serialization import Budget, Serializer

from .test_v1_core import MIXED, TINY, gemma_tok, rec, tiny_model  # noqa: F401  (fixture re-export)

ROOT = Path(__file__).resolve().parents[2]
D = TINY.encoder.hidden_size   # 32


def mm_all(**over) -> MultimodalConfig:
    return MultimodalConfig(vision=VisionConfig(enabled=True, model="mock", projector_dim=D, tokens=4),
                            document=DocumentConfig(enabled=True, min_chars=20),
                            audio=AudioConfig(enabled=True, model="mock", projector_dim=D, tokens=4), **over)


def text_batch(tok, records):
    s = Serializer(tok, Budget(max_len=256, head_max_len=160))
    qs = [s.encode_record(r) for r in records]
    flat, where = chunk_index(qs)
    return collate_chunks(flat, tok.pad_token_id), where, qs


LONG_DOC = ("Invoice INV-2026-0042. Customer ACME Ltda. Amount due: 1,250.00 EUR. Due date: 2026-10-15. "
            "Payment overdue reminder: please settle the outstanding balance. ") * 2


# ---------------------------------------------------------------------------------------------- defaults / text path
def test_config_disabled_by_default_and_yaml():
    c = MultimodalConfig()
    assert not c.any_enabled and not c.vision.enabled and not c.audio.enabled and not c.document.enabled
    y = MultimodalConfig.from_yaml(ROOT / "configs/v1/multimodal.yaml")
    assert y.to_dict() == c.to_dict()
    assert MultimodalConfig.from_dict(mm_all().to_dict()).to_dict() == mm_all().to_dict()   # round trip


def test_text_only_has_zero_extra_params_and_no_towers():
    core = tiny_model()
    mm = KrunOneV1Multimodal(core, MultimodalConfig(vision=VisionConfig(projector_dim=D), audio=AudioConfig(projector_dim=D)))
    assert mm.fusion is None and mm._vision is None and mm._audio is None
    assert sum(p.numel() for p in mm.parameters()) == sum(p.numel() for p in core.parameters())
    assert mm.multimodal_state_dict() == {}


def test_text_only_outputs_bitwise_unchanged(gemma_tok):
    """Same weights: pre-multimodal model code == new core == multimodal wrapper (text-only)."""
    spec = importlib.util.spec_from_file_location("pre_mm", ROOT / "results/v1/multimodal/model_pre_multimodal.py",
                                                  submodule_search_locations=None)
    pre = importlib.util.module_from_spec(spec)
    import sys

    sys.modules["pre_mm_model"] = pre
    pre.__package__ = "krun_one.v1"
    spec.loader.exec_module(pre)
    core = tiny_model().eval()
    old = pre.KrunOneV1Model.from_spec(TINY, attn_implementation="eager").eval()
    old.load_state_dict(core.state_dict(), strict=True)
    wrapper = KrunOneV1Multimodal(core, mm_all()).eval()          # multimodal ENABLED, but a text-only request
    b, where, qs = text_batch(gemma_tok, MIXED)
    with torch.no_grad():
        a = old(**b)
        c = core(**b)
        w = wrapper(b)
    for k in a:
        assert torch.equal(a[k], c[k]) and torch.equal(a[k], w[k]), k
    assert wrapper.towers_loaded == {"vision": False, "audio": False}


def test_text_request_with_multimodal_enabled_loads_no_tower(gemma_tok):
    mm = KrunOneV1Multimodal(tiny_model(), mm_all())
    ctx, left = mm.prepare(["quero cancelar"], [None])
    assert mm.encode_media(left) is None
    assert mm.towers_loaded == {"vision": False, "audio": False}


def test_document_text_fast_path_does_not_load_vision():
    mm = KrunOneV1Multimodal(tiny_model(), mm_all())
    ctx, left = mm.prepare(["the customer sent this invoice"],
                           [RequestMedia(documents=[DocumentInput(LONG_DOC, "text/plain", document_type="invoice")])])
    assert "INV-2026-0042" in ctx[0] and left[0]["visual"] == []
    assert mm.encode_media(left) is None
    assert mm.towers_loaded["vision"] is False


def test_document_router_visual_cases():
    r = DocumentRouter(DocumentConfig(enabled=True, min_chars=200))
    assert r.route(DocumentInput(b"x", "image/png")).route == "visual"
    assert r.route(DocumentInput("tiny receipt", "text/plain")).route == "visual"          # too little text
    assert r.route(DocumentInput(LONG_DOC * 3, "text/plain", visual_required=True)).route == "visual"
    assert r.route(DocumentInput("<p>" + LONG_DOC * 3 + "</p>", "text/html")).route == "text"
    with pytest.raises(ValueError):
        DocumentRouter(DocumentConfig(enabled=True, visual_fallback=False)).route(DocumentInput(b"x", "image/png"))


def test_disabled_modality_rejected():
    mm = KrunOneV1Multimodal(tiny_model(), MultimodalConfig(vision=VisionConfig(projector_dim=D), audio=AudioConfig(projector_dim=D)))
    with pytest.raises(ValueError):
        mm.prepare(["x"], [RequestMedia(images=[torch.zeros(3, 32, 32)])])


# ---------------------------------------------------------------------------------------------- projector / fusion
def test_projector_output_is_krun_hidden_1536():
    f = LightweightFusion(1536, num_queries=32, heads=8)
    f.ensure_adapter("image", 768)
    x = VisionFeatures(torch.randn(2, 196, 768), torch.ones(2, 196, dtype=torch.bool))
    t, m = f([x])
    assert t.shape == (2, 32, 1536) and m.shape == (2, 32) and m.all()
    n_fusion = sum(p.numel() for p in f.parameters())
    assert n_fusion < 40e6   # budget: fusion/projectors/heads 10-40M


def test_mixed_modality_masks_and_sequence_expansion():
    mm = KrunOneV1Multimodal(tiny_model(), mm_all())
    img = torch.randn(3, 32, 32)
    wav = AudioInput(torch.randn(16000), 16000, language="pt-BR")
    media = [RequestMedia(images=[img, img]), RequestMedia(), RequestMedia(audio=[wav]), RequestMedia(images=[img], audio=[wav])]
    ctx, left = mm.prepare(["a", "b", "c", "d"], media)
    tokens, mask = mm.encode_media(left)
    assert tokens.shape == (4, 8, D) and mask.shape == (4, 8)          # 4 image tokens + 4 audio tokens
    img_m, aud_m = mask[:, :4], mask[:, 4:]
    assert img_m.tolist() == [[True] * 4, [False] * 4, [False] * 4, [True] * 4]
    assert aud_m.tolist() == [[False] * 4, [False] * 4, [True] * 4, [True] * 4]
    assert mm.towers_loaded == {"vision": True, "audio": True}
    t, m = expand_to_sequences(tokens, mask, [0, 0, 3])               # 3 sequences: 2 of request 0, 1 of request 3
    assert t.shape == (3, 8, D) and torch.equal(t[0], t[1]) and torch.equal(m[2], mask[3])


def test_empty_media_rows_are_masked_not_nan():
    mm = KrunOneV1Multimodal(tiny_model(), mm_all())
    ctx, left = mm.prepare(["a", "b"], [RequestMedia(images=[torch.randn(3, 32, 32)]), RequestMedia()])
    tokens, mask = mm.encode_media(left)
    assert torch.isfinite(tokens).all() and not mask[1].any() and tokens[1].abs().sum() == 0


# ---------------------------------------------------------------------------------------------- smoke: fused decisions
@pytest.mark.parametrize("combo", ["image", "document_visual", "audio", "image+audio"])
def test_smoke_fused_decision_forward_backward(gemma_tok, combo):
    torch.manual_seed(0)
    mm = KrunOneV1Multimodal(tiny_model(), mm_all())
    img = torch.randn(3, 32, 32)
    wav = AudioInput(torch.randn(24000), 24000, language="fr-CA")
    m = RequestMedia(images=[img] if "image" in combo else [],
                     documents=[DocumentInput(b"scan", "image/png", pages=[img], document_type="receipt")] if "document" in combo else [],
                     audio=[wav] if "audio" in combo else [])
    records = [rec(), rec("multi", labels=("a", "c"), n=4, text="billing overdue enterprise"),
               rec("noul", labels=("true",), text="needs review?"), rec("score", labels=("3",), n=5, instructions="Priority?", text="urgent")]
    ctx, left = mm.prepare([r["context"]["text"] for r in records], [m] * len(records))
    for r, c in zip(records, ctx):
        r["context"]["text"] = c
    b, where, qs = text_batch(gemma_tok, records)
    media = mm.encode_media(left)
    assert media is not None
    seq_req = [qi for qi, _, _ in where]
    out = mm(b, media, seq_req)
    q = merge_chunks(out, where, len(qs), [x.n_candidates for x in qs])
    kmax = max(x.n_candidates for x in qs)
    tgt = torch.zeros(len(records), kmax)
    for i, r in enumerate(records):
        ids = [c["id"] for c in r["candidates"]]
        for lab in r["target"]["labels"]:
            tgt[i, ids.index(lab)] = 1
    from krun_one.v1.arch import QTYPES_V1
    qt = torch.tensor([QTYPES_V1[r["task"]["type"]] for r in records])
    loss, _ = decision_loss(q, qt, tgt, torch.ones(len(records), dtype=torch.bool), LossConfig())
    loss.backward()
    grads = [p.grad for p in mm.fusion.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)          # gradients reach the projectors/poolers
    # multimodal tokens change the answer only through the head: text-only answer differs from fused
    with torch.no_grad():
        text_only = mm.core(**b)["rank"]
    assert not torch.equal(text_only, out["rank"].detach())


# ---------------------------------------------------------------------------------------------- schema / checkpoints
def test_schema_backward_compatible_and_media():
    for r in MIXED:
        assert schema.validate(r) == []                    # existing text records unchanged
    r = rec(text="Is this invoice overdue?")
    r["context"]["images"] = [{"ref": "s3://bucket/inv.png", "mime": "image/png", "width": 1200, "height": 1600}]
    r["context"]["audio"] = [{"ref": "a.wav", "mime": "audio/wav", "duration_s": 4.2, "language": "pt-BR"}]
    r["modalities"] = schema.derive_modalities(r["context"])
    assert r["modalities"] == ["text", "image", "audio"] and schema.validate(r) == []
    bad = rec()
    bad["context"]["images"] = [{"ref": "x", "mime": "audio/wav"}]
    assert any("mime" in e for e in schema.validate(bad))
    bad2 = rec()
    bad2["modalities"] = ["text", "image"]
    assert any("inconsistent" in e for e in schema.validate(bad2))


def test_checkpoint_loads_without_multimodal_weights(tmp_path):
    from safetensors.torch import load_file, save_file

    core = tiny_model()
    save_file(core.state_dict(), str(tmp_path / "model.safetensors"))     # a text-only artifact
    mm = KrunOneV1Multimodal(tiny_model(seed=9), mm_all())
    mm.load_text_core(load_file(str(tmp_path / "model.safetensors")))    # strict, no multimodal keys needed
    for (k, a), (_, b2) in zip(core.state_dict().items(), mm.core.state_dict().items()):
        assert torch.equal(a, b2), k
    assert all(k.startswith("fusion.") for k in mm.multimodal_state_dict())


def test_document_shares_vision_tower_and_adapter():
    mm = KrunOneV1Multimodal(tiny_model(), mm_all())
    img = torch.randn(3, 32, 32)
    ctx, left = mm.prepare(["a"], [RequestMedia(images=[img], documents=[DocumentInput(b"scan", "image/png", pages=[img])])])
    tokens, mask = mm.encode_media(left)
    assert list(mm.fusion.adapters.keys()) == ["image"]           # one adapter for image + visual document
    assert tokens.shape == (1, 8, D)                               # image tokens + document tokens (distinct emb)
    assert not torch.equal(tokens[0, :4], tokens[0, 4:])
