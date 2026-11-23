"""M1: functional head == frozen head (LoRA zero-init), text path untouched, gradients only in the fusion."""

import torch

from krun_one.v1 import arch
from krun_one.v1.model import KrunOneV1Model
from krun_one.v1.multimodal.features import ModalityFeatures
from krun_one.v1.multimodal.m1 import M1Fusion, M1Model, core_digest, head_layer_forward, trainable_parameter_count


def _core():
    torch.manual_seed(0)
    return KrunOneV1Model.from_spec(arch.SPECS["proxy"], attn_implementation="sdpa").eval()


def _batch(core, B=3, L=12, K=3):
    ids = torch.randint(5, 100, (B, L))
    att = torch.ones(B, L, dtype=torch.long)
    att[0, -3:] = 0
    return {"input_ids": ids, "attention_mask": att, "marker_pos": torch.tensor([[1, 3, 5]] * B),
            "marker_mask": torch.ones(B, K, dtype=torch.bool), "qtype": torch.tensor([0, 2, 3]),
            "null_pos": torch.tensor([6, -1, 6])}


def test_functional_head_matches_core():
    core = _core()
    b = _batch(core)
    fus = M1Fusion(core, 16, 12, tokens=4, heads=2, lora_r=4)
    m = M1Model(core, fus)
    with torch.no_grad():
        ref = core(**b)
        got = m.forward_mm(b, None, None)
    for k in ("rank", "apply", "null_rank", "ans"):
        assert torch.allclose(ref[k], got[k], atol=1e-4), k


def test_lora_zero_init_and_text_path_bitwise():
    core = _core()
    b = _batch(core)
    fus = M1Fusion(core, 16, 12, tokens=4, heads=2, lora_r=4)
    m = M1Model(core, fus)
    feats = ModalityFeatures("image", torch.randn(3, 10, 16), torch.ones(3, 10, dtype=torch.bool))
    with torch.no_grad():
        t, mk = fus.media_tokens(feats)
        a = m.forward_mm(b, (t, mk), [0, 1, 2], use_lora=True)
        c = m.forward_mm(b, (t, mk), [0, 1, 2], use_lora=False)
        assert torch.allclose(a["rank"], c["rank"], atol=1e-5)        # B = 0 at init
        assert torch.equal(core(**b)["rank"], m.forward_text(b)["rank"])


def test_only_fusion_gets_gradients():
    core = _core()
    d0 = core_digest(core)
    b = _batch(core)
    fus = M1Fusion(core, 16, 12, tokens=4, heads=2, lora_r=4)
    m = M1Model(core, fus).train()
    feats = ModalityFeatures("audio", torch.randn(3, 10, 12), torch.ones(3, 10, dtype=torch.bool))
    t, mk = fus.media_tokens(feats)
    out = m.forward_mm(b, (t, mk), [0, 1, 2])
    out["rank"].sum().backward()
    assert all(p.grad is None for p in core.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in fus.adapters["audio"].parameters())
    assert core_digest(core) == d0
    n = trainable_parameter_count(fus)
    assert n["total"] == sum(p.numel() for p in fus.parameters())
