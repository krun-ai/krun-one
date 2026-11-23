"""M1.5 gated fusion: text mode == frozen text decision, gate interpolates, only gates / fusion receive gradients."""

import torch

from krun_one.v1 import arch
from krun_one.v1.model import KrunOneV1Model
from krun_one.v1.multimodal.features import ModalityFeatures
from krun_one.v1.multimodal.m1 import M1Fusion, M1Model, core_digest
from krun_one.v1.multimodal.m15 import M15Model


def _setup():
    torch.manual_seed(0)
    core = KrunOneV1Model.from_spec(arch.SPECS["proxy"], attn_implementation="sdpa").eval()
    fus = M1Fusion(core, 16, 12, tokens=4, heads=2, lora_r=4)
    torch.nn.init.normal_(fus.lora.layers[0]["in_proj"].b, std=0.05)          # non-zero LoRA: fused != text
    m = M15Model(M1Model(core, fus))
    B, L = 3, 12
    att = torch.ones(B, L, dtype=torch.long)
    att[0, -3:] = 0
    b = {"input_ids": torch.randint(5, 100, (B, L)), "attention_mask": att, "marker_pos": torch.tensor([[1, 3, 5]] * B),
         "marker_mask": torch.ones(B, 3, dtype=torch.bool), "qtype": torch.tensor([0, 2, 3]), "null_pos": torch.tensor([6, -1, 6])}
    f = ModalityFeatures("audio", torch.randn(B, 10, 12), torch.ones(B, 10, dtype=torch.bool))
    return core, fus, m, b, f


def test_text_mode_equals_core():
    core, fus, m, b, f = _setup()
    with torch.no_grad():
        ref = core(**b)
        t, mk = fus.media_tokens(f)
        got = m(b, (t, mk), [0, 1, 2], "audio", mode="text")
    assert torch.allclose(ref["rank"], got["rank"], atol=1e-4)
    assert torch.allclose(ref["apply"], got["apply"], atol=1e-4)


def test_gate_interpolates_between_text_and_fused():
    core, fus, m, b, f = _setup()
    with torch.no_grad():
        t, mk = fus.media_tokens(f)
        zt = m(b, (t, mk), [0, 1, 2], "audio", mode="text")["rank"]
        zf = m(b, (t, mk), [0, 1, 2], "audio", mode="fused")["rank"]
        out = m(b, (t, mk), [0, 1, 2], "audio")
    a = out["alpha"][:, None]
    assert torch.allclose(out["rank"], zt + a * (zf - zt), atol=1e-4)
    assert not torch.allclose(zt, zf)
    assert torch.allclose(out["alpha"], torch.sigmoid(torch.tensor(-1.0)).expand(3), atol=1e-6)   # zero-init last layer


def test_only_trainable_get_gradients():
    core, fus, m, b, f = _setup()
    d0 = core_digest(core)
    m.train()
    t, mk = fus.media_tokens(f)
    m(b, (t, mk), [0, 1, 2], "audio")["rank"].sum().backward()
    assert all(p.grad is None for p in core.parameters())
    assert any(p.grad is not None for p in m.gates["audio"].parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in fus.adapters["audio"].parameters())
    assert core_digest(core) == d0


def test_remove_gate_equals_fused():
    core, fus, m, b, f = _setup()
    ng = M15Model(m.m1, gate=False)
    assert len(ng.gates) == 0 and sum(p.numel() for p in ng.gates.parameters()) == 0
    with torch.no_grad():
        t, mk = fus.media_tokens(f)
        zf = m(b, (t, mk), [0, 1, 2], "audio", mode="fused")["rank"]
        zn = ng(b, (t, mk), [0, 1, 2], "audio")["rank"]
    assert torch.equal(zf, zn)
