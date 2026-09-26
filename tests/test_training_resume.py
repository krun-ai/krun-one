"""Stop + resume must reproduce the continuous run exactly (tiny ModernBERT, CPU, dropout on)."""

import copy

import torch
from safetensors.torch import load_file

from krun_one.checkpoint import list_checkpoints, resolve_resume
from krun_one.model import DecisionModel, build_encoder
from krun_one.training import configure_trainable, decision_loss, train

CFG = {"seed": 7, "epochs": 2, "sequence": {"max_len": 128, "head_max_len": 64},
       "objective": {"unanswerable_weight": 0.5}, "batching": {"max_tokens": 400, "max_seqs": 8},
       "optimizer": {"weight_decay": 0.01, "lr_encoder": 1e-3, "lr_head": 1e-3, "warmup_ratio": 0.1, "max_grad_norm": 1.0,
                     "amp": "bf16"},
       "checkpoint": {"save_every": 3, "keep_last": 0}, "configs": {"A": {"train_top_encoder_layers": 1},
                                                                    "B": {"train_top_encoder_layers": "all"}}}


def items(n=37):
    g = torch.Generator().manual_seed(1)
    out = []
    for i in range(n):
        L = int(torch.randint(10, 40, (1,), generator=g))
        k = int(torch.randint(2, 5, (1,), generator=g))
        ids = [50281] + torch.randint(100, 5000, (L,), generator=g).tolist() + [50282]
        markers = sorted(torch.randperm(L, generator=g)[:k].add(1).tolist())
        y = i % (k + 1)
        ans = y < k
        out.append({"id": f"x{i}", "ids": ids, "markers": markers, "label": y if ans else -1, "answerable": ans,
                    "target": [1.0 if j == y else 0.0 for j in range(k)] if ans else [1.0 / k] * k})
    return out


def fresh(enc_dir):
    torch.manual_seed(0)
    return DecisionModel(build_encoder(enc_dir), head_layers=1, n_act=2)


def test_resume_is_exact(tiny_model, tmp_path):
    _, enc_dir = tiny_model
    data = items()
    ref = train(fresh(enc_dir), data, 50283, CFG, "B", tmp_path / "ref", device="cpu", log_every=1)
    total = ref["total_steps"]
    part = train(fresh(enc_dir), data, 50283, CFG, "B", tmp_path / "res", max_steps=5, device="cpu", log_every=1)
    assert not part["completed"] and part["global_step"] == 5
    ck = resolve_resume("latest", tmp_path / "res")
    assert ck.name == "checkpoint-5"
    m = fresh(enc_dir)
    res = train(m, data, 50283, CFG, "B", tmp_path / "res", resume_from=ck, device="cpu", log_every=1)
    assert res["completed"] and res["global_step"] == total
    assert [e["train_loss"] for e in res["log"]] == [e["train_loss"] for e in ref["log"]]
    # final weights identical to the continuous run: compare against its last checkpoint + remaining steps via the log,
    # and directly by re-running the continuous model
    m_ref = fresh(enc_dir)
    train(m_ref, data, 50283, CFG, "B", tmp_path / "ref2", device="cpu", log_every=1)
    for (k1, a), (k2, b) in zip(m_ref.state_dict().items(), m.state_dict().items()):
        assert k1 == k2 and torch.equal(a, b), k1


def test_checkpoint_contents_and_mismatch(tiny_model, tmp_path):
    _, enc_dir = tiny_model
    data = items()
    train(fresh(enc_dir), data, 50283, CFG, "A", tmp_path / "r", max_steps=3, device="cpu")
    ck = list_checkpoints(tmp_path / "r")[-1]
    assert {p.name for p in ck.iterdir()} >= {"model.safetensors", "optimizer.pt", "scheduler.pt", "scaler.pt", "rng.pt",
                                              "trainer_state.json", "config.json"}
    rng = torch.load(ck / "rng.pt", weights_only=False)
    assert set(rng) >= {"python", "numpy", "torch_cpu", "torch_cuda", "shuffle_generator_epoch_start"}
    bad = copy.deepcopy(CFG)
    bad["optimizer"]["lr_head"] = 5e-4
    try:
        train(fresh(enc_dir), data, 50283, bad, "A", tmp_path / "r", resume_from=ck, device="cpu")
        raise AssertionError("mismatching config must be refused")
    except ValueError as e:
        assert "mismatch" in str(e)


def test_config_a_freezes_lower_layers(tiny_model):
    m, _ = tiny_model
    c = configure_trainable(m, 1)
    assert not any(p.requires_grad for p in m.encoder.layers[0].parameters())
    assert all(p.requires_grad for p in m.encoder.layers[-1].parameters())
    assert all(p.requires_grad for p in m.scorer.parameters())
    assert not any(p.requires_grad for p in m.act_head.parameters())
    assert c["trainable_parameters"] < c["total_parameters"]


def test_loss_uniform_target_for_unanswerable():
    logits = torch.tensor([[2.0, 0.0, -1e4], [0.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    target = torch.tensor([[1.0, 0.0, 0.0], [1 / 3, 1 / 3, 1 / 3]])
    ans = torch.tensor([True, False])
    l = decision_loss(logits, target, mask, ans, 0.5)
    ce0 = -torch.log_softmax(torch.tensor([2.0, 0.0]), -1)[0]
    ce1 = torch.log(torch.tensor(3.0))  # uniform logits vs uniform target = log K (the minimum)
    assert torch.isclose(l, (ce0 + 0.5 * ce1) / 1.5)
