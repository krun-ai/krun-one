"""Krun One v1 — unit tests for the low-risk building blocks (CPU, tiny models, no network)."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import pytest
import torch

from krun_one.v1 import arch, growth, leakage, losses, schema
from krun_one.v1.arch import QTYPES_V1, DecisionSpec, EncoderSpec, ModelSpec
from krun_one.v1.model import KrunOneV1Model, chunk_index, collate_chunks, merge_chunks
from krun_one.v1.serialization import Budget, Serializer

HUB = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def gemma_tok():
    snaps = glob.glob(str(HUB / "models--jhu-clsp--mmBERT-small/snapshots/*/tokenizer.json"))
    if not snaps:
        pytest.skip("mmBERT-small tokenizer not in the HF cache")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(Path(snaps[0]).parent))


TINY = ModelSpec("tiny", EncoderSpec(vocab_size=256_000, hidden_size=32, num_layers=3, num_heads=2,
                                     intermediate_size=48, local_window=16),
                 DecisionSpec(head_layers=1, head_ffn_mult=2))


def tiny_model(seed: int = 0) -> KrunOneV1Model:
    torch.manual_seed(seed)
    return KrunOneV1Model.from_spec(TINY, attn_implementation="eager")


def rec(qtype="choice", labels=("a",), n=3, text="quero cancelar minha assinatura", **kw):
    if qtype == "noul":
        cands = [{"id": "false"}, {"id": "true"}]
    elif qtype == "score":
        cands = [{"id": str(i), "label": f"level text {i}"} for i in range(n)]
    else:
        cands = [{"id": c, "label": c, "description": f"Description of tool {c}.", "language": "en"}
                 for c in "abcdefghijklmnopqrstuvwxyz"[:n]]
    r = {"schema_version": "krun.decision.v1", "id": kw.get("id", f"t/{qtype}/{text[:8]}"), "split": kw.get("split", "train"),
         "task": {"type": qtype, "instructions": kw.get("instructions", "Is the user asking to cancel?")},
         "context": {"text": text, "language": kw.get("lang", "pt-BR")}, "candidates": cands,
         "target": {"labels": list(labels), "distribution": None}, "answerable": bool(labels),
         "semantic_group_id": kw.get("group", "g1"), "split_group_id": kw.get("group", "g1"),
         "provenance": {"source_dataset": "unit", "revision": "0", "license": "CC0-1.0", "synthetic": None},
         "candidate_set": {"strategy": kw.get("strategy", "hard_v1"), "k": len(cands)}}
    return schema.derive_fields(r)


# ---------------------------------------------------------------------------------------------- parameter counts
@pytest.mark.parametrize("spec", arch.CANDIDATES + [arch.PROXY, TINY], ids=lambda s: s.name)
def test_param_count_analytic_equals_instantiated(spec):
    a = arch.count_parameters(spec)
    r = arch.instantiate_counts(spec)
    assert a["total"] == r["total"]
    assert a["encoder_total"] == r["encoder_total"]
    assert a["encoder_embeddings"] == r["encoder_embeddings"]


def test_reference_v03_count_matches_production_artifact():
    # real artifact: 421,293,827 parameters incl. the unused act head (Linear(d+4,256) + Linear(256,2));
    # `temperature` is a buffer, not a parameter
    act = (1024 + 4) * 256 + 256 + 256 * 2 + 2
    assert arch.count_parameters(arch.REFERENCE_V03)["total"] + act == 421_293_827


def test_candidates_in_budget():
    main = arch.CANDIDATES[0]
    assert 750e6 <= arch.count_parameters(main)["total"] <= 900e6


# ---------------------------------------------------------------------------------------------- serialization
def test_layout_and_null_marker(gemma_tok):
    s = Serializer(gemma_tok)
    q = s.encode_record(rec())
    assert len(q.chunks) == 1
    c = q.chunks[0]
    assert c.ids[0] == gemma_tok.cls_token_id
    assert [c.ids[m] for m in c.markers] == [gemma_tok.mask_token_id] * 3
    assert c.ids[c.null_pos] == s.null_id and c.null_pos > max(c.markers)
    assert c.ids[c.null_pos + 1] == gemma_tok.sep_token_id and c.ids[-1] == gemma_tok.sep_token_id
    assert "cancelar" in gemma_tok.decode(c.ids[c.null_pos + 2:])
    head = gemma_tok.decode(c.ids[1:c.markers[0]]).strip()
    assert head.startswith("choice question: Is the user asking to cancel?")


def test_noul_score_have_no_null_and_render_like_v03(gemma_tok):
    s = Serializer(gemma_tok)
    qn = s.encode_record(rec("noul", labels=("true",)))
    assert qn.chunks[0].null_pos == -1 and len(qn.chunks[0].markers) == 2
    txt = gemma_tok.decode(qn.chunks[0].ids)
    assert "false: no, the statement does not hold" in txt and "true: yes, the statement holds" in txt
    qs = s.encode_record(rec("score", labels=("1",), n=5, instructions="How severe?"))
    assert qs.chunks[0].null_pos == -1 and len(qs.chunks[0].markers) == 5
    assert "level 4: level text 4" in gemma_tok.decode(qs.chunks[0].ids)


def test_chunking_keeps_full_descriptions(gemma_tok):
    s = Serializer(gemma_tok, Budget(max_len=1024, head_max_len=896))
    r = rec(n=26)
    for c in r["candidates"]:
        c["description"] = "A fairly long tool description that explains what the function does. " * 4
    q = s.encode_record(r)
    assert len(q.chunks) > 1
    got = sorted(i for c in q.chunks for i in c.candidate_index)
    assert got == list(range(26))
    for c in q.chunks:
        assert len(c.ids) <= 1024
        # every candidate keeps its capped length (48 text tokens + marker): no even shrinking
        spans = c.markers + [c.null_pos]
        assert all(spans[j + 1] - spans[j] == 49 for j in range(len(c.markers)))


def test_default_instructions_by_family(gemma_tok):
    s = Serializer(gemma_tok)
    r = rec(instructions=None)
    r["task"]["instructions"] = None
    head = gemma_tok.decode(s.encode_record(r).chunks[0].ids[:30])
    assert "Choose the function that should be called" in head


# ---------------------------------------------------------------------------------------------- model + losses
def _batch(gemma_tok, records):
    s = Serializer(gemma_tok, Budget(max_len=256, head_max_len=160))
    qs = [s.encode_record(r) for r in records]
    flat, where = chunk_index(qs)
    b = collate_chunks(flat, gemma_tok.pad_token_id)
    kmax = max(q.n_candidates for q in qs)
    tgt = torch.zeros(len(qs), kmax)
    for i, r in enumerate(records):
        ids = [c["id"] for c in r["candidates"]]
        for lab in r["target"]["labels"]:
            tgt[i, ids.index(lab)] = 1
    qt = torch.tensor([QTYPES_V1[r["task"]["type"]] for r in records])
    ans = torch.tensor([r["answerable"] for r in records])
    return b, where, qs, tgt, qt, ans


MIXED = [rec(), rec(labels=(), text="qual a previsão do tempo amanhã"),
         rec("multi", labels=("a", "c"), n=4, text="cancel my plan and send me the invoice"),
         rec("multi", labels=(), n=3, text="tell me a joke about cats"),
         rec("noul", labels=("true",), text="je veux parler à un humain"),
         rec("score", labels=("3",), n=5, instructions="How severe?", text="el sistema está caído desde ayer")]


@pytest.mark.parametrize("mode", ["oe", "null", "head"])
def test_forward_backward_all_types(gemma_tok, mode):
    m = tiny_model()
    b, where, qs, tgt, qt, ans = _batch(gemma_tok, MIXED)
    out = m(**b)
    q = merge_chunks(out, where, len(qs), [x.n_candidates for x in qs])
    loss, parts = losses.decision_loss(q, qt, tgt, ans, losses.LossConfig(unanswerable_mode=mode))
    assert torch.isfinite(loss)
    loss.backward()
    assert m.scorer[-1].weight.grad.abs().sum() > 0
    assert m.answer_head[-1].weight.grad.abs().sum() > 0
    sc = losses.answerability_scores({k: v.detach() for k, v in q.items()})
    assert set(sc) >= {"margin", "null_softmax", "ans_head", "max_apply"}


def test_merge_single_chunk_is_identity(gemma_tok):
    m = tiny_model().eval()
    b, where, qs, *_ = _batch(gemma_tok, [rec(), rec("noul", labels=("true",))])
    with torch.no_grad():
        out = m(**b)
        q = merge_chunks(out, where, len(qs), [x.n_candidates for x in qs])
    assert torch.equal(q["rank"][0, :3], out["rank"][0, :3])
    assert torch.equal(q["ans"], out["ans"])


def test_padding_invariance_fp32(gemma_tok):
    m = tiny_model().eval()
    b1, w1, q1, *_ = _batch(gemma_tok, [rec()])
    b2, w2, q2, *_ = _batch(gemma_tok, [rec(), rec(text="x " * 120)])
    with torch.no_grad():
        a = m(**b1)["rank"][0, :3]
        c = m(**b2)["rank"][0, :3]
    assert torch.allclose(a, c, atol=1e-5)


def test_overfit_tiny(gemma_tok):
    """The whole stack can drive every objective to ~0 on a handful of decisions (catches wiring bugs)."""
    torch.manual_seed(0)
    m = tiny_model()
    b, where, qs, tgt, qt, ans = _batch(gemma_tok, MIXED)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    cfg = losses.LossConfig(unanswerable_mode="head", w_apply=1.0, w_ans=1.0)
    first = None
    for _ in range(150):
        q = merge_chunks(m(**b), where, len(qs), [x.n_candidates for x in qs])
        loss, _ = losses.decision_loss(q, qt, tgt, ans, cfg)
        first = first if first is not None else float(loss.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss.detach()) < 0.05 * first
    m.eval()
    with torch.no_grad():
        q = merge_chunks(m(**b), where, len(qs), [x.n_candidates for x in qs])
    assert torch.sigmoid(q["ans"][0]) > 0.9 and torch.sigmoid(q["ans"][1]) < 0.1


# ---------------------------------------------------------------------------------------------- growth
def test_hyperclone_preserves_function():
    from transformers import ModernBertConfig, ModernBertForMaskedLM

    torch.manual_seed(0)
    cfg = ModernBertConfig(vocab_size=500, hidden_size=48, intermediate_size=72, num_hidden_layers=4,
                           num_attention_heads=3, local_attention=16, pad_token_id=0, bos_token_id=2,
                           cls_token_id=2, eos_token_id=1, sep_token_id=1, layer_types=None)
    cfg._attn_implementation = "eager"
    src = ModernBertForMaskedLM(cfg).eval()
    dst = growth.widen_modernbert(src).eval()
    assert dst.config.hidden_size == 96 and dst.config.num_attention_heads == 6
    ids = torch.randint(5, 500, (2, 40))
    att = torch.ones_like(ids)
    att[1, 30:] = 0
    with torch.no_grad():
        a = src(input_ids=ids, attention_mask=att)
        b = dst(input_ids=ids, attention_mask=att)
        ha = src.model(input_ids=ids, attention_mask=att).last_hidden_state
        hb = dst.model(input_ids=ids, attention_mask=att).last_hidden_state
    assert torch.allclose(hb[..., :48], ha, atol=1e-4) and torch.allclose(hb[..., 48:], ha, atol=1e-4)
    assert torch.allclose(a.logits, b.logits, atol=1e-3)


# ---------------------------------------------------------------------------------------------- schema
@pytest.mark.parametrize("r", MIXED, ids=lambda r: r["task"]["type"] + str(r["answerable"]))
def test_schema_valid(r):
    assert schema.validate(r) == []


def test_schema_rejects():
    bad = rec(labels=("a", "b"))
    assert any("at most one" in e for e in schema.validate(bad))
    bad = rec(labels=())
    bad["answerable"] = True
    assert schema.validate(bad)
    bad = rec(lang="portuguese")
    assert any("BCP-47" in e for e in schema.validate(bad))
    bad = rec()
    bad["provenance"]["synthetic"] = {"teacher": "x", "prompt_id": "p", "verifier": "v", "verdict": "rejected",
                                      "source_example_id": "s", "transformation": "minimal_pair"}
    assert any("verifier" in e for e in schema.validate(bad))


@pytest.mark.skipif(not (ROOT / "data/v0/val_tools.jsonl").exists(), reason="v0 data not built")
def test_convert_v0_files_validate():
    for path, conv in [(ROOT / "data/v0/val_tools.jsonl", schema.from_krunbench_v0),
                       (ROOT / "data/primitives_v2/noul_val.jsonl", schema.from_primitive_v0),
                       (ROOT / "data/primitives_v2/score_val.jsonl", schema.from_primitive_v0)]:
        if not path.exists():
            continue
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= 300:
                    break
                r = conv(json.loads(line))
                assert schema.validate(r) == [], (path.name, r["id"], schema.validate(r))


# ---------------------------------------------------------------------------------------------- leakage
def test_leakage_checks():
    prot = [rec(text="Can a transfer be cancelled after it was sent to another bank?", id="p1", group="pg1",
                split="protected")]
    train = [
        rec(text="can a transfer be cancelled after it was sent to another bank", id="t1", group="g1"),  # exact
        rec(text="Can a transfr be canceled after it was sent to anther bank?", id="t2", group="g2"),   # near
        rec(text="Qual é a previsão do tempo para amanhã em Lisboa?", id="t3", group="pg1"),             # group
        rec(text="Please book me a table for two tonight at an Italian place.", id="t4", group="g4"),    # clean
    ]
    rep = leakage.check_leakage(train, prot, near_threshold=0.5)
    v = rep.summary()["violations"]
    assert v.get("exact") == 1 and v.get("near") == 1 and v.get("group") == 1
    assert not any(x["id"] == "t4" for k in rep.violations for x in rep.violations[k])


def test_leakage_candidate_protection():
    prot = [rec(id="p1", group="pg")]
    train = [rec(id="t1", group="g", text="something unrelated entirely here")]
    rep = leakage.check_leakage(train, prot, protect_candidates=True)
    assert rep.summary()["violations"].get("candidate") == 3


def test_split_group_check():
    rs = [rec(id="a", group="x", split="train"), rec(id="b", group="x", split="test")]
    assert not leakage.check_split_groups(rs).ok


# ---------------------------------------------------------------------------------------------- checkpoint / resume
def test_v1_resume_is_bit_identical_on_cpu(gemma_tok, tmp_path):
    """Continuous 6 steps == 3 steps + kill + resume 3 steps (CPU, deterministic)."""
    from krun_one.v1.training import encode_records, train

    ser = Serializer(gemma_tok, Budget(max_len=256, head_max_len=160))
    recs = [dict(r, id=f"{r['id']}/{i}") for i in range(4) for r in MIXED]
    items = encode_records(ser, recs)
    cfg = {"seed": 7, "epochs": 1, "batching": {"max_tokens": 600, "max_questions": 4},
           "optimizer": {"lr_encoder": 1e-3, "lr_head": 1e-3, "weight_decay": 0.01, "warmup_ratio": 0.1,
                         "max_grad_norm": 1.0},
           "checkpoint": {"save_every": 3, "keep_last": 2}, "loss": {"unanswerable_mode": "head"}}

    def fresh():
        return tiny_model(seed=3)

    a = fresh()
    ra = train(a, items, gemma_tok.pad_token_id, cfg, tmp_path / "a", max_steps=6, device="cpu")
    b = fresh()
    train(b, items, gemma_tok.pad_token_id, cfg, tmp_path / "b", max_steps=3, device="cpu")
    b2 = fresh()
    from krun_one.checkpoint import resolve_resume
    rb = train(b2, items, gemma_tok.pad_token_id, cfg, tmp_path / "b", max_steps=6, device="cpu",
               resume_from=resolve_resume("latest", tmp_path / "b"))
    assert ra["global_step"] == rb["global_step"] == 6
    for (n, pa), (_, pb) in zip(a.state_dict().items(), b2.state_dict().items()):
        assert torch.equal(pa, pb), n
    # a different loss config must be refused on resume
    bad = dict(cfg, loss={"unanswerable_mode": "oe"})
    with pytest.raises(ValueError):
        train(fresh(), items, gemma_tok.pad_token_id, bad, tmp_path / "b", max_steps=7, device="cpu",
              resume_from=resolve_resume("latest", tmp_path / "b"))


# ---------------------------------------------------------------------------------------------- v1 metadata
def test_derived_fields_and_consistency():
    r = rec(lang="pt-BR")                       # EN tool candidates, PT context
    assert r["schema_language"] == "en" and r["cross_language"] and "cross_language" in r["slices"]
    assert r["difficulty"] == "hard"
    r2 = rec(lang="en-US", strategy="random")
    assert not r2["cross_language"] and r2["difficulty"] == "easy"
    bad = dict(r, cross_language=False)
    assert any("inconsistent" in e for e in schema.validate(bad))
    bad = {k: v for k, v in r.items() if k != "difficulty"}
    assert schema.validate(bad)


# ---------------------------------------------------------------------------------------------- unseen tools
def test_tool_index_statuses():
    from krun_one.v1.tools import SEEN_EQUIVALENT, SEEN_EXACT, UNSEEN, TrainToolIndex

    idx = TrainToolIndex()
    idx.add("get_weather", "Get the current weather forecast for a given city and date.")
    idx.add("send_email", "Send an email message to a recipient with a subject and body.")  # a distractor counts too
    assert idx.status("get_weather", "Get the current weather forecast for a given city and date.") == SEEN_EXACT
    assert idx.status("weather_now", "Get the current weather forecast for a given city and date") == SEEN_EQUIVALENT
    assert idx.status("wx", "Get the current weather forecast for a given city and a date.") == SEEN_EQUIVALENT
    assert idx.status("send_mail", "Send an email message to a recipient with a subject and body!") == SEEN_EQUIVALENT
    assert idx.status("book_flight", "Book a flight between two airports on a given date.") == UNSEEN


def test_proxy_unseen_tool_marking_and_invariants():
    import importlib.util

    spec = importlib.util.spec_from_file_location("bpd", ROOT / "scripts/v1/build_proxy_data.py")
    bpd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bpd)
    train = [rec(id="tr", group="g0")]           # tools a/b/c with "Description of tool X."
    v_seen = rec(id="v1", group="g1", split="validation")
    v_new = rec(id="v2", group="g2", split="validation")
    v_new["candidates"][0].update(id="zz", label="zz", description="A totally different brand new capability here.")
    v_new["target"]["labels"] = ["zz"]
    for r in (train[0], v_seen, v_new):
        r["task"]["family"] = "tool"
    noul = rec("noul", labels=("true",), id="v3", group="g3", split="validation")
    noul["metadata"] = {"tool": "yy", "tool_description": "Another unseen capability that nobody trained on."}
    val = [v_seen, v_new, noul]
    stats = bpd.mark_unseen_tools(train, val)
    assert "unseen_tool" not in v_seen["slices"] and "unseen_tool" in v_new["slices"] and "unseen_tool" in noul["slices"]
    assert stats["choice/unseen"] == 1 and stats["choice/seen_exact"] == 1
    assert bpd.check_unseen_tool_invariants(train, val)["unseen_tool_slice"] == {"choice": 1, "noul": 1}
    with pytest.raises(SystemExit):                  # empty noul slice must abort
        bpd.check_unseen_tool_invariants(train, [v_new])


@pytest.mark.skipif(not (ROOT / "data/v0/train.jsonl").exists(), reason="v0 data not built")
def test_noul_builder_unseen_tool_slice_regression():
    """Regression for the v0.3 bug: the noul unseen-tool slice was always empty."""
    import importlib.util

    import yaml

    spec = importlib.util.spec_from_file_location("ppd", ROOT / "scripts/prepare_primitives_data.py")
    ppd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ppd)
    cfg = yaml.safe_load((ROOT / "configs/primitives_data_v2.yaml").read_text())
    out = ppd.build_noul_tools(cfg, {})
    from krun_one.v1.tools import TrainToolIndex

    idx = TrainToolIndex().add_v0_decisions(json.loads(l) for l in open(ROOT / "data/v0/train.jsonl"))
    n = {sp: sum(r["metadata"]["unseen_tool"] for r in out[sp]) for sp in out}
    assert n["train"] == 0 and n["val"] > 0 and n["test"] > 0, n
    for sp in ("val", "test"):
        for r in out[sp]:
            if r["metadata"]["unseen_tool"]:
                desc = r["instructions"].split("Function description:", 1)[1].strip()
                assert idx.is_unseen(r["provenance"]["tool"], desc)


# ---------------------------------------------------------------------------------------------- continued MLM (G1)
def test_continued_mlm_smoke_milestones_and_bitwise_resume(tmp_path):
    import subprocess
    import sys

    from safetensors.torch import load_file

    def run(d, *extra):
        r = subprocess.run([sys.executable, str(ROOT / "scripts/v1/continued_mlm.py"), "--smoke", "--smoke-dir", str(d),
                            *extra], capture_output=True, text=True, env={**os.environ, "TORCH_DISABLE_NATIVE_JIT": "1"})
        assert r.returncode == 0, r.stderr[-2000:]
        return r.stdout

    run(tmp_path / "a")
    run(tmp_path / "b", "--max-tokens", "3072")
    run(tmp_path / "b", "--resume-from-checkpoint", "latest")
    for m in (0, 2048, 5120):
        assert (tmp_path / "a/run" / f"milestone-{m}/eval.json").exists()
        ea = json.loads((tmp_path / "a/run" / f"milestone-{m}/eval.json").read_text())["heldout_mlm_loss"]
        eb = json.loads((tmp_path / "b/run" / f"milestone-{m}/eval.json").read_text())["heldout_mlm_loss"]
        assert ea == eb, (m, ea, eb)
    wa = load_file(str(tmp_path / "a/run/milestone-5120/model.safetensors"))
    wb = load_file(str(tmp_path / "b/run/milestone-5120/model.safetensors"))
    assert all(torch.equal(wa[k], wb[k]) for k in wa)
    l0 = json.loads((tmp_path / "a/run/milestone-0/eval.json").read_text())["heldout_mlm_loss"]
    l1 = json.loads((tmp_path / "a/run/milestone-5120/eval.json").read_text())["heldout_mlm_loss"]
    assert all(l1[k] < l0[k] for k in l0)


def test_continued_mlm_refuses_without_license_gate(tmp_path):
    import subprocess
    import sys

    gate = ROOT / "docs/KRUN_ONE_V1_TOKENIZER_LICENSE.md"
    if gate.exists() and "GATE: CLEARED" in gate.read_text():
        pytest.skip("license gate cleared")
    r = subprocess.run([sys.executable, str(ROOT / "scripts/v1/continued_mlm.py")], capture_output=True, text=True)
    assert r.returncode != 0 and "blocked" in (r.stderr + r.stdout)


def test_grown_checkpoint_loads_into_v1_model(tmp_path):
    """G0 -> G2 wiring: a widened MLM saved with save_pretrained loads exactly into the v1 decision model's encoder."""
    from transformers import ModernBertConfig, ModernBertForMaskedLM

    from krun_one.v1.training import load_mmbert_encoder

    torch.manual_seed(0)
    cfg = ModernBertConfig(vocab_size=256_000, hidden_size=16, intermediate_size=24, num_hidden_layers=3,
                           num_attention_heads=1, local_attention=16, pad_token_id=0, bos_token_id=2,
                           cls_token_id=2, eos_token_id=1, sep_token_id=1)
    cfg._attn_implementation = "eager"
    grown = growth.widen_modernbert(ModernBertForMaskedLM(cfg)).eval()
    grown.save_pretrained(tmp_path / "g")
    spec = ModelSpec("g", EncoderSpec(hidden_size=32, num_layers=3, num_heads=2, intermediate_size=48, local_window=16,
                                      rope_theta_local=cfg.rope_parameters["sliding_attention"]["rope_theta"]),
                     DecisionSpec(head_layers=1, head_ffn_mult=2))
    m = KrunOneV1Model.from_spec(spec, attn_implementation="eager").eval()
    load_mmbert_encoder(m, tmp_path / "g/model.safetensors")
    ids = torch.randint(5, 1000, (1, 12))
    with torch.no_grad():
        a = grown.model(input_ids=ids).last_hidden_state
        b = m.encoder(input_ids=ids).last_hidden_state
    assert torch.equal(a, b)


def test_excluded_sources_removed_as_context_gold_and_distractor(tmp_path):
    """xLAM exclusion (owner decision 2026-09-26): excluded tools may not survive even as distractors."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("bpd2", ROOT / "scripts/v1/build_proxy_data.py")
    bpd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bpd)
    reg = tmp_path / "sources.yaml"
    reg.write_text("decision_sources:\n  xlam: {status: excluded_pending_license_clarification}\n  toolace: {status: approved}\n")
    x = rec(id="x1", group="gx")                                   # xLAM decision: tools a/b/c
    x["provenance"]["source_dataset"] = "xlam"
    t = rec(id="t1", group="gt", n=4)                              # ToolACE decision with a/b/c as distractors, gold d
    t["provenance"]["source_dataset"] = "toolace"
    t["target"]["labels"] = ["d"]
    t2 = rec(id="t2", group="gt2", n=3)                            # ToolACE decision whose gold 'a' is ALSO an xLAM tool
    t2["provenance"]["source_dataset"] = "toolace"
    tr, va, st = bpd.apply_exclusions([x, t, t2], [], str(reg))
    ids = {r["id"] for r in tr}
    assert "x1" not in ids                                          # excluded as context
    kept = {r["id"]: r for r in tr}
    # 'a' is gold of a kept decision -> stays; b and c are xLAM-only -> removed from t1
    assert [c["id"] for c in kept["t1"]["candidates"]] == ["a", "d"]
    assert st["dropped_context"] == 1 and st["candidates_removed"] == 2


def test_pod_runners_run_gpu_preflight_before_work():
    """Regression: a broken-CUDA host once made the probe train on CPU for 25 min. Every GPU runner must abort early."""
    for name in ("runpod_mlm_1b.sh",):
        s = (ROOT / "scripts/v1" / name).read_text()
        assert "gpu_preflight.py" in s
        assert s.index("gpu_preflight.py") < s.index("prepare_mlm_corpus.py")
    s = (ROOT / "scripts/v1/runpod_pilot.sh").read_text()
    assert s.index("gpu_preflight.py") < s.index("synthetic_pilot.py generate")
    # Regression: vLLM JIT-compiles kernels at engine start; without `ninja` on PATH generation died in 2 min.
    assert "ninja" in s and "export PATH=/root/pilot/.venv/bin:$PATH" in s


def test_mlm_checkpoint_policy_keeps_pre_decay_branch_point():
    """The +2B continuation resumes from the pre-decay checkpoint: keep_last must still hold it when the 1B run ends."""
    import yaml

    cfg = yaml.safe_load((ROOT / "configs/v1/continued_mlm_budget.yaml").read_text())
    cfg["total_tokens_stage1"] = max(cfg["milestones_stage1"])            # as in scripts/v1/continued_mlm.py
    total, every, keep = cfg["total_tokens_stage1"], cfg["checkpoint"]["save_every_tokens"], cfg["checkpoint"]["keep_last"]
    step_tokens = cfg["sequence_length"] * cfg["batch_sequences"]
    pre_decay = int(total * (1 - cfg["schedule"]["decay_fraction"]))
    saves, tokens = [], 0
    while tokens < total:                                   # mirrors the save rule in scripts/v1/continued_mlm.py
        prev, tokens = tokens, tokens + step_tokens
        if prev // every != tokens // every or tokens >= total:
            saves.append(tokens)
    kept = saves[-keep:]
    assert any(0 <= t - pre_decay < step_tokens for t in kept), (kept, pre_decay)
    assert every <= 100_000_000                             # <= ~$0.6 lost per interruption on a 4090 or a 3090


def test_krunbench_v1_check_catches_every_leak_kind(tmp_path):
    """--check must flag: exact text, near-duplicate, group, unseen-tool id/name/description/equivalent description."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("kb", ROOT / "scripts/v1/build_krunbench_v1.py")
    kb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kb)
    final = [rec(id="f1", group="fg1", split="protected", text="Where is the nearest charging station for my car tonight?")]
    (tmp_path / "final_protected.jsonl").write_text(json.dumps(final[0]) + "\n")
    man = {"unseen_tools": [{"id": "ev_charge_find", "name": "ev_charge_find",
                             "description": "Find electric vehicle charging stations near a location."}]}
    (tmp_path / "protected_manifest.json").write_text(json.dumps(man))

    def run(rs):
        p = tmp_path / "train.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in rs))
        return kb.check_training_side([p], tmp_path)

    clean = rec(id="t0", group="g0", text="Please book a table for two at an Italian restaurant.")
    assert run([clean])["LEAKAGE"] == 0
    assert run([rec(id="t1", group="g1", text="where is the nearest charging station for my car tonight")])["violations"].get("exact")
    assert run([rec(id="t2", group="fg1", text="unrelated text entirely different here")])["violations"].get("group")
    same_desc = rec(id="t3", group="g3")
    same_desc["candidates"][0].update(id="x", label="x", description="Find electric vehicle charging stations near a location.")
    assert run([same_desc])["violations"].get("tool_seen_equivalent") or run([same_desc])["violations"].get("tool_seen_exact")
    same_name = rec(id="t4", group="g4")
    same_name["candidates"][0].update(id="ev_charge_find", label="ev_charge_find", description="Totally different behaviour.")
    assert run([same_name])["violations"].get("tool_name")


def test_enable_sparse_prediction_takes_effect_after_from_pretrained(tmp_path):
    """Regression: setting only config.sparse_prediction after loading kept full-vocab logits -> OOM (15.6 GiB)."""
    from transformers import ModernBertConfig, ModernBertForMaskedLM

    from krun_one.v1.mlm import enable_sparse_prediction

    cfg = ModernBertConfig(vocab_size=512, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                           num_attention_heads=2, max_position_embeddings=64, pad_token_id=0)
    ModernBertForMaskedLM(cfg).save_pretrained(tmp_path)
    m = ModernBertForMaskedLM.from_pretrained(tmp_path, attn_implementation="sdpa").eval()
    x = torch.randint(5, 512, (2, 16))
    y = torch.full_like(x, -100)
    y[0, 3], y[1, 7], y[1, 9] = 11, 12, 13
    assert m(input_ids=x, labels=y).logits.shape[:2] == (2, 16)          # default: full logits
    enable_sparse_prediction(m)
    out = m(input_ids=x, labels=y)
    assert out.logits.shape == (3, 512)                                  # masked positions only
    assert torch.isfinite(out.loss)


def test_continued_mlm_optimizer_matches_benchmarked_setup():
    """Regression: non-fused AdamW + no expandable segments went OOM on a 22 GiB 4090 (benchmark: fused, 17.6 GiB)."""
    s = (ROOT / "scripts/v1/continued_mlm.py").read_text()
    assert 'fused=dev.type == "cuda"' in s and "enable_sparse_prediction(model)" in s
    assert "expandable_segments:True" in (ROOT / "scripts/v1/runpod_mlm_1b.sh").read_text()
    # Regression: deleting only datasets--* left ~34 GB of blobs/ and the volume filled up at the 300M checkpoint
    assert "! -name 'models--*' -exec rm -rf" in (ROOT / "scripts/v1/runpod_mlm_1b.sh").read_text()
