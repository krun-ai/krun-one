"""Krun One v1 — M1.5 hybrid training / evaluation (configs/v1/m15.yaml; registered before training).

    python scripts/v1/train_m15.py teacher             # frozen text teacher on the GROUND-TRUTH text -> data/m15/teacher.jsonl.gz
    python scripts/v1/train_m15.py health              # CUDA, text fwd, vision fwd, audio encoder, ASR decode, OCR smoke, fusion, backward
    python scripts/v1/train_m15.py train [--max-steps N] [--run-name NAME]
    python scripts/v1/train_m15.py dump --run runs/v1/m15/<name>   # val_fit / val_eval: C, B, A (M1), oracle, fused

Frozen: text core, SigLIP2, Whisper-small (encoder + decoder), OCR. Starts from the M1 fusion weights (sha checked).
Media context: audio -> ASR transcript (precomputed in data/m15 with the same frozen ASR used for serving),
documents / screenshots -> OCR text (precomputed, frozen OCR), natural images -> empty.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/v1"))
import train_m1 as T1  # noqa: E402
from krun_one.v1.arch import QTYPES_V1  # noqa: E402
from krun_one.v1.losses import LossConfig, decision_loss  # noqa: E402
from krun_one.v1.model import chunk_index, collate_chunks, merge_chunks  # noqa: E402
from krun_one.v1.multimodal.features import ModalityFeatures  # noqa: E402
from krun_one.v1.multimodal.m1 import M1Fusion, M1Model, core_digest, trainable_parameter_count  # noqa: E402
from krun_one.v1.multimodal.m15 import M15Model  # noqa: E402

import os  # noqa: E402

CFG_PATH = ROOT / os.environ.get("KRUN_M15_CONFIG", "configs/v1/m15.yaml")   # configs/v1/m15b.yaml for M1.5b
CFG = yaml.safe_load(CFG_PATH.read_text())
VARIANT = CFG["name"].rsplit("-", 1)[-1]                 # m15 | m15b | m15c
IS_B = VARIANT != "m15"                                  # M1.5b-style rounds (no gate, replay data, P system)
D = ROOT / "data" / VARIANT
MEDIA_ROOT = ROOT / "data/m1"
RUNS = ROOT / "runs/v1" / VARIANT
LOSS_CFG = CFG["loss"]["base"] if "base" in CFG["loss"] else CFG["loss"]
GATE = not CFG.get("gate", {}).get("REMOVE_GATE", False)


def read_split(name: str, limit: int = 0) -> list[dict]:
    if name == "protected":
        raise SystemExit("protected split is never read")
    out = []
    for l in open(D / f"{name}.jsonl", encoding="utf-8").read().split("\n"):
        if l:
            out.append(json.loads(l))
            if limit and len(out) >= limit:
                break
    return out


def with_context(r: dict, text: str) -> dict:
    return dict(r, context={"text": text, "language": r["context"]["language"]})


# ------------------------------------------------------------------------------------------------ data
class Data(torch.utils.data.Dataset):
    """variants: which text contexts to encode per item: ctx (ASR/OCR), empty (M1 / blind), gt (oracle)."""

    def __init__(self, recs, ser, teacher, train: bool, variants=("ctx",)):
        self.recs, self.ser, self.teacher, self.train, self.variants = recs, ser, teacher, train, variants

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        qs = {}
        for v in self.variants:
            txt = {"ctx": r["context"]["text"], "empty": "", "gt": r.get("gt_text") or ""}[v]
            qs[v] = self.ser.encode_record(with_context(r, txt))
        t = self.teacher.get(r["id"]) if self.teacher is not None else None
        item = {"i": i, "qs": qs, "target": T1.target_of(r), "qtype": QTYPES_V1[r["task"]["type"]],
                "answerable": bool(r["answerable"]), "modality": r["modality"], "teacher": t,
                "weight": float(r.get("loss_weight", 1.0))}
        p = MEDIA_ROOT / r["media"]["path"]
        if r["modality"] == "audio":
            from krun_one.v1.multimodal.m1_towers import load_audio

            w = load_audio(str(p))
            if self.train:
                rng = np.random.default_rng()
                w = w * rng.uniform(0.6, 1.4)
                if rng.random() < 0.5:
                    snr = rng.uniform(15, 40)
                    w = w + rng.normal(0, np.sqrt(np.mean(w ** 2) / 10 ** (snr / 10) + 1e-12), len(w)).astype(np.float32)
            item["wav"] = w.astype(np.float32)
        else:
            from PIL import Image

            item["image"] = Image.open(p).convert("RGB")
        return item


def make_collate(pad_id: int, variants=("ctx",)):
    state = {}

    def collate(items):
        from krun_one.v1.multimodal.m1_towers import PATCHES, image_preprocessor

        out = {"variants": {}}
        for v in variants:
            qs = [it["qs"][v] for it in items]
            flat, where = chunk_index(qs)
            out["variants"][v] = {"batch": collate_chunks(flat, pad_id), "where": where, "ncand": [q.n_candidates for q in qs]}
        kmax = max(len(it["target"]) for it in items)
        tgt = torch.zeros(len(items), kmax)
        tr = torch.full((len(items), kmax), -1e4)
        ta = torch.full((len(items), kmax), -1e4)
        has_t = torch.zeros(len(items), dtype=torch.bool)
        for i, it in enumerate(items):
            tgt[i, :len(it["target"])] = torch.from_numpy(it["target"])
            t = it["teacher"]
            if t is not None and t["use"]:
                has_t[i] = True
                tr[i, :len(t["rank"])] = torch.tensor(t["rank"])
                if t.get("apply"):
                    ta[i, :len(t["apply"])] = torch.tensor(t["apply"])
        mod = items[0]["modality"]
        assert all(it["modality"] == mod for it in items)
        if mod == "audio":
            media = {"wavs": [it["wav"] for it in items]}
        else:
            if "ip" not in state:
                state["ip"] = image_preprocessor(CFG["towers"]["vision"])
            media = dict(state["ip"](images=[it["image"] for it in items], max_num_patches=PATCHES[mod], return_tensors="pt"))
        out.update({"target": tgt, "qtype": torch.tensor([it["qtype"] for it in items]),
                    "weight": torch.tensor([it["weight"] for it in items]),
                    "answerable": torch.tensor([it["answerable"] for it in items]), "teacher_rank": tr, "teacher_apply": ta,
                    "has_teacher": has_t, "modality": mod, "media": media, "idx": [it["i"] for it in items]})
        return out

    return collate


def batches(recs, rng, shuffle=True):
    bs = CFG["batching"]["batch_size"]
    by = {}
    for i, r in enumerate(recs):
        by.setdefault(r["modality"], []).append(i)
    out = []
    for mod, ids in by.items():
        if shuffle:
            rng.shuffle(ids)
        else:
            ids.sort(key=lambda i: len(recs[i]["candidates"]))
        out += [ids[j:j + bs[mod]] for j in range(0, len(ids), bs[mod])]
    if shuffle:
        rng.shuffle(out)
    return out


# ------------------------------------------------------------------------------------------------ towers + model
class Towers15:
    def __init__(self, device="cuda"):
        from krun_one.v1.multimodal.m1_towers import SigLIP2Tower
        from krun_one.v1.multimodal.m15_asr import WhisperASR

        self.vision = SigLIP2Tower(CFG["towers"]["vision"], device=device)
        self.asr = WhisperASR(CFG["towers"]["asr"], device=device)

    def features(self, B) -> ModalityFeatures:
        m = B["media"]
        if B["modality"] == "audio":
            feats, n = self.asr.logmel(m["wavs"])
            enc = self.asr.encode(feats)
            h, mask = self.asr.acoustic(enc, n)
            h = h.clone()                                    # leave inference mode tensors behind
            mask = mask.clone()
        else:
            h, mask = self.vision.encode(m)
        return ModalityFeatures(B["modality"], h, mask)


def build(device="cuda", freeze_core=True, load_m1=True, gate=None, init_m15=True):
    from safetensors.torch import load_file

    from krun_one.v1.runtime import sha256_file

    core, tok, ser = T1.load_core(device)
    towers = Towers15(device)
    fus = M1Fusion(core, towers.vision.output_dim, towers.asr.output_dim, **T1.CFG["fusion"]).to(device)
    if load_m1:
        p = ROOT / CFG["m1_init"]
        if sha256_file(p) != CFG["m1_init_sha256"]:
            raise SystemExit("M1 init weights sha mismatch")
        fus.load_state_dict(load_file(str(p)))
    model = M15Model(M1Model(core, fus, freeze_core=freeze_core), gate=GATE if gate is None else gate).to(device)
    if init_m15 and CFG.get("m15_init"):                  # M1.5b starts from the M1.5 selected checkpoint (fusion part)
        load_m15(model, ROOT / CFG["m15_init"], strict_gates=False)
    return model, towers, tok, ser


def fwd(model: M15Model, towers: Towers15, B, dev, mode="gated", variant="ctx", feats=None):
    V = B["variants"][variant]
    b = {k: v.to(dev, non_blocking=True) for k, v in V["batch"].items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if mode == "exact_text":
            out = model.core(**b)
        else:
            f = feats if feats is not None else towers.features(B)
            tok, m = model.fusion.media_tokens(f)
            out = model(b, (tok, m), T1.seq_request(V["where"]), B["modality"], mode=mode)
    q = merge_chunks(out, V["where"], len(V["ncand"]), V["ncand"])
    if "alpha" in out:
        a = torch.zeros(len(V["ncand"]), device=dev).index_add(0, torch.tensor([qi for qi, _, _ in V["where"]], device=dev), out["alpha"].float())
        n = torch.zeros(len(V["ncand"]), device=dev).index_add(0, torch.tensor([qi for qi, _, _ in V["where"]], device=dev),
                                                              torch.ones_like(out["alpha"].float()))
        q["alpha"] = a / n.clamp(min=1)
    return q


# ------------------------------------------------------------------------------------------------ teacher (GT text)
@torch.no_grad()
def cmd_teacher(args) -> None:
    core, tok, ser = T1.load_core()
    out = D / "teacher.jsonl.gz"
    stats = {}
    with gzip.open(out, "wt") as f:
        for sp in ("train", "val_fit", "val_eval"):
            recs = [r for r in read_split(sp) if r.get("gt_text") and r.get("teacher_filter") != "never"]
            order = sorted(range(len(recs)), key=lambda i: len(recs[i]["gt_text"]))
            for s in range(0, len(order), 64):
                rs = [recs[i] for i in order[s:s + 64]]
                enc = [ser.encode_record(with_context(r, r["gt_text"])) for r in rs]
                flat, where = chunk_index(enc)
                b = {k: v.cuda() for k, v in collate_chunks(flat, tok.pad_token_id).items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = core(**b)
                q = merge_chunks(o, where, len(rs), [e.n_candidates for e in enc])
                for j, r in enumerate(rs):
                    K = len(r["candidates"])
                    rank = q["rank"][j, :K].float().cpu().numpy()
                    app = q["apply"][j, :K].float().cpu().numpy()
                    gold = T1.target_of(r)
                    t = r["task"]["type"]
                    ok = bool(((1 / (1 + np.exp(-app)) > 0.35) == (gold > 0.5)).all()) if t == "multi" else \
                        bool(r["answerable"] and gold[int(np.argmax(rank))] > 0.5)
                    use = r.get("teacher_filter") == "always" or ok
                    st = stats.setdefault(f"{sp}/{r['source']}/{t}", [0, 0, 0])
                    st[0] += 1; st[1] += ok; st[2] += use
                    f.write(json.dumps({"id": r["id"], "rank": [round(float(x), 4) for x in rank],
                                        "apply": [round(float(x), 4) for x in app], "correct": ok, "use": use}) + "\n")
    res = {k: {"n": v[0], "teacher_acc": v[1] / v[0], "used": v[2]} for k, v in sorted(stats.items())}
    (ROOT / "results/v1/m15").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/v1/m15/teacher_stats.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def load_teacher():
    p = D / "teacher.jsonl.gz"
    return {json.loads(l)["id"]: json.loads(l) for l in gzip.open(p, "rt")} if p.exists() else {}


# ------------------------------------------------------------------------------------------------ health
def cmd_health(args) -> None:
    import torch.nn.functional as F
    from PIL import Image

    from krun_one.v1.multimodal.m1_towers import PATCHES, image_preprocessor
    from krun_one.v1.multimodal.m15_ocr import build_ocr, chosen_ocr

    res = {"cuda": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0)}
    model, towers, tok, ser = build()
    d0 = core_digest(model.core)
    rec = {"task": {"type": "choice", "instructions": "What does the user want?"}, "context": {"text": "quero cancelar minha assinatura"},
           "candidates": [{"id": "cancel_subscription"}, {"id": "check_balance"}]}
    enc = [ser.encode_record(rec), ser.encode_record(rec)]
    flat, where = chunk_index(enc)
    b = {k: v.cuda() for k, v in collate_chunks(flat, tok.pad_token_id).items()}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        qt = merge_chunks(model.core(**b), where, 2, [2, 2])
    res["text_forward_argmax"] = int(qt["rank"][0].argmax())
    im = Image.new("RGB", (640, 480), (120, 30, 200))
    mv = dict(image_preprocessor(CFG["towers"]["vision"])(images=[im, im], max_num_patches=PATCHES["image"], return_tensors="pt"))
    hv, mvk = towers.vision.encode(mv)
    res["vision_forward"] = list(hv.shape)
    from krun_one.v1.multimodal.m1_towers import load_audio
    probe = sorted((ROOT / "data/m1/media/tts").glob("*.flac"))[:1] if (ROOT / "data/m1/media/tts").exists() else []
    w = load_audio(str(probe[0])) if probe else (np.random.randn(16000 * 3).astype(np.float32) * 0.05)
    feats, n = towers.asr.logmel([w, w])
    e = towers.asr.encode(feats)
    res["audio_encoder"] = list(e.shape)
    res["asr_decode"] = towers.asr.transcribe(e[:1])[0][:80]
    ocr = build_ocr(args.ocr or chosen_ocr(ROOT))
    page = Image.new("RGB", (640, 200), (255, 255, 255))
    from PIL import ImageDraw, ImageFont
    ImageDraw.Draw(page).text((20, 60), "Fatura total R$ 123,45", font=ImageFont.truetype(str(ROOT / "data/m1/fonts/DejaVuSans.ttf"), 40), fill=(0, 0, 0))
    tmp = Path("/tmp/ocr_smoke.png"); page.save(tmp)
    res["ocr_smoke"] = ocr.read([str(tmp)])[0]
    model.train()
    ha, ma = towers.asr.acoustic(e, n)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        t, m = model.fusion.media_tokens(ModalityFeatures("audio", ha.clone(), ma.clone()))
        o = model(b, (t, m), T1.seq_request(where), "audio")
        q = merge_chunks(o, where, 2, [2, 2])
        loss = F.cross_entropy(q["rank"], torch.tensor([0, 1], device="cuda"))
        tv, mv_ = model.fusion.media_tokens(ModalityFeatures("image", hv.clone(), mvk.clone()))
        loss = loss + merge_chunks(model(b, (tv, mv_), T1.seq_request(where), "image"), where, 2, [2, 2])["rank"].sum() * 0
    res["fusion_forward_alpha"] = [round(float(x), 3) for x in o["alpha"]] if "alpha" in o else None
    loss.backward()
    core_g = [k for k, p in model.core.named_parameters() if p.grad is not None]
    tower_g = [k for k, p in list(towers.vision.named_parameters()) + list(towers.asr.model.named_parameters()) if p.grad is not None]
    gate_g = sum(1 for p in model.gates.parameters() if p.grad is not None)
    fus_g = sum(1 for p in model.fusion.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    res["backward_only_trainable"] = not core_g and not tower_g and fus_g > 0 and (gate_g > 0 or not len(model.gates))
    res["gate"] = bool(len(model.gates))
    res["core_digest_unchanged"] = core_digest(model.core) == d0
    res["trainable_params"] = {**trainable_parameter_count(model.fusion), "gates": sum(p.numel() for p in model.gates.parameters())}
    res["trainable_params"]["total_m15"] = res["trainable_params"]["total"] + res["trainable_params"]["gates"]
    res["frozen_params"] = {"text_core": sum(p.numel() for p in model.core.parameters()),
                            "siglip2_vision": sum(p.numel() for p in towers.vision.model.parameters()),
                            "whisper_small_total": sum(p.numel() for p in towers.asr.model.parameters()),
                            "whisper_encoder": sum(p.numel() for p in towers.asr.model.model.encoder.parameters()),
                            "ocr": ocr.n_params}
    res["PASS"] = bool(res["cuda"] and res["backward_only_trainable"] and res["core_digest_unchanged"] and res["ocr_smoke"]
                       and res["asr_decode"] is not None)
    print(json.dumps(res, indent=1))
    Path(args.out or ROOT / "results/v1/m15/health.json").write_text(json.dumps(res, indent=1))
    if not res["PASS"]:
        raise SystemExit(1)


# ------------------------------------------------------------------------------------------------ train
def cell(r):
    return f"{r['modality']}/{r['task_class']}/{r['task']['type']}"


def correct(r, q, j) -> float | None:
    K = len(r["candidates"])
    g = T1.target_of(r)
    t = r["task"]["type"]
    if t == "multi":
        return float(((torch.sigmoid(q["apply"][j, :K]).cpu().numpy() > 0.5) == (g > 0.5)).all())
    if t == "choice" and not r["answerable"]:
        return None
    return float(g[int(q["rank"][j, :K].argmax())] > 0.5)


def evaluate_quick(model, towers, recs, ser, tok, dev, limit=5000):
    rng = random.Random(0)
    sub = recs if len(recs) <= limit else rng.sample(recs, limit)
    dl = torch.utils.data.DataLoader(Data(sub, ser, None, False), batch_sampler=batches(sub, rng, False),
                                     collate_fn=make_collate(tok.pad_token_id), num_workers=CFG["batching"]["workers"])
    cells = {}
    model.eval()
    with torch.no_grad():
        for B in dl:
            q = fwd(model, towers, B, dev)
            for j, i in enumerate(B["idx"]):
                c = correct(sub[i], q, j)
                if c is not None:
                    cells.setdefault(cell(sub[i]), []).append(c)
    model.train()
    acc = {k: float(np.mean(v)) for k, v in cells.items()}
    return {"cells": acc, "mean": float(np.mean(list(acc.values())))}


def cmd_train(args) -> None:
    from safetensors.torch import save_file

    torch.manual_seed(CFG["seed"])
    rng = random.Random(CFG["seed"])
    dev = torch.device("cuda")
    model, towers, tok, ser = build()
    d0 = core_digest(model.core)
    run = RUNS / (args.run_name or CFG["name"])
    run.mkdir(parents=True, exist_ok=True)
    train = read_split("train", args.limit)
    val_fit = read_split("val_fit")
    teacher = load_teacher()
    oc = CFG["optimizer"]
    plan = [batches(train, rng) for _ in range(oc["epochs"])]
    total = sum(len(p) for p in plan)
    if args.max_steps:
        total = min(total, args.max_steps)
    lora = [p for n, p in model.fusion.named_parameters() if n.startswith("lora")]
    ada = [p for n, p in model.fusion.named_parameters() if not n.startswith("lora")]
    groups = [{"params": ada, "lr": oc["lr_adapters"]}, {"params": lora, "lr": oc["lr_lora"]}]
    if len(model.gates):
        groups.append({"params": list(model.gates.parameters()), "lr": oc["lr_gate"]})
    opt = torch.optim.AdamW(groups, weight_decay=oc["weight_decay"], betas=(0.9, 0.98))
    base = [g["lr"] for g in opt.param_groups]
    warm = max(1, int(oc["warmup_ratio"] * total))
    lcfg = LossConfig(**LOSS_CFG)
    lam = CFG["consistency_weight"]
    miles = {max(1, int(f * total)) for f in CFG["milestones"]}
    res = {"trainable": {**trainable_parameter_count(model.fusion), "gates": sum(p.numel() for p in model.gates.parameters())},
           "total_steps": total, "n_train": len(train), "core_digest_start": d0, "evals": {}}
    ev0 = evaluate_quick(model, towers, val_fit, ser, tok, dev)
    res["evals"]["step_000000"] = ev0
    print("EVAL step_000000", json.dumps(ev0), flush=True)
    log, step, t0 = [], 0, time.time()
    model.train()
    for ep, pl in enumerate(plan):
        dl = torch.utils.data.DataLoader(Data(train, ser, teacher, True), batch_sampler=pl, collate_fn=make_collate(tok.pad_token_id),
                                         num_workers=CFG["batching"]["workers"], prefetch_factor=4)
        for B in dl:
            if step >= total:
                break
            f = step / warm if step < warm else max(0.0, (total - step) / max(1, total - warm))
            for g, b0 in zip(opt.param_groups, base):
                g["lr"] = b0 * f
            q = fwd(model, towers, B, dev)
            ld, _ = decision_loss(q, B["qtype"].to(dev), B["target"].to(dev), B["answerable"].to(dev), lcfg,
                                  sample_weight=B["weight"].to(dev))
            lc = T1.consistency(q, B, dev)
            loss = ld + lam * lc
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], oc["max_grad_norm"])
            opt.step()
            step += 1
            if step % 50 == 0 or step == 1:
                e = {"step": step, "epoch": ep, "loss": float(loss), "decision": float(ld), "consistency": float(lc), "grad_norm": float(gn),
                     "alpha_mean": float(q["alpha"].mean()) if "alpha" in q else 1.0, "modality": B["modality"], "sec": round(time.time() - t0, 1),
                     "vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
                log.append(e)
                print(json.dumps(e), flush=True)
            if step in miles:
                ck = run / f"step_{step:06d}"
                ck.mkdir(exist_ok=True)
                save_file({k: v.detach().cpu().contiguous() for k, v in model.trainable_state_dict().items()}, str(ck / "m15.safetensors"))
                ev = evaluate_quick(model, towers, val_fit, ser, tok, dev)
                res["evals"][ck.name] = ev
                print("EVAL", ck.name, json.dumps(ev), flush=True)
        if step >= total:
            break
    res["train_seconds"] = time.time() - t0
    res["core_digest_end"] = core_digest(model.core)
    res["core_unchanged"] = res["core_digest_end"] == d0
    if IS_B:
        res["selected"], res["selection_detail"] = select_b(res["evals"])
    else:
        res["selected"] = max((k for k in res["evals"] if k != "step_000000"), key=lambda k: (res["evals"][k]["mean"], k))
    (run / "train_result.json").write_text(json.dumps(res, indent=1))
    (run / "train_log.jsonl").write_text("\n".join(json.dumps(x) for x in log))
    print(json.dumps({k: v for k, v in res.items() if k != "evals"}, indent=1))
    if not res["core_unchanged"]:
        raise SystemExit("TEXT CORE CHANGED")


def select_b(evals: dict) -> tuple[str, dict]:
    """configs/v1/m15b.yaml checkpoint_selection (val_fit only)."""
    def grp(cells, prefix):
        v = [x for k, x in cells.items() if k.startswith(prefix)]
        return float(np.mean(v)) if v else None

    base = evals["step_000000"]["cells"]
    groups = ("audio/text_dependent", "document/text_dependent", "document/hybrid", "image/visual_only")
    detail = {}
    for k, ev in evals.items():
        if k == "step_000000":
            continue
        c = ev["cells"]
        ok = {g: (grp(c, g) or 0) >= (grp(base, g) or 0) - 0.01 for g in groups}
        detail[k] = {"eligible": all(ok.values()), "preservation": ok, "layout": c.get("document/visual_only/multi"),
                     "doc_visual_only": grp(c, "document/visual_only")}
    el = [k for k, v in detail.items() if v["eligible"]] or list(detail)
    best = max(detail[k]["layout"] or 0 for k in el)
    tied = [k for k in el if (detail[k]["layout"] or 0) >= best - 0.005]
    return max(tied, key=lambda k: (detail[k]["doc_visual_only"] or 0, k)), detail


def load_m15(model: M15Model, path: Path, strict_gates: bool = True) -> None:
    from safetensors.torch import load_file

    sd = load_file(str(path))
    model.fusion.load_state_dict({k[len("fusion."):]: v for k, v in sd.items() if k.startswith("fusion.")})
    if len(model.gates):
        model.gates.load_state_dict({k[len("gates."):]: v for k, v in sd.items() if k.startswith("gates.")}, strict=strict_gates)


# ------------------------------------------------------------------------------------------------ dump (serving stack)
def cmd_dump(args) -> None:
    dev = torch.device("cuda")
    model, towers, tok, ser = build(freeze_core=False)
    run = Path(args.run)
    tr = json.loads((run / "train_result.json").read_text())
    load_m15(model, run / tr["selected"] / "m15.safetensors")
    model.eval()
    # A = the M1 model exactly as trained (empty context, truncated Whisper encoder, M1 weights, M1 LoRA)
    m1_towers = T1.Towers("cuda")
    from safetensors.torch import load_file
    m1_fus = M1Fusion(model.core, 768, 768, **T1.CFG["fusion"]).to(dev)
    m1_fus.load_state_dict(load_file(str(ROOT / CFG["m1_init"])))
    m1 = M1Model(model.core, m1_fus, freeze_core=False).eval()
    P = None
    if IS_B:                                              # P = the M1.5 selected checkpoint (gated), same host / harness
        p_fus = M1Fusion(model.core, 768, 768, **T1.CFG["fusion"]).to(dev)
        P = M15Model(M1Model(model.core, p_fus, freeze_core=False), gate=True).to(dev).eval()
        load_m15(P, ROOT / CFG["m15_init"])
    teacher = load_teacher()
    variants = ("ctx", "empty", "gt")
    for sp in ("val_fit", "val_eval"):
        recs = read_split(sp)
        rng = random.Random(0)
        dl = torch.utils.data.DataLoader(Data(recs, ser, None, False, variants), batch_sampler=batches(recs, rng, False),
                                         collate_fn=make_collate(tok.pad_token_id, variants), num_workers=CFG["batching"]["workers"])
        with gzip.open(run / f"signals_{sp}.jsonl.gz", "wt") as f, torch.inference_mode():
            for B in dl:
                feats = towers.features(B)
                qC = fwd(model, towers, B, dev, "gated", "ctx", feats)
                qF = fwd(model, towers, B, dev, "fused", "ctx", feats)
                qB = fwd(model, towers, B, dev, "exact_text", "ctx")
                qO = fwd(model, towers, B, dev, "exact_text", "gt")
                # A: M1 (empty context + M1 media path)
                V = B["variants"]["empty"]
                bA = {"batch": V["batch"], "where": V["where"], "ncand": V["ncand"], "modality": B["modality"],
                      "media": B["media"] if B["modality"] != "audio" else _m1_audio_media(B["media"]["wavs"])}
                qA = T1.forward(m1, m1_towers, bA, dev)
                qP = fwd(P, towers, B, dev, "gated", "ctx", feats) if P is not None else None
                for j, i in enumerate(B["idx"]):
                    r = recs[i]
                    K = len(r["candidates"])
                    row = {k: r.get(k) for k in ("id", "source", "modality", "task_class", "speech", "semantic_group_id", "answerable")}
                    row.update({"type": r["task"]["type"], "lang_media": r["context"]["language"], "lang_schema": r["task"].get("instructions_language"),
                                "slices": r.get("slices") or [], "gold": [int(x) for x in np.flatnonzero(T1.target_of(r) > 0.5)], "k": K,
                                "has_ctx": bool(r["context"]["text"]), "has_gt": bool(r.get("gt_text")), "alpha": round(float(qC["alpha"][j]), 4)})
                    row["candidate_ids"] = [c["id"] for c in r["candidates"]]
                    for tag, q in (("C", qC), ("F", qF), ("B", qB), ("O", qO), ("A", qA)) + ((("P", qP),) if qP is not None else ()):
                        row[f"{tag}_rank"] = [round(float(x), 4) for x in q["rank"][j, :K]]
                        row[f"{tag}_apply"] = [round(float(x), 4) for x in q["apply"][j, :K]]
                    t = teacher.get(r["id"])
                    if t:
                        row["teacher_correct"] = t["correct"]
                    f.write(json.dumps(row) + "\n")
        print("dumped", sp, len(recs), flush=True)


_FE = {}


def _m1_audio_media(wavs):
    from krun_one.v1.multimodal.m1_towers import AudioFrontend

    if "fe" not in _FE:
        _FE["fe"] = AudioFrontend(T1.CFG["towers"]["audio"])
    feats, n = _FE["fe"](wavs)
    return {"feats": feats, "n_frames": n}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["teacher", "health", "train", "dump"])
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--run-name")
    ap.add_argument("--run")
    ap.add_argument("--ocr")
    ap.add_argument("--out")
    a = ap.parse_args()
    {"teacher": cmd_teacher, "health": cmd_health, "train": cmd_train, "dump": cmd_dump}[a.cmd](a)
