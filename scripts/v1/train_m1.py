"""Krun One v1 — M1 multimodal alignment training (configs/v1/m1.yaml; registered before training).

    python scripts/v1/train_m1.py teacher               # frozen text teacher on paired texts -> data/m1/teacher.jsonl.gz
    python scripts/v1/train_m1.py health                # pod health check: CUDA, tiny forward, towers, fusion, backward
    python scripts/v1/train_m1.py train [--max-steps N] [--run-name NAME]
    python scripts/v1/train_m1.py dump --run runs/v1/m1/<name> [--ckpt best]     # val_fit + val_eval signals

Frozen: the text core (artifacts/krun-one-v1-text-final, sha-verified; digest checked before/after), SigLIP2 and
Whisper encoders. Trainable: image adapter (shared by visual documents), audio adapter, modality embeddings, head LoRA.
Loss = decision loss (the text objective, configs/v1/stage2.yaml loss) + lambda * consistency (KL to the frozen text
teacher on the paired text; for teacher_filter=if_correct only where the teacher is right). protected is never read.
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
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from krun_one.v1.arch import QTYPES_V1, ModelSpec  # noqa: E402
from krun_one.v1.losses import LossConfig, decision_loss  # noqa: E402
from krun_one.v1.model import NEG, KrunOneV1Model, chunk_index, collate_chunks, merge_chunks  # noqa: E402
from krun_one.v1.multimodal.features import ModalityFeatures  # noqa: E402
from krun_one.v1.multimodal.m1 import M1Fusion, M1Model, core_digest, trainable_parameter_count  # noqa: E402
from krun_one.v1.serialization import Budget, Serializer  # noqa: E402

CFG = yaml.safe_load((ROOT / "configs/v1/m1.yaml").read_text())
D = ROOT / "data/m1"
ART = ROOT / CFG["base_artifact"]
KIND = {"image": "image", "document": "document", "audio": "audio"}


# ------------------------------------------------------------------------------------------------ loading
def load_core(device="cuda") -> tuple[KrunOneV1Model, object, Serializer]:
    from safetensors.torch import load_model
    from transformers import AutoTokenizer

    from krun_one.v1.runtime import sha256_file

    if sha256_file(ART / "model.safetensors") != CFG["base_sha256"]:
        raise SystemExit("base artifact sha mismatch: M1 may only start from krun-one-v1-text-final")
    m = KrunOneV1Model.from_spec(ModelSpec.from_dict(json.loads((ART / "spec.json").read_text())))
    load_model(m, str(ART / "model.safetensors"))
    m.to(device).eval()
    tok = AutoTokenizer.from_pretrained(str(ART / "tokenizer"))
    sq = json.loads((ART / "decision_config.json").read_text())["sequence"]
    ser = Serializer(tok, Budget(sq["max_len"], sq["head_max_len"], max_candidates_per_chunk=sq["max_candidates_per_chunk"]))
    return m, tok, ser


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


def target_of(r: dict) -> np.ndarray:
    ids = [c["id"] for c in r["candidates"]]
    t = np.zeros(len(ids), np.float32)
    for lab in r["target"]["labels"]:
        t[ids.index(lab)] = 1
    return t


# ------------------------------------------------------------------------------------------------ data
class M1Data(torch.utils.data.Dataset):
    def __init__(self, recs, ser, teacher, train: bool):
        self.recs, self.ser, self.teacher, self.train = recs, ser, teacher, train
        self._img = None
        self._aud = None

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        q = self.ser.encode_record(r)
        t = self.teacher.get(r["id"]) if self.teacher is not None else None
        item = {"i": i, "q": q, "target": target_of(r), "qtype": QTYPES_V1[r["task"]["type"]],
                "answerable": bool(r["answerable"]), "modality": r["modality"], "teacher": t}
        p = D / r["media"]["path"]
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


def make_collate(pad_id: int):
    state = {}

    def collate(items):
        from krun_one.v1.multimodal.m1_towers import PATCHES, AudioFrontend, image_preprocessor

        qs = [it["q"] for it in items]
        flat, where = chunk_index(qs)
        b = collate_chunks(flat, pad_id)
        kmax = max(q.n_candidates for q in qs)
        tgt = torch.zeros(len(items), kmax)
        tr = torch.full((len(items), kmax), NEG)
        ta = torch.full((len(items), kmax), NEG)
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
        assert all(it["modality"] == mod for it in items), "batches are single-modality"
        if mod == "audio":
            if "fe" not in state:
                state["fe"] = AudioFrontend(CFG["towers"]["audio"])
            feats, n = state["fe"]([it["wav"] for it in items])
            media = {"feats": feats, "n_frames": n}
        else:
            if "ip" not in state:
                state["ip"] = image_preprocessor(CFG["towers"]["vision"])
            media = dict(state["ip"](images=[it["image"] for it in items], max_num_patches=PATCHES[mod], return_tensors="pt"))
        return {"batch": b, "where": where, "ncand": [q.n_candidates for q in qs], "target": tgt,
                "qtype": torch.tensor([it["qtype"] for it in items]), "answerable": torch.tensor([it["answerable"] for it in items]),
                "teacher_rank": tr, "teacher_apply": ta, "has_teacher": has_t, "modality": mod, "media": media,
                "idx": [it["i"] for it in items]}

    return collate


def batches(recs, rng: random.Random, shuffle=True) -> list[list[int]]:
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


# ------------------------------------------------------------------------------------------------ model step
class Towers:
    def __init__(self, device="cuda"):
        from krun_one.v1.multimodal.m1_towers import SigLIP2Tower, WhisperEncoderTower

        self.vision = SigLIP2Tower(CFG["towers"]["vision"], device=device)
        self.audio = WhisperEncoderTower(CFG["towers"]["audio"], device=device)

    def features(self, B) -> ModalityFeatures:
        m = B["media"]
        if B["modality"] == "audio":
            h, mask = self.audio.encode(m["feats"], m["n_frames"])
        else:
            h, mask = self.vision.encode(m)
        return ModalityFeatures(B["modality"], h, mask)


def seq_request(where) -> list[int]:
    rows = sorted(where, key=lambda x: x[1])
    return [qi for qi, _, _ in rows]


def forward(model: M1Model, towers: Towers, B, dev, blind=False, use_lora=True):
    b = {k: v.to(dev, non_blocking=True) for k, v in B["batch"].items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if blind:
            out = model.forward_text(b)
        else:
            f = towers.features(B)
            tok, m = model.fusion.media_tokens(f)
            out = model.forward_mm(b, (tok, m), seq_request(B["where"]), use_lora=use_lora)
    return merge_chunks(out, B["where"], len(B["ncand"]), B["ncand"])


def consistency(q, B, dev) -> torch.Tensor:
    """KL(teacher || student) on the rank distribution (choice / noul / score); BCE to teacher sigmoid (multi)."""
    ht = B["has_teacher"].to(dev)
    if not ht.any():
        return torch.zeros((), device=dev)
    mask = q["mask"]
    tr = B["teacher_rank"].to(dev)[:, :mask.size(1)].masked_fill(~mask, NEG)
    qt = B["qtype"].to(dev)
    zs = q["rank"].masked_fill(~mask, NEG)
    pt = torch.softmax(tr, -1)
    kl = (pt * (torch.log_softmax(tr, -1) - torch.log_softmax(zs, -1))).masked_fill(~mask, 0).sum(-1)
    is_multi = qt == QTYPES_V1["multi"]
    per = torch.where(is_multi, torch.zeros_like(kl), kl)
    if is_multi.any() and "apply" in q:
        ta = B["teacher_apply"].to(dev)[:, :mask.size(1)]
        bce = F.binary_cross_entropy_with_logits(q["apply"], torch.sigmoid(ta), reduction="none") * mask.float()
        bce = bce.sum(-1) / mask.sum(-1).clamp(min=1).float()
        per = torch.where(is_multi, bce, per)
    return (per * ht.float()).sum() / ht.float().sum().clamp(min=1)


def build(device="cuda", freeze_core=True):
    core, tok, ser = load_core(device)
    towers = Towers(device)
    fusion = M1Fusion(core, towers.vision.output_dim, towers.audio.output_dim, **CFG["fusion"]).to(device)
    return M1Model(core, fusion, freeze_core=freeze_core), towers, tok, ser


# ------------------------------------------------------------------------------------------------ teacher
@torch.no_grad()
def cmd_teacher(args) -> None:
    core, tok, ser = load_core()
    out = D / "teacher.jsonl.gz"
    n = n_use = 0
    stats = {}
    with gzip.open(out, "wt") as f:
        for sp in ("train", "val_fit", "val_eval"):
            recs = [r for r in read_split(sp) if r.get("paired_text")]
            order = sorted(range(len(recs)), key=lambda i: len(recs[i]["paired_text"]))
            for s in range(0, len(order), 64):
                rs = [recs[i] for i in order[s:s + 64]]
                enc = [ser.encode_record(dict(r, context={"text": r["paired_text"], "language": r["context"]["language"]})) for r in rs]
                flat, where = chunk_index(enc)
                b = {k: v.cuda() for k, v in collate_chunks(flat, tok.pad_token_id).items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = core(**b)
                q = merge_chunks(o, where, len(rs), [e.n_candidates for e in enc])
                for j, r in enumerate(rs):
                    K = len(r["candidates"])
                    rank = q["rank"][j, :K].float().cpu().numpy()
                    app = q["apply"][j, :K].float().cpu().numpy() if "apply" in q else None
                    gold = target_of(r)
                    t = r["task"]["type"]
                    if t == "multi":
                        ok = bool(((1 / (1 + np.exp(-app)) > 0.35) == (gold > 0.5)).all())
                    else:
                        ok = bool(r["answerable"] and gold[int(np.argmax(rank))] > 0.5)
                    use = r.get("teacher_filter") == "always" or ok
                    key = f"{sp}/{r['source']}/{t}"
                    st = stats.setdefault(key, [0, 0, 0])
                    st[0] += 1
                    st[1] += ok
                    st[2] += use
                    f.write(json.dumps({"id": r["id"], "rank": [round(float(x), 4) for x in rank],
                                        "apply": None if app is None else [round(float(x), 4) for x in app],
                                        "correct": ok, "use": use}) + "\n")
                    n += 1
                    n_use += use
    res = {"n": n, "used": n_use, "by": {k: {"n": v[0], "teacher_acc": v[1] / v[0], "used": v[2]} for k, v in sorted(stats.items())}}
    (ROOT / "results/v1/m1").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/v1/m1/teacher_stats.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def load_teacher() -> dict:
    p = D / "teacher.jsonl.gz"
    return {json.loads(l)["id"]: json.loads(l) for l in gzip.open(p, "rt")} if p.exists() else {}


# ------------------------------------------------------------------------------------------------ health
def cmd_health(args) -> None:
    res = {"cuda": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0)}
    x = torch.randn(256, 256, device="cuda")
    res["tiny_forward"] = float((x @ x).sum().item()) == float((x @ x).sum().item())
    model, towers, tok, ser = build()
    d0 = core_digest(model.core)
    from PIL import Image

    from krun_one.v1.multimodal.m1_towers import PATCHES, AudioFrontend, image_preprocessor

    ip = image_preprocessor(CFG["towers"]["vision"])
    im = Image.new("RGB", (640, 480), (120, 30, 200))
    mv = dict(ip(images=[im, im], max_num_patches=PATCHES["image"], return_tensors="pt"))
    hv, mvk = towers.vision.encode(mv)
    res["vision_forward"] = list(hv.shape)
    feats, n = AudioFrontend(CFG["towers"]["audio"])([np.random.randn(16000 * 5).astype(np.float32) * 0.1] * 2)
    ha, mak = towers.audio.encode(feats, n)
    res["audio_forward"] = list(ha.shape)
    rec = {"task": {"type": "choice", "instructions": "Which object is the main subject of the image?"}, "context": {"text": ""},
           "candidates": [{"id": "dog"}, {"id": "cat"}]}
    enc = [ser.encode_record(rec), ser.encode_record(rec)]
    flat, where = chunk_index(enc)
    b = {k: v.cuda() for k, v in collate_chunks(flat, tok.pad_token_id).items()}
    model.fusion.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        t, m = model.fusion.media_tokens(ModalityFeatures("image", hv, mvk))
        o = model.forward_mm(b, (t, m), seq_request(where))
        q = merge_chunks(o, where, 2, [2, 2])
        loss = F.cross_entropy(q["rank"], torch.tensor([0, 1], device="cuda"))
        ta, ma = model.fusion.media_tokens(ModalityFeatures("audio", ha, mak))
        oa = model.forward_mm(b, (ta, ma), seq_request(where))
        loss = loss + merge_chunks(oa, where, 2, [2, 2])["rank"].sum() * 0
    res["fusion_forward"] = list(t.shape)
    loss.backward()
    core_grads = [n_ for n_, p in model.core.named_parameters() if p.grad is not None]
    fus_grads = sum(1 for p in model.fusion.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    tower_grads = [n_ for n_, p in list(towers.vision.named_parameters()) + list(towers.audio.named_parameters()) if p.grad is not None]
    res["backward_only_trainable"] = not core_grads and not tower_grads and fus_grads > 0
    res["core_digest_unchanged"] = core_digest(model.core) == d0
    res["trainable_params"] = trainable_parameter_count(model.fusion)
    res["frozen_params"] = {"text_core": sum(p.numel() for p in model.core.parameters()),
                            "vision_tower": sum(p.numel() for p in towers.vision.model.parameters()),
                            "audio_tower": sum(p.numel() for p in towers.audio.enc.parameters())}
    res["PASS"] = bool(res["cuda"] and res["tiny_forward"] and res["backward_only_trainable"] and res["core_digest_unchanged"])
    print(json.dumps(res, indent=1))
    (ROOT / "results/v1/m1").mkdir(parents=True, exist_ok=True)
    Path(args.out or ROOT / "results/v1/m1/health.json").write_text(json.dumps(res, indent=1))
    if not res["PASS"]:
        raise SystemExit(1)


# ------------------------------------------------------------------------------------------------ train
def evaluate_quick(model, towers, recs, ser, tok, dev, limit=4000) -> dict:
    rng = random.Random(0)
    sub = recs if len(recs) <= limit else rng.sample(recs, limit)
    ds = M1Data(sub, ser, None, train=False)
    dl = torch.utils.data.DataLoader(ds, batch_sampler=batches(sub, rng, shuffle=False), collate_fn=make_collate(tok.pad_token_id),
                                     num_workers=CFG["batching"]["workers"])
    cells = {}
    model.eval()
    with torch.no_grad():
        for B in dl:
            q = forward(model, towers, B, dev)
            for j, i in enumerate(B["idx"]):
                r = sub[i]
                K = len(r["candidates"])
                g = target_of(r)
                t = r["task"]["type"]
                if t == "multi":
                    pr = (torch.sigmoid(q["apply"][j, :K]).cpu().numpy() > 0.5)
                    ok = float((pr == (g > 0.5)).all())
                elif t == "choice" and not r["answerable"]:
                    continue
                else:
                    ok = float(g[int(q["rank"][j, :K].argmax())] > 0.5)
                cells.setdefault(f"{r['modality']}/{t}", []).append(ok)
    model.train()
    acc = {k: float(np.mean(v)) for k, v in cells.items()}
    return {"cells": acc, "mean": float(np.mean(list(acc.values())))}


def cmd_train(args) -> None:
    from safetensors.torch import save_file

    torch.manual_seed(CFG["seed"])
    rng = random.Random(CFG["seed"])
    dev = torch.device("cuda")
    model, towers, tok, ser = build()
    digest0 = core_digest(model.core)
    run = ROOT / "runs/v1/m1" / (args.run_name or CFG["name"])
    run.mkdir(parents=True, exist_ok=True)
    train = read_split("train", args.limit)
    val_fit = read_split("val_fit")
    teacher = load_teacher()
    epochs = CFG["optimizer"]["epochs"]
    plan = [batches(train, rng) for _ in range(epochs)]
    total = sum(len(p) for p in plan)
    if args.max_steps:
        total = min(total, args.max_steps)
    oc = CFG["optimizer"]
    lora = [p for n, p in model.fusion.named_parameters() if n.startswith("lora")]
    rest = [p for n, p in model.fusion.named_parameters() if not n.startswith("lora")]
    opt = torch.optim.AdamW([{"params": rest, "lr": oc["lr_adapters"]}, {"params": lora, "lr": oc["lr_lora"]}],
                            weight_decay=oc["weight_decay"], betas=(0.9, 0.98))
    base = [g["lr"] for g in opt.param_groups]
    warm = max(1, int(oc["warmup_ratio"] * total))
    lcfg = LossConfig(**CFG["loss"])
    lam = CFG["consistency_weight"]
    milestones = {max(1, int(f * total)) for f in CFG["milestones"]}
    log = []
    step = 0
    t0 = time.time()
    model.train()
    coll = make_collate(tok.pad_token_id)
    res = {"trainable_params": trainable_parameter_count(model.fusion), "total_steps": total, "n_train": len(train),
           "core_digest_start": digest0, "evals": {}}
    for ep, plan_ep in enumerate(plan):
        ds = M1Data(train, ser, teacher, train=True)
        dl = torch.utils.data.DataLoader(ds, batch_sampler=plan_ep, collate_fn=coll, num_workers=CFG["batching"]["workers"],
                                         prefetch_factor=4, persistent_workers=False)
        for B in dl:
            if step >= total:
                break
            lr_f = step / warm if step < warm else max(0.0, (total - step) / max(1, total - warm))
            for g, b0 in zip(opt.param_groups, base):
                g["lr"] = b0 * lr_f
            q = forward(model, towers, B, dev)
            ld, parts = decision_loss(q, B["qtype"].to(dev), B["target"].to(dev), B["answerable"].to(dev), lcfg)
            lc = consistency(q, B, dev)
            loss = ld + lam * lc
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.fusion.parameters(), oc["max_grad_norm"])
            opt.step()
            step += 1
            if step % 50 == 0 or step == 1:
                e = {"step": step, "epoch": ep, "loss": float(loss), "decision": float(ld), "consistency": float(lc),
                     "grad_norm": float(gn), "lr": opt.param_groups[0]["lr"], "modality": B["modality"],
                     "sec": round(time.time() - t0, 1), "vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
                log.append(e)
                print(json.dumps(e), flush=True)
            if step in milestones:
                ck = run / f"step_{step:06d}"
                ck.mkdir(exist_ok=True)
                save_file({k: v.detach().cpu().contiguous() for k, v in model.fusion.state_dict().items()}, str(ck / "fusion.safetensors"))
                ev = evaluate_quick(model, towers, val_fit, ser, tok, dev)
                res["evals"][ck.name] = ev
                print("EVAL", ck.name, json.dumps(ev), flush=True)
        if step >= total:
            break
    res["train_seconds"] = time.time() - t0
    res["core_digest_end"] = core_digest(model.core)
    res["core_unchanged"] = res["core_digest_end"] == digest0
    best = max(res["evals"], key=lambda k: (res["evals"][k]["mean"], k))   # registered rule: val_fit mean cell accuracy
    res["selected"] = best
    (run / "fusion_config.json").write_text(json.dumps(model.fusion.config, indent=1))
    (run / "train_result.json").write_text(json.dumps(res, indent=1))
    (run / "train_log.jsonl").write_text("\n".join(json.dumps(x) for x in log))
    print(json.dumps({k: v for k, v in res.items() if k != "evals"}, indent=1))
    if not res["core_unchanged"]:
        raise SystemExit("TEXT CORE CHANGED — M1 invalid")


# ------------------------------------------------------------------------------------------------ dump
def cmd_dump(args) -> None:
    from safetensors.torch import load_file

    dev = torch.device("cuda")
    model, towers, tok, ser = build(freeze_core=False)          # dumps = the SERVING stack (core exactly as loaded)
    run = Path(args.run)
    tr = json.loads((run / "train_result.json").read_text())
    ck = run / (tr["selected"] if args.ckpt == "best" else args.ckpt)
    model.fusion.load_state_dict(load_file(str(ck / "fusion.safetensors")))
    model.eval()
    teacher = load_teacher()
    for sp in ("val_fit", "val_eval"):
        recs = read_split(sp)
        rng = random.Random(0)
        dl = torch.utils.data.DataLoader(M1Data(recs, ser, None, train=False), batch_sampler=batches(recs, rng, shuffle=False),
                                         collate_fn=make_collate(tok.pad_token_id), num_workers=CFG["batching"]["workers"])
        with gzip.open(run / f"signals_{sp}.jsonl.gz", "wt") as f, torch.inference_mode():
            for B in dl:
                qs = forward(model, towers, B, dev)
                qb = forward(model, towers, B, dev, blind=True)
                qn = forward(model, towers, B, dev, use_lora=False) if args.ablate_lora else None
                for j, i in enumerate(B["idx"]):
                    r = recs[i]
                    K = len(r["candidates"])
                    row = {"id": r["id"], "source": r["source"], "modality": r["modality"], "type": r["task"]["type"],
                           "family": r["task"].get("family"), "lang_media": r["context"]["language"],
                           "lang_schema": r["task"].get("instructions_language"), "slices": r.get("slices") or [],
                           "answerable": r["answerable"], "gold": [int(x) for x in np.flatnonzero(target_of(r) > 0.5)],
                           "k": K, "cross_modal_group": r.get("cross_modal_group"),
                           "rank": [round(float(x), 4) for x in qs["rank"][j, :K]],
                           "apply": [round(float(x), 4) for x in qs["apply"][j, :K]],
                           "blind_rank": [round(float(x), 4) for x in qb["rank"][j, :K]],
                           "blind_apply": [round(float(x), 4) for x in qb["apply"][j, :K]]}
                    if qn is not None:
                        row["nolora_rank"] = [round(float(x), 4) for x in qn["rank"][j, :K]]
                        row["nolora_apply"] = [round(float(x), 4) for x in qn["apply"][j, :K]]
                    t = teacher.get(r["id"])
                    if t:
                        row["teacher_rank"], row["teacher_apply"], row["teacher_correct"] = t["rank"], t["apply"], t["correct"]
                    f.write(json.dumps(row) + "\n")
        print("dumped", sp, len(recs), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["teacher", "health", "train", "dump"])
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--run-name")
    ap.add_argument("--run")
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--ablate-lora", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    {"teacher": cmd_teacher, "health": cmd_health, "train": cmd_train, "dump": cmd_dump}[a.cmd](a)
