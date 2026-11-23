"""Control for G2: are two plain loads of the frozen artifact (no M1) bitwise identical on this GPU? And is the
text path deterministic within one process? Dumps val_fit + val_eval for A1 (process 1, twice) and compares with the
existing textreg A / B dumps.

    python scripts/v1/m1_textreg_control.py --run runs/v1/m1/<run> --tag A2
"""
import argparse, gzip, hashlib, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts/v1"))
from krun_one.v1.runtime import TextArtifact  # noqa: E402
from krun_one.v1.training import dump_signals, encode_records, load_records  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--deterministic", action="store_true")
ap.add_argument("--towers", action="store_true"); ap.add_argument("--fusion", action="store_true"); ap.add_argument("--mm", action="store_true")
ap.add_argument("--reps", type=int, default=2); ap.add_argument("--nograd-params", action="store_true")
a = ap.parse_args()
if a.deterministic:
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.use_deterministic_algorithms(True, warn_only=True)
A = TextArtifact.load(ROOT / "artifacts/krun-one-v1-text-final", device="cuda", dtype="bf16", verify=False)
if a.towers or a.fusion or a.mm:
    import m1_serving_checks as S
    import train_m1 as T
    towers = T.Towers("cuda")
    if a.fusion or a.mm:
        from safetensors.torch import load_file
        from krun_one.v1.multimodal.m1 import M1Fusion, M1Model
        from krun_one.v1.multimodal.m1_towers import AudioFrontend, image_preprocessor
        fus = M1Fusion(A.model, towers.vision.output_dim, towers.audio.output_dim, **T.CFG["fusion"]).cuda()
        tr = json.loads((Path(a.run) / "train_result.json").read_text())
        fus.load_state_dict(load_file(str(Path(a.run) / tr["selected"] / "fusion.safetensors")))
        m = M1Model(A.model, fus, freeze_core=False).eval()
        if a.mm:
            ip, fe = image_preprocessor(T.CFG["towers"]["vision"]), AudioFrontend(T.CFG["towers"]["audio"])
            for kind in ("image", "document", "audio"):
                S.mm_request(m, towers, ip, fe, A, kind)
    print("modes", a.towers, a.fusion, a.mm, "train_mode_core", A.model.training)
if a.nograd_params:
    for p_ in A.model.parameters():
        p_.requires_grad_(False)
    rc = getattr(A.model.encoder.config, "reference_compile", "n/a")
    print("requires_grad False; reference_compile =", rc)
out = Path(a.run) / "textreg" / a.tag
out.mkdir(parents=True, exist_ok=True)
res = {}
for rep in range(1, a.reps + 1):
    for name, p in (("val_fit", ROOT / "data/krunbench_v1/val_fit.jsonl"), ("val_eval", ROOT / "data/krunbench_v1/val_eval.jsonl")):
        recs = load_records(p); items = encode_records(A.ser, recs)
        f = out / f"signals_{name}.rep{rep}.jsonl.gz"
        dump_signals(A.model, items, recs, A.tok.pad_token_id, f, device="cuda")
        res[f"{name}.rep{rep}"] = hashlib.sha256(gzip.open(f).read()).hexdigest()
print(json.dumps(res, indent=1))
