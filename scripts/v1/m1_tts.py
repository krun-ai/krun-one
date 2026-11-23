"""M1 synthetic speech with Kokoro-82M (Apache-2.0; separate venv: kokoro + soundfile + scipy + espeak-ng).

    python scripts/v1/m1_tts.py --jobs data/m1/tts_jobs.jsonl --shard 0 --nshards 3 --budget-min 30

Reads jobs in file order (evaluation-side first), writes data/m1/media/tts/<id>.flac (16 kHz mono) and appends
{"id", "voice", "lang", "seconds"} to data/m1/tts_manifest.<shard>.jsonl. Stops at the time budget. Voice and speed are
a pure function of the job id. Kokoro languages: a/b = en-US/en-GB, e = es, f = fr, p = pt-BR (also used for pt-PT).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
VOICES = {"a": ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "am_adam", "am_michael", "am_fenrir", "am_puck"],
          "b": ["bf_emma", "bf_isabella", "bm_george", "bm_lewis"],
          "e": ["ef_dora", "em_alex", "em_santa"], "f": ["ff_siwis"], "p": ["pf_dora", "pm_alex", "pm_santa"]}


def lang_code(tag: str, h: int) -> str:
    l = tag.split("-")[0]
    return {"en": "a" if h % 3 else "b", "es": "e", "fr": "f", "pt": "p"}[l]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default=str(ROOT / "data/m1/tts_jobs.jsonl"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--budget-min", type=float, default=30)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from kokoro import KPipeline

    out = ROOT / "data/m1/media/tts"
    out.mkdir(parents=True, exist_ok=True)
    man = ROOT / f"data/m1/tts_manifest.{a.shard}.jsonl"
    done = {json.loads(l)["id"] for l in open(man)} if man.exists() else set()
    jobs = [json.loads(l) for l in open(a.jobs, encoding="utf-8")]
    jobs = [j for i, j in enumerate(jobs) if i % a.nshards == a.shard]
    if a.limit:
        jobs = jobs[: a.limit]
    pipes: dict = {}
    t0 = time.time()
    n = secs = 0
    with open(man, "a") as mf:
        for j in jobs:
            if j["id"] in done:
                continue
            if time.time() - t0 > a.budget_min * 60:
                break
            h = int(hashlib.sha1(j["id"].encode()).hexdigest()[:8], 16)
            lc = lang_code(j["lang"], h)
            if lc not in pipes:
                pipes[lc] = KPipeline(lang_code=lc, repo_id="hexgrad/Kokoro-82M")
            voice = VOICES[lc][h % len(VOICES[lc])]
            speed = 0.85 + (h % 31) / 100.0         # 0.85 .. 1.15
            try:
                chunks = [np.asarray(audio, dtype=np.float32) for _, _, audio in pipes[lc](j["text"], voice=voice, speed=speed)
                          if audio is not None]
            except Exception as e:  # noqa: BLE001
                print("fail", j["id"], e, flush=True)
                continue
            if not chunks:
                continue
            w = resample_poly(np.concatenate(chunks), 2, 3).astype(np.float32)   # 24 kHz -> 16 kHz
            sf.write(out / f"{j['id']}.flac", w, 16000)
            mf.write(json.dumps({"id": j["id"], "voice": voice, "lang": j["lang"], "speed": round(speed, 2),
                                 "seconds": round(len(w) / 16000, 2)}) + "\n")
            n += 1
            secs += len(w) / 16000
            if n % 500 == 0:
                mf.flush()
                print(f"shard {a.shard}: {n} clips, {secs / 3600:.2f} h audio, {time.time() - t0:.0f} s", flush=True)
    print(f"shard {a.shard} done: {n} clips, {secs / 3600:.2f} h audio in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
