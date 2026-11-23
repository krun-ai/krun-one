"""Krun Decisions v1 — GPU text jobs (RunPod). Standalone: needs only vllm (no krun_one import).

Phase ``qwen``    (Qwen/Qwen3-8B, Apache-2.0, thinking off):
    describe        one-sentence EN description per intent label (from train examples only)
    translate       contexts EN -> pt-BR / es-419 / fr-FR (train + KrunBench v1 VALIDATION items; output keeps the split)
    translate_desc  ToolACE tool descriptions + generated intent descriptions -> pt-BR / es-419 / fr-FR
    pilot_generate  synthetic pilot v2 prompts (2,000)
Phase ``granite`` (ibm-granite/granite-3.3-8b-instruct, Apache-2.0; other family, blind):
    judge           adequacy / fluency of every translation (translate + translate_desc)
    pilot_verify    blind verification of the pilot items (never sees the intended label)

Every job writes <work>/out/<job>.part-NNN.jsonl per chunk and skips finished chunks, so a restarted pod resumes.

    python kd1_gpu_jobs.py qwen --work /workspace/kd1
    python kd1_gpu_jobs.py granite --work /workspace/kd1
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

QWEN = "Qwen/Qwen3-8B"
GRANITE = "ibm-granite/granite-3.3-8b-instruct"
LOCALE_NAME = {"pt-BR": "Brazilian Portuguese", "es-419": "Latin American Spanish", "fr-FR": "French (France)",
               "pt-PT": "European Portuguese (Portugal)", "es-ES": "Spanish (Spain)", "fr-CA": "Canadian French (Québec)"}
CATALOG_DOMAIN = {"banking77": "an online banking customer-support assistant",
                  "clinc150": "a general-purpose virtual assistant (banking, travel, home, work, utilities, small talk)",
                  "massive": "a voice assistant", "snips": "a voice assistant",
                  "bitext": "an e-commerce customer-support assistant"}
CHUNK = 8000


def jl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def parse_json(s: str):
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class Runner:
    def __init__(self, model: str, work: Path, max_len: int = 4096):
        from vllm import LLM

        self.model, self.work = model, work
        t0 = time.time()
        self.llm = LLM(model=model, dtype="bfloat16", max_model_len=max_len, gpu_memory_utilization=0.90,
                       seed=20260927, enable_prefix_caching=True)
        self.load_s = time.time() - t0
        self.stats: dict[str, dict] = {}

    def run(self, job: str, rows: list[dict], make_prompt, sampling, post, thinking_off: bool = True) -> list[dict]:
        """Chunked, resumable chat generation. post(row, text) -> dict merged into the output row."""
        out_dir = self.work / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        results = []
        st = self.stats.setdefault(job, {"requests": 0, "seconds": 0.0, "prompt_tokens": 0, "output_tokens": 0})
        for ci in range(0, len(rows), CHUNK):
            part = out_dir / f"{job}.part-{ci // CHUNK:03d}.jsonl"
            if part.exists():
                results += jl(part)
                continue
            chunk = rows[ci:ci + CHUNK]
            msgs = [[{"role": "user", "content": make_prompt(r)}] for r in chunk]
            kw = {"chat_template_kwargs": {"enable_thinking": False}} if thinking_off else {}
            t0 = time.time()
            res = self.llm.chat(msgs, sampling, use_tqdm=False, **kw)
            dt = time.time() - t0
            rows_out = []
            for r, o in zip(chunk, res):
                txt = o.outputs[0].text
                rows_out.append({**r, "raw": txt, **post(r, txt)})
            st["requests"] += len(chunk)
            st["seconds"] += dt
            st["prompt_tokens"] += sum(len(o.prompt_token_ids) for o in res)
            st["output_tokens"] += sum(len(o.outputs[0].token_ids) for o in res)
            tmp = part.with_suffix(".tmp")
            with open(tmp, "w") as f:
                for r in rows_out:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            tmp.rename(part)
            results += rows_out
            print(json.dumps({"job": job, "chunk": ci // CHUNK, "n": len(chunk), "s": round(dt, 1),
                              "out_tok_s": round(st["output_tokens"] / max(st["seconds"], 1e-9))}), flush=True)
        return results

    def dump_stats(self, name: str):
        p = self.work / "out" / f"stats_{name}.json"
        old = json.loads(p.read_text()) if p.exists() else {}
        old.update({"model": self.model, "load_seconds": self.load_s, "jobs": {**old.get("jobs", {}), **self.stats}})
        p.write_text(json.dumps(old, indent=2))


# ------------------------------------------------------------------------------------------------ prompts
def p_describe(r):
    ex = "\n".join(f'- "{e}"' for e in r["examples"])
    return (f"You document the intents of {CATALOG_DOMAIN.get(r['catalog'], 'an assistant')}.\n"
            f"Intent label: {r['label']}\nExample user messages with this intent:\n{ex}\n\n"
            "Write ONE sentence (at most 20 words) that a developer would put as this intent's description in an API "
            "schema: say precisely what the user wants, so it can be told apart from similar intents. Start with a "
            "verb (e.g. 'Report ...', 'Check ...'). Output only the sentence.")


def p_translate(r):
    lang = LOCALE_NAME[r["target_locale"]]
    return (f"Translate the user message below from English into {lang}, the way a native speaker would type it to an "
            "assistant. Rules: keep the meaning exactly; keep the register (casual stays casual); keep names, numbers, "
            "dates, codes, URLs, e-mail addresses, product/brand names and placeholders in double curly braces (like "
            "{{Order Number}}) unchanged; do not answer the message; no quotes, no notes. Output only the translation.\n\n"
            f"Message: {r['text']}")


def p_translate_desc(r):
    lang = LOCALE_NAME[r["target_locale"]]
    return (f"Translate this API/intent description from English into {lang}. Keep function names, parameter names, "
            "code identifiers, product names and units unchanged. Output only the translation.\n\n"
            f"Description: {r['text']}")


def p_judge(r):
    lang = LOCALE_NAME[r["target_locale"]]
    return ("You check machine translations of chatbot data.\n"
            f"Source (English): {r['text']}\nTranslation ({lang}): {r['output']}\n\n"
            "Rate ADEQUACY (1 = meaning lost or changed, 5 = same meaning, nothing added or missing, names/numbers/"
            f"placeholders preserved) and FLUENCY (1 = broken or not {lang}, 5 = natural {lang}).\n"
            'Return ONLY JSON: {"adequacy": 1-5, "fluency": 1-5}')


def p_verify(it):
    d = it["decision"]
    opts = "\n".join(f"- {cid}: {desc}" if desc else f"- {cid}" for cid, desc in d["options"])
    multi = it["kind"] == "underspecified"
    ask = ("List EVERY option that is a reasonable interpretation of the message (empty list if none)" if multi else
           "Choose the ONE option the message asks for, or NONE if no option applies")
    fmt = '["option_id", ...]' if multi else '"option_id" or "NONE"'
    return ("You are a careful annotator. " + ask + ". Also rate how natural the message sounds for a real user "
            "(1 = broken/unnatural, 5 = perfectly natural).\n"
            f'Return ONLY JSON: {{"answer": {fmt}, "naturalness": 1-5}}\n\n'
            f"Message: \"{d['text']}\"\n\nOptions:\n{opts}")


def clean_line(t: str) -> str:
    t = t.strip().split("\n")[0].strip()
    return t.strip('"“”«»').strip()


def pilot_decision(it: dict) -> dict | None:
    p, k = it.get("parsed"), it["kind"]
    if not isinstance(p, dict):
        return None
    opts = [(cid, it["verifier_options"].get(cid)) for cid in it["candidates"]]
    if k == "hard_negative":
        name, desc = p.get("name"), p.get("description")
        if not name or not desc:
            return None
        nid = re.sub(r"[^\w\-/. ]+", "", str(name)).strip()[:64]
        if not nid or nid in it["candidates"]:
            return None
        opts = opts + [(nid, str(desc)[:300])]
        return {"text": it["source_item"]["text"], "options": opts, "candidate_ids": [c for c, _ in opts],
                "intended": [it["gold"]], "new_tool": {"id": nid, "description": str(desc)[:300]}}
    text = p.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    intended = {"minimal_pair": [it.get("target")], "counterfactual": [it.get("target")], "near_ood": [],
                "underspecified": sorted(it.get("pair") or [])}.get(k, [it.get("gold")])
    return {"text": text.strip(), "options": opts, "candidate_ids": [c for c, _ in opts], "intended": intended,
            "why_none": p.get("why_none")}


# ------------------------------------------------------------------------------------------------ phases
def phase_qwen(work: Path):
    from vllm import SamplingParams

    R = Runner(QWEN, work)
    jobs = work / "gpu_jobs"
    desc = R.run("describe", jl(jobs / "describe.jsonl"), p_describe, SamplingParams(temperature=0.2, max_tokens=60),
                 lambda r, t: {"output": clean_line(t)})
    R.dump_stats("qwen")
    tr = jl(jobs / "translate.jsonl")
    R.run("translate", tr, p_translate, SamplingParams(temperature=0.2, top_p=0.9, max_tokens=400),
          lambda r, t: {"output": t.strip().strip('"“”')})
    R.dump_stats("qwen")
    td = jl(jobs / "translate_desc.jsonl") + [
        {"job": "translate_desc", "key": f"intent::{d['key']}>{loc}", "name": d["label"], "catalog": d["catalog"],
         "text": d["output"], "target_locale": loc} for d in desc for loc in ("pt-BR", "es-419", "fr-FR")]
    R.run("translate_desc", td, p_translate_desc, SamplingParams(temperature=0.2, max_tokens=200),
          lambda r, t: {"output": clean_line(t)})
    R.dump_stats("qwen")
    pil = work / "pilot/prompts.jsonl"
    if pil.exists():
        R.run("pilot_generate", jl(pil), lambda r: r["prompt"],
              SamplingParams(temperature=0.8, top_p=0.95, max_tokens=220, seed=20260927),
              lambda r, t: {"parsed": parse_json(t)})
    R.dump_stats("qwen")


def phase_granite(work: Path):
    from vllm import SamplingParams

    R = Runner(GRANITE, work)
    out = work / "out"
    rows = []
    for job in ("translate", "translate_desc"):
        for f in sorted(out.glob(f"{job}.part-*.jsonl")):
            rows += [{"key": r["key"], "job": job, "text": r["text"], "output": r["output"],
                      "target_locale": r["target_locale"]} for r in jl(f)]
    R.run("judge", rows, p_judge, SamplingParams(temperature=0.0, max_tokens=30),
          lambda r, t: {"verdict": parse_json(t)})
    R.dump_stats("granite")
    gen = sorted(out.glob("pilot_generate.part-*.jsonl"))
    if gen:
        items = [it for f in gen for it in jl(f)]
        todo = []
        for it in items:
            it["abstained"] = isinstance(it.get("parsed"), dict) and it["parsed"].get("text", "x") is None \
                and it["kind"] != "hard_negative"
            it["decision"] = None if it["abstained"] else pilot_decision(it)
            if it["decision"]:
                todo.append(it)
        done = R.run("pilot_verify", todo, p_verify, SamplingParams(temperature=0.0, max_tokens=80),
                     lambda r, t: {"verifier": parse_json(t)})
        by = {d["pid"]: d for d in done}
        with open(work / "pilot/verified.jsonl", "w") as f:
            for it in items:
                f.write(json.dumps(by.get(it["pid"], it), ensure_ascii=False) + "\n")
    R.dump_stats("granite")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["qwen", "granite"])
    ap.add_argument("--work", default="/workspace/kd1")
    a = ap.parse_args()
    {"qwen": phase_qwen, "granite": phase_granite}[a.phase](Path(a.work))


if __name__ == "__main__":
    main()
