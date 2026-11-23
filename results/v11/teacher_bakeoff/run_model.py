"""Run one Ollama model over the fixed sample. Usage:
  run_model.py MODEL [--think false|true|low|medium|high] [--subset] [--conc N] [--tag TAG] [--num-predict N] [--num-ctx N] [--deadline SECONDS] [--ids-file F]
Shows the model ONLY: instructions, context, candidate names (software_repair: "name: path"). Never feat/value for other tracks."""
import json, sys, time, re, argparse, urllib.request, threading, os
from concurrent.futures import ThreadPoolExecutor
ap = argparse.ArgumentParser()
ap.add_argument('model'); ap.add_argument('--think', default='false'); ap.add_argument('--subset', action='store_true')
ap.add_argument('--conc', type=int, default=1); ap.add_argument('--tag', default=''); ap.add_argument('--num-predict', type=int, default=64)
ap.add_argument('--num-ctx', type=int, default=0); ap.add_argument('--deadline', type=float, default=1e9); ap.add_argument('--timeout', type=float, default=300)
ap.add_argument('--limit', type=int, default=0); ap.add_argument('--ids-file', default='')
a = ap.parse_args()
VAL = str(__import__('pathlib').Path(__file__).resolve().parents[3] / 'data/v11/techbench/val.jsonl')
S = json.load(open('sample_ids.json'))
ids = S['subset60_ids'] if a.subset else S['ids']
if a.limit: ids = ids[:a.limit]
if a.ids_file: ids = json.load(open(a.ids_file))
want = set(ids)
ex = {}
for l in open(VAL):
    r = json.loads(l)
    if r['id'] in want: ex[r['id']] = r
label = re.sub(r'[^A-Za-z0-9._-]+', '_', a.model) + (('__' + a.tag) if a.tag else '')
out_path = f'raw_{label}.jsonl'
done = set()
if os.path.exists(out_path):
    for l in open(out_path):
        d = json.loads(l)
        if not d.get('error'): done.add(d['id'])
SYSTEM = ('You are a careful senior engineer acting as a decision judge. Read the task and the numbered candidates, '
          'and pick the single candidate that best satisfies the instructions. If NONE of the candidates is acceptable '
          '(none satisfies the requirements), answer 0. Reply with JSON only: {"choice": <candidate number, or 0 for none>}.')
SCHEMA = {'type': 'object', 'properties': {'choice': {'type': 'integer'}}, 'required': ['choice']}
def prompt(r):
    lines = []
    for i, c in enumerate(r['candidates'], 1):
        txt = f"{c['name']}: {c['feat']}" if r['track'] == 'software_repair' else c['name']
        lines.append(f'{i}. {txt}')
    return (f"Task: {r['instructions']}\n\nContext:\n{r['context']}\n\nCandidates:\n" + '\n'.join(lines) +
            '\n0. none of the candidates is acceptable\n\nAnswer with JSON only: {"choice": <number>}')
think = {'false': False, 'true': True}.get(a.think, a.think)
lock = threading.Lock(); t0 = time.time()
fout = open(out_path, 'a')
def one(i):
    if time.time() - t0 > a.deadline: return
    r = ex[i]; k = len(r['candidates'])
    body = {'model': a.model, 'stream': False, 'think': think, 'format': SCHEMA,
            'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt(r)}],
            'options': {'temperature': 0, 'num_predict': a.num_predict, 'seed': 20261010}}
    if a.num_ctx: body['options']['num_ctx'] = a.num_ctx
    rec = {'id': i, 'model': a.model, 'variant': a.tag or 'default', 'think': a.think}
    err = None
    for attempt in range(3):
        ts = time.time()
        try:
            req = urllib.request.Request('http://localhost:11434/api/chat', json.dumps(body).encode(), {'Content-Type': 'application/json'})
            d = json.load(urllib.request.urlopen(req, timeout=a.timeout))
            err = None; break
        except Exception as e:
            err = f'{type(e).__name__}: {e}'
            try: err += ' ' + e.read().decode()[:300]
            except Exception: pass
            time.sleep(3 * (attempt + 1))
    lat = time.time() - ts
    gold = (r['gold'] + 1) if (r['answerable'] and r['gold'] is not None) else 0
    if err:
        rec.update(choice=None, correct=False, latency_s=round(lat, 3), valid_json=False, error=err, gold=gold, attempts=attempt + 1)
    else:
        content = (d.get('message') or {}).get('content', '') or ''
        choice, valid = None, False
        try:
            j = json.loads(content)
            c = j.get('choice') if isinstance(j, dict) else None
            if isinstance(c, int) and not isinstance(c, bool) and 0 <= c <= k: choice, valid = c, True
            elif isinstance(c, int): choice = c
        except Exception: pass
        rec.update(choice=choice, correct=bool(valid and choice == gold), latency_s=round(lat, 3),
                   prompt_tokens=d.get('prompt_eval_count'), completion_tokens=d.get('eval_count'), valid_json=valid,
                   gold=gold, k=k, done_reason=d.get('done_reason'), raw=content[:200],
                   thinking_chars=len((d.get('message') or {}).get('thinking') or ''), attempts=attempt + 1)
    with lock:
        fout.write(json.dumps(rec) + '\n'); fout.flush()
todo = [i for i in ids if i not in done]
with ThreadPoolExecutor(a.conc) as p: list(p.map(one, todo))
n = c = v = 0
for l in open(out_path):
    d = json.loads(l)
    if d.get('error'): continue
    n += 1; c += d['correct']; v += d['valid_json']
print(f'{label}: n={n} acc={c/max(n,1):.3f} valid={v/max(n,1):.3f} wall={time.time()-t0:.0f}s')
