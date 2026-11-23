"""Aggregate raw_*.jsonl -> summary.json (+ tables printed). Accuracy is always vs dataset gold/answerable."""
import json, glob, random, statistics as st, collections, itertools, re
S = json.load(open('sample_ids.json')); META = S['meta']; FULL = S['ids']; SUB = set(S['subset60_ids'])
def load(f):
    recs = {}
    nerr = 0
    for l in open(f):
        d = json.loads(l)
        if d.get('error'):
            nerr += 1
            if d['id'] not in recs: recs[d['id']] = d
        else: recs[d['id']] = d
    return recs, nerr
def pct(xs, p):
    xs = sorted(xs); 
    if not xs: return None
    i = min(len(xs) - 1, max(0, int(round(p / 100 * (len(xs) - 1))))); return xs[i]
def acc(rs): return (sum(r['correct'] for r in rs) / len(rs)) if rs else None
def boot(rs, B=5000, seed=20261010):
    rng = random.Random(seed); v = [int(r['correct']) for r in rs]; n = len(v)
    if not n: return None
    m = sorted(sum(rng.choices(v, k=n)) / n for _ in range(B))
    return [round(m[int(0.025 * B)], 4), round(m[int(0.975 * B) - 1], 4)]
def r4(x): return None if x is None else round(x, 4)
def summarize(recs, ids):
    rs = [recs[i] for i in ids if i in recs]
    for r in rs: r.update({k: META[r['id']][k] for k in ('track', 'slice', 'family', 'answerable')})
    out = {'n': len(rs), 'n_expected': len(ids), 'n_request_errors': sum(1 for r in rs if r.get('error')),
           'acc': r4(acc(rs)), 'acc_ci95': boot(rs),
           'acc_answerable': r4(acc([r for r in rs if r['answerable']])), 'n_answerable': sum(r['answerable'] for r in rs),
           'acc_unanswerable': r4(acc([r for r in rs if not r['answerable']])), 'n_unanswerable': sum(not r['answerable'] for r in rs),
           'said_none_rate': r4(sum(r['choice'] == 0 for r in rs) / len(rs)) if rs else None,
           'false_none_rate_on_answerable': r4(acc([{'correct': r['choice'] == 0} for r in rs if r['answerable']])),
           'valid_json_rate': r4(sum(r['valid_json'] for r in rs) / len(rs)) if rs else None}
    out['per_track'] = {t: {'acc': r4(acc(v)), 'n': len(v)} for t, v in sorted(_grp(rs, 'track').items())}
    out['per_family'] = {t: {'acc': r4(acc(v)), 'n': len(v)} for t, v in sorted(_grp(rs, 'family').items())}
    out['per_slice'] = {t: {'acc': r4(acc(v)), 'n': len(v)} for t, v in sorted(_grp(rs, 'slice').items())}
    pairs = collections.defaultdict(list)
    for r in rs:
        fl = META[r['id']]['flip']
        if fl: pairs[fl['pair']].append(r)
    full = [v for v in pairs.values() if len(v) == 2]
    out['flip_pairs'] = len(full)
    out['flip_both_correct'] = r4(sum(all(x['correct'] for x in v) for v in full) / len(full)) if full else None
    out['flip_same_choice_both'] = r4(sum(v[0]['choice'] == v[1]['choice'] for v in full) / len(full)) if full else None
    lat = [r['latency_s'] for r in rs if not r.get('error')]
    out['latency_median_s'] = r4(st.median(lat)) if lat else None; out['latency_p95_s'] = r4(pct(lat, 95)); out['latency_mean_s'] = r4(st.mean(lat)) if lat else None
    pt = [r.get('prompt_tokens') or 0 for r in rs if not r.get('error')]; ct = [r.get('completion_tokens') or 0 for r in rs if not r.get('error')]
    out['prompt_tokens_mean'] = r4(st.mean(pt)) if pt else None; out['completion_tokens_mean'] = r4(st.mean(ct)) if ct else None
    out['completion_tokens_p95'] = pct(ct, 95); out['prompt_tokens_max'] = max(pt) if pt else None
    out['truncated_rate'] = r4(sum(r.get('done_reason') == 'length' for r in rs) / len(rs)) if rs else None
    if lat:
        out['per_10k'] = {'sequential_hours': r4(st.mean(lat) * 10000 / 3600), 'prompt_tokens_M': r4(st.mean(pt) * 10000 / 1e6), 'completion_tokens_M': r4(st.mean(ct) * 10000 / 1e6)}
    return out
def add_cost(label, out):
    p = price(label)
    if out.get('per_10k') is None: return
    if p: out['per_10k']['usd'] = r4((out['prompt_tokens_mean'] * p[0] + out['completion_tokens_mean'] * p[1]) * 10000 / 1e6); out['per_10k']['usd_per_Mtok_in_out'] = list(p)
    elif 'cloud' not in label: out['per_10k']['usd'] = 0.0
    cl = latprobe.get(label)
    if cl: out['latency_clean_sequential'] = cl
    if 'cloud' in label:
        m = cl['mean_s'] if cl else out['latency_mean_s']
        out['per_10k']['hours_at_pro_concurrency3'] = r4(m * 10000 / 3 / 3600)
        out['per_10k']['latency_basis'] = 'clean sequential probe (8 req)' if cl else 'run latency (run had the 3 cloud slots to itself)'
        if cl: out['per_10k']['sequential_hours'] = r4(cl['mean_s'] * 10000 / 3600)
def _grp(rs, k):
    g = collections.defaultdict(list)
    for r in rs: g[r[k]].append(r)
    return g
PRICE = {  # USD per 1M tokens (input, output), ollama.com/pricing fetched 2026-10-01; local models = 0
 'kimi-k3': (3.00, 15.00), 'glm-5.3': (1.40, 4.40), 'deepseek-v4-pro': (1.32, 3.96), 'deepseek-v4.1-flash': (0.30, 1.20), 'gpt-oss_120b-cloud': (0.15, 0.60)}
def price(label):
    for k, v in PRICE.items():
        if label.startswith(k): return v
    return None
runs = {}; latprobe = {}
for f in sorted(glob.glob('raw_*.jsonl')):
    label = f[4:-6]; recs, nerr = load(f)
    if label.endswith('__latprobe'):
        lat = [r['latency_s'] for r in recs.values() if not r.get('error')]
        latprobe[label[:-10]] = {'n': len(lat), 'median_s': r4(st.median(lat)), 'mean_s': r4(st.mean(lat)), 'max_s': max(lat)}
        continue
    if 'retry' in label: continue
    runs[label] = recs
# glm-5.3 thinking: merged variant where items that were invalid at the 4000-token cap are replaced by the 16000-token retry
try:
    retry, _ = load('raw_glm-5.3_cloud__think16k_retry.jsonl')
    base = runs['glm-5.3_cloud__think']; merged = {}
    for i, r in base.items():
        if i in retry and not retry[i].get('error'):
            m = dict(retry[i]); m['latency_s'] = r['latency_s'] + m['latency_s']
            m['prompt_tokens'] = (r.get('prompt_tokens') or 0) + (m.get('prompt_tokens') or 0)
            m['completion_tokens'] = (r.get('completion_tokens') or 0) + (m.get('completion_tokens') or 0)
            merged[i] = m
        else: merged[i] = dict(r)
    runs['glm-5.3_cloud__think_plus16k_retry'] = merged
except FileNotFoundError: pass
summary = {'sample': {'n': len(FULL), 'subset60': len(SUB), 'seed': S['seed']}, 'models_full240': {}, 'models_subset60': {}}
for label, recs in runs.items():
    if len(recs) >= 200: summary['models_full240'][label] = summarize(recs, FULL); add_cost(label, summary['models_full240'][label])
    sub = summarize(recs, [i for i in FULL if i in SUB])
    if sub['n'] >= 40: summary['models_subset60'][label] = sub; add_cost(label, sub)
    elif len(recs) < 200: summary.setdefault('incomplete', {})[label] = {'n': len(recs)}
# pairwise agreement (all pairs of full-240 runs)
def agree(a, b, ids):
    ids = [i for i in ids if i in runs[a] and i in runs[b]]
    ra, rb = runs[a], runs[b]
    ag = [i for i in ids if ra[i]['choice'] is not None and ra[i]['choice'] == rb[i]['choice']]
    dis = [i for i in ids if i not in set(ag)]
    d = {'n': len(ids), 'agreement_rate': r4(len(ag) / len(ids)), 'n_agree': len(ag),
         'acc_when_agree': r4(sum(ra[i]['correct'] for i in ag) / len(ag)) if ag else None,
         'acc_when_agree_ci95': boot([ra[i] for i in ag]),
         'n_disagree': len(dis), 'either_correct_when_disagree': r4(sum(ra[i]['correct'] or rb[i]['correct'] for i in dis) / len(dis)) if dis else None,
         'acc_a_when_disagree': r4(sum(ra[i]['correct'] for i in dis) / len(dis)) if dis else None,
         'acc_b_when_disagree': r4(sum(rb[i]['correct'] for i in dis) / len(dis)) if dis else None}
    bt = collections.defaultdict(lambda: [0, 0, 0])
    for i in ids:
        t = META[i]['track']; bt[t][0] += 1
        if i in set(ag): bt[t][1] += 1; bt[t][2] += ra[i]['correct']
    d['per_track'] = {t: {'n': v[0], 'n_agree': v[1], 'acc_when_agree': r4(v[2] / v[1]) if v[1] else None} for t, v in sorted(bt.items())}
    ans = [i for i in ag if META[i]['answerable']]; un = [i for i in ag if not META[i]['answerable']]
    d['agree_acc_answerable'] = r4(sum(ra[i]['correct'] for i in ans) / len(ans)) if ans else None
    d['agree_acc_unanswerable'] = r4(sum(ra[i]['correct'] for i in un) / len(un)) if un else None; d['n_agree_unanswerable'] = len(un)
    return d
fullm = sorted(summary['models_full240'], key=lambda m: -summary['models_full240'][m]['acc'])
summary['ranking_full240'] = fullm
agm = [m for m in fullm if not m.endswith('plus16k_retry')]
summary['agreement'] = {f'{a}|{b}': agree(a, b, FULL) for a, b in itertools.combinations(agm, 2)}
summary['top2_full240'] = agm[:2]
if len(agm) >= 3:
    a, b, c = agm[:3]; ids = [i for i in FULL if all(i in runs[m] for m in (a, b, c))]
    ag = [i for i in ids if runs[a][i]['choice'] is not None and runs[a][i]['choice'] == runs[b][i]['choice'] == runs[c][i]['choice']]
    summary['agreement_top3_unanimous'] = {'models': [a, b, c], 'n': len(ids), 'n_agree': len(ag), 'agreement_rate': r4(len(ag) / len(ids)),
                                           'acc_when_agree': r4(sum(runs[a][i]['correct'] for i in ag) / len(ag)) if ag else None,
                                           'acc_when_agree_ci95': boot([runs[a][i] for i in ag])}
def paired(a, b):
    ids = [i for i in FULL if i in SUB and i in runs.get(a, {}) and i in runs.get(b, {})]
    if not ids: return None
    A, B = runs[a], runs[b]
    return {'a': a, 'b': b, 'n': len(ids), 'acc_a': r4(sum(A[i]['correct'] for i in ids) / len(ids)), 'acc_b': r4(sum(B[i]['correct'] for i in ids) / len(ids)),
            'a_wrong_b_right': sum((not A[i]['correct']) and B[i]['correct'] for i in ids), 'a_right_b_wrong': sum(A[i]['correct'] and not B[i]['correct'] for i in ids),
            'b_truncated': sum(B[i].get('done_reason') == 'length' for i in ids),
            'acc_b_on_valid': r4(sum(B[i]['correct'] for i in ids if B[i]['valid_json']) / max(1, sum(B[i]['valid_json'] for i in ids))), 'n_b_valid': sum(B[i]['valid_json'] for i in ids)}
summary['thinking_effect_subset60'] = [p for p in (paired('kimi-k3_cloud', 'kimi-k3_cloud__think'), paired('gpt-oss_120b-cloud', 'gpt-oss_120b-cloud__think_high')) if p]
summary['latency_probe_clean'] = latprobe
json.dump(summary, open('summary.json', 'w'), indent=1)
def f(x): return '-' if x is None else f'{100*x:.1f}'
L = []
for key, title in (('models_full240', 'Full sample (240 examples)'), ('models_subset60', 'Fixed 60-example subset')):
    M = summary[key]; order = sorted(M, key=lambda m: -M[m]['acc']); tracks = sorted({t for m in M.values() for t in m['per_track']})
    L += [f'### {title}: headline', '', '| model / variant | n | acc % | 95% CI | answerable | unanswerable (says none) | says none overall | false none on answerable | flip both-correct | valid JSON | truncated |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for m in order:
        s_ = M[m]; L.append(f"| {m} | {s_['n']} | {f(s_['acc'])} | {f(s_['acc_ci95'][0])}-{f(s_['acc_ci95'][1])} | {f(s_['acc_answerable'])} | {f(s_['acc_unanswerable'])} | {f(s_['said_none_rate'])} | {f(s_['false_none_rate_on_answerable'])} | {f(s_['flip_both_correct'])} ({s_['flip_pairs']} pairs) | {f(s_['valid_json_rate'])} | {f(s_['truncated_rate'])} |")
    L += ['', f'### {title}: accuracy % per track', '', '| model / variant | ' + ' | '.join(tracks) + ' |', '|---|' + '---|' * len(tracks)]
    for m in order: L.append(f'| {m} | ' + ' | '.join(f(M[m]['per_track'].get(t, {}).get('acc')) for t in tracks) + ' |')
    L.append('| n per track | ' + ' | '.join(str(M[order[0]]['per_track'].get(t, {}).get('n')) for t in tracks) + ' |')
    fams = ['rule_table', 'simulator', 'swe', 'tetris']
    L += ['', f'### {title}: accuracy % per slice family', '', '| model / variant | ' + ' | '.join(fams) + ' |', '|---|' + '---|' * len(fams)]
    for m in order: L.append(f'| {m} | ' + ' | '.join(f(M[m]['per_family'].get(t, {}).get('acc')) for t in fams) + ' |')
    L.append('| n | ' + ' | '.join(str(M[order[0]]['per_family'].get(t, {}).get('n')) for t in fams) + ' |')
    L += ['', f'### {title}: latency, tokens, cost', '', '| model / variant | median s | p95 s | clean sequential median s | prompt tok (mean) | completion tok (mean, incl. thinking) | USD / 10k examples | hours / 10k (sequential) | hours / 10k (3 concurrent cloud slots) |', '|---|---|---|---|---|---|---|---|---|']
    for m in order:
        s_ = M[m]; p = s_.get('per_10k') or {}; c = s_.get('latency_clean_sequential') or {}
        L.append(f"| {m} | {s_['latency_median_s']} | {s_['latency_p95_s']} | {c.get('median_s', '-')} | {round(s_['prompt_tokens_mean'])} | {round(s_['completion_tokens_mean'])} | {p.get('usd', '-')} | {p.get('sequential_hours', '-')} | {p.get('hours_at_pro_concurrency3', '-')} |")
    L.append('')
M = summary['models_full240']; order = sorted(M, key=lambda m: -M[m]['acc']); slices = sorted({t for m in M.values() for t in m['per_slice']})
L += ['### Full sample: accuracy % per slice', '', '| slice (n) | ' + ' | '.join(order) + ' |', '|---|' + '---|' * len(order)]
for t in slices: L.append(f"| {t} ({M[order[0]]['per_slice'].get(t, {}).get('n')}) | " + ' | '.join(f(M[m]['per_slice'].get(t, {}).get('acc')) for m in order) + ' |')
L += ['', '### Pairwise agreement (full sample)', '', '| pair | agreement % | n agree | acc when agree % | 95% CI | n disagree | at least one right when they disagree % |', '|---|---|---|---|---|---|---|']
for k, d in summary['agreement'].items():
    L.append(f"| {k.replace('|', ' + ')} | {f(d['agreement_rate'])} | {d['n_agree']} | {f(d['acc_when_agree'])} | {f(d['acc_when_agree_ci95'][0])}-{f(d['acc_when_agree_ci95'][1])} | {d['n_disagree']} | {f(d['either_correct_when_disagree'])} |")
k = '|'.join(summary['top2_full240']); d = summary['agreement'][k]
L += ['', f"### Top-2 agreement per track ({k.replace('|', ' + ')})", '', '| track | n | n agree | acc when agree % |', '|---|---|---|---|']
for t, v in d['per_track'].items(): L.append(f"| {t} | {v['n']} | {v['n_agree']} | {f(v['acc_when_agree'])} |")
open('report_tables.md', 'w').write('\n'.join(L) + '\n')
print('ranking', [(m, M[m]['acc']) for m in order]); print('top2', summary['top2_full240'], {x: d[x] for x in d if x != 'per_track'})
print('top3', summary.get('agreement_top3_unanimous')); print('thinking', summary['thinking_effect_subset60']); print('incomplete', summary.get('incomplete'))
