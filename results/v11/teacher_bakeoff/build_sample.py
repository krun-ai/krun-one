"""Fixed evaluation sample for the teacher bake-off (seed 20261010). Reads ONLY val.jsonl."""
import json, random, collections
SEED = 20261010
VAL = str(__import__('pathlib').Path(__file__).resolve().parents[3] / 'data/v11/techbench/val.jsonl')
PER_TRACK = 24
rows = [json.loads(l) for l in open(VAL)]

def eligible(r):
    if r.get('qtype') not in (None, 'choice'):
        return False
    if r['track'] == 'software_repair':
        return r['render'] == 'feat'
    return r['render'] == 'raw'

def family(r):
    s = r['slice']
    if s == 'tetris': return 'tetris'
    if s.startswith('swe_'): return 'swe'
    if (r.get('evidence') or {}).get('type') == 'simulation': return 'simulator'
    return 'rule_table'

el = sorted([r for r in rows if eligible(r)], key=lambda r: r['id'])
rng = random.Random(SEED)
tracks = sorted({r['track'] for r in el})
by_track = {t: [r for r in el if r['track'] == t] for t in tracks}

# quotas
flip_tracks = [t for t in tracks if any(r.get('flip') for r in by_track[t])]
unans_tracks = [t for t in tracks if any(not r['answerable'] for r in by_track[t])]
flip_q = {t: 4 for t in flip_tracks}
for t in ['api', 'distributed']: flip_q[t] += 1            # -> 30 pairs
unans_q = {t: 4 for t in unans_tracks}
for t in ['api', 'mlinfra', 'software_repair', 'security']: unans_q[t] += 1   # -> 40

sample, subset = [], []
for t in tracks:
    pool = by_track[t]
    chosen = []
    # flip pairs (both members)
    pairs = collections.defaultdict(list)
    for r in pool:
        if r.get('flip'): pairs[r['flip']['pair']].append(r)
    pids = sorted(p for p, v in pairs.items() if len(v) == 2)
    rng.shuffle(pids)
    # spread pairs across slices
    pids_by_slice = collections.defaultdict(list)
    for p in pids: pids_by_slice[pairs[p][0]['slice']].append(p)
    picked_pairs = []
    sl = sorted(pids_by_slice)
    i = 0
    while len(picked_pairs) < flip_q.get(t, 0) and any(pids_by_slice.values()):
        s = sl[i % len(sl)]; i += 1
        if pids_by_slice[s]: picked_pairs.append(pids_by_slice[s].pop())
    for p in picked_pairs: chosen += sorted(pairs[p], key=lambda r: r['flip']['role'])
    # unanswerable, spread across slices
    un = [r for r in pool if not r['answerable']]
    rng.shuffle(un)
    un_by_slice = collections.defaultdict(list)
    for r in un: un_by_slice[r['slice']].append(r)
    sl = sorted(un_by_slice); i = 0; picked_un = []
    while len(picked_un) < unans_q.get(t, 0) and any(un_by_slice.values()):
        s = sl[i % len(sl)]; i += 1
        if un_by_slice[s]: picked_un.append(un_by_slice[s].pop())
    chosen += picked_un
    # fill: answerable non-flip, balancing slice counts
    rest = [r for r in pool if r['answerable'] and not r.get('flip')]
    rng.shuffle(rest)
    rest_by_slice = collections.defaultdict(list)
    for r in rest: rest_by_slice[r['slice']].append(r)
    cnt = collections.Counter(r['slice'] for r in chosen)
    fill = []
    while len(chosen) + len(fill) < PER_TRACK:
        cands = sorted([s for s in rest_by_slice if rest_by_slice[s]], key=lambda s: (cnt[s], s))
        s = cands[0]; fill.append(rest_by_slice[s].pop()); cnt[s] += 1
    chosen += fill
    assert len(chosen) == PER_TRACK and len({r['id'] for r in chosen}) == PER_TRACK
    sample += chosen
    # 60-example subset: 6 per track = 1 flip pair (if any) + 1 unanswerable (if any) + fill
    sub = []
    if picked_pairs: sub += [r for r in chosen if r.get('flip') and r['flip']['pair'] == picked_pairs[0]]
    if picked_un: sub.append(picked_un[0])
    # fill subset balancing slices
    c2 = collections.Counter(r['slice'] for r in sub)
    for r in fill:
        if len(sub) >= 6: break
        if c2[r['slice']] <= min(c2[x['slice']] for x in fill if x not in sub):
            sub.append(r); c2[r['slice']] += 1
    for r in fill:
        if len(sub) >= 6: break
        if r not in sub: sub.append(r)
    assert len(sub) == 6
    subset += sub

meta = {r['id']: {'track': r['track'], 'slice': r['slice'], 'family': family(r), 'answerable': r['answerable'],
                  'k': len(r['candidates']), 'flip': r.get('flip'), 'render': r['render']} for r in sample}
out = {'seed': SEED, 'source': VAL, 'n': len(sample), 'ids': [r['id'] for r in sample],
       'subset60_ids': [r['id'] for r in subset], 'meta': meta}
json.dump(out, open('sample_ids.json', 'w'), indent=1)
print('n', len(sample), 'subset', len(subset))
print('unans', sum(not r['answerable'] for r in sample), 'flip pairs', len({r['flip']['pair'] for r in sample if r.get('flip')}))
print('family', collections.Counter(family(r) for r in sample))
print('slices', sorted(collections.Counter((r['track'], r['slice']) for r in sample).items()))
print('subset: unans', sum(not r['answerable'] for r in subset), 'flip pairs', len({r['flip']['pair'] for r in subset if r.get('flip')}), collections.Counter(family(r) for r in subset))
print('K', sorted(collections.Counter(len(r['candidates']) for r in sample).items()))
print('gold pos (1-based, 0=none)', sorted(collections.Counter((r['gold']+1 if r['gold'] is not None else 0) for r in sample).items()))
print('family by slice', sorted({(r['slice'], family(r)) for r in el}))
