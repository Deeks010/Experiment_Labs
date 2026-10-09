"""Data-layer stress test: which facts survive tracker noise? 20 seeds x 3 noise levels x 2 scenarios."""
import json, statistics
from sim import run, T, OPS, zone_of
from reps import coast_positions, smooth_counts, good_tracklets, edge_of, stitch

LEVELS = {'normal': dict(miss=0.03, crowd_miss=0.22, flicker=0.004),
          'bad': dict(miss=0.08, crowd_miss=0.40, flicker=0.012),
          'very bad': dict(miss=0.15, crowd_miss=0.55, flicker=0.025)}

def evaluate(s, seed, p):
    R = run(s, seed=seed, **p); gt = R['gt']; off = R['offline']
    o = lambda t: any(a <= t < b for a, b in off)
    truth = [len(x) for x in gt]
    cnt = smooth_counts([len(d) for d in coast_positions(R)], off)
    ok = [v == truth[t] for t, v in cnt if v != 'unknown']
    within1 = [abs(v - truth[t]) <= 1 for t, v in cnt if v != 'unknown']
    # zone occupancy (occupied vs empty) per operator zone
    pos = coast_positions(R); zok = []
    for z in OPS:
        sm = smooth_counts([sum(1 for x, y, _ in d if zone_of(x, y) == z) for d in pos], off)
        for t, v in sm:
            if v == 'unknown': continue
            zok.append((v > 0) == any(q[2] == z for q in gt[t].values()))
    # edge crossings
    ent = {'west': 0, 'east': 0}; ex = {'west': 0, 'east': 0}
    for k, tr in good_tracklets(R).items():
        e0, e1 = edge_of(tr[0][1], tr[0][2]), edge_of(tr[-1][1], tr[-1][2])
        if e0: ent[e0] += 1
        if e1 and tr[-1][0] < T - 3: ex[e1] += 1
    edges_ok = ent == R['truth']['entries'] and ex == R['truth']['exits']
    # identity: stitched persons vs truth, and purity of stitched persons
    ps = stitch(R)
    pure = sum(1 for p in ps if len({q[3] for q in p['pts'] if q[3] != 'FP'}) == 1) / len(ps)
    return dict(count_exact=sum(ok) / len(ok), count_pm1=sum(within1) / len(within1), zone_state=sum(zok) / len(zok),
                edges_exact=edges_ok, n_ids=len(R['tracks']), stitched=len(ps), true_people=R['truth']['distinct'],
                stitched_right=len(ps) == R['truth']['distinct'], stitch_purity=pure)

out = {}
for lv, p in LEVELS.items():
    rs = [evaluate(s, seed, p) for s in ['S1', 'S2'] for seed in range(1, 21)]
    m = lambda k: statistics.mean(float(r[k]) for r in rs)
    out[lv] = {k: round(m(k), 3) for k in rs[0]}
    print(f"{lv:9} count exact {m('count_exact'):.1%}  count ±1 {m('count_pm1'):.1%}  op-zone state {m('zone_state'):.1%}  "
          f"edge counts exact {m('edges_exact'):.0%}  | tracker IDs {m('n_ids'):.0f} for ~{m('true_people'):.0f} people, "
          f"stitched persons {m('stitched'):.1f}, stitched count right {m('stitched_right'):.0%}, stitched persons pure {m('stitch_purity'):.0%}")
json.dump(out, open('stress.json', 'w'), indent=1)
