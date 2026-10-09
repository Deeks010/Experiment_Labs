"""Build the five data representations from NOISY detector/tracker output only."""
import math, statistics
from sim import run, zone_of, hh, ZN, OPS, T, W0, W1

MAJOR_EDGE = 40


def tl_zones(tr):
    return [(t, zone_of(x, y)) for t, x, y, _ in tr]


def rle(seq, min_run):
    """seq: list of (t, value). Returns [(start, end_exclusive, value)] merging runs shorter than min_run into previous."""
    runs = []
    for t, v in seq:
        if runs and runs[-1][2] == v and t == runs[-1][1]:
            runs[-1][1] = t + 1
        else:
            runs.append([t, t + 1, v])
    out = []
    for r in runs:
        if out and (r[1] - r[0] < min_run) and out[-1][2] != 'unknown' and r[2] != 'unknown' and r[0] == out[-1][1]:
            out[-1][1] = r[1]
        elif out and out[-1][2] == r[2] and r[0] == out[-1][1]:
            out[-1][1] = r[1]
        else:
            out.append(r)
    # second pass to merge equal neighbours
    m = []
    for r in out:
        if m and m[-1][2] == r[2] and m[-1][1] == r[0]:
            m[-1][1] = r[1]
        else:
            m.append(list(r))
    return m


def smooth_counts(raw, offline):
    off = lambda t: any(a <= t < b for a, b in offline)
    out = []
    for t in range(T):
        if off(t):
            out.append((t, 'unknown')); continue
        win = [raw[u] for u in range(max(0, t - 2), min(T, t + 3)) if not off(u)]
        out.append((t, int(round(statistics.median(win)))))
    return out


def health(R):
    if R['offline']:
        a, b = R['offline'][0]
        return f'CAMERA HEALTH: online except {hh(a)}–{hh(b)} (camera OFFLINE, no data). No blockage, no blur.'
    return 'CAMERA HEALTH: online 100% of window, no blockage, no blur.'


def coast_positions(R):
    """per second: list of (x,y) for every track (>=3 s) seen within the last 2 s. Uses only local continuity, not identity."""
    pos = [[] for _ in range(T)]
    for k, v in R['tracks'].items():
        if len(v) < 3: continue
        i = 0
        for t in range(v[0][0], v[-1][0] + 1):
            while i + 1 < len(v) and v[i + 1][0] <= t: i += 1
            if t - v[i][0] <= 2: pos[t].append((v[i][1], v[i][2], None))
    return pos


def counts_block(R):
    dets, off = coast_positions(R), R['offline']
    frame = smooth_counts([len(d) for d in dets], off)
    lines = ['PEOPLE IN FRAME (people per second from short local tracks, gaps up to 2 s bridged, 5 s median, runs under 5 s merged; ID-free)']
    for a, b, v in rle(frame, 5):
        lines.append(f'  {hh(a)}–{hh(b)}  {v if v != "unknown" else "UNKNOWN (camera offline)"}')
    lines.append('')
    lines.append('ZONE STATES (people count per zone, same method; ID-free). Zones not listed stayed at 0.')
    for z in ZN:
        raw = [sum(1 for x, y, _ in d if zone_of(x, y) == z) for d in dets]
        sm = smooth_counts(raw, off)
        runs = rle(sm, 5 if z in OPS or z.startswith('Bench') else 3)
        if all(v == 0 for _, _, v in runs if v != 'unknown'):
            continue
        lines.append(f'  {z}:')
        for a, b, v in runs:
            lab = 'UNKNOWN (offline)' if v == 'unknown' else ('empty' if v == 0 else f'{v} person' + ('s' if v > 1 else ''))
            lines.append(f'    {hh(a)}–{hh(b)}  {lab}')
    return '\n'.join(lines)


def good_tracklets(R, min_len=3):
    return {k: v for k, v in R['tracks'].items() if len(v) >= min_len}


def edge_of(x, y):
    if y < 290: return None
    if x < MAJOR_EDGE + W0: return 'west'
    if x > W1 - MAJOR_EDGE: return 'east'
    return None


def zone_path(tr, min_dwell=2, max_gap=4):
    """zone runs for one track; same-zone runs split by short detection gaps are joined"""
    runs = []
    for t, z in tl_zones(tr):
        if runs and runs[-1][2] == z and t - runs[-1][1] <= max_gap:
            runs[-1][1] = t + 1
        else:
            runs.append([t, t + 1, z])
    out = []
    for r in runs:
        if out and r[1] - r[0] < min_dwell and r[0] - out[-1][1] <= max_gap:
            out[-1][1] = r[1]
        elif out and out[-1][2] == r[2] and r[0] - out[-1][1] <= max_gap:
            out[-1][1] = r[1]
        else:
            out.append(r)
    return out


def edges_block(R):
    ev = []
    for k, tr in good_tracklets(R).items():
        t0, x0, y0, _ = tr[0]; t1, x1, y1, _ = tr[-1]
        e0, e1 = edge_of(x0, y0), edge_of(x1, y1)
        if e0: ev.append((t0, f'1 person ENTERED view at {e0} edge of Yellow aisle'))
        if e1 and t1 < T - 3: ev.append((t1 + 1, f'1 person LEFT view at {e1} edge of Yellow aisle'))
    ev.sort()
    lines = ['EDGE CROSSINGS (judged from a few seconds of track next to the image edge; ID-free)']
    lines += [f'  {hh(t)}  {s}' for t, s in ev] or ['  none']
    return '\n'.join(lines)


def movement_block(R):
    """Anonymous movement episodes from single tracklets. Short Op->Bench->Op round trips are summarised."""
    eps, bench = [], {}
    for k, tr in sorted(good_tracklets(R).items(), key=lambda kv: kv[1][0][0]):
        runs = zone_path(tr)
        i = 0
        # detect short bench round trips
        segs = [(a, b, z) for a, b, z in runs]
        j = 0
        keep = []
        while j < len(segs):
            a, b, z = segs[j]
            if (z.startswith('Bench') and 0 < j < len(segs) - 1 and segs[j - 1][2] == segs[j + 1][2] and segs[j - 1][2] in OPS and b - a <= 25):
                bench.setdefault(segs[j - 1][2], []).append(a)
                # merge neighbours
                prev = keep.pop()
                nxt = segs[j + 1]
                keep.append((prev[0], nxt[1], prev[2]))
                j += 2
                continue
            keep.append((a, b, z)); j += 1
        if len(keep) <= 1:
            continue
        e0 = edge_of(tr[0][1], tr[0][2]); e1 = edge_of(tr[-1][1], tr[-1][2])
        parts = []
        if e0: parts.append(f'entered at {e0} edge')
        for a, b, z in keep:
            parts.append(f'{z} ({hh(a)}–{hh(b)})')
        if e1 and tr[-1][0] < T - 3: parts.append(f'left view at {e1} edge {hh(tr[-1][0] + 1)}')
        elif tr[-1][0] < T - 3: parts.append(f'track ends {hh(tr[-1][0] + 1)} inside view (lost: occlusion, crowd or ID break)')
        start = tr[0][0]
        if not e0 and start > 3: parts.insert(0, f'track starts {hh(start)} inside view')
        eps.append((keep[0][0], ' → '.join(parts)))
    lines = ['MOVEMENT EPISODES (one line = one continuous short track = "a person moved this way". Different lines can be the same person or different people. Do not join lines into identities.)']
    lines += [f'  {s}' for _, s in sorted(eps)] or ['  none']
    lines.append('')
    lines.append('SHORT BENCH VISITS (Op zone → its bench → same Op zone, under 25 s), start times:')
    for z in OPS:
        if z in bench:
            lines.append(f'  {z}: ' + ', '.join(hh(t) for t in sorted(bench[z])))
    return '\n'.join(lines)


def raw_block(R):
    out = ['RAW TRACKER OUTPUT (zone visits per tracker ID; IDs are assigned by the tracker)']
    for k, tr in sorted(R['tracks'].items(), key=lambda kv: kv[1][0][0]):
        runs = zone_path(tr, 2)
        out.append(f'Track ID {k} (first seen {hh(tr[0][0])}, last seen {hh(tr[-1][0] + 1)})')
        for a, b, z in runs:
            out.append(f'  {hh(a)}–{hh(b)}  {z}')
    return '\n'.join(out)


def stitch(R):
    """Simple offline stitcher: spatial-temporal gap join, plus edge re-entry join by 'appearance' (assumed)."""
    tl = sorted(good_tracklets(R).items(), key=lambda kv: kv[1][0][0])
    persons = []   # each: dict(ids, pts, conf, notes)
    for k, tr in tl:
        t0, x0, y0, _ = tr[0]
        best = None; nc = 0
        for p in persons:
            pt, px, py, _ = p['pts'][-1]
            gap = t0 - pt
            span_off = sum(max(0, min(b, t0) - max(a, pt)) for a, b in R['offline'])
            if 0 < gap - span_off <= 20 and math.hypot(x0 - px, y0 - py) <= 60 + 5 * gap:
                nc += 1
                d = math.hypot(x0 - px, y0 - py) + 3 * gap
                if best is None or d < best[0]: best = (d, p, 'gap', gap)
        e0 = edge_of(x0, y0)
        if best is None and e0:
            for p in persons:
                pt, px, py, _ = p['pts'][-1]
                if 20 < t0 - pt <= 600 and edge_of(px, py) == e0 and pt < T - 3:
                    nc += 1
                    if best is None or (t0 - pt) < best[3]:
                        best = (0, p, 'reentry', t0 - pt)
        if best:
            _, p, kind, gap = best
            if kind == 'reentry':
                p['notes'].append((t0, f'[re-entered at {e0} edge; matched to earlier exit by appearance, confidence 0.65]'))
                p['conf'] *= 0.65
            elif nc > 1:
                p['notes'].append((t0, '[junction: several candidates; identity may be swapped, confidence 0.55]'))
                p['conf'] *= 0.55
            else:
                p['conf'] *= 0.97
            p['ids'].append(k); p['pts'] += tr
        else:
            persons.append(dict(ids=[k], pts=list(tr), conf=0.95, notes=[]))
    return persons


def stitched_block(R):
    ps = stitch(R)
    out = [f'STITCHED PERSONS (tracker fragments joined offline; {len(ps)} persons; tracks under 3 s dropped as false detections)']
    for i, p in enumerate(ps):
        name = chr(65 + i) if i < 26 else f'P{i}'
        runs = zone_path(p['pts'], 2)
        out.append(f'Person {name} (from tracker IDs {", ".join(map(str, p["ids"]))}; overall confidence {p["conf"]:.2f})')
        t_first = p['pts'][0][0]
        e0 = edge_of(p['pts'][0][1], p['pts'][0][2])
        if t_first > 3:
            out.append(f'  before {hh(t_first)}  NOT OBSERVED' + (f' (entered at {e0} edge)' if e0 else ''))
        prev_end = None
        for a, b, z in runs:
            if prev_end is not None and a - prev_end > 3:
                out.append(f'  {hh(prev_end)}–{hh(a)}  NOT OBSERVED')
            note = ' '.join(n for t, n in p['notes'] if a <= t < b + 1)
            out.append(f'  {hh(a)}–{hh(b)}  {z}' + ('  ' + note if note else ''))
            prev_end = b
        tl = p['pts'][-1]
        if tl[0] < T - 3:
            e1 = edge_of(tl[1], tl[2])
            out.append(f'  after {hh(tl[0] + 1)}  NOT OBSERVED' + (f' (left at {e1} edge)' if e1 else ' (track lost inside view)'))
    return '\n'.join(out)


LIM_ID = ('SYSTEM LIMITS: Position is the foot point mapped to zones. Hands and objects are not visible. '
          'Tracker IDs are about 80% reliable: one person can get several IDs, and IDs can swap when people are close. '
          'Detections in crowds can be missed; short false detections occur. Nothing is known while the camera is offline or a person is NOT OBSERVED.')
LIM_FREE = ('SYSTEM LIMITS: Position is the foot point mapped to zones. Hands and objects are not visible. '
            'People are NOT identified. Counts per second are the most reliable data (about ±1 for a few seconds in crowds). '
            'Movement episodes are short, anonymous tracks: they show that "a person moved this way", not who. '
            'Nothing is known while the camera is offline.')


def build(scn):
    R = run(scn)
    h = health(R)
    F = {
        'R0': h + '\n' + LIM_ID + '\n\n' + raw_block(R),
        'R1': h + '\n' + LIM_ID + '\n\n' + stitched_block(R),
        'R2': h + '\n' + LIM_FREE + '\n\n' + counts_block(R),
        'R3': h + '\n' + LIM_FREE + '\n\n' + counts_block(R) + '\n\n' + edges_block(R) + '\n\n' + movement_block(R),
    }
    F['R4'] = F['R3'] + '\n\n' + 'IDENTITY HYPOTHESES (low trust; use only if a question needs identity, and say how uncertain it is)\n' + stitched_block(R)
    return R, F


if __name__ == '__main__':
    import sys
    R, F = build(sys.argv[1] if len(sys.argv) > 1 else 'S1')
    for k, v in F.items():
        print('=====', k, len(v), 'chars')
    print(F[sys.argv[2] if len(sys.argv) > 2 else 'R3'])


def image_R5(R, path):
    """ID-free zone state timeline image: count in each run printed as text; offline hatched; edge events listed."""
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    pos, off = coast_positions(R), R['offline']
    rows = [('People in frame', smooth_counts([len(d) for d in pos], off))]
    for z in ZN:
        raw = [sum(1 for x, y, _ in d if zone_of(x, y) == z) for d in pos]
        rows.append((z, smooth_counts(raw, off)))
    shade = {0: '#FFFFFF', 1: '#C9D8F0', 2: '#7FA3DB', 3: '#3E6FBF'}
    fig, ax = plt.subplots(figsize=(16, 7.5), dpi=100)
    for i, (name, ser) in enumerate(rows):
        y = len(rows) - i
        for a, b, v in rle(ser, 5 if name in OPS or name.startswith('Bench') or i == 0 else 3):
            if v == 'unknown':
                ax.add_patch(Rectangle((a, y - .4), b - a, .8, facecolor='#EEEEEE', hatch='///', edgecolor='#888', lw=0))
                continue
            col = '#DDDDDD' if i == 0 else shade[min(v, 3)]
            ax.add_patch(Rectangle((a, y - .4), b - a, .8, facecolor=col, edgecolor='#FFFFFF', lw=1.5))
            if v > 0 and b - a >= 6:
                ax.text((a + b) / 2, y, str(v), ha='center', va='center', fontsize=9, color='#111' if (i == 0 or min(v, 3) < 3) else '#fff')
        ax.text(-4, y, name, ha='right', va='center', fontsize=11, fontweight='bold' if i == 0 else 'normal')
    ev = []
    for k, tr in good_tracklets(R).items():
        e0, e1 = edge_of(tr[0][1], tr[0][2]), edge_of(tr[-1][1], tr[-1][2])
        if e0: ev.append((tr[0][0], f'IN {e0}'))
        if e1 and tr[-1][0] < T - 3: ev.append((tr[-1][0] + 1, f'OUT {e1}'))
    for j, (t, s) in enumerate(sorted(ev)):
        ax.plot([t, t], [0.2, 0.6], color='#B3261E', lw=2)
        ax.text(t, -0.05 - 0.42 * (j % 3), f'{s} {hh(t)[3:]}', ha='center', va='top', fontsize=8.5, color='#B3261E')
    ax.text(-125, -1.7, 'Edge crossings in order: ' + ';  '.join(f'{hh(t)} {s}' for t, s in sorted(ev)), fontsize=10, color='#222')
    ax.text(-4, 0.4, 'Edge crossings', ha='right', va='center', fontsize=11)
    for m in range(11):
        ax.axvline(m * 60, color='#E0E0E0', lw=.8, zorder=0)
        ax.text(m * 60, len(rows) + .8, f'10:{m:02d}', ha='center', fontsize=10)
    ax.set_xlim(-130, 605); ax.set_ylim(-2.0, len(rows) + 1.3); ax.axis('off')
    ax.set_title('cam07  10:00–10:10   ID-free zone states. Number in bar = people in that zone. Hatched = camera OFFLINE (unknown). '
                 'Red ticks = a person crossed the image edge.', fontsize=11, loc='left')
    fig.savefig(path, bbox_inches='tight'); plt.close(fig)
