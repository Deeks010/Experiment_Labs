"""Round 4: exact placement between machine, workstation (bench) and aisle. Metric top-down map (metres)."""
import random, math, json
random.seed(11)

T = 600
def hh(t): t = int(round(t)); return '10:%02d:%02d' % (t // 60, t % 60)

# Static objects (x0, y0, x1, y1). y grows from back wall (0) toward the aisle.
OBJ = {'Press 15': (5, 0, 8, 2), 'Bench 15': (4.5, 3.2, 8.5, 4.2),
       'Press 16': (11, 0, 14, 2), 'Bench 16': (10.5, 3.2, 14.5, 4.2),
       'Control panel': (17, 1, 19, 4)}
AISLE = (0, 5.5, 20, 7.0)
# Exclusive floor cells, named by their neighbours (every floor point is in exactly one cell)
CELLS = [('Gap Press 15 ↔ Bench 15', 4.5, 2.0, 8.5, 3.2), ('On/over Bench 15', 4.5, 3.2, 8.5, 4.2),
         ('Front of Bench 15 (aisle side)', 4.5, 4.2, 8.5, 5.5),
         ('Gap Press 16 ↔ Bench 16', 10.5, 2.0, 14.5, 3.2), ('On/over Bench 16', 10.5, 3.2, 14.5, 4.2),
         ('Front of Bench 16 (aisle side)', 10.5, 4.2, 14.5, 5.5),
         ('Passage between benches', 8.5, 2.0, 10.5, 5.5), ('Yellow aisle', 0, 5.5, 20, 7.0),
         ('Front of Control panel', 15.5, 1, 17, 5.5)]
COARSE = [('Workstation 15', 4.5, 2.0, 8.5, 5.5), ('Workstation 16', 10.5, 2.0, 14.5, 5.5),
          ('Passage', 8.5, 2.0, 10.5, 5.5), ('Yellow aisle', 0, 5.5, 20, 7.0), ('Control panel', 15.5, 1, 17, 5.5)]

def cell(x, y, cells=CELLS):
    for n, a, b, c, d in cells:
        if a <= x < c and b <= y < d: return n
    return 'Other floor'

# ---- ground truth (truth label per second for each person) ----
W1 = [(0, 120, 6.5, 2.6, 'gap'), (120, 210, 6.5, 4.8, 'front'), (210, 300, 6.6, 3.0, 'gap_hidden'),
      (300, 360, 6.5, 6.2, 'aisle'), (360, 600, 6.5, 2.6, 'gap')]
W2 = [(0, 240, 12.5, 2.6, 'gap'), (240, 420, 12.4, 4.9, 'front'), (420, 600, 12.5, 2.6, 'gap')]

def series(plan):
    out = []
    for t in range(T):
        for i, (a, b, x, y, lab) in enumerate(plan):
            if a <= t < b:
                # 6 s walk between segments
                if i and t - a < 6:
                    px, py = plan[i - 1][2], plan[i - 1][3]; f = (t - a) / 6
                    x, y = px + f * (x - px), py + f * (y - py)
                out.append((t, x, y, lab)); break
    return out

def observe(s):
    """foot point estimate. When feet are behind the bench (gap_hidden), the estimate is biased toward the bench."""
    o = []
    for t, x, y, lab in s:
        hidden = lab == 'gap_hidden'
        ex = x + random.gauss(0, 0.12); ey = y + (0.55 if hidden else 0) + random.gauss(0, 0.3 if hidden else 0.12)
        o.append((t, ex, ey, not hidden))
    return o

TR = {'a': series(W1), 'b': series(W2)}
OB = {k: observe(v) for k, v in TR.items()}
# passer-by in aisle 10:03:00-10:03:40 west to east
OB['c'] = [(t, 0.3 + (t - 180) * 0.48, 6.3 + random.gauss(0, 0.1), True) for t in range(180, 220)]

def segments(obs, cells=CELLS, min_run=5):
    runs = []
    for t, x, y, vis in obs:
        c = cell(x, y, cells)
        if runs and runs[-1]['c'] == c and t - runs[-1]['b'] <= 2:
            r = runs[-1]; r['b'] = t + 1; r['pts'].append((x, y, vis))
        else:
            runs.append(dict(c=c, a=t, b=t + 1, pts=[(x, y, vis)]))
    m = []
    for r in runs:
        if m and r['b'] - r['a'] < min_run:
            m[-1]['b'] = r['b']; m[-1]['pts'] += r['pts']
        elif m and m[-1]['c'] == r['c']:
            m[-1]['b'] = r['b']; m[-1]['pts'] += r['pts']
        else:
            m.append(r)
    return m

def med(v): v = sorted(v); return v[len(v) // 2]

def rel_desc(r):
    xs = [p[0] for p in r['pts']]; ys = [p[1] for p in r['pts']]; vis = sum(p[2] for p in r['pts']) / len(r['pts'])
    x, y = med(xs), med(ys); parts = []
    for name in ['Press 15', 'Press 16', 'Bench 15', 'Bench 16']:
        a, b, c, d = OBJ[name]
        if a - 1 <= x <= c + 1:
            if y < b: parts.append(f'{b - y:.1f} m in front of... ')
            if name.startswith('Press'): parts.append(f'{y - d:.1f} m from {name} front face')
            else:
                if y < b: parts.append(f'{b - y:.1f} m behind {name} (press side)')
                elif y > d: parts.append(f'{y - d:.1f} m in front of {name} (aisle side)')
                else: parts.append(f'over {name} footprint')
    parts = [p for p in parts if not p.startswith('0.') or 'front of...' not in p]
    parts = [p for p in parts if '...' not in p]
    unc = 0.15 if vis > 0.8 else 0.6
    feet = 'feet visible' if vis > 0.8 else f'FEET HIDDEN by bench ({100 * (1 - vis):.0f}% of time): position ±{unc} m, true spot may be the press-side gap'
    return f"{', '.join(parts)}; spread ±{max(0.1, (max(ys) - min(ys)) / 4):.1f} m; {feet}"

def text_rel():
    L = ['POSITION LOG cam07  10:00:00–10:10:00  (foot point on floor plan, metres; people NOT identified)',
         'FLOOR CELLS (exclusive; every floor point is in exactly one cell; named by the objects on each side):',
         '  Back wall → Press 15/16 (machines, front face at y=2.0) → Gap Press↔Bench (y 2.0–3.2, operator stands here to run the press)',
         '  → Bench 15/16 (workstation top, y 3.2–4.2) → Front of Bench (aisle side, y 4.2–5.5) → Yellow aisle (y 5.5–7.0)',
         'TRACKS (one block = one continuous anonymous track)']
    for k, obs in OB.items():
        L.append(f'  track {k}:')
        for r in segments(obs):
            L.append(f'    {hh(r["a"])}–{hh(r["b"])}  {r["c"]:32}  {rel_desc(r)}')
    return '\n'.join(L)

def text_coarse():
    L = ['ZONE LOG cam07  10:00:00–10:10:00  (people NOT identified)',
         'ZONES: Workstation 15, Workstation 16 (area around each workstation), Passage, Yellow aisle, Control panel.',
         'TRACKS (one block = one continuous anonymous track)']
    for k, obs in OB.items():
        L.append(f'  track {k}:')
        for r in segments(obs, COARSE):
            L.append(f'    {hh(r["a"])}–{hh(r["b"])}  near {r["c"]}')
    return '\n'.join(L)

def image(path):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Ellipse
    fig, ax = plt.subplots(figsize=(15, 8.2), dpi=100)
    ax.add_patch(Rectangle((0, 5.5), 20, 1.5, color='#F6E39A', zorder=0)); ax.text(0.2, 6.85, 'YELLOW AISLE', fontsize=11, va='top')
    for n, a, b, c, d in CELLS:
        if n.startswith('Gap'): ax.add_patch(Rectangle((a, b), c - a, d - b, facecolor='#E3F1E6', edgecolor='#2E7D4F', ls='--', lw=1))
        if n.startswith('Front of Bench'): ax.add_patch(Rectangle((a, b), c - a, d - b, facecolor='#EEF0FA', edgecolor='#5560B0', ls='--', lw=1))
    for n, (a, b, c, d) in OBJ.items():
        ax.add_patch(Rectangle((a, b), c - a, d - b, facecolor='#B8B8B4' if 'Press' in n else '#D9CBB3', edgecolor='#555'))
        ax.text((a + c) / 2, (b + d) / 2, n.upper() if 'Press' in n else n, ha='center', va='center', fontsize=11, fontweight='bold')
    for x0 in (4.5, 10.5):
        ax.text(x0 + 4.0, 2.1, 'GAP press↔bench', fontsize=9, color='#2E7D4F', va='top', ha='right')
        ax.text(x0 + 4.0, 5.42, 'FRONT of bench (aisle side)', fontsize=9, color='#3C46A0', va='bottom', ha='right')
    cols = {'a': '#C2410C', 'b': '#7C3AED', 'c': '#0F766E'}
    for k, obs in OB.items():
        segs = segments(obs); groups = {}
        for i, r in enumerate(segs): groups.setdefault(r['c'], []).append((i, r))
        for c, items in groups.items():
            pts = [p for _, r in items for p in r['pts']]
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; vis = sum(p[2] for p in pts) / len(pts)
            x, y = med(xs), med(ys)
            ax.scatter(xs[::3], ys[::3], s=10, color=cols[k], alpha=.35, zorder=3)
            if vis < 0.8:
                ax.add_patch(Ellipse((x, y - 0.3), 1.4, 1.6, fill=False, ls=':', lw=2, color=cols[k], zorder=4))
                ax.text(x + 0.8, y + 0.15, 'feet hidden by bench:\ntrue spot may be in gap', fontsize=9, color=cols[k])
            ax.scatter([x], [y], s=60, color='white' if vis < .8 else cols[k], edgecolor=cols[k], lw=2, zorder=5)
            if k != 'c':
                lab = ' + '.join(f'{k}{i + 1} {hh(r["a"])[3:]}–{hh(r["b"])[3:]}' for i, r in items)
                ax.text(x - 0.2, y, lab, ha='right', va='center', fontsize=9.5,
                        color=cols[k], bbox=dict(fc='white', ec='none', alpha=.85, pad=1))
    ax.annotate('', xy=(19.5, 6.25), xytext=(0.5, 6.25), arrowprops=dict(arrowstyle='->', color=cols['c']))
    ax.text(10, 6.55, 'track c 03:00–03:40 walked west → east', color=cols['c'], fontsize=9.5, ha='center')
    ax.set_xlim(-0.2, 20.2); ax.set_ylim(7.2, -0.3); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('cam07 floor plan, top view (metres). Dot = median foot point of one stay; label = track+stay, mm:ss after 10:00. '
                 'Hollow dot + dotted ring = feet hidden.', fontsize=10.5, loc='left')
    fig.savefig(path, bbox_inches='tight'); plt.close(fig)

if __name__ == '__main__':
    print(text_coarse()); print(); print(text_rel()); image('map.png')
