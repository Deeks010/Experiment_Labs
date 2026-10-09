"""Round 6: is a heatmap a good overview for the LLM? 1 hour of dwell, aggregated (no time, no identity)."""
import random, math, json, os, sys
import os as _o, sys as _s; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), '..', 'round4'))
from place import OBJ, CELLS, cell
random.seed(3)
OBJ = dict(OBJ)
# truth stays: (x, y, minutes, label)
STAYS = [(7.5, 2.6, 35, 'gap P15 right part'), (6.0, 4.8, 10, 'front of Bench 15 middle'),
         (11.2, 2.6, 30, 'gap P16 left part'), (12.5, 4.9, 8, 'front of Bench 16 middle'),
         (14.9, 3.7, 5, 'beyond right end of Bench 16 (unusual)'), (9.0, 4.8, 3, 'passage next to Bench 15 right end (unusual)'),
         (16.6, 4.6, 4, 'in front of control panel lower part (unusual)')]
G = 0.2
heat = {}
def add(x, y, s):
    k = (int(x // G), int(y // G)); heat[k] = heat.get(k, 0) + s
for x, y, m, _ in STAYS:
    for _ in range(int(m * 60)):
        add(x + random.gauss(0, 0.22), y + random.gauss(0, 0.18), 1)
for _ in range(360):                       # 6 min of walking in the aisle, spread out
    add(random.uniform(0.5, 19.5), random.uniform(5.7, 6.8), 1)

def rect_dist(x, y, r):
    a, b, c, d = r; return math.hypot(max(a - x, 0, x - c), max(b - y, 0, y - d))

def relation(x, y):
    n = min(OBJ, key=lambda k: rect_dist(x, y, OBJ[k])); a, b, c, d = OBJ[n]
    bench = n if n.startswith('Bench') else ('Bench ' + n.split()[-1] if n.startswith('Press') else None)
    s = f'nearest {n} ({rect_dist(x, y, OBJ[n]):.1f} m)'
    if bench:
        a, b, c, d = OBJ[bench]
        if x < a: side = f'beyond LEFT end of {bench}'
        elif x > c: side = f'beyond RIGHT end of {bench}'
        elif y < b: side = f'press side of {bench}, {100 * (x - a) / (c - a):.0f}% along it'
        elif y > d: side = f'aisle side of {bench}, {100 * (x - a) / (c - a):.0f}% along it'
        else: side = f'over {bench}'
        s = side + '; ' + s
    return s

def hotspots(min_s=90):
    cells = {k for k, v in heat.items() if v >= 8}
    seen, comps = set(), []
    for k in cells:
        if k in seen: continue
        st, comp = [k], []
        seen.add(k)
        while st:
            c = st.pop(); comp.append(c)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    n = (c[0] + dx, c[1] + dy)
                    if n in cells and n not in seen: seen.add(n); st.append(n)
        tot = sum(heat[c] for c in comp)
        if tot >= min_s:
            x = sum((c[0] + .5) * G * heat[c] for c in comp) / tot; y = sum((c[1] + .5) * G * heat[c] for c in comp) / tot
            comps.append((tot, x, y))
    return sorted(comps, reverse=True)

def text_summary():
    per = {}
    for (i, j), v in heat.items():
        z = cell((i + .5) * G, (j + .5) * G); per[z] = per.get(z, 0) + v
    L = ['DWELL OVERVIEW cam07 09:00–10:00 (all people together; no identities; no time order)',
         'TIME PER FLOOR CELL (minutes):']
    L += [f'  {z:34} {v / 60:5.1f}' for z, v in sorted(per.items(), key=lambda kv: -kv[1])]
    L.append('HOTSPOTS (connected areas with ≥1.5 min; centre position described relative to objects):')
    for i, (t, x, y) in enumerate(hotspots()):
        L.append(f'  H{i + 1}  {t / 60:5.1f} min  {relation(x, y)}  xy=({x:.1f},{y:.1f})')
    return '\n'.join(L)

def image(path, labelled):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np
    fig, ax = plt.subplots(figsize=(14, 6.6), dpi=100)
    ax.add_patch(Rectangle((0, 5.5), 20, 1.5, color='#F6E39A', zorder=0)); ax.text(0.2, 6.9, 'YELLOW AISLE', fontsize=11, va='top')
    for n, (a, b, c, d) in OBJ.items():
        ax.add_patch(Rectangle((a, b), c - a, d - b, facecolor='#B8B8B4' if 'Press' in n else '#D9CBB3', edgecolor='#555', zorder=1))
        ax.text((a + c) / 2, (b + d) / 2, n, ha='center', va='center', fontsize=11, fontweight='bold', zorder=6)
    mx = max(heat.values())
    for (i, j), v in heat.items():
        f = (v / mx) ** 0.5
        ax.add_patch(Rectangle((i * G, j * G), G, G, color=(0.85, 0.25 * (1 - f) + 0.1, 0.1, 0.15 + 0.8 * f), lw=0, zorder=3))
    if labelled:
        for k, (t, x, y) in enumerate(hotspots()):
            ax.text(x + 0.35, y - 0.35, f'H{k + 1}: {t / 60:.0f} min', fontsize=10, fontweight='bold', color='#111',
                    bbox=dict(fc='white', ec='#333', pad=1.5), zorder=7)
    ax.set_xlim(-0.2, 20.2); ax.set_ylim(7.2, -0.3); ax.set_aspect('equal'); ax.set_xticks(range(0, 21, 2)); ax.set_yticks(range(0, 8))
    ax.grid(alpha=.2)
    ax.set_title('cam07 09:00–10:00 dwell heatmap, top view (metres). Darker red = more time spent there (all people, no time order).'
                 + (' Boxes = hotspot total minutes.' if labelled else ''), loc='left', fontsize=10.5)
    fig.savefig(path, bbox_inches='tight'); plt.close(fig)

BRIEF = '''You are an analysis assistant for a factory CCTV system. Camera cam07, top-view floor plan in metres.
Press 15 and Press 16 are machines at the back wall. Bench 15 and Bench 16 are work benches in front of them. Operators normally stand in the gap between press and bench (to run the press) or on the aisle side of the bench. The yellow aisle runs along the front. A control panel is on the right.
Answer only from the data.'''
Q = '''QUESTIONS (one hour, 09:00–10:00)
1. Name the 3 places where people spent the most time, in order, with approximate minutes. Describe each place exactly (which object, which side, which part).
2. At Bench 15: how many minutes on the press side and how many on the aisle side?
3. In the gap between Press 16 and Bench 16, which part was used most: left end, middle, or right end?
4. List every place OUTSIDE the normal work spots (normal = press-side gaps, aisle sides of benches, yellow aisle) where people spent 2 minutes or more. Give minutes and an exact description.
5. About how many minutes in total were spent in the yellow aisle?
Reply with JSON only: {"q1":[{"place":"...","minutes":n},...],"q2":{"press_side":n,"aisle_side":n},"q3":"left end"|"middle"|"right end"|"unknown","q4":[{"place":"...","minutes":n}],"q5":n}'''

if __name__ == '__main__':
    os.makedirs('out', exist_ok=True); os.makedirs('out_answers', exist_ok=True)
    image('out/heat_plain.png', False); image('out/heat_lab.png', True)
    T = text_summary()
    F = {'plain': 'DATA: one image (attached): dwell heatmap on the floor plan.',
         'lab': 'DATA: one image (attached): dwell heatmap on the floor plan, with hotspot totals.',
         'text': T, 'hyb': T + '\n\nAlso attached: one image, the dwell heatmap with hotspot labels H1.. matching the list above.'}
    for k, v in F.items():
        open(f'out/{k}.txt', 'w').write(BRIEF + '\n\nDATA:\n' + v + '\n\n' + Q)
    print(T)
