"""Round 5: edge positions. Does an LLM locate a person better from cell names, precomputed relations, raw coordinates, or an image?"""
import json, math, sys
import os as _o, sys as _s; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), '..', 'round4'))
from place import OBJ, CELLS, cell

# 12 positions (x, y) in metres, chosen to hit ends, corners and boundaries around Press 15 / Bench 15
P = [(6.5, 2.6), (4.3, 2.6), (8.7, 3.0), (8.6, 3.7), (6.5, 5.45), (5.0, 2.15), (7.9, 3.1), (4.6, 4.4),
     (8.4, 4.9), (9.6, 2.4), (4.0, 3.7), (7.2, 5.7)]

def rect_dist(x, y, r):
    a, b, c, d = r
    dx = max(a - x, 0, x - c); dy = max(b - y, 0, y - d); return math.hypot(dx, dy)

def nearest(x, y):
    return min(OBJ, key=lambda n: rect_dist(x, y, OBJ[n]))

def side15(x, y):
    a, b, c, d = OBJ['Bench 15']
    if x < a: return 'beyond_left_end'
    if x > c: return 'beyond_right_end'
    if y < b: return 'press_side'
    if y > d: return 'aisle_side'
    return 'on_bench'

def reach_press15(x, y):
    a, b, c, d = OBJ['Press 15']
    return 'yes' if (a - 0.3 <= x <= c + 0.3 and 0 <= y - d <= 0.8) else 'no'

def part15(x, y):
    a, b, c, d = OBJ['Bench 15']
    if x < a or x > c: return 'outside_length'
    f = (x - a) / (c - a)
    return 'left_third' if f < 1 / 3 else ('middle_third' if f < 2 / 3 else 'right_third')

TRUTH = [dict(nearest=nearest(x, y), side=side15(x, y), reach=reach_press15(x, y), part=part15(x, y)) for x, y in P]

def hh(i): return f'10:{i:02d}:00'

def fmt_cell():
    L = ['STAYS (anonymous; one line = one stay of about 40 s)']
    for i, (x, y) in enumerate(P):
        L.append(f'  s{i + 1:<2} {hh(i)}  {cell(x, y)}')
    return '\n'.join(L)

def fmt_rel():
    L = ['STAYS (anonymous; one line = one stay of about 40 s). Relations are computed by the system from the foot point.']
    for i, (x, y) in enumerate(P):
        ds = sorted(((rect_dist(x, y, r), n) for n, r in OBJ.items()))[:2]
        a, b, c, d = OBJ['Bench 15']
        along = (x - a) / (c - a) * 100
        flags = []
        for n, ca, cb, cc, cd in CELLS:
            m = min(abs(x - ca), abs(x - cc), abs(y - cb), abs(y - cd))
            if ca - .2 <= x <= cc + .2 and cb - .2 <= y <= cd + .2 and m < 0.2 and cell(x, y) != n:
                flags.append(f'within {m:.1f} m of cell "{n}"')
        L.append(f'  s{i + 1:<2} {hh(i)}  cell: {cell(x, y)} | nearest: {ds[0][1]} {ds[0][0]:.1f} m, {ds[1][1]} {ds[1][0]:.1f} m'
                 f' | vs Bench 15: {side15(x, y).replace("_", " ")}, {along:.0f}% along its length (0% = left end, 100% = right end)'
                 f' | {"within" if reach_press15(x, y) == "yes" else "NOT within"} 0.8 m reach of Press 15 front face'
                 + (f' | boundary: {"; ".join(flags)}' if flags else ''))
    return '\n'.join(L)

def fmt_raw():
    L = ['OBJECTS (rectangles, metres, x0,y0 – x1,y1; y grows from back wall toward the aisle):']
    for n, r in OBJ.items(): L.append(f'  {n}: {r[0]},{r[1]} – {r[2]},{r[3]}')
    L.append('  Yellow aisle: 0,5.5 – 20,7.0')
    L.append('STAYS (anonymous; median foot point, metres)')
    for i, (x, y) in enumerate(P): L.append(f'  s{i + 1:<2} {hh(i)}  x={x:.2f} y={y:.2f}')
    return '\n'.join(L)

def image(path):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(13, 7), dpi=100)
    ax.add_patch(Rectangle((0, 5.5), 20, 1.5, color='#F6E39A', zorder=0)); ax.text(0.2, 6.85, 'YELLOW AISLE', fontsize=11, va='top')
    for n, (a, b, c, d) in OBJ.items():
        ax.add_patch(Rectangle((a, b), c - a, d - b, facecolor='#B8B8B4' if 'Press' in n else '#D9CBB3', edgecolor='#555'))
        ax.text((a + c) / 2, (b + d) / 2, n, ha='center', va='center', fontsize=11, fontweight='bold')
    for i, (x, y) in enumerate(P):
        ax.scatter([x], [y], s=55, color='#C2410C', zorder=5)
        ax.text(x + 0.12, y - 0.12, f's{i + 1}', fontsize=10, color='#7A2A08', fontweight='bold', zorder=6)
    ax.set_xlim(2.5, 15.5); ax.set_ylim(7.2, -0.3); ax.set_aspect('equal')
    ax.set_xticks(range(3, 16)); ax.set_yticks(range(0, 8)); ax.grid(alpha=.25)
    ax.set_title('cam07 floor plan, top view, metres. Red dot = median foot point of stay (anonymous).', loc='left', fontsize=11)
    fig.savefig(path, bbox_inches='tight'); plt.close(fig)

BRIEF = '''You are an analysis assistant for a factory CCTV system. Camera cam07, top-view floor plan in metres.
Press 15 and Press 16 are machines at the back wall. Bench 15 and Bench 16 are work benches in front of them. The operator runs a press from the gap between press and bench. The yellow aisle runs along the front.
Answer only from the data.'''
Q = '''QUESTIONS. For EACH stay s1–s12 answer:
 nearest: which object was the person closest to? one of "Press 15","Bench 15","Press 16","Bench 16","Control panel"
 side: where relative to Bench 15? one of "press_side","aisle_side","on_bench","beyond_left_end","beyond_right_end"
 reach: was the person within 0.8 m of the front face of Press 15 (and in front of it, not beside it)? "yes"/"no"
 part: which part of Bench 15's length were they level with? "left_third","middle_third","right_third","outside_length"
Reply with JSON only: {"s1":{"nearest":"...","side":"...","reach":"...","part":"..."}, ... ,"s12":{...}}'''

if __name__ == '__main__':
    import os, shutil
    os.makedirs('out12', exist_ok=True); os.makedirs('out12_answers', exist_ok=True)
    image('out12/plan.png')
    F = {'cell': fmt_cell(), 'rel': fmt_rel(), 'raw': fmt_raw(),
         'img': 'DATA: one image (attached): floor plan with the stays drawn as numbered dots. A metre grid is drawn.'}
    for k, v in F.items():
        open(f'out12/{k}.txt', 'w').write(BRIEF + '\n\nDATA:\n' + v + '\n\n' + Q)
    json.dump(TRUTH, open('truth.json', 'w'), indent=1)
    print(fmt_rel()); print(json.dumps(TRUTH))
