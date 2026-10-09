"""Data-layer fix: a person cannot stand inside a solid object. Snap foot points out of benches/presses.
If the bench hides the feet, the person is on the far side of the bench from the camera (camera is on the aisle side) = press-side gap."""
import place
from place import OBJ, OB, CELLS, segments, hh, med

SOLID = {k: v for k, v in OBJ.items() if k.startswith('Bench') or k.startswith('Press')}

def snap(obs):
    out = []
    for t, x, y, vis in obs:
        for n, (a, b, c, d) in SOLID.items():
            if a <= x < c and b <= y < d:
                if n.startswith('Bench'):
                    y = b - 0.25 if (not vis or y - b < d - y) else d + 0.25   # hidden feet -> press side
                else:
                    y = d + 0.25
                out.append((t, x, y, vis, 'snapped')); break
        else:
            out.append((t, x, y, vis, ''))
    return out

def text_rel_snap():
    L = ['POSITION LOG cam07  10:00:00–10:10:00  (foot point on floor plan, metres; people NOT identified)',
         'FLOOR CELLS (exclusive, walkable only; benches and presses are solid, nobody can stand inside them):',
         '  Press (front face y=2.0) → Gap Press↔Bench (y 2.0–3.2, operator runs the press here) → Bench (solid, y 3.2–4.2)',
         '  → Front of Bench (aisle side, y 4.2–5.5) → Yellow aisle (y 5.5–7.0). Camera looks from the aisle side.',
         'TRACKS (one block = one continuous anonymous track)']
    for k, obs in OB.items():
        L.append(f'  track {k}:')
        so = snap(obs)
        for r in segments([(t, x, y, v) for t, x, y, v, _ in so]):
            ys = [p[1] for p in r['pts']]; vis = sum(p[2] for p in r['pts']) / len(r['pts']); y = med(ys)
            n = sum(1 for t, *_ , s in so if r['a'] <= t < r['b'] and s)
            if r['c'].startswith('Gap'): where = f'{y - 2.0:.1f} m from press face, {3.2 - y:.1f} m behind bench'
            elif r['c'].startswith('Front'): where = f'{y - 4.2:.1f} m in front of bench (aisle side)'
            else: where = ''
            if vis < 0.8:
                q = f'FEET HIDDEN by bench {100 * (1 - vis):.0f}% of time; placed in press-side gap because the bench hides the feet and the camera is on the aisle side (inferred, confidence 0.8; check frames to confirm)'
            else:
                q = 'feet visible, ±0.15 m'
            L.append(f'    {hh(r["a"])}–{hh(r["b"])}  {r["c"]:32}  {where}; {q}')
    return '\n'.join(L)

if __name__ == '__main__':
    print(text_rel_snap())
