"""Simulator (from research round 3) + extra people for the new detectors.
World units: x 0..680, y 0..345 (aisle at y≈315). simulate.py converts to camera pixels.
Ground truth is kept separate from what the representations see."""
import random, math, json

W0, W1 = 14, 676                     # visible x range (outside = out of view)
ZONES = [('Op 14', 40, 108, 150, 168), ('Op 15', 150, 108, 270, 168), ('Op 16', 330, 108, 445, 168),
         ('Op 17', 445, 108, 560, 168), ('Bench L', 40, 168, 270, 230), ('Bench R', 330, 168, 560, 230),
         ('Passage', 270, 108, 330, 300), ('Control panel', 570, 40, 680, 220), ('Yellow aisle', 0, 300, 680, 330)]
ZN = [z[0] for z in ZONES] + ['Unmapped floor']
OPS = ['Op 14', 'Op 15', 'Op 16', 'Op 17']
STX = {'Op 14': 95, 'Op 15': 212, 'Op 16': 387, 'Op 17': 507}
T = 600


def zone_of(x, y):
    for z in ZONES:
        if z[1] <= x < z[3] and z[2] <= y < z[4]:
            return z[0]
    return 'Unmapped floor'


def hh(t):
    t = int(round(t)); return '10:%02d:%02d' % (t // 60, t % 60)


def interp(k, t):
    if t <= k[0][0]: return k[0][1:]
    for a, b in zip(k, k[1:]):
        if a[0] <= t <= b[0]:
            f = (t - a[0]) / ((b[0] - a[0]) or 1)
            return (a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]))
    return k[-1][1:]


def operator(x, t0, t1, period, phase, away=()):
    """keyframes for an operator at station x, with 12 s bench visits; away = list of (ta, tb, path_out, path_back)"""
    k = []
    t = t0
    k.append((t0, x, 135))
    nxt = phase
    while nxt < t1:
        if not any(a - 15 <= nxt <= b + 15 for a, b, *_ in away):
            k += [(nxt, x, 135), (nxt + 3, x, 190), (nxt + 12, x, 190), (nxt + 15, x, 135)]
        nxt += period
    k.append((t1, x, 135))
    k.sort()
    for a, b, out, back in away:   # remove keyframes inside away window, splice paths
        k = [q for q in k if not (a <= q[0] <= b)]
        k += [(a, x, 135)] + out + back + [(b, x, 135)]
        k.sort()
    return k


def scenario(name):
    P = {}
    if name == 'S1':
        # B leaves Op 15 by Passage to west edge; a DIFFERENT worker F later takes over Op 15.
        P['A'] = operator(95, 0, 599, 90, 54)
        P['B'] = [(0, 212, 135), (54, 212, 135), (57, 212, 190), (66, 212, 190), (69, 212, 135), (190, 212, 138), (205, 300, 170),
                  (218, 300, 280), (226, 300, 315), (236, 10, 315), (240, -40, 315), (599, -40, 315)]
        P['C'] = operator(387, 0, 599, 90, 54)
        P['D'] = operator(507, 0, 599, 90, 54)
        P['F'] = [(0, -40, 315), (400, -40, 315), (404, 10, 315), (430, 300, 315), (440, 300, 280), (455, 300, 170), (470, 212, 138), (599, 212, 138)]
        # visitor group of 3 from east edge, to control panel, then two go to Op 16/17, all leave east
        P['G'] = [(0, 720, 315), (128, 720, 315), (132, 672, 315), (146, 590, 315), (160, 600, 190), (300, 600, 190), (330, 470, 150),
                  (400, 470, 150), (420, 590, 315), (436, 690, 315), (599, 720, 315)]
        P['H'] = [(0, 720, 315), (130, 720, 315), (134, 672, 315), (148, 590, 315), (162, 625, 180), (300, 625, 180), (332, 430, 150),
                  (405, 430, 150), (425, 590, 315), (442, 690, 315), (599, 720, 315)]
        P['I'] = [(0, 720, 315), (133, 720, 315), (137, 672, 315), (151, 590, 315), (165, 612, 205), (310, 612, 205), (318, 600, 315),
                  (330, 690, 315), (599, 720, 315)]
        # runner: fast movement west -> east along the aisle
        P['Q'] = [(0, -40, 318), (520, -40, 318), (521, 10, 318), (525, 670, 318), (526, 720, 318), (599, 720, 318)]
        offline = [(345, 385)]
        fp = [(48, 3, 322, 150), (251, 4, 322, 150), (300, 2, 322, 150), (512, 5, 322, 150), (230, 2, 652, 260), (480, 3, 652, 260)]
        truth = dict(distinct=9, entries={'west': 2, 'east': 3}, exits={'west': 1, 'east': 4})
    else:
        # S2: A leaves east and returns (same person); material handler M twice; supervisor K walks through; C+D chat
        P['A'] = operator(95, 0, 599, 90, 40, away=[(100, 265,
                  [(103, 95, 190), (118, 290, 200), (130, 300, 280), (138, 300, 315), (160, 670, 315), (163, 720, 315)],
                  [(225, 720, 315), (228, 670, 315), (250, 300, 315), (256, 300, 280), (262, 200, 190)])])
        P['B'] = operator(212, 0, 599, 90, 70)
        P['C'] = operator(387, 0, 599, 90, 25, away=[(480, 562, [(486, 470, 150)], [(556, 470, 150)])])
        P['D'] = operator(507, 0, 599, 90, 25, away=[(478, 564, [(482, 520, 150)], [(560, 520, 150)])])
        P['M'] = [(0, -40, 315), (56, -40, 315), (60, 10, 315), (78, 300, 315), (85, 300, 250), (92, 400, 200), (150, 400, 200),
                  (158, 300, 250), (165, 300, 315), (196, 670, 315), (200, 720, 315),
                  (376, 720, 315), (380, 670, 315), (400, 300, 315), (406, 300, 250), (412, 160, 200), (440, 160, 200),
                  (447, 300, 250), (452, 300, 315), (476, 10, 315), (480, -40, 315), (599, -40, 315)]
        P['K'] = [(0, -40, 315), (298, -40, 315), (302, 10, 315), (336, 670, 315), (340, 720, 315), (599, 720, 315)]
        # long stationary person on unmapped floor (rest / sleep candidate)
        P['R'] = [(0, 720, 315), (110, 720, 315), (114, 672, 315), (124, 600, 300), (134, 585, 262), (400, 585, 262), (410, 600, 300), (420, 672, 315), (424, 720, 315), (599, 720, 315)]
        offline = []
        fp = [(30, 3, 322, 150), (95, 2, 322, 150), (170, 4, 322, 150), (233, 2, 322, 150), (290, 5, 322, 150), (355, 2, 322, 150),
              (420, 3, 322, 150), (530, 4, 322, 150), (120, 3, 652, 260), (470, 2, 652, 260)]
        truth = dict(distinct=7, entries={'west': 2, 'east': 3}, exits={'west': 1, 'east': 4})
    return P, offline, fp, truth


def run(name, seed=7, miss=0.03, crowd_miss=0.22, flicker=0.004, step=0.25):
    """Simulate detector + tracker at 1/step Hz. Returns ground truth per second (gt) and
    tracker output per frame: frames = [(time_sec, [(tid, x, y, src, hidden_feet)])]."""
    rnd = random.Random(seed)
    P, offline, fp, truth = scenario(name)
    off = lambda t: any(a <= t < b for a, b in offline)
    gt = []
    for t in range(T):
        o = {}
        for pid, k in P.items():
            x, y = interp(k, t)
            if W0 <= x <= W1:
                o[pid] = (x, y, zone_of(x, y))
        gt.append(o)
    frames = []
    active, nid, hist = {}, [1], {}
    for n in range(int(T / step)):
        t = round(n * step, 3)
        if off(t):
            continue
        pts = []
        for pid, k in P.items():
            x, y = interp(k, t)
            if W0 <= x <= W1:
                pts.append((pid, x, y))
        dets = []
        for pid, x, y in pts:
            crowd = any(q != pid and math.hypot(x - u, y - v) < 32 for q, u, v in pts)
            if rnd.random() < (crowd_miss if crowd else miss):
                continue
            hidden = 150 <= y < 168 and rnd.random() < 0.6        # feet behind the bench edge
            dets.append((x + rnd.gauss(0, 3), y + rnd.gauss(0, 3) + (12 if hidden else 0), pid, hidden))
        for a0, du, x, y in fp:
            if a0 <= t < a0 + du:
                dets.append((x + rnd.gauss(0, 3), y + rnd.gauss(0, 3), 'FP', False))
        for tid in [k for k, v in active.items() if t - v[2] > 2]:
            del active[tid]                       # lost tracks expire before association
        cand = []
        for tid, (lx, ly, lt, vx, vy, npts) in active.items():
            px_, py_ = lx + vx * (t - lt), ly + vy * (t - lt)
            gate = (60 if npts < 2 else 25) + 40 * (t - lt)
            for j, (x, y, _, _) in enumerate(dets):
                dd = math.hypot(x - px_, y - py_)
                if dd < gate:
                    cand.append((dd, tid, j))
        cand.sort(); used_t, used_d = set(), set()
        out = []
        for dd, tid, j in cand:
            if tid in used_t or j in used_d:
                continue
            used_t.add(tid); used_d.add(j)
            x, y, src, hid = dets[j]
            old = active[tid]
            if rnd.random() < flicker * step:
                active.pop(tid); tid = nid[0]; nid[0] += 1; old = None
            vx, vy = ((x - old[0]) / max(t - old[2], step), (y - old[1]) / max(t - old[2], step)) if old else (0.0, 0.0)
            active[tid] = (x, y, t, vx, vy, (old[5] + 1) if old else 1)
            out.append((tid, x, y, src, hid))
        for j, (x, y, src, hid) in enumerate(dets):
            if j in used_d:
                continue
            tid = nid[0]; nid[0] += 1
            active[tid] = (x, y, t, 0.0, 0.0, 1)
            out.append((tid, x, y, src, hid))
        for tid in [k for k, v in active.items() if t - v[2] > 2]:
            del active[tid]
        frames.append((t, out))
    return dict(P=P, offline=offline, gt=gt, frames=frames, truth=truth, step=step)


if __name__ == '__main__':
    for s_ in ['S1', 'S2']:
        r = run(s_)
        ids = {d[0] for _, f in r['frames'] for d in f}
        print(s_, 'tracker IDs:', len(ids), 'true people:', r['truth']['distinct'])
