"""Questions, ground truth (from simulator truth, not from detections) and auto-grader."""
import json, re
from sim import run, hh, OPS, T

QS = {
 'S1': dict(t1=180, zone3='Op 15', w3=(240, 420), t7=360, z7='Op 15',
            q6='Is the person working at Op 15 at 10:09 the same person who worked at Op 15 at 10:01?',
            q8='Around 10:03 a person left the operator area. In which zones did they move, in order, by which edge did they leave the camera view, and at what time?'),
 'S2': dict(t1=90, zone3='Op 16', w3=(490, 540), t7=180, z7='Op 14',
            q6='Is the person working at Op 14 at 10:06 the same person who worked at Op 14 at 10:01?',
            q8='Around 10:02 a person left the operator area. In which zones did they move, in order, by which edge did they leave the camera view, and at what time?'),
}


def questions(s):
    q = QS[s]
    return [
        f'How many people were in the camera view at {hh(q["t1"])}?',
        'What was the largest number of people in view at the same time, and in which period?',
        f'Was {q["zone3"]} occupied between {hh(q["w3"][0])} and {hh(q["w3"][1])}?',
        'Which operator zones (Op 14–17) were empty for more than 1 minute, and when?',
        'How many times did a person enter the camera view, and how many times did a person leave it, at each edge (west/east)?',
        q['q6'],
        f'Was {q["z7"]} occupied at {hh(q["t7"])}?',
        q['q8'],
        'How many different people appeared in this camera between 10:00 and 10:10?',
    ]


SHAPE = '''Reply with JSON only, in exactly this shape (use null when the data cannot answer; times as "10:mm:ss"):
{"q1":{"count":int|null,"note":"..."},
 "q2":{"max":int|null,"from":"10:mm:ss"|null,"to":"10:mm:ss"|null,"note":"..."},
 "q3":{"answer":"yes"|"no"|"partly"|"unknown","note":"..."},
 "q4":{"zones":[{"zone":"Op 1x","from":"10:mm:ss","to":"10:mm:ss"}],"note":"..."},
 "q5":{"entered_west":int|null,"entered_east":int|null,"left_west":int|null,"left_east":int|null,"note":"..."},
 "q6":{"answer":"same"|"different"|"unknown","note":"..."},
 "q7":{"answer":"occupied"|"empty"|"unknown","note":"..."},
 "q8":{"zones":["..."],"edge":"west"|"east"|null,"time_out_of_view":"10:mm:ss"|null,"note":"..."},
 "q9":{"best":int|null,"min":int|null,"max":int|null,"note":"..."}}
Each "note" is 1–2 short sentences explaining the answer and its certainty.'''

BRIEF = '''You are an analysis assistant for a factory CCTV system.
Camera: cam07 "Moulding press 14–17". Fixed overhead camera. Window: 10:00:00 to 10:10:00.
Static zones mapped on this camera:
- Op 14, Op 15, Op 16, Op 17: standing area in front of each moulding press where its operator works.
- Bench L (in front of Op 14 and Op 15) and Bench R (in front of Op 16 and Op 17): work benches.
- Passage: gap between Bench L and Bench R that leads to the yellow aisle.
- Control panel: electrical control cabinet on the right side.
- Yellow aisle: walkway along the front; it leads out of the camera view at the west edge and the east edge.
- Unmapped floor: any floor area not in a named zone.
The data below comes from automated person detection and tracking. It can contain errors.
Answer only from the data. If the data cannot answer a question, say so (use null or "unknown").'''


def sec(s):
    if not s or not isinstance(s, str): return None
    m = re.search(r'(\d+):(\d\d):(\d\d)', s)
    return None if not m else (int(m.group(2)) * 60 + int(m.group(3)))


def ground_truth(s):
    R = run(s); gt = R['gt']; q = QS[s]
    n = [len(o) for o in gt]
    occ = lambda z, t: any(v[2] == z for v in gt[t].values())
    empties = []
    for z in OPS:
        a = None
        for t in range(T + 1):
            e = t < T and not occ(z, t)
            if e and a is None: a = t
            if not e and a is not None:
                if t - a > 60: empties.append((z, a, t))
                a = None
    w0, w1 = q['w3']
    occ3 = any(occ(q['zone3'], t) for t in range(w0, w1))
    off = any(a <= q['t7'] < b for a, b in R['offline'])
    return dict(q1=n[q['t1']], q2=max(n), q3='yes' if occ3 else 'no', q4=empties, q5=R['truth'],
                q6='different' if s == 'S1' else 'same',
                q7='unknown' if off else ('occupied' if occ(q['z7'], q['t7']) else 'empty'),
                q7_real='occupied' if occ(q['z7'], q['t7']) else 'empty',
                q8=dict(S1=('west', 236, ['Op 15', 'Passage', 'Yellow aisle']), S2=('east', 161, ['Op 14', 'Bench L', 'Passage', 'Yellow aisle']))[s],
                q9=R['truth']['distinct'])


def grade(s, a):
    G = ground_truth(s); sc = {}
    g = lambda k: a.get(k) or {}
    c = g('q1').get('count'); sc['q1'] = 2 if c == G['q1'] else (1 if isinstance(c, int) and abs(c - G['q1']) == 1 else 0)
    m = g('q2').get('max'); sc['q2'] = 2 if m == G['q2'] else (1 if isinstance(m, int) and abs(m - G['q2']) == 1 else 0)
    ans = (g('q3').get('answer') or '').lower(); sc['q3'] = {G['q3']: 2, 'partly': 1, 'unknown': 0.5}.get(ans, 0)
    got = g('q4').get('zones') or []; pts = 0
    for z, a0, a1 in G['q4']:
        hit = [x for x in got if x.get('zone') == z]
        fr = [sec(x.get('from')) for x in hit if sec(x.get('from')) is not None]
        to = [sec(x.get('to')) for x in hit if sec(x.get('to')) is not None]
        if fr and to and abs(min(fr) - a0) <= 60 and abs(max(to) - a1) <= 60:
            pts += 1
        elif hit: pts += 0.5
    extra = len({x.get('zone') for x in got} - {z for z, *_ in G['q4']})
    sc['q4'] = max(0, round(2 * pts / len(G['q4']) - extra, 2))
    t5 = G['q5']; v = g('q5')
    right = sum([v.get('entered_west') == t5['entries']['west'], v.get('entered_east') == t5['entries']['east'],
                 v.get('left_west') == t5['exits']['west'], v.get('left_east') == t5['exits']['east']])
    sc['q5'] = 2 if right == 4 else (1 if right == 3 else 0)
    a6 = (g('q6').get('answer') or '').lower(); sc['q6'] = 2 if a6 == G['q6'] else (1 if a6 == 'unknown' else 0)
    a7 = (g('q7').get('answer') or '').lower()
    if G['q7'] == 'unknown': sc['q7'] = 2 if a7 == 'unknown' else (0.5 if a7 == G['q7_real'] else 0)
    else: sc['q7'] = 2 if a7 == G['q7'] else (0.5 if a7 == 'unknown' else 0)
    e, tt, zs = G['q8']; v = g('q8'); p = 0
    if (v.get('edge') or '').lower() == e: p += 1
    if sec(v.get('time_out_of_view')) is not None and abs(sec(v.get('time_out_of_view')) - tt) <= 30: p += 0.5
    if all(z in (v.get('zones') or []) for z in zs if z != 'Bench L'): p += 0.5
    sc['q8'] = p
    v = g('q9'); b = v.get('best'); lo, hi = v.get('min'), v.get('max')
    sc['q9'] = 2 if b == G['q9'] else (1 if isinstance(b, int) and abs(b - G['q9']) == 1 and (lo or 0) <= G['q9'] <= (hi or 99) else 0)
    sc['total'] = round(sum(sc[k] for k in sc), 2)
    return sc


if __name__ == '__main__':
    for s in QS:
        print(s, json.dumps(ground_truth(s)))
