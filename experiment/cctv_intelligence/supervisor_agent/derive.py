"""Build the derived (sa_*) tables for one tracking run.

Input: frame_observations of the existing tracker (raw, never modified).
Output: ID-free facts the tools serve: counts, zone states, edge events, movement episodes,
short visits, dwell grid, detector events and camera health.

Run:  python -m supervisor_agent.derive --camera CAMERA_1 [--run RUN_ID]   (from cctv_intelligence/)
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone

from . import db as D
from .geometry import edge_of, place_label, point_in_poly, poly_dist, zones_at

VERSION = 1
VISIBLE_FEET = {"ankle_midpoint", "single_ankle"}
WORK_ROLES = {"operator_station", "work_surface", "walkway", "exit"}


# ---------------------------------------------------------------- helpers
def rle(series: list, start_t: int, min_run: int) -> list[list]:
    """series[i] is the value at second start_t+i (None = unknown). Runs shorter than min_run are merged
    into the previous run (never across unknown)."""
    runs = []
    for i, v in enumerate(series):
        t = start_t + i
        if runs and runs[-1][2] == v:
            runs[-1][1] = t + 1
        else:
            runs.append([t, t + 1, v])
    out = []
    for r in runs:
        if out and r[1] - r[0] < min_run and r[2] is not None and out[-1][2] is not None:
            out[-1][1] = r[1]
        elif out and out[-1][2] == r[2]:
            out[-1][1] = r[1]
        else:
            out.append(r)
    merged = []
    for r in out:
        if merged and merged[-1][2] == r[2]:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return merged


def median_filter(vals: list, half: int) -> list:
    out = []
    for i, v in enumerate(vals):
        if v is None:
            out.append(None); continue
        win = [u for u in vals[max(0, i - half): i + half + 1] if u is not None]
        out.append(int(round(statistics.median(win))))
    return out


def _intervals(flags: list[bool], t0: int):
    a = None
    for i, f in enumerate(flags + [False]):
        if f and a is None:
            a = i
        if not f and a is not None:
            yield t0 + a, t0 + i
            a = None


# ---------------------------------------------------------------- main build
def build(conn, run_id: str, sem: dict | None = None) -> dict:
    sem = sem or D.load_semantics()
    S, TH = sem["sampling"], sem["thresholds"]
    D.init_derived(conn)
    run = next(r for c in D.list_camera_names(conn) for r in D.camera_runs(conn, c) if r["run_id"] == run_id)
    camera = run["camera"]
    zones = D.load_zones(conn, camera)
    cfg = run["config"]
    W, H = int(cfg.get("width") or 1920), int(cfg.get("height") or 1080)

    rows = conn.execute("SELECT time_sec, detections_json FROM frame_observations WHERE run_id=? ORDER BY time_sec",
                        (run_id,)).fetchall()
    if not rows:
        raise ValueError(f"Run {run_id} has no frame_observations.")
    t_start = int(math.floor(min(run["start_sec"], rows[0]["time_sec"])))
    t_end = int(math.ceil(max(run["end_sec"], rows[-1]["time_sec"])))
    N = t_end - t_start

    # --- camera health: gaps in processed frames = offline
    times = [r["time_sec"] for r in rows]
    online = [False] * N
    for t in times:
        i = int(t) - t_start
        if 0 <= i < N:
            online[i] = True
    # fill short holes (frame stride < 1 s means every second should have frames)
    gap = S["offline_gap_sec"]
    i = 0
    while i < N:
        if not online[i]:
            j = i
            while j < N and not online[j]:
                j += 1
            if j - i < gap and i > 0 and j < N:
                for k in range(i, j):
                    online[k] = True
            i = j
        else:
            i += 1
    health = [(a, b, "offline", "no frames processed") for a, b in _intervals([not o for o in online], t_start)]

    # --- per-track per-second samples
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        sec = int(r["time_sec"])
        for d in json.loads(r["detections_json"] or "[]"):
            key = str(d.get("subject_ref") or f"tracker_{d.get('tracker_id')}")
            fx, fy = d.get("foot") or d.get("raw_foot")
            bx1, by1, bx2, by2 = d.get("bbox") or (fx, fy - 170, fx, fy)
            acc[key][sec].append((fx, fy, by2 - by1, d.get("foot_source") in VISIBLE_FEET))
    # never join a tracker id across an offline gap: split it into separate pieces
    split = {}
    for key, per in acc.items():
        secs_all = sorted(per); piece = 0; prev = None
        for s_ in secs_all:
            if prev is not None and any(not online[i - t_start] for i in range(prev + 1, s_) if 0 <= i - t_start < N):
                piece += 1
            split.setdefault(f"{key}#{piece}", {})[s_] = per[s_]
            prev = s_
    acc = split
    tracks = {}
    tid = 0
    for key in sorted(acc, key=lambda k: min(acc[k])):
        secs = sorted(acc[key])
        if len(secs) < S["min_track_sec"]:
            continue                              # ghost / false detection
        tid += 1
        pts = {}
        for s in secs:
            v = acc[key][s]
            pts[s] = (statistics.median(p[0] for p in v), statistics.median(p[1] for p in v),
                      statistics.median(p[2] for p in v), sum(p[3] for p in v) / len(v) >= 0.5, 0)
        # coasting: bridge gaps up to coast_sec with the last position (never into offline seconds)
        for a, b in zip(secs, secs[1:]):
            if 1 < b - a <= S["coast_sec"] + 1:
                for s in range(a + 1, b):
                    if online[s - t_start]:
                        x, y, h, fv, _ = pts[a]
                        pts[s] = (x, y, h, fv, 1)
        tracks[f"p{tid}"] = dict(sorted(pts.items()))

    for tb in D.DERIVED_TABLES:
        conn.execute(f"DELETE FROM {tb} WHERE run_id=?", (run_id,))
    conn.executemany("INSERT INTO sa_positions VALUES (?,?,?,?,?,?,?,?)",
                     [(run_id, s, k, x, y, h, int(fv), c) for k, p in tracks.items() for s, (x, y, h, fv, c) in p.items()])
    conn.executemany("INSERT INTO sa_health VALUES (?,?,?,?,?)", [(run_id, a, b, st, n) for a, b, st, n in health])

    # --- people in frame
    half = S["count_median_sec"] // 2
    per_sec = [[] for _ in range(N)]
    for k, p in tracks.items():
        for s, v in p.items():
            if 0 <= s - t_start < N:
                per_sec[s - t_start].append((k, v))
    cnt = [len(per_sec[i]) if online[i] else None for i in range(N)]
    cnt_s = median_filter(cnt, half)
    conn.executemany("INSERT INTO sa_counts VALUES (?,?,?,?)",
                     [(run_id, a, b, v) for a, b, v in rle(cnt_s, t_start, S["min_run_sec_station"])])

    # --- zone states (floor areas)
    floor = [z for z in zones if z["kind"] == "floor_area"]
    zone_series = {}
    for z in floor:
        raw = [sum(1 for _, v in per_sec[i] if point_in_poly(v[0], v[1], z["points"])) if online[i] else None for i in range(N)]
        sm = median_filter(raw, half)
        zone_series[z["name"]] = sm
        mr = S["min_run_sec_station"] if z["role"] in ("operator_station",) else S["min_run_sec_other"]
        conn.executemany("INSERT INTO sa_zone_states VALUES (?,?,?,?,?)",
                         [(run_id, z["name"], a, b, v) for a, b, v in rle(sm, t_start, mr)])

    # --- edge events, episodes, short visits, dwell
    def near_offline(s):
        return any(not online[i] for i in range(max(0, s - t_start - 2), min(N, s - t_start + 3)))

    def edge_near(p, sa, sb):
        # the point itself, or one second extrapolated outward along the track's motion
        xa, ya = p[sa][:2]
        e = edge_of(xa, ya, W, H, TH["edge_margin_frac"], zones)
        if e or sb not in p:
            return e
        vx, vy = xa - p[sb][0], ya - p[sb][1]
        xe, ye = xa + vx, ya + vy
        if xe < 0 or xe > W or ye > H:
            exits = [z for z in zones if z.get("role") == "exit"]
            if exits:
                z = min(exits, key=lambda z: poly_dist(xa, ya, z["points"]))
                if poly_dist(xa, ya, z["points"]) / max(p[sa][2], 1) <= 3:
                    return z["name"]
            return "left image edge" if xe < 0 else ("right image edge" if xe > W else "bottom image edge")
        return edge_of(xe, ye, W, H, TH["edge_margin_frac"], zones)

    # edge events for all tracks first, then remove track switches at exits:
    # an OUT and an IN at the same exit within 3 s is one person whose ID changed, not two crossings
    edge_ev = []
    for k, p in tracks.items():
        secs = list(p); s0, s1 = secs[0], secs[-1]
        if s0 > t_start + 2 and not near_offline(s0):
            e = edge_near(p, s0, secs[1] if len(secs) > 1 else s0)
            if e:
                edge_ev.append([s0, "in", e, k])
        if s1 < t_end - 3 and not near_offline(s1):
            e = edge_near(p, s1, secs[-2] if len(secs) > 1 else s1)
            if e:
                edge_ev.append([s1 + 1, "out", e, k])
    edge_ev.sort()
    drop = set()
    for i1, e1 in enumerate(edge_ev):
        for i2, e2 in enumerate(edge_ev):
            if i1 < i2 and e1[2] == e2[2] and e1[1] != e2[1] and abs(e1[0] - e2[0]) <= 3 and not ({i1, i2} & drop):
                drop |= {i1, i2}
    edge_ev = [e for i, e in enumerate(edge_ev) if i not in drop]
    conn.executemany("INSERT INTO sa_edge_events VALUES (?,?,?,?,?)", [(run_id, t, d, e, k) for t, d, e, k in edge_ev])
    e_in_of = {k: e for t, d, e, k in edge_ev if d == "in"}
    e_out_of = {k: e for t, d, e, k in edge_ev if d == "out"}

    cell = TH["dwell_cell_px"]
    dwell = defaultdict(float)
    for k, p in tracks.items():
        secs = list(p)
        s0, s1 = secs[0], secs[-1]
        e_in, e_out = e_in_of.get(k), e_out_of.get(k)
        for s, (x, y, h, fv, c) in p.items():
            dwell[(s // 60, int(x // cell), int(y // cell))] += 1.0
        # places
        labels = [(s, place_label(v[0], v[1], zones)) for s, v in p.items()]
        runs = []
        for s, lab in labels:
            if runs and runs[-1][2] == lab and s - runs[-1][1] <= S["coast_sec"] + 1:
                runs[-1][1] = s + 1
            else:
                runs.append([s, s + 1, lab])
        merged = []
        for r in runs:
            if merged and r[1] - r[0] < S["min_run_sec_other"]:
                merged[-1][1] = r[1]
            elif merged and merged[-1][2] == r[2]:
                merged[-1][1] = r[1]
            else:
                merged.append(r)
        # fold short visits station -> X -> same station
        roles = {z["name"]: z["role"] for z in floor}
        folded, j = [], 0
        while j < len(merged):
            a, b, lab = merged[j]
            if (folded and j + 1 < len(merged) and roles.get(folded[-1][2]) == "operator_station"
                    and merged[j + 1][2] == folded[-1][2] and b - a <= TH["short_visit_max_sec"]):
                conn.execute("INSERT INTO sa_short_visits VALUES (?,?,?,?,?)", (run_id, folded[-1][2], lab, a, b))
                folded[-1][1] = merged[j + 1][1]
                j += 2
                continue
            folded.append([a, b, lab]); j += 1
        trivial = all(roles.get(lab) == "operator_station" or lab.startswith("at Bench") or lab.startswith("at ") and "footprint" in lab
                      for _, _, lab in folded)
        if (len(folded) > 1 or e_in or e_out) and not (trivial and not e_in and not e_out):
            parts = []
            if e_in:
                parts.append(f"entered at {e_in}")
            elif s0 > t_start + 2:
                parts.append(f"track starts {{t:{s0}}} inside view" + (" after camera offline" if near_offline(s0) else ""))
            parts += [f"{lab} ({{t:{a}}}–{{t:{b}}})" for a, b, lab in folded]
            if e_out:
                parts.append(f"left view at {e_out} {{t:{s1 + 1}}}")
            elif s1 < t_end - 3:
                parts.append(f"track ends {{t:{s1 + 1}}} inside view" + (" (camera offline)" if near_offline(s1) else " (lost: occlusion, crowd or ID break)"))
            conn.execute("INSERT INTO sa_episodes VALUES (?,?,?,?,?,?)",
                         (run_id, k, s0, s1 + 1, " → ".join(parts), json.dumps(folded)))
    conn.executemany("INSERT INTO sa_dwell VALUES (?,?,?,?,?)", [(run_id, m, cx, cy, v) for (m, cx, cy), v in dwell.items()])

    # --- detector events
    ev = []
    for z in floor:
        sm = zone_series[z["name"]]
        if z["role"] == "operator_station":
            flags = [v == 0 or v is None for v in sm]
            for a, b in _intervals(flags, t_start):
                seg = sm[a - t_start: b - t_start]
                empty = sum(1 for v in seg if v == 0)
                if b - a >= TH["absence_min_sec"] and empty > 0 and seg[0] == 0 and seg[-1] == 0:
                    unk = sum(1 for v in seg if v is None)
                    ev.append(("absence", a, b, z["name"], {"empty_sec": empty, "unknown_sec": unk}, 0.9 if unk == 0 else 0.7))
        if z["role"] == "restricted":
            for a, b in _intervals([bool(v) for v in sm], t_start):
                ev.append(("restricted_presence", a, b, z["name"], {}, 0.8))
    for k, p in tracks.items():
        items = [(s, v) for s, v in p.items()]
        # fast movement
        fast = []
        for (s1_, v1), (s2_, v2) in zip(items, items[1:]):
            if s2_ - s1_ == 1 and not v2[4] and not v1[4]:
                sp = math.hypot(v2[0] - v1[0], v2[1] - v1[1]) / max((v1[2] + v2[2]) / 2, 1)
                fast.append((s1_, sp >= TH["fast_bh_per_sec"]))
        flags = {s: f for s, f in fast}
        run_a = None
        for s, _ in items + [(10 ** 9, None)]:
            f = flags.get(s, False)
            if f and run_a is None:
                run_a = s
            if not f and run_a is not None:
                if s - run_a >= TH["fast_min_sec"]:
                    ev.append(("fast_movement", run_a, s, place_label(*p[run_a][:2], zones), {"track": k}, 0.6))
                run_a = None
    # long stationary spots (ID-free: any person keeps a spot occupied; survives ID switches)
    spots, done = [], []
    for i in range(N):
        sec = t_start + i
        for _, v in per_sec[i]:
            if v[4]:
                continue
            bh = max(v[2], 1)
            m = next((sp for sp in spots if math.hypot(v[0] - sp["x"], v[1] - sp["y"]) / bh <= TH["stationary_radius_bh"]), None)
            if m:
                m["last"] = sec; m["sx"] += v[0]; m["sy"] += v[1]; m["n"] += 1
            else:
                spots.append({"x": v[0], "y": v[1], "start": sec, "last": sec, "sx": v[0], "sy": v[1], "n": 1})
        keep = []
        for sp in spots:
            (keep if sec - sp["last"] <= S["coast_sec"] + 1 and online[i] else done).append(sp)
        spots = keep
    for sp in done + spots:
        if sp["last"] - sp["start"] + 1 >= TH["stationary_min_sec"]:
            in_work = any(z["role"] in ("operator_station", "work_surface") for z in zones_at(sp["x"], sp["y"], zones, "floor_area"))
            ev.append(("long_stationary", sp["start"], sp["last"] + 1, place_label(sp["x"], sp["y"], zones),
                       {"in_work_zone": in_work, "note": "someone stayed within %.2f bh; check frames before calling it rest/sleep" % TH["stationary_radius_bh"]}, 0.6))
    # outside-work-area dwell (ID-free spots, radius 1 bh, only positions outside work roles)
    roles_all = {z["name"]: z["role"] for z in zones}

    def work_role(lab):
        key = lab[3:].split(" (")[0] if lab.startswith("at ") and "footprint" in lab else lab
        return roles_all.get(key) in WORK_ROLES

    spots, done = [], []
    for i in range(N):
        sec = t_start + i
        for _, v in per_sec[i]:
            if work_role(place_label(v[0], v[1], zones)):
                continue
            bh = max(v[2], 1)
            m = next((sp for sp in spots if math.hypot(v[0] - sp["x"], v[1] - sp["y"]) / bh <= 1.0), None)
            if m:
                m["last"] = sec; m["sx"] += v[0]; m["sy"] += v[1]; m["n"] += 1
            else:
                spots.append({"x": v[0], "y": v[1], "start": sec, "last": sec, "sx": v[0], "sy": v[1], "n": 1})
        keep = []
        for sp in spots:
            (keep if sec - sp["last"] <= S["coast_sec"] + 2 and online[i] else done).append(sp)
        spots = keep
    for sp in done + spots:
        if sp["last"] - sp["start"] + 1 >= TH["outside_dwell_min_sec"]:
            cx, cy = sp["sx"] / sp["n"], sp["sy"] / sp["n"]
            ev.append(("outside_work_area_dwell", sp["start"], sp["last"] + 1, place_label(cx, cy, zones),
                       {"xy_px": [round(cx), round(cy)]}, 0.7))
    # pairwise per second: close pairs and crowding (anonymous, by location)
    pair_secs, crowd_secs = defaultdict(list), []
    for i in range(N):
        people = [v for _, v in per_sec[i] if not v[4]]
        s = t_start + i
        for a_ in range(len(people)):
            for b_ in range(a_ + 1, len(people)):
                pa, pb = people[a_], people[b_]
                bh = max((pa[2] + pb[2]) / 2, 1)
                if math.hypot(pa[0] - pb[0], pa[1] - pb[1]) / bh <= TH["close_pair_bh"]:
                    mx, my = (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2
                    pair_secs[(int(mx // (bh)), int(my // (bh)))].append((s, mx, my))
        for pa in people:
            near = [q for q in people if math.hypot(q[0] - pa[0], q[1] - pa[1]) / max(pa[2], 1) <= TH["crowd_radius_bh"]]
            if len(near) >= TH["crowd_min_people"]:
                crowd_secs.append((s, statistics.mean(q[0] for q in near), statistics.mean(q[1] for q in near), len(near)))
                break

    def runs_of(points, min_len, gap=5):
        points = sorted(points)
        out, cur = [], []
        for pnt in points:
            if cur and (pnt[0] - cur[-1][0] > gap or math.hypot(pnt[1] - cur[-1][1], pnt[2] - cur[-1][2]) > 150):
                if cur[-1][0] - cur[0][0] + 1 >= min_len:
                    out.append(cur)
                cur = []
            cur.append(pnt)
        if cur and cur[-1][0] - cur[0][0] + 1 >= min_len:
            out.append(cur)
        return out

    all_pairs = [p_ for v in pair_secs.values() for p_ in v]
    for grp in runs_of(all_pairs, TH["close_pair_min_sec"]):
        mx = statistics.median(g[1] for g in grp); my = statistics.median(g[2] for g in grp)
        ev.append(("close_pair", grp[0][0], grp[-1][0] + 1, place_label(mx, my, zones),
                   {"note": "distance only; chit-chat candidate"}, 0.6))
    for grp in runs_of(crowd_secs, TH["crowd_min_sec"]):
        mx = statistics.median(g[1] for g in grp); my = statistics.median(g[2] for g in grp)
        ev.append(("crowding", grp[0][0], grp[-1][0] + 1, place_label(mx, my, zones),
                   {"peak_people": max(g[3] for g in grp)}, 0.7))
    conn.executemany("INSERT INTO sa_events VALUES (?,?,?,?,?,?,?,?)",
                     [(run_id, t, a, b, z, json.dumps(d), c, "supervisor_agent.derive") for t, a, b, z, d, c in ev])

    conn.execute("INSERT OR REPLACE INTO sa_runs VALUES (?,?,?,?,?,?,?,?,?)",
                 (run_id, camera, datetime.now(timezone.utc).isoformat(), VERSION, t_start, t_end, W, H,
                  json.dumps({"semantics_version": sem["version"]})))
    conn.commit()
    return dict(run_id=run_id, camera=camera, tracks=len(tracks), raw_ids=len(acc), events=len(ev),
                offline=[(a, b) for a, b, *_ in health])


def import_events(conn, run_id: str, events: list[dict]) -> int:
    """Import detector events from the main system. Each: {type,start_sec,end_sec,zone,details,confidence}."""
    D.init_derived(conn)
    conn.executemany("INSERT INTO sa_events VALUES (?,?,?,?,?,?,?,?)",
                     [(run_id, e["type"], int(e["start_sec"]), int(e["end_sec"]), e.get("zone", ""),
                       json.dumps(e.get("details", {})), float(e.get("confidence", 0.8)), "main_system") for e in events])
    conn.commit()
    return len(events)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--run", default="all", help="run_id or 'all'")
    ap.add_argument("--db")
    a = ap.parse_args()
    conn = D.connect(a.db)
    cam = D.resolve_camera(conn, a.camera)
    runs = [r["run_id"] for r in D.camera_runs(conn, cam)] if a.run == "all" else [a.run]
    for r in runs:
        print(build(conn, r))


if __name__ == "__main__":
    main()
