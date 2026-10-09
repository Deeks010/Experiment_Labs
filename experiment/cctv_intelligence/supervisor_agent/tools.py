"""The agent's tools. Each returns {"text": str, "images": [paths]}.

All numbers are computed here (the data layer). The LLM only reads the text.
Times: wall-clock 'HH:MM[:SS]' or 'YYYY-MM-DD HH:MM[:SS]' when a recording clock is bound,
else video time 'v00:05:00' or seconds.
"""
from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict

from . import db as D
from .geometry import relation_text

WORK_ROLES = {"operator_station", "work_surface", "walkway", "exit"}
MAX_SCENE_WINDOW_SEC = 30 * 60


def _r(text, images=None):
    return {"text": text.rstrip() + "\n", "images": images or []}


def _ft(run, s):
    return D.fmt_t(run, s)


def _sub_times(run, text):
    return re.sub(r"\{t:(\d+)\}", lambda m: _ft(run, int(m.group(1))), text)


def _span(run, a, b):
    return f"{_ft(run, a)}–{_ft(run, b)}"


def _clip(rows, a, b):
    out = []
    for r in rows:
        s, e = max(r[0], a), min(r[1], b)
        if e > s:
            out.append((s, e) + tuple(r[2:]))
    return out


def _windows(conn, camera, start, end):
    cam = D.resolve_camera(conn, camera)
    wins = D.runs_for_window(conn, cam, start, end)
    if not wins:
        cov = D.coverage_text(conn, cam)
        raise ValueError(f"No recording of {cam} overlaps {start}–{end}. Available recordings:\n{cov}")
    D.init_derived(conn)
    for run, *_ in wins:
        if not conn.execute("SELECT 1 FROM sa_runs WHERE run_id=?", (run["run_id"],)).fetchone():
            from .derive import build
            build(conn, run["run_id"])
    return cam, wins


def _health_lines(conn, run, a, b):
    off = _clip(conn.execute("SELECT start,end,note FROM sa_health WHERE run_id=? ORDER BY start", (run["run_id"],)).fetchall(), a, b)
    if not off:
        return "HEALTH   online for the whole window"
    tot = sum(e - s for s, e, _ in off)
    return "HEALTH   OFFLINE " + ", ".join(_span(run, s, e) for s, e, _ in off) + f" ({tot} s, nothing known then); online otherwise"


# ---------------------------------------------------------------- 1. list_cameras
def list_cameras(conn) -> dict:
    D.init_derived(conn)
    lines = ["CAMERAS"]
    for cam in D.list_camera_names(conn):
        info = conn.execute("SELECT description FROM sa_camera_info WHERE camera=?", (cam,)).fetchone()
        zones = D.load_zones(conn, cam)
        lines.append(f"- {cam}: {(info['description'][:220] + '…') if info and len(info['description']) > 220 else (info['description'] if info else 'no description')}")
        lines.append(f"  zones: {len(zones)} ({', '.join(z['name'] for z in zones[:12])}{'…' if len(zones) > 12 else ''})")
        lines.append("  recordings:\n" + D.coverage_text(conn, cam))
    return _r("\n".join(lines))


# ---------------------------------------------------------------- 2. get_camera_info
def get_camera_info(conn, camera: str, place: str | None = None, include_reference_frame: bool = False) -> dict:
    cam = D.resolve_camera(conn, camera)
    D.init_derived(conn)
    zones = D.load_zones(conn, cam)
    info = conn.execute("SELECT description, metadata_json FROM sa_camera_info WHERE camera=?", (cam,)).fetchone()
    sem = D.load_semantics()
    L = [f"CAMERA {cam}", f"DESCRIPTION {info['description'] if info else 'not set'}"]
    if info and info["metadata_json"]:
        L.append(f"METADATA {info['metadata_json']}")
    L.append("RECORDINGS\n" + D.coverage_text(conn, cam))
    L.append("STATIC OBJECTS (solid; people stand around them)")
    for z in zones:
        if z["kind"] == "static_object":
            L.append(f"  {z['name']:22} role={z['role'] or '-':16} {z['description']}" + (f"  aka: {', '.join(z['synonyms'])}" if z['synonyms'] else ""))
    L.append("FLOOR AREAS (where people stand or walk)")
    for z in zones:
        if z["kind"] == "floor_area":
            L.append(f"  {z['name']:22} role={z['role'] or '-':16} {z['description']}" + (f"  aka: {', '.join(z['synonyms'])}" if z['synonyms'] else ""))
    L.append("IMAGE TERMS  'far side' = above an object in the image (away from camera); 'camera side' = below it; "
             "distances in body heights (bh, ~1.7 m).")
    L.append("ROLES  " + "; ".join(f"{k}: {v}" for k, v in sem["zone_roles"].items() if not k.startswith("_")))
    if place:
        q = place.lower()
        words = [w for w in re.findall(r"[a-z0-9.]+", q) if len(w) > 1]
        scored = []
        for z in zones:
            hay = " ".join([z["name"], z["description"], " ".join(z["synonyms"])]).lower()
            sc = sum(1 for w in words if w in hay) + (3 if z["name"].lower() in q or any(s in q for s in z["synonyms"]) else 0)
            if sc:
                scored.append((sc, z))
        scored.sort(key=lambda t: -t[0])
        L.append(f"PLACE MATCH for '{place}': " + (", ".join(f"{z['name']} ({z['kind']}, role {z['role'] or '-'})" for _, z in scored[:4]) or "no match — ask the user or use the zone list"))
    images = []
    if include_reference_frame:
        runs = D.camera_runs(conn, cam)
        if runs:
            from .render import frame_sheet
            p, notes = frame_sheet(runs[-1], [runs[-1]["start_sec"] + 1], zones, {}, tag="reference")
            if p:
                images.append(p); L.append(f"REFERENCE FRAME attached: {p}")
    return _r("\n".join(L), images)


# ---------------------------------------------------------------- 3. get_scene_log
def get_scene_log(conn, camera: str, start, end, focus: str | None = None, detail: str = "normal") -> dict:
    cam, wins = _windows(conn, camera, start, end)
    sem = D.load_semantics()
    zones = D.load_zones(conn, cam)
    roles = {z["name"]: z["role"] for z in zones}
    out = []
    for run, a, b in wins:
        a, b = int(a), int(b)
        if b - a > MAX_SCENE_WINDOW_SEC:
            b = a + MAX_SCENE_WINDOW_SEC
            out.append(f"NOTE window cut to 30 min ({_span(run, a, b)}). For longer periods use get_dwell_overview or get_events.")
        rid = run["run_id"]
        L = [f"SCENE LOG {cam} {D.fmt_t(run, a, True)} – {_ft(run, b)}" + (f"  focus={focus}" if focus else ""),
             _health_lines(conn, run, a, b),
             "LIMITS   " + sem["limits_text"]]
        counts = _clip(conn.execute("SELECT start,end,value FROM sa_counts WHERE run_id=? ORDER BY start", (rid,)).fetchall(), a, b)
        L.append("\nPEOPLE IN FRAME")
        L += [f"  {_span(run, s, e)}  {'UNKNOWN (offline)' if v is None else v}" for s, e, v in counts]
        zs = defaultdict(list)
        for z, s, e, v in conn.execute("SELECT zone,start,end,value FROM sa_zone_states WHERE run_id=? ORDER BY start", (rid,)):
            zs[z].append((s, e, v))
        sv = defaultdict(list)
        for st, vz, s, e in conn.execute("SELECT station,visit_zone,start,end FROM sa_short_visits WHERE run_id=? AND start<? AND end>?", (rid, b, a)):
            sv[st].append(s)
        L.append("\nOPERATOR STATIONS")
        for z in [z for z in zones if z["role"] == "operator_station"]:
            if focus and focus.lower() not in z["name"].lower() and detail != "all":
                pass
            full = zs.get(z["name"], [])
            short = sem["thresholds"]["short_visit_max_sec"]
            stepaway = {(s, e) for i_, (s, e, v) in enumerate(full)
                        if v == 0 and e - s <= short and 0 < i_ < len(full) - 1 and full[i_ - 1][2] and full[i_ + 1][2]}
            runs_ = [(max(s, a), min(e, b), v, (s, e) in stepaway) for s, e, v in full if min(e, b) > max(s, a)]
            # ID-free short step-away: an EMPTY run of at most short_visit_max_sec between non-empty runs
            segs, steps = [], []
            for s, e, v, is_step in runs_:
                lab = "UNKNOWN" if v is None else ("EMPTY" if v == 0 else ("occupied" if v == 1 else f"{v} people"))
                if is_step:
                    lab = "occupied"; steps.append(s)
                if v and v >= 1 and segs and segs[-1][2] in ("occupied",) and lab == "occupied":
                    pass
                if segs and segs[-1][2] == lab:
                    segs[-1][1] = e
                else:
                    segs.append([s, e, lab])
            if len(segs) == 1:
                txt = f"{segs[0][2]} whole window"
            else:
                txt = " | ".join(f"{lab} {_span(run, s, e)}" for s, e, lab in segs)
            L.append(f"  {z['name']:10} {txt}" + (f"   short step-aways (≤{sem['thresholds']['short_visit_max_sec']}s, counted as occupied) at {', '.join(_ft(run, x) for x in steps)}" if steps else ""))
        L.append("\nOTHER AREAS (non-zero periods)")
        for z in [z for z in zones if z["kind"] == "floor_area" and z["role"] != "operator_station"]:
            nz = [(s, e, v) for s, e, v in _clip(zs.get(z["name"], []), a, b) if v]
            if nz:
                L.append(f"  {z['name']:20} " + " | ".join(f"{v} {'person' if v == 1 else 'people'} {_span(run, s, e)}" for s, e, v in nz))
        edges = conn.execute("SELECT t,direction,edge FROM sa_edge_events WHERE run_id=? AND t>=? AND t<? ORDER BY t", (rid, a, b)).fetchall()
        L.append("\nEDGE CROSSINGS (people entering or leaving the camera view)")
        L += [f"  {_ft(run, t)}  {'IN ' if d == 'in' else 'OUT'}  {e}" for t, d, e in edges] or ["  none"]
        eps = conn.execute("SELECT track,start,end,text,places_json FROM sa_episodes WHERE run_id=? AND start<? AND end>? ORDER BY start",
                           (rid, b, a)).fetchall()
        if focus:
            eps = [e for e in eps if focus.lower() in e["text"].lower()]
        L.append("\nMOVEMENTS (one line = one short continuous track = 'a person moved this way'. Lines are NOT joined into persons.)")
        for i, e in enumerate(eps[:40]):
            L.append(f"  m{i + 1}  {_sub_times(run, e['text'])}")
        if len(eps) > 40:
            L.append(f"  … {len(eps) - 40} more (narrow the window or set focus)")
        if not eps:
            L.append("  none")
        if detail in ("positions", "all"):
            from .geometry import place_label
            L.append("\nPOSITIONS (each stay of each visible person in the window, median foot point, relative to the map; p-numbers are per-track only, not identities)")
            rows = conn.execute("SELECT track,t,x,y,body_h,feet_visible FROM sa_positions WHERE run_id=? AND t>=? AND t<? ORDER BY track,t",
                                (rid, a, b)).fetchall()
            by = defaultdict(list)
            for r_ in rows:
                by[r_[0]].append(r_)
            stays = []
            for trk, pts in by.items():
                cur = []
                for r_ in pts + [None]:
                    lab = place_label(r_[2], r_[3], zones) if r_ else None
                    if cur and (r_ is None or lab != cur[0][1] or r_[1] - cur[-1][0][1] > 3):
                        if cur[-1][0][1] - cur[0][0][1] + 1 >= 3:
                            stays.append((cur[0][0][1], cur[-1][0][1] + 1, trk, [c[0] for c in cur]))
                        cur = []
                    if r_:
                        cur.append((r_, lab))
            stays.sort()
            for s2, e2, trk, pts in stays[:40]:
                x = statistics.median(p[2] for p in pts); y = statistics.median(p[3] for p in pts)
                bh = statistics.median(p[4] for p in pts); fv = sum(p[5] for p in pts) / len(pts)
                feet = "feet visible" if fv >= 0.8 else f"FEET HIDDEN {100 - fv * 100:.0f}% (position uncertain; check frames)"
                L.append(f"  {trk:4} {_span(run, s2, e2)}  {relation_text(x, y, bh, zones)} | {feet} | xy=({x:.0f},{y:.0f})px")
            if len(stays) > 40:
                L.append(f"  … {len(stays) - 40} more stays (narrow the window)")
        evs = conn.execute("SELECT type,start,end,zone FROM sa_events WHERE run_id=? AND start<? AND end>? ORDER BY start", (rid, b, a)).fetchall()
        L.append("\nDETECTOR EVENTS IN WINDOW (details: get_events)")
        L += [f"  {_span(run, s, e)}  {t:24} {z}" for t, s, e, z in evs] or ["  none"]
        ins = sum(1 for _, d, _ in edges if d == "in")
        start_cnt = next((v for s, e, v in counts if s <= a < e and v is not None), None)
        peak = max((v for *_, v in counts if v is not None), default=0)
        L.append("\nDERIVED")
        if start_cnt is not None:
            L.append(f"  different people in window: at least {peak} (most at one time), at most {start_cnt + ins} "
                     f"({start_cnt} at start + {ins} entries; re-entries of the same person cannot be told apart; "
                     f"edge crossings are about 90% reliable, so treat the bounds as ±1)")
        L.append("  identity: NOT available. Use 'a person'. For identity questions use get_image_frames and say how uncertain it is.")
        out.append("\n".join(L))
    return _r("\n\n".join(out))


# ---------------------------------------------------------------- 4. get_dwell_overview
def get_dwell_overview(conn, camera: str, start, end, include_heatmap: bool = False) -> dict:
    cam, wins = _windows(conn, camera, start, end)
    sem = D.load_semantics(); TH = sem["thresholds"]
    zones = D.load_zones(conn, cam)
    roles = {z["name"]: z["role"] for z in zones}
    from .geometry import place_label

    def role_of(lab):
        if lab.startswith("at ") and "footprint" in lab:
            return roles.get(lab[3:].split(" (")[0], "")
        return roles.get(lab, "")
    out, images = [], []
    for run, a, b in wins:
        a, b = int(a), int(b); rid = run["run_id"]
        L = [f"DWELL OVERVIEW {cam} {D.fmt_t(run, a, True)} – {_ft(run, b)}", _health_lines(conn, run, a, b),
             "LIMITS   All people added together. No identities. No time order. Use get_scene_log for WHEN / ORDER / HOW MANY."]
        per = defaultdict(float); bh_at = defaultdict(list)
        for x, y, bh in conn.execute("SELECT x,y,body_h FROM sa_positions WHERE run_id=? AND t>=? AND t<?", (rid, a, b)):
            lab = place_label(x, y, zones)
            per[lab] += 1
        merged_per = defaultdict(float)
        for lab, s in per.items():
            merged_per["unmapped floor (outside all mapped areas; see hotspots for where)" if lab.startswith("unmapped floor") else lab] += s
        per = merged_per
        L.append("\nTIME PER PLACE (person-minutes)")
        for lab, s in sorted(per.items(), key=lambda kv: -kv[1]):
            role = role_of(lab)
            tag = ("normal work spot" if role in ("operator_station", "work_surface") else "walkway" if role in ("walkway", "exit")
                   else "control area (not a work station)" if role == "control" else "← not a work spot")
            L.append(f"  {lab:58} {s / 60:6.1f}   {tag}")
        cell = TH["dwell_cell_px"]
        cells = defaultdict(float)
        for cx, cy, s in conn.execute("SELECT cx,cy,sum(seconds) FROM sa_dwell WHERE run_id=? AND minute>=? AND minute<? GROUP BY cx,cy",
                                      (rid, a // 60, (b + 59) // 60)):
            cells[(cx, cy)] = s
        keep = {k for k, v in cells.items() if v >= 8}
        seen, hs = set(), []
        for k in keep:
            if k in seen:
                continue
            st, comp = [k], []
            seen.add(k)
            while st:
                c = st.pop(); comp.append(c)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        n = (c[0] + dx, c[1] + dy)
                        if n in keep and n not in seen:
                            seen.add(n); st.append(n)
            tot = sum(cells[c] for c in comp)
            if tot >= TH["hotspot_min_sec"]:
                x = sum((c[0] + .5) * cell * cells[c] for c in comp) / tot; y = sum((c[1] + .5) * cell * cells[c] for c in comp) / tot
                hs.append((tot, x, y, comp))
        hs.sort(key=lambda h: -h[0])
        L.append(f"\nHOTSPOTS (connected areas with ≥{TH['hotspot_min_sec'] / 60:.1f} min; centre described relative to the map)")
        for i, (tot, x, y, _) in enumerate(hs[:12]):
            near = conn.execute("SELECT avg(body_h) FROM sa_positions WHERE run_id=? AND abs(x-?)<60 AND abs(y-?)<60", (rid, x, y)).fetchone()[0] or 150
            lab = place_label(x, y, zones)
            unusual = role_of(lab) not in WORK_ROLES
            L.append(f"  H{i + 1:<2} {tot / 60:5.1f} min  {relation_text(x, y, near, zones)}  xy=({x:.0f},{y:.0f})px" + ("   UNUSUAL" if unusual else ""))
        if not hs:
            L.append("  none")
        if include_heatmap:
            cfg = run["config"]
            from .render import heatmap
            p = heatmap(
                run, cells, cell, zones, hs[:12], int(cfg.get("width") or 1920), int(cfg.get("height") or 1080), f"{rid}_{a}_{b}")
            images.append(p); L.append(f"\nIMAGE attached: labelled heatmap (H1.. match the list above): {p}")
        L.append("\nNEXT STEP  To see WHEN a hotspot happened: get_scene_log(camera, start, end, focus=<place>) or get_events.")
        out.append("\n".join(L))
    return _r("\n\n".join(out), images)


# ---------------------------------------------------------------- 5. get_events
def get_events(conn, start, end, camera: str | None = None, types: list[str] | None = None) -> dict:
    cams = [D.resolve_camera(conn, camera)] if camera else D.list_camera_names(conn)
    sem = D.load_semantics()
    L = ["DETECTOR EVENTS" + (f" types={','.join(types)}" if types else "")]
    n = 0
    for cam in cams:
        try:
            _, wins = _windows(conn, cam, start, end)
        except ValueError:
            continue
        for run, a, b in wins:
            q = "SELECT type,start,end,zone,details_json,confidence,source FROM sa_events WHERE run_id=? AND start<? AND end>?"
            rows = conn.execute(q + " ORDER BY start", (run["run_id"], b, a)).fetchall()
            for t, s, e, z, d, c, src in rows:
                if types and t not in types:
                    continue
                n += 1
                det = json.loads(d or "{}")
                det.pop("track", None)
                L.append(f"  {cam}  {D.fmt_t(run, s, True)}–{_ft(run, e)} ({e - s} s)  {t:24} {z}  conf {c:.2f}  src={src}"
                         + (f"  {json.dumps(det)}" if det else ""))
    if n == 0:
        L.append("  none in this window")
    L.append("DEFINITIONS " + "; ".join(f"{k}: {v}" for k, v in sem["definitions"].items() if (not types or k in types)))
    L.append("NOTE Events are candidates from positions only. Verify important ones with get_image_frames.")
    return _r("\n".join(L))


# ---------------------------------------------------------------- 6. compare_periods
def _metrics(conn, cam, start, end) -> dict:
    _, wins = _windows(conn, cam, start, end)
    zones = D.load_zones(conn, cam)
    m = defaultdict(float)
    for run, a, b in wins:
        a, b = int(a), int(b); rid = run["run_id"]
        off = sum(e - s for s, e, *_ in _clip(conn.execute("SELECT start,end FROM sa_health WHERE run_id=?", (rid,)).fetchall(), a, b))
        m["window_min"] += (b - a) / 60; m["online_min"] += (b - a - off) / 60
        cnt = _clip(conn.execute("SELECT start,end,value FROM sa_counts WHERE run_id=?", (rid,)).fetchall(), a, b)
        m["person_min_in_view"] += sum((e - s) * v for s, e, v in cnt if v) / 60
        m["peak_people"] = max(m["peak_people"], max((v for *_, v in cnt if v), default=0))
        short = D.load_semantics()["thresholds"]["short_visit_max_sec"]
        for z in zones:
            if z["role"] == "operator_station":
                st = _clip(conn.execute("SELECT start,end,value FROM sa_zone_states WHERE run_id=? AND zone=? ORDER BY start", (rid, z["name"])).fetchall(), a, b)
                m[f"{z['name']} empty_min (runs >{short}s)"] += sum(e - s for s, e, v in st if v == 0 and e - s > short) / 60
                m[f"{z['name']} short step-aways"] += sum(1 for i, (s, e, v) in enumerate(st)
                                                          if v == 0 and e - s <= short and 0 < i < len(st) - 1 and st[i - 1][2] and st[i + 1][2])
        for d, n in conn.execute("SELECT direction,count(*) FROM sa_edge_events WHERE run_id=? AND t>=? AND t<? GROUP BY direction", (rid, a, b)):
            m[f"edge {d}"] += n
        for t, n, dur in conn.execute("SELECT type,count(*),sum(min(end,?)-max(start,?)) FROM sa_events WHERE run_id=? AND start<? AND end>? GROUP BY type",
                                      (b, a, rid, b, a)):
            m[f"event {t} count"] += n; m[f"event {t} min"] += dur / 60
    m["avg_people_in_view"] = m["person_min_in_view"] / m["online_min"] if m["online_min"] else 0
    return dict(m)


def compare_periods(conn, camera: str, a_start, a_end, b_start, b_end) -> dict:
    cam = D.resolve_camera(conn, camera)
    A, B = _metrics(conn, cam, a_start, a_end), _metrics(conn, cam, b_start, b_end)
    keys = ["window_min", "online_min", "avg_people_in_view", "peak_people", "edge in", "edge out"]
    keys += sorted(k for k in set(A) | set(B) if "empty_min" in k)
    keys += sorted(k for k in set(A) | set(B) if k.endswith("short step-aways"))
    keys += sorted(k for k in set(A) | set(B) if k.startswith("event "))
    L = [f"COMPARE {cam}", f"  A = {a_start} → {a_end}", f"  B = {b_start} → {b_end}",
         "  Same metrics, same code, both periods. Differences in online minutes change totals: compare per online hour when they differ.",
         f"\n  {'metric':34} {'A':>8} {'B':>8} {'B−A':>8}"]
    for k in keys:
        va, vb = A.get(k, 0.0), B.get(k, 0.0)
        L.append(f"  {k:34} {va:8.1f} {vb:8.1f} {vb - va:+8.1f}")
    L.append("\n  Event types: absence, long_stationary, close_pair, crowding, fast_movement, outside_work_area_dwell, restricted_presence, plus imported main-system events.")
    L.append("  For where time went in each period: get_dwell_overview for A and for B.")
    return _r("\n".join(L))


# ---------------------------------------------------------------- 7. get_image_frames
def get_image_frames(conn, camera: str, time=None, start=None, end=None, count: int = 1, crop_zone: str | None = None,
                     overlay: bool = True) -> dict:
    from .render import frame_sheet
    cam = D.resolve_camera(conn, camera)
    count = max(1, min(int(count or 1), 9))
    if time is not None:
        start = end = time
    run = None
    for r in D.camera_runs(conn, cam):
        try:
            s = D.parse_time_arg(start, r)
        except ValueError:
            continue
        if r["start_sec"] <= s <= r["end_sec"]:
            run = r; break
    if run is None:
        return _r(f"No recording of {cam} at {start}. Recordings:\n{D.coverage_text(conn, cam)}")
    s = D.parse_time_arg(start, run)
    e = D.parse_time_arg(end, run) if end is not None else s
    times = [s] if count == 1 or e <= s else [s + (e - s) * i / (count - 1) for i in range(count)]
    zones = D.load_zones(conn, cam)
    dets = {}
    for t in times:
        row = conn.execute("SELECT time_sec, detections_json FROM frame_observations WHERE run_id=? ORDER BY abs(time_sec-?) LIMIT 1",
                           (run["run_id"], t)).fetchone()
        if row and abs(row["time_sec"] - t) <= 1:
            dets[t] = [d["bbox"] for d in json.loads(row["detections_json"]) if d.get("bbox")]
    crop = None
    if crop_zone:
        z = next((z for z in zones if z["name"].lower() == crop_zone.lower()), None)
        if z:
            x1, y1, x2, y2 = z["box"]; pad = max(0.8 * max(x2 - x1, y2 - y1), 260)
            crop = (x1 - pad, y1 - pad * 1.5, x2 + pad, y2 + pad)
    path, notes = frame_sheet(run, times, zones, dets, crop=crop, overlay=overlay, tag="frames")
    L = [f"FRAMES {cam}  {len(times)} frame(s): " + ", ".join(_ft(run, t) for t in times),
         "  Green boxes = person detections at that moment (#n is per-frame only, not an identity). Zone outlines drawn.",
         "  Frames are in time order; read the time printed on each tile." + ("  SIMULATED VIDEO." if run["source_kind"] == "synthetic" else "")]
    L += [f"  {n}" for n in notes]
    if path:
        L.append(f"  image: {path}")
    return _r("\n".join(L), [path] if path else [])


TOOLS = {
    "list_cameras": list_cameras,
    "get_camera_info": get_camera_info,
    "get_scene_log": get_scene_log,
    "get_dwell_overview": get_dwell_overview,
    "get_events": get_events,
    "compare_periods": compare_periods,
    "get_image_frames": get_image_frames,
}


def call(conn, name: str, args: dict) -> dict:
    if name not in TOOLS:
        return _r(f"Unknown tool {name}. Tools: {', '.join(TOOLS)}")
    try:
        return TOOLS[name](conn, **args)
    except (ValueError, TypeError) as e:
        return _r(f"ERROR in {name}: {e}")
