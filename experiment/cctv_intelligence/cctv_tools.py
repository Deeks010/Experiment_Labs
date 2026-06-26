"""
cctv_tools.py

Tool layer for the CCTV Intelligence bot.
Converts raw DB data (pixel coordinates, zone boxes) into human-readable
spatial facts. No interpretation of what a person was doing is ever made.

STRICT RULE:
  - Zones are STATIC OBJECTS (machines, workstations, panels, areas).
  - We NEVER say "person was in zone X" or "person was at zone X".
  - We ONLY report: frame region + which objects were nearby + proximity label.
  - The AI bot reads this and draws its own conclusions.

Tools:
  1. get_map(camera, t_start, t_end)         — dynamic ASCII floor map
  2. get_camera_info(camera=None)            — camera metadata
  3. get_zones_info(camera)                  — zone descriptions
  4. get_people_count(camera, t_start, t_end, zone=None) — segment count timeline
  5. get_day_summary(camera, date=None)      — rich narrative session summary
  6. get_video_frame(camera, t_sec)          — extract real video frame image
  7. get_activity_table(camera, t_start, t_end) — translated spatial facts table
"""

import json
import math
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "cctv_maps.sqlite3"

GRID_W = 90
GRID_H = 30
SYMBOLS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮"

# Proximity bands as % of frame width
# (upper_bound_pct, label)
PROXIMITY_BANDS = [
    (2.0,  "RIGHT NEXT TO"),
    (6.0,  "VERY CLOSE"),
    (12.0, "NEARBY"),
]

STITCH_CONFIG = {
    "direction_points": 8,
    "max_time_gap_sec": 3.0,
    "max_distance_px": 260.0,
    "max_speed_px_sec": 180.0,
    "max_angle_deg": 110.0,
    "conflict_radius_px": 130.0,
    "stationary_direction_px": 18.0,
}


# ─────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────

def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _dist_to_box_edge(px, py, box):
    """Euclidean distance from point to nearest edge of a bounding box."""
    x1, y1, x2, y2 = box
    cx = max(x1, min(px, x2))
    cy = max(y1, min(py, y2))
    return math.hypot(px - cx, py - cy)


def _proximity_label(dist_px, frame_w):
    """Convert pixel distance to a human-readable proximity label, or None if too far."""
    pct = (dist_px / frame_w) * 100
    for upper, label in PROXIMITY_BANDS:
        if pct <= upper:
            return label
    return None  # beyond 12% → don't include


def _frame_region(x, y, frame_w, frame_h):
    """Return a 3×3 grid region label for a coordinate."""
    col = "LEFT" if x < frame_w / 3 else ("CENTER" if x < 2 * frame_w / 3 else "RIGHT")
    row = "UPPER" if y < frame_h / 3 else ("MIDDLE" if y < 2 * frame_h / 3 else "LOWER")
    return f"{col}-{row}"


def _friendly_region(region: str) -> str:
    col, row = region.split("-")
    col_words = {"LEFT": "left side", "CENTER": "middle", "RIGHT": "right side"}
    row_words = {"UPPER": "far end", "MIDDLE": "middle stretch", "LOWER": "near end"}
    return f"{row_words.get(row, row.lower())} of the {col_words.get(col, col.lower())}"


def _translate_point(x, y, zones, frame_w, frame_h):
    """
    Translate a pixel coordinate into spatial facts.
    Returns (region_label, nearby_list) where nearby_list contains
    strings like 'Press No.9 (VERY CLOSE)' for objects within 12% of frame.
    """
    region = _frame_region(x, y, frame_w, frame_h)
    nearby = []
    for z in zones:
        d = _dist_to_box_edge(x, y, z["box"])
        label = _proximity_label(d, frame_w)
        if label:
            nearby.append(f"{z['name']} ({label})")
    return region, nearby


def _load_zones(camera_id, cur):
    rows = cur.execute(
        "SELECT zone_name, geometry_json, metadata_json FROM camera_zones WHERE camera_id=?",
        (camera_id,)
    ).fetchall()
    zones = []
    for r in rows:
        geom = json.loads(r["geometry_json"])
        box = geom.get("box")
        if box:
            meta = json.loads(r["metadata_json"])
            zones.append({
                "name": r["zone_name"],
                "box": box,
                "description": meta.get("description", ""),
            })
    return zones


def _get_camera(camera_name, cur):
    row = cur.execute("SELECT id FROM cameras WHERE name=?", (camera_name,)).fetchone()
    return row["id"] if row else None


def _infer_frame_dims(zones):
    if not zones:
        return 2560.0, 1440.0
    xs = [z["box"][0] for z in zones] + [z["box"][2] for z in zones]
    ys = [z["box"][1] for z in zones] + [z["box"][3] for z in zones]
    return max(xs) * 1.02, max(ys) * 1.02


def _load_segments(camera_id, t_start, t_end, cur):
    """
    Load all tracking segments that overlap [t_start, t_end].
    Returns a list of dicts with segment_id, points [(t,x,y), ...].
    """
    t_s = t_start if t_start is not None else 0.0
    t_e = t_end if t_end is not None else 999999.0
    rows = cur.execute(
        """SELECT id, person_path_json, metadata_json
           FROM floor_data
           WHERE camera_id=? AND end_time_sec >= ? AND start_time_sec <= ?""",
        (camera_id, t_s, t_e)
    ).fetchall()

    segments = []
    for row in rows:
        data = json.loads(row["person_path_json"])
        for sub in data.get("subjects", []):
            pts = [
                (pt["t"], pt["foot"][0], pt["foot"][1])
                for pt in sub.get("path_points", [])
                if t_s <= pt["t"] <= t_e
            ]
            if pts:
                segments.append({
                    "segment_id": f"row{row['id']}_{sub['subject_ref']}",
                    "start": pts[0][0],
                    "end": pts[-1][0],
                    "points": pts,
                    "confidence": sub.get("confidence", 0.0),
                })
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# Track stitching / worker consolidation enrichment layer
# ─────────────────────────────────────────────────────────────────────────────

def _point_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _path_distance(points):
    total = 0.0
    for (_, x1, y1), (_, x2, y2) in zip(points, points[1:]):
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def _mean_direction_vector(points):
    if len(points) < 2:
        return None
    angles = []
    total_move = 0.0
    for (_, x1, y1), (_, x2, y2) in zip(points, points[1:]):
        dx = x2 - x1
        dy = y2 - y1
        step = math.hypot(dx, dy)
        if step < 1.0:
            continue
        total_move += step
        angles.append(math.atan2(dy, dx))
    if not angles or total_move < STITCH_CONFIG["stationary_direction_px"]:
        return None
    sin_mean = sum(math.sin(a) for a in angles) / len(angles)
    cos_mean = sum(math.cos(a) for a in angles) / len(angles)
    mag = math.hypot(cos_mean, sin_mean)
    if mag < 0.05:
        return None
    return (cos_mean / mag, sin_mean / mag)


def _angle_diff_deg(v1, v2):
    if not v1 or not v2:
        return None
    dot = max(-1.0, min(1.0, v1[0] * v2[0] + v1[1] * v2[1]))
    return math.degrees(math.acos(dot))


def _track_position_at(profile, t_sec):
    pts = profile["points"]
    if not pts or t_sec < profile["start"] or t_sec > profile["end"]:
        return None
    best = min(pts, key=lambda p: abs(p[0] - t_sec))
    return (best[1], best[2])


def _profile_segments(segments):
    n_dir = STITCH_CONFIG["direction_points"]
    profiles = {}
    for seg in segments:
        pts = sorted(seg["points"], key=lambda p: p[0])
        if not pts:
            continue
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        profiles[seg["segment_id"]] = {
            "segment_id": seg["segment_id"],
            "start": pts[0][0],
            "end": pts[-1][0],
            "first_xy": (pts[0][1], pts[0][2]),
            "last_xy": (pts[-1][1], pts[-1][2]),
            "points": pts,
            "path_distance": _path_distance(pts),
            "entry_dir": _mean_direction_vector(pts[:n_dir]),
            "exit_dir": _mean_direction_vector(pts[-n_dir:]),
            "bbox": (min(xs), min(ys), max(xs), max(ys)),
        }
    return profiles


def _evaluate_stitch_pair(a, b, profiles):
    if b["start"] <= a["end"]:
        return None, "Gate 1 failed: second path started before first path ended"

    time_gap = b["start"] - a["end"]
    if time_gap > STITCH_CONFIG["max_time_gap_sec"]:
        return None, f"Gate 2 failed: time gap {time_gap:.2f}s is too long"

    distance = _point_distance(a["last_xy"], b["first_xy"])
    if distance > STITCH_CONFIG["max_distance_px"]:
        return None, f"Gate 3 failed: distance gap {distance:.0f}px is too far"

    speed = distance / max(time_gap, 0.001)
    if speed > STITCH_CONFIG["max_speed_px_sec"]:
        return None, f"Gate 4 failed: required speed {speed:.0f}px/s is too fast"

    angle = _angle_diff_deg(a["exit_dir"], b["entry_dir"])
    if angle is not None and angle > STITCH_CONFIG["max_angle_deg"]:
        return None, f"Gate 5 failed: direction changed by {angle:.0f} degrees"

    for other_id, other in profiles.items():
        if other_id in {a["segment_id"], b["segment_id"]}:
            continue
        pos = _track_position_at(other, b["start"])
        if pos and _point_distance(pos, b["first_xy"]) <= STITCH_CONFIG["conflict_radius_px"]:
            return None, f"Gate 6 failed: another worker was already near the re-entry point"

    distance_score = 1.0 - min(distance / STITCH_CONFIG["max_distance_px"], 1.0)
    time_score = 1.0 - min(time_gap / STITCH_CONFIG["max_time_gap_sec"], 1.0)
    if angle is None:
        angle_score = 0.65
    else:
        angle_score = 1.0 - min(angle / STITCH_CONFIG["max_angle_deg"], 1.0)
    speed_score = 1.0 - min(speed / STITCH_CONFIG["max_speed_px_sec"], 1.0)

    confidence = (
        0.35 * distance_score +
        0.25 * time_score +
        0.25 * angle_score +
        0.15 * speed_score
    )
    return {
        "from": a["segment_id"],
        "to": b["segment_id"],
        "confidence": confidence,
        "time_gap": time_gap,
        "distance_px": distance,
        "speed_px_sec": speed,
        "angle_deg": angle,
    }, "passed"


def _generate_stitch_candidates(profiles):
    candidates = []
    rejected = []
    ordered = sorted(profiles.values(), key=lambda p: (p["start"], p["end"]))
    for a in ordered:
        for b in ordered:
            if b["start"] <= a["end"]:
                continue
            loose_time = b["start"] - a["end"]
            if loose_time > STITCH_CONFIG["max_time_gap_sec"] * 2:
                continue
            loose_dist = _point_distance(a["last_xy"], b["first_xy"])
            if loose_dist > STITCH_CONFIG["max_distance_px"] * 2:
                continue
            candidate, reason = _evaluate_stitch_pair(a, b, profiles)
            if candidate:
                candidates.append(candidate)
            else:
                rejected.append({
                    "from": a["segment_id"],
                    "to": b["segment_id"],
                    "reason": reason,
                })
    return candidates, rejected


def _exclusive_stitch_assignment(candidates):
    by_from = {}
    for c in candidates:
        by_from.setdefault(c["from"], []).append(c)
    rows = sorted(by_from, key=lambda sid: max(c["confidence"] for c in by_from[sid]), reverse=True)
    for sid in rows:
        by_from[sid].sort(key=lambda c: c["confidence"], reverse=True)

    best_score = -1.0
    best_pairs = []

    def search(idx, used_to, chosen, score):
        nonlocal best_score, best_pairs
        if idx >= len(rows):
            if score > best_score:
                best_score = score
                best_pairs = list(chosen)
            return
        source = rows[idx]
        search(idx + 1, used_to, chosen, score)
        for cand in by_from[source]:
            if cand["to"] in used_to:
                continue
            used_to.add(cand["to"])
            chosen.append(cand)
            search(idx + 1, used_to, chosen, score + cand["confidence"])
            chosen.pop()
            used_to.remove(cand["to"])

    if len(rows) <= 24:
        search(0, set(), [], 0.0)
        return best_pairs

    chosen = []
    used_from = set()
    used_to = set()
    for cand in sorted(candidates, key=lambda c: c["confidence"], reverse=True):
        if cand["from"] in used_from or cand["to"] in used_to:
            continue
        chosen.append(cand)
        used_from.add(cand["from"])
        used_to.add(cand["to"])
    return chosen


def _build_consolidated_workers(segments):
    profiles = _profile_segments(segments)
    candidates, rejected = _generate_stitch_candidates(profiles)
    assigned = _exclusive_stitch_assignment(candidates)

    parent = {sid: sid for sid in profiles}
    merge_by_child = {}
    for pair in assigned:
        parent[pair["to"]] = pair["from"]
        merge_by_child[pair["to"]] = pair

    def root_of(sid):
        seen = set()
        while parent.get(sid, sid) != sid and sid not in seen:
            seen.add(sid)
            sid = parent[sid]
        return sid

    groups = {}
    for sid in profiles:
        groups.setdefault(root_of(sid), []).append(sid)

    workers = []
    for idx, (root, ids) in enumerate(sorted(groups.items(), key=lambda item: min(profiles[s]["start"] for s in item[1])), 1):
        ids = sorted(ids, key=lambda sid: profiles[sid]["start"])
        merges = [merge_by_child[sid] for sid in ids if sid in merge_by_child]
        min_conf = min([m["confidence"] for m in merges], default=1.0)
        if min_conf >= 0.75:
            tier = "High"
        elif min_conf >= 0.50:
            tier = "Medium"
        else:
            tier = "Low"
        total_distance = sum(profiles[sid]["path_distance"] for sid in ids)
        start = min(profiles[sid]["start"] for sid in ids)
        end = max(profiles[sid]["end"] for sid in ids)
        all_points = [pt for sid in ids for pt in profiles[sid]["points"]]
        all_points.sort(key=lambda p: p[0])
        workers.append({
            "worker_id": f"worker_{idx}",
            "root_segment": root,
            "segments": ids,
            "start": start,
            "end": end,
            "total_distance": total_distance,
            "points": all_points,
            "confidence": min_conf,
            "confidence_tier": tier,
            "merge_count": len(merges),
            "merges": merges,
        })
    return workers, assigned, rejected, profiles


# ─────────────────────────────────────────────────
# Map rendering internals
# ─────────────────────────────────────────────────

def _norm(val, val_max, out_max):
    return max(0, min(int((val / val_max) * (out_max - 1)), out_max - 1))


def _build_zone_grid(zones, frame_w, frame_h):
    grid = [[" "] * GRID_W for _ in range(GRID_H)]
    for z in zones:
        x1, y1, x2, y2 = z["box"]
        gx1 = _norm(x1, frame_w, GRID_W)
        gx2 = _norm(x2, frame_w, GRID_W)
        gy1 = _norm(y1, frame_h, GRID_H)
        gy2 = _norm(y2, frame_h, GRID_H)
        label = z["name"].split()[0][:10]
        # borders
        for gx in range(max(0, gx1), min(GRID_W, gx2 + 1)):
            if gy1 < GRID_H: grid[gy1][gx] = "─"
            if gy2 < GRID_H: grid[gy2][gx] = "─"
        for gy in range(max(0, gy1), min(GRID_H, gy2 + 1)):
            if gx1 < GRID_W: grid[gy][gx1] = "│"
            if gx2 < GRID_W: grid[gy][gx2] = "│"
        for gx, gy in [(gx1, gy1), (gx2, gy1), (gx1, gy2), (gx2, gy2)]:
            if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                grid[gy][gx] = "+"
        # label inside box
        if gy1 + 1 < min(gy2, GRID_H):
            for i, ch in enumerate(label):
                c = gx1 + 2 + i
                if gx1 < c < min(gx2, GRID_W):
                    grid[gy1 + 1][c] = ch
    return grid


def _grid_to_str(grid, header):
    border = "+" + "─" * GRID_W + "+"
    lines = [header, border]
    for row in grid:
        lines.append("│" + "".join(row) + "│")
    lines.append(border)
    return "\n".join(lines)


def _snapshot_map(segments, zones, frame_w, frame_h, t_start, t_end):
    grid = _build_zone_grid(zones, frame_w, frame_h)
    legend = []
    for idx, seg in enumerate(segments):
        symbol = SYMBOLS[idx % len(SYMBOLS)]
        legend.append(f"  {symbol} = {seg['segment_id']}")
        pts = seg["points"]
        for i, (t, x, y) in enumerate(pts):
            gx = _norm(x, frame_w, GRID_W)
            gy = _norm(y, frame_h, GRID_H)
            if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                if i == len(pts) - 1:
                    grid[gy][gx] = symbol
                elif grid[gy][gx] == " ":
                    grid[gy][gx] = "·"

    header = f"MAP  {t_start:.1f}s → {t_end:.1f}s  [snapshot]"
    out = _grid_to_str(grid, header)
    if legend:
        out += "\nLEGEND (tracking segments — NOT necessarily unique individuals):\n"
        out += "\n".join(legend)
    else:
        out += "\n  (no tracking data in this window)"
    out += "\nNOTE: Boxes = static objects/areas on the floor. Dots = foot path trail."
    return out


def _heatmap(segments, zones, frame_w, frame_h, t_start, t_end):
    hits = [[0] * GRID_W for _ in range(GRID_H)]
    for seg in segments:
        for _, x, y in seg["points"]:
            gx = _norm(x, frame_w, GRID_W)
            gy = _norm(y, frame_h, GRID_H)
            if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                hits[gy][gx] += 1

    max_h = max(max(r) for r in hits) or 1
    DENSITY = [" ", "░", "▒", "▓", "█"]
    zone_grid = _build_zone_grid(zones, frame_w, frame_h)
    grid = []
    for gy in range(GRID_H):
        row = []
        for gx in range(GRID_W):
            if zone_grid[gy][gx] != " ":
                row.append(zone_grid[gy][gx])
            elif hits[gy][gx] > 0:
                lvl = int((hits[gy][gx] / max_h) * (len(DENSITY) - 1))
                row.append(DENSITY[lvl])
            else:
                row.append(" ")
        grid.append(row)

    header = f"HEATMAP  {t_start:.1f}s → {t_end:.1f}s  [aggregated foot-traffic density]"
    out = _grid_to_str(grid, header)
    out += "\nDensity key: ░ light  ▒ medium  ▓ heavy  █ peak"
    out += "\nNOTE: Density shows where foot points were most detected. Boxes = static objects."
    return out


# ─────────────────────────────────────────────────
# TOOL 1: get_map
# ─────────────────────────────────────────────────

def get_map(camera_name: str, t_start_sec: float, t_end_sec: float) -> str:
    """
    Returns an ASCII floor map for the given time window.

    Auto mode:
      duration ≤ 60s  → snapshot map (individual segment symbols + trail dots)
      duration ≤ 300s → map series (one snapshot per 20s bucket)
      duration > 300s → heatmap (density aggregation, no individual symbols)

    The map shows:
      - Static object/area boxes (zones) with short labels
      - Foot position trails and last-known positions per segment
    """
    duration = t_end_sec - t_start_sec
    if duration <= 0:
        return "ERROR: t_end_sec must be greater than t_start_sec."

    conn = _db()
    cur = conn.cursor()
    camera_id = _get_camera(camera_name, cur)
    if not camera_id:
        conn.close()
        return f"ERROR: Camera '{camera_name}' not found."

    zones = _load_zones(camera_id, cur)
    frame_w, frame_h = _infer_frame_dims(zones)

    if duration <= 60:
        segments = _load_segments(camera_id, t_start_sec, t_end_sec, cur)
        conn.close()
        return _snapshot_map(segments, zones, frame_w, frame_h, t_start_sec, t_end_sec)

    elif duration <= 300:
        parts = []
        t = t_start_sec
        while t < t_end_sec:
            t_b = min(t + 20, t_end_sec)
            segs = _load_segments(camera_id, t, t_b, cur)
            parts.append(_snapshot_map(segs, zones, frame_w, frame_h, t, t_b))
            t = t_b
        conn.close()
        return "\n\n".join(parts)

    else:
        segments = _load_segments(camera_id, t_start_sec, t_end_sec, cur)
        conn.close()
        return _heatmap(segments, zones, frame_w, frame_h, t_start_sec, t_end_sec)


# ─────────────────────────────────────────────────
# TOOL 2: get_camera_info
# ─────────────────────────────────────────────────

def get_camera_info(camera_name: str = None) -> str:
    """
    If camera_name is None: lists all cameras with zone count and last activity.
    If camera_name is given: returns details for that specific camera.
    """
    conn = _db()
    cur = conn.cursor()

    if camera_name:
        row = cur.execute("SELECT * FROM cameras WHERE name=?", (camera_name,)).fetchone()
        if not row:
            conn.close()
            return f"Camera '{camera_name}' not found."
        cam_id = row["id"]
        zone_count = cur.execute(
            "SELECT COUNT(*) FROM camera_zones WHERE camera_id=?", (cam_id,)
        ).fetchone()[0]
        run_count = cur.execute(
            "SELECT COUNT(*) FROM floor_data WHERE camera_id=?", (cam_id,)
        ).fetchone()[0]
        last = cur.execute(
            "SELECT created_at, start_time_sec, end_time_sec FROM floor_data WHERE camera_id=? ORDER BY created_at DESC LIMIT 1",
            (cam_id,)
        ).fetchone()
        conn.close()
        lines = [
            f"Camera: {row['name']}",
            f"  Source video: {row['source_path']}",
            f"  Mapped static objects/areas: {zone_count}",
            f"  Tracking sessions in DB: {run_count}",
        ]
        if last:
            lines.append(f"  Most recent session: recorded at {last['created_at']}")
            lines.append(f"  Session video span: {last['start_time_sec']:.1f}s → {last['end_time_sec']:.1f}s")
        return "\n".join(lines)

    else:
        rows = cur.execute("SELECT id, name, source_path, created_at FROM cameras").fetchall()
        if not rows:
            conn.close()
            return "No cameras found in database."
        lines = ["Available cameras:"]
        for r in rows:
            zc = cur.execute(
                "SELECT COUNT(*) FROM camera_zones WHERE camera_id=?", (r["id"],)
            ).fetchone()[0]
            lines.append(f"  - {r['name']}  |  Mapped objects: {zc}  |  Added: {r['created_at']}")
        conn.close()
        return "\n".join(lines)


# ─────────────────────────────────────────────────
# TOOL 3: get_zones_info
# ─────────────────────────────────────────────────

def get_zones_info(camera_name: str) -> str:
    """
    Returns all mapped static objects/areas for a camera with their
    descriptions and approximate position on the camera frame.

    These zones represent PHYSICAL OBJECTS — machines, workstations,
    panels, aisles, etc. People move around and interact with them
    from the outside, not inside them (except open floor areas).
    """
    conn = _db()
    cur = conn.cursor()
    camera_id = _get_camera(camera_name, cur)
    if not camera_id:
        conn.close()
        return f"Camera '{camera_name}' not found."

    zones = _load_zones(camera_id, cur)
    conn.close()

    if not zones:
        return f"No zones mapped for camera '{camera_name}' yet."

    frame_w, frame_h = _infer_frame_dims(zones)
    lines = [
        f"Static objects/areas mapped on camera '{camera_name}':",
        f"  Frame estimated size: {int(frame_w)} × {int(frame_h)} px",
        f"",
        f"  IMPORTANT: These are OBJECTS and AREAS on the floor.",
        f"  A person detected near a zone is likely interacting with it from outside,",
        f"  not standing inside it (exception: open walkway/aisle areas).",
        f"",
    ]
    for i, z in enumerate(zones, 1):
        x1, y1, x2, y2 = z["box"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        region = _frame_region(cx, cy, frame_w, frame_h)
        lines.append(f"  {i}. {z['name']}")
        lines.append(f"     Frame position: {region}")
        if z["description"]:
            lines.append(f"     Description: {z['description']}")
        lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────
# TOOL 4: get_people_count
# ─────────────────────────────────────────────────

def get_people_count(
    camera_name: str,
    t_start_sec: float = None,
    t_end_sec: float = None,
    zone_name: str = None,
) -> str:
    """
    Returns a timeline of how many tracking segments were concurrently active
    in 5-second buckets within the given time range.

    If zone_name is given, counts only segments whose foot positions were
    detected within 12% of frame width from that object's boundary.

    IMPORTANT:
      - Count = tracking segments, NOT unique people.
      - One physical person leaving and re-entering = multiple segments.
      - Proximity to a zone does NOT mean the person was using that zone.
    """
    conn = _db()
    cur = conn.cursor()
    camera_id = _get_camera(camera_name, cur)
    if not camera_id:
        conn.close()
        return f"Camera '{camera_name}' not found."

    zones = _load_zones(camera_id, cur)
    frame_w, frame_h = _infer_frame_dims(zones)

    t_s = t_start_sec if t_start_sec is not None else 0.0
    t_e = t_end_sec if t_end_sec is not None else 999999.0

    segments = _load_segments(camera_id, t_s, t_e, cur)
    conn.close()

    # Cap t_e to the actual max time in the segments so we don't generate millions of empty buckets
    actual_max_t = max([pt_t for seg in segments for pt_t, x, y in seg.get("points", [])], default=t_s)
    if t_end_sec is None:
        t_e = actual_max_t

    # Zone proximity filter
    target_zone = None
    if zone_name:
        for z in zones:
            if zone_name.lower() in z["name"].lower():
                target_zone = z
                break
        if not target_zone:
            return f"Zone matching '{zone_name}' not found for camera '{camera_name}'."

    threshold_px = frame_w * 0.12

    BUCKET = 5.0
    t = t_s
    timeline = []
    while t < t_e:
        t_b = t + BUCKET
        count = 0
        for seg in segments:
            pts = [(x, y) for pt_t, x, y in seg["points"] if t <= pt_t < t_b]
            if not pts:
                continue
            if target_zone:
                if any(_dist_to_box_edge(x, y, target_zone["box"]) <= threshold_px
                       for x, y in pts):
                    count += 1
            else:
                count += 1
        timeline.append((t, t_b, count))
        t = t_b

    counts = [c for _, _, c in timeline]
    peak = max(counts) if counts else 0
    peak_windows = [(ta, tb) for ta, tb, c in timeline if c == peak]

    lines = [
        f"PEOPLE COUNT — Camera: {camera_name}",
        f"  Time range: {t_s:.1f}s → {min(t_e, t + BUCKET):.1f}s",
    ]
    if target_zone:
        lines.append(
            f"  Filtered to: segments detected within 12% frame-width of '{target_zone['name']}'"
        )
    lines += [
        f"",
        f"  Peak concurrent segments: {peak}",
        f"  Peak window(s): " + ", ".join(f"{ta:.1f}s–{tb:.1f}s" for ta, tb in peak_windows[:3]),
        f"  Average: {sum(counts)/len(counts):.1f} segments per 5s window" if counts else "",
        f"",
        f"  Timeline (5s buckets):",
    ]
    for ta, tb, c in timeline:
        bar = "█" * c
        lines.append(f"    {ta:7.1f}s – {tb:6.1f}s :  {bar} ({c})")

    lines += [
        f"",
        f"NOTE: 'Segments' ≠ unique individuals.",
        f"  The same physical person re-entering the frame creates a new segment.",
        f"  Proximity to a zone does not mean they were using it.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────
# TOOL 5: get_day_summary
# ─────────────────────────────────────────────────

def _get_day_summary_legacy(camera_name: str, date: str = None) -> str:
    """
    Returns a rich summary of all floor activity for a given date.

    date format: 'YYYY-MM-DD'
    If date is None, uses the most recent date with data.

    Date is derived from 'created_at' (when tracker was run).
    This is an approximation — it reflects when the tracker processed the
    footage, not necessarily when the footage was recorded.

    Summary includes:
      - Session span and duration
      - Segment count and total tracked points
      - Concurrent activity timeline
      - Floor region distribution (where foot points were detected)
      - Peak and idle periods
    No zone assignments or behavioral conclusions are made.
    """
    conn = _db()
    cur = conn.cursor()
    camera_id = _get_camera(camera_name, cur)
    if not camera_id:
        conn.close()
        return f"Camera '{camera_name}' not found."

    if not date:
        row = cur.execute(
            "SELECT substr(created_at,1,10) as d FROM floor_data WHERE camera_id=? ORDER BY created_at DESC LIMIT 1",
            (camera_id,)
        ).fetchone()
        if not row:
            conn.close()
            return f"No tracking data found for camera '{camera_name}'."
        date = row["d"]

    rows = cur.execute(
        "SELECT * FROM floor_data WHERE camera_id=? AND created_at LIKE ?",
        (camera_id, f"{date}%")
    ).fetchall()
    zones = _load_zones(camera_id, cur)
    conn.close()

    if not rows:
        return f"No tracking data for camera '{camera_name}' on {date}."

    frame_w, frame_h = _infer_frame_dims(zones)

    # Collect all segments
    all_segs = []
    total_pts = 0
    for row in rows:
        data = json.loads(row["person_path_json"])
        for sub in data.get("subjects", []):
            pts = sub.get("path_points", [])
            if pts:
                all_segs.append({
                    "start": sub["start_time_sec"],
                    "end": sub["end_time_sec"],
                    "points": pts,
                    "confidence": sub.get("confidence", 0.0),
                })
                total_pts += len(pts)

    if not all_segs:
        return f"No segments found for camera '{camera_name}' on {date}."

    session_start = min(s["start"] for s in all_segs)
    session_end = max(s["end"] for s in all_segs)
    duration = session_end - session_start
    avg_conf = sum(s["confidence"] for s in all_segs) / len(all_segs)

    # Frame region distribution
    region_hits: dict = {}
    for seg in all_segs:
        for pt in seg["points"]:
            x, y = pt["foot"]
            r = _frame_region(x, y, frame_w, frame_h)
            region_hits[r] = region_hits.get(r, 0) + 1
    top_regions = sorted(region_hits.items(), key=lambda x: -x[1])

    # Peak concurrent (5s buckets)
    BUCKET = 5.0
    t = session_start
    peak_count = 0
    peak_t = session_start
    idle_gap = 0.0
    prev_end = session_start

    bucket_counts = []
    while t < session_end:
        t_b = t + BUCKET
        active = sum(1 for s in all_segs if s["start"] <= t_b and s["end"] >= t)
        bucket_counts.append((t, t_b, active))
        if active > peak_count:
            peak_count = active
            peak_t = t
        if active == 0:
            idle_gap += BUCKET
        t = t_b

    lines = [
        "═" * 65,
        f"  DAY SUMMARY",
        f"  Camera : {camera_name}",
        f"  Date   : {date}  (based on tracker run timestamp)",
        "═" * 65,
        "",
        "OVERVIEW",
        f"  Video span tracked   : {session_start:.1f}s → {session_end:.1f}s  ({duration:.0f}s total)",
        f"  Tracking segments    : {len(all_segs)}",
        f"  Total path points    : {total_pts}",
        f"  Avg detection conf.  : {avg_conf:.2f}",
        f"  Mapped static objects: {len(zones)}",
        "",
        "ACTIVITY PATTERN",
        f"  Peak concurrent segments : {peak_count}  (around t={peak_t:.1f}s)",
        f"  Estimated idle time      : {idle_gap:.0f}s with zero detected movement",
        "",
        "FLOOR REGION DISTRIBUTION",
        "  Camera frame divided into 9 regions (LEFT/CENTER/RIGHT × UPPER/MIDDLE/LOWER)",
        "  Showing where foot detections were concentrated:",
        "",
    ]

    for region, count in top_regions:
        pct = (count / total_pts) * 100
        bar = "█" * max(1, int(pct / 3))
        lines.append(f"    {region:<22} {bar}  {pct:.1f}%")

    lines += [
        "",
        "CONCURRENT ACTIVITY TIMELINE (5s buckets)",
    ]
    for ta, tb, c in bucket_counts:
        bar = "█" * c if c > 0 else "·"
        lines.append(f"    {ta:7.1f}s – {tb:6.1f}s :  {bar} ({c})")

    lines += [
        "",
        "IMPORTANT NOTES",
        "  · Segments ≠ unique people. One person re-entering = new segment.",
        "  · No zone assignments are made in this summary.",
        "  · Use get_map() to see spatial positions visually.",
        "  · Timestamps are relative to video start, not wall-clock time.",
        "═" * 65,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────
# TOOL 6: get_video_frame
# ─────────────────────────────────────────────────

def get_day_summary(camera_name: str, date: str = None) -> str:
    """
    Returns neutral evidence for a factory-floor daily/session summary.

    This tool gives the agent useful map-aware facts without concluding that
    a worker belongs to a zone or was operating an object.
    """
    conn = _db()
    cur = conn.cursor()
    camera_id = _get_camera(camera_name, cur)
    if not camera_id:
        conn.close()
        return f"Camera '{camera_name}' not found."

    if not date:
        row = cur.execute(
            "SELECT substr(created_at,1,10) as d FROM floor_data WHERE camera_id=? ORDER BY created_at DESC LIMIT 1",
            (camera_id,)
        ).fetchone()
        if not row:
            conn.close()
            return f"No tracking data found for camera '{camera_name}'."
        date = row["d"]

    rows = cur.execute(
        "SELECT * FROM floor_data WHERE camera_id=? AND created_at LIKE ?",
        (camera_id, f"{date}%")
    ).fetchall()
    zones = _load_zones(camera_id, cur)
    conn.close()

    if not rows:
        return f"No tracking data for camera '{camera_name}' on {date}."

    frame_w, frame_h = _infer_frame_dims(zones)

    all_tracks = []
    total_readings = 0
    for row in rows:
        data = json.loads(row["person_path_json"])
        for sub in data.get("subjects", []):
            pts = sub.get("path_points", [])
            if pts:
                all_tracks.append({
                    "start": sub["start_time_sec"],
                    "end": sub["end_time_sec"],
                    "points": pts,
                })
                total_readings += len(pts)

    if not all_tracks:
        return f"No worker movement records found for camera '{camera_name}' on {date}."

    session_start = min(s["start"] for s in all_tracks)
    session_end = max(s["end"] for s in all_tracks)
    duration = session_end - session_start

    def video_mark(t_sec: float) -> str:
        total = max(0, int(round(t_sec)))
        return f"{total // 60:02d}:{total % 60:02d}"

    def friendly_region(region: str) -> str:
        col, row = region.split("-")
        col_words = {"LEFT": "left side", "CENTER": "middle", "RIGHT": "right side"}
        row_words = {"UPPER": "far end", "MIDDLE": "middle stretch", "LOWER": "near end"}
        return f"{row_words.get(row, row.lower())} of the {col_words.get(col, col.lower())}"

    nearby_readings = {z["name"]: 0 for z in zones}
    nearby_tracks = {z["name"]: 0 for z in zones}
    proximity_words = {z["name"]: set() for z in zones}
    away_region_hits = {}

    for track in all_tracks:
        track_near = set()
        for pt in track["points"]:
            x, y = pt["foot"]
            near_any_mapped_item = False
            for z in zones:
                label = _proximity_label(_dist_to_box_edge(x, y, z["box"]), frame_w)
                if label:
                    nearby_readings[z["name"]] += 1
                    proximity_words[z["name"]].add(label)
                    track_near.add(z["name"])
                    near_any_mapped_item = True
            if not near_any_mapped_item:
                region = _frame_region(x, y, frame_w, frame_h)
                away_region_hits[region] = away_region_hits.get(region, 0) + 1
        for name in track_near:
            nearby_tracks[name] += 1

    object_evidence = []
    for z in zones:
        name = z["name"]
        if nearby_readings[name] > 0:
            labels = ", ".join(sorted(proximity_words[name]))
            object_evidence.append((nearby_readings[name], nearby_tracks[name], name, labels))
    object_evidence.sort(reverse=True)

    bucket = 5.0
    bucket_rows = []
    t = session_start
    while t < session_end:
        t_b = min(t + bucket, session_end)
        active_tracks = 0
        nearby_names = set()
        for track in all_tracks:
            pts_in_window = [
                pt for pt in track["points"]
                if t <= pt["t"] < t_b
            ]
            if not pts_in_window:
                continue
            active_tracks += 1
            for pt in pts_in_window:
                x, y = pt["foot"]
                for z in zones:
                    if _proximity_label(_dist_to_box_edge(x, y, z["box"]), frame_w):
                        nearby_names.add(z["name"])
        bucket_rows.append((t, t_b, active_tracks, sorted(nearby_names)))
        t = t_b

    busiest = sorted(bucket_rows, key=lambda r: (-r[2], r[0]))[:5]
    quietest = sorted(bucket_rows, key=lambda r: (r[2], r[0]))[:5]
    away_regions = sorted(away_region_hits.items(), key=lambda x: -x[1])[:4]

    lines = [
        "=" * 72,
        "DAY SUMMARY EVIDENCE - NO ZONE ASSIGNMENT",
        f"Camera: {camera_name}",
        f"Date: {date} (based on tracker run timestamp)",
        "=" * 72,
        "",
        "HOW TO READ THIS",
        "  This shows where workers were seen in the camera view.",
        "  It can show which floor areas were busy, but it cannot prove who",
        "  the worker was or exactly what job they were doing.",
        "  Use visual frames when the final answer needs behavior or action details.",
        "",
        "VIDEO COVERED",
        f"  From video mark {video_mark(session_start)} to {video_mark(session_end)}",
        f"  Approximate length: {max(1, round(duration / 60, 1))} minute(s)",
        f"  Worker visits or movements noticed by the system: {len(all_tracks)}",
        "  The same person may be counted more than once if they left and came back.",
        "",
        "MAPPED FLOOR OBJECTS / AREAS FOR THIS CAMERA",
    ]

    if zones:
        for z in zones:
            clean_desc = z["description"]
            clean_desc = clean_desc.replace("foot position alone", "where a worker was seen")
            clean_desc = clean_desc.replace("Nearby foot positions", "Workers seen nearby")
            clean_desc = clean_desc.replace("nearby feet alone", "being seen nearby")
            clean_desc = clean_desc.replace("Proximity", "Being nearby")
            clean_desc = clean_desc.replace("proximity", "being nearby")
            clean_desc = clean_desc.replace("being seen nearby do not", "being seen nearby does not")
            desc = f" - {clean_desc}" if clean_desc else ""
            lines.append(f"  - {z['name']}{desc}")
    else:
        lines.append("  No mapped floor objects/areas found for this camera.")

    lines += [
        "",
        "WHERE WORKERS WERE SEEN MOST",
        "  Ranked by how often and how long workers were seen around each area.",
    ]
    if object_evidence:
        for rank, (_readings, _tracks, name, _labels) in enumerate(object_evidence[:8], 1):
            lines.append(
                f"  - #{rank} {name}: workers were seen around this area often"
            )
    else:
        lines.append("  Workers were not seen close to the mapped floor objects/areas.")

    lines += [
        "",
        "BUSIEST VIDEO MARKS",
    ]
    for ta, tb, active, nearby_names in busiest:
        near = ", ".join(nearby_names[:4]) if nearby_names else "no mapped object close enough"
        lines.append(
            f"  - {video_mark(ta)} to {video_mark(tb)}: about {active} worker(s) visible at the same time; around: {near}"
        )

    lines += [
        "",
        "QUIETER VIDEO MARKS",
    ]
    for ta, tb, active, nearby_names in quietest:
        near = ", ".join(nearby_names[:4]) if nearby_names else "no mapped object close enough"
        lines.append(
            f"  - {video_mark(ta)} to {video_mark(tb)}: about {active} worker(s) visible at the same time; around: {near}"
        )

    if away_regions:
        lines += [
            "",
            "WORKERS SEEN AWAY FROM MAPPED OBJECTS / AREAS",
        ]
        for region, count in away_regions:
            pct = (count / total_readings) * 100
            lines.append(f"  - {friendly_region(region)}: about {pct:.1f}% of worker sightings")

    lines += [
        "",
        "LIMITS FOR THE AGENT",
        "  - Do not say a worker was inside or assigned to a mapped object/area.",
        "  - Do not say what a worker was doing unless visual evidence is checked.",
        "  - Worker visits/movements are not unique people; re-entry can be counted again.",
        "  - Video marks are relative to the footage, not factory clock time.",
        "=" * 72,
    ]
    return "\n".join(lines)


def get_worker_movement_summary(
    camera_name: str,
    t_start_sec: float = None,
    t_end_sec: float = None,
    focus: str = "movement",
) -> str:
    """
    Enrichment-layer worker movement summary.

    This does not modify raw tracker data. It reads fragmented tracking segments,
    stitches likely continuations into consolidated worker records, and reports
    movement/stillness metrics with confidence.
    """
    conn = _db()
    cur = conn.cursor()
    camera_id = _get_camera(camera_name, cur)
    if not camera_id:
        conn.close()
        return f"Camera '{camera_name}' not found."

    zones = _load_zones(camera_id, cur)
    frame_w, frame_h = _infer_frame_dims(zones)
    t_s = t_start_sec if t_start_sec is not None else 0.0
    t_e = t_end_sec if t_end_sec is not None else 999999.0
    segments = _load_segments(camera_id, t_s, t_e, cur)
    conn.close()

    if not segments:
        return f"No worker movement data for camera '{camera_name}' in the requested time range."

    actual_start = min(seg["start"] for seg in segments)
    actual_end = max(seg["end"] for seg in segments)
    if t_end_sec is None:
        t_e = actual_end

    workers, assigned, rejected, _profiles = _build_consolidated_workers(segments)
    focus_norm = (focus or "movement").lower()

    def main_area(worker):
        hits = {}
        for _, x, y in worker["points"]:
            region, nearby = _translate_point(x, y, zones, frame_w, frame_h)
            if nearby:
                name = nearby[0].split(" (", 1)[0]
            else:
                name = _friendly_region(region)
            hits[name] = hits.get(name, 0) + 1
        if not hits:
            return "an unmapped part of the camera view"
        return max(hits.items(), key=lambda item: item[1])[0]

    def stationary_score(worker):
        duration = max(worker["end"] - worker["start"], 0.1)
        return worker["total_distance"] / duration

    def sample_times(worker):
        start = worker["start"]
        end = worker["end"]
        mid = start + ((end - start) / 2.0)
        values = []
        for t in (start, mid, end):
            if not values or abs(t - values[-1]) >= 1.0:
                values.append(t)
        return values

    if "still" in focus_norm or "stand" in focus_norm or "stationary" in focus_norm or "idle" in focus_norm:
        ranked = sorted(workers, key=lambda w: (stationary_score(w), -len(w["points"])))
        ranking_label = "least movement / most stationary"
    else:
        ranked = sorted(workers, key=lambda w: w["total_distance"], reverse=True)
        ranking_label = "most movement"

    lines = [
        "=" * 72,
        "CONSOLIDATED WORKER MOVEMENT SUMMARY",
        f"Camera: {camera_name}",
        f"Time window: {t_s:.1f}s to {t_e:.1f}s",
        "=" * 72,
        "",
        "HOW THIS WAS BUILT",
        "  Raw tracker segments were not changed.",
        "  The enrichment layer linked likely broken paths when time, distance,",
        "  speed, direction, and nearby-worker checks all passed.",
        "  Consolidated worker IDs below are analysis labels, not employee names.",
        "",
        f"RANKING: {ranking_label}",
    ]

    for idx, worker in enumerate(ranked[:8], 1):
        area = main_area(worker)
        duration = max(worker["end"] - worker["start"], 0.0)
        lines.append(
            f"  #{idx} {worker['worker_id']}: around {area}; "
            f"seen from {worker['start']:.1f}s to {worker['end']:.1f}s "
            f"({duration:.1f}s); movement distance ~{worker['total_distance']:.0f}px; "
            f"confidence {worker['confidence_tier']}"
        )
        lines.append(
            "      visual-check timestamps: "
            + ", ".join(f"{t:.1f}s" for t in sample_times(worker))
        )
        if worker["merge_count"]:
            lines.append(
                f"      stitched from {len(worker['segments'])} broken camera path(s); "
                f"weakest merge confidence {worker['confidence']:.2f}"
            )

    lines += [
        "",
        "MERGE AUDIT",
    ]
    if assigned:
        for pair in sorted(assigned, key=lambda p: (p["from"], p["to"]))[:12]:
            lines.append(
                f"  - Combined one broken path into another: gap {pair['time_gap']:.2f}s, "
                f"distance {pair['distance_px']:.0f}px, speed {pair['speed_px_sec']:.0f}px/s, "
                f"confidence {pair['confidence']:.2f}"
            )
    else:
        lines.append("  No broken paths were confidently stitched in this window.")

    if rejected:
        lines += [
            "",
            "REJECTED MERGE EXAMPLES",
        ]
        for item in rejected[:6]:
            lines.append(f"  - Not combined: {item['reason']}")

    lines += [
        "",
        "IMPORTANT FOR THE AI",
        "  Use these consolidated worker records for movement/stillness questions.",
        "  Do not expose raw segment names in owner-facing answers.",
        "  For 'which worker' answers, call get_visual_grid at representative",
        "  timestamps and describe the worker by visible clothing/location.",
        "  If confidence is Medium or Low, say the relevant time window should be reviewed.",
        "=" * 72,
    ]
    return "\n".join(lines)


def get_video_frame(camera_name: str, t_sec: float) -> str:
    """
    Extracts a JPEG frame from the source video at the given timestamp (seconds).
    Returns the saved image file path so the bot or user can view it.

    Useful when a visual confirmation of the floor state at a specific moment is needed.
    """
    try:
        import cv2
    except ImportError:
        return "ERROR: opencv-python not available. Install it with: pip install opencv-python"

    conn = _db()
    cur = conn.cursor()
    camera_id = _get_camera(camera_name, cur)
    if not camera_id:
        conn.close()
        return f"Camera '{camera_name}' not found."

    meta_row = cur.execute(
        "SELECT metadata_json FROM floor_data WHERE camera_id=? ORDER BY created_at DESC LIMIT 1",
        (camera_id,)
    ).fetchone()
    conn.close()

    if not meta_row:
        return f"No tracking runs found for camera '{camera_name}'. Cannot determine video path."

    meta = json.loads(meta_row["metadata_json"])
    video_path = meta.get("video_path", "")
    fps = meta.get("fps", 20.0)

    if not video_path or not Path(video_path).exists():
        return (
            f"Video file not accessible: {video_path}\n"
            f"The video path is stored from when the tracker was run. "
            f"Ensure the file still exists at that location."
        )

    cap = cv2.VideoCapture(video_path)
    frame_idx = int(t_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return f"Could not extract frame at t={t_sec}s (frame index {frame_idx})."

    out_dir = BASE_DIR / "frame_exports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{camera_name}_t{int(t_sec * 1000):09d}ms.jpg"
    cv2.imwrite(str(out_path), frame)

    return (
        f"Frame saved: {out_path}\n"
        f"  Camera   : {camera_name}\n"
        f"  Timestamp: {t_sec}s  (frame #{frame_idx} @ {fps:.1f}fps)\n"
        f"  Video    : {video_path}"
    )


# ─────────────────────────────────────────────────
# TOOL 7: get_activity_table
# ─────────────────────────────────────────────────

def get_activity_table(camera_name: str, t_start_sec: float, t_end_sec: float) -> str:
    """
    Returns a flat, translated table of every tracked foot position
    in the given time range.

    Each row contains:
      - segment     : internal tracking episode ID (not a unique person)
      - t_sec       : time in video (seconds from start)
      - frame_region: where on the camera frame (3×3 grid label)
      - nearby_objects: static objects within 12% of frame width from foot

    STRICT POLICY:
      - No zone assignments are made.
      - No conclusions about what the person was doing.
      - Segment IDs are only internal clues for analysis. They must not be
        shown to the owner as worker identities.
      - 'nearby_objects' lists objects that were spatially close — the AI
        should decide what that means based on context.
      - Proximity labels: RIGHT NEXT TO (0-2%) | VERY CLOSE (2-6%) | NEARBY (6-12%)
      - Objects beyond 12% of frame width are not listed.
    """
    conn = _db()
    cur = conn.cursor()
    camera_id = _get_camera(camera_name, cur)
    if not camera_id:
        conn.close()
        return f"Camera '{camera_name}' not found."

    zones = _load_zones(camera_id, cur)
    frame_w, frame_h = _infer_frame_dims(zones)
    segments = _load_segments(camera_id, t_start_sec, t_end_sec, cur)
    conn.close()

    if not segments:
        return (
            f"No tracking data for camera '{camera_name}' "
            f"between {t_start_sec}s and {t_end_sec}s."
        )

    lines = [
        f"ACTIVITY TABLE  —  Camera: {camera_name}",
        f"  Time range  : {t_start_sec:.1f}s → {t_end_sec:.1f}s",
        f"  Frame size  : ~{int(frame_w)} × {int(frame_h)} px",
        f"",
        f"COLUMN DEFINITIONS:",
        f"  IMPORTANT   - segment names are internal clues, not worker identities",
        f"                do not show them to the owner as person names",
        f"  segment      — tracking episode (NOT a unique person; same person may = multiple segments)",
        f"  t_sec        — seconds from start of video",
        f"  frame_region — 3×3 grid position: [LEFT|CENTER|RIGHT]-[UPPER|MIDDLE|LOWER]",
        f"  nearby_objects — mapped static objects within 12% of frame width from foot position",
        f"                   Labels: RIGHT NEXT TO (0–2%) | VERY CLOSE (2–6%) | NEARBY (6–12%)",
        f"                   Empty = no mapped object within that range",
        f"",
        f"NO ZONE ASSIGNMENTS ARE MADE. These are spatial facts only.",
        f"For person-level answers, use video frames and describe visible clothing/location.",
        f"",
        f"{'segment':<28} {'t_sec':>6}  {'frame_region':<22} nearby_objects",
        f"{'─'*28} {'─'*6}  {'─'*22} {'─'*45}",
    ]

    total_rows = 0
    for seg in segments:
        for t, x, y in seg["points"]:
            region, nearby = _translate_point(x, y, zones, frame_w, frame_h)
            nearby_str = " | ".join(nearby) if nearby else "—"
            lines.append(
                f"{seg['segment_id']:<28} {t:>6.2f}  {region:<22} {nearby_str}"
            )
            total_rows += 1

    lines += [
        f"",
        f"Total rows: {total_rows}",
        f"Use get_map() alongside this table for visual spatial context.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────
# TOOL 8: get_visual_grid
# ─────────────────────────────────────────────────

def get_visual_grid(
    camera_name: str,
    timestamps: list,
    question: str,
) -> str:
    """
    Extracts video frames at the given timestamps, assembles them into a
    single labeled grid image, then uses GPT-4o Vision to visually analyze
    the grid and answer the question.

    Use this whenever the question requires seeing what is actually happening
    in the footage — posture, interactions, behaviour, equipment, groupings, etc.
    Position data alone cannot answer these; only looking at the frames can.

    Each cell in the grid is labeled with its timestamp so the analysis can
    reference specific moments.

    timestamps: list of floats (seconds from video start), up to 9 values.
    question:   the specific visual question to answer from the frames.
    """
    try:
        import cv2
    except ImportError:
        return "ERROR: opencv-python not installed."

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return "ERROR: Pillow not installed. Run: pip install Pillow"

    try:
        import base64, io
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv(BASE_DIR / ".env")
        import os
        oai_key = os.environ.get("OPENAI_API_KEY", "")
        if not oai_key:
            return "ERROR: OPENAI_API_KEY not found in .env"
        oai = OpenAI(api_key=oai_key)
    except ImportError:
        return "ERROR: openai package not installed."

    # ── Get video path ───────────────────────────────────────────
    conn = _db()
    cur = conn.cursor()
    camera_id = _get_camera(camera_name, cur)
    if not camera_id:
        conn.close()
        return f"ERROR: Camera '{camera_name}' not found."

    meta_row = cur.execute(
        "SELECT metadata_json FROM floor_data WHERE camera_id=? ORDER BY created_at DESC LIMIT 1",
        (camera_id,)
    ).fetchone()
    conn.close()

    if not meta_row:
        return f"No tracking runs found for camera '{camera_name}'."

    meta = json.loads(meta_row["metadata_json"])
    video_path = meta.get("video_path", "")
    fps = meta.get("fps", 20.0)

    if not video_path or not Path(video_path).exists():
        return f"Video file not accessible: {video_path}"

    # ── Extract frames ───────────────────────────────────────────
    timestamps = timestamps[:9]  # max 9 cells in a 3x3 grid
    frames = []
    cap = cv2.VideoCapture(video_path)
    for t in timestamps:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((t, frame_rgb))
    cap.release()

    if not frames:
        return "Could not extract any frames from the video at the given timestamps."

    # ── Build grid image ─────────────────────────────────────────
    n = len(frames)
    cols = min(n, 3)
    rows = math.ceil(n / cols)

    cell_w, cell_h = 640, 360
    label_h = 28
    grid_w = cols * cell_w
    grid_h = rows * (cell_h + label_h)

    grid = Image.new("RGB", (grid_w, grid_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for i, (t, frame_arr) in enumerate(frames):
        col = i % cols
        row = i // cols
        x_off = col * cell_w
        y_off = row * (cell_h + label_h)

        # Resize frame to cell
        img = Image.fromarray(frame_arr)
        img = img.resize((cell_w, cell_h), Image.LANCZOS)
        grid.paste(img, (x_off, y_off))

        # Draw label bar below the frame
        label_y = y_off + cell_h
        draw.rectangle([x_off, label_y, x_off + cell_w, label_y + label_h],
                        fill=(20, 20, 20))
        draw.text((x_off + 8, label_y + 5),
                  f"t = {t:.1f}s  (frame {int(t*fps)})",
                  fill=(200, 220, 255), font=font)

    # ── Save and encode ──────────────────────────────────────────
    out_dir = BASE_DIR / "frame_exports"
    out_dir.mkdir(exist_ok=True)
    grid_path = out_dir / f"{camera_name}_grid_{int(timestamps[0]*1000)}.jpg"
    grid.save(str(grid_path), "JPEG", quality=88)

    buf = io.BytesIO()
    grid.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # ── Send to GPT-4o Vision ────────────────────────────────────
    context = (
        f"This is a grid of {n} frames from a factory floor CCTV recording "
        f"(camera: {camera_name}). Each cell is labeled with its timestamp in seconds. "
        f"The factory has molding presses, workstations, an operator aisle, and a control panel.\n\n"
        f"Question: {question}\n\n"
        f"Answer based only on what you can actually see in the images. "
        f"Be specific about which timestamp(s) support your answer. "
        f"If the question asks which worker/person, describe them by visible clothing, color, "
        f"hat/helmet, position, and nearby machine/workstation. Do not invent names or identities. "
        f"If the same person cannot be confidently followed across frames because they are blocked, "
        f"blurred, or leave the view, say that clearly. "
        f"If something is unclear or not visible, say so."
    )

    response = oai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": context},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
                ],
            }
        ],
        max_tokens=1024,
    )

    analysis = response.choices[0].message.content or "No analysis returned."

    return (
        f"Visual analysis of {n} frame(s) from camera '{camera_name}':\n"
        f"Timestamps: {', '.join(f'{t:.1f}s' for t,_ in frames)}\n"
        f"Grid saved: {grid_path}\n\n"
        f"ANALYSIS:\n{analysis}"
    )


# ─────────────────────────────────────────────────
# Quick CLI test
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    camera = sys.argv[1] if len(sys.argv) > 1 else "d23_run"
    tool = sys.argv[2] if len(sys.argv) > 2 else "summary"

    if tool == "map":
        print(get_map(camera, 0, 30))
    elif tool == "map-heat":
        print(get_map(camera, 0, 400))
    elif tool == "camera":
        print(get_camera_info())
    elif tool == "zones":
        print(get_zones_info(camera))
    elif tool == "count":
        print(get_people_count(camera, 0, 60))
    elif tool == "summary":
        print(get_day_summary(camera))
    elif tool == "table":
        print(get_activity_table(camera, 0, 30))
    elif tool == "workers":
        focus = sys.argv[3] if len(sys.argv) > 3 else "movement"
        start = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
        end = float(sys.argv[5]) if len(sys.argv) > 5 else None
        print(get_worker_movement_summary(camera, start, end, focus))
    elif tool == "frame":
        t = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
        print(get_video_frame(camera, t))
    elif tool == "grid":
        ts = [float(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [5.0, 20.0, 40.0]
        q = sys.argv[4] if len(sys.argv) > 4 else "What are the workers doing in each frame?"
        print(get_visual_grid(camera, ts, q))

