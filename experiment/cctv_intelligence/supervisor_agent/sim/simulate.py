"""Write simulated tracker output into the SAME SQLite schema the real tracker uses.

Creates: cameras, camera_zones (pixel polygons + roles), tracking_runs, frame_observations,
recording_clocks (simulated clock), a rendered video per run (for frame sheets), sa_camera_info,
and a ground-truth JSON per run (kept OUTSIDE the database, for the bench only).

Two runs of camera 'cam07':
  yesterday  2026-10-08 10:00-10:10  scenario S2 (operator leaves and returns, material handler,
             supervisor walk-through, C+D close pair, a person standing still 4.5 min near the notice board)
  today      2026-10-09 10:00-10:10  scenario S1 (Op 15 operator leaves, a different worker takes over,
             3 visitors at control panel = crowding, camera offline 40 s, a runner in the aisle)

Run (from cctv_intelligence/):  python -m supervisor_agent.sim.simulate --out sim_data
Then:                           CCTV_DB_PATH=sim_data/sim.sqlite3 python -m supervisor_agent.derive --camera cam07
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from . import scenarios as SC

W, H = 1440, 810
FPS = 4


def to_px(x, y):
    return 40 + x * 2.0, 60 + y * 2.0


def body_h(py):
    return 90 + 0.15 * py


ZONES = [  # name, kind, role, world box (x1,y1,x2,y2), description, synonyms
    ("Press 14", "static_object", "machine", (40, 35, 145, 100), "Blue moulding press No.14 at the back wall.", ["press no.14", "machine 14"]),
    ("Press 15", "static_object", "machine", (160, 35, 265, 100), "Blue moulding press No.15 at the back wall.", ["press no.15", "machine 15"]),
    ("Press 16", "static_object", "machine", (335, 35, 440, 100), "Blue moulding press No.16 at the back wall.", ["press no.16", "machine 16"]),
    ("Press 17", "static_object", "machine", (455, 35, 560, 100), "Blue moulding press No.17 at the back wall.", ["press no.17", "machine 17"]),
    ("Bench L", "static_object", "work_surface", (40, 172, 270, 230), "Steel bench in front of Op 14 and Op 15 for trimming parts.", ["left bench", "bench 14/15"]),
    ("Bench R", "static_object", "work_surface", (330, 172, 560, 230), "Steel bench in front of Op 16 and Op 17.", ["right bench", "bench 16/17"]),
    ("Control panel", "static_object", "control", (600, 40, 680, 150), "Electrical control cabinet on the right side.", ["electrical panel", "cabinet"]),
    ("Notice board", "static_object", "", (628, 232, 676, 290), "Notice board on the right.", []),
    ("Op 14", "floor_area", "operator_station", (40, 108, 150, 172), "Operator standing area for Press 14 (between press and bench).", ["press 14 operator", "station 14"]),
    ("Op 15", "floor_area", "operator_station", (150, 108, 270, 172), "Operator standing area for Press 15.", ["press 15 operator", "station 15"]),
    ("Op 16", "floor_area", "operator_station", (330, 108, 445, 172), "Operator standing area for Press 16.", ["press 16 operator", "station 16"]),
    ("Op 17", "floor_area", "operator_station", (445, 108, 560, 172), "Operator standing area for Press 17.", ["press 17 operator", "station 17"]),
    ("Passage", "floor_area", "walkway", (270, 108, 330, 300), "Gap between Bench L and Bench R leading to the yellow aisle.", ["gap between benches"]),
    ("Control panel front", "floor_area", "control", (560, 150, 680, 230), "Floor in front of the control panel.", ["near control panel"]),
    ("Yellow aisle", "floor_area", "walkway", (30, 300, 650, 335), "Marked yellow walkway along the front.", ["aisle", "walkway"]),
    ("West exit", "floor_area", "exit", (0, 295, 30, 340), "Aisle leaves the view at the west (left) side.", ["left exit"]),
    ("East exit", "floor_area", "exit", (650, 295, 680, 340), "Aisle leaves the view at the east (right) side.", ["right exit"]),
]

RUNS = [("S2", "2026-10-08T10:00:00", 11), ("S1", "2026-10-09T10:00:00", 7)]


def schema(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS cameras (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE, source_path TEXT, created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS camera_zones (id INTEGER PRIMARY KEY AUTOINCREMENT, camera_id INTEGER NOT NULL, zone_name TEXT NOT NULL,
        geometry_json TEXT NOT NULL, metadata_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS tracking_runs (run_id TEXT PRIMARY KEY, camera_id INTEGER NOT NULL, source_path TEXT NOT NULL,
        source_kind TEXT NOT NULL, config_json TEXT NOT NULL, status TEXT NOT NULL, started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        finished_at TEXT, error TEXT);
    CREATE TABLE IF NOT EXISTS frame_observations (run_id TEXT NOT NULL, frame_index INTEGER NOT NULL, time_sec REAL NOT NULL,
        detections_json TEXT NOT NULL, PRIMARY KEY(run_id, frame_index));
    CREATE TABLE IF NOT EXISTS recording_clocks (run_id TEXT PRIMARY KEY, source_zero_at TEXT NOT NULL, provenance TEXT NOT NULL,
        uncertainty_sec REAL NOT NULL);
    """)


def zone_px(box):
    x1, y1 = to_px(box[0], box[1]); x2, y2 = to_px(box[2], box[3])
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def render_video(path: Path, res: dict, zero: datetime):
    import cv2
    import numpy as np
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    base = np.full((H, W, 3), (205, 208, 204), np.uint8)
    for name, kind, role, box, *_ in ZONES:
        p = np.array(zone_px(box), np.int32)
        if kind == "static_object":
            col = (150, 90, 40) if role == "machine" else ((170, 170, 175) if role == "work_surface" else (80, 80, 200))
            cv2.fillPoly(base, [p], col)
            cv2.putText(base, name.upper(), (int(p[0][0]) + 6, int(p[0][1]) + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        elif name == "Yellow aisle":
            cv2.fillPoly(base, [p], (60, 200, 235))
    colors = [(30, 30, 200), (30, 150, 30), (200, 60, 30), (150, 30, 150), (30, 160, 200), (120, 120, 30), (60, 60, 60), (200, 120, 200), (0, 90, 160)]
    pids = list(res["P"])
    off = lambda t: any(a <= t < b for a, b in res["offline"])
    for n in range(int(SC.T * FPS)):
        t = n / FPS
        img = base.copy()
        if off(t):
            img[:] = (20, 20, 20)
            cv2.putText(img, "NO SIGNAL", (W // 2 - 120, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
        else:
            people = []
            for i, pid in enumerate(pids):
                x, y = SC.interp(res["P"][pid], t)
                if SC.W0 <= x <= SC.W1:
                    people.append((y, i, *to_px(x, y)))
            for _, i, px, py in sorted(people):
                bh = body_h(py)
                c = colors[i % len(colors)]
                cv2.ellipse(img, (int(px), int(py - bh * 0.45)), (int(bh * 0.14), int(bh * 0.33)), 0, 0, 360, c, -1)
                cv2.circle(img, (int(px), int(py - bh * 0.88)), int(bh * 0.09), (150, 180, 220), -1)
                cv2.line(img, (int(px - bh * 0.06), int(py - bh * 0.15)), (int(px - bh * 0.06), int(py)), (40, 40, 40), 4)
                cv2.line(img, (int(px + bh * 0.06), int(py - bh * 0.15)), (int(px + bh * 0.06), int(py)), (40, 40, 40), 4)
        cv2.putText(img, f"cam07  {(zero + timedelta(seconds=t)):%Y-%m-%d %H:%M:%S}  SIMULATED", (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        vw.write(img)
    vw.release()


def truth_summary(name: str, res: dict, zero: datetime) -> dict:
    gt = res["gt"]
    off = [list(o) for o in res["offline"]]
    count = [len(o) for o in gt]
    zocc = {}
    for zn, kind, role, box, *_ in ZONES:
        if kind != "floor_area":
            continue
        x1, y1, x2, y2 = box
        zocc[zn] = [sum(1 for v in o.values() if x1 <= v[0] < x2 and y1 <= v[1] < y2) for o in gt]
    return dict(scenario=name, zero=zero.isoformat(), offline=off, count_per_sec=count, zone_count_per_sec=zocc,
                distinct_people=res["truth"]["distinct"], entries=res["truth"]["entries"], exits=res["truth"]["exits"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="sim_data")
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "truth").mkdir(exist_ok=True)
    dbp = out / "sim.sqlite3"
    if dbp.exists():
        dbp.unlink()
    conn = sqlite3.connect(dbp)
    schema(conn)
    now = datetime.now().isoformat()
    conn.execute("INSERT INTO cameras(name, source_path, created_at) VALUES (?,?,?)", ("cam07", "simulated", now))
    cam_id = conn.execute("SELECT id FROM cameras WHERE name='cam07'").fetchone()[0]
    for name, kind, role, box, desc, syn in ZONES:
        pts = zone_px(box)
        geo = {"shape": "box", "points": pts, "box": [pts[0][0], pts[0][1], pts[2][0], pts[2][1]]}
        meta = {"kind": kind, "role": role, "category": role, "description": desc, "synonyms": syn}
        conn.execute("INSERT INTO camera_zones(camera_id, zone_name, geometry_json, metadata_json, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                     (cam_id, name, json.dumps(geo), json.dumps(meta), now, now))
    for scen, zero_s, seed in RUNS:
        zero = datetime.fromisoformat(zero_s)
        res = SC.run(scen, seed=seed)
        run_id = f"cam07_{zero:%Y%m%d}_{scen.lower()}_sim"
        video = out / f"{run_id}.mp4"
        cfg = {"fps": FPS, "frame_stride": 1, "start_sec": 0, "end_sec": SC.T, "width": W, "height": H, "simulated": True}
        conn.execute("INSERT INTO tracking_runs(run_id, camera_id, source_path, source_kind, config_json, status) VALUES (?,?,?,?,?,?)",
                     (run_id, cam_id, str(video.resolve()), "synthetic", json.dumps(cfg), "complete"))
        rows = []
        for t, dets in res["frames"]:
            js = []
            for tid, x, y, src, hidden in dets:
                px, py = to_px(x, y); bh = body_h(py)
                js.append({"subject_ref": f"subject_{tid}", "tracker_id": str(tid), "foot": [round(px, 1), round(py, 1)],
                           "raw_foot": [round(px, 1), round(py, 1)], "bbox": [round(px - bh * 0.2, 1), round(py - bh, 1), round(px + bh * 0.2, 1), round(py, 1)],
                           "confidence": 0.8, "foot_source": "knee_projected" if hidden else "ankle_midpoint"})
            rows.append((run_id, int(round(t * FPS)), t, json.dumps(js)))
        conn.executemany("INSERT INTO frame_observations VALUES (?,?,?,?)", rows)
        conn.execute("INSERT INTO recording_clocks VALUES (?,?,?,?)", (run_id, zero.isoformat(), "simulated", 0))
        (out / "truth" / f"{run_id}.json").write_text(json.dumps(truth_summary(scen, res, zero)))
        if not a.no_video:
            render_video(video, res, zero)
        print("wrote", run_id, len(rows), "frames")
    conn.execute("CREATE TABLE IF NOT EXISTS sa_camera_info (camera TEXT PRIMARY KEY, description TEXT, metadata_json TEXT)")
    conn.execute("INSERT OR REPLACE INTO sa_camera_info VALUES (?,?,?)", ("cam07",
                 "SIMULATED. Moulding press line 14–17. Fixed camera, front view. Four blue presses at the back wall, an operator "
                 "standing area in front of each press, two steel benches (Bench L for Op 14/15, Bench R for Op 16/17), "
                 "a passage between the benches, a control panel and notice board on the right, and a yellow aisle along the front "
                 "that leaves the view at the west and east exits.",
                 json.dumps({"line": "Moulding line A", "shift_plan": "A 06-14, B 14-22", "simulated": True})))
    conn.commit()
    print("database:", dbp.resolve())


if __name__ == "__main__":
    main()
