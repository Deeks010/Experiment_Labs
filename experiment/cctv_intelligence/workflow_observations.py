"""Run-scoped, frame-backed measurements. No productivity or identity inference."""
from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np


def init_observations(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tracking_runs (
            run_id TEXT PRIMARY KEY, camera_id INTEGER NOT NULL,
            source_path TEXT NOT NULL, source_kind TEXT NOT NULL,
            config_json TEXT NOT NULL, status TEXT NOT NULL,
            started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            finished_at TEXT, error TEXT,
            FOREIGN KEY(camera_id) REFERENCES cameras(id)
        );
        CREATE TABLE IF NOT EXISTS frame_observations (
            run_id TEXT NOT NULL, frame_index INTEGER NOT NULL,
            time_sec REAL NOT NULL, detections_json TEXT NOT NULL,
            PRIMARY KEY(run_id, frame_index),
            FOREIGN KEY(run_id) REFERENCES tracking_runs(run_id)
        );
        CREATE INDEX IF NOT EXISTS observation_time
            ON frame_observations(run_id, time_sec);
    """)


def select_run(conn, camera_name, run_id=None):
    exists = conn.execute("SELECT 1 FROM sqlite_master WHERE name='tracking_runs'").fetchone()
    if not exists:
        raise ValueError("Frame-level observations are unavailable. Process this video with the updated tracker.")
    query = """SELECT r.*, c.name AS camera_name FROM tracking_runs r
               JOIN cameras c ON c.id=r.camera_id WHERE c.name=?"""
    params = [camera_name]
    if run_id and run_id != "latest":
        query += " AND r.run_id=?"
        params.append(run_id)
    query += " ORDER BY r.rowid DESC LIMIT 1"
    row = conn.execute(query, params).fetchone()
    if not row:
        raise ValueError("No matching tracking run for this camera.")
    return dict(row)


def list_runs(db_path, camera_name=None):
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='tracking_runs'").fetchone():
            return []
        query = """SELECT r.run_id, c.name AS camera, r.source_kind, r.status,
                   r.started_at, r.config_json, count(f.frame_index) AS processed_frames
                   FROM tracking_runs r JOIN cameras c ON c.id=r.camera_id
                   LEFT JOIN frame_observations f ON f.run_id=r.run_id"""
        params = []
        if camera_name:
            query += " WHERE c.name=?"
            params.append(camera_name)
        query += " GROUP BY r.run_id ORDER BY r.rowid DESC"
        result = []
        for row in conn.execute(query, params):
            item = dict(row)
            config = json.loads(item.pop("config_json"))
            item["period"] = [config["start_sec"], config["end_sec"]]
            item["time_basis"] = "source video seconds; cameras are not assumed synchronized"
            result.append(item)
        return result


def inside(point, geometry):
    if geometry.get("shape") == "polygon" and geometry.get("points"):
        polygon = np.asarray(geometry["points"], dtype=np.float32)
        return cv2.pointPolygonTest(polygon, tuple(map(float, point)), False) >= 0
    box = geometry.get("box")
    return bool(box and box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3])


def _merge(intervals, start, end):
    if intervals and abs(intervals[-1][1] - start) < 1e-6:
        intervals[-1][1] = end
    else:
        intervals.append([start, end])


def measure_frames(frames, zones, start, end, sample_step, low_motion_rate=0.12):
    """Integrate only adjacent samples with the same local label at both ends."""
    if not all(math.isfinite(v) for v in (start, end, sample_step)) or end <= start or sample_step <= 0:
        raise ValueError("A finite, positive reporting window and sampling interval are required.")
    frames = sorted((f for f in frames if start <= f["t"] < end), key=lambda f: f["t"])
    people = {}
    area = {z["name"]: {"name": z["name"], "kind": z["kind"], "peak_detected": 0,
                          "observed_person_sec": 0.0, "occupied_sec": 0.0} for z in zones}
    timeline = []
    coverage = 0.0
    for index, frame in enumerate(frames):
        detections = {d["subject_ref"]: d for d in frame["detections"]}
        timeline.append({"time_sec": round(frame["t"], 3), "detected_people": len(detections)})
        zone_refs = defaultdict(set)
        for ref, detection in detections.items():
            entry = people.setdefault(ref, {"subject_ref": ref, "first_seen_sec": frame["t"],
                "last_seen_sec": frame["t"], "observed_sec": 0.0, "low_motion_sec": 0.0,
                "moving_sec": 0.0, "samples": 0, "confidence_sum": 0.0,
                "tracker_ids": set(), "low_motion_intervals": [], "zone_visits": {},
                "path": []})
            entry["last_seen_sec"] = frame["t"]
            entry["samples"] += 1
            entry["confidence_sum"] += detection["confidence"]
            entry["tracker_ids"].add(detection["tracker_id"])
            entry["path"].append({"t": round(frame["t"], 3), "foot": detection["foot"]})
            for zone in zones:
                if inside(detection["foot"], zone["geometry"]):
                    zone_refs[zone["name"]].add(ref)
        for name, refs in zone_refs.items():
            area[name]["peak_detected"] = max(area[name]["peak_detected"], len(refs))
        if index + 1 == len(frames):
            continue
        next_frame = frames[index + 1]
        dt = next_frame["t"] - frame["t"]
        if dt <= 0 or dt > sample_step * 1.5 + 1e-6:
            continue
        coverage += dt
        following = {d["subject_ref"]: d for d in next_frame["detections"]}
        for ref in detections.keys() & following.keys():
            a, b = detections[ref], following[ref]
            entry = people[ref]
            entry["observed_sec"] += dt
            height = max(1, (a["bbox"][3] - a["bbox"][1] + b["bbox"][3] - b["bbox"][1]) / 2)
            rate = math.dist(a["foot"], b["foot"]) / height / dt
            if rate <= low_motion_rate:
                entry["low_motion_sec"] += dt
                _merge(entry["low_motion_intervals"], frame["t"], next_frame["t"])
            else:
                entry["moving_sec"] += dt
        for zone in zones:
            name = zone["name"]
            refs = zone_refs[name] & following.keys()
            refs = {ref for ref in refs if inside(following[ref]["foot"], zone["geometry"])}
            if refs:
                area[name]["occupied_sec"] += dt
            area[name]["observed_person_sec"] += len(refs) * dt
            for ref in refs:
                intervals = people[ref]["zone_visits"].setdefault(name, [])
                _merge(intervals, frame["t"], next_frame["t"])
    for entry in people.values():
        span_samples = sum(entry["first_seen_sec"] <= f["t"] <= entry["last_seen_sec"] for f in frames)
        entry["unobserved_samples_within_span"] = span_samples - entry["samples"]
        entry["tracker_ids"] = sorted(entry["tracker_ids"])
        entry["mean_detection_confidence"] = round(entry.pop("confidence_sum") / entry["samples"], 4)
        for key in ("observed_sec", "low_motion_sec", "moving_sec", "first_seen_sec", "last_seen_sec"):
            entry[key] = round(entry[key], 3)
        entry["longest_low_motion_sec"] = round(max((b-a for a,b in entry["low_motion_intervals"]), default=0), 3)
    for row in area.values():
        for key in ("observed_person_sec", "occupied_sec"):
            row[key] = round(row[key], 3)
    return {"period": {"start_sec": start, "end_sec": end, "duration_sec": end-start},
            "processed_frames": len(frames), "adjacent_sample_coverage_sec": round(coverage, 3),
            "frames_without_detections": sum(not f["detections"] for f in frames),
            "peak_detected_people": max((len(f["detections"]) for f in frames), default=0),
            "local_track_labels": len(people), "people": list(people.values()),
            "areas": list(area.values()), "count_timeline": timeline}


def get_observations(db_path, camera_name, run_id=None, start_sec=None, end_sec=None):
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        run = select_run(conn, camera_name, run_id)
        config = json.loads(run["config_json"])
        start = config["start_sec"] if start_sec is None else max(float(start_sec), config["start_sec"])
        end = config["end_sec"] if end_sec is None else min(float(end_sec), config["end_sec"])
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError("Requested time window is outside the run or invalid.")
        frames = [{"t": row["time_sec"], "detections": json.loads(row["detections_json"])}
                  for row in conn.execute("SELECT * FROM frame_observations WHERE run_id=? AND time_sec>=? AND time_sec<? ORDER BY frame_index",
                                          (run["run_id"], start, end))]
        zones = [{"name": row["zone_name"], "geometry": json.loads(row["geometry_json"]),
                  "kind": json.loads(row["metadata_json"]).get("kind", "static_object"),
                  "description": json.loads(row['metadata_json']).get('description',''),
                  "category": json.loads(row['metadata_json']).get('category','')}
                 for row in conn.execute("SELECT * FROM camera_zones WHERE camera_id=?", (run["camera_id"],))]
    summary = measure_frames(frames, zones, start, end, config["frame_stride"] / config["fps"])
    limitations = [
        "No validated normal reference or operational limits supplied; normal/abnormal is undetermined.",
        "Low motion is image-space stillness, not confirmed waiting, idleness or lost production.",
        "Labels are camera-local estimates, not verified people or cross-camera matches.",
        "Separate clips have independent clocks; no causal chain or travel time across cameras is established.",
        "Intervals estimate activity between adjacent detections only; missing people are not assumed absent.",
        "Static-object overlap means image proximity only; floor-area membership is geometric, not task evidence.",
    ]
    if run["source_kind"] == "synthetic":
        limitations.append("Synthetic clips test software operation, not real-factory accuracy or financial value.")
    summary.update({"camera": camera_name, "run_id": run["run_id"], "status": run["status"],
                    "source_kind": run["source_kind"], "source_path": run["source_path"],
                    "frame_size": [config["width"], config["height"]], "zones": zones,
                    "normality": "insufficient_evidence", "baseline": None,
                    "limitations": limitations, "measurement_policy": {"low_motion_body_heights_per_sec": 0.12,
                        "gap_policy": "Never integrate across a missing processed frame or missing person detection.",
                        "crowding": "Counts only; no site capacity supplied, so no congestion verdict."}})
    return summary


def evidence_frame(db_path, camera_name, run_id, t_sec, output_dir, include_map=True):
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        run = select_run(conn, camera_name, run_id)
        config = json.loads(run["config_json"])
        if not math.isfinite(t_sec) or not config["start_sec"] <= t_sec < config["end_sec"]:
            raise ValueError("Evidence timestamp must be within the processed video range.")
        sample = conn.execute("SELECT * FROM frame_observations WHERE run_id=? ORDER BY ABS(time_sec-?) LIMIT 1",
                              (run["run_id"], t_sec)).fetchone()
        if sample is None:
            raise ValueError("No processed frames available for evidence.")
        zones = conn.execute("SELECT * FROM camera_zones WHERE camera_id=?", (run["camera_id"],)).fetchall()
        if not include_map:
            zones = []
    cap = cv2.VideoCapture(run["source_path"])
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample["frame_index"])
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise ValueError("The source frame could not be read.")
    for zone in zones:
        geometry = json.loads(zone["geometry_json"])
        points = geometry.get("points")
        if not points:
            x1, y1, x2, y2 = geometry["box"]
            points = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        cv2.polylines(frame, [np.asarray(points, np.int32)], True, (80,180,230), 2)
        cv2.putText(frame, zone["zone_name"], tuple(map(int, points[0])), cv2.FONT_HERSHEY_SIMPLEX, .5, (80,180,230), 1)
    for detection in json.loads(sample["detections_json"]):
        x1,y1,x2,y2 = map(int, detection["bbox"])
        cv2.rectangle(frame, (x1,y1), (x2,y2), (50,220,90), 2)
        cv2.putText(frame, detection["subject_ref"], (x1,max(20,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, .6, (50,220,90), 2)
        cv2.circle(frame, tuple(map(int,detection["foot"])), 4, (30,60,240), -1)
    cv2.putText(frame, f"{camera_name}  {sample['time_sec']:.3f}s  {run['source_kind']}", (15,30), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = '' if include_map else '_people_only'
    path = output_dir / f"{run['run_id']}_{sample['frame_index']:06d}{suffix}.jpg"
    if not cv2.imwrite(str(path), frame):
        raise IOError("Could not save evidence frame.")
    from workflow_knowledge import frame_ref, map_version, scope_id
    map_zones = [{'name': z['zone_name'], 'geometry': json.loads(z['geometry_json']),
                  'kind': json.loads(z['metadata_json']).get('kind', 'static_object'),
                  'description': json.loads(z['metadata_json']).get('description',''),
                  'category': json.loads(z['metadata_json']).get('category','')} for z in zones]
    return {"path": str(path.resolve()), "requested_time_sec": t_sec, "actual_time_sec": sample["time_sec"],
            "frame_index": sample["frame_index"], "run_id": run["run_id"],
            "scope_id": scope_id(camera_name, run['run_id']), "map_version": map_version(map_zones),
            "evidence_kind": "labeled_frame_artifact", "image_interpreted": False,
            "evidence_ref": frame_ref(camera_name, run['run_id'], sample['frame_index']),
            "person_refs": [d['subject_ref'] for d in json.loads(sample['detections_json'])]}
