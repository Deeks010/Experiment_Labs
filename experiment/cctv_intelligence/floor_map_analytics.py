from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB = BASE_DIR / "cctv_maps.sqlite3"
DEFAULT_OUTPUT = BASE_DIR / "map_analytics"


@dataclass
class Track:
    ref: str
    start: float
    end: float
    points: list[tuple[float, float, float]]
    confidence: float
    quality: str
    tracker_ids: list[str]
    reconnect_count: int


def parse_json(value: str | None) -> dict[str, Any]:
    try:
        return json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}


def parse_created(value: str | None) -> datetime:
    text = (value or "").strip()
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return datetime.min


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def evidence_run_key(row: sqlite3.Row) -> str | None:
    metadata = parse_json(row["metadata_json"])
    if metadata.get("run_id"):
        return str(metadata["run_id"])
    evidence = parse_json(row["evidence_json"])
    for frame in evidence.get("frames", []):
        path = frame.get("path")
        if path:
            return Path(path).parent.name
    return None


def discover_runs(rows: list[sqlite3.Row]) -> dict[str, list[sqlite3.Row]]:
    explicit: list[tuple[sqlite3.Row, str]] = []
    unassigned: list[sqlite3.Row] = []
    for row in rows:
        key = evidence_run_key(row)
        if key:
            explicit.append((row, key))
        else:
            unassigned.append(row)

    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row, key in explicit:
        grouped[key].append(row)

    for row in unassigned:
        row_meta = parse_json(row["metadata_json"])
        row_video = row_meta.get("video_path")
        row_time = parse_created(row["created_at"])
        candidates = []
        for known_row, key in explicit:
            known_meta = parse_json(known_row["metadata_json"])
            if known_meta.get("video_path") != row_video:
                continue
            delta = abs((parse_created(known_row["created_at"]) - row_time).total_seconds())
            candidates.append((delta, key))
        if candidates and min(candidates)[0] <= 3600:
            grouped[min(candidates)[1]].append(row)
        else:
            fallback = f"legacy_{row_time.date().isoformat()}_{row_video or 'unknown'}"
            grouped[fallback].append(row)
    return dict(grouped)


def load_camera(conn: sqlite3.Connection, camera_name: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM cameras WHERE name=?", (camera_name,)).fetchone()
    if row is None:
        names = [r[0] for r in conn.execute("SELECT name FROM cameras ORDER BY name")]
        raise ValueError(f"Camera '{camera_name}' not found. Available: {', '.join(names) or 'none'}")
    return row


def load_zones(conn: sqlite3.Connection, camera_id: int) -> list[dict[str, Any]]:
    zones = []
    rows = conn.execute(
        "SELECT zone_name, geometry_json, metadata_json FROM camera_zones WHERE camera_id=? ORDER BY id",
        (camera_id,),
    ).fetchall()
    for row in rows:
        geometry = parse_json(row["geometry_json"])
        box = geometry.get("box")
        if not box or len(box) != 4:
            continue
        zones.append(
            {
                "name": row["zone_name"],
                "box": [float(v) for v in box],
                "metadata": parse_json(row["metadata_json"]),
            }
        )
    return zones


def load_tracks(rows: list[sqlite3.Row]) -> list[Track]:
    tracks = []
    used_refs: dict[str, int] = defaultdict(int)
    for row in rows:
        payload = parse_json(row["person_path_json"])
        for subject in payload.get("subjects", []):
            raw_ref = str(subject.get("subject_ref") or f"row_{row['id']}")
            used_refs[raw_ref] += 1
            ref = raw_ref if used_refs[raw_ref] == 1 else f"{raw_ref}_row{row['id']}"
            points = []
            for point in subject.get("path_points", []):
                foot = point.get("foot")
                if foot and len(foot) >= 2:
                    points.append((float(point["t"]), float(foot[0]), float(foot[1])))
            points.sort()
            if not points:
                continue
            tracks.append(
                Track(
                    ref=ref,
                    start=float(subject.get("start_time_sec", points[0][0])),
                    end=float(subject.get("end_time_sec", points[-1][0])),
                    points=points,
                    confidence=float(subject.get("confidence") or row["confidence"] or 0.0),
                    quality=str(subject.get("quality") or "unknown"),
                    tracker_ids=[str(value) for value in subject.get("tracker_ids", [])],
                    reconnect_count=len(subject.get("reconnect_events", [])),
                )
            )
    return tracks


def frame_dimensions(camera: sqlite3.Row, tracks: list[Track], zones: list[dict[str, Any]]) -> tuple[int, int]:
    source = Path(camera["source_path"] or "")
    if source.exists():
        cap = cv2.VideoCapture(str(source))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        if width > 0 and height > 0:
            return width, height

    xs = [point[1] for track in tracks for point in track.points]
    ys = [point[2] for track in tracks for point in track.points]
    xs.extend(zone["box"][2] for zone in zones)
    ys.extend(zone["box"][3] for zone in zones)
    return max(1, int(max(xs, default=1920) * 1.03)), max(1, int(max(ys, default=1080) * 1.03))


def interpolate(track: Track, sample_sec: float = 1.0) -> list[tuple[float, float, float]]:
    if len(track.points) == 1:
        return [track.points[0]]
    samples = []
    point_index = 0
    t = track.start
    while t <= track.end + 1e-6:
        while point_index + 1 < len(track.points) and track.points[point_index + 1][0] < t:
            point_index += 1
        left = track.points[point_index]
        right = track.points[min(point_index + 1, len(track.points) - 1)]
        if right[0] - left[0] > 3.0:
            t += sample_sec
            continue
        ratio = 0.0 if right[0] == left[0] else (t - left[0]) / (right[0] - left[0])
        ratio = max(0.0, min(1.0, ratio))
        samples.append((t, left[1] + ratio * (right[1] - left[1]), left[2] + ratio * (right[2] - left[2])))
        t += sample_sec
    return samples


def clip_tracks(tracks: list[Track], start_sec: float | None, end_sec: float | None) -> list[Track]:
    if start_sec is None and end_sec is None:
        return tracks
    clipped = []
    for track in tracks:
        window_start = track.start if start_sec is None else max(track.start, start_sec)
        window_end = track.end if end_sec is None else min(track.end, end_sec)
        if window_end < window_start:
            continue
        points = [point for point in interpolate(track) if window_start <= point[0] <= window_end]
        if not points:
            continue
        clipped.append(
            Track(
                ref=track.ref,
                start=window_start,
                end=window_end,
                points=points,
                confidence=track.confidence,
                quality=track.quality,
                tracker_ids=track.tracker_ids,
                reconnect_count=track.reconnect_count,
            )
        )
    return clipped


def distance_to_box(x: float, y: float, box: list[float]) -> float:
    x1, y1, x2, y2 = box
    near_x = max(x1, min(x, x2))
    near_y = max(y1, min(y, y2))
    return math.hypot(x - near_x, y - near_y)


def point_inside_box(x: float, y: float, box: list[float]) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def round_position(t: float, x: float, y: float, width: int, height: int) -> dict[str, Any]:
    return {
        "time_sec": round(t, 3),
        "pixel": {"x": round(x, 1), "y": round(y, 1)},
        "normalized": {"x": round(x / max(1, width), 4), "y": round(y / max(1, height), 4)},
    }


def sample_positions(
    samples: list[tuple[float, float, float]], width: int, height: int, interval_sec: float = 5.0
) -> list[dict[str, Any]]:
    selected: list[tuple[float, float, float]] = []
    last_time = -math.inf
    for point in samples:
        if point[0] - last_time >= interval_sec:
            selected.append(point)
            last_time = point[0]
    if samples and (not selected or selected[-1] != samples[-1]):
        selected.append(samples[-1])
    return [round_position(t, x, y, width, height) for t, x, y in selected]


def track_zone_intervals(
    samples: list[tuple[float, float, float]], zones: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    for zone in zones:
        seconds = sorted({int(t) for t, x, y in samples if point_inside_box(x, y, zone["box"])})
        for item in observation_spans(seconds):
            intervals.append({"mapped_area": zone["name"], **item})
    return sorted(intervals, key=lambda item: (item["start_sec"], item["mapped_area"]))


def consecutive_ranges(seconds: list[int], minimum_sec: int) -> list[dict[str, int]]:
    if not seconds:
        return []
    ranges = []
    start = previous = seconds[0]
    for value in seconds[1:]:
        if value != previous + 1:
            if previous - start + 1 >= minimum_sec:
                ranges.append({"start_sec": start, "end_sec": previous + 1, "duration_sec": previous - start + 1})
            start = value
        previous = value
    if previous - start + 1 >= minimum_sec:
        ranges.append({"start_sec": start, "end_sec": previous + 1, "duration_sec": previous - start + 1})
    return ranges


def observation_spans(seconds: list[int], max_gap_sec: int = 5) -> list[dict[str, int]]:
    """Compress nearby inside samples without claiming the gap was observed inside."""
    if not seconds:
        return []
    groups = [[seconds[0]]]
    for value in seconds[1:]:
        if value - groups[-1][-1] > max_gap_sec:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [
        {
            "start_sec": values[0],
            "end_sec": values[-1] + 1,
            "span_sec": values[-1] - values[0] + 1,
            "sampled_inside_seconds": len(values),
        }
        for values in groups
    ]


def build_summary(
    camera_name: str,
    run_name: str,
    tracks: list[Track],
    zones: list[dict[str, Any]],
    width: int,
    height: int,
    near_ratio: float,
    crowd_threshold: int,
    sustained_sec: int,
    period_start: float | None = None,
    period_end: float | None = None,
    mapped_area_source_camera: str | None = None,
) -> tuple[dict[str, Any], dict[str, list[tuple[float, float, float]]]]:
    samples_by_track = {track.ref: interpolate(track) for track in tracks}
    start = period_start if period_start is not None else min((track.start for track in tracks), default=0.0)
    end = period_end if period_end is not None else max((track.end for track in tracks), default=0.0)
    timeline: dict[int, set[str]] = defaultdict(set)
    for ref, samples in samples_by_track.items():
        for t, _, _ in samples:
            timeline[int(t)].add(ref)

    area_rows = []
    area_timelines: dict[str, dict[int, int]] = {}
    for zone in zones:
        zone_timeline: dict[int, set[str]] = defaultdict(set)
        for ref, samples in samples_by_track.items():
            for t, x, y in samples:
                if point_inside_box(x, y, zone["box"]):
                    second = int(t)
                    zone_timeline[second].add(ref)
        counts = {second: len(refs) for second, refs in zone_timeline.items()}
        area_timelines[zone["name"]] = counts
        crowded_seconds = sorted(second for second, count in counts.items() if count >= crowd_threshold)
        area_rows.append(
            {
                "mapped_area": zone["name"],
                "geometry": {"type": "axis_aligned_box", "pixel_box": [round(v, 1) for v in zone["box"]]},
                "footpoint_inside_person_seconds": sum(counts.values()),
                "seconds_with_any_footpoint_inside": len(counts),
                "percent_of_period_with_any_footpoint_inside": round(100.0 * len(counts) / max(1.0, end - start), 1),
                "peak_simultaneous_footpoints_inside": max(counts.values(), default=0),
                "subject_refs_observed_inside": sorted({ref for ref, samples in samples_by_track.items() if any(point_inside_box(x, y, zone["box"]) for _, x, y in samples)}),
                "sustained_multi_person_intervals": consecutive_ranges(crowded_seconds, sustained_sec),
            }
        )

    second_counts = {second: len(refs) for second, refs in timeline.items()}
    first_second = int(math.floor(start))
    last_second = int(math.ceil(end))
    report_seconds = list(range(first_second, last_second))
    peak = max(second_counts.values(), default=0)
    peak_seconds = sorted(second for second in report_seconds if peak > 0 and second_counts.get(second, 0) == peak)
    duration = max(0.0, end - start)
    trend_bucket_sec = 60 if duration <= 900 else (300 if duration <= 7200 else 3600)
    time_trends = []
    bucket_start = first_second
    while bucket_start < last_second:
        bucket_end = min(bucket_start + trend_bucket_sec, last_second)
        bucket_seconds = list(range(bucket_start, bucket_end))
        area_presence = [
            {
                "mapped_area": name,
                "footpoint_inside_person_seconds": sum(counts.get(second, 0) for second in bucket_seconds),
            }
            for name, counts in area_timelines.items()
        ]
        area_presence = sorted(area_presence, key=lambda row: row["footpoint_inside_person_seconds"], reverse=True)
        time_trends.append(
            {
                "start_sec": bucket_start,
                "end_sec": bucket_end,
                "average_people_visible": round(
                    sum(second_counts.get(second, 0) for second in bucket_seconds) / max(1, len(bucket_seconds)), 2
                ),
                "peak_people_visible": max((second_counts.get(second, 0) for second in bucket_seconds), default=0),
                "mapped_area_observations": [row for row in area_presence if row["footpoint_inside_person_seconds"] > 0],
            }
        )
        bucket_start = bucket_end

    area_rows.sort(key=lambda row: row["footpoint_inside_person_seconds"], reverse=True)
    track_rows = []
    for track in sorted(tracks, key=lambda value: (value.start, value.ref)):
        samples = samples_by_track.get(track.ref, [])
        duration = max(0.0, track.end - track.start)
        track_rows.append(
            {
                "subject_ref": track.ref,
                "identity_status": "persistent_observation" if duration >= 30.0 else ("brief_observation" if duration < 10.0 else "limited_observation"),
                "first_seen_sec": round(track.start, 3),
                "last_seen_sec": round(track.end, 3),
                "elapsed_span_sec": round(duration, 3),
                "sampled_visible_seconds": len({int(t) for t, _, _ in samples}),
                "detection_confidence": round(track.confidence, 4),
                "track_quality": track.quality,
                "raw_tracker_ids_joined": track.tracker_ids,
                "reconnect_event_count": track.reconnect_count,
                "first_position": round_position(*samples[0], width, height) if samples else None,
                "last_position": round_position(*samples[-1], width, height) if samples else None,
                "position_samples_5_sec": sample_positions(samples, width, height),
                "mapped_area_intervals": track_zone_intervals(samples, zones),
            }
        )
    persistent = sum(row["identity_status"] == "persistent_observation" for row in track_rows)
    brief = sum(row["identity_status"] == "brief_observation" for row in track_rows)
    summary = {
        "schema_version": 3,
        "camera": camera_name,
        "tracking_run": run_name,
        "mapped_area_source_camera": mapped_area_source_camera or camera_name,
        "time_basis": "seconds from the start of the source video",
        "period": {"start_sec": round(start, 3), "end_sec": round(end, 3), "duration_sec": round(max(0.0, end - start), 3)},
        "coordinate_system": {
            "space": "source_video_pixels",
            "frame_size": {"width": width, "height": height},
            "origin": "top_left",
            "x_direction": "right",
            "y_direction": "down",
            "person_position": "estimated floor-contact point from pose/bounding-box evidence",
        },
        "identity_observations": {
            "estimated_distinct_subject_references": len(track_rows),
            "persistent_references_30_sec_or_more": persistent,
            "brief_references_under_10_sec": brief,
            "limited_references_10_to_30_sec": len(track_rows) - persistent - brief,
            "raw_tracker_id_count": len({tracker_id for track in tracks for tracker_id in track.tracker_ids}),
            "reconnect_events_applied": sum(track.reconnect_count for track in tracks),
            "meaning": "Subject references are camera-local tracking estimates, not biometric or ground-truth unique-person identities.",
        },
        "observed_presence": {
            "average_people_visible": round(
                sum(second_counts.get(second, 0) for second in report_seconds) / max(1, len(report_seconds)), 2
            ),
            "peak_people_visible": peak,
            "peak_periods": consecutive_ranges(peak_seconds, 1),
        },
        "data_quality": {
            "average_detection_confidence": round(
                sum(track.confidence * len(samples_by_track.get(track.ref, [])) for track in tracks)
                / max(1, sum(len(samples_by_track.get(track.ref, [])) for track in tracks)),
                4,
            ),
            "identity_limit": "The estimated subject-reference count may contain brief false detections or unresolved fragments and is not a verified worker count.",
        },
        "tracks": track_rows,
        "mapped_area_observations": area_rows,
        "time_windows": {
            "bucket_seconds": trend_bucket_sec,
            "segments": time_trends,
        },
        "visual_evidence": {
            "occupancy_map": "occupancy_map.png",
            "track_position_map": "track_position_map.png",
        },
        "metric_definitions": {
            "footpoint_inside_person_seconds": "At each one-second sample, the number of tracked floor-contact points geometrically inside the stored area box, summed over time.",
            "sampled_visible_seconds": "Distinct one-second samples for which this subject reference had an interpolated position; gaps over three seconds are not filled.",
            "occupancy_map": "Relative density of sampled person floor-contact positions; it is not physical floor area occupied.",
        },
        "analysis_limits": [
            "A foot point inside a mapped box states position only; it does not establish which machine, task, or object the person interacted with.",
            "Mapped boxes can overlap, so one position can be listed in more than one mapped area at the same time.",
            "Occlusion and missed detections can reduce occupancy or split one visit into multiple episodes.",
            "Image-pixel distance and density are perspective-distorted unless a floor-plane calibration is added.",
            "The summary describes camera observations only and makes no claim about work activity or intent.",
        ],
    }
    return summary, samples_by_track


def heat_color(value: float) -> tuple[int, int, int]:
    value = max(0.0, min(1.0, value))
    if value < 0.5:
        ratio = value * 2.0
        return (int(255 * (1.0 - ratio)), 255, int(70 * ratio))
    ratio = (value - 0.5) * 2.0
    return (0, int(255 * (1.0 - ratio)), 255)


def render_map(
    output_path: Path,
    title: str,
    mode: str,
    tracks: list[Track],
    samples_by_track: dict[str, list[tuple[float, float, float]]],
    zones: list[dict[str, Any]],
    width: int,
    height: int,
    background: np.ndarray | None = None,
) -> None:
    scale = min(1400 / width, 820 / height)
    map_w, map_h = max(1, int(width * scale)), max(1, int(height * scale))
    legend_w = 460
    canvas = np.full((map_h + 80, map_w + legend_w, 3), (246, 247, 249), dtype=np.uint8)
    if background is not None:
        resized = cv2.resize(background, (map_w, map_h), interpolation=cv2.INTER_AREA)
        pale = np.full_like(resized, (246, 247, 249))
        canvas[80:, :map_w] = cv2.addWeighted(resized, 0.32, pale, 0.68, 0)
    field = np.zeros((map_h, map_w), dtype=np.float32)

    if mode == "occupancy":
        for samples in samples_by_track.values():
            for _, x, y in samples:
                cv2.circle(field, (int(x * scale), int(y * scale)), max(5, int(24 * scale)), 1.0, -1)

    sigma = max(5.0, 18.0 * scale)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if field.max() > 0:
        normalized = field / field.max()
        heat = np.zeros((map_h, map_w, 3), dtype=np.uint8)
        for y in range(map_h):
            for x in range(map_w):
                heat[y, x] = heat_color(float(normalized[y, x]))
        alpha = np.clip(normalized[..., None] * 0.82, 0.0, 0.82)
        map_region = canvas[80:, :map_w]
        canvas[80:, :map_w] = (map_region * (1.0 - alpha) + heat * alpha).astype(np.uint8)

    cv2.putText(canvas, title, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (30, 34, 40), 2, cv2.LINE_AA)
    subtitle = (
        "Blue boxes are fixed mapped objects; color depth shows relative position density."
        if mode == "occupancy"
        else "Blue boxes are fixed mapped objects; colored lines are estimated floor-contact paths."
    )
    cv2.putText(canvas, subtitle, (18, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 86, 94), 1, cv2.LINE_AA)
    cv2.line(canvas, (map_w, 0), (map_w, map_h + 80), (214, 218, 224), 1)
    cv2.putText(canvas, "MAPPED AREAS", (map_w + 22, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 34, 40), 2, cv2.LINE_AA)
    for index, zone in enumerate(zones, start=1):
        x1, y1, x2, y2 = zone["box"]
        p1 = (int(x1 * scale), int(y1 * scale) + 80)
        p2 = (int(x2 * scale), int(y2 * scale) + 80)
        cv2.rectangle(canvas, p1, p2, (190, 92, 25), 2, cv2.LINE_AA)
        cv2.circle(canvas, (p1[0] + 17, p1[1] + 17), 13, (190, 92, 25), -1, cv2.LINE_AA)
        cv2.putText(canvas, str(index), (p1[0] + 11, p1[1] + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
        legend_y = 75 + (index - 1) * 84
        cv2.circle(canvas, (map_w + 36, legend_y), 14, (190, 92, 25), -1, cv2.LINE_AA)
        cv2.putText(canvas, str(index), (map_w + 30, legend_y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
        words = zone["name"].split()
        lines = []
        current = ""
        for word in words:
            if len(current) + len(word) + 1 > 40:
                lines.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            lines.append(current)
        for line_index, line in enumerate(lines[:3]):
            cv2.putText(canvas, line, (map_w + 60, legend_y - 8 + line_index * 21), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (48, 53, 61), 1, cv2.LINE_AA)

    if mode == "tracks":
        palette = [
            (42, 67, 173), (40, 140, 70), (32, 126, 196), (168, 74, 50),
            (140, 65, 150), (35, 145, 150), (90, 90, 205), (120, 105, 35),
        ]
        for index, track in enumerate(sorted(tracks, key=lambda value: value.ref)):
            color = palette[index % len(palette)]
            points = np.array(
                [[int(x * scale), int(y * scale) + 80] for _, x, y in track.points], dtype=np.int32
            )
            if len(points) >= 2:
                cv2.polylines(canvas, [points], False, color, 2, cv2.LINE_AA)
            if len(points):
                end_x, end_y = (int(points[-1][0]), int(points[-1][1]))
                cv2.circle(canvas, (end_x, end_y), 4, color, -1, cv2.LINE_AA)
                cv2.putText(canvas, track.ref, (end_x + 5, end_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(canvas, track.ref, (end_x + 5, end_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Could not write {output_path}")


def list_database(conn: sqlite3.Connection) -> None:
    cameras = conn.execute("SELECT * FROM cameras ORDER BY name").fetchall()
    if not cameras:
        print("No cameras found.")
        return
    for camera in cameras:
        rows = conn.execute("SELECT * FROM floor_data WHERE camera_id=? ORDER BY created_at, id", (camera["id"],)).fetchall()
        zones = conn.execute("SELECT COUNT(*) FROM camera_zones WHERE camera_id=?", (camera["id"],)).fetchone()[0]
        runs = discover_runs(rows)
        print(f"Camera {camera['name']}: zones={zones}, floor_rows={len(rows)}, runs={len(runs)}")
        for name, run_rows in sorted(runs.items(), key=lambda item: max(parse_created(r["created_at"]) for r in item[1]), reverse=True):
            tracks = load_tracks(run_rows)
            start = min((track.start for track in tracks), default=0.0)
            end = max((track.end for track in tracks), default=0.0)
            created = max((r["created_at"] for r in run_rows), default="")
            print(f"  {name}: rows={len(run_rows)}, tracks={len(tracks)}, video={start:.1f}-{end:.1f}s, latest_write={created}")


def load_background(source_path: str | None) -> np.ndarray | None:
    source = Path(source_path or "")
    if not source.exists():
        return None
    cap = cv2.VideoCapture(str(source))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    identity = summary["identity_observations"]
    presence = summary["observed_presence"]
    period = summary["period"]
    lines = [
        f"# Camera Observation Summary: {summary['camera']}",
        "",
        f"- Video interval: `{period['start_sec']}-{period['end_sec']}s` ({period['duration_sec']}s)",
        f"- Estimated subject references: **{identity['estimated_distinct_subject_references']}**",
        f"- Persistent references (30s or more): **{identity['persistent_references_30_sec_or_more']}**",
        f"- Brief references (under 10s): **{identity['brief_references_under_10_sec']}**",
        f"- Average / peak visible detections: **{presence['average_people_visible']} / {presence['peak_people_visible']}**",
        "",
        "> Subject references are tracking estimates, not verified unique workers. Position does not establish activity or intent.",
        "",
        "## Mapped Position Observations",
        "",
        "| Mapped area | Footpoint person-seconds | Seconds with any footpoint | Peak simultaneous |",
        "|---|---:|---:|---:|",
    ]
    for area in summary["mapped_area_observations"]:
        lines.append(
            f"| {area['mapped_area']} | {area['footpoint_inside_person_seconds']} | "
            f"{area['seconds_with_any_footpoint_inside']} | {area['peak_simultaneous_footpoints_inside']} |"
        )
    lines.extend(
        [
            "",
            "## Track Index",
            "",
            "| Subject reference | Status | First seen | Last seen | Sampled visible seconds | Raw IDs joined |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for track in summary["tracks"]:
        raw_ids = ", ".join(track["raw_tracker_ids_joined"]) or "none"
        lines.append(
            f"| {track['subject_ref']} | {track['identity_status']} | {track['first_seen_sec']} | "
            f"{track['last_seen_sec']} | {track['sampled_visible_seconds']} | {raw_ids} |"
        )
    lines.extend(
        [
            "",
            "## Visual Evidence",
            "",
            "- [Occupancy map](occupancy_map.png)",
            "- [Track position map](track_position_map.png)",
            "- Full timestamped coordinates and mapped-area spans are in [summary.json](summary.json).",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate neutral floor occupancy and movement analytics from SQLite tracking data.")
    parser.add_argument("camera", nargs="?", help="Camera name. Omit with --list.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--run", default="latest", help="Tracking run ID from --list, or latest.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--near-ratio", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--zones-from", help="Camera whose stored mapped areas should be applied to this run.")
    parser.add_argument("--crowd-threshold", type=int, default=2)
    parser.add_argument("--sustained-sec", type=int, default=120)
    parser.add_argument("--start-sec", type=float, help="Optional start of the reporting window in video seconds.")
    parser.add_argument("--end-sec", type=float, help="Optional end of the reporting window in video seconds.")
    parser.add_argument("--list", action="store_true", help="List cameras and inferred tracking runs.")
    args = parser.parse_args()

    with connect(args.db) as conn:
        if args.list:
            list_database(conn)
            return
        if not args.camera:
            parser.error("camera is required unless --list is used")
        camera = load_camera(conn, args.camera)
        zone_camera = load_camera(conn, args.zones_from) if args.zones_from else camera
        zones = load_zones(conn, zone_camera["id"])
        rows = conn.execute("SELECT * FROM floor_data WHERE camera_id=? ORDER BY created_at, id", (camera["id"],)).fetchall()
        runs = discover_runs(rows)
        if not runs:
            raise ValueError(f"Camera '{args.camera}' has no floor tracking data")
        if args.run == "latest":
            run_name, selected_rows = max(runs.items(), key=lambda item: max(parse_created(r["created_at"]) for r in item[1]))
        elif args.run in runs:
            run_name, selected_rows = args.run, runs[args.run]
        else:
            raise ValueError(f"Run '{args.run}' not found. Use --list to see available runs.")
        tracks = load_tracks(selected_rows)
        width, height = frame_dimensions(camera, tracks, zones)
        background = load_background(camera["source_path"])

    if args.start_sec is not None and args.end_sec is not None and args.end_sec <= args.start_sec:
        parser.error("--end-sec must be greater than --start-sec")
    tracks = clip_tracks(tracks, args.start_sec, args.end_sec)

    summary, samples = build_summary(
        args.camera,
        run_name,
        tracks,
        zones,
        width,
        height,
        args.near_ratio,
        args.crowd_threshold,
        args.sustained_sec,
        args.start_sec,
        args.end_sec,
        args.zones_from,
    )
    summary["provenance"] = {
        "source_video": camera["source_path"],
        "sqlite_database": str(args.db.resolve()),
        "floor_data_row_ids": [int(row["id"]) for row in selected_rows],
        "tracking_run": run_name,
    }
    output_dir = args.output_dir / args.camera / run_name
    if args.start_sec is not None or args.end_sec is not None:
        start_label = "start" if args.start_sec is None else f"{args.start_sec:g}"
        end_label = "end" if args.end_sec is None else f"{args.end_sec:g}"
        output_dir = output_dir / f"window_{start_label}_{end_label}"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    markdown_path = output_dir / "SUMMARY.md"
    occupancy_path = output_dir / "occupancy_map.png"
    track_position_path = output_dir / "track_position_map.png"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_summary(markdown_path, summary)
    period_label = f"{summary['period']['start_sec']:g}-{summary['period']['end_sec']:g}s"
    render_map(
        occupancy_path,
        f"{args.camera} - occupancy depth - {period_label}",
        "occupancy",
        tracks,
        samples,
        zones,
        width,
        height,
        background,
    )
    render_map(
        track_position_path,
        f"{args.camera} - track positions - {period_label}",
        "tracks",
        tracks,
        samples,
        zones,
        width,
        height,
        background,
    )
    print(f"Run: {run_name}")
    print(f"Tracks: {len(tracks)}; static areas: {len(zones)}; frame: {width}x{height}")
    print(f"Summary: {summary_path}")
    print(f"Readable summary: {markdown_path}")
    print(f"Occupancy map: {occupancy_path}")
    print(f"Track position map: {track_position_path}")


if __name__ == "__main__":
    main()
