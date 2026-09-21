from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import threading
import time
import webbrowser
import uuid
from dataclasses import dataclass, field
from contextlib import closing, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .workflow_observations import init_observations
except ImportError:
    from workflow_observations import init_observations


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("CCTV_DB_PATH", BASE_DIR / "cctv_maps.sqlite3"))
OUTPUT_DIR = BASE_DIR / "activity_runs"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


@contextmanager
def connect_db():
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        with conn:
            yield conn


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect_db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS cameras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                source_path TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS camera_zones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id INTEGER NOT NULL,
                zone_name TEXT NOT NULL,
                geometry_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(camera_id) REFERENCES cameras(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS floor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id INTEGER NOT NULL,
                start_time_sec REAL NOT NULL,
                end_time_sec REAL NOT NULL,
                data_kind TEXT NOT NULL,
                subject_count INTEGER,
                person_path_json TEXT,
                confidence REAL,
                evidence_json TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(camera_id) REFERENCES cameras(id) ON DELETE CASCADE
            );
            """
        )
        init_observations(conn)
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(floor_data)").fetchall()}
        # Older experimental DBs had these columns. Leaving them is harmless; new inserts do not use them.
        if "zone_context_json" in columns or "change_reason" in columns:
            print("DB note: old floor_data columns exist, but this tracker writes only the finalized fields.")


def get_or_create_camera(name: str, source_path: str) -> int:
    with connect_db() as conn:
        row = conn.execute("SELECT id FROM cameras WHERE name = ?", (name,)).fetchone()
        if row:
            conn.execute("UPDATE cameras SET source_path = ? WHERE id = ?", (source_path, row["id"]))
            return int(row["id"])
        cur = conn.execute(
            "INSERT INTO cameras (name, source_path, created_at) VALUES (?, ?, ?)",
            (name, source_path, now_iso()),
        )
        return int(cur.lastrowid)


def insert_floor_segment(camera_id: int, segment: dict[str, Any], evidence: dict[str, Any], metadata: dict[str, Any]) -> int:
    subjects = segment["subjects"]
    start_time = min(s["start_time_sec"] for s in subjects)
    end_time = max(s["end_time_sec"] for s in subjects)
    confidence_values = [s.get("confidence", 0.0) for s in subjects if s.get("confidence") is not None]
    confidence = sum(confidence_values) / len(confidence_values) if confidence_values else None
    with connect_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO floor_data
                (camera_id, start_time_sec, end_time_sec, data_kind, subject_count,
                 person_path_json, confidence, evidence_json, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                camera_id,
                start_time,
                end_time,
                "person_activity_segment",
                len(subjects),
                json.dumps(segment),
                confidence,
                json.dumps(evidence),
                json.dumps(metadata),
                now_iso(),
            ),
        )
        return int(cur.lastrowid)


def save_subject_group(
    camera_id: int,
    subjects: list["ActiveSubject"],
    frame: np.ndarray | None,
    run_dir: Path,
    saved_count: int,
    metadata: dict[str, Any],
    frame_time_sec: float | None = None,
) -> tuple[int | None, int]:
    if not subjects:
        return None, saved_count
    segment = {"subjects": [subject.to_json() for subject in subjects]}
    evidence = {"frames": []}
    if frame is not None:
        if frame_time_sec is None:
            raise ValueError("A saved evidence frame requires its actual source timestamp.")
        evidence_path = run_dir / f"evidence_{saved_count + 1:04d}_{int(frame_time_sec * 1000)}.jpg"
        cv2.imwrite(str(evidence_path), frame)
        evidence["frames"].append({"time_sec": round(frame_time_sec, 3), "path": str(evidence_path)})
    row_id = insert_floor_segment(camera_id=camera_id, segment=segment, evidence=evidence, metadata=metadata)
    return row_id, saved_count + 1


def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bbox_overlap(a: list[float], b: list[float]) -> tuple[float, float]:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    iou = intersection / union if union > 0 else 0.0
    containment = intersection / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0
    return iou, containment


def matching_pose_distance(a: dict[str, Any], b: dict[str, Any]) -> tuple[int, float | None]:
    keypoints_a = a.get("keypoints") or []
    keypoints_b = b.get("keypoints") or []
    distances = []
    for point_a, point_b in zip(keypoints_a, keypoints_b):
        if len(point_a) < 3 or len(point_b) < 3 or point_a[2] < 0.35 or point_b[2] < 0.35:
            continue
        distances.append(dist((float(point_a[0]), float(point_a[1])), (float(point_b[0]), float(point_b[1]))))
    if not distances:
        return 0, None
    x1, y1, x2, y2 = a["bbox"]
    diagonal = max(1.0, math.hypot(x2 - x1, y2 - y1))
    return len(distances), float(np.median(distances)) / diagonal


def same_origin_nested_boxes(a: list[float], b: list[float]) -> bool:
    _, containment = bbox_overlap(a, b)
    if containment < 0.92:
        return False
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    min_width = max(1.0, min(ax2 - ax1, bx2 - bx1))
    min_height = max(1.0, min(ay2 - ay1, by2 - by1))
    return abs(ax1 - bx1) / min_width <= 0.08 and abs(ay1 - by1) / min_height <= 0.08


def detections_are_duplicates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    iou, containment = bbox_overlap(a["bbox"], b["bbox"])
    if iou >= 0.88:
        return True
    if containment < 0.88:
        return False
    shared_keypoints, pose_distance = matching_pose_distance(a, b)
    if same_origin_nested_boxes(a["bbox"], b["bbox"]) and (
        shared_keypoints < 4 or (pose_distance is not None and pose_distance <= 0.08)
    ):
        return True
    return shared_keypoints >= 4 and pose_distance is not None and pose_distance <= 0.08


def suppress_duplicate_detections(
    detections: list[dict[str, Any]],
    preferred_ids: set[str],
) -> list[dict[str, Any]]:
    ranked = sorted(
        enumerate(detections),
        key=lambda item: (
            item[1]["internal_id"] in preferred_ids,
            float(item[1].get("conf", 0.0)),
        ),
        reverse=True,
    )
    kept: list[tuple[int, dict[str, Any]]] = []
    for original_index, detection in ranked:
        if any(detections_are_duplicates(detection, existing) for _, existing in kept):
            continue
        kept.append((original_index, detection))
    return [detection for _, detection in sorted(kept, key=lambda item: item[0])]


def smooth_point(prev: tuple[float, float] | None, curr: tuple[float, float], alpha: float) -> tuple[float, float]:
    if prev is None:
        return curr
    return (prev[0] * (1.0 - alpha) + curr[0] * alpha, prev[1] * (1.0 - alpha) + curr[1] * alpha)


def clamped_crop(frame: np.ndarray, bbox: list[float]) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 - x1 < 12 or y2 - y1 < 24:
        return None
    return frame[y1:y2, x1:x2]


def normalized_hist(values: np.ndarray) -> list[float]:
    total = float(values.sum())
    if total <= 0:
        return []
    return [round(float(v / total), 5) for v in values.flatten()]


def hsv_hist(crop: np.ndarray) -> list[float]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256])
    return normalized_hist(hist)


def keypoint_xyc(keypoints: list[list[float]] | np.ndarray | None, idx: int, min_conf: float = 0.25) -> tuple[float, float] | None:
    if keypoints is None or len(keypoints) <= idx:
        return None
    x, y, conf = keypoints[idx]
    if float(conf) < min_conf:
        return None
    return float(x), float(y)


def keypoint_region_crop(
    frame: np.ndarray,
    points: list[tuple[float, float]],
    x_pad: float = 0.28,
    y_pad: float = 0.22,
) -> np.ndarray | None:
    if len(points) < 2:
        return None
    h, w = frame.shape[:2]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    bw = max(18.0, x2 - x1)
    bh = max(24.0, y2 - y1)
    x1 = int(round(x1 - bw * x_pad))
    x2 = int(round(x2 + bw * x_pad))
    y1 = int(round(y1 - bh * y_pad))
    y2 = int(round(y2 + bh * y_pad))
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 - x1 < 12 or y2 - y1 < 16:
        return None
    return frame[y1:y2, x1:x2]


def pose_part_crops(frame: np.ndarray, keypoints: list[list[float]] | np.ndarray | None) -> dict[str, np.ndarray]:
    if keypoints is None:
        return {}
    left_shoulder = keypoint_xyc(keypoints, 5)
    right_shoulder = keypoint_xyc(keypoints, 6)
    left_hip = keypoint_xyc(keypoints, 11)
    right_hip = keypoint_xyc(keypoints, 12)
    left_knee = keypoint_xyc(keypoints, 13)
    right_knee = keypoint_xyc(keypoints, 14)
    left_ankle = keypoint_xyc(keypoints, 15)
    right_ankle = keypoint_xyc(keypoints, 16)

    crops = {}
    upper_points = [p for p in (left_shoulder, right_shoulder, left_hip, right_hip) if p is not None]
    lower_points = [p for p in (left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle) if p is not None]
    body_points = upper_points + [p for p in (left_knee, right_knee, left_ankle, right_ankle) if p is not None]
    upper = keypoint_region_crop(frame, upper_points, x_pad=0.35, y_pad=0.28)
    lower = keypoint_region_crop(frame, lower_points, x_pad=0.30, y_pad=0.18)
    body = keypoint_region_crop(frame, body_points, x_pad=0.25, y_pad=0.18)
    if upper is not None:
        crops["upper"] = upper
    if lower is not None:
        crops["lower"] = lower
    if body is not None:
        crops["body"] = body
    return crops


def color_name(crop: np.ndarray) -> str:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)
    sat = float(np.median(pixels[:, 1]))
    val = float(np.median(pixels[:, 2]))
    if val < 85:
        return "dark"
    if sat < 35:
        return "light" if val > 170 else "gray"
    saturated = pixels[pixels[:, 1] >= 45]
    if len(saturated) < max(10, len(pixels) * 0.08):
        return "light" if val > 170 else "gray"
    hue = float(np.median(saturated[:, 0])) * 2.0
    if hue < 20 or hue >= 340:
        return "red"
    if hue < 45:
        return "orange"
    if hue < 70:
        return "yellow"
    if hue < 165:
        return "green"
    if hue < 255:
        return "blue"
    if hue < 290:
        return "purple"
    return "pink"


def appearance_from_bbox(
    frame: np.ndarray,
    bbox: list[float],
    keypoints: list[list[float]] | np.ndarray | None = None,
) -> dict[str, Any] | None:
    pose_crops = pose_part_crops(frame, keypoints)
    if pose_crops.get("upper") is not None and pose_crops.get("body") is not None:
        upper = pose_crops["upper"]
        lower = pose_crops.get("lower", pose_crops["upper"])
        body = pose_crops["body"]
        return {
            "full_hsv_hist": hsv_hist(body),
            "upper_hsv_hist": hsv_hist(upper),
            "lower_hsv_hist": hsv_hist(lower),
            "upper_color": color_name(upper),
            "lower_color": color_name(lower),
            "sample_method": "pose_keypoint_regions",
            "visible_parts": sorted(pose_crops.keys()),
        }

    crop = clamped_crop(frame, bbox)
    if crop is None:
        return None
    height, width = crop.shape[:2]
    x1 = int(width * 0.22)
    x2 = max(x1 + 1, int(width * 0.78))
    body = crop[int(height * 0.08): max(int(height * 0.96), int(height * 0.08) + 1), x1:x2]
    upper = crop[int(height * 0.12): max(int(height * 0.58), int(height * 0.12) + 1), x1:x2]
    lower = crop[int(height * 0.48): max(int(height * 0.95), int(height * 0.48) + 1), x1:x2]
    return {
        "full_hsv_hist": hsv_hist(body),
        "upper_hsv_hist": hsv_hist(upper),
        "lower_hsv_hist": hsv_hist(lower),
        "upper_color": color_name(upper),
        "lower_color": color_name(lower),
        "sample_method": "center_body_regions",
        "visible_parts": ["body", "upper", "lower"],
    }


def mean_hist(samples: list[dict[str, Any]], key: str) -> list[float]:
    hists = [sample.get(key) for sample in samples if sample.get(key)]
    if not hists:
        return []
    return normalized_hist(np.array(hists, dtype=np.float32).mean(axis=0))


def mode_text(values: list[str]) -> str | None:
    if not values:
        return None
    return max(set(values), key=values.count)


def hist_similarity(a: list[float] | None, b: list[float] | None) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    return float(sum(min(float(x), float(y)) for x, y in zip(a, b)))


def subject_appearance_signature(subject: "ActiveSubject") -> dict[str, Any] | None:
    if not subject.appearance_samples:
        return None
    return {
        "full_hsv_hist": mean_hist(subject.appearance_samples, "full_hsv_hist"),
        "upper_hsv_hist": mean_hist(subject.appearance_samples, "upper_hsv_hist"),
        "lower_hsv_hist": mean_hist(subject.appearance_samples, "lower_hsv_hist"),
        "upper_color": mode_text([sample.get("upper_color") for sample in subject.appearance_samples if sample.get("upper_color")]),
        "lower_color": mode_text([sample.get("lower_color") for sample in subject.appearance_samples if sample.get("lower_color")]),
    }


def appearance_similarity(a: dict[str, Any] | None, b: dict[str, Any] | None) -> float | None:
    if not a or not b:
        return None
    weighted = []
    weights = []
    for key, weight in (("upper_hsv_hist", 0.45), ("full_hsv_hist", 0.35), ("lower_hsv_hist", 0.20)):
        score = hist_similarity(a.get(key), b.get(key))
        if score is not None:
            weighted.append(score * weight)
            weights.append(weight)
    if not weights:
        return None
    score = sum(weighted) / sum(weights)
    if a.get("upper_color") and a.get("upper_color") == b.get("upper_color"):
        score = min(1.0, score + 0.04)
    if a.get("lower_color") and a.get("lower_color") == b.get("lower_color"):
        score = min(1.0, score + 0.02)
    return score


def foot_from_pose(bbox: list[float], keypoints: np.ndarray | None, min_kpt_conf: float = 0.25) -> tuple[float, float, str]:
    x1, y1, x2, y2 = bbox
    if keypoints is not None and len(keypoints) >= 17:
        candidates = []
        for idx in (15, 16):  # ankles
            x, y, c = keypoints[idx]
            if c >= min_kpt_conf:
                candidates.append((float(x), float(y)))
        if candidates:
            return (
                sum(p[0] for p in candidates) / len(candidates),
                sum(p[1] for p in candidates) / len(candidates),
                "ankle_midpoint" if len(candidates) == 2 else "single_ankle",
            )
        candidates = []
        for idx in (13, 14):  # knees
            x, y, c = keypoints[idx]
            if c >= min_kpt_conf:
                candidates.append((float(x), float(y)))
        if candidates:
            knee_x = sum(p[0] for p in candidates) / len(candidates)
            knee_y = sum(p[1] for p in candidates) / len(candidates)
            return knee_x, min(float(y2), knee_y + (float(y2) - knee_y) * 0.75), "knee_projected"
        candidates = []
        for idx in (11, 12):  # hips
            x, y, c = keypoints[idx]
            if c >= min_kpt_conf:
                candidates.append((float(x), float(y)))
        if candidates:
            hip_x = sum(p[0] for p in candidates) / len(candidates)
            hip_y = sum(p[1] for p in candidates) / len(candidates)
            return hip_x, min(float(y2), hip_y + (float(y2) - hip_y) * 0.9), "hip_projected"
    return (float(x1 + x2) / 2.0, float(y2), "bbox_bottom_center")


def direction_from_points(points: list[dict[str, Any]]) -> str:
    if len(points) < 2:
        return "static"
    x0, y0 = points[0]["foot"]
    x1, y1 = points[-1]["foot"]
    dx, dy = x1 - x0, y1 - y0
    if math.hypot(dx, dy) < 25:
        return "static"
    if abs(dx) > abs(dy) * 1.4:
        return "left_to_right" if dx > 0 else "right_to_left"
    if abs(dy) > abs(dx) * 1.4:
        return "top_to_bottom" if dy > 0 else "bottom_to_top"
    horizontal = "right" if dx > 0 else "left"
    vertical = "down" if dy > 0 else "up"
    return f"{vertical}_{horizontal}"


def path_length(points: list[dict[str, Any]]) -> float:
    total = 0.0
    for a, b in zip(points, points[1:]):
        total += dist(tuple(a["foot"]), tuple(b["foot"]))
    return total


def reconnect_score(
    subject: "ActiveSubject",
    detection: dict[str, Any],
    t_sec: float,
    max_gap_sec: float,
    max_distance_px: float,
    min_score: float,
    min_appearance: float | None,
) -> tuple[float, dict[str, float]] | None:
    if subject.last_smoothed is None:
        return None
    gap = t_sec - subject.last_time
    if gap < 0 or gap > max_gap_sec:
        return None
    appearance = appearance_similarity(subject_appearance_signature(subject), detection.get("appearance"))
    nested_continuation = (
        gap <= 1.5
        and subject.last_bbox is not None
        and detection.get("bbox") is not None
        and same_origin_nested_boxes(subject.last_bbox, detection["bbox"])
    )
    if nested_continuation and (appearance is None or appearance >= 0.35):
        distance_px = dist(subject.last_smoothed, detection["foot"])
        speed = distance_px / max(gap, 0.1)
        return 0.92, {
            "score": 0.92,
            "appearance": round(appearance, 4) if appearance is not None else -1.0,
            "time_gap_sec": round(gap, 3),
            "distance_px": round(distance_px, 2),
            "speed_px_sec": round(speed, 2),
            "nested_bbox_continuation": 1.0,
        }
    distance_px = dist(subject.last_smoothed, detection["foot"])
    if distance_px > max_distance_px:
        return None
    speed = distance_px / max(gap, 0.1)
    if speed > 520.0:
        return None

    if min_appearance is not None and (appearance is None or appearance < min_appearance):
        return None
    # A strong appearance disagreement is also evidence against a short-range reconnect.
    if appearance is not None and appearance < 0.35:
        return None
    if appearance is None and gap > 1.5:
        return None

    time_score = 1.0 - min(gap / max(max_gap_sec, 0.1), 1.0)
    distance_score = 1.0 - min(distance_px / max(max_distance_px, 1.0), 1.0)
    speed_score = 1.0 - min(speed / 520.0, 1.0)
    appearance_score = appearance if appearance is not None else 0.50
    score = 0.44 * appearance_score + 0.22 * time_score + 0.24 * distance_score + 0.10 * speed_score
    if score < min_score:
        return None
    return score, {
        "score": round(score, 4),
        "appearance": round(appearance, 4) if appearance is not None else -1.0,
        "time_gap_sec": round(gap, 3),
        "distance_px": round(distance_px, 2),
        "speed_px_sec": round(speed, 2),
    }


def assign_reconnections(
    detections: list[dict[str, Any]],
    active: dict[str, "ActiveSubject"],
    lost: dict[str, "ActiveSubject"],
    seen_active_ids: set[str],
    t_sec: float,
    max_missing_sec: float,
    max_reconnect_px: float,
    reid_lost_sec: float,
    reid_max_distance_px: float,
    reid_min_appearance: float,
    reid_min_score: float,
) -> dict[int, tuple[str, str, dict[str, float]]]:
    ambiguity_margin = 0.04
    candidates: list[tuple[float, int, str, str, dict[str, float]]] = []
    for detection_index, detection in enumerate(detections):
        if detection["internal_id"] in active:
            continue
        for old_id, subject in active.items():
            if old_id in seen_active_ids:
                continue
            scored = reconnect_score(
                subject,
                detection,
                t_sec,
                max_missing_sec,
                max_reconnect_px,
                min_score=0.50,
                min_appearance=None,
            )
            if scored:
                score, reason = scored
                candidates.append((score, detection_index, "active", old_id, reason))
        for old_id, subject in lost.items():
            scored = reconnect_score(
                subject,
                detection,
                t_sec,
                reid_lost_sec,
                reid_max_distance_px,
                min_score=reid_min_score,
                min_appearance=reid_min_appearance,
            )
            if scored:
                score, reason = scored
                candidates.append((score, detection_index, "lost", old_id, reason))

    detection_scores: dict[int, list[float]] = {}
    source_scores: dict[tuple[str, str], list[float]] = {}
    for score, detection_index, source_pool, old_id, _ in candidates:
        detection_scores.setdefault(detection_index, []).append(score)
        source_scores.setdefault((source_pool, old_id), []).append(score)
    for scores in detection_scores.values():
        scores.sort(reverse=True)
    for scores in source_scores.values():
        scores.sort(reverse=True)

    # Highest-confidence one-to-one assignment avoids detection-order identity swaps.
    # Ambiguous pairs are deliberately left split instead of risking a false merge.
    matches: dict[int, tuple[str, str, dict[str, float]]] = {}
    used_sources: set[tuple[str, str]] = set()
    for score, detection_index, source_pool, old_id, reason in sorted(candidates, key=lambda item: item[0], reverse=True):
        source = (source_pool, old_id)
        if detection_index in matches or source in used_sources:
            continue
        det_options = detection_scores[detection_index]
        source_options = source_scores[source]
        det_ambiguous = len(det_options) > 1 and score - det_options[1] < ambiguity_margin
        source_ambiguous = len(source_options) > 1 and score - source_options[1] < ambiguity_margin
        if det_ambiguous or source_ambiguous:
            continue
        matches[detection_index] = (source_pool, old_id, reason)
        used_sources.add(source)
    return matches


@dataclass
class ActiveSubject:
    internal_id: str
    subject_ref: str
    first_time: float
    last_time: float
    last_seen_frame: int
    points: list[dict[str, Any]] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    foot_sources: dict[str, int] = field(default_factory=dict)
    uncertain_notes: list[str] = field(default_factory=list)
    last_bbox: list[float] | None = None
    last_smoothed: tuple[float, float] | None = None
    appearance_samples: list[dict[str, Any]] = field(default_factory=list)
    last_appearance_sample_time: float | None = None
    reconnect_events: list[dict[str, Any]] = field(default_factory=list)
    tracker_ids: list[str] = field(default_factory=list)

    def add_point(
        self,
        t: float,
        frame_idx: int,
        foot: tuple[float, float],
        bbox: list[float],
        conf: float,
        foot_source: str,
        smooth_alpha: float,
        min_point_gap_px: float,
        appearance: dict[str, Any] | None = None,
        keypoints: list[list[float]] | None = None,
    ) -> None:
        smoothed = smooth_point(self.last_smoothed, foot, smooth_alpha)
        should_append = not self.points
        if self.points:
            should_append = dist(tuple(self.points[-1]["foot"]), smoothed) >= min_point_gap_px or t - self.points[-1]["t"] >= 1.0
        if should_append:
            point = {
                "t": round(t, 3),
                "frame": int(frame_idx),
                "foot": [round(smoothed[0], 2), round(smoothed[1], 2)],
                "raw_foot": [round(foot[0], 2), round(foot[1], 2)],
                "bbox": [round(float(v), 2) for v in bbox],
                "foot_source": foot_source,
                "confidence": round(float(conf), 4),
            }
            if keypoints:
                point["keypoints"] = [
                    [round(float(x), 2), round(float(y), 2), round(float(c), 4)]
                    for x, y, c in keypoints
                ]
            self.points.append(point)
        self.last_time = t
        self.last_seen_frame = frame_idx
        self.last_smoothed = smoothed
        self.last_bbox = bbox
        self.confidences.append(float(conf))
        self.foot_sources[foot_source] = self.foot_sources.get(foot_source, 0) + 1
        if appearance and (
            self.last_appearance_sample_time is None or t - self.last_appearance_sample_time >= 0.5
        ):
            self.appearance_samples.append(appearance)
            self.appearance_samples = self.appearance_samples[-48:]
            self.last_appearance_sample_time = t

    def net_displacement_px(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return dist(tuple(self.points[0]["foot"]), tuple(self.points[-1]["foot"]))

    def path_distance_px(self) -> float:
        return path_length(self.points)

    def to_json(self) -> dict[str, Any]:
        confidence = sum(self.confidences) / len(self.confidences) if self.confidences else 0.0
        quality = "good"
        if self.uncertain_notes:
            quality = "uncertain"
        if not self.uncertain_notes and (len(self.points) < 2 or self.path_distance_px() < 20):
            quality = "low_motion"
        upper_color = mode_text([sample.get("upper_color") for sample in self.appearance_samples if sample.get("upper_color")])
        lower_color = mode_text([sample.get("lower_color") for sample in self.appearance_samples if sample.get("lower_color")])
        appearance_signature = None
        if self.appearance_samples:
            appearance_signature = {
                "sample_count": len(self.appearance_samples),
                "upper_color": upper_color,
                "lower_color": lower_color,
                "full_hsv_hist": mean_hist(self.appearance_samples, "full_hsv_hist"),
                "upper_hsv_hist": mean_hist(self.appearance_samples, "upper_hsv_hist"),
                "lower_hsv_hist": mean_hist(self.appearance_samples, "lower_hsv_hist"),
                "note": "day-level clothing appearance only; not a permanent identity",
            }
        payload = {
            "subject_ref": self.subject_ref,
            "start_time_sec": round(self.first_time, 3),
            "end_time_sec": round(self.last_time, 3),
            "start_point": self.points[0]["foot"] if self.points else None,
            "end_point": self.points[-1]["foot"] if self.points else None,
            "direction": direction_from_points(self.points),
            "distance_px": round(self.path_distance_px(), 2),
            "net_displacement_px": round(self.net_displacement_px(), 2),
            "path_points": self.points,
            "quality": quality,
            "confidence": round(confidence, 4),
            "tracking_notes": self.uncertain_notes,
            "reconnect_events": self.reconnect_events,
            "tracker_ids": self.tracker_ids or [self.internal_id],
            "foot_source_counts": self.foot_sources,
        }
        if appearance_signature:
            payload["appearance_signature"] = appearance_signature
        return payload


class PreviewState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.status = "waiting"

    def update(self, frame: np.ndarray, status: str) -> None:
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return
        with self.lock:
            self.jpeg = encoded.tobytes()
            self.status = status


class PreviewHandler(BaseHTTPRequestHandler):
    state: PreviewState | None = None

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/":
            body = b"""<!doctype html><html><head><title>CCTV Activity Preview</title>
<style>body{margin:0;background:#111;color:#eee;font-family:Arial}header{padding:10px 14px;background:#20242a}img{max-width:100%;display:block;margin:auto}</style>
</head><body><header>Live detection preview</header><img src="/stream"></body></html>"""
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path != "/stream" or PreviewHandler.state is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        while True:
            with PreviewHandler.state.lock:
                jpeg = PreviewHandler.state.jpeg
            if jpeg:
                try:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
            time.sleep(0.1)


def start_preview_server(port: int, state: PreviewState) -> ThreadingHTTPServer:
    PreviewHandler.state = state
    server = ThreadingHTTPServer(("127.0.0.1", port), PreviewHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def draw_subjects(frame: np.ndarray, subjects: dict[str, ActiveSubject], detections: list[dict[str, Any]], saved_count: int) -> np.ndarray:
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        foot_x, foot_y = [int(v) for v in det["foot"]]
        label = det["subject_ref"]
        color = (0, 220, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.circle(out, (foot_x, foot_y), 5, (0, 0, 255), -1)
        cv2.putText(out, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    for subject in subjects.values():
        pts = [tuple(map(int, p["foot"])) for p in subject.points[-40:]]
        for a, b in zip(pts, pts[1:]):
            cv2.line(out, a, b, (0, 255, 0), 2)
    cv2.putText(out, f"active={len(subjects)} saved={saved_count}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return out


def patch_torch_load() -> None:
    try:
        import torch
        import torch.serialization

        if hasattr(torch, "load") and not hasattr(torch, "_cctv_original_load_patched"):
            original_load = torch.load

            def custom_load(*args: Any, **kwargs: Any) -> Any:
                kwargs["weights_only"] = False
                return original_load(*args, **kwargs)

            torch.load = custom_load
            torch.serialization.load = custom_load
            torch._cctv_original_load_patched = True
    except Exception:
        pass


def load_yolo(model_path: str):
    patch_torch_load()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed in this environment. Install it in .venv-sam with: "
            ".\\.venv-sam\\Scripts\\python.exe -m pip install ultralytics"
        ) from exc
    return YOLO(model_path)


def extract_detections(result: Any) -> list[dict[str, Any]]:
    detections = []
    if result is None or result.boxes is None:
        return detections
    boxes = result.boxes
    keypoints = result.keypoints.data.cpu().numpy() if getattr(result, "keypoints", None) is not None and result.keypoints is not None else None
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.arange(len(xyxy))
    for idx, bbox in enumerate(xyxy):
        kpts = keypoints[idx] if keypoints is not None and idx < len(keypoints) else None
        foot_x, foot_y, foot_source = foot_from_pose([float(v) for v in bbox], kpts)
        detections.append(
            {
                "internal_id": str(int(ids[idx])),
                "bbox": [float(v) for v in bbox],
                "conf": float(confs[idx]),
                "foot": (foot_x, foot_y),
                "foot_source": foot_source,
                "keypoints": [[float(x), float(y), float(c)] for x, y, c in kpts] if kpts is not None else None,
            }
        )
    return detections


def process_video(args: argparse.Namespace) -> dict[str, Any]:
    init_db()
    video_path = Path(args.video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    camera_id = get_or_create_camera(args.camera, str(video_path))
    if args.frame_stride < 1 or args.start_sec < 0:
        raise ValueError("Frame stride must be positive and start time must be nonnegative.")
    run_dir = OUTPUT_DIR / f"{args.camera}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    run_id = run_dir.name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Camera: {args.camera} db_id={camera_id}")
    print(f"Video: {video_path}")
    print(f"Output: {run_dir}")
    print(f"Loading model: {args.model}")
    if str(args.device) == "cpu":
        import torch
        torch.set_num_threads(getattr(args, "cpu_threads", 4))
    model = load_yolo(args.model)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if args.start_sec:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start_sec * fps))
    end_sec = args.end_sec if args.end_sec is not None else (total_frames / fps if total_frames else None)
    if total_frames:
        end_sec = min(end_sec, total_frames / fps)
    if end_sec is None or not math.isfinite(end_sec) or end_sec <= args.start_sec:
        cap.release()
        raise ValueError("Video must have a finite, nonempty processing range.")
    config = {**vars(args), "fps": fps, "end_sec": end_sec,
              "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    with connect_db() as conn:
        conn.execute("INSERT INTO tracking_runs (run_id,camera_id,source_path,source_kind,config_json,status) VALUES (?,?,?,?,?,?)",
                     (run_id, camera_id, str(video_path), getattr(args, "source_kind", "unknown"),
                      json.dumps(config, default=str), "processing"))

    preview_state = PreviewState()
    preview_server = None
    if args.preview:
        preview_server = start_preview_server(args.preview_port, preview_state)
        url = f"http://127.0.0.1:{args.preview_port}"
        print(f"Preview: {url}")
        if args.open_browser:
            webbrowser.open(url)

    active: dict[str, ActiveSubject] = {}
    lost: dict[str, ActiveSubject] = {}
    next_subject_num = 1
    saved_count = 0
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
    last_log = time.time()
    tracker_arg = args.tracker if args.tracker != "none" else None
    writer = None
    observation_batch = []
    processed_frames = 0
    processing_started = time.perf_counter()

    def flush_observations():
        if observation_batch:
            with connect_db() as conn:
                conn.executemany("INSERT INTO frame_observations VALUES (?,?,?,?)", observation_batch)
            observation_batch.clear()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            t_sec = frame_idx / fps
            if t_sec >= end_sec:
                break
            if frame_idx % args.frame_stride != 0:
                continue

            results = model.track(
                frame,
                persist=True,
                classes=[0],
                conf=args.conf,
                imgsz=args.imgsz,
                tracker=tracker_arg,
                verbose=False,
                device=args.device,
            )
            detections = extract_detections(results[0] if results else None)
            detections = suppress_duplicate_detections(detections, set(active))
            for detection in detections:
                detection["appearance"] = appearance_from_bbox(
                    frame,
                    detection["bbox"],
                    detection.get("keypoints"),
                )
            seen_ids = {
                detection["internal_id"]
                for detection in detections
                if detection["internal_id"] in active
            }
            reconnect_matches = assign_reconnections(
                detections=detections,
                active=active,
                lost=lost,
                seen_active_ids=seen_ids,
                t_sec=t_sec,
                max_missing_sec=args.max_missing_sec,
                max_reconnect_px=args.max_reconnect_px,
                reid_lost_sec=args.reid_lost_sec,
                reid_max_distance_px=args.reid_max_distance_px,
                reid_min_appearance=args.reid_min_appearance,
                reid_min_score=args.reid_min_score,
            )
            drawn_detections = []
            for detection_index, det in enumerate(detections):
                internal_id = det["internal_id"]
                match = reconnect_matches.get(detection_index)
                if internal_id not in active and match is not None:
                    source_pool, old_id, reason = match
                    subject = active.pop(old_id) if source_pool == "active" else lost.pop(old_id)
                    reconnect_event = {
                        "subject_ref": subject.subject_ref,
                        "source_pool": source_pool,
                        "old_tracker_id": old_id,
                        "new_tracker_id": internal_id,
                        "before_time_sec": round(subject.last_time, 3),
                        "reconnect_time_sec": round(t_sec, 3),
                        "before_frame": int(subject.last_seen_frame),
                        "reconnect_frame": int(frame_idx),
                        "before_bbox": [round(float(v), 2) for v in subject.last_bbox] if subject.last_bbox else None,
                        "reconnect_bbox": [round(float(v), 2) for v in det["bbox"]],
                        **reason,
                    }
                    subject.reconnect_events.append(reconnect_event)
                    if internal_id not in subject.tracker_ids:
                        subject.tracker_ids.append(internal_id)
                    subject.uncertain_notes.append(
                        f"tracker_id_changed_reidentified_{source_pool}:{json.dumps(reason)}"
                    )
                    subject.internal_id = internal_id
                    active[internal_id] = subject
                    print(
                        f"[{t_sec:8.2f}s] REID-{source_pool.upper()} {subject.subject_ref} "
                        f"old_id={old_id} new_id={internal_id} score={reason.get('score')}"
                    )
                seen_ids.add(internal_id)
                if internal_id not in active:
                    subject_ref = f"subject_{next_subject_num}"
                    next_subject_num += 1
                    active[internal_id] = ActiveSubject(
                        internal_id=internal_id,
                        subject_ref=subject_ref,
                        first_time=t_sec,
                        last_time=t_sec,
                        last_seen_frame=frame_idx,
                        tracker_ids=[internal_id],
                    )
                    print(f"[{t_sec:8.2f}s] NEW {subject_ref}")
                subject = active[internal_id]
                subject.add_point(
                    t=t_sec,
                    frame_idx=frame_idx,
                    foot=det["foot"],
                    bbox=det["bbox"],
                    conf=det["conf"],
                    foot_source=det["foot_source"],
                    smooth_alpha=args.smooth_alpha,
                    min_point_gap_px=args.min_point_gap_px,
                    appearance=det.get("appearance"),
                    keypoints=det.get("keypoints"),
                )
                drawn_detections.append({**det, "subject_ref": subject.subject_ref})

            observation_batch.append((run_id, frame_idx, t_sec, json.dumps([
                {"subject_ref": det["subject_ref"], "tracker_id": det["internal_id"],
                 "foot": list(active[det["internal_id"]].last_smoothed), "raw_foot": list(det["foot"]),
                 "bbox": det["bbox"], "confidence": det["conf"], "foot_source": det["foot_source"]}
                for det in drawn_detections])))
            processed_frames += 1
            if len(observation_batch) >= 32:
                flush_observations()

            stale_ids = []
            for internal_id, subject in active.items():
                missing_sec = (frame_idx - subject.last_seen_frame) / fps
                if internal_id not in seen_ids and missing_sec > args.max_missing_sec:
                    stale_ids.append(internal_id)

            ended_subjects = []
            for internal_id in stale_ids:
                subject = active.pop(internal_id)
                if subject.last_time - subject.first_time < args.min_segment_sec:
                    print(f"[{t_sec:8.2f}s] DROP {subject.subject_ref} short segment")
                    continue
                if args.discard_low_motion and subject.path_distance_px() < args.min_movement_px:
                    print(f"[{t_sec:8.2f}s] DROP {subject.subject_ref} low movement")
                    continue
                lost[internal_id] = subject
                print(f"[{t_sec:8.2f}s] LOST {subject.subject_ref} id={internal_id}")

            expired_lost_ids = [
                lost_id
                for lost_id, subject in lost.items()
                if t_sec - subject.last_time > args.reid_lost_sec
            ]
            for lost_id in expired_lost_ids:
                ended_subjects.append(lost.pop(lost_id))

            if ended_subjects:
                row_id, saved_count = save_subject_group(
                    camera_id=camera_id,
                    subjects=ended_subjects,
                    frame=frame,
                    run_dir=run_dir,
                    saved_count=saved_count,
                    frame_time_sec=t_sec,
                    metadata={
                        "run_id": run_id,
                        "video_path": str(video_path),
                        "model": args.model,
                        "tracker": args.tracker,
                        "fps": fps,
                        "frame_stride": args.frame_stride,
                        "processing_start_sec": args.start_sec,
                        "processing_end_sec": end_sec,
                        "preview_fps": max(1.0, fps / args.frame_stride),
                        "storage_policy": "saved_after_lost_reid_window_expired",
                        "reid_policy": {
                            "mode": "global_one_to_one_motion_appearance",
                            "lost_sec": args.reid_lost_sec,
                            "max_distance_px": args.reid_max_distance_px,
                            "min_appearance": args.reid_min_appearance,
                            "min_score": args.reid_min_score,
                        },
                    },
                )
                refs = ", ".join(s.subject_ref for s in ended_subjects)
                print(
                    f"[{t_sec:8.2f}s] SAVE row={row_id} subjects={len(ended_subjects)} "
                    f"refs={refs}"
                )

            preview = draw_subjects(frame, active, drawn_detections, saved_count)
            if args.preview:
                preview_state.update(preview, f"t={t_sec:.2f}s saved={saved_count}")
            if args.save_preview_video:
                # Lazy-create the writer after the first preview frame.
                if writer is None:
                    h, w = preview.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(run_dir / "preview.mp4"), fourcc, fps / args.frame_stride, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError("Could not create tracking preview video.")
                writer.write(preview)

            if time.time() - last_log > 2.0:
                print(f"[{t_sec:8.2f}s] active={len(active)} lost={len(lost)} detections={len(detections)} saved={saved_count}")
                last_log = time.time()

    except Exception as exc:
        with connect_db() as conn:
            conn.execute("UPDATE tracking_runs SET status='failed',error=?,finished_at=CURRENT_TIMESTAMP WHERE run_id=?", (str(exc), run_id))
        raise
    finally:
        flush_observations()
        cap.release()
        if writer is not None:
            writer.release()
        if preview_server is not None:
            preview_server.shutdown()

    # Flush remaining active subjects at end of video/range.
    final_subjects = []
    for subject in list(active.values()) + list(lost.values()):
        if subject.last_time - subject.first_time < args.min_segment_sec:
            continue
        if args.discard_low_motion and subject.path_distance_px() < args.min_movement_px:
            continue
        final_subjects.append(subject)

    if final_subjects:
        row_id, saved_count = save_subject_group(
            camera_id=camera_id,
            subjects=final_subjects,
            frame=None,
            run_dir=run_dir,
            saved_count=saved_count,
            metadata={
                "run_id": run_id,
                "video_path": str(video_path),
                "model": args.model,
                "tracker": args.tracker,
                "fps": fps,
                "frame_stride": args.frame_stride,
                "processing_start_sec": args.start_sec,
                "processing_end_sec": end_sec,
                "preview_fps": max(1.0, fps / args.frame_stride),
                "storage_policy": "saved_at_video_end",
                "reid_policy": {
                    "mode": "global_one_to_one_motion_appearance",
                    "lost_sec": args.reid_lost_sec,
                    "max_distance_px": args.reid_max_distance_px,
                    "min_appearance": args.reid_min_appearance,
                    "min_score": args.reid_min_score,
                },
            },
        )
        print(f"FINAL SAVE row={row_id} subjects={len(final_subjects)}")
    print(f"Done. Saved {saved_count} floor_data rows.")
    elapsed = time.perf_counter() - processing_started
    config.update({"processed_frames": processed_frames, "elapsed_sec": elapsed,
                   "processing_fps": processed_frames / max(elapsed, 1e-9)})
    with connect_db() as conn:
        conn.execute("UPDATE tracking_runs SET status='complete',config_json=?,finished_at=CURRENT_TIMESTAMP WHERE run_id=?",
                     (json.dumps(config, default=str), run_id))
    result = {"camera": args.camera, "run_id": run_id, "run_dir": str(run_dir.resolve()),
              "processed_frames": processed_frames, "elapsed_sec": elapsed}
    (run_dir / "run.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def default_model_path() -> str:
    candidates = [
        BASE_DIR / "models" / "yolo11m-pose.pt",
        BASE_DIR.parent.parent / "backend" / "workstation" / "person" / "models" / "yolo11m-pose.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "yolo11m-pose.pt"


def main() -> None:
    global DB_PATH, OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Read CCTV video, smooth YOLO pose detections, and store floor activity segments.")
    parser.add_argument("camera", help="Camera/source name, for example D23")
    parser.add_argument("video", help="Video path")
    parser.add_argument("--model", default=default_model_path(), help="YOLO pose model path")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--source-kind", choices=["real", "synthetic", "unknown"], default="unknown")
    parser.add_argument("--tracker", default="botsort.yaml", help="Ultralytics tracker config: botsort.yaml, bytetrack.yaml, or none")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, 0, etc.")
    parser.add_argument("--cpu-threads", type=int, default=4, help="Limit CPU thread oversubscription")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--frame-stride", type=int, default=3, help="Process every Nth frame")
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float)
    parser.add_argument("--max-missing-sec", type=float, default=4.0)
    parser.add_argument("--max-reconnect-px", type=float, default=250.0)
    parser.add_argument("--min-segment-sec", type=float, default=1.0)
    parser.add_argument("--min-movement-px", type=float, default=35.0, help="Path-distance threshold used only with --discard-low-motion")
    parser.add_argument("--min-point-gap-px", type=float, default=12.0)
    parser.add_argument("--smooth-alpha", type=float, default=0.45)
    parser.add_argument("--save-low-motion", action="store_true", help="Deprecated compatibility flag; low-motion tracks are now saved by default")
    parser.add_argument("--discard-low-motion", action="store_true", help="Opt in to discarding short-path tracks as noise; not recommended for occupancy")
    parser.add_argument("--reid-lost-sec", type=float, default=30.0, help="How long to keep lost tracklets for appearance-assisted reconnect")
    parser.add_argument("--reid-max-distance-px", type=float, default=680.0)
    parser.add_argument("--reid-min-appearance", type=float, default=0.68)
    parser.add_argument("--reid-min-score", type=float, default=0.58)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-port", type=int, default=8770)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--save-preview-video", action="store_true")
    args = parser.parse_args()
    DB_PATH, OUTPUT_DIR = args.db.resolve(), args.output_dir.resolve()
    process_video(args)


if __name__ == "__main__":
    main()
