from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "cctv_maps.sqlite3"
PROFILE_SAMPLE_DIR = BASE_DIR / "identity_samples"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_identity_db() -> None:
    with connect_db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS worker_identity_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id INTEGER NOT NULL,
                run_name TEXT NOT NULL,
                video_path TEXT NOT NULL,
                start_time_sec REAL,
                end_time_sec REAL,
                config_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(camera_id) REFERENCES cameras(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS worker_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                identity_run_id INTEGER NOT NULL,
                profile_label TEXT NOT NULL,
                camera_id INTEGER NOT NULL,
                start_time_sec REAL NOT NULL,
                end_time_sec REAL NOT NULL,
                segment_count INTEGER NOT NULL,
                sample_count INTEGER NOT NULL,
                profile_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(identity_run_id) REFERENCES worker_identity_runs(id) ON DELETE CASCADE,
                FOREIGN KEY(camera_id) REFERENCES cameras(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS worker_profile_segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                identity_run_id INTEGER NOT NULL,
                profile_id INTEGER NOT NULL,
                floor_data_id INTEGER NOT NULL,
                subject_ref TEXT NOT NULL,
                link_confidence REAL NOT NULL,
                link_status TEXT NOT NULL,
                link_reason_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(identity_run_id) REFERENCES worker_identity_runs(id) ON DELETE CASCADE,
                FOREIGN KEY(profile_id) REFERENCES worker_profiles(id) ON DELETE CASCADE,
                FOREIGN KEY(floor_data_id) REFERENCES floor_data(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS worker_profile_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                identity_run_id INTEGER NOT NULL,
                profile_id INTEGER,
                floor_data_id INTEGER NOT NULL,
                subject_ref TEXT NOT NULL,
                time_sec REAL NOT NULL,
                sample_path TEXT NOT NULL,
                quality REAL NOT NULL,
                feature_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(identity_run_id) REFERENCES worker_identity_runs(id) ON DELETE CASCADE,
                FOREIGN KEY(profile_id) REFERENCES worker_profiles(id) ON DELETE SET NULL,
                FOREIGN KEY(floor_data_id) REFERENCES floor_data(id) ON DELETE CASCADE
            );
            """
        )


def normalized_hist(values: np.ndarray) -> list[float]:
    total = float(values.sum())
    if total <= 0:
        return []
    return [float(v / total) for v in values.flatten()]


def hist_intersection(a: list[float] | None, b: list[float] | None) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    return float(sum(min(float(x), float(y)) for x, y in zip(a, b)))


def cosine_similarity(a: list[float] | None, b: list[float] | None) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    av = np.array(a, dtype=np.float32)
    bv = np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1e-9:
        return None
    return float(np.dot(av, bv) / denom)


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


def crop_from_bbox(frame: np.ndarray, bbox: list[float], pad: float = 0.08) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw = x2 - x1
    bh = y2 - y1
    x1 = int(round(x1 - bw * pad))
    x2 = int(round(x2 + bw * pad))
    y1 = int(round(y1 - bh * pad))
    y2 = int(round(y2 + bh * pad))
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 - x1 < 16 or y2 - y1 < 32:
        return None
    return frame[y1:y2, x1:x2]


def keypoint_xyc(keypoints: list[list[float]] | None, idx: int, min_conf: float = 0.25) -> tuple[float, float] | None:
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


def pose_body_parts(frame: np.ndarray, keypoints: list[list[float]] | None) -> dict[str, np.ndarray]:
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


def body_parts(crop: np.ndarray) -> dict[str, np.ndarray]:
    h, w = crop.shape[:2]
    x1 = int(w * 0.20)
    x2 = max(x1 + 1, int(w * 0.80))
    return {
        "body": crop[int(h * 0.08): max(int(h * 0.96), int(h * 0.08) + 1), x1:x2],
        "upper": crop[int(h * 0.12): max(int(h * 0.58), int(h * 0.12) + 1), x1:x2],
        "lower": crop[int(h * 0.48): max(int(h * 0.95), int(h * 0.48) + 1), x1:x2],
    }


def hsv_feature(crop: np.ndarray) -> list[float]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 4, 4], [0, 180, 0, 256, 0, 256])
    return normalized_hist(hist)


def texture_feature(crop: np.ndarray) -> list[float]:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 128), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=3)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    bins = np.floor((angle % 180.0) / 20.0).astype(np.int32)
    features = []
    for y in range(0, 128, 32):
        for x in range(0, 64, 32):
            cell_bins = bins[y:y + 32, x:x + 32]
            cell_mag = mag[y:y + 32, x:x + 32]
            hist = np.zeros(9, dtype=np.float32)
            for idx in range(9):
                hist[idx] = float(cell_mag[cell_bins == idx].sum())
            features.extend(normalized_hist(hist))
    return features


def crop_quality(crop: np.ndarray, bbox: list[float], frame_shape: tuple[int, int, int]) -> float:
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    area_ratio = min((bw * bh) / max(1.0, frame_w * frame_h), 0.08) / 0.08
    aspect = bw / bh
    aspect_score = 1.0 - min(abs(aspect - 0.42) / 0.7, 1.0)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = min(blur / 180.0, 1.0)
    edge_penalty = 0.0
    if x1 <= 2 or y1 <= 2 or x2 >= frame_w - 2 or y2 >= frame_h - 2:
        edge_penalty = 0.25
    return max(0.0, min(1.0, 0.35 * area_ratio + 0.25 * aspect_score + 0.40 * blur_score - edge_penalty))


def extract_sample_feature(
    frame: np.ndarray,
    bbox: list[float],
    keypoints: list[list[float]] | None = None,
) -> dict[str, Any] | None:
    pose_parts = pose_body_parts(frame, keypoints)
    crop = crop_from_bbox(frame, bbox)
    if crop is None and not pose_parts:
        return None
    if pose_parts.get("upper") is not None and pose_parts.get("body") is not None:
        parts = {
            "body": pose_parts["body"],
            "upper": pose_parts["upper"],
            "lower": pose_parts.get("lower", pose_parts["upper"]),
        }
        sample_method = "pose_keypoint_hsv_texture_quality_v1"
        quality_crop = parts["body"]
    else:
        if crop is None:
            return None
        parts = body_parts(crop)
        sample_method = "center_body_hsv_texture_quality_v1"
        quality_crop = crop
    quality = crop_quality(quality_crop, bbox, frame.shape)
    return {
        "quality": quality,
        "crop": crop if crop is not None else quality_crop,
        "upper_color": color_name(parts["upper"]),
        "lower_color": color_name(parts["lower"]),
        "body_hsv": hsv_feature(parts["body"]),
        "upper_hsv": hsv_feature(parts["upper"]),
        "lower_hsv": hsv_feature(parts["lower"]),
        "texture": texture_feature(parts["body"]),
        "sample_method": sample_method,
        "visible_parts": sorted(pose_parts.keys()) if pose_parts else ["body", "upper", "lower"],
    }


def weighted_mean(features: list[dict[str, Any]], key: str) -> list[float]:
    items = [(f.get(key), max(float(f.get("quality") or 0.0), 0.05)) for f in features if f.get(key)]
    if not items:
        return []
    arr = np.array([item[0] for item in items], dtype=np.float32)
    weights = np.array([item[1] for item in items], dtype=np.float32)
    mean = np.average(arr, axis=0, weights=weights)
    norm = float(mean.sum())
    if norm > 1e-9 and key.endswith("_hsv"):
        mean = mean / norm
    return [float(v) for v in mean]


def weighted_mode(features: list[dict[str, Any]], key: str) -> str | None:
    scores: dict[str, float] = {}
    for feature in features:
        value = feature.get(key)
        if not value:
            continue
        scores[value] = scores.get(value, 0.0) + max(float(feature.get("quality") or 0.0), 0.05)
    if not scores:
        return None
    return max(scores.items(), key=lambda item: item[1])[0]


@dataclass
class Segment:
    segment_id: str
    floor_data_id: int
    subject_ref: str
    start: float
    end: float
    points: list[dict[str, Any]]
    raw_appearance_signature: dict[str, Any] | None = None
    features: list[dict[str, Any]] = field(default_factory=list)
    profile_feature: dict[str, Any] = field(default_factory=dict)

    @property
    def first_xy(self) -> tuple[float, float]:
        p = self.points[0]["foot"]
        return float(p[0]), float(p[1])

    @property
    def last_xy(self) -> tuple[float, float]:
        p = self.points[-1]["foot"]
        return float(p[0]), float(p[1])


def get_camera(conn: sqlite3.Connection, camera_name: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM cameras WHERE name=?", (camera_name,)).fetchone()
    if not row:
        raise ValueError(f"Camera not found: {camera_name}")
    return row


def load_segments(conn: sqlite3.Connection, camera_id: int, start: float | None, end: float | None) -> list[Segment]:
    t_s = 0.0 if start is None else start
    t_e = 999999.0 if end is None else end
    rows = conn.execute(
        """
        SELECT id, start_time_sec, end_time_sec, person_path_json
        FROM floor_data
        WHERE camera_id=? AND data_kind='person_activity_segment'
          AND end_time_sec >= ? AND start_time_sec <= ?
        ORDER BY start_time_sec, id
        """,
        (camera_id, t_s, t_e),
    ).fetchall()
    segments: list[Segment] = []
    for row in rows:
        data = json.loads(row["person_path_json"] or "{}")
        for subject in data.get("subjects", []):
            points = [p for p in subject.get("path_points", []) if t_s <= float(p.get("t", 0.0)) <= t_e and p.get("bbox")]
            if not points:
                continue
            segments.append(
                Segment(
                    segment_id=f"row{row['id']}_{subject['subject_ref']}",
                    floor_data_id=int(row["id"]),
                    subject_ref=subject["subject_ref"],
                    start=float(subject["start_time_sec"]),
                    end=float(subject["end_time_sec"]),
                    points=sorted(points, key=lambda p: float(p["t"])),
                    raw_appearance_signature=subject.get("appearance_signature"),
                )
            )
    return segments


def sample_points(points: list[dict[str, Any]], max_samples: int) -> list[dict[str, Any]]:
    if len(points) <= max_samples:
        return points
    indices = np.linspace(0, len(points) - 1, max_samples).round().astype(int)
    return [points[int(i)] for i in indices]


def build_segment_features(
    segments: list[Segment],
    video_path: Path,
    output_dir: Path,
    max_samples_per_segment: int,
) -> list[dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    sample_records: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for segment in segments:
            for idx, point in enumerate(sample_points(segment.points, max_samples_per_segment), 1):
                frame_idx = int(point.get("frame") or 0)
                if frame_idx <= 0:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
                ok, frame = cap.read()
                if not ok:
                    continue
                feature = extract_sample_feature(
                    frame,
                    [float(v) for v in point["bbox"]],
                    point.get("keypoints"),
                )
                if not feature:
                    continue
                sample_path = output_dir / f"{segment.segment_id}_{idx:02d}_{int(float(point['t']) * 1000)}.jpg"
                cv2.imwrite(str(sample_path), feature.pop("crop"))
                feature["time_sec"] = float(point["t"])
                feature["frame"] = frame_idx
                feature["sample_path"] = str(sample_path)
                segment.features.append(feature)
                sample_records.append(
                    {
                        "segment_id": segment.segment_id,
                        "floor_data_id": segment.floor_data_id,
                        "subject_ref": segment.subject_ref,
                        "time_sec": float(point["t"]),
                        "sample_path": str(sample_path),
                        "quality": float(feature["quality"]),
                        "feature": feature,
                    }
                )
            segment.profile_feature = apply_raw_signature_prior(
                aggregate_features(segment.features),
                segment.raw_appearance_signature,
            )
    finally:
        cap.release()
    return sample_records


def aggregate_features(features: list[dict[str, Any]]) -> dict[str, Any]:
    if not features:
        return {}
    return {
        "sample_count": len(features),
        "mean_quality": float(sum(float(f.get("quality") or 0.0) for f in features) / len(features)),
        "upper_color": weighted_mode(features, "upper_color"),
        "lower_color": weighted_mode(features, "lower_color"),
        "body_hsv": weighted_mean(features, "body_hsv"),
        "upper_hsv": weighted_mean(features, "upper_hsv"),
        "lower_hsv": weighted_mean(features, "lower_hsv"),
        "texture": weighted_mean(features, "texture"),
        "sample_method": "center_body_hsv_texture_quality_v1",
    }


def apply_raw_signature_prior(feature: dict[str, Any], raw_signature: dict[str, Any] | None) -> dict[str, Any]:
    if not raw_signature:
        return feature
    merged = dict(feature)
    if raw_signature.get("upper_color"):
        merged["upper_color"] = raw_signature["upper_color"]
    if raw_signature.get("lower_color"):
        merged["lower_color"] = raw_signature["lower_color"]
    merged["raw_signature_prior"] = {
        "upper_color": raw_signature.get("upper_color"),
        "lower_color": raw_signature.get("lower_color"),
        "sample_count": raw_signature.get("sample_count"),
    }
    return merged


def appearance_score(a: dict[str, Any], b: dict[str, Any]) -> float:
    scores = []
    for key, weight in [("upper_hsv", 0.30), ("body_hsv", 0.28), ("lower_hsv", 0.18)]:
        score = hist_intersection(a.get(key), b.get(key))
        if score is not None:
            scores.append(score * weight)
    texture = cosine_similarity(a.get("texture"), b.get("texture"))
    if texture is not None:
        scores.append(max(0.0, texture) * 0.16)
    color_bonus = 0.0
    if a.get("upper_color") and a.get("upper_color") == b.get("upper_color"):
        color_bonus += 0.05
    if a.get("lower_color") and a.get("lower_color") == b.get("lower_color"):
        color_bonus += 0.03
    base = sum(scores) / 0.92 if scores else 0.0
    return max(0.0, min(1.0, base + color_bonus))


def segment_overlap(a: Segment, b: Segment) -> bool:
    return not (a.end < b.start or b.end < a.start)


def point_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def evaluate_pair(a: Segment, b: Segment, config: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    if segment_overlap(a, b):
        return 0.0, {"status": "rejected", "reason": "segments overlap in time"}
    first, second = (a, b) if a.end <= b.start else (b, a)
    gap = second.start - first.end
    if gap > config["max_gap_sec"]:
        return 0.0, {"status": "rejected", "reason": "time gap too long", "time_gap": gap}
    app = appearance_score(first.profile_feature, second.profile_feature)
    dist_px = point_distance(first.last_xy, second.first_xy)
    max_dist = config["max_distance_px_appearance"] if app >= config["appearance_gate"] else config["max_distance_px"]
    if dist_px > max_dist:
        return 0.0, {"status": "rejected", "reason": "distance too far", "distance_px": dist_px, "appearance": app}
    speed = dist_px / max(gap, 0.1)
    max_speed = config["max_speed_px_sec"] * (2.5 if app >= config["appearance_gate"] and gap <= 2.0 else 1.6)
    if speed > max_speed:
        return 0.0, {"status": "rejected", "reason": "speed too high", "speed_px_sec": speed, "appearance": app}
    time_score = 1.0 - min(gap / max(config["max_gap_sec"], 0.1), 1.0)
    dist_score = 1.0 - min(dist_px / max(max_dist, 1.0), 1.0)
    speed_score = 1.0 - min(speed / max(max_speed, 1.0), 1.0)
    score = 0.48 * app + 0.20 * time_score + 0.22 * dist_score + 0.10 * speed_score
    status = "confirmed" if score >= config["confirmed_threshold"] else "probable" if score >= config["probable_threshold"] else "uncertain"
    return score, {
        "status": status,
        "appearance": app,
        "time_gap": gap,
        "distance_px": dist_px,
        "speed_px_sec": speed,
        "score_parts": {
            "appearance": app,
            "time": time_score,
            "distance": dist_score,
            "speed": speed_score,
        },
    }


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def cluster_segments(segments: list[Segment], config: dict[str, Any]) -> tuple[list[list[Segment]], dict[tuple[str, str], dict[str, Any]]]:
    by_id = {segment.segment_id: segment for segment in segments}
    pairs = []
    pair_reasons: dict[tuple[str, str], dict[str, Any]] = {}
    for i, a in enumerate(segments):
        for b in segments[i + 1:]:
            score, reason = evaluate_pair(a, b, config)
            if score >= config["probable_threshold"]:
                pairs.append((score, a.segment_id, b.segment_id, reason))
                pair_reasons[tuple(sorted((a.segment_id, b.segment_id)))] = reason
    pairs.sort(reverse=True, key=lambda item: item[0])

    uf = UnionFind([segment.segment_id for segment in segments])
    cluster_members: dict[str, set[str]] = {segment.segment_id: {segment.segment_id} for segment in segments}
    accepted: dict[tuple[str, str], dict[str, Any]] = {}

    for score, a_id, b_id, reason in pairs:
        ra = uf.find(a_id)
        rb = uf.find(b_id)
        if ra == rb:
            continue
        merged = cluster_members[ra] | cluster_members[rb]
        has_overlap = False
        merged_list = [by_id[sid] for sid in merged]
        for i, a in enumerate(merged_list):
            for b in merged_list[i + 1:]:
                if segment_overlap(a, b):
                    has_overlap = True
                    break
            if has_overlap:
                break
        if has_overlap:
            continue
        uf.union(ra, rb)
        root = uf.find(ra)
        other = rb if root == ra else ra
        cluster_members[root] = merged
        cluster_members.pop(other, None)
        accepted[tuple(sorted((a_id, b_id)))] = {**reason, "link_score": score}

    groups: dict[str, list[Segment]] = {}
    for segment in segments:
        groups.setdefault(uf.find(segment.segment_id), []).append(segment)
    clusters = [sorted(group, key=lambda s: (s.start, s.end)) for group in groups.values()]
    clusters.sort(key=lambda group: min(s.start for s in group))
    return clusters, accepted


def aggregate_profile(segments: list[Segment]) -> dict[str, Any]:
    features = [segment.profile_feature for segment in segments if segment.profile_feature]
    profile = {
        "segment_ids": [segment.segment_id for segment in segments],
        "sample_count": sum(int(feature.get("sample_count") or 0) for feature in features),
        "mean_quality": float(sum(float(feature.get("mean_quality") or 0.0) for feature in features) / max(len(features), 1)),
        "upper_color": weighted_mode(features, "upper_color"),
        "lower_color": weighted_mode(features, "lower_color"),
        "body_hsv": weighted_mean(features, "body_hsv"),
        "upper_hsv": weighted_mean(features, "upper_hsv"),
        "lower_hsv": weighted_mean(features, "lower_hsv"),
        "texture": weighted_mean(features, "texture"),
        "note": "Day/session visible-worker profile; not permanent human identity.",
    }
    raw_priors = [segment.raw_appearance_signature for segment in segments if segment.raw_appearance_signature]
    upper_prior = weighted_mode(
        [
            {"upper_color": prior.get("upper_color"), "quality": float(prior.get("sample_count") or 1)}
            for prior in raw_priors
            if prior.get("upper_color")
        ],
        "upper_color",
    )
    lower_prior = weighted_mode(
        [
            {"lower_color": prior.get("lower_color"), "quality": float(prior.get("sample_count") or 1)}
            for prior in raw_priors
            if prior.get("lower_color")
        ],
        "lower_color",
    )
    if upper_prior:
        profile["upper_color"] = upper_prior
    if lower_prior:
        profile["lower_color"] = lower_prior
    return profile


def default_config() -> dict[str, Any]:
    return {
        "max_gap_sec": 45.0,
        "max_distance_px": 300.0,
        "max_distance_px_appearance": 680.0,
        "max_speed_px_sec": 180.0,
        "appearance_gate": 0.70,
        "probable_threshold": 0.54,
        "confirmed_threshold": 0.72,
        "max_samples_per_segment": 10,
    }


def clear_previous_identity(camera_id: int, run_name: str | None) -> None:
    with connect_db() as conn:
        if run_name:
            rows = conn.execute(
                "SELECT id FROM worker_identity_runs WHERE camera_id=? AND run_name=?",
                (camera_id, run_name),
            ).fetchall()
        else:
            rows = conn.execute("SELECT id FROM worker_identity_runs WHERE camera_id=?", (camera_id,)).fetchall()
        for row in rows:
            run_id = row["id"]
            profile_ids = [p["id"] for p in conn.execute("SELECT id FROM worker_profiles WHERE identity_run_id=?", (run_id,)).fetchall()]
            conn.execute("DELETE FROM worker_profile_samples WHERE identity_run_id=?", (run_id,))
            conn.execute("DELETE FROM worker_profile_segments WHERE identity_run_id=?", (run_id,))
            for profile_id in profile_ids:
                conn.execute("DELETE FROM worker_profiles WHERE id=?", (profile_id,))
            conn.execute("DELETE FROM worker_identity_runs WHERE id=?", (row["id"],))
        conn.execute(
            """
            DELETE FROM worker_profile_samples
            WHERE identity_run_id NOT IN (SELECT id FROM worker_identity_runs)
            """
        )
        conn.execute(
            """
            DELETE FROM worker_profile_segments
            WHERE identity_run_id NOT IN (SELECT id FROM worker_identity_runs)
            """
        )
        conn.execute(
            """
            DELETE FROM worker_profiles
            WHERE identity_run_id NOT IN (SELECT id FROM worker_identity_runs)
            """
        )


def write_identity_results(
    camera_id: int,
    run_name: str,
    video_path: Path,
    t_start: float | None,
    t_end: float | None,
    config: dict[str, Any],
    clusters: list[list[Segment]],
    accepted_links: dict[tuple[str, str], dict[str, Any]],
    sample_records: list[dict[str, Any]],
) -> int:
    created = now_iso()
    with connect_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO worker_identity_runs
                (camera_id, run_name, video_path, start_time_sec, end_time_sec, config_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (camera_id, run_name, str(video_path), t_start, t_end, json.dumps(config), created),
        )
        identity_run_id = int(cur.lastrowid)
        profile_by_segment: dict[str, int] = {}
        for idx, cluster in enumerate(clusters, 1):
            profile = aggregate_profile(cluster)
            label = f"worker_profile_{idx}"
            profile_start = min(segment.start for segment in cluster)
            profile_end = max(segment.end for segment in cluster)
            cur = conn.execute(
                """
                INSERT INTO worker_profiles
                    (identity_run_id, profile_label, camera_id, start_time_sec, end_time_sec,
                     segment_count, sample_count, profile_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identity_run_id,
                    label,
                    camera_id,
                    profile_start,
                    profile_end,
                    len(cluster),
                    int(profile.get("sample_count") or 0),
                    json.dumps(profile),
                    created,
                ),
            )
            profile_id = int(cur.lastrowid)
            for segment in cluster:
                profile_by_segment[segment.segment_id] = profile_id
            for segment in cluster:
                link_reasons = []
                link_confidence = 1.0 if len(cluster) == 1 else 0.60
                link_status = "raw_segment"
                for other in cluster:
                    if other.segment_id == segment.segment_id:
                        continue
                    key = tuple(sorted((segment.segment_id, other.segment_id)))
                    if key in accepted_links:
                        link_reasons.append(accepted_links[key])
                        link_confidence = max(link_confidence, float(accepted_links[key].get("link_score") or 0.0))
                        link_status = accepted_links[key].get("status") or "probable"
                conn.execute(
                    """
                    INSERT INTO worker_profile_segments
                        (identity_run_id, profile_id, floor_data_id, subject_ref,
                         link_confidence, link_status, link_reason_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        identity_run_id,
                        profile_id,
                        segment.floor_data_id,
                        segment.subject_ref,
                        link_confidence,
                        link_status,
                        json.dumps(link_reasons),
                        created,
                    ),
                )
        for sample in sample_records:
            conn.execute(
                """
                INSERT INTO worker_profile_samples
                    (identity_run_id, profile_id, floor_data_id, subject_ref,
                     time_sec, sample_path, quality, feature_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identity_run_id,
                    profile_by_segment.get(sample["segment_id"]),
                    sample["floor_data_id"],
                    sample["subject_ref"],
                    sample["time_sec"],
                    sample["sample_path"],
                    sample["quality"],
                    json.dumps(sample["feature"]),
                    created,
                ),
            )
        return identity_run_id


def build_identity_profiles(args: argparse.Namespace) -> str:
    init_identity_db()
    config = default_config()
    config["max_samples_per_segment"] = args.max_samples_per_segment
    with connect_db() as conn:
        camera = get_camera(conn, args.camera)
        camera_id = int(camera["id"])
        video_path = Path(args.video or camera["source_path"]).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        if args.replace:
            clear_previous_identity(camera_id, args.run_name)
        segments = load_segments(conn, camera_id, args.start_sec, args.end_sec)
    if not segments:
        return f"No raw tracking segments found for camera {args.camera}."

    run_name = args.run_name or f"{args.camera}_{int(time.time())}"
    sample_dir = PROFILE_SAMPLE_DIR / run_name
    sample_records = build_segment_features(segments, video_path, sample_dir, config["max_samples_per_segment"])
    usable_segments = [segment for segment in segments if segment.profile_feature]
    clusters, accepted_links = cluster_segments(usable_segments, config)
    identity_run_id = write_identity_results(
        camera_id=camera_id,
        run_name=run_name,
        video_path=video_path,
        t_start=args.start_sec,
        t_end=args.end_sec,
        config=config,
        clusters=clusters,
        accepted_links=accepted_links,
        sample_records=sample_records,
    )
    linked_count = sum(max(0, len(cluster) - 1) for cluster in clusters)
    lines = [
        f"Identity run {identity_run_id}: {run_name}",
        f"Camera: {args.camera}",
        f"Raw segments: {len(segments)}",
        f"Usable segments with samples: {len(usable_segments)}",
        f"Profiles: {len(clusters)}",
        f"Linked segment joins: {linked_count}",
        f"Samples saved: {len(sample_records)}",
        f"Sample folder: {sample_dir}",
        "",
        "Profiles:",
    ]
    for idx, cluster in enumerate(clusters, 1):
        profile = aggregate_profile(cluster)
        lines.append(
            f"  worker_profile_{idx}: {len(cluster)} segment(s), "
            f"{min(s.start for s in cluster):.1f}-{max(s.end for s in cluster):.1f}s, "
            f"upper={profile.get('upper_color')}, lower={profile.get('lower_color')}, "
            f"samples={profile.get('sample_count')}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build day/session worker identity profiles from raw CCTV tracking rows.")
    parser.add_argument("camera", help="Camera/source name already present in floor_data")
    parser.add_argument("--video", help="Video path. Defaults to cameras.source_path")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--start-sec", type=float, default=None)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--max-samples-per-segment", type=int, default=10)
    parser.add_argument("--replace", action="store_true", help="Delete existing identity runs for this camera/run name before writing")
    args = parser.parse_args()
    print(build_identity_profiles(args))


if __name__ == "__main__":
    main()
