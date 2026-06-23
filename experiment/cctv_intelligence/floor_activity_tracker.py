from __future__ import annotations

import argparse
import json
import math
import sqlite3
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "cctv_maps.sqlite3"
OUTPUT_DIR = BASE_DIR / "activity_runs"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
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
) -> tuple[int | None, int]:
    if not subjects:
        return None, saved_count
    segment = {"subjects": [subject.to_json() for subject in subjects]}
    evidence = {"frames": []}
    if frame is not None:
        evidence_path = run_dir / f"evidence_{saved_count + 1:04d}_{int(max(s.last_time for s in subjects) * 1000)}.jpg"
        cv2.imwrite(str(evidence_path), frame)
        evidence["frames"].append({"time_sec": round(max(s.last_time for s in subjects), 3), "path": str(evidence_path)})
    row_id = insert_floor_segment(camera_id=camera_id, segment=segment, evidence=evidence, metadata=metadata)
    return row_id, saved_count + 1


def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def smooth_point(prev: tuple[float, float] | None, curr: tuple[float, float], alpha: float) -> tuple[float, float]:
    if prev is None:
        return curr
    return (prev[0] * (1.0 - alpha) + curr[0] * alpha, prev[1] * (1.0 - alpha) + curr[1] * alpha)


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


def find_reconnect_subject(
    active: dict[str, "ActiveSubject"],
    seen_ids: set[str],
    foot: tuple[float, float],
    frame_idx: int,
    fps: float,
    max_missing_sec: float,
    max_reconnect_px: float,
) -> str | None:
    best_id = None
    best_distance = max_reconnect_px
    for internal_id, subject in active.items():
        if internal_id in seen_ids or subject.last_smoothed is None:
            continue
        missing_sec = (frame_idx - subject.last_seen_frame) / fps
        if missing_sec > max_missing_sec:
            continue
        candidate_distance = dist(subject.last_smoothed, foot)
        if candidate_distance < best_distance:
            best_distance = candidate_distance
            best_id = internal_id
    return best_id


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
    ) -> None:
        smoothed = smooth_point(self.last_smoothed, foot, smooth_alpha)
        should_append = not self.points
        if self.points:
            should_append = dist(tuple(self.points[-1]["foot"]), smoothed) >= min_point_gap_px or t - self.points[-1]["t"] >= 1.0
        if should_append:
            self.points.append(
                {
                    "t": round(t, 3),
                    "frame": int(frame_idx),
                    "foot": [round(smoothed[0], 2), round(smoothed[1], 2)],
                    "raw_foot": [round(foot[0], 2), round(foot[1], 2)],
                    "bbox": [round(float(v), 2) for v in bbox],
                    "foot_source": foot_source,
                    "confidence": round(float(conf), 4),
                }
            )
        self.last_time = t
        self.last_seen_frame = frame_idx
        self.last_smoothed = smoothed
        self.last_bbox = bbox
        self.confidences.append(float(conf))
        self.foot_sources[foot_source] = self.foot_sources.get(foot_source, 0) + 1

    def movement_px(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return dist(tuple(self.points[0]["foot"]), tuple(self.points[-1]["foot"]))

    def to_json(self) -> dict[str, Any]:
        confidence = sum(self.confidences) / len(self.confidences) if self.confidences else 0.0
        quality = "good"
        if self.uncertain_notes:
            quality = "uncertain"
        if len(self.points) < 2 or self.movement_px() < 20:
            quality = "low_motion"
        return {
            "subject_ref": self.subject_ref,
            "start_time_sec": round(self.first_time, 3),
            "end_time_sec": round(self.last_time, 3),
            "start_point": self.points[0]["foot"] if self.points else None,
            "end_point": self.points[-1]["foot"] if self.points else None,
            "direction": direction_from_points(self.points),
            "distance_px": round(path_length(self.points), 2),
            "path_points": self.points,
            "quality": quality,
            "confidence": round(confidence, 4),
            "tracking_notes": self.uncertain_notes,
            "foot_source_counts": self.foot_sources,
        }


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
            }
        )
    return detections


def process_video(args: argparse.Namespace) -> None:
    init_db()
    video_path = Path(args.video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    camera_id = get_or_create_camera(args.camera, str(video_path))
    run_dir = OUTPUT_DIR / f"{args.camera}_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Camera: {args.camera} db_id={camera_id}")
    print(f"Video: {video_path}")
    print(f"Output: {run_dir}")
    print(f"Loading model: {args.model}")
    model = load_yolo(args.model)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if args.start_sec:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start_sec * fps))
    end_sec = args.end_sec if args.end_sec is not None else (total_frames / fps if total_frames else None)

    preview_state = PreviewState()
    preview_server = None
    if args.preview:
        preview_server = start_preview_server(args.preview_port, preview_state)
        url = f"http://127.0.0.1:{args.preview_port}"
        print(f"Preview: {url}")
        if args.open_browser:
            webbrowser.open(url)

    active: dict[str, ActiveSubject] = {}
    next_subject_num = 1
    saved_count = 0
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
    last_log = time.time()
    tracker_arg = args.tracker if args.tracker != "none" else None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            t_sec = frame_idx / fps
            if end_sec is not None and t_sec > end_sec:
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
            seen_ids = set()
            drawn_detections = []
            for det in detections:
                internal_id = det["internal_id"]
                if internal_id not in active:
                    reconnect_id = find_reconnect_subject(
                        active=active,
                        seen_ids=seen_ids,
                        foot=det["foot"],
                        frame_idx=frame_idx,
                        fps=fps,
                        max_missing_sec=args.max_missing_sec,
                        max_reconnect_px=args.max_reconnect_px,
                    )
                    if reconnect_id is not None:
                        subject = active.pop(reconnect_id)
                        subject.uncertain_notes.append("tracker_id_changed_reconnected_by_nearby_foot_point")
                        active[internal_id] = subject
                        print(f"[{t_sec:8.2f}s] RECONNECT {subject.subject_ref} old_id={reconnect_id} new_id={internal_id}")
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
                )
                drawn_detections.append({**det, "subject_ref": subject.subject_ref})

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
                if subject.movement_px() < args.min_movement_px and not args.save_low_motion:
                    print(f"[{t_sec:8.2f}s] DROP {subject.subject_ref} low movement")
                    continue
                ended_subjects.append(subject)

            if ended_subjects:
                row_id, saved_count = save_subject_group(
                    camera_id=camera_id,
                    subjects=ended_subjects,
                    frame=frame,
                    run_dir=run_dir,
                    saved_count=saved_count,
                    metadata={
                        "video_path": str(video_path),
                        "model": args.model,
                        "tracker": args.tracker,
                        "fps": fps,
                        "frame_stride": args.frame_stride,
                        "storage_policy": "saved_when_visible_subject_segment_ended",
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
                if not hasattr(process_video, "_writer"):
                    h, w = preview.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    process_video._writer = cv2.VideoWriter(str(run_dir / "preview.mp4"), fourcc, max(1.0, fps / args.frame_stride), (w, h))
                process_video._writer.write(preview)

            if time.time() - last_log > 2.0:
                print(f"[{t_sec:8.2f}s] active={len(active)} detections={len(detections)} saved={saved_count}")
                last_log = time.time()

    finally:
        cap.release()
        writer = getattr(process_video, "_writer", None)
        if writer is not None:
            writer.release()
            delattr(process_video, "_writer")
        if preview_server is not None:
            preview_server.shutdown()

    # Flush remaining active subjects at end of video/range.
    final_subjects = []
    for subject in list(active.values()):
        if subject.last_time - subject.first_time < args.min_segment_sec:
            continue
        if subject.movement_px() < args.min_movement_px and not args.save_low_motion:
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
                "video_path": str(video_path),
                "model": args.model,
                "tracker": args.tracker,
                "fps": fps,
                "frame_stride": args.frame_stride,
                "storage_policy": "saved_at_video_end",
            },
        )
        print(f"FINAL SAVE row={row_id} subjects={len(final_subjects)}")
    print(f"Done. Saved {saved_count} floor_data rows.")


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
    parser = argparse.ArgumentParser(description="Read CCTV video, smooth YOLO pose detections, and store floor activity segments.")
    parser.add_argument("camera", help="Camera/source name, for example D23")
    parser.add_argument("video", help="Video path")
    parser.add_argument("--model", default=default_model_path(), help="YOLO pose model path")
    parser.add_argument("--tracker", default="botsort.yaml", help="Ultralytics tracker config: botsort.yaml, bytetrack.yaml, or none")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, 0, etc.")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--frame-stride", type=int, default=3, help="Process every Nth frame")
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float)
    parser.add_argument("--max-missing-sec", type=float, default=4.0)
    parser.add_argument("--max-reconnect-px", type=float, default=250.0)
    parser.add_argument("--min-segment-sec", type=float, default=1.0)
    parser.add_argument("--min-movement-px", type=float, default=35.0)
    parser.add_argument("--min-point-gap-px", type=float, default=12.0)
    parser.add_argument("--smooth-alpha", type=float, default=0.45)
    parser.add_argument("--save-low-motion", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-port", type=int, default=8770)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--save-preview-video", action="store_true")
    args = parser.parse_args()
    process_video(args)


if __name__ == "__main__":
    main()
