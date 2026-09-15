from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "cctv_maps.sqlite3"


def load_run(camera: str, run_id: str) -> tuple[list[dict[str, Any]], dict[str, Any], list[int]]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT fd.id, fd.person_path_json, fd.metadata_json
            FROM floor_data fd
            JOIN cameras c ON c.id = fd.camera_id
            WHERE c.name = ? AND fd.data_kind = 'person_activity_segment'
            ORDER BY fd.id
            """,
            (camera,),
        ).fetchall()

    subjects: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    row_ids: list[int] = []
    for row in rows:
        row_metadata = json.loads(row["metadata_json"] or "{}")
        if row_metadata.get("run_id") != run_id:
            continue
        row_ids.append(int(row["id"]))
        metadata = row_metadata
        payload = json.loads(row["person_path_json"] or "{}")
        subjects.extend(payload.get("subjects", []))
    if not row_ids:
        raise RuntimeError(f"No floor_data rows found for camera={camera!r}, run_id={run_id!r}")
    return subjects, metadata, row_ids


def read_frame(video_path: Path, time_sec: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_sec) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read {video_path} at {time_sec:.3f}s")
    return frame


def annotate_frame(
    frame: np.ndarray,
    bbox: list[float] | None,
    subject_ref: str,
    tracker_id: str,
    time_sec: float,
    phase: str,
) -> np.ndarray:
    out = frame.copy()
    if bbox:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 255), 4)
        cv2.putText(
            out,
            f"{subject_ref} / tracker {tracker_id}",
            (max(8, x1), max(28, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.rectangle(out, (0, 0), (out.shape[1], 58), (20, 20, 20), -1)
    cv2.putText(
        out,
        f"{phase}  t={time_sec:.3f}s",
        (18, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def fit_height(frame: np.ndarray, height: int) -> np.ndarray:
    if frame.shape[0] == height:
        return frame
    scale = height / frame.shape[0]
    return cv2.resize(frame, (int(round(frame.shape[1] * scale)), height))


def write_event_image(video_path: Path, event: dict[str, Any], output_path: Path) -> None:
    before_time = float(event["before_time_sec"])
    reconnect_time = float(event["reconnect_time_sec"])
    before = annotate_frame(
        read_frame(video_path, before_time),
        event.get("before_bbox"),
        event["subject_ref"],
        str(event["old_tracker_id"]),
        before_time,
        "LAST DETECTION BEFORE GAP",
    )
    after = annotate_frame(
        read_frame(video_path, reconnect_time),
        event.get("reconnect_bbox"),
        event["subject_ref"],
        str(event["new_tracker_id"]),
        reconnect_time,
        "RECONNECTED DETECTION",
    )
    target_height = min(before.shape[0], after.shape[0])
    combined = np.hstack([fit_height(before, target_height), fit_height(after, target_height)])
    footer = np.full((74, combined.shape[1], 3), 24, dtype=np.uint8)
    appearance = event.get("appearance", -1)
    message = (
        f"same {event['subject_ref']} | gap={event.get('time_gap_sec')}s | "
        f"appearance={appearance} | total score={event.get('score')}"
    )
    cv2.putText(footer, message, (18, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(output_path), np.vstack([combined, footer]))


def write_event_clip(
    preview_path: Path,
    event: dict[str, Any],
    output_path: Path,
    processing_start_sec: float,
    padding_sec: float,
) -> bool:
    cap = cv2.VideoCapture(str(preview_path))
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    event_elapsed = float(event["reconnect_time_sec"]) - processing_start_sec
    start = max(0.0, event_elapsed - padding_sec)
    end = max(start, event_elapsed + padding_sec)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    wrote = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        position_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if position_sec > end:
            break
        cv2.rectangle(frame, (0, height - 48), (width, height), (20, 20, 20), -1)
        cv2.putText(
            frame,
            f"Reconnect proof: {event['subject_ref']}  tracker {event['old_tracker_id']} -> {event['new_tracker_id']}",
            (18, height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        wrote = True
    writer.release()
    cap.release()
    if not wrote and output_path.exists():
        output_path.unlink()
    return wrote


def write_detection_samples(
    preview_path: Path,
    output_dir: Path,
    processing_start_sec: float,
    processing_end_sec: float,
    interval_sec: float = 60.0,
) -> list[dict[str, Any]]:
    if not preview_path.exists():
        return []
    cap = cv2.VideoCapture(str(preview_path))
    if not cap.isOpened():
        return []
    duration = max(0.0, processing_end_sec - processing_start_sec)
    elapsed_times = [0.0]
    elapsed = interval_sec
    while elapsed < duration:
        elapsed_times.append(elapsed)
        elapsed += interval_sec
    if duration > 0:
        elapsed_times.append(max(0.0, duration - 0.2))
    samples = []
    for elapsed in elapsed_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, elapsed * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        source_time = processing_start_sec + elapsed
        sample_number = len(samples) + 1
        cv2.rectangle(frame, (0, frame.shape[0] - 48), (frame.shape[1], frame.shape[0]), (20, 20, 20), -1)
        cv2.putText(
            frame,
            f"Detection sample {sample_number}  source t={source_time:.1f}s",
            (18, frame.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        path = output_dir / f"detection_sample_{sample_number:02d}_{int(source_time * 1000)}.jpg"
        cv2.imwrite(str(path), frame)
        samples.append({"time_sec": round(source_time, 3), "image": path.name})
    cap.release()
    return samples


def summarize(
    subjects: list[dict[str, Any]],
    row_ids: list[int],
    analysis_duration_sec: float,
) -> dict[str, Any]:
    events = [event for subject in subjects for event in subject.get("reconnect_events", [])]
    tracker_ids = {
        str(tracker_id)
        for subject in subjects
        for tracker_id in (subject.get("tracker_ids") or [])
    }
    starts = [float(subject["start_time_sec"]) for subject in subjects]
    ends = [float(subject["end_time_sec"]) for subject in subjects]
    scores = [float(event["score"]) for event in events if event.get("score") is not None]
    appearances = [
        float(event["appearance"])
        for event in events
        if event.get("appearance") is not None and float(event["appearance"]) >= 0
    ]
    durations = [float(subject["end_time_sec"]) - float(subject["start_time_sec"]) for subject in subjects]
    persistence_threshold_sec = min(30.0, max(5.0, analysis_duration_sec * 0.5))
    return {
        "database_row_ids": row_ids,
        "saved_subject_reference_count": len(subjects),
        "persistence_threshold_sec": round(persistence_threshold_sec, 3),
        "persistent_reference_count": sum(duration >= persistence_threshold_sec for duration in durations),
        "brief_reference_count_under_10s": sum(duration < 10.0 for duration in durations),
        "raw_tracker_id_count": len(tracker_ids),
        "raw_tracklets_joined": max(0, len(tracker_ids) - len(subjects)),
        "reconnect_event_count": len(events),
        "distinct_tracker_id_merges": sum(
            str(event.get("old_tracker_id")) != str(event.get("new_tracker_id")) for event in events
        ),
        "same_tracker_id_gap_recoveries": sum(
            str(event.get("old_tracker_id")) == str(event.get("new_tracker_id")) for event in events
        ),
        "review_required_reconnects": sum(
            float(event.get("score", 0.0)) < 0.65 or float(event.get("appearance", -1.0)) < 0.60
            for event in events
        ),
        "active_gap_reconnects": sum(event.get("source_pool") == "active" for event in events),
        "lost_track_reconnects": sum(event.get("source_pool") == "lost" for event in events),
        "observed_start_sec": min(starts) if starts else None,
        "observed_end_sec": max(ends) if ends else None,
        "mean_reconnect_score": round(sum(scores) / len(scores), 4) if scores else None,
        "minimum_reconnect_score": round(min(scores), 4) if scores else None,
        "mean_appearance_similarity": round(sum(appearances) / len(appearances), 4) if appearances else None,
        "minimum_appearance_similarity": round(min(appearances), 4) if appearances else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build auditable images, clips, JSON, and Markdown for a tracking run.")
    parser.add_argument("camera")
    parser.add_argument("run_id")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--clip-padding-sec", type=float, default=3.0)
    args = parser.parse_args()

    subjects, metadata, row_ids = load_run(args.camera, args.run_id)
    video_path = (args.video or Path(metadata["video_path"])).resolve()
    run_dir = (args.run_dir or BASE_DIR / "activity_runs" / args.run_id).resolve()
    proof_dir = run_dir / "proof"
    image_dir = proof_dir / "images"
    clip_dir = proof_dir / "clips"
    image_dir.mkdir(parents=True, exist_ok=True)
    clip_dir.mkdir(parents=True, exist_ok=True)

    events = [event for subject in subjects for event in subject.get("reconnect_events", [])]
    evidence: list[dict[str, Any]] = []
    preview_path = run_dir / "preview.mp4"
    for index, event in enumerate(events, 1):
        stem = f"reconnect_{index:03d}_{event['subject_ref']}_{int(float(event['reconnect_time_sec']) * 1000)}"
        image_path = image_dir / f"{stem}.jpg"
        write_event_image(video_path, event, image_path)
        clip_path = clip_dir / f"{stem}.mp4"
        clip_written = preview_path.exists() and write_event_clip(
            preview_path,
            event,
            clip_path,
            float(metadata.get("processing_start_sec", 0.0)),
            args.clip_padding_sec,
        )
        evidence.append(
            {
                **event,
                "image": str(image_path.relative_to(proof_dir)),
                "clip": str(clip_path.relative_to(proof_dir)) if clip_written else None,
            }
        )

    processing_start_sec = float(metadata.get("processing_start_sec", 0.0))
    processing_end_sec = float(metadata.get("processing_end_sec") or max(
        (subject.get("end_time_sec", processing_start_sec) for subject in subjects),
        default=processing_start_sec,
    ))
    detection_samples = write_detection_samples(
        preview_path,
        image_dir,
        processing_start_sec,
        processing_end_sec,
    )

    metrics = summarize(subjects, row_ids, processing_end_sec - processing_start_sec)
    result = {
        "camera": args.camera,
        "run_id": args.run_id,
        "video_path": str(video_path),
        "run_metadata": metadata,
        "metrics": metrics,
        "subjects": subjects,
        "reconnect_evidence": evidence,
        "detection_samples": detection_samples,
        "interpretation": {
            "what_is_proven": "Every listed raw tracker-ID change was joined into one subject reference using the recorded motion and appearance scores.",
            "accuracy_limit": "This is auditable reconnect evidence and a fragmentation measurement, not formal identity accuracy. Formal accuracy requires human-labeled ground truth for the full video.",
        },
    }
    (proof_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        f"# Tracking Proof: {args.run_id}",
        "",
    ]
    if (proof_dir / "VISUAL_REVIEW.md").exists():
        lines.extend(["Human inspection notes: [VISUAL_REVIEW.md](VISUAL_REVIEW.md)", ""])
    lines.extend([
        "## Run",
        "",
        f"- Camera: `{args.camera}`",
        f"- Video: `{video_path}`",
        f"- Database rows: `{', '.join(map(str, row_ids))}`",
        f"- Model: `{metadata.get('model')}`",
        f"- Tracker: `{metadata.get('tracker')}`",
        "",
        "## Measured Results",
        "",
        f"- Saved subject references: **{metrics['saved_subject_reference_count']}** (not a unique-worker count)",
        f"- Persistent references ({metrics['persistence_threshold_sec']}s or more): **{metrics['persistent_reference_count']}**",
        f"- Brief observations (under 10s): **{metrics['brief_reference_count_under_10s']}**",
        f"- Raw tracker IDs observed: **{metrics['raw_tracker_id_count']}**",
        f"- Fragmented raw tracklets joined: **{metrics['raw_tracklets_joined']}**",
        f"- Reconnect events: **{metrics['reconnect_event_count']}** "
        f"({metrics['active_gap_reconnects']} short-gap, {metrics['lost_track_reconnects']} lost-track)",
        f"- Distinct ID merges / same-ID gap recoveries: **{metrics['distinct_tracker_id_merges']} / {metrics['same_tracker_id_gap_recoveries']}**",
        f"- Reconnects requiring visual review: **{metrics['review_required_reconnects']}**",
        f"- Reconnect score, mean/minimum: **{metrics['mean_reconnect_score']} / {metrics['minimum_reconnect_score']}**",
        f"- Appearance similarity, mean/minimum: **{metrics['mean_appearance_similarity']} / {metrics['minimum_appearance_similarity']}**",
        "",
        "## Reconnect Evidence",
        "",
        "| # | Subject | Tracker change | Gap | Appearance | Score | Proof | Clip |",
        "|---:|---|---|---:|---:|---:|---|---|",
    ])
    for index, item in enumerate(evidence, 1):
        image_link = item["image"].replace("\\", "/")
        clip_link = item["clip"].replace("\\", "/") if item.get("clip") else ""
        lines.append(
            f"| {index} | {item['subject_ref']} | {item['old_tracker_id']} -> {item['new_tracker_id']} | "
            f"{item.get('time_gap_sec')}s | {item.get('appearance')} | {item.get('score')} | "
            f"[image]({image_link}) | " + (f"[clip]({clip_link}) |" if clip_link else "not generated |")
        )
    if not evidence:
        lines.append("| - | - | - | - | - | - | No reconnect events occurred | - |")
    lines.extend(
        [
            "",
            "## Detection Samples",
            "",
        ]
    )
    for sample in detection_samples:
        image_link = f"images/{sample['image']}".replace("\\", "/")
        lines.append(f"- t={sample['time_sec']}s: [annotated frame]({image_link})")
    if not detection_samples:
        lines.append("- No preview video was available for timeline samples.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The raw tracker-ID count is the fragmented baseline. The saved subject-reference count is the result after reconnect logic, but brief references can be partial detections or real passers-by and must not be reported as unique workers. Each merge above can be checked visually in its before/after image and annotated clip.",
            "",
            "This report does not claim formal identity accuracy because the video has no human-labeled identity ground truth. It proves which reconnect decisions were made and provides the evidence needed to accept or reject each one.",
            "",
        ]
    )
    (proof_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Proof report: {proof_dir / 'REPORT.md'}")
    print(f"Detailed results: {proof_dir / 'results.json'}")
    print(f"Images: {len(evidence)}; clips: {sum(item['clip'] is not None for item in evidence)}")


if __name__ == "__main__":
    main()
