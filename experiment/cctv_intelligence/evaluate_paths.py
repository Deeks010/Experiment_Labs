import argparse
import json
import sqlite3
import cv2
import numpy as np
import colorsys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "cctv_maps.sqlite3"
OUTPUT_DIR = BASE_DIR / "activity_runs"

def get_color(subject_ref):
    """Generate a unique deterministic color for a subject string."""
    try:
        # e.g. "subject_1" -> 1
        num = int(subject_ref.split("_")[-1])
        hue = (num * 137.508) % 360
        rgb = colorsys.hsv_to_rgb(hue / 360.0, 0.8, 0.9)
        return (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255)) # BGR
    except:
        return (0, 255, 0)

def main():
    parser = argparse.ArgumentParser(description="Evaluate path tracking against original video.")
    parser.add_argument("--camera", default="D24", help="Camera name in the database")
    parser.add_argument("--video", default=r"D:\hppindia-danny\backend\workstation\videos\D24_cut_01-00_to_01-10.mp4", help="Path to source video")
    parser.add_argument("--trail-sec", type=float, default=3.0, help="Seconds of path trail to show")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"evaluated_paths_{args.camera}.mp4"

    print(f"Connecting to DB {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Find camera ID
    row = conn.execute("SELECT id FROM cameras WHERE name = ?", (args.camera,)).fetchone()
    if not row:
        print(f"Error: Camera {args.camera} not found in database.")
        return
    camera_id = row["id"]

    # Load all subjects
    rows = conn.execute("SELECT person_path_json FROM floor_data WHERE camera_id = ? AND data_kind = 'person_activity_segment'", (camera_id,)).fetchall()
    
    subjects = []
    for r in rows:
        data = json.loads(r["person_path_json"])
        if "subjects" in data:
            subjects.extend(data["subjects"])

    print(f"Loaded {len(subjects)} tracking segments from database.")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    trail_frames = int(args.trail_sec * fps)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    print(f"Processing video {total_frames} frames at {fps} FPS. Outputting to {out_path}...")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # For each subject, find points up to the current frame
        for subj in subjects:
            ref = subj["subject_ref"]
            pts = subj["path_points"]
            color = get_color(ref)

            # Filter points that have already happened, but keep within trail_frames
            valid_pts = [p for p in pts if p["frame"] <= frame_idx and (frame_idx - p["frame"]) <= trail_frames]

            if not valid_pts:
                continue

            # Draw trail
            for i in range(len(valid_pts) - 1):
                p1 = tuple(map(int, valid_pts[i]["foot"]))
                p2 = tuple(map(int, valid_pts[i+1]["foot"]))
                
                # Make older points thinner
                age_frames = frame_idx - valid_pts[i]["frame"]
                thickness = max(1, int(3 * (1.0 - (age_frames / trail_frames))))
                
                cv2.line(frame, p1, p2, color, thickness)

            # Draw current bounding box if the last point is very recent (within 5 frames)
            last_pt = valid_pts[-1]
            if frame_idx - last_pt["frame"] <= 5:
                bbox = last_pt["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw foot coordinate as a dot
                foot_x, foot_y = map(int, last_pt["foot"])
                cv2.circle(frame, (foot_x, foot_y), 5, (0, 0, 255), -1)

                # Label
                cv2.putText(frame, ref, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Progress overlay
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        writer.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    writer.release()
    print(f"Done! Evaluated video saved to: {out_path}")

if __name__ == "__main__":
    main()
