"""Render camera-local movement tracks from saved frame observations."""
from __future__ import annotations

import argparse
import colorsys
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def color(index: int, total: int) -> tuple[int, int, int]:
    red, green, blue = colorsys.hsv_to_rgb(index / max(total, 1), 0.82, 1.0)
    return int(blue * 255), int(green * 255), int(red * 255)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--background-sec", type=float, default=80.0)
    args = parser.parse_args()

    with sqlite3.connect(args.db) as connection:
        connection.row_factory = sqlite3.Row
        run = connection.execute(
            "SELECT * FROM tracking_runs WHERE run_id=?", (args.run_id,)
        ).fetchone()
        if run is None:
            raise ValueError(f"Unknown run: {args.run_id}")
        rows = connection.execute(
            "SELECT time_sec, detections_json FROM frame_observations "
            "WHERE run_id=? ORDER BY frame_index", (args.run_id,)
        ).fetchall()
        zones = connection.execute(
            "SELECT zone_name, geometry_json, metadata_json FROM camera_zones "
            "WHERE camera_id=?", (run["camera_id"],)
        ).fetchall()

    capture = cv2.VideoCapture(run["source_path"])
    capture.set(cv2.CAP_PROP_POS_MSEC, args.background_sec * 1000)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError("Could not read the background frame.")

    frame = cv2.addWeighted(frame, 0.34, np.zeros_like(frame), 0.66, 0)
    tracks: dict[str, list[tuple[float, tuple[int, int]]]] = defaultdict(list)
    for row in rows:
        for detection in json.loads(row["detections_json"]):
            point = tuple(round(value) for value in detection["foot"])
            tracks[detection["subject_ref"]].append((row["time_sec"], point))

    scale = 3
    canvas = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    legend_width = 310
    output = np.full((canvas.shape[0], canvas.shape[1] + legend_width, 3), 24, np.uint8)
    output[:, :canvas.shape[1]] = canvas

    for zone in zones:
        geometry = json.loads(zone["geometry_json"])
        points = geometry.get("points")
        if not points:
            x1, y1, x2, y2 = geometry["box"]
            points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        polygon = np.asarray(points, np.float32) * scale
        kind = json.loads(zone["metadata_json"]).get("kind", "static_object")
        zone_color = (150, 150, 150) if kind == "static_object" else (80, 200, 230)
        cv2.polylines(output, [polygon.astype(np.int32)], True, zone_color, 2)

    ordered = sorted(tracks, key=lambda value: int(value.split("_")[-1]))
    for index, subject in enumerate(ordered):
        track_color = color(index, len(ordered))
        samples = tracks[subject]
        for (time_a, point_a), (time_b, point_b) in zip(samples, samples[1:]):
            if time_b - time_a <= 0.6:
                cv2.line(output, tuple(value * scale for value in point_a),
                         tuple(value * scale for value in point_b), track_color, 4, cv2.LINE_AA)
        start = tuple(value * scale for value in samples[0][1])
        end = tuple(value * scale for value in samples[-1][1])
        cv2.circle(output, start, 8, track_color, -1)
        cv2.circle(output, end, 11, track_color, 3)

        y = 58 + index * 47
        cv2.line(output, (canvas.shape[1] + 25, y),
                 (canvas.shape[1] + 70, y), track_color, 7)
        cv2.putText(output, subject.replace("subject_", "Track "),
                    (canvas.shape[1] + 85, y + 7), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (235, 235, 235), 2, cv2.LINE_AA)

    cv2.putText(output, "Saved movement tracks", (canvas.shape[1] + 22, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output, "dot=start  ring=end", (canvas.shape[1] + 22, output.shape[0] - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.53, (190, 190, 190), 1, cv2.LINE_AA)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), output):
        raise RuntimeError("Could not save track image.")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
