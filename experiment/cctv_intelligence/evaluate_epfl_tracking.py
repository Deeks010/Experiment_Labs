"""Evaluate an EPFL tracking run against its labelled person counts."""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path


def load_ground_truth(path: Path) -> tuple[int, list[list[int]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = [int(value) for value in lines[1].split()]
    annotation_step = header[4]
    rows = [[int(value) for value in line.split()] for line in lines[2:]]
    return annotation_step, rows


def evaluate(db_path: Path, run_id: str, ground_truth_path: Path) -> dict:
    annotation_step, truth = load_ground_truth(ground_truth_path)
    with sqlite3.connect(db_path) as connection:
        run = connection.execute(
            "SELECT camera_id, config_json, status FROM tracking_runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if run is None:
            raise ValueError(f"Unknown run: {run_id}")
        samples = connection.execute(
            "SELECT frame_index, time_sec, detections_json FROM frame_observations "
            "WHERE run_id=? ORDER BY frame_index",
            (run_id,),
        ).fetchall()

    comparisons = []
    track_labels = set()
    for frame_index, time_sec, detections_json in samples:
        detections = json.loads(detections_json)
        track_labels.update(item["subject_ref"] for item in detections)
        if frame_index % annotation_step or frame_index >= len(truth):
            continue
        true_count = sum(value >= 0 for value in truth[frame_index])
        comparisons.append({
            "frame_index": frame_index,
            "time_sec": time_sec,
            "predicted_count": len(detections),
            "true_count": true_count,
        })

    if not comparisons:
        raise ValueError("The run has no frames aligned with ground-truth annotations.")

    deltas = [row["predicted_count"] - row["true_count"] for row in comparisons]
    true_unique_people = max(row["true_count"] for row in comparisons)
    predicted_labels = len(track_labels)
    exact = sum(delta == 0 for delta in deltas)
    result = {
        "run_id": run_id,
        "run_status": run[2],
        "labelled_samples": len(comparisons),
        "count_accuracy": {
            "exact_match_percent": round(100 * exact / len(deltas), 1),
            "mean_absolute_error": round(sum(abs(delta) for delta in deltas) / len(deltas), 3),
            "undercount_percent": round(100 * sum(delta < 0 for delta in deltas) / len(deltas), 1),
            "overcount_percent": round(100 * sum(delta > 0 for delta in deltas) / len(deltas), 1),
            "predicted_peak": max(row["predicted_count"] for row in comparisons),
            "true_peak": max(row["true_count"] for row in comparisons),
            "error_distribution": dict(sorted(Counter(deltas).items())),
        },
        "identity_check": {
            "true_people": true_unique_people,
            "predicted_local_labels": predicted_labels,
            "extra_labels": predicted_labels - true_unique_people,
            "labels_per_true_person": round(predicted_labels / true_unique_people, 2),
            "verdict": "failed" if predicted_labels != true_unique_people else "passed",
            "meaning": "Extra labels show identity fragmentation. This does not measure which label belongs to which person.",
        },
        "decision": {
            "usable": [
                "Peak visible-person count for this clip.",
                "Approximate room activity and broad occupied-area trends.",
            ],
            "not_yet_reliable": [
                "Unique-person totals.",
                "Complete individual routes or time spent in an area.",
                "Worker-level operational conclusions.",
            ],
        },
        "comparisons": comparisons,
    }
    return result


def markdown(result: dict) -> str:
    count = result["count_accuracy"]
    identity = result["identity_check"]
    return "\n".join([
        "# EPFL tracking evaluation",
        "",
        f"Run: {result['run_id']} ({result['run_status']})",
        f"Labelled moments checked: {result['labelled_samples']}",
        "",
        "## Result",
        f"- Exact person count: {count['exact_match_percent']}%",
        f"- Average count error: {count['mean_absolute_error']} people",
        f"- Undercount: {count['undercount_percent']}%; overcount: {count['overcount_percent']}%",
        f"- Peak count: {count['predicted_peak']} detected; {count['true_peak']} true",
        f"- Identity labels: {identity['predicted_local_labels']} detected; {identity['true_people']} true people",
        "",
        "## Decision",
        "Peak occupancy and broad room activity are usable for this test. Individual movement histories are not reliable yet because identities split during crowding and occlusion.",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(args.db, args.run_id, args.ground_truth)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output_dir / "evaluation.md").write_text(markdown(result) + "\n", encoding="utf-8")
    print(markdown(result))


if __name__ == "__main__":
    main()
