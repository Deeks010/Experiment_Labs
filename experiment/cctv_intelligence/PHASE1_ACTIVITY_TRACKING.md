# CCTV Intelligence Phase 1 Notes

## Goal

Build an independent CCTV intelligence experiment that can:

- create/edit a high-level camera zone map
- run a video through YOLO pose/person tracking
- smooth noisy detections
- store only useful floor activity data
- avoid claiming real human identity when tracking is uncertain

This phase is separate from the existing backend project.

## Current Files

- `grid_sam_mapper.py`
  - Generates initial map annotations from a CCTV frame using the grid/LLM method.
  - SAM is optional and not required for the current high-level map workflow.

- `map_editor_server.py`
  - Local HTTP server.
  - Serves the map editor UI.
  - Creates/uses SQLite DB at `cctv_maps.sqlite3`.

- `map_editor_ui.html`
  - Browser canvas editor.
  - Loads generated map runs.
  - Allows box editing, renaming, adding, deleting, and saving zones.

- `floor_activity_tracker.py`
  - Reads a video file.
  - Runs YOLO pose tracking.
  - Calculates person foot points.
  - Smooths movement paths.
  - Groups visible people into activity segments.
  - Saves useful movement data into `floor_data`.
  - Can show a live browser preview and save a preview video.

## Final DB Shape

### `cameras`

One row per camera/source.

```text
id
name
source_path
created_at
```

### `camera_zones`

One row per saved map zone/object.

```text
id
camera_id
zone_name
geometry_json
metadata_json
created_at
updated_at
```

`geometry_json` stores coordinates:

```json
{
  "shape": "box",
  "box": [362, 32, 820, 616],
  "points": [[362, 32], [820, 32], [820, 616], [362, 616]]
}
```

`metadata_json` stores LLM/user details useful for later agent reasoning:

```json
{
  "description": "Blue press machine at rear-left",
  "visual_disambiguation": "leftmost press in the row",
  "importance": "high",
  "source": "llm_grid_then_user_corrected"
}
```

### `floor_data`

One row per meaningful floor activity segment.

```text
id
camera_id
start_time_sec
end_time_sec
data_kind
subject_count
person_path_json
confidence
evidence_json
metadata_json
created_at
```

No `track_id` is stored as real identity.
No `zone_context_json` is stored.
Zone relation should be calculated later by tools using `person_path_json` + `camera_zones`.

Example `person_path_json`:

```json
{
  "subjects": [
    {
      "subject_ref": "subject_1",
      "start_time_sec": 0.3,
      "end_time_sec": 3.9,
      "start_point": [243.59, 1264.05],
      "end_point": [330.63, 1245.2],
      "direction": "left_to_right",
      "distance_px": 102.6,
      "path_points": [
        {
          "t": 0.3,
          "frame": 9,
          "foot": [243.59, 1264.05],
          "raw_foot": [243.59, 1264.05],
          "bbox": [204.1, 900.2, 290.4, 1264.0],
          "foot_source": "ankle_midpoint",
          "confidence": 0.89
        }
      ],
      "quality": "good",
      "confidence": 0.89,
      "tracking_notes": [],
      "foot_source_counts": {
        "ankle_midpoint": 5
      }
    }
  ]
}
```

## Tracking Method

The tracker follows the common industrial pattern:

```text
YOLO pose detection
→ Ultralytics tracker, default BoT-SORT
→ foot-point extraction from pose keypoints
→ smoothing and jitter removal
→ short tracker-ID reconnect by nearby foot point
→ conservative segment saving
```

Foot point priority:

```text
both ankles midpoint
single ankle
knee projected down
hip projected down
bbox bottom-center fallback
```

If tracker identity becomes unreliable, the path is split or marked uncertain. The system does not claim that a subject is a known real person.

## Test Command Used

```powershell
.\.venv-sam\Scripts\python.exe experiment\cctv_intelligence\floor_activity_tracker.py D24_TEST backend\workstation\videos\D24_cut_01-00_to_01-10.mp4 --end-sec 4 --frame-stride 6 --tracker botsort.yaml --device cpu --save-preview-video --save-low-motion
```

Result:

- YOLO pose model loaded from `backend\workstation\person\models\yolo11m-pose.pt`
- short D24 clip processed
- one grouped `floor_data` row was saved for the test segment
- the row contained 6 visible subject paths
- preview video was written under `activity_runs`

## Usage

Run activity tracking with preview:

```powershell
.\.venv-sam\Scripts\python.exe experiment\cctv_intelligence\floor_activity_tracker.py D23 backend\workstation\videos\D23_cut_02-30_to_03-00.mp4 --preview --open-browser --save-preview-video
```

Run map editor:

```powershell
.\.venv-sam\Scripts\python.exe experiment\cctv_intelligence\map_editor_server.py --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

## Next Phase

Build agent tools that dynamically compare:

```text
person foot/path coordinates + camera_zones geometry
```

Those tools should answer questions like:

- which zone did the person pass through?
- who was near a machine during a time range?
- what movement happened on the floor?
- did anyone enter or cross a mapped area?

