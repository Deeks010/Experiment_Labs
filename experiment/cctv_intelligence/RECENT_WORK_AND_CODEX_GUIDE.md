# CCTV Intelligence: Recent Work and Codex Guide

## Purpose

This module converts CCTV video into conservative, auditable floor observations:

```text
camera video
-> person pose detection and local tracking
-> appearance-assisted track reconnection
-> timestamped floor-contact positions in SQLite
-> day/session identity profiles
-> map occupancy and track-position summaries
-> chatbot tools and visual evidence
```

The system records what the camera supports. It must not infer that a person operated a
machine, performed a task, or is a known employee merely because of position or clothing.

## Recent Work

The recent CCTV update added or improved:

- YOLO pose tracking with BoT-SORT and pose-derived floor-contact points.
- One-to-one reconnection of interrupted tracks using time, position, motion, pose-region
  clothing histograms, and conflict checks.
- Duplicate suppression for overlapping full-person and partial-person detections.
- Day/session appearance signatures. These describe visible clothing for that session and
  are not permanent or biometric identities.
- Persisted identity-profile tables linking likely fragments without rewriting raw logs.
- A chatbot tool for inspecting identity profiles and the evidence behind fragment links.
- SQLite-backed map editing. `camera_zones` is the authoritative static map store.
- Neutral analytics with per-track timestamps, normalized/pixel positions, exact foot-point
  containment, one-minute windows, occupancy maps, and track-position maps.
- Auditable reconnect reports with images, clips, JSON, and Markdown outputs.
- Regression tests for reconnection, duplicate suppression, map calculations, and missing
  identity-profile tables.
- Ignore rules preventing credentials, databases, videos, models, samples, and generated
  run outputs from being committed.

## Supported Core Files

| File | Responsibility |
|---|---|
| `floor_activity_tracker.py` | Detect, track, reconnect, and store person observations. |
| `identity_profile_builder.py` | Build day/session identity profiles from stored tracks. |
| `map_editor_server.py` | Edit and save static camera-map objects in SQLite. |
| `map_editor_ui.html` | Browser interface for the map editor. |
| `floor_map_analytics.py` | Produce factual JSON/Markdown summaries and maps. |
| `tracking_proof_report.py` | Generate auditable tracking and reconnect evidence. |
| `cctv_tools.py` | Query and translate stored data for the chatbot. |
| `cctv_agent.py` | OpenAI/Gemini tool-calling agent. |
| `agent_web_server.py` | Serve the chatbot API and built frontend. |
| `agent_frontend/` | React/Vite chatbot interface. |

`cross_camera_reid_proof.py` and `prepare_multicam_sample.py` may exist in a developer
worktree, but they are experimental utilities and are not part of the supported main-branch
pipeline unless their external OSNet source, weights, and dataset setup are documented.

## Installation

### Required Software

- Git
- Python 3.11 or 3.12 recommended
- Node.js 20.19+ or 22.12+ and npm, only for the chatbot web interface
- NVIDIA driver and a CUDA-compatible PyTorch installation, only for GPU processing

### Python Environment

From the repository root on Windows PowerShell:

```powershell
python -m venv .venv-cctv
.\.venv-cctv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r experiment\cctv_intelligence\requirements.txt
```

The main requirements install OpenCV, Pillow, Ultralytics, dotenv, and the OpenAI/Gemini
SDKs. `requirements-sam.txt` is optional and is only needed for SAM-assisted static map
generation.

For NVIDIA execution, install the PyTorch build matching the machine's CUDA environment,
then verify:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Pose Model

Model weights are intentionally not committed. The tracker searches in this order:

1. `experiment/cctv_intelligence/models/yolo11m-pose.pt`
2. `backend/workstation/person/models/yolo11m-pose.pt`
3. `yolo11m-pose.pt`, which Ultralytics may download on first use

Pass an explicit model path when repeatability matters:

```powershell
--model D:\models\yolo11m-pose.pt
```

### Environment Variables

Copy `.env.example` to `.env` inside `experiment/cctv_intelligence` and set only the
provider used by the chatbot:

```env
OPENAI_API_KEY=your_key
GEMINI_API_KEY=your_key
GEMINI_MODEL=your_supported_model
OPENAI_VISION_MODEL=your_supported_vision_model
```

Never commit `.env`. Tracking, map editing, identity building, and map analytics do not
require an LLM key.

## Database

The local database is:

```text
experiment/cctv_intelligence/cctv_maps.sqlite3
```

It is generated locally and ignored by Git. Important tables are:

- `cameras`: camera name and source path.
- `camera_zones`: static mapped objects and their geometry.
- `floor_data`: raw timestamped subject paths and tracking evidence.
- `worker_identity_runs`: identity-profile build metadata.
- `worker_profiles`: estimated session-level visible-person profiles.
- `worker_profile_segments`: links from raw fragments to profiles.
- `worker_profile_samples`: appearance evidence used by profile building.

Raw `floor_data` must remain immutable during identity enrichment. Profile tables are an
evidence layer over raw observations.

## End-to-End Workflow

### 1. Track a Video

The tracker creates the database schema, registers the camera, processes the video, and
stores the run:

```powershell
python experiment\cctv_intelligence\floor_activity_tracker.py CAMERA_1 D:\videos\camera_1.mp4 --model D:\models\yolo11m-pose.pt --tracker botsort.yaml --device 0 --frame-stride 3 --save-preview-video
```

Use `--device cpu` when CUDA is unavailable. For a short smoke test, add `--end-sec 10`.
Low-motion observations are retained by default because stationary people matter for
occupancy. Do not use `--discard-low-motion` without a specific reason.

Generated previews and evidence are written below `activity_runs/` and are ignored.

### 2. Define Static Map Objects

Start the map editor:

```powershell
python experiment\cctv_intelligence\map_editor_server.py --port 8765
```

Open `http://127.0.0.1:8765`, select the registered camera, draw or correct static objects,
and save. New map data must be written to `camera_zones`, not to a new JSON sidecar.

### 3. Build Session Identity Profiles

```powershell
python experiment\cctv_intelligence\identity_profile_builder.py CAMERA_1 --run-name camera_1_day_2026_09_15
```

The builder samples useful person crops, compares appearance and continuity, writes profile
tables, and preserves links back to every source `floor_data` row. Use `--replace` only when
deliberately rebuilding the same named profile run.

### 4. Generate Analytics

List available camera tracking runs:

```powershell
python experiment\cctv_intelligence\floor_map_analytics.py --list
```

Generate a complete summary:

```powershell
python experiment\cctv_intelligence\floor_map_analytics.py CAMERA_1 --run RUN_ID
```

If tracking and static zones are stored under different camera records:

```powershell
python experiment\cctv_intelligence\floor_map_analytics.py TRACK_CAMERA --run RUN_ID --zones-from MAP_CAMERA
```

Outputs below `map_analytics/` include:

- `summary.json`: compact LLM-ready facts and provenance.
- `SUMMARY.md`: readable observation summary.
- `occupancy_map.png`: relative density of sampled floor-contact points.
- `track_position_map.png`: camera-local subject paths over the mapped scene.

### 5. Generate Tracking Proof

```powershell
python experiment\cctv_intelligence\tracking_proof_report.py CAMERA_1 RUN_ID --video D:\videos\camera_1.mp4
```

Use the generated images and clips to inspect reconnect decisions. A proof report audits
what the algorithm did; it is not ground-truth identity accuracy unless labeled identity
annotations are supplied separately.

### 6. Run the Chatbot

Command-line agent:

```powershell
python experiment\cctv_intelligence\cctv_agent.py --provider openai --camera CAMERA_1
```

Use `--provider gemini` for Gemini.

Web interface setup:

```powershell
cd experiment\cctv_intelligence\agent_frontend
npm install
npm run build
cd ..
python agent_web_server.py --port 8788
```

Open `http://127.0.0.1:8788`.

## Verification

Run before committing CCTV changes:

```powershell
python -m unittest discover -s experiment\cctv_intelligence -p "test*.py"
python experiment\cctv_intelligence\floor_activity_tracker.py --help
python experiment\cctv_intelligence\floor_map_analytics.py --help
python experiment\cctv_intelligence\identity_profile_builder.py --help
```

For frontend changes:

```powershell
cd experiment\cctv_intelligence\agent_frontend
npm run build
```

Before a long video run, check GPU and concurrent workloads:

```powershell
nvidia-smi
```

Do not start a heavy run if another model is already consuming most GPU memory.

## Rules for Codex Working Here

1. Read `git status` before editing. Do not revert unrelated user changes.
2. Keep SQLite as the source of truth for camera maps and tracking logs.
3. Never commit `.env`, `*.sqlite3`, videos, model weights, generated crops, previews, or
   analytics output.
4. Treat `subject_ref`, tracker IDs, and worker profiles as camera/session estimates. Never
   describe them as employee identity, face recognition, or verified unique humans.
5. Reconnect only when appearance, time, position, motion, and conflict evidence support the
   same assignment. Preserve uncertain cases rather than forcing a match.
6. Store raw observations first. Derive identity profiles and analytics without rewriting
   the original `floor_data` records.
7. A floor-contact point inside a mapped box establishes geometric position only. Do not
   claim the person operated, used, inspected, or worked at that object.
8. Keep long detection gaps unfilled. Summaries must expose confidence and limitations.
9. Use explicit file staging. Generated or unrelated worktree files must not enter CCTV
   commits.
10. Run unit tests, in-memory syntax compilation if Windows locks `__pycache__`, and at least
    one read-only SQLite smoke test before pushing.

## Current Limitations

- Clothing appearance can help reconnect a person during one day but can collide when
  uniforms look similar.
- Occlusion, reflections, partial bodies, poor lighting, and low resolution can split or
  incorrectly merge tracks.
- Camera-pixel movement is perspective-distorted until floor-plane calibration is added.
- Exact unique-person accuracy requires human-labeled ground truth.
- Cross-camera identity is not part of the supported core pipeline in `main`.
- Position analytics describe visible presence and movement, not productivity or intent.
