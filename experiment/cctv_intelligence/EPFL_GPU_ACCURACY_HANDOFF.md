# EPFL single-camera identity accuracy handoff

## Objective

Find the best technically achievable person detection and identity continuity on the
EPFL camera-01 video. Accuracy matters more than latency. Process offline and use the
GPU. The video contains six real people. A successful result should keep close to six
continuous identities while also detecting the correct concurrent headcount.

Do not tune the output by hard-coding that six people exist. Use the supplied labels
only for evaluation after each run.

## Required local data

The video is intentionally not stored in Git. Ask the user for the original file if it
is absent, then place it at:

`demo_assets/public_samples/epfl_6person_indoor/6p-c0.avi`

Keep the original AVI. Do not recompress it. The true positions and camera calibration
must be present beside it as `gt_lab_6p.txt` and `calibration-6p.txt`. Model weights are
also not stored in Git. Let the model library download them, or ask the user to place a
weight file locally if downloading is unavailable.

## Recreate the supplied map

Run the one-camera preparation step with `epfl_c0_accuracy_test.json`. It imports the
five polygons already drawn by the user. The bench, table, and projector are static
objects. The patterned and blue mats are floor areas. Do not redraw or reinterpret
them for the first benchmark.

From the repository root, a clean starting setup is:

```powershell
python experiment/cctv_intelligence/prepare_public_tracking_workspace.py `
  --db experiment/cctv_intelligence/activity_runs/epfl_gpu_test/observations.sqlite3 `
  --config experiment/cctv_intelligence/epfl_c0_accuracy_test.json
```

## Baseline already measured

- Video interval: 4.0 to 118.2 seconds.
- Detector: smallest YOLO pose model.
- Input size: 320.
- Sampling: one of every five frames.
- Device: CPU.
- Exact concurrent-count match: 44.3% across 115 labelled moments.
- Mean absolute count error: 0.774 people.
- Peak count: six detected and six true.
- Overcount at labelled moments: 0%.
- Identity result: 14 saved identities for six real people.
- The custom reconnection step successfully joined 48 tracker-ID changes, but still
  split several real people into new identities.

## Accuracy experiment

1. Confirm CUDA is actually used and record the GPU, library version, model, tracker,
   input size, thresholds, and run time.
2. Process every frame. Start at 4 seconds. Do not reduce quality for real-time speed.
3. Test stronger person or pose detectors that fit the GPU. Begin with medium, then
   large or extra-large when memory permits.
4. Begin at 640 input size. Test a larger size only if it improves measured accuracy;
   the source itself is only 360x288.
5. Enable a real appearance ReID branch. The supplied BoT-SORT ReID configuration is
   a starting point, not an assumed winner.
6. Compare at least BoT-SORT with ReID against one strong alternative available in the
   installed tracking library. Tune lost-track memory, new-track threshold, matching,
   proximity, and appearance thresholds from measured results.
7. Keep the existing custom reconnection layer only when an ablation proves it improves
   identity continuity. Compare tracker alone versus tracker plus custom reconnection.
8. Do not choose a result from visual smoothness. Use the supplied truth data.

Use this only as the first high-quality run. Change one major setting at a time after
recording its result:

```powershell
python experiment/cctv_intelligence/floor_activity_tracker.py EPFL_C0 `
  demo_assets/public_samples/epfl_6person_indoor/6p-c0.avi `
  --db experiment/cctv_intelligence/activity_runs/epfl_gpu_test/observations.sqlite3 `
  --output-dir experiment/cctv_intelligence/activity_runs/epfl_gpu_test/tracking `
  --source-kind real --device 0 --model yolo11m-pose.pt `
  --tracker experiment/cctv_intelligence/botsort_reid_accuracy.yaml `
  --imgsz 640 --frame-stride 1 --start-sec 4 --end-sec 118.2 `
  --conf 0.2 --save-preview-video
```

The model name may download automatically. If it does not, place the downloaded model
outside Git and pass its local path through `--model`.

## Required evaluation

For every run, report concurrent-count exact match, mean absolute count error,
undercount, overcount, peak count, total saved identities, identity switches, and
fragmentation. Add IDF1 and HOTA if the ground-plane truth can be converted correctly.
Never call total saved identities a headcount.

Use the included evaluator and movement-track renderer. Improve the evaluator when
needed, but preserve the raw results. Save a comparison table for every attempted
configuration. The final recommendation must show why its settings won.

After each run, take the printed run ID and execute:

```powershell
python experiment/cctv_intelligence/evaluate_epfl_tracking.py `
  --db experiment/cctv_intelligence/activity_runs/epfl_gpu_test/observations.sqlite3 `
  --run-id REPLACE_WITH_RUN_ID `
  --ground-truth demo_assets/public_samples/epfl_6person_indoor/gt_lab_6p.txt `
  --output-dir experiment/cctv_intelligence/activity_runs/epfl_gpu_test/evaluations/REPLACE_WITH_RUN_ID
```

## Acceptance target

The first target is six stable identities with no simultaneous duplicate person boxes,
while maintaining substantially better count accuracy than the baseline. If six
identities cannot be achieved without identity swaps, report that honestly and provide
the best measured trade-off. Include labelled evidence frames around each remaining
identity break.
