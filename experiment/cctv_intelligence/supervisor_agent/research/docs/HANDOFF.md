# Handoff: CCTV factory agent – data format and visual tests

## 0. How to talk to me (the user)
- Always explain in brief.
- Always write in ASD-STE100 style: short sentences, simple words, active voice, one instruction per sentence.
- Be honest. Say when something is weak.

## 1. What we build
- We put AI on factory CCTV cameras.
- We store 48 hours of video in one place.
- An agent gets tools (MCP style): it gives a camera ID and a timestamp, and it gets the frames it needs.
- We track workers in each camera with vision models.
- New idea: we map all STATIC things in each camera (yellow aisle, machines, benches, control panel, etc.) as named zones (polygons). Each camera has a description and a reference frame.
- We store worker foot paths relative to these zones. So we know where a person was and near what.
- We already detect: chit-chat, absence at workstation, loitering.
- Goal: a production agent that can answer almost any supervisor question, including "today vs yesterday".

## 2. Decisions so far
1. The data layer does the math. The agent plans, queries, checks frames, and explains.
2. Data layer (SQL): cameras + zones, zone visits (not raw points), one events table for all detectors, daily aggregates, CAMERA HEALTH table (offline/blocked), keep structured data 90+ days.
3. Few GENERIC tools are better than many specific tools:
   - get_map / get_camera_info
   - query (read-only SQL on views)
   - render(camera_id, start, end, mode, zones?, persons?, show_events?)  – modes: paths, slices, heatmap, snapshot, timeline
   - get_frames(camera_id, timestamp, count)
4. Semantic document in the system prompt: fixed definitions (absence, idle, loitering, chit-chat), shift times, schema.
5. Agent loop: understand → plan → query → drill down → verify with frames → answer with evidence + confidence.
6. Testing: a set of real supervisor questions with correct answers, run after every change.

## 3. Main data problem: tracking IDs
- ID accuracy is about 80–85%.
- One person gets several IDs (flicker, occlusion). Example: one path split into ID 1, ID 2, ID 3.
- Two people meet ("X" crossing) and IDs can swap.
- Agreed approach:
  - Treat tracker IDs as TRACKLETS, not persons.
  - Stitch tracklets OFFLINE (position + time gap, direction/speed, appearance/re-ID).
  - One-in-one-out → join, high confidence. Two-in-two-out → best guess, mark as JUNCTION, low confidence. Never hide uncertainty.
  - Count people with ID-free data (people per frame, zone occupancy per second).
  - Give the agent "zone stories" (time ranges per zone), not raw x,y dots.
  - Mark gaps as NOT OBSERVED with a reason. Separate "absent" from "not seen".
  - Remove ghost tracklets (< 3 s).

## 4. The simulated test scenario (ground truth)
Camera: cam07 "Moulding press 14–17" (real camera: 4 blue presses at the back, operators in front, 2 steel benches, control cabinet on the right, notice board, yellow floor tape). Window 10:00:00–10:10:00.
Zones: Press 14–17, Op 14–17 (operator standing areas), Bench L (Op 14/15), Bench R (Op 16/17), Passage (gap between benches to the aisle), Control panel, Yellow aisle (exits at west and east edge), Unmapped floor.
What really happened:
- A (Press 14): mostly at Op 14, short bench visits. Occluded 10:04:02–10:04:22. Raw IDs 3 → 11.
- B (Press 15): leaves Op 15 at 10:03:10 via Passage and aisle, out of view (west edge) 10:03:56–10:07:06, back at Op 15 10:07:52. Raw IDs 5 → 14.
- C (Press 16): goes to Op 17 at 10:05:10, talks with D until 10:07:00, back to Op 16 at ~10:07:06. Raw IDs 17 → 18 → 20.
- D (Press 17): stays at Op 17. Raw IDs 16 → 19 → 21. C/D junctions J1 10:05:10, J2 10:07:00.
- E (visitor): enters at east edge of the aisle 10:02:02, at Control panel 10:02:26–10:06:04, leaves east 10:06:34. Raw ID 9.
- Ghost: ID 12, 4 s near the fan in Passage at 10:04:10.
- Around 10:01 all four operators make a short (10–15 s) bench visit.
- Raw result: 12 IDs for 5 real people.
Events: E1 loitering E at Control panel 10:02:26–10:06:04; E2 absence Op 15 10:03:10–10:07:50; E3 absence Op 16 10:05:10–10:07:05; E4 chit-chat C+D Op 17 10:05:10–10:07:00.

## 5. Visual iterations (what we learned)
1. First path map: colour = TIME (dark → light). FAILED. The user could not tell persons apart (B, C, D confusing), and overplotting hid earlier dots (A looked orange only). Lesson: one visual channel = one meaning.
2. Fix: colour = PERSON, time = separate panels (time slices). Person filter. Max 5–6 persons per image. Label person letter on the path.
3. LLM view: LLMs are weak at counting dots, reading exact positions, fine colour gradients, and overlapping lines. They are strong at text, tables, timelines, and real frames. Without explicit gaps they fill gaps with guesses.
4. Zone occupancy timeline (rows = zones, columns = time bins). First version used MAX per bin. FLAW found by the user: a person who moves between Op 14 and Bench L inside one bin is counted in both rows, so rows add up to more people than the frame. Fix: use AVERAGE per bin (rows sum to "People in frame"), or instant snapshots.
5. Ideal package for the LLM: camera health + system limits + events (text) + zone occupancy (average) + stitched zone stories with confidence and NOT OBSERVED gaps + frames on request + path map on request (one person).

## 6. Test bench (built and published)
- Page: https://claude.ai/artifact/5QfaPUATFLcBjBZs8jF3e7  (file: llm_format_test.html in this folder; it contains all simulation, data formats, image drawing, questions, and ground truth).
- Method: a fresh model gets a neutral brief + one format + 8 questions. A second fresh model grades each answer (0/1/2) against the ground truth and flags invented facts. The tested model never sees the truth.
- Questions: Q1 people in view at 10:01 (4). Q2 where/pattern at 10:01. Q3 Op 15 occupied 10:04–10:07? (no). Q4 anyone at control panel, when, from where. Q5 operator zones empty > 1 min (Op 15, Op 16). Q6 anyone in yellow aisle 10:04:15–10:06:00? (no). Q7 different people in 10 min (5). Q8 what was Press 15 operator doing at 10:05? (unknown – not in view).
- Text formats: F1 raw tracker zone stories, F2 max-per-bin table, F3 average table, F4 stitched stories + events, F5 full package (health + limits + F3 + F4 + events).
- Visual formats (images in this folder): V1 path map colour = person (stitched, gap + junction notes), V2 path map colour = raw tracker ID, V3 dots coloured by time no IDs, V4 heatmap no IDs no time, V5 four time slices colour = person, V6 zone occupancy timeline image, V7 person timeline (Gantt) + events, H1 = F5 text + V1 + V7.

## 7. Round 1 results (text only, score /16, invented facts) – raw JSON in round1_results/
| Format | Run 1 | Run 2 | Invented |
|---|---|---|---|
| F1 raw IDs | 15 | failed | 0 |
| F2 max table | 13 | 14 | 0 |
| F3 avg table | 15 | 13 | 2 |
| F4 stitched + events | 13 | 15 | 2 |
| F5 full package | 16 | 16 | 0 |
Findings: F5 best. F2 double count was handled by using "People in frame", but time precision was lower. Most common error in all formats: time boundaries (Q6). Without stitching, models could not say "5 people". Sample size is small (n ≤ 2).

## 8. Round 2 results (DONE 2026-10-09) – raw JSON in round2_results.json
Method: run with subagents, not the page. Prompts were made from the page code (same brief, captions, questions, rubric). Tested model: claude-opus-5-5, fresh agent per run, no access to truth. Grader: separate fresh agent per run. n = 2.
| Rank | Format | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 | Mean /16 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | F5 text package | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 16.0 |
| 1 | H1 F5 + V1 + V7 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 16.0 |
| 3 | V7 Gantt + events | 2 | 2 | 2 | 1.5 | 2 | 2 | 2 | 2 | 15.5 |
| 4 | V6 occupancy image | 2 | 2 | 2 | 2 | 2 | 2 | 1 | 2 | 15.0 |
| 5 | V2 raw-ID path map | 2 | 1 | 2 | 2 | 2 | 2 | 1 | 2 | 14.0 |
| 6 | V1 stitched path map | 2 | 1 | 2 | 2 | 1 | 0.5 | 2 | 2 | 12.5 |
| 7 | V5 time slices | 1 | 0.5 | 1.5 | 2 | 1 | 0 | 2 | 1.5 | 9.5 |
| 8 | V3 dots by time | 0 | 0 | 0 | 1 | 0 | 0 | 1.5 | 2 | 4.5 |
| 9 | V4 heatmap | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 2 | 3.0 |
Key findings:
- An image scores well only when it carries printed times and text labels (V7, V6, V2 legend). Pure geometry (V1, V5) loses time questions (Q5, Q6). Colour-as-time (V3) and heatmap (V4) fail.
- V2 beat V1 because its legend prints start/end time per ID. V1 has no time on the paths.
- V5 slices invent presence: B is drawn "in the aisle" in 10:05–10:07:30 because the slice merges his return (10:07:06). Q6 = 0 in both runs. Slice boundaries hide when things happen.
- V3 run 1 invented a second control-panel trip (dark vs tan dots misread).
- Adding images to F5 (H1) did not raise the score and did not hurt it. Ceiling effect: F5 is already 16/16. The test is now too easy for text.
- Grader noise: 2 of 4 "invented" flags were false (true facts flagged). Use 2 graders or a stricter rubric.

## 8b. Round 3 (DONE 2026-10-09) – ID-free representations under real noise. Code + data in round3_bench.zip
User premise: we cannot say "worker X took this path". We can say "a worker moved this way", and counts per second are reliable.
Setup: new simulator (sim.py) with a noisy detector (3% miss, 22% miss in crowds, false detections at the fan and notice board) and a naive tracker (greedy, 2 s memory, ID flicker). Result: 32 tracker IDs for 8 people.
- S1: B leaves west; a DIFFERENT worker takes Op 15 later; 3 visitors at the control panel; camera OFFLINE 10:05:45–10:06:25.
- S2: Op 14 leaves east and comes back (same person); material handler twice (crosses benches, causes ID swaps); supervisor walks through; C+D chat at Op 17 (Op 16 empty 73 s).
9 structured questions, AUTO-GRADED from simulator truth (no grader model). 6 formats x 2 scenarios x 2 runs.
| Rank | Format | Mean /18 |
|---|---|---|
| 1 | R4 ID-free package + identity hypotheses (low trust) | 16.5 |
| 2 | R3 ID-free package: counts + zone states + edge crossings + anonymous movement episodes | 15.75 |
| 3 | R5 ID-free zone timeline image | 15.0 |
| 4 | R1 stitched persons | 13.1 |
| 5 | R0 raw tracker IDs | 12.25 |
| 6 | R2 counts + zone states only | 11.5 |
- Without edge events and movement episodes, entries/exits and "which way did someone go" cannot be answered (R0, R1, R2 lose Q5 and Q8).
- Identity question (Q6): every format answered "unknown". The stitcher joined B and the new worker F as one person (wrong). Stitched data gave 11 persons for 8 people.
- R4 gains only on "how many different people" (best guess). Everything else is equal to R3.
Data-layer stress test (stress.py, 20 seeds x 3 noise levels x 2 scenarios):
| Noise | Count exact | Count ±1 | Op zone occupied/empty | Edge events right | Tracker IDs per ~7 people | Stitched count right |
|---|---|---|---|---|---|---|
| normal | 94.8% | 100% | 99.4% | 95% | 32 | 0% |
| bad | 87.2% | 99.1% | 99.2% | 89% | 57 | 0% |
| very bad | 77.8% | 95.3% | 98.3% | 86% | 92 | 0% |
Important data-layer fix found: count people from SHORT LOCAL TRACKS with 2 s coasting, not raw detections. Exact count in crowd scenario went from 75% to 95%.

## 8c. FINAL METHOD: ID-free scene log ("zones are the entities, people are anonymous")
1. People-in-frame intervals (run-length encoded, from coasting short tracks, 5 s median).
2. Zone state intervals per zone: empty / 1 / 2 / 3+ / UNKNOWN (offline).
3. Edge crossing events: time, edge, in/out.
4. Anonymous movement episodes: one short continuous track = "a person moved A → B → C" with times. Never joined into identities.
5. Short repeating visits summarised (e.g. Op→Bench→Op under 25 s: start times).
6. Camera health + system limits.
7. Derived bounds (TO ADD): distinct people ≥ max simultaneous, ≤ start count + entries.
Identity hypotheses: separate tool, only on request, always labelled low trust. Verify identity with frames (re-ID / keyframes), never with tracks.

## 8e. Round 4 (DONE) – exact placement: machine ↔ bench ↔ aisle. Code in r4 (place.py, snap.py)
Problem: coarse zone "near Workstation 15" hides whether the person is between press and bench (running the press) or on the aisle side of the bench.
Test: metric floor plan, 3 tracks, one stay with feet hidden by the bench (detector puts the foot point ON the bench). 9 questions, auto-graded, 3 runs per format.
| Format | Score /18 |
|---|---|
| Coarse zone text ("near Workstation 15") | 4.2 (answers "unknown" to all placement questions) |
| Coarse text + floor-plan image | 15.7 |
| Floor-plan image only | 16.0 |
| Relational text (exclusive cells named by neighbours + metres + feet visibility) | 16.0 |
| Relational text + image | 16.0 |
| Relational text + solid-object snap | 16.0 |
Findings: the fix is in the MAP and DATA, not in the picture. Exclusive cells named by both neighbours ("Gap Press 15 ↔ Bench 15", "Front of Bench 15 (aisle side)") + distance to each object + feet visible/hidden. All formats with exact geometry gave the same answers. Image adds nothing over good text here, but it rescues coarse text.
Hidden-feet stay: every format answered "uncertain" (truth: in the gap). Correct behaviour. Resolve with a real frame crop (get_frames) or pose keypoints, not with more text.

## 8f. Round 5 (DONE) – edge positions: text vs coordinates vs image. Code in r5 (edge.py, edge40.py)
Question: cell names cannot cover a person at the END or CORNER of a station. Is an image better for the LLM?
Test A: 12 tricky stays (bench ends, corners, cell boundaries). 4 questions each: nearest object, side of bench, within 0.8 m reach of press face, which third of bench. 2 runs.
| Format | Accuracy |
|---|---|
| Cell name only | 77% (silent guesses: "middle third" everywhere, bench-edge stay called "within reach of press") |
| Raw x,y + object rectangles | 100% |
| Clean top-view image with metre grid | 100% |
| Precomputed relations (nearest + distance, side, % along bench, reach flag, boundary flag) | 100% |
Test B: 40 stays, denser, labels overlap in the image. 2 runs.
| Format | Accuracy | Errors |
|---|---|---|
| Precomputed relations | 100% | none |
| Raw x,y | 99.4% | one stay 0.04 m from the reach-rule limit |
| Image | 98.4% | stays exactly at the bench end / reach limit; one label in a crowded cluster |
Findings: cell names alone fail at edges. A strong model can do the geometry from raw numbers or a clean plan, but errors sit exactly at boundaries, where a decision matters. Rules ("reach", "at station") must be computed in the data layer, not interpreted by the LLM. Images here were clean schematic top views; real camera perspective images were NOT tested.

## 8g. Round 6 (DONE) – heatmap as overview. Code in r6/heat.py
1 hour of aggregated dwell (no time, no identity). 7 true spots incl. 3 unusual spots (beyond Bench 16 end 5 min, control panel front 4 min, passage by Bench 15 end 3 min) + 6 min spread along the aisle. 5 overview questions, 2 runs, graded by hand (/10).
| Format | Score /10 | Notes |
|---|---|---|
| Plain heatmap image | ~5 | places and ranking right, but NO minutes (colour cannot be read as numbers); aisle total unknown |
| Heatmap + printed hotspot minutes | 8 | all right except aisle total (diffuse time is invisible in a heatmap) |
| Text hotspot summary (minutes per cell + hotspots with relations) | 10 | |
| Text + labelled heatmap (hybrid) | 10 | |
Findings: heatmap shows WHERE time concentrated; it hides diffuse time, numbers, order in time, and identity. Use a labelled heatmap only as an overview image next to the text hotspot list, and only for "where did time go" questions.

## 8d. WHERE WE STOP NOW
- Next: add derived bounds line; test 1-hour window with per-minute rollups; test two-camera handover; test real tracker output from one real camera.

## 9. Your next tasks
1. Open the page in Chrome. Allow model access (it sends images). Keep default selection, 2 runs. Click "Run test". Wait for "Finished".
2. Read the results table and "answers and grades" on the page. Rank all formats. Report per question where each visual helps or fails.
3. Look at the 7 PNG images yourself and explain WHY each method won or lost.
4. Next iteration ideas to test: hybrid "text first + one image"; V7 timeline with exact times printed on bars; per-person path map (one person per image); event-centred keyframe sheets (real frames before/after a junction); more runs (3–5) for stable scores; harder scenarios (more workers, more crossings, camera offline period, two cameras).
5. Recommend the final render tool design and the final package format for the production agent.
6. Report to the user in brief, in ASD-STE100 style.
