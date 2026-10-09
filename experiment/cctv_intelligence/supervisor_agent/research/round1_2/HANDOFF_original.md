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

## 8. WHERE WE STOP NOW
- Round 2 (visual formats V1–V7, H1, plus F5 baseline, 2 runs each) is published on the page but NOT yet run / NOT yet analysed.

## 9. Your next tasks
1. Open the page in Chrome. Allow model access (it sends images). Keep default selection, 2 runs. Click "Run test". Wait for "Finished".
2. Read the results table and "answers and grades" on the page. Rank all formats. Report per question where each visual helps or fails.
3. Look at the 7 PNG images yourself and explain WHY each method won or lost.
4. Next iteration ideas to test: hybrid "text first + one image"; V7 timeline with exact times printed on bars; per-person path map (one person per image); event-centred keyframe sheets (real frames before/after a junction); more runs (3–5) for stable scores; harder scenarios (more workers, more crossings, camera offline period, two cameras).
5. Recommend the final render tool design and the final package format for the production agent.
6. Report to the user in brief, in ASD-STE100 style.
