# CLAUDE.md — CCTV supervisor agent (Kenesis / hppindia)

You are continuing work that started in a claude.ai cloud session
(session_012sPavTb3yigaVsqnpnrRmZ, 2026-10-09) with Dheekshith (dheekshith@kenesis.ai).
If this session was teleported, the full chat history is also in your context. If not,
THIS FILE and `docs/` are the memory. Read this file fully, then `docs/conversation_log.md`,
then `docs/agent_design.md`. Then continue from "NEXT STEPS" below.

## 0. How to talk to the user (strict)
- Brief answers. ASD-STE100 style: short sentences, simple words, active voice, one instruction per sentence.
- Be honest. Say when something is weak, untested or uncertain. Never overclaim.
- The user is new to agents. Explain in simple terms, with tables. Avoid jargon or explain it.
- The user likes evidence: when a design question can be tested, build a small test and show numbers.
- Test method we use: blind subagents answer from one format only; grade by script against simulator truth
  (preferred) or by a separate grader agent. Never let the tested agent see the truth.

## 1. Product goal
AI on factory CCTV. 48 h video stored. An agent answers ANY supervisor question that the data can answer
(incl. "today vs yesterday"), with evidence and confidence. Detectors already exist: chit-chat, absence at
workstation, loitering. Static map per camera: named zones (polygons), objects (presses, benches,
control panel, aisle), description, reference frame. People are tracked by a vision model.

## 2. Core findings (all tested; details + tables in docs/HANDOFF.md)
1. Tracker IDs are ~80–85% reliable. Long identity is NOT usable. Never say "worker X did Y".
   Say "a person at Op 15 moved …". Stitched tracks gave 11 persons for 8 (0% correct counts under noise).
2. Reliable, ID-free facts: people count per second (95% exact at normal noise, from SHORT tracks with 2 s
   coasting — not raw detections), zone occupied/empty (99%), edge crossings in/out (95% of events).
3. LLMs answer best from TEXT that the data layer pre-computes. Images help only when they carry printed
   numbers/labels. Plain heatmaps, colour=time dots, overlapping paths fail (Round 2: V3 4.5/16, V4 3/16).
4. Winning format = ID-free scene log: people-in-frame intervals + zone state intervals + edge events +
   anonymous movement episodes + short-visit summaries + camera health/limits (Round 3: 15.75/18;
   raw IDs 12.25, stitched 13.1).
5. Exact placement: coarse zones ("near Workstation 15") cause guessing. Use exclusive floor cells named by
   both neighbours ("Gap Press 15 ↔ Bench 15", "Front of Bench 15 (aisle side)") + relations computed in code
   (nearest object + metres, side, % along bench, reach flag, boundary flag, feet visible/hidden) + xy.
   Round 4: coarse 4.2/18 vs exact 16/18. Round 5 edge positions: cell-only 77%, relations 100%,
   raw xy 99.4%, clean image 98.4% (errors exactly at boundaries). Rules (reach, at-station) live in code.
6. Hidden feet (bench occludes): every format says "uncertain". Resolve with real frame crops, not more text.
7. Heatmap overview (Round 6): plain heatmap ~5/10 (no numbers), labelled heatmap 8/10 (misses diffuse
   aisle time), text hotspot list 10/10, text+labelled heatmap 10/10. Heatmap = optional overview only.

## 3. Agreed architecture (see docs/agent_design.md)
- ONE main agent (not multi-agent; multi-agent ≈15× tokens, our task is sequential).
- Data layer does all math: Postgres + PostGIS + TimescaleDB (proposal), jobs that build zone states,
  edge events, movement episodes, dwell rollups, detector events (incl. new: long stationary/"sleeping",
  crowding, fast movement), camera health.
- Semantic layer: one definitions file (absence, idle, loitering, chit-chat, sleeping, crowding, reach, shift times,
  zone synonyms).
- Tools (MCP server), user's list + additions:
  get_image_frames (single or frame sheet grid, timestamps burned in, crop), get_camera_info (description,
  zones, objects, synonyms/place resolver), list_cameras, get_dwell_overview (text hotspots + optional labelled
  heatmap), get_scene_log (time-ordered ID-free log + relations), get_events, compare_periods (pure comparison).
- Framework: Claude API (client SDK / tool runner) + own MCP tool server. Agent SDK acceptable alternative.
- Optional subagent later: frame checker (vision) to keep images out of main context. Answer checker = code
  (every number in answer must appear in a tool output).
- Loop: hard cap (~20 tool calls, time, cost) as safety net; effort guide in prompt (simple 1–3, medium 3–8,
  open ≤15); stop when 2 calls add nothing new.
- Prompt: short, "right altitude", sections (role, facts, how_to_work, rules, answer_format, 2–3 examples).
  Grow it only from observed failures. Tool descriptions carry most of the "how".
- Answer format: Answer · Evidence (tool + times) · Confidence · What we cannot know · Suggested check.

## 4. Repo map
- `round1_2/` original test page (llm_format_test.html), images V1–V7, prompts, round-2 results.
- `round3/` simulator (sim.py), formats (reps.py), questions+auto-grader (truth.py), stress test (stress.py),
  prompts/answers/scores, example scene log.
- `round4/` exact placement (place.py, snap.py). `round5/` edge positions (edge.py, edge40.py).
- `round6/` heatmap overview (heat.py), example dwell overview.
- `docs/HANDOFF.md` full results log. `docs/agent_design.md` build plan, tools, prompt skeleton.
  `docs/conversation_log.md` chronological summary of the original chat. `docs/research_sources.md` sources.
- Run any round from inside its folder: `cd round3 && python3 sim.py` (needs matplotlib, numpy).

## 5. Known gaps / not tested
- Real camera images in perspective; real tracker output; weaker models; agent tool-choice accuracy;
  the added overview lines (HEALTH/LIMITS/UNUSUAL/NEXT STEPS) and the "derived bounds" line; 1-hour windows
  with per-minute rollups; two-camera handover.

## 6. NEXT STEPS (in order)
1. Ask the user for: real data format (tracker output schema, zone polygons, camera list), DB choice confirmation,
   and where the GPU pipeline writes results.
2. Build v0 data layer + 4 tools (get_camera_info, get_scene_log, get_dwell_overview, get_image_frames) on the
   simulator, then on ONE real recorded camera hour.
3. Agent v0 (Claude API + MCP tool server) with the prompt skeleton; bench of 30 questions (sim truth + one
   hand-labelled real hour); script grading for numbers, LLM judge for text.
4. Add get_events, compare_periods, list_cameras, frame-checker subagent; new detectors (sleeping, crowding,
   fast movement) in the data layer.
5. Pilot with one supervisor; log every question; add misses to the bench; weekly loop: run bench → read
   transcripts → fix tool/data first, prompt last.
