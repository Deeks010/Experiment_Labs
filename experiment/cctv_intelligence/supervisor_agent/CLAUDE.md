# CLAUDE.md — Supervisor agent (project memory + instructions)

You continue work started in a claude.ai session (2026-10-09, session_012sPavTb3yigaVsqnpnrRmZ) with
Dheekshith (Kenesis). If the session was teleported, the chat history is in your context too. If not, this file,
`research/RESEARCH_MEMORY.md`, `research/docs/HANDOFF.md` and `research/docs/conversation_log.md` are the memory.

## 0. How to work with the user (strict)
- Brief replies. ASD-STE100 style: short sentences, simple words, active voice, one instruction per sentence.
- Be honest: say what is weak, untested or uncertain. The user is new to agents: explain simply, use tables.
- The user likes evidence: when a design question can be tested, build a small test and report numbers.
- Repo rules from `../RECENT_WORK_AND_CODEX_GUIDE.md` apply: SQLite is the source of truth; never modify raw
  `floor_data` / `frame_observations`; never commit `.env`, databases, videos, weights or generated outputs;
  never describe tracks as identities; position ≠ activity.

## 1. Goal
An agent that answers ANY supervisor question the CCTV data can answer (incl. today vs yesterday), with evidence
and confidence, and says "unknown / not observed" when the data cannot answer.

## 2. What is built (v0, this folder)
| File | What |
|---|---|
| `semantics.json` | Semantic layer: definitions, thresholds (body-height units), zone roles, shifts, limits text. |
| `db.py` | Reads the existing SQLite schema (cameras, camera_zones, tracking_runs, frame_observations, recording_clocks); owns derived `sa_*` tables; time parsing (wall clock via recording_clocks, else video time `v00:05:00`). |
| `geometry.py` | Point/polygon tests; relations in camera pixels scaled by body height (bh ≈ 1.7 m): side (far/camera side, ends), part, BETWEEN, distance. |
| `derive.py` | Builds `sa_*` from frame_observations: 1 Hz positions with 2 s coasting, counts, zone states (RLE), edge in/out (exit zones or image border, extrapolation, track-switch pairs removed), movement episodes, dwell grid, health (offline), detector events: absence, long_stationary (ID-free spots), close_pair, crowding, fast_movement, outside_work_area_dwell (ID-free spots), restricted_presence. `import_events()` for main-system events. |
| `tools.py` | The 7 tools → `{"text","images"}`: list_cameras, get_camera_info (+ place resolver), get_scene_log (+ detail='positions'), get_dwell_overview (+ labelled heatmap), get_events, compare_periods, get_image_frames (frame sheet). |
| `tool_specs.py` | Tool descriptions + JSON schemas (single source for agent and MCP). |
| `render.py` | Frame sheets from the recorded video (time burned in, zone outlines, detection boxes, crop) and heatmaps. |
| `mcp_server.py` | MCP stdio server with the 7 tools (mcp 1.x FastMCP or 2.x MCPServer). |
| `agent.py` | Claude API agent loop: system prompt, tool calls with images, caps (20 calls / 240 s), code checker → one revision, JSONL logs, events compatible with `agent_web_server.py` NDJSON. |
| `checker.py` | Flags times/numbers in an answer that appear in no tool output. |
| `prompts/system_prompt.md` | Short "right altitude" prompt: role, facts, how_to_work, rules, answer_format, 2 examples. |
| `sim/` | Simulator writing the SAME SQLite schema as the real tracker + rendered video + hidden truth JSON. |
| `bench/` | 16 questions with truth + rubric (`questions.json`), `run_bench.py` (agent + LLM judge), first agent test record. |
| `tests/` | 12 unit/e2e tests (no GPU, no video, no API key). |
| `research/` | All earlier research (rounds 1–6), handoff, sources, conversation log. |

## 3. Results so far
- Unit/e2e tests: 12/12 pass.
- Data layer vs simulator truth: offline exact; absence intervals within ~5 s; edge entry/exit counts exact on both
  runs; counts per second >90% exact; new detectors fire (runner, crowd, 4.5 min stationary person).
- Agent behaviour test (subagents played the model using only the CLI tools and the real system prompt; LLM grader with
  rubrics): **31/32** on 16 questions; 1 invented detail (q14 position) → fixed (positions now cover all tracks) and
  re-test correct. q11 scored 1 (said "about as busy" for 5.1 vs 4.9). Record: `bench/agent_test_2026-10-09/`.
- Not yet run: the real Claude API loop end-to-end (no API key in the build session). Mechanics tested with a fake client.

## 4. How to run (from experiment/cctv_intelligence/)
```bash
pip install -r supervisor_agent/requirements.txt
python -m supervisor_agent.sim.simulate --out sim_data            # simulated DB + videos (≈30 s)
export CCTV_DB_PATH=sim_data/sim.sqlite3 SA_TODAY=2026-10-09
python -m supervisor_agent.derive --camera cam07                  # build sa_* tables
python -m supervisor_agent.cli get_scene_log '{"camera":"cam07","start":"2026-10-09 10:00","end":"10:10"}'
python -m unittest supervisor_agent.tests.test_supervisor_agent -v
export ANTHROPIC_API_KEY=...  SA_MODEL=<current Claude model id>
python -m supervisor_agent.agent "Was Op 15 manned between 10:04 and 10:07 today?"
python -m supervisor_agent.bench.run_bench
```
Real data: point `CCTV_DB_PATH` at `cctv_maps.sqlite3`, make sure the run has `frame_observations` (new tracker) and a
recording clock (`site_context.bind_clock`), add `role` (+ optional `synonyms`) to zone metadata in the map editor
(operator_station, work_surface, walkway, exit, restricted, control), then run `derive`.

## 5. Known gaps / honest weak points
- Only simulated data so far. Real tracker noise, perspective and occlusion will differ.
- Relations use image terms ("far side" = above in the image). Correct for front-facing cameras; check per camera.
- Edge events ~90% per event; crowding includes groups walking together; long_stationary inside work zones is normal work.
- `derive` is pure Python, fine for minutes–hours; for 8 h × many cameras move hot loops to numpy or SQL.
- Agent tool-choice accuracy on many more question types is untested; prompt is v0.
- Web UI integration not done (run_turn() already yields the same NDJSON-style events).

## 6. NEXT STEPS
1. Run `run_bench` with a real API key; read 5–10 transcripts in `agent_logs/`; fix tools/data first, prompt last.
2. One real camera hour: run tracker → bind clock → add zone roles → derive → compare tool output with the video.
3. Hand-label that hour (counts, station empty intervals, entries/exits) and add 10–20 real questions to the bench.
4. Wire `agent.run_turn` into `agent_web_server.py` as a third provider ("claude").
5. Multi-camera: list_cameras + get_events across cameras; later a frame-checker subagent if images bloat context.
