# Supervisor agent (v0)

Answers factory supervisors' questions from the CCTV tracking data, with evidence and honest uncertainty.

```
frame_observations (existing tracker)  ──derive.py──▶  sa_* tables (ID-free facts)
                                                          │
                                   tools.py (7 tools, text + images) ◀── semantics.json
                                     │                 │
                              agent.py (Claude)   mcp_server.py (any MCP client)
```

Design rules (tested in `research/`):
- People are anonymous. Counts, station states and entries/exits are reliable; long identities are not.
- The data layer does all math. The LLM reads short pre-computed text (relations like "BETWEEN Press 15 and Bench L").
- Images are for checking doubtful cases (frame sheets) and for overviews (labelled heatmap), not as the main data.

Quick start (from `experiment/cctv_intelligence/`):
```bash
pip install -r supervisor_agent/requirements.txt
python -m supervisor_agent.sim.simulate --out sim_data
export CCTV_DB_PATH=sim_data/sim.sqlite3 SA_TODAY=2026-10-09
python -m supervisor_agent.derive --camera cam07
python -m supervisor_agent.cli list_cameras
python -m unittest supervisor_agent.tests.test_supervisor_agent -v
ANTHROPIC_API_KEY=... SA_MODEL=<claude model id> python -m supervisor_agent.agent
```

Tools: `list_cameras`, `get_camera_info`, `get_scene_log`, `get_dwell_overview`, `get_events`, `compare_periods`,
`get_image_frames`. Example outputs: `research/docs/sim_scene_log_example.txt`, `research/docs/sim_dwell_overview_example.txt`,
`research/docs/sim_heatmap_example.png`, `research/docs/sim_frame_example.jpg`.

Environment: `CCTV_DB_PATH`, `SA_MODEL`, `SA_MAX_TOOL_CALLS` (20), `SA_MAX_SECONDS` (240), `SA_TODAY`, `SA_TIMEZONE`,
`SA_EXPORT_DIR` (frame_exports/), `SA_LOG_DIR` (agent_logs/).

See `CLAUDE.md` for status, results, gaps and next steps.
