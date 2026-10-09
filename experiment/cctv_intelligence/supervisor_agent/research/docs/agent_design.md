# Agent design (agreed in session 2026-10-09)

## 1. Roles (simple picture)
| Part | Role | What |
|---|---|---|
| Data layer | Accountant | DB + jobs. Computes counts, zone states, edges, relations. Never guesses. |
| Tools | Standard forms | 7–8 functions returning clear text for one question type. |
| Agent | Analyst | Reads question, picks tools, checks frames, writes answer with evidence. |

## 2. Data layer (proposal)
- Postgres + PostGIS (zone polygons) + TimescaleDB (time series).
- Tables: cameras, zones, objects (polygons), positions (1 Hz foot point x,y in metres + feet_visible),
  tracklets, zone_states (RLE intervals: empty/1/2/3+/unknown), edge_events, movement_episodes,
  events (all detectors), camera_health, dwell_minute (per cell per minute), daily rollups. Keep 90+ days structured.
- Counting rule: count people from short local tracks with 2 s coasting (not raw detections).
- Placement rule: exclusive floor cells named by neighbours; solid objects (snap foot points out of benches/presses);
  feet hidden → flag + uncertainty, resolve with frames.
- New detectors to add as events: long stationary in same exact spot (possible sleeping), crowding (n people
  within r metres for t s), fast movement (speed > v), movement outside workstation areas.

## 3. Tools (user's list + additions)
| Tool | Returns | Use for |
|---|---|---|
| list_cameras() | cameras, names, lines, what they cover | factory-wide questions |
| get_camera_info(camera) | description, zones, objects, synonyms/place resolver, reference frame | map supervisor words to zones |
| get_scene_log(camera, start, end, focus?, detail?) | ID-free time-ordered log: counts, zone states, edges, movements + relations | WHEN / ORDER / HOW MANY, short windows |
| get_dwell_overview(camera, start, end, include_heatmap?) | minutes per cell, hotspots with relations, UNUSUAL flags, optional labelled heatmap | WHERE time went, long windows |
| get_events(type?, cameras, start, end) | detector events list | absence, loitering, chit-chat, sleeping, crowding … |
| compare_periods(camera/zone, metric(s), period_a, period_b) | pure comparison computed in code | today vs yesterday |
| get_image_frames(camera, time or range, count, crop?) | single frame or frame sheet (grid, timestamps burned in) | verify doubtful cases |

Tool-writing rules (Anthropic): few strong tools; clear names; readable text not raw IDs; concise/detailed mode;
actionable errors; each description says when to use, when not, one example.

Example outputs: `round3/example_scene_log_S1.txt`, `round6/example_dwell_overview.txt`,
`docs/dwell_heatmap_labelled.png`.

## 4. Agent loop
- Single agent. Hard cap ~20 tool calls + time + cost (safety net; log every hit).
- Effort guide in prompt: simple 1–3 calls, medium 3–8, open ≤15. Stop when 2 calls add nothing new.
- Routing rules: when/order/how many → scene log or events; where/how much time over long period → overview;
  open long question → overview first, then scene log on the hotspot window, then frames.

## 5. System prompt skeleton (~40 lines, grow from failures)
```
<role> You answer factory supervisors' questions using CCTV-derived data tools. </role>
<facts> Shift times. Definitions: absence, idle, loitering, chit-chat, sleeping, crowding.
People are NOT identified. A track is "a person", never a named worker. </facts>
<how_to_work>
- Map the supervisor's words to cameras and zones first (get_camera_info / list_cameras).
- Long periods: overview first, then zoom in with the scene log.
- "When / order / how many": scene log or events. "Where did time go": overview.
- If a position or event is doubtful, check frames before you claim it.
- Effort: simple 1–3 calls, medium 3–8, open questions up to ~15.
- Stop when you can answer, or when 2 calls add nothing new.
</how_to_work>
<rules>
- Every number must come from a tool output. Never estimate silently.
- "Not observed" and "unknown" are valid answers. Never fill gaps with reasons.
- Never say a specific worker did something; say "a person at Op 15".
</rules>
<answer_format> Answer · Evidence (tool + times) · Confidence · What we cannot know · Suggested check </answer_format>
<examples> 2–3 short good answers </examples>
```

## 6. Evaluation (professional part)
- Bench: start 20–30 real supervisor questions with truth (simulator + 1–2 hand-labelled real hours).
- Script grading for numbers; LLM judge (0–1 + pass/fail) for free text; code check that every number in an
  answer appears in a tool output.
- Run on every prompt/tool/model change. Trace every run (question, tool calls, tokens, cost, answer).
- Weekly: run bench → read 5–10 transcripts → fix tool/data first, prompt last → rerun.

## 7. Build order
1. Weeks 1–2: data layer + 4 tools on simulator + one real hour.
2. Week 3: agent v0 + 30-question bench.
3. Week 4: events, compare, positions/place resolver, frame-checker subagent.
4. Weeks 5–6: pilot with one supervisor, one line.
5. Then: more cameras, long-period rollups, access control, cost tuning.
