"""Tool descriptions and JSON schemas. One source for the Claude agent and the MCP server.
The descriptions carry most of the 'how to use' guidance (when to use, when not, example)."""

TIME = ("Time as wall clock 'YYYY-MM-DD HH:MM[:SS]' or 'HH:MM[:SS]' (an end time without a date takes the start's date). "
        "If a recording has no clock, use video time 'v00:05:00'.")

SPECS = [
    {
        "name": "list_cameras",
        "description": "List all cameras with a short description, mapped zones and the recorded time ranges. "
                       "Use first for factory-wide questions or when you do not know which camera or date has data. "
                       "Do not use for details of one camera (use get_camera_info).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_camera_info",
        "description": "Full description of one camera: what it shows, static objects (machines, benches, panels), floor areas "
                       "(operator stations, walkways, exits) with roles and synonyms, recorded time ranges. Set 'place' to map the "
                       "supervisor's words (e.g. 'press 15 operator', 'near the panel') to zone names. Use before any analysis when the "
                       "question names a place. include_reference_frame=true attaches one image with zone outlines.",
        "input_schema": {"type": "object", "properties": {
            "camera": {"type": "string"}, "place": {"type": "string"},
            "include_reference_frame": {"type": "boolean"}}, "required": ["camera"]},
    },
    {
        "name": "get_scene_log",
        "description": "Time-ordered, ID-free log for ONE camera and a SHORT window (max 30 min): camera health, people in frame per "
                       "interval, operator station occupied/EMPTY intervals (short step-aways ≤25 s counted as occupied), other areas, "
                       "edge crossings (people entering/leaving the view), anonymous movement lines, detector events, and derived bounds "
                       "on how many different people. Use for WHEN / IN WHAT ORDER / HOW MANY questions. detail='positions' adds exact "
                       "positions relative to machines and benches (between which objects, side, distance in body heights, feet visible). "
                       "focus='<zone>' keeps only movement lines that touch that zone. Not for long periods (use get_dwell_overview / get_events). "
                       "Example: get_scene_log(camera='cam07', start='2026-10-09 10:00', end='10:10').",
        "input_schema": {"type": "object", "properties": {
            "camera": {"type": "string"}, "start": {"type": "string", "description": TIME}, "end": {"type": "string", "description": TIME},
            "focus": {"type": "string"}, "detail": {"type": "string", "enum": ["normal", "positions", "all"]}},
            "required": ["camera", "start", "end"]},
    },
    {
        "name": "get_dwell_overview",
        "description": "Where time went over a period (any length): person-minutes per place, hotspots described relative to the map, "
                       "UNUSUAL flags for time outside work spots. No time order, no identities, no counts of people. Use for 'where did "
                       "people spend time', 'anything unusual this shift', and as the FIRST call for open questions over long periods; "
                       "then zoom in with get_scene_log. include_heatmap=true attaches a labelled heatmap image (H1.. match the text).",
        "input_schema": {"type": "object", "properties": {
            "camera": {"type": "string"}, "start": {"type": "string", "description": TIME}, "end": {"type": "string", "description": TIME},
            "include_heatmap": {"type": "boolean"}}, "required": ["camera", "start", "end"]},
    },
    {
        "name": "get_events",
        "description": "Detector events in a period, for one camera or all cameras: absence (operator station empty > 60 s), "
                       "long_stationary (someone still ≥3 min: possible rest/sleep, verify with frames), close_pair (chit-chat candidate), "
                       "crowding, fast_movement, outside_work_area_dwell, restricted_presence, plus imported main-system events. "
                       "Use for 'did anything happen', 'any absence', 'was anyone sleeping'. types filters event types.",
        "input_schema": {"type": "object", "properties": {
            "start": {"type": "string", "description": TIME}, "end": {"type": "string", "description": TIME},
            "camera": {"type": "string"}, "types": {"type": "array", "items": {"type": "string"}}}, "required": ["start", "end"]},
    },
    {
        "name": "compare_periods",
        "description": "Same metrics computed the same way for period A and period B on one camera: online minutes, average and peak "
                       "people, entries/exits, empty minutes and short step-aways per operator station, event counts and minutes per type. "
                       "Use for 'today vs yesterday' or 'this shift vs last shift'. Pair with get_dwell_overview for where-time-went differences.",
        "input_schema": {"type": "object", "properties": {
            "camera": {"type": "string"}, "a_start": {"type": "string"}, "a_end": {"type": "string"},
            "b_start": {"type": "string"}, "b_end": {"type": "string"}},
            "required": ["camera", "a_start", "a_end", "b_start", "b_end"]},
    },
    {
        "name": "get_image_frames",
        "description": "Real camera frames to VERIFY a doubtful case (hidden feet, possible sleep, identity questions, what an object or "
                       "person looks like). One frame ('time') or a frame sheet of up to 9 evenly spaced frames between 'start' and 'end', "
                       "in time order, wall-clock printed on each tile, detections boxed (#n is per frame, not an identity), zone outlines "
                       "drawn. crop_zone zooms around one zone. Costly: ask for the fewest frames that answer the doubt.",
        "input_schema": {"type": "object", "properties": {
            "camera": {"type": "string"}, "time": {"type": "string", "description": TIME},
            "start": {"type": "string"}, "end": {"type": "string"}, "count": {"type": "integer", "minimum": 1, "maximum": 9},
            "crop_zone": {"type": "string"}, "overlay": {"type": "boolean"}}, "required": ["camera"]},
    },
]
