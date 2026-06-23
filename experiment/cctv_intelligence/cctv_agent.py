"""
cctv_agent.py

Factory Floor Intelligence Agent.

An AI agent that answers questions about factory floor activity using
CCTV tracking data stored in the local SQLite database.

Usage:
    python cctv_agent.py                          # uses OpenAI by default
    python cctv_agent.py --camera d23_run         # pre-sets default camera
    python cctv_agent.py --provider gemini        # use Gemini instead
    python cctv_agent.py --provider openai        # use OpenAI (default)

The agent uses 7 tools:
    1. get_map              — ASCII floor map for any time window
    2. get_camera_info      — available cameras and their metadata
    3. get_zones_info       — static objects/areas mapped on a camera
    4. get_people_count     — tracking segment count timeline
    5. get_day_summary      — full session narrative summary
    6. get_video_frame      — extract real frame image from footage
    7. get_activity_table   — translated spatial facts table for analysis

The agent decides which tools to call based on the question.
It never over-calls (does not pull the full activity table for a simple
headcount question) and never under-calls (will chain map + activity_table
for complex analytical questions).
"""

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path
from dotenv import load_dotenv

# ── Load env ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ── Imports ───────────────────────────────────────────────────────────────────
sys.path.insert(0, str(BASE_DIR))
from cctv_tools import (
    get_camera_info,
    get_zones_info,
    get_map,
    get_people_count,
    get_day_summary,
    get_video_frame,
    get_activity_table,
    get_visual_grid,
)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a Factory Floor Assistant. Your job is to help factory owners understand what happened on their floor.

THE SITUATION & THE DATA
We are tracking factory workers via CCTV. 
- PROS (What we know): We store foot coordinates of detected workers and precise timestamps. We also store "zones" which are strictly the coordinates of static objects (machines, workstations, panels) to give you a map idea of the floor. We know WHERE people were, WHEN they were there, and HOW MANY people were visible.
- CONS (What we don't know): We do not have audio. We do not know employee names. Foot tracking alone cannot tell you exactly WHAT a person was doing with their hands or who they were talking to.

3 RULES
1. Never say a person was inside a machine or zone. They stand next to static objects, not inside them.
2. Never make up data.
3. Plain Language: Do not speak in the language of the stored data. Speak in the language a factory owner understands. The owner knows camera names and the actual object/area names from that floor's map, not technical words like segments, subjects, foot coordinates, zone coordinates, pixels, path points, LEFT-LOWER, concurrent segments, or bounding boxes. Use the actual mapped object/area names from the relevant camera's zone data when describing where people were. If the answer needs to say what a person was doing, first use get_visual_grid for visual evidence instead of guessing from position data. Note that if someone stepped briefly out of view they may be counted twice. Tell a story, not a data dump.

TOOLS
get_camera_info: list cameras or get details for one.
get_zones_info: see what static objects are mapped on a camera. Call once and remember.
get_map: main tool for WHERE questions. Shows floor layout and worker positions visually.
get_people_count: for HOW MANY people and WHEN was it busiest.
get_day_summary: full picture of a session. Call once and reuse.
get_activity_table: detailed analysis like idle detection. Short windows only (30-60s). Not for location.
get_video_frame: extract a single image frame.
get_visual_grid: extracts multiple frames into a grid and analyzes them visually. Use for ANY question requiring you to literally SEE behavior, interactions, or equipment. Up to 9 frames.

SMART REASONING - THINK BEFORE YOU ACT
Be highly efficient. Do not waste tools. When the user asks a question, mentally map it against the PROS and CONS of our data:
- Is this fundamentally impossible? (e.g., "What were their names?"). If so, stop and say you cannot answer.
- Can this be answered by foot positions and timestamps? Use data tools (get_map, get_people_count).
- Does this require seeing behavior? Do NOT give up. First, use data tools to find the interesting moments (e.g., peak activity times, or when people stood close to a machine). Then, call get_visual_grid at those exact timestamps to look and solve the problem visually.
- If an image at a specific timestamp would clearly give a more detailed or reliable answer, use the image/visual tool. Avoid using it when the position/count data already answers the owner's question and the image would not add useful value.
Always exhaust the possibilities of combining data logic + visual confirmation before saying "I cannot answer."
"""

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_camera_info",
            "description": (
                "Returns a list of all available cameras with metadata, or details for a specific camera. "
                "Call with no camera_name to list all cameras. "
                "Call first when you don't know which cameras exist."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "camera_name": {"type": "string", "description": "Camera name (e.g. 'd23_run'). Omit to list all cameras."}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_zones_info",
            "description": (
                "Returns all static objects and areas mapped on a camera — machines, workstations, "
                "panels, aisles — with descriptions and frame positions. "
                "Call to understand factory layout. Remember result; don't call again for same camera."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "camera_name": {"type": "string", "description": "Camera name."}
                },
                "required": ["camera_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_map",
            "description": (
                "Returns an ASCII floor map for a time window. Zones shown as labeled boxes. "
                "Segment positions shown as symbols with trail dots. "
                "Auto-selects: snapshot (≤60s) / map series (≤300s) / heatmap (>300s). "
                "PRIMARY tool for any spatial or location question. Preserve the map exactly in your response."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "camera_name": {"type": "string"},
                    "t_start_sec": {"type": "number", "description": "Start time in seconds from video start."},
                    "t_end_sec": {"type": "number", "description": "End time in seconds from video start."},
                },
                "required": ["camera_name", "t_start_sec", "t_end_sec"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_people_count",
            "description": (
                "Returns a 5-second bucket timeline of concurrent active tracking segments. "
                "Optional zone_name filters to segments near that object (within 12% of frame width). "
                "Use for: headcount questions, busiest period, proximity to a specific object. "
                "NOTE: count = segments, not unique individuals."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "camera_name": {"type": "string"},
                    "t_start_sec": {"type": "number", "description": "Start time. Omit for full session."},
                    "t_end_sec": {"type": "number", "description": "End time. Omit for full session."},
                    "zone_name": {"type": "string", "description": "Partial zone name to filter proximity (e.g. 'Press No.9', 'aisle', 'control panel')."},
                },
                "required": ["camera_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_day_summary",
            "description": (
                "Returns a rich narrative summary of floor activity for a given date: "
                "duration, segment count, region distribution, concurrent timeline, peaks, idle periods. "
                "Use for overview/summary questions. Call once and reuse for follow-ups."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "camera_name": {"type": "string"},
                    "date": {"type": "string", "description": "Date as YYYY-MM-DD. Omit for most recent."},
                },
                "required": ["camera_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_video_frame",
            "description": (
                "Extracts a real JPEG frame from the source video at a given timestamp. "
                "Returns the saved image file path. "
                "Use only when visual confirmation adds real value."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "camera_name": {"type": "string"},
                    "t_sec": {"type": "number", "description": "Timestamp in seconds from video start."},
                },
                "required": ["camera_name", "t_sec"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_activity_table",
            "description": (
                "Returns a flat translated table of foot positions: segment, timestamp, frame region, nearby objects. "
                "NO zone assignments — spatial facts only. "
                "Use for analytical questions: idle detection, movement patterns, dwell distribution. "
                "Use SHORT windows (30–60s). Do NOT use for simple location questions — use get_map instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "camera_name": {"type": "string"},
                    "t_start_sec": {"type": "number"},
                    "t_end_sec": {"type": "number"},
                },
                "required": ["camera_name", "t_start_sec", "t_end_sec"],
            },
        },
    }
]

OPENAI_TOOLS.append({
        "type": "function",
        "function": {
            "name": "get_visual_grid",
            "description": (
                "Extracts video frames at given timestamps, combines them into a labeled grid image, "
                "and uses GPT-4o Vision to answer a visual question. Use for ANY question about "
                "worker behaviour, posture, interactions, groupings, or equipment. "
                "Use other tools first to find relevant timestamps, then call this to actually look."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "camera_name": {"type": "string"},
                    "timestamps": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Timestamps in seconds from video start (up to 9).",
                    },
                    "question": {
                        "type": "string",
                        "description": "The specific visual question to answer from the frames.",
                    },
                },
                "required": ["camera_name", "timestamps", "question"],
            },
        },
    })


# ── Gemini tool declarations ───────────────────────────────────────────────────
from google import genai
from google.genai import types

GEMINI_TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="get_camera_info",
        description="Returns list of all cameras or details for a specific camera. Call with no argument to list all.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={"camera_name": types.Schema(type=types.Type.STRING, description="Camera name. Omit to list all.")},
        ),
    ),
    types.FunctionDeclaration(
        name="get_zones_info",
        description="Returns all mapped static objects/areas for a camera with descriptions and frame positions.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={"camera_name": types.Schema(type=types.Type.STRING)},
            required=["camera_name"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_map",
        description="PRIMARY spatial tool. ASCII floor map for a time window. Auto-picks snapshot/series/heatmap.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "camera_name": types.Schema(type=types.Type.STRING),
                "t_start_sec": types.Schema(type=types.Type.NUMBER),
                "t_end_sec": types.Schema(type=types.Type.NUMBER),
            },
            required=["camera_name", "t_start_sec", "t_end_sec"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_people_count",
        description="5s-bucket timeline of concurrent tracking segments. Optional zone_name for proximity filter.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "camera_name": types.Schema(type=types.Type.STRING),
                "t_start_sec": types.Schema(type=types.Type.NUMBER),
                "t_end_sec": types.Schema(type=types.Type.NUMBER),
                "zone_name": types.Schema(type=types.Type.STRING),
            },
            required=["camera_name"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_day_summary",
        description="Rich narrative summary of floor activity for a given date.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "camera_name": types.Schema(type=types.Type.STRING),
                "date": types.Schema(type=types.Type.STRING),
            },
            required=["camera_name"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_video_frame",
        description="Extracts JPEG frame from video at given timestamp. Returns image file path.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "camera_name": types.Schema(type=types.Type.STRING),
                "t_sec": types.Schema(type=types.Type.NUMBER),
            },
            required=["camera_name", "t_sec"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_activity_table",
        description="Flat translated spatial facts table. Use for analytical questions with SHORT windows (30-60s).",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "camera_name": types.Schema(type=types.Type.STRING),
                "t_start_sec": types.Schema(type=types.Type.NUMBER),
                "t_end_sec": types.Schema(type=types.Type.NUMBER),
            },
            required=["camera_name", "t_start_sec", "t_end_sec"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_visual_grid",
        description="Extracts frames into a grid image and uses GPT-4o Vision to answer visual questions about worker behaviour, interactions, posture, or equipment.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "camera_name": types.Schema(type=types.Type.STRING),
                "timestamps": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.NUMBER),
                    description="Timestamps in seconds, up to 9.",
                ),
                "question": types.Schema(type=types.Type.STRING),
            },
            required=["camera_name", "timestamps", "question"],
        ),
    ),
]
GEMINI_TOOLS = types.Tool(function_declarations=GEMINI_TOOL_DECLARATIONS)

# ── Placeholder to keep old reference names intact ─────────────────────────
TOOL_DECLARATIONS = GEMINI_TOOL_DECLARATIONS
TOOLS = GEMINI_TOOLS


# ── Tool declarations (Gemini format kept for reference) ──────────────────────
# Declarations moved above — see GEMINI_TOOL_DECLARATIONS
FAKE_MARKER = True  # marker so the next block parses correctly
if False:
    TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="get_camera_info",
        description=(
            "Returns a list of all available cameras with their metadata (zone count, last activity), "
            "or detailed info for a specific camera. Call with no argument to list all cameras. "
            "Call with a camera name to get that camera's details."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "camera_name": types.Schema(
                    type=types.Type.STRING,
                    description="Camera name (e.g. 'd23_run'). Leave empty to list all cameras.",
                ),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="get_zones_info",
        description=(
            "Returns all static objects and areas mapped on a given camera — machines, workstations, "
            "panels, aisles — with their descriptions and frame positions. "
            "Call this to understand the factory layout before answering location questions. "
            "Cache this mentally — do not call again for the same camera in the same conversation."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "camera_name": types.Schema(
                    type=types.Type.STRING,
                    description="Camera name (e.g. 'd23_run').",
                ),
            },
            required=["camera_name"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_map",
        description=(
            "Returns an ASCII floor map showing where tracking segments were detected in a time window. "
            "Zones (objects/areas) are shown as labeled boxes. Segment positions shown as symbols with trail dots. "
            "Auto-selects rendering mode: snapshot (≤60s), map series (≤300s), heatmap (>300s). "
            "Use this as your primary tool for any spatial or location question. "
            "The map is sent as preformatted text — preserve it exactly in your response."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "camera_name": types.Schema(
                    type=types.Type.STRING,
                    description="Camera name (e.g. 'd23_run').",
                ),
                "t_start_sec": types.Schema(
                    type=types.Type.NUMBER,
                    description="Start time in seconds from video start.",
                ),
                "t_end_sec": types.Schema(
                    type=types.Type.NUMBER,
                    description="End time in seconds from video start.",
                ),
            },
            required=["camera_name", "t_start_sec", "t_end_sec"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_people_count",
        description=(
            "Returns a timeline of how many tracking segments were concurrently active in 5-second buckets. "
            "Optionally filter to segments detected near a specific zone/object. "
            "Use for: headcount questions, 'when was it busiest', 'was anyone near X'. "
            "NOTE: Count = segments, not unique people."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "camera_name": types.Schema(
                    type=types.Type.STRING,
                    description="Camera name.",
                ),
                "t_start_sec": types.Schema(
                    type=types.Type.NUMBER,
                    description="Start time in seconds. Omit for full session.",
                ),
                "t_end_sec": types.Schema(
                    type=types.Type.NUMBER,
                    description="End time in seconds. Omit for full session.",
                ),
                "zone_name": types.Schema(
                    type=types.Type.STRING,
                    description=(
                        "Optional. Partial zone name to filter by proximity "
                        "(e.g. 'Press No.9', 'aisle', 'control panel'). "
                        "Only counts segments detected within 12% of frame width from this object."
                    ),
                ),
            },
            required=["camera_name"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_day_summary",
        description=(
            "Returns a rich narrative summary of all floor activity for a given date. "
            "Includes: session duration, segment count, floor region distribution, "
            "concurrent activity timeline, peak moments, and idle periods. "
            "Use for overview/summary questions. Call ONCE and use for multiple follow-ups."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "camera_name": types.Schema(
                    type=types.Type.STRING,
                    description="Camera name.",
                ),
                "date": types.Schema(
                    type=types.Type.STRING,
                    description=(
                        "Date in YYYY-MM-DD format. "
                        "Leave empty to use the most recent date with data."
                    ),
                ),
            },
            required=["camera_name"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_video_frame",
        description=(
            "Extracts a real JPEG frame from the source video at a given timestamp. "
            "Returns the saved image file path. "
            "Use only when visual confirmation adds real value — not for every answer."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "camera_name": types.Schema(
                    type=types.Type.STRING,
                    description="Camera name.",
                ),
                "t_sec": types.Schema(
                    type=types.Type.NUMBER,
                    description="Timestamp in seconds from video start.",
                ),
            },
            required=["camera_name", "t_sec"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_activity_table",
        description=(
            "Returns a flat translated table of every tracked foot position in a time range. "
            "Each row: segment ID, timestamp, frame region (LEFT/CENTER/RIGHT × UPPER/MIDDLE/LOWER), "
            "and nearby static objects with proximity labels (RIGHT NEXT TO / VERY CLOSE / NEARBY). "
            "NO zone assignments — spatial facts only. "
            "Use for analytical questions: idle detection, movement patterns, region distribution. "
            "Use SHORT windows (30–60s max) to keep the table manageable. "
            "Do NOT use for simple 'where was someone' questions — use get_map instead."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "camera_name": types.Schema(
                    type=types.Type.STRING,
                    description="Camera name.",
                ),
                "t_start_sec": types.Schema(
                    type=types.Type.NUMBER,
                    description="Start time in seconds.",
                ),
                "t_end_sec": types.Schema(
                    type=types.Type.NUMBER,
                    description="End time in seconds.",
                ),
            },
            required=["camera_name", "t_start_sec", "t_end_sec"],
        ),
    ),
]

TOOLS = types.Tool(function_declarations=TOOL_DECLARATIONS)

# ── Tool dispatcher ───────────────────────────────────────────────────────────
TOOL_FUNCTIONS = {
    "get_camera_info": lambda args: get_camera_info(**args),
    "get_zones_info": lambda args: get_zones_info(**args),
    "get_map": lambda args: get_map(**args),
    "get_people_count": lambda args: get_people_count(**args),
    "get_day_summary": lambda args: get_day_summary(**args),
    "get_video_frame": lambda args: get_video_frame(**args),
    "get_activity_table": lambda args: get_activity_table(**args),
    "get_visual_grid": lambda args: get_visual_grid(**args),
}


def call_tool(name: str, args: dict) -> str:
    fn = TOOL_FUNCTIONS.get(name)
    if not fn:
        return f"ERROR: Unknown tool '{name}'"
    try:
        result = fn(args)
        if name in ("get_map", "get_activity_table", "get_people_count",
                    "get_day_summary", "get_zones_info"):
            return f"```\n{result}\n```"
        return result
    except Exception as e:
        return f"ERROR calling {name}: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# OPENAI AGENT LOOP
# ─────────────────────────────────────────────────────────────────────────────

def print_tool_data(name: str, args: dict, result: str) -> None:
    """Print the exact tool input and output that will be sent back to the agent."""
    print("\n  [Tool data shown to agent]")
    print(f"  Tool: {name}")
    print(f"  Args: {json.dumps(args, ensure_ascii=False)}")
    print("  Result:")
    print(textwrap.indent(result, "    "))
    print("  [End tool data]\n")


def run_agent_turn_openai(history: list, user_message: str) -> tuple[str, list]:
    """OpenAI-backed agent turn with full tool-call loop."""
    from openai import OpenAI
    oai = OpenAI(api_key=OPENAI_API_KEY)

    history.append({"role": "user", "content": user_message})

    while True:
        response = oai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
            tools=OPENAI_TOOLS,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=8192,
        )

        msg = response.choices[0].message
        history.append(msg.model_dump(exclude_unset=False))

        tool_calls = msg.tool_calls or []

        if not tool_calls:
            return (msg.content or "").strip(), history

        print(f"\n  [Tools called: {', '.join(tc.function.name for tc in tool_calls)}]")

        for tc in tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            print(f"  → {tc.function.name}({', '.join(f'{k}={v!r}' for k,v in args.items())})")
            result = call_tool(tc.function.name, args)
            print_tool_data(tc.function.name, args, result)
            history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
        # Loop → model reasons over results


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI AGENT LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_turn_gemini(history: list, user_message: str) -> tuple[str, list]:
    """Gemini-backed agent turn with full tool-call loop."""
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    history.append(
        types.Content(role="user", parts=[types.Part(text=user_message)])
    )

    while True:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[GEMINI_TOOLS],
                temperature=0.2,
                max_output_tokens=8192,
            ),
        )

        content = response.candidates[0].content
        tool_calls = []
        text_parts = []
        for part in content.parts:
            if part.function_call:
                tool_calls.append(part.function_call)
            elif part.text:
                text_parts.append(part.text)

        history.append(content)

        if not tool_calls:
            return "\n".join(text_parts).strip(), history

        print(f"\n  [Tools called: {', '.join(tc.name for tc in tool_calls)}]")

        tool_parts = []
        for tc in tool_calls:
            args = dict(tc.args) if tc.args else {}
            print(f"  → {tc.name}({', '.join(f'{k}={v!r}' for k,v in args.items())})")
            result = call_tool(tc.name, args)
            print_tool_data(tc.name, args, result)
            tool_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=tc.name, response={"result": result}
                    )
                )
            )
        history.append(types.Content(role="tool", parts=tool_parts))


# ── Unified entry point ──────────────────────────────────────────────────────
_PROVIDER = "openai"  # default; overridden by --provider arg


def run_agent_turn(history: list, user_message: str) -> tuple[str, list]:
    if _PROVIDER == "gemini":
        return run_agent_turn_gemini(history, user_message)
    return run_agent_turn_openai(history, user_message)


# ── CLI ───────────────────────────────────────────────────────────────────────
def print_banner():
    provider_label = "OpenAI GPT-4o" if _PROVIDER == "openai" else "Gemini 2.0 Flash"
    print("\n" + "═" * 65)
    print("  FACTORY FLOOR INTELLIGENCE AGENT")
    print(f"  Powered by {provider_label} · CCTV Spatial Tracking Data")
    print("═" * 65)
    print("  Type your question about the factory floor.")
    print("  Type 'exit' or 'quit' to stop.")
    print("  Type 'clear' to reset conversation history.")
    print("═" * 65 + "\n")


def wrap_response(text: str) -> str:
    """Pretty-print agent response, preserving code blocks."""
    in_block = False
    lines = text.split("\n")
    out = []
    for line in lines:
        if line.startswith("```"):
            in_block = not in_block
            out.append(line)
        elif in_block:
            out.append(line)          # preserve exactly
        else:
            # Wrap long prose lines
            if len(line) > 100:
                out.extend(textwrap.wrap(line, width=100))
            else:
                out.append(line)
    return "\n".join(out)


def main():
    global _PROVIDER

    parser = argparse.ArgumentParser(description="Factory Floor Intelligence Agent")
    parser.add_argument("--camera", default=None, help="Pre-set a default camera (e.g. d23_run)")
    parser.add_argument(
        "--provider", default="openai", choices=["openai", "gemini"],
        help="LLM provider to use (default: openai)"
    )
    args = parser.parse_args()

    _PROVIDER = args.provider

    if _PROVIDER == "openai" and not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not found. Add it to .env or use --provider gemini")
        sys.exit(1)
    if _PROVIDER == "gemini" and not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not found. Add it to .env or use --provider openai")
        sys.exit(1)

    print_banner()

    history = []

    # If camera pre-set, prime the conversation
    if args.camera:
        primer = (
            f"The user is working with camera '{args.camera}'. "
            f"Keep this as the default camera for all queries unless the user specifies otherwise."
        )
        if _PROVIDER == "openai":
            history.append({"role": "user", "content": primer})
            history.append({"role": "assistant", "content": f"Understood. I'll use '{args.camera}' as the default camera."})
        else:
            history.append(types.Content(role="user", parts=[types.Part(text=primer)]))
            history.append(types.Content(role="model", parts=[types.Part(text=f"Understood. I'll use '{args.camera}' as the default camera.")]))
        print(f"  Default camera set to: {args.camera}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "bye"):
            print("\nGoodbye.")
            break

        if user_input.lower() == "clear":
            history = []
            print("  [Conversation history cleared]\n")
            continue

        print()
        try:
            response_text, history = run_agent_turn(history, user_input)
            print(f"\nAgent:\n{wrap_response(response_text)}\n")
        except Exception as e:
            print(f"\n[ERROR] {e}\n")
            import traceback
            traceback.print_exc()

        print("─" * 65)


if __name__ == "__main__":
    main()
