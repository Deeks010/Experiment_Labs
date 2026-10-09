"""MCP server exposing the 7 supervisor tools (stdio). Works with mcp 1.x (FastMCP) and 2.x (MCPServer).

  python -m supervisor_agent.mcp_server            (from cctv_intelligence/, CCTV_DB_PATH set)

Claude Desktop / Claude Code config example:
  {"mcpServers": {"cctv": {"command": "python", "args": ["-m", "supervisor_agent.mcp_server"],
                           "cwd": "<repo>/experiment/cctv_intelligence", "env": {"CCTV_DB_PATH": "<db>"}}}}
"""
from __future__ import annotations

try:  # mcp 2.x
    from mcp.server.mcpserver import Image, MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server, Image

from . import db as D
from . import tools as T
from .tool_specs import SPECS

DESC = {s["name"]: s["description"] for s in SPECS}
server = _Server("cctv-supervisor")


def _out(res: dict):
    items = [res["text"]]
    for p in res["images"]:
        items.append(Image(path=p))
    return items


def _run(name, **kw):
    with D.connect() as conn:
        return _out(T.call(conn, name, {k: v for k, v in kw.items() if v is not None}))


@server.tool(description=DESC["list_cameras"])
def list_cameras():
    return _run("list_cameras")


@server.tool(description=DESC["get_camera_info"])
def get_camera_info(camera: str, place: str | None = None, include_reference_frame: bool = False):
    return _run("get_camera_info", camera=camera, place=place, include_reference_frame=include_reference_frame)


@server.tool(description=DESC["get_scene_log"])
def get_scene_log(camera: str, start: str, end: str, focus: str | None = None, detail: str = "normal"):
    return _run("get_scene_log", camera=camera, start=start, end=end, focus=focus, detail=detail)


@server.tool(description=DESC["get_dwell_overview"])
def get_dwell_overview(camera: str, start: str, end: str, include_heatmap: bool = False):
    return _run("get_dwell_overview", camera=camera, start=start, end=end, include_heatmap=include_heatmap)


@server.tool(description=DESC["get_events"])
def get_events(start: str, end: str, camera: str | None = None, types: list[str] | None = None):
    return _run("get_events", start=start, end=end, camera=camera, types=types)


@server.tool(description=DESC["compare_periods"])
def compare_periods(camera: str, a_start: str, a_end: str, b_start: str, b_end: str):
    return _run("compare_periods", camera=camera, a_start=a_start, a_end=a_end, b_start=b_start, b_end=b_end)


@server.tool(description=DESC["get_image_frames"])
def get_image_frames(camera: str, time: str | None = None, start: str | None = None, end: str | None = None,
                     count: int = 1, crop_zone: str | None = None, overlay: bool = True):
    return _run("get_image_frames", camera=camera, time=time, start=start, end=end, count=count, crop_zone=crop_zone, overlay=overlay)


if __name__ == "__main__":
    server.run()
