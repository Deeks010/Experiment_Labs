"""Database access for the supervisor agent.

Reads the existing CCTV SQLite schema (cameras, camera_zones, tracking_runs,
frame_observations, recording_clocks) and owns the derived tables (prefix sa_).
Raw tables are never modified.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
DEFAULT_DB = PKG_DIR.parent / "cctv_maps.sqlite3"


def db_path(path: str | os.PathLike | None = None) -> Path:
    return Path(path or os.environ.get("CCTV_DB_PATH") or DEFAULT_DB)


def connect(path=None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path(path)))
    conn.row_factory = sqlite3.Row
    return conn


def load_semantics() -> dict:
    return json.loads((PKG_DIR / "semantics.json").read_text(encoding="utf-8"))


DERIVED_SCHEMA = """
CREATE TABLE IF NOT EXISTS sa_runs (
    run_id TEXT PRIMARY KEY, camera TEXT NOT NULL, built_at TEXT NOT NULL,
    version INTEGER NOT NULL, start_sec REAL, end_sec REAL,
    frame_w INTEGER, frame_h INTEGER, params_json TEXT
);
CREATE TABLE IF NOT EXISTS sa_positions (
    run_id TEXT, t INTEGER, track TEXT, x REAL, y REAL, body_h REAL,
    feet_visible INTEGER, coasted INTEGER, PRIMARY KEY(run_id, t, track)
);
CREATE TABLE IF NOT EXISTS sa_counts (run_id TEXT, start INTEGER, end INTEGER, value INTEGER);
CREATE TABLE IF NOT EXISTS sa_zone_states (run_id TEXT, zone TEXT, start INTEGER, end INTEGER, value INTEGER);
CREATE TABLE IF NOT EXISTS sa_edge_events (run_id TEXT, t INTEGER, direction TEXT, edge TEXT, track TEXT);
CREATE TABLE IF NOT EXISTS sa_episodes (run_id TEXT, track TEXT, start INTEGER, end INTEGER, text TEXT, places_json TEXT);
CREATE TABLE IF NOT EXISTS sa_short_visits (run_id TEXT, station TEXT, visit_zone TEXT, start INTEGER, end INTEGER);
CREATE TABLE IF NOT EXISTS sa_dwell (run_id TEXT, minute INTEGER, cx INTEGER, cy INTEGER, seconds REAL);
CREATE TABLE IF NOT EXISTS sa_events (
    run_id TEXT, type TEXT, start INTEGER, end INTEGER, zone TEXT, details_json TEXT,
    confidence REAL, source TEXT
);
CREATE TABLE IF NOT EXISTS sa_health (run_id TEXT, start INTEGER, end INTEGER, status TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS sa_camera_info (camera TEXT PRIMARY KEY, description TEXT, metadata_json TEXT);
CREATE INDEX IF NOT EXISTS sa_pos_t ON sa_positions(run_id, t);
CREATE INDEX IF NOT EXISTS sa_ev_t ON sa_events(run_id, start);
"""

DERIVED_TABLES = ["sa_positions", "sa_counts", "sa_zone_states", "sa_edge_events", "sa_episodes",
                  "sa_short_visits", "sa_dwell", "sa_events", "sa_health"]


def init_derived(conn: sqlite3.Connection) -> None:
    conn.executescript(DERIVED_SCHEMA)


def has_table(conn, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


# ---------------------------------------------------------------- cameras, zones, runs
def list_camera_names(conn) -> list[str]:
    return [r["name"] for r in conn.execute("SELECT name FROM cameras ORDER BY name")]


def resolve_camera(conn, name: str) -> str:
    names = list_camera_names(conn)
    if name in names:
        return name
    low = {n.lower(): n for n in names}
    if name.lower() in low:
        return low[name.lower()]
    hits = [n for n in names if name.lower() in n.lower()]
    if len(hits) == 1:
        return hits[0]
    raise ValueError(f"Unknown camera '{name}'. Known cameras: {', '.join(names) or 'none'}. Call list_cameras.")


def load_zones(conn, camera: str) -> list[dict]:
    rows = conn.execute(
        """SELECT z.zone_name, z.geometry_json, z.metadata_json FROM camera_zones z
           JOIN cameras c ON c.id = z.camera_id WHERE c.name=? ORDER BY z.id""", (camera,)).fetchall()
    zones = []
    for r in rows:
        g = json.loads(r["geometry_json"] or "{}")
        m = json.loads(r["metadata_json"] or "{}")
        pts = g.get("points") or []
        if not pts and g.get("box"):
            x1, y1, x2, y2 = g["box"]
            pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        if len(pts) < 3:
            continue
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        zones.append(dict(name=r["zone_name"], points=[(float(p[0]), float(p[1])) for p in pts],
                          box=(min(xs), min(ys), max(xs), max(ys)), kind=m.get("kind", "static_object"),
                          role=m.get("role", ""), category=m.get("category", ""),
                          description=m.get("description", ""), synonyms=m.get("synonyms", []),
                          meta=m))
    return zones


def camera_runs(conn, camera: str) -> list[dict]:
    if not has_table(conn, "tracking_runs"):
        return []
    rows = conn.execute(
        """SELECT r.run_id, r.source_path, r.source_kind, r.config_json, r.status, c.name AS camera
           FROM tracking_runs r JOIN cameras c ON c.id = r.camera_id WHERE c.name=? ORDER BY r.rowid""",
        (camera,)).fetchall()
    out = []
    for r in rows:
        cfg = json.loads(r["config_json"] or "{}")
        clock = None
        if has_table(conn, "recording_clocks"):
            k = conn.execute("SELECT * FROM recording_clocks WHERE run_id=?", (r["run_id"],)).fetchone()
            if k:
                clock = dict(k)
        start = float(cfg.get("start_sec") or 0)
        end = cfg.get("end_sec")
        if end is None:
            row = conn.execute("SELECT max(time_sec) FROM frame_observations WHERE run_id=?", (r["run_id"],)).fetchone()
            end = row[0] or start
        out.append(dict(run_id=r["run_id"], camera=r["camera"], source_path=r["source_path"],
                        source_kind=r["source_kind"], status=r["status"], config=cfg,
                        start_sec=start, end_sec=float(end), clock=clock))
    return out


# ---------------------------------------------------------------- time handling
def _zero(run: dict) -> datetime | None:
    if run.get("clock"):
        return datetime.fromisoformat(run["clock"]["source_zero_at"].replace("Z", "+00:00")).replace(tzinfo=None)
    return None


def run_window_wall(run: dict) -> tuple[datetime, datetime] | None:
    z = _zero(run)
    if z is None:
        return None
    return z + timedelta(seconds=run["start_sec"]), z + timedelta(seconds=run["end_sec"])


def fmt_t(run: dict, sec: float, with_date: bool = False) -> str:
    """Format run seconds as wall-clock (HH:MM:SS) when a clock is bound, else as video time."""
    z = _zero(run)
    if z is None:
        s = int(round(sec)); return f"v{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"
    t = z + timedelta(seconds=round(sec))
    return t.strftime("%Y-%m-%d %H:%M:%S" if with_date else "%H:%M:%S")


def parse_time_arg(value, run: dict, default_date: str | None = None) -> float:
    """Accept seconds (number or 'v00:01:30' video time) or wall-clock 'HH:MM[:SS]' / 'YYYY-MM-DD HH:MM[:SS]'.
    Returns run seconds."""
    if value is None:
        raise ValueError("time missing")
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace("T", " ")
    if s.startswith("v"):
        h, m, sec = (s[1:].split(":") + ["0", "0"])[:3]
        return int(h) * 3600 + int(m) * 60 + float(sec)
    try:
        return float(s)
    except ValueError:
        pass
    z = _zero(run)
    if z is None:
        raise ValueError("This run has no wall-clock binding. Use video time like 'v00:05:00' or seconds.")
    parts = s.split(" ")
    if len(parts) == 2:
        d, tm = parts
    else:
        d, tm = (default_date or z.strftime("%Y-%m-%d")), parts[0]
    tm = tm if tm.count(":") == 2 else tm + ":00"
    dt = datetime.fromisoformat(f"{d} {tm}")
    return (dt - z).total_seconds()


def runs_for_window(conn, camera: str, start, end) -> list[tuple[dict, float, float]]:
    """Return [(run, start_sec, end_sec)] for runs of this camera that overlap the requested window.
    start/end may be wall-clock strings or run seconds. With no clock, the latest run is used."""
    runs = camera_runs(conn, camera)
    if not runs:
        raise ValueError(f"No tracking runs for camera '{camera}'.")
    # an end time without a date takes the date of the start time
    if isinstance(start, str) and isinstance(end, str):
        s_, e_ = start.strip().replace("T", " "), end.strip().replace("T", " ")
        if " " in s_ and " " not in e_ and not e_.startswith("v"):
            end = s_.split(" ")[0] + " " + e_
    out = []
    for run in runs:
        try:
            a = parse_time_arg(start, run) if start is not None else run["start_sec"]
            b = parse_time_arg(end, run) if end is not None else run["end_sec"]
        except ValueError:
            continue
        a2, b2 = max(a, run["start_sec"]), min(b, run["end_sec"])
        if b2 > a2:
            out.append((run, a2, b2))
    if not out and all(r["clock"] is None for r in runs):
        run = runs[-1]
        a = parse_time_arg(start, run) if start is not None else run["start_sec"]
        b = parse_time_arg(end, run) if end is not None else run["end_sec"]
        out.append((run, max(a, run["start_sec"]), min(b, run["end_sec"])))
    return out


def coverage_text(conn, camera: str) -> str:
    lines = []
    for run in camera_runs(conn, camera):
        w = run_window_wall(run)
        span = f"{w[0]:%Y-%m-%d %H:%M:%S} → {w[1]:%H:%M:%S}" if w else f"video {fmt_t(run, run['start_sec'])} → {fmt_t(run, run['end_sec'])} (no wall clock)"
        built = conn.execute("SELECT 1 FROM sa_runs WHERE run_id=?", (run["run_id"],)).fetchone() if has_table(conn, "sa_runs") else None
        lines.append(f"  {span}  run={run['run_id']}  source={run['source_kind']}  derived={'yes' if built else 'NO (run derive)'}")
    return "\n".join(lines) or "  no recordings"
