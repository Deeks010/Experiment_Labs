from __future__ import annotations

import argparse
import json
import mimetypes
import sqlite3
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"
DB_PATH = BASE_DIR / "cctv_maps.sqlite3"
EDITOR_HTML = BASE_DIR / "map_editor_ui.html"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with connect_db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS cameras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                source_path TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS camera_zones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id INTEGER NOT NULL,
                zone_name TEXT NOT NULL,
                geometry_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(camera_id) REFERENCES cameras(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS floor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id INTEGER NOT NULL,
                start_time_sec REAL NOT NULL,
                end_time_sec REAL NOT NULL,
                data_kind TEXT NOT NULL,
                subject_count INTEGER,
                person_path_json TEXT,
                confidence REAL,
                evidence_json TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(camera_id) REFERENCES cameras(id) ON DELETE CASCADE
            );
            """
        )


def safe_json_load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def path_to_url(path: str | Path) -> str:
    resolved = Path(path).resolve()
    return f"/api/image?path={quote_path(str(resolved))}"


def quote_path(value: str) -> str:
    from urllib.parse import quote

    return quote(value, safe="")


def require_local_path(raw_path: str) -> Path:
    candidate = Path(raw_path).resolve()
    base = BASE_DIR.resolve()
    if candidate == base or base in candidate.parents:
        return candidate
    raise ValueError("Path is outside cctv_intelligence folder")


def list_runs() -> list[dict]:
    runs = []
    if not RUNS_DIR.exists():
        return runs
    # Sort by modification time, descending (newest first)
    dirs = sorted(RUNS_DIR.iterdir(), key=lambda d: d.stat().st_mtime if d.is_dir() else 0, reverse=True)
    for item in dirs:
        map_path = item / "grid_camera_map.json"
        frame_path = item / "extracted_frame.jpg"
        if item.is_dir() and map_path.exists() and frame_path.exists():
            runs.append({"name": item.name, "map_path": str(map_path), "frame_path": str(frame_path)})
    return runs


def map_item_to_zone(item: dict, index: int) -> dict:
    box = item.get("box") or [0, 0, 100, 100]
    points = item.get("points") or [
        [box[0], box[1]],
        [box[2], box[1]],
        [box[2], box[3]],
        [box[0], box[3]],
    ]
    metadata = {
        "source_id": item.get("id") or f"zone_{index + 1}",
        "description": item.get("description", ""),
        "visual_disambiguation": item.get("visual_disambiguation", ""),
        "why_map_anchor": item.get("why_map_anchor", ""),
        "importance": item.get("importance", ""),
        "needs_review": item.get("needs_review", False),
        "quality_flags": item.get("quality_flags", []),
        "source": item.get("source", {}),
    }
    return {
        "id": item.get("id") or f"zone_{index + 1}",
        "number": index + 1,
        "name": item.get("label") or item.get("id") or f"Zone {index + 1}",
        "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
        "points": points,
        "metadata": metadata,
    }


def zone_to_serializable(zone: dict, index: int) -> dict:
    box = zone.get("box") or [0, 0, 100, 100]
    return {
        "id": zone.get("id") or f"zone_{index + 1}",
        "number": index + 1,
        "name": zone.get("name") or f"Zone {index + 1}",
        "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
        "points": zone.get("points") or [
            [float(box[0]), float(box[1])],
            [float(box[2]), float(box[1])],
            [float(box[2]), float(box[3])],
            [float(box[0]), float(box[3])],
        ],
        "metadata": zone.get("metadata") or {},
    }


def get_or_create_camera(conn: sqlite3.Connection, name: str, source_path: str | None = None) -> int:
    row = conn.execute("SELECT id FROM cameras WHERE name = ?", (name,)).fetchone()
    if row:
        if source_path:
            conn.execute("UPDATE cameras SET source_path = COALESCE(NULLIF(?, ''), source_path) WHERE id = ?", (source_path, row["id"]))
        return int(row["id"])
    cur = conn.execute(
        "INSERT INTO cameras (name, source_path, created_at) VALUES (?, ?, ?)",
        (name, source_path or "", now_iso()),
    )
    return int(cur.lastrowid)


def load_run(run_name: str) -> dict:
    run_dir = RUNS_DIR / run_name
    map_path = run_dir / "grid_camera_map.json"
    if not map_path.exists():
        raise FileNotFoundError(f"No grid_camera_map.json found for run {run_name}")
    data = safe_json_load(map_path)
    frame_path = run_dir / "extracted_frame.jpg"
    if not frame_path.exists():
        frame_path = Path(data.get("frame_path", ""))
    edited_path = run_dir / "edited_zones.json"
    if edited_path.exists():
        edited = safe_json_load(edited_path)
        zones = [
            zone_to_serializable(zone, index)
            for index, zone in enumerate(edited.get("zones", []))
        ]
    else:
        zones = [map_item_to_zone(item, index) for index, item in enumerate(data.get("items", []))]
    return {
        "run_name": run_name,
        "camera_name": data.get("camera_id") or run_name,
        "frame_width": data.get("frame_width"),
        "frame_height": data.get("frame_height"),
        "frame_path": str(frame_path.resolve()),
        "image_url": path_to_url(frame_path),
        "zones": zones,
    }


def load_saved_camera(camera_name: str) -> dict | None:
    with connect_db() as conn:
        camera = conn.execute("SELECT * FROM cameras WHERE name = ?", (camera_name,)).fetchone()
        if not camera:
            return None
        rows = conn.execute(
            "SELECT * FROM camera_zones WHERE camera_id = ? ORDER BY id",
            (camera["id"],),
        ).fetchall()
    zones = []
    for index, row in enumerate(rows):
        geometry = json.loads(row["geometry_json"])
        metadata = json.loads(row["metadata_json"])
        box = geometry.get("box") or [0, 0, 100, 100]
        zones.append(
            {
                "id": f"zone_{row['id']}",
                "number": index + 1,
                "name": row["zone_name"],
                "box": box,
                "points": geometry.get("points", []),
                "metadata": metadata,
                "db_id": row["id"],
            }
        )
    return {"camera": dict(camera), "zones": zones}


def save_camera_zones(payload: dict) -> dict:
    camera_name = (payload.get("camera_name") or payload.get("run_name") or "").strip()
    if not camera_name:
        raise ValueError("camera_name is required")

    zones = payload.get("zones")
    if not isinstance(zones, list):
        raise ValueError("zones must be a list")

    saved_run_path = None
    run_name = (payload.get("run_name") or "").strip()
    if run_name:
        run_dir = (RUNS_DIR / run_name).resolve()
        if RUNS_DIR.resolve() not in run_dir.parents:
            raise ValueError("run_name resolves outside runs folder")
        run_dir.mkdir(parents=True, exist_ok=True)
        saved_run_path = run_dir / "edited_zones.json"
        run_payload = {
            "camera_name": camera_name,
            "run_name": run_name,
            "frame_path": payload.get("frame_path"),
            "updated_at": now_iso(),
            "zones": [zone_to_serializable(zone, index) for index, zone in enumerate(zones)],
        }
        saved_run_path.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")

    with connect_db() as conn:
        camera_id = get_or_create_camera(conn, camera_name, payload.get("source_path") or payload.get("frame_path"))
        conn.execute("DELETE FROM camera_zones WHERE camera_id = ?", (camera_id,))
        timestamp = now_iso()
        for zone in zones:
            name = (zone.get("name") or "Unnamed zone").strip()
            box = zone.get("box") or [0, 0, 100, 100]
            geometry = {
                "shape": "box",
                "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                "points": zone.get("points") or [
                    [float(box[0]), float(box[1])],
                    [float(box[2]), float(box[1])],
                    [float(box[2]), float(box[3])],
                    [float(box[0]), float(box[3])],
                ],
            }
            metadata = zone.get("metadata") or {}
            metadata["editor_number"] = zone.get("number")
            conn.execute(
                """
                INSERT INTO camera_zones
                    (camera_id, zone_name, geometry_json, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (camera_id, name, json.dumps(geometry), json.dumps(metadata), timestamp, timestamp),
            )
    return {
        "ok": True,
        "camera_name": camera_name,
        "zone_count": len(zones),
        "saved_run_path": str(saved_run_path) if saved_run_path else None,
    }


class MapEditorHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def send_json(self, payload: dict | list, status: int = 200) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, message: str, status: int = 400) -> None:
        self.send_json({"ok": False, "error": message}, status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self.serve_file(EDITOR_HTML)
            elif parsed.path == "/api/runs":
                self.send_json({"runs": list_runs()})
            elif parsed.path == "/api/load-run":
                run = parse_qs(parsed.query).get("run", [""])[0]
                self.send_json(load_run(run))
            elif parsed.path == "/api/load-saved":
                camera = parse_qs(parsed.query).get("camera", [""])[0]
                data = load_saved_camera(camera)
                self.send_json(data or {"camera": None, "zones": []})
            elif parsed.path == "/api/image":
                raw_path = parse_qs(parsed.query).get("path", [""])[0]
                self.serve_file(require_local_path(raw_path))
            else:
                self.send_error_json("Not found", 404)
        except Exception as exc:
            self.send_error_json(str(exc), 500)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            if parsed.path == "/api/save-zones":
                self.send_json(save_camera_zones(payload))
            else:
                self.send_error_json("Not found", 404)
        except Exception as exc:
            self.send_error_json(str(exc), 500)

    def serve_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error_json("File not found", 404)
            return
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local CCTV map editor")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    init_db()
    server = ThreadingHTTPServer((args.host, args.port), MapEditorHandler)
    print(f"Map editor running at http://{args.host}:{args.port}")
    print(f"SQLite DB: {DB_PATH}")
    server.serve_forever()


if __name__ == "__main__":
    main()
