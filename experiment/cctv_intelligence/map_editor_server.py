from __future__ import annotations

import argparse
import hashlib
import math
import json
import mimetypes
import os
import sqlite3
import time
from contextlib import closing, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"
DB_PATH = Path(os.environ.get("CCTV_DB_PATH", BASE_DIR / "cctv_maps.sqlite3"))
EDITOR_HTML = BASE_DIR / "map_editor_ui.html"
EDITOR_FRAME_CACHE = BASE_DIR / "editor_frame_cache"
FALLBACK_FRAME = BASE_DIR / "extracted_frame.jpg"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


@contextmanager
def connect_db(db_path=None):
    with closing(sqlite3.connect(db_path or DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        with conn:
            yield conn


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
    if is_known_camera_source(candidate):
        return candidate
    raise ValueError("Path is outside cctv_intelligence folder")


def is_known_camera_source(path: Path) -> bool:
    with connect_db() as conn:
        rows = conn.execute("SELECT source_path FROM cameras WHERE source_path IS NOT NULL").fetchall()
    candidate = path.resolve()
    for row in rows:
        source_path = row["source_path"]
        if source_path and Path(source_path).expanduser().resolve() == candidate:
            return True
    return False


def image_size(path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except Exception:
        return None, None


def extract_video_frame(video_path: Path, camera_name: str) -> Path | None:
    if not video_path.exists():
        return None
    try:
        import cv2

        EDITOR_FRAME_CACHE.mkdir(parents=True, exist_ok=True)
        source_key = hashlib.sha256(str(video_path.resolve()).encode()).hexdigest()[:16]
        frame_path = EDITOR_FRAME_CACHE / f"{source_key}_frame.jpg"
        if frame_path.exists() and frame_path.stat().st_mtime >= video_path.stat().st_mtime:
            return frame_path

        capture = cv2.VideoCapture(str(video_path))
        try:
            ok, frame = capture.read()
            if not ok or frame is None:
                return None
            if not cv2.imwrite(str(frame_path), frame):
                return None
            return frame_path
        finally:
            capture.release()
    except Exception:
        return None


def frame_path_for_camera(camera: sqlite3.Row) -> Path | None:
    raw_source = camera["source_path"] or ""
    source_path = Path(raw_source).expanduser() if raw_source else None
    if source_path and source_path.suffix.lower() in IMAGE_EXTENSIONS and source_path.exists():
        return source_path
    if source_path and source_path.suffix.lower() in VIDEO_EXTENSIONS:
        frame_path = extract_video_frame(source_path, camera["name"])
        if frame_path:
            return frame_path
    if raw_source:
        return None
    if FALLBACK_FRAME.exists():
        return FALLBACK_FRAME
    return None


def list_runs() -> list[dict]:
    runs = []
    with connect_db() as conn:
        cameras = conn.execute(
            """
            SELECT c.*, COUNT(z.id) AS zone_count
            FROM cameras c
            LEFT JOIN camera_zones z ON z.camera_id = c.id
            GROUP BY c.id
            ORDER BY c.id DESC
            """
        ).fetchall()
    for camera in cameras:
        runs.append(
            {
                "name": camera["name"],
                "map_path": "sqlite:camera_zones",
                "frame_path": camera["source_path"] or "",
                "source": "db",
                "zone_count": camera["zone_count"],
            }
        )

    if not RUNS_DIR.exists():
        return runs
    # Sort by modification time, descending (newest first)
    dirs = sorted(RUNS_DIR.iterdir(), key=lambda d: d.stat().st_mtime if d.is_dir() else 0, reverse=True)
    existing_names = {run["name"] for run in runs}
    for item in dirs:
        map_path = item / "grid_camera_map.json"
        frame_path = item / "extracted_frame.jpg"
        if item.name in existing_names:
            continue
        if item.is_dir() and map_path.exists() and frame_path.exists():
            runs.append({"name": item.name, "map_path": str(map_path), "frame_path": str(frame_path), "source": "run"})
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
    run_dir = (RUNS_DIR / run_name).resolve()
    if run_dir.parent != RUNS_DIR.resolve():
        raise ValueError('Invalid run name.')
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


def load_saved_camera(camera_name: str, db_path=None) -> dict | None:
    with connect_db(db_path) as conn:
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
                "shape": geometry.get('shape', 'box'),
                "points": geometry.get("points", []),
                "metadata": metadata,
                "db_id": row["id"],
            }
        )
    frame_path = frame_path_for_camera(camera)
    frame_width, frame_height = image_size(frame_path) if frame_path else (None, None)
    return {
        "run_name": None,
        "camera_name": camera["name"],
        "camera": dict(camera),
        "frame_width": frame_width,
        "frame_height": frame_height,
        "frame_path": str(frame_path.resolve()) if frame_path else "",
        "image_url": path_to_url(frame_path) if frame_path else "",
        "zones": zones,
        "source": "db",
        "revision": map_revision(zones),
    }


def map_revision(zones):
    from workflow_knowledge import map_version
    return map_version([{'name': z['name'], 'kind': z.get('metadata', {}).get('kind', 'static_object'),
        'description': z.get('metadata', {}).get('description', ''),
        'category': z.get('metadata', {}).get('category', ''),
        'geometry': {'shape': z.get('shape','box'), 'box': z['box'], 'points': z.get('points',[])}}
        for z in zones])


def normalize_zones(zones, width, height):
    import cv2
    import numpy as np
    if not isinstance(zones, list) or len(zones)>256:
        raise ValueError('Supply at most 256 objects or areas.')
    result, names = [], set()
    for index, source in enumerate(zones):
        name = str(source.get('name','')).strip()
        if not name or len(name)>100 or name.casefold() in names:
            raise ValueError('Every object needs a unique name of 1-100 characters.')
        names.add(name.casefold())
        shape = source.get('shape','box')
        if shape not in {'box','polygon'}:
            raise ValueError('Choose a rectangle or polygon.')
        if shape == 'box':
            x1,y1,x2,y2 = map(float, source['box'])
            points = [[min(x1,x2),min(y1,y2)],[max(x1,x2),min(y1,y2)],
                      [max(x1,x2),max(y1,y2)],[min(x1,x2),max(y1,y2)]]
        else:
            points = source.get('points', [])
        if not 3 <= len(points) <= 64 or any(len(p)!=2 for p in points):
            raise ValueError('A polygon needs 3-64 vertices.')
        points = [[float(x),float(y)] for x,y in points]
        if any(not math.isfinite(v) for p in points for v in p):
            raise ValueError('Coordinates must be finite.')
        if any(not (0<=x<=width and 0<=y<=height) for x,y in points):
            raise ValueError('Keep all vertices inside the camera image.')
        if len(set(map(tuple,points))) != len(points) or cv2.contourArea(np.asarray(points,np.float32)) < 4:
            raise ValueError('The shape has duplicate vertices or no usable area.')
        def cross(a,b,c): return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
        for i,a in enumerate(points):
            b=points[(i+1)%len(points)]
            for j in range(i+1,len(points)):
                if j in {(i+1)%len(points)} or (j+1)%len(points)==i:
                    continue
                c,d=points[j],points[(j+1)%len(points)]
                if cross(a,b,c)*cross(a,b,d)<=0 and cross(c,d,a)*cross(c,d,b)<=0 and \
                   max(min(a[0],b[0]),min(c[0],d[0]))<=min(max(a[0],b[0]),max(c[0],d[0])) and \
                   max(min(a[1],b[1]),min(c[1],d[1]))<=min(max(a[1],b[1]),max(c[1],d[1])):
                    raise ValueError('Polygon edges must not cross.')
        metadata = dict(source.get('metadata') or {})
        metadata.setdefault('kind','static_object')
        if metadata['kind'] not in {'static_object','floor_area'}:
            raise ValueError('Choose static object or floor area.')
        metadata['description'] = str(metadata.get('description',''))[:2000]
        metadata['category'] = str(metadata.get('category',''))[:100]
        xs,ys=zip(*points)
        result.append({'name':name,'number':index+1,'shape':shape,'points':points,
                       'box':[min(xs),min(ys),max(xs),max(ys)],'metadata':metadata})
    return result


def load_camera_or_run(name: str) -> dict:
    saved = load_saved_camera(name)
    if saved:
        return saved
    return load_run(name)


def save_camera_zones(payload: dict, db_path=None) -> dict:
    camera_name = str(payload.get('camera_name') or payload.get('run_name') or '').strip()
    if not camera_name:
        raise ValueError('camera_name is required')
    saved = load_saved_camera(camera_name, db_path)
    new_camera = saved is None
    if not saved:
        saved = load_run(str(payload.get('run_name') or camera_name))
    width, height = saved.get('frame_width'), saved.get('frame_height')
    if not width or not height:
        raise ValueError('The camera frame is unavailable. No annotations were changed.')
    zones = normalize_zones(payload.get('zones'), width, height)
    revision = map_revision(zones)
    with connect_db(db_path) as conn:
        from site_context import init_site_tables
        init_site_tables(conn)
        conn.execute('BEGIN IMMEDIATE')
        camera_id = get_or_create_camera(conn, camera_name,
            saved.get('frame_path') if new_camera else None)
        current = conn.execute('SELECT * FROM camera_zones WHERE camera_id=? ORDER BY id', (camera_id,)).fetchall()
        existing = [{'name':r['zone_name'], **json.loads(r['geometry_json']),
                     'metadata':json.loads(r['metadata_json'])} for r in current]
        if 'expected_revision' in payload and payload['expected_revision'] != map_revision(existing):
            raise ValueError('Map changed since loading. Reload before saving.')
        conn.execute('DELETE FROM camera_zones WHERE camera_id=?', (camera_id,))
        for zone in zones:
            geometry = {k:zone[k] for k in ('shape','points','box')}
            conn.execute('''INSERT INTO camera_zones
                (camera_id,zone_name,geometry_json,metadata_json,created_at,updated_at)
                VALUES (?,?,?,?,?,?)''', (camera_id,zone['name'],json.dumps(geometry),
                json.dumps(zone['metadata']),now_iso(),now_iso()))
        conn.execute('DELETE FROM camera_map_reviews WHERE camera_name=?', (camera_name,))
        if payload.get('reviewed'):
            if not zones:
                raise ValueError('Add at least one object or area before confirming the map.')
            conn.execute('INSERT INTO camera_map_reviews VALUES (?,?,?)', (camera_name,revision,now_iso()))
    saved_run_path = None
    if saved.get('run_name'):
        saved_run_path = (RUNS_DIR / saved['run_name'] / 'edited_zones.json').resolve()
        saved_run_path.write_text(json.dumps({'camera_name':camera_name,'zones':zones}, indent=2), encoding='utf-8')
    return {'ok':True,'camera_name':camera_name,'zone_count':len(zones),'revision':revision,
            'reviewed':bool(payload.get('reviewed')), 'saved_run_path':str(saved_run_path) if saved_run_path else None}


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
                self.send_json(load_camera_or_run(run))
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
