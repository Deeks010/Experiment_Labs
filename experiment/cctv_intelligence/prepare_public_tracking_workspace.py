"""Prepare public videos for manual mapping without running person tracking."""
import argparse
import json
import sqlite3
from pathlib import Path

from site_context import load_site, save_site
from workflow_observations import init_observations

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]


def prepare(db: Path, config: Path) -> None:
    payload = json.loads(config.read_text(encoding="utf-8"))
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        conn.executescript("""
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
        """)
        init_observations(conn)
        for source in payload["sources"]:
            video = (REPO / source["video"]).resolve()
            if not video.is_file():
                raise FileNotFoundError(f"Missing video for {source['camera']}: {video}")
            conn.execute("""INSERT INTO cameras(name,source_path,created_at)
                VALUES(?,?,CURRENT_TIMESTAMP)
                ON CONFLICT(name) DO UPDATE SET source_path=excluded.source_path""",
                (source["camera"], str(video)))
            camera_id = conn.execute(
                "SELECT id FROM cameras WHERE name=?", (source["camera"],)
            ).fetchone()[0]
            has_zones = conn.execute(
                "SELECT 1 FROM camera_zones WHERE camera_id=? LIMIT 1", (camera_id,)
            ).fetchone()
            if source.get("zones") and not has_zones:
                for zone in source["zones"]:
                    points = zone["points"]
                    xs, ys = zip(*points)
                    geometry = {
                        "shape": "polygon",
                        "points": points,
                        "box": [min(xs), min(ys), max(xs), max(ys)],
                    }
                    metadata = {
                        "kind": zone["kind"],
                        "description": zone.get("description", ""),
                        "category": zone.get("category", ""),
                    }
                    conn.execute("""INSERT INTO camera_zones
                        (camera_id,zone_name,geometry_json,metadata_json,created_at,updated_at)
                        VALUES(?,?,?, ?,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)""",
                        (camera_id, zone["name"], json.dumps(geometry), json.dumps(metadata)))
    if load_site(db) is None:
        save_site(db, payload["site"])
    print(f"Prepared {len(payload['sources'])} new camera videos for annotation.")
    print(f"Database: {db}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=BASE / "activity_runs" / "public_tracking" / "observations.sqlite3")
    parser.add_argument("--config", type=Path, default=BASE / "public_tracking_site.json")
    args = parser.parse_args()
    prepare(args.db.resolve(), args.config.resolve())
