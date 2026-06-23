from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST = BASE_DIR / "agent_frontend" / "dist"

sys.path.insert(0, str(BASE_DIR))

import cctv_agent
from cctv_tools import get_camera_info


SESSIONS: dict[str, list] = {}
IMAGE_PATH_RE = re.compile(r"^(?:Grid saved|Frame saved):\s*(.+)$", re.MULTILINE)


def json_response(handler: BaseHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def chunk_text(text: str, size: int = 18):
    words = text.split(" ")
    current = []
    current_len = 0
    for word in words:
        current.append(word)
        current_len += len(word) + 1
        if current_len >= size:
            yield " ".join(current) + " "
            current = []
            current_len = 0
    if current:
        yield " ".join(current)


def require_local_path(raw_path: str) -> Path:
    candidate = Path(unquote(raw_path)).resolve()
    base = BASE_DIR.resolve()
    if candidate == base or base in candidate.parents:
        return candidate
    raise ValueError("Path is outside cctv_intelligence folder")


def extract_image_paths(history_items: list) -> list[Path]:
    paths: list[Path] = []
    seen = set()

    for item in history_items:
        content = ""
        if isinstance(item, dict):
            content = str(item.get("content") or "")
        else:
            content = str(item)

        for match in IMAGE_PATH_RE.finditer(content):
            raw_path = match.group(1).strip()
            try:
                path = require_local_path(raw_path)
            except Exception:
                continue
            if path.exists() and path.is_file() and path not in seen:
                paths.append(path)
                seen.add(path)

    return paths


class AgentWebHandler(BaseHTTPRequestHandler):
    server_version = "CCTVAgentWeb/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/cameras":
            try:
                json_response(self, {"ok": True, "text": get_camera_info()})
            except Exception as exc:
                json_response(self, {"ok": False, "error": str(exc)}, 500)
            return

        if parsed.path == "/api/image":
            try:
                raw_path = parse_qs(parsed.query).get("path", [""])[0]
                self.serve_local_image(require_local_path(raw_path))
            except Exception as exc:
                json_response(self, {"ok": False, "error": str(exc)}, 400)
            return

        self.serve_frontend(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/chat":
            json_response(self, {"ok": False, "error": "Not found"}, 404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            message = str(payload.get("message") or "").strip()
            if not message:
                json_response(self, {"ok": False, "error": "message is required"}, 400)
                return

            session_id = str(payload.get("session_id") or uuid.uuid4())
            provider = str(payload.get("provider") or "openai").strip().lower()
            camera = str(payload.get("camera") or "").strip()

            if provider not in {"openai", "gemini"}:
                provider = "openai"

            history = SESSIONS.setdefault(session_id, [])
            user_message = message
            if camera:
                user_message = (
                    f"The current camera is '{camera}'. Use it as the camera for this question "
                    f"unless the user says otherwise.\n\nUser question: {message}"
                )

            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            header = json.dumps({"type": "session", "session_id": session_id}) + "\n"
            self.wfile.write(header.encode("utf-8"))
            self.wfile.flush()

            status = json.dumps({"type": "status", "text": "Checking CCTV tools and preparing the brief..."}) + "\n"
            self.wfile.write(status.encode("utf-8"))
            self.wfile.flush()

            cctv_agent._PROVIDER = provider
            history_start = len(history)
            response_text, updated_history = cctv_agent.run_agent_turn(history, user_message)
            SESSIONS[session_id] = updated_history

            for chunk in chunk_text(response_text):
                line = json.dumps({"type": "chunk", "text": chunk}) + "\n"
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()
                time.sleep(0.018)

            for image_path in extract_image_paths(updated_history[history_start:]):
                line = json.dumps(
                    {
                        "type": "image",
                        "url": f"/api/image?path={quote(str(image_path), safe='')}",
                        "name": image_path.name,
                    }
                ) + "\n"
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()

            done = json.dumps({"type": "done"}) + "\n"
            self.wfile.write(done.encode("utf-8"))
            self.wfile.flush()
        except Exception as exc:
            try:
                line = json.dumps({"type": "error", "error": str(exc)}) + "\n"
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()
            except Exception:
                json_response(self, {"ok": False, "error": str(exc)}, 500)

    def serve_frontend(self, path: str) -> None:
        if not FRONTEND_DIST.exists():
            json_response(
                self,
                {
                    "ok": False,
                    "error": "Frontend build not found. Run: cd agent_frontend && npm install && npm run build",
                },
                404,
            )
            return

        if path in {"", "/"}:
            file_path = FRONTEND_DIST / "index.html"
        else:
            candidate = (FRONTEND_DIST / path.lstrip("/")).resolve()
            if FRONTEND_DIST.resolve() not in candidate.parents and candidate != FRONTEND_DIST.resolve():
                json_response(self, {"ok": False, "error": "Invalid path"}, 400)
                return
            file_path = candidate if candidate.exists() else FRONTEND_DIST / "index.html"

        if not file_path.exists() or not file_path.is_file():
            json_response(self, {"ok": False, "error": "Not found"}, 404)
            return

        suffix = file_path.suffix.lower()
        mime = {
            ".html": "text/html; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".svg": "image/svg+xml",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }.get(suffix, "application/octet-stream")
        body = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_local_image(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            json_response(self, {"ok": False, "error": "Image not found"}, 404)
            return
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="CCTV Intelligence Agent web server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), AgentWebHandler)
    print(f"CCTV Agent web app: http://{args.host}:{args.port}")
    print(f"Frontend dist: {FRONTEND_DIST}")
    server.serve_forever()


if __name__ == "__main__":
    main()
