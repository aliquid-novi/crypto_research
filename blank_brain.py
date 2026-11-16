#!/usr/bin/env python3
"""
Blank Brain — local edition.

Runs your Obsidian-style Brain on this computer:
  * serves brain.html at http://localhost:8765 (camera access needs localhost)
  * saves every brain, note and group to brain_data.json next to this script
  * keeps one dated backup per day in ./backups

Hand control (webcam gestures) runs inside the page — press G or the hand
button in the toolbar. No pip installs needed: standard-library Python only.

Usage:
  python blank_brain.py                 # start and open the browser
  python blank_brain.py --port 9000
  python blank_brain.py --data ~/Notes/brain_data.json --no-browser
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGE = HERE / "brain.html"
MAX_BODY = 8 * 1024 * 1024  # 8 MB per write is plenty for notes


class Vault:
    """All docs in one JSON object: {"brains/<id>": {...}, "brains/<id>/notes/<id>": {...}, ...}"""

    def __init__(self, path: Path):
        self.path = path
        self.backups = path.parent / "backups"
        self.lock = threading.Lock()
        self.docs: dict = {}
        if path.exists():
            try:
                self.docs = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(self.docs, dict):
                    raise ValueError("top level is not an object")
            except Exception as e:  # noqa: BLE001
                bad = path.with_suffix(f".corrupt-{int(time.time())}.json")
                shutil.copy2(path, bad)
                print(f"! Could not read {path.name} ({e}). Kept a copy as {bad.name}; starting empty.")
                self.docs = {}

    def snapshot(self) -> bytes:
        with self.lock:
            return json.dumps(self.docs, ensure_ascii=False).encode("utf-8")

    def apply(self, path: str, data) -> None:
        with self.lock:
            if data is None:
                self.docs.pop(path, None)
            else:
                self.docs[path] = data
            self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.backups.mkdir(exist_ok=True)
            daily = self.backups / f"{self.path.stem}-{time.strftime('%Y-%m-%d')}.json"
            if not daily.exists():
                shutil.copy2(self.path, daily)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.docs, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)


def valid_doc_path(p) -> bool:
    if not isinstance(p, str) or not p.startswith("brains/") or len(p) > 300:
        return False
    parts = p.split("/")
    return all(part and part not in (".", "..") for part in parts) and len(parts) in (2, 4)


def make_handler(vault: Vault, port: int):
    allowed_origins = {f"http://localhost:{port}", f"http://127.0.0.1:{port}"}

    class Handler(BaseHTTPRequestHandler):
        server_version = "BlankBrain/1.0"

        def log_message(self, fmt, *args):  # quiet
            if os.environ.get("BRAIN_VERBOSE"):
                super().log_message(fmt, *args)

        def _send(self, code, body: bytes = b"", ctype="application/json; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if body and self.command != "HEAD":
                self.wfile.write(body)

        def _host_ok(self) -> bool:
            host = (self.headers.get("Host") or "").split(":")[0]
            return host in ("localhost", "127.0.0.1")

        def do_GET(self):
            if not self._host_ok():
                return self._send(HTTPStatus.FORBIDDEN, b'{"error":"bad host"}')
            route = self.path.split("?")[0]
            if route in ("/", "/index.html", "/brain.html"):
                try:
                    return self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
                except FileNotFoundError:
                    return self._send(HTTPStatus.NOT_FOUND, b"brain.html is missing - keep it next to blank_brain.py",
                                      "text/plain; charset=utf-8")
            if route == "/api/data":
                return self._send(HTTPStatus.OK, vault.snapshot())
            if route == "/favicon.ico":
                return self._send(HTTPStatus.NO_CONTENT)
            return self._send(HTTPStatus.NOT_FOUND, b'{"error":"not found"}')

        do_HEAD = do_GET

        def do_PUT(self):
            if not self._host_ok():
                return self._send(HTTPStatus.FORBIDDEN, b'{"error":"bad host"}')
            origin = self.headers.get("Origin")
            if origin and origin not in allowed_origins:
                return self._send(HTTPStatus.FORBIDDEN, b'{"error":"bad origin"}')
            if self.path.split("?")[0] != "/api/doc":
                return self._send(HTTPStatus.NOT_FOUND, b'{"error":"not found"}')
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if n <= 0 or n > MAX_BODY:
                    raise ValueError("bad length")
                msg = json.loads(self.rfile.read(n))
                path, data = msg.get("path"), msg.get("data")
                if not valid_doc_path(path) or not (data is None or isinstance(data, dict)):
                    raise ValueError("bad doc")
            except Exception as e:  # noqa: BLE001
                return self._send(HTTPStatus.BAD_REQUEST, json.dumps({"error": str(e)}).encode())
            try:
                vault.apply(path, data)
            except OSError as e:
                return self._send(HTTPStatus.INTERNAL_SERVER_ERROR, json.dumps({"error": str(e)}).encode())
            return self._send(HTTPStatus.OK, b'{"ok":true}')

    return Handler


def main():
    ap = argparse.ArgumentParser(description="Run the Blank Brain locally.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--data", type=Path, default=HERE / "brain_data.json", help="where notes are saved")
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = ap.parse_args()

    if not PAGE.exists():
        sys.exit(f"brain.html not found. Keep it in the same folder as this script:\n  {HERE}")

    vault = Vault(args.data.expanduser().resolve())
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(vault, args.port))
    except OSError as e:
        sys.exit(f"Port {args.port} is busy ({e}). Try --port {args.port + 1}, or close the other Brain window.")

    url = f"http://localhost:{args.port}/"
    print(f"Blank Brain running at {url}")
    print(f"Notes save to {vault.path}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
