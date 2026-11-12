"""Lightweight HTTP collector and diagnostics server for whyTrace.ai."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from .trace import Trace, load_traces, save_traces

DEFAULT_STORE = Path("runs/live/traces.jsonl")


class WhyTraceRequestHandler(BaseHTTPRequestHandler):
    store_path: Path = DEFAULT_STORE

    def _send_json(self, status: int, data: Any) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "service": "whyTrace.ai collector"})
        elif self.path.startswith("/api/v1/traces"):
            if self.store_path.exists():
                traces = load_traces(self.store_path)
                self._send_json(200, {"count": len(traces), "traces": [t.model_dump() for t in traces]})
            else:
                self._send_json(200, {"count": 0, "traces": []})
        else:
            self._send_json(404, {"error": "Not Found"})

    def do_POST(self) -> None:
        if self.path == "/api/v1/traces":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
                trace = Trace.model_validate(data)
                self.store_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.store_path, "a", encoding="utf-8") as f:
                    f.write(trace.model_dump_json() + "\n")
                self._send_json(201, {"status": "stored", "task_id": trace.task_id})
            except Exception as e:
                self._send_json(400, {"error": f"Invalid trace payload: {e}"})
        else:
            self._send_json(404, {"error": "Not Found"})


def start_server(host: str = "127.0.0.1", port: int = 8080, store_path: Path = DEFAULT_STORE) -> HTTPServer:
    """Start the whyTrace live collector server."""
    WhyTraceRequestHandler.store_path = Path(store_path)
    server = HTTPServer((host, port), WhyTraceRequestHandler)
    print(f"whyTrace.ai collector listening on http://{host}:{port}")
    return server

