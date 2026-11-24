# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file events.py
# @brief Event bus: agents emit JSON lines that the extension forwards to the webview.
# @author prachi-t17 <prachi.t@samsung.com>

"""
Event bus: agents emit structured JSON lines to stdout; the VS Code
extension host reads them line-by-line and forwards each event into the
webview via postMessage. No server process and no WebSocket: the
simplest channel that works.

agents/events.py re-exports this bus for the agents.
"""
import json
import sys
import threading
import time
from typing import Callable, Dict, List


class EventBus:
    """Thread-safe, stdout-based event bus."""

    def __init__(self) -> None:
        self._seq = 0
        self._lock = threading.Lock()
        self._subscribers: Dict[str, List[Callable]] = {}

    # ------------------------------------------------------------------ emit
    def emit(self, payload: dict) -> None:
        """Write one event line to stdout and pass it to the in-process subscribers."""
        with self._lock:
            self._seq += 1
            payload["seq"] = self._seq
            payload["ts"] = time.time()
            sys.stdout.write(json.dumps(payload) + "\n")
            sys.stdout.flush()
        # local in-process subscribers (used by tests / other agents)
        for callback in self._subscribers.get(payload.get("event", ""), []):
            try:
                callback(payload)
            # a subscriber's error must not stop the event
            except Exception:  # pylint: disable=broad-except
                pass

    def subscribe(self, event: str, callback: Callable[[dict], None]) -> None:
        """Call `callback` for every event of this kind (in-process)."""
        self._subscribers.setdefault(event, []).append(callback)

    # ----------------------------------------------------------------- typed helpers
    def log(self, message: str, level: str = "info") -> None:
        """Send a log line."""
        self.emit({"event": "log", "level": level, "message": message})

    def agent_status(self, agent: str, status: str, detail: str = "") -> None:
        """Send a pipeline step's status."""
        self.emit({"event": "agent_status", "agent": agent, "status": status, "detail": detail})

    def chat(self, role: str, content: str, agent: str = "orchestrator") -> None:
        """Send a chat message."""
        self.emit({"event": "chat", "role": role, "content": content, "agent": agent})

    def graph(self, data: dict, target: str = "ir") -> None:
        """Send a graph for one of the webview's panes."""
        self.emit({"event": "graph", "data": data, "target": target})

    def node_mappings(self, mappings: list) -> None:
        """Source<->target node id groups for click-to-highlight between
        the Model Graph and nntrainer Graph tabs (see agents/dual_graph.py)."""
        self.emit({"event": "node_mappings", "mappings": mappings})

    def code(self, filename: str, content: str) -> None:
        """Send generated code."""
        self.emit({"event": "code", "filename": filename, "content": content})

    def file_content(self, target: str, filename: str, content: str) -> None:
        """Send a file's content for one of the webview's panes."""
        self.emit(
            {"event": "file_content", "target": target, "filename": filename, "content": content}
        )

    def profile(self, data: dict) -> None:
        """Send a profiling result."""
        self.emit({"event": "profile", "data": data})

    def artifacts(self, items: list) -> None:
        """Send the list of output files."""
        self.emit({"event": "artifacts", "items": items})

    # pylint: disable-next=too-many-arguments
    def graph_focus(self, graph: str, passed: bool, matched_ids: list,
                    explanation: str, source: str = "rule") -> None:
        """Send the result of a Focus query."""
        self.emit({"event": "graph_focus", "graph": graph, "ok": passed,
                    "matched_ids": matched_ids, "explanation": explanation, "source": source})

    def codegen(self, review: dict) -> None:
        """Claude-CLI code generation result / review state (knowledge/codegen.py
        review.json) for the webview's Review tab."""
        self.emit({"event": "codegen", "data": review})

    def pipeline_complete(self, summary: dict) -> None:
        """Send the summary at the end of the pipeline."""
        self.emit({"event": "pipeline_complete", "summary": summary})

    def error(self, stage: str, message: str) -> None:
        """Send an error from a pipeline step."""
        self.emit({"event": "error", "stage": stage, "message": message})


# Module-level singleton — import this everywhere
bus = EventBus()
