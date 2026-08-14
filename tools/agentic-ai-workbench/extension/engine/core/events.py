"""
Event bus: agents emit structured JSON lines to stdout; the VS Code
extension host reads them line-by-line and forwards each event into the
webview via postMessage.  No server process, no WebSocket — the cheapest
approach that's still honest about what it's doing.

Same wire format as the original agents/events.py so the webview needs no changes.
"""
import json
import sys
import threading
import time
from typing import Any, Callable, Dict, List


class EventBus:
    """Thread-safe, stdout-based event bus."""

    def __init__(self) -> None:
        self._seq = 0
        self._lock = threading.Lock()
        self._subscribers: Dict[str, List[Callable]] = {}

    # ------------------------------------------------------------------ emit
    def _emit(self, payload: dict) -> None:
        with self._lock:
            self._seq += 1
            payload["seq"] = self._seq
            payload["ts"] = time.time()
            sys.stdout.write(json.dumps(payload) + "\n")
            sys.stdout.flush()
        # local in-process subscribers (used by tests / other agents)
        for cb in self._subscribers.get(payload.get("event", ""), []):
            try:
                cb(payload)
            except Exception:
                pass

    def subscribe(self, event: str, callback: Callable[[dict], None]) -> None:
        self._subscribers.setdefault(event, []).append(callback)

    # ----------------------------------------------------------------- typed helpers
    def log(self, message: str, level: str = "info") -> None:
        self._emit({"event": "log", "level": level, "message": message})

    def agent_status(self, agent: str, status: str, detail: str = "") -> None:
        self._emit({"event": "agent_status", "agent": agent, "status": status, "detail": detail})

    def chat(self, role: str, content: str, agent: str = "orchestrator") -> None:
        self._emit({"event": "chat", "role": role, "content": content, "agent": agent})

    def graph(self, data: dict, target: str = "ir") -> None:
        self._emit({"event": "graph", "data": data, "target": target})

    def node_mappings(self, mappings: list) -> None:
        """Source<->target node id groups for click-to-highlight between
        the Model Graph and nntrainer Graph tabs (see agents/dual_graph.py)."""
        self._emit({"event": "node_mappings", "mappings": mappings})

    def code(self, filename: str, content: str) -> None:
        self._emit({"event": "code", "filename": filename, "content": content})

    def file_content(self, target: str, filename: str, content: str) -> None:
        self._emit({"event": "file_content", "target": target, "filename": filename, "content": content})

    def profile(self, data: dict) -> None:
        self._emit({"event": "profile", "data": data})

    def artifacts(self, items: list) -> None:
        self._emit({"event": "artifacts", "items": items})

    def graph_focus(self, graph: str, ok: bool, matched_ids: list,
                    explanation: str, source: str = "rule") -> None:
        self._emit({"event": "graph_focus", "graph": graph, "ok": ok,
                    "matched_ids": matched_ids, "explanation": explanation, "source": source})

    def pipeline_complete(self, summary: dict) -> None:
        self._emit({"event": "pipeline_complete", "summary": summary})

    def error(self, stage: str, message: str) -> None:
        self._emit({"event": "error", "stage": stage, "message": message})


# Module-level singleton — import this everywhere
bus = EventBus()
