"""
Artifact Manager Agent (no LLM).

Walks <out_dir> and reports every file the pipeline produced (or
downloaded), with size and modified time, for the webview's
"Artifacts" panel and for the final pipeline summary.
"""
import os
import time

from .events import bus


def run(state: dict) -> dict:
    bus.agent_status("artifact_manager", "running")
    out_dir = state["out_dir"]

    items = []
    if os.path.isdir(out_dir):
        for root, _dirs, files in os.walk(out_dir):
            for name in files:
                full_path = os.path.join(root, name)
                rel_path = os.path.relpath(full_path, out_dir)
                try:
                    stat = os.stat(full_path)
                    items.append({
                        "path": rel_path,
                        "type": _kind(name),
                        "size_bytes": stat.st_size,
                        "modified": time.strftime("%H:%M:%S", time.localtime(stat.st_mtime)),
                    })
                except OSError:
                    continue

    items.sort(key=lambda i: i["path"])
    state["artifacts"] = items
    bus.artifacts(items)
    bus.log(f"Collected {len(items)} artifact(s) in {out_dir}")
    bus.agent_status("artifact_manager", "done", f"{len(items)} files")
    return state


def _kind(name: str) -> str:
    ext = os.path.splitext(name)[1].lower()
    return {
        ".cpp": "C++ Source", ".h": "C++ Header", ".ini": "INI File",
        ".json": "JSON File", ".bin": "Binary File", ".safetensors": "Weights",
        ".log": "Log File", ".txt": "Text File",
    }.get(ext, "File")
