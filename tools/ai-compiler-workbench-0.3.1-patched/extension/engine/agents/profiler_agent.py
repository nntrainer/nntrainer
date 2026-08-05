"""
Profiler Agent (no LLM).

If the Compiler Agent produced a real binary, runs it with `time` and
reports wall-clock latency plus a per-layer breakdown derived from the
IR's declared attributes (param counts / dtypes) -- nntrainer itself
doesn't emit a JSON trace from this smoke-test main(), so the
per-layer numbers here are a static estimate proportional to parameter
count per layer, clearly labeled as such, not a claim of measured
per-layer timing. If nothing was compiled, this agent says so plainly
instead of inventing numbers.
"""
import os
import subprocess
import time

from .events import bus


def run(state: dict) -> dict:
    bus.agent_status("profiler", "running")

    if not state.get("compiled") or not state.get("binary_path"):
        bus.log("Nothing was compiled -- profiler has no binary to run", "warn")
        state["profile"] = {"ran": False, "reason": state.get("compile_log", "not compiled")}
        bus.agent_status("profiler", "error", "no binary")
        return state

    binary_path = state["binary_path"]
    total_ms = None
    try:
        start = time.perf_counter()
        subprocess.run([binary_path], capture_output=True, timeout=60, check=True)
        total_ms = (time.perf_counter() - start) * 1000
    except Exception as exc:
        bus.log(f"Profiling run failed: {exc}", "warn")
        state["profile"] = {"ran": False, "reason": str(exc)}
        bus.agent_status("profiler", "error", str(exc))
        return state

    graph_view = state.get("graph_view", {"nodes": []})
    weighted_nodes = [n for n in graph_view["nodes"] if n["status"] == "mapped"]
    total_weight = sum(max(1, len(n.get("attributes", {}))) for n in weighted_nodes) or 1

    layers = []
    for n in weighted_nodes:
        weight = max(1, len(n.get("attributes", {})))
        est_ms = round(total_ms * weight / total_weight, 3) if total_ms else None
        layers.append({"name": n["label"], "type": n["type"], "estimated_ms": est_ms})

    profile = {
        "ran": True,
        "total_latency_ms": round(total_ms, 3),
        "layers": layers,
        "note": "per-layer figures are proportional estimates from op weight, not a traced measurement",
    }
    state["profile"] = profile
    bus.profile(profile)
    bus.log(f"Profiling run complete -- total latency {total_ms:.2f} ms")
    bus.agent_status("profiler", "done", f"{total_ms:.2f} ms")
    return state
