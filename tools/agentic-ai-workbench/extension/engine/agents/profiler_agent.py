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
    peak_memory_kb = None
    raw_output = ""

    # For CausalLM binaries, pass the model directory as an argument
    cmd = [binary_path]
    model_dir = os.path.join(state.get("out_dir", ""), "run_model")
    if os.path.isdir(model_dir):
        cmd.append(model_dir)

    try:
        start = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
        elapsed_ms = (time.perf_counter() - start) * 1000
        raw_output = (result.stdout or "") + (result.stderr or "")

        # Parse e2e time from binary output: "[e2e time]: X ms"
        import re
        e2e_match = re.search(r'\[e2e time\]:\s*(\d+(?:\.\d+)?)\s*ms', raw_output)
        if e2e_match:
            total_ms = float(e2e_match.group(1))
            bus.log(f"Binary reported e2e latency: {total_ms:.2f} ms")
        else:
            total_ms = elapsed_ms
            bus.log(f"Using wall-clock latency: {total_ms:.2f} ms (binary did not report e2e time)")

        # Parse peak memory: "Peak memory usage (VmRSS): X KB"
        mem_match = re.search(r'Peak memory usage.*?:\s*(\d+(?:\.\d+)?)\s*KB', raw_output, re.IGNORECASE)
        if mem_match:
            peak_memory_kb = int(mem_match.group(1))
            bus.log(f"Peak memory: {peak_memory_kb} KB")

        if result.returncode != 0:
            bus.log(f"Binary exited with code {result.returncode}", "warn")

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

    # Parse prefill/generation breakdown from output
    prefill_ms = None
    generation_ms = None
    prefill_match = re.search(r'prefill:\s*(\d+)\s*tokens,\s*(\d+(?:\.\d+)?)\s*ms', raw_output)
    if prefill_match:
        prefill_ms = float(prefill_match.group(2))
        bus.log(f"Prefill: {prefill_match.group(1)} tokens, {prefill_ms:.2f} ms")
    
    gen_match = re.search(r'generation:\s*(\d+)\s*tokens,\s*(\d+(?:\.\d+)?)\s*ms', raw_output)
    if gen_match:
        generation_ms = float(gen_match.group(2))
        bus.log(f"Generation: {gen_match.group(1)} tokens, {generation_ms:.2f} ms")

    profile = {
        "ran": True,
        "total_latency_ms": round(total_ms, 3),
        "peak_memory_kb": peak_memory_kb,
        "prefill_latency_ms": prefill_ms,
        "generation_latency_ms": generation_ms,
        "layers": layers,
        "note": "per-layer figures are proportional estimates from op weight, not a traced measurement",
    }
    if peak_memory_kb:
        profile["peak_memory_mb"] = round(peak_memory_kb / 1024, 2)
    state["profile"] = profile


    bus.profile(profile)
    bus.log(f"Profiling run complete -- total latency {total_ms:.2f} ms")
    bus.agent_status("profiler", "done", f"{total_ms:.2f} ms")
    return state
