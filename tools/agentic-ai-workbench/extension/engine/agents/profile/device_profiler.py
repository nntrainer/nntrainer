"""
Device Profiler Agent (no LLM).

Takes the already-generated createModel() from generated_model.cpp and
wraps it in a real, runnable profiling harness: construct -> compile ->
initialize -> load the real converted weights -> run N warm-up + M
timed forward passes -> report wall-clock latency, measured on this
actual machine (x86_64 or arm64, auto-detected) against a real local
nntrainer install.

This reuses the exact same createLayer()/addLayer()/NeuralNetwork
pattern the C++ Generator Agent already emits (the same one used
throughout nntrainer's own Applications/*, including Applications/
CausalLM) -- nothing new is invented for model *construction*. Two
lines are genuinely new: loading real weights and running a timed
forward pass. Those two calls are isolated in their own clearly
commented block so that if your installed nntrainer version's exact
load()/forwarding() signature differs slightly, the compiler error
points at exactly that block, not the whole file.

Per-layer timing figures are NOW REAL measurements from nntrainer's
built-in profiler (PROFILE_TIME_START/END macros). The profiler captures
actual per-layer forward pass times using nntrainer's internal profiling
hooks. Requires nntrainer built with -Dprofile=true.
"""
import json
import os
import platform
import shutil
import subprocess
import time

from ..events import bus
from ..nntrainer_env import discover_flags

WARMUP_ITERS = 3
TIMED_ITERS = 10


def detect_arch() -> str:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x86_64"
    if machine in ("arm64", "aarch64"):
        return "arm64"
    return machine or "unknown"


def run(state: dict, nntrainer_path: str) -> dict:
    bus.agent_status("device_profiler", "running")

    emission_mode = state.get("cpp_emission_mode", "")
    if hasattr(emission_mode, "value"):
        emission_mode = emission_mode.value

    if str(emission_mode).lower() == "causallm_component":
        # Check if causallm_so_builder already built the binary
        # If so, profile it directly instead of skipping
        causallm_binary = state.get("causallm_exe_path") or state.get("binary_path")
        if causallm_binary and os.path.exists(causallm_binary):
            bus.log(f"CausalLM binary found: {causallm_binary} -- profiling directly", "info")
            return _profile_causallm_binary(state, causallm_binary)
        
        # No binary available - skip profiling
        message = (
            "Profiling skipped for CausalLM component: no standalone binary "
            "is produced. Build and profile it through the CausalLM project."
        )
        state["device_profile_skipped"] = True
        bus.log(message, "info")
        bus.agent_status("device_profiler", "skipped", "no CausalLM executable")
        return state

    if state.get("compile_skipped") and not state.get("binary_path"):
        message = "Profiling skipped because no runnable binary was produced."
        state["device_profile_skipped"] = True
        bus.log(message, "info")
        bus.agent_status("device_profiler", "skipped", "no binary")
        return state

    cpp_path = state.get("cpp_path")
    if not cpp_path or not os.path.exists(cpp_path):
        bus.log("No generated_model.cpp available -- run the pipeline first", "error")
        bus.agent_status("device_profiler", "error", "no generated_model.cpp")
        return state

    arch = detect_arch()
    bus.log(f"Target: this machine ({arch}) -- native build, no cross-compilation")

    # Shared nntrainer discovery -- same include/lib/link flags (and multiarch
    # lib probing) the Compiler Agent uses, so the three compile paths can't
    # drift. discover_flags(prefix=...) resolves lib/<arch>-linux-gnu for us.
    cflags, libs, source = discover_flags(prefix=nntrainer_path)
    if cflags is None:
        bus.log(f"nntrainer not found under '{nntrainer_path}' -- check the path in Settings", "error")
        bus.agent_status("device_profiler", "error", "nntrainer include dir not found")
        return state
    bus.log(f"Using nntrainer via {source}")
    # Derive the resolved lib dir (the token right after "-L") for the rpath.
    lib_dir = libs[libs.index("-L") + 1] if "-L" in libs else os.path.join(nntrainer_path, "lib")

    out_dir = state["out_dir"]
    profile_dir = os.path.join(out_dir, "profile")
    os.makedirs(profile_dir, exist_ok=True)
    harness_path = os.path.join(profile_dir, "profiling_harness.cpp")
    binary_path = os.path.join(profile_dir, "profiling_harness_bin")
    report_path = os.path.join(profile_dir, "profile_report.json")

    weights_path = state.get("converted_weights_path") or ""

    _write_harness(cpp_path, harness_path, report_path, weights_path)

    if not shutil.which("g++"):
        bus.log("g++ not found on PATH -- cannot build the profiling harness", "error")
        bus.agent_status("device_profiler", "error", "g++ not found")
        return state

    cmd = ["g++", "-std=c++17", "-O2", harness_path, "-o", binary_path]
    cmd += cflags + libs + ["-Wl,-rpath," + lib_dir]
    bus.log(f"Compiling profiling harness: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except Exception as exc:
        bus.log(f"Harness compilation failed to start: {exc}", "error")
        bus.agent_status("device_profiler", "error", str(exc))
        return state

    if proc.returncode != 0:
        bus.log(f"Harness compilation failed (exit {proc.returncode})", "error")
        for line in (proc.stdout + proc.stderr).splitlines()[:40]:
            bus.log(f"  {line}", "error")
        bus.log(
            "If the error is on the 'load(...)' or 'forwarding(...)' line near the bottom "
            "of profiling_harness.cpp, your nntrainer version's exact signature differs "
            "slightly -- check neuralnet.h in your install and adjust that one call.",
            "warn",
        )
        bus.agent_status("device_profiler", "error", f"compile exit {proc.returncode}")
        return state

    bus.log("Harness compiled -- running on-device profiling now")
    try:
        run_proc = subprocess.run(
            [binary_path], capture_output=True, text=True, timeout=180,
            env={**os.environ, "LD_LIBRARY_PATH": lib_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")},
        )
    except Exception as exc:
        bus.log(f"Running the profiling harness failed: {exc}", "error")
        bus.agent_status("device_profiler", "error", str(exc))
        return state

    for line in run_proc.stdout.splitlines():
        if not line.startswith("PROFILE_JSON:"):
            bus.log(line)

    if run_proc.returncode != 0:
        bus.log(f"Profiling harness exited with code {run_proc.returncode}", "error")
        for line in run_proc.stderr.splitlines()[:40]:
            bus.log(f"  {line}", "error")
        bus.agent_status("device_profiler", "error", f"runtime exit {run_proc.returncode}")
        return state

    measured = None
    for line in run_proc.stdout.splitlines():
        if line.startswith("PROFILE_JSON:"):
            try:
                measured = json.loads(line[len("PROFILE_JSON:"):])
            except json.JSONDecodeError:
                pass

    if not measured:
        bus.log("Harness ran but produced no PROFILE_JSON line -- see log above", "error")
        bus.agent_status("device_profiler", "error", "no profile output")
        return state

    # Use real per-layer data from nntrainer profiler if available
    layers_data = measured.get("layers", [])
    if layers_data:
        # Real per-layer timing from nntrainer profiler
        layers = [
            {
                "name": layer["name"],
                "type": layer.get("type", "unknown"),
                "time_ms": layer["time_ms"],
                "measured": True
            }
            for layer in layers_data
        ]
        note = (
            "avg/min/max latency is measured on this machine via std::chrono around real "
            "forwarding() calls. Per-layer figures are REAL measurements from nntrainer's "
            "built-in profiler (PROFILE_TIME_START/END macros). Requires nntrainer built "
            "with -Dprofile=true."
        )
        bus.log(f"Got real per-layer profiling data for {len(layers)} layers", "info")
    else:
        # Fall back to estimated bottlenecks if no profiler data
        layers = _estimate_bottlenecks(state, measured["avg_ms"])
        note = (
            "avg/min/max latency is measured on this machine via std::chrono around real "
            "forwarding() calls. Per-layer figures are ESTIMATES proportional to parameter "
            "count (nntrainer profiler not available - rebuild nntrainer with -Dprofile=true)."
        )
        bus.log("No per-layer profiler data - using estimated bottlenecks", "warn")

    report = {
        "ran": True,
        "arch": arch,
        "weights_loaded": bool(weights_path),
        "warmup_iters": WARMUP_ITERS,
        "timed_iters": TIMED_ITERS,
        "avg_ms": measured["avg_ms"],
        "min_ms": measured["min_ms"],
        "max_ms": measured["max_ms"],
        "iterations_ms": measured.get("iterations_ms", []),
        "layers": layers,
        "note": note,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    state["device_profile"] = report
    bus.profile(report)

    # Show top bottleneck layers
    if layers and all("time_ms" in l for l in layers):
        # Real measured times
        top = sorted(layers, key=lambda l: l["time_ms"], reverse=True)[:5]
        bus.log("Top bottleneck layers (real measured times):")
        for layer in top:
            bus.log(f"  {layer['name']} [{layer['type']}] -- {layer['time_ms']:.2f} ms")
    elif layers:
        # Estimated times
        top = sorted(layers, key=lambda l: l["estimated_ms"], reverse=True)[:5]
        bus.log("Likely bottleneck layers (estimated compute share):")
        for layer in top:
            bus.log(f"  {layer['name']} [{layer['type']}] -- ~{layer['estimated_ms']:.2f} ms ({layer['share_pct']:.1f}%, {layer['params']:,} params)")
    bus.log(f"Measured on {arch}: avg {measured['avg_ms']:.2f} ms over {TIMED_ITERS} runs (min {measured['min_ms']:.2f}, max {measured['max_ms']:.2f})")
    bus.chat(
        "assistant",
        f"On-device profiling on **{arch}** complete: {measured['avg_ms']:.2f} ms avg "
        f"over {TIMED_ITERS} runs. Top bottleneck: **{top[0]['name']}** (~{top[0]['share_pct']:.1f}%)."
        if top else f"On-device profiling on **{arch}** complete: {measured['avg_ms']:.2f} ms avg.",
        "device_profiler",
    )

    bus.agent_status("device_profiler", "done", f"{measured['avg_ms']:.2f} ms on {arch}")
    return state


def _write_harness(cpp_path: str, harness_path: str, report_path: str, weights_path: str):
    with open(cpp_path, "r", encoding="utf-8") as f:
        source = f.read()

    # Strip the old smoke-test main() -- this harness supplies its own,
    # real, weight-loading, timed main() instead.
    marker = "#ifdef NNTRAINER_STANDALONE_SMOKE_TEST"
    if marker in source:
        source = source.split(marker)[0]

    weights_literal = json.dumps(weights_path)
    report_literal = json.dumps(report_path)

    main_block = f'''
// ---------------------------------------------------------------------
// Device Profiler Agent's harness main(). Model construction above this
// point is unchanged, generator-emitted code (createLayer/addLayer),
// identical to the pattern used throughout nntrainer's own Applications.
// 
// NEW: Per-layer profiling using nntrainer's built-in profiler.
// Requires nntrainer built with -Dprofile=true
// ---------------------------------------------------------------------
#include <chrono>
#include <fstream>
#include <numeric>
#include <vector>
#include <sstream>
#include <regex>
#include <profiler.h>

int main() {{
    auto model = createModel();
    model->compile();
    model->initialize();

    const std::string weights_path = {weights_literal};
    if (!weights_path.empty()) {{
        // NOTE: adjust this call if your nntrainer version's load() signature differs.
        model->load(weights_path);
        std::cout << "Loaded weights from " << weights_path << std::endl;
    }} else {{
        std::cout << "No converted weights available -- profiling with randomly initialized parameters" << std::endl;
    }}

    const int warmup_iters = {WARMUP_ITERS};
    const int timed_iters = {TIMED_ITERS};

    // Warm-up runs
    for (int i = 0; i < warmup_iters; ++i) {{
        model->forwarding();
    }}

    // Timed runs with per-layer profiling
    std::vector<double> iterations_ms;
    for (int i = 0; i < timed_iters; ++i) {{
        auto start = std::chrono::high_resolution_clock::now();
        model->forwarding();
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        iterations_ms.push_back(ms);
    }}

    double sum = std::accumulate(iterations_ms.begin(), iterations_ms.end(), 0.0);
    double avg = sum / iterations_ms.size();
    double min_ms = *std::min_element(iterations_ms.begin(), iterations_ms.end());
    double max_ms = *std::max_element(iterations_ms.begin(), iterations_ms.end());

    // Capture per-layer profiling data from nntrainer's profiler
    std::ostringstream profiler_out;
    profiler_out << nntrainer::profile::Profiler::Global();
    std::string profiler_text = profiler_out.str();

    // Parse per-layer times from profiler output
    // Format: "LayerName(FORWARD): XXX us"
    std::vector<std::pair<std::string, double>> layer_times;
    std::regex layer_regex(R"((\\w+).*?:\\s*(\\d+)\\s*us)");
    auto begin = std::sregex_iterator(profiler_text.begin(), profiler_text.end(), layer_regex);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {{
        std::string name = (*it)[1].str();
        double time_us = std::stod((*it)[2].str());
        layer_times.push_back({{name, time_us / 1000.0}});  // Convert to ms
    }}

    // Build JSON output with per-layer data
    std::ofstream report({report_literal});
    report << "{{\\n";
    report << "  \\"avg_ms\\": " << avg << ",\\n";
    report << "  \\"min_ms\\": " << min_ms << ",\\n";
    report << "  \\"max_ms\\": " << max_ms << ",\\n";
    report << "  \\"layers\\": [\\n";
    for (size_t i = 0; i < layer_times.size(); ++i) {{
        report << "    {{\\"name\\": \\"" << layer_times[i].first << "\\", \\"time_ms\\": " << layer_times[i].second << "}}";
        if (i < layer_times.size() - 1) report << ",";
        report << "\\n";
    }}
    report << "  ]\\n";
    report << "}}\\n";
    report.close();

    std::cout << "PROFILE_JSON:{{\\"avg_ms\\":" << avg << ",\\"min_ms\\":" << min_ms
              << ",\\"max_ms\\":" << max_ms << ",\\"layers\\":[";
    for (size_t i = 0; i < layer_times.size(); ++i) {{
        if (i > 0) std::cout << ",";
        std::cout << "{{\\"name\\":\\"" << layer_times[i].first << "\\",\\"time_ms\\":" << layer_times[i].second << "}}";
    }}
    std::cout << "]}}" << std::endl;
    
    // Also print profiler output for debugging
    std::cout << "\\n=== nntrainer Profiler Output ===" << std::endl;
    std::cout << profiler_text << std::endl;
    
    return 0;
}}
'''
    with open(harness_path, "w", encoding="utf-8") as f:
        f.write(source)
        f.write("\n#include <algorithm>\n")
        f.write(main_block)


def _profile_causallm_binary(state: dict, binary_path: str) -> dict:
    """
    Profile an already-built CausalLM binary directly.
    
    Args:
        state: Pipeline state dictionary
        binary_path: Path to the CausalLM executable (nntr_causallm)
    
    Returns:
        Updated state with profiling results
    """
    bus.log(f"Running CausalLM binary: {binary_path}", "info")
    
    arch = detect_arch()
    out_dir = state.get("out_dir", "")
    
    # Set up environment with LD_LIBRARY_PATH for nntrainer shared libraries
    env = os.environ.copy()
    nntrainer_root = state.get("nntrainer_root", "/storage_data/snap/Prachi/nntrainer/nntrainer")
    
    # Add nntrainer build directory to library path (where .so files are built)
    build_lib_dir = os.path.join(nntrainer_root, "build", "nntrainer")
    if os.path.isdir(build_lib_dir):
        env["LD_LIBRARY_PATH"] = f"{build_lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"
    
    # Also add the lib and lib64 directories if they exist
    for lib_dir in [os.path.join(nntrainer_root, "lib"), os.path.join(nntrainer_root, "lib64")]:
        if os.path.isdir(lib_dir):
            env["LD_LIBRARY_PATH"] = f"{lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"
    
    bus.log(f"LD_LIBRARY_PATH: {env.get('LD_LIBRARY_PATH', 'not set')}", "info")
    
    # Prepare model directory argument
    cmd = [binary_path]
    model_dir = os.path.join(out_dir, "run_model")
    if os.path.isdir(model_dir):
        cmd.append(model_dir)
    
    total_ms = None
    peak_memory_kb = None
    raw_output = ""
    prefill_ms = None
    generation_ms = None
    
    try:
        start = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
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
        
        # Parse prefill/generation breakdown from output
        prefill_match = re.search(r'prefill:\s*(\d+)\s*tokens,\s*(\d+(?:\.\d+)?)\s*ms', raw_output)
        if prefill_match:
            prefill_ms = float(prefill_match.group(2))
            bus.log(f"Prefill: {prefill_match.group(1)} tokens, {prefill_ms:.2f} ms")
        
        gen_match = re.search(r'generation:\s*(\d+)\s*tokens,\s*(\d+(?:\.\d+)?)\s*ms', raw_output)
        if gen_match:
            generation_ms = float(gen_match.group(2))
            bus.log(f"Generation: {gen_match.group(1)} tokens, {generation_ms:.2f} ms")
        
        if result.returncode != 0:
            bus.log(f"Binary exited with code {result.returncode}", "warn")
        
    except Exception as exc:
        bus.log(f"Profiling run failed: {exc}", "error")
        bus.agent_status("device_profiler", "error", str(exc))
        state["device_profile"] = {"ran": False, "reason": str(exc)}
        return state
    
    # Build profiling report
    report = {
        "ran": True,
        "arch": arch,
        "binary": binary_path,
        "total_latency_ms": round(total_ms, 3) if total_ms else None,
        "peak_memory_kb": peak_memory_kb,
        "prefill_latency_ms": prefill_ms,
        "generation_latency_ms": generation_ms,
        "note": "Direct profiling of CausalLM binary built by causallm_so_builder",
    }
    
    if peak_memory_kb:
        report["peak_memory_mb"] = round(peak_memory_kb / 1024, 2)
    
    state["device_profile"] = report
    bus.profile(report)
    
    # Save the raw model output (including generated text) to a file
    output_file = os.path.join(out_dir, "model_output.txt")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=== Model Execution Output ===\n\n")
            f.write(raw_output)
        bus.log(f"Model output saved to: {output_file}", "info")
        state["model_output_file"] = output_file
        
        # Display the generated text in the chat
        bus.log("Generated text output:", "info")
        for line in raw_output.splitlines()[:50]:  # Show first 50 lines
            bus.log(f"  {line}", "info")
    except Exception as exc:
        bus.log(f"Failed to save model output: {exc}", "warn")
    
    bus.log(f"Profiling complete -- total latency: {total_ms:.2f} ms", "info")
    bus.agent_status("device_profiler", "done", f"{total_ms:.2f} ms")
    
    return state


def _estimate_bottlenecks(state: dict, total_ms: float) -> list:
    graph_ir = state.get("graph_ir") or {}
    nodes = [n for n in graph_ir.get("nodes", []) if n.get("supported") and n.get("parameter_count")]

    total_params = sum(n["parameter_count"] for n in nodes) or 1
    results = []
    for n in nodes:
        share = n["parameter_count"] / total_params
        results.append({
            "name": n["name"],
            "type": n["node_type"],
            "params": n["parameter_count"],
            "share_pct": round(share * 100, 2),
            "estimated_ms": round(total_ms * share, 3),
        })
    return results
