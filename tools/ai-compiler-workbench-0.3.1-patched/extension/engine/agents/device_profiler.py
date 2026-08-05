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

Per-layer "bottleneck" figures are a real number (measured total
latency on your machine) distributed across layers proportional to
each layer's real parameter count (captured during tracing from the
model's actual nn.Module parameters) -- a reasonable, defensible proxy
for compute share, but still a proportional estimate, not a traced
per-layer measurement. nntrainer's public C++ API doesn't expose a
documented stable "time this one layer" hook we could call instead
without risking exactly the kind of confident-but-wrong code this tool
exists to avoid.
"""
import json
import os
import platform
import shutil
import subprocess

from .events import bus
from .nntrainer_env import discover_flags

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

    bottlenecks = _estimate_bottlenecks(state, measured["avg_ms"])
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
        "layers": bottlenecks,
        "note": (
            "avg/min/max latency is measured on this machine via std::chrono around real "
            "forwarding() calls. Per-layer figures below distribute that real measured total "
            "proportional to each layer's real parameter count -- a reasonable compute-share "
            "proxy, not a traced per-layer measurement (nntrainer's public API doesn't expose "
            "a documented stable per-layer timing hook)."
        ),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    state["device_profile"] = report
    bus.profile(report)

    top = sorted(bottlenecks, key=lambda l: l["estimated_ms"], reverse=True)[:5]
    bus.log(f"Measured on {arch}: avg {measured['avg_ms']:.2f} ms over {TIMED_ITERS} runs (min {measured['min_ms']:.2f}, max {measured['max_ms']:.2f})")
    if top:
        bus.log("Likely bottleneck layers (estimated compute share):")
        for layer in top:
            bus.log(f"  {layer['name']} [{layer['type']}] -- ~{layer['estimated_ms']:.2f} ms ({layer['share_pct']:.1f}%, {layer['params']:,} params)")
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
// Only the two calls below (load real weights, run a timed forwarding
// pass) are new -- if either fails to compile, check neuralnet.h in
// your nntrainer install for the exact signature on your version.
// ---------------------------------------------------------------------
#include <chrono>
#include <fstream>
#include <numeric>
#include <vector>

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

    for (int i = 0; i < warmup_iters; ++i) {{
        // NOTE: adjust this call if your nntrainer version's forwarding() signature differs.
        model->forwarding();
    }}

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

    std::ofstream report({report_literal});
    report << "{{\\n";
    report << "  \\"avg_ms\\": " << avg << ",\\n";
    report << "  \\"min_ms\\": " << min_ms << ",\\n";
    report << "  \\"max_ms\\": " << max_ms << "\\n";
    report << "}}\\n";
    report.close();

    std::cout << "PROFILE_JSON:{{\\"avg_ms\\":" << avg << ",\\"min_ms\\":" << min_ms
              << ",\\"max_ms\\":" << max_ms << "}}" << std::endl;
    return 0;
}}
'''
    with open(harness_path, "w", encoding="utf-8") as f:
        f.write(source)
        f.write("\n#include <algorithm>\n")
        f.write(main_block)


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
