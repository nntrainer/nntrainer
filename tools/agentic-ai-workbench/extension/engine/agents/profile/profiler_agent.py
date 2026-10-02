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

from ..events import bus


def run(state: dict) -> dict:
    bus.agent_status("profiler", "running")

    if not state.get("weights_verified"):
        bus.log("Verified model weights are unavailable -- profiler will not run the model", "error")
        state["profile"] = {"ran": False, "reason": "weights not verified"}
        bus.agent_status("profiler", "error", "weights not verified")
        return state

    # Check for binary from either compiler_agent (compiled) or causallm_so_builder (causallm_exe_path)
    binary_path = state.get("binary_path")
    
    # If causallm_so_builder built the binary, use that path
    if not binary_path and state.get("causallm_exe_path"):
        binary_path = state["causallm_exe_path"]
        state["binary_path"] = binary_path  # Ensure it's set for downstream
    
    if not binary_path or not os.path.exists(binary_path):
        bus.log("Nothing was compiled -- profiler has no binary to run", "warn")
        state["profile"] = {"ran": False, "reason": state.get("compile_log", "not compiled")}
        bus.agent_status("profiler", "error", "no binary")
        return state

    bus.log(f"Running profiler on: {binary_path}", "info")
    total_ms = None
    peak_memory_kb = None
    raw_output = ""

    # For CausalLM binaries, pass the model directory as an argument
    cmd = [binary_path]
    
    # Find the model directory - check multiple possible locations
    out_dir = state.get("out_dir", "")
    model_dir = None
    
    # First check: run_model directory (standard location)
    standard_model_dir = os.path.join(out_dir, "run_model")
    if os.path.isdir(standard_model_dir):
        model_dir = standard_model_dir
        bus.log(f"Found model directory: {model_dir}")
    else:
        # Second check: look for any .bin file in out_dir and use its parent directory
        bin_path = state.get("causallm_bin_path") or state.get("weights_path")
        if bin_path and os.path.exists(bin_path):
            # Use the directory containing the .bin file as model directory
            model_dir = os.path.dirname(bin_path)
            bus.log(f"Using .bin file directory as model directory: {model_dir}")
        else:
            # Fallback: search for .bin files in out_dir
            import glob
            bin_files = glob.glob(os.path.join(out_dir, "*.bin"))
            if bin_files:
                bin_path = bin_files[0]
                model_dir = os.path.dirname(bin_path)
                bus.log(f"Found .bin file, using directory: {model_dir}")
    
    if model_dir and os.path.isdir(model_dir):
        cmd.append(model_dir)
        bus.log(f"Model directory: {model_dir}")
    else:
        bus.log("No valid model directory found -- profiler may fail", "warn")
        # Try to find config.json to pair with the .bin file
        config_path = state.get("causallm_config_path")
        if config_path and os.path.exists(config_path):
            bus.log(f"Config file found at: {config_path}, but binary expects a directory", "warn")

    # Set up environment with LD_LIBRARY_PATH for nntrainer shared libraries
    env = os.environ.copy()
    nntrainer_root = state.get("nntrainer_root", "/storage_data/snap/Prachi/nntrainer")

    lib_dirs = []

    # Add CausalLM custom layer .so files (e.g., librms_norm_layer.so, libmha_core_layer.so)
    causallm_layers_dir = os.path.join(nntrainer_root, "build_docker", "Applications", "CausalLM", "layers")
    if os.path.isdir(causallm_layers_dir):
        lib_dirs.append(causallm_layers_dir)

    # Add CausalLM main library directory
    causallm_dir = os.path.join(nntrainer_root, "build_docker", "Applications", "CausalLM")
    if os.path.isdir(causallm_dir):
        lib_dirs.append(causallm_dir)

    # Add nntrainer build directory to library path (where .so files are built).
    # Check both a host meson build ("build") and a Docker build
    # ("build_docker") -- the layer plugins above are built alongside
    # whichever libnntrainer.so lives in the SAME tree, so that lib must be
    # found first or the process falls back to a stale system install that
    # can be missing symbols the freshly-built plugins need (e.g.
    # ThreadManager::Global()), causing a "symbol lookup error" at load time.
    for build_name in ("build", "build_docker"):
        build_lib_dir = os.path.join(nntrainer_root, build_name, "nntrainer")
        if os.path.isdir(build_lib_dir):
            lib_dirs.append(build_lib_dir)

    # Also add the lib and lib64 directories if they exist
    for lib_dir in [os.path.join(nntrainer_root, "lib"), os.path.join(nntrainer_root, "lib64")]:
        if os.path.isdir(lib_dir):
            lib_dirs.append(lib_dir)

    if lib_dirs:
        lib_path = ":".join(lib_dirs)
        existing_path = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{lib_path}:{existing_path}" if existing_path else lib_path

    bus.log(f"LD_LIBRARY_PATH: {env.get('LD_LIBRARY_PATH', 'not set')}", "info")

    try:
        start = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False, env=env)
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
            # The binary crashed or exited abnormally -- any latency figure
            # above is wall-clock time for a run that never actually
            # produced a real inference, so it must not be shown as measured
            # on-device data. Surface the real failure (e.g. a dynamic
            # linker "symbol lookup error") and stop here.
            reason = f"binary exited with code {result.returncode}: {raw_output[-500:].strip()}"
            bus.log(f"Binary exited with code {result.returncode} -- not a valid on-device run", "error")
            for line in raw_output.strip().splitlines()[-20:]:
                bus.log(f"  {line}", "error")
            state["profile"] = {"ran": False, "reason": reason}
            bus.profile(state["profile"])
            bus.agent_status("profiler", "error", f"exit {result.returncode}")
            return state

    except Exception as exc:
        bus.log(f"Profiling run failed: {exc}", "warn")
        state["profile"] = {"ran": False, "reason": str(exc)}
        bus.profile(state["profile"])
        bus.agent_status("profiler", "error", str(exc))
        return state


    # Profile model graph (semantic/high-level)
    graph_view = state.get("graph_view", {"nodes": []})
    weighted_nodes = [n for n in graph_view["nodes"] if n["status"] == "mapped"]
    total_weight = sum(max(1, len(n.get("attributes", {}))) for n in weighted_nodes) or 1

    layers = []
    for n in weighted_nodes:
        weight = max(1, len(n.get("attributes", {})))
        share_pct = round(weight / total_weight * 100, 2)
        est_ms = round(total_ms * weight / total_weight, 3) if total_ms else None
        layers.append({
            "name": n["label"],
            "type": n["type"],
            "estimated_ms": est_ms,
            # share_pct drives the Profiler Dashboard's per-layer bottleneck
            # list (rendered below the summary tiles) -- without it the UI
            # silently skips rendering any per-layer breakdown.
            "share_pct": share_pct,
        })

    # Also profile nntrainer graph (lowered/target) if available
    nntrainer_layers = []
    cpp_graph_view = state.get("cpp_graph_view") or state.get("nntrainer_graph_view", {"nodes": []})
    cpp_nodes = [n for n in cpp_graph_view["nodes"] if n.get("status") == "mapped" or "mapped" not in [n.get("status")]]
    cpp_total_weight = sum(max(1, len(n.get("attributes", {}))) for n in cpp_nodes) or 1

    for n in cpp_nodes[:100]:  # Limit to top 100 nodes to avoid huge lists
        weight = max(1, len(n.get("attributes", {})))
        est_ms = round(total_ms * weight / cpp_total_weight, 3) if total_ms else None
        nntrainer_layers.append({"name": n["label"], "type": n["type"], "estimated_ms": est_ms})

    if nntrainer_layers:
        state["cpp_profile"] = {
            "ran": True,
            "total_latency_ms": round(total_ms, 3),
            "layers": nntrainer_layers,
        }
        bus.log(f"Profiled nntrainer graph: {len(nntrainer_layers)} nodes")

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

    # Identify bottleneck layers (top 5 slowest)
    sorted_layers = sorted(layers, key=lambda l: l.get("estimated_ms", 0), reverse=True)
    bottleneck_layers = sorted_layers[:5]

    # Calculate per-layer memory estimates (proportional to latency as proxy)
    total_estimated_ms = sum(l.get("estimated_ms", 0) for l in layers) or 1
    for layer in layers:
        layer_ms = layer.get("estimated_ms", 0)
        # Estimate memory as proportional to compute time
        layer_memory_percent = (layer_ms / total_estimated_ms) * 100 if total_estimated_ms > 0 else 0
        layer["estimated_memory_mb"] = round(layer_memory_percent * (peak_memory_kb or 0) / 100 / 1024, 2)
        layer["memory_percent"] = round(layer_memory_percent, 2)

    profile = {
        "ran": True,
        "total_latency_ms": round(total_ms, 3),
        "peak_memory_kb": peak_memory_kb,
        "peak_memory_mb": round(peak_memory_kb / 1024, 2) if peak_memory_kb else 0,
        "prefill_latency_ms": prefill_ms,
        "generation_latency_ms": generation_ms,
        "prefill_percent": round(prefill_ms / total_ms * 100, 1) if prefill_ms else 0,
        "generation_percent": round(generation_ms / total_ms * 100, 1) if generation_ms else 0,
        "layers": layers,
        "bottleneck_layers": bottleneck_layers,
        "num_bottleneck_layers": len(bottleneck_layers),
        "note": "per-layer figures are proportional estimates from op weight, not a traced measurement",
    }
    state["profile"] = profile

    # Log bottleneck summary
    bus.log(f"Top bottleneck layers (by latency):")
    for i, layer in enumerate(bottleneck_layers, 1):
        bus.log(f"  {i}. {layer['name']}: {layer.get('estimated_ms', 0):.2f}ms ({layer.get('memory_percent', 0):.1f}% memory)")

    bus.profile(profile)
    prefill_str = f"{prefill_ms:.0f}ms" if prefill_ms is not None else "n/a"
    generation_str = f"{generation_ms:.0f}ms" if generation_ms is not None else "n/a"
    bus.log(f"Profiling run complete -- total latency {total_ms:.2f} ms (prefill: {prefill_str}, generation: {generation_str})")
    bus.agent_status("profiler", "done", f"{total_ms:.2f} ms (P:{prefill_str} G:{generation_str})")

    # Save the raw model output (including generated text) to a file
    out_dir = state.get("out_dir", "")
    if raw_output and out_dir:
        output_file = os.path.join(out_dir, "model_output.txt")
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("=== Model Execution Output ===\n\n")
                f.write(raw_output)
            bus.log(f"Model output saved to: {output_file}", "info")
            state["model_output_file"] = output_file
            
            # Display the generated text in the logs
            bus.log("Generated text output:", "info")
            for line in raw_output.splitlines()[:50]:  # Show first 50 lines
                bus.log(f"  {line}", "info")
        except Exception as exc:
            bus.log(f"Failed to save model output: {exc}", "warn")

    return state
