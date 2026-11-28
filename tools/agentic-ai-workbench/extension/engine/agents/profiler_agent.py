# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file profiler_agent.py
# @brief Profiler Agent (no LLM).
# @author prachi-t17 <prachi.t@samsung.com>

"""
Profiler Agent (no LLM).

A generated CausalLM model (the Claude CLI codegen, review.json): once it
has passed Verify, it is built, loaded and run with its real weights in
Docker and profiled (knowledge/run_model.py, the same as Build, Run & Profile
in the Review tab) -- tokens/s, peak memory, load time, then the time of
every layer from one run against the profiling build (unless
aiCompilerWorkbench.autoProfilePerLayer is off) -- unless
aiCompilerWorkbench.autoProfile is off. Otherwise one line says why not.

MODEL_API code from the template generator: if the Compiler Agent produced
a real standalone binary, runs it with `time` and reports wall-clock latency
plus a per-layer breakdown derived from the IR's declared attributes --
a static estimate proportional to parameter count per layer, clearly
labeled as such. If nothing was compiled, this agent says so plainly
instead of inventing numbers.
"""
import os
import re
import subprocess
import time

from knowledge import codegen as cg
from knowledge import docker_host
from knowledge import run_model as rm
from knowledge import verify

from . import codegen_review
from .causallm_codegen import publish
from .events import bus


# The same prompt the Review tab and the Build, Run & Profile command start
# with: an empty prompt would fall back to the model's sample_input or
# "Hello" -- one token, which makes prefill speed meaningless.
DEFAULT_PROMPT = "What is the capital of France?"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except ValueError:
        return default


def _env_on(name: str) -> bool:
    return os.environ.get(name, "1").lower() not in ("0", "false", "off", "no")


def profile_generated(state: dict) -> dict:
    """Build, Run & Profile the verified generated model with its real weights."""
    def skip(why: str) -> dict:
        bus.log(f"Profiling skipped: {why}")
        state["profile"] = {"ran": False, "reason": why}
        bus.agent_status("profiler", "skipped", why[:80])
        return state

    if not _env_on("WORKBENCH_AUTO_PROFILE"):
        return skip(
            "turned off (aiCompilerWorkbench.autoProfile); Build, Run & Profile is in the"
            " Review tab"
        )
    review = cg.load_review(state["codegen_review_path"])
    verify_res = review.get("verify") or {}
    if verify_res.get("status") != "passed" or verify_res.get(
        "files_hash"
    ) != verify.current_verify_hash(review):
        return skip("Verify has not passed for this code yet")
    if not docker_host.daemon_status()[0]:
        return skip("Docker is not running (runs with real weights happen in Docker)")
    model_dir = rm.find_model_dir(review, state.get("weights_path"))
    if model_dir is None:
        return skip(f"the weights of {review.get('model')} are not on this computer "
                    "(Build, Run & Profile in the Review tab offers to download them)")
    tokens, repeat = _env_int("WORKBENCH_RUN_TOKENS", 64), _env_int("WORKBENCH_RUN_REPEATS", 3)
    cpus, memory = (
        os.environ.get("WORKBENCH_RUN_CPUS") or None,
        os.environ.get("WORKBENCH_RUN_MEMORY") or None,
    )
    per_layer = _env_on(
        "WORKBENCH_AUTO_PROFILE_PER_LAYER"
    )  # aiCompilerWorkbench.autoProfilePerLayer
    try:
        codegen_review.docker_ready("docker", "run")
        bus.log(
            f"Profiling {review.get('model')} with its real weights in Docker ({tokens} tokens, "
            f"{repeat} run{'s' if repeat != 1 else ''}"
            f"{', then per-layer timing' if per_layer else ''}) ..."
        )
        review = rm.run_generated(
            state["codegen_review_path"],
            prompt=DEFAULT_PROMPT,
            tokens=tokens,
            repeat=repeat,
            cpus=cpus,
            memory=memory,
            log=bus.log,
            model_dir=model_dir,
            per_layer=per_layer,
        )
    except (rm.RunError, cg.CodegenError, OSError, ValueError) as exc:
        msg = str(exc)
        bus.log(f"Profiling did not run: {msg}", "warn")
        state["profile"] = {"ran": False, "reason": msg}
        bus.agent_status("profiler", "error", msg[:80])
        return state
    rec = review.get("run") or {}
    codegen_review.report_run(rec)
    bus.profile(rm.profile_event(rec, review.get("model", "")))
    publish(review)                                       # the Review tab's Run section
    summary = rec.get("summary") or {}
    layer_profile = rec.get("layer_profile") or {}
    state["profile"] = {
        "ran": bool(rec.get("ok")),
        "summary": summary,
        "peak_memory_kb": rec.get("peak_memory_kb"),
        "reason": rec.get("error", ""),
        "per_layer": bool(layer_profile.get("layers")),
        "per_layer_error": layer_profile.get("error", ""),
    }
    if rec.get("ok"):
        bus.agent_status("profiler", "done", f"{summary.get('generation_tps')} tok/s generation, "
                                             f"{summary.get('prefill_tps')} tok/s prefill")
    else:
        bus.agent_status("profiler", "error", str(rec.get("error") or "run failed")[:80])
    return state


# pylint: disable-next=too-many-locals,too-many-statements
def run(state: dict) -> dict:
    """Pipeline step: profile the generated model (or the compiled binary)."""
    bus.agent_status("profiler", "running")

    if state.get("codegen_review_path"):
        return profile_generated(state)

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
        e2e_match = re.search(r'\[e2e time\]:\s*(\d+(?:\.\d+)?)\s*ms', raw_output)
        if e2e_match:
            total_ms = float(e2e_match.group(1))
            bus.log(f"Binary reported e2e latency: {total_ms:.2f} ms")
        else:
            total_ms = elapsed_ms
            bus.log(f"Using wall-clock latency: {total_ms:.2f} ms (binary did not report e2e time)")

        # Parse peak memory: "Peak memory usage (VmRSS): X KB"
        mem_match = re.search(
            r'Peak memory usage.*?:\s*(\d+(?:\.\d+)?)\s*KB', raw_output, re.IGNORECASE
        )
        if mem_match:
            peak_memory_kb = int(mem_match.group(1))
            bus.log(f"Peak memory: {peak_memory_kb} KB")

        if result.returncode != 0:
            bus.log(f"Binary exited with code {result.returncode}", "warn")

    # a failed profiling run must not fail the pipeline
    except Exception as exc:  # pylint: disable=broad-except
        bus.log(f"Profiling run failed: {exc}", "warn")
        state["profile"] = {"ran": False, "reason": str(exc)}
        bus.agent_status("profiler", "error", str(exc))
        return state

    graph_view = state.get("graph_view", {"nodes": []})

    weighted_nodes = [n for n in graph_view["nodes"] if n["status"] == "mapped"]
    total_weight = sum(max(1, len(n.get("attributes", {}))) for n in weighted_nodes) or 1

    layers = []
    for weighted_node in weighted_nodes:
        weight = max(1, len(weighted_node.get("attributes", {})))
        est_ms = round(total_ms * weight / total_weight, 3) if total_ms else None
        layers.append(
            {"name": weighted_node["label"], "type": weighted_node["type"], "estimated_ms": est_ms}
        )

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
        "note": (
            "per-layer figures are proportional estimates from op weight, not a traced measurement"
        ),
    }
    if peak_memory_kb:
        profile["peak_memory_mb"] = round(peak_memory_kb / 1024, 2)
    state["profile"] = profile

    bus.profile(profile)
    bus.log(f"Profiling run complete -- total latency {total_ms:.2f} ms")
    bus.agent_status("profiler", "done", f"{total_ms:.2f} ms")
    return state
