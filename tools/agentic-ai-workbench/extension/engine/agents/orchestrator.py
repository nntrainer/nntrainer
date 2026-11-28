# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file orchestrator.py
# @brief Orchestrator Agent - Coordinates all agents in the pipeline.
# @author prachi-t17 <prachi.t@samsung.com>

"""
Orchestrator Agent - Coordinates all agents in the pipeline.
"""
import json
import os
import sys
import threading
import traceback

from . import (
    model_discovery,
    compatibility,
    nntrainer_lowering,
    weight_download,
    weight_converter,
    ini_generator,
    graph_builder,
    cpp_generator_agent,
    causallm_codegen,
    causallm_install,
    dual_graph,
    compiler_agent,
    auto_fix,
    profiler_agent,
    artifact_manager,
    weight_preview,
)

from .events import bus
from .state import PipelineState, new_state

MAX_FIX_ITERATIONS = 2

_bg_lock = threading.Lock()
_bg_threads: dict = {}
_bg_results: dict = {}


def _weight_worker(run_id: str, model_name: str, out_dir: str, api_key, custom_weights_path=None):
    local_state = {
        "model_name": model_name,
        "out_dir": out_dir,
        "anthropic_api_key": api_key,
        "custom_weights_path": custom_weights_path,
        "errors": [],
    }
    local_state = weight_download.run(local_state)
    local_state = weight_converter.run(local_state)
    with _bg_lock:
        _bg_results[run_id] = {
            "weights_path": local_state.get("weights_path"),
            "converted_weights_path": local_state.get("converted_weights_path"),
            "errors": local_state.get("errors", []),
        }


def start_weight_download(state: dict) -> dict:
    """Start downloading the weights in the background while the next steps run."""
    run_id = state["out_dir"]
    thread = threading.Thread(
        target=_weight_worker,
        args=(
            run_id,
            state["model_name"],
            state["out_dir"],
            state.get("anthropic_api_key"),
            state.get("custom_weights_path"),
        ),
        daemon=True,
    )
    with _bg_lock:
        _bg_threads[run_id] = thread
    thread.start()
    bus.log(
        "Weight download started in the background -- graph/.ini/.cpp construction continues in"
        " parallel"
    )
    return state


def join_weight_download(state: dict) -> dict:
    """Wait for the background weight download and merge its result into the state."""
    run_id = state["out_dir"]
    with _bg_lock:
        thread = _bg_threads.pop(run_id, None)
    if thread and thread.is_alive():
        bus.log(
            "Waiting for the background weight download to finish before collecting artifacts..."
        )
        thread.join()
    with _bg_lock:
        result = _bg_results.pop(run_id, None)
    if result:
        state["weights_path"] = result.get("weights_path")
        state["converted_weights_path"] = result.get("converted_weights_path")
        if result.get("errors"):
            state.setdefault("errors", []).extend(result["errors"])
    return state


def _should_retry_compile(state: dict) -> str:
    if state.get("compiled"):
        return "profiler"
    if state.get("codegen_review_path"):
        return "profiler"   # generated code is fixed through the Review tab (Fix with Claude)
    log = state.get("compile_log", "")
    if state.get("nntrainer_missing") or "not found" in log or "NNTRAINER_INCLUDE_DIR" in log:
        return "profiler"
    if state.get("fix_iterations", 0) >= MAX_FIX_ITERATIONS:
        return "profiler"
    if not state.get("anthropic_api_key"):
        return "profiler"
    return "auto_fix"


def build_graph():
    """The pipeline as a LangGraph StateGraph (used when langgraph is installed)."""
    # pylint: disable-next=import-outside-toplevel,import-error  # optional package
    from langgraph.graph import StateGraph, END

    graph = StateGraph(PipelineState)
    graph.add_node("model_discovery", _guard("model_discovery", model_discovery.run))
    graph.add_node("start_weight_download", _guard("start_weight_download", start_weight_download))
    graph.add_node("compatibility", _guard("compatibility", compatibility.run))
    graph.add_node("nntrainer_lowering", _guard("nntrainer_lowering", nntrainer_lowering.run))
    graph.add_node("ini_generator", _guard("ini_generator", ini_generator.run))
    graph.add_node("graph_builder", _guard("graph_builder", graph_builder.run))
    graph.add_node("cpp_generator", _guard("cpp_generator", cpp_generator_agent.run))
    graph.add_node("causallm_codegen", _guard("causallm_codegen", causallm_codegen.run))
    graph.add_node("causallm_install", _guard("causallm_install", causallm_install.run))
    graph.add_node("dual_graph", _guard("dual_graph", dual_graph.run))

    graph.add_node("compiler", _guard("compiler", compiler_agent.run))
    graph.add_node("auto_fix", _guard("auto_fix", auto_fix.run))
    graph.add_node("profiler", _guard("profiler", profiler_agent.run))
    graph.add_node("join_weight_download", _guard("join_weight_download", join_weight_download))
    graph.add_node("weight_preview", _guard("weight_preview", weight_preview.run))
    graph.add_node("artifact_manager", _guard("artifact_manager", artifact_manager.run))

    graph.set_entry_point("model_discovery")
    graph.add_edge("model_discovery", "start_weight_download")
    graph.add_edge("start_weight_download", "compatibility")
    graph.add_edge("compatibility", "nntrainer_lowering")
    graph.add_edge("nntrainer_lowering", "ini_generator")
    graph.add_edge("ini_generator", "graph_builder")
    graph.add_edge("graph_builder", "cpp_generator")
    graph.add_edge("cpp_generator", "causallm_codegen")
    graph.add_edge("causallm_codegen", "causallm_install")
    graph.add_edge("causallm_install", "dual_graph")

    graph.add_edge("dual_graph", "compiler")
    # The weights must be on disk before the profiler runs the model with them.
    graph.add_conditional_edges("compiler", _should_retry_compile, {
        "auto_fix": "auto_fix", "profiler": "join_weight_download",
    })
    graph.add_edge("auto_fix", "dual_graph")
    graph.add_edge("join_weight_download", "profiler")
    graph.add_edge("profiler", "weight_preview")
    graph.add_edge("weight_preview", "artifact_manager")
    graph.add_edge("artifact_manager", END)

    return graph.compile()


def _guard(name: str, func):
    """Wrap a step so that a crash is reported for that step (agent row turns
    red, error in the log, traceback in the output panel) and the run goes on
    with the remaining steps instead of ending."""
    def step(state: dict) -> dict:
        try:
            out = func(state)
            return out if isinstance(out, dict) else state
        # any crash is reported as a pipeline error event
        except Exception as exc:  # pylint: disable=broad-except
            label = (
                "weight_download"
                if name in ("start_weight_download", "join_weight_download")
                else name
            )
            bus.error(label, f"{type(exc).__name__}: {exc}")
            bus.agent_status(label, "error", f"{type(exc).__name__}: {str(exc)[:90]}")
            state.setdefault("errors", []).append(f"{label}: {exc}")
            sys.stderr.write(traceback.format_exc())
            return state
    return step


def _run_sequential(state: dict) -> dict:
    state = _guard("model_discovery", model_discovery.run)(state)
    if state.get("errors"):
        return state
    steps = [
        ("start_weight_download", start_weight_download), ("compatibility", compatibility.run),
        ("nntrainer_lowering", nntrainer_lowering.run), ("ini_generator", ini_generator.run),
        ("graph_builder", graph_builder.run), ("cpp_generator", cpp_generator_agent.run),
        ("causallm_codegen", causallm_codegen.run), ("causallm_install", causallm_install.run),
    ]
    steps += [("dual_graph", dual_graph.run), ("compiler", compiler_agent.run)]
    for name, func in steps:
        state = _guard(name, func)(state)
    for _ in range(MAX_FIX_ITERATIONS):
        if _should_retry_compile(state) != "auto_fix":
            break
        state = _guard("auto_fix", auto_fix.run)(state)
        state = _guard("dual_graph", dual_graph.run)(state)
        state = _guard("compiler", compiler_agent.run)(state)
    for name, func in [
        ("join_weight_download", join_weight_download),
        ("profiler", profiler_agent.run),
        ("weight_preview", weight_preview.run),
        ("artifact_manager", artifact_manager.run),
    ]:
        state = _guard(name, func)(state)
    return state


def run_check_only(model_name: str, out_dir: str, api_key: str = None) -> dict:
    """Quick check: config + tensor names/shapes only (no weight download, no
    model built in memory), then the checklist, the plan and the Claude CLI's
    code, and a compile check when nothing waits for review. Works for models
    of any size; the graphs need a full Run."""
    state = dict(new_state(model_name, out_dir, api_key))
    state["check_only"] = True
    bus.chat(
        "assistant",
        f"Checking **{model_name}** -- config and tensor headers only: nothing is downloaded "
        "and the model is not loaded into memory.",
        "orchestrator",
    )
    state = _guard("model_discovery", model_discovery.run)(state)
    if not state.get("errors"):
        state = _guard("causallm_codegen", causallm_codegen.run)(state)
        if state.get("codegen_review_path"):
            state = _guard("compiler", compiler_agent.run)(state)
    note = "Check only: the graphs are built by a full Run."
    bus.graph({"nodes": [], "edges": [], "message": note}, "model")
    bus.graph({"nodes": [], "edges": [], "message": note}, "nntrainer")
    review_ok = state.get("codegen_ok")
    summary = {
        "model_name": model_name,
        "architecture": state.get("architecture"),
        "check_only": True,
        "compatibility": {},
        "compiled": False,
        "profiled": False,
        "artifact_count": 0,
        "codegen_ok": review_ok,
        "errors": state.get("errors", []),
    }
    bus.chat("assistant", f"Check finished for **{model_name}**" +
             ("" if review_ok is None else (" -- generated code passed its checks." if review_ok
                                            else " -- generated code failed its checks.")) +
             " See the Review tab.", "orchestrator")
    bus.pipeline_complete(summary)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "state.json"), "w", encoding="utf-8") as handle:
        json.dump(_json_safe(state), handle, indent=2)
    return state


# pylint: disable-next=too-many-arguments
def run_pipeline(
    model_name: str,
    out_dir: str,
    api_key: str = None,
    custom_weights_path: str = None,
    causallm_project_root: str = None,
    install_generated_files: bool = False,
    generated_header_directory: str = "include/generated",
    generated_source_directory: str = "src/generated",
) -> dict:
    """Run the whole pipeline for a model and save the final state.json."""
    state = dict(new_state(
        model_name, out_dir, api_key,
        custom_weights_path=custom_weights_path,
        causallm_project_root=causallm_project_root,
        install_generated_files=install_generated_files,
        generated_header_directory=generated_header_directory,
        generated_source_directory=generated_source_directory,
    ))

    bus.chat("assistant", f"Starting pipeline for **{model_name}**...", "orchestrator")

    try:
        graph = build_graph()
        final_state = graph.invoke(state, config={"recursion_limit": 60})
    except ImportError:
        bus.log("langgraph not installed -- running agents sequentially instead", "warn")
        final_state = _run_sequential(state)
    # any crash is reported as a pipeline error event
    except Exception as exc:  # pylint: disable=broad-except
        bus.error("orchestrator", str(exc))
        final_state = state
        final_state.setdefault("errors", []).append(str(exc))

    summary = {
        "model_name": model_name,
        "architecture": final_state.get("architecture"),
        "compatibility": (final_state.get("report") or {}).get("summary", {}),
        "compiled": final_state.get("compiled", False),
        "profiled": (final_state.get("profile") or {}).get("ran", False),
        "artifact_count": len(final_state.get("artifacts", [])),
        "errors": final_state.get("errors", []),
    }
    bus.chat(
        "assistant",
        f"Pipeline finished for **{model_name}**. "
        f"{summary['compatibility'].get('supported_nodes', 0)} ops mapped, "
        f"{summary['compatibility'].get('unsupported_nodes', 0)} unresolved, "
        f"compiled={summary['compiled']}.",
        "orchestrator",
    )
    bus.pipeline_complete(summary)

    with open(os.path.join(out_dir, "state.json"), "w", encoding="utf-8") as handle:
        json.dump(_json_safe(final_state), handle, indent=2)

    return final_state


def _json_safe(state: dict) -> dict:
    safe = json.loads(json.dumps(state, default=str))
    safe.pop("anthropic_api_key", None)
    return safe
