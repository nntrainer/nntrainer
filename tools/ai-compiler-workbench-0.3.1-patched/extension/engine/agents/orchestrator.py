"""
Orchestrator Agent.

Plans, coordinates, schedules and monitors every other agent, exactly
as the "Orchestrator Agent (LangGraph)" box in the architecture
diagram describes. Built on LangGraph's StateGraph: each agent is a
node over the shared PipelineState, edges encode the fixed pipeline
order, and one conditional edge implements the Compiler -> Auto-Fix ->
Compiler loop (bounded by MAX_FIX_ITERATIONS) before falling through
to profiling and artifact collection regardless of whether compilation
ultimately succeeded.

Pipeline order: Model Discovery (config only, fast) -> kick off Weight
Download + Weight Converter in a background thread -> Compatibility
(builds the graph from the config alone via AutoModel.from_config, so
it never waits on a weight download) -> INI Generator -> Graph Builder
-> C++ Generator -> Compiler/Auto-Fix loop -> Profiler -> join the
background weight thread -> Artifact Manager. Weights are only needed
for the final artifact listing, not for anything before it, so the
whole graph/.ini/.cpp/compile path runs fully in parallel with the
(often much slower) weight download.

The background thread's results are handed back through a small
process-global store keyed by output directory, not by mutating the
LangGraph state dict directly from another thread -- LangGraph may or
may not preserve object identity of the state dict across node
boundaries, so relying on that would be fragile. Reading the result
back into state happens synchronously inside the "join" node instead.

Falls back to a plain sequential run of the same node functions if
langgraph isn't installed, so the extension still works with just
`pip install langchain langchain-anthropic` and no langgraph -- but
the intended, documented path is the StateGraph one.
"""
import threading

from . import (
    model_discovery,
    compatibility,
    nntrainer_lowering,
    weight_download,
    weight_converter,
    ini_generator,
    graph_builder,
    cpp_generator_agent,
    causallm_install,
    dual_graph,
    compiler_agent,
    auto_fix,
    profiler_agent,
    artifact_manager,
)
from .events import bus
from .state import PipelineState, new_state

MAX_FIX_ITERATIONS = 2

# Process-global (this extension always spawns one fresh Python process per
# "Run Pipeline" invocation, so there's no cross-run collision risk here).
_bg_lock = threading.Lock()
_bg_threads: dict = {}
_bg_results: dict = {}


def _weight_worker(run_id: str, model_name: str, out_dir: str, api_key):
    local_state = {"model_name": model_name, "out_dir": out_dir, "anthropic_api_key": api_key, "errors": []}
    local_state = weight_download.run(local_state)
    local_state = weight_converter.run(local_state)
    with _bg_lock:
        _bg_results[run_id] = {
            "weights_path": local_state.get("weights_path"),
            "converted_weights_path": local_state.get("converted_weights_path"),
            "errors": local_state.get("errors", []),
        }


def start_weight_download(state: dict) -> dict:
    run_id = state["out_dir"]
    thread = threading.Thread(
        target=_weight_worker,
        args=(run_id, state["model_name"], state["out_dir"], state.get("anthropic_api_key")),
        daemon=True,
    )
    with _bg_lock:
        _bg_threads[run_id] = thread
    thread.start()
    bus.log("Weight download started in the background -- graph/.ini/.cpp construction continues in parallel")
    return state


def join_weight_download(state: dict) -> dict:
    run_id = state["out_dir"]
    with _bg_lock:
        thread = _bg_threads.pop(run_id, None)
    if thread and thread.is_alive():
        bus.log("Waiting for the background weight download to finish before collecting artifacts...")
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
    # only retry when there's a real compiler error to react to (not
    # "nntrainer not found", which no amount of code patching fixes).
    # Prefer the explicit flag the compiler agent sets; fall back to the
    # log-substring check so this still works with an older compiler agent.
    log = state.get("compile_log", "")
    if state.get("nntrainer_missing") or "not found" in log or "NNTRAINER_INCLUDE_DIR" in log:
        return "profiler"
    if state.get("fix_iterations", 0) >= MAX_FIX_ITERATIONS:
        return "profiler"
    if not state.get("anthropic_api_key"):
        return "profiler"
    return "auto_fix"


def build_graph():
    from langgraph.graph import StateGraph, END

    g = StateGraph(PipelineState)
    g.add_node("model_discovery", model_discovery.run)
    g.add_node("start_weight_download", start_weight_download)
    g.add_node("compatibility", compatibility.run)
    g.add_node("nntrainer_lowering", nntrainer_lowering.run)
    g.add_node("ini_generator", ini_generator.run)
    g.add_node("graph_builder", graph_builder.run)
    g.add_node("cpp_generator", cpp_generator_agent.run)
    g.add_node("causallm_install", causallm_install.run)
    g.add_node("dual_graph", dual_graph.run)
    g.add_node("compiler", compiler_agent.run)
    g.add_node("auto_fix", auto_fix.run)
    g.add_node("profiler", profiler_agent.run)
    g.add_node("join_weight_download", join_weight_download)
    g.add_node("artifact_manager", artifact_manager.run)

    g.set_entry_point("model_discovery")
    g.add_edge("model_discovery", "start_weight_download")
    g.add_edge("start_weight_download", "compatibility")
    g.add_edge("compatibility", "nntrainer_lowering")
    g.add_edge("nntrainer_lowering", "ini_generator")
    g.add_edge("ini_generator", "graph_builder")
    g.add_edge("graph_builder", "cpp_generator")
    g.add_edge("cpp_generator", "causallm_install")
    g.add_edge("causallm_install", "dual_graph")
    g.add_edge("dual_graph", "compiler")
    g.add_conditional_edges("compiler", _should_retry_compile, {
        "auto_fix": "auto_fix", "profiler": "profiler",
    })
    g.add_edge("auto_fix", "dual_graph")
    g.add_edge("profiler", "join_weight_download")
    g.add_edge("join_weight_download", "artifact_manager")
    g.add_edge("artifact_manager", END)

    return g.compile()


def _run_sequential(state: dict) -> dict:
    """Fallback path used when langgraph isn't installed."""
    state = model_discovery.run(state)
    if state.get("errors"):
        return state
    state = start_weight_download(state)
    state = compatibility.run(state)
    state = nntrainer_lowering.run(state)
    state = ini_generator.run(state)
    state = graph_builder.run(state)
    state = cpp_generator_agent.run(state)
    state = causallm_install.run(state)
    state = dual_graph.run(state)
    state = compiler_agent.run(state)
    for _ in range(MAX_FIX_ITERATIONS):
        if _should_retry_compile(state) != "auto_fix":
            break
        state = auto_fix.run(state)
        state = dual_graph.run(state)
        state = compiler_agent.run(state)
    state = profiler_agent.run(state)
    state = join_weight_download(state)
    state = artifact_manager.run(state)
    return state


def run_pipeline(
    model_name: str,
    out_dir: str,
    api_key: str = None,
    causallm_project_root: str = None,
    install_generated_files: bool = False,
    generated_header_directory: str = "include/generated",
    generated_source_directory: str = "src/generated",
) -> dict:
    state = dict(new_state(
        model_name, out_dir, api_key,
        causallm_project_root=causallm_project_root,
        install_generated_files=install_generated_files,
        generated_header_directory=generated_header_directory,
        generated_source_directory=generated_source_directory,
    ))
    # By default, skip C++ compilation; use nntrainer's native CausalLM instead.
    # Set to False to compile the generated code (for validation).
    state["skip_cpp_compilation"] = True

    bus.chat("assistant", f"Starting pipeline for **{model_name}**...", "orchestrator")

    try:
        graph = build_graph()
        final_state = graph.invoke(state, config={"recursion_limit": 60})
    except ImportError:
        bus.log("langgraph not installed -- running agents sequentially instead", "warn")
        final_state = _run_sequential(state)
    except Exception as exc:
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

    import json, os
    with open(os.path.join(out_dir, "state.json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe(final_state), f, indent=2)

    return final_state


def _json_safe(state: dict) -> dict:
    import json as _json

    def default(o):
        return str(o)

    safe = _json.loads(_json.dumps(state, default=default))
    # Never write the API key to disk. Post-run agents (chat, graph focus)
    # receive it from the ANTHROPIC_API_KEY environment variable instead.
    safe.pop("anthropic_api_key", None)
    return safe
