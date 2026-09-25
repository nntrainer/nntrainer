"""
Orchestrator Agent - Coordinates all agents in the pipeline.

All LLM calls use Cline with UMS token (Gauss models only).
"""
import threading
import os
import json
from typing import Optional

from . import (
    model_discovery,
    compatibility,
    nntrainer_lowering,
    ini_generator,
    graph_builder,
    dual_graph,
)
from ..cpp import cpp_generator_agent
from ..causallm import causallm_install, causallm_weight_converter, causallm_so_builder
from ..quantization import nntr_quantize_agent
from ..build import compiler_agent
from ..fix import auto_fix
from ..profile import profiler_agent
from ..utils import artifact_manager, weight_download, weight_converter, state
from .. import graph_views, integrated_builder
from ..events import bus

# Memory store for RAG-style run history
# Use absolute import for engine.knowledge module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from knowledge import get_memory_store, record_event

PipelineState = state.PipelineState
new_state = state.new_state

# =============================================================================
# GLOBAL CLINE CONFIGURATION - Single token for all LLM operations
# =============================================================================
# All LLM calls use Cline (UMS token) with Ollama models
# Set CLINE_UMS_TOKEN environment variable or pass via VS Code settings
# VS Code settings take precedence over environment variables
CLINE_UMS_TOKEN = os.environ.get("CLINE_UMS_TOKEN", "")
CLINE_API_BASE = os.environ.get("CLINE_API_BASE", "http://localhost:6543/v1")
CLINE_MODEL = os.environ.get("CLINE_MODEL", "claude-opus-4-6")
CLINE_MAX_TOKENS = int(os.environ.get("CLINE_MAX_TOKENS", "8192"))
# Capped at 2: auto_fix is a bounded, conservative mechanical-fix loop
# (see agents/fix/auto_fix.py), not an open-ended LLM retry loop.
MAX_FIX_ITERATIONS = int(os.environ.get("AUTO_FIX_MAX_ITERATIONS", "2"))
# =============================================================================

_bg_lock = threading.Lock()
_bg_threads: dict = {}
_bg_results: dict = {}


def _weight_worker(run_id: str, model_name: str, out_dir: str, api_key, custom_weights_path=None):
    """
    Background worker: ONLY downloads and verifies weights.
    
    Does NOT run weight_converter - that runs in the main pipeline AFTER
    join_weight_download ensures weights are ready.
    """
    local_state = {
        "model_name": model_name,
        "out_dir": out_dir,
        "cline_ums_token": api_key,
        "custom_weights_path": custom_weights_path,
        "errors": [],
    }
    # ONLY download - converter runs in main pipeline after join
    local_state = weight_download.run(local_state)
    
    # CRITICAL: Preserve weights_verified from download result
    # Do NOT run weight_converter here - it would fail and clear the flag
    with _bg_lock:
        _bg_results[run_id] = {
            "weights_path": local_state.get("weights_path"),
            "converted_weights_path": local_state.get("converted_weights_path"),
            "weights_verified": local_state.get("weights_verified", False),  # From download, not converter
            "weight_manifest": local_state.get("weight_manifest"),
            "errors": local_state.get("errors", []),
        }


def start_weight_download(state: dict) -> dict:
    run_id = state["out_dir"]
    thread = threading.Thread(
        target=_weight_worker,
        args=(run_id, state["model_name"], state["out_dir"], state.get("cline_ums_token"), state.get("custom_weights_path")),
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
        bus.log("Waiting for verified weight download before starting weight-dependent stages...")
        thread.join()
    with _bg_lock:
        result = _bg_results.pop(run_id, None)
    if result:
        state["weights_path"] = result.get("weights_path")
        state["converted_weights_path"] = result.get("converted_weights_path")
        state["weights_verified"] = result.get("weights_verified", False)
        state["weight_manifest"] = result.get("weight_manifest")
        if result.get("errors"):
            state.setdefault("errors", []).extend(result["errors"])
    return state


def _fix_budget_exhausted(state: dict) -> bool:
    max_iterations = state.get("max_fix_iterations", MAX_FIX_ITERATIONS)
    return state.get("fix_iterations", 0) >= max_iterations


def _should_retry_fast_compile(state: dict) -> str:
    """
    Gate right after the `compiler` node (fast g++ check: a real standalone
    compile for MODEL_API mode, a syntax-only check for CausalLM component
    mode). Loops into `auto_fix` (mechanical/rule-based first, see
    agents/fix/auto_fix.py) only while there's a plausibly-fixable failure
    and fix budget remains; otherwise the pipeline continues regardless of
    outcome -- the real, authoritative build gate is `causallm_so_builder`
    (see _should_retry_causallm_build below).
    """
    # Already compiled, syntax checked, or compilation was skipped (pre-existing code)
    if state.get("compiled") or state.get("syntax_ok") or state.get("compile_skipped"):
        return "next"

    log = state.get("compile_log", "") or ""

    # System/environment errors are not fixable by rewriting the generated
    # file -- looping here would just waste fix budget.
    if (
        state.get("nntrainer_missing")
        or "not found" in log
        or "NNTRAINER_INCLUDE_DIR" in log
        or "No such file or directory" in log
    ):
        return "next"

    if not log.strip():
        # Nothing to fix (e.g. compile_skipped with no compile_log at all).
        return "next"

    if _fix_budget_exhausted(state):
        return "next"

    return "auto_fix"


def _should_retry_causallm_build(state: dict) -> str:
    """
    Gate right after `causallm_so_builder` (the real meson+ninja build of
    the generated component as part of the full CausalLM project). Only
    loops into auto_fix when the failure was attributed to the generated
    file itself (see causallm_so_builder.py's error filtering) -- rewriting
    unrelated CausalLM project files is out of scope.
    """
    if state.get("causallm_so_built"):
        return "next"

    if not state.get("causallm_build_fixable"):
        return "next"

    if _fix_budget_exhausted(state):
        return "next"

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

    # Integrated builder: builds nntrainer + CausalLM + profiles in one step
    g.add_node("integrated_builder", integrated_builder.run)

    g.add_node("causallm_weight_converter", causallm_weight_converter.run)
    g.add_node("nntr_quantize", nntr_quantize_agent.run)
    g.add_node("causallm_so_builder", causallm_so_builder.run)
    g.add_node("compiler", compiler_agent.run)
    g.add_node("auto_fix", auto_fix.run)
    # Same underlying auto_fix.run, registered under a second name so the
    # so_builder retry loop has its own return edge distinct from the fast
    # compiler-check retry loop below.
    g.add_node("auto_fix_causallm", auto_fix.run)
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

    # causallm_install runs right after cpp_generator (still produces the
    # generated file, kept for later use) and BEFORE compiler. When
    # use_pre_existing_code=True (the default), causallm_install overwrites
    # cpp_path/causallm_source_path with the real, hand-written CausalLM
    # source -- so the fast compile/syntax-check gate below checks that real
    # file instead of the just-generated one. This decouples "compile" from
    # "generated code" without removing code generation itself.
    g.add_edge("cpp_generator", "causallm_install")

    # Fast compile/syntax-check gate: loops into auto_fix (mechanical fixes
    # first, see agents/fix/auto_fix.py) while there's a fixable failure and
    # fix budget remains, then always continues into the real pipeline --
    # the authoritative build gate is causallm_so_builder below.
    g.add_edge("causallm_install", "compiler")
    g.add_conditional_edges(
        "compiler",
        _should_retry_fast_compile,
        {"auto_fix": "auto_fix", "next": "dual_graph"},
    )
    g.add_edge("auto_fix", "compiler")
    
    # CausalLM .bin generation: runs after dual_graph
    # causallm_weight_converter converts HF weights to .bin
    # nntr_quantize optionally quantizes the .bin file
    # Graph construction and C++ generation remain parallel with the download.
    # From this point on every stage consumes weights directly or indirectly,
    # so join only after the two visualizer graphs have been produced.
    g.add_edge("dual_graph", "join_weight_download")
    g.add_edge("join_weight_download", "causallm_weight_converter")
    g.add_edge("causallm_weight_converter", "nntr_quantize")
    
    # CausalLM .so build: runs after quantization, before integrated_builder
    # This builds libcausallm.so from CausalLM source code
    g.add_edge("nntr_quantize", "causallm_so_builder")

    # Real build gate: retries auto_fix only when the failure is attributed
    # to the generated file itself (causallm_so_builder.py sets
    # causallm_build_fixable), bounded by the shared fix_iterations budget.
    g.add_conditional_edges(
        "causallm_so_builder",
        _should_retry_causallm_build,
        {"auto_fix": "auto_fix_causallm", "next": "integrated_builder"},
    )
    g.add_edge("auto_fix_causallm", "causallm_so_builder")

    # New flow: integrated_builder replaces compiler for full build + profile
    g.add_edge("integrated_builder", "profiler")
    g.add_edge("profiler", "artifact_manager")
    g.add_edge("artifact_manager", END)

    return g.compile()


def _run_sequential(state: dict) -> dict:
    state = model_discovery.run(state)
    if state.get("errors"):
        return state
    state = start_weight_download(state)
    state = compatibility.run(state)
    state = nntrainer_lowering.run(state)
    state = ini_generator.run(state)
    state = graph_builder.run(state)
    state = cpp_generator_agent.run(state)

    # causallm_install runs before compiler so that, when
    # use_pre_existing_code=True (default), the compile/syntax-check step
    # targets the real pre-existing CausalLM source instead of the
    # just-generated file (see build_graph() for the matching edge order).
    state = causallm_install.run(state)

    state = compiler_agent.run(state)
    while _should_retry_fast_compile(state) == "auto_fix":
        state = auto_fix.run(state)
        state = compiler_agent.run(state)

    state = dual_graph.run(state)

    # The download/validation may still be in progress while graphs and C++
    # were built. Do not let any weight-consuming stage proceed before it.
    state = join_weight_download(state)

    # CausalLM .bin generation (sequential fallback)
    state = causallm_weight_converter.run(state)
    state = nntr_quantize_agent.run(state)

    # CausalLM .so build from source (sequential fallback)
    state = causallm_so_builder.run(state)
    while _should_retry_causallm_build(state) == "auto_fix":
        state = auto_fix.run(state)
        state = causallm_so_builder.run(state)

    state = integrated_builder.run(state)
    state = profiler_agent.run(state)
    state = artifact_manager.run(state)
    return state


def run_pipeline(
    model_name: str,
    out_dir: str,
    cline_ums_token: str = None,
    custom_weights_path: str = None,
    causallm_project_root: str = None,
    install_generated_files: bool = False,
    generated_header_directory: str = "include/generated",
    generated_source_directory: str = "src/generated",
    # Cline configuration (Gauss models only)
    cline_api_base: str = None,
    cline_model: str = None,
    cline_max_tokens: int = None,
    max_fix_iterations: int = None,
    # NOOA configuration (for LLM-driven C++ generation)
    use_nooa: bool = None,
    # CausalLM .bin generation configuration
    causallm_path: str = None,
    enable_quantization: bool = None,
    quantization_preset: str = None,
    target_isa: str = None,
    nntr_quantize_path: str = None,
    # Fully-automatic compile/build configuration
    auto_compile: bool = False,
    use_docker_build: bool = True,
    auto_fix_use_llm: bool = False,
    nntrainer_path: str = None,
        # Pre-existing code configuration - Generate new code by default
        use_pre_existing_code: bool = False,  # Default: generate new code from graph_ir
    nntrainer_root: str = None,
) -> dict:
    graph_views.clear_weight_cache()

    if auto_compile:
        # This is the one flag that turns "generate code" into "generate,
        # install into the real CausalLM project, build, and profile" with
        # no manual step in between. It only forces install_generated_files
        # for this run -- it does not touch the persisted VS Code setting,
        # and still requires causallm_project_root to be configured.
        install_generated_files = True
        bus.log(
            "auto_compile enabled -- generated files will be installed into "
            "causallm_project_root and built automatically.",
            "info",
        )

    # Get CausalLM settings from environment (set by extension.js)
    env_causallm_path = os.environ.get("CAUSALLM_PATH", "")
    env_enable_quant = os.environ.get("ENABLE_QUANTIZATION", "1") == "1"
    env_quant_preset = os.environ.get("DEFAULT_QUANTIZATION_PRESET", "Q4_0-FP32")
    env_target_isa = os.environ.get("TARGET_ISA", "x86")
    env_nntr_quantize = os.environ.get("NNTR_QUANTIZE_PATH", "")

    # Use provided values, fall back to global defaults
    # All LLM calls use Cline with UMS token (Gauss models only)
    state = dict(new_state(
        model_name, out_dir, cline_ums_token,
        custom_weights_path=custom_weights_path,
        causallm_project_root=causallm_project_root,
        install_generated_files=install_generated_files,
        generated_header_directory=generated_header_directory,
        generated_source_directory=generated_source_directory,
        # Cline configuration (Gauss models only)
        cline_api_base=cline_api_base if cline_api_base is not None else CLINE_API_BASE,
        cline_model=cline_model if cline_model is not None else CLINE_MODEL,
        cline_max_tokens=cline_max_tokens if cline_max_tokens is not None else CLINE_MAX_TOKENS,
        max_fix_iterations=max_fix_iterations if max_fix_iterations is not None else MAX_FIX_ITERATIONS,
        # NOOA configuration - ALWAYS enabled by default
        use_nooa=use_nooa if use_nooa is not None else True,
        # CausalLM .bin generation settings (env takes precedence)
        causallm_path=causallm_path or env_causallm_path,
        enable_quantization=enable_quantization if enable_quantization is not None else env_enable_quant,
        quantization_preset=quantization_preset or env_quant_preset,
        target_isa=target_isa or env_target_isa,
        nntr_quantize_path=nntr_quantize_path or env_nntr_quantize,
        auto_compile=auto_compile,
        use_docker_build=use_docker_build,
        auto_fix_use_llm=auto_fix_use_llm,
        nntrainer_path=nntrainer_path or os.environ.get("NNTRAINER_PATH"),
        # Pre-existing code configuration
        use_pre_existing_code=use_pre_existing_code,
        nntrainer_root=nntrainer_root or os.environ.get("NNTRAINER_ROOT", "/storage_data/snap/Prachi/nntrainer"),
    ))

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

    # Record to memory store for RAG-style recall
    _record_to_memory(model_name, final_state)

    import json
    with open(os.path.join(out_dir, "state.json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe(final_state), f, indent=2)

    return final_state


def _record_to_memory(model_name: str, final_state: dict):
    """
    Record pipeline run outcome to memory store for future recall.
    
    Captures:
    - Success/failure status
    - Errors encountered
    - Fixes applied
    - Duration and token usage
    """
    try:
        memory = get_memory_store()
        
        # Determine overall success
        errors = final_state.get("errors", [])
        success = len(errors) == 0 and final_state.get("compiled", False)
        
        # Record compile errors
        if final_state.get("compile_log"):
            record_event({
                "model_name": model_name,
                "event_type": "compile_error" if not final_state.get("compiled") else "compile_success",
                "error_type": _classify_error(final_state.get("compile_log", "")),
                "error_msg": final_state.get("compile_log", "")[:500],  # Truncate for storage
                "fix_applied": final_state.get("fix_applied", ""),
                "file_path": final_state.get("cpp_file_path", ""),
                "content": f"Compilation {'failed' if not final_state.get('compiled') else 'succeeded'} for {model_name}. Log: {final_state.get('compile_log', '')[:1000]}",
                "agent_name": "compiler_agent",
                "success": final_state.get("compiled", False),
            })
        
        # Record build errors
        if final_state.get("causallm_build_log"):
            record_event({
                "model_name": model_name,
                "event_type": "build_error" if not final_state.get("causallm_so_built") else "build_success",
                "error_type": _classify_error(final_state.get("causallm_build_log", "")),
                "error_msg": final_state.get("causallm_build_log", "")[:500],
                "fix_applied": final_state.get("fix_applied", ""),
                "file_path": final_state.get("cpp_file_path", ""),
                "content": f"CausalLM build {'failed' if not final_state.get('causallm_so_built') else 'succeeded'} for {model_name}. Log: {final_state.get('causallm_build_log', '')[:1000]}",
                "agent_name": "causallm_so_builder",
                "success": final_state.get("causallm_so_built", False),
            })
        
        # Record auto-fix events
        if final_state.get("fix_iterations", 0) > 0:
            record_event({
                "model_name": model_name,
                "event_type": "auto_fix_applied",
                "error_type": _classify_error(final_state.get("compile_log", "")),
                "error_msg": final_state.get("compile_log", "")[:500],
                "fix_applied": f"Auto-fix applied ({final_state.get('fix_iterations', 0)} iterations)",
                "file_path": final_state.get("cpp_file_path", ""),
                "content": f"Auto-fix applied {final_state.get('fix_iterations', 0)} times for {model_name}",
                "agent_name": "auto_fix",
                "success": True,
            })
        
        # Record overall pipeline result
        record_event({
            "model_name": model_name,
            "event_type": "pipeline_complete",
            "error_type": None,
            "error_msg": "; ".join(errors) if errors else "",
            "fix_applied": "",
            "file_path": "",
            "content": f"Pipeline completed for {model_name}. Architecture: {final_state.get('architecture', 'unknown')}. Compiled: {final_state.get('compiled', False)}. Errors: {errors}",
            "agent_name": "orchestrator",
            "success": success,
            "duration_sec": final_state.get("duration_sec"),
            "tokens_used": final_state.get("tokens_used"),
        })
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.debug("Failed to record to memory: %s", e)


def _classify_error(log: str) -> str:
    """Classify error type from log message."""
    log_lower = log.lower()
    
    if "undefined reference" in log_lower or "undefined symbol" in log_lower:
        return "undefined_reference"
    elif "not found" in log_lower or "no such file" in log_lower:
        return "missing_file"
    elif "missing include" in log_lower or "#include" in log_lower:
        return "missing_include"
    elif "undefined" in log_lower:
        return "undefined_symbol"
    elif "type mismatch" in log_lower or "expected" in log_lower:
        return "type_error"
    elif "segmentation fault" in log_lower or "segfault" in log_lower:
        return "segfault"
    elif "timeout" in log_lower:
        return "timeout"
    elif "permission" in log_lower or "access denied" in log_lower:
        return "permission_error"
    elif "memory" in log_lower or "out of memory" in log_lower:
        return "memory_error"
    else:
        return "unknown"


def _json_safe(state: dict) -> dict:
    import json as _json

    def default(o):
        return str(o)

    safe = _json.loads(_json.dumps(state, default=default))
    safe.pop("anthropic_api_key", None)
    return safe
