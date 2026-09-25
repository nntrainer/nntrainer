"""
Compiler Agent (no LLM).

Compiles ordinary generated nntrainer MODEL_API C++ files against a real
nntrainer installation.

CausalLM components are not standalone programs. They must be installed and
built through a configured CausalLM project, so this agent skips the generic
standalone g++ command for that emission mode.
"""
import os
import shutil
import subprocess

from ..events import bus
from ..nntrainer_env import discover_flags


def _normalise_emission_mode(state: dict) -> str:
    mode = state.get("cpp_emission_mode", "")
    bus.log(f"Compiler received cpp_emission_mode={mode!r}","info",)
    if hasattr(mode, "value"):
        mode = mode.value
    return str(mode or "").strip().lower()


def _syntax_check_causallm_component(state: dict) -> None:
    """
    Best-effort fast syntax-only check for a freshly generated CausalLM
    component. A component can't be linked standalone (it needs the rest
    of the CausalLM project), but a syntax-only pass (`-fsyntax-only`)
    catches obvious bracket/syntax mistakes immediately, before the much
    slower meson/ninja build (see causallm_so_builder.py, the authoritative
    build gate) runs. Populates state["compile_log"]/state["syntax_ok"];
    silently skips if nntrainer or g++ aren't available on the host (the
    real build may only be reachable inside Docker, in which case this
    fast check simply has nothing to run against).
    """
    cpp_path = state.get("causallm_source_path") or state.get("causallm_installed_source_path")
    if not cpp_path or not os.path.isfile(cpp_path):
        return

    compiler = shutil.which("g++")
    if compiler is None:
        return

    cflags, _libs, _source = discover_flags()
    if cflags is None:
        bus.log("nntrainer not found on host -- skipping fast syntax check (full build still runs)", "info")
        state["nntrainer_missing"] = True
        return

    include_flags = list(cflags)
    
    # Add CausalLM project include paths for llm_util.hpp and other headers
    project_root = state.get("causallm_project_root")
    if project_root and os.path.isdir(project_root):
        for candidate in (project_root, os.path.join(project_root, "include")):
            if os.path.isdir(candidate):
                include_flags += ["-I", candidate]
    
    # Add nntrainer CausalLM include directory (where llm_util.hpp lives).
    # nntrainer_root is the repo root itself here (Applications/CausalLM sits
    # directly under it) -- there is no extra nested "nntrainer/" component.
    nntrainer_root = state.get("nntrainer_root", "/storage_data/snap/Prachi/nntrainer")
    causallm_include = os.path.join(nntrainer_root, "Applications", "CausalLM")
    if os.path.isdir(causallm_include):
        include_flags += ["-I", causallm_include]
        layers_include = os.path.join(causallm_include, "layers")
        if os.path.isdir(layers_include):
            include_flags += ["-I", layers_include]
        # Add models subdirectory for causal_lm.h and model-specific headers
        models_include = os.path.join(causallm_include, "models")
        if os.path.isdir(models_include):
            include_flags += ["-I", models_include]
    
    # Add CCAPI include directory for layer.h (required by app_context.h)
    ccapi_include = os.path.join(nntrainer_root, "api", "ccapi", "include")
    if os.path.isdir(ccapi_include):
        include_flags += ["-I", ccapi_include]
    
    # Add the directory of the cpp file being checked
    include_flags += ["-I", os.path.dirname(cpp_path)]

    command = [compiler, "-std=c++17", "-fsyntax-only", *include_flags, cpp_path]
    bus.log(f"Fast syntax check: {' '.join(command)}")

    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    except (subprocess.TimeoutExpired, OSError) as exc:
        bus.log(f"Syntax check could not run: {exc}", "warn")
        return

    compile_log = (process.stdout or "") + (process.stderr or "")
    state["compile_log"] = compile_log
    state["syntax_ok"] = process.returncode == 0

    if process.returncode == 0:
        bus.log("Fast syntax check passed", "info")
    else:
        bus.log(f"Fast syntax check found issues (exit {process.returncode})", "warn")
        for line in compile_log.splitlines()[:20]:
            bus.log(f"  {line}", "warn")


def _handle_causallm_component(state: dict) -> dict:
    project_root = state.get("causallm_project_root")
    install_enabled = bool(state.get("install_generated_files", False))

    # Preserve compiled/binary_path if already set by causallm_build_run
    already_compiled = state.get("compiled", False)
    existing_binary = state.get("binary_path")

    state["compile_skipped"] = True
    state["requires_causallm_build"] = True
    state["compile_attempted"] = True  # Mark as attempted to prevent infinite retry loop

    # Only set compiled=False if causallm_build_run hasn't already run
    if not already_compiled:
        state["compiled"] = False
        state["binary_path"] = None

    bus.log("Processing CausalLM component (standalone nntrainer compilation not applicable)", "info")

    _syntax_check_causallm_component(state)

    if not install_enabled:
        message = (
            "CausalLM component generated successfully. Standalone compilation "
            "is not applicable. Install Generated Files is disabled, so the "
            "component remains in the Workbench output directory. "
            "Build the component using your CausalLM project's CMake/Meson build system."
        )
        state["compile_log"] = message
        bus.log(message, "info")
        bus.agent_status("compiler", "skipped", "CausalLM project build required")
        return state

    if not project_root:
        message = (
            "CausalLM component generated successfully, but "
            "causallmProjectRoot is not configured. Build was skipped."
        )
        state["compile_log"] = message
        bus.log(message, "warn")
        bus.agent_status("compiler", "skipped", "CausalLM project root missing")
        return state

    if not os.path.isdir(project_root):
        message = f"Configured CausalLM project root does not exist: {project_root}"
        state["compile_log"] = message
        bus.log(message, "error")
        bus.agent_status("compiler", "error", "invalid CausalLM project root")
        return state

    message = (
        "Generated files were installed into the CausalLM project. Standalone "
        "nntrainer compilation was skipped; build the component with the "
        "CausalLM project's own CMake or Meson target."
    )
    state["compile_log"] = message
    bus.log(message, "info")
    bus.agent_status("compiler", "skipped", "use CausalLM project build")
    return state


def run(state: dict) -> dict:
    bus.agent_status("compiler", "running")

    if _normalise_emission_mode(state) == "causallm_component":
        return _handle_causallm_component(state)

    if state.get("skip_cpp_compilation"):
        message = "C++ compilation skipped (skip_cpp_compilation=True)"
        bus.log(message, "warn")
        state["compiled"] = False
        state["compile_skipped"] = True
        state["compile_attempted"] = True
        state["compile_log"] = "Compilation skipped per configuration"
        state["binary_path"] = None
        bus.agent_status("compiler", "skipped", "C++ compilation disabled")
        return state

    cpp_path = state.get("cpp_path")
    if not cpp_path or not os.path.isfile(cpp_path):
        message = "Generated C++ file was not found"
        state["compiled"] = False
        state["compile_skipped"] = False
        state["compile_log"] = message
        state["binary_path"] = None
        bus.log(message, "error")
        bus.agent_status("compiler", "error", "no generated file")
        return state

    compiler = shutil.which("g++")
    if compiler is None:
        message = "g++ not found on PATH"
        state["compiled"] = False
        state["compile_skipped"] = True
        state["compile_attempted"] = True
        state["compile_log"] = message
        state["binary_path"] = None
        bus.log(f"{message} -- skipping compilation", "warn")
        bus.agent_status("compiler", "skipped", "g++ not found")
        return state

    cflags, libs, source = discover_flags()
    if cflags is None:
        message = (
            "nntrainer not found. Set NNTRAINER_INCLUDE_DIR and optionally "
            "NNTRAINER_LIB_DIR, or install nntrainer so its pkg-config file "
            "is available."
        )
        state["compiled"] = False
        state["compile_skipped"] = True
        state["compile_attempted"] = True
        state["nntrainer_missing"] = True
        state["compile_log"] = message
        state["binary_path"] = None
        bus.log(message, "warn")
        bus.agent_status("compiler", "skipped", "nntrainer not found")
        return state

    state["nntrainer_missing"] = False
    state["compile_skipped"] = False
    bus.log(f"Found nntrainer via {source}")

    out_dir = os.path.dirname(cpp_path)
    binary_path = os.path.join(out_dir, "model_bin")
    command = [
        compiler,
        "-std=c++17",
        "-DNNTRAINER_STANDALONE_SMOKE_TEST",
        *cflags,
        cpp_path,
        "-o",
        binary_path,
        *libs,
    ]
    command_text = " ".join(command)
    state["compile_command"] = command_text
    bus.log(f"Compiling: {command_text}")

    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except subprocess.TimeoutExpired:
        message = "Compilation timed out after 180 seconds"
        state["compiled"] = False
        state["compile_log"] = message
        state["binary_path"] = None
        bus.log(message, "error")
        bus.agent_status("compiler", "error", "compilation timeout")
        return state
    except OSError as exc:
        message = f"Failed to start compiler: {exc}"
        state["compiled"] = False
        state["compile_log"] = message
        state["binary_path"] = None
        bus.log(message, "error")
        bus.agent_status("compiler", "error", "compiler invocation failed")
        return state

    compile_log = (process.stdout or "") + (process.stderr or "")
    state["compile_log"] = compile_log

    if process.returncode == 0:
        state["compiled"] = True
        state["binary_path"] = binary_path
        bus.log("Compilation successful")
        bus.agent_status("compiler", "done", "success")
        return state

    state["compiled"] = False
    state["binary_path"] = None
    bus.log(f"Compilation failed (exit {process.returncode})", "error")
    for line in compile_log.splitlines()[:30]:
        bus.log(f"  {line}", "error")
    bus.agent_status("compiler", "error", f"exit {process.returncode}")
    return state
