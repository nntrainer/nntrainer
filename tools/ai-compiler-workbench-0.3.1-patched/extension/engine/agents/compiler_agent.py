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

from .events import bus
from .nntrainer_env import discover_flags


def _normalise_emission_mode(state: dict) -> str:
    mode = state.get("cpp_emission_mode", "")
    bus.log(f"Compiler received cpp_emission_mode={mode!r}","info",)
    if hasattr(mode, "value"):
        mode = mode.value
    return str(mode or "").strip().lower()


def _handle_causallm_component(state: dict) -> dict:
    project_root = state.get("causallm_project_root")
    install_enabled = bool(state.get("install_generated_files", False))

    state["compiled"] = False
    state["compile_skipped"] = True
    state["binary_path"] = None
    state["requires_causallm_build"] = True

    if not install_enabled:
        message = (
            "CausalLM component generated successfully. Standalone compilation "
            "is not applicable. Install Generated Files is disabled, so the "
            "component remains in the Workbench output directory."
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
