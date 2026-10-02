"""
CausalLM Install Agent (no LLM, optional).

Copies the generated CausalLM component (written by cpp_generator_agent
to <out_dir>/generated/causallm/<arch>/) into a real CausalLM project's
generated-header/source directories -- but only when explicitly asked
to. Three independent things all have to be true, or this is a no-op:

  1. state["install_generated_files"] is True (default False --
     aiCompilerWorkbench.installGeneratedFiles)
  2. state["causallm_project_root"] is set and exists on disk
  3. this run actually produced a CausalLM component (causallm_header_path
     / causallm_source_path in state -- nothing to install for MODEL_API
     runs or a failed/skipped generation)

None of these being true is not an error -- it's the default, safe
behavior (see doc item 17: "safer default installGeneratedFiles=false").
This agent never touches the CausalLM project's *handwritten* files
(e.g. src/qwen3_causallm.cpp); it only ever writes into the configured
generated-header/generated-source subdirectories, and only ever writes
files this same run just generated.
"""
import os
import shutil

from .events import bus


def run(state: dict) -> dict:
    bus.agent_status("causallm_install", "running")

    header_path = state.get("causallm_header_path")
    source_path = state.get("causallm_source_path")
    if not (header_path and source_path):
        bus.agent_status("causallm_install", "done", "nothing to install")
        return state

    if not state.get("install_generated_files"):
        bus.log(
            "Install Generated Files is off (default) -- generated component stays "
            "under <output>/generated/causallm/ only. Enable "
            "aiCompilerWorkbench.installGeneratedFiles to copy it into your CausalLM project.",
            "info",
        )
        bus.agent_status("causallm_install", "done", "disabled")
        return state

    project_root = state.get("causallm_project_root")
    if not project_root:
        bus.log(
            "installGeneratedFiles is on but aiCompilerWorkbench.causallmProjectRoot "
            "isn't set -- nothing to install into", "warn",
        )
        bus.agent_status("causallm_install", "error", "no causallmProjectRoot configured")
        return state

    if not os.path.isdir(project_root):
        bus.log(f"causallmProjectRoot '{project_root}' doesn't exist -- skipping install", "warn")
        bus.agent_status("causallm_install", "error", "causallmProjectRoot not found")
        return state

    header_dir = os.path.join(project_root, state.get("generated_header_directory", "include/generated"))
    source_dir = os.path.join(project_root, state.get("generated_source_directory", "src/generated"))
    os.makedirs(header_dir, exist_ok=True)
    os.makedirs(source_dir, exist_ok=True)

    installed_header = os.path.join(header_dir, os.path.basename(header_path))
    installed_source = os.path.join(source_dir, os.path.basename(source_path))
    shutil.copyfile(header_path, installed_header)
    shutil.copyfile(source_path, installed_source)

    state["causallm_installed_header_path"] = installed_header
    state["causallm_installed_source_path"] = installed_source

    bus.log(f"Installed generated component into CausalLM project: {installed_header}, {installed_source}")
    bus.log(
        "Remember to add these to your build (see doc item 15 -- CMake target_sources / "
        "Meson generated_causallm_sources) and rebuild through your CausalLM project's own "
        "build system; this extension does not invoke that build for you."
    )
    bus.agent_status("causallm_install", "done", f"{installed_header}, {installed_source}")
    return state
