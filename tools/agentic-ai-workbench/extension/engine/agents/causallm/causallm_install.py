"""
CausalLM Install Agent (no LLM, optional).

Copies the generated CausalLM component (written by cpp_generator_agent
to <out_dir>/generated/causallm/<arch>/) directly into a real CausalLM
project's models/<arch>/ directory -- but only when explicitly asked to.
Three independent things all have to be true, or this is a no-op:

  1. state["install_generated_files"] is True (default False --
     aiCompilerWorkbench.installGeneratedFiles)
  2. state["causallm_project_root"] is set and exists on disk
  3. this run actually produced a CausalLM component (causallm_header_path
     / causallm_source_path in state -- nothing to install for MODEL_API
     runs or a failed/skipped generation)

The generated files (e.g., Qwen3_causallm.h/cpp) are placed directly into
the models/qwen3/ directory, replacing only the model-specific architecture
code. All framework code (causal_lm.h, transformer.h, etc.) comes from
nntrainer and remains unchanged.

None of these being true is not an error -- it's the default, safe
behavior (see doc item 17: "safer default installGeneratedFiles=false").

USE_PRE_EXISTING_CODE MODE:
  When state["use_pre_existing_code"] is True, this agent will use the
  pre-existing code from the nntrainer repository instead of generated code.
  This is useful when you have manually created or modified code in the
  models/<arch>/ directory and want to use that for compilation.
"""
import json
import os
import re
import shutil

from ..events import bus


def _append_once(path: str, marker: str, text: str) -> None:
    """Append generated integration text once; never rewrite user code."""
    with open(path, "r", encoding="utf-8") as f:
        current = f.read()
    if marker in current:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n" + text)


def _register_generated_model(project_root: str, model_manifest: dict,
                              header_path: str, source_path: str) -> None:
    """Make an installed component visible to the existing Meson executable.

    This uses idempotent, clearly marked additions only. Existing handwritten
    model entries and the build command remain untouched.
    """
    architecture = model_manifest.get("architecture", "")
    class_name = model_manifest.get("class", "")
    implementation = model_manifest.get("implementation", "")
    slug = implementation.rsplit("/", 1)[-1]
    if not re.fullmatch(r"[a-z0-9_]+", slug):
        raise ValueError(f"unsafe generated model slug: {slug!r}")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*CausalLM", class_name):
        raise ValueError(f"unsafe or missing generated CausalLM class: {class_name!r}")
    header = os.path.basename(header_path)
    source = os.path.basename(source_path)
    # Generated models deliberately live outside handwritten model folders.
    # That makes a generated Qwen3 independently buildable/reviewable without
    # overwriting Applications/CausalLM/models/qwen3.
    model_dir = os.path.join(project_root, "models", "generated", slug)
    meson_path = os.path.join(model_dir, "meson.build")
    if not os.path.exists(meson_path):
        with open(meson_path, "w", encoding="utf-8") as f:
            f.write(
                f"{slug}_src = [meson.current_source_dir() / '{source}']\n"
                f"{slug}_inc = include_directories('.')\n"
                f"causallm_src += {slug}_src\ncausallm_inc += {slug}_inc\n")
    _append_once(os.path.join(project_root, "models", "generated", "meson.build"),
                 f"# workbench-generated-model:{slug}",
                 f"# workbench-generated-model:{slug}\nsubdir('{slug}')")
    registry_source = f"{slug}_workbench_registry.cpp"
    registry_path = os.path.join(model_dir, registry_source)
    with open(registry_path, "w", encoding="utf-8") as f:
        f.write(f'''// workbench-generated-model:{slug}
#include <memory>
#include <factory.h>
#include <json.hpp>
#include "{header}"
namespace {{
const causallm::Factory::GeneratedModelManifest kManifest{{
  "{architecture}", "{implementation}", "{class_name}",
  "{model_manifest.get('tensor_map', 'tensor_map.json')}",
  "{model_manifest.get('weight_converter', 'weight_converter.py')}"
}};
struct WorkbenchRegistrar {{
  WorkbenchRegistrar() {{
    causallm::Factory::Instance().registerGeneratedModel(kManifest,
      [](nlohmann::json &cfg, nlohmann::json &generation_cfg, nlohmann::json &nntr_cfg) {{
        return std::make_unique<causallm::{class_name}>(cfg, generation_cfg, nntr_cfg);
      }});
  }}
}} workbench_registrar;
}} // namespace
''')
    with open(meson_path, "r", encoding="utf-8") as f:
        meson = f.read()
    if registry_source not in meson:
        meson = meson.replace(f"'{source}'", f"'{source}', meson.current_source_dir() / '{registry_source}'")
        with open(meson_path, "w", encoding="utf-8") as f:
            f.write(meson)


def _extract_model_slug(architecture: str) -> str:
    """Extract capitalized model name from architecture.
    E.g., 'Qwen3ForCausalLM' -> 'Qwen3', 'Gemma3CausalLM' -> 'Gemma3'.
    Falls back to lowercase if extraction fails."""
    slug = architecture.replace("ForCausalLM", "").replace("CausalLM", "").replace("Model", "")
    if slug and len(slug) > 0:
        return slug[0].upper() + slug[1:] if len(slug) > 1 else slug.upper()
    return architecture.lower()


def run(state: dict) -> dict:
    bus.agent_status("causallm_install", "running")

    # Generated components are authoritative for a generation run. Reusing a
    # handwritten model remains opt-in only; otherwise this stage would
    # silently replace the just-validated source before compilation.
    use_pre_existing = state.get("use_pre_existing_code", False)
    
    if use_pre_existing:
        return _use_pre_existing_code(state)
    
    # Install generated code into CausalLM project
    header_path = state.get("causallm_header_path")
    source_path = state.get("causallm_source_path")
    
    if not (header_path and source_path):
        bus.log("No generated code found (causallm_header_path/source_path not set)", "warn")
        bus.agent_status("causallm_install", "done", "nothing to install")
        return state
    
    # Check if install_generated_files is enabled
    if not state.get("install_generated_files"):
        bus.log(
            "Install Generated Files is off (default) -- generated component stays "
            "under <output>/generated/causallm/ only. Enable "
            "aiCompilerWorkbench.installGeneratedFiles to copy it into your CausalLM project.",
            "info",
        )
        bus.agent_status("causallm_install", "done", "disabled")
        return state
    
    # Get CausalLM project root
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
    
    # Get architecture and determine model directory
    architecture = state.get("architecture", "")
    if not architecture:
        bus.log("No architecture found in state -- cannot determine model directory", "warn")
        bus.agent_status("causallm_install", "error", "no architecture")
        return state
    
    model_slug = _extract_model_slug(architecture).lower() + "_workbench"
    # Never install generated output over a handwritten architecture. The
    # generated tree is compiled through its own Meson subdirectory.
    model_dir = os.path.join(project_root, "models", "generated", model_slug)
    os.makedirs(model_dir, exist_ok=True)
    
    # Validate the generated bundle before copying any part of it into the
    # CausalLM tree. A partial model target must never be build-selectable.
    manifest_path = state.get("generated_model_manifest_path")
    if not manifest_path or not os.path.isfile(manifest_path):
        bus.log("Generated model_manifest.json is missing -- refusing installation", "error")
        bus.agent_status("causallm_install", "error", "missing model manifest")
        return state
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            bundle = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        bus.log(f"Generated model manifest is invalid: {exc}", "error")
        bus.agent_status("causallm_install", "error", "invalid model manifest")
        return state
    if bundle.get("implementation", "").rsplit("/", 1)[-1] != model_slug:
        bus.log("Generated model manifest does not match installation target", "error")
        bus.agent_status("causallm_install", "error", "manifest target mismatch")
        return state
    required_artifacts = (
        ("tensor_map_path", "tensor_map.json"),
        ("generated_weight_converter_path", "weight_converter.py"),
    )
    missing = [filename for key, filename in required_artifacts
               if not state.get(key) or not os.path.isfile(state[key])]
    if missing:
        bus.log(f"Generated model bundle is incomplete: missing {', '.join(missing)}", "error")
        bus.agent_status("causallm_install", "error", "incomplete generated bundle")
        return state

    # Install header and source files
    installed_header = os.path.join(model_dir, os.path.basename(header_path))
    installed_source = os.path.join(model_dir, os.path.basename(source_path))
    shutil.copyfile(header_path, installed_header)
    shutil.copyfile(source_path, installed_source)

    # The model is a bundle: source, manifest, tensor map and converter must
    # travel together. Do not substitute a handwritten CausalLM converter.
    for state_key, filename in (
        ("tensor_map_path", "tensor_map.json"),
        ("generated_weight_converter_path", "weight_converter.py"),
        ("generated_model_manifest_path", "model_manifest.json"),
    ):
        artifact = state.get(state_key)
        shutil.copyfile(artifact, os.path.join(model_dir, filename))

    try:
        _register_generated_model(project_root, bundle, installed_header,
                                  installed_source)
    except (OSError, ValueError) as exc:
        # Keep the generated artifacts for review, but do not claim that they
        # are build-integrated when registration metadata could not be written.
        state.setdefault("errors", []).append(f"generated model integration: {exc}")
        bus.log(f"Generated model integration incomplete: {exc}", "error")
    else:
        state["causallm_model_build_registered"] = True
        bus.log(f"Registered generated source in models/generated/{model_slug}/meson.build", "info")
    
    state["causallm_installed_header_path"] = installed_header
    state["causallm_installed_source_path"] = installed_source
    
    # Copy .bin file if available
    bin_path = state.get("causallm_bin_path")
    if bin_path and os.path.exists(bin_path):
        installed_bin = os.path.join(model_dir, "nntr_model.bin")
        shutil.copyfile(bin_path, installed_bin)
        state["causallm_installed_bin_path"] = installed_bin
        bus.log(f"Installed .bin file: {installed_bin}")
    
    bus.log(f"Installed generated component into CausalLM project: {installed_header}, {installed_source}")
    bus.agent_status("causallm_install", "done", f"{installed_header}, {installed_source}")
    return state


def _use_pre_existing_code(state: dict) -> dict:
    """
    Use pre-existing code from the nntrainer repository instead of generated code.
    
    This is useful when you have manually created or modified code in the
    models/<arch>/ directory and want to use that for compilation.
    """
    bus.log("Using pre-existing code from nntrainer repository", "info")
    
    # Get the nntrainer repository root
    nntrainer_root = state.get("nntrainer_root")
    if not nntrainer_root:
        # Try to infer from common locations
        nntrainer_root = _find_nntrainer_root()
    
    if not nntrainer_root or not os.path.isdir(nntrainer_root):
        bus.log("NNTrainer repository root not found -- cannot use pre-existing code", "error")
        bus.agent_status("causallm_install", "error", "nntrainer_root not found")
        return state
    
    # Get architecture and model slug
    architecture = state.get("architecture", "")
    if not architecture:
        bus.log("No architecture found in state -- cannot determine model directory", "error")
        bus.agent_status("causallm_install", "error", "no architecture")
        return state
    
    model_slug = _extract_model_slug(architecture).lower()
    
    # Path to pre-existing model code
    pre_existing_model_dir = os.path.join(nntrainer_root, "Applications", "CausalLM", "models", model_slug)
    
    if not os.path.isdir(pre_existing_model_dir):
        bus.log(f"Pre-existing model directory not found: {pre_existing_model_dir}", "error")
        bus.agent_status("causallm_install", "error", f"model dir not found: {model_slug}")
        return state
    
    # Find the header and source files
    header_file = f"{model_slug}_causallm.h"
    source_file = f"{model_slug}_causallm.cpp"
    
    header_path = os.path.join(pre_existing_model_dir, header_file)
    source_path = os.path.join(pre_existing_model_dir, source_file)
    
    if not os.path.exists(header_path):
        bus.log(f"Pre-existing header not found: {header_path}", "warn")
    if not os.path.exists(source_path):
        bus.log(f"Pre-existing source not found: {source_path}", "warn")
    
    if os.path.exists(source_path):
        # Read the pre-existing source code
        with open(source_path, "r", encoding="utf-8") as f:
            source_content = f.read()
        
        # Read the header if it exists
        header_content = ""
        if os.path.exists(header_path):
            with open(header_path, "r", encoding="utf-8") as f:
                header_content = f.read()
        
        # Set state as if we generated this code
        state["causallm_header"] = header_content
        state["causallm_source"] = source_content
        state["causallm_header_path"] = header_path
        state["causallm_source_path"] = source_path
        state["causallm_installed_header_path"] = header_path
        state["causallm_installed_source_path"] = source_path
        state["cpp_emission_mode"] = "causallm_component"
        state["requires_causallm_build"] = True
        state["cpp_generation_method"] = "pre_existing"
        state["cpp_code"] = header_content + "\n\n" + source_content
        state["cpp_path"] = source_path
        
        bus.log(f"Using pre-existing code from: {pre_existing_model_dir}")
        bus.log(f"  Header: {header_path}")
        bus.log(f"  Source: {source_path}")
        bus.agent_status("causallm_install", "done", f"pre-existing {model_slug}")
    else:
        bus.log(f"No pre-existing source code found for {model_slug}", "error")
        bus.agent_status("causallm_install", "error", "no pre-existing source")
        state.setdefault("errors", []).append(f"No pre-existing source for {model_slug}")
    
    return state


def _find_nntrainer_root() -> str:
    """Try to find the nntrainer repository root."""
    # Common locations
    search_paths = [
        "/storage_data/snap/Prachi/nntrainer",
        os.path.expanduser("~/nntrainer"),
        os.path.expanduser("~/projects/nntrainer"),
        "/workspace/nntrainer",
    ]
    
    for path in search_paths:
        if os.path.isdir(path):
            # Verify it looks like nntrainer root
            if os.path.exists(os.path.join(path, "Applications", "CausalLM")):
                return path
    
    return ""
