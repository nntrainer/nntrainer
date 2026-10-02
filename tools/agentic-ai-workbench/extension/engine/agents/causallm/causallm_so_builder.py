"""
CausalLM .so Builder Agent - Builds libcausallm.so from source.

This agent builds the CausalLM runtime library (libcausallm.so) and
executable (nntr_causallm) from the CausalLM source code using meson + ninja.

Unlike using pre-built binaries, this ensures the .so is generated fresh
for the current platform and configuration.
"""
import os
import subprocess
import shutil
from typing import Optional

from ..events import bus
from ..build import docker_builder


def _run_build_step(command_args, cwd: str, use_docker: bool, timeout: int) -> dict:
    """
    Run one build step (a meson or ninja invocation) either on the host or
    inside the shared Docker build image, mounting only `cwd` (the CausalLM
    source tree) -- not the whole filesystem.

    Returns {"success": bool, "stdout": str, "stderr": str} either way, so
    callers don't need to branch on how it ran.
    """
    if use_docker:
        return docker_builder.run_in_container(
            command=" ".join(command_args),
            mounts={cwd: "/workspace/causallm"},
            workdir="/workspace/causallm",
            log_fn=lambda m, lvl="info": bus.log(f"  {m}", lvl),
            timeout=timeout,
        )

    result = subprocess.run(command_args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return {"success": result.returncode == 0, "stdout": result.stdout, "stderr": result.stderr}


def _mark_build_failure(state: dict, stage: str, full_log: str, short_error: str) -> None:
    """
    Record a build failure and decide whether it's attributable to the
    generated component file -- only then is it worth looping into
    auto_fix; rewriting unrelated CausalLM project files is out of scope.
    """
    bus.log(f"{stage} failed: {short_error}", "error")
    bus.agent_status("causallm_so_builder", "error", f"{stage.lower()} failed")
    state["causallm_so_built"] = False
    state["causallm_build_error"] = short_error

    installed_source = state.get("causallm_installed_source_path")
    generated_basename = os.path.basename(installed_source) if installed_source else ""
    fixable = bool(generated_basename) and generated_basename in full_log

    state["causallm_build_fixable"] = fixable
    if fixable:
        state["compile_log"] = full_log
        state["cpp_path"] = installed_source
        try:
            with open(installed_source, "r", encoding="utf-8") as f:
                state["cpp_code"] = f.read()
        except OSError as exc:
            bus.log(f"Could not read installed source for auto-fix: {exc}", "warn")
            state["causallm_build_fixable"] = False
    else:
        bus.log(
            "Build failure isn't attributable to the generated file -- not looping into auto-fix",
            "warn",
        )


def _create_model_directory(state: dict) -> Optional[str]:
    """
    Create a proper model directory for the profiler.
    
    The nntr_causallm binary expects a directory containing:
    - nntr_model.bin (or the actual .bin file)
    - config.json
    
    This function creates a run_model/ directory with symlinks/copies of these files.
    """
    out_dir = state.get("out_dir", "")
    if not out_dir:
        return None

    if not state.get("weights_verified"):
        bus.log("Weights are not verified -- refusing to prepare a runnable model directory", "error")
        return None
    
    model_dir = os.path.join(out_dir, "run_model")
    
    # Find the .bin file
    bin_path = state.get("converted_weights_path") or state.get("weights_path")
    if not bin_path or not os.path.exists(bin_path):
        # Search for .bin files in out_dir
        import glob
        bin_files = glob.glob(os.path.join(out_dir, "*.bin"))
        if bin_files:
            bin_path = bin_files[0]
    
    if not bin_path or not os.path.exists(bin_path):
        bus.log("No .bin file found for model directory", "warn")
        return None
    
    # Find config.json - search in generated/causallm/<arch>/ directories
    config_path = None
    causallm_generated_dir = os.path.join(out_dir, "generated", "causallm")
    
    if os.path.isdir(causallm_generated_dir):
        # Search for config.json in any subdirectory
        import glob
        config_matches = glob.glob(os.path.join(causallm_generated_dir, "*", "config.json"))
        if config_matches:
            config_path = config_matches[0]
            bus.log(f"Found config.json at: {config_path}")
    
    if not config_path or not os.path.exists(config_path):
        # Try root out_dir as fallback
        config_path = os.path.join(out_dir, "config.json")
        if not os.path.exists(config_path):
            bus.log("config.json not found for model directory", "warn")
            return None
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Copy .bin file (rename to nntr_model.bin for standard name)
    model_bin_path = os.path.join(model_dir, "nntr_model.bin")
    shutil.copy2(bin_path, model_bin_path)
    bus.log(f"Copied {os.path.basename(bin_path)} -> {model_dir}/nntr_model.bin")
    
    # Copy model.ini (binary expects nntr_model.ini, not JSON!)
    # The generated model.ini contains the full nntrainer graph definition
    ini_source = os.path.join(out_dir, "generated", "model.ini")
    if os.path.exists(ini_source):
        model_ini_path = os.path.join(model_dir, "nntr_model.ini")
        shutil.copy2(ini_source, model_ini_path)
        bus.log(f"Copied model.ini -> {model_dir}/nntr_model.ini")
    else:
        bus.log("model.ini not found - binary may fail without it", "warn")
    
    state["model_dir"] = model_dir
    state["causallm_bin_path"] = model_bin_path
    state["causallm_ini_path"] = os.path.join(model_dir, "nntr_model.ini")
    
    bus.log(f"Created model directory: {model_dir}")
    return model_dir


def find_causallm_source(state: dict) -> Optional[str]:
    """
    Find CausalLM source directory.
    
    Priority order:
    1. nntrainer_root from state (user-configured path) - try both flat and nested structures
    2. Default: /storage_data/snap/Prachi/nntrainer/Applications/CausalLM
    3. Fallback search paths
    
    Args:
        state: Pipeline state dictionary
        
    Returns:
        Path to CausalLM source if found, None otherwise
    """
    def is_valid_causallm_path(path: str) -> bool:
        """Check if path is a valid CausalLM source directory."""
        return (os.path.isdir(path) and 
                os.path.exists(os.path.join(path, "meson.build")) and 
                os.path.exists(os.path.join(path, "main.cpp")))
    
    # First, try the nntrainer_root from state (this is the user's existing nntrainer path)
    nntrainer_root = state.get("nntrainer_root")
    if nntrainer_root:
        # Try flat structure first (nntrainer_root/Applications/CausalLM)
        causallm_path = os.path.join(nntrainer_root, "Applications", "CausalLM")
        if is_valid_causallm_path(causallm_path):
            return causallm_path
        
        # Try nested structure (nntrainer_root/nntrainer/Applications/CausalLM)
        causallm_path = os.path.join(nntrainer_root, "nntrainer", "Applications", "CausalLM")
        if is_valid_causallm_path(causallm_path):
            return causallm_path
    
    # Fallback: try common locations
    workspace_root = state.get("out_dir", "")
    workspace_root = os.path.dirname(workspace_root) if workspace_root else ""
    
    search_paths = [
        os.path.join(workspace_root, "Applications", "CausalLM"),
        os.path.join(workspace_root, "nntrainer", "Applications", "CausalLM"),
        os.path.join(workspace_root, "..", "Applications", "CausalLM"),
        os.path.expanduser("~/nntrainer/Applications/CausalLM"),
        "/storage_data/snap/Prachi/nntrainer/Applications/CausalLM",  # Fixed: flat structure
    ]
    
    for path in search_paths:
        if is_valid_causallm_path(path):
            return path
    
    return None


def run(state: dict) -> dict:
    """
    Build nntrainer with CausalLM using meson + ninja (same as manual build).
    
    This runs meson setup + ninja -C build from the nntrainer root directory,
    exactly like the user's manual build:
      cd /storage_data/snap/Prachi/nntrainer/nntrainer
      meson setup build -Dthread-backend=omp -Denable-transformer=true -Denable-profile=true -Dnntr-num-threads=4
      ninja -C build
    
    Args:
        state: Pipeline state dictionary
        
    Returns:
        Updated state with causallm_so_path and causallm_exe_path set
    """
    bus.agent_status("causallm_so_builder", "running")
    
    # Get nntrainer root from state
    nntrainer_root = state.get("nntrainer_root")
    
    if not nntrainer_root:
        # Try hardcoded default - this is where the user's nntrainer lives
        # Note: nntrainer is at /storage_data/snap/Prachi/nntrainer (NOT nested)
        nntrainer_root = "/storage_data/snap/Prachi/nntrainer"
    
    if not os.path.isdir(nntrainer_root):
        bus.log(f"NNTrainer root not found: {nntrainer_root}", "error")
        bus.agent_status("causallm_so_builder", "error", "nntrainer root not found")
        state["causallm_so_built"] = False
        return state
    
    # Verify this looks like nntrainer root (has meson.build and Applications/CausalLM)
    if not os.path.exists(os.path.join(nntrainer_root, "meson.build")):
        bus.log(f"Path doesn't look like nntrainer root (no meson.build): {nntrainer_root}", "error")
        bus.agent_status("causallm_so_builder", "error", "not nntrainer root")
        state["causallm_so_built"] = False
        return state
    
    bus.log(f"Building nntrainer (with CausalLM) from: {nntrainer_root}")
    
    build_dir = os.path.join(nntrainer_root, "build")
    out_dir = state.get("out_dir")
    
    # Check if already built (look for the CausalLM executable)
    exe_dest = os.path.join(out_dir, "nntr_causallm")
    
    # FORCE REBUILD: Don't skip - always rebuild to ensure binary matches current config
    # This is important for testing - cached binaries may be from older code/config
    if os.path.exists(exe_dest):
        bus.log("Removing existing binary to force fresh rebuild...", "info")
        try:
            os.remove(exe_dest)
            bus.log(f"Removed: {exe_dest}")
        except Exception as e:
            bus.log(f"Could not remove existing binary: {e}", "warn")
    
    # if os.path.exists(exe_dest):
    #     bus.log("nntrainer+CausalLM already built -- skipping rebuild")
    #     state["causallm_so_built"] = True
    #     state["causallm_build_fixable"] = False
    #     state["causallm_exe_path"] = exe_dest
    #     if not state.get("binary_path"):
    #         state["binary_path"] = exe_dest
    #     bus.agent_status("causallm_so_builder", "done", "cached")
    #     return state
    
    # Get build configuration from state
    enable_fp16 = state.get("causallm_fp16", True)
    enable_transformer = state.get("causallm_transformer", True)
    enable_profile = state.get("causallm_enable_profile", True)
    thread_backend = state.get("causallm_thread_backend", "omp")
    nntr_num_threads = state.get("causallm_nntr_num_threads", "4")
    use_docker = state.get("use_docker_build", False)  # Default to host build
    
    # Remove existing build directory for clean build
    # If permission denied (e.g., Docker build), just reuse the build dir
    if os.path.exists(build_dir):
        try:
            bus.log(f"Removing existing build directory: {build_dir}", "info")
            shutil.rmtree(build_dir)
        except PermissionError as e:
            bus.log(f"Cannot remove build directory (Docker permissions): {e}", "warn")
            bus.log("Will attempt to rebuild in place...", "info")
    
    # Step 1: meson setup (exactly like user's manual build)
    bus.log("Configuring nntrainer build (meson setup)...")
    bus.agent_status("causallm_so_builder", "running", "meson setup")
    
    # Match user's exact command:
    # meson setup build -Dthread-backend=omp -Denable-transformer=true -Denable-profile=true -Dnntr-num-threads=4
    meson_args = ["meson", "setup", "build"]
    if enable_transformer:
        meson_args.append("-Denable-transformer=true")
    if enable_profile:
        meson_args.append("-Denable-profile=true")
    meson_args.append(f"-Dthread-backend={thread_backend}")
    meson_args.append(f"-Dnntr-num-threads={nntr_num_threads}")
    
    bus.log(f"Running: {' '.join(meson_args)} in {nntrainer_root}")
    
    try:
        result = subprocess.run(
            meson_args,
            cwd=nntrainer_root,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            full_log = (result.stdout or "") + "\n" + (result.stderr or "")
            short_error = (result.stderr or full_log).strip()[:500] or "meson setup failed"
            bus.log(f"Meson setup failed: {short_error}", "error")
            bus.agent_status("causallm_so_builder", "error", "meson setup failed")
            state["causallm_so_built"] = False
            return state
        
        if result.stdout:
            for line in result.stdout.strip().split("\n")[-10:]:
                if line.strip():
                    bus.log(f"  {line}")
        
    except subprocess.TimeoutExpired:
        bus.log("Meson setup timed out (5 min limit)", "error")
        bus.agent_status("causallm_so_builder", "error", "meson timeout")
        state["causallm_so_built"] = False
        return state
    except Exception as e:
        bus.log(f"Meson setup error: {e}", "error")
        bus.agent_status("causallm_so_builder", "error", str(e))
        state["causallm_so_built"] = False
        return state
    
    # Step 2: ninja build (exactly like user's manual build)
    bus.log("Building nntrainer (ninja -C build)...")
    bus.agent_status("causallm_so_builder", "running", "ninja build")
    
    try:
        result = subprocess.run(
            ["ninja", "-C", "build"],
            cwd=nntrainer_root,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes for full build
        )
        
        if result.returncode != 0:
            full_log = (result.stdout or "") + "\n" + (result.stderr or "")
            short_error = (result.stderr or full_log).strip()[:500] or "ninja build failed"
            bus.log(f"Ninja build failed: {short_error}", "error")
            bus.agent_status("causallm_so_builder", "error", "ninja build failed")
            state["causallm_so_built"] = False
            return state
        
        # Log final output
        if result.stdout:
            for line in result.stdout.strip().split("\n")[-10:]:
                if line.strip():
                    bus.log(f"  {line}")
        
    except subprocess.TimeoutExpired:
        bus.log("Ninja build timed out (30 min limit)", "error")
        bus.agent_status("causallm_so_builder", "error", "ninja timeout")
        state["causallm_so_built"] = False
        return state
    except Exception as e:
        bus.log(f"Ninja build error: {e}", "error")
        bus.agent_status("causallm_so_builder", "error", str(e))
        state["causallm_so_built"] = False
        return state
    
    # Step 3: Copy built files to output directory
    # The CausalLM executable is built at: build/Applications/CausalLM/nntr_causallm
    exe_src = os.path.join(build_dir, "Applications", "CausalLM", "nntr_causallm")
    so_src = os.path.join(build_dir, "Applications", "CausalLM", "libcausallm.so")
    api_so_src = os.path.join(build_dir, "Applications", "CausalLM", "libcausallm_api.so")
    
    built_count = 0
    
    # Track missing files for error reporting
    missing_files = []
    
    if os.path.exists(exe_src):
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy2(exe_src, exe_dest)
        bus.log(f"Copied nntr_causallm to {out_dir}")
        built_count += 1
    else:
        missing_files.append("nntr_causallm (executable)")
        bus.log("nntr_causallm not found after build - build may be incomplete", "error")
    
    if os.path.exists(so_src):
        so_dest = os.path.join(out_dir, "libcausallm.so")
        shutil.copy2(so_src, so_dest)
        bus.log(f"Copied libcausallm.so to {out_dir}")
        built_count += 1
    else:
        missing_files.append("libcausallm.so (shared library)")
        bus.log("libcausallm.so not found after build - build may be incomplete", "error")
    
    # libcausallm_api.so is optional (not all builds produce it)
    if os.path.exists(api_so_src):
        api_so_dest = os.path.join(out_dir, "libcausallm_api.so")
        shutil.copy2(api_so_src, api_so_dest)
        bus.log(f"Copied libcausallm_api.so to {out_dir}")
        built_count += 1
    
    # Determine build success: at minimum, we need the executable OR the .so
    if built_count >= 1:
        if missing_files:
            bus.log(f"Build partially complete: {built_count} file(s) copied, missing: {', '.join(missing_files)}", "warn")
        else:
            bus.log(f"nntrainer+CausalLM build complete: {built_count} file(s) copied")
        
        state["causallm_so_built"] = True
        state["causallm_build_fixable"] = False
        state["causallm_exe_path"] = exe_dest if os.path.exists(exe_dest) else None
        state["causallm_so_path"] = os.path.join(out_dir, "libcausallm.so") if os.path.exists(os.path.join(out_dir, "libcausallm.so")) else None
        # profiler_agent only looks at binary_path -- point it at the freshly
        # built CausalLM executable so profiling runs automatically.
        if state.get("causallm_exe_path") and not state.get("binary_path"):
            state["binary_path"] = state["causallm_exe_path"]
        bus.agent_status("causallm_so_builder", "done", f"{built_count} files")
    else:
        bus.log("Build completed but no output files found - check build logs for errors", "error")
        bus.agent_status("causallm_so_builder", "error", "no output files")
        state["causallm_so_built"] = False
        state.setdefault("errors", []).append("causallm_so_builder: no output files produced")
    
    return state
