"""
Integrated Builder Agent - Builds nntrainer, CausalLM, and profiles the model.

Replaces manual Docker/build steps by automating:
1. Cloning/setting up nntrainer if needed
2. Building nntrainer
3. Building CausalLM with generated component
4. Running the model
5. Collecting profiling data
"""
import os
import subprocess
import json
from pathlib import Path

from .events import bus
from .build import docker_builder


def run(state: dict) -> dict:
    """
    Run integrated build and profiling.
    
    If causallm_so_builder already built the binary, use it directly.
    Otherwise, try to build with integrated_builder's own logic.
    """
    bus.agent_status("integrated_builder", "running")

    try:
        # Check if causallm_so_builder already built the binary
        # If so, just use it for profiling - no need to rebuild
        if state.get("causallm_exe_path") and os.path.exists(state["causallm_exe_path"]):
            bus.log("Using already-built CausalLM binary from causallm_so_builder", "info")
            bus.log(f"Binary path: {state['causallm_exe_path']}", "info")
            
            # Set up for profiling
            state["causallm_build_success"] = True
            state["binary_path"] = state["causallm_exe_path"]
            
            # Run profiling with the existing binary
            profile_data = _run_model_and_profile_existing(state)
            if profile_data:
                state["profile"] = profile_data
                bus.log("Model profiling completed", "info")
                bus.agent_status("integrated_builder", "done", "profile success")
            else:
                bus.log("Model profiling failed or returned no data", "warn")
                bus.agent_status("integrated_builder", "done", "profile failed")
            
            return state
        
        # No existing binary - fall back to building
        nntrainer_root = _setup_nntrainer(state)
        if not nntrainer_root:
            bus.log("Failed to setup nntrainer", "error")
            state.setdefault("errors", []).append("nntrainer setup failed")
            return state

        state["nntrainer_root"] = nntrainer_root
        bus.log(f"nntrainer ready at: {nntrainer_root}")

        # Step 2: Build nntrainer with CausalLM using meson + ninja
        build_success = _build_causallm(state, nntrainer_root)
        if not build_success:
            bus.log("nntrainer build failed", "error")
            state.setdefault("errors", []).append("nntrainer build failed")
            state["causallm_build_success"] = False
            bus.agent_status("integrated_builder", "error", "build failed")
            return state

        state["causallm_build_success"] = True
        bus.log("nntrainer built successfully", "info")

        # Step 3: Run the model and collect profiling data
        # Note: _run_model_and_profile now uses nntrainer_root, not causallm_root
        profile_data = _run_model_and_profile(state, nntrainer_root)
        if profile_data:
            state["profile"] = profile_data
            bus.log("Model profiling completed", "info")
            bus.agent_status("integrated_builder", "done", "build + profile success")
        else:
            bus.log("Model profiling failed or returned no data", "warn")
            bus.agent_status("integrated_builder", "done", "build success, profile failed")

        return state

    except Exception as e:
        bus.log(f"Integrated builder error: {e}", "error")
        state.setdefault("errors", []).append(f"integrated_builder: {e}")
        bus.agent_status("integrated_builder", "error", str(e))
        return state


def _setup_nntrainer(state: dict) -> str:
    """
    Use the existing nntrainer installation - NO CLONING.
    
    The nntrainer source is expected to be at the nntrainer_root from state,
    which defaults to /storage_data/snap/Prachi/nntrainer/nntrainer/
    """
    bus.log("Using existing nntrainer installation...", "info")
    
    # Use nntrainer_root from state (set by orchestrator)
    nntrainer_root = state.get("nntrainer_root")
    if nntrainer_root and os.path.isdir(nntrainer_root):
        bus.log(f"Using nntrainer at: {nntrainer_root}", "info")
        return nntrainer_root
    
    # Fallback: try nntrainer_path from state
    nntrainer_path = state.get("nntrainer_path")
    if nntrainer_path and os.path.isdir(nntrainer_path):
        bus.log(f"Using nntrainer at: {nntrainer_path}", "info")
        return nntrainer_path
    
    # Last resort: hardcoded default path
    default_path = "/storage_data/snap/Prachi/nntrainer/nntrainer"
    if os.path.isdir(default_path):
        bus.log(f"Using default nntrainer at: {default_path}", "info")
        return default_path
    
    bus.log("nntrainer not found at any expected location", "error")
    return None


def _check_nntrainer_installation(root: str) -> bool:
    """Check if nntrainer is installed at given root."""
    root = Path(root)
    # Check for nntrainer headers
    include_dir = root / "include" / "nntrainer"
    if not include_dir.exists():
        return False

    # Check for main library in lib or lib64
    lib_dir = root / "lib"
    lib64_dir = root / "lib64"
    has_lib = (lib_dir.exists() and any(f.name.startswith("libnntrainer") for f in lib_dir.glob("*"))) or \
              (lib64_dir.exists() and any(f.name.startswith("libnntrainer") for f in lib64_dir.glob("*")))

    return has_lib


def _clone_and_build_nntrainer(state: dict) -> str:
    """Clone and build nntrainer from source using Docker."""
    out_dir = Path(state["out_dir"])
    nntrainer_dir = out_dir / "nntrainer_build"
    install_prefix = out_dir / "nntrainer_install"

    bus.log(f"Cloning nntrainer to {nntrainer_dir}...", "info")

    if not nntrainer_dir.exists():
        cmd = [
            "git", "clone",
            "https://github.com/nnstreamer/nntrainer.git",
            str(nntrainer_dir)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            bus.log(f"Failed to clone nntrainer: {result.stderr}", "error")
            return None

    # Sync git submodules
    bus.log("Syncing git submodules...", "info")
    try:
        subprocess.run(
            ["git", "submodule", "sync"],
            cwd=str(nntrainer_dir),
            check=True,
            capture_output=True,
            timeout=60
        )
        subprocess.run(
            ["git", "submodule", "update", "--init", "--depth", "1"],
            cwd=str(nntrainer_dir),
            check=True,
            capture_output=True,
            timeout=300
        )
        bus.log("Git submodules synced", "info")
    except subprocess.CalledProcessError as e:
        bus.log(f"Submodule sync failed: {e.stderr.decode() if isinstance(e.stderr, bytes) else str(e)}", "error")
        return None

    # Build nntrainer using Docker. Reuses docker_builder's build-if-missing
    # image + scoped mounts (only nntrainer_dir and install_prefix -- not
    # the whole filesystem, and no hardcoded personal image name).
    install_prefix.mkdir(parents=True, exist_ok=True)

    bus.log("Building nntrainer with Docker...", "info")
    ok = docker_builder.build_nntrainer_in_docker(
        str(nntrainer_dir),
        str(install_prefix),
        log_fn=lambda m, lvl="info": bus.log(m, lvl),
    )

    if not ok:
        bus.log("Docker build failed", "error")
        return None

    bus.log("nntrainer built and installed successfully in Docker", "info")
    return str(install_prefix)


def _build_causallm(state: dict, nntrainer_root: str) -> bool:
    """
    Build nntrainer with CausalLM using meson + ninja (same as manual build).
    
    This runs meson setup + ninja -C build from the nntrainer root directory,
    exactly like the user's manual build:
      cd /storage_data/snap/Prachi/nntrainer
      meson setup build -Denable-transformer=true -Denable-profile=true
      ninja -C build
    
    Args:
        state: Pipeline state dictionary
        nntrainer_root: Path to nntrainer source repository
        
    Returns:
        True if build succeeded, False otherwise
    """
    if not nntrainer_root:
        nntrainer_root = "/storage_data/snap/Prachi/nntrainer"
    
    if not os.path.isdir(nntrainer_root):
        bus.log(f"NNTrainer root not found: {nntrainer_root}", "error")
        return False
    
    if not os.path.exists(os.path.join(nntrainer_root, "meson.build")):
        bus.log(f"Path doesn't look like nntrainer root (no meson.build): {nntrainer_root}", "error")
        return False
    
    bus.log(f"Building nntrainer (with CausalLM) from: {nntrainer_root}")
    
    build_dir = os.path.join(nntrainer_root, "build")
    
    # Remove existing build directory for clean rebuild
    if os.path.exists(build_dir):
        bus.log(f"Removing existing build directory: {build_dir}", "info")
        import shutil
        shutil.rmtree(build_dir)
    
    # Step 1: meson setup
    bus.log("Configuring nntrainer build (meson setup)...")
    
    meson_args = ["meson", "setup", "build"]
    meson_args.append("-Denable-transformer=true")
    meson_args.append("-Denable-profile=true")
    meson_args.append("-Dthread-backend=omp")
    meson_args.append("-Dnntr-num-threads=4")
    
    if state.get("use_docker_build", True):
        # Build inside Docker container
        return _build_nntrainer_in_docker(nntrainer_root, meson_args)
    
    # Build on host
    try:
        result = subprocess.run(
            meson_args,
            cwd=nntrainer_root,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            bus.log(f"Meson setup failed: {result.stderr[:500]}", "error")
            return False
        
        if result.stdout:
            for line in result.stdout.strip().split("\n")[-10:]:
                if line.strip():
                    bus.log(f"  {line}")
        
    except subprocess.TimeoutExpired:
        bus.log("Meson setup timed out (5 min limit)", "error")
        return False
    except Exception as e:
        bus.log(f"Meson setup error: {e}", "error")
        return False
    
    # Step 2: ninja build
    bus.log("Building nntrainer (ninja -C build)...")
    
    try:
        result = subprocess.run(
            ["ninja", "-C", "build"],
            cwd=nntrainer_root,
            capture_output=True,
            text=True,
            timeout=1800
        )
        
        if result.returncode != 0:
            bus.log(f"Ninja build failed: {result.stderr[:500]}", "error")
            return False
        
        if result.stdout:
            for line in result.stdout.strip().split("\n")[-10:]:
                if line.strip():
                    bus.log(f"  {line}")
        
    except subprocess.TimeoutExpired:
        bus.log("Ninja build timed out (30 min limit)", "error")
        return False
    except Exception as e:
        bus.log(f"Ninja build error: {e}", "error")
        return False
    
    bus.log("nntrainer built successfully", "info")
    return True


def _build_nntrainer_in_docker(nntrainer_root: str, meson_args: list) -> bool:
    """
    Build nntrainer inside the shared Docker build image,
    mounting only the nntrainer repository.
    """
    container_nntrainer = "/workspace/nntrainer"
    
    command = (
        f"git submodule sync && git submodule update --init --depth 1 && "
        f"{' '.join(meson_args)} && "
        f"ninja -C build"
    )
    
    result = docker_builder.run_in_container(
        command=command,
        mounts={
            str(nntrainer_root): container_nntrainer,
        },
        workdir=container_nntrainer,
        log_fn=lambda m, lvl="info": bus.log(m, lvl),
        timeout=1800,
    )
    
    if not result["success"]:
        bus.log(f"nntrainer build failed (Docker): {result['stderr'][:500]}", "error")
        return False
    
    bus.log("nntrainer built successfully (Docker)", "info")
    return True


def _setup_library_path(state: dict) -> dict:
    """Set up LD_LIBRARY_PATH environment for running CausalLM binaries."""
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

    return env


def _run_model_and_profile_existing(state: dict) -> dict:
    """
    Run the already-built CausalLM binary and collect profiling data.

    The binary was built by causallm_so_builder at state["causallm_exe_path"].
    """
    model_binary = state.get("causallm_exe_path") or state.get("binary_path")

    if not model_binary or not os.path.exists(model_binary):
        bus.log(f"Model binary not found: {model_binary}", "warn")
        return None

    bus.log(f"Running model: {model_binary}", "info")

    try:
        # Set up environment with proper library paths
        env = _setup_library_path(state)

        # Prepare command: binary expects model directory as first positional argument
        cmd = [str(model_binary)]
        out_dir = state.get("out_dir", "")
        model_dir = os.path.join(out_dir, "run_model")

        bus.log(f"out_dir from state: {out_dir}", "debug")
        bus.log(f"Checking for model directory: {model_dir}", "debug")

        if os.path.isdir(model_dir):
            cmd.append(model_dir)
            bus.log(f"Using model directory: {model_dir}", "info")
        else:
            bus.log(f"Model directory not found at {model_dir} -- binary will fail without it", "error")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )

        if result.returncode != 0:
            bus.log(f"Model execution failed: {result.stderr[:500]}", "error")
            return None

        # Parse profiling output
        profile_data = _parse_profile_output(result.stdout, result.stderr)

        return profile_data

    except subprocess.TimeoutExpired:
        bus.log("Model execution timed out", "error")
        return None
    except Exception as e:
        bus.log(f"Error running model: {e}", "error")
        return None


def _run_model_and_profile(state: dict, nntrainer_root: str) -> dict:
    """
    Run the built nntr_causallm model and collect profiling data.
    
    Args:
        state: Pipeline state dictionary
        nntrainer_root: Path to nntrainer source (where build/ directory is)
    """
    # The nntr_causallm binary is built at: build/Applications/CausalLM/nntr_causallm
    build_dir = Path(nntrainer_root) / "build"
    model_binary = build_dir / "Applications" / "CausalLM" / "nntr_causallm"

    if not model_binary.exists():
        bus.log(f"nntr_causallm binary not found at {model_binary}", "warn")
        # Try fallback: maybe it's in the out_dir
        out_dir = state.get("out_dir", "")
        if out_dir:
            model_binary = Path(out_dir) / "nntr_causallm"
            if model_binary.exists():
                bus.log(f"Using nntr_causallm from output dir: {model_binary}", "info")
            else:
                bus.log(f"nntr_causallm not found in output dir either", "error")
                return None
        else:
            return None

    bus.log(f"Running model: {model_binary}", "info")

    try:
        # Set up environment with proper library paths
        env = _setup_library_path(state)

        # Prepare command: binary expects model directory as first positional argument
        cmd = [str(model_binary)]
        out_dir = state.get("out_dir", "")
        model_dir = os.path.join(out_dir, "run_model")

        bus.log(f"out_dir from state: {out_dir}", "debug")
        bus.log(f"Checking for model directory: {model_dir}", "debug")

        if os.path.isdir(model_dir):
            cmd.append(model_dir)
            bus.log(f"Using model directory: {model_dir}", "info")
        else:
            bus.log(f"Model directory not found at {model_dir} -- binary will fail without it", "error")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(build_dir),
            env=env,
        )

        if result.returncode != 0:
            bus.log(f"Model execution failed: {result.stderr[:500]}", "error")
            return None

        # Parse profiling output
        profile_data = _parse_profile_output(result.stdout, result.stderr)

        return profile_data

    except subprocess.TimeoutExpired:
        bus.log("Model execution timed out", "error")
        return None
    except Exception as e:
        bus.log(f"Error running model: {e}", "error")
        return None


def _parse_profile_output(stdout: str, stderr: str) -> dict:
    """Parse profiling data from model output."""
    output = stdout + "\n" + stderr

    profile_data = {
        "ran": True,
        "output": output[:1000],  # Store first 1000 chars
        "metrics": {},
    }

    # Try to extract key metrics
    lines = output.split("\n")
    for line in lines:
        if "latency" in line.lower() or "time" in line.lower():
            profile_data["output"] += "\n" + line

        # Look for pattern: "Operation: XXms"
        if "ms" in line.lower():
            try:
                parts = line.split()
                for i, part in enumerate(parts):
                    if "ms" in part.lower() and i > 0:
                        metric_name = parts[i-1]
                        metric_value = float(part.replace("ms", "").strip())
                        profile_data["metrics"][metric_name] = metric_value
            except (ValueError, IndexError):
                pass

    bus.log(f"Profiling data collected: {len(profile_data['metrics'])} metrics", "info")
    return profile_data
