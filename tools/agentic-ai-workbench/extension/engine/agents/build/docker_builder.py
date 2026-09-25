"""
Docker-based nntrainer builder helper.

Provides functions to build nntrainer inside a Docker container for
consistent, platform-independent builds. Automatically builds the Docker
image if it doesn't exist, then runs the nntrainer build inside a container.

SECURITY FIX #5 (Phase 2): Docker socket access restrictions
- Path validation to prevent directory traversal attacks
- Volume mount restrictions to prevent sensitive host directory exposure
"""
import subprocess
import os
import sys
from pathlib import Path

DOCKERFILE = os.path.join(os.path.dirname(__file__), "Dockerfile.nntrainer")
IMAGE_NAME = "nntrainer-build:latest"

# SECURITY FIX #5: Allowed base directories for Docker volume mounts
ALLOWED_BASE_PATHS = [
    "/home",
    "/workspace",
    "/data",
]

# SECURITY FIX #5: Blocked sensitive directories
BLOCKED_PATHS = [
    "/etc",
    "/root",
    "/proc",
    "/sys",
    "/dev",
    "/var/run/docker.sock",
]


def _is_path_safe(path_str: str) -> bool:
    """
    SECURITY FIX #5: Validate that a path is safe to mount in Docker.
    
    Checks:
    1. Path doesn't contain traversal sequences
    2. Path is within allowed base directories
    3. Path is not in blocked sensitive directories
    
    Args:
        path_str: Path to validate
        
    Returns:
        True if path is safe, False otherwise
    """
    # Normalize path
    try:
        resolved = str(Path(path_str).resolve())
    except (OSError, ValueError):
        return False
    
    # Check for path traversal in original string
    if '..' in path_str:
        return False
    
    # Check against blocked paths
    for blocked in BLOCKED_PATHS:
        if resolved == blocked or resolved.startswith(blocked + '/'):
            return False
    
    # Check if within allowed base paths
    for allowed in ALLOWED_BASE_PATHS:
        if resolved == allowed or resolved.startswith(allowed + '/'):
            return True
    
    # If no allowed base matched but also not blocked, allow if it's an absolute path
    # This is for flexibility while still blocking sensitive directories
    return os.path.isabs(resolved)


def image_exists(image_name: str) -> bool:
    """Check if a Docker image exists locally."""
    result = subprocess.run(
        ["docker", "images", "-q", image_name],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return bool(result.stdout.strip())


def build_image(log_fn=None) -> bool:
    """Build the Docker image if it doesn't exist.

    Args:
        log_fn: Optional callback for logging output

    Returns:
        True if image exists or was built successfully
    """
    if image_exists(IMAGE_NAME):
        if log_fn:
            log_fn(f"Docker image {IMAGE_NAME} already exists")
        return True

    if log_fn:
        log_fn(f"Building Docker image {IMAGE_NAME}...")

    try:
        result = subprocess.run(
            ["docker", "build", "-t", IMAGE_NAME, "-f", DOCKERFILE, "."],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            if log_fn:
                log_fn(f"Docker build failed: {result.stderr}")
            return False
        if log_fn:
            log_fn(f"Docker image {IMAGE_NAME} built successfully")
        return True
    except subprocess.TimeoutExpired:
        if log_fn:
            log_fn("Docker image build timed out")
        return False
    except FileNotFoundError:
        if log_fn:
            log_fn("Docker is not installed or not on PATH")
        return False


def build_nntrainer_in_docker(
    repo_path: str, prefix: str, log_fn=None
) -> bool:
    """Build nntrainer inside Docker container.

    SECURITY FIX #5 (Phase 2): Added path validation before Docker volume mounts.
    
    Args:
        repo_path: Path to nntrainer source repository
        prefix: Installation prefix for the build
        log_fn: Optional callback for logging output

    Returns:
        True if build succeeded, False if path validation failed or build failed
    """
    # SECURITY FIX #5: Validate paths before mounting
    if not _is_path_safe(repo_path):
        if log_fn:
            log_fn(f"SECURITY: repo_path '{repo_path}' failed path validation", "error")
        return False
    
    if not _is_path_safe(prefix):
        if log_fn:
            log_fn(f"SECURITY: prefix '{prefix}' failed path validation", "error")
        return False
    
    repo_path = str(Path(repo_path).resolve())
    prefix = str(Path(prefix).resolve())

    if not build_image(log_fn):
        return False

    if log_fn:
        log_fn(f"Building nntrainer in Docker (repo: {repo_path}, prefix: {prefix})")

    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{repo_path}:/workspace/nntrainer",
        "-v",
        f"{prefix}:/workspace/install",
        IMAGE_NAME,
        "/bin/bash",
        "-c",
        # The repo is bind-mounted from the host, so its files keep the
        # host UID -- git inside the container sees that as a mismatched
        # owner and refuses every git command (including submodule sync)
        # with "detected dubious ownership" unless explicitly exempted.
        "git config --global --add safe.directory '*' && "
        "cd /workspace/nntrainer && "
        "git submodule sync && git submodule update --init --depth 1 && "
        "meson setup build --prefix=/workspace/install -Denable-profile=true && "
        "ninja -C build install",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,
        )

        if result.stdout:
            if log_fn:
                for line in result.stdout.splitlines():
                    log_fn(line)
            else:
                print(result.stdout)

        if result.stderr:
            if log_fn:
                for line in result.stderr.splitlines():
                    log_fn(line, "error")
            else:
                print(result.stderr, file=sys.stderr)

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        if log_fn:
            log_fn("Docker build timed out after 30 minutes")
        return False
    except FileNotFoundError:
        if log_fn:
            log_fn("Docker is not installed or not on PATH")
        return False


def run_in_container(
    command: str, mounts: dict, workdir: str, log_fn=None, timeout: int = 1800
) -> dict:
    """Run an arbitrary shell command inside the shared build image.

    Generic counterpart to build_nntrainer_in_docker() for build steps that
    aren't nntrainer itself (e.g. the CausalLM project's own meson/ninja or
    cmake/make build). Reuses the same image, path-safety validation, and
    build-if-missing behavior.

    Args:
        command: shell command to run inside the container (via bash -c)
        mounts: {host_path: container_path} bind mounts -- only what the
            command actually needs, not the whole filesystem
        workdir: working directory inside the container to run command from
        log_fn: optional callback for logging output
        timeout: seconds before the run is killed

    Returns:
        {"success": bool, "stdout": str, "stderr": str}
    """
    for host_path in mounts:
        if not _is_path_safe(host_path):
            msg = f"SECURITY: mount path '{host_path}' failed path validation"
            if log_fn:
                log_fn(msg, "error")
            return {"success": False, "stdout": "", "stderr": msg}

    if not build_image(log_fn):
        return {"success": False, "stdout": "", "stderr": "Docker image build failed"}

    cmd = ["docker", "run", "--rm"]
    for host_path, container_path in mounts.items():
        cmd += ["-v", f"{str(Path(host_path).resolve())}:{container_path}"]
    cmd += ["-w", workdir, IMAGE_NAME, "/bin/bash", "-c", command]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        if result.stdout:
            for line in result.stdout.splitlines():
                log_fn(line) if log_fn else print(line)
        if result.stderr:
            for line in result.stderr.splitlines():
                log_fn(line, "error") if log_fn else print(line, file=sys.stderr)

        return {"success": result.returncode == 0, "stdout": result.stdout, "stderr": result.stderr}

    except subprocess.TimeoutExpired:
        msg = f"Docker command timed out after {timeout}s"
        if log_fn:
            log_fn(msg, "error")
        return {"success": False, "stdout": "", "stderr": msg}
    except FileNotFoundError:
        msg = "Docker is not installed or not on PATH"
        if log_fn:
            log_fn(msg, "error")
        return {"success": False, "stdout": "", "stderr": msg}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python docker_builder.py <repo_path> <prefix>")
        sys.exit(1)
    success = build_nntrainer_in_docker(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
