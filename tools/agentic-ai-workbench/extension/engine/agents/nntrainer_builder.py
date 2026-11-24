# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file nntrainer_builder.py
# @brief NNTrainer Builder Agent (no LLM, no LangChain required).
# @author prachi-t17 <prachi.t@samsung.com>

"""
NNTrainer Builder Agent (no LLM, no LangChain required).

Builds nntrainer from source on Ubuntu/Linux in two phases:
  python -m agents.nntrainer_builder <repo> <prefix> prep
  python -m agents.nntrainer_builder <repo> <prefix> build

"prep"  — validate repo, check tools, detect missing apt packages,
          sync git submodules. Returns apt commands only if needed.

"build" — detect build state, run meson + ninja with auto-retry
          (up to 2 retries before reporting failure).
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from core.events import bus

logger = logging.getLogger(__name__)

AGENT_ID = "nntrainer_builder"

APT_DEPS: List[str] = [
    "meson", "ninja-build", "gcc", "g++", "pkg-config",
    "libopenblas-dev", "libiniparser-dev", "libjsoncpp-dev", "libcurl3-dev",
    "tensorflow2-lite-dev", "nnstreamer-dev", "libglib2.0-dev",
    "libgstreamer1.0-dev", "libgtest-dev", "ml-api-common-dev",
    "flatbuffers-compiler", "ml-inference-api-dev",
]

REQUIRED_TOOLS: List[str] = ["git", "meson", "ninja", "gcc"]


# ------------------------------------------------------------------ validation
def _validate_repo(repo: str) -> Optional[str]:
    """Return an error string, or None if the repo looks valid."""
    path = Path(repo)
    if not path.is_dir():
        return f"'{repo}' is not a directory"
    if not (path / "meson.build").exists():
        return "No meson.build — not the nntrainer source root"
    if not (path / ".git").is_dir():
        return "No .git — must be a real 'git clone', not a downloaded zip"
    return None


def _validate_prefix(prefix: str) -> List[str]:
    """Return a list of missing paths (empty = OK)."""
    path = Path(prefix)
    missing = []
    if not (path / "include" / "nntrainer").exists():
        missing.append("include/nntrainer")
    if not (path / "lib").exists():
        missing.append("lib/")
    return missing


# ------------------------------------------------------------------ tool checks
def _missing_tools() -> List[str]:
    missing = []
    for required_tool in REQUIRED_TOOLS:
        try:
            subprocess.run(
                [required_tool, "--version"], capture_output=True, timeout=5, check=False
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            missing.append(required_tool)
    return missing


def _missing_packages() -> List[str]:
    missing = []
    for pkg in APT_DEPS:
        proc = subprocess.run(["dpkg", "-l", pkg], capture_output=True, timeout=5, check=False)
        if proc.returncode != 0:
            missing.append(pkg)
    return missing


def _apt_commands(missing: List[str]) -> List[str]:
    deps = " ".join(missing)
    return [
        "sudo apt-add-repository -y ppa:nnstreamer/ppa",
        "(sudo apt-get update || true)  # tolerates broken mirrors",
        f"sudo apt-get install -y --no-install-recommends {deps}",
    ]


# ------------------------------------------------------------------ submodules
def _sync_submodules(repo: str) -> bool:
    for cmd in (
        ["git", "submodule", "sync"],
        ["git", "submodule", "update", "--init", "--depth", "1"],
    ):
        bus.log(f"Running: {' '.join(cmd)}")
        try:
            proc = subprocess.run(
                cmd, cwd=repo, capture_output=True, text=True, timeout=300, check=False
            )
        except subprocess.TimeoutExpired:
            bus.log(f"Timed out: {' '.join(cmd)}", "error")
            return False
        for line in proc.stdout.splitlines():
            bus.log(line)
        if proc.returncode != 0:
            bus.log(f"Command failed (exit {proc.returncode})", "error")
            for line in proc.stderr.splitlines()[:30]:
                bus.log(line, "error")
            return False
    return True


# ------------------------------------------------------------------ build state
def _build_state(repo: str) -> str:
    """Returns 'fresh' | 'meson_failed' | 'ninja_build'."""
    build = Path(repo) / "build"
    if not build.exists():
        return "fresh"
    meson_log = build / "meson-log.txt"
    if meson_log.exists() and "Build targets in project:" in meson_log.read_text(errors="ignore"):
        return "ninja_build"
    return "meson_failed"


# ------------------------------------------------------------------ build runner
def _run_with_retry(label: str, cmd: List[str], cwd: str, max_retries: int = 2) -> Tuple[bool, str]:
    """Run cmd with up to max_retries automatic retries. Returns (success, stderr)."""
    timeout = 1800 if "ninja" in label else 600
    for attempt in range(1, max_retries + 2):
        if attempt > 1:
            bus.log(f"Retry {attempt}/{max_retries + 1}: {label}")
        bus.log(f"Running: {' '.join(cmd)}")
        try:
            proc = subprocess.run(
                cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired:
            bus.log(f"{label} timed out", "warn")
            if attempt <= max_retries:
                continue
            return False, f"{label} timed out"
        for line in proc.stdout.splitlines()[:80]:
            bus.log(line)
        if proc.returncode == 0:
            return True, ""
        err = proc.stderr
        bus.log(f"{label} failed (exit {proc.returncode}, attempt {attempt})", "warn")
        for line in err.splitlines()[:30]:
            bus.log(line, "error")
        if attempt <= max_retries:
            continue
        return False, err
    return False, "max retries exceeded"


# ------------------------------------------------------------------ phases
def run_prep(repo: str, _prefix: str) -> dict:
    """Check the prerequisites (every phase takes the same repo + prefix arguments)."""
    bus.agent_status(AGENT_ID, "running", "Checking prerequisites …")

    err = _validate_repo(repo)
    if err:
        bus.log(err, "error")
        bus.agent_status(AGENT_ID, "error", err)
        return {"ok": False, "error": err}
    bus.log(f"Repo valid: {repo}")

    missing_tools = _missing_tools()
    if missing_tools:
        msg = f"Missing tools on PATH: {', '.join(missing_tools)}"
        bus.log(msg, "error")
        bus.agent_status(AGENT_ID, "error", msg)
        return {"ok": False, "error": msg}
    bus.log("Build tools available (git, meson, ninja, gcc)")

    bus.agent_status(AGENT_ID, "running", "Checking system packages …")
    missing_pkgs = _missing_packages()
    if missing_pkgs:
        bus.log(f"Missing {len(missing_pkgs)} packages: {', '.join(missing_pkgs)}")
        return {"ok": True, "apt_commands": _apt_commands(missing_pkgs), "error": None}

    bus.log("All system packages present — skipping apt step")

    bus.agent_status(AGENT_ID, "running", "Syncing git submodules …")
    if not _sync_submodules(repo):
        msg = "git submodule sync failed — see log above"
        bus.agent_status(AGENT_ID, "error", msg)
        return {"ok": False, "error": msg}
    bus.log("Submodules synced")

    return {"ok": True, "apt_commands": [], "error": None}


def run_build(repo: str, prefix: str) -> dict:
    """Build nntrainer with meson and ninja and install it into `prefix`."""
    prefix = str(Path(prefix).resolve())
    bus.agent_status(AGENT_ID, "running", "Building nntrainer …")

    state = _build_state(repo)
    bus.log(f"Build state: {state}")

    # --reconfigure is only valid on an already-configured build dir; on a
    # fresh (or never-successfully-configured) dir it errors "Directory does
    # not contain a valid build tree" on older meson. Only add it when there's
    # a configured tree to reconfigure.
    meson_cmd = ["meson", "setup", "build", f"--prefix={prefix}"]
    if state != "fresh":
        meson_cmd.append("--reconfigure")

    passed, err = _run_with_retry("meson setup", meson_cmd, repo, max_retries=2)
    if not passed:
        msg = f"meson setup failed: {err[:200]}"
        bus.log(msg, "error")
        bus.agent_status(AGENT_ID, "error", msg)
        return {"ok": False, "error": msg, "needs_user_help": True}

    passed, err = _run_with_retry(
        "ninja install",
        ["ninja", "-C", "build", "install"],
        repo,
        max_retries=2,
    )
    if not passed:
        msg = f"ninja build failed: {err[:200]}"
        bus.log(msg, "error")
        bus.agent_status(AGENT_ID, "error", msg)
        return {"ok": False, "error": msg, "needs_user_help": True}

    missing = _validate_prefix(prefix)
    if missing:
        msg = f"Build incomplete — missing: {', '.join(missing)}"
        bus.log(msg, "error")
        bus.agent_status(AGENT_ID, "error", msg)
        return {"ok": False, "error": msg}

    bus.log(f"nntrainer installed to {prefix}")
    bus.agent_status(AGENT_ID, "done", f"Installed to {prefix}")
    return {"ok": True, "prefix": prefix, "error": None}


# ------------------------------------------------------------------ entrypoint
if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[3] not in ("prep", "build"):
        print("usage: python -m agents.nntrainer_builder <repo> <prefix> <prep|build>")
        sys.exit(1)
    fn = run_prep if sys.argv[3] == "prep" else run_build
    print(json.dumps(fn(sys.argv[1], sys.argv[2])))
