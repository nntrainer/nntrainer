# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file verify.py
# @brief Verify: build the generated model, run it, and compare it with Hugging Face.
# @author prachi-t17 <prachi.t@samsung.com>

# This module holds one whole workbench step, kept together on purpose.
# pylint: disable=too-many-lines

"""
Verify: build the generated model, run it, and compare
it with Hugging Face. A pass is what turns approved code into *verified*
bank entries, the only kind the workbench reuses.

    1. fixture   knowledge/verify_fixture.py (its own process) makes a tiny,
                 random-weight copy of the model's architecture, converts it
                 with the generated weight_converter.py and records Hugging
                 Face's prefill logits and greedy tokens.
    2. harness   unittest_<slug>_workbench.cpp binds the generated CausalLM
                 class to upstream's runFp32DifferentialChecks
                 (test/unittest/models/causallm_test_utils.cpp), so the pass
                 rule is upstream's: fp32 atol 0.01 on the logits + the tokens.
    3. build     compiled and linked with the exact rules, flags and
                 libraries of the workbench's nntrainer build
                 (tools/agentic-ai-workbench/nb/build.ninja): the compile
                 flags of a libcausallm model source, the link line of
                 nntr_causallm, plus gtest.
    4. run       the harness against the fixture; the result goes to
                 review.json ("verify") and the bank (mark_verified /
                 mark_failed).

Nothing is written outside the workbench folder: the build is configured
with enable-test=false (tests on would make meson re-extract upstream's
fixtures into test/), upstream sources are only read, and every output lands
in generated/<slug>/verify/ or the build folder.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from . import codegen          # codegen and preflight import this module too; the three
from . import compile_check    # only use each other at call time, never while loading
from . import docker_host
from . import preflight
from .code_bank import CodeBank, workbench_dir
from .docker_host import DockerHost, wait_streaming

FIXTURE_SCRIPT = Path(__file__).with_name("verify_fixture.py")
ATOL_FP32 = 0.01
MESON_VERSION = "1.7.2"
MESON_OPTIONS = [
    "-Denable-transformer=true",
    "-Denable-test=false",
    "-Denable-app=true",
    "-Denable-fp16=false",
    "-Dopenblas-num-threads=1",
    # prepare_encoder puts json.hpp in the build root; ninja compiles from there.
    "-Dcpp_args=-I.",
]
LINK_TARGET = "Applications/CausalLM/nntr_causallm"
Log = Callable[[str], None]


class VerifyError(RuntimeError):
    """Verify could not run, or its result could not be read."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ ninja
def _expand(template: str, values: Dict[str, str]) -> str:
    """Ninja's $-syntax: $$ $<space> $: escapes, $name and ${name} references."""
    out, i = [], 0
    while i < len(template):
        char = template[i]
        if char == "$" and i + 1 < len(template):
            next_char = template[i + 1]
            if next_char in " :$":
                out.append(next_char)
                i += 2
                continue
            if next_char == "{":
                j = template.index("}", i)
                out.append(values.get(template[i + 2:j], ""))
                i = j + 1
                continue
            match = re.match(r"[A-Za-z0-9_-]+", template[i + 1:])
            if match:
                out.append(values.get(match.group(0), ""))
                i += 1 + len(match.group(0))
                continue
        out.append(char)
        i += 1
    return "".join(out)


# pylint: disable-next=too-few-public-methods
class _Build:
    # pylint: disable-next=too-many-arguments
    def __init__(self, outs, implicit_outs, rule, ins, implicit, order, variables):
        self.outs, self.implicit_outs, self.rule = outs, implicit_outs, rule
        self.ins, self.implicit, self.order, self.vars = ins, implicit, order, variables


class Ninja:
    """Just enough of build.ninja: rules and build statements with their variables."""

    def __init__(self, path: Path):
        self.rules: Dict[str, Dict[str, str]] = {}
        self.builds: List[_Build] = []
        text = (
            Path(path)
            .read_text(encoding="utf-8", errors="replace")
            .replace("$\r\n", "")
            .replace("$\n", "")
        )
        scratch: Dict[str, str] = {}            # variables outside a rule / build block are ignored
        current = scratch
        for line in text.splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if line[0] in " \t":
                if "=" in line:
                    k, _, val = line.strip().partition("=")
                    current[k.strip()] = val.strip()
                continue
            current = scratch
            if line.startswith("rule "):
                current = self.rules.setdefault(line[5:].strip(), {})
            elif line.startswith("build "):
                builder = self._parse_build(line[6:])
                if builder:
                    self.builds.append(builder)
                    current = builder.vars
        for builder in self.builds:            # build-level variables are evaluated once
            builder.vars = {k: _expand(v, {}) for k, v in builder.vars.items()}

    @staticmethod
    def _parse_build(text: str) -> Optional[_Build]:
        toks: List[str] = []
        cur: List[str] = []
        colon_at = -1
        i = 0
        while i < len(text):
            char = text[i]
            if char == "$" and i + 1 < len(text):
                cur.append(text[i + 1] if text[i + 1] in " :$" else text[i:i + 2])
                i += 2
                continue
            if char == " ":
                if cur:
                    toks.append("".join(cur))
                    cur = []
            elif char == ":" and colon_at < 0:
                if cur:
                    toks.append("".join(cur))
                    cur = []
                colon_at = len(toks)
            else:
                cur.append(char)
            i += 1
        if cur:
            toks.append("".join(cur))
        if colon_at < 0 or colon_at >= len(toks):
            return None
        out_part, rest = toks[:colon_at], toks[colon_at:]
        outs = out_part[:out_part.index("|")] if "|" in out_part else out_part
        implicit_outs = out_part[out_part.index("|") + 1:] if "|" in out_part else []
        rule, rest = rest[0], rest[1:]
        order = rest[rest.index("||") + 1:] if "||" in rest else []
        rest = rest[:rest.index("||")] if "||" in rest else rest
        implicit = rest[rest.index("|") + 1:] if "|" in rest else []
        ins = rest[:rest.index("|")] if "|" in rest else rest
        return _Build(outs, implicit_outs, rule, ins, implicit, order, {})

    def outputs(self) -> set:
        """Every output of the build."""
        return {o for b in self.builds for o in b.outs + b.implicit_outs}

    def link_statement(self) -> Optional[_Build]:
        """The build statement that links nntr_causallm, or None."""
        for builder in self.builds:
            for out in builder.outs:
                if out.replace("\\", "/") in (LINK_TARGET, LINK_TARGET + ".exe"):
                    return builder
        return None

    def model_compile_statement(self) -> Optional[_Build]:
        """A libcausallm compile of an upstream model source: the flags generated model code
        needs."""
        for builder in self.builds:
            if (
                builder.rule.startswith("cpp_COMPILER")
                and builder.ins
                and "libcausallm" in builder.outs[0]
                and "/Applications/CausalLM/models/" in builder.ins[0].replace("\\", "/")
            ):
                return builder
        return None


# ------------------------------------------------------------- toolchain
def resolve_root(nntrainer_root: Optional[os.PathLike]) -> Path:
    """The nntrainer checkout: the argument, NNTRAINER_ROOT, or the one the workbench is in."""
    return Path(
        nntrainer_root
        or os.environ.get("NNTRAINER_ROOT")
        or (workbench_dir() or Path(".")).parent.parent
    )


def build_tools_dir() -> Optional[Path]:
    """tools/agentic-ai-workbench/.buildenv: a venv with meson + ninja for the workbench build."""
    wb_dir = workbench_dir()
    return wb_dir / ".buildenv" if wb_dir else None


def tool_env() -> Dict[str, str]:
    """Environment for meson, ninja, the compiler and the harness: MSVC's
    (vcvars) on Windows outside a Developer prompt, Windows' own tar.exe before
    any Git tar (meson extracts archives with `cmd /C tar`), the build venv's
    meson/ninja, and cargo for the Windows tokenizer build."""
    base = None
    if os.name == "nt":
        base = compile_check.env_for("cl", "msvc")
    env = dict(base if base is not None else os.environ)
    if os.name == "nt":
        env = {k.upper(): v for k, v in env.items()}
    front = []
    if os.name == "nt":
        front.append(os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32"))
        front.append(str(Path.home() / ".cargo" / "bin"))
    venv = build_tools_dir()
    if venv:
        front.append(str(venv / ("Scripts" if os.name == "nt" else "bin")))
    key = "PATH"
    env[key] = os.pathsep.join(front + [env.get(key, "")])
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _which(name: str, env: Dict[str, str]) -> Optional[str]:
    return shutil.which(name, path=env.get("PATH"))


def run_command(
    cmd, cwd: Path, env: Dict[str, str], timeout: int, log: Optional[Log] = None
) -> Tuple[int, str]:
    """Run, streaming output lines to `log`; never raises for a failing command."""
    lines: List[str] = []

    def on_line(line: str) -> None:
        if line.startswith("Note: including file:"):          # MSVC /showIncludes noise
            return
        lines.append(line)
        if log:
            log(line)

    try:
        with subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        ) as proc:
            if not wait_streaming(proc, timeout, on_line):
                lines.append(f"timed out after {timeout}s")
                return 124, "\n".join(lines)
    except OSError as exc:
        return 127, f"could not start {cmd if isinstance(cmd, str) else cmd[0]}: {exc}"
    return proc.returncode, "\n".join(lines)


def _resolve_command(cmd: str, env: Dict[str, str]):
    """A rule's command line with its program resolved on `env`'s PATH
    (CreateProcess would search the parent's PATH, not the one passed)."""
    match = re.match(r'\s*(?:"([^"]+)"|(\S+))', cmd)
    if not match:
        return cmd
    exe = match.group(1) or match.group(2)
    full = _which(exe, env) or exe
    if os.name == "nt":
        return f'"{full}"' + cmd[match.end():]
    return [full] + shlex.split(cmd[match.end():])


# --------------------------------------------------------------- executors
class NativeExecutor:
    """Builds and runs on this machine (MSVC on Windows, gcc/clang elsewhere)."""
    kind = "native"

    def __init__(self, root: Path):
        self.root = Path(root)
        self.windows = os.name == "nt"
        self._env: Optional[Dict[str, str]] = None

    @property
    def env(self) -> Dict[str, str]:
        """The build environment (worked out once)."""
        if self._env is None:
            self._env = tool_env()
        return self._env

    @property
    def build_dir(self) -> Path:
        """The workbench's native build folder."""
        return compile_check.build_dir(self.root)

    def describe(self) -> str:
        """The build host, for the log."""
        return "native MSVC" if self.windows else "native"

    @staticmethod
    def path(part: os.PathLike) -> str:
        """A host path as the build sees it (the same path natively)."""
        return str(part)

    @staticmethod
    def quote(text: str) -> str:
        """Quote an argument for this computer's shell."""
        return _q(text)

    def lib_env(self, dirs: List[str]) -> Dict[str, str]:
        """Where the runtime finds the build's shared libraries."""
        key = "PATH" if self.windows else "LD_LIBRARY_PATH"
        return {key: os.pathsep.join([str(d) for d in dirs] + [self.env.get(key, "")])}

    # The build is on this machine: nothing to copy in or out.
    def ninja_file(self) -> Path:
        """The build's build.ninja."""
        return self.build_dir / "build.ninja"

    def missing(self, rels: List[str]) -> List[str]:
        """Which of these build outputs are missing."""
        return [r for r in rels if not (self.build_dir / r).exists()]

    @staticmethod
    def exists(host_path: os.PathLike) -> bool:
        """True when the path exists."""
        return Path(host_path).exists()

    def lib_paths(self) -> List[str]:
        """The folders that hold the build's shared libraries."""
        return [str(d) for d in lib_dirs(self.build_dir)]

    # Natively the build reads and writes the checkout itself: nothing to prepare,
    # push or pull (the Docker executor syncs its copy in these).
    @staticmethod
    def prepare(_log: Optional[Log] = None) -> Tuple[bool, str]:
        """Nothing to prepare natively."""
        return True, ""

    def after_build(self) -> None:
        """Nothing to do after a native build."""

    @staticmethod
    def push(_files) -> Tuple[bool, str]:
        """Nothing to copy natively: the build reads the checkout itself."""
        return True, ""

    @staticmethod
    def pull(_host_path: os.PathLike) -> Tuple[bool, str]:
        """Nothing to copy back natively."""
        return True, ""

    @staticmethod
    def live(host_path: os.PathLike) -> str:
        """A path the build writes and the host reads (the same path natively)."""
        return str(host_path)

    @staticmethod
    def mkdirs(dirs) -> None:
        """Create the folders."""
        for dir_name in dirs:
            Path(dir_name).mkdir(parents=True, exist_ok=True)

    # pylint: disable-next=too-many-arguments
    def run(self, cmd, cwd: Path, timeout: int, log: Optional[Log] = None,
            env_extra: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
        """Run a command in the build environment; returns (exit code, output)."""
        env = dict(self.env)
        env.update(env_extra or {})
        if isinstance(cmd, str):
            cmd = _resolve_command(cmd, env)
        elif cmd and not os.path.isabs(cmd[0]):
            cmd = [_which(cmd[0], env) or cmd[0], *cmd[1:]]
        return run_command(cmd, cwd, env, timeout, log)


class DockerExecutor:
    """Builds and runs in the Ubuntu build container (knowledge/docker_host.py),
    on its private copy of the checkout: host paths are translated to that
    copy, workbench files are pushed in before a build and results pulled back."""
    kind = "docker"
    windows = False

    def __init__(self, root: Path, variant: str = ""):
        self.root = Path(root)
        self.variant = variant
        self.host = DockerHost(root, variant)

    @property
    def build_dir(self) -> Path:
        """The host-side mirror of the build (build.ninja); paths under it map to the build."""
        return self.host.build_dir

    def describe(self) -> str:
        """The build host, for the log."""
        return "Docker (Ubuntu 24.04, gcc-13" + (
            ", nntrainer profiler on)" if self.variant == "prof" else ")"
        )

    def path(self, part: os.PathLike) -> str:
        """A host path as the container sees it."""
        return self.host.to_container(part)

    @staticmethod
    def quote(text: str) -> str:
        """Quote an argument for the container's shell."""
        return shlex.quote(text)

    @staticmethod
    def lib_env(dirs: List[str]) -> Dict[str, str]:
        """The environment that lets programs in the container find the libraries."""
        return {"LD_LIBRARY_PATH": ":".join(dirs)}

    def ninja_file(self) -> Path:
        """The host copy of the container build's build.ninja (copied over when it is missing)."""
        handle = self.build_dir / "build.ninja"
        if not handle.is_file():
            if docker_host.daemon_status()[0]:
                self.host.mirror_build()
        return handle

    def missing(self, rels: List[str]) -> List[str]:
        """Which of these build outputs are missing in the container."""
        return self.host.missing(rels)

    def exists(self, host_path: os.PathLike) -> bool:
        """True when the path exists in the container."""
        return self.host.exec(["test", "-e", self.path(host_path)], timeout=60)[0] == 0

    def lib_paths(self) -> List[str]:
        """The container folders that hold the build's shared libraries."""
        return self.host.lib_dirs()

    def prepare(self, log: Optional[Log] = None) -> Tuple[bool, str]:
        """Bring the container's copy of the checkout up to date."""
        return self.host.sync_checkout(log)

    def after_build(self) -> None:
        """Copy the build's ninja files back to the host."""
        self.host.mirror_build()

    def push(self, files) -> Tuple[bool, str]:
        """Copy files into the container's copy of the checkout."""
        return self.host.push(files)

    def pull(self, host_path: os.PathLike) -> Tuple[bool, str]:
        """Copy a file from the container back to the host."""
        return self.host.pull(host_path)

    def live(self, host_path: os.PathLike) -> str:
        """A workbench path as the container sees it (the bind mount)."""
        return self.host.live(host_path)

    def mkdirs(self, dirs) -> None:
        """Output folders on the host and in the container's copy."""
        dirs = list(dirs)
        for dir_name in dirs:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
        if dirs:
            self.host.exec(
                "mkdir -p " + " ".join(shlex.quote(self.path(d)) for d in dirs), timeout=60
            )

    # pylint: disable-next=too-many-arguments
    def run(self, cmd, cwd: Path, timeout: int, log: Optional[Log] = None,
            env_extra: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
        """Run a command in the container; returns (exit code, output)."""
        return self.host.exec(cmd, workdir=self.path(cwd), env=env_extra, timeout=timeout, log=log)


def select_executor(nntrainer_root: Optional[os.PathLike] = None, host: Optional[str] = None):
    """The build host: `host` or $WORKBENCH_BUILD_HOST = docker | native | auto.
    auto: a Docker build that is set up, else a native one that is, else Docker
    when it is running, else native."""
    root = resolve_root(nntrainer_root)
    host = (host or os.environ.get("WORKBENCH_BUILD_HOST") or "auto").lower()
    if host == "docker":
        return DockerExecutor(root)
    if host == "native":
        return NativeExecutor(root)
    running = docker_host.daemon_status()[0]
    docker = DockerExecutor(root)
    if running and (docker.build_dir / "build.ninja").is_file():
        return docker
    native = NativeExecutor(root)
    if (native.build_dir / "build.ninja").is_file():
        return native
    return docker if running else native


def build_status(
    nntrainer_root: Optional[os.PathLike] = None, ex=None, host: Optional[str] = None
) -> dict:
    """Whether the workbench build is set up, and which libraries it has."""
    root = resolve_root(nntrainer_root)
    ex = ex or select_executor(root, host)
    bdir = ex.build_dir
    base = {"host": ex.kind, "host_label": ex.describe(), "build_dir": str(bdir)}
    nfile = ex.ninja_file()
    if not nfile.is_file():
        return dict(
            base,
            ok=False,
            status="no_build",
            why=f"the workbench's nntrainer build ({ex.describe()}) is not set up yet: run "
            + ("'Set Up Docker Build'" if ex.kind == "docker" else "'Set Up nntrainer Build'"),
        )
    ninja = Ninja(nfile)
    link, comp = ninja.link_statement(), ninja.model_compile_statement()
    if not link or not comp:
        return dict(
            base,
            ok=False,
            status="no_build",
            why=f"{bdir} has no CausalLM targets (configure it with -Denable-transformer=true)",
        )
    outs = ninja.outputs()
    libs = [p for p in link.implicit if p in outs and not p.endswith(".pdb")]
    missing = ex.missing(libs)
    return dict(
        base,
        ok=True,
        status="ready" if not missing else "libs_missing",
        libs=libs,
        missing=missing,
        why="" if not missing else f"{len(missing)} libraries not built",
    )


def ensure_libraries(
    nntrainer_root: Optional[os.PathLike] = None,
    log: Log = print,
    timeout: int = 7200,
    ex=None,
    host: Optional[str] = None,
) -> dict:
    """Build the libraries nntr_causallm links (not the app itself: upstream's
    main.cpp does not build on Windows) with the build's ninja."""
    root = resolve_root(nntrainer_root)
    ex = ex or select_executor(root, host)
    status = build_status(root, ex=ex)
    if not status["ok"]:
        return status
    passed, detail = ex.prepare(log)             # Docker: bring its copy of the checkout up to date
    if not passed:
        return dict(status, ok=False, status="build_failed", why=detail)
    if ex.kind == "native" and not status["missing"]:
        return status
    if ex.kind == "native" and not _which("ninja", ex.env):
        return dict(
            status, ok=False, status="no_ninja", why="ninja not found: run 'Set Up nntrainer Build'"
        )
    jobs = docker_jobs(ex)
    if status["missing"]:
        log(f"Building {len(status['libs'])} nntrainer libraries ({ex.describe()}"
            + (f", {jobs} parallel jobs" if jobs else "") + "; the first build takes a while) ...")
    elif jobs:
        log(
            f"Checking the nntrainer libraries are up to date ({ex.describe()}, {jobs} parallel"
            " jobs) ..."
        )
    code, out, jobs = run_ninja(
        ex,
        Path(status["build_dir"]),
        status["libs"],
        timeout,
        log if status["missing"] else None,
        jobs=jobs,
    )
    ex.after_build()
    if code != 0:
        return dict(
            status,
            ok=False,
            status="build_failed",
            why=build_failure_why(out, code, jobs, "the nntrainer libraries did not build"),
            log=out[-6000:],
        )
    return build_status(root, ex=ex)


# Signs that the compiler was killed for lack of memory (Linux OOM killer in Docker).
_OUT_OF_MEMORY = re.compile(
    r"Killed signal terminated program|fatal error: Killed|virtual memory exhausted|"
    r"cannot allocate memory|out of memory allocating|std::bad_alloc",
    re.I,
)


def out_of_memory(text: str, code: Optional[int] = None) -> bool:
    """True when a build or run output shows that memory ran out."""
    return code == 137 or bool(_OUT_OF_MEMORY.search(text or ""))


def docker_jobs(ex) -> int:
    """Parallel compile jobs for a Docker build (0 = ninja's own default, natively)."""
    if getattr(ex, "kind", "") != "docker":
        return 0
    return docker_host.build_jobs()


# pylint: disable-next=too-many-arguments
def run_ninja(ex, build_dir: Path, targets: List[str], timeout: int, log: Optional[Log] = None,
              jobs: Optional[int] = None) -> Tuple[int, str, int]:
    """ninja in the build: in Docker with at most `jobs` parallel compiles (by
    default from Docker's memory) and, if the compiler still ran out of memory,
    once more with half the jobs. Returns (exit code, output, jobs used)."""
    jobs = docker_jobs(ex) if jobs is None else jobs

    def build_with(num_jobs: int) -> Tuple[int, str]:
        cmd = (
            ["ninja", "-C", ex.path(build_dir)]
            + (["-j", str(num_jobs)] if num_jobs else [])
            + list(targets)
        )
        return ex.run(cmd, build_dir, timeout, log)

    code, out = build_with(jobs)
    if code != 0 and jobs > 1 and out_of_memory(out, code):
        jobs = max(1, jobs // 2)
        msg = (
            f"The compiler ran out of memory in Docker; building again with {jobs} parallel"
            " job(s) ..."
        )
        (log or print)(msg)
        code, out = build_with(jobs)
    return code, out, jobs


def build_failure_why(out: str, code: int, jobs: int, default: str) -> str:
    """Why a build failed, with the memory fix when memory ran out."""
    if out_of_memory(out, code):
        return (f"the compiler ran out of memory in Docker (even with {jobs} parallel job(s)). "
                + preflight.memory_fix())
    return default


def _long_paths_enabled() -> bool:
    if os.name != "nt":
        return True
    try:
        # pylint: disable-next=import-outside-toplevel  # Windows only
        import winreg
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem"
        ) as k:
            return bool(winreg.QueryValueEx(k, "LongPathsEnabled")[0])
    except OSError:
        return False


def _paths_too_long(bdir: Path) -> int:
    """How many build outputs would exceed MAX_PATH (0 when long paths are on)."""
    if _long_paths_enabled() or not (bdir / "build.ninja").is_file():
        return 0
    base = len(str(bdir.resolve())) + 1
    return sum(1 for o in Ninja(bdir / "build.ninja").outputs() if base + len(o) > 259)


def _git_status(root: Path) -> Optional[List[str]]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.splitlines() if proc.returncode == 0 else None


# pylint: disable-next=too-many-locals,too-many-branches,too-many-return-statements
def setup_build(
    nntrainer_root: Optional[os.PathLike] = None,
    log: Log = print,
    python: Optional[str] = None,
    host: Optional[str] = None,
) -> dict:
    """One-time setup of the workbench's nntrainer build: in Docker (a copy of
    the checkout in a Docker volume, built at /work/src/build-wb; see
    knowledge/docker_host.py) or natively (tools/agentic-ai-workbench/nb,
    meson + ninja in .buildenv).
    Anything that would write outside the workbench folder (git submodules,
    Rust, compilers) is reported as missing, not done."""
    root = resolve_root(nntrainer_root)
    ex = select_executor(root, host)
    if ex.kind == "docker":
        status = docker_host.setup(root, log)
        if not status.get("ok"):
            return dict(status, host="docker", host_label=ex.describe())
        return ensure_libraries(root, log, ex=ex)
    bdir = compile_check.build_dir(root)
    env = tool_env()
    missing: List[str] = []
    if not (root / "subprojects" / "googletest" / "googletest" / "include").is_dir() or \
            (os.name == "nt" and not (root / "nntrainer-windows-resource" / "x64").is_dir()):
        missing.append(
            "git submodules: run `git submodule sync && git submodule update --init --depth 1` "
            "in the nntrainer checkout"
        )
    if os.name == "nt":
        if compile_check.env_for("cl", "msvc") is None and not os.environ.get("INCLUDE"):
            missing.append("MSVC: install Visual Studio Build Tools with the C++ workload "
                           "(winget install Microsoft.VisualStudio.2022.BuildTools)")
        tok = bdir / "tokenizers_c_win" / "target" / "release" / "tokenizers_c.lib"
        if not tok.is_file() and not _which("cargo", env):
            missing.append("Rust (builds the tokenizer library): winget install Rustlang.Rustup, "
                           "then `rustup default stable`")
    elif not (shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")):
        missing.append("a C++ compiler (g++ or clang++)")
    if missing:
        return {"ok": False, "status": "missing", "missing": missing, "build_dir": str(bdir)}

    if not (_which("meson", env) and _which("ninja", env)):
        venv = build_tools_dir()
        if venv is None:
            return {
                "ok": False,
                "status": "missing",
                "missing": ["cannot find tools/agentic-ai-workbench"],
            }
        log(f"Installing meson {MESON_VERSION} and ninja into {venv} ...")
        python_exe = python or sys.executable
        code, out = run_command([python_exe, "-m", "venv", str(venv)], root, env, 600, log)
        vpy = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if code == 0:
            code, out = run_command(
                [
                    str(vpy),
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "-q",
                    f"meson=={MESON_VERSION}",
                    "ninja",
                ],
                root,
                env,
                900,
                log,
            )
        if code != 0 or not (_which("meson", env) and _which("ninja", env)):
            return {
                "ok": False,
                "status": "missing",
                "missing": ["meson/ninja could not be installed"],
                "log": out[-4000:],
            }

    changed: List[str] = []
    if not (bdir / "build.ninja").is_file():
        before = _git_status(root)
        cmd = [_which("meson", env), "setup"]
        if os.name == "nt":
            cmd += ["--native-file", str(root / "configurations" / "windows-native.ini")]
        cmd += MESON_OPTIONS + [str(bdir)]
        log("Configuring: " + " ".join(cmd[1:]))
        code, out = run_command(cmd, root, env, 3600, log)
        after = _git_status(root)
        if before is not None and after is not None:
            changed = sorted(set(after) - set(before))
        if code != 0:
            return {
                "ok": False,
                "status": "setup_failed",
                "build_dir": str(bdir),
                "log": out[-6000:],
                "changed_outside": changed,
            }
    too_long = _paths_too_long(bdir)
    if too_long:
        return {
            "ok": False,
            "status": "path_too_long",
            "build_dir": str(bdir),
            "changed_outside": changed,
            "why": (
                f"{too_long} build outputs would exceed Windows' 260-character path limit under"
                f" {bdir}. Enable long paths (gpedit: Enable Win32 long paths, or"
                " LongPathsEnabled=1) or move the nntrainer checkout to a shorter folder."
            ),
        }
    status = ensure_libraries(root, log, ex=ex)
    status["changed_outside"] = changed
    if changed:
        log(
            "WARNING: the nntrainer build changed files in the checkout: " + "; ".join(changed[:10])
        )
    return status


# ---------------------------------------------------------------- harness
HARNESS = """// Generated by the Agentic AI Workbench (knowledge/verify.py) -- do not edit.
// Upstream's FP32 differential check (test/unittest/models/causallm_test_utils.cpp)
// of {cls} against a tiny Hugging Face fixture of {model}.
#include <causallm_test_utils.h>

#include <gtest/gtest.h>

#include "{header}"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {{

const char *kFixture = "{fixture}";

std::unique_ptr<causallm_test::TinyCausalLMRunner>
makeModel(causallm::json &cfg, causallm::json &gen, causallm::json &nntr) {{
  return std::make_unique<
    causallm_test::CausalLMTestAdapter<causallm::{cls}>>(cfg, gen, nntr);
}}

causallm_test::DifferentialModel workbenchModel() {{
  return {{kFixture, makeModel}};
}}

TEST(WorkbenchVerify, FP32MatchesHFReference) {{
  causallm_test::runFp32DifferentialChecks(workbenchModel());
}}

template <typename T> std::string toList(const std::vector<T> &v) {{
  std::ostringstream os;
  os << std::setprecision(9) << "[";
  for (size_t i = 0; i < v.size(); ++i)
    os << (i ? "," : "") << v[i];
  os << "]";
  return os.str();
}}

std::string escape(const std::string &s) {{
  std::string out;
  for (char c : s) {{
    if (c == '"' || c == '\\\\')
      out += '\\\\';
    out += (c == '\\n' || c == '\\r') ? ' ' : c;
  }}
  return out;
}}

long argmaxOf(const std::vector<float> &v) {{
  return v.empty() ? -1L
                   : static_cast<long>(std::max_element(v.begin(), v.end()) -
                                       v.begin());
}}

// Numbers for the workbench (the pass/fail verdict is the test above).
TEST(WorkbenchVerify, Report) {{
  std::ostringstream os;
  os << std::setprecision(9);
  try {{
    auto dir = causallm_test::findFixtureDir(kFixture);
    auto fix = causallm_test::loadReferenceFixture(dir);
    auto cfg = causallm::LoadJsonFile((dir / "config.json").string());
    auto gen = causallm::LoadJsonFile((dir / "generation_config.json").string());
    auto nntr = causallm::LoadJsonFile((dir / "nntr_config.json").string());
    nntr["tokenizer_file"] = (dir / "tokenizer.json").string();
    auto weights = (dir / nntr["model_file_name"].get<std::string>()).string();

    auto m = makeModel(cfg, gen, nntr);
    m->initializeModel();
    m->loadWeight(weights);
    auto got = m->prefillLogitsFromIds(fix.input_ids);
    double max_diff = 0.0;
    size_t n = std::min(got.size(), fix.reference_logits.size());
    for (size_t i = 0; i < n; ++i)
      max_diff = std::max(
        max_diff, static_cast<double>(std::fabs(got[i] - fix.reference_logits[i])));

    auto m2 = makeModel(cfg, gen, nntr);
    m2->initializeModel();
    m2->loadWeight(weights);
    auto toks = m2->greedyGenerateFromIds(fix.input_ids, fix.reference_tokens.size());

    os << "{{\\"max_abs_diff\\":" << max_diff << ",\\"n_logits\\":" << got.size()
       << ",\\"n_ref\\":" << fix.reference_logits.size()
       << ",\\"argmax_got\\":" << argmaxOf(got)
       << ",\\"argmax_ref\\":" << argmaxOf(fix.reference_logits)
       << ",\\"tokens_got\\":" << toList(toks)
       << ",\\"tokens_ref\\":" << toList(fix.reference_tokens)
       << ",\\"logits_got\\":" << toList(got) << "}}";
  }} catch (const std::exception &e) {{
    os.str("");
    os << "{{\\"error\\":\\"" << escape(e.what()) << "\\"}}";
  }}
  std::cout << "WORKBENCH_RESULT " << os.str() << std::endl;
}}

}} // namespace
"""


def header_for(review: dict, cls: str) -> Optional[Path]:
    """The generated header that defines class `cls`, or None."""
    pat = re.compile(r"\bclass\s+" + re.escape(cls) + r"\b[^;]*\{")
    for name, path in sorted(review.get("files", {}).items()):
        part = Path(path)
        if (
            name.endswith(".h")
            and part.is_file()
            and pat.search(part.read_text(encoding="utf-8", errors="replace"))
        ):
            return part
    return None


def harness_source(review: dict, fixture: str) -> str:
    """The gtest source that compares the generated model with the reference fixture."""
    cls = (review.get("classes") or {}).get("causallm")
    if not cls:
        raise VerifyError("review.json names no CausalLM class")
    header = header_for(review, cls)
    if not header:
        raise VerifyError(f"no generated header declares class {cls}")
    return HARNESS.format(
        cls=cls, model=review.get("model", ""), header=header.name, fixture=fixture
    )


# ---------------------------------------------------------------- compile
def _q(path: str) -> str:
    return f'"{path}"' if os.name == "nt" else shlex.quote(path)


# pylint: disable-next=too-many-locals,too-many-arguments
def compile_source(ninja: Ninja, bdir: Path, src: Path, obj: Path, includes: List[Path], ex,
             timeout: int) -> Tuple[bool, List[dict], str]:
    """Compile one source with the build's own rule and the flags of a libcausallm model source."""
    stmt = ninja.model_compile_statement()
    rule = ninja.rules.get(stmt.rule, {})
    args = stmt.vars.get("ARGS", "").replace('"/showIncludes"', "").replace("/showIncludes", "")
    args = args.replace(
        "-fdiagnostics-color=always", "-fdiagnostics-color=never"
    )  # parseable errors
    args = " ".join(ex.quote("-I" + ex.path(p)) for p in includes) + " " + args
    obj.parent.mkdir(parents=True, exist_ok=True)
    obj_path, src_path = ex.path(obj), ex.path(src)
    values = dict(stmt.vars, ARGS=args, out=ex.quote(obj_path), **{"in": ex.quote(src_path)},
                  DEPFILE=ex.quote(obj_path + ".d"), DEPFILE_UNQUOTED=obj_path + ".d")
    code, out = ex.run(_expand(rule.get("command", ""), values), bdir, timeout)
    errors = []
    for line in out.splitlines():
        match = compile_check.DIAG_RE.match(line.strip())
        if match and match.group("kind") in ("error", "fatal error"):
            errors.append(
                {
                    "file": os.path.basename(match.group("file")),
                    "line": int(match.group("line")),
                    "message": match.group("msg").strip(),
                }
            )
    if code != 0 and not errors:
        errors.append(
            {"file": src.name, "line": 0, "message": out.strip()[-400:] or f"exit {code}"}
        )
    return code == 0, errors, f"$ compile {src.name} (exit {code})\n{out.strip()}"


def link_extras(ex, gtest: bool) -> List[str]:
    """Libraries added to upstream's link line: gtest for the verify harness;
    on Windows synchronization.lib, which Rust's std (1.78+, inside
    tokenizers_c.lib) needs for WaitOnAddress/WakeByAddress* and upstream's
    link line predates."""
    if ex.windows:
        return (["gtest_main.lib", "gtest.lib"] if gtest else []) + ["synchronization.lib"]
    return (["-lgtest_main", "-lgtest"] if gtest else []) + ["-pthread"]


# pylint: disable-next=too-many-locals,too-many-arguments
def link_objects(ninja: Ninja, bdir: Path, objs: List[Path], exe: Path, ex, timeout: int,
          extra: Optional[List[str]] = None) -> Tuple[bool, List[dict], str]:
    """Link with nntr_causallm's rule and libraries, its objects swapped for ours."""
    stmt = ninja.link_statement()
    rule = ninja.rules.get(stmt.rule, {})
    extra = link_extras(ex, gtest=True) if extra is None else extra
    paths = [ex.quote(ex.path(o)) for o in objs]
    values = dict(
        stmt.vars,
        out=ex.quote(ex.path(exe)),
        **{"in": " ".join(paths), "in_newline": "\n".join(paths)},
    )
    link_args, dropped = _drop_absent_import_libs(values.get("LINK_ARGS", ""), bdir) if ex.windows \
        else (values.get("LINK_ARGS", ""), [])
    values["LINK_ARGS"] = link_args + "".join(" " + ex.quote(x) for x in extra)
    if rule.get("rspfile_content"):
        rsp = Path(str(exe) + ".rsp")
        rsp.write_text(_expand(rule["rspfile_content"], values), encoding="utf-8")
        prog = re.match(r'\s*(?:"([^"]+)"|(\S+))', rule.get("command", "")).groups()
        cmd = [prog[0] or prog[1], "@" + ex.path(rsp)]
    else:
        cmd = _expand(rule.get("command", ""), values)
    code, out = ex.run(cmd, bdir, timeout)
    if dropped:
        out = (
            "(left out import libraries of DLLs that export nothing: "
            + ", ".join(dropped)
            + ")\n"
            + out
        )
    errors = [{"file": exe.name, "line": 0, "message": line.strip()} for line in out.splitlines()
              if re.search(r"\berror\b|undefined reference|unresolved external", line, re.I)][:30]
    if code != 0 and not errors:
        errors.append(
            {"file": exe.name, "line": 0, "message": out.strip()[-400:] or f"exit {code}"}
        )
    return code == 0, errors, f"$ link {exe.name} (exit {code})\n{out.strip()}"


def _no_space_path(path: Path) -> str:
    """nntrainer's weight loading strips spaces from file paths, so on Windows
    the fixture folder is passed by its 8.3 short name (same folder)."""
    text = str(path)
    if " " not in text or os.name != "nt":
        return text
    try:
        buf = ctypes.create_unicode_buffer(32768)
        length = ctypes.windll.kernel32.GetShortPathNameW(text, buf, len(buf))
        return buf.value if length and " " not in buf.value else text
    except (OSError, AttributeError):
        return text


def _drop_absent_import_libs(link_args: str, bdir: Path) -> Tuple[str, List[str]]:
    """MSVC writes no import library for a DLL that exports nothing (upstream's
    causal_conv1d_layer.dll), but meson still lists X.lib on the link line.
    There is nothing to import from such a DLL, so the missing .lib is left out."""
    if os.name != "nt":
        return link_args, []
    keep, dropped = [], []
    for tok in re.findall(r'"[^"]*"|\S+', link_args):
        val = tok.strip('"')
        if val.lower().endswith(".lib"):
            path = Path(val) if os.path.isabs(val) else bdir / val
            if not path.exists() and path.with_suffix(".dll").exists():
                dropped.append(val)
                continue
        keep.append(tok)
    return " ".join(keep), dropped


def lib_dirs(bdir: Path) -> List[Path]:
    """Build folders that hold shared libraries (.dll on Windows, .so on Linux)."""
    dirs = set()
    for data, subs, files in os.walk(bdir):
        subs[:] = [
            s
            for s in subs
            if s not in ("meson-private", "meson-logs", "meson-info", "tokenizers_c_win")
            and not s.endswith(".p")
        ]
        if any(f.lower().endswith(".dll") or ".so" in f for f in files):
            dirs.add(Path(data))
    return sorted(dirs)


def _parse_result(stdout: str) -> Optional[dict]:
    for line in reversed(stdout.splitlines()):
        if line.startswith("WORKBENCH_RESULT "):
            text = re.sub(
                r"(?<![\w.])-?(?:nan|inf)(?![\w.])",
                "null",
                line[len("WORKBENCH_RESULT ") :],
                flags=re.I,
            )
            try:
                return json.loads(text)
            except ValueError:
                return {"error": "unreadable result line: " + line[:300]}
    return None


_GTEST_NEAR = re.compile(
    r"difference between got\[i\] and ref\[i\] is ([-+\w.]+).*?got\[i\] evaluates to ([-+\w.]+)"
    r".*?ref\[i\] evaluates to ([-+\w.]+).*?index (\d+)",
    re.S,
)


def summarize_failure(text: str) -> str:
    """One readable line from a gtest failure, such as
    'logit 7: nntrainer 0.163, Hugging Face 0.117 (diff 0.0464)'."""
    match = _GTEST_NEAR.search(text)
    if match:
        try:
            diff, got, ref = (float(match.group(i)) for i in (1, 2, 3))
            return (
                f"logit {match.group(4)}: nntrainer {got:.4g}, Hugging Face {ref:.4g} (diff"
                f" {diff:.3g})"
            )
        except ValueError:
            pass
    lines = text.strip().splitlines()
    if len(lines) > 1 and re.search(r"\.(cpp|h)[:(]\d+", lines[0]):
        lines = lines[1:]                                   # drop the "<file>:<line>" location line
    return " ".join(" ".join(lines).split())[:300]


def parse_gtest(path: Path) -> Dict[str, dict]:
    """Test results from a gtest JSON file, by test name."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for suite in data.get("testsuites", []):
        for testcase in suite.get("testsuite", []):
            out[testcase.get("name", "")] = {
                "result": testcase.get("result", ""),
                "status": testcase.get("status", ""),
                "failures": [f.get("failure", "") for f in testcase.get("failures", [])],
            }
    return out


def verify_hash(files: Dict[str, str]) -> str:
    """What a verify result is valid for: the generated C++ and the converter."""
    hasher = hashlib.sha256()
    for name in sorted(n for n in files if n.endswith((".h", ".cpp", ".py"))):
        hasher.update(name.encode("utf-8") + b"\0" + files[name].encode("utf-8") + b"\0")
    return hasher.hexdigest()[:16]


def current_verify_hash(review: dict) -> str:
    """Hash of the generated files that Verify checks."""
    files = {}
    for name, path in review.get("files", {}).items():
        part = Path(path)
        if name.endswith((".h", ".cpp", ".py")) and part.is_file():
            files[name] = part.read_text(encoding="utf-8", errors="replace")
    return verify_hash(files)


# ----------------------------------------------------------------- verify
# pylint: disable-next=R0914,R0913,R0912,R0915,R0911
def verify_generated(
    review_path: os.PathLike,
    bank: CodeBank,
    nntrainer_root: Optional[os.PathLike] = None,
    python: Optional[str] = None,
    log: Log = print,
    timeout: int = 900,
    host: Optional[str] = None,
) -> dict:
    """Build + run the generated model against a tiny Hugging Face fixture.
    Stores review["verify"]; a pass marks the approved bank entries verified,
    a logits/token mismatch marks the workbench-written ones failed."""
    review_path = Path(review_path)
    review = codegen.load_review(review_path)
    passed, why = codegen.compile_readiness(review)
    if not passed:
        raise codegen.CodegenError(f"Not verified: {why}")
    root = resolve_root(nntrainer_root)
    out_dir = Path(review["out_dir"])
    work = out_dir / "verify"
    work.mkdir(parents=True, exist_ok=True)
    slug = review.get("slug") or out_dir.name
    res = {"at": _now(), "files_hash": current_verify_hash(review), "atol": ATOL_FP32, "ok": False}

    def save(update: dict) -> dict:
        review["verify"] = dict(res, **update)
        review["updated_at"] = _now()
        codegen.refresh_review_summary(review)
        review_path.write_text(json.dumps(review, indent=2, default=str), encoding="utf-8")
        return review

    ex = select_executor(root, host)
    res["host"], res["host_label"] = ex.kind, ex.describe()
    status = build_status(root, ex=ex)
    if status["ok"] and (status["missing"] or ex.kind == "docker"):
        status = ensure_libraries(root, log, ex=ex)
    if not status["ok"]:
        return save(
            {
                "status": status["status"] if status["status"] != "libs_missing" else "no_build",
                "log": status.get("why", "")
                + ("\n" + status.get("log", "") if status.get("log") else ""),
            }
        )
    bdir = Path(status["build_dir"])
    ninja = Ninja(bdir / "build.ninja")

    # 1. fixture
    fixture = f"{slug}_tiny"
    froot = work / "fixture"
    fdir = froot / fixture
    if fdir.exists():
        shutil.rmtree(fdir)
    conv = out_dir / "weight_converter.py"
    if not conv.is_file():
        return save({"status": "fixture_failed", "log": "there is no weight_converter.py"})
    log(f"Making a tiny {review.get('architecture', '')} fixture with Hugging Face ...")
    fenv = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONIOENCODING="utf-8")
    code, out = run_command(
        [
            python or sys.executable,
            "-E",
            str(FIXTURE_SCRIPT),
            "--config",
            str(out_dir / "res" / "config.json"),
            "--converter",
            str(conv),
            "--nntr-config",
            str(out_dir / "res" / "nntr_config.json"),
            "--out",
            str(fdir),
        ],
        work,
        fenv,
        900,
    )
    summary = None
    for line in reversed(out.splitlines()):
        if line.startswith("{"):
            try:
                summary = json.loads(line)
                break
            except ValueError:
                pass
    if not summary or not summary.get("ok"):
        return save(
            {
                "status": "fixture_failed",
                "log": (summary or {}).get("error") or out[-3000:] or f"fixture script exit {code}",
            }
        )
    res["fixture"] = {
        k: summary.get(k)
        for k in (
            "fixture",
            "converter_fn",
            "argmax",
            "max_abs_logit",
            "tokens",
            "layers",
            "params",
        )
    }

    # 2. harness
    try:
        harness = work / f"unittest_{slug}_workbench.cpp"
        harness.write_text(harness_source(review, fixture), encoding="utf-8", newline="\n")
    except VerifyError as exc:
        return save(
            {
                "status": "build_failed",
                "errors": [{"file": "", "line": 0, "message": str(exc)}],
                "log": str(exc),
            }
        )

    # 3. build
    utils = root / "test" / "unittest" / "models" / "causallm_test_utils.cpp"
    if not utils.is_file():
        return save({"status": "no_build", "log": f"{utils} is missing from the checkout"})
    gen_sources = [
        Path(p)
        for n, p in sorted(review["files"].items())
        if n.endswith(".cpp")
        and not os.path.basename(n).startswith("register_")
        and Path(p).is_file()
    ]
    includes = [out_dir, work, root / "test" / "unittest" / "models"]
    if ex.windows:  # gtest.lib comes from nntrainer-windows-resource; the image has its own gtest
        includes.append(root / "subprojects" / "googletest" / "googletest" / "include")
    ext = ".obj" if ex.windows else ".o"
    utils_sha = hashlib.sha256(utils.read_bytes()).hexdigest()[:10]
    jobs = [(harness, work / "obj" / (harness.stem + ext))]
    jobs += [(s, work / "obj" / (s.stem + ext)) for s in gen_sources]
    utils_obj = bdir / "wb_verify" / f"causallm_test_utils-{utils_sha}{ext}"
    if not ex.exists(utils_obj):
        jobs.append((utils, utils_obj))
    ok_, detail = ex.push(
        [
            *gen_sources,
            *[Path(p) for n, p in review["files"].items() if n.endswith(".h")],
            harness,
            fdir,
        ]
    )  # Docker: into its copy of the checkout
    if not ok_:
        return save(
            {
                "status": "build_failed",
                "errors": [{"file": "", "line": 0, "message": detail[-400:]}],
                "log": detail,
            }
        )
    ex.mkdirs(sorted({o.parent for _s, o in jobs}))
    log(f"Compiling {len(jobs)} file(s) with the nntrainer build's flags ({ex.describe()}) ...")
    logs, errors = [], []
    workers = min(4, len(jobs))
    if ex.kind == "docker":                    # parallel compiles must fit Docker's memory
        workers = max(1, min(workers, docker_jobs(ex)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for ok_, errs, text in pool.map(
            lambda j: compile_source(ninja, bdir, j[0], j[1], includes, ex, timeout), jobs
        ):
            logs.append(text)
            errors += errs
    if errors:
        return save(
            {"status": "build_failed", "errors": errors[:40], "log": "\n\n".join(logs)[-20000:]}
        )
    exe = work / (f"unittest_{slug}_workbench" + (".exe" if ex.windows else ""))
    objs = [o for _s, o in jobs if o != utils_obj] + [utils_obj]
    log("Linking against the nntrainer libraries ...")
    ok_, errs, text = link_objects(ninja, bdir, objs, exe, ex, timeout, link_extras(ex, gtest=True))
    logs.append(text)
    if not ok_:
        return save({"status": "build_failed", "errors": errs, "log": "\n\n".join(logs)[-20000:]})

    # 4. run
    renv = ex.lib_env(ex.lib_paths())
    fixture_dir = _no_space_path(froot) if ex.windows else ex.path(froot)
    if " " in fixture_dir:
        return save(
            {
                "status": "error",
                "log": f"nntrainer cannot load weights from a path with spaces ({froot}) "
                "and Windows has no short (8.3) name for it",
            }
        )
    renv["NNTRAINER_CAUSALLM_FIXTURE_DIR"] = fixture_dir
    gjson = work / "gtest.json"
    if gjson.exists():
        gjson.unlink()
    log("Running the generated model against the Hugging Face fixture ...")
    code, out = ex.run(
        [ex.path(exe), "--gtest_output=json:" + ex.path(gjson)], work, timeout, env_extra=renv
    )
    ex.pull(gjson)  # Docker: the result back to the workbench
    tests = parse_gtest(gjson)
    report = _parse_result(out) or {}
    official = tests.get("FP32MatchesHFReference")
    numbers = {
        k: report.get(k)
        for k in (
            "max_abs_diff",
            "argmax_got",
            "argmax_ref",
            "tokens_got",
            "tokens_ref",
            "n_logits",
            "n_ref",
        )
    }
    numbers["tokens_match"] = bool(report.get("tokens_got")) and report.get(
        "tokens_got"
    ) == report.get("tokens_ref")
    numbers["argmax_match"] = report.get("argmax_got") is not None and \
        report.get("argmax_got") == report.get("argmax_ref")
    run_log = out[-12000:]
    if official is None:
        return save(dict(numbers, status="crashed", exit_code=code,
                         log=f"the harness exited with {code} before finishing the test\n{run_log}",
                         error=report.get("error", "")))
    if official["result"] == "SKIPPED" or "SKIP" in official["status"].upper():
        return save(
            dict(
                numbers,
                status="error",
                log="upstream's check skipped (fixture not found)\n" + run_log,
            )
        )
    failures = official["failures"]
    if failures and (report.get("error") or report.get("max_abs_diff") is None):
        # It threw (e.g. weights did not load): a real problem, but not
        # evidence that the code computes the wrong thing -- the bank is left alone.
        return save(dict(numbers, status="crashed", exit_code=code, error=report.get("error", ""),
                         failures=[summarize_failure(f) for f in failures[:20]], log=run_log))
    evidence = {
        "method": "hf-logits",
        "fixture": "tiny",
        "atol": ATOL_FP32,
        "max_abs_diff": numbers.get("max_abs_diff"),
        "tokens_match": numbers["tokens_match"],
        "model": review.get("model"),
        "files_hash": res["files_hash"],
        "init_std": 0.2,
        "layers": res["fixture"].get("layers"),
        "at": res["at"],
        "host": ex.describe(),
    }
    marked = []
    if not failures and code == 0:
        for part in review["parts"]:
            eid = part.get("entry_id")
            entry = bank.get(eid) if eid else None
            if entry and entry["status"] == "approved":
                bank.mark_verified(eid, evidence)
                marked.append(eid)
            if (
                entry
                and (bank.get(eid) or {}).get("status") == "verified"
                and part.get("status")
                in ("approved", "accepted", "needs_verification", "failed_verification")
            ):
                part["status"] = "verified"
        log(f"Verified: logits match Hugging Face (max |diff| {numbers.get('max_abs_diff')}) "
            f"and the greedy tokens match.")
        return save(dict(numbers, status="passed", ok=True, marked_verified=marked, log=run_log))
    for part in review["parts"]:
        eid = part.get("entry_id")
        entry = bank.get(eid) if eid else None
        if part.get("action") == "write" and entry and entry["status"] == "approved":
            bank.mark_failed(eid, dict(evidence, failures=failures[:5]))
            marked.append(eid)
            part["status"] = "failed_verification"
    log("Not verified: the generated model does not match Hugging Face.")
    return save(
        dict(
            numbers,
            status="mismatch",
            failures=[summarize_failure(f) for f in failures[:20]],
            failed_checks=len(failures),
            marked_failed=marked,
            log=run_log,
        )
    )


def verify_feedback(review: dict) -> str:
    """What Claude is told when it is asked to fix a failed verification."""
    verify = review.get("verify") or {}
    status = verify.get("status")
    if status == "build_failed":
        errs = verify.get("errors") or []
        return ("The generated code does not build against nntrainer. Errors:\n"
                + "\n".join(f"{e['file']}:{e['line']}: {e['message']}" for e in errs[:30]))
    if status == "fixture_failed":
        return ("Running weight_converter.py on a tiny random-weight copy of the model failed:\n"
                + str(verify.get("log", ""))[-2000:])
    if status in ("mismatch", "crashed"):
        lines = ["The generated model builds, but on a tiny random-weight copy of the architecture "
                 f"(2 layers, hidden 64) it does not match Hugging Face (fp32, atol {ATOL_FP32})."]
        if verify.get("max_abs_diff") is not None:
            lines.append(f"- max |logit difference| after prefill: {verify['max_abs_diff']}")
        if verify.get("argmax_got") is not None:
            lines.append(
                f"- argmax: nntrainer {verify.get('argmax_got')}, Hugging Face"
                f" {verify.get('argmax_ref')}"
            )
        if verify.get("tokens_got") is not None:
            lines.append(
                f"- greedy tokens: nntrainer {verify.get('tokens_got')}, Hugging Face"
                f" {verify.get('tokens_ref')}"
            )
        if status == "crashed":
            lines.append(
                "- the model crashed or threw: "
                + (verify.get("error") or str(verify.get("log", ""))[-1500:])
            )
        for failure in (verify.get("failures") or [])[:6]:
            lines.append("- " + failure.replace("\n", " ")[:300])
        lines.append(
            "Compare each hook with the Hugging Face reference code (norm placement and epsilon,"
            " RoPE, biases, scaling, activation, tied embeddings) and the weight converter's tensor"
            " order."
        )
        return "\n".join(lines)
    raise VerifyError("there is no failed verification to fix")
