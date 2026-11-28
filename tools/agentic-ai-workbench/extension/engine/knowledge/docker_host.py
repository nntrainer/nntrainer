# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file docker_host.py
# @brief Docker build host: builds nntrainer and generated models in an Ubuntu container.
# @author prachi-t17 <prachi.t@samsung.com>

"""
Docker build host: nntrainer and the generated models are built in an
Ubuntu 24.04 container (engine/docker/Dockerfile.build, gcc-13 like
upstream's Ubuntu CI) instead of natively.

    image      nntr-workbench-build:<sha8 of the Dockerfile>, built on first use
    volume     nntr-wb-<sha8 of the checkout path> at /work: a private copy of
               the checkout (/work/src, synced file by file, build scripts
               converted to LF line endings) and the build (/work/src/build-wb).
               Builds run on the container's own filesystem: compiling through
               a Windows bind mount is ~50x slower. The checkout itself is never
               mounted, so the container cannot change it.
    workbench  tools/agentic-ai-workbench is bind-mounted at /wb (read-write):
               generated code is pushed into the copy before a build, and
               results (test output, the deploy bundle) are copied back there.
    mirror     tools/agentic-ai-workbench/docker/build holds copies of the
               build's build.ninja / compile_commands.json for the host side.

knowledge/verify.py and knowledge/run_model.py run their compile/link/run
steps here through verify.DockerExecutor.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import threading
import urllib.parse
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

DOCKERFILE = Path(__file__).resolve().parent.parent / "docker" / "Dockerfile.build"
IMAGE_REPO = "nntr-workbench-build"
LAYOUT = "2"  # bump when mounts/paths change: the container is recreated
WORK = PurePosixPath("/work")
CSRC = WORK / "src"                            # the copy of the checkout
# Builds of the one copy: variant -> (folder in /work/src, host mirror under docker/).
# "prof" compiles nntrainer's own profiler in (meson -Denable-profile=true,
# i.e. -DPROFILE=1) for Run's per-layer timing; it never replaces the normal
# build, whose timings stay free of profiling overhead.
VARIANTS = {"": ("build-wb", "build"), "prof": ("build-wb-prof", "build-prof")}
PROFILE_OPTIONS = ["-Denable-profile=true"]
CWB = PurePosixPath("/wb")                     # the workbench folder (bind mount)
WB_REL = PurePosixPath("tools/agentic-ai-workbench")
# Left out of the copy: version control, this workbench (pushed file by file
# when needed), and what the Linux build does not use (Windows/Android-only
# folders, docs, vendored libraries the image has as system packages).
SKIP_NAMES = {".git", "__pycache__", ".gradle", "node_modules", ".pytest_cache"}
SKIP_PATHS = {
    "tools/agentic-ai-workbench",
    "build-wb",
    "build-wb-prof",
    "nntrainer-windows-resource",
    "docs",
    "Applications/Android",
    "subprojects/OpenBLAS",
    "subprojects/CLBlast",
    "subprojects/benchmark",
    "subprojects/ruy",
    "subprojects/googletest",
}
LF_SUFFIXES = (".sh",)                          # Linux cannot run `#!/bin/bash\r`
# openblas-num-threads=1 is how upstream builds CausalLM (run_unittest_x86.sh,
# tools/package_android.sh): nntrainer runs its own thread pool, and letting
# OpenBLAS add a thread per core on top oversubscribes the CPU -- measured on
# Qwen2.5-0.5B: 12 -> 1 token/s.
MESON_OPTIONS = [
    "-Denable-transformer=true",
    "-Denable-test=false",
    "-Denable-app=true",
    "-Denable-fp16=false",
    "-Dopenblas-num-threads=1",
    "-Dinstall-app=false",
    "-Denable-tflite-backbone=false",
    "-Denable-tflite-interpreter=false",
    "-Denable-nnstreamer-backbone=false",
    "-Denable-capi=disabled",
    "-Dml-api-support=disabled",
    "-Denable-nnstreamer-tensor-filter=disabled",
    "-Denable-nnstreamer-tensor-trainer=disabled",
]
Log = Callable[[str], None]


class DockerError(RuntimeError):
    """A Docker command failed, or a path is not visible in the container."""


def _sha8(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:8]


def docker_exe() -> Optional[str]:
    """Path of the docker CLI, or None."""
    return shutil.which("docker")


def run_docker(args: Sequence[str], timeout: int = 60, input_text: Optional[str] = None,
            env: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    """Run docker; returns (exit code, output)."""
    exe = docker_exe()
    if not exe:
        return 127, "docker is not installed (or not on PATH)"
    try:
        proc = subprocess.run(
            [exe, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            input=input_text,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 124, f"docker {args[0]} timed out after {timeout}s"
    except OSError as exc:
        return 127, f"could not run docker: {exc}"
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


# ------------------------------------------------------------------ proxy
# A build behind a company proxy needs the proxy inside Docker too: the image
# build (apt-get) and the container. The values travel in the docker
# process's environment and are named, not spelled out, on its command line
# (`--build-arg HTTPS_PROXY`, `-e HTTPS_PROXY`), so a proxy password never
# shows up in a process listing.
PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy")
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")
DOCKER_HOST_ALIAS = "host.docker.internal"


def _container_proxy_url(url: str) -> str:
    """A proxy on this computer's localhost, as a container reaches it."""
    full = url if "://" in url else "http://" + url
    parts = urllib.parse.urlsplit(full)
    if (parts.hostname or "").lower() not in _LOCAL_HOSTS:
        return url
    pos = parts.netloc.rfind("@")
    userinfo = parts.netloc[:pos + 1] if pos >= 0 else ""
    port = f":{parts.port}" if parts.port else ""
    new = urllib.parse.urlunsplit((parts.scheme, userinfo + DOCKER_HOST_ALIAS + port, parts.path,
                                   parts.query, parts.fragment))
    return new if "://" in url else new.split("://", 1)[1]


def proxy_env(env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """The proxy variables that are set, ready for Docker. Windows environment
    names ignore case (HTTPS_PROXY is https_proxy), so there each is named
    once, in capitals, whichever spelling set it."""
    env = os.environ if env is None else env
    windows = os.name == "nt"
    names = tuple(k for k in PROXY_VARS if k.isupper()) if windows else PROXY_VARS
    out = {}
    for k in names:
        val = env.get(k) or (env.get(k.lower()) if windows else None)
        if val:
            out[k] = val if k.lower() == "no_proxy" else _container_proxy_url(val)
    return out


def docker_env() -> Optional[Dict[str, str]]:
    """The environment for a docker command: this one, proxies made usable
    from inside Docker (None when there is no proxy: inherit as is)."""
    proxies = proxy_env()
    return dict(os.environ, **proxies) if proxies else None


# ------------------------------------------------------- engine resources
def engine_info(
    docker: Optional[Callable[[Sequence[str], int], Tuple[int, str]]] = None
) -> Tuple[bool, dict, str]:
    """(running, `docker info` as a dict, error text). With the daemon down the
    CLI still prints JSON with empty fields and exits non-zero."""
    run = docker or (lambda args, timeout: run_docker(list(args), timeout=timeout))
    code, out = run(["info", "--format", "{{json .}}"], 30)
    info: dict = {}
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                info = json.loads(line)
                break
            except ValueError:
                continue
    errors = info.get("ServerErrors") or []
    running = code == 0 and bool(info.get("ServerVersion")) and not errors
    text = "; ".join(str(e) for e in errors) or out.strip()[-300:]
    return running, info, text


GB_PER_BUILD_JOB = 1.5          # gcc on nntrainer's larger sources peaks near this
DEFAULT_BUILD_JOBS = 4          # when Docker does not report its resources


def jobs_for(cpus: int, mem_bytes: int) -> int:
    """Parallel compile jobs that fit Docker's memory: one per 1.5 GB, at most one per CPU."""
    if not mem_bytes:
        return max(1, min(cpus or DEFAULT_BUILD_JOBS, DEFAULT_BUILD_JOBS))
    by_mem = int(mem_bytes / (GB_PER_BUILD_JOB * 1024 ** 3))
    return max(1, min(cpus or by_mem, by_mem))


def engine_resources() -> Tuple[int, int]:
    """(CPUs, memory in bytes) Docker gives containers; (0, 0) when unknown."""
    running, info, _ = engine_info()
    if not running:
        return 0, 0
    return int(info.get("NCPU") or 0), int(info.get("MemTotal") or 0)


def build_jobs() -> int:
    """How many parallel build jobs fit Docker's CPUs and memory."""
    return jobs_for(*engine_resources())


def daemon_status() -> Tuple[bool, str]:
    """(running, detail): the Docker CLI exists and its daemon answers."""
    if not docker_exe():
        return (
            False,
            "Docker is not installed. Install Docker Desktop (https://docs.docker.com/desktop/).",
        )
    code, out = run_docker(["version", "--format", "{{.Server.Version}}"], timeout=30)
    if code != 0 or not out.strip() or "error" in out.lower():
        return False, "Docker is installed but not running. Start Docker Desktop and try again."
    return True, f"Docker {out.strip().splitlines()[-1]}"


def image_tag() -> str:
    """The build image's tag (it changes with the Dockerfile)."""
    return f"{IMAGE_REPO}:{_sha8(DOCKERFILE.read_bytes())}"


def base_image() -> str:
    """The build image's pinned base (its FROM line), reused by the run images."""
    for line in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("FROM "):
            return line.split()[1]
    return "ubuntu:24.04"


def image_exists(tag: str) -> bool:
    """True when the image is present locally."""
    code, _ = run_docker(["image", "inspect", "--format", "{{.Id}}", tag], timeout=30)
    return code == 0


def wait_streaming(proc, timeout: int, on_line: Callable[[str], None]) -> bool:
    """Feed `proc`'s output to `on_line`, line by line, while waiting at most
    `timeout` seconds for it to finish. A reader thread does the reading, so a
    process that hangs without printing anything is still stopped on time.
    False when it timed out (the process is then killed)."""
    def pump() -> None:
        for line in proc.stdout:
            on_line(line.rstrip("\r\n"))

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        reader.join(5)            # a child process still holding the pipe must not hang us
        return False
    reader.join()
    return True


def stream(args: Sequence[str], timeout: int, log: Optional[Log] = None, cwd: Optional[Path] = None,
           env: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    """Run docker with output streamed line by line to `log`."""
    exe = docker_exe()
    if not exe:
        return 127, "docker is not installed (or not on PATH)"
    lines: List[str] = []

    def on_line(line: str) -> None:
        lines.append(line)
        if log:
            log(line)

    try:
        with subprocess.Popen([exe, *args], cwd=str(cwd) if cwd else None, stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              encoding="utf-8", errors="replace", env=env) as proc:
            if not wait_streaming(proc, timeout, on_line):
                lines.append(f"timed out after {timeout}s")
                return 124, "\n".join(lines)
    except OSError as exc:
        return 127, f"could not run docker: {exc}"
    return proc.returncode, "\n".join(lines)


def build_image(log: Optional[Log] = None, timeout: int = 3600) -> Tuple[bool, str]:
    """Build the build image unless it exists; returns (ok, tag or error)."""
    tag = image_tag()
    if image_exists(tag):
        return True, tag
    if log:
        log(f"Building the Docker build image {tag} (Ubuntu 24.04 + gcc-13, first time only) ...")
    proxy_args = [a for k in proxy_env() for a in ("--build-arg", k)]  # names only: values in env
    code, out = stream(["build", "--progress=plain", *proxy_args, "-t", tag, "-f", str(DOCKERFILE),
                        str(DOCKERFILE.parent)], timeout, log, env=docker_env())
    return code == 0, tag if code == 0 else out[-4000:]


def _skipped(rel: str) -> bool:
    return any(rel == p or rel.startswith(p + "/") for p in SKIP_PATHS)


def checkout_manifest(root: Path) -> Dict[str, List[int]]:
    """{relative path: [size, mtime_ns]} of every file that goes into the copy."""
    out: Dict[str, List[int]] = {}
    for data, subs, files in os.walk(root):
        rel_d = Path(data).relative_to(root).as_posix()
        rel_d = "" if rel_d == "." else rel_d
        subs[:] = [
            s for s in subs if s not in SKIP_NAMES and not _skipped(f"{rel_d}/{s}".lstrip("/"))
        ]
        for file in files:
            rel = f"{rel_d}/{file}".lstrip("/")
            try:
                status = os.stat(os.path.join(data, file))
            except OSError:
                continue
            out[rel] = [status.st_size, status.st_mtime_ns]
    return out


def tar_members(root: Path, rels: Iterable[str], base: str = "") -> bytes:
    """A tar of root/<rel> for each rel, stored as <base>/<rel>; build scripts with LF line
    endings."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for rel in rels:
            path = root / rel
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if rel.endswith(LF_SUFFIXES):
                data = data.replace(b"\r\n", b"\n")
            info = tarfile.TarInfo(str(PurePosixPath(base, rel)) if base else rel)
            info.size = len(data)
            info.mode = 0o755 if rel.endswith(LF_SUFFIXES) else 0o644
            info.mtime = int(path.stat().st_mtime)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


# pylint: disable-next=too-many-instance-attributes
class DockerHost:
    """The build container of one checkout. `variant` picks one of its builds
    (VARIANTS): "" the normal one, "prof" a second build of the same copy with
    nntrainer's own profiler compiled in (per-layer timing for Run). Both
    live in the same container and volume."""

    def __init__(self, root: os.PathLike, variant: str = ""):
        if variant not in VARIANTS:
            raise DockerError(f"unknown build variant {variant!r}")
        self.root = Path(root).resolve()
        self.wb_dir = self.root / Path(*WB_REL.parts)
        key = _sha8(os.path.normcase(str(self.root)).encode("utf-8"))
        self.name = f"nntr-wb-build-{key}"
        self.volume = f"nntr-wb-{key}"
        self.variant = variant
        self.cbuild = CSRC / VARIANTS[variant][0]
        self.mirror_rel = Path("docker") / VARIANTS[variant][1]
        self._ready = False

    # -- paths
    @property
    def build_dir(self) -> Path:
        """Host-side mirror of the build (build.ninja, compile_commands.json)."""
        return self.wb_dir / self.mirror_rel

    def to_container(self, host_path: Union[str, os.PathLike]) -> str:
        """Where a host file is in the container's copy (the build mirror maps to the build)."""
        path = Path(host_path).resolve()
        try:
            return str(self.cbuild.joinpath(*path.relative_to(self.build_dir.resolve()).parts))
        except ValueError:
            pass
        try:
            rel = path.relative_to(self.root)
        except ValueError:
            raise DockerError(
                f"{path} is outside the nntrainer checkout, so the build container cannot see it"
            ) from None
        return str(CSRC.joinpath(*rel.parts)) if rel.parts else str(CSRC)

    def live(self, host_path: Union[str, os.PathLike]) -> str:
        """The workbench folder itself (bind mount), for results that must reach the host."""
        path = Path(host_path).resolve()
        try:
            rel = path.relative_to(self.wb_dir)
        except ValueError:
            raise DockerError(f"{path} is outside the workbench folder") from None
        return str(CWB.joinpath(*rel.parts)) if rel.parts else str(CWB)

    # -- container
    def _state(self) -> Tuple[Optional[str], Optional[str], str]:
        code, out = run_docker(
            [
                "container",
                "inspect",
                "--format",
                (
                    '{{.State.Status}}|{{.Config.Image}}|{{index .Config.Labels'
                    ' "nntr-workbench-layout"}}'
                ),
                self.name,
            ]
        )
        if code != 0:
            return None, None, ""
        status, image, layout = (out.strip().split("|") + ["", "", ""])[:3]
        return status, image, layout

    def ensure_container(self, log: Optional[Log] = None) -> Tuple[bool, str]:
        """Start (or create) the build container; returns (ok, its name or the error)."""
        if self._ready:
            return True, self.name
        passed, detail = self._ensure_container(log)
        self._ready = passed
        return passed, detail

    def _ensure_container(self, log: Optional[Log] = None) -> Tuple[bool, str]:
        passed, tag = build_image(log)
        if not passed:
            return False, "the Docker build image did not build:\n" + tag
        status, image, layout = self._state()
        if status and (image != tag or layout != LAYOUT):
            run_docker(["rm", "-f", self.name])
            status = None
        if status == "running":
            return True, self.name
        if status:
            code, out = run_docker(["start", self.name])
            return code == 0, self.name if code == 0 else out
        self.wb_dir.mkdir(parents=True, exist_ok=True)
        code, out = run_docker(["run", "-d", "--name", self.name, "--label", "nntr-workbench=build",
                             "--label", f"nntr-workbench-layout={LAYOUT}",
                             "--mount", f"type=volume,source={self.volume},target={WORK}",
                             "--mount", f"type=bind,source={self.wb_dir},target={CWB}",
                             "-w", str(CSRC), tag, "sleep", "infinity"], timeout=120)
        return code == 0, self.name if code == 0 else out

    # pylint: disable-next=too-many-arguments
    def exec(self, cmd: Union[str, Sequence[str]], workdir: Optional[str] = None,
             env: Optional[Dict[str, str]] = None, timeout: int = 3600,
             log: Optional[Log] = None) -> Tuple[int, str]:
        """Run `cmd` (a shell string or an argv list) in the build container."""
        passed, detail = self.ensure_container(log)
        if not passed:
            return 125, detail
        args = ["exec", "-w", workdir or str(CSRC)]
        for k, val in (env or {}).items():
            args += ["-e", f"{k}={val}"]
        for k in proxy_env():  # names only: the values come from docker_env()
            args += ["-e", k]
        script = cmd if isinstance(cmd, str) else shlex.join(cmd)
        return stream(args + [self.name, "sh", "-c", script], timeout, log, env=docker_env())

    def _untar(self, data: bytes, dest: str, timeout: int = 1800) -> Tuple[int, str]:
        passed, detail = self.ensure_container()
        if not passed:
            return 125, detail
        try:
            proc = subprocess.run(
                [
                    docker_exe(),
                    "exec",
                    "-i",
                    self.name,
                    "sh",
                    "-c",
                    f"mkdir -p {shlex.quote(dest)} && tar -x -C {shlex.quote(dest)} -f -",
                ],
                input=data,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return 1, str(exc)
        return proc.returncode, (proc.stdout + proc.stderr).decode("utf-8", "replace")

    # -- the copy of the checkout
    def sync_checkout(self, log: Optional[Log] = None) -> Tuple[bool, str]:
        """Bring the container's copy up to date with the checkout, file by file."""
        manifest = checkout_manifest(self.root)
        mfile = self.wb_dir / "docker" / "src_manifest.json"
        mid = _sha8(json.dumps(manifest, sort_keys=True).encode("utf-8"))
        prev: Dict[str, List[int]] = {}
        code, have = self.exec(f"cat {WORK}/.src_manifest_id 2>/dev/null || true", timeout=60)
        if code != 0:
            return False, have
        have = have.strip()
        try:
            saved = json.loads(mfile.read_text(encoding="utf-8"))
            if saved.get("id") == have:
                prev = saved.get("files", {})
        except (OSError, ValueError):
            pass
        if have == mid and prev:
            return True, "the copy of the checkout is up to date"
        changed = [r for r, v in manifest.items() if prev.get(r) != v]
        deleted = [r for r in prev if r not in manifest]
        if log:
            log(f"Copying {len(changed)} changed file(s) of the checkout into the build container"
                + (f", removing {len(deleted)}" if deleted else "") + " ...")
        for i in range(0, len(changed), 2000):
            code, out = self._untar(tar_members(self.root, changed[i:i + 2000]), str(CSRC))
            if code != 0:
                return False, "copying the checkout failed: " + out[-2000:]
        if deleted:
            self.exec("rm -f " + " ".join(shlex.quote(str(CSRC / r)) for r in deleted), timeout=300)
        mfile.parent.mkdir(parents=True, exist_ok=True)
        mfile.write_text(json.dumps({"id": mid, "files": manifest}), encoding="utf-8")
        code, out = self.exec(f"echo {mid} > {WORK}/.src_manifest_id", timeout=60)
        return code == 0, f"{len(changed)} file(s) copied"

    def push(self, host_files: Iterable[os.PathLike]) -> Tuple[bool, str]:
        """Copy workbench files (generated code, a harness, a fixture) into the container's copy."""
        rels = []
        for host_file in host_files:
            path = Path(host_file).resolve()
            if path.is_dir():
                rels += [
                    q.relative_to(self.root).as_posix() for q in path.rglob("*") if q.is_file()
                ]
            elif path.is_file():
                rels.append(path.relative_to(self.root).as_posix())
        if not rels:
            return True, ""
        code, out = self._untar(tar_members(self.root, rels), str(CSRC))
        return code == 0, out

    def pull(self, host_path: os.PathLike) -> Tuple[bool, str]:
        """Copy a file the build wrote in the container's copy back to the workbench folder."""
        src = self.to_container(host_path)
        dst = self.live(host_path)
        code, out = self.exec(f"mkdir -p {shlex.quote(str(PurePosixPath(dst).parent))} && "
                              f"cp -a {shlex.quote(src)} {shlex.quote(dst)}", timeout=600)
        return code == 0, out

    def mirror_build(self) -> None:
        """Copies of build.ninja / compile_commands.json for the host side."""
        self.build_dir.mkdir(parents=True, exist_ok=True)
        dst = self.live(self.build_dir)
        self.exec(
            f"for f in build.ninja compile_commands.json; do "
            f"[ -f {self.cbuild}/$f ] && cp {self.cbuild}/$f {shlex.quote(dst)}/$f; done; true",
            timeout=120,
        )

    def missing(self, rel_paths: Sequence[str]) -> List[str]:
        """Which of these build outputs (relative to the build) do not exist yet."""
        if not rel_paths:
            return []
        script = (
            "for f in "
            + " ".join(shlex.quote(r) for r in rel_paths)
            + "; do [ -e \"$f\" ] || echo \"$f\"; done"
        )
        code, out = self.exec(script, workdir=str(self.cbuild), timeout=120)
        if code != 0:
            return list(rel_paths)
        return [line.strip() for line in out.splitlines() if line.strip()]

    def lib_dirs(self) -> List[str]:
        """Container folders of the build that hold shared libraries."""
        code, out = self.exec(
            f"find {self.cbuild} -name '*.so*' -not -path '*.p/*' -printf '%h\\n' | sort -u",
            timeout=120,
        )
        return (
            [line.strip() for line in out.splitlines() if line.strip().startswith("/")]
            if code == 0
            else []
        )


# ------------------------------------------------------------------ setup
def meson_setup_command(variant: str = "") -> List[str]:
    """The meson setup command for a build variant."""
    extra = PROFILE_OPTIONS if variant == "prof" else []
    return ["meson", "setup", *MESON_OPTIONS, *extra, str(CSRC / VARIANTS[variant][0])]


def setup(root: os.PathLike, log: Log = print, variant: str = "") -> dict:
    """Image + container + the copy of the checkout + meson setup (the
    libraries are then built by verify.ensure_libraries). variant "prof":
    the second, profiler-enabled build in the same volume."""
    passed, detail = daemon_status()
    if not passed:
        return {"ok": False, "status": "no_docker", "missing": [detail]}
    host = DockerHost(root, variant)
    ini = host.root / "subprojects" / "iniparser"
    if not ini.is_dir() or not any(ini.iterdir()):
        return {
            "ok": False,
            "status": "missing",
            "missing": [
                "git submodules: run `git submodule sync && git submodule update --init --depth 1`"
                " in the checkout"
            ],
        }
    passed, detail = host.ensure_container(log)
    if not passed:
        return {"ok": False, "status": "setup_failed", "log": detail}
    passed, detail = host.sync_checkout(log)
    if not passed:
        return {"ok": False, "status": "setup_failed", "log": detail}
    code, out = host.exec(f"test -f {host.cbuild}/build.ninja", timeout=60)
    if code != 0:
        cmd = meson_setup_command(variant)
        log("Configuring in Docker: " + " ".join(cmd[1:]))
        code, out = host.exec(cmd, workdir=str(CSRC), timeout=3600, log=log)
        if code != 0:
            return {
                "ok": False,
                "status": "setup_failed",
                "build_dir": str(host.build_dir),
                "log": out[-6000:],
            }
    host.mirror_build()
    return {
        "ok": True,
        "status": "configured",
        "build_dir": str(host.build_dir),
        "container": host.name,
    }


# ------------------------------------------------------- disk use / clean up
RUN_IMAGE_REPO = "nntr-workbench-run"          # the run images of knowledge/run_model.py
_SIZE_UNITS = {"b": 1, "kb": 1e3, "mb": 1e6, "gb": 1e9, "tb": 1e12,
               "kib": 1024, "mib": 1024 ** 2, "gib": 1024 ** 3, "tib": 1024 ** 4}
_REMOVE_ORDER = ("container", "run_image", "build_image", "volume")


def parse_size(text) -> int:
    """Docker's sizes ('213MB', '1.2GB', '0B', 'N/A'; powers of 1000) in bytes."""
    match = re.match(r"\s*([\d.]+)\s*([kmgt]?i?b)\b", str(text or ""), re.I)
    return int(float(match.group(1)) * _SIZE_UNITS[match.group(2).lower()]) if match else 0


def fmt_size(size: int) -> str:
    """Bytes the way Docker prints them (powers of 1000)."""
    return f"{size / 1e9:.1f} GB" if size >= 1e9 else f"{size / 1e6:.0f} MB"


def _created_key(text: str) -> float:
    try:
        return datetime.strptime(str(text)[:25], "%Y-%m-%d %H:%M:%S %z").timestamp()
    except ValueError:
        return 0.0


# pylint: disable-next=too-many-locals,too-many-branches
def usage(
    root: os.PathLike, docker: Optional[Callable[[Sequence[str], int], Tuple[int, str]]] = None
) -> dict:
    """What the workbench keeps in Docker, with sizes (`docker system df -v`).
    An item is `old` when Clean Up may remove it without losing the build:
    a build image of an earlier Dockerfile, or a run image older than the
    newest one of its model."""
    run = docker or (lambda args, timeout: run_docker(list(args), timeout=timeout))
    host = DockerHost(root)
    code, out = run(["system", "df", "-v", "--format", "{{json .}}"], 180)
    data = None
    for line in out.splitlines():
        if line.strip().startswith("{"):
            try:
                data = json.loads(line)
                break
            except ValueError:
                continue
    if code != 0 or data is None:
        return {
            "ok": False,
            "error": out.strip()[-300:] or "docker system df failed",
            "items": [],
            "total": 0,
            "old_total": 0,
        }
    current = image_tag()
    items: List[dict] = []
    runs: Dict[str, List[dict]] = {}
    for image in data.get("Images") or []:
        repo, tag = image.get("Repository"), image.get("Tag")
        if repo not in (IMAGE_REPO, RUN_IMAGE_REPO) or not tag or tag == "<none>":
            continue
        name = f"{repo}:{tag}"
        item = {"kind": "build_image" if repo == IMAGE_REPO else "run_image", "name": name,
                "size": parse_size(image.get("UniqueSize") or image.get("Size")), "old": False,
                "created": image.get("CreatedAt", "")}
        if repo == IMAGE_REPO:
            item["old"] = name != current
        else:
            runs.setdefault(tag.rsplit("-", 1)[0], []).append(item)
        items.append(item)
    for group in runs.values():                         # keep the newest run image of each model
        group.sort(key=lambda i: _created_key(i["created"]), reverse=True)
        for i in group[1:]:
            i["old"] = True
    for volume in data.get("Volumes") or []:
        if volume.get("Name") == host.volume:
            items.append(
                {
                    "kind": "volume",
                    "name": host.volume,
                    "size": parse_size(volume.get("Size")),
                    "old": False,
                    "created": "",
                }
            )
    for container in data.get("Containers") or []:
        if host.name in str(container.get("Names") or "").split(","):
            items.append(
                {
                    "kind": "container",
                    "name": host.name,
                    "size": parse_size(container.get("Size")),
                    "old": False,
                    "created": container.get("CreatedAt", ""),
                }
            )
    return {
        "ok": True,
        "items": items,
        "total": sum(i["size"] for i in items),
        "old_total": sum(i["size"] for i in items if i["old"]),
        "volume": host.volume,
        "container": host.name,
    }


# pylint: disable-next=too-many-locals
def cleanup(
    root: os.PathLike,
    scope: str,
    docker: Optional[Callable[[Sequence[str], int], Tuple[int, str]]] = None,
    log: Optional[Log] = None,
) -> dict:
    """scope "old": old build/run images only (the build stays). scope "all":
    also the build container, its volume (the copy of the checkout + the
    build) and the host-side mirror of the build -- the next Docker step
    sets everything up again."""
    if scope not in ("old", "all"):
        raise ValueError(f"unknown scope {scope!r} (old or all)")
    run = docker or (lambda args, timeout: run_docker(list(args), timeout=timeout))
    use = usage(root, run)
    if not use["ok"]:
        return dict(use, scope=scope, removed=[], failed=[], freed=0)
    host = DockerHost(root)
    chosen = [i for i in use["items"] if i["old"] or scope == "all"]
    removed: List[dict] = []
    failed: List[dict] = []
    for i in sorted(chosen, key=lambda i: _REMOVE_ORDER.index(i["kind"])):
        args = {"container": ["rm", "-f", i["name"]], "volume": ["volume", "rm", i["name"]]}.get(
            i["kind"], ["image", "rm", i["name"]])
        code, out = run(args, 300)
        if code == 0:
            removed.append(i)
            if log:
                log(f"Removed {i['kind'].replace('_', ' ')} {i['name']}")
        else:
            failed.append(dict(i, error=out.strip()[-300:]))
            if log:
                log(f"Could not remove {i['name']}: {out.strip()[-300:]}")
    if scope == "all" and not any(f["kind"] == "volume" for f in failed):
        # The host-side mirrors tell "auto" that a Docker build is set up; they go with the volume.
        for _cdir, mdir in VARIANTS.values():
            shutil.rmtree(host.wb_dir / "docker" / mdir, ignore_errors=True)
        shutil.rmtree(host.wb_dir / "docker" / "selftest", ignore_errors=True)
        try:
            (host.wb_dir / "docker" / "src_manifest.json").unlink()
        except OSError:
            pass
    return {"ok": not failed, "scope": scope, "removed": removed, "failed": failed,
            "freed": sum(i["size"] for i in removed)}
