# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file preflight.py
# @brief System check for the workbench's x86 Docker build.
# @author prachi-t17 <prachi.t@samsung.com>

"""
System check for the workbench's x86 Docker build: is this computer ready
to set up the build, verify a model and run it in Docker?

    result = run_checks(step="setup")          # setup | verify | run | all
    python -m knowledge.preflight [--step all] [--json] [--no-network]

Every check reports pass / warn / fail / skip, what it found and, when
something is wrong, a one-line fix. A failed check stops a step only when
that step needs it (`Check.needs`):

    Docker installed, running, building for x86-64   every Docker step
    the git submodule the build uses                  every Docker step
    free disk (only when almost full)                 every Docker step
    the Python environment                            verify, run

Docker's memory, the Claude CLI and the network only warn. With step "all"
(the "Check Docker Setup" command) nothing is stopped; `ok` then says
whether every Docker step could start.

x86-64 only for now: Docker on an ARM computer (Apple Silicon, ARM Linux)
builds ARM binaries, which the workbench does not support yet.

Docker is called through `docker` (default: docker_host.run_docker) and the
network through `http_get`, so tests can replace both. Standard library only.
"""
from __future__ import annotations

import argparse
import http.client
import importlib
import importlib.util
import json
import os
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import docker_host
from . import verify  # verify imports this module too; both only use each other at call time

DockerFn = Callable[[Sequence[str], int], Tuple[int, str]]
HttpFn = Callable[[str, float], Tuple[Optional[int], str]]

STEPS = ("setup", "verify", "run", "all")
DOCKER_STEPS = ("setup", "verify", "run")
GB = 1024 ** 3
MIN_DOCKER_MEMORY_GB = (
    6  # below: warn (builds limit their jobs and runs check the model fits; less is tight)
)
RECLAIM_WARN_GB = 1               # old images worth a Clean Up
MIN_FREE_DISK_GB = 10             # below: warn
CRITICAL_FREE_DISK_GB = 2         # below: fail
PYTHON_MODULES = ("torch", "transformers", "huggingface_hub", "safetensors")
X86 = ("x86_64", "amd64")
HTTP_TIMEOUT = 6.0
USER_AGENT = "nntrainer-workbench-check"
# (id, label, url, what it is needed for)
NETWORK_TARGETS = (
    (
        "net_docker_hub",
        "Docker Hub",
        "https://registry-1.docker.io/v2/",
        "building the Docker image",
    ),
    (
        "net_ubuntu",
        "Ubuntu package servers",
        "http://archive.ubuntu.com/ubuntu/",
        "building the Docker image",
    ),
    (
        "net_hugging_face",
        "Hugging Face",
        "https://huggingface.co/",
        "downloading model configs and weights",
    ),
)
CHECKOUT_HINT = (
    "Open the nntrainer folder in VS Code, or set aiCompilerWorkbench.nntrainerRepoPath to the "
    "nntrainer checkout."
)


@dataclass
class Check:
    """One system check result: status, detail, hint and the steps it blocks."""
    id: str  # pylint: disable=invalid-name  # the key the webview's system check reads
    label: str
    status: str                       # pass | warn | fail | skip
    detail: str
    fix: str = ""
    needs: Tuple[str, ...] = ()       # steps a failure stops

    def blocks(self, step: str) -> bool:
        """True when this check stops `step`."""
        if self.status != "fail":
            return False
        return bool(self.needs) if step == "all" else step in self.needs


# ------------------------------------------------------------------ helpers
def _default_docker(args: Sequence[str], timeout: int) -> Tuple[int, str]:
    return docker_host.run_docker(list(args), timeout=timeout)


def _default_http_get(url: str, timeout: float) -> Tuple[Optional[int], str]:
    """(HTTP status, "") when the server answered -- any status, 401 included --
    or (None, reason) when it could not be reached. Honours the proxy settings
    (HTTP(S)_PROXY, or the system's on Windows)."""
    req = urllib.request.Request(url, method="GET", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, ""
    except urllib.error.HTTPError as exc:
        return exc.code, ""
    except (OSError, ValueError, http.client.HTTPException) as exc:  # URLError, timeout, SSL, ...
        return None, str(getattr(exc, "reason", exc) or exc)


def mask_proxy(url: str) -> str:
    """A proxy URL with any user name / password replaced by ***."""
    if not url:
        return url
    parts = urllib.parse.urlsplit(url if "://" in url else "http://" + url)
    if parts.username is None and parts.password is None:
        return url
    host = parts.hostname or ""
    netloc = "***@" + host + (f":{parts.port}" if parts.port else "")
    return urllib.parse.urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def proxy_settings(env: Optional[Dict[str, str]] = None) -> Tuple[str, str]:
    """(proxy URL masked, where it came from) or ("", "")."""
    env = os.environ if env is None else env
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        if env.get(key):
            return mask_proxy(env[key]), key
    if env is os.environ:
        found = urllib.request.getproxies()       # includes the Windows / macOS system settings
        url = found.get("https") or found.get("http") or ""
        if url:
            return mask_proxy(url), "system proxy settings"
    return "", ""


def _gb(size: float) -> str:
    return f"{size / GB:.1f} GB"


def _docker_info(docker: DockerFn) -> Tuple[bool, dict, str]:
    """(running, info, error text) from `docker info` (docker_host.engine_info)."""
    return docker_host.engine_info(docker)


def memory_fix() -> str:
    """How to give Docker more memory on this operating system."""
    if os.name == "nt":
        return (
            "Give Docker more memory: with the WSL 2 backend add `memory=12GB` under [wsl2] in"
            " %UserProfile%\\.wslconfig, run `wsl --shutdown` and restart Docker Desktop; with the"
            " Hyper-V backend use Docker Desktop > Settings > Resources."
        )
    return (
        "Give Docker more memory (Docker Desktop > Settings > Resources), or close other programs."
    )


def _parse_df(out: str) -> Optional[int]:
    """Available bytes from `df -Pk` output."""
    for line in reversed(out.strip().splitlines()):
        fields = line.split()
        if len(fields) >= 4 and fields[3].isdigit():
            return int(fields[3]) * 1024
    return None


def _disk_check(cid: str, label: str, free: Optional[int], where: str) -> Check:
    if free is None:
        return Check(cid, label, "skip", f"could not read the free space {where}")
    needs = DOCKER_STEPS
    if free < CRITICAL_FREE_DISK_GB * GB:
        return Check(
            cid,
            label,
            "fail",
            f"only {_gb(free)} free {where}",
            f"Free up space {where}: the build, the weights and the Docker image need at least "
            f"{MIN_FREE_DISK_GB} GB.",
            needs,
        )
    if free < MIN_FREE_DISK_GB * GB:
        return Check(
            cid,
            label,
            "warn",
            f"{_gb(free)} free {where}",
            f"Free up space {where}: {MIN_FREE_DISK_GB} GB or more is recommended for the build and"
            " the model weights.",
            needs,
        )
    return Check(cid, label, "pass", f"{_gb(free)} free {where}")


# ------------------------------------------------------------------- checks
def check_checkout(root: Path) -> List[Check]:
    """Check that `root` is an nntrainer checkout with its git submodules."""
    if not (root / "Applications" / "CausalLM").is_dir():
        return [
            Check(
                "checkout",
                "nntrainer checkout",
                "fail",
                f"{root} is not an nntrainer checkout",
                CHECKOUT_HINT,
                DOCKER_STEPS,
            )
        ]
    ini = root / "subprojects" / "iniparser"
    if not ini.is_dir() or not any(ini.iterdir()):
        return [
            Check(
                "submodules",
                "Git submodules",
                "fail",
                "subprojects/iniparser is empty",
                "In the nntrainer folder run: git submodule sync && git submodule update --init"
                " --depth 1",
                DOCKER_STEPS,
            )
        ]
    return [Check("submodules", "Git submodules", "pass", "subprojects/iniparser is there")]


def check_docker(root: Path, docker: DockerFn) -> Tuple[List[Check], dict]:
    """Installed / running / CPU type / memory / free space inside Docker."""
    checks: List[Check] = []
    code, out = docker(["--version"], 20)
    if code != 0:
        checks.append(
            Check(
                "docker_installed",
                "Docker installed",
                "fail",
                "the docker command was not found",
                "Install Docker Desktop (https://docs.docker.com/desktop/) or, on Linux, Docker"
                " Engine.",
                DOCKER_STEPS,
            )
        )
        for cid, label in (
            ("docker_running", "Docker running"),
            ("docker_cpu", "Docker CPU type"),
            ("docker_memory", "Docker memory"),
            ("docker_disk", "Free disk in Docker"),
            ("docker_usage", "Workbench data in Docker"),
        ):
            checks.append(Check(cid, label, "skip", "needs Docker"))
        return checks, {}
    checks.append(Check("docker_installed", "Docker installed", "pass",
                        (out.strip().splitlines() or ["docker found"])[0][:120]))

    running, info, _err = _docker_info(docker)
    if not running:
        checks.append(
            Check(
                "docker_running",
                "Docker running",
                "fail",
                "Docker is installed but not running",
                "Start Docker Desktop (or the Docker service) and wait until it says it is"
                " running.",
                DOCKER_STEPS,
            )
        )
        for cid, label in (
            ("docker_cpu", "Docker CPU type"),
            ("docker_memory", "Docker memory"),
            ("docker_disk", "Free disk in Docker"),
            ("docker_usage", "Workbench data in Docker"),
        ):
            checks.append(Check(cid, label, "skip", "needs Docker running"))
        return checks, info
    engine = info.get("OperatingSystem") or "unknown engine"
    checks.append(Check("docker_running", "Docker running", "pass",
                        f"Docker {info.get('ServerVersion')} is running ({engine})"))

    arch = str(info.get("Architecture") or "")
    if arch.lower() in X86:
        checks.append(Check("docker_cpu", "Docker CPU type", "pass", f"builds for {arch}"))
    else:
        checks.append(
            Check(
                "docker_cpu",
                "Docker CPU type",
                "fail",
                f"Docker builds for {arch or 'an unknown CPU'}, not x86-64",
                "The Docker build supports x86-64 computers only for now (Windows or Linux PCs); "
                "ARM computers such as Apple Silicon Macs come later.",
                DOCKER_STEPS,
            )
        )

    mem, cpus = int(info.get("MemTotal") or 0), int(info.get("NCPU") or 0)
    if not mem:
        checks.append(
            Check("docker_memory", "Docker memory", "skip", "Docker did not report its memory")
        )
    elif mem < MIN_DOCKER_MEMORY_GB * GB:
        checks.append(
            Check(
                "docker_memory",
                "Docker memory",
                "warn",
                f"{_gb(mem)} and {cpus} CPUs: the build may run out of memory and larger models "
                "will not fit",
                memory_fix(),
            )
        )
    else:
        checks.append(
            Check("docker_memory", "Docker memory", "pass", f"{_gb(mem)} and {cpus} CPUs")
        )

    checks.append(_docker_disk(root, docker))
    checks.append(_docker_usage(root, docker))
    return checks, info


def _docker_usage(root: Path, docker: DockerFn) -> Check:
    """How much the workbench keeps in Docker, and what Clean Up could free."""
    label = "Workbench data in Docker"
    usage = docker_host.usage(root, docker)
    if not usage["ok"]:
        return Check("docker_usage", label, "skip", "could not read Docker's disk use")
    images = sum(1 for i in usage["items"] if i["kind"].endswith("image"))
    has_volume = any(i["kind"] == "volume" for i in usage["items"])
    detail = (
        f"{docker_host.fmt_size(usage['total'])} in {images} image{'s' if images != 1 else ''}"
        + (" and the build volume" if has_volume else "")
    )
    if usage["old_total"] >= RECLAIM_WARN_GB * 1e9:
        return Check(
            "docker_usage",
            label,
            "warn",
            f"{detail}; {docker_host.fmt_size(usage['old_total'])} of it is old images",
            "Run 'AI Compiler Workbench: Clean Up Docker' and pick 'Old images only' (the build"
            " stays).",
        )
    return Check("docker_usage", label, "pass", detail)


def check_selftest(root: Path) -> Check:
    """The last Docker self-test (knowledge/selftest.py), if any."""
    label = "Docker self-test"
    handle = root / "tools" / "agentic-ai-workbench" / "docker" / "selftest" / "result.json"
    try:
        data = json.loads(handle.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Check(
            "selftest",
            label,
            "skip",
            "not run yet: 'AI Compiler Workbench: Run Docker Self-Test' "
            "proves the whole Docker path once",
        )
    when = str(data.get("at", ""))[:16].replace("T", " ")
    minutes = round((data.get("seconds") or 0) / 60)
    if data.get("ok"):
        return Check("selftest", label, "pass", f"passed on {when} UTC ({minutes} min)")
    return Check(
        "selftest",
        label,
        "warn",
        f"failed on {when} UTC: {data.get('summary', '')}"[:240],
        "See docker/selftest/log.txt in the workbench folder, fix it, then run the self-test"
        " again.",
    )


def _docker_disk(root: Path, docker: DockerFn) -> Check:
    """Free space where Docker keeps its data: measured in the workbench's
    build container when it runs, else in a throw-away container of the
    build image when that exists (nothing is downloaded for this)."""
    label = "Free disk in Docker"
    name = docker_host.DockerHost(root).name
    code, out = docker(["container", "inspect", "--format", "{{.State.Running}}", name], 20)
    if code == 0 and out.strip() == "true":
        code, out = docker(["exec", name, "df", "-Pk", str(docker_host.WORK)], 30)
        return _disk_check("docker_disk", label, _parse_df(out) if code == 0 else None, "in Docker")
    tag = docker_host.image_tag()
    code, _ = docker(["image", "inspect", "--format", "{{.Id}}", tag], 20)
    if code == 0:
        code, out = docker(["run", "--rm", tag, "df", "-Pk", "/"], 60)
        return _disk_check("docker_disk", label, _parse_df(out) if code == 0 else None, "in Docker")
    return Check(
        "docker_disk",
        label,
        "skip",
        "checked once the Docker build image exists (Set Up Docker Build)",
    )


def check_host_disk(root: Path) -> Check:
    """Check the free disk space for the workbench folder."""
    wb_dir = root / "tools" / "agentic-ai-workbench"
    try:
        free = shutil.disk_usage(wb_dir if wb_dir.is_dir() else root).free
    except OSError:
        free = None
    return _disk_check("host_disk", "Free disk on this computer", free, "on this computer's drive")


def check_python(modules: Sequence[str] = PYTHON_MODULES) -> Check:
    """Check the Python packages that Verify and Run need."""
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if missing:
        return Check(
            "python_env",
            "Python environment",
            "fail",
            f"{sys.executable} lacks {', '.join(missing)}",
            "Run 'AI Compiler Workbench: Set Up / Check Python Environment' (or point "
            "aiCompilerWorkbench.pythonPath at an environment with these packages).",
            ("verify", "run"),
        )
    return Check(
        "python_env", "Python environment", "pass", f"{', '.join(modules)} found ({sys.executable})"
    )


def check_claude(find: Optional[Callable[[], Optional[str]]] = None) -> Check:
    """Check that the Claude CLI can be found (only a warning when not)."""
    if find is None:
        try:  # agents/ is a sibling package, used only here
            find = importlib.import_module("agents.cpp.claude_cli_backend").find_claude
        except ImportError:                             # the engine package is not importable
            return Check("claude_cli", "Claude CLI", "skip", "could not look for the Claude CLI")
    path = find()
    if not path:
        return Check(
            "claude_cli",
            "Claude CLI",
            "warn",
            "not found (needed to generate code, not to build)",
            "Install Claude Code and sign in once by running `claude`, or set"
            " aiCompilerWorkbench.claudePath.",
        )
    return Check("claude_cli", "Claude CLI", "pass", path)


def check_proxy(info: dict, env: Optional[Dict[str, str]] = None) -> Check:
    """Report the proxy this computer and Docker use (passwords hidden)."""
    url, source = proxy_settings(env)
    docker_proxy = mask_proxy(str(info.get("HttpsProxy") or info.get("HttpProxy") or ""))
    parts = [f"{url} (from {source})" if url else "no proxy set on this computer"]
    if docker_proxy:
        parts.append(f"Docker uses {docker_proxy}")
    return Check("proxy", "Proxy", "pass", "; ".join(parts))


def check_network(http_get: HttpFn, image_built: bool) -> List[Check]:
    """Check that the sites the build downloads from can be reached."""
    def one(target):
        cid, label, url, purpose = target
        status, reason = http_get(url, HTTP_TIMEOUT)
        later = (
            " (only needed when the image is rebuilt)"
            if image_built and cid != "net_hugging_face"
            else ""
        )
        if status is None:
            return Check(
                cid,
                label,
                "warn",
                f"not reachable: {reason}"[:200] + later,
                f"Needed for {purpose}: check the internet connection or the proxy settings.",
            )
        if status == 407:
            return Check(
                cid,
                label,
                "warn",
                "the proxy asks for a user name and password" + later,
                "Put the proxy credentials in HTTPS_PROXY, e.g. http://user:password@proxy:port.",
            )
        if status >= 500:
            return Check(cid, label, "warn", f"answered with HTTP {status}" + later,
                         f"Needed for {purpose}: try again later.")
        return Check(cid, label, "pass", "reachable")
    with ThreadPoolExecutor(max_workers=len(NETWORK_TARGETS)) as pool:
        return list(pool.map(one, NETWORK_TARGETS))


# --------------------------------------------------------------------- run
# pylint: disable-next=too-many-arguments
def run_checks(
    root: Optional[os.PathLike] = None,
    step: str = "all",
    network: Optional[bool] = None,
    docker: Optional[DockerFn] = None,
    http_get: Optional[HttpFn] = None,
    python_modules: Sequence[str] = PYTHON_MODULES,
    find_claude: Optional[Callable[[], Optional[str]]] = None,
    env: Optional[Dict[str, str]] = None,
) -> dict:
    """All the checks for `step`. The network is checked for setup and all."""
    if step not in STEPS:
        raise ValueError(f"unknown step {step!r} (one of {STEPS})")
    root = Path(verify.resolve_root(root)).resolve()
    docker = docker or _default_docker
    http_get = http_get or _default_http_get
    network = step in ("setup", "all") if network is None else network

    checks = check_checkout(root)
    docker_checks, info = check_docker(root, docker)
    checks += docker_checks
    checks.append(check_selftest(root))
    checks.append(check_host_disk(root))
    checks.append(check_python(python_modules))
    checks.append(check_claude(find_claude))
    checks.append(check_proxy(info, env))
    if network:
        running = any(c.id == "docker_running" and c.status == "pass" for c in checks)
        built = (
            running
            and docker(["image", "inspect", "--format", "{{.Id}}", docker_host.image_tag()], 20)[0]
            == 0
        )
        checks += check_network(http_get, built)

    blocking = [c for c in checks if c.blocks(step)]
    counts = {s: sum(1 for c in checks if c.status == s) for s in ("pass", "warn", "fail", "skip")}
    return {
        "ok": not blocking,
        "step": step,
        "root": str(root),
        "summary": summarize(step, blocking, counts),
        "blocking": [c.id for c in blocking],
        "counts": counts,
        "checks": [asdict(c) for c in checks],
    }


STEP_NAMES = {
    "setup": "setting up the Docker build",
    "verify": "verifying in Docker",
    "run": "running in Docker",
    "all": "the Docker build",
}


def summarize(step: str, blocking: List[Check], counts: Dict[str, int]) -> str:
    """One line that says whether `step` can go ahead."""
    if blocking:
        what = "; ".join(f"{c.label}: {c.detail}" for c in blocking)
        return (f"Not ready for {STEP_NAMES[step]}: {what}" if step == "all"
                else f"Stopped before {STEP_NAMES[step]}: {what}")
    warns = counts.get("warn", 0)
    if warns:
        return f"Ready for {STEP_NAMES[step]}, with {warns} warning{'s' if warns != 1 else ''}."
    return f"Ready for {STEP_NAMES[step]}: every check passed."


MARK = {"pass": "ok", "warn": "warn", "fail": "FAIL", "skip": "skip"}
LEVEL = {"pass": "info", "warn": "warn", "fail": "error", "skip": "info"}


def log_lines(result: dict) -> List[Tuple[str, str]]:
    """(line, level) pairs for the Output panel / Logs."""
    out = [(f"Docker setup check ({result['step']}):", "info")]
    for check in result["checks"]:
        line = f"  [{MARK[check['status']]}] {check['label']}: {check['detail']}"
        if check["fix"] and check["status"] in ("warn", "fail"):
            line += f" -- fix: {check['fix']}"
        out.append((line, LEVEL[check["status"]]))
    out.append((result["summary"], "info" if result["ok"] else "error"))
    return out


def gate(
    step: str, log: Callable[..., None], emit: Optional[Callable[[dict], None]] = None, **kw
) -> dict:
    """Run the checks for `step`, log them and (optionally) emit the result."""
    result = run_checks(step=step, **kw)
    for line, level in log_lines(result):
        log(line, level)
    if emit:
        emit(result)
    return result


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m knowledge.preflight",
        description="Is this computer ready for the workbench's x86 Docker build?",
    )
    parser.add_argument("--step", default="all", choices=STEPS)
    parser.add_argument(
        "--root", help="the nntrainer checkout (default: found from this file or $NNTRAINER_ROOT)"
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-network", action="store_true")
    args = parser.parse_args(argv)
    result = run_checks(args.root, args.step, network=False if args.no_network else None)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for line, _ in log_lines(result):
            print(line)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
