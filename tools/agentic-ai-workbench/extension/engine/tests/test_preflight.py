# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file test_preflight.py
# @brief Tests for the x86 Docker build's system check.
# @author prachi-t17 <prachi.t@samsung.com>

# Tests call private helpers and import the engine from the folder that run_all.py
# puts on sys.path at run time (some only after installing stand-ins), which pylint
# cannot follow when it checks this file on its own.
# pylint: disable=protected-access,import-error,wrong-import-position,import-outside-toplevel

"""
System check for the x86 Docker build (knowledge/preflight.py) and where it
runs by itself (agents/build_setup.py, agents/codegen_review.py). Docker,
the network, the host disk and the Claude CLI lookup are all faked: nothing
here calls Docker or the internet. Standard library only.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(_HERE), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from knowledge import preflight as pf  # noqa: E402

GB = pf.GB
_TMP = tempfile.mkdtemp(prefix="preflight_test_")


def _checkout(submodule=True, causallm=True):
    root = Path(tempfile.mkdtemp(dir=_TMP))
    (root / "tools" / "agentic-ai-workbench").mkdir(parents=True)
    if causallm:
        (root / "Applications" / "CausalLM").mkdir(parents=True)
    ini = root / "subprojects" / "iniparser"
    ini.mkdir(parents=True)
    if submodule:
        (ini / "iniparser.h").write_text("", encoding="utf-8")
    return root


def _df(kbytes):
    return (
        "Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/sdd 100000000 1"
        f" {kbytes} 1% /work\n"
    )


# pylint: disable-next=too-many-arguments
def fake_docker(
    installed=True,
    running=True,
    arch="x86_64",
    mem=16 * GB,
    cpus=8,
    container=True,
    image=True,
    df_kb=50 * 1024 * 1024,
    info_extra=None,
):
    """A stand-in for the docker CLI in the given state (it records every call)."""
    calls = []

    # pylint: disable-next=too-many-return-statements
    def docker(args, *_args, **_kwargs):
        attn = list(args)
        calls.append(attn)
        if attn == ["--version"]:
            return (
                (0, "Docker version 29.0.0, build abc")
                if installed
                else (127, "docker is not installed")
            )
        if attn[0] == "info":
            if not running:
                return 1, json.dumps(
                    {"ServerVersion": "", "ServerErrors": ["Cannot connect to the Docker daemon"]}
                )
            data = {"ServerVersion": "29.0.0", "Architecture": arch, "MemTotal": mem, "NCPU": cpus,
                 "OperatingSystem": "Docker Desktop"}
            data.update(info_extra or {})
            return 0, json.dumps(data)
        if attn[:2] == ["container", "inspect"]:
            return (0, "true\n") if container else (1, "Error: No such container")
        if attn[:2] == ["image", "inspect"]:
            return (0, "sha256:abc\n") if image else (1, "Error: No such image")
        if (attn[0] == "exec" or attn[0] == "run") and "df" in attn:
            return 0, _df(df_kb)
        return 1, "unexpected docker " + " ".join(attn)
    docker.calls = calls
    return docker


# pylint: disable-next=too-many-arguments
def _run(
    root=None,
    step="all",
    docker=None,
    http_get=None,
    modules=(),
    claude="/usr/bin/claude",
    env=None,
    network=False,
    host_free=100 * GB,
):
    real = pf.check_host_disk
    pf.check_host_disk = lambda r: pf._disk_check(
        "host_disk", "Free disk on this computer", host_free, "here"
    )
    try:
        return pf.run_checks(
            root or _checkout(),
            step,
            network=network,
            docker=docker or fake_docker(),
            http_get=http_get,
            python_modules=modules,
            find_claude=lambda: claude,
            env=env if env is not None else {},
        )
    finally:
        pf.check_host_disk = real


def _by_id(result):
    return {c["id"]: c for c in result["checks"]}


def test_everything_ready():
    """Everything ready."""
    res = _run()
    by_id = _by_id(res)
    assert res["ok"] and res["blocking"] == [], res
    assert res["summary"].startswith("Ready for the Docker build: every check passed"), res[
        "summary"
    ]
    for cid in (
        "submodules",
        "docker_installed",
        "docker_running",
        "docker_cpu",
        "docker_memory",
        "docker_disk",
        "host_disk",
        "python_env",
        "claude_cli",
        "proxy",
    ):
        assert by_id[cid]["status"] == "pass", (cid, by_id[cid])
    assert "Docker 29.0.0 is running (Docker Desktop)" in by_id["docker_running"]["detail"]
    assert "16.0 GB and 8 CPUs" in by_id["docker_memory"]["detail"]


def test_docker_not_installed_stops_every_docker_step():
    """Docker not installed stops every Docker step."""
    data = fake_docker(installed=False)
    for step in ("setup", "verify", "run", "all"):
        res = _run(step=step, docker=data)
        by_id = _by_id(res)
        assert not res["ok"] and "docker_installed" in res["blocking"], (step, res)
        assert "Install Docker Desktop" in by_id["docker_installed"]["fix"]
        assert by_id["docker_running"]["status"] == by_id["docker_cpu"]["status"] == "skip"
    assert _run(step="all", docker=data)["summary"].startswith("Not ready for the Docker build")
    assert _run(step="setup", docker=data)["summary"].startswith(
        "Stopped before setting up the Docker build"
    )


def test_docker_not_running():
    """Docker not running."""
    res = _run(step="setup", docker=fake_docker(running=False))
    by_id = _by_id(res)
    assert res["blocking"] == ["docker_running"]
    assert "Start Docker Desktop" in by_id["docker_running"]["fix"]
    assert by_id["docker_disk"]["status"] == "skip"


def test_arm_docker_is_not_supported_yet():
    """Arm Docker is not supported yet."""
    res = _run(step="verify", docker=fake_docker(arch="aarch64"))
    by_id = _by_id(res)
    assert res["blocking"] == ["docker_cpu"]
    assert (
        "not x86-64" in by_id["docker_cpu"]["detail"]
        and "x86-64 computers only" in by_id["docker_cpu"]["fix"]
    )
    assert _by_id(_run(docker=fake_docker(arch="amd64")))["docker_cpu"]["status"] == "pass"


def test_low_memory_only_warns():
    """Low memory only warns."""
    res = _run(step="setup", docker=fake_docker(mem=4 * GB, cpus=4))
    by_id = _by_id(res)
    assert res["ok"] and by_id["docker_memory"]["status"] == "warn"
    assert "4.0 GB and 4 CPUs" in by_id["docker_memory"]["detail"] and by_id["docker_memory"]["fix"]
    assert res["summary"] == "Ready for setting up the Docker build, with 1 warning."


def test_memory_warning_starts_below_6_gb():
    """Memory warning starts below 6 GB."""
    this_pc = 8169426944                                            # Docker Desktop here: 7.6 GB
    assert _by_id(_run(docker=fake_docker(mem=this_pc)))["docker_memory"]["status"] == "pass"
    assert _by_id(_run(docker=fake_docker(mem=int(5.9 * GB))))["docker_memory"]["status"] == "warn"


def test_disk_in_docker_from_the_container_or_the_image():
    """Disk in Docker from the container or the image."""
    data = fake_docker(df_kb=1 * 1024 * 1024)                          # 1 GB: almost full
    res = _run(step="run", docker=data)
    assert res["blocking"] == ["docker_disk"] and _by_id(res)["docker_disk"]["status"] == "fail"
    assert any(a[0] == "exec" and "df" in a for a in data.calls)
    data = fake_docker(
        container=False, df_kb=5 * 1024 * 1024
    )  # no container yet: the image is used
    res = _run(docker=data)
    assert res["ok"] and _by_id(res)["docker_disk"]["status"] == "warn"
    assert any(a[0] == "run" and "--rm" in a for a in data.calls)
    data = fake_docker(container=False, image=False)  # nothing built yet: skipped, no download
    by_id = _by_id(_run(docker=data))
    assert by_id["docker_disk"]["status"] == "skip" and not any(a[0] == "run" for a in data.calls)


def test_host_disk_thresholds():
    """Host disk thresholds."""
    assert _by_id(_run(host_free=1 * GB))["host_disk"]["status"] == "fail"
    assert "host_disk" in _run(step="setup", host_free=1 * GB)["blocking"]
    assert _by_id(_run(host_free=5 * GB))["host_disk"]["status"] == "warn"
    real = pf.check_host_disk(_checkout())  # the real one reads this computer's disk
    assert real.status in ("pass", "warn", "fail") and "free" in real.detail


def test_python_environment_is_needed_for_verify_and_run_only():
    """Python environment is needed for verify and run only."""
    missing = ("surely_not_an_installed_module_xyz",)
    assert _run(step="setup", modules=missing)["ok"]
    for step in ("verify", "run"):
        res = _run(step=step, modules=missing)
        assert res["blocking"] == ["python_env"], res
        assert "surely_not_an_installed_module_xyz" in _by_id(res)["python_env"]["detail"]
    assert _by_id(_run(modules=("json",)))["python_env"]["status"] == "pass"


def test_checkout_and_submodules():
    """Checkout and submodules."""
    res = _run(root=_checkout(submodule=False), step="setup")
    assert res["blocking"] == ["submodules"]
    assert "git submodule update --init" in _by_id(res)["submodules"]["fix"]
    res = _run(root=_checkout(causallm=False), step="verify")
    assert res["blocking"] == ["checkout"] and "nntrainerRepoPath" in _by_id(res)["checkout"]["fix"]


def test_claude_cli_only_warns():
    """Claude CLI only warns."""
    res = _run(step="run", claude=None)
    assert res["ok"] and _by_id(res)["claude_cli"]["status"] == "warn"


def test_network_checks():
    """Network checks."""
    seen = []

    def http_get(url, *_args, **_kwargs):
        seen.append(url)
        if "docker.io" in url:
            return 401, ""                                          # answered: reachable
        if "ubuntu" in url:
            return None, "timed out"
        return 407, ""                                              # the proxy wants credentials
    res = _run(step="setup", http_get=http_get, network=True)
    by_id = _by_id(res)
    assert res["ok"], res                                               # the network only warns
    assert by_id["net_docker_hub"]["status"] == "pass"
    assert by_id["net_ubuntu"]["status"] == "warn" and "timed out" in by_id["net_ubuntu"]["detail"]
    assert (
        "only needed when the image is rebuilt" in by_id["net_ubuntu"]["detail"]
    )  # the image exists
    assert (
        by_id["net_hugging_face"]["status"] == "warn"
        and "user name and password" in by_id["net_hugging_face"]["detail"]
    )
    assert len(seen) == 3
    # by default the network is checked for setup and all, not before verify / run
    res = pf.run_checks(
        _checkout(),
        "verify",
        docker=fake_docker(),
        http_get=http_get,
        python_modules=(),
        find_claude=lambda: "claude",
        env={},
    )
    assert not any(ch["id"].startswith("net_") for ch in res["checks"])


def test_proxy_passwords_are_never_shown():
    """Proxy passwords are never shown."""
    assert pf.mask_proxy("http://alice:s3cret@proxy.local:8080") == "http://***@proxy.local:8080"
    assert pf.mask_proxy("http://proxy.local:8080") == "http://proxy.local:8080"
    assert pf.mask_proxy("proxy.local:3128") == "proxy.local:3128"
    env = {"HTTPS_PROXY": "http://alice:s3cret@proxy.local:8080"}
    data = fake_docker(info_extra={"HttpsProxy": "http://bob:hunter2@corp:3128"})
    res = _run(env=env, docker=data)
    text = json.dumps(res) + "\n".join(line for line, _ in pf.log_lines(res))
    assert "s3cret" not in text and "hunter2" not in text
    detail = _by_id(res)["proxy"]["detail"]
    assert (
        "http://***@proxy.local:8080 (from HTTPS_PROXY)" in detail
        and "Docker uses http://***@corp:3128" in detail
    )
    assert _by_id(_run(env={}))["proxy"]["detail"] == "no proxy set on this computer"


def test_log_lines_and_gate():
    """Log lines and gate."""
    res = _run(step="setup", docker=fake_docker(mem=4 * GB))
    lines = pf.log_lines(res)
    assert lines[0][0] == "Docker setup check (setup):"
    mem = next(line for line, _ in lines if "Docker memory" in line)
    assert mem.startswith("  [warn]") and "-- fix:" in mem
    assert lines[-1][0] == res["summary"]

    logged, emitted = [], []
    real = pf.run_checks
    pf.run_checks = lambda step, **kw: dict(res, step=step)
    try:
        out = pf.gate("setup", lambda m, level="info": logged.append((m, level)), emitted.append)
    finally:
        pf.run_checks = real
    assert out["step"] == "setup" and emitted == [out] and len(logged) == len(lines)


def test_unknown_step_is_rejected():
    """Unknown step is rejected."""
    try:
        pf.run_checks(_checkout(), "deploy")
    except ValueError as exc:
        assert "deploy" in str(exc)
    else:
        raise AssertionError("expected ValueError")


# ------------------------------------------------- where the check runs by itself
def _capture(event):
    from agents.events import bus
    got = []
    bus.subscribe(event, got.append)
    return got


def test_build_setup_check_action_emits_the_result():
    """Build setup check action emits the result."""
    from agents import build_setup
    got = _capture("preflight")
    canned = _run()
    real = pf.run_checks
    pf.run_checks = lambda step, **kw: dict(canned, step=step)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            code = build_setup.main(["check", "--step", "all"])
    finally:
        pf.run_checks = real
    assert code == 0 and got and got[-1]["step"] == "all" and got[-1]["ok"]


def test_docker_setup_stops_when_the_check_fails():
    """Docker setup stops when the check fails."""
    from agents import build_setup
    from knowledge import docker_host, verify
    status = _capture("build_status")
    blocked = _run(step="setup", docker=fake_docker(running=False))
    real = (pf.run_checks, verify.select_executor, verify.setup_build, docker_host.daemon_status)
    calls = []
    pf.run_checks = lambda step, **kw: dict(blocked, step=step)
    verify.select_executor = lambda *a, **kw: type("Ex", (), {"kind": "docker"})()
    verify.setup_build = lambda **kw: calls.append(kw) or {"ok": True}
    docker_host.daemon_status = lambda: (False, "not running")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            code = build_setup.main(["setup", "--host", "docker"])
    finally:
        pf.run_checks, verify.select_executor, verify.setup_build, docker_host.daemon_status = real
    assert code == 1 and not calls                                  # setup never started
    assert status[-1]["status"] == "preflight_failed" and "Docker running" in status[-1]["why"]


def test_review_actions_check_docker_first():
    """Review actions check Docker first."""
    from agents import codegen_review as cr
    from knowledge import codegen as cg
    blocked = _run(step="run", docker=fake_docker(running=False))
    real = pf.run_checks
    pf.run_checks = lambda step, **kw: dict(blocked, step=step)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            cr.docker_ready("native", "verify")                    # a native build: no Docker check
            try:
                cr.docker_ready("docker", "run")
            except cg.CodegenError as exc:
                assert "Stopped before running in Docker" in str(exc)
            else:
                raise AssertionError("expected CodegenError")
    finally:
        pf.run_checks = real
