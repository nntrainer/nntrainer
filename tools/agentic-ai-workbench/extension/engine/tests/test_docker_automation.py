# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file test_docker_automation.py
# @brief Tests for the Docker automation: job limits, retries, guards and clean-up.
# @author prachi-t17 <prachi.t@samsung.com>

# Tests call private helpers and import the engine from the folder that run_all.py
# puts on sys.path at run time (some only after installing stand-ins), which pylint
# cannot follow when it checks this file on its own.
# pylint: disable=protected-access,import-error,wrong-import-position,import-outside-toplevel

"""
Docker automation: job limits and the out-of-memory retry
(knowledge/verify.py), the compile check in Docker (codegen.compile_in_docker),
the run memory guard (knowledge/run_model.py), proxy passing, disk use and
Clean Up (knowledge/docker_host.py), the self-test's stages
(knowledge/selftest.py) and the pinned build image. Docker is faked
throughout; nothing here calls Docker or the network. Standard library only.
"""
import contextlib
import io
import json
import os
import re
import shlex
import struct
import sys
import tempfile
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(_HERE), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import test_codegen as tg  # noqa: E402
import test_compile_and_chat as tcc  # noqa: E402
import test_preflight as tp  # noqa: E402
from knowledge import codegen as cg  # noqa: E402
from knowledge import compile_check  # noqa: E402
from knowledge import docker_host as dh  # noqa: E402
from knowledge import run_model as rm  # noqa: E402
from knowledge import selftest as stt  # noqa: E402
from knowledge import verify  # noqa: E402

GIB = 1024 ** 3
THIS_PC_DOCKER_MEMORY = 8169426944            # Docker Desktop on the development PC: 7.6 GB


def _tmp() -> Path:
    return Path(tempfile.mkdtemp(dir=tg._TMP))


class _Ex:
    """A Docker executor that records what it is asked to run."""
    kind = "docker"
    windows = False

    def __init__(self, results=(), build_dir=None):
        self.results = list(results)
        self.calls = []
        self.pushed = []
        self.build_dir = Path(build_dir or _tmp())

    @staticmethod
    def path(part):
        """A host path as a container-style path."""
        return "/c/" + str(Path(part)).replace("\\", "/").split(":", 1)[-1].lstrip("/")

    @staticmethod
    def quote(text):
        """Quote an argument for a POSIX shell."""
        return shlex.quote(text)

    def run(self, cmd, *_args, env_extra=None, **_kwargs):
        """Record the command and return the next canned result."""
        self.calls.append((cmd, env_extra))
        return self.results.pop(0) if self.results else (0, "")

    def ninja_file(self):
        """The fake build's build.ninja."""
        return self.build_dir / "build.ninja"

    def push(self, files):
        """Record the files pushed."""
        self.pushed += [Path(f) for f in files]
        return True, ""

    def mkdirs(self, dirs):
        """Nothing to create in the fake."""

    def after_build(self):
        """Nothing to do after the fake build."""

    @staticmethod
    def exists(_part):
        """Nothing exists in the fake container."""
        return False

    @staticmethod
    def lib_env(dirs):
        """The library path environment, as the real executor sets it."""
        return {"LD_LIBRARY_PATH": ":".join(dirs)}

    @staticmethod
    def lib_paths():
        """The fake build's library folder."""
        return ["/work/src/build-wb/nntrainer"]

    @staticmethod
    def live(part):
        """A workbench path as the fake container sees it."""
        return "/wb/" + Path(part).name


# -------------------------------------------------------------------- builds
def test_jobs_fit_dockers_memory():
    """Jobs fit Docker's memory."""
    assert dh.jobs_for(24, THIS_PC_DOCKER_MEMORY) == 5         # one job per 1.5 GB
    assert dh.jobs_for(2, 16 * GIB) == 2                       # never more than the CPUs
    assert dh.jobs_for(8, 1 * GIB) == 1
    assert dh.jobs_for(0, 0) == dh.DEFAULT_BUILD_JOBS


def test_out_of_memory_is_recognised():
    """Out of memory is recognised."""
    assert verify.out_of_memory("c++: fatal error: Killed signal terminated program cc1plus")
    assert verify.out_of_memory("virtual memory exhausted: Cannot allocate memory")
    assert verify.out_of_memory("", 137)
    assert not verify.out_of_memory("x.cpp:3:5: error: 'x' was not declared in this scope", 1)


def test_ninja_retries_with_half_the_jobs_when_the_compiler_runs_out_of_memory():
    """Ninja retries with half the jobs when the compiler runs out of memory."""
    killed = "FAILED: a.o\nc++: fatal error: Killed signal terminated program cc1plus"
    ex = _Ex([(1, killed), (0, "ok")])
    code, _out, jobs = verify.run_ninja(
        ex, ex.build_dir, ["libnntrainer.so"], 60, lambda m: None, jobs=6
    )
    assert code == 0 and jobs == 3
    assert ex.calls[0][0][-3:] == ["-j", "6", "libnntrainer.so"]
    assert ex.calls[1][0][-3:] == ["-j", "3", "libnntrainer.so"]
    ex = _Ex([(1, "x.cpp:1:1: error: expected ';'")])          # a real compile error: no retry
    code, _out, _jobs = verify.run_ninja(ex, ex.build_dir, [], 60, jobs=6)
    assert code == 1 and len(ex.calls) == 1
    ex = _Ex()
    ex.kind = "native"                                          # natively: ninja's own default
    verify.run_ninja(ex, ex.build_dir, ["x"], 60)
    assert "-j" not in ex.calls[0][0]


def test_a_build_that_still_runs_out_of_memory_says_so():
    """A build that still runs out of memory says so."""
    why = verify.build_failure_why(
        "fatal error: Killed signal terminated program cc1plus", 1, 2, "default"
    )
    assert "ran out of memory in Docker" in why and "2 parallel job" in why
    assert verify.build_failure_why("x.cpp:1:1: error: nope", 1, 2, "default") == "default"


_LINUX_NINJA = """ninja_required_version = 1.8.2

rule cpp_COMPILER
 command = c++ $ARGS -MD -MQ $out -MF $DEPFILE -o $out -c $in
 deps = gcc
 depfile = $DEPFILE_UNQUOTED

build Applications/CausalLM/libcausallm.so.p/models_qwen3_qwen3_causallm.cpp.o: cpp_COMPILER \
../Applications/CausalLM/models/qwen3/qwen3_causallm.cpp
 DEPFILE = Applications/CausalLM/libcausallm.so.p/q.o.d
 DEPFILE_UNQUOTED = Applications/CausalLM/libcausallm.so.p/q.o.d
 ARGS = -IApplications/CausalLM -std=c++17 -fdiagnostics-color=always
"""


def _generated():
    tmp = _tmp()
    out = tmp / "generated" / "m"
    out.mkdir(parents=True)
    cpp, header = out / "m_workbench_causallm.cpp", out / "m_workbench_causallm.h"
    cpp.write_text("int x;\n", encoding="utf-8")
    header.write_text("#pragma once\n", encoding="utf-8")
    return (
        tmp,
        {"out_dir": str(out), "slug": "m", "files": {cpp.name: str(cpp), header.name: str(header)}},
        [cpp],
        header,
    )


def test_the_compile_check_runs_in_docker_with_the_builds_flags():
    """The compile check runs in Docker with the build's flags."""
    tmp, review, sources, header = _generated()
    ex = _Ex([(0, "")])
    ex.ninja_file().write_text(_LINUX_NINJA, encoding="utf-8")
    compile_in_docker = cg.compile_in_docker(review, sources, tmp, ex=ex)
    assert (
        compile_in_docker["status"] == "ok"
        and compile_in_docker["ok"]
        and compile_in_docker["compiler"] == "g++-13 (Docker)"
        and compile_in_docker["mode"] == "the Docker build"
    )
    assert header in ex.pushed and sources[0] in ex.pushed  # the generated code goes into the copy
    cmd = ex.calls[0][0]
    assert cmd.startswith("c++ ") and "-std=c++17" in cmd and "-fdiagnostics-color=never" in cmd
    assert "-c " in cmd and "m_workbench_causallm.cpp" in cmd and "wb_compile_check" in cmd

    error = ("/work/src/tools/agentic-ai-workbench/generated/m/m_workbench_causallm.cpp:3:5: "
             "error: 'y' was not declared in this scope")
    ex = _Ex([(1, error)])
    ex.ninja_file().write_text(_LINUX_NINJA, encoding="utf-8")
    compile_in_docker = cg.compile_in_docker(review, sources, tmp, ex=ex)
    assert compile_in_docker["status"] == "failed" and not compile_in_docker["ok"]
    assert compile_in_docker["errors"][0] == {"file": "m_workbench_causallm.cpp", "line": 3,
                              "message": "'y' was not declared in this scope"}


def test_the_compile_check_stays_on_this_computer_without_a_docker_build():
    """The compile check stays on this computer without a Docker build."""
    tmp, review, sources, _h = _generated()
    ex = _Ex()
    ex.kind = "native"
    assert cg.compile_in_docker(review, sources, tmp, ex=ex) is None
    assert cg.compile_in_docker(review, sources, tmp, ex=_Ex()) is None  # Docker, but no build yet


def test_without_docker_or_a_compiler_the_message_points_to_docker():
    """Without Docker or a compiler the message points to Docker."""
    bank = tg.tc._bank_copy()
    review = tcc._olmo_review(bank)
    path = review["review_path"]
    tcc._approve_all(path, bank)
    real = (cg.compile_in_docker, compile_check.check)
    cg.compile_in_docker = lambda *a, **k: None
    compile_check.check = lambda sources, root, compiler=None, timeout=300: {
        "status": "no_compiler", "ok": False, "errors": [], "log": "No C++ compiler found."}
    try:
        with tcc._env(WORKBENCH_CXX=None):
            compile_res = cg.compile_generated(path)["compile"]
    finally:
        cg.compile_in_docker, compile_check.check = real
    assert compile_res["status"] == "no_compiler" and "Set Up Docker Build" in compile_res["log"]


def test_an_explicit_compiler_never_goes_to_docker():
    """An explicit compiler never goes to Docker."""
    bank = tg.tc._bank_copy()
    review = tcc._olmo_review(bank)
    path = review["review_path"]
    tcc._approve_all(path, bank)
    real = cg.compile_in_docker

    def boom(*a, **k):
        raise AssertionError("Docker must not be used with an explicit compiler")
    cg.compile_in_docker = boom
    try:
        with tcc._env(WORKBENCH_CXX=tcc._fake_compiler(), FAKE_CXX_FAIL=None):
            assert cg.compile_generated(path)["compile"]["status"] == "ok"
    finally:
        cg.compile_in_docker = real


# ---------------------------------------------------------------------- runs
QWEN25 = {
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "hidden_size": 896,
}
QWEN25_PARAMS = 494032768
QWEN25_MEASURED_PEAK = 2542996 * 1024          # generated/qwen2/run/report.json


def _checkpoint(params, config):
    """A safetensors file with only a header: shapes are all the estimate reads."""
    tmp = _tmp()
    header = json.dumps({"w": {"dtype": "BF16", "shape": [params], "data_offsets": [0, 0]}}).encode(
        "utf-8"
    )
    (tmp / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    (tmp / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp


def test_the_memory_estimate_stays_just_above_qwen25s_measured_peak():
    """The memory estimate stays just above Qwen2.5's measured peak."""
    checkpoint = _checkpoint(QWEN25_PARAMS, QWEN25)
    assert rm.checkpoint_bytes_fp32(checkpoint) == QWEN25_PARAMS * 4
    assert rm.kv_cache_bytes(QWEN25, 512) == 2 * 24 * 2 * 64 * 512 * 4
    need = rm.memory_needed({"out_dir": str(checkpoint)}, checkpoint, 32)
    assert QWEN25_MEASURED_PEAK < need < QWEN25_MEASURED_PEAK * 1.1, need


def test_docker_memory_values():
    """Docker memory values."""
    assert rm.parse_docker_memory("4g") == 4 * GIB
    assert rm.parse_docker_memory("512m") == 512 * 1024 ** 2
    assert rm.parse_docker_memory("4096MB") == 4096 * 1024 ** 2
    assert rm.parse_docker_memory("1.5G") == int(1.5 * GIB)
    assert rm.parse_docker_memory("") == 0 and rm.parse_docker_memory("lots") == 0


def test_a_run_that_cannot_fit_stops_before_anything_slow():
    """A run that cannot fit stops before anything slow."""
    checkpoint = _checkpoint(QWEN25_PARAMS, QWEN25)
    review = {"out_dir": str(checkpoint), "model": "Qwen/Qwen2.5-0.5B-Instruct"}
    real = dh.engine_resources
    dh.engine_resources = lambda: (24, THIS_PC_DOCKER_MEMORY)
    try:
        try:
            rm.check_memory(review, checkpoint, 32, "1g", lambda m: None)
        except rm.RunError as exc:
            msg = str(exc)
        else:
            raise AssertionError("expected RunError")
        assert "needs about 2.5 GB" in msg and "allows 1.0 GB" in msg and "runMemory" in msg
        logs = []
        rm.check_memory(review, checkpoint, 32, "2900m", logs.append)  # fits, but only just
        assert len(logs) == 1 and "Memory is tight" in logs[0]
        _need, limit, where = rm.check_memory(
            review, checkpoint, 32, "", logs.append
        )  # Docker's 7.6 GB
        assert where == "Docker" and limit == THIS_PC_DOCKER_MEMORY and len(logs) == 1
    finally:
        dh.engine_resources = real


def test_an_out_of_memory_kill_is_explained():
    """An out of memory kill is explained."""
    msg = rm.oom_error(
        int(2.5 * GIB), 2 * GIB, "the run's memory limit (2g, aiCompilerWorkbench.runMemory)", "2g"
    )
    assert msg.startswith(
        "the model ran out of memory: it needs about 2.5 GB and the run's memory limit"
    )
    assert "Raise aiCompilerWorkbench.runMemory" in msg
    assert rm.oom_error(0, 0, "Docker", "").startswith("the model ran out of memory. ")


# --------------------------------------------------------------------- proxy
def test_proxies_reach_docker_by_name_only():
    """Proxies reach Docker by name only."""
    saved = {k: os.environ.get(k) for k in dh.PROXY_VARS}
    real = (dh.stream, dh.image_exists)
    calls = []
    try:
        for k in dh.PROXY_VARS:
            os.environ.pop(k, None)
        assert dh.proxy_env() == {} and dh.docker_env() is None
        os.environ["HTTPS_PROXY"] = "http://alice:s3cret@127.0.0.1:3128"
        os.environ["NO_PROXY"] = "localhost,.corp"
        assert dh.proxy_env() == {"HTTPS_PROXY": "http://alice:s3cret@host.docker.internal:3128",
                                  "NO_PROXY": "localhost,.corp"}
        dh.stream = lambda args, timeout, log=None, cwd=None, env=None: calls.append(
            (list(args), env)
        ) or (0, "")
        dh.image_exists = lambda tag: False
        dh.build_image()
        args, env = calls[-1]
        assert args[:2] == ["build", "--progress=plain"]
        assert ["--build-arg", "HTTPS_PROXY"] == args[2:4] and "NO_PROXY" in args
        assert not any("s3cret" in a for a in args)                  # never on the command line
        assert env["HTTPS_PROXY"] == "http://alice:s3cret@host.docker.internal:3128"
        host = dh.DockerHost(_tmp())
        host._ready = True
        host.exec("printenv HTTPS_PROXY")
        args, env = calls[-1]
        named = [args[i + 1] for i, a in enumerate(args) if a == "-e"]
        assert set(named) == {"HTTPS_PROXY", "NO_PROXY"} and not any("s3cret" in a for a in args)
        assert env["HTTPS_PROXY"].endswith("@host.docker.internal:3128")
        assert dh._container_proxy_url("proxy.corp:8080") == "proxy.corp:8080"
        assert dh._container_proxy_url("localhost:3128") == "host.docker.internal:3128"
    finally:
        dh.stream, dh.image_exists = real
        for k, val in saved.items():
            if val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = val


# ---------------------------------------------------------------------- image
def test_the_build_image_is_pinned_by_digest():
    """The build image is pinned by digest."""
    text = dh.DOCKERFILE.read_text(encoding="utf-8")
    from_line = next(line for line in text.splitlines() if line.startswith("FROM "))
    assert re.fullmatch(r"FROM ubuntu:24\.04@sha256:[0-9a-f]{64}", from_line), from_line
    assert "mounted read-only" not in text  # the checkout is copied, not mounted


# ------------------------------------------------------------------- clean up
def _df(root):
    host = dh.DockerHost(root)
    cur = dh.image_tag().split(":", 1)[1]
    return {
        "Images": [
            {
                "Repository": "nntr-workbench-build",
                "Tag": cur,
                "Size": "971MB",
                "UniqueSize": "850MB",
                "CreatedAt": "2026-11-22 10:00:00 +0530 IST",
            },
            {
                "Repository": "nntr-workbench-build",
                "Tag": "3f7055c5",
                "Size": "971MB",
                "UniqueSize": "950MB",
                "CreatedAt": "2026-11-19 23:21:52 +0530 IST",
            },
            {
                "Repository": "nntr-workbench-run",
                "Tag": "qwen2-aaaaaaaaaaaa",
                "Size": "213MB",
                "UniqueSize": "95MB",
                "CreatedAt": "2026-11-18 10:30:55 +0530 IST",
            },
            {
                "Repository": "nntr-workbench-run",
                "Tag": "qwen2-bbbbbbbbbbbb",
                "Size": "213MB",
                "UniqueSize": "96MB",
                "CreatedAt": "2026-11-20 10:30:55 +0530 IST",
            },
            {
                "Repository": "nntr-workbench-run",
                "Tag": "qwen3-cccccccccccc",
                "Size": "230MB",
                "UniqueSize": "110MB",
                "CreatedAt": "2026-11-19 10:30:55 +0530 IST",
            },
            {
                "Repository": "ubuntu",
                "Tag": "24.04",
                "Size": "78MB",
                "UniqueSize": "78MB",
                "CreatedAt": "",
            },
        ],
        "Volumes": [
            {"Name": host.volume, "Size": "6.2GB"},
            {"Name": "someone_elses", "Size": "1GB"},
        ],
        "Containers": [
            {"Names": host.name, "Size": "0B", "CreatedAt": ""},
            {"Names": "other", "Size": "1MB"},
        ],
    }


def _docker_with(disk_usage, fail=()):
    calls = []

    def docker(args, *_args, **_kwargs):
        calls.append(list(args))
        if list(args[:2]) == ["system", "df"]:
            return 0, json.dumps(disk_usage)
        if any(f in args for f in fail):
            return 1, "Error response from daemon: conflict: unable to remove, image is being used"
        return 0, ""
    docker.calls = calls
    return docker


def test_usage_lists_what_the_workbench_keeps_and_what_is_old():
    """Usage lists what the workbench keeps and what is old."""
    root = _tmp()
    usage = dh.usage(root, _docker_with(_df(root)))
    names = {i["name"]: i for i in usage["items"]}
    assert (
        usage["ok"]
        and "ubuntu:24.04" not in names
        and "someone_elses" not in names
        and "other" not in names
    )
    assert names["nntr-workbench-build:3f7055c5"]["old"] and not names[dh.image_tag()]["old"]
    assert names["nntr-workbench-run:qwen2-aaaaaaaaaaaa"]["old"]  # an older qwen2 image
    assert not names["nntr-workbench-run:qwen2-bbbbbbbbbbbb"][
        "old"
    ]  # the newest of each model stays
    assert not names["nntr-workbench-run:qwen3-cccccccccccc"]["old"]
    assert names[dh.DockerHost(root).volume]["size"] == 6_200_000_000
    assert usage["old_total"] == 950_000_000 + 95_000_000
    assert (
        dh.parse_size("1.2GB") == 1_200_000_000
        and dh.parse_size("N/A") == 0
        and dh.parse_size("0B") == 0
    )
    assert dh.fmt_size(1_045_000_000) == "1.0 GB" and dh.fmt_size(95_000_000) == "95 MB"
    assert not dh.usage(root, lambda a, t: (1, "Cannot connect to the Docker daemon"))["ok"]


def test_clean_up_old_keeps_the_build():
    """Clean up old keeps the build."""
    root = _tmp()
    docker = _docker_with(_df(root))
    cleanup = dh.cleanup(root, "old", docker)
    assert cleanup["ok"] and {i["name"] for i in cleanup["removed"]} == {
        "nntr-workbench-build:3f7055c5",
        "nntr-workbench-run:qwen2-aaaaaaaaaaaa",
    }
    assert cleanup["freed"] == 1_045_000_000
    assert not any(a[0] in ("rm", "volume") for a in docker.calls)  # container and volume stay
    docker = _docker_with(_df(root), fail=("nntr-workbench-build:3f7055c5",))
    cleanup = dh.cleanup(root, "old", docker)
    assert (
        not cleanup["ok"]
        and cleanup["failed"][0]["name"] == "nntr-workbench-build:3f7055c5"
        and "being used" in cleanup["failed"][0]["error"]
    )


def test_clean_up_everything_removes_the_build_and_its_mirror():
    """Clean up everything removes the build and its mirror."""
    root = _tmp()
    wb_dir = root / "tools" / "agentic-ai-workbench"
    mirror = wb_dir / "docker" / "build"
    mirror.mkdir(parents=True)
    (mirror / "build.ninja").write_text("", encoding="utf-8")
    (wb_dir / "docker" / "src_manifest.json").write_text("{}", encoding="utf-8")
    docker = _docker_with(_df(root))
    cleanup = dh.cleanup(root, "all", docker)
    order = [a[0] for a in docker.calls if list(a[:2]) != ["system", "df"]]
    assert (
        cleanup["ok"] and order[0] == "rm" and order[-1] == "volume" and order.count("image") == 5
    )
    assert (
        not mirror.exists() and not (wb_dir / "docker" / "src_manifest.json").exists()
    )  # "auto" sees no build
    try:
        dh.cleanup(root, "some", docker)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_clean_up_needs_docker_running():
    """Clean up needs Docker running."""
    from agents import build_setup
    from agents.events import bus
    got = []
    bus.subscribe("docker_usage", got.append)
    real = dh.daemon_status
    dh.daemon_status = lambda: (
        False,
        "Docker is installed but not running. Start Docker Desktop and try again.",
    )
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            code = build_setup.main(["cleanup", "--list"])
    finally:
        dh.daemon_status = real
    assert code == 1 and got[-1]["ok"] is False and "not running" in got[-1]["error"]


def test_the_system_check_reports_dockers_disk_use_and_the_self_test():
    """The system check reports Docker's disk use and the self test."""
    root = tp._checkout()
    base = tp.fake_docker()
    disk_usage = _df(root)

    def docker(args, timeout):
        if list(args[:2]) == ["system", "df"]:
            return 0, json.dumps(disk_usage)
        return base(args, timeout)
    checks = {x["id"]: x for x in tp._run(root=root, docker=docker)["checks"]}
    assert (
        checks["docker_usage"]["status"] == "warn"
        and "of it is old images" in checks["docker_usage"]["detail"]
    )
    assert "Clean Up Docker" in checks["docker_usage"]["fix"]
    assert (
        checks["selftest"]["status"] == "skip"
        and "Run Docker Self-Test" in checks["selftest"]["detail"]
    )

    out = stt.result_dir(root)
    out.mkdir(parents=True)
    (out / "result.json").write_text(
        json.dumps({"ok": True, "at": "2026-11-22T10:00:00+00:00", "seconds": 600}),
        encoding="utf-8",
    )
    checks = {x["id"]: x for x in tp._run(root=root, docker=docker)["checks"]}
    assert (
        checks["selftest"]["status"] == "pass"
        and checks["selftest"]["detail"] == "passed on 2026-11-22 10:00 UTC (10 min)"
    )
    (out / "result.json").write_text(json.dumps({"ok": False, "at": "2026-11-22T10:00:00+00:00",
                                                 "summary": "the build failed"}), encoding="utf-8")
    res = tp._run(root=root, docker=docker, step="setup")
    checks = {x["id"]: x for x in res["checks"]}
    assert (
        checks["selftest"]["status"] == "warn"
        and "the build failed" in checks["selftest"]["detail"]
    )
    assert res["ok"]  # a failed self-test only warns


# ------------------------------------------------------------------ self-test
def _selftest_checkout():
    root = _tmp()
    (root / "Applications" / "CausalLM").mkdir(parents=True)
    (root / "tools" / "agentic-ai-workbench").mkdir(parents=True)
    tests = root / stt.FIXTURE_DEST
    tests.mkdir(parents=True)
    (root / stt.UTILS_SRC).write_text("// utils\n", encoding="utf-8")
    (root / stt.TEST_SRC).write_text("// test\n", encoding="utf-8")
    return root


def _with_fake_build(func):
    real = (verify.compile_source, verify.link_objects, verify.Ninja, verify.docker_jobs)
    compiled = []
    verify.compile_source = lambda nj, bdir, src, obj, includes, ex, timeout: compiled.append(
        src.name
    ) or (True, [], "$ c")
    verify.link_objects = lambda nj, bdir, objs, exe, ex, timeout, extra=None: (True, [], "$ link")
    verify.Ninja = lambda path: object()
    verify.docker_jobs = lambda ex: 5
    try:
        return func(compiled)
    finally:
        verify.compile_source, verify.link_objects, verify.Ninja, verify.docker_jobs = real


def test_the_self_test_runs_every_stage_in_order():
    """The self test runs every stage in order."""
    root = _selftest_checkout()
    ex = _Ex()
    gtest = {
        "testsuites": [
            {
                "testsuite": [
                    {"name": stt.TEST_NAME, "result": "COMPLETED", "status": "RUN", "failures": []}
                ]
            }
        ]
    }

    def run(cmd, *_args, env_extra=None, **_kwargs):
        ex.calls.append((cmd, env_extra))
        if isinstance(cmd, list) and any(str(c).startswith("--gtest_output=") for c in cmd):
            (stt.result_dir(root) / "gtest.json").write_text(json.dumps(gtest), encoding="utf-8")
        return 0, ""
    ex.run = run
    logs = []

    def check_build(compiled):
        selftest = stt.run_selftest(
            root,
            log=logs.append,
            ex=ex,
            gate=lambda step: {"ok": True, "summary": "ok"},
            setup=lambda: {"ok": True},
        )
        return selftest, compiled
    res, compiled = _with_fake_build(check_build)
    assert res["ok"] and res["status"] == "passed", res
    assert [s["stage"] for s in res["stages"]] == [
        "system check",
        "Docker build set up",
        "full build",
        "fixture unpacked",
        "test compiled",
        "test linked",
        "test run",
    ]
    cmds = [c for c, _env in ex.calls]
    assert cmds[0][:2] == ["ninja", "-C"] and cmds[0][-2:] == ["-j", "5"]  # every target, 5 jobs
    assert cmds[1] == ["tar", "-xzf", stt.FIXTURE_TAR, "-C", stt.FIXTURE_DEST]
    assert "--gtest_filter=" + stt.TEST_FILTER in cmds[2]
    assert ex.calls[2][1]["NNTRAINER_CAUSALLM_FIXTURE_DIR"].endswith(
        "test/unittest/models/causallm_reference"
    )
    assert compiled == ["unittest_causallm_qwen3_reference.cpp", "causallm_test_utils.cpp"]
    saved = json.loads((stt.result_dir(root) / "result.json").read_text(encoding="utf-8"))
    assert (
        saved["ok"]
        and saved["test"] == stt.TEST_FILTER
        and (stt.result_dir(root) / "log.txt").is_file()
    )


def test_the_self_test_stops_at_the_first_failing_stage():
    """The self test stops at the first failing stage."""
    root = _selftest_checkout()

    def check_build(_compiled):
        selftest = stt.run_selftest(root, log=lambda m: None, ex=_Ex(), setup=lambda: {"ok": True},
                              gate=lambda step: {"ok": False, "summary": "Docker is not running"})
        killed = "c++: fatal error: Killed signal terminated program cc1plus"
        res2 = stt.run_selftest(
            root,
            log=lambda m: None,
            ex=_Ex([(1, killed), (1, killed)]),
            gate=lambda step: {"ok": True, "summary": ""},
            setup=lambda: {"ok": True},
        )
        mismatch = _Ex()
        gtest = {
            "testsuites": [
                {
                    "testsuite": [
                        {
                            "name": stt.TEST_NAME,
                            "result": "COMPLETED",
                            "status": "RUN",
                            "failures": [{"failure": "max |diff| 0.5 > atol 0.01"}],
                        }
                    ]
                }
            ]
        }

        def run(cmd, *_args, **_kwargs):
            if isinstance(cmd, list) and any(str(c).startswith("--gtest_output=") for c in cmd):
                (stt.result_dir(root) / "gtest.json").write_text(
                    json.dumps(gtest), encoding="utf-8"
                )
                return 1, ""
            return 0, ""
        mismatch.run = run
        res3 = stt.run_selftest(root, log=lambda m: None, ex=mismatch, setup=lambda: {"ok": True},
                              gate=lambda step: {"ok": True, "summary": ""})
        return selftest, res2, res3
    res1, res2, res3 = _with_fake_build(check_build)
    assert res1["status"] == "preflight_failed" and len(res1["stages"]) == 1
    assert res2["status"] == "build_failed" and "ran out of memory in Docker" in res2["summary"]
    assert res3["status"] == "mismatch" and "does not match Hugging Face" in res3["summary"]
