# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file test_hosts.py
# @brief Tests for the Docker and native build hosts, with Docker faked in-process.
# @author prachi-t17 <prachi.t@samsung.com>

# Tests call private helpers and import the engine from the folder that run_all.py
# puts on sys.path at run time (some only after installing stand-ins), which pylint
# cannot follow when it checks this file on its own.
# pylint: disable=protected-access,import-error,wrong-import-position

"""
The build hosts with Docker and the toolchain faked in-process:
- knowledge/docker_host.py against a fake `docker` (subprocess replaced): the
  daemon check, resources and build jobs, the image, the container's life
  cycle, exec with streamed output, syncing the checkout copy file by file,
  push / pull / mirror / missing outputs / library folders, and setup;
- knowledge/verify.py's executors (native and Docker), its process runner,
  tool environment, path-length and git checks, the link step, library builds
  and the build setup's early exits.
Nothing here starts Docker or a compiler. Standard library only.
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path
import time
import shutil

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(_HERE), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import test_codegen as tg  # noqa: E402
import test_docker_automation as tda  # noqa: E402
import test_run_model as trm  # noqa: E402
import test_verify as tv  # noqa: E402
from knowledge import docker_host as dh  # noqa: E402
from knowledge import verify  # noqa: E402


def _tmp() -> Path:
    return Path(tempfile.mkdtemp(dir=tg._TMP))


@contextlib.contextmanager
def _patched(obj, **attrs):
    old = {k: getattr(obj, k) for k in attrs}
    for k, attr in attrs.items():
        setattr(obj, k, attr)
    try:
        yield
    finally:
        for k, attr in old.items():
            setattr(obj, k, attr)


# ------------------------------------------------------------- a fake docker
# pylint: disable-next=too-many-instance-attributes,too-few-public-methods
class _FakeDocker:
    """Answers `docker ...` like a daemon with one build container would."""

    # pylint: disable-next=too-many-arguments
    def __init__(self, running=True, image=False, container=None, container_image="", layout="",
                 configured=False, fail=(), timeouts=(), missing_out=""):
        self.running, self.image, self.container = running, image, container
        self.container_image, self.layout, self.configured = container_image, layout, configured
        self.fail, self.timeouts, self.missing_out = fail, timeouts, missing_out
        self.calls, self.manifest_id, self.untarred = [], "", 0

    # pylint: disable-next=too-many-branches,too-many-return-statements
    def handle(self, args, data=None):
        """Answer one docker command line from the fake's state."""
        self.calls.append(list(args))
        cmd = args[0] if args else ""
        if cmd in self.timeouts:
            raise subprocess.TimeoutExpired(args, 1)
        if cmd == "version":
            return (0, "29.1.0\n") if self.running else (1, "error during connect\n")
        if cmd == "info":
            if self.running:
                return (
                    0,
                    json.dumps({"ServerVersion": "29.1.0", "NCPU": 8, "MemTotal": 12 * 1024**3})
                    + "\n",
                )
            return 1, json.dumps({"ServerVersion": "", "ServerErrors": ["daemon down"]}) + "\n"
        if args[:2] == ["image", "inspect"]:
            return (0, "sha256:1\n") if self.image else (1, "Error: No such image\n")
        if cmd == "build":
            self.image = True
            return 0, "Step 1/3 : FROM ubuntu\nSuccessfully built 1\n"
        if args[:2] == ["container", "inspect"]:
            if not self.container:
                return 1, "Error: No such container\n"
            return 0, f"{self.container}|{self.container_image}|{self.layout}\n"
        if cmd == "rm":
            self.container = None
            return 0, args[-1] + "\n"
        if cmd == "start":
            self.container = "running"
            return 0, args[-1] + "\n"
        if cmd == "run":
            self.container, self.container_image = "running", args[-3]
            self.layout = [
                a.split("=", 1)[1] for a in args if a.startswith("nntr-workbench-layout=")
            ][0]
            return 0, "containerid\n"
        if cmd == "exec":
            if "-i" in args:
                self.untarred += len(data or b"")
                return (1, "tar: error\n") if "untar" in self.fail else (0, "")
            script = args[-1]
            for bad in self.fail:
                if bad in script:
                    return 1, f"failed: {bad}\n"
            if ".src_manifest_id 2>/dev/null" in script:
                return 0, self.manifest_id + "\n"
            if script.startswith("echo ") and ".src_manifest_id" in script:
                self.manifest_id = script.split()[1]
                return 0, ""
            if script.startswith("test -f"):
                return (0, "") if self.configured else (1, "")
            if script.startswith("meson setup"):
                self.configured = True
                return 0, "Build targets in project: 5\n"
            if "*.so*" in script:
                return 0, "/work/src/build-wb/nntrainer\n/work/src/build-wb/api\n"
            if script.startswith("for f in") and "[ -e" in script:
                return 0, self.missing_out
            return 0, "ok\n"
        return 0, ""


class _Proc:
    def __init__(self, code, out, timeout=False):
        self.stdout, self.returncode, self._timeout, self.killed = (
            io.StringIO(out),
            code,
            timeout,
            False,
        )

    def wait(self, timeout=None):
        """Wait for the fake process (or time out when told to)."""
        if self._timeout:
            raise subprocess.TimeoutExpired("docker", timeout)
        return self.returncode

    def kill(self):
        """Record that the process was killed."""
        self.killed = True

    def __enter__(self):                       # like subprocess.Popen: `with Popen(...) as proc`
        return self

    def __exit__(self, *exc):
        self.stdout.close()
        return False


class _FakeSubprocess:
    PIPE, STDOUT, DEVNULL = subprocess.PIPE, subprocess.STDOUT, subprocess.DEVNULL
    TimeoutExpired = subprocess.TimeoutExpired

    def __init__(self, docker, popen_timeout=False):
        self.docker, self.popen_timeout = docker, popen_timeout

    def run(self, argv, **kw):
        """subprocess.run for docker commands, answered by the fake docker."""
        code, out = self.docker.handle(argv[1:], kw.get("input"))
        if kw.get("text"):
            return types.SimpleNamespace(returncode=code, stdout=out, stderr="")
        return types.SimpleNamespace(returncode=code, stdout=out.encode(), stderr=b"")

    def Popen(
        self, argv, **_kwargs
    ):  # pylint: disable=invalid-name  # stands in for subprocess.Popen
        """subprocess.Popen for docker commands, answered by the fake docker."""
        code, out = self.docker.handle(argv[1:])
        return _Proc(code, out, self.popen_timeout)


@contextlib.contextmanager
def _docker(fake, popen_timeout=False):
    with _patched(dh, docker_exe=lambda: "docker", subprocess=_FakeSubprocess(fake, popen_timeout)):
        yield fake


def _checkout():
    root = trm._checkout()
    (root / "subprojects" / "iniparser").mkdir(parents=True)
    (root / "subprojects" / "iniparser" / "iniparser.c").write_text("int x;\n", encoding="utf-8")
    (root / "nntrainer").mkdir()
    (root / "nntrainer" / "a.cpp").write_text("int a;\n", encoding="utf-8")
    return root


# ------------------------------------------------------------ docker basics
def test_without_docker_installed_every_call_says_so():
    """Without Docker installed every call says so."""
    with _patched(dh, docker_exe=lambda: None):
        assert dh.run_docker(["version"]) == (127, "docker is not installed (or not on PATH)")
        assert dh.stream(["build"], 10)[0] == 127
        passed, detail = dh.daemon_status()
    assert not passed and "not installed" in detail


def test_daemon_status_resources_and_build_jobs():
    """Daemon status resources and build jobs."""
    with _docker(_FakeDocker(running=False)):
        assert dh.daemon_status() == (
            False,
            "Docker is installed but not running. Start Docker Desktop and try again.",
        )
        assert dh.engine_resources() == (0, 0)
        assert dh.build_jobs() == dh.DEFAULT_BUILD_JOBS
    with _docker(_FakeDocker(running=True)):
        assert dh.daemon_status() == (True, "Docker 29.1.0")
        assert dh.engine_resources() == (8, 12 * 1024 ** 3)
        assert dh.build_jobs() == 8  # 12 GB / 1.5 GB, capped by 8 CPUs


def test_timeouts_and_start_failures_are_reported():
    """Timeouts and start failures are reported."""
    with _docker(_FakeDocker(timeouts=("version",))):
        code, out = dh.run_docker(["version"], timeout=7)
    assert code == 124 and "timed out after 7s" in out

    class _Raises(_FakeSubprocess):
        def run(self, argv, **kw):
            raise OSError("access denied")

        def Popen(self, argv, **kw):
            raise OSError("access denied")
    with _patched(dh, docker_exe=lambda: "docker", subprocess=_Raises(_FakeDocker())):
        assert dh.run_docker(["version"])[0] == 127
        assert dh.stream(["build"], 10)[0] == 127
    lines = []
    with _docker(_FakeDocker(), popen_timeout=True):
        code, out = dh.stream(["build"], 3, log=lines.append)
    assert code == 124 and out.endswith("timed out after 3s") and lines


def test_the_build_image_is_built_once():
    """The build image is built once."""
    fake = _FakeDocker(image=False)
    lines = []
    with _docker(fake):
        assert dh.image_exists(dh.image_tag()) is False
        passed, tag = dh.build_image(log=lines.append)
        assert passed and tag == dh.image_tag() and dh.image_exists(tag)
        assert dh.build_image() == (True, tag)                              # second time: no build
    assert [c[0] for c in fake.calls].count("build") == 1
    assert any("Building the Docker build image" in line for line in lines)


# --------------------------------------------------------- the build container
def test_the_container_is_created_started_or_recreated():
    """The container is created started or recreated."""
    root = _checkout()
    fake = _FakeDocker(image=True)
    with _docker(fake):
        host = dh.DockerHost(root)
        assert host._state() == (None, None, "")
        passed, name = host.ensure_container()
        assert passed and name == host.name
        run = [c for c in fake.calls if c[0] == "run"][0]
        assert f"type=volume,source={host.volume},target=/work" in run and "sleep" in run
        assert host.ensure_container() == (True, host.name)                  # cached
        assert host._state()[0] == "running"

        fake.container = "exited"
        assert host._ensure_container() == (True, host.name)
        assert [c for c in fake.calls if c[0] == "start"]

        fake.container_image = "old-image"  # an image of an earlier Dockerfile
        assert host._ensure_container()[0]
        assert [c for c in fake.calls if c[0] == "rm"] and fake.container_image == dh.image_tag()

    fake = _FakeDocker(image=False)
    with _docker(fake), _patched(dh, build_image=lambda log=None: (False, "no space left")):
        passed, why = dh.DockerHost(root)._ensure_container()
    assert not passed and "did not build" in why and "no space left" in why


def test_exec_passes_the_workdir_env_and_proxy_names_only():
    """Exec passes the workdir env and proxy names only."""
    root = _checkout()
    fake = _FakeDocker(image=True)
    lines = []
    with (
        _docker(fake),
        _patched(dh, proxy_env=lambda: {"HTTPS_PROXY": "http://user:pw@proxy:8080"}),
    ):
        host = dh.DockerHost(root)
        code, _out = host.exec(
            ["echo", "a b"], workdir="/work/src/x", env={"K": "v"}, log=lines.append
        )
    assert code == 0 and lines == ["ok"]
    call = [c for c in fake.calls if c[0] == "exec"][-1]
    assert call[:3] == ["exec", "-w", "/work/src/x"] and ["-e", "K=v"] == call[3:5]
    assert "HTTPS_PROXY" in call and not any("pw@" in a for a in call)
    assert call[-1] == "echo 'a b'"
    with _docker(_FakeDocker(image=True)), _patched(dh, build_image=lambda log=None: (False, "x")):
        assert dh.DockerHost(root).exec("true")[0] == 125


def test_the_checkout_copy_is_synced_file_by_file():
    """The checkout copy is synced file by file."""
    root = _checkout()
    fake = _FakeDocker(image=True)
    lines = []
    with _docker(fake):
        host = dh.DockerHost(root)
        passed, detail = host.sync_checkout(lines.append)
        assert passed and fake.untarred > 0 and fake.manifest_id
        first = fake.untarred
        assert host.sync_checkout() == (True, "the copy of the checkout is up to date")
        assert fake.untarred == first

        (root / "nntrainer" / "a.cpp").write_text("int a = 2;\n", encoding="utf-8")
        (root / "jni" / "prepare_lf.sh").unlink()
        passed, detail = host.sync_checkout(lines.append)
        assert passed and detail == "1 file(s) copied" and fake.untarred > first
        assert any(
            c[-1].startswith("rm -f ") and "prepare_lf.sh" in c[-1]
            for c in fake.calls
            if c[0] == "exec"
        )
    assert any("Copying" in line and "removing 1" in line for line in lines)

    with _docker(_FakeDocker(image=True, fail=("untar",))):
        passed, why = dh.DockerHost(_checkout()).sync_checkout()
    assert not passed and "copying the checkout failed" in why
    with _docker(_FakeDocker(image=True, fail=(".src_manifest_id",))):
        assert dh.DockerHost(_checkout()).sync_checkout()[0] is False


def test_push_pull_mirror_missing_and_library_folders():
    """Push pull mirror missing and library folders."""
    root = _checkout()
    fake = _FakeDocker(image=True, missing_out="lib/b.so\n")
    with _docker(fake):
        host = dh.DockerHost(root)
        gen = host.wb_dir / "generated" / "m"
        gen.mkdir(parents=True)
        (gen / "a.cpp").write_text("x", encoding="utf-8")
        assert host.push([]) == (True, "")
        assert host.push([gen])[0] and host.push([gen / "a.cpp"])[0]
        passed, _out = host.pull(gen / "a.cpp")
        assert passed and any("cp -a" in c[-1] for c in fake.calls if c[0] == "exec")
        host.mirror_build()
        assert host.build_dir.is_dir()
        assert host.missing([]) == [] and host.missing(["lib/a.so", "lib/b.so"]) == ["lib/b.so"]
        assert host.lib_dirs() == ["/work/src/build-wb/nntrainer", "/work/src/build-wb/api"]
    with _docker(_FakeDocker(image=True, fail=("[ -e", "*.so*"))):
        host = dh.DockerHost(root)
        assert host.missing(["x"]) == ["x"] and host.lib_dirs() == []

    class _Raises(_FakeSubprocess):
        def run(self, argv, **kw):
            if "-i" in argv:
                raise OSError("pipe broke")
            return super().run(argv, **kw)
    with _patched(dh, docker_exe=lambda: "docker", subprocess=_Raises(_FakeDocker(image=True))):
        assert dh.DockerHost(root)._untar(b"x", "/work/src") == (1, "pipe broke")


def test_setup_configures_the_copy_once():
    """Setup configures the copy once."""
    root = _checkout()
    lines = []
    with _docker(_FakeDocker(running=False)):
        assert dh.setup(root, lines.append)["status"] == "no_docker"
    bare = trm._checkout()
    with _docker(_FakeDocker()):
        status = dh.setup(bare, lines.append)
    assert status["status"] == "missing" and "submodule" in status["missing"][0]
    fake = _FakeDocker()
    with _docker(fake):
        status = dh.setup(root, lines.append)
        assert status["ok"] and status["status"] == "configured" and fake.configured
        assert any(line.startswith("Configuring in Docker:") for line in lines)
        meson = [c for c in fake.calls if c[0] == "exec" and c[-1].startswith("meson setup")]
        assert len(meson) == 1 and "-Denable-test=false" in meson[0][-1]
        assert dh.setup(root, lines.append)["ok"]  # configured: no second meson
        assert (
            len([c for c in fake.calls if c[0] == "exec" and c[-1].startswith("meson setup")]) == 1
        )
    with _docker(_FakeDocker(fail=("meson setup",))):
        assert dh.setup(_checkout(), lines.append)["status"] == "setup_failed"
    with _docker(_FakeDocker(fail=("untar",))):
        assert dh.setup(_checkout(), lines.append)["status"] == "setup_failed"


# ------------------------------------------------------ verify's executors
def test_the_docker_executor_delegates_to_the_build_container():
    """The Docker executor delegates to the build container."""
    root = _checkout()
    fake = _FakeDocker(image=True, missing_out="")
    with _docker(fake):
        ex = verify.DockerExecutor(root)
        assert ex.describe() == "Docker (Ubuntu 24.04, gcc-13)"
        assert verify.DockerExecutor(root, "prof").describe().endswith("nntrainer profiler on)")
        handle = ex.host.wb_dir / "generated" / "x.txt"
        handle.parent.mkdir(parents=True)
        handle.write_text("x", encoding="utf-8")
        assert ex.exists(handle) and ex.lib_paths()[0] == "/work/src/build-wb/nntrainer"
        assert ex.prepare()[0] and ex.push([handle])[0] and ex.pull(handle)[0]
        assert ex.live(handle) == "/wb/generated/x.txt" and ex.missing(["a"]) == []
        ex.mkdirs([ex.host.wb_dir / "generated" / "out"])
        assert (ex.host.wb_dir / "generated" / "out").is_dir()
        ex.after_build()
        assert ex.ninja_file() == ex.build_dir / "build.ninja"


def test_the_native_executor_runs_here():
    """The native executor runs here."""
    root = _checkout()
    ex = verify.NativeExecutor(root)
    work = _tmp()
    assert ex.path(work) == str(work) and ex.live(work) == str(work)
    assert (
        ex.prepare() == (True, "") and ex.push([work]) == (True, "") and ex.pull(work) == (True, "")
    )
    ex.after_build()
    ex.mkdirs([work / "a" / "b"])
    assert (work / "a" / "b").is_dir() and ex.exists(work) and not ex.exists(work / "nope")
    assert ex.missing(["build.ninja"]) == ["build.ninja"] and ex.lib_paths() == []
    assert ex.ninja_file().name == "build.ninja" and ex.describe() in ("native", "native MSVC")
    assert ex.quote("a b") in ('"a b"', "'a b'")
    code, out = ex.run(
        [sys.executable, "-c", "print('one'); print('Note: including file: x'); print('two')"],
        work,
        60,
    )
    assert code == 0 and out.splitlines() == ["one", "two"]  # MSVC include notes dropped
    code, out = ex.run(f'"{sys.executable}" -c "print(42)"', work, 60)
    assert code == 0 and out.strip() == "42"


def test_the_process_runner_reports_timeouts_and_missing_programs():
    """The process runner reports timeouts and missing programs."""
    env = dict(os.environ)
    start = time.monotonic()
    code, out = verify.run_command(
        [sys.executable, "-c", "import time; time.sleep(30)"], _tmp(), env, 1
    )
    assert code == 124 and out.endswith("timed out after 1s")
    assert time.monotonic() - start < 15, "a silent, hanging process must be stopped at the timeout"
    code, out = verify.run_command(["definitely-not-a-program-xyz"], _tmp(), env, 10)
    assert code == 127 and "could not start" in out
    lines = []
    verify.run_command([sys.executable, "-c", "print('a')"], _tmp(), env, 60, lines.append)
    assert lines == ["a"]
    assert verify._which("definitely-not-a-program-xyz", env) is None
    assert verify._which(
        os.path.basename(sys.executable), {"PATH": os.path.dirname(sys.executable)}
    )
    resolved = verify._resolve_command(f'"{sys.executable}" -c 1', env)
    assert (resolved.startswith('"') if os.name == "nt" else resolved[0] == sys.executable)
    assert verify._resolve_command("   ", env) == "   "


def test_tool_environment_and_build_tools_folder():
    """Tool environment and build tools folder."""
    env = verify.tool_env()
    tools = verify.build_tools_dir()
    assert tools is not None and tools.name == ".buildenv"
    assert str(tools / ("Scripts" if os.name == "nt" else "bin")) in env["PATH"].split(os.pathsep)
    assert env["PYTHONIOENCODING"] == os.environ.get(
        "PYTHONIOENCODING", "utf-8"
    )  # a caller's value is kept
    with (
        _patched(verify.compile_check, env_for=lambda cxx, kind: None),
        _patched(
            verify.os, environ={k: v for k, v in os.environ.items() if k != "PYTHONIOENCODING"}
        ),
    ):
        assert verify.tool_env()["PYTHONIOENCODING"] == "utf-8"  # and set when there is none
    assert verify._q("a b") in ('"a b"', "'a b'")


def test_long_paths_and_git_status():
    """Long paths and git status."""
    assert isinstance(verify._long_paths_enabled(), bool)
    bdir = _tmp()
    with _patched(verify, _long_paths_enabled=lambda: False):
        assert verify._paths_too_long(bdir) == 0                            # no build yet
        long_out = "a/" + "x" * 250 + ".obj"
        (bdir / "build.ninja").write_text(f"rule cc\n command = cc $in\nbuild {long_out}: cc in.c\n"
                                          "build short.obj: cc in.c\n", encoding="utf-8")
        assert verify._paths_too_long(bdir) == 1
    with _patched(verify, _long_paths_enabled=lambda: True):
        assert verify._paths_too_long(bdir) == 0
    assert verify._git_status(_tmp()) is None                               # not a git checkout
    with _patched(verify.subprocess, run=lambda *a, **k: (_ for _ in ()).throw(OSError("no git"))):
        assert verify._git_status(_tmp()) is None


# --------------------------------------------------------- link and libraries
def test_the_link_step_uses_the_builds_rule_and_reports_errors():
    """The link step uses the build's rule and reports errors."""
    ninja = verify.Ninja(tv._build_dir() / "build.ninja")
    bdir = _tmp()
    ex = tda._Ex(build_dir=bdir)
    exe = bdir / "unittest_m.exe"
    passed, errors, log = verify.link_objects(ninja, bdir, [bdir / "a.obj"], exe, ex, 60)
    assert passed and errors == [] and "$ link unittest_m.exe (exit 0)" in log
    cmd = ex.calls[0][0]
    assert cmd[0] == "link" and cmd[1].startswith("@")  # an rsp file, as the rule says
    rsp = Path(str(exe) + ".rsp").read_text(encoding="utf-8")
    assert "-lgtest_main" in rsp and "/OUT:" in rsp
    ex = tda._Ex(
        results=[(1, "a.obj : error LNK2019: unresolved external symbol foo\n")], build_dir=bdir
    )
    passed, errors, _log = verify.link_objects(ninja, bdir, [bdir / "a.obj"], exe, ex, 60, extra=[])
    assert not passed and "unresolved external" in errors[0]["message"]
    ex = tda._Ex(results=[(2, "")], build_dir=bdir)
    assert (
        verify.link_objects(ninja, bdir, [bdir / "a.obj"], exe, ex, 60)[1][0]["message"] == "exit 2"
    )


class _LibEx(tda._Ex):
    def __init__(self, kind, missing, results=(), prepared=(True, "")):
        super().__init__(results=results, build_dir=tv._build_dir())
        self.kind, self._missing, self.prepared = kind, list(missing), prepared
        self.env = {"PATH": ""}

    def describe(self):
        """The fake executor's kind."""
        return self.kind

    def missing(self, rels):
        """The build outputs the test marked as missing."""
        return [r for r in rels if r in self._missing]

    def prepare(self, _log=None):
        """The canned result of preparing the build."""
        return self.prepared


def test_libraries_are_built_only_when_missing():
    """Libraries are built only when missing."""
    root = _checkout()
    lines = []
    with _patched(dh, build_jobs=lambda: 4):
        ex = _LibEx("docker", ["nntrainer/libnntrainer.a"])
        status = verify.ensure_libraries(root, lines.append, ex=ex)
        assert ex.calls and ex.calls[0][0][:2] == ["ninja", "-C"] and "-j" in ex.calls[0][0]
        assert any("Building" in line for line in lines)
        ex = _LibEx("docker", ["nntrainer/libnntrainer.a"], results=[(1, "error")])
        status = verify.ensure_libraries(root, lines.append, ex=ex)
        assert status["status"] == "build_failed"
        ex = _LibEx("docker", [], prepared=(False, "Docker stopped"))
        assert verify.ensure_libraries(root, lines.append, ex=ex)["why"] == "Docker stopped"
        ex = _LibEx("docker", [])
        verify.ensure_libraries(root, lines.append, ex=ex)
        assert any("up to date" in line for line in lines)
    ex = _LibEx("native", [])
    assert verify.ensure_libraries(root, lines.append, ex=ex)["ok"] and not ex.calls
    ex = _LibEx("native", ["nntrainer/libnntrainer.a"])
    assert verify.ensure_libraries(root, lines.append, ex=ex)["status"] == "no_ninja"
    nobuild = _LibEx("native", [])
    nobuild.build_dir = _tmp()
    assert verify.ensure_libraries(root, lines.append, ex=nobuild)["status"] == "no_build"


def test_build_setup_early_exits():
    """Build setup early exits."""
    root = _checkout()
    with (
        _patched(verify, select_executor=lambda r, h=None: _LibEx("docker", [])),
        _patched(
            dh,
            setup=lambda r, log: {"ok": False, "status": "no_docker", "missing": ["start Docker"]},
        ),
    ):
        status = verify.setup_build(root, log=lambda m: None)
    assert status["status"] == "no_docker" and status["host"] == "docker"
    with (
        _patched(verify, select_executor=lambda r, h=None: _LibEx("docker", [])),
        _patched(dh, setup=lambda r, log: {"ok": True}),
        _patched(verify, ensure_libraries=lambda r, log, ex=None: {"ok": True, "status": "ready"}),
    ):
        assert verify.setup_build(root, log=lambda m: None)["status"] == "ready"
    with _patched(verify, select_executor=lambda r, h=None: verify.NativeExecutor(r)):
        status = verify.setup_build(root, log=lambda m: None)
    assert status["status"] == "missing" and any("submodule" in m for m in status["missing"])


def test_docker_is_looked_up_on_the_path():
    """Docker is looked up on the path."""
    assert dh.docker_exe() == shutil.which("docker")
