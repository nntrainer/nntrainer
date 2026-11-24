# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file test_runs_and_tools.py
# @brief Tests for running generated models and the workbench's smaller tools.
# @author prachi-t17 <prachi.t@samsung.com>

# Tests call private helpers and import the engine from the folder that run_all.py
# puts on sys.path at run time (some only after installing stand-ins), which pylint
# cannot follow when it checks this file on its own.
# pylint: disable=protected-access,import-error,wrong-import-position,import-outside-toplevel

"""
Running generated models and the workbench's smaller tools, with Docker, the
network, Hugging Face and compilers faked:
- knowledge/run_model.py: weights (download, conversion and its reuse), the
  runner bundle and its reuse key, staging into Docker, the profiling build
  executor, and a whole run_generated with its averages;
- knowledge/compile_check.py's MSVC discovery and header folders;
- knowledge/preflight.py's defaults (a local HTTP server stands in for the
  internet) and its command line;
- knowledge/code_bank.py's embedding backends (a local Ollama stand-in) and
  command line; knowledge/checklist.py's Hub reader and command line;
  knowledge/capture_causallm.py's command line;
- the two entry scripts the extension starts, and that every command a message
  names really exists. Standard library only.
"""
import contextlib
import http.server
import io
import json
import os
import re
import sys
import tempfile
import threading
import types
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENGINE = os.path.dirname(_HERE)
for p in (_ENGINE, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import test_checklist as tc  # noqa: E402
import test_codegen as tg  # noqa: E402
import test_compile_and_chat as tcc  # noqa: E402
import test_docker_automation as tda  # noqa: E402
import test_profiling as tp  # noqa: E402
import test_verify as tv  # noqa: E402
from knowledge import code_bank as cb  # noqa: E402
from knowledge import compile_check  # noqa: E402
from knowledge import docker_host as dh  # noqa: E402
from knowledge import preflight  # noqa: E402
from knowledge import run_model as rm  # noqa: E402
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


@contextlib.contextmanager
def _module(name, **attrs):
    """A stand-in module in sys.modules for the duration (an optional package)."""
    old = sys.modules.get(name)
    sys.modules[name] = types.SimpleNamespace(**attrs)
    try:
        yield
    finally:
        if old is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old


def _stdout(func):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = func()
    return result, buf.getvalue()


# ----------------------------------------------------------- a local server
class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):  # pylint: disable=invalid-name  # http.server's method name
        """Answer 200 for /ok and 404 for anything else."""
        self.send_response(200 if self.path == "/ok" else 404)
        self.end_headers()

    def do_POST(self):  # pylint: disable=invalid-name  # http.server's method name
        """Answer an embedding request like a local Ollama would."""
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        vec = [] if body["prompt"] == "empty" else [1.0, float(len(body["prompt"]))]
        data = json.dumps({"embedding": vec}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)


@contextlib.contextmanager
def _server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


# --------------------------------------------------------------- weights
def _review(out_dir: Path, model="Qwen/Qwen2.5-0.5B-Instruct"):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "weight_converter.py").write_text("def save_x(): pass\n", encoding="utf-8")
    return {"model": model, "slug": "qwen2", "out_dir": str(out_dir), "files": {}}


def test_small_run_helpers():
    """Small run helpers."""
    assert re.match(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", rm._now())
    assert (
        rm._last_json("noise\n{\"a\": 1}\n{bad\nend") == {"a": 1} and rm._last_json("none") is None
    )
    tmp = _tmp()
    (tmp / "a.safetensors").write_bytes(b"x" * 10)
    (tmp / "b.safetensors").write_bytes(b"x" * 5)
    assert rm.weights_size(tmp) == 15
    review = _review(_tmp() / "gen")
    weights_key = rm._weights_key(review, tmp)
    (tmp / "b.safetensors").write_bytes(b"x" * 6)
    assert rm._weights_key(review, tmp) != weights_key  # a weight changed


def test_weights_are_downloaded_into_the_workbench_cache():
    """Weights are downloaded into the workbench cache."""
    wb_dir = _tmp()
    with _patched(rm, workbench_dir=lambda: wb_dir):
        try:
            rm.download_weights({"model": str(_tmp())})
            raise AssertionError("a local folder cannot be downloaded")
        except rm.RunError as exc:
            assert "local folder" in str(exc)

        def snapshot(local_dir, allow_patterns, **_kwargs):
            assert "*.safetensors" in allow_patterns
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            (Path(local_dir) / "model.safetensors").write_bytes(b"w")
            (Path(local_dir) / "config.json").write_text("{}")
        lines = []
        with _module("huggingface_hub", snapshot_download=snapshot):
            dest = rm.download_weights({"model": "Qwen/x"}, log=lines.append)
        assert (
            dest == rm.short_weights_dir(wb_dir, "Qwen/x")
            and (dest / "model.safetensors").is_file()
        )
        assert lines and "Downloading the weights of Qwen/x" in lines[0]
        with _module("huggingface_hub", snapshot_download=lambda **k: None):
            try:
                rm.download_weights({"model": "Qwen/empty"}, log=lambda m: None)
                raise AssertionError("no weights downloaded")
            except rm.RunError as exc:
                assert "no *.safetensors" in str(exc)
    with _patched(rm, workbench_dir=lambda: None):
        try:
            rm.download_weights({"model": "Qwen/x"})
        except rm.RunError as exc:
            assert "agentic-ai-workbench" in str(exc)


def test_weights_are_converted_once_per_version():
    """Weights are converted once per version."""
    review = _review(_tmp() / "gen")
    weights, run_dir = _tmp(), _tmp() / "run"
    (weights / "model.safetensors").write_bytes(b"w" * 100)
    calls, lines = [], []

    def fake_run(cmd, *_args, **_kwargs):
        calls.append(cmd)
        Path(cmd[cmd.index("--out") + 1]).write_bytes(b"bin")
        return (
            0,
            (
                'converting\n{"ok": true, "tensors": 290, "converter_fn":'
                ' "save_qwen2_for_nntrainer", "bytes": 2000000000}'
            ),
        )
    with _patched(verify, run_command=fake_run):
        out = rm.convert_weights(review, weights, run_dir, log=lines.append)
        assert (
            out.name == "nntr_qwen2_fp32.bin" and len(calls) == 1 and "--real-weights" in calls[0]
        )
        assert rm.convert_weights(review, weights, run_dir, log=lines.append) == out
        assert len(calls) == 1 and any("Reusing the converted weights" in line for line in lines)
    with _patched(
        verify, run_command=lambda *a, **k: (1, '{"ok": false, "error": "KeyError: lm_head"}')
    ):
        try:
            rm.convert_weights(review, weights, _tmp() / "run2", log=lambda m: None)
            raise AssertionError("a failed conversion must raise")
        except rm.RunError as exc:
            assert "KeyError: lm_head" in str(exc)


# ----------------------------------------------------------- runner bundle
class _BundleEx(tda._Ex):
    kind, variant = "native", ""

    def __init__(self, root, results=(), push_ok=True):
        super().__init__(results=results, build_dir=tv._build_dir())
        self.root, self.push_ok = root, push_ok
        self.host = None                  # a DockerHost when a test plays the Docker executor

    @staticmethod
    def describe():
        """The fake host, for the log."""
        return "fake host"

    def push(self, files):
        self.pushed += [Path(f) for f in files]
        return (True, "") if self.push_ok else (False, "no space")

    def run(self, cmd, *_args, env_extra=None, **_kwargs):
        self.calls.append((cmd, env_extra))
        if self.results:
            return self.results.pop(0)
        if isinstance(cmd, list) and cmd[0] == "ldd":
            return 0, "\tlibnntrainer.so => /work/src/build-wb/nntrainer/libnntrainer.so (0x1)\n" \
                      "\tlibc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x2)\n"
        return 0, ""


def _gen_review(bank):
    review = tcc._olmo_review(bank)
    return review


def test_the_runner_is_bundled_once_and_rebuilt_when_the_code_changes():
    """The runner is bundled once and rebuilt when the code changes."""
    bank = tc._bank_copy()
    review = _gen_review(bank)
    root = _tmp()
    status = {"ok": True, "missing": [], "build_dir": str(_tmp()), "libs": [], "status": "ready"}
    lines = []
    with _patched(
        verify,
        build_status=lambda r, ex=None, host=None: status,
        compile_source=lambda *a, **k: (True, [], "compiled"),
        link_objects=lambda *a, **k: (True, [], "linked"),
    ):
        ex = _BundleEx(root)
        deploy, digest = rm.bundle(review, ex, log=lines.append)
        assert (
            (deploy / "Dockerfile").is_file()
            and (deploy / ".bundle_key").is_file()
            and len(digest) == 12
        )
        copy_cmds = [c for c, _e in ex.calls if isinstance(c, str) and c.startswith("cp -L")][0]
        assert (
            "libnntrainer.so" in copy_cmds and "libc.so.6" not in copy_cmds
        )  # glibc stays the image's own
        again = _BundleEx(root)
        assert rm.bundle(review, again, log=lines.append) == (deploy, digest)
        assert not again.calls and any("Reusing the built runner" in line for line in lines)

        for results, why in (([(1, "ldd: not a dynamic executable")], "ldd failed"),
                             ([(0, "\tlibfoo.so => not found\n")], "needs libraries"),
                             ([(0, ""), (1, "cp: no space")], "collecting the runner")):
            (deploy / ".bundle_key").unlink(missing_ok=True)
            try:
                rm.bundle(review, _BundleEx(root, results=list(results)), log=lines.append)
                raise AssertionError(why)
            except rm.RunError as exc:
                assert why in str(exc)
        try:
            rm._build_runner(review, _BundleEx(root, push_ok=False), deploy, lines.append, 60)
            raise AssertionError("push failure")
        except rm.RunError as exc:
            assert "copying the generated code" in str(exc)
    bad = {"file": "run_qwen2.cpp", "line": 3, "message": "boom"}
    with _patched(
        verify,
        build_status=lambda r, ex=None, host=None: status,
        compile_source=lambda *a, **k: (False, [bad], "x"),
        link_objects=lambda *a, **k: (True, [], ""),
    ):
        try:
            rm._build_runner(review, _BundleEx(root), deploy, lines.append, 60)
        except rm.RunError as exc:
            assert "run_qwen2.cpp:3: boom" in str(exc)
    with _patched(verify, build_status=lambda r, ex=None, host=None: status,
                  compile_source=lambda *a, **k: (True, [], "x"),
                  link_objects=lambda *a, **k: (False, [{"message": "undefined reference"}], "")):
        try:
            rm._build_runner(review, _BundleEx(root), deploy, lines.append, 60)
        except rm.RunError as exc:
            assert "did not link" in str(exc)
    with _patched(
        verify, build_status=lambda r, ex=None, host=None: dict(status, ok=False, why="no build")
    ):
        try:
            rm._build_runner(review, _BundleEx(root), deploy, lines.append, 60)
        except rm.RunError as exc:
            assert "no build" in str(exc)


def test_the_bundle_key_follows_the_docker_copy_of_the_checkout():
    """The bundle key follows the Docker copy of the checkout."""
    bank = tc._bank_copy()
    review = _gen_review(bank)
    root = _tmp()
    status = {
        "ok": True,
        "missing": [],
        "build_dir": str(_tmp()),
        "libs": ["lib/a.so"],
        "status": "ready",
    }
    ex = _BundleEx(root)
    ex.kind = "docker"
    ex.host = dh.DockerHost(root)
    with _patched(verify, build_status=lambda r, ex=None, host=None: status):
        bundle_key = rm._bundle_key(review, ex)
        mfile = ex.host.wb_dir / "docker" / "src_manifest.json"
        mfile.parent.mkdir(parents=True)
        mfile.write_text(json.dumps({"id": "abc12345"}), encoding="utf-8")
        assert rm._bundle_key(review, ex) != bundle_key
    tmp = _tmp()
    (tmp / "x").write_text("1")
    (tmp / ".bundle_key").write_text("ignored")
    digest = rm._dir_digest(tmp)
    (tmp / ".bundle_key").write_text("changed")
    assert rm._dir_digest(tmp) == digest


def test_staging_copies_the_weights_only_when_they_changed():
    """Staging copies the weights only when they changed."""
    lines = []

    # pylint: disable-next=too-few-public-methods
    class Host:
        """Stand-in Docker host whose staging script copies the weights."""
        @staticmethod
        def exec(script, *_args, log=None, **_kwargs):
            """Check the staging script, report the copy and succeed."""
            assert "cmp -s" in script and "/work/models/qwen2" in script
            log("COPYING_WEIGHTS")
            return 0, ""

    ex = types.SimpleNamespace(host=Host(), live=lambda p: "/wb/run")
    assert rm.stage_model(ex, _tmp(), "qwen2", lines.append) == "/work/models/qwen2"
    assert any("Copying the converted weights" in line for line in lines)

    # pylint: disable-next=too-few-public-methods
    class Failing:
        """Stand-in Docker host whose disk is full."""
        @staticmethod
        def exec(*_args, **_kwargs):
            """Fail as a full disk would."""
            return 1, "no space left on device"
    try:
        rm.stage_model(types.SimpleNamespace(host=Failing(), live=lambda p: "/wb/run"), _tmp(), "q")
    except rm.RunError as exc:
        assert "no space left" in str(exc)


def test_the_profiling_build_is_set_up_on_first_use():
    """The profiling build is set up on first use."""
    root = _tmp()
    lines = []
    with _patched(dh, daemon_status=lambda: (False, "down"),
                  setup=lambda r, log, variant="": {"ok": variant == "prof"}):
        ex = rm.profile_executor(root, lines.append)
    assert ex.variant == "prof" and any("second nntrainer build" in line for line in lines)
    with _patched(dh, daemon_status=lambda: (False, "down"),
                  setup=lambda r, log, variant="": {"ok": False, "log": "meson failed"}):
        try:
            rm.profile_executor(root, lines.append)
        except rm.RunError as exc:
            assert "meson failed" in str(exc)


def test_a_whole_run_records_the_average_of_the_runs():
    """A whole run records the average of the runs."""
    bank = tc._bank_copy()
    review = _gen_review(bank)
    path = review["review_path"]
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    data["verify"] = {"status": "passed", "files_hash": verify.current_verify_hash(review)}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle)
    weights = _tmp()
    outs = [tp._run_line(900), tp._run_line(100), tp._run_line(300)]        # warm-up, then two runs
    seen = []

    def fake_stream(args, *_args, **_kwargs):
        seen.append(args)
        return 0, outs[len(seen) - 1]

    def fake_convert(_review, _model_dir, run_dir, *_args, **_kwargs):
        run_dir.mkdir(parents=True, exist_ok=True)  # as the real conversion does
        return run_dir / "nntr.bin"
    with (
        _patched(dh, daemon_status=lambda: (True, "Docker 29"), stream=fake_stream),
        _patched(
            rm,
            find_model_dir=lambda r, hint=None: weights,
            check_memory=lambda *a: (0, 0, ""),
            convert_weights=fake_convert,
            prepare_run_dir=lambda *a: None,
            bundle=lambda review, ex, log, t: (Path("d"), "abc"),
            load_image=lambda review, deploy, digest, log: "nntr-workbench-run:m-abc",
            stage_model=lambda ex, rd, slug, log: "/work/models/m",
        ),
    ):
        with tcc._env(WORKBENCH_CODE_BANK=str(bank.path)):
            out = rm.run_generated(path, prompt="Hi", repeat=2, log=lambda m: None)
    rec = out["run"]
    assert rec["ok"] and len(rec["runs"]) == 2 and len(seen) == 3
    assert rec["summary"]["prefill_ms"] == 200.0  # (100 + 300) / 2, warm-up left out


# --------------------------------------------------------------- MSVC lookup
def test_msvc_discovery_and_header_folders():
    """MSVC discovery and header folders."""
    with _patched(compile_check.os, environ=dict(os.environ, **{"ProgramFiles(x86)": str(_tmp())})):
        assert compile_check._vswhere() is None
    assert compile_check.vs_installations() == [] or os.name == "nt"
    if os.name == "nt":
        fake = types.SimpleNamespace(run=lambda *a, **k: types.SimpleNamespace(
            stdout="C:\\VS\\BuildTools\n\n", returncode=0), TimeoutExpired=TimeoutError)
        with _patched(compile_check, _vswhere=lambda: "C:\\vswhere.exe", subprocess=fake):
            assert compile_check.vs_installations() == [Path("C:\\VS\\BuildTools")]
    passed = types.SimpleNamespace(
        run=lambda *a, **k: types.SimpleNamespace(
            stdout="INCLUDE=C:\\inc\nPATH=C:\\bin\n=C:=junk\n", returncode=0
        ),
        TimeoutExpired=TimeoutError,
    )
    with _patched(compile_check, _vswhere=lambda: None, subprocess=passed):
        env = compile_check._capture_env(Path("vcvars64.bat"))
    assert env["INCLUDE"] == "C:\\inc" and "=C:" not in env
    no_include = types.SimpleNamespace(
        run=lambda *a, **k: types.SimpleNamespace(stdout="PATH=x\n", returncode=0),
        TimeoutExpired=TimeoutError,
    )
    with _patched(compile_check, _vswhere=lambda: "C:\\VS\\vswhere.exe", subprocess=no_include):
        assert compile_check._capture_env(Path("vcvars64.bat")) is None
    raising = types.SimpleNamespace(run=lambda *a, **k: (_ for _ in ()).throw(OSError("no cmd")),
                                    TimeoutExpired=TimeoutError)
    with _patched(compile_check, subprocess=raising):
        assert compile_check._capture_env(Path("vcvars64.bat")) is None

    root = _tmp()
    for data in (
        "Applications/CausalLM/layers",
        "api/ccapi/include",
        "nntrainer/tensor",
        "nntrainer/tensor/test",
    ):
        (root / data).mkdir(parents=True)
    (root / "nntrainer" / "tensor" / "t.h").write_text("")
    (root / "nntrainer" / "tensor" / "test" / "x.h").write_text("")
    dirs = compile_check._header_dirs(root)
    assert (
        str(root / "nntrainer" / "tensor") in dirs
        and str(root / "Applications" / "CausalLM") in dirs
    )
    assert not any(d.endswith("test") for d in dirs) and len(dirs) == len(set(dirs))


# ------------------------------------------------------------------ preflight
def test_preflight_defaults_and_command_line():
    """Preflight defaults and command line."""
    with _patched(dh, run_docker=lambda args, timeout=60: (0, " ".join(args))):
        assert preflight._default_docker(["version"], 5) == (0, "version")
    with _server() as url:
        assert preflight._default_http_get(url + "/ok", 5) == (200, "")
        assert preflight._default_http_get(url + "/missing", 5) == (404, "")
    status, why = preflight._default_http_get("http://127.0.0.1:1/", 2)
    assert status is None and why

    result = {
        "ok": True,
        "step": "all",
        "summary": "Ready",
        "checks": [
            {
                "id": "docker_installed",
                "label": "Docker installed",
                "status": "pass",
                "detail": "29",
                "fix": "",
                "needs": [],
            }
        ],
        "blocking": [],
        "counts": {"pass": 1, "warn": 0, "fail": 0, "skip": 0},
    }
    with _patched(preflight, run_checks=lambda root, step, network=None: result):
        code, out = _stdout(lambda: preflight._main(["--json", "--no-network"]))
        assert code == 0 and json.loads(out)["summary"] == "Ready"
        code, out = _stdout(lambda: preflight._main([]))
        assert code == 0 and "Docker installed" in out
    with _patched(preflight, run_checks=lambda root, step, network=None: dict(result, ok=False)):
        assert _stdout(lambda: preflight._main(["--step", "verify"]))[0] == 1


# ------------------------------------------------------------------ code bank
def test_the_embedding_backends():
    """The embedding backends."""
    with _server() as url:
        emb = cb._OllamaEmbedder("nomic", url)
        assert emb.model == "ollama:nomic" and emb.embed(["ab", "abc"]) == [[1.0, 2.0], [1.0, 3.0]]
        try:
            emb.embed(["empty"])
            raise AssertionError("an empty embedding must raise")
        except RuntimeError:
            pass
        cb._EMBEDDER_CACHE.pop("ollama:nomic", None)
        with tcc._env(OLLAMA_HOST=url.replace("http://", "")):
            assert cb.resolve_embedder("ollama:nomic").model == "ollama:nomic"
        cb._EMBEDDER_CACHE.pop("ollama:nomic", None)

    # pylint: disable-next=too-few-public-methods
    class _ST:
        def __init__(self, name):
            self.name = name

        @staticmethod
        def encode(texts):
            """Vectors made from the text lengths."""
            return [[len(t), 0] for t in texts]
    with _module("sentence_transformers", SentenceTransformer=_ST):
        status = cb._SentenceTransformersEmbedder("mini")
        assert status.model == "st:mini" and status.embed(["abc"]) == [[3.0, 0.0]]
        cb._EMBEDDER_CACHE.pop("st:mini", None)
        assert cb.resolve_embedder("st:mini").model == "st:mini"
        cb._EMBEDDER_CACHE.pop("st:mini", None)
    try:
        cb.Embedder().embed(["x"])
        raise AssertionError("the interface has no backend")
    except NotImplementedError:
        pass


def test_the_code_bank_command_line_and_context_manager():
    """The code bank command line and context manager."""
    bank = tc._bank_copy()
    db_path = str(bank.path)
    with cb.CodeBank(db_path) as builder:
        total = builder.stats()["total"]
        some = builder.list(kind="hook")[0]["id"]
    code, out = _stdout(lambda: cb._main(["--db", db_path, "stats"]))
    assert code == 0 and json.loads(out)["total"] == total
    code, out = _stdout(lambda: cb._main(["--db", db_path, "list", "--kind", "hook"]))
    assert code == 0 and some in out
    code, out = _stdout(lambda: cb._main(["--db", db_path, "show", some]))
    assert code == 0 and json.loads(out.split("\n\n")[0])["id"] == some
    assert _stdout(lambda: cb._main(["--db", db_path, "show", "no:such"]))[0] == 1
    code, out = _stdout(
        lambda: cb._main(
            ["--db", db_path, "search", "attention", "--embeddings", "off", "--limit", "2"]
        )
    )
    assert code == 0 and len(out.strip().splitlines()) == 2


# ------------------------------------------------------------------ checklist
def test_the_checklist_reads_the_hub_and_has_a_command_line():
    """The checklist reads the Hub and has a command line."""
    folder = _tmp()
    cfg = tc._base_cfg(architectures=["Qwen3ForCausalLM"], model_type="qwen3")
    (folder / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    def download(*_args, **_kwargs):
        return str(folder / "config.json")

    tensors = {"model.embed_tokens.weight": types.SimpleNamespace(shape=[32, 64])}
    meta = types.SimpleNamespace(
        files_metadata={"model.safetensors": types.SimpleNamespace(tensors=tensors)}
    )
    with _module(
        "huggingface_hub", hf_hub_download=download, get_safetensors_metadata=lambda *a, **k: meta
    ):
        facts = tc.ck.facts_from_hub("Qwen/Qwen3-test")
    assert (
        facts.tensors == {"embed_tokens.weight": [32, 64]}
        and facts.source == "Hub safetensors headers"
    )

    bank = tc._bank_copy()
    snap = _tmp()
    tc._write_safetensors(os.path.join(snap, "model.safetensors"), tc._decoder(qk_norm="per_head"))
    (snap / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    code, out = _stdout(lambda: tc.ck._main([str(snap), "--db", str(bank.path)]))
    assert code == 0 and "(Qwen3ForCausalLM)" in out and "[x] Token embedding" in out
    code, out = _stdout(
        lambda: tc.ck._main([str(snap / "config.json"), "--db", str(bank.path), "--json"])
    )
    assert code == 0 and "report" in json.loads(out)


def test_capture_has_a_command_line():
    """Capture has a command line."""
    from knowledge import capture_causallm
    db_path = str(_tmp() / "bank.db")
    code, out = _stdout(lambda: capture_causallm._main(["--db", db_path, "--json"]))
    summary = json.loads(out)
    assert code == 0 and summary["stats"]["total"] > 0 and summary["bank"].endswith("bank.db")
    code, out = _stdout(lambda: capture_causallm._main(["--db", db_path]))
    assert code == 0 and "Reference-tested classes:" in out


# --------------------------------------------------------------- entry points
def test_the_entry_scripts_the_extension_starts():
    """The entry scripts the extension starts."""
    import orchestrator_main
    calls = []
    with (
        _patched(
            orchestrator_main,
            run_pipeline=lambda *a, **k: calls.append(("run", a, k)),
            run_check_only=lambda *a, **k: calls.append(("check", a, k)),
        ),
        _patched(sys, argv=["orchestrator_main.py", "Qwen/x", "--out", "o", "--install-generated"]),
        tcc._env(ANTHROPIC_API_KEY="k"),
        _module("transformers", AutoModel=object),     # main() checks for it before starting
    ):
        orchestrator_main.main()
        sys.argv = ["orchestrator_main.py", "Qwen/x", "--check-only"]
        orchestrator_main.main()
    assert (
        calls[0][0] == "run"
        and calls[0][1][:3] == ("Qwen/x", "o", "k")
        and calls[0][2]["install_generated_files"]
    )
    assert calls[1][0] == "check"

    import device_profiler_main
    out = _tmp()
    for argv, needle in ((["d"], "usage"), (["d", str(out), "/n"], "No completed pipeline")):
        buf = io.StringIO()
        with _patched(sys, argv=argv), contextlib.redirect_stdout(buf):
            try:
                device_profiler_main.main()
                raise AssertionError("must exit")
            except SystemExit as exc:
                assert exc.code == 1
        assert needle in buf.getvalue()
    (out / "state.json").write_text(
        json.dumps({"cpp_path": str(out / "missing.cpp")}), encoding="utf-8"
    )
    with _patched(sys, argv=["d", str(out), "/n"]):
        try:
            _r, _text = _stdout(device_profiler_main.main)
        except SystemExit as exc:
            assert exc.code == 1
    cpp = out / "generated_model.cpp"
    cpp.write_text("int x;\n")
    (out / "state.json").write_text(json.dumps({"cpp_path": str(cpp)}), encoding="utf-8")
    ran = []
    with _patched(sys, argv=["d", str(out), "/n"]), \
            _patched(device_profiler_main, run_device_profiler=lambda s, p: ran.append((s, p))):
        device_profiler_main.main()
    assert ran[0][1] == "/n"


def test_every_command_a_message_names_exists():
    """Every command a message names exists."""
    with open(os.path.join(os.path.dirname(_ENGINE), "package.json"), encoding="utf-8") as handle:
        titles = [c["title"].replace("AI Compiler Workbench: ", "")
                  for c in json.load(handle)["contributes"]["commands"]]
    for root, dirs, files in os.walk(_ENGINE):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", "tests")]
        for name in files:
            if name.endswith(".py"):
                with open(os.path.join(root, name), encoding="utf-8") as handle:
                    for named in re.findall(
                        r"AI Compiler Workbench: ([A-Z][A-Za-z /&,-]+?)(?:['\")]|\*\*| to | --|$)",
                        handle.read(),
                        re.M,
                    ):
                        assert any(
                            t.startswith(named.strip()) for t in titles
                        ), f"{name}: {named!r}"
