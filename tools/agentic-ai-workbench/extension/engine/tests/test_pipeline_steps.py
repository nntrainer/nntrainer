# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file test_pipeline_steps.py
# @brief Tests for the pipeline's steps and helpers, with outside dependencies faked.
# @author prachi-t17 <prachi.t@samsung.com>

# Tests call private helpers and import the engine from the folder that run_all.py
# puts on sys.path at run time (some only after installing stand-ins), which pylint
# cannot follow when it checks this file on its own.
# pylint: disable=protected-access,import-error,wrong-import-position,import-outside-toplevel

"""
The pipeline's own steps and helpers, with every outside dependency faked:
the workspace cache, the artifact list, the pipeline state, the event bus
helpers, the IR graph classes, the nntrainer builder (subprocess faked), the
orchestrator (steps replaced by stand-ins, both sequential and through
LangGraph when it is installed), graph Focus, the device profiler's skip paths
and bottleneck estimate, and the Review tab's actions. Standard library only.
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import importlib.util
import tempfile
import time
import types
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(_HERE), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import test_codegen as tg  # noqa: E402
import test_compile_and_chat as tcc  # noqa: E402
from agents import (
    artifact_manager,
    cache,
    codegen_review,
    device_profiler,
    graph_focus_agent,
)  # noqa: E402
from agents import nntrainer_builder as nb  # noqa: E402
from agents import orchestrator as orch  # noqa: E402
from agents.cpp import claude_cli_backend as cli  # noqa: E402
from agents.events import bus  # noqa: E402
from agents.state import new_state  # noqa: E402
from api.graph.edge import GraphEdge  # noqa: E402
from api.graph.graph import Graph  # noqa: E402
from api.graph.node import GraphNode  # noqa: E402
from api.lowering.nntrainer.concurrency import ThreadSafeGraphBuilder  # noqa: E402
from api.lowering.nntrainer.validation import Diagnostic, ValidationResult  # noqa: E402


def _tmp() -> str:
    return tempfile.mkdtemp(dir=tg._TMP)


def _events(func):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = func()
    return result, [
        json.loads(line) for line in buf.getvalue().splitlines() if line.startswith("{")
    ]


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


def _status(events, agent):
    return [
        e["status"] for e in events if e.get("event") == "agent_status" and e.get("agent") == agent
    ]


# ------------------------------------------------------------------ the cache
def test_cache_keys_roots_and_freshness():
    """Cache keys roots and freshness."""
    assert cache.cache_key("Qwen/Qwen3-0.6B") == "Qwen__Qwen3-0.6B"
    assert cache._sanitize("a\\b:c/d") == "a__b_c__d"
    folder = os.path.join(_tmp(), "My Model (v2)")
    os.makedirs(folder)
    key = cache.cache_key(folder)
    assert key.startswith("My_Model_v2_-") and len(key.rsplit("-", 1)[1]) == 8
    assert cache.cache_key(folder) == key                                    # stable
    out = os.path.join(_tmp(), "out")
    root = cache.cache_root_for(out, "Qwen/x")
    assert root == os.path.join(
        os.path.dirname(os.path.abspath(out)), ".ai_compiler_cache", "Qwen__x"
    )

    assert cache.read_meta(root) == {}                                       # nothing yet
    meta = cache.set_entry(root, "graph", {"n": 1})
    assert meta["graph"]["n"] == 1 and cache.read_meta(root)["graph"]["n"] == 1
    assert cache.is_fresh(meta, "graph") and 0 <= cache.age_days(meta, "graph") < 0.01
    old = {"graph": {"cached_at": time.time() - 40 * 86400}}
    assert not cache.is_fresh(old, "graph") and cache.age_days(old, "graph") > 39
    assert not cache.is_fresh({}, "graph") and cache.age_days({}, "graph") == -1
    with open(cache._meta_path(root), "w", encoding="utf-8") as handle:
        handle.write("{not json")
    assert cache.read_meta(root) == {}  # a broken file is ignored


# ------------------------------------------------------- artifacts and state
def test_artifacts_lists_every_file_with_its_kind():
    """Artifacts lists every file with its kind."""
    out = _tmp()
    for rel in (
        "generated_model.cpp",
        "model.ini",
        os.path.join("sub", "w.safetensors"),
        "notes.xyz",
    ):
        os.makedirs(os.path.dirname(os.path.join(out, rel)) or out, exist_ok=True)
        with open(os.path.join(out, rel), "w", encoding="utf-8") as handle:
            handle.write("x")
    state, events = _events(lambda: artifact_manager.run({"out_dir": out}))
    kinds = {a["path"].replace(os.sep, "/"): a["type"] for a in state["artifacts"]}
    assert kinds == {"generated_model.cpp": "C++ Source", "model.ini": "INI File",
                     "sub/w.safetensors": "Weights", "notes.xyz": "File"}
    assert any(e["event"] == "artifacts" and len(e["items"]) == 4 for e in events)
    assert _status(events, "artifact_manager")[-1] == "done"
    state, _e = _events(lambda: artifact_manager.run({"out_dir": os.path.join(out, "missing")}))
    assert state["artifacts"] == []


def test_new_state_starts_safe():
    """New state starts safe."""
    state = new_state("m", "/o", None)
    assert (
        state["compiled"] is False
        and state["errors"] == []
        and state["install_generated_files"] is False
    )
    assert state["generated_header_directory"] == "include/generated"
    assert (
        new_state("m", "/o", "k", install_generated_files=True)["install_generated_files"] is True
    )


# ------------------------------------------------------------------ the bus
def test_every_bus_helper_writes_one_json_event():
    """Every bus helper writes one JSON event."""
    def emit():
        bus.graph({"nodes": []}, "model")
        bus.node_mappings([{"sourceIds": ["a"], "targetIds": ["b"]}])
        bus.artifacts([{"path": "x"}])
        bus.graph_focus("ini", True, ["n1"], "why", source="llm")
        bus.pipeline_complete({"ok": True})
        bus.error("stage", "boom")
    _r, events = _events(emit)
    assert [e["event"] for e in events] == ["graph", "node_mappings", "artifacts", "graph_focus",
                                            "pipeline_complete", "error"]
    assert events[3]["source"] == "llm" and events[5]["stage"] == "stage"
    assert all(isinstance(e["seq"], int) and e["ts"] > 0 for e in events)


# --------------------------------------------------------- IR graph classes
def test_graph_classes_describe_themselves():
    """Graph classes describe themselves."""
    graph = Graph()
    graph.model_name, graph.architecture = "m", "Arch"
    node_a, node_b = GraphNode(name="a", node_type="fc", id="a"), GraphNode(
        name="b", node_type="act", id="b"
    )
    graph.add_node(node_a)
    graph.add_node(node_b)
    graph.connect(node_a, node_b)
    assert len(graph) == 2 and "nodes=2" in repr(graph) and "edges=1" in repr(graph)
    assert "a" in repr(node_a) and "fc" in repr(node_a)
    assert "a" in repr(GraphEdge(source="a", target="b")) and "b" in repr(
        GraphEdge(source="a", target="b")
    )
    builder = ThreadSafeGraphBuilder(graph)
    assert builder.graph is graph
    result = ValidationResult([Diagnostic("warning", 0, "w"), Diagnostic("error", None, "e")])
    assert result.to_dict() == {"ok": False, "diagnostics": [
        {"severity": "warning", "layer_index": 0, "message": "w"},
        {"severity": "error", "layer_index": None, "message": "e"}]}


# --------------------------------------------------------- nntrainer builder
# pylint: disable-next=too-few-public-methods
class _Run:
    """A fake subprocess.run: answers from a script of (returncode, stdout, stderr) per
    command word."""

    def __init__(self, script=None, raise_for=()):
        self.script, self.raise_for, self.calls = script or {}, raise_for, []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        if cmd[0] in self.raise_for:
            raise self.raise_for[cmd[0]]
        code, out, err = self.script.get(cmd[0], (0, "", ""))
        if callable(code):
            code = code()
        return types.SimpleNamespace(returncode=code, stdout=out, stderr=err)


def _repo(git=True, meson=True):
    tmp = _tmp()
    if meson:
        Path(os.path.join(tmp, "meson.build")).write_text("", encoding="utf-8")
    if git:
        os.makedirs(os.path.join(tmp, ".git"))
    return tmp


def test_builder_checks_the_repo_and_the_prefix():
    """Builder checks the repo and the prefix."""
    assert "not a directory" in nb._validate_repo(os.path.join(_tmp(), "nope"))
    assert "meson.build" in nb._validate_repo(_repo(meson=False))
    assert ".git" in nb._validate_repo(_repo(git=False))
    assert nb._validate_repo(_repo()) is None
    prefix = _tmp()
    assert nb._validate_prefix(prefix) == ["include/nntrainer", "lib/"]
    os.makedirs(os.path.join(prefix, "include", "nntrainer"))
    os.makedirs(os.path.join(prefix, "lib"))
    assert nb._validate_prefix(prefix) == []


def test_builder_tools_packages_and_apt_commands():
    """Builder tools packages and apt commands."""
    run = _Run(raise_for={"ninja": FileNotFoundError(), "gcc": subprocess.TimeoutExpired("gcc", 5)})
    with _patched(nb.subprocess, run=run):
        assert nb._missing_tools() == ["ninja", "gcc"]
    run = _Run({"dpkg": (lambda: 1 if len(run.calls) % 2 else 0, "", "")})
    with _patched(nb.subprocess, run=run):
        missing = nb._missing_packages()
    assert 0 < len(missing) < len(nb.APT_DEPS)
    cmds = nb._apt_commands(["meson", "gcc"])
    assert (
        cmds[-1].endswith("install -y --no-install-recommends meson gcc")
        and "ppa:nnstreamer" in cmds[0]
    )


def test_builder_submodules_state_and_retries():
    """Builder submodules state and retries."""
    repo = _repo()
    with _patched(nb.subprocess, run=_Run({"git": (0, "synced\n", "")})):
        assert _events(lambda: nb._sync_submodules(repo))[0] is True
    with _patched(nb.subprocess, run=_Run({"git": (1, "", "fatal: no\n")})):
        passed, events = _events(lambda: nb._sync_submodules(repo))
    assert passed is False and any("fatal" in e.get("message", "") for e in events)
    with _patched(
        nb.subprocess, run=_Run(raise_for={"git": subprocess.TimeoutExpired("git", 300)})
    ):
        assert _events(lambda: nb._sync_submodules(repo))[0] is False

    assert nb._build_state(repo) == "fresh"
    os.makedirs(os.path.join(repo, "build"))
    assert nb._build_state(repo) == "meson_failed"
    with open(os.path.join(repo, "build", "meson-log.txt"), "w", encoding="utf-8") as handle:
        handle.write("Build targets in project: 12\n")
    assert nb._build_state(repo) == "ninja_build"

    flaky = _Run({"ninja": (lambda: 1 if len(flaky.calls) < 2 else 0, "ok\n", "err\n")})
    with _patched(nb.subprocess, run=flaky):
        assert _events(lambda: nb._run_with_retry("ninja install", ["ninja"], repo))[0] == (
            True,
            "",
        )
    with _patched(nb.subprocess, run=_Run({"meson": (1, "", "bad\n")})):
        passed, err = _events(
            lambda: nb._run_with_retry("meson setup", ["meson"], repo, max_retries=1)
        )[0]
    assert passed is False and "bad" in err
    with _patched(
        nb.subprocess, run=_Run(raise_for={"meson": subprocess.TimeoutExpired("meson", 1)})
    ):
        assert (
            _events(lambda: nb._run_with_retry("meson setup", ["meson"], repo, max_retries=0))[0][0]
            is False
        )


def test_builder_prep_and_build_phases():
    """Builder prep and build phases."""
    repo, prefix = _repo(), _tmp()
    assert _events(lambda: nb.run_prep(os.path.join(repo, "nope"), prefix))[0]["ok"] is False
    with _patched(nb, _missing_tools=lambda: ["meson"]):
        assert "meson" in _events(lambda: nb.run_prep(repo, prefix))[0]["error"]
    with _patched(nb, _missing_tools=lambda: [], _missing_packages=lambda: ["gcc"]):
        res = _events(lambda: nb.run_prep(repo, prefix))[0]
    assert res["ok"] and res["apt_commands"][-1].endswith("gcc")
    with _patched(
        nb,
        _missing_tools=lambda: [],
        _missing_packages=lambda: [],
        _sync_submodules=lambda r: False,
    ):
        assert _events(lambda: nb.run_prep(repo, prefix))[0]["ok"] is False
    with _patched(
        nb, _missing_tools=lambda: [], _missing_packages=lambda: [], _sync_submodules=lambda r: True
    ):
        assert _events(lambda: nb.run_prep(repo, prefix))[0] == {
            "ok": True,
            "apt_commands": [],
            "error": None,
        }

    calls = []

    def fake_retry(label, cmd, *_args, **_kwargs):
        calls.append(cmd)
        if label == "ninja install":
            os.makedirs(os.path.join(prefix, "include", "nntrainer"), exist_ok=True)
            os.makedirs(os.path.join(prefix, "lib"), exist_ok=True)
        return True, ""
    with _patched(nb, _run_with_retry=fake_retry):
        res = _events(lambda: nb.run_build(repo, prefix))[0]
    assert res["ok"] and "--reconfigure" not in calls[0] and calls[1][:2] == ["ninja", "-C"]
    os.makedirs(os.path.join(repo, "build"))
    with _patched(
        nb, _run_with_retry=lambda label, cmd, cwd, max_retries=2: (label != "meson setup", "x")
    ):
        res = _events(lambda: nb.run_build(repo, prefix))[0]
    assert res["ok"] is False and res["needs_user_help"] and "meson setup failed" in res["error"]
    with _patched(
        nb, _run_with_retry=lambda label, cmd, cwd, max_retries=2: (label != "ninja install", "y")
    ):
        assert "ninja build failed" in _events(lambda: nb.run_build(repo, prefix))[0]["error"]
    with _patched(nb, _run_with_retry=lambda *a, **k: (True, "")):
        assert "missing" in _events(lambda: nb.run_build(repo, _tmp()))[0]["error"]


# ------------------------------------------------------------- orchestrator
_STEPS = (
    "model_discovery",
    "compatibility",
    "nntrainer_lowering",
    "ini_generator",
    "graph_builder",
    "cpp_generator_agent",
    "causallm_codegen",
    "causallm_install",
    "dual_graph",
    "compiler_agent",
    "auto_fix",
    "profiler_agent",
    "weight_preview",
    "artifact_manager",
)


@contextlib.contextmanager
def _fake_pipeline(order, **overrides):
    mods = {name: getattr(orch, name) for name in _STEPS}
    real = {name: m.run for name, m in mods.items()}

    def step(name):
        def run(state):
            order.append(name)
            return overrides[name](state) if name in overrides else state
        return run
    try:
        for name, mod in mods.items():
            mod.run = step(name)
        yield
    finally:
        for name, mod in mods.items():
            mod.run = real[name]


def test_weight_download_runs_in_the_background_and_is_joined():
    """Weight download runs in the background and is joined."""
    out = _tmp()

    def download(step_state):
        step_state["weights_path"] = "/w"
        return step_state

    def convert(step_state):
        step_state["converted_weights_path"] = "/w.bin"
        step_state["errors"].append("convert: warning")
        return step_state
    with _patched(orch.weight_download, run=download), _patched(orch.weight_converter, run=convert):
        state = {"model_name": "m", "out_dir": out, "errors": []}
        _events(lambda: orch.start_weight_download(state))
        state = _events(lambda: orch.join_weight_download(state))[0]
    assert state["weights_path"] == "/w" and state["converted_weights_path"] == "/w.bin"
    assert state["errors"] == ["convert: warning"]
    assert orch.join_weight_download({"out_dir": out}) == {"out_dir": out}  # nothing pending


def test_the_pipeline_runs_every_step_and_saves_a_key_free_state():
    """The pipeline runs every step and saves a key free state."""
    out, order = _tmp(), []

    def discover(step_state):
        step_state["architecture"] = "Qwen2ForCausalLM"
        return step_state

    def compiled(step_state):
        step_state["compiled"] = True
        return step_state
    with _fake_pipeline(order, model_discovery=discover, compiler_agent=compiled), \
            _patched(orch, start_weight_download=lambda s: s, join_weight_download=lambda s: s):
        state, events = _events(lambda: orch.run_pipeline("Qwen/x", out, api_key="secret-key"))
    assert order[0] == "model_discovery" and order[-1] == "artifact_manager"
    assert "auto_fix" not in order and state["compiled"] is True
    done = [e for e in events if e["event"] == "pipeline_complete"][0]["summary"]
    assert done["architecture"] == "Qwen2ForCausalLM" and done["compiled"] is True
    with open(os.path.join(out, "state.json"), encoding="utf-8") as handle:
        saved = json.load(handle)
    assert "anthropic_api_key" not in saved and saved["architecture"] == "Qwen2ForCausalLM"


def test_a_failed_compile_is_auto_fixed_at_most_twice_when_a_key_is_set():
    """A failed compile is auto fixed at most twice when a key is set."""
    out, order = _tmp(), []

    def fail(state):
        state["compiled"] = False
        state["compile_log"] = "error: x"
        return state

    def fix(state):
        state["fix_iterations"] = state.get("fix_iterations", 0) + 1
        return state
    with _fake_pipeline(order, compiler_agent=fail, auto_fix=fix), \
            _patched(orch, start_weight_download=lambda s: s, join_weight_download=lambda s: s), \
            _patched(orch, build_graph=lambda: (_ for _ in ()).throw(ImportError("no langgraph"))):
        _events(lambda: orch.run_pipeline("m", out, api_key="k"))
    assert order.count("auto_fix") == orch.MAX_FIX_ITERATIONS
    assert order.count("compiler_agent") == orch.MAX_FIX_ITERATIONS + 1


def test_a_crashing_graph_is_reported_and_the_state_still_saved():
    """A crashing graph is reported and the state still saved."""
    out = _tmp()
    with _patched(orch, build_graph=lambda: (_ for _ in ()).throw(RuntimeError("graph broke"))):
        state, events = _events(lambda: orch.run_pipeline("m", out))
    assert "graph broke" in state["errors"] and any(e["event"] == "error" for e in events)
    assert os.path.exists(os.path.join(out, "state.json"))


def test_check_only_reads_headers_generates_and_compiles():
    """Check only reads headers generates and compiles."""
    out, order = _tmp(), []

    def codegen(step_state):
        step_state["codegen_review_path"], step_state["codegen_ok"] = "/r/review.json", True
        return step_state
    with _fake_pipeline(order, causallm_codegen=codegen):
        state, events = _events(
            lambda: orch.run_check_only("m", os.path.join(out, "new"), api_key="k")
        )
    assert (
        order == ["model_discovery", "causallm_codegen", "compiler_agent"] and state["check_only"]
    )
    graphs = [e for e in events if e["event"] == "graph"]
    assert {g["target"] for g in graphs} == {"model", "nntrainer"} and "full Run" in graphs[0][
        "data"
    ]["message"]
    assert any("passed its checks" in e.get("content", "") for e in events if e["event"] == "chat")
    with open(os.path.join(out, "new", "state.json"), encoding="utf-8") as handle:
        assert "anthropic_api_key" not in json.load(handle)
    assert orch._json_safe({"a": {1, 2}, "anthropic_api_key": "k"})["a"].startswith("{")


def test_the_langgraph_pipeline_when_installed():
    """The LangGraph pipeline when installed."""
    if importlib.util.find_spec("langgraph") is None:
        return                                      # optional package: nothing to check here
    out, order = _tmp(), []
    with (
        _fake_pipeline(order),
        _patched(orch, start_weight_download=lambda s: s, join_weight_download=lambda s: s),
    ):
        state, _e = _events(lambda: orch.run_pipeline("m", out))
    assert (
        order[0] == "model_discovery"
        and order[-1] == "artifact_manager"
        and not state.get("errors")
    )


# ------------------------------------------------------- CLI backend helpers
def test_cli_backend_helpers():
    """CLI backend helpers."""
    assert cli._version_key("/x/anthropic.claude-code-2.10.3-win32-x64/claude") == (2, 10, 3)
    assert cli._version_key("/usr/bin/claude") == (0, 0, 0)
    _r, events = _events(lambda: cli._log("hello", "warn"))
    assert events[0]["message"] == "hello" and events[0]["level"] == "warn"


# -------------------------------------------------------------- graph Focus
def _focus_state(out):
    state = {
        "model_graph_view": {
            "nodes": [
                {"id": "embed", "type": "embedding", "label": "embed_tokens"},
                {
                    "id": "q",
                    "type": "fully_connected",
                    "label": "q_proj",
                    "attributes": {"hf_module": "self_attn.q_proj"},
                },
                {"id": "mlp", "type": "fully_connected", "label": "mlp_up"},
            ]
        },
        "device_profile": {
            "layers": [
                {"name": "q_proj", "estimated_ms": 9.0},
                {"name": "mlp_up", "estimated_ms": 1.0},
                {"name": "embed_tokens", "estimated_ms": 0.5},
            ]
        },
    }
    with open(os.path.join(out, "state.json"), "w", encoding="utf-8") as handle:
        json.dump(state, handle)
    return state


def test_graph_focus_reset_rules_keywords_and_the_llm_fallback():
    """Graph focus reset rules keywords and the LLM fallback."""
    out = _tmp()
    _focus_state(out)
    focus = lambda q, key="ini": _events(lambda: graph_focus_agent.run(out, key, q))  # noqa: E731
    assert focus("   ")[0] == {"ok": False}
    res, events = focus("reset")
    assert res["ok"] and events[-1]["source"] == "reset"
    res, events = focus("show the bottleneck")
    assert res["ok"] and events[-1]["matched_ids"] == ["q"]
    res, events = focus("self attention")
    assert res["ok"] and events[-1]["matched_ids"] == ["q"]
    old = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        res, events = focus("zzzz nothing")
        assert res["ok"] is False and "Set Anthropic API Key" in events[-1]["explanation"]
        os.environ["ANTHROPIC_API_KEY"] = "k"
        with _patched(graph_focus_agent, _llm_match=lambda *a: (["mlp"], "")):
            assert focus("zzzz nothing")[1][-1]["matched_ids"] == ["mlp"]
        with _patched(graph_focus_agent, _llm_match=lambda *a: ([], "")):
            assert focus("zzzz nothing")[0]["ok"] is False
        with _patched(
            graph_focus_agent, _llm_match=lambda *a: (None, "langchain-anthropic not installed")
        ):
            assert focus("zzzz nothing")[1][-1]["source"] == "llm"
    finally:
        os.environ.pop("ANTHROPIC_API_KEY", None)
        if old is not None:
            os.environ["ANTHROPIC_API_KEY"] = old
    assert focus("anything", key="cpp")[0]["ok"] is False                 # no nntrainer graph saved
    assert graph_focus_agent._load_state(_tmp()) == {}
    bad = _tmp()
    with open(os.path.join(bad, "state.json"), "w", encoding="utf-8") as handle:
        handle.write("{oops")
    assert graph_focus_agent._load_state(bad) == {}
    state = _focus_state(_tmp())
    state["device_profile"] = {}
    assert graph_focus_agent._bottleneck_match(state, "ini", "slowest") == []
    assert graph_focus_agent._bottleneck_match(state, "cpp", "slowest") is None


def test_graph_focus_llm_match_without_langchain_says_so():
    """Graph focus LLM match without LangChain says so."""
    from agents import optional_deps
    if not optional_deps.missing("langchain_anthropic"):
        return  # installed: the real call would need a key and the network
    ids, reason = graph_focus_agent._llm_match({}, "ini", "q", "k")
    assert ids is None and "not installed" in reason


# ----------------------------------------------------------- device profiler
def test_device_profiler_skips_and_estimates():
    """Device profiler skips and estimates."""
    assert device_profiler.detect_arch() in ("x86_64", "arm64") or device_profiler.detect_arch()
    with _patched(device_profiler.platform, machine=lambda: "AMD64"):
        assert device_profiler.detect_arch() == "x86_64"
    with _patched(device_profiler.platform, machine=lambda: "aarch64"):
        assert device_profiler.detect_arch() == "arm64"
    with _patched(device_profiler.platform, machine=lambda: ""):
        assert device_profiler.detect_arch() == "unknown"
    state, events = _events(
        lambda: device_profiler.run({"cpp_emission_mode": "causallm_component"}, "/n")
    )
    assert state["device_profile_skipped"] and _status(events, "device_profiler")[-1] == "skipped"
    state, events = _events(lambda: device_profiler.run({"compile_skipped": True}, "/n"))
    assert state["device_profile_skipped"]
    state, events = _events(lambda: device_profiler.run({"cpp_path": "/does/not/exist.cpp"}, "/n"))
    assert _status(events, "device_profiler")[-1] == "error"
    cpp = os.path.join(_tmp(), "generated_model.cpp")
    Path(cpp).write_text("", encoding="utf-8")
    with _patched(device_profiler, discover_flags=lambda prefix=None: (None, None, None)):
        state, events = _events(
            lambda: device_profiler.run({"cpp_path": cpp, "out_dir": _tmp()}, "/n")
        )
    assert "nntrainer not found" in " ".join(e.get("message", "") for e in events)
    est = device_profiler._estimate_bottlenecks({"graph_ir": {"nodes": [
        {"name": "a", "node_type": "fc", "supported": True, "parameter_count": 3},
        {"name": "b", "node_type": "fc", "supported": True, "parameter_count": 1},
        {"name": "c", "node_type": "act", "supported": True, "parameter_count": 0}]}}, 8.0)
    assert [(e["name"], e["estimated_ms"], e["share_pct"]) for e in est] == [
        ("a", 6.0, 75.0),
        ("b", 2.0, 25.0),
    ]


# -------------------------------------------------------------- Review tab
def test_review_messages_for_compile_and_verify():
    """Review messages for compile and verify."""
    _r, events = _events(lambda: [codegen_review.report_compile(c) for c in (
        {"status": "ok", "compiler": "/x/clang++", "mode": "build"},
        {"status": "failed", "errors": [{"file": "a.cpp", "line": 3, "message": "boom"}]},
        {"status": "skipped", "log": "no compiler"})])
    msgs = [e["message"] for e in events]
    assert (
        "Compile check passed (clang++" in msgs[0]
        and "a.cpp:3: boom" in msgs[2]
        and msgs[3] == "no compiler"
    )
    _r, events = _events(
        lambda: [
            codegen_review.report_verify(v)
            for v in (
                {
                    "status": "passed",
                    "marked_verified": ["a", "b"],
                    "max_abs_diff": 0.001,
                    "atol": 0.01,
                },
                {"status": "passed", "marked_verified": []},
                {
                    "status": "mismatch",
                    "failures": ["logit 3:  far  off"],
                    "tokens_got": [1],
                    "tokens_ref": [2],
                },
                {
                    "status": "build_failed",
                    "errors": [{"file": "b.cpp", "line": 1, "message": "x"}],
                },
                {"status": "crashed", "log": "segfault"},
            )
        ]
    )
    msgs = [e["message"] for e in events]
    assert "2 code bank entries newly verified" in msgs[0] and "already verified" in msgs[1]
    assert (
        "logit 3: far off" in msgs[3]
        and "b.cpp:1: x" in msgs[5]
        and msgs[6].startswith("Verify crashed")
    )
    assert codegen_review._user()


def test_review_actions_through_the_command_line():
    """Review actions through the command line."""
    bank = tg.tc._bank_copy()
    review = tcc._olmo_review(bank)
    path = review["review_path"]
    part = [p["id"] for p in review["parts"] if p.get("needs_review")][0]
    main = lambda *argv: _events(lambda: codegen_review.main([path, *argv]))  # noqa: E731
    with tcc._env(WORKBENCH_CODE_BANK=str(bank.path)):
        code, events = main("show")
        assert code == 0 and any(e["event"] == "codegen" for e in events)
        assert main("approve")[0] == 1                                    # needs a part id
        code, events = main("approve", part, "--actor", "reviewer")
        assert code == 0 and any("approved by reviewer" in e.get("message", "") for e in events)
        assert main("recheck")[0] == 0
        with tcc._env(WORKBENCH_CXX=tcc._fake_compiler()):
            code, events = main("compile")
            assert code == 1 and any(
                "waiting for review" in e.get("message", "")
                for e in events
                if e["event"] == "error"
            )
            tcc._approve_all(path, bank)
            code, events = main("compile")
        assert code == 0 and any("Compile check passed" in e.get("message", "") for e in events)
        assert main("regenerate")[0] == 1                                 # needs a part id

        from knowledge import preflight
        from knowledge import run_model as rm
        gate = lambda step, log, emit: {
            "ok": False,
            "summary": "Docker is not running",
        }  # noqa: E731
        with _patched(preflight, gate=gate):
            code, events = main("run", "--prompt", "hi")
        assert code == 1 and any(
            e.get("status") == "error" for e in events if e["event"] == "run_status"
        )
        with _patched(
            rm, download_weights=lambda review, log=None: _tmp(), weights_size=lambda d: 2e9
        ):
            code, events = main("fetchweights")
        assert (
            code == 0
            and [e for e in events if e["event"] == "run_status"][0]["status"] == "weights_ready"
        )
        ran = {
            "ok": True,
            "device": "this PC",
            "runs": [{"generation_tps": 10.0}],
            "summary": {"generation_tps": 10.0},
            "text": "Paris",
        }
        with _patched(preflight, gate=lambda step, log, emit: {"ok": True}), \
                _patched(rm, run_generated=lambda *a, **k: dict(review, run=ran)):
            code, events = main("run")
        statuses = [e["status"] for e in events if e["event"] == "run_status"]
        assert code == 0 and statuses == ["running", "done"]
        assert any(e["event"] == "profile" and e["data"]["ran"] for e in events)
        assert any(e.get("message") == "Output: Paris" for e in events)


# ------------------------------------------------------------- old settings
def test_no_message_points_at_the_old_key_setting():
    """No message points at the old key setting."""
    engine = os.path.dirname(_HERE)
    for root, dirs, files in os.walk(engine):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", "tests")]
        for name in files:
            if name.endswith(".py"):
                with open(os.path.join(root, name), encoding="utf-8") as handle:
                    assert (
                        "set aicompilerworkbench.anthropicapikey" not in handle.read().lower()
                    ), name
