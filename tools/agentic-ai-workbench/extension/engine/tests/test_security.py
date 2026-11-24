# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file test_security.py
# @brief Tests for the security and production hygiene of the extension and engine.
# @author prachi-t17 <prachi.t@samsung.com>

# Tests call private helpers and import the engine from the folder that run_all.py
# puts on sys.path at run time (some only after installing stand-ins), which pylint
# cannot follow when it checks this file on its own.
# pylint: disable=import-error,wrong-import-position,import-outside-toplevel

"""
Security and production hygiene of the extension and the engine:
- settings that pick a program or a folder to build, run or write cannot come
  from a repository's .vscode/settings.json, and the extension needs a trusted
  workspace;
- the optional Anthropic API key is read only from VS Code's secret storage
  and reaches the engine only through the environment, never a command line;
- the engine never installs packages by itself; a missing optional package is
  reported with the command that installs it.
Standard library only.
"""
import ast
import contextlib
import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENGINE = os.path.dirname(_HERE)
_EXT = os.path.dirname(_ENGINE)
for p in (_ENGINE, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from agents import optional_deps  # noqa: E402


def _package():
    with open(os.path.join(_EXT, "package.json"), encoding="utf-8") as handle:
        return json.load(handle)


def _extension_js():
    with open(os.path.join(_EXT, "extension.js"), encoding="utf-8") as handle:
        return handle.read()


def _events(func):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = func()
    return result, [
        json.loads(line) for line in buf.getvalue().splitlines() if line.startswith("{")
    ]


# ------------------------------------------------------------ workspace trust
MACHINE_SETTINGS = (
    "pythonPath",
    "venvPath",
    "claudePath",
    "cxxCompiler",
    "buildHost",
    "nntrainerPath",
    "nntrainerRepoPath",
    "causallmPath",
    "causallmProjectRoot",
    "outputDirectory",
    "installGeneratedFiles",
)


def test_settings_that_pick_a_program_or_a_folder_are_machine_only():
    """Settings that pick a program or a folder are machine only."""
    props = _package()["contributes"]["configuration"]["properties"]
    for name in MACHINE_SETTINGS:
        assert props["aiCompilerWorkbench." + name].get("scope") == "machine", name


def test_the_extension_runs_only_in_a_trusted_workspace():
    """The extension runs only in a trusted workspace."""
    caps = _package()["capabilities"]
    assert caps["untrustedWorkspaces"]["supported"] is False
    assert caps["untrustedWorkspaces"]["description"]


def test_no_unused_runtime_dependencies():
    """No unused runtime dependencies."""
    pkg = _package()
    used = set(re.findall(r'require\("([^"]+)"\)', _extension_js()))
    assert used <= {"vscode", "path", "fs", "child_process", "readline", "crypto", "os"}, used
    assert not pkg.get("dependencies"), "extension.js only needs Node's built-ins"


# ------------------------------------------------------------- the API key
def test_the_api_key_is_read_only_from_secret_storage():
    """The API key is read only from secret storage."""
    extension_js = _extension_js()
    assert 'get("anthropicApiKey")' not in extension_js
    assert "context.secrets.get(ANTHROPIC_KEY_SECRET)" in extension_js
    assert extension_js.count("env.ANTHROPIC_API_KEY =") == 1  # only engineEnv() sets it
    props = _package()["contributes"]["configuration"]["properties"]
    old = props["aiCompilerWorkbench.anthropicApiKey"]
    assert old["scope"] == "application" and "markdownDeprecationMessage" in old
    commands = {c["command"] for c in _package()["contributes"]["commands"]}
    assert {
        "aiCompilerWorkbench.setAnthropicApiKey",
        "aiCompilerWorkbench.clearAnthropicApiKey",
    } <= commands


def test_no_key_on_a_command_line():
    """No key on a command line."""
    with open(os.path.join(_ENGINE, "orchestrator_main.py"), encoding="utf-8") as handle:
        assert "--api-key" not in handle.read()
    with open(os.path.join(_ENGINE, "agents", "chat_agent.py"), encoding="utf-8") as handle:
        assert "sys.argv[3]" not in handle.read()


# ------------------------------------------------------- no silent installs
def test_the_engine_never_installs_into_the_users_python():
    """The engine never installs into the user's Python."""
    # The one install allowed: "Set Up nntrainer Build" puts pinned meson + ninja
    # into the workbench's own venv (knowledge/verify.py, tools/.../.buildenv).
    allowed = {os.path.join("knowledge", "verify.py")}
    for root, dirs, files in os.walk(_ENGINE):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", "tests")]
        for name in files:
            rel = os.path.relpath(os.path.join(root, name), _ENGINE)
            if name.endswith(".py") and rel not in allowed:
                with open(os.path.join(root, name), encoding="utf-8") as handle:
                    assert not re.search(r'"pip",\s*"install"', handle.read()), rel


def test_missing_optional_packages_are_named_with_the_install_command():
    """Missing optional packages are named with the install command."""
    assert optional_deps.missing("json", "a_module_that_does_not_exist_xyz") == [
        "a_module_that_does_not_exist_xyz"
    ]
    hint = optional_deps.install_hint()
    assert "pip install -r" in hint and hint.count("requirements-agentic.txt") == 1
    assert os.path.isfile(optional_deps.REQUIREMENTS)


def test_the_api_key_chat_sends_the_same_prompt_as_the_cli_chat():
    """The API key chat sends the same prompt as the CLI chat."""
    from agents import chat_agent

    # pylint: disable-next=too-few-public-methods
    class _Reply:
        content = [{"type": "text", "text": "the answer"}]  # content blocks, as newer models return

    # pylint: disable-next=too-few-public-methods
    class _FakeLLM:
        sent = []

        def invoke(self, messages):
            """Record the messages and return a canned reply."""
            self.sent.append(messages)
            return _Reply()

    out_dir = tempfile.mkdtemp()
    llm = _FakeLLM()
    _r, events = _events(
        lambda: chat_agent.run_api(
            out_dir,
            "what was converted?",
            {"model_name": "Qwen/x", "architecture": "Qwen2"},
            "k",
            llm=llm,
        )
    )
    (system, user), = [(m[0][1], m[1][1]) for m in llm.sent]
    assert system == chat_agent.CHAT_SYSTEM
    assert "Model: Qwen/x (Qwen2)" in user and user.endswith("# Question\nwhat was converted?")
    assert [e["content"] for e in events if e.get("event") == "chat"] == ["the answer"]
    with open(os.path.join(out_dir, chat_agent.HISTORY_FILE), encoding="utf-8") as handle:
        assert json.load(handle) == [{"user": "what was converted?", "assistant": "the answer"}]


def test_every_state_key_a_step_writes_is_declared():
    """Every state key a step writes is declared."""
    # LangGraph passes only the keys PipelineState declares from one step to the next.
    tree = ast.parse(Path(os.path.join(_ENGINE, "agents", "state.py")).read_text(encoding="utf-8"))
    declared = {
        s.target.id
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "PipelineState"
        for s in n.body
        if isinstance(s, ast.AnnAssign)
    }
    agents_dir = os.path.join(_ENGINE, "agents")
    for name in os.listdir(agents_dir):
        if name.endswith(".py"):
            with open(os.path.join(agents_dir, name), encoding="utf-8") as handle:
                for key in re.findall(r'state\["(\w+)"\]\s*=', handle.read()):
                    assert (
                        key in declared
                    ), f"{name} writes state[{key!r}], which PipelineState does not declare"


def test_chat_and_auto_fix_report_missing_langchain_instead_of_installing_it():
    """Chat and auto fix report missing LangChain instead of installing it."""
    from agents import auto_fix, chat_agent
    real = optional_deps.missing
    fake = lambda *mods: list(mods)  # noqa: E731  everything missing
    chat_agent.missing = auto_fix.missing = fake
    try:
        out_dir = tempfile.mkdtemp()
        with open(os.path.join(out_dir, "state.json"), "w", encoding="utf-8") as handle:
            json.dump({"model_name": "m"}, handle)
        os.environ["WORKBENCH_CHAT_BACKEND"] = "api"
        _r, chat = _events(lambda: chat_agent.run(out_dir, "hi", api_key="test-key"))
        state, fix = _events(
            lambda: auto_fix.run(
                {
                    "anthropic_api_key": "test-key",
                    "cpp_code": "x",
                    "compile_log": "error",
                    "cpp_path": os.path.join(out_dir, "a.cpp"),
                }
            )
        )
    finally:
        chat_agent.missing = auto_fix.missing = real
        os.environ.pop("WORKBENCH_CHAT_BACKEND", None)
    said = [e["content"] for e in chat if e.get("event") == "chat"]
    assert len(said) == 1 and "pip install -r" in said[0] and "not installed" in said[0]
    logs = [e["message"] for e in fix if e.get("event") == "log"]
    assert any("pip install -r" in m for m in logs)
    assert [e["status"] for e in fix if e.get("event") == "agent_status"][-1] == "skipped"
    assert state["cpp_code"] == "x" and not os.path.exists(os.path.join(out_dir, "a.cpp"))
