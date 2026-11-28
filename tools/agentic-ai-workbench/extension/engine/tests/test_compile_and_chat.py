# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file test_compile_and_chat.py
# @brief Tests for compile gating, the compile check and the Claude CLI chat.
# @author prachi-t17 <prachi.t@samsung.com>

# Tests call private helpers and import the engine from the folder that run_all.py
# puts on sys.path at run time (some only after installing stand-ins), which pylint
# cannot follow when it checks this file on its own.
# pylint: disable=protected-access,import-error,wrong-import-position,import-outside-toplevel

"""
Compile gating + compile check (knowledge/codegen.py, compile_check.py,
agents/compiler_agent.py) and the Claude-CLI chat (agents/chat_agent.py).
A fake compiler (a small script) stands in for clang++/g++, and a fake CLI
for Claude. Standard library only.
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

import test_codegen as tg  # noqa: E402
from knowledge import codegen as cg  # noqa: E402
from knowledge import compile_check  # noqa: E402

_FAKE_CXX = r'''
import os, sys
src = sys.argv[-1]
if os.environ.get("FAKE_CXX_FAIL"):
    sys.stderr.write(f"{src}:12:5: error: no member named 'NORM_EPSILON' "
                     "in 'causallm::Transformer'\n")
    sys.stderr.write(f"{src}:30:1: error: expected ';' after expression\n")
    sys.exit(1)
sys.exit(0)
'''


def _fake_compiler():
    tmp_dir = tempfile.mkdtemp(dir=tg._TMP)
    script = os.path.join(tmp_dir, "fake_cxx.py")
    with open(script, "w", encoding="utf-8") as handle:
        handle.write(_FAKE_CXX)
    if os.name == "nt":
        exe = os.path.join(tmp_dir, "clang++.cmd")
        with open(exe, "w", encoding="utf-8") as handle:
            handle.write(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n')
    else:
        exe = os.path.join(tmp_dir, "clang++")
        with open(exe, "w", encoding="utf-8") as handle:
            handle.write(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        os.chmod(exe, 0o755)
    return exe


@contextlib.contextmanager
def _env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    for k, val in kv.items():
        if val is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = val
    try:
        yield
    finally:
        for k, val in old.items():
            if val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = val


def _olmo_review(bank):
    return cg.generate(tg._facts(tg.OLMO2, tg.OLMO2_TENSORS), bank, tg._fake_llm(tg._olmo_files),
                       out_root=tg._out(), model_id="allenai/OLMo-2-test", log=lambda m: None)


def _approve_all(path, bank):
    for part in cg.load_review(path)["parts"]:
        if part.get("needs_review"):
            cg.set_part_status(path, part["id"], "approve", "reviewer", bank)
    return cg.load_review(path)


# ------------------------------------------------------------------ gating
def test_compile_waits_until_nothing_needs_review():
    """Compile waits until nothing needs review."""
    bank = tg.tc._bank_copy()
    review = _olmo_review(bank)
    assert review["compile_ready"]["ok"] is False and review["compile_ready"]["why"].startswith(
        "waiting for review"
    )
    assert review["compile"]["status"] == "not_run"
    try:
        cg.compile_generated(review["review_path"])
        raise AssertionError("compiled code that still needs review")
    except cg.CodegenError as exc:
        assert "waiting for review" in str(exc)
    review = _approve_all(review["review_path"], bank)
    assert review["compile_ready"] == {"ok": True, "why": "ready to compile"}


def test_a_rejected_part_blocks_compiling():
    """A rejected part blocks compiling."""
    bank = tg.tc._bank_copy()
    review = _olmo_review(bank)
    _approve_all(review["review_path"], bank)
    review = cg.set_part_status(
        review["review_path"], "createAttention", "reject", "reviewer", bank
    )
    assert review["compile_ready"]["ok"] is False and "rejected" in review["compile_ready"]["why"]


def test_known_model_is_ready_to_compile_right_away():
    """Known model is ready to compile right away."""
    bank = tg.tc._bank_copy()
    review = cg.generate(
        tg._facts(tg.tc.QWEN3, tg.tc._decoder(qk_norm="per_head")),
        bank,
        tg._fake_llm(lambda: tg._qwen3_reply(bank)),
        out_root=tg._out(),
        log=lambda m: None,
    )
    assert review["compile_ready"]["ok"] is True


# ------------------------------------------------------------- compile check
def test_compile_passes_and_failures_are_parsed_and_sent_to_claude():
    """Compile passes and failures are parsed and sent to Claude."""
    bank = tg.tc._bank_copy()
    review = _olmo_review(bank)
    path = review["review_path"]
    _approve_all(path, bank)
    cxx = _fake_compiler()
    root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
    )
    with _env(WORKBENCH_CXX=cxx, FAKE_CXX_FAIL=None, NNTRAINER_ROOT=root):
        review = cg.compile_generated(path)
    compile_res = review["compile"]
    assert (
        compile_res["status"] == "ok"
        and compile_res["ok"]
        and "olmo2_workbench_causallm.cpp" in compile_res["files"]
    ), compile_res
    with _env(WORKBENCH_CXX=cxx, FAKE_CXX_FAIL="1", NNTRAINER_ROOT=root):
        review = cg.compile_generated(path)
    compile_res = review["compile"]
    assert compile_res["status"] == "failed" and len(compile_res["errors"]) >= 2
    assert (
        compile_res["errors"][0]["line"] == 12
        and "NORM_EPSILON" in compile_res["errors"][0]["message"]
    )
    llm = tg._fake_llm(tg._olmo_files)
    cg.fix_compile_errors(path, bank, llm, log=lambda m: None)
    prompt = llm.calls[0][1]
    assert (
        "does not compile" in prompt
        and "NORM_EPSILON" in prompt
        and "# Fix the compile errors" in prompt
    )


def test_editing_the_files_invalidates_the_compile_result():
    """Editing the files invalidates the compile result."""
    bank = tg.tc._bank_copy()
    review = _olmo_review(bank)
    path, out = review["review_path"], review["out_dir"]
    _approve_all(path, bank)
    with _env(WORKBENCH_CXX=_fake_compiler(), FAKE_CXX_FAIL=None):
        assert cg.compile_generated(path)["compile"]["status"] == "ok"
    cpp = os.path.join(out, "olmo2_workbench_causallm.cpp")
    with open(cpp, "a", encoding="utf-8") as handle:
        handle.write("\n// edited\n")
    review = cg.recheck(path, bank)
    assert review["compile"]["status"] == "not_run"


def test_no_compiler_is_reported_not_hidden():
    """No compiler is reported not hidden."""
    real = compile_check.find_compiler
    compile_check.find_compiler = lambda explicit=None: None
    try:
        check = compile_check.check([], ".")
    finally:
        compile_check.find_compiler = real
    assert check["status"] == "no_compiler" and not check["ok"] and "clang" in check["log"]


def test_flags_come_from_an_nntrainer_build_when_there_is_one():
    """Flags come from an nntrainer build when there is one."""
    root = tempfile.mkdtemp(dir=tg._TMP)
    os.makedirs(os.path.join(root, "build"))
    src = os.path.join(
        root, "Applications", "CausalLM", "models", "qwen3", "qwen3_causallm.cpp"
    ).replace("\\", "/")
    cmd = [
        {
            "directory": os.path.join(root, "build"),
            "file": src,
            "arguments": [
                "c++",
                "-Inntrainer/include",
                "-DENABLE_FP16=1",
                "-std=c++17",
                "-O2",
                "-o",
                "x.o",
                "-c",
                src,
            ],
        }
    ]
    with open(
        os.path.join(root, "build", "compile_commands.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(cmd, handle)
    flags, mode = compile_check.flags_for(__import__("pathlib").Path(root), "gnu", ["/gen"])
    assert mode == "compile_commands.json"
    assert (
        "-DENABLE_FP16=1" in flags
        and "-std=c++17" in flags
        and "-O2" not in flags
        and "-c" not in flags
    )
    assert any(
        f.startswith("-I") and f.endswith(os.path.join("build", "nntrainer", "include"))
        for f in flags
    )
    assert flags[0] == "-I/gen"


def test_msvc_outside_a_developer_prompt_is_found_through_vswhere():
    """MSVC outside a developer prompt is found through vswhere."""
    if os.name != "nt":
        return
    inst = tempfile.mkdtemp(dir=tg._TMP)
    build = os.path.join(inst, "VC", "Auxiliary", "Build")
    bindir = os.path.join(inst, "VC", "Tools", "MSVC", "14.50", "bin", "Hostx64", "x64")
    os.makedirs(build)
    os.makedirs(bindir)
    Path(os.path.join(build, "vcvars64.bat")).write_text("", encoding="utf-8")
    Path(os.path.join(bindir, "cl.exe")).write_text("", encoding="utf-8")
    real = (
        compile_check.shutil.which,
        compile_check.vs_installations,
        compile_check._capture_env,
        compile_check._WIN_GNU_PATHS,
        os.environ.pop("CXX", None),
        os.environ.pop("INCLUDE", None),
    )
    captured = []

    def fake_capture(bat):
        captured.append(str(bat))
        return {"Path": bindir, "INCLUDE": r"C:\fake\include", "LIB": r"C:\fake\lib"}

    compile_check._MSVC_ENV.clear()
    compile_check.shutil.which = lambda name, path=None, **kw: (
        real[0](name, path=path) if path else None
    )
    compile_check.vs_installations = lambda: [__import__("pathlib").Path(inst)]
    compile_check._capture_env = fake_capture
    compile_check._WIN_GNU_PATHS = ()
    try:
        cxx, kind = compile_check.find_compiler()
        env = compile_check.env_for(cxx, kind)
        compile_check.env_for(cxx, kind)
    finally:
        (compile_check.shutil.which, compile_check.vs_installations, compile_check._capture_env,
         compile_check._WIN_GNU_PATHS) = real[:4]
        for key, val in (("CXX", real[4]), ("INCLUDE", real[5])):
            if val is not None:
                os.environ[key] = val
        compile_check._MSVC_ENV.clear()
    assert kind == "msvc" and os.path.normcase(cxx) == os.path.normcase(
        os.path.join(bindir, "cl.exe")
    )
    assert compile_check.env_get(env, "include") == r"C:\fake\include"
    assert captured == [os.path.join(build, "vcvars64.bat")]       # vcvars runs once per process
    assert compile_check.env_for("clang++", "gnu") is None


def test_msvc_flags_come_from_the_workbench_build():
    """MSVC flags come from the workbench build."""
    root = tempfile.mkdtemp(dir=tg._TMP)
    bdir = os.path.join(root, "tools", "agentic-ai-workbench", compile_check.BUILD_DIR_NAME)
    os.makedirs(bdir)
    src = os.path.join(root, "Applications", "CausalLM", "models", "qwen3", "qwen3_causallm.cpp")
    inc = os.path.join(root, "Applications", "CausalLM")
    command = (
        f'"cl" "-I{inc}" "-I..\\..\\..\\nntrainer" "/MD" "/nologo" "/showIncludes" "/utf-8" "/W2" '
        f'"/EHsc" "/std:c++20" "/permissive-" "/O2" "-DENABLE_FP16=0" "/arch:AVX2" '
        f'"/FoApplications/x.obj" "/c" "{src}"'
    )
    with open(os.path.join(bdir, "compile_commands.json"), "w", encoding="utf-8") as handle:
        json.dump([{"directory": bdir, "file": src, "command": command}], handle)
    flags, mode = compile_check.flags_for(__import__("pathlib").Path(root), "msvc", ["C:\\gen"])
    assert mode == "compile_commands.json" and flags[0] == "/IC:\\gen"
    for want in (
        "/std:c++20",
        "/EHsc",
        "/utf-8",
        "/permissive-",
        "-DENABLE_FP16=0",
        "/arch:AVX2",
        f"-I{inc}",
    ):
        assert want in flags, want
    for drop in ("/O2", "/c", "/showIncludes", "/MD", "/W2"):
        assert drop not in flags, drop
    assert not any(f.startswith("/Fo") or f.endswith(".cpp") for f in flags)
    rel = next(f for f in flags if f.endswith("nntrainer") and f.startswith("-I"))
    assert os.path.isabs(rel[2:])


# ----------------------------------------------------- pipeline compile step
def test_pipeline_compile_step_waits_for_review():
    """Pipeline compile step waits for review."""
    from agents import compiler_agent
    bank = tg.tc._bank_copy()
    review = _olmo_review(bank)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        state = compiler_agent.run({"cpp_emission_mode": "causallm_component",
                                    "codegen_review_path": review["review_path"]})
    events = [json.loads(line) for line in out.getvalue().splitlines() if line.startswith("{")]
    status = [e for e in events if e["event"] == "agent_status" and e["agent"] == "compiler"][-1]
    assert status["status"] == "skipped" and status["detail"] == "waiting for review"
    assert state["compiled"] is False and "Review tab" in state["compile_log"]


# -------------------------------------------------------------------- chat
# pylint: disable-next=too-few-public-methods
class _FakeCLI:
    def __init__(self):
        self.calls = []

    def run(self, system, user):
        """Record the request and answer with its number."""
        self.calls.append((system, user))
        return f"answer {len(self.calls)}"


def test_chat_answers_with_the_run_as_context_and_remembers_the_conversation():
    """Chat answers with the run as context and remembers the conversation."""
    from agents import chat_agent
    bank = tg.tc._bank_copy()
    review = _olmo_review(bank)
    state = {
        "model_name": "allenai/OLMo-2-test",
        "architecture": "Olmo2ForCausalLM",
        "hf_config": tg.OLMO2,
        "codegen_review_path": review["review_path"],
    }
    out_dir = tempfile.mkdtemp(dir=tg._TMP)
    cli = _FakeCLI()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert chat_agent.run_claude_cli(out_dir, "why does attention need review?", state, cli=cli)
        assert chat_agent.run_claude_cli(out_dir, "and the block?", state, cli=cli)
    system, first = cli.calls[0]
    assert "cannot change files" in system
    assert "createAttention: write -> generated" in first and "waiting for review" in first
    assert "=== olmo2_workbench_causallm.h ===" in first and "Olmo2WorkbenchCausalLM" in first
    second = cli.calls[1][1]
    assert "User: why does attention need review?" in second and "Assistant: answer 1" in second
    replies = [json.loads(line)["content"] for line in buf.getvalue().splitlines()
               if line.startswith("{") and json.loads(line).get("event") == "chat"]
    assert replies == ["answer 1", "answer 2"]
    with open(os.path.join(out_dir, "chat_history.json"), encoding="utf-8") as handle:
        assert len(json.load(handle)) == 2


# --------------------------------------------------- weight download path
def test_weight_folder_avoids_the_windows_path_limit():
    """Weight folder avoids the Windows path limit."""
    from agents import weight_download as wd
    out = os.path.join(
        tempfile.gettempdir(), "nntrainer", "tools", "agentic-ai-workbench", "workbench_output"
    )
    model = "allenai/" + "OLMo-2-0425-1B-" + "x" * 40
    root = os.path.join(os.path.dirname(out), ".ai_compiler_cache", model.replace("/", "__"))
    weights_dir_for = wd.weights_dir_for(root, out, model)
    if os.name == "nt" and not wd._long_paths_enabled():
        assert (
            os.sep + ".w" + os.sep in weights_dir_for
            and len(os.path.basename(weights_dir_for)) == 8
        )
        assert weights_dir_for.startswith(os.path.dirname(out))  # still inside the workbench folder
    else:
        assert weights_dir_for == os.path.join(root, "weights")
    short_model = "Qwen/Qwen3-0.6B"
    short_root = os.path.join(os.path.dirname(out), ".ai_compiler_cache", "Qwen__Qwen3-0.6B")
    if len(os.path.join(short_root, "weights")) + wd._HF_STAGING_CHARS <= wd._WIN_MAX_PATH:
        assert wd.weights_dir_for(short_root, out, short_model) == os.path.join(
            short_root, "weights"
        )
