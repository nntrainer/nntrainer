# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file test_run_model.py
# @brief Tests for the Docker build host and running a generated model.
# @author prachi-t17 <prachi.t@samsung.com>

# Tests call private helpers and import the engine from the folder that run_all.py
# puts on sys.path at run time (some only after installing stand-ins), which pylint
# cannot follow when it checks this file on its own.
# pylint: disable=protected-access,import-error,wrong-import-position

"""
Docker build host (knowledge/docker_host.py) and running a generated model
with its real weights (knowledge/run_model.py): path mapping, the CRLF
script overlays, the executor's commands, the runner, the run folder, ldd
parsing, the metrics and the verify gate. Docker is never called. Standard
library only.
"""
import json
import os
import sys
import tempfile
from pathlib import Path
import io
import tarfile

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(_HERE), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import test_codegen as tg  # noqa: E402
import test_compile_and_chat as tcc  # noqa: E402
from knowledge import codegen as cg  # noqa: E402
from knowledge import docker_host as dh  # noqa: E402
from knowledge import run_model as rm  # noqa: E402
from knowledge import verify  # noqa: E402


def _checkout():
    root = Path(tempfile.mkdtemp(dir=tg._TMP))
    (root / "tools" / "agentic-ai-workbench").mkdir(parents=True)
    (root / "jni").mkdir()
    (root / "jni" / "prepare_encoder.sh").write_bytes(b"#!/bin/bash\r\nset -e\r\necho hi\r\n")
    (root / "jni" / "prepare_lf.sh").write_bytes(b"#!/bin/bash\necho fine\n")
    return root


def test_container_paths_map_to_the_copy_and_results_go_to_the_workbench():
    """Container paths map to the copy and results go to the workbench."""
    root = _checkout()
    host = dh.DockerHost(root)
    gen = root / "tools" / "agentic-ai-workbench" / "generated" / "qwen2" / "x.cpp"
    assert host.to_container(gen) == "/work/src/tools/agentic-ai-workbench/generated/qwen2/x.cpp"
    assert host.to_container(root) == "/work/src"
    assert host.to_container(host.build_dir / "build.ninja") == "/work/src/build-wb/build.ninja"
    assert host.live(gen) == "/wb/generated/qwen2/x.cpp"
    for bad in (lambda: host.to_container(Path(tempfile.gettempdir()) / "elsewhere"),
                lambda: host.live(root / "Applications" / "x.h")):
        try:
            bad()
            raise AssertionError("expected DockerError")
        except dh.DockerError:
            pass


def test_the_copy_skips_what_the_linux_build_does_not_use_and_fixes_script_line_endings():
    """The copy skips what the Linux build does not use and fixes script line endings."""
    root = _checkout()
    for rel in (
        ".git/config",
        "tools/agentic-ai-workbench/generated/a.cpp",
        "docs/x.md",
        "subprojects/OpenBLAS/a.c",
        "subprojects/iniparser/src/iniparser.c",
        "Applications/CausalLM/a.cpp",
    ):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("x", encoding="utf-8")
    files = dh.checkout_manifest(root)
    assert (
        "subprojects/iniparser/src/iniparser.c" in files and "Applications/CausalLM/a.cpp" in files
    )
    assert "jni/prepare_encoder.sh" in files
    for rel in (
        ".git/config",
        "tools/agentic-ai-workbench/generated/a.cpp",
        "docs/x.md",
        "subprojects/OpenBLAS/a.c",
    ):
        assert rel not in files, rel
    data = dh.tar_members(root, ["jni/prepare_encoder.sh", "jni/prepare_lf.sh"])
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        script = tar.getmember("jni/prepare_encoder.sh")
        assert (
            tar.extractfile(script).read() == b"#!/bin/bash\nset -e\necho hi\n"
            and script.mode == 0o755
        )
    assert (root / "jni" / "prepare_encoder.sh").read_bytes().count(
        b"\r\n"
    ) == 3  # the checkout is untouched


def test_meson_setup_runs_in_the_copy_with_tests_off():
    """Meson setup runs in the copy with tests off."""
    cmd = dh.meson_setup_command()
    assert cmd[:2] == ["meson", "setup"] and cmd[-1] == "/work/src/build-wb"
    assert "-Denable-test=false" in cmd and "-Denable-transformer=true" in cmd
    assert not any(
        c.startswith("-Dcpp_args") for c in cmd
    )  # json.hpp lands where upstream expects it


def test_the_docker_executor_translates_paths_and_runs_through_docker_exec():
    """The Docker executor translates paths and runs through Docker exec."""
    root = _checkout()
    ex = verify.DockerExecutor(root)
    calls = []
    ex.host.exec = lambda cmd, workdir=None, env=None, timeout=0, log=None: calls.append(
        (cmd, workdir, env)
    ) or (0, "")
    work = root / "tools" / "agentic-ai-workbench" / "generated" / "m"
    ex.run(
        ["ldd", ex.path(work / "run")],
        work,
        10,
        env_extra=ex.lib_env(["/work/src/build-wb/nntrainer"]),
    )
    cmd, workdir, env = calls[0]
    assert cmd == ["ldd", "/work/src/tools/agentic-ai-workbench/generated/m/run"]
    assert workdir == "/work/src/tools/agentic-ai-workbench/generated/m"
    assert env == {"LD_LIBRARY_PATH": "/work/src/build-wb/nntrainer"}
    assert not ex.windows and ex.quote("a b") == "'a b'"
    assert verify.link_extras(ex, gtest=False) == ["-pthread"]
    assert verify.link_extras(ex, gtest=True)[:2] == ["-lgtest_main", "-lgtest"]


def test_auto_host_prefers_a_docker_build_that_is_set_up():
    """Auto host prefers a Docker build that is set up."""
    root = _checkout()
    real = dh.daemon_status
    try:
        dh.daemon_status = lambda: (True, "Docker 29")
        assert (
            verify.select_executor(root, "auto").kind == "docker"
        )  # nothing set up, Docker running
        (root / "tools" / "agentic-ai-workbench" / "nb").mkdir()
        (root / "tools" / "agentic-ai-workbench" / "nb" / "build.ninja").write_text(
            "", encoding="utf-8"
        )
        assert verify.select_executor(root, "auto").kind == "native"  # only the native build exists
        bdir = root / "tools" / "agentic-ai-workbench" / "docker" / "build"  # the host-side mirror
        bdir.mkdir(parents=True)
        (bdir / "build.ninja").write_text("", encoding="utf-8")
        assert verify.select_executor(root, "auto").kind == "docker"
        dh.daemon_status = lambda: (False, "not running")
        assert verify.select_executor(root, "auto").kind == "native"
        assert verify.select_executor(root, "docker").kind == "docker"
    finally:
        dh.daemon_status = real


def test_runner_constructs_the_generated_class_and_reports_metrics():
    """Runner constructs the generated class and reports metrics."""
    bank = tg.tc._bank_copy()
    review = tcc._olmo_review(bank)
    src = rm.runner_source(review)
    assert (
        '#include "olmo2_workbench_causallm.h"' in src
        and "std::make_unique<causallm::Olmo2WorkbenchCausalLM>" in src
    )
    for call in (
        "initialize()",
        "load_weight(weights)",
        "repack_weight()",
        "getPerformanceMetrics()",
        "getOutput(0)",
        "ChatTemplate::Exists(dir)",
    ):
        assert call in src, call
    assert (
        'causallm::json req = {{"messages", causallm::json::array({{{"role", "user"}, {"content",'
        ' input}}})}};'
        in src
    )
    assert src.count("{") == src.count("}") and "WORKBENCH_RUN {" in src


def test_run_output_is_parsed_into_tokens_per_second():
    """Run output is parsed into tokens per second."""
    out = (
        'prefill ...\nWORKBENCH_TEXT_BEGIN\nParis.\nWORKBENCH_TEXT_END\n'
        'WORKBENCH_RUN'
        ' {"prefill_tokens":20,"prefill_ms":400,"generation_tokens":64,"generation_ms":3200,'
        '"total_ms":3650,"peak_memory_kb":2500000,"init_ms":12.5,"load_ms":900,'
        '"prompt":"p","text":"Paris."}\n'
    )
    metrics = rm.parse_run(out)
    assert (
        metrics["prefill_tps"] == 50.0
        and metrics["generation_tps"] == 20.0
        and metrics["text"] == "Paris."
    )
    assert "error" in rm.parse_run("Segmentation fault\n")
    assert rm.parse_run('WORKBENCH_RUN {"error":"weights"}')["error"] == "weights"


def test_ldd_output_keeps_every_library_but_glibc():
    """Ldd output keeps every library but glibc."""
    text = """\tlinux-vdso.so.1 (0x00007ffc)
\tlibcausallm.so => /src/tools/agentic-ai-workbench/docker/nb/Applications/CausalLM/libcausallm.so \
(0x7f)
\tlibopenblas.so.0 => /lib/x86_64-linux-gnu/libopenblas.so.0 (0x7f)
\tlibstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x7f)
\tlibc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x7f)
\tlibm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x7f)
\t/lib64/ld-linux-x86-64.so.2 (0x7f)"""
    assert rm.parse_ldd(text) == [
        "/src/tools/agentic-ai-workbench/docker/nb/Applications/CausalLM/libcausallm.so",
        "/lib/x86_64-linux-gnu/libopenblas.so.0",
        "/lib/x86_64-linux-gnu/libstdc++.so.6",
    ]


def test_the_run_folder_gets_the_model_files_and_lengths():
    """The run folder gets the model files and lengths."""
    bank = tg.tc._bank_copy()
    review = tcc._olmo_review(bank)
    model = Path(tempfile.mkdtemp(dir=tg._TMP))
    for name, text in (
        ("config.json", "{}"),
        ("tokenizer.json", "{}"),
        ("tokenizer_config.json", "{}"),
    ):
        (model / name).write_text(text, encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"x")
    run_dir = Path(tempfile.mkdtemp(dir=tg._TMP)) / "model"
    nntr = json.loads(
        rm.prepare_run_dir(review, model, run_dir, run_dir / "nntr_olmo2_fp32.bin", 100).read_text()
    )
    assert (
        nntr["model_file_name"] == "nntr_olmo2_fp32.bin"
        and nntr["tokenizer_file"] == "tokenizer.json"
    )
    assert nntr["num_to_generate"] == 100 and nntr["max_seq_len"] >= nntr["init_seq_len"] + 100
    assert {"config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"} <= \
        {p.name for p in run_dir.iterdir()}
    assert rm._has_weights(model) and rm.find_model_dir({"model": str(model)}) == model


def test_only_verified_code_runs():
    """Only verified code runs."""
    bank = tg.tc._bank_copy()
    review = tcc._olmo_review(bank)
    try:
        rm.run_generated(review["review_path"], log=lambda m: None)
        raise AssertionError("expected CodegenError")
    except cg.CodegenError as exc:
        assert "verify" in str(exc).lower()


def test_profile_event_carries_the_run_numbers():
    """Profile event carries the run numbers."""
    rec = {
        "ok": True,
        "device": "this PC",
        "summary": {"generation_tps": 20.0, "prefill_tps": 50.0},
        "peak_memory_kb": 2_500_000,
        "runs": [{"generation_tps": 20.0, "prefill_ms": 400}],
        "text": "Paris.",
    }
    events = rm.profile_event(rec, "Qwen/Qwen2.5-0.5B-Instruct")
    assert (
        events["kind"] == "llm_run"
        and events["ran"]
        and events["summary"]["generation_tps"] == 20.0
    )
    assert events["runs"][0]["prefill_ms"] == 400 and events["text"] == "Paris."
