# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file test_verify.py
# @brief Tests for Verify: build parsing, link fix-ups, the harness and results.
# @author prachi-t17 <prachi.t@samsung.com>

# Tests call private helpers and import the engine from the folder that run_all.py
# puts on sys.path at run time (some only after installing stand-ins), which pylint
# cannot follow when it checks this file on its own.
# pylint: disable=protected-access,import-error,wrong-import-position

"""
Verify (knowledge/verify.py, knowledge/verify_fixture.py): build.ninja
parsing, the link fix-ups, the harness, the tiny-config shrink, and what a
pass / mismatch / crash does to review.json and the code bank. The compiler,
the fixture script and the harness run are faked. Standard library only.
"""
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
import test_compile_and_chat as tcc  # noqa: E402
from knowledge import codegen as cg  # noqa: E402
from knowledge import verify  # noqa: E402
from knowledge import verify_fixture as vf  # noqa: E402

_NINJA = '''ninja_required_version = 1.8.2

rule cpp_COMPILER
 command = "cl" $ARGS /Fo$out "/c" $in
 deps = msvc

rule cpp_LINKER_RSP
 command = "link" @$out.rsp
 rspfile = $out.rsp
 rspfile_content = $ARGS /MACHINE:x64 /OUT:$out $in_newline $LINK_ARGS

build Applications/CausalLM/models/gpt_oss/gptoss_moe_layer.dll.p/a.obj: cpp_COMPILER \
C$:/Users/Some$ One/nntrainer/Applications/CausalLM/models/gpt_oss/gptoss_moe_layer.cpp
 ARGS = "-Ilayers_only"

build Applications/CausalLM/libcausallm.a.p/x_qwen2_causallm.cpp.obj: cpp_COMPILER C$:/Users/Some$ \
One/nntrainer/Applications/CausalLM/models/qwen2/qwen2_causallm.cpp
 DEPFILE = "x.d"
 ARGS = "-IApplications\\CausalLM" "-I..\\..\\..\\Applications\\CausalLM" "/showIncludes" \
"/std:c++20" "-DPATH=\\"C:/Some$ One\\""

build Applications/CausalLM/nntr_causallm.exe: cpp_LINKER_RSP \
Applications/CausalLM/nntr_causallm.exe.p/main.cpp.obj | Applications/CausalLM/libcausallm.a \
nntrainer/libnntrainer.a C$:/abs/tokenizers_c.lib
 LINK_ARGS = "/nologo" "nntrainer/libnntrainer.a" "Applications/CausalLM/layers/empty_layer.lib" \
"Applications/CausalLM/libcausallm.a"

build Applications/CausalLM/libcausallm.a: STATIC_LINKER a.obj
build nntrainer/libnntrainer.a: STATIC_LINKER b.obj
'''


def _build_dir():
    path = Path(tempfile.mkdtemp(dir=tg._TMP))
    (path / "build.ninja").write_text(_NINJA, encoding="utf-8")
    return path


def test_ninja_rules_builds_and_escapes_are_read():
    """Ninja rules builds and escapes are read."""
    ninja = verify.Ninja(_build_dir() / "build.ninja")
    assert ninja.rules["cpp_LINKER_RSP"]["rspfile"] == "$out.rsp"
    comp = ninja.model_compile_statement()
    assert comp.ins == [
        "C:/Users/Some One/nntrainer/Applications/CausalLM/models/qwen2/qwen2_causallm.cpp"
    ]
    assert "libcausallm" in comp.outs[0]                    # not the gpt_oss layer DLL listed first
    assert '"-DPATH=\\"C:/Some One\\""' in comp.vars["ARGS"]
    link = ninja.link_statement()
    assert link.ins == ["Applications/CausalLM/nntr_causallm.exe.p/main.cpp.obj"]
    assert link.implicit == [
        "Applications/CausalLM/libcausallm.a",
        "nntrainer/libnntrainer.a",
        "C:/abs/tokenizers_c.lib",
    ]
    assert "nntrainer/libnntrainer.a" in ninja.outputs()
    assert (
        verify._expand('"link" @$out.rsp $$x ${in}', {"out": "a.exe", "in": "b.obj"})
        == '"link" @a.exe.rsp $x b.obj'
    )


def test_build_status_lists_the_libraries_still_to_build():
    """Build status lists the libraries still to build."""
    bdir = _build_dir()
    old = os.environ.get("WORKBENCH_NNTR_BUILD")
    os.environ["WORKBENCH_NNTR_BUILD"] = str(bdir)
    try:
        status = verify.build_status(tg._TMP, host="native")
        assert status["ok"] and status["status"] == "libs_missing"
        assert status["libs"] == [
            "Applications/CausalLM/libcausallm.a",
            "nntrainer/libnntrainer.a",
        ]  # tokenizer: not built here
        for lib in status["libs"]:
            (bdir / lib).parent.mkdir(parents=True, exist_ok=True)
            (bdir / lib).write_text("", encoding="utf-8")
        assert verify.build_status(tg._TMP, host="native")["status"] == "ready"
        os.environ["WORKBENCH_NNTR_BUILD"] = str(bdir / "nowhere")
        assert verify.build_status(tg._TMP, host="native")["status"] == "no_build"
    finally:
        if old is None:
            os.environ.pop("WORKBENCH_NNTR_BUILD", None)
        else:
            os.environ["WORKBENCH_NNTR_BUILD"] = old


def test_import_library_of_a_dll_that_exports_nothing_is_left_out():
    """Import library of a DLL that exports nothing is left out."""
    if os.name != "nt":
        return
    bdir = _build_dir()
    layers = bdir / "Applications" / "CausalLM" / "layers"
    layers.mkdir(parents=True)
    (layers / "empty_layer.dll").write_text("", encoding="utf-8")      # DLL, but no .lib
    (layers / "rms_norm_layer.dll").write_text("", encoding="utf-8")
    (layers / "rms_norm_layer.lib").write_text("", encoding="utf-8")
    args = (
        '"/nologo" "Applications/CausalLM/layers/empty_layer.lib"'
        ' "Applications/CausalLM/layers/rms_norm_layer.lib" "openblas.lib"'
    )
    kept, dropped = verify._drop_absent_import_libs(args, bdir)
    assert dropped == ["Applications/CausalLM/layers/empty_layer.lib"]
    assert (
        "rms_norm_layer.lib" in kept and "openblas.lib" in kept
    )  # no DLL next to it: left for the linker


def test_harness_binds_the_generated_class_to_upstreams_check():
    """Harness binds the generated class to upstream's check."""
    path = Path(tempfile.mkdtemp(dir=tg._TMP))
    (path / "olmo2_workbench_causallm.h").write_text(
        "class Olmo2WorkbenchCausalLM : public CausalLM, public Olmo2WorkbenchTransformer {\n};\n",
        encoding="utf-8",
    )
    review = {"model": "allenai/OLMo-2", "classes": {"causallm": "Olmo2WorkbenchCausalLM"},
              "files": {"olmo2_workbench_causallm.h": str(path / "olmo2_workbench_causallm.h")}}
    src = verify.harness_source(review, "olmo2_tiny")
    assert '#include "olmo2_workbench_causallm.h"' in src
    assert "CausalLMTestAdapter<causallm::Olmo2WorkbenchCausalLM>" in src
    assert "runFp32DifferentialChecks(workbenchModel())" in src and 'kFixture = "olmo2_tiny"' in src
    assert src.count("{") == src.count("}") and "WORKBENCH_RESULT" in src
    try:
        verify.harness_source(dict(review, classes={"causallm": "Missing"}), "x")
        raise AssertionError("expected VerifyError")
    except verify.VerifyError:
        pass


def test_tiny_config_keeps_the_architecture_and_shrinks_the_sizes():
    """Tiny config keeps the architecture and shrinks the sizes."""
    qwen2 = {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "hidden_size": 896,
             "intermediate_size": 4864, "num_attention_heads": 14, "num_key_value_heads": 2,
             "num_hidden_layers": 24, "vocab_size": 151936, "max_position_embeddings": 32768,
             "sliding_window": 32768, "use_sliding_window": False, "max_window_layers": 21,
             "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "tie_word_embeddings": True,
             "torch_dtype": "bfloat16", "pad_token_id": 151643}
    shrink_config = vf.shrink_config(qwen2)
    assert (
        shrink_config["hidden_size"],
        shrink_config["num_hidden_layers"],
        shrink_config["vocab_size"],
        shrink_config["intermediate_size"],
    ) == (64, 2, 32, 64)
    assert (
        shrink_config["num_key_value_heads"] < shrink_config["num_attention_heads"]
    )  # GQA stays GQA
    assert shrink_config["sliding_window"] == 32768  # unused window: unchanged
    assert shrink_config["max_window_layers"] == 2 and shrink_config["max_position_embeddings"] == 8
    assert (
        shrink_config["rope_theta"] == 1000000.0
        and shrink_config["rms_norm_eps"] == 1e-6
        and shrink_config["tie_word_embeddings"] is True
    )
    assert (
        "torch_dtype" not in shrink_config
        and "pad_token_id" not in shrink_config
        and shrink_config["initializer_range"] == 0.2
    )

    gemma = {
        "architectures": ["X"],
        "hidden_size": 1152,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "query_pre_attn_scalar": 256,
        "num_hidden_layers": 26,
        "vocab_size": 262144,
        "sliding_window": 512,
        "layer_types": ["sliding_attention"] * 5 + ["full_attention"] + ["sliding_attention"] * 20,
        "intermediate_size": 6912,
        "moe_intermediate_size": 768,
        "num_experts": 64,
        "num_experts_per_tok": 8,
    }
    shrink_config = vf.shrink_config(gemma)
    assert (
        shrink_config["head_dim"] == 32 and shrink_config["query_pre_attn_scalar"] == 32
    )  # head_dim != hidden/heads is kept that way
    assert (
        shrink_config["layer_types"] == ["sliding_attention", "full_attention"]
        and shrink_config["sliding_window"] == 4
    )
    assert (
        shrink_config["moe_intermediate_size"],
        shrink_config["num_experts"],
        shrink_config["num_experts_per_tok"],
    ) == (32, 4, 2)
    for bad in ({**qwen2, "auto_map": {"AutoModel": "x.Y"}}, {**qwen2, "text_config": {}}):
        try:
            vf.shrink_config(bad)
            raise AssertionError("expected FixtureError")
        except vf.FixtureError:
            pass


def test_generated_sources_get_the_includes_copied_hooks_need():
    """Generated sources get the includes copied hooks need."""
    text = (
        '// c\n#include "olmo2_workbench_causallm.h"\n#include <model.h>\n\nnamespace causallm {}\n'
    )
    out = cg.ensure_common_includes(text, "olmo2_workbench_causallm.h")
    assert (
        out.index('"olmo2_workbench_causallm.h"')
        < out.index("<llm_util.hpp>")
        < out.index("namespace")
    )
    assert out.count("<model.h>") == 1 and "<engine.h>" in out and "<app_context.h>" in out
    assert cg.ensure_common_includes(out, "olmo2_workbench_causallm.h") == out


def test_a_written_converter_needs_the_entry_point_verify_calls():
    """A written converter needs the entry point verify calls."""
    def files_without_entry_point(*a, **k):
        files = tg._olmo_files(*a, **k)
        files["weight_converter.py"] = (
            files["weight_converter.py"]
            .replace("def save_olmo2_for_nntrainer(", "def _write_all(")
            .replace("save_olmo2_for_nntrainer(", "_write_all(")
        )
        return files
    bank = tg.tc._bank_copy()
    review = cg.generate(
        tg._facts(tg.OLMO2, tg.OLMO2_TENSORS),
        bank,
        tg._fake_llm(files_without_entry_point),
        out_root=tg._out(),
        model_id="allenai/OLMo-2-test",
        log=lambda m: None,
        max_attempts=1,
    )
    assert any("save_<model>_for_nntrainer" in e for e in review["validation"]["errors"])


# ------------------------------------------------------- verify, faked run
class _FakeRun:
    """Stands in for the build + run: the fixture script, compiler, linker and harness."""

    def __init__(self, report: dict, failures=(), exit_code=0):
        self.report, self.failures, self.exit_code = report, list(failures), exit_code
        self.saved = {}

    def __enter__(self):
        names = (
            "build_status",
            "Ninja",
            "tool_env",
            "compile_source",
            "link_objects",
            "run_command",
            "select_executor",
        )
        self.saved = {n: getattr(verify, n) for n in names}
        verify.select_executor = lambda root=None, host=None: verify.NativeExecutor(
            Path(root or tg._TMP)
        )
        verify.build_status = lambda root=None, **kw: {
            "ok": True,
            "status": "ready",
            "build_dir": tg._TMP,
            "libs": [],
            "missing": [],
        }
        verify.Ninja = lambda path: None
        verify.tool_env = lambda: dict(os.environ)
        verify.compile_source = lambda *a: (True, [], "compiled")
        verify.link_objects = lambda *a: (True, [], "linked")
        verify.run_command = self._run
        return self

    def __exit__(self, *exc):
        for key, handle in self.saved.items():
            setattr(verify, key, handle)

    def _run(self, cmd, *_args, **_kwargs):
        if str(verify.FIXTURE_SCRIPT) in [str(c) for c in cmd]:
            return 0, json.dumps(
                {
                    "ok": True,
                    "fixture": "x",
                    "converter_fn": "save_olmo2_for_nntrainer",
                    "argmax": 3,
                    "max_abs_logit": 2.0,
                    "tokens": [3, 3, 1, 2],
                    "layers": 2,
                }
            )
        gjson = cmd[1].split("json:", 1)[1]
        fails = [{"failure": f} for f in self.failures]
        Path(gjson).write_text(
            json.dumps(
                {
                    "testsuites": [
                        {
                            "testsuite": [
                                {
                                    "name": "FP32MatchesHFReference",
                                    "result": "COMPLETED",
                                    "status": "RUN",
                                    "failures": fails,
                                },
                                {"name": "Report", "result": "COMPLETED", "status": "RUN"},
                            ]
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return self.exit_code, "[ RUN ]\nWORKBENCH_RESULT " + json.dumps(self.report) + "\n"


def _approved_olmo(bank):
    review = tcc._olmo_review(bank)
    review = tcc._approve_all(review["review_path"], bank)
    assert review["compile_ready"]["ok"]
    return review


def _checkout():
    root = Path(tempfile.mkdtemp(dir=tg._TMP))
    utils = root / "test" / "unittest" / "models" / "causallm_test_utils.cpp"
    utils.parent.mkdir(parents=True)
    utils.write_text("// upstream test utils\n", encoding="utf-8")
    return root


_MATCH = {"max_abs_diff": 0.002, "argmax_got": 3, "argmax_ref": 3, "tokens_got": [3, 3, 1, 2],
          "tokens_ref": [3, 3, 1, 2], "n_logits": 32, "n_ref": 32}


def test_a_pass_verifies_the_approved_code_in_the_bank():
    """A pass verifies the approved code in the bank."""
    bank = tg.tc._bank_copy()
    review = _approved_olmo(bank)
    written = [p["entry_id"] for p in review["parts"] if p["action"] == "write"]
    assert written and all(bank.get(e)["status"] == "approved" for e in written)
    with _FakeRun(_MATCH):
        review = verify.verify_generated(
            review["review_path"], bank, nntrainer_root=_checkout(), log=lambda m: None
        )
    verify_res = review["verify"]
    assert (
        verify_res["status"] == "passed"
        and verify_res["ok"]
        and verify_res["tokens_match"]
        and verify_res["max_abs_diff"] == 0.002
    )
    assert set(written) <= set(verify_res["marked_verified"])
    for written_item in written:
        entry = bank.get(written_item)
        assert entry["status"] == "verified" and entry["verification"]["method"] == "hf-logits"
    assert all(p["status"] == "verified" for p in review["parts"] if p["action"] == "write")
    assert review["compile_ready"]["ok"] and review["summary"]["to_review"] == 0
    harness = Path(review["out_dir"]) / "verify" / "unittest_olmo2_workbench.cpp"
    assert "Olmo2WorkbenchCausalLM" in harness.read_text(encoding="utf-8")

    again = cg.recheck(review["review_path"], bank)               # unchanged files keep the result
    assert again["verify"]["status"] == "passed"
    cpp = Path(review["out_dir"]) / "olmo2_workbench_causallm.cpp"
    cpp.write_text(cpp.read_text(encoding="utf-8") + "\n// edited\n", encoding="utf-8")
    assert cg.recheck(review["review_path"], bank)["verify"]["status"] == "not_run"


def test_a_mismatch_fails_the_written_parts_and_asks_for_review_again():
    """A mismatch fails the written parts and asks for review again."""
    bank = tg.tc._bank_copy()
    review = _approved_olmo(bank)
    written = [p["entry_id"] for p in review["parts"] if p["action"] == "write"]
    off = dict(_MATCH, max_abs_diff=0.31, tokens_got=[3, 1, 1, 1])
    with _FakeRun(
        off,
        failures=["causallm_test_utils.cpp:409\nThe difference between got[i] and ref[i] is 0.31"],
        exit_code=1,
    ):
        review = verify.verify_generated(
            review["review_path"], bank, nntrainer_root=_checkout(), log=lambda m: None
        )
    verify_res = review["verify"]
    assert (
        verify_res["status"] == "mismatch"
        and not verify_res["ok"]
        and not verify_res["tokens_match"]
    )
    assert set(verify_res["marked_failed"]) == set(written)
    assert all(bank.get(e)["status"] == "failed_verification" for e in written)
    assert not review["compile_ready"]["ok"] and review["compile_ready"]["why"].startswith(
        "waiting for review"
    )

    seen = []
    def llm(_system, user):
        seen.append(user)
        return tg._wrap(tg._olmo_files())
    cg.fix_verify_errors(review["review_path"], bank, llm, max_attempts=1, log=lambda m: None)
    assert (
        "Fix the mismatch against Hugging Face" in seen[0]
        and "0.31" in seen[0]
        and "[3, 1, 1, 1]" in seen[0]
    )


def test_a_crash_is_reported_without_touching_the_bank():
    """A crash is reported without touching the bank."""
    bank = tg.tc._bank_copy()
    review = _approved_olmo(bank)
    written = [p["entry_id"] for p in review["parts"] if p["action"] == "write"]
    with _FakeRun(
        {"error": "Failed to load model weights"},
        failures=["C++ exception: Failed to load"],
        exit_code=1,
    ):
        review = verify.verify_generated(
            review["review_path"], bank, nntrainer_root=_checkout(), log=lambda m: None
        )
    assert review["verify"]["status"] == "crashed" and "Failed to load" in review["verify"]["error"]
    assert all(bank.get(e)["status"] == "approved" for e in written)


def test_verify_waits_for_review_like_compile():
    """Verify waits for review like compile."""
    bank = tg.tc._bank_copy()
    review = tcc._olmo_review(bank)
    try:
        verify.verify_generated(
            review["review_path"], bank, nntrainer_root=_checkout(), log=lambda m: None
        )
        raise AssertionError("expected CodegenError")
    except cg.CodegenError as exc:
        assert "waiting for review" in str(exc)
