# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file test_codegen.py
# @brief Tests for Claude CLI code generation, with a fake LLM and no network.
# @author prachi-t17 <prachi.t@samsung.com>

# Tests call private helpers and import the engine from the folder that run_all.py
# puts on sys.path at run time (some only after installing stand-ins), which pylint
# cannot follow when it checks this file on its own.
# pylint: disable=protected-access,import-error,wrong-import-position,import-outside-toplevel

"""
Claude-CLI code generation (knowledge/codegen.py, codegen_validator.py) and
the CLI backend (agents/cpp/claude_cli_backend.py), with a stand-in for Claude:
a fake LLM that answers from the code bank, so the whole loop -- plan, prompt,
reply parsing, validation, retry, files, review, bank entries -- runs without
network access. Standard library only.
"""
import atexit
import json
import os
import shutil
import sys
import contextlib
import io
import tempfile
from pathlib import Path

_ENGINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_ENGINE_ROOT, os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)

import test_checklist as tc  # noqa: E402  (configs, tensor builders, captured bank)
from knowledge import checklist as ck  # noqa: E402
from knowledge import codegen as cg  # noqa: E402
from knowledge.codegen_validator import _body_of, validate  # noqa: E402

_TMP = tempfile.mkdtemp(prefix="codegen_test_")
atexit.register(lambda: shutil.rmtree(_TMP, ignore_errors=True))

SIG_ATT = (
    "Tensor {cls}::createAttention(const int layer_id, int seq_len, int n_heads, int head_dim, "
    "Tensor query, Tensor key, Tensor value) {body}"
)


def _header(prefix, t_decls, c_decls, extra_include=""):
    transformer_cls, causallm_cls = f"{prefix}WorkbenchTransformer", f"{prefix}WorkbenchCausalLM"
    return f"""#ifndef __GEN_H__
#define __GEN_H__
#include <causal_lm.h>
{extra_include}
namespace causallm {{
class {transformer_cls} : virtual public Transformer {{
public:
  {transformer_cls}(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg) {{}}
{t_decls}
}};
class {causallm_cls} : public CausalLM, public {transformer_cls} {{
public:
  {causallm_cls}(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM),
    CausalLM(cfg, generation_cfg, nntr_cfg),
    {transformer_cls}(cfg, generation_cfg, nntr_cfg) {{}}
{c_decls}
}};
}} // namespace causallm
#endif
"""


def _register_pair(prefix, classes=("ReshapedRMSNormLayer",)):
    transformer_cls, causallm_cls = f"{prefix}WorkbenchTransformer", f"{prefix}WorkbenchCausalLM"
    regs = "\n".join(
        f"  app_context->registerFactory(nntrainer::createLayer<causallm::{c}>);" for c in classes
    )
    return (
        f"void {transformer_cls}::registerCustomLayers() {{\n  auto &ct_engine ="
        " nntrainer::Engine::Global();\n  auto app_context = static_cast<nntrainer::AppContext"
        f" *>(ct_engine.getRegisteredContext(\"cpu\"));\n{regs}\n}}\n\nvoid"
        f" {causallm_cls}::registerCustomLayers() {{\n  CausalLM::registerCustomLayers();\n "
        f" {transformer_cls}::registerCustomLayers();\n}}\n"
    )


def _wrap(files):
    return "\n".join(f"<<<FILE: {n}>>>\n{t}\n<<<END FILE>>>" for n, t in files.items())


def _qwen3_reply(bank):
    body = _body_of(bank.get("hook:Qwen3Transformer::createAttention")["code"])
    decl = (
        "  Tensor createAttention(const int layer_id, int seq_len, int n_heads, int head_dim,"
        " Tensor query,\n                         Tensor key, Tensor value) override;\n  void"
        " registerCustomLayers() override;"
    )
    header = _header("Qwen3", decl, "  void registerCustomLayers() override;")
    cpp = (
        '#include "qwen3_workbench_causallm.h"\n#include <reshaped_rms_norm.h>\n\nnamespace'
        ' causallm {\n\n'
        + SIG_ATT.format(cls="Qwen3WorkbenchTransformer", body=body)
        + "\n\n"
        + _register_pair("Qwen3")
        + "\n} // namespace causallm\n"
    )
    return {"qwen3_workbench_causallm.h": header, "qwen3_workbench_causallm.cpp": cpp}


def _fake_llm(make_files, first=None):
    calls = []

    def llm(system, user):
        calls.append((system, user))
        if first is not None and len(calls) == 1:
            return first
        return _wrap(make_files())
    llm.calls = calls
    return llm


def _facts(cfg, tensors, name="test-model"):
    return ck.make_facts(name, cfg, tensors, "test tensors")


def _out():
    return tempfile.mkdtemp(dir=_TMP)


# --------------------------------------------------------------- known model
def test_known_model_prompt_carries_verified_code_and_inherits_base_hooks():
    """Known model prompt carries verified code and inherits base hooks."""
    bank = tc._bank_copy()
    facts = _facts(tc.QWEN3, tc._decoder(qk_norm="per_head"))
    report = ck.run_checklist(facts, bank.learned_features())
    spec = cg.make_spec("Qwen/Qwen3-test", facts, report, ck.match_bank(report, bank), bank)
    assert spec.transformer_class == "Qwen3WorkbenchTransformer" and spec.slug == "qwen3"
    assert spec.hooks["createAttention"].action == "copy"
    assert {
        h: spec.hooks[h].action
        for h in ("createMlp", "createTransformerDecoderBlock", "constructModel")
    } == {
        "createMlp": "inherit",
        "createTransformerDecoderBlock": "inherit",
        "constructModel": "inherit",
    }
    user = cg.build_user_prompt(spec, report, bank, tc.QWEN3, {})
    assert "## createAttention: COPY" in user and "reshaped_rms_norm" in user
    assert (
        "## createMlp: INHERIT" in user and "Do not write one" in user
    )  # verified converter reused
    system = cg.build_system_prompt(spec)
    assert "Do NOT override setupParameters" in system and "<<<FILE: file_name>>>" in system


def test_known_model_generates_on_first_attempt_and_reuses_everything():
    """Known model generates on first attempt and reuses everything."""
    bank = tc._bank_copy()
    llm = _fake_llm(lambda: _qwen3_reply(bank))
    review = cg.generate(
        _facts(tc.QWEN3, tc._decoder(qk_norm="per_head")),
        bank,
        llm,
        out_root=_out(),
        model_id="Qwen/Qwen3-test",
        log=lambda m: None,
    )
    assert review["validation"]["ok"], review["validation"]
    assert review["attempts"] == 1 and len(llm.calls) == 1
    status = {p["id"]: p["status"] for p in review["parts"]}
    assert status == {
        "constructModel": "verified",
        "createTransformerDecoderBlock": "verified",
        "createAttention": "verified",
        "createMlp": "verified",
        "weight_converter": "verified",
    }
    out = review["out_dir"]
    for name in (
        "qwen3_workbench_causallm.h",
        "qwen3_workbench_causallm.cpp",
        "meson.build",
        "register_qwen3_workbench.cpp",
        "weight_converter.py",
        "res/nntr_config.json",
        "review.json",
    ):
        assert os.path.exists(os.path.join(out, name)), name
    reg = Path(os.path.join(out, "register_qwen3_workbench.cpp")).read_text(encoding="utf-8")
    assert '"Qwen3ForCausalLM"' in reg and "make_unique<Qwen3WorkbenchCausalLM>" in reg
    assert not bank.list(status="generated")                       # nothing new to review


# ---------------------------------------------------------------- validator
def _spec_and_files(bank):
    facts = _facts(tc.QWEN3, tc._decoder(qk_norm="per_head"))
    report = ck.run_checklist(facts, bank.learned_features())
    return cg.make_spec("m", facts, report, ck.match_bank(report, bank), bank), _qwen3_reply(bank)


def test_validator_rejects_a_changed_copy():
    """Validator rejects a changed copy."""
    bank = tc._captured_bank()
    spec, files = _spec_and_files(bank)
    files["qwen3_workbench_causallm.cpp"] = files["qwen3_workbench_causallm.cpp"].replace(
        '"disable_bias", "true"', '"disable_bias", "false"', 1)
    errors = validate(files, spec, bank).errors
    assert any("must be copied exactly" in e for e in errors), errors


def test_validator_rejects_overriding_an_inherited_hook_and_setup():
    """Validator rejects overriding an inherited hook and setup."""
    bank = tc._captured_bank()
    spec, files = _spec_and_files(bank)
    files["qwen3_workbench_causallm.cpp"] = files["qwen3_workbench_causallm.cpp"].replace(
        "} // namespace causallm",
        "Tensor Qwen3WorkbenchTransformer::createMlp(const int layer_id, int dim, int hidden_dim,"
        " Tensor input) { return input; }\nvoid Qwen3WorkbenchTransformer::setupParameters(json &a,"
        " json &b, json &c) {}\n} // namespace causallm",
    )
    errors = validate(files, spec, bank).errors
    assert any(e.startswith("do not override createMlp") for e in errors), errors
    assert any("must not override setupParameters" in e for e in errors), errors


def test_validator_checks_layer_types_properties_names_and_registration():
    """Validator checks layer types properties names and registration."""
    bank = tc._captured_bank()
    spec, files = _spec_and_files(bank)
    cpp = files["qwen3_workbench_causallm.cpp"]
    files["qwen3_workbench_causallm.cpp"] = cpp.replace(
        "createLayer<causallm::ReshapedRMSNormLayer>", "createLayer<X>"
    )
    files["qwen3_workbench_causallm.cpp"] += (
        '\nvoid helper() { auto a = createLayer("rotary_magic", {withKey("name", "x")});\n  auto b'
        ' = createLayer("fully_connected", {withKey("name", "y"), withKey("feature_size", "3")});\n'
        '  auto c = createLayer("addition", {});\n}\n'
    )
    errors = validate(files, spec, bank).errors
    assert any('unknown layer type "rotary_magic"' in e for e in errors), errors
    assert any('does not take feature_size' in e for e in errors), errors
    assert any('createLayer("addition", ...) has no withKey("name"' in e for e in errors), errors
    assert any("ReshapedRMSNormLayer is not registered" in e for e in errors), errors


def test_validator_checks_class_layout_and_constructor_order():
    """Validator checks class layout and constructor order."""
    bank = tc._captured_bank()
    spec, files = _spec_and_files(bank)
    header = files["qwen3_workbench_causallm.h"]
    header = header.replace(": virtual public Transformer", ": public Transformer")
    header = header.replace(
        "Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM),\n    CausalLM(cfg,"
        " generation_cfg, nntr_cfg),",
        "CausalLM(cfg, generation_cfg, nntr_cfg),",
    )
    files["qwen3_workbench_causallm.h"] = header
    errors = validate(files, spec, bank).errors
    assert any("virtual public Transformer" in e for e in errors), errors
    assert any("must initialise the virtual base explicitly" in e for e in errors), errors


# ------------------------------------------------------------- new parts
OLMO2 = tc._base_cfg(architectures=["Olmo2ForCausalLM"], model_type="olmo2", attention_bias=False,
                     hidden_act="silu", rope_scaling=None, tie_word_embeddings=False)
OLMO2_TENSORS = tc._decoder(
    tied=False, qk_norm="full", norms=("post_attention_layernorm", "post_feedforward_layernorm")
)

_OLMO_ATT = '''{
  auto Q = query; auto K = key; auto V = value;
  LayerHandle wq(createLayer("fully_connected", {withKey("name", "layer" + \
std::to_string(layer_id) + "_wq"),
                                                 withKey("unit", head_dim * n_heads), \
withKey("disable_bias", "true")}));
  LayerHandle qn(createLayer("rms_norm", {withKey("name", "layer" + std::to_string(layer_id) + \
"_q_norm"),
                                          withKey("epsilon", std::to_string(NORM_EPS))}));
  Tensor q = qn(wq(Q));
  LayerHandle wk(createLayer("fully_connected", {withKey("name", "layer" + \
std::to_string(layer_id) + "_wk"),
                                                 withKey("unit", head_dim * n_heads / GQA_SIZE), \
withKey("disable_bias", "true")}));
  LayerHandle kn(createLayer("rms_norm", {withKey("name", "layer" + std::to_string(layer_id) + \
"_k_norm"),
                                          withKey("epsilon", std::to_string(NORM_EPS))}));
  Tensor k = kn(wk(K));
  LayerHandle wv(createLayer("fully_connected", {withKey("name", "layer" + \
std::to_string(layer_id) + "_wv"),
                                                 withKey("unit", head_dim * n_heads / GQA_SIZE), \
withKey("disable_bias", "true")}));
  Tensor v = wv(V);
  auto [cache_k, cache_v] = createKVCachePlaceholders(layer_id, n_heads);
  LayerHandle mha(createLayer("mha_core", {withKey("name", "layer" + std::to_string(layer_id) + \
"_attention"),
                                           withKey("num_heads", n_heads), withKey("num_heads_kv", \
n_heads / GQA_SIZE),
                                           withKey("rope_theta", ROPE_THETA)}));
  Tensor a = mha({q, k, v, cache_k, cache_v});
  LayerHandle wo(createLayer("fully_connected", {withKey("name", "layer" + \
std::to_string(layer_id) + "_attention_out"),
                                                 withKey("unit", DIM), withKey("disable_bias", \
"true")}));
  return wo(a);
}'''
_OLMO_BLOCK = '''{
  Tensor att = createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM, input, input, input);
  LayerHandle pn(createLayer("rms_norm", {withKey("name", "layer" + std::to_string(layer_id) + \
"_post_attention_norm"),
                                          withKey("epsilon", std::to_string(NORM_EPS))}));
  LayerHandle add1(createLayer("addition", {withKey("name", "layer" + std::to_string(layer_id) + \
"_decoder_add")}));
  Tensor h = add1({input, pn(att)});
  Tensor m = createMlp(layer_id, DIM, INTERMEDIATE_SIZE, h);
  LayerHandle fn(createLayer("rms_norm", {withKey("name", "layer" + std::to_string(layer_id) + \
"_post_ffn_norm"),
                                          withKey("epsilon", std::to_string(NORM_EPS))}));
  LayerHandle add2(createLayer("addition", {withKey("name", "layer" + std::to_string(layer_id) + \
"_decoder_output")}));
  return add2({h, fn(m)});
}'''

_CONVERTER = '''import json, sys
import numpy as np
from safetensors.numpy import load_file

ORDER = ["embed_tokens", "self_attn.q_proj", "self_attn.q_norm", "self_attn.k_proj", \
"self_attn.k_norm",
         "self_attn.v_proj", "self_attn.o_proj", "post_attention_layernorm", "mlp.up_proj", \
"mlp.gate_proj",
         "mlp.down_proj", "post_feedforward_layernorm", "norm", "lm_head"]

def save_olmo2_for_nntrainer(params, n_layers, dtype, file):
    for key in sorted(params):
        arr = params[key]
        if arr.ndim == 2 and "proj" in key:
            arr = arr.T
        arr.astype(np.float32).tofile(file)

def main(src, dst):
    params = load_file(src)
    with open(dst, "wb") as f:
        save_olmo2_for_nntrainer(params, 0, "float32", f)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
'''


def _olmo_files(bad_layer=False):
    transformer_cls = "Olmo2WorkbenchTransformer"
    decl = (
        "  Tensor createAttention(const int layer_id, int seq_len, int n_heads, int head_dim,"
        " Tensor query,\n                         Tensor key, Tensor value) override;\n  Tensor"
        " createTransformerDecoderBlock(const int layer_id, Tensor input) override;"
    )
    att = _OLMO_ATT.replace('"mha_core"', '"rotary_attention"') if bad_layer else _OLMO_ATT
    cpp = (
        '#include "olmo2_workbench_causallm.h"\n\nnamespace causallm {\n\n'
        + SIG_ATT.format(cls=transformer_cls, body=att)
        + "\n\n"
        + f"Tensor {transformer_cls}::createTransformerDecoderBlock(const int layer_id, Tensor"
        f" input) {_OLMO_BLOCK}\n\n"
        + "} // namespace causallm\n"
    )
    return {
        "olmo2_workbench_causallm.h": _header("Olmo2", decl, ""),
        "olmo2_workbench_causallm.cpp": cpp,
        "weight_converter.py": _CONVERTER,
    }


def test_new_parts_retry_on_failed_checks_then_land_in_review_and_bank():
    """New parts retry on failed checks then land in review and bank."""
    bank = tc._bank_copy()
    llm = _fake_llm(_olmo_files, first=_wrap(_olmo_files(bad_layer=True)))
    hf_sources = {
        "Olmo2Attention": "class Olmo2Attention(nn.Module): ...",
        "Olmo2DecoderLayer": "class Olmo2DecoderLayer: ...",
    }
    review = cg.generate(
        _facts(OLMO2, OLMO2_TENSORS),
        bank,
        llm,
        out_root=_out(),
        model_id="allenai/OLMo-2-test",
        hf_sources=hf_sources,
        log=lambda m: None,
    )
    assert review["validation"]["ok"], review["validation"]
    assert review["attempts"] == 2
    assert (
        'unknown layer type "rotary_attention"' in llm.calls[1][1]
    )  # the error went back to Claude
    parts = {p["id"]: p for p in review["parts"]}
    assert (
        parts["createAttention"]["action"] == "write"
        and parts["createAttention"]["status"] == "generated"
    )
    assert parts["createAttention"]["hf_reference"].startswith("class Olmo2Attention")
    assert parts["createAttention"]["learned_feature"]["claims"]["tensors"] == [
        "layers.*.self_attn.k_norm.weight", "layers.*.self_attn.q_norm.weight"]
    assert parts["createTransformerDecoderBlock"]["status"] == "generated"
    assert parts["weight_converter"]["action"] == "write"
    eid = parts["createAttention"]["entry_id"]
    assert bank.get(eid)["status"] == "generated" and not bank.find_reusable(
        "hook", bank.get(eid)["features"]
    )
    assert review["summary"]["to_review"] >= 3
    prompt = llm.calls[0][1]
    assert "## createAttention: WRITE" in prompt and "class Olmo2Attention" in prompt
    assert "tensor layers.*.self_attn.q_norm.weight" in prompt                  # the evidence


def test_approving_saves_the_learned_box_and_the_next_model_reuses_after_verification():
    """Approving saves the learned box and the next model reuses after verification."""
    bank = tc._bank_copy()
    review = cg.generate(
        _facts(OLMO2, OLMO2_TENSORS),
        bank,
        _fake_llm(_olmo_files),
        out_root=_out(),
        model_id="allenai/OLMo-2-test",
        log=lambda m: None,
    )
    path = review["review_path"]
    for part in ("createAttention", "createTransformerDecoderBlock", "weight_converter"):
        cg.set_part_status(path, part, "approve", actor="reviewer", bank=bank)
    review = cg.load_review(path)
    att = next(p for p in review["parts"] if p["id"] == "createAttention")
    assert att["status"] == "approved" and bank.get(att["entry_id"])["status"] == "approved"
    assert {f["id"] for f in bank.learned_features()} >= {
        "learned_olmo2_createAttention",
        "learned_olmo2_createTransformerDecoderBlock",
    }
    # A second OLMo-2-style checkpoint: the learned boxes explain it, but the
    # approved code is only an example until it is verified.
    report = ck.run_checklist(_facts(OLMO2, OLMO2_TENSORS), bank.learned_features())
    assert report.is_fully_explained
    assert ck.match_bank(report, bank)["hooks"]["createAttention"]["state"] == "needs_verification"
    bank.mark_verified(att["entry_id"], {"method": "hf-logits", "max_abs_diff": 0.004})
    plan = ck.match_bank(report, bank)
    assert plan["hooks"]["createAttention"]["reuse"] == att["entry_id"]


def test_reject_and_recheck_after_editing():
    """Reject and recheck after editing."""
    bank = tc._bank_copy()
    review = cg.generate(
        _facts(OLMO2, OLMO2_TENSORS),
        bank,
        _fake_llm(_olmo_files),
        out_root=_out(),
        model_id="allenai/OLMo-2-test",
        log=lambda m: None,
    )
    path, out = review["review_path"], review["out_dir"]
    old = next(p for p in review["parts"] if p["id"] == "createAttention")["entry_id"]
    cpp_path = os.path.join(out, "olmo2_workbench_causallm.cpp")
    text = Path(cpp_path).read_text(encoding="utf-8")
    Path(cpp_path).write_text(text.replace('"mha_core"', '"rotary_attention"'), encoding="utf-8")
    review = cg.recheck(path, bank)
    assert not review["validation"]["ok"]
    try:
        cg.set_part_status(path, "createAttention", "approve", actor="reviewer", bank=bank)
        raise AssertionError("approved code that fails the checks")
    except cg.CodegenError:
        pass
    Path(cpp_path).write_text(
        text.replace(
            "withKey(\"rope_theta\", ROPE_THETA)",
            "withKey(\"rope_theta\", ROPE_THETA), withKey(\"max_timestep\", MAX_SEQ_LEN)",
        ),
        encoding="utf-8",
    )
    review = cg.recheck(path, bank)
    assert review["validation"]["ok"], review["validation"]
    new = next(p for p in review["parts"] if p["id"] == "createAttention")["entry_id"]
    assert new != old and bank.get(old)["status"] == "rejected"  # the superseded version is retired
    assert bank.get(new)["source"] == "edited"
    cg.set_part_status(
        path, "createTransformerDecoderBlock", "reject", actor="reviewer", bank=bank, reason="wrong"
    )
    blk = next(
        p for p in cg.load_review(path)["parts"] if p["id"] == "createTransformerDecoderBlock"
    )
    assert blk["status"] == "rejected" and bank.get(blk["entry_id"])["status"] == "rejected"


def test_regenerate_sends_the_reviewer_comment():
    """Regenerate sends the reviewer comment."""
    bank = tc._bank_copy()
    review = cg.generate(
        _facts(OLMO2, OLMO2_TENSORS),
        bank,
        _fake_llm(_olmo_files),
        out_root=_out(),
        model_id="allenai/OLMo-2-test",
        log=lambda m: None,
    )
    llm = _fake_llm(_olmo_files)
    cg.regenerate(
        review["review_path"],
        "createAttention",
        "use the HF epsilon",
        bank,
        llm,
        log=lambda m: None,
    )
    assert "use the HF epsilon" in llm.calls[0][1] and "# Current files" in llm.calls[0][1]


def test_whole_model_match_is_copied_without_the_llm():
    """Whole model match is copied without the LLM."""
    bank = tc._bank_copy()
    cfg = {
        "architectures": ["Lfm2ForCausalLM"],
        "model_type": "lfm2",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 100,
    }
    tensors = {
        "model.embed_tokens.weight": [100, 64],
        "model.layers.0.conv.conv.weight": [64, 1, 3],
    }
    review = cg.generate(
        _facts(cfg, tensors),
        bank,
        llm=None,
        out_root=_out(),
        model_id="LiquidAI/LFM2-test",
        log=lambda m: None,
    )
    assert review["generator"].startswith("copied")
    out = review["out_dir"]
    header = Path(os.path.join(out, "lfm2_causallm_workbench.h")).read_text(encoding="utf-8")
    assert "class Lfm2WorkbenchCausalLM" in header and "class Lfm2CausalLM " not in header
    reg = Path(os.path.join(out, "register_lfm2_workbench.cpp")).read_text(encoding="utf-8")
    assert '"Lfm2ForCausalLM"' in reg and "Lfm2WorkbenchCausalLM" in reg


def test_parse_files_only_accepts_plain_names_and_strips_fences():
    """Parse files only accepts plain names and strips fences."""
    files = cg.parse_files(
        "noise\n<<<FILE: a.h>>>\n```cpp\nint a;\n```\n<<<END FILE>>>\n"
        "<<<FILE: ../../etc/passwd>>>\nx\n<<<END FILE>>>\n<<<FILE: b c.h>>>\ny\n<<<END FILE>>>"
    )
    assert files == {"a.h": "int a;\n", "passwd": "x\n"}  # path parts dropped; bad names skipped


# --------------------------------------------------------------- CLI backend
from agents.cpp import claude_cli_backend as cli  # noqa: E402

_FAKE = r'''
import sys, json
if "--help" in sys.argv:
    print("  --tools <tools...>  x\n  --no-session-persistence\n"
          "  --system-prompt <prompt>\n  --model <model>")
    sys.exit(0)
data = sys.stdin.buffer.read().decode("utf-8")
reply = {"argv": sys.argv[1:], "stdin_len": len(data), "tail": data[-12:]}
sys.stdout.buffer.write(json.dumps(reply).encode("utf-8"))
'''


def test_cli_backend_sends_the_prompt_on_stdin_with_supported_flags_only():
    """CLI backend sends the prompt on stdin with supported flags only."""
    script = os.path.join(_TMP, "fake_claude.py")
    with open(script, "w", encoding="utf-8") as handle:
        handle.write(_FAKE)
    claude = cli.ClaudeCLI(exe=[sys.executable, script], model="opus", timeout=60)
    long_user = "x" * 50000 + " ünïcode →"
    out = json.loads(claude.run("rules " * 2000, long_user))
    argv = out["argv"]
    assert (
        out["tail"].endswith(" ünïcode →") and out["stdin_len"] > 60000
    )  # all of it, UTF-8, through stdin
    assert argv[:2] == ["-p", "--output-format"] and "opus" in argv
    assert argv[argv.index("--tools") + 1] == "" and "--no-session-persistence" in argv
    assert "--strict-mcp-config" not in argv  # not listed by this CLI's --help
    assert len(" ".join(argv)) < 1000  # nothing long on the command line


def test_cli_backend_reports_failures():
    """CLI backend reports failures."""
    script = os.path.join(_TMP, "failing_claude.py")
    with open(script, "w", encoding="utf-8") as handle:
        handle.write(
            'import sys\nif "--help" in sys.argv: sys.exit(0)\nsys.stderr.write("not logged in");'
            ' sys.exit(3)\n'
        )
    claude = cli.ClaudeCLI(exe=[sys.executable, script], timeout=60)
    try:
        claude.run("s", "u")
        raise AssertionError("no error raised")
    except cli.ClaudeCLIError as exc:
        assert "exited with 3" in str(exc) and "not logged in" in str(exc)


def test_find_claude_honours_an_explicit_path():
    """Find Claude honours an explicit path."""
    tmp_dir = tempfile.mkdtemp(dir=_TMP)
    exe = os.path.join(tmp_dir, "claude.exe" if os.name == "nt" else "claude")
    Path(exe).write_text("", encoding="utf-8")
    assert cli.find_claude(exe) == exe


# ------------------------------------------------------- the pipeline step
# pylint: disable-next=too-many-locals
def test_pipeline_step_end_to_end_with_a_wrapped_cli():
    """agents/causallm_codegen.run on a local snapshot: safetensors header ->
    empty bank filled from the repo -> plan -> Claude CLI found through
    CLAUDE_PATH (a .cmd wrapper on Windows, as npm installs it) -> files,
    review and events. Output and bank go to temp folders."""
    work = tempfile.mkdtemp(dir=_TMP)
    snap = os.path.join(work, "olmo2-snapshot")
    os.makedirs(snap)
    with open(os.path.join(snap, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(OLMO2, handle)
    tc._write_safetensors(os.path.join(snap, "model.safetensors"), OLMO2_TENSORS)
    reply = os.path.join(work, "reply.txt")
    with open(reply, "w", encoding="utf-8") as handle:
        handle.write(_wrap(_olmo_files()))
    script = os.path.join(work, "fake_claude.py")
    with open(script, "w", encoding="utf-8") as handle:
        handle.write(
            "import sys\nif '--help' in sys.argv:\n    print('  --tools <tools...> "
            " --system-prompt <p>')\n   "
            f" sys.exit(0)\nsys.stdin.buffer.read()\nsys.stdout.buffer.write(open({reply!r},"
            " 'rb').read())\n"
        )
    if os.name == "nt":
        exe = os.path.join(work, "claude.cmd")
        with open(exe, "w", encoding="utf-8") as handle:
            handle.write(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n')
    else:
        exe = os.path.join(work, "claude")
        with open(exe, "w", encoding="utf-8") as handle:
            handle.write(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        os.chmod(exe, 0o755)
    env = {"CLAUDE_PATH": exe, "WORKBENCH_CODE_BANK": os.path.join(work, "bank.db"),
           "WORKBENCH_GENERATED_DIR": os.path.join(work, "generated"), "WORKBENCH_CODEGEN": "1"}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    from agents import causallm_codegen
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            state = causallm_codegen.run({"model_name": snap, "hf_config": OLMO2})
    finally:
        for k, val in old.items():
            if val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = val
    events = [json.loads(line) for line in out.getvalue().splitlines() if line.startswith("{")]
    kinds = {e["event"] for e in events}
    assert state.get("codegen_ok") is True, [e for e in events if e["event"] == "error"]
    assert {"codegen", "file_content", "code"} <= kinds
    review = next(e["data"] for e in events if e["event"] == "codegen")
    assert review["out_dir"].startswith(os.path.join(work, "generated"))
    assert state["causallm_header_path"].endswith("olmo2_workbench_causallm.h")
    assert "Olmo2WorkbenchCausalLM" in state["causallm_header"]
    assert {p["id"]: p["status"] for p in review["parts"]}["createAttention"] == "generated"


# ------------------------------------------------- fixes from the first real runs
def test_only_causal_lms_are_generated():
    """Only causal LMs are generated."""
    from agents.causallm_codegen import not_causal_lm_reason
    assert not_causal_lm_reason({"architectures": ["Qwen3ForCausalLM"]}) is None
    assert not_causal_lm_reason({"architectures": ["GPT2LMHeadModel"]}) is None
    assert not_causal_lm_reason({"architectures": ["Gemma3ForConditionalGeneration"],
                                 "text_config": {"model_type": "gemma3_text"}}) is None
    bert = not_causal_lm_reason({"architectures": ["BertForMaskedLM"], "model_type": "bert"})
    assert bert and "BertForMaskedLM is not a causal" in bert
    t5_reason = not_causal_lm_reason(
        {"architectures": ["T5ForConditionalGeneration"], "is_encoder_decoder": True}
    )
    assert t5_reason and "encoder-decoder" in t5_reason


def test_checker_accepts_nntrainers_own_layers():
    """Checker accepts nntrainer's own layers."""
    from knowledge.capture_causallm import core_layer_types, find_nntrainer_root
    core = core_layer_types(find_nntrainer_root())
    assert {"layer_normalization", "embedding", "concat", "reshape"} <= set(core)
    bank = tc._captured_bank()
    assert "layer_normalization" in bank.get_meta("nntrainer_core_layer_types")
    spec, files = _spec_and_files(bank)
    assert "layer_normalization" in spec.core_layer_types
    files["qwen3_workbench_causallm.cpp"] += (
        '\nvoid helper() { auto n = createLayer("layer_normalization", '
        '{withKey("name", "x_ln"), withKey("epsilon", "1e-5")}); }\n'
    )
    errors = validate(files, spec, bank).errors
    assert not any("layer_normalization" in e for e in errors), errors
    assert "layer_normalization" in cg.build_system_prompt(spec)


def test_shape_probe_uses_a_few_tokens_not_max_position_embeddings():
    """Shape probe uses a few tokens not max position embeddings."""
    try:
        import torch
    except ImportError:
        return                                    # torch-free environment: nothing to check
    from api.parsers import generic_fx_parser as gfp

    # pylint: disable-next=too-few-public-methods
    class Cfg:
        """Stand-in config with a long max_position_embeddings."""
        max_position_embeddings = 40960
        vocab_size = 100

    # pylint: disable-next=too-few-public-methods
    class Tiny(torch.nn.Module):
        """A one-layer model that records the input shape it is called with."""
        def __init__(self):
            super().__init__()
            self.config = Cfg()
            self.emb = torch.nn.Embedding(100, 8)
            self.seen = None

        def forward(self, ids):
            """Record the input shape and embed the ids."""
            self.seen = tuple(ids.shape)
            return self.emb(ids)

    tiny = Tiny()
    parser = gfp.GenericFxParser(tiny, model_name="tiny")
    parser._infer_shapes()
    assert tiny.seen == (1, gfp.SHAPE_PROBE_SEQ_LEN), tiny.seen


def test_a_learned_box_does_not_pull_other_models_off_verified_code():
    """Approving OLMo-2 teaches 'q_norm/k_norm over the full width'. Qwen3 also
    has q_norm/k_norm (per head, explained by a built-in box): it must still
    reuse its verified code, while another OLMo-2-style model uses the learned box."""
    bank = tc._bank_copy()
    review = cg.generate(
        _facts(OLMO2, OLMO2_TENSORS),
        bank,
        _fake_llm(_olmo_files),
        out_root=_out(),
        model_id="allenai/OLMo-2-test",
        log=lambda m: None,
    )
    for part in ("createAttention", "createTransformerDecoderBlock", "weight_converter"):
        cg.set_part_status(review["review_path"], part, "approve", actor="reviewer", bank=bank)
    att_box = next(f for f in bank.learned_features() if f["id"] == "learned_olmo2_createAttention")
    assert {"shape": "layers.*.self_attn.q_norm.weight", "dims": ["@q_width"]} in att_box["when"]
    qwen = ck.run_checklist(
        _facts(tc.QWEN3, tc._decoder(qk_norm="per_head")), bank.learned_features()
    )
    assert not any(f.ticked for f in qwen.features if f.source == "learned")
    plan = ck.match_bank(qwen, bank)
    assert plan["hooks"]["createAttention"]["reuse"] == "hook:Qwen3Transformer::createAttention"
    assert plan["converter"]["state"] == "reuse"
    olmo = ck.run_checklist(_facts(OLMO2, OLMO2_TENSORS), bank.learned_features())
    assert {"learned_olmo2_createAttention", "learned_olmo2_createTransformerDecoderBlock"} <= \
        {f.id for f in olmo.features if f.ticked}
