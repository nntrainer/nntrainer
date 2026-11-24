# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file test_checklist.py
# @brief Tests for the feature checklist, coverage check and CausalLM capture.
# @author prachi-t17 <prachi.t@samsung.com>

# Tests call private helpers and import the engine from the folder that run_all.py
# puts on sys.path at run time (some only after installing stand-ins), which pylint
# cannot follow when it checks this file on its own.
# pylint: disable=import-error,wrong-import-position

"""
Feature checklist + coverage check (knowledge/checklist.py) and the capture
from Applications/CausalLM (knowledge/capture_causallm.py), planned against a
bank captured from this checkout.

Tensor lists here are written from Hugging Face naming conventions for small
versions of each model (no torch/transformers needed); configs use the real
HF key sets. Standard library only.
"""
import atexit
import json
import os
import shutil
import sqlite3
import struct
import sys
import tempfile

_ENGINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ENGINE_ROOT not in sys.path:
    sys.path.insert(0, _ENGINE_ROOT)

from knowledge import checklist as ck  # noqa: E402
from knowledge.code_bank import CodeBank  # noqa: E402
from knowledge.capture_causallm import capture, find_nntrainer_root  # noqa: E402

H, HEADS, KV, HD, FFN, V = 64, 4, 2, 16, 128, 100

# Every temp bank/snapshot lives under one folder that is removed at exit
# (connections are closed first; Windows cannot delete an open database).
_TMP = tempfile.mkdtemp(prefix="checklist_test_")
_OPEN = []


def _cleanup():
    for bank in _OPEN:
        try:
            bank.close()
        except sqlite3.Error:
            pass
    shutil.rmtree(_TMP, ignore_errors=True)


atexit.register(_cleanup)


def _new_bank():
    bank = CodeBank(os.path.join(tempfile.mkdtemp(dir=_TMP), "bank.db"))
    _OPEN.append(bank)
    return bank


# pylint: disable-next=too-many-locals,too-many-arguments
def _decoder(
    layers=2,
    hidden=H,
    heads=HEADS,
    kv_heads=KV,
    head_dim=HD,
    ffn=FFN,
    vocab=V,
    tied=True,
    qkv_bias=False,
    qk_norm=None,
    norms=("input_layernorm", "post_attention_layernorm"),
    mlp=True,
    prefix="model.",
):
    tensors = {f"{prefix}embed_tokens.weight": [vocab, hidden], f"{prefix}norm.weight": [hidden]}
    if not tied:
        tensors["lm_head.weight"] = [vocab, hidden]
    for i in range(layers):
        path = f"{prefix}layers.{i}."
        for norm in norms:
            tensors[path + norm + ".weight"] = [hidden]
        tensors[path + "self_attn.q_proj.weight"] = [heads * head_dim, hidden]
        tensors[path + "self_attn.k_proj.weight"] = [kv_heads * head_dim, hidden]
        tensors[path + "self_attn.v_proj.weight"] = [kv_heads * head_dim, hidden]
        tensors[path + "self_attn.o_proj.weight"] = [hidden, heads * head_dim]
        if qkv_bias:
            tensors[path + "self_attn.q_proj.bias"] = [heads * head_dim]
            tensors[path + "self_attn.k_proj.bias"] = [kv_heads * head_dim]
            tensors[path + "self_attn.v_proj.bias"] = [kv_heads * head_dim]
        if qk_norm == "per_head":
            tensors[path + "self_attn.q_norm.weight"] = [head_dim]
            tensors[path + "self_attn.k_norm.weight"] = [head_dim]
        elif qk_norm == "full":
            tensors[path + "self_attn.q_norm.weight"] = [heads * head_dim]
            tensors[path + "self_attn.k_norm.weight"] = [kv_heads * head_dim]
        if mlp:
            tensors[path + "mlp.gate_proj.weight"] = [ffn, hidden]
            tensors[path + "mlp.up_proj.weight"] = [ffn, hidden]
            tensors[path + "mlp.down_proj.weight"] = [hidden, ffn]
    return tensors


def _base_cfg(**kw):
    cfg = {
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_size": H,
        "initializer_range": 0.02,
        "intermediate_size": FFN,
        "max_position_embeddings": 4096,
        "num_attention_heads": HEADS,
        "num_hidden_layers": 2,
        "num_key_value_heads": KV,
        "head_dim": HD,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.51.0",
        "use_cache": True,
        "vocab_size": V,
    }
    cfg.update(kw)
    return cfg


QWEN3 = _base_cfg(
    architectures=["Qwen3ForCausalLM"],
    model_type="qwen3",
    attention_bias=False,
    hidden_act="silu",
    max_window_layers=28,
    rope_scaling=None,
    sliding_window=None,
    tie_word_embeddings=True,
    use_sliding_window=False,
    rope_theta=1000000,
)
LLAMA2 = _base_cfg(
    architectures=["LlamaForCausalLM"],
    model_type="llama",
    attention_bias=False,
    hidden_act="silu",
    mlp_bias=False,
    pretraining_tp=1,
    rope_scaling=None,
    tie_word_embeddings=False,
)
LLAMA32 = dict(LLAMA2, tie_word_embeddings=True,
               rope_scaling={"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                             "original_max_position_embeddings": 8192, "rope_type": "llama3"})
GEMMA3 = _base_cfg(
    architectures=["Gemma3ForCausalLM"],
    model_type="gemma3_text",
    attention_bias=False,
    attn_logit_softcapping=None,
    final_logit_softcapping=None,
    hidden_activation="gelu_pytorch_tanh",
    query_pre_attn_scalar=HD,
    rope_local_base_freq=10000,
    rope_scaling=None,
    rope_theta=1000000,
    sliding_window=512,
    sliding_window_pattern=6,
    cache_implementation="hybrid",
)
GEMMA3_NORMS = (
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
)
QWEN3_MOE = dict(
    QWEN3,
    architectures=["Qwen3MoeForCausalLM"],
    model_type="qwen3_moe",
    decoder_sparse_step=1,
    mlp_only_layers=[],
    moe_intermediate_size=32,
    norm_topk_prob=True,
    num_experts=4,
    num_experts_per_tok=2,
    output_router_logits=False,
    router_aux_loss_coef=0.001,
)

_BANK = {}


def _captured_bank():
    """A bank captured from this checkout (once per test run)."""
    if "bank" not in _BANK:
        root = find_nntrainer_root()
        assert root, "tests must run inside an nntrainer checkout"
        new_bank = _new_bank()
        _BANK["summary"] = capture(root, new_bank)
        _BANK["bank"] = new_bank
    return _BANK["bank"]


def _bank_copy():
    copy = _new_bank()
    _captured_bank().conn.backup(copy.conn)
    return copy


def _plan(cfg, tensors, extra=(), bank=None):
    report = ck.run_checklist(ck.make_facts("m", cfg, tensors, "test"), extra)
    return report, ck.match_bank(report, bank or _captured_bank())


def _ticked(report):
    return {f.id for f in report.ticked}


def _moe_tensors(experts=4):
    decoder = _decoder(qk_norm="per_head", mlp=False)
    for i in range(2):
        path = f"model.layers.{i}.mlp."
        decoder[path + "gate.weight"] = [experts, H]
        for expert in range(experts):
            for proj in ("gate_proj", "up_proj"):
                decoder[f"{path}experts.{expert}.{proj}.weight"] = [32, H]
            decoder[f"{path}experts.{expert}.down_proj.weight"] = [H, 32]
    return decoder


# --------------------------------------------------------- known models
def test_qwen3_is_fully_explained_and_reuses_qwen3_code():
    """Qwen3 is fully explained and reuses Qwen3 code."""
    report, plan = _plan(QWEN3, _decoder(qk_norm="per_head"))
    assert report.is_fully_explained, (report.unclaimed_tensors, report.unclaimed_config)
    assert {
        "qk_norm_per_head",
        "block_pre_norm",
        "mlp_gated_silu",
        "lm_head_tied",
        "rope_default",
    } <= _ticked(report)
    hooks = plan["hooks"]
    assert hooks["createAttention"]["reuse"] == "hook:Qwen3Transformer::createAttention"
    assert hooks["createMlp"]["reuse"] == "hook:Transformer::createMlp"
    assert (
        hooks["createTransformerDecoderBlock"]["reuse"]
        == "hook:Transformer::createTransformerDecoderBlock"
    )
    assert hooks["constructModel"]["reuse"] == "hook:CausalLM::constructModel"
    assert plan["converter"]["reuse"] == "converter:res/qwen3/qwen3-0.6b/weight_converter.py"
    assert plan["summary"]["new"] == 0 and plan["summary"]["reuse"] == 4


def test_gemma3_reuses_gemma3_code_and_needs_the_norm_plus_one_converter():
    """Gemma 3 reuses Gemma 3 code and needs the norm plus one converter."""
    report, plan = _plan(GEMMA3, _decoder(qk_norm="per_head", norms=GEMMA3_NORMS))
    assert report.is_fully_explained, (report.unclaimed_tensors, report.unclaimed_config)
    ticked = _ticked(report)
    assert {
        "block_sandwich_norm",
        "mlp_gated_gelu_tanh",
        "local_rope_theta_10000",
        "embedding_scale_sqrt_hidden",
        "norm_one_plus_weight",
        "query_scale_standard",
        "sliding_window_pattern",
    } <= ticked
    assert "mlp_gated_silu" not in ticked
    assert plan["hooks"]["createAttention"]["reuse"] == "hook:Gemma3Transformer::createAttention"
    assert plan["hooks"]["createMlp"]["reuse"] == "hook:Gemma3Transformer::createMlp"
    assert (
        plan["hooks"]["createTransformerDecoderBlock"]["reuse"]
        == "hook:Gemma3Transformer::createTransformerDecoderBlock"
    )
    assert plan["converter"]["reuse"] == "converter:res/gemma3/weight_converter.py"


def test_qwen3_moe_reuses_the_verified_moe_layer_not_the_slim_variants():
    """Qwen3 MoE reuses the verified MoE layer not the slim variants."""
    report, plan = _plan(QWEN3_MOE, _moe_tensors())
    assert report.is_fully_explained, (report.unclaimed_tensors, report.unclaimed_config)
    assert plan["hooks"]["createMlp"]["reuse"] == "hook:Qwen3MoECausalLM::createMlp"
    assert (
        "hook:Qwen3SlimMoECausalLM::createMlp" in plan["hooks"]["createMlp"]["unverified_matches"]
    )


# --------------------------------------------- the reuse rule in practice
def test_llama2_matches_base_attention_but_it_is_not_verified():
    """Llama 2 matches base attention but it is not verified."""
    report, plan = _plan(LLAMA2, _decoder(tied=False))
    assert report.is_fully_explained, (report.unclaimed_tensors, report.unclaimed_config)
    att = plan["hooks"]["createAttention"]
    assert att["state"] == "needs_verification" and att["reuse"] is None
    assert att["unverified_matches"] == ["hook:Transformer::createAttention"]
    assert plan["hooks"]["createMlp"]["state"] == "reuse"


# ------------------------------------------------ what the check catches
def test_llama3_rope_scaling_is_a_new_part():
    """Llama 3 RoPE scaling is a new part."""
    report, plan = _plan(LLAMA32, _decoder())
    assert "rope_default" not in _ticked(report)
    assert set(report.unclaimed_config) == {"rope_scaling", "rope_theta"}
    assert plan["hooks"]["createAttention"]["state"] == "new"
    assert any(
        "rope_scaling" in p["what"] and p["hook"] == "createAttention" for p in plan["new_parts"]
    )


def test_gemma3_27b_query_scale_is_a_new_part():
    """Gemma 3 27B query scale is a new part."""
    cfg = dict(GEMMA3, query_pre_attn_scalar=168)  # head_dim is 16 here, 128 in the real 27B
    report, plan = _plan(cfg, _decoder(qk_norm="per_head", norms=GEMMA3_NORMS))
    assert report.unclaimed_config == {"query_pre_attn_scalar": 168}
    assert plan["hooks"]["createAttention"]["state"] == "new"


def test_gemma_softcapping_is_not_silently_reused():
    """Gemma softcapping is not silently reused."""
    cfg = dict(GEMMA3, attn_logit_softcapping=50.0)
    report, plan = _plan(cfg, _decoder(qk_norm="per_head", norms=GEMMA3_NORMS))
    assert "attn_logit_softcapping" in _ticked(report)
    assert (
        plan["hooks"]["createAttention"]["state"] == "new"
    )  # no verified hook handles a non-null value


def test_olmo2_style_model_flags_full_width_qk_norm_and_post_only_norms():
    """OLMo2 style model flags full width Q/K norm and post only norms."""
    cfg = _base_cfg(architectures=["Olmo2ForCausalLM"], model_type="olmo2", attention_bias=False,
                    hidden_act="silu", rope_scaling=None, tie_word_embeddings=False)
    tensors = _decoder(
        tied=False, qk_norm="full", norms=("post_attention_layernorm", "post_feedforward_layernorm")
    )
    report, plan = _plan(cfg, tensors)
    ticked = _ticked(report)
    assert (
        "qk_norm_per_head" not in ticked
        and "block_pre_norm" not in ticked
        and "block_sandwich_norm" not in ticked
    )
    patterns = {u["pattern"]: u["hook"] for u in report.unclaimed_tensors}
    assert patterns["layers.*.self_attn.q_norm.weight"] == "createAttention"
    assert patterns["layers.*.post_feedforward_layernorm.weight"] == "createTransformerDecoderBlock"
    assert patterns["layers.*.post_attention_layernorm.weight"] == "createTransformerDecoderBlock"
    assert plan["hooks"]["createAttention"]["state"] == "new"
    assert plan["hooks"]["createTransformerDecoderBlock"]["state"] == "new"
    assert plan["hooks"]["createAttention"]["examples"]  # nearest existing code to start from


def test_moe_without_topk_renormalisation_is_a_new_part():
    """MoE without top-k renormalisation is a new part."""
    report, plan = _plan(dict(QWEN3_MOE, norm_topk_prob=False), _moe_tensors())
    assert "moe_qwen" not in _ticked(report)
    assert plan["hooks"]["createMlp"]["state"] == "new"


def test_multimodal_checkpoint_reports_the_vision_tower():
    """Multimodal checkpoint reports the vision tower."""
    cfg = {"architectures": ["Gemma3ForConditionalGeneration"], "model_type": "gemma3",
           "text_config": dict(GEMMA3), "vision_config": {"hidden_size": 32}}
    tensors = _decoder(qk_norm="per_head", norms=GEMMA3_NORMS, prefix="language_model.model.")
    tensors["vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight"] = [32, 32]
    tensors["multi_modal_projector.mm_input_projection_weight"] = [32, H]
    report, plan = _plan(cfg, tensors)
    unclaimed = {u["pattern"] for u in report.unclaimed_tensors}
    assert unclaimed == {"vision_tower.vision_model.encoder.layers.*.self_attn.q_proj.weight",
                         "multi_modal_projector.mm_input_projection_weight"}
    assert plan["hooks"]["createAttention"]["state"] == "new"      # the vision part needs code


def test_names_without_shapes_tick_but_are_flagged_unconfirmed():
    """Names without shapes tick but are flagged unconfirmed."""
    names_only = {k: None for k in _decoder(qk_norm="per_head")}
    report, plan = _plan(QWEN3, names_only)
    qk_norm = next(f for f in report.features if f.id == "qk_norm_per_head")
    assert qk_norm.ticked and qk_norm.uncertain
    assert any(p["what"].startswith("check qk_norm_per_head") for p in plan["new_parts"])


def test_no_tensor_list_ticks_nothing_that_needs_tensors():
    """No tensor list ticks nothing that needs tensors."""
    report = ck.run_checklist(ck.make_facts("m", QWEN3, {}, "config only"))
    assert not report.tensors_known and not report.is_fully_explained
    assert "attn_gqa_rope" not in _ticked(report)


# ---------------------------------------------------- the learning loop
def test_approved_and_verified_new_code_is_reused_next_time():
    """Approved and verified new code is reused next time."""
    bank = _bank_copy()
    cfg = _base_cfg(
        architectures=["PostNormForCausalLM"],
        model_type="postnorm",
        attention_bias=False,
        hidden_act="silu",
        rope_scaling=None,
        tie_word_embeddings=True,
    )
    tensors = _decoder(
        qk_norm="per_head", norms=("post_attention_layernorm", "post_feedforward_layernorm")
    )
    learned = {
        "id": "block_post_norm_only",
        "label": "Norms only after attention and after the MLP",
        "hooks": ["createTransformerDecoderBlock"],
        "when": [
            {"tensor": "layers.*.post_feedforward_layernorm.weight"},
            {"no_tensor": "layers.*.input_layernorm.weight"},
        ],
        "claims": {"tensors": ["layers.*.post_{attention,feedforward}_layernorm.weight"]},
    }
    bank.put_feature(learned, source_entry="hook:test::postNormBlock")
    report, plan = _plan(cfg, tensors, bank.learned_features(), bank)
    assert report.is_fully_explained  # the learned box explains the norms
    assert (
        plan["hooks"]["createTransformerDecoderBlock"]["state"] == "new"
    )  # but no code for it yet
    bank.add_generated(
        {
            "id": "hook:test::postNormBlock",
            "kind": "hook",
            "name": "post-norm block",
            "hook": "createTransformerDecoderBlock",
            "features": ["block_post_norm_only"],
            "code": "Tensor createTransformerDecoderBlock(...) { /* generated */ }",
        }
    )
    _, plan = _plan(cfg, tensors, bank.learned_features(), bank)
    assert (
        plan["hooks"]["createTransformerDecoderBlock"]["state"] == "new"
    )  # generated, not approved
    bank.approve("hook:test::postNormBlock", actor="reviewer")
    _, plan = _plan(cfg, tensors, bank.learned_features(), bank)
    assert plan["hooks"]["createTransformerDecoderBlock"]["state"] == "needs_verification"
    bank.mark_verified("hook:test::postNormBlock", {"method": "hf-logits", "max_abs_diff": 0.003})
    _, plan = _plan(cfg, tensors, bank.learned_features(), bank)
    assert plan["hooks"]["createTransformerDecoderBlock"]["reuse"] == "hook:test::postNormBlock"


# --------------------------------------------------------- reading files
def _write_safetensors(path, tensors):
    header = {k: {"dtype": "BF16", "shape": s, "data_offsets": [0, 0]} for k, s in tensors.items()}
    header["__metadata__"] = {"format": "pt"}
    raw = json.dumps(header).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)


def test_facts_from_a_snapshot_read_only_headers():
    """Facts from a snapshot read only headers."""
    tmp_dir = tempfile.mkdtemp(dir=_TMP)
    with open(os.path.join(tmp_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(QWEN3, handle)
    _write_safetensors(os.path.join(tmp_dir, "model.safetensors"), _decoder(qk_norm="per_head"))
    facts = ck.facts_from_dir(tmp_dir)
    assert facts.shapes_known and facts.tensors["layers.*.self_attn.q_norm.weight"] == [HD]
    assert facts.tensor_counts["layers.*.self_attn.q_norm.weight"] == 2
    assert ck.run_checklist(facts).is_fully_explained


def test_tensor_names_are_normalised():
    """Tensor names are normalised."""
    assert (
        ck.normalise_tensor_name("model.layers.12.mlp.experts.3.up_proj.weight")
        == "layers.*.mlp.experts.*.up_proj.weight"
    )
    assert (
        ck.normalise_tensor_name("language_model.model.layers.0.self_attn.q_proj.weight")
        == "layers.*.self_attn.q_proj.weight"
    )
    assert ck.normalise_tensor_name("lm_head.weight") == "lm_head.weight"


# ------------------------------------------------------------------ capture
def test_capture_marks_verified_only_what_reference_tests_reach():
    """Capture marks verified only what reference tests reach."""
    bank = _captured_bank()
    status = lambda i: bank.get(i)["status"]                       # noqa: E731
    assert status("hook:Qwen3Transformer::createAttention") == "verified"
    assert status("hook:Gemma3Transformer::createAttention") == "verified"
    assert (
        status("hook:Transformer::createAttention") == "approved"
    )  # only Llama uses it; no reference test
    assert status("hook:GptOssForCausalLM::createAttention") == "approved"
    assert status("layer:qwen_moe") == "verified" and status("layer:moe_slim") == "approved"
    assert status("layer:tie_word_embeddings") == "verified"             # chosen through a variable
    assert status("framework:causallm-base") == "verified"
    verification = bank.get("hook:Qwen3Transformer::createAttention")["verification"]
    assert (
        verification["method"] == "upstream-reference-tests"
        and "unittest_causallm_qwen3_reference.cpp" in verification["tests"]
    )


def test_capture_keeps_everything_needed_without_the_models_folder():
    """Capture keeps everything needed without the models folder."""
    bank = _captured_bank()
    base = bank.get("framework:causallm-base")["files"]
    assert {
        "models/transformer.h",
        "models/transformer.cpp",
        "models/causal_lm.h",
        "models/causal_lm.cpp",
    } <= set(base)
    assert "models/qwen3_moe/qwen_moe_layer.cpp" in bank.get("layer:qwen_moe")["files"]
    assert "models/gemma3/gemma3_causallm.cpp" in bank.get("model:gemma3")["files"]
    assert (
        bank.get("hook:Qwen3Transformer::createAttention")["code"]
        .lstrip("/* \n")
        .count("createAttention")
        >= 1
    )


def test_capture_is_idempotent():
    """Capture is idempotent."""
    bank = _captured_bank()
    again = capture(find_nntrainer_root(), bank)
    assert set(again["writes"]) == {"unchanged"}, again["writes"]
