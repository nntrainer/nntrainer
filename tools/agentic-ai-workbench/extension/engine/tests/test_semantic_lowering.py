# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file test_semantic_lowering.py
# @brief Tests for the semantic IR and its lowering to nntrainer.
# @author prachi-t17 <prachi.t@samsung.com>

# Tests call private helpers and import the engine from the folder that run_all.py
# puts on sys.path at run time (some only after installing stand-ins), which pylint
# cannot follow when it checks this file on its own.
# pylint: disable=import-error,wrong-import-position,import-outside-toplevel

"""
These modules cross-import within the `api` package (api.semantic,
api.lowering.nntrainer, api.graph), so -- unlike the single-file
`_util.load()` tests elsewhere in this folder -- they need the engine
root on sys.path rather than being loaded in isolation. Still zero
third-party dependencies: the lowerer/validator only ever touch the
plain-data CausalLMIR, never a real torch model, so no torch import
happens anywhere in this test.
"""
import os
import sys
import shutil
import tempfile

_ENGINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ENGINE_ROOT not in sys.path:
    sys.path.insert(0, _ENGINE_ROOT)

from converters.cpp_generator import CPPGenerator  # noqa: E402

from api.lowering.nntrainer.lowerer import NNTrainerLowerer  # noqa: E402
from api.lowering.nntrainer.manifest import (  # noqa: E402
    build_model_metadata, build_weight_manifest, validate_weight_manifest,
)
from api.lowering.nntrainer.validation import validate  # noqa: E402
from api.semantic.model import (  # noqa: E402
    AttentionIR, CausalLMIR, DecoderLayerIR, MLPIR, NormIR, ProjectionIR,
)


def _make_layer(index: int, qk_norm: bool = True) -> DecoderLayerIR:
    attn = AttentionIR(
        source_name=f"model.layers.{index}.self_attn",
        q_proj=ProjectionIR(f"model.layers.{index}.self_attn.q_proj", 128),
        k_proj=ProjectionIR(f"model.layers.{index}.self_attn.k_proj", 64),
        v_proj=ProjectionIR(f"model.layers.{index}.self_attn.v_proj", 64),
        o_proj=ProjectionIR(f"model.layers.{index}.self_attn.o_proj", 256),
        num_heads=8,
        num_kv_heads=4,
        head_dim=16,
        # reshaped=True: this is the real adapter's behavior (see
        # api/adapters/llama_family.py) -- Q/K norm operates per-head on
        # the reshaped tensor, unlike the plain decoder-level norms.
        q_norm=(
            NormIR(f"model.layers.{index}.self_attn.q_norm", "rms_norm", 16, 1e-6, reshaped=True)
            if qk_norm
            else None
        ),
        k_norm=(
            NormIR(f"model.layers.{index}.self_attn.k_norm", "rms_norm", 16, 1e-6, reshaped=True)
            if qk_norm
            else None
        ),
        rope_theta=10000.0,
        max_position_embeddings=4096,
    )
    mlp = MLPIR(
        source_name=f"model.layers.{index}.mlp",
        up_proj=ProjectionIR(f"model.layers.{index}.mlp.up_proj", 512),
        down_proj=ProjectionIR(f"model.layers.{index}.mlp.down_proj", 256),
        activation="silu", gated=True,
        gate_proj=ProjectionIR(f"model.layers.{index}.mlp.gate_proj", 512),
    )
    return DecoderLayerIR(
        index=index,
        input_norm=NormIR(f"model.layers.{index}.input_layernorm", "rms_norm", 256, 1e-6),
        attention=attn,
        post_attention_norm=NormIR(
            f"model.layers.{index}.post_attention_layernorm", "rms_norm", 256, 1e-6
        ),
        mlp=mlp,
    )


def _make_model_ir(num_layers=3, qk_norm=True) -> CausalLMIR:
    return CausalLMIR(
        architecture="qwen3",
        hidden_size=256,
        vocab_size=32000,
        num_layers=num_layers,
        embedding_name="model.embed_tokens",
        decoder_layers=[_make_layer(i, qk_norm) for i in range(num_layers)],
        final_norm=NormIR("model.norm", "rms_norm", 256, 1e-6),
        lm_head_name="lm_head",
        tied_embeddings=False,
    )


def test_semantic_ir_round_trips_through_dict():
    """Semantic IR round trips through dict."""
    model_ir = _make_model_ir()
    restored = CausalLMIR.from_dict(model_ir.to_dict())
    assert restored.architecture == "qwen3"
    assert len(restored.decoder_layers) == 3
    assert restored.decoder_layers[1].attention.q_norm.epsilon == 1e-6


def test_uniform_layers_detected_when_structurally_identical():
    """Uniform layers detected when structurally identical."""
    model_ir = _make_model_ir(num_layers=4)
    assert model_ir.uniform_layer_signature() is not None


def test_uniform_layers_none_when_layers_differ():
    """Uniform layers none when layers differ."""
    model_ir = _make_model_ir(num_layers=2, qk_norm=True)
    model_ir.decoder_layers[1] = _make_layer(1, qk_norm=False)
    assert model_ir.uniform_layer_signature() is None


def test_lowerer_wires_qkv_from_the_same_normalized_input():
    """Lowerer wires QKV from the same normalized input."""
    model_ir = _make_model_ir(num_layers=1)
    graph = NNTrainerLowerer(model_ir).lower()

    wq_nodes = next(n for n in graph.get_nodes() if n.template_id.endswith(".wq"))
    wk_nodes = next(n for n in graph.get_nodes() if n.template_id.endswith(".wk"))
    wv_nodes = next(n for n in graph.get_nodes() if n.template_id.endswith(".wv"))
    assert wq_nodes.inputs == wk_nodes.inputs == wv_nodes.inputs
    assert len(wq_nodes.inputs) == 1


def test_lowerer_produces_two_residual_adds_per_layer():
    """Lowerer produces two residual adds per layer."""
    model_ir = _make_model_ir(num_layers=1)
    graph = NNTrainerLowerer(model_ir).lower()
    adds = [n for n in graph.get_nodes() if n.group_id == "decoder_0" and n.node_type == "addition"]
    assert len(adds) == 2


def test_lowered_graph_passes_validation():
    """Lowered graph passes validation."""
    model_ir = _make_model_ir(num_layers=2)
    graph = NNTrainerLowerer(model_ir).lower()
    result = validate(graph, model_ir)
    assert result.passed, [d.message for d in result.diagnostics if d.severity == "error"]


def test_validation_catches_broken_qkv_wiring():
    """Validation catches broken QKV wiring."""
    model_ir = _make_model_ir(num_layers=1)
    graph = NNTrainerLowerer(model_ir).lower()
    wk_nodes = next(n for n in graph.get_nodes() if n.template_id.endswith(".wk"))
    wk_nodes.inputs = []  # deliberately break K's input wiring
    result = validate(graph, model_ir)
    assert not result.passed
    assert any("share the same input" in d.message for d in result.diagnostics)


def test_mha_core_receives_q_k_v_and_one_cache_node():
    """MHA core receives Q, K, V and one cache node."""
    model_ir = _make_model_ir(num_layers=1)
    graph = NNTrainerLowerer(model_ir).lower()
    mha = next(n for n in graph.get_nodes() if n.node_type == "mha_core")
    # q, k, v, kv_cache_placeholders -- the two cache tensors are one
    # node (see api/lowering/nntrainer/lowerer.py), not two, so the C++
    # emitter can destructure them together via createKVCachePlaceholders()
    # instead of pretending they're two ordinary createLayer() calls.
    assert len(mha.inputs) == 4
    cache_pred = next(
        n
        for n in graph.get_nodes()
        if n.id in mha.inputs and n.node_type == "kv_cache_placeholders"
    )
    assert cache_pred.status == "host_managed"


def test_qwen3_qk_norm_lowered_as_reshaped_rms_norm():
    """Qwen3 Q/K norm lowered as reshaped RMS norm."""
    model_ir = _make_model_ir(num_layers=1)
    graph = NNTrainerLowerer(model_ir).lower()
    q_norm = next(n for n in graph.get_nodes() if n.template_id.endswith(".q_norm"))
    k_norm = next(n for n in graph.get_nodes() if n.template_id.endswith(".k_norm"))
    assert q_norm.node_type == "reshaped_rms_norm"
    assert k_norm.node_type == "reshaped_rms_norm"


def test_decoder_level_norm_stays_plain_rms_norm():
    """Decoder level norm stays plain RMS norm."""
    # Only Q/K norm is reshaped -- the ordinary per-token input/post-attn
    # norms must NOT be swept into reshaped_rms_norm by the same fix.
    model_ir = _make_model_ir(num_layers=1)
    graph = NNTrainerLowerer(model_ir).lower()
    input_norm = next(n for n in graph.get_nodes() if n.template_id.endswith(".input_norm"))
    assert input_norm.node_type == "rms_norm"


def test_silu_lowered_to_activation_layer_with_swish_property():
    """SiLU lowered to activation layer with swish property."""
    model_ir = _make_model_ir(num_layers=1)
    graph = NNTrainerLowerer(model_ir).lower()
    activation = next(n for n in graph.get_nodes() if n.template_id.endswith(".act"))
    assert activation.node_type == "activation"
    assert activation.attributes["activation"] == "swish"


def test_projection_bias_lowered_to_disable_bias_property():
    """Projection bias lowered to disable bias property."""
    model_ir = _make_model_ir(num_layers=1)
    graph = NNTrainerLowerer(model_ir).lower()
    wq_nodes = next(n for n in graph.get_nodes() if n.template_id.endswith(".wq"))
    assert "disable_bias" in wq_nodes.attributes
    assert "bias" not in wq_nodes.attributes
    assert wq_nodes.attributes["disable_bias"] in ("true", "false")


def test_unknown_activation_raises_instead_of_guessing():
    """Unknown activation raises instead of guessing."""
    model_ir = _make_model_ir(num_layers=1)
    model_ir.decoder_layers[0].mlp.activation = "totally_made_up_activation"
    try:
        NNTrainerLowerer(model_ir).lower()
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_cpp_does_not_create_external_tensor_layer():
    """C++ does not create external tensor layer."""
    model_ir = _make_model_ir(num_layers=1)
    graph = NNTrainerLowerer(model_ir).lower()
    code = CPPGenerator(graph).generate()
    assert 'createLayer("external_tensor"' not in code
    assert "createKVCachePlaceholders" in code


def test_disable_bias_property_appears_in_generated_cpp():
    """Disable bias property appears in generated C++."""
    model_ir = _make_model_ir(num_layers=1)
    graph = NNTrainerLowerer(model_ir).lower()
    code = CPPGenerator(graph).generate()
    assert "disable_bias=" in code


def test_decoder_layer_method_declared_class_qualified():
    """Decoder layer method declared class qualified."""
    model_ir = _make_model_ir(num_layers=3)  # uniform -> component generation succeeds
    graph = NNTrainerLowerer(model_ir).lower()
    files = CPPGenerator(graph).generate_component()
    assert f"{files.transformer_class}::createDecoderLayer" in files.source
    assert f"{files.transformer_class}::createAttention" in files.source
    assert f"{files.transformer_class}::createMLP" in files.source
    # createAttention/createMLP are called from createDecoderLayer, and
    # C++ only needs them declared (in the header) before that call, not
    # defined-before-use within the .cpp -- so it's enough that the
    # header declares all three before the source defines any of them.
    assert files.header.index("createAttention") < files.header.index("createDecoderLayer")


def test_causallm_component_never_references_build_model_or_main():
    """CausalLM component never references build model or main."""
    model_ir = _make_model_ir(num_layers=6)
    graph = NNTrainerLowerer(model_ir).lower()
    files = CPPGenerator(graph).generate_component()
    combined = files.header + files.source
    assert "buildModel()" not in combined
    assert "int main(" not in combined
    assert "Tensor forward(Tensor input)" not in combined


def test_non_uniform_layers_refuse_component_generation():
    """Non uniform layers refuse component generation."""
    model_ir = _make_model_ir(num_layers=2, qk_norm=True)
    model_ir.decoder_layers[1] = _make_layer(1, qk_norm=False)
    graph = NNTrainerLowerer(model_ir).lower()
    try:
        CPPGenerator(graph).generate_component()
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_generated_class_names_follow_architecture():
    """Generated class names follow architecture."""
    model_ir = _make_model_ir(num_layers=1)
    graph = NNTrainerLowerer(model_ir).lower()
    files = CPPGenerator(graph).generate_component()
    assert files.transformer_class == "GeneratedQwen3Transformer"
    assert files.causal_lm_class == "GeneratedQwen3CausalLM"


def test_register_custom_layers_emitted_when_reshaped_norm_present():
    """Register custom layers emitted when reshaped norm present."""
    model_ir = _make_model_ir(num_layers=1, qk_norm=True)
    graph = NNTrainerLowerer(model_ir).lower()
    files = CPPGenerator(graph).generate_component()
    assert "ReshapedRMSNormLayer" in files.source
    assert f"{files.causal_lm_class}::registerCustomLayers" in files.source


def test_model_metadata_includes_core_architecture_fields():
    """Model metadata includes core architecture fields."""
    model_ir = _make_model_ir(num_layers=4)
    metadata = build_model_metadata(model_ir, source_model="Qwen/Qwen3-test")
    assert metadata["architecture"] == "qwen3"
    assert metadata["source_model"] == "Qwen/Qwen3-test"
    assert metadata["num_hidden_layers"] == 4
    assert metadata["qk_norm"] is True
    assert metadata["emission_mode"] == "causallm_component"


def test_weight_manifest_templates_layer_index_and_covers_ungrouped_weights():
    """Weight manifest templates layer index and covers ungrouped weights."""
    model_ir = _make_model_ir(num_layers=5)
    manifest = build_weight_manifest(model_ir)

    wq_entries = [e for e in manifest if e["target"] == "layer{layer_id}_wq/weight"]
    assert len(wq_entries) == 1  # one templated entry covers all 5 layers, not 5 entries
    assert wq_entries[0]["source"] == "model.layers.{layer_id}.self_attn.q_proj.weight"
    assert wq_entries[0]["layer_count"] == 5

    targets = {e["target"] for e in manifest}
    assert "embedding/weight" in targets
    assert "final_norm/weight" in targets
    assert "lm_head/weight" in targets


def test_weight_manifest_marks_tied_embeddings_instead_of_duplicating():
    """Weight manifest marks tied embeddings instead of duplicating."""
    model_ir = _make_model_ir(num_layers=1)
    model_ir.tied_embeddings = True
    manifest = build_weight_manifest(model_ir)
    lm_head_entry = next(e for e in manifest if e["target"] == "lm_head/weight")
    assert lm_head_entry["transform"] == "tied_with_embedding"
    assert lm_head_entry["source"] == model_ir.embedding_name + ".weight"


def test_validate_weight_manifest_catches_shape_mismatch():
    """Validate weight manifest catches shape mismatch."""
    manifest = [{
        "source": "a", "target": "b",
        "source_shape": [128, 64], "target_shape": [128, 32],
    }]
    problems = validate_weight_manifest(manifest)
    assert problems and "shape mismatch" in problems[0]


def test_causallm_install_noop_when_disabled_by_default():
    """CausalLM install no-op when disabled by default."""
    from agents import causallm_install
    state = {
        "causallm_header_path": "/tmp/does-not-matter.h",
        "causallm_source_path": "/tmp/does-not-matter.cpp",
        "install_generated_files": False,
    }
    result = causallm_install.run(state)
    assert "causallm_installed_header_path" not in result


def test_causallm_install_noop_when_nothing_was_generated():
    """CausalLM install no-op when nothing was generated."""
    from agents import causallm_install
    state = {"install_generated_files": True, "causallm_project_root": "/tmp"}
    result = causallm_install.run(state)  # no causallm_header_path/source_path
    assert "causallm_installed_header_path" not in result


def test_causallm_install_copies_files_when_enabled_and_root_exists():
    """CausalLM install copies files when enabled and root exists."""
    from agents import causallm_install

    tmp = tempfile.mkdtemp()
    try:
        src_dir = os.path.join(tmp, "src_out")
        os.makedirs(src_dir)
        header_path = os.path.join(src_dir, "generated_qwen3_causallm.h")
        source_path = os.path.join(src_dir, "generated_qwen3_causallm.cpp")
        with open(header_path, "w", encoding="utf-8") as handle:
            handle.write("// header\n")
        with open(source_path, "w", encoding="utf-8") as handle:
            handle.write("// source\n")

        project_root = os.path.join(tmp, "causallm_project")
        os.makedirs(project_root)

        state = {
            "causallm_header_path": header_path,
            "causallm_source_path": source_path,
            "install_generated_files": True,
            "causallm_project_root": project_root,
            "generated_header_directory": "include/generated",
            "generated_source_directory": "src/generated",
        }
        result = causallm_install.run(state)
        assert os.path.exists(result["causallm_installed_header_path"])
        assert os.path.exists(result["causallm_installed_source_path"])
        assert result["causallm_installed_header_path"].startswith(
            os.path.join(project_root, "include", "generated")
        )
    finally:
        shutil.rmtree(tmp)


def test_causallm_install_skips_gracefully_when_project_root_missing():
    """CausalLM install skips gracefully when project root missing."""
    from agents import causallm_install
    state = {
        "causallm_header_path": "/tmp/does-not-matter.h",
        "causallm_source_path": "/tmp/does-not-matter.cpp",
        "install_generated_files": True,
        "causallm_project_root": "/definitely/does/not/exist/on/this/machine",
    }
    result = causallm_install.run(state)
    assert "causallm_installed_header_path" not in result
    # Must not raise, and must not add a hard pipeline error for a config problem.
