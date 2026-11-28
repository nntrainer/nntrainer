# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file test_graph_views.py
# @brief Tests for the graph-pane payloads the webview renders.
# @author prachi-t17 <prachi.t@samsung.com>

# Tests call private helpers and import the engine from the folder that run_all.py
# puts on sys.path at run time (some only after installing stand-ins), which pylint
# cannot follow when it checks this file on its own.
# pylint: disable=protected-access,import-error,wrong-import-position

"""
Graph-pane payloads the webview renders: node ids that line up between the
two panes (click-to-highlight), the inspector's Properties/Weights data,
weight-preview merging, and which saved view the focus box searches.
No third-party dependencies, like test_semantic_lowering.py.
"""
import os
import sys

_ENGINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ENGINE_ROOT not in sys.path:
    sys.path.insert(0, _ENGINE_ROOT)

from agents import dual_graph, graph_focus_agent, graph_views  # noqa: E402

from api.lowering.nntrainer.lowerer import NNTrainerLowerer  # noqa: E402
from api.semantic.model import (  # noqa: E402
    AttentionIR, CausalLMIR, DecoderLayerIR, MLPIR, NormIR, ProjectionIR,
)

HIDDEN, VOCAB, HEADS, KV_HEADS, HEAD_DIM, FFN = 256, 1000, 8, 4, 32, 512


def _proj(name, out, inp, bias=False):
    return ProjectionIR(name, out, bias=bias, weight_name=f"{name}.weight", input_size=inp)


def _layer(i):
    path = f"model.layers.{i}"
    return DecoderLayerIR(
        index=i,
        input_norm=NormIR(f"{path}.input_layernorm", "rms_norm", HIDDEN, 1e-6),
        attention=AttentionIR(
            source_name=f"{path}.self_attn",
            q_proj=_proj(f"{path}.self_attn.q_proj", HEADS * HEAD_DIM, HIDDEN),
            k_proj=_proj(f"{path}.self_attn.k_proj", KV_HEADS * HEAD_DIM, HIDDEN),
            v_proj=_proj(f"{path}.self_attn.v_proj", KV_HEADS * HEAD_DIM, HIDDEN),
            o_proj=_proj(f"{path}.self_attn.o_proj", HIDDEN, HEADS * HEAD_DIM),
            num_heads=HEADS, num_kv_heads=KV_HEADS, head_dim=HEAD_DIM,
            q_norm=NormIR(f"{path}.self_attn.q_norm", "rms_norm", HEAD_DIM, 1e-6, reshaped=True),
            k_norm=NormIR(f"{path}.self_attn.k_norm", "rms_norm", HEAD_DIM, 1e-6, reshaped=True),
            rope_theta=10000.0, max_position_embeddings=4096,
        ),
        post_attention_norm=NormIR(f"{path}.post_attention_layernorm", "rms_norm", HIDDEN, 1e-6),
        mlp=MLPIR(
            source_name=f"{path}.mlp",
            up_proj=_proj(f"{path}.mlp.up_proj", FFN, HIDDEN),
            down_proj=_proj(f"{path}.mlp.down_proj", HIDDEN, FFN),
            activation="silu", gated=True,
            gate_proj=_proj(f"{path}.mlp.gate_proj", FFN, HIDDEN),
        ),
    )


def _model_ir(num_layers=5, tied=False):
    return CausalLMIR(
        architecture="qwen3", hidden_size=HIDDEN, vocab_size=VOCAB, num_layers=num_layers,
        embedding_name="model.embed_tokens",
        decoder_layers=[_layer(i) for i in range(num_layers)],
        final_norm=NormIR("model.norm", "rms_norm", HIDDEN, 1e-6),
        lm_head_name="lm_head", tied_embeddings=tied,
    )


def _build(num_layers=5, tied=False):
    model_ir = _model_ir(num_layers, tied)
    graph = NNTrainerLowerer(model_ir).lower()
    model_view = graph_views.build_model_graph_view(model_ir, "bfloat16")
    nntrainer_view = graph_views.build_nntrainer_graph_view(graph.export(), "bfloat16")
    mappings = graph_views.build_node_mappings(graph, model_view)
    return model_ir, graph, model_view, nntrainer_view, mappings


def _by_id(view):
    return {n["id"]: n for n in view["nodes"]}


# ------------------------------------------------------- click-to-highlight
def test_mapping_ids_are_the_ids_both_panes_render():
    """Mapping ids are the ids both panes render."""
    _, _, model_view, nntrainer_view, mappings = _build()
    model_ids, nntrainer_ids = set(_by_id(model_view)), set(_by_id(nntrainer_view))
    assert mappings
    for mapping in mappings:
        assert set(mapping["sourceIds"]) <= model_ids, mapping
        assert set(mapping["targetIds"]) <= nntrainer_ids, mapping


def test_every_box_in_either_pane_has_a_counterpart():
    """Every box in either pane has a counterpart."""
    _, _, model_view, nntrainer_view, mappings = _build()
    mapped_sources = {s for m in mappings for s in m["sourceIds"]}
    mapped_targets = {t for m in mappings for t in m["targetIds"]}
    assert set(_by_id(model_view)) == mapped_sources
    assert set(_by_id(nntrainer_view)) == mapped_targets


def test_mapping_is_per_box_not_per_layer():
    """Mapping is per box not per layer."""
    _, graph, _, _, mappings = _build()
    q_proj = next(n for n in graph.get_nodes() if n.name == "model.layers.0.self_attn.q_proj")
    entry = next(m for m in mappings if q_proj.id in m["targetIds"])
    assert entry["sourceIds"] == ["decoder_0.wq"]
    assert entry["targetIds"] == [q_proj.id]


def test_kv_cache_maps_to_the_attention_box():
    """KV cache maps to the attention box."""
    _, graph, _, _, mappings = _build()
    by_name = {n.name: n for n in graph.get_nodes()}
    attention = next(m for m in mappings if m["sourceIds"] == ["decoder_0.attention"])
    assert set(attention["targetIds"]) == {
        by_name["model.layers.0.self_attn.mha_core"].id,
        by_name["model.layers.0.self_attn.kv_cache"].id,
    }


def test_both_panes_collapse_the_same_middle_layers():
    """Both panes collapse the same middle layers."""
    _, _, model_view, nntrainer_view, mappings = _build(num_layers=5)
    collapsed = {f"decoder_{i}_collapsed" for i in (1, 2, 3)}
    assert collapsed <= set(_by_id(model_view))
    assert collapsed <= set(_by_id(nntrainer_view))
    for cid in collapsed:
        entry = next(m for m in mappings if m["sourceIds"] == [cid])
        assert entry["targetIds"] == [cid]


def test_short_models_are_not_collapsed():
    """Short models are not collapsed."""
    _, _, model_view, nntrainer_view, _ = _build(num_layers=3)
    for view in (model_view, nntrainer_view):
        assert not any(n["id"].endswith("_collapsed") for n in view["nodes"])


# ---------------------------------------------------------------- inspector
def test_nntrainer_pane_weight_info_has_shape_params_and_dtype():
    """Nntrainer pane weight info has shape params and dtype."""
    _, _, _, nntrainer_view, _ = _build()
    q_node = next(
        n for n in nntrainer_view["nodes"] if n["label"] == "model.layers.0.self_attn.q_proj"
    )
    assert q_node["weightInfo"] == {
        "name": "model.layers.0.self_attn.q_proj.weight",
        "shape": [HEADS * HEAD_DIM, HIDDEN], "dtype": "bfloat16",
        "params": HEADS * HEAD_DIM * HIDDEN,
    }
    assert q_node["inputShape"] == ["seq", HIDDEN]
    assert q_node["outputShape"] == ["seq", HEADS * HEAD_DIM]


def test_q_norm_keeps_the_projection_output_shape():
    """Q norm keeps the projection output shape."""
    _, _, _, nntrainer_view, _ = _build()
    q_norm = next(
        n for n in nntrainer_view["nodes"] if n["label"] == "model.layers.0.self_attn.q_norm"
    )
    assert q_norm["inputShape"] == q_norm["outputShape"] == ["seq", HEADS * HEAD_DIM]
    assert q_norm["weightInfo"]["shape"] == [HEAD_DIM]


def test_model_pane_shows_properties_and_weights():
    """Model pane shows properties and weights."""
    _, _, model_view, _, _ = _build()
    nodes = _by_id(model_view)
    decoder_0_wq = nodes["decoder_0.wq"]
    assert decoder_0_wq["attributes"]["hf_module"] == "model.layers.0.self_attn.q_proj"
    assert decoder_0_wq["attributes"]["in_features"] == HIDDEN
    assert decoder_0_wq["inputShape"] == ["seq", HIDDEN]
    assert decoder_0_wq["weightInfo"]["shape"] == [HEADS * HEAD_DIM, HIDDEN]
    attention = nodes["decoder_0.attention"]
    assert attention["attributes"]["num_kv_heads"] == KV_HEADS
    assert attention["outputShape"] == ["seq", HEADS * HEAD_DIM]
    assert nodes["decoder_1_collapsed"]["attributes"]["parameters"] > 0


def test_tied_lm_head_is_labelled_and_not_double_counted():
    """Tied LM head is labelled and not double counted."""
    _, _, model_view, nntrainer_view, _ = _build(tied=True)
    for view in (model_view, nntrainer_view):
        lm_head = next(n for n in view["nodes"] if n.get("semanticType") == "lm_head")
        assert lm_head["weightInfo"]["name"] == "model.embed_tokens.weight"
        assert lm_head["weightInfo"]["params"] == 0
        assert "tied" in lm_head["weightInfo"]["note"]


def test_display_fields_never_become_layer_properties():
    """Display fields never become layer properties."""
    # `attributes` on lowered decoder nodes are emitted verbatim as C++
    # createLayer properties, so shapes/weight info must stay out of them.
    _, graph, _, _, _ = _build()
    q_node = next(n for n in graph.get_nodes() if n.name == "model.layers.0.self_attn.q_proj")
    assert set(q_node.attributes) == {"unit", "disable_bias"}


def test_weight_previews_merge_into_matching_nodes():
    """Weight previews merge into matching nodes."""
    _, _, model_view, _, _ = _build()
    stats = {
        "dtype": "float16",
        "min": -1.0,
        "max": 1.0,
        "mean": 0.0,
        "std": 0.5,
        "preview": [0.1, 0.2],
    }
    updated = graph_views.apply_weight_previews(
        model_view, {"model.layers.0.self_attn.q_proj.weight": stats}
    )
    assert updated == 1
    info = _by_id(model_view)["decoder_0.wq"]["weightInfo"]
    assert info["preview"] == [0.1, 0.2] and info["dtype"] == "float16"


# ---------------------------------------------------------------- focus box
def test_focus_box_searches_the_graph_each_pane_draws():
    """Focus box searches the graph each pane draws."""
    state = {
        "model_graph_view": {"nodes": [{"id": "m", "type": "x", "label": "Q Projection"}]},
        "nntrainer_graph_view": {"nodes": [{"id": "n", "type": "fully_connected", "label": "q"}]},
        "graph_view": {"nodes": [{"id": "internal", "type": "x", "label": "Q Projection"}]},
    }
    assert [n["id"] for n in graph_focus_agent._graph_nodes(state, "ini")] == ["m"]
    assert [n["id"] for n in graph_focus_agent._graph_nodes(state, "cpp")] == ["n"]


def test_focus_keywords_match_role_and_hf_module():
    """Focus keywords match role and HF module."""
    _, _, model_view, _, _ = _build()
    state = {"model_graph_view": model_view}
    ids = graph_focus_agent._keyword_match(state, "ini", "self attention")
    assert "decoder_0.wq" in ids and "decoder_0.attention" in ids
    assert "decoder_0.up" not in ids


# ------------------------------------------------------------ fallback path
def test_fallback_mapping_pairs_cpp_layers_with_ini_sections():
    """Fallback mapping pairs C++ layers with INI sections."""
    model_view = {"nodes": [{"id": "encoder_layer_0_dense"}]}
    nntrainer_view = {
        "nodes": [{"id": "encoder.layer.0.dense"}, {"id": "encoder.layer.0.unsupported_op"}]
    }
    mappings = dual_graph._fallback_node_mappings(model_view, nntrainer_view)
    assert mappings == [{
        "sourceIds": ["encoder_layer_0_dense"], "targetIds": ["encoder.layer.0.dense"],
        "mappingType": "one_to_one", "description": "encoder.layer.0.dense",
    }]


def test_fallback_views_get_shapes_and_full_weight_names():
    """Fallback views get shapes and full weight names."""
    view = {"nodes": [{"id": "encoder.dense", "weightInfo": {"name": "weight", "shape": [4, 8]}}]}
    traced = {"encoder.dense": {"name": "encoder.dense", "weight_name": "weight",
                                "input_shape": [1, 8], "output_shape": [1, 4]}}
    dual_graph._enrich_from_traced_graph(view, traced)
    node = view["nodes"][0]
    assert node["inputShape"] == [1, 8] and node["outputShape"] == [1, 4]
    assert node["weightInfo"]["name"] == "encoder.dense.weight"
