# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file dual_graph.py
# @brief Dual Graph Agent (no LLM).
# @author prachi-t17 <prachi.t@samsung.com>

"""
Dual Graph Agent (no LLM).

Publishes exactly two graphs to the webview -- "Model Graph" (target
"model") and "nntrainer Graph" (target "nntrainer") -- plus the
node-mapping table that drives click-to-highlight between them (see
agents/nntrainer_lowering.py and agents/graph_views.py).

For architectures with a matched semantic adapter, both graphs come
straight from the IR built earlier in the pipeline (semantic_ir /
nntrainer_graph_ir) -- never by re-parsing generated text. For
architectures with no adapter (state["semantic_ir"] is None), the two
tabs fall back to parsing the generated files: "Model Graph" from
model.ini, "nntrainer Graph" from generated_model.cpp. Either
way the webview only ever sees two tabs.
"""
import os

from .events import bus
from .graph_views import build_ini_graph, build_cpp_graph
from .ini_generator import _section_name


def run(state: dict) -> dict:
    """Pipeline step: pair the Model Graph and nntrainer Graph nodes for click-to-highlight."""
    bus.agent_status("dual_graph", "running")

    if state.get("nntrainer_graph_ir") is not None:
        _publish_semantic_graphs(state)
    else:
        _publish_fallback_graphs(state)

    mappings = state.get("node_mappings") or []
    bus.node_mappings(mappings)
    if mappings:
        bus.log(
            f"Node mapping table: {len(mappings)} source<->target group(s) for click-to-highlight"
        )

    bus.agent_status("dual_graph", "done")
    return state


def _publish_semantic_graphs(state: dict) -> None:
    model_view = state.get("model_graph_view") or {"nodes": [], "edges": []}
    nntrainer_view = state.get("nntrainer_graph_view") or {"nodes": [], "edges": []}

    bus.graph(model_view, target="model")
    bus.log(
        f"Model Graph (semantic): {len(model_view['nodes'])} nodes,"
        f" {len(model_view['edges'])} edges"
    )

    bus.graph(nntrainer_view, target="nntrainer")
    bus.log(
        f"nntrainer Graph (target): {len(nntrainer_view['nodes'])} nodes,"
        f" {len(nntrainer_view['edges'])} edges"
    )

    if state.get("ini_content"):
        bus.file_content("model", "model.ini", state["ini_content"])
    _publish_cpp_files(state)


def _publish_cpp_files(state: dict) -> None:
    """The files behind the nntrainer pane's Export button, under the names
    they were written to disk with. A CausalLM component is a header plus
    a source file, so it's sent as two files rather than one concatenated
    blob labelled generated_model.cpp."""
    if state.get("cpp_emission_mode") == "causallm_component" and state.get("causallm_source"):
        for content_key, path_key, fallback in (
            ("causallm_header", "causallm_header_path", "generated_causallm.h"),
            ("causallm_source", "causallm_source_path", "generated_causallm.cpp"),
        ):
            if state.get(content_key):
                name = os.path.basename(state.get(path_key) or "") or fallback
                bus.file_content("nntrainer", name, state[content_key])
    elif state.get("cpp_code"):
        name = os.path.basename(state.get("cpp_path") or "") or "generated_model.cpp"
        bus.file_content("nntrainer", name, state["cpp_code"])


def _publish_fallback_graphs(state: dict) -> None:
    ini_content = state.get("ini_content")
    cpp_code = state.get("cpp_code")
    model_view = nntrainer_view = None
    traced = _traced_nodes_by_name(state.get("graph_ir"))

    if ini_content:
        model_view = build_ini_graph(ini_content)
        _enrich_from_traced_graph(
            model_view, {_section_name(name): n for name, n in traced.items()}
        )
        state["model_graph_view"] = model_view
        bus.graph(model_view, target="model")
        bus.file_content("model", "model.ini", ini_content)
        bus.log(
            f"Model Graph (.ini): {len(model_view['nodes'])} nodes,"
            f" {len(model_view['edges'])} edges"
        )
    else:
        bus.log("No model.ini content available -- skipping Model Graph", "warn")

    if cpp_code:
        nntrainer_view = build_cpp_graph(cpp_code)
        _enrich_from_traced_graph(nntrainer_view, traced)
        state["nntrainer_graph_view"] = nntrainer_view
        bus.graph(nntrainer_view, target="nntrainer")
        _publish_cpp_files(state)
        bus.log(
            f"nntrainer Graph (.cpp): {len(nntrainer_view['nodes'])} nodes,"
            f" {len(nntrainer_view['edges'])} edges"
        )
    else:
        bus.log("No generated_model.cpp content available -- skipping nntrainer Graph", "warn")

    if model_view and nntrainer_view:
        state["node_mappings"] = _fallback_node_mappings(model_view, nntrainer_view)

    # Kept for backward compatibility with anything still reading these keys.
    state["ini_graph_view"] = state.get("model_graph_view")
    state["cpp_graph_view"] = state.get("nntrainer_graph_view")


def _traced_nodes_by_name(graph_ir) -> dict:
    return {n["name"]: n for n in (graph_ir or {}).get("nodes", []) if n.get("name")}


def _enrich_from_traced_graph(view: dict, traced_by_id: dict) -> None:
    """The .ini/.cpp text these views are parsed from carries no tensor
    shapes, and its weight comments name only the parameter ("weight"),
    not the module. Fill both in from the traced module-tree graph, keyed
    by the id each pane uses (layer name for .cpp, section name for .ini)."""
    for node in view.get("nodes", []):
        traced = traced_by_id.get(node["id"])
        if not traced:
            continue
        node.setdefault("inputShape", traced.get("input_shape"))
        node.setdefault("outputShape", traced.get("output_shape"))
        info = node.get("weightInfo")
        param = traced.get("weight_name")
        if info and param and info.get("name") == param:
            info["name"] = f"{traced['name']}.{param}"


def _fallback_node_mappings(model_view: dict, nntrainer_view: dict) -> list:
    """Click-to-highlight for the .ini/.cpp graphs: a .cpp layer and the
    .ini section generated from the same traced node. Unsupported
    (TODO) nodes have no .ini section, so they have no counterpart."""
    ini_ids = {n["id"] for n in model_view.get("nodes", [])}
    mappings = []
    for node in nntrainer_view.get("nodes", []):
        section = _section_name(node["id"])
        if section in ini_ids:
            mappings.append({
                "sourceIds": [section], "targetIds": [node["id"]],
                "mappingType": "one_to_one", "description": node["id"],
            })
    return mappings
