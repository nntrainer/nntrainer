# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file nntrainer_lowering.py
# @brief NNTrainer Lowering Agent - Converts semantic IR to nntrainer graph.
# @author prachi-t17 <prachi.t@samsung.com>

"""
NNTrainer Lowering Agent - Converts semantic IR to nntrainer graph.
"""
from concurrent.futures import ThreadPoolExecutor

from api.lowering.nntrainer.lowerer import NNTrainerLowerer
from api.lowering.nntrainer.validation import validate
from api.semantic.model import CausalLMIR

from . import graph_views
from .events import bus


def run(state: dict) -> dict:
    """Pipeline step: lower the semantic IR to the nntrainer graph and validate it."""
    bus.agent_status("nntrainer_lowering", "running")

    semantic_ir_dict = state.get("semantic_ir")
    if not semantic_ir_dict:
        bus.log(
            "No semantic IR available -- skipping nntrainer lowering (using module-tree path)",
            "info",
        )
        bus.agent_status("nntrainer_lowering", "done", "skipped")
        return state

    model_ir = CausalLMIR.from_dict(semantic_ir_dict)

    # The checkpoint's declared dtype, until agents/weight_preview.py reads
    # the real tensors (the weights are still downloading at this point).
    hf_config = state.get("hf_config") or {}
    declared_dtype = hf_config.get("torch_dtype") or hf_config.get("dtype")
    declared_dtype = str(declared_dtype) if declared_dtype else None

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="lowering-stage") as pool:
        model_view_future = pool.submit(
            graph_views.build_model_graph_view, model_ir, declared_dtype
        )
        target_future = pool.submit(_lower_and_validate, model_ir)

        state["model_graph_view"] = model_view_future.result()
        nntrainer_graph, diagnostics = target_future.result()

    nntrainer_graph_ir = nntrainer_graph.export()
    state["nntrainer_graph_ir"] = nntrainer_graph_ir
    state["lowering_diagnostics"] = diagnostics.to_dict()
    state["node_mappings"] = graph_views.build_node_mappings(
        nntrainer_graph, state["model_graph_view"]
    )
    state["nntrainer_graph_view"] = graph_views.build_nntrainer_graph_view(
        nntrainer_graph_ir, declared_dtype
    )

    errors = [d for d in diagnostics.diagnostics if d.severity == "error"]
    if errors:
        for error in errors:
            bus.log(f"  [lowering] {error.message}", "error")
        bus.agent_status(
            "nntrainer_lowering", "error", f"{len(errors)} validation error(s) -- see log"
        )
        state.setdefault("errors", []).extend(f"nntrainer_lowering: {d.message}" for d in errors)
    else:
        bus.log(
            f"Lowered {model_ir.architecture} to nntrainer graph: "
            f"{len(nntrainer_graph_ir['nodes'])} nodes, {len(nntrainer_graph_ir['edges'])} edges "
            f"({model_ir.num_layers} decoder layers)"
        )
        bus.agent_status("nntrainer_lowering", "done", f"{len(nntrainer_graph_ir['nodes'])} nodes")

    return state


def _lower_and_validate(model_ir: CausalLMIR):
    graph = NNTrainerLowerer(model_ir).lower()
    diagnostics = validate(graph, model_ir)
    return graph, diagnostics
