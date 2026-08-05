"""
NNTrainer Lowering Agent (no LLM).

Runs right after Compatibility. Takes `state["semantic_ir"]` (built by
an ArchitectureAdapter -- see api/adapters) and lowers it into
`state["nntrainer_graph_ir"]`: the actual target graph, with real
nntrainer layer types and genuine tensor dataflow. This is the ONLY
graph the C++ generator and the "nntrainer Graph" webview tab consume
from here on -- see converters/cpp_generator.py and
agents/graph_views.py.

Two independent pieces of work only depend on `semantic_ir` and not on
each other -- the source-side "Model Graph" view, and the target-side
lowering + validation -- so they run concurrently on a small thread
pool rather than sequentially.

If no adapter matched (state["semantic_ir"] is None), this agent is a
no-op: the pipeline falls back to the pre-existing graph_ir/.ini/.cpp
path for that architecture, unchanged.
"""
from concurrent.futures import ThreadPoolExecutor

from api.lowering.nntrainer.lowerer import NNTrainerLowerer
from api.lowering.nntrainer.validation import validate
from api.semantic.model import CausalLMIR

from . import graph_views
from .events import bus


def run(state: dict) -> dict:
    bus.agent_status("nntrainer_lowering", "running")

    semantic_ir_dict = state.get("semantic_ir")
    if not semantic_ir_dict:
        bus.log("No semantic IR available -- skipping nntrainer lowering (using module-tree path)", "info")
        bus.agent_status("nntrainer_lowering", "done", "skipped")
        return state

    model_ir = CausalLMIR.from_dict(semantic_ir_dict)

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="lowering-stage") as pool:
        model_view_future = pool.submit(graph_views.build_model_graph_view, model_ir)
        target_future = pool.submit(_lower_and_validate, model_ir)

        state["model_graph_view"] = model_view_future.result()
        nntrainer_graph, diagnostics = target_future.result()

    nntrainer_graph_ir = nntrainer_graph.export()
    state["nntrainer_graph_ir"] = nntrainer_graph_ir
    state["lowering_diagnostics"] = diagnostics.to_dict()
    state["node_mappings"] = graph_views.build_node_mappings(model_ir, nntrainer_graph)
    state["nntrainer_graph_view"] = graph_views.build_nntrainer_graph_view(nntrainer_graph_ir)

    errors = [d for d in diagnostics.diagnostics if d.severity == "error"]
    if errors:
        for d in errors:
            bus.log(f"  [lowering] {d.message}", "error")
        bus.agent_status("nntrainer_lowering", "error", f"{len(errors)} validation error(s) -- see log")
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
