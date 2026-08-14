"""
Visualization / Graph Builder Agent (no LLM).

Takes the traced Compiler IR (a dict of nodes/edges with no notion of
"where should this sit on screen") and produces the interactive
knowledge-graph payload used internally by the Profiler Agent's
op-weight estimate: per-node color class (supported / unsupported /
skipped) plus x/y coordinates.

Layout is delegated to the shared `graph_views._layout_vertical`
(layered + barycenter crossing reduction + row centering) so this view
and the two user-facing graphs share one implementation and can't drift.
This IR-level view is not sent to the webview as its own tab -- the two
user-facing graphs (Model Graph from .ini, C++ Graph from
generated_model.cpp) are built from those generated files directly by
agents/graph_views.py, later in the pipeline.
"""
import os

from .events import bus
from .graph_views import _layout_vertical


def run(state: dict) -> dict:
    bus.agent_status("graph_builder", "running")
    graph_ir = state.get("graph_ir") or state.get("nntrainer_graph_ir")

    if not graph_ir:
        bus.log("No graph IR available -- skipping graph build", "warn")
        bus.agent_status("graph_builder", "error", "no graph_ir")
        return state

    view_nodes = []
    for node in graph_ir["nodes"]:
        status = "unmapped" if not node["supported"] else "skipped" if node.get("dead") else "mapped"
        view_nodes.append({
            "id": node["id"],
            "label": node["name"],
            "type": node["node_type"] or "passthrough",
            "status": status,
            "attributes": node.get("attributes", {}),
            "inputShape": node.get("input_shape"),
            "outputShape": node.get("output_shape"),
            "reason": node.get("compatibility_reason", ""),
        })

    view_edges = [
        {"id": f"{e['source']}-{e['target']}", "source": e["source"], "target": e["target"]}
        for e in graph_ir["edges"]
    ]

    # Shared layout assigns x/y in place (layered + crossing reduction).
    _layout_vertical(view_nodes, view_edges, [n["id"] for n in view_nodes])

    graph_view = {"nodes": view_nodes, "edges": view_edges}
    state["graph_view"] = graph_view

    generated_dir = os.path.join(state["out_dir"], "generated")
    os.makedirs(generated_dir, exist_ok=True)
    graph_json_path = os.path.join(generated_dir, "graph.json")
    import json
    with open(graph_json_path, "w", encoding="utf-8") as f:
        json.dump(graph_view, f, indent=2)

    bus.log(f"Built internal model graph: {len(view_nodes)} nodes, {len(view_edges)} edges")
    bus.agent_status("graph_builder", "done", f"{len(view_nodes)} nodes")
    return state
