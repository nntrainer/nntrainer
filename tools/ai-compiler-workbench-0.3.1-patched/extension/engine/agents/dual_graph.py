"""
Dual Graph Agent (no LLM).

Publishes exactly two graphs to the webview -- "Model Graph" (target
"model") and "nntrainer Graph" (target "nntrainer") -- plus the
node-mapping table that drives click-to-highlight between them. No
third "C++ Audit" or "Mapping" tab: two tabs cover everything per the
simplified design (see agents/nntrainer_lowering.py and
agents/graph_views.py).

For architectures with a matched semantic adapter, both graphs come
straight from the IR built earlier in the pipeline (semantic_ir /
nntrainer_graph_ir) -- never by re-parsing generated text. For
architectures with no adapter (state["semantic_ir"] is None), the two
tabs fall back to the previous behavior: "Model Graph" parsed from
model.ini, "nntrainer Graph" parsed from generated_model.cpp. Either
way the webview only ever sees two tabs.
"""
from .events import bus
from .graph_views import build_ini_graph, build_cpp_graph


def run(state: dict) -> dict:
    bus.agent_status("dual_graph", "running")

    if state.get("nntrainer_graph_ir") is not None:
        _publish_semantic_graphs(state)
    else:
        _publish_fallback_graphs(state)

    mappings = state.get("node_mappings") or []
    bus.node_mappings(mappings)
    if mappings:
        bus.log(f"Node mapping table: {len(mappings)} source<->target group(s) for click-to-highlight")

    bus.agent_status("dual_graph", "done")
    return state


def _publish_semantic_graphs(state: dict) -> None:
    model_view = state.get("model_graph_view") or {"nodes": [], "edges": []}
    nntrainer_view = state.get("nntrainer_graph_view") or {"nodes": [], "edges": []}

    bus.graph(model_view, target="model")
    bus.log(f"Model Graph (semantic): {len(model_view['nodes'])} nodes, {len(model_view['edges'])} edges")

    bus.graph(nntrainer_view, target="nntrainer")
    bus.log(f"nntrainer Graph (target): {len(nntrainer_view['nodes'])} nodes, {len(nntrainer_view['edges'])} edges")

    if state.get("ini_content"):
        bus.file_content("nntrainer", "model.ini", state["ini_content"])
    if state.get("cpp_code"):
        bus.file_content("nntrainer", "generated_model.cpp", state["cpp_code"])


def _publish_fallback_graphs(state: dict) -> None:
    ini_content = state.get("ini_content")
    cpp_code = state.get("cpp_code")

    if ini_content:
        model_view = build_ini_graph(ini_content)
        state["model_graph_view"] = model_view
        bus.graph(model_view, target="model")
        bus.file_content("model", "model.ini", ini_content)
        bus.log(f"Model Graph (.ini): {len(model_view['nodes'])} nodes, {len(model_view['edges'])} edges")
    else:
        bus.log("No model.ini content available -- skipping Model Graph", "warn")

    if cpp_code:
        nntrainer_view = build_cpp_graph(cpp_code)
        state["nntrainer_graph_view"] = nntrainer_view
        bus.graph(nntrainer_view, target="nntrainer")
        bus.file_content("nntrainer", "generated_model.cpp", cpp_code)
        bus.log(f"nntrainer Graph (.cpp): {len(nntrainer_view['nodes'])} nodes, {len(nntrainer_view['edges'])} edges")
    else:
        bus.log("No generated_model.cpp content available -- skipping nntrainer Graph", "warn")

    # Kept for backward compatibility with anything still reading these keys.
    state["ini_graph_view"] = state.get("model_graph_view")
    state["cpp_graph_view"] = state.get("nntrainer_graph_view")
