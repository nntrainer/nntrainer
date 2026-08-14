"""
INI Generator Agent (no LLM).

Generates nntrainer's model.ini from the traced IR graph: one
[layer_name] section per supported node, in graph order, with its
type, properties, and -- importantly -- an explicit `input_layers`
property reflecting the *real* predecessor edges from the traced
graph (not just declaration order). This matters for any model with
branching (residual adds, etc.): without it, the .ini would silently
describe a straight chain even when the traced graph isn't one.
"""
import os

from .events import bus


def _section_name(name: str) -> str:
    return name.replace(".", "_").replace("/", "_").replace("-", "_")


def run(state: dict) -> dict:
    bus.agent_status("ini_generator", "running")
    graph_ir = state.get("nntrainer_graph_ir") or state.get("graph_ir")

    if not graph_ir:
        bus.log("No graph IR available -- skipping .ini generation", "warn")
        bus.agent_status("ini_generator", "error", "no graph_ir")
        return state

    if (graph_ir.get("metadata") or {}).get("emission_mode") == "causallm_component":
        # .ini's declarative "one section, one input_layers string" format
        # has no representation for multi-input tensor calls (mha_core's
        # 4 inputs, kv_cache_placeholders' 2-output destructure) -- and a
        # real CausalLM integration builds the model programmatically in
        # C++ anyway, never from an .ini. Skip rather than emit a
        # section-per-node file that looks plausible but can't actually
        # describe this graph.
        bus.log(
            "Skipping .ini generation for CausalLM-component graph -- "
            "this architecture is built programmatically via the generated "
            "C++ class, not from model.ini (see agents/cpp_generator_agent.py)",
            "info",
        )
        bus.agent_status("ini_generator", "done", "skipped (causallm_component)")
        return state

    nodes = graph_ir["nodes"]
    # id -> section name, but only for nodes that actually emit a section
    # (supported + has a node_type); unsupported/passthrough nodes have no
    # section of their own, so edges through them resolve to their nearest
    # supported ancestor instead.
    id_to_node = {n["id"]: n for n in nodes}
    id_to_section = {
        n["id"]: _section_name(n["name"])
        for n in nodes
        if n["supported"] and n.get("node_type")
    }

    def resolve_inputs(node, seen=None):
        """Walk back through unsupported/passthrough predecessors to find
        the nearest ancestor(s) that actually have a section."""
        seen = seen or set()
        resolved = []
        for pred_id in node.get("inputs", []):
            if pred_id in seen:
                continue
            seen.add(pred_id)
            if pred_id in id_to_section:
                resolved.append(id_to_section[pred_id])
            elif pred_id in id_to_node:
                resolved.extend(resolve_inputs(id_to_node[pred_id], seen))
        return resolved

    lines = [
        "[Model]",
        f"Model = {state.get('architecture', 'model')}",
        "Type = NeuralNetwork",
        "Epochs = 1",
        "Loss = cross",
        "Save_Path = model.bin",
        "",
    ]

    for node in nodes:
        if not node["supported"] or not node.get("node_type"):
            continue
        section = _section_name(node["name"])
        lines.append(f"[{section}]")
        lines.append(f"Type = {node['node_type']}")
        for key, value in (node.get("attributes") or {}).items():
            lines.append(f"{key} = {value}")

        if node.get("weight_shape"):
            lines.append(
                f"; weight: name={node.get('weight_name')} shape={list(node['weight_shape'])} "
                f"dtype={node.get('weight_dtype')} params={node.get('parameter_count', 0)}"
            )

        input_layers = resolve_inputs(node)
        if input_layers:
            # de-dupe while preserving order
            seen_names = []
            for n in input_layers:
                if n not in seen_names:
                    seen_names.append(n)
            lines.append(f"input_layers = {','.join(seen_names)}")
        lines.append("")

    content = "\n".join(lines)
    generated_dir = os.path.join(state["out_dir"], "generated")
    os.makedirs(generated_dir, exist_ok=True)
    ini_path = os.path.join(generated_dir, "model.ini")
    with open(ini_path, "w", encoding="utf-8") as f:
        f.write(content)

    state["ini_content"] = content
    state["ini_path"] = ini_path
    bus.log(f"Generated model.ini ({len(graph_ir['nodes'])} nodes considered) -> {ini_path}")
    bus.agent_status("ini_generator", "done")
    return state
