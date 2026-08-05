"""
C++ Generator Agent (LLM optional).

Two independent output shapes, matching converters.cpp_generator's two
CPPEmissionModes:

  MODEL_API           -- unchanged: a single generated_model.cpp with
                 buildModel()/main(), annotated with the Compatibility
                 Agent's suggestions on any TODO(unsupported) block.
                 Used whenever there's no nntrainer_graph_ir (no
                 semantic adapter matched this architecture).

  CAUSALLM_COMPONENT  -- a header + source pair
                 (generated_<arch>_causallm.{h,cpp}) with class-
                 qualified method overrides, plus a model-metadata and
                 weight-mapping manifest alongside them (see
                 api/lowering/nntrainer/manifest.py). No buildModel()/
                 main() -- there's no Model to compile() and no
                 standalone entry point for this mode; see
                 converters/cpp_generator.py's module docstring for why.
"""
import json
import os

from api.lowering.nntrainer.manifest import (
    build_model_metadata, build_weight_manifest, validate_weight_manifest,
)
from api.semantic.model import CausalLMIR
from converters.cpp_generator import CPPGenerator

from .events import bus


def run(state: dict) -> dict:
    bus.agent_status("cpp_generator", "running")

    nntrainer_graph_ir = state.get("nntrainer_graph_ir")
    graph_ir = nntrainer_graph_ir or state.get("graph_ir")

    if not graph_ir:
        bus.log("No graph IR available -- skipping C++ generation", "warn")
        bus.agent_status("cpp_generator", "error", "no graph_ir")
        return state

    graph = _graph_from_ir(graph_ir)
    is_causallm_component = graph.metadata.get("emission_mode") == "causallm_component"

    generated_dir = os.path.join(state["out_dir"], "generated")
    os.makedirs(generated_dir, exist_ok=True)

    if is_causallm_component:
        return _run_causallm_component(state, graph, generated_dir)
    return _run_model_api(state, graph, generated_dir)


def _graph_from_ir(graph_ir: dict):
    from api.graph.graph import Graph
    from api.graph.node import GraphNode

    graph = Graph()
    graph.model_name = graph_ir["summary"]["model_name"]
    graph.architecture = graph_ir["summary"]["architecture"]
    graph.metadata = dict(graph_ir.get("metadata") or {})
    for n in graph_ir["nodes"]:
        node = GraphNode(name=n["name"], node_type=n["node_type"], id=n["id"])
        node.attributes = n.get("attributes", {})
        node.supported = n["supported"]
        node.compatibility_reason = n.get("compatibility_reason", "")
        node.input_shape = n.get("input_shape")
        node.output_shape = n.get("output_shape")
        node.weight_name = n.get("weight_name")
        node.weight_shape = tuple(n["weight_shape"]) if n.get("weight_shape") else None
        node.weight_dtype = n.get("weight_dtype")
        node.parameter_count = n.get("parameter_count", 0)
        node.semantic_type = n.get("semantic_type", "")
        node.group_id = n.get("group_id", "")
        node.template_id = n.get("template_id", "")
        node.source_node_ids = n.get("source_node_ids") or []
        node.status = n.get("status", "supported")
        node.cpp_symbol = n.get("cpp_symbol", "")
        node.repeat_index = n.get("repeat_index")
        graph.add_node(node)
    for e in graph_ir["edges"]:
        src, tgt = e["source"], e["target"]
        if src in graph.nodes and tgt in graph.nodes:
            graph.connect(graph.nodes[src], graph.nodes[tgt])
    return graph


def _run_model_api(state: dict, graph, generated_dir: str) -> dict:
    code = CPPGenerator(graph).generate()

    suggestions = state.get("suggestions") or {}
    if suggestions:
        code = _inject_suggestions(code, suggestions)

    # buildModel() alone isn't a linkable program -- append a thin main()
    # so the Compiler/Profiler agents have something real to smoke-test
    # against a real nntrainer install.
    code += (
        "\n\n"
        "#ifdef NNTRAINER_STANDALONE_SMOKE_TEST\n"
        "int main() {\n"
        "    auto model = buildModel();\n"
        "    model->compile();\n"
        "    model->initialize();\n"
        "    std::cout << \"Model constructed OK\" << std::endl;\n"
        "    return 0;\n"
        "}\n"
        "#endif\n"
    )

    cpp_path = os.path.join(generated_dir, "generated_model.cpp")
    with open(cpp_path, "w", encoding="utf-8") as f:
        f.write(code)

    state["cpp_code"] = code
    state["cpp_path"] = cpp_path
    state["cpp_emission_mode"] = "model_api"
    state["requires_causallm_build"] = False
    bus.code("generated_model.cpp", code)
    bus.log(f"Generated {cpp_path} ({len(code.splitlines())} lines)")
    bus.agent_status("cpp_generator", "done", f"{len(code.splitlines())} lines")
    return state


def _run_causallm_component(state: dict, graph, generated_dir: str) -> dict:
    try:
        files = CPPGenerator(graph).generate_component()
    except ValueError as exc:
        # e.g. non-uniform decoder layers -- a real limitation of this
        # generation mode (see converters/cpp_generator.py), not
        # something to paper over with a guess.
        bus.log(f"CausalLM component generation failed: {exc}", "error")
        bus.agent_status("cpp_generator", "error", str(exc))
        state.setdefault("errors", []).append(f"cpp_generator: {exc}")
        return state

    causallm_dir = os.path.join(generated_dir, "causallm", files.architecture.lower())
    os.makedirs(causallm_dir, exist_ok=True)
    header_path = os.path.join(causallm_dir, files.header_filename)
    source_path = os.path.join(causallm_dir, files.source_filename)
    with open(header_path, "w", encoding="utf-8") as f:
        f.write(files.header)
    with open(source_path, "w", encoding="utf-8") as f:
        f.write(files.source)

    state["causallm_header"] = files.header
    state["causallm_source"] = files.source
    state["causallm_header_path"] = header_path
    state["causallm_source_path"] = source_path
    state["cpp_emission_mode"] = "causallm_component"
    state["requires_causallm_build"] = True
    # Kept populated too, so anything reading the older single-file keys
    # (dual_graph's optional file_content forwarding, chat context, etc.)
    # still gets something meaningful.
    state["cpp_code"] = files.header + "\n\n" + files.source
    state["cpp_path"] = source_path

    bus.code(files.header_filename, files.header)
    bus.code(files.source_filename, files.source)
    bus.log(
        f"Generated CausalLM component for '{files.architecture}': "
        f"{files.transformer_class} / {files.causal_lm_class} "
        f"-> {header_path}, {source_path}"
    )
    bus.log(
        "CAUSALLM_COMPONENT mode emits class-qualified method overrides, not "
        "buildModel()/main() -- there's no standalone entry point to smoke-test "
        "here; build this file inside your CausalLM project instead."
    )

    _write_manifests(state, causallm_dir)

    bus.agent_status(
        "cpp_generator", "done",
        f"{files.header_filename}, {files.source_filename}",
    )
    return state


def _write_manifests(state: dict, causallm_dir: str) -> None:
    semantic_ir_dict = state.get("semantic_ir")
    if not semantic_ir_dict:
        return
    model_ir = CausalLMIR.from_dict(semantic_ir_dict)

    metadata = build_model_metadata(model_ir, source_model=state.get("model_name"))
    metadata_path = os.path.join(causallm_dir, "config.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    state["model_metadata"] = metadata

    manifest = build_weight_manifest(model_ir, None)
    problems = validate_weight_manifest(manifest)
    manifest_path = os.path.join(causallm_dir, "weight_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    state["weight_manifest"] = manifest

    bus.log(f"Wrote {metadata_path} and {manifest_path} ({len(manifest)} weight mapping entries)")
    if problems:
        for p in problems:
            bus.log(f"  [weight manifest] {p}", "warn")


def _inject_suggestions(code: str, suggestions: dict) -> str:
    lines = code.split("\n")
    out = []
    for line in lines:
        out.append(line)
        if line.startswith("    // TODO(unsupported):"):
            for name, suggestion in suggestions.items():
                if name in line:
                    out.append(f"    // Suggested approach: {suggestion}")
                    break
    return "\n".join(out)
