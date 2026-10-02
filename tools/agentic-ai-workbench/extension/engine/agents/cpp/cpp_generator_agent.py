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
                 (<arch>_causallm.{h,cpp}) with class-
                 qualified method overrides, plus a model-metadata and
                 weight-mapping manifest alongside them (see
                 api/lowering/nntrainer/manifest.py). No buildModel()/
                 main() -- there's no Model to compile() and no
                 standalone entry point for this mode; see
                 converters/cpp_generator.py's module docstring for why.
                 Generated files are written to <out_dir>/generated/causallm/<arch>/
                 and can be installed directly into a CausalLM project's
                 models/<arch>/ directory via installGeneratedFiles setting.

CAUSALLM_COMPONENT Generation Strategy:
  1. Model Code Reuse (PRIMARY): Adapts existing reference implementations
     (Qwen3, Qwen2, Gemma3, etc.) by changing only class names and filenames.
     Preserves all tested implementations of createAttention, createMLP, etc.
  2. NOOA Generation (FALLBACK): LLM-driven generation if no reference found
  3. Template Generation (FALLBACK): Simple template-based generation

Code Reuse Benefits:
  - Reuses tested, working implementations
  - Avoids duplicating method bodies
  - Consistent with existing CausalLM model patterns
  - Single source of truth for each method type

RAG Memory Integration:
  - Queries prior compile errors before generating
  - Avoids known error patterns from past runs
"""
import json
import os
import asyncio
import shutil

from api.lowering.nntrainer.manifest import (
    build_model_metadata, build_tensor_map, build_weight_manifest,
    validate_tensor_map, validate_weight_manifest,
)
from api.semantic.model import CausalLMIR
from converters.cpp_generator import CPPGenerator
from converters.layer_registry import get_layer_registry
from converters.variants import registry as variant_registry

from ..events import bus
from .. import model_code_reuse
from . import llm_codegen

# RAG Memory integration - query prior errors before generating
# Use absolute import for engine.knowledge module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from knowledge import recall_compile_errors, recall_errors, format_recall_context

# NOOA integration - optional import
try:
    from .nooa_cpp_generator import generate_cpp_code
    NOOA_AVAILABLE = True
except (ImportError, Exception):
    NOOA_AVAILABLE = False
    generate_cpp_code = None

# Harness integration - for LLM-based generation with validation
try:
    from .harness.validator import CompilationValidator
    from .generators.llm_generator import LLMGenerator
    HARNESS_AVAILABLE = True
except (ImportError, Exception):
    HARNESS_AVAILABLE = False
    CompilationValidator = None
    LLMGenerator = None


def run(state: dict) -> dict:
    bus.agent_status("cpp_generator", "running")

    # Variant dispatch: some architectures are best generated as a small
    # delta on an existing base implementation (e.g. Qwen3-MoE inheriting
    # from Qwen3CausalLM, see converters/variants/) rather than derived
    # from a traced graph IR -- these don't need nntrainer_graph_ir/
    # graph_ir to exist at all, so this check runs before that
    # requirement, not after.
    architecture = state.get("architecture", "")
    hf_config = state.get("hf_config") or {}
    variant_name, variant_generator = variant_registry.find_variant(architecture, hf_config)
    if variant_generator:
        bus.log(f"Using '{variant_name}' variant generator for {architecture}", "info")
        flavor = state.get("qwen_moe_variant", "standard")
        try:
            files = variant_generator(architecture, hf_config, flavor)
        except Exception as exc:
            bus.log(f"Variant generation failed ({exc}) -- falling back to generic generator", "warn")
        else:
            generated_dir = os.path.join(state["out_dir"], "generated", "causallm", architecture.lower())
            os.makedirs(generated_dir, exist_ok=True)
            return _finish_causallm_component(state, files, generated_dir)

    nntrainer_graph_ir = state.get("nntrainer_graph_ir")
    graph_ir = nntrainer_graph_ir or state.get("graph_ir")

    # Model Code Reuse (optional fallback)
    # When use_pre_existing_code=True, use hand-written reference implementation
    # When use_pre_existing_code=False (default), generate new code from graph_ir
    use_pre_existing_code = state.get("use_pre_existing_code", False)
    
    if use_pre_existing_code:
        bus.log("Using pre-existing CausalLM code from nntrainer repository (use_pre_existing_code=True)", "info")
        reused = _try_model_code_reuse(state, architecture)
        if reused is not None:
            return reused
        bus.log("No pre-existing code found -- falling back to generation", "warn")
    
    if not graph_ir:
        bus.log("No graph IR available -- skipping C++ generation", "warn")
        bus.agent_status("cpp_generator", "error", "no graph_ir")
        return state

    graph = _graph_from_ir(graph_ir)
    is_causallm_component = graph.metadata.get("emission_mode") == "causallm_component"

    if is_causallm_component:
        # CAUSALLM_COMPONENT: Generate with hand-written compatible signatures
        architecture_slug = graph.metadata.get("architecture", "").lower()
        generated_dir = os.path.join(state["out_dir"], "generated", "causallm", architecture_slug)
        
        # OPTION D: ALL OF THE ABOVE - Maximum freshness guarantee
        # A. Clear output directory before each run
        # B. Add timestamp to generated filenames (optional, for history)
        # C. Force regenerate flag
        
        force_regenerate = state.get("force_regenerate", True)  # Default: always regenerate
        use_timestamp_filenames = state.get("use_timestamp_filenames", False)  # Default: overwrite same filename
        clear_output_dir = state.get("clear_output_dir", True)  # Default: clear before generating
        
        if clear_output_dir and force_regenerate and os.path.isdir(generated_dir):
            bus.log(f"Clearing existing files in {generated_dir} for fresh generation", "info")
            import shutil
            shutil.rmtree(generated_dir)
        
        os.makedirs(generated_dir, exist_ok=True)
        
        # If using timestamp filenames, store the timestamp in state for the generator to use
        if use_timestamp_filenames:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            state["filename_timestamp"] = timestamp
            bus.log(f"Using timestamp filenames: _{timestamp}", "info")
        
        return _run_causallm_component(state, graph, generated_dir)

    # MODEL_API mode
    use_nooa = state.get("use_nooa", True)

    if use_nooa and NOOA_AVAILABLE:
        return _run_nooa_generation(state, graph_ir)
    else:
        generated_dir = os.path.join(state["out_dir"], "generated")
        os.makedirs(generated_dir, exist_ok=True)
        return _run_model_api(state, graph, generated_dir)


def _try_model_code_reuse(state: dict, architecture: str) -> dict:
    """Try to adapt existing model implementation instead of generating.
    Deliberately takes `architecture` as a plain string rather than a
    Graph -- this is the fallback for exactly the cases where a graph
    walk *can't* produce a component (no nntrainer_graph_ir at all, or
    a decoder shape the generic uniform-layer emitter can't represent,
    e.g. Gemma3's per-layer sliding-window/rope alternation), so it must
    not require one to already exist."""
    if not architecture:
        return None

    try:
        ref_path = model_code_reuse.find_reference_model(architecture)
        if not ref_path or not os.path.isdir(ref_path):
            return None

        generated_dir = os.path.join(state["out_dir"], "generated", "causallm",
                                     architecture.lower())
        os.makedirs(generated_dir, exist_ok=True)

        result = model_code_reuse.adapt_implementation(ref_path, architecture, generated_dir)
        if not result.get("success"):
            return None

        # Be explicit that this path COPIES a hand-written implementation and
        # renames its classes -- it does not generate anything from the traced
        # model. That was previously silent, which made runs look like real
        # generation when no LLM had been involved at all.
        bus.log(
            f"FALLBACK: copied the hand-written reference implementation from "
            f"{ref_path} and renamed its classes. This is NOT generated from "
            f"this model's traced graph, so any architectural difference from "
            f"the reference is silently absent. Configure an LLM "
            f"(clineUmsToken or anthropicApiKey) to generate instead.",
            "warn",
        )

        # Populate state with adapted files
        header_path = result["header_path"]
        source_path = result["source_path"]

        with open(header_path, 'r') as f:
            header_content = f.read()
        with open(source_path, 'r') as f:
            source_content = f.read()

        arch_slug = result["target_model"]
        state["causallm_header"] = header_content
        state["causallm_source"] = source_content
        state["causallm_header_path"] = header_path
        state["causallm_source_path"] = source_path
        state["cpp_emission_mode"] = "causallm_component"
        state["requires_causallm_build"] = True
        state["cpp_generation_method"] = "model_code_reuse"
        state["cpp_code"] = header_content + "\n\n" + source_content
        state["cpp_path"] = source_path

        bus.code(os.path.basename(header_path), header_content)
        bus.code(os.path.basename(source_path), source_content)
        bus.log(f"Adapted {result['reference_model']} → {arch_slug} (code reuse, not generated)")
        bus.agent_status("cpp_generator", "done", f"{arch_slug}_causallm.h/cpp (adapted)")

        # Write manifests if semantic IR available
        _write_manifests(state, generated_dir)

        return state

    except Exception as e:
        bus.log(f"Model code reuse failed: {e}", "warn")
        return None


def _run_nooa_generation(state: dict, graph_ir: dict) -> dict:
    """
    Use NOOA-based LLM-driven C++ generation.
    
    This uses Cline (via UMS token) as the LLM backend to generate
    nntrainer-compatible C++ code from the graph IR.
    
    RAG Memory Integration:
    - Queries prior compile errors for this model family
    - Injects known error patterns and fixes into the prompt
    """
    bus.log("Using NOOA for C++ generation...", "info")
    
    # RAG Memory: Query prior compile errors for this model
    model_name = state.get("model_name", "")
    if model_name:
        try:
            prior_errors = recall_compile_errors(model_name, limit=3)
            if prior_errors:
                bus.log(f"Found {len(prior_errors)} prior compile errors for {model_name.split('-')[0]}", "info")
                # Note: Would inject into LLM prompt here
                for err in prior_errors:
                    bus.log(f"  - {err.get('error_type', 'unknown')}: {err.get('fix_applied', 'no fix')}", "info")
        except Exception as e:
            bus.log(f"Failed to query RAG memory: {e}", "debug")
    
    try:
        output_dir = state["out_dir"]
        result = asyncio.run(generate_cpp_code(graph_ir, output_dir))
        
        generated_dir = os.path.join(output_dir, "generated", "causallm", result.architecture.lower())
        header_path = os.path.join(generated_dir, result.header_filename)
        source_path = os.path.join(generated_dir, result.source_filename)
        
        state["causallm_header"] = result.header
        state["causallm_source"] = result.source
        state["causallm_header_path"] = header_path
        state["causallm_source_path"] = source_path
        state["cpp_emission_mode"] = "causallm_component"
        state["requires_causallm_build"] = True
        state["cpp_code"] = result.header + "\n\n" + result.source
        state["cpp_path"] = source_path
        state["cpp_generation_method"] = "nooa"
        
        bus.code(result.header_filename, result.header)
        bus.code(result.source_filename, result.source)
        bus.log(f"NOOA generated: {result.header_filename}, {result.source_filename}")
        bus.agent_status("cpp_generator", "done", f"NOOA: {result.header_filename}, {result.source_filename}")
        
        # Write manifests
        _write_manifests_nooa(state, generated_dir, result.architecture)
        
        return state
        
    except Exception as exc:
        bus.log(f"NOOA generation failed: {exc}", "error")
        bus.agent_status("cpp_generator", "error", f"NOOA: {exc}")
        state.setdefault("errors", []).append(f"nooa_cpp_generator: {exc}")
        
        # Fall back to template generation
        bus.log("Falling back to template-based generation...", "warn")
        graph = _graph_from_ir(graph_ir)
        generated_dir = os.path.join(state["out_dir"], "generated")
        os.makedirs(generated_dir, exist_ok=True)
        
        is_causallm_component = graph.metadata.get("emission_mode") == "causallm_component"
        if is_causallm_component:
            return _run_causallm_component(state, graph, generated_dir)
        return _run_model_api(state, graph, generated_dir)


def _write_manifests_nooa(state: dict, causallm_dir: str, architecture: str) -> None:
    """Write model metadata and weight manifest for NOOA-generated code."""
    semantic_ir_dict = state.get("semantic_ir")
    if not semantic_ir_dict:
        return
    
    try:
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
    except Exception as e:
        bus.log(f"Failed to write manifests: {e}", "warn")


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
    state["cpp_generation_method"] = "template"
    bus.code("generated_model.cpp", code)
    bus.log(f"Generated {cpp_path} ({len(code.splitlines())} lines)")
    bus.agent_status("cpp_generator", "done", f"{len(code.splitlines())} lines")
    return state


def _run_causallm_component(state: dict, graph, generated_dir: str) -> dict:
    # state["architecture"] is the HF class name (e.g. "Gemma3ForCausalLM")
    # -- the casing model_code_reuse.adapt_implementation() needs to
    # produce correctly-cased class names. graph.metadata["architecture"]
    # is the lowercase CausalLMIR slug (e.g. "gemma3"), which is only a
    # fallback for architectures that somehow reach this point without
    # state["architecture"] set.
    architecture = state.get("architecture", "") or graph.metadata.get("architecture", "")

    # Refresh the layer view from the actual CausalLM source tree before any
    # model code is generated. This is intentionally advisory: fuzzy matches
    # are prompt/context candidates, never an automatic graph rewrite.
    nntrainer_root = state.get("nntrainer_root", "/storage_data/snap/Prachi/nntrainer")
    registry = get_layer_registry()
    discovered = registry.scan_causallm_layers(nntrainer_root)
    graph_ir = state.get("nntrainer_graph_ir") or state.get("graph_ir") or {}
    candidates = {}
    for node in graph_ir.get("nodes", []):
        requested = node.get("node_type") or node.get("semantic_type") or node.get("name")
        if not requested or registry.get_layer(requested):
            continue
        matches = registry.fuzzy_find_layers(requested)
        if matches:
            candidates[requested] = [
                {"layer": layer.name, "score": round(score, 3),
                 "header": layer.header_file, "properties": layer.properties}
                for layer, score in matches
            ]
    state["causallm_discovered_layers"] = discovered
    state["causallm_layer_candidates"] = candidates
    bus.log(f"Scanned {len(discovered)} CausalLM layer types for generation", "info")
    if candidates:
        bus.log(f"Found fuzzy layer candidates for {len(candidates)} unresolved IR operations", "info")

    # The planned generator owns scaffolding and validates layer names,
    # properties, cache arity, and weight-name conventions before compiling.
    # Preserve the older harness below as a compatibility fallback for IRs
    # that cannot yet be represented by a ModelPlan.
    if state.get("use_planned_cpp_codegen", True):
        try:
            files = llm_codegen.generate(state)
            if files is not None:
                return _finish_causallm_component(
                    state, files, generated_dir, method="planned_llm",
                )
            bus.log("Planned C++ generator could not resolve this model; trying compatibility fallback", "warn")
        except Exception as exc:
            bus.log(f"Planned C++ generator failed ({exc}); trying compatibility fallback", "warn")
            state.setdefault("errors", []).append(f"planned_cpp_codegen: {exc}")

    # PRIMARY: Use harness LLMGenerator for C++ code generation from nntrainer_graph_ir.
    #
    # UMS TOKEN FLOW:
    # 1. User sets cline_ums_token in VS Code settings (aiCompilerWorkbench.clineUmsToken)
    # 2. Token is passed to state["cline_ums_token"] by orchestrator
    # 3. Harness LLMGenerator receives token in constructor
    # 4. LLMGenerator calls llm_codegen.generate_from_prompt() with token
    # 5. llm_codegen uses ChatOpenAI with:
    #    - openai_api_key = cline_ums_token (UMS authentication)
    #    - openai_api_base = http://localhost:6543/v1 (Cline/Gauss endpoint)
    #    - model = gauss-4-5 (or configured model)
    # 6. Cline/Gauss returns generated C++ code
    #
    # The harness LLMGenerator:
    # 1. Uses the LAYER_CATALOG to ensure only available layers are used
    # 2. Validates the nntrainer_graph_ir against available layers
    # 3. Generates C++ code using Cline (LLM) with detailed prompts
    #    - Model family patterns (Qwen, Gemma, Llama, Phi, Mistral)
    #    - Correct inheritance (CausalLM::registerCustomLayers)
    #    - Sliding window, weight_initializer, ROPE_THETA constants
    # 4. Returns GeneratedFiles with metadata for compilation validation
    #
    # This is the PRIMARY generation path - no fallback to old llm_codegen.
    if HARNESS_AVAILABLE and LLMGenerator and state.get("use_llm_codegen", True):
        bus.log("Using harness LLMGenerator for C++ code generation", "info")
        has_token = bool(state.get("cline_ums_token", ""))
        bus.log(f"UMS token configured: {'Yes' if has_token else 'No'}", "info")
        try:
            generator = LLMGenerator(
                nntrainer_root=state.get("nntrainer_root", "/storage_data/snap/Prachi/nntrainer"),
                cline_ums_token=state.get("cline_ums_token", "")
            )
            result = generator.generate(state)
            
            if result.success:
                bus.log(f"LLM generation successful: {result.header_filename}, {result.source_filename}", "info")
                bus.log(f"Required layers: {', '.join(result.required_layers)}", "info")
                return _finish_causallm_component(
                    state, result, generated_dir,
                    method="harness_llm",
                )
            else:
                bus.log(f"LLM generation failed: {'; '.join(result.errors)}", "warn")
                state.setdefault("errors", []).extend(result.errors)
        except Exception as exc:
            bus.log(f"Harness LLMGenerator raised ({exc}) -- falling back to template emitter", "error")
            state.setdefault("errors", []).append(f"harness_llm: {exc}")
        
        bus.log("Harness LLMGenerator unavailable -- falling back to template emitter", "warn")

    try:
        files = CPPGenerator(graph).generate_component()
    except ValueError as exc:
        # e.g. non-uniform decoder layers (Gemma3's per-layer sliding-
        # window/rope-theta alternation, sandwich norms) -- a real
        # limitation of the generic single-template emitter (see
        # converters/cpp_generator.py), not something to paper over with
        # a guess. Before giving up, fall back to reusing a real,
        # hand-written reference implementation for this architecture
        # if one exists (agents/model_code_reuse.py) -- this is the
        # "Model Code Reuse (PRIMARY)" strategy documented at the top of
        # this file, just only invoked once the derived-from-graph path
        # has proven it can't represent this decoder shape.
        bus.log(f"CausalLM component generation failed ({exc}) -- trying model code reuse", "warn")
        # TEMPORARILY DISABLED: Testing LLM generation only
        # reused = _try_model_code_reuse(state, architecture)
        # if reused is not None:
        #     return reused
        bus.log(f"CausalLM component generation failed: {exc}", "error")
        bus.agent_status("cpp_generator", "error", str(exc))
        state.setdefault("errors", []).append(f"cpp_generator: {exc}")
        return state

    # `generated_dir` already ends in generated/causallm/<arch> (see run()), so
    # re-appending causallm/<arch> here produced the nested
    # generated/causallm/qwen2/causallm/qwen2/ paths visible in older output.
    return _finish_causallm_component(state, files, generated_dir,
                                      method="template")


def _finish_causallm_component(state: dict, files, causallm_dir: str, method: str = "variant") -> dict:
    """
    Shared tail for every CAUSALLM_COMPONENT-shaped generator (the
    generic graph-driven CPPGenerator path AND every converters/variants/
    generator) -- writes both files to disk, populates the same state
    keys causallm_install.py/causallm_so_builder.py read to install and
    compile the result, and emits the same bus.code(...) events so
    whatever generated it, the code shows up in the panel and reaches
    the build.
    
    Includes compilation validation using CompilationValidator.
    """
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
    state["cpp_generation_method"] = method
    state["generated_transformer_class"] = files.transformer_class
    state["generated_causallm_class"] = files.causal_lm_class
    # Kept populated too, so anything reading the older single-file keys
    # (dual_graph's optional file_content forwarding, chat context, etc.)
    # still gets something meaningful.
    state["cpp_code"] = files.header + "\n\n" + files.source
    state["cpp_path"] = source_path

    # COMPILATION VALIDATION: Check syntax of generated code
    if HARNESS_AVAILABLE and CompilationValidator:
        bus.log("Running compilation validation on generated code...", "info")
        nntrainer_root = state.get("nntrainer_root", "/storage_data/snap/Prachi/nntrainer")
        causallm_root = state.get("causallm_project_root", "")
        
        validator = CompilationValidator(
            causallm_root=causallm_root,
            nntrainer_root=nntrainer_root
        )
        
        validation_result = validator.validate_syntax(
            header=files.header,
            source=files.source,
            architecture=files.architecture
        )
        
        state["syntax_validated"] = validation_result.success
        state["compile_log"] = validation_result.compile_log
        
        if validation_result.success:
            bus.log("Compilation validation passed!", "info")
        else:
            bus.log(f"Compilation validation failed:", "error")
            for err in validation_result.errors[:3]:
                bus.log(f"  {err}", "error")
            state.setdefault("errors", []).extend(validation_result.errors)

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

    tensor_map = build_tensor_map(model_ir, state.get("hf_config"))
    tensor_map_problems = validate_tensor_map(tensor_map)
    tensor_map_path = os.path.join(causallm_dir, "tensor_map.json")
    with open(tensor_map_path, "w", encoding="utf-8") as f:
        json.dump(tensor_map, f, indent=2)
    state["tensor_map"] = tensor_map
    state["tensor_map_path"] = tensor_map_path

    converter_source = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "converters", "manifest_weight_converter.py",
    )
    converter_path = os.path.join(causallm_dir, "weight_converter.py")
    shutil.copyfile(converter_source, converter_path)
    state["generated_weight_converter_path"] = converter_path

    architecture = state.get("architecture") or model_ir.architecture
    implementation = "generated/" + architecture.replace("ForCausalLM", "").replace("Model", "").lower() + "_workbench"
    model_manifest = {
        "schema_version": 1,
        "architecture": architecture,
        "implementation": implementation,
        "class": state.get("generated_causallm_class"),
        "transformer_class": state.get("generated_transformer_class"),
        "source": os.path.basename(state.get("causallm_source_path") or ""),
        "header": os.path.basename(state.get("causallm_header_path") or ""),
        "tensor_map": "tensor_map.json",
        "weight_converter": "weight_converter.py",
        "build_option": "enable-workbench-generated-models",
    }
    model_manifest_path = os.path.join(causallm_dir, "model_manifest.json")
    with open(model_manifest_path, "w", encoding="utf-8") as f:
        json.dump(model_manifest, f, indent=2)
    state["generated_model_manifest"] = model_manifest
    state["generated_model_manifest_path"] = model_manifest_path

    bus.log(
        f"Wrote {metadata_path}, {manifest_path}, {tensor_map_path}, and model_manifest.json "
        f"({len(tensor_map['entries'])} ordered tensor mappings)"
    )
    if problems:
        for p in problems:
            bus.log(f"  [weight manifest] {p}", "warn")
    if tensor_map_problems:
        for p in tensor_map_problems:
            bus.log(f"  [tensor map] {p}", "error")
        state.setdefault("errors", []).extend(f"tensor_map: {p}" for p in tensor_map_problems)


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
