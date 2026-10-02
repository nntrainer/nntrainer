"""
Shared state object passed between every node in the orchestrator's
LangGraph StateGraph. This is the "Knowledge Graph Agent" / shared
state store from the architecture diagram, minus the standalone
DB process -- for a VS Code extension driving one conversion at a
time, an in-memory dict that gets checkpointed to
<out_dir>/state.json after every step is the right amount of
infrastructure. The checkpoint file is also what lets the chat
panel answer follow-up questions after the pipeline has finished
(a fresh process load state.json instead of re-running everything).
"""
from typing import Any, Optional, TypedDict


class PipelineState(TypedDict, total=False):
    model_name: str
    out_dir: str
    anthropic_api_key: Optional[str]
    custom_weights_path: Optional[str]

    hf_config: dict
    architecture: str

    graph_ir: Optional[dict]  # exported api.graph.graph.Graph.export() -- module-tree inspection view; None for known transformer architectures once semantic lowering takes over
    report: dict             # OpLevelCompatibilityChecker output
    suggestions: dict        # unsupported-op-name -> LLM suggestion

    semantic_ir: Optional[dict]          # api.semantic.model.CausalLMIR.to_dict(), or None if no adapter matched
    semantic_capabilities: Optional[dict]  # api.compatibility.semantic_capabilities.describe() output

    nntrainer_graph_ir: Optional[dict]     # exported target Graph -- the ONLY graph the C++ generator consumes
    lowering_diagnostics: dict              # api.lowering.nntrainer.validation.ValidationResult.to_dict()
    node_mappings: list                     # [{sourceIds: [...], targetIds: [...]}] for click-to-highlight between the two graphs

    model_graph_view: dict       # webview payload, Tab 1 "Model Graph"
    nntrainer_graph_view: dict   # webview payload, Tab 2 "nntrainer Graph"

    weights_path: Optional[str]
    converted_weights_path: Optional[str]

    ini_content: Optional[str]
    ini_path: Optional[str]

    graph_view: dict          # React-Flow-style {nodes, edges} for the webview
    ini_graph_view: dict      # from dual_graph agent — parsed from model.ini
    cpp_graph_view: dict      # from dual_graph agent — parsed from generated_model.cpp

    cpp_code: Optional[str]
    cpp_path: Optional[str]
    cpp_emission_mode: str  # "model_api" or "causallm_component" -- set by cpp_generator_agent, read by compiler_agent
    fix_iterations: int
    requires_causallm_build: bool
    skip_cpp_compilation: bool
    nntrainer_missing: bool
    compile_skipped: bool
    compile_command: str


    causallm_header: Optional[str]        # generated_<arch>_causallm.h contents
    causallm_source: Optional[str]        # generated_<arch>_causallm.cpp contents
    causallm_header_path: Optional[str]
    causallm_source_path: Optional[str]
    model_metadata: Optional[dict]        # api.lowering.nntrainer.manifest.build_model_metadata()
    weight_manifest: Optional[list]       # api.lowering.nntrainer.manifest.build_weight_manifest()

    # aiCompilerWorkbench.installGeneratedFiles / causallmProjectRoot / etc.
    # -- see agents/causallm_install.py. All default to "do nothing".
    install_generated_files: bool
    causallm_project_root: Optional[str]
    generated_header_directory: str
    generated_source_directory: str
    causallm_installed_header_path: Optional[str]
    causallm_installed_source_path: Optional[str]

    # CausalLM Build & Run agent (PC x86 build + inference)
    causallm_build_success: bool
    causallm_build_log: str
    causallm_run_success: bool
    causallm_run_log: str

    compiled: bool

    compile_log: str
    binary_path: Optional[str]

    profile: dict

    artifacts: list

    errors: list


def new_state(
    model_name: str,
    out_dir: str,
    api_key: Optional[str],
    custom_weights_path: Optional[str] = None,
    causallm_project_root: Optional[str] = None,
    install_generated_files: bool = False,
    generated_header_directory: str = "include/generated",
    generated_source_directory: str = "src/generated",
) -> PipelineState:
    return PipelineState(
        model_name=model_name,
        out_dir=out_dir,
        anthropic_api_key=api_key,
        custom_weights_path=custom_weights_path,
        fix_iterations=0,
        compiled=False,
        errors=[],
        cpp_emission_mode="",
        requires_causallm_build=False,
        skip_cpp_compilation=True,
        nntrainer_missing=False,
        compile_skipped=False,
        compile_command="",
        causallm_project_root=causallm_project_root,
        install_generated_files=install_generated_files,
        generated_header_directory=generated_header_directory,
        generated_source_directory=generated_source_directory,
    )

