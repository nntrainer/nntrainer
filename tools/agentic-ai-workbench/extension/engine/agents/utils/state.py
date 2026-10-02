"""
Shared state object passed between every node in the orchestrator's
LangGraph StateGraph. All LLM calls use Cline with UMS token (Gauss models only).
"""
from typing import Any, Optional, TypedDict


class PipelineState(TypedDict, total=False):
    model_name: str
    out_dir: str
    cline_ums_token: Optional[str]  # Single token for all Cline LLM operations
    custom_weights_path: Optional[str]

    hf_config: dict
    architecture: str

    graph_ir: Optional[dict]
    report: dict
    suggestions: dict

    semantic_ir: Optional[dict]
    semantic_capabilities: Optional[dict]

    nntrainer_graph_ir: Optional[dict]
    lowering_diagnostics: dict
    node_mappings: list

    model_graph_view: dict
    nntrainer_graph_view: dict

    weights_path: Optional[str]
    converted_weights_path: Optional[str]

    ini_content: Optional[str]
    ini_path: Optional[str]

    graph_view: dict
    ini_graph_view: dict
    cpp_graph_view: dict

    cpp_code: Optional[str]
    cpp_path: Optional[str]
    cpp_emission_mode: str
    fix_iterations: int
    max_fix_iterations: int
    requires_causallm_build: bool
    skip_cpp_compilation: bool
    nntrainer_missing: bool
    compile_skipped: bool
    compile_command: str

    # Cline configuration (Gauss models only)
    cline_api_base: str  # Cline API base URL (default: http://localhost:6543/v1)
    cline_model: str  # Gauss model name (default: gauss-4-5)
    cline_max_tokens: int  # Max tokens for Cline LLM

    # NOOA configuration
    use_nooa: bool  # Use NOOA-based C++ generation (default: true)

    # Variant-generator configuration (see converters/variants/) -- which
    # pre-built expert-routing layer a MoE architecture should target.
    # "slim"/"cached_slim" are deployment choices, not something detected
    # from the HF config; default "standard" picks the plain MoELayer.
    qwen_moe_variant: str

    # Fix history for learning
    fix_history: list

    # CausalLM .bin generation configuration
    causallm_path: Optional[str]
    enable_quantization: bool
    quantization_preset: str
    target_isa: str
    nntr_quantize_path: Optional[str]

    causallm_header: Optional[str]
    causallm_source: Optional[str]
    causallm_header_path: Optional[str]
    causallm_source_path: Optional[str]
    model_metadata: Optional[dict]
    weight_manifest: Optional[list]
    # These four are set later in the same cpp_generator_agent._write_manifests
    # call as model_metadata/weight_manifest above. LangGraph's StateGraph only
    # tracks fields declared in this TypedDict as channels between nodes --
    # anything a node sets that isn't declared here is silently dropped when
    # the graph hands state to the next node, so causallm_install.py's very
    # next-node check for generated_model_manifest_path always saw it as
    # missing even though it was set moments earlier in the same run.
    tensor_map: Optional[dict]
    tensor_map_path: Optional[str]
    generated_weight_converter_path: Optional[str]
    generated_model_manifest: Optional[dict]
    generated_model_manifest_path: Optional[str]

    install_generated_files: bool
    causallm_project_root: Optional[str]
    generated_header_directory: str
    generated_source_directory: str
    causallm_installed_header_path: Optional[str]
    causallm_installed_source_path: Optional[str]

    causallm_build_success: bool
    causallm_build_log: str
    causallm_run_success: bool
    causallm_run_log: str
    causallm_so_built: bool
    causallm_build_error: str
    causallm_build_fixable: bool
    causallm_so_path: Optional[str]
    causallm_exe_path: Optional[str]
    causallm_api_so_path: Optional[str]

    compiled: bool
    compile_log: str
    binary_path: Optional[str]
    syntax_ok: Optional[bool]

    # Fully-automatic compile/build configuration
    auto_compile: bool  # forces install_generated_files=True for this run only
    use_docker_build: bool  # route meson/ninja/cmake/make through Docker (default: true)
    auto_fix_use_llm: bool  # allow auto_fix to fall back to an LLM rewrite (default: false)
    nntrainer_path: Optional[str]  # user-configured nntrainer install/checkout root

    # Pre-existing code configuration
    use_pre_existing_code: bool  # use pre-existing code from nntrainer repo (default: true)
    nntrainer_root: Optional[str]  # path to nntrainer repository root

    profile: dict
    artifacts: list
    errors: list


def new_state(
    model_name: str,
    out_dir: str,
    cline_ums_token: Optional[str],
    custom_weights_path: Optional[str] = None,
    causallm_project_root: Optional[str] = None,
    install_generated_files: bool = False,
    generated_header_directory: str = "include/generated",
    generated_source_directory: str = "src/generated",
    # Cline configuration (Ollama by default - CodeLlama 13B for best code generation)
    cline_api_base: str = "http://localhost:11434/v1",
    cline_model: str = "codellama:13b",
    cline_max_tokens: int = 8192,
    max_fix_iterations: int = 2,
    # NOOA configuration
    use_nooa: bool = True,
    # CausalLM .bin generation configuration
    causallm_path: Optional[str] = None,
    enable_quantization: bool = True,
    quantization_preset: str = "Q4_0-FP32",
    target_isa: str = "x86",
    nntr_quantize_path: Optional[str] = None,
    auto_compile: bool = False,
    use_docker_build: bool = True,
    auto_fix_use_llm: bool = False,
    nntrainer_path: Optional[str] = None,
    qwen_moe_variant: str = "standard",
    # Pre-existing code configuration
    use_pre_existing_code: bool = True,
    nntrainer_root: Optional[str] = None,
) -> PipelineState:
    return PipelineState(
        model_name=model_name,
        out_dir=out_dir,
        cline_ums_token=cline_ums_token,
        custom_weights_path=custom_weights_path,
        fix_iterations=0,
        max_fix_iterations=max_fix_iterations,
        compiled=False,
        errors=[],
        cpp_emission_mode="",
        requires_causallm_build=False,
        skip_cpp_compilation=True,
        nntrainer_missing=False,
        compile_skipped=False,
        compile_command="",
        syntax_ok=None,
        auto_compile=auto_compile,
        use_docker_build=use_docker_build,
        auto_fix_use_llm=auto_fix_use_llm,
        nntrainer_path=nntrainer_path,
        qwen_moe_variant=qwen_moe_variant,
        causallm_project_root=causallm_project_root,
        install_generated_files=install_generated_files,
        generated_header_directory=generated_header_directory,
        generated_source_directory=generated_source_directory,
        # Cline configuration (Gauss models only)
        cline_api_base=cline_api_base,
        cline_model=cline_model,
        cline_max_tokens=cline_max_tokens,
        # NOOA configuration
        use_nooa=use_nooa,
        # CausalLM .bin generation configuration
        causallm_path=causallm_path,
        enable_quantization=enable_quantization,
        quantization_preset=quantization_preset,
        target_isa=target_isa,
        nntr_quantize_path=nntr_quantize_path,
        fix_history=[],
        # Pre-existing code configuration
        use_pre_existing_code=use_pre_existing_code,
        nntrainer_root=nntrainer_root,
    )

