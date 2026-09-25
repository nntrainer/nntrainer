"""
NOOA-based C++ Generator Agent using Cline (UMS Token) as LLM client.
Generates nntrainer-compatible C++ code from graph IR.

Usage:
    from .nooa_cpp_generator import CppGenerationAgent
    import asyncio
    
    agent = CppGenerationAgent()
    result = asyncio.run(agent.generate(graph_ir, output_dir))
"""
from __future__ import annotations
import os
import asyncio
from pathlib import Path
from typing import Annotated, Dict, Any
from pydantic import BaseModel, Field

from nooa import Agent
from nooa.strategies import CodeActStrategy, PredictStrategy
from nooa.config import CodeActConfig, PredictConfig
from nooa.unifiedllm import get_llm_client


def get_cline_llm_client() -> Any:
    """
    Get LLM client using UMS token from environment.
    
    The UMS token is set by the VS Code extension from settings:
    - Extension reads: aiCompilerWorkbench.umsToken
    - Passed to Python agents via CLINE_UMS_TOKEN environment variable
    """
    ums_token = os.environ.get("CLINE_UMS_TOKEN")
    
    if not ums_token:
        raise RuntimeError(
            "CLINE_UMS_TOKEN not found. Set aiCompilerWorkbench.umsToken in VS Code settings "
            "or export CLINE_UMS_TOKEN environment variable."
        )
    
    # Cline exposes OpenAI-compatible API at localhost:6543
    return get_llm_client(
        "openai_compat/cline",
        api_base="http://localhost:6543/v1",
        api_key=ums_token,
    )


class GeneratedFiles(BaseModel):
    """Structured output for generated C++ files."""
    header: str = Field(description="Header file content")
    source: str = Field(description="Source file content")
    architecture: str = Field(description="Model architecture name")
    header_filename: str = Field(description="Header filename")
    source_filename: str = Field(description="Source filename")


class ArchitectureAnalysis(BaseModel):
    """Structured output for architecture analysis."""
    architecture: str = Field(description="Model architecture name (e.g., 'qwen3', 'llama')")
    emission_mode: str = Field(description="'model_api' or 'causallm_component'")
    uniform_layers: bool = Field(description="Whether all decoder layers share same structure")
    custom_layers_needed: list[str] = Field(description="List of custom layer types needed")
    complexity: str = Field(description="'simple', 'moderate', or 'complex'")


class CppGenerationAgent(Agent, llm=get_cline_llm_client()):
    """
    Generate nntrainer-compatible C++ code from graph IR.
    Uses Cline (via UMS token) as the LLM backend.
    """
    
    # nntrainer API reference - visible to LLM as context
    NNTRAINER_LAYER_TYPES: Dict[str, Dict[str, str]] = {
        "embedding": {"header": "<layer.h>", "create": 'createLayer("embedding")'},
        "fully_connected": {"header": "<layer.h>", "create": 'createLayer("fully_connected")'},
        "layer_norm": {"header": "<layer.h>", "create": 'createLayer("layer_norm")'},
        "rms_norm": {"header": "<layer.h>", "create": 'createLayer("rms_norm")'},
        "reshaped_rms_norm": {"header": "<reshaped_rms_norm.h>", "create": 'createLayer("reshaped_rms_norm")'},
        "mha_core": {"header": "<mha_core.h>", "create": 'createLayer("mha_core")'},
        "addition": {"header": "<layer.h>", "create": 'createLayer("addition")'},
        "multiply": {"header": "<layer.h>", "create": 'createLayer("multiply")'},
        "activation": {"header": "<layer.h>", "create": 'createLayer("activation")'},
        "kv_cache_placeholders": {"header": "<kv_cache_manager.h>", "create": "createKVCachePlaceholders"},
    }
    
    NNTRAINER_HEADERS = """#include <nntrainer/nntrainer.h>
#include <nntrainer/layer.h>
#include <nntrainer/tensor.h>
#include <nntrainer/model.h>
"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set context blocks with nntrainer API reference
        self.context["nntrainer_api"] = self.NNTRAINER_LAYER_TYPES
        self.context["nntrainer_headers"] = self.NNTRAINER_HEADERS
        self.context["nntrainer_patterns"] = {
            "layer_creation": 'LayerHandle name(createLayer("type", {"name=layer_name"}));',
            "tensor_call": "Tensor out = layer(input);",
            "multi_input": "Tensor out = layer({a, b, c});",
            "kv_cache": "auto [cache_k, cache_v] = createKVCachePlaceholders(layer_id, num_kv_heads);",
            "residual": "Tensor residual = addition({input, layer_output});",
        }

    @staticmethod
    def _extract_model_slug(architecture: str) -> str:
        """Extract capitalized model name from architecture.
        E.g., 'Qwen3ForCausalLM' -> 'Qwen3', 'Gemma3CausalLM' -> 'Gemma3'.
        Falls back to lowercase if extraction fails."""
        slug = architecture.replace("ForCausalLM", "").replace("CausalLM", "").replace("Model", "")
        if slug and len(slug) > 0:
            return slug[0].upper() + slug[1:] if len(slug) > 1 else slug.upper()
        return architecture.lower()

    def validate_graph_ir(self, graph_ir: dict) -> dict:
        """Validate and preprocess graph IR for generation."""
        required_keys = ["summary", "nodes", "edges"]
        for key in required_keys:
            if key not in graph_ir:
                raise ValueError(f"Missing required key: {key}")
        
        metadata = graph_ir.get("metadata", {})
        return {
            "architecture": graph_ir["summary"].get("architecture", "unknown"),
            "num_layers": len([n for n in graph_ir["nodes"] if "decoder" in n.get("group_id", "")]),
            "nodes": graph_ir["nodes"],
            "edges": graph_ir["edges"],
            "metadata": metadata,
            "emission_mode": metadata.get("emission_mode", "causallm_component"),
            "uniform_layers": metadata.get("uniform_layers", False),
        }
    
    def get_nntrainer_type(self, node_type: str) -> str:
        """Map semantic node type to nntrainer layer type."""
        type_mapping = {
            "embedding": "embedding",
            "linear": "fully_connected",
            "projection": "fully_connected",
            "normalization": "rms_norm",
            "layer_norm": "layer_norm",
            "attention": "mha_core",
            "matmul": "multiply",
            "activation": "activation",
            "residual": "addition",
        }
        return type_mapping.get(node_type, "fully_connected")
    
    @strategy(PredictStrategy(config=PredictConfig(max_retries=3)))
    async def analyze_architecture(self, graph_ir: dict) -> ArchitectureAnalysis:
        """Analyze the model architecture and determine generation strategy.
        
        Graph IR: {graph_ir}
        
        Return analysis with architecture, emission_mode, uniform_layers, 
        custom_layers_needed, and complexity.
        """
        ...
    
    @strategy(CodeActStrategy(config=CodeActConfig(
        max_iterations=10,
        max_retries=5,
        max_consecutive_text_only=0,  # Force tool usage
    )))
    async def generate_header(self, analysis: dict, graph_ir: dict) -> Annotated[str, "C++ header code"]:
        """Generate the C++ header file for the model.
        
        Analysis: {analysis}
        Graph IR: {graph_ir}
        
        nntrainer API: {self.context["nntrainer_api"]}
        
        Requirements:
        - Use #pragma once
        - Include architecture-specific headers (e.g., <qwen3_causallm.h>)
        - Define Generated{Arch}Transformer and Generated{Arch}CausalLM classes
        - Declare: createAttention, createMLP, createDecoderLayer, registerCustomLayers
        - Use proper inheritance from base transformer/causal_lm classes
        """
        ...
    
    @strategy(CodeActStrategy(config=CodeActConfig(
        max_iterations=15,
        max_retries=5,
    )))
    async def generate_source(self, analysis: dict, graph_ir: dict, header: str) -> Annotated[str, "C++ source code"]:
        """Generate the C++ source file implementing the model.
        
        Analysis: {analysis}
        Graph IR: {graph_ir}
        Generated header: {header}
        
        nntrainer patterns: {self.context["nntrainer_patterns"]}
        
        Requirements:
        - Include generated header and nntrainer headers
        - Implement createAttention, createMLP, createDecoderLayer
        - Use LayerHandle for RAII
        - Proper weight naming with layer_id templatization
        - Handle KV cache placeholders for attention
        - Implement registerCustomLayers() if custom layers needed
        """
        ...
    
    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=5)))
    async def fix_compile_errors(self, header: str, source: str, 
                                 errors: str) -> Annotated[dict, "Fixed files"]:
        """Fix compilation errors in the generated code.
        
        Header: {header}
        Source: {source}
        Compile errors: {errors}
        
        Fix ONLY what errors indicate. Preserve existing structure.
        Return dict with 'header' and 'source' keys.
        """
        ...
    
    async def generate(self, graph_ir: dict, output_dir: str) -> GeneratedFiles:
        """
        Main generation pipeline - orchestrates all steps.
        
        Args:
            graph_ir: nntrainer graph IR with nodes, edges, metadata
            output_dir: Directory to write generated files
            
        Returns:
            GeneratedFiles with header, source, and filenames
        """
        # Validate input
        validated_ir = self.validate_graph_ir(graph_ir)
        
        # Analyze architecture
        analysis = await self.analyze_architecture(validated_ir)
        analysis_dict = analysis.model_dump()
        
        # Generate header
        header = await self.generate_header(analysis_dict, validated_ir)
        
        # Generate source
        source = await self.generate_source(analysis_dict, validated_ir, header)
        
        # Write files
        output_path = Path(output_dir) / "generated"
        output_path.mkdir(parents=True, exist_ok=True)

        arch = self._extract_model_slug(analysis.architecture)
        header_path = output_path / f"{arch}_causallm.h"
        source_path = output_path / f"{arch}_causallm.cpp"

        header_path.write_text(header)
        source_path.write_text(source)

        return GeneratedFiles(
            header=header,
            source=source,
            architecture=analysis.architecture,
            header_filename=header_path.name,
            source_filename=source_path.name,
        )


async def generate_cpp_code(graph_ir: dict, output_dir: str) -> GeneratedFiles:
    """
    Convenience function for generating C++ code from graph IR.
    
    Args:
        graph_ir: nntrainer graph IR dictionary
        output_dir: Output directory for generated files
        
    Returns:
        GeneratedFiles with header, source, and metadata
    """
    agent = CppGenerationAgent()
    return await agent.generate(graph_ir, output_dir)
