"""
Abstract base class for C++ code generation harnesses.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple


@dataclass
class GeneratedFiles:
    """Result of C++ generation"""
    header: str
    source: str
    header_filename: str
    source_filename: str
    architecture: str
    transformer_class: str
    causal_lm_class: str
    required_layers: List[str]
    constants: Dict[str, Any]
    success: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class CppGeneratorHarness(ABC):
    """
    Abstract harness for CausalLM C++ code generation.
    
    RESPONSIBILITIES:
    1. Read nntrainer_graph_ir from state
    2. Generate C++ code using only layers from LAYER_CATALOG
    3. Validate generated code compiles and uses valid layers
    4. Return GeneratedFiles with header, source, and metadata
    """
    
    @abstractmethod
    def get_generator_id(self) -> str:
        """Unique identifier for this generator"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Capabilities this generator supports:
        - 'causallm_component': Can generate <model>_causallm.{h,cpp}
        - 'custom_decoder': Supports custom decoder blocks
        - 'sliding_window': Supports sliding window attention
        - 'gqa': Supports grouped query attention (GQA)
        - 'moe': Supports mixture of experts
        """
        pass
    
    @abstractmethod
    def generate(self, state: dict) -> GeneratedFiles:
        """
        Generate C++ code from nntrainer_graph_ir.
        
        Args:
            state: Pipeline state containing:
                - architecture: HF architecture name
                - hf_config: HuggingFace config dict
                - nntrainer_graph_ir: Graph IR from nntrainer_lowering
                - nntrainer_root: Path to nntrainer repository
        
        Returns:
            GeneratedFiles with header, source, and metadata
        """
        pass
    
    def validate_nntrainer_graph_ir(self, graph_ir: dict) -> Dict[str, Any]:
        """
        Validate that nntrainer_graph_ir is compatible with available layers.
        
        This is CRITICAL - the graph_ir must only use layers that exist in
        Applications/CausalLM/layers/
        """
        from .layer_catalog import LAYER_CATALOG, validate_layer_usage
        
        errors = []
        warnings = []
        layer_usage = {}
        
        available_layers = set(LAYER_CATALOG.keys())
        
        for node in graph_ir.get("nodes", []):
            node_type = node.get("node_type", "")
            node_name = node.get("name", "unknown")
            
            # Check 1: Node type exists in layers/
            if node_type and node_type not in available_layers:
                errors.append(
                    f"Node '{node_name}' uses unknown layer type '{node_type}'. "
                    f"Available layers: {sorted(available_layers)}"
                )
            else:
                layer_usage[node_type] = layer_usage.get(node_type, 0) + 1
            
            # Check 2: Validate layer attributes
            # NOTE: Attributes from graph_ir are Python values that will become C++ code.
            # We only check for unknown properties, not type mismatches, since any Python
            # value can be converted to a C++ expression (strings, numbers, bools all work).
            if node_type in available_layers:
                attributes = node.get("attributes", {})
                # Only check for unknown properties - skip type validation
                # Type validation happens at C++ compile time, not here
                from .layer_catalog import get_layer_info
                layer_info = get_layer_info(node_type)
                if layer_info:
                    props = layer_info.get("properties", {})
                    valid_props = set(props.get("required", []) + props.get("optional", []))
                    for prop_name in attributes:
                        if prop_name not in valid_props:
                            errors.append(f"Node '{node_name}': Layer '{node_type}' has unknown property: '{prop_name}'. Valid properties: {sorted(valid_props)}")
                    # Check required properties are present
                    for req_prop in props.get("required", []):
                        if req_prop not in attributes:
                            errors.append(f"Node '{node_name}': Layer '{node_type}' missing required property: '{req_prop}'")
        
        # Check 3: Required layers for CausalLM
        used_layers = set(layer_usage.keys())
        required_for_causallm = {"embedding", "mha_core", "fully_connected"}
        missing_required = required_for_causallm - used_layers
        if missing_required:
            warnings.append(f"Missing typical CausalLM layers: {missing_required}")
        
        # Check 4: Graph connectivity
        node_ids = {n["id"] for n in graph_ir.get("nodes", [])}
        for edge in graph_ir.get("edges", []):
            if edge["source"] not in node_ids:
                errors.append(f"Edge references non-existent source: {edge['source']}")
            if edge["target"] not in node_ids:
                errors.append(f"Edge references non-existent target: {edge['target']}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "layer_usage": layer_usage
        }
    
    def extract_constants_from_config(self, hf_config: dict) -> Dict[str, Any]:
        """
        Extract C++ constants from HuggingFace config.
        
        These constants will be written as `constexpr auto` in the generated code.
        """
        constants = {}
        
        # Required constants
        constants['NORM_EPS'] = hf_config.get('rms_norm_eps', 1e-5)
        constants['NUM_LAYERS'] = hf_config.get('num_hidden_layers', 32)
        constants['NUM_HEADS'] = hf_config.get('num_attention_heads', 32)
        constants['HIDDEN_SIZE'] = hf_config.get('hidden_size', 4096)
        constants['INTERMEDIATE_SIZE'] = hf_config.get('intermediate_size', 11008)
        constants['MAX_POSITION_EMBEDDINGS'] = hf_config.get('max_position_embeddings', 2048)
        constants['NUM_VOCAB'] = hf_config.get('vocab_size', 32000)
        
        # GQA calculation
        num_kv_heads = hf_config.get('num_key_value_heads', constants['NUM_HEADS'])
        # Avoid division by zero - default to NUM_HEADS if num_kv_heads is 0 or missing
        if not num_kv_heads or num_kv_heads == 0:
            num_kv_heads = constants['NUM_HEADS']
        constants['GQA_SIZE'] = max(1, int(constants['NUM_HEADS']) // int(num_kv_heads))
        
        # Optional constants
        if 'sliding_window' in hf_config:
            constants['SLIDING_WINDOW'] = hf_config['sliding_window']
        else:
            constants['SLIDING_WINDOW'] = 'UINT_MAX'
        
        if 'rope_theta' in hf_config:
            constants['ROPE_THETA'] = int(hf_config['rope_theta'])
        
        if 'max_new_tokens' in hf_config:
            constants['NUM_TO_GENERATE'] = hf_config['max_new_tokens']
        else:
            constants['NUM_TO_GENERATE'] = 512
        
        # Architecture-specific
        constants['IS_CAUSAL'] = hf_config.get('is_causal', True)
        constants['TIE_WORD_EMBEDDINGS'] = hf_config.get('tie_word_embeddings', False)
        
        if 'attn_logit_softcapping' in hf_config:
            constants['ATTN_LOGIT_SOFTCAPPING'] = hf_config['attn_logit_softcapping']
        else:
            constants['ATTN_LOGIT_SOFTCAPPING'] = 0.0
        
        # Layer types (for models like Gemma3 with per-layer variation)
        if 'layer_types' in hf_config:
            constants['HAS_LAYER_TYPES'] = True
            constants['LAYER_TYPES'] = hf_config['layer_types']
        else:
            constants['HAS_LAYER_TYPES'] = False
        
        return constants
    
    def format_constants_cpp(self, constants: Dict[str, Any]) -> str:
        """Format constants as C++ constexpr declarations."""
        lines = ["// Model-specific constants derived from HuggingFace config"]
        
        for name, value in constants.items():
            # Skip None values - they're not valid C++
            if value is None:
                continue
            
            # Skip complex types that can't be constexpr
            if isinstance(value, (list, dict, tuple)):
                continue
            
            if isinstance(value, bool):
                cpp_value = "true" if value else "false"
            elif isinstance(value, float):
                cpp_value = f"{value}f"
            elif isinstance(value, str):
                cpp_value = f'"{value}"'
            elif value == 'UINT_MAX':
                cpp_value = 'UINT_MAX'
            else:
                cpp_value = str(value)
            
            lines.append(f"constexpr auto {name} = {cpp_value};")
        
        return "\n".join(lines)
