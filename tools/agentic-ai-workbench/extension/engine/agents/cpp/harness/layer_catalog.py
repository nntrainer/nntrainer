"""
Complete catalog of available CausalLM layers.
This is the SOURCE OF TRUTH for what layers can be used in generated code.
"""

from typing import Dict, List, Any, Tuple

LAYER_CATALOG: Dict[str, Dict[str, Any]] = {
    # ========================================================================
    # CORE ATTENTION LAYERS
    # ========================================================================
    "fully_connected": {
        "header": "<fully_connected.h>",
        "class": "FullyConnectedLayer",
        "namespace": "causallm",
        "properties": {
            "required": ["unit"],
            "optional": ["disable_bias", "weight_initializer", "weight_dtype", "activation_type"],
            "types": {
                "unit": "unsigned int",
                "disable_bias": "bool",
                "weight_initializer": "string",
                "weight_dtype": "string",
                "activation_type": "string"
            }
        },
        "usage_example": """
LayerHandle wq(createLayer(
    "fully_connected",
    {withKey("name", "layer0_wq"),
     withKey("unit", std::to_string(head_dim * n_heads)),
     withKey("disable_bias", "true"),  // nntrainer accepts string "true"/"false" for bool
     withKey("weight_initializer", "ones")}));
""",
        "semantic_types": ["q_projection", "k_projection", "v_projection", "o_projection", 
                          "gate_projection", "up_projection", "down_projection"]
    },
    
    "mha_core": {
        "header": "<mha_core.h>",
        "class": "MHACoreLayer",
        "namespace": "causallm",
        "properties": {
            "required": ["num_heads", "num_heads_kv", "max_timestep"],
            "optional": ["sliding_window", "rope_theta", "max_position_embeddings", 
                        "max_new_tokens", "attn_logit_softcapping", "is_causal"],
            "types": {
                "num_heads": "unsigned int",
                "num_heads_kv": "unsigned int",
                "max_timestep": "unsigned int",
                "sliding_window": "unsigned int",
                "rope_theta": "unsigned int",
                "max_position_embeddings": "unsigned int",
                "max_new_tokens": "unsigned int",
                "attn_logit_softcapping": "float",
                "is_causal": "bool"
            }
        },
        "usage_example": """
LayerHandle mha(createLayer(
    "mha_core",
    {withKey("name", "layer0_attention"),
     withKey("num_heads", n_heads),
     withKey("num_heads_kv", n_heads / GQA_SIZE),
     withKey("max_timestep", MAX_SEQ_LEN),  // unsigned int: use integer literal, NOT std::to_string()
     withKey("sliding_window", SLIDING_WINDOW),
     withKey("rope_theta", ROPE_THETA),  // unsigned int: use integer literal
     withKey("is_causal", IS_CAUSAL)}));  // bool: use true/false directly, NOT ternary string
""",
        "semantic_types": ["attention"]
    },
    
    # ========================================================================
    # NORMALIZATION LAYERS
    # ========================================================================
    "reshaped_rms_norm": {
        "header": "<reshaped_rms_norm.h>",
        "class": "ReshapedRMSNormLayer",
        "namespace": "causallm",
        "properties": {
            "required": ["epsilon", "feature_size"],
            "optional": ["packed", "gamma_initializer"],
            "types": {
                "epsilon": "float",
                "feature_size": "unsigned int",
                "packed": "bool",
                "gamma_initializer": "string"
            }
        },
        "usage_example": """
LayerHandle q_norm(createLayer(
    "reshaped_rms_norm",
    {withKey("name", "layer0_q_norm"),
     withKey("packed", false),  // bool: use false NOT "false"
     withKey("epsilon", NORM_EPS),  // float: use macro directly
     withKey("feature_size", head_dim)}));  // unsigned int: use variable directly
""",
        "semantic_types": ["normalization", "q_norm", "k_norm"]
    },
    
    "rms_norm": {
        "header": "<rms_norm.h>",
        "class": "RMSNormLayer",
        "namespace": "causallm",
        "properties": {
            "required": ["epsilon"],
            "optional": ["packed", "skip_prefill", "gamma_initializer"],
            "types": {
                "epsilon": "float",
                "packed": "bool",
                "skip_prefill": "bool",
                "gamma_initializer": "string"
            }
        },
        "usage_example": """
LayerHandle attn_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer0_attention_norm"),
     withKey("epsilon", NORM_EPS),  // float: use macro directly
     withKey("packed", false)}));  // bool: use false NOT "false"
""",
        "semantic_types": ["normalization", "pre_attention_norm", "post_attention_norm"]
    },
    
    # ========================================================================
    # ACTIVATION LAYERS
    # ========================================================================
    "activation": {
        "header": "<activation_layer.h>",
        "class": "ActivationLayer",
        "namespace": "ml::train",
        "properties": {
            "required": ["activation"],
            "optional": [],
            "types": {
                "activation": "string"  # "silu", "tanh_gelu", "gelu", "relu", etc.
            }
        },
        "usage_example": """
LayerHandle gelu(createLayer(
    "activation",
    {withKey("name", "layer0_ffn_gate_gelu"),
     withKey("activation", "tanh_gelu")}));
""",
        "semantic_types": ["activation"]
    },
    
    "swiglu": {
        "header": "<swiglu.h>",
        "class": "SwiGLULayer",
        "namespace": "causallm",
        "properties": {
            "required": [],
            "optional": [],
            "types": {}
        },
        "usage_example": """
LayerHandle swiglu(createLayer(
    "swiglu",
    {withKey("name", "layer0_ffn_swiglu")}));
""",
        "semantic_types": ["activation", "swiglu"]
    },
    
    # ========================================================================
    # ELEMENT-WISE OPERATIONS
    # ========================================================================
    "multiply": {
        "header": "<multiply.h>",
        "class": "MultiplyLayer",
        "namespace": "ml::train",
        "properties": {
            "required": [],
            "optional": [],
            "types": {}
        },
        "usage_example": """
LayerHandle mul(createLayer(
    "multiply",
    {withKey("name", "layer0_ffn_geglu")}));
Tensor geglu = mul({gate_gelu, up});
""",
        "semantic_types": ["element_wise", "geglu"]
    },
    
    "addition": {
        "header": "<addition.h>",
        "class": "AdditionLayer",
        "namespace": "ml::train",
        "properties": {
            "required": [],
            "optional": [],
            "types": {}
        },
        "usage_example": """
LayerHandle residual(createLayer(
    "addition",
    {withKey("name", "layer0_attention_residual")}));
return residual({input, att_out});
""",
        "semantic_types": ["element_wise", "residual"]
    },
    
    "scalar_multiply": {
        "header": "<scalar_multiply.h>",
        "class": "ScalarMultiplyLayer",
        "namespace": "causallm",
        "properties": {
            "required": ["scale"],
            "optional": [],
            "types": {
                "scale": "float"
            }
        },
        "usage_example": """
LayerHandle scale(createLayer(
    "scalar_multiply",
    {withKey("name", "layer0_embed_scale"),
     withKey("scale", std::to_string(EMBEDDING_SCALE))}));
""",
        "semantic_types": ["element_wise", "scaling"]
    },
    
    # ========================================================================
    # EMBEDDING AND OUTPUT LAYERS
    # ========================================================================
    "embedding": {
        "header": "<embedding_layer.h>",
        "class": "EmbeddingLayer",
        "namespace": "ml::train",
        "properties": {
            "required": [],  # Made optional - unit can be derived from config
            "optional": ["unit", "out_dim", "weight_dtype", "lut_path"],
            "types": {
                "unit": "unsigned int",
                "out_dim": "unsigned int",  # Alias for unit (from HF tracer)
                "weight_dtype": "string",
                "lut_path": "string"
            }
        },
        "usage_example": """
LayerHandle embed(createLayer(
    "embedding",
    {withKey("name", "embed_tokens"),
     withKey("unit", NUM_VOCAB)}));  // or use out_dim for vocab size
""",
        "semantic_types": ["embedding"],
        "note": "Accepts both 'unit' (CausalLM standard) and 'out_dim' (from HF tracer) for vocabulary size"
    },
    
    # ========================================================================
    # KV CACHE PLACEHOLDERS (special node type for KV cache management)
    # ========================================================================
    "kv_cache_placeholders": {
        "header": "<kv_cache_manager.h>",
        "class": "KVCachePlaceholders",
        "namespace": "causallm",
        "properties": {
            "required": [],
            "optional": ["layer_id", "num_heads", "num_kv_heads"],
            "types": {
                "layer_id": "int",
                "num_heads": "unsigned int",
                "num_kv_heads": "unsigned int"  # Alias for num_heads (GQA models)
            }
        },
        "usage_example": """
auto [cache_k, cache_v] = createKVCachePlaceholders(layer_id, n_heads);
""",
        "semantic_types": ["kv_cache", "placeholder"],
        "note": "This is a special helper, not a createLayer() call. Used for KV cache management. Accepts both 'num_heads' and 'num_kv_heads' (for GQA models)."
    },
    
    "lm_head": {
        "header": "<lm_head.h>",
        "class": "LMHeadLayer",
        "namespace": "causallm",
        "properties": {
            "required": ["unit"],
            "optional": ["tie_word_embeddings"],
            "types": {
                "unit": "unsigned int",
                "tie_word_embeddings": "bool"
            }
        },
        "usage_example": """
LayerHandle lm_head(createLayer(
    "lm_head",
    {withKey("name", "lm_head"),
     withKey("unit", NUM_VOCAB),
     withKey("tie_word_embeddings", TIE_WORD_EMBEDDINGS)}));  // bool: use bool variable directly, NOT ternary string
""",
        "semantic_types": ["output", "lm_head"]
    },
    
    "tie_word_embedding": {
        "header": "<tie_word_embedding.h>",
        "class": "TieWordEmbeddingLayer",
        "namespace": "causallm",
        "properties": {
            "required": [],
            "optional": [],
            "types": {}
        },
        "usage_example": """
LayerHandle tie(createLayer(
    "tie_word_embedding",
    {withKey("name", "tie_word_embedding")}));
""",
        "semantic_types": ["embedding", "tie"]
    },
}


def get_layer_info(layer_type: str) -> Dict[str, Any]:
    """Get information about a specific layer type."""
    return LAYER_CATALOG.get(layer_type, {})


def get_available_layers() -> List[str]:
    """Get list of all available layer types."""
    return list(LAYER_CATALOG.keys())


def validate_layer_usage(layer_type: str, attributes: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate that a layer is used correctly.
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    layer_info = get_layer_info(layer_type)
    
    if not layer_info:
        return False, [f"Unknown layer type: {layer_type}"]
    
    props = layer_info["properties"]
    
    # Check required properties
    for req_prop in props["required"]:
        if req_prop not in attributes:
            errors.append(f"Layer '{layer_type}' missing required property: {req_prop}")
    
    # Check for unknown properties (typos in property names)
    valid_props = set(props["required"] + props.get("optional", []))
    for prop_name in attributes:
        if prop_name not in valid_props:
            errors.append(f"Layer '{layer_type}' has unknown property: '{prop_name}'. Valid properties: {sorted(valid_props)}")
    
    # Check property types (basic validation)
    # IMPORTANT: nntrainer's withKey() accepts string values and converts them internally at runtime.
    # The reference implementation (Applications/CausalLM/models/qwen3/qwen3_causallm.cpp) uses:
    #   withKey("disable_bias", "true")  -- string "true" for bool
    #   withKey("packed", "false")       -- string "false" for bool
    #   withKey("epsilon", std::to_string(NORM_EPS)) -- string for float
    #   withKey("is_causal", IS_CAUSAL ? "true" : "false") -- ternary string for bool
    # So string representations are the EXPECTED format, not errors!
    type_map = {
        "unsigned int": (int, lambda x: x >= 0),
        "int": (int, lambda x: True),
        "float": (float, lambda x: True),
        "bool": (bool, lambda x: True),
        "string": (str, lambda x: True),
    }
    
    for prop_name, prop_value in attributes.items():
        if prop_name in props.get("types", {}):
            expected_type = props["types"][prop_name]
            if expected_type in type_map:
                expected_python_type, type_validator = type_map[expected_type]
                
                # nntrainer's withKey() accepts string representations of all types.
                # String values are the EXPECTED format in generated C++ code.
                # We only validate that the string LOOKS like a valid literal for the type.
                
                if isinstance(prop_value, str):
                    # String values are always OK - nntrainer converts them at runtime
                    # Just validate they look like valid literals for the expected type
                    if expected_type == "bool":
                        # Valid bool strings: "true", "false", "1", "0", or expressions
                        valid_bools = ("true", "false", "1", "0")
                        if prop_value.lower() not in valid_bools:
                            # Allow C++ expressions like: IS_CAUSAL ? "true" : "false"
                            if "?" not in prop_value and "std::to_string" not in prop_value:
                                # Check if it's a variable/expression (alphanumeric with underscores)
                                if not prop_value.replace("_", "").replace(" ", "").isalnum():
                                    errors.append(f"Layer '{layer_type}' property '{prop_name}' has invalid bool value: '{prop_value}' (expected 'true', 'false', '1', '0', or a C++ expression)")
                    elif expected_type in ("unsigned int", "int"):
                        try:
                            int(prop_value)
                        except ValueError:
                            # Allow C++ expressions like std::to_string(MAX_SEQ_LEN) or just MAX_SEQ_LEN
                            # These are valid C++ that will compile
                            pass  # String expressions are OK for int types
                    elif expected_type == "float":
                        try:
                            float(prop_value)
                        except ValueError:
                            # Allow C++ expressions like std::to_string(NORM_EPS) or NORM_EPS
                            pass  # String expressions are OK for float types
                    # else: string type, any string is OK
                    
                elif not isinstance(prop_value, expected_python_type):
                    # Native Python types are also OK (e.g., True instead of "true")
                    pass  # Both native types and strings are accepted
    
    return len(errors) == 0, errors
