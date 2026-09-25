"""
Model-specific strategies for C++ code generation.

This module provides architecture-specific configuration and code generation
logic for different model families (Qwen, Gemma, Llama, etc.). Instead of
relying solely on graph metadata, this encodes architectural knowledge
directly to ensure correct code generation.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any


@dataclass
class ModelStrategy:
    """
    Architecture-specific configuration for code generation.
    
    Attributes:
        family: Model family name (e.g., "qwen", "gemma", "llama")
        norm_type: Type of normalization ("rms_norm", "reshaped_rms_norm")
        norm_eps: RMSNorm epsilon value
        mlp_type: MLP activation type ("swiglu", "geglu", "gelu")
        has_bias: Whether FC layers have bias (False = disable_bias=true)
        has_sliding_window: Whether model uses sliding window attention
        has_qk_norm: Whether Q/K projections have normalization
        custom_layers: List of custom layer types that need registration
        architecture_suffix: Suffix for class names (e.g., "ForCausalLM")
    """
    family: str
    norm_type: str = "rms_norm"
    norm_eps: float = 1e-6
    mlp_type: str = "swiglu"
    has_bias: bool = False  # False means disable_bias=true
    has_sliding_window: bool = False
    has_qk_norm: bool = False
    custom_layers: List[str] = field(default_factory=list)
    architecture_suffix: str = "ForCausalLM"
    
    def get_disable_bias(self) -> str:
        """Returns the disable_bias property value for this model."""
        return "false" if self.has_bias else "true"
    
    def needs_mlp_override(self) -> bool:
        """
        Returns True if this model needs a custom createMlp() implementation.
        
        The base Transformer::createMlp implements standard SwiGLU.
        Models with different MLP types need their own implementation.
        """
        return self.mlp_type.lower() not in ("swiglu", "gated_swiglu")
    
    def get_mlp_activation(self) -> str:
        """Returns the activation function for MLP."""
        activations = {
            "swiglu": "swish",  # nntrainer uses "swish" for SiLU
            "geglu": "gelu",
            "gelu": "gelu",
            "relu": "relu",
        }
        return activations.get(self.mlp_type.lower(), "swish")


# Registry of model-specific strategies
MODEL_STRATEGIES: Dict[str, ModelStrategy] = {
    # Qwen family
    "qwen": ModelStrategy(
        family="qwen",
        norm_type="rms_norm",
        norm_eps=1e-6,
        mlp_type="swiglu",
        has_bias=False,
        has_sliding_window=False,
        has_qk_norm=False,
        custom_layers=[],
        architecture_suffix="ForCausalLM",
    ),
    "qwen2": ModelStrategy(
        family="qwen2",
        norm_type="rms_norm",
        norm_eps=1e-6,
        mlp_type="swiglu",
        has_bias=False,
        has_sliding_window=False,
        has_qk_norm=False,
        custom_layers=[],
        architecture_suffix="CausalLM",  # Qwen2 uses "Qwen2CausalLM"
    ),
    "qwen3": ModelStrategy(
        family="qwen3",
        norm_type="reshaped_rms_norm",  # Qwen3 uses ReshapedRMSNorm for Q/K
        norm_eps=1e-5,
        mlp_type="swiglu",
        has_bias=False,
        has_sliding_window=True,  # Optional, model-dependent
        has_qk_norm=True,  # Qwen3 has Q/K norm before attention
        custom_layers=["reshaped_rms_norm"],
        architecture_suffix="ForCausalLM",
    ),
    
    # Gemma family
    "gemma": ModelStrategy(
        family="gemma",
        norm_type="rms_norm",
        norm_eps=1e-6,
        mlp_type="geglu",  # Gemma uses GeGLU
        has_bias=True,  # Gemma has bias in FC layers
        has_sliding_window=True,
        has_qk_norm=False,
        custom_layers=[],
        architecture_suffix="CausalLM",
    ),
    "gemma2": ModelStrategy(
        family="gemma2",
        norm_type="rms_norm",
        norm_eps=1e-6,
        mlp_type="geglu",
        has_bias=True,
        has_sliding_window=True,
        has_qk_norm=False,
        custom_layers=[],
        architecture_suffix="CausalLM",
    ),
    "gemma3": ModelStrategy(
        family="gemma3",
        norm_type="rms_norm",
        norm_eps=1e-6,
        mlp_type="geglu",
        has_bias=True,
        has_sliding_window=True,
        has_qk_norm=False,
        custom_layers=[],
        architecture_suffix="CausalLM",
    ),
    "gemma4": ModelStrategy(
        family="gemma4",
        norm_type="rms_norm",
        norm_eps=1e-6,
        mlp_type="geglu",
        has_bias=True,
        has_sliding_window=True,
        has_qk_norm=False,
        custom_layers=[],
        architecture_suffix="CausalLM",
    ),
    
    # Llama family
    "llama": ModelStrategy(
        family="llama",
        norm_type="rms_norm",
        norm_eps=1e-5,
        mlp_type="swiglu",
        has_bias=False,
        has_sliding_window=False,
        has_qk_norm=False,
        custom_layers=[],
        architecture_suffix="ForCausalLM",
    ),
    
    # Mistral family
    "mistral": ModelStrategy(
        family="mistral",
        norm_type="rms_norm",
        norm_eps=1e-5,
        mlp_type="swiglu",
        has_bias=False,
        has_sliding_window=True,
        has_qk_norm=False,
        custom_layers=[],
        architecture_suffix="ForCausalLM",
    ),
}


def get_strategy(architecture: str) -> ModelStrategy:
    """
    Get the model-specific strategy for an architecture.
    
    Args:
        architecture: Architecture name (e.g., "qwen2", "gemma3", "llama")
    
    Returns:
        ModelStrategy for the architecture, or a sensible default
    """
    # Normalize architecture name
    arch_lower = architecture.lower()
    
    # Try exact match first
    if arch_lower in MODEL_STRATEGIES:
        return MODEL_STRATEGIES[arch_lower]
    
    # Try prefix match (e.g., "qwen2.5" -> "qwen2")
    for key in MODEL_STRATEGIES:
        if arch_lower.startswith(key):
            return MODEL_STRATEGIES[key]
    
    # Try to infer from architecture name
    for key in MODEL_STRATEGIES:
        if key in arch_lower:
            return MODEL_STRATEGIES[key]
    
    # Default to generic transformer strategy
    return ModelStrategy(
        family=arch_lower,
        norm_type="rms_norm",
        norm_eps=1e-6,
        mlp_type="swiglu",
        has_bias=False,
        has_sliding_window=False,
        has_qk_norm=False,
        custom_layers=[],
        architecture_suffix="ForCausalLM",
    )


def get_mlp_code_template(strategy: ModelStrategy, layer_id: str = "layer_id") -> str:
    """
    Generate MLP implementation code for a specific model strategy.
    
    Args:
        strategy: ModelStrategy for the architecture
        layer_id: C++ expression for layer ID (e.g., "layer_id" or std::to_string(i))
    
    Returns:
        C++ code string for the MLP implementation
    """
    if strategy.mlp_type.lower() == "swiglu":
        # Standard SwiGLU - base class handles this, no override needed
        return ""
    
    if strategy.mlp_type.lower() == "geglu":
        # GeGLU: gate × GELU(up)
        disable_bias = strategy.get_disable_bias()
        mlp_activation = strategy.get_mlp_activation()
        return f'''
  // Gate projection
  LayerHandle gate_proj(createLayer(
    "fully_connected",
    {{withKey("name", "model.layers." + std::to_string({layer_id}) + ".mlp.gate_proj"),
     withKey("unit", hidden_dim), withKey("disable_bias", "{disable_bias}")}}));
  Tensor gate = gate_proj(input);

  // Up projection
  LayerHandle up_proj(createLayer(
    "fully_connected",
    {{withKey("name", "model.layers." + std::to_string({layer_id}) + ".mlp.up_proj"),
     withKey("unit", hidden_dim), withKey("disable_bias", "{disable_bias}")}}));
  Tensor up = up_proj(input);

  // GeGLU activation: gate × GELU(up)
  LayerHandle gelu(createLayer(
    "activation",
    {{withKey("name", "model.layers." + std::to_string({layer_id}) + ".mlp.gelu"),
     withKey("activation", "{mlp_activation}")}}));
  Tensor gate_act = gelu(gate);

  LayerHandle mul(createLayer(
    "multiply",
    {{withKey("name", "model.layers." + std::to_string({layer_id}) + ".mlp.mul")}}));
  Tensor mlp_mid = mul({{gate_act, up}});

  // Down projection
  LayerHandle down_proj(createLayer(
    "fully_connected",
    {{withKey("name", "model.layers." + std::to_string({layer_id}) + ".mlp.down_proj"),
     withKey("unit", dim), withKey("disable_bias", "{disable_bias}")}}));
  return down_proj(mlp_mid);
'''
    
    # Generic fallback for unknown MLP types
    disable_bias = strategy.get_disable_bias()
    mlp_activation = strategy.get_mlp_activation()
    return f'''
  // Generic MLP (architecture: {strategy.family})
  // Activation type: {strategy.mlp_type}
  LayerHandle mlp_gate(createLayer(
    "fully_connected",
    {{withKey("name", "model.layers." + std::to_string({layer_id}) + ".mlp.gate"),
     withKey("unit", hidden_dim), withKey("disable_bias", "{disable_bias}")}}));
  Tensor gate = mlp_gate(input);

  LayerHandle mlp_up(createLayer(
    "fully_connected",
    {{withKey("name", "model.layers." + std::to_string({layer_id}) + ".mlp.up"),
     withKey("unit", hidden_dim), withKey("disable_bias", "{disable_bias}")}}));
  Tensor up = mlp_up(input);

  // Activation: {mlp_activation}
  LayerHandle activation(createLayer(
    "activation",
    {{withKey("name", "model.layers." + std::to_string({layer_id}) + ".mlp.activation"),
     withKey("activation", "{mlp_activation}")}}));
  Tensor gate_act = activation(gate);

  LayerHandle mul(createLayer(
    "multiply",
    {{withKey("name", "model.layers." + std::to_string({layer_id}) + ".mlp.mul")}}));
  Tensor mlp_mid = mul({{gate_act, up}});

  LayerHandle down_proj(createLayer(
    "fully_connected",
    {{withKey("name", "model.layers." + std::to_string({layer_id}) + ".mlp.down"),
     withKey("unit", dim), withKey("disable_bias", "{disable_bias}")}}));
  return down_proj(mlp_mid);
'''


def get_attention_code_template(
    strategy: ModelStrategy,
    layer_id: str = "layer_id",
    use_constants: bool = True
) -> str:
    """
    Generate attention implementation code for a specific model strategy.
    
    Args:
        strategy: ModelStrategy for the architecture
        layer_id: C++ expression for layer ID
        use_constants: If True, use base class constants (GQA_SIZE, etc.)
    
    Returns:
        C++ code string for the attention implementation
    """
    lines = []
    gqa_expr = "n_heads / GQA_SIZE" if use_constants else "num_kv_heads"
    disable_bias = strategy.get_disable_bias()
    
    # Q projection
    lines.append(f'''  // Q layer projection
  LayerHandle wq(createLayer(
    "fully_connected",
    {{withKey("name", "model.layers." + std::to_string({layer_id}) + ".self_attn.q_proj"),
     withKey("unit", head_dim * n_heads), withKey("disable_bias", "{disable_bias}")}}));
  Tensor q = wq(query);
''')
    
    # Q norm (if applicable)
    if strategy.has_qk_norm:
        norm_layer = strategy.norm_type
        lines.append(f'''
  // Q RMSNorm (q_norm) - {strategy.family} specific
  LayerHandle q_norm(createLayer(
    "{norm_layer}",
    {{withKey("name", "model.layers." + std::to_string({layer_id}) + ".self_attn.q_norm"),
     withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("feature_size", std::to_string(head_dim))}}));
  Tensor q_normed = q_norm(q);
''')
        q_var = "q_normed"
    else:
        q_var = "q"
    
    # K projection
    lines.append(f'''
  // K layer projection
  LayerHandle wk(createLayer(
    "fully_connected",
    {{withKey("name", "model.layers." + std::to_string({layer_id}) + ".self_attn.k_proj"),
     withKey("unit", head_dim * {gqa_expr}),
     withKey("disable_bias", "{disable_bias}")}}));
  Tensor k = wk(key);
''')
    
    # K norm (if applicable)
    if strategy.has_qk_norm:
        norm_layer = strategy.norm_type
        lines.append(f'''
  // K RMSNorm (k_norm) - {strategy.family} specific
  LayerHandle k_norm(createLayer(
    "{norm_layer}",
    {{withKey("name", "model.layers." + std::to_string({layer_id}) + ".self_attn.k_norm"),
     withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("feature_size", std::to_string(head_dim))}}));
  Tensor k_normed = k_norm(k);
''')
        k_var = "k_normed"
    else:
        k_var = "k"
    
    # V projection
    lines.append(f'''
  // V layer projection
  LayerHandle wv(createLayer(
    "fully_connected",
    {{withKey("name", "model.layers." + std::to_string({layer_id}) + ".self_attn.v_proj"),
     withKey("unit", head_dim * {gqa_expr}),
     withKey("disable_bias", "{disable_bias}")}}));
  Tensor v = wv(value);
''')
    
    # KV cache placeholders
    lines.append('''
  // External KV cache placeholders (per-layer)
  auto [cache_k, cache_v] = createKVCachePlaceholders(layer_id, n_heads);
''')
    
    # MHA core parameters
    mha_params = [
        f'withKey("name", "model.layers." + std::to_string({layer_id}) + ".self_attn.mha_core")',
        f'withKey("num_heads", n_heads)',
        f'withKey("num_heads_kv", {gqa_expr})',
        f'withKey("max_timestep", std::to_string(MAX_SEQ_LEN))',
    ]
    
    if strategy.has_sliding_window:
        mha_params.append('withKey("sliding_window", SLIDING_WINDOW)')
    
    mha_params.extend([
        'withKey("rope_theta", ROPE_THETA)',
        'withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS)',
        'withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE))',
        'withKey("is_causal", IS_CAUSAL ? "true" : "false")',
    ])
    
    # MHA core
    params_str = ",\n     ".join(mha_params)
    lines.append(f'''
  // MHA core layer
  LayerHandle mha(createLayer(
    "mha_core",
    {{{params_str}}}));
  Tensor a = mha({{{q_var}, {k_var}, v, cache_k, cache_v}});
''')
    
    # O projection
    lines.append(f'''
  // O layer (output projection)
  LayerHandle wo(createLayer(
    "fully_connected",
    {{withKey("name", "model.layers." + std::to_string({layer_id}) + ".self_attn.o_proj"),
     withKey("unit", DIM), withKey("disable_bias", "{disable_bias}")}}));
  return wo(a);
''')
    
    return "\n".join(lines)


def get_register_custom_layers_code(strategy: ModelStrategy) -> str:
    """
    Generate registerCustomLayers() implementation for a specific model strategy.
    
    Args:
        strategy: ModelStrategy for the architecture
    
    Returns:
        C++ code string for registerCustomLayers()
    """
    if not strategy.custom_layers:
        return '''  // This architecture uses only stock nntrainer layers -- nothing to register.
}'''
    
    # Generate registration code for each custom layer
    registrations = []
    for layer_type in strategy.custom_layers:
        if layer_type == "reshaped_rms_norm":
            registrations.append('''    app_context->registerFactory(
        nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);''')
    
    if not registrations:
        return '''  // This architecture uses only stock nntrainer layers -- nothing to register.
}'''
    
    code = '''  auto &engine = nntrainer::Engine::Global();

  auto app_context = static_cast<nntrainer::AppContext *>(
      engine.getRegisteredContext("cpu"));

  try {
'''
    code += "\n".join(f"    {r}" for r in registrations)
    code += '''
  } catch (const std::invalid_argument &error) {
    std::cerr << "Failed to register custom layer: " << error.what() << std::endl;
  }
}'''
    
    return code
