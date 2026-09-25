"""
Gemma3Adapter -- text-only Gemma3 (`Gemma3ForCausalLM` / `Gemma3TextModel`).

Structurally a different decoder shape from the Llama family
(api/adapters/llama_family.py), not a variant of it:

  * Four norms per layer, not two -- Gemma3DecoderLayer sandwiches a
    *second* norm around both the attention and the MLP branch
    (post_attention_layernorm, post_feedforward_layernorm), applied to
    the branch's own output *before* it's added back to the residual,
    on top of the usual pre-norm (input_layernorm,
    pre_feedforward_layernorm). See DecoderLayerIR.attn_post_norm/
    mlp_post_norm.
  * Attention alternates per layer between "sliding_attention" (a
    local window, rope_theta ~= 1e4) and "full_attention" (no window,
    rope_theta ~= 1e6), per `config.layer_types` -- read directly off
    each layer's own `self_attn.sliding_window`/`is_sliding`, not
    re-derived from the pattern, since the live module already computed
    it the same way nntrainer's own hand-written Gemma3Transformer does
    (Applications/CausalLM/models/gemma3/gemma3_causallm.cpp).
  * Q/K RMSNorm (like Qwen3), and GeGLU-tanh MLP (gate/up/down with
    HF's "gelu_pytorch_tanh", not SwiGLU's silu).
  * Scaled input embeddings (Gemma3TextScaledWordEmbedding multiplies
    by sqrt(hidden_size)) -- CausalLMIR.embedding_scale.

Because of the per-layer attention heterogeneity, this model's decoder
layers are NOT structurally uniform (CausalLMIR.uniform_layer_signature()
returns None) -- the generic single-template nntrainer C++ emitter
(converters/cpp_generator.py) can't represent that, so C++ generation
for this architecture falls back to reusing the real hand-written
Gemma3Transformer/Gemma3CausalLM implementation (see
agents/model_code_reuse.py) rather than deriving one from this IR. This
adapter still matters on its own: it's what makes the compatibility
report and the graph visualizer describe Gemma3 as a supported
architecture with a handful of named semantic features, instead of
falling back to raw module-tree op counting (which sees the fused
building blocks -- Gemma3RMSNorm, Gemma3TextScaledWordEmbedding,
Gemma3RotaryEmbedding, tanh-GELU -- as unknown ops nntrainer has no
1:1 layer for).
"""
from __future__ import annotations

from typing import Any

from api.adapters.base import ArchitectureAdapter
from api.adapters.llama_family import _linear_projection, _rms_norm
from api.semantic.model import (
    AttentionIR, CausalLMIR, DecoderLayerIR, MLPIR,
)

#: Gemma3's own defaults (Gemma3TextConfig.default_theta) for the two
#: rope bases when a config predates the rope_parameters dict format.
_DEFAULT_GLOBAL_ROPE_THETA = 1_000_000.0
_DEFAULT_LOCAL_ROPE_THETA = 10_000.0


def _layer_rope_theta(config: Any, layer_type: str) -> float:
    """Reads config.rope_parameters[layer_type]["rope_theta"] (current
    transformers versions); falls back to the old flat
    rope_theta/rope_local_base_freq attributes for configs/transformers
    versions that predate the rope_parameters dict."""
    rope_parameters = getattr(config, "rope_parameters", None) or {}
    params = rope_parameters.get(layer_type) if isinstance(rope_parameters, dict) else None
    if params and params.get("rope_theta") is not None:
        return float(params["rope_theta"])
    if layer_type == "sliding_attention":
        return float(getattr(config, "rope_local_base_freq", _DEFAULT_LOCAL_ROPE_THETA))
    return float(getattr(config, "rope_theta", _DEFAULT_GLOBAL_ROPE_THETA))


class Gemma3Adapter(ArchitectureAdapter):
    # "gemma3_text" is the real `config.model_type` for a text-only
    # checkpoint (google/gemma-3-270m, Gemma3TextConfig) -- "gemma3" is
    # the multimodal wrapper's model_type (Gemma3Config, text_config +
    # vision_config) and would only reach here if a caller handed us the
    # text sub-config directly under that name. Both map onto the same
    # decoder shape.
    model_types: tuple[str, ...] = ("gemma3_text", "gemma3")

    def build_semantic_ir(self, config: Any, model: Any) -> CausalLMIR:
        hidden_size = int(config.hidden_size)
        num_layers = int(getattr(config, "num_hidden_layers", len(model.layers)))
        eps = float(getattr(config, "rms_norm_eps", 1e-6))

        layers = [
            self._build_decoder_layer(index, decoder, config, hidden_size, eps)
            for index, decoder in enumerate(model.layers)
        ]

        return CausalLMIR(
            # Normalized to the short slug nntrainer_names.py's
            # TRANSFORMER_BASE_CLASSES/CAUSAL_LM_BASE_CLASSES key on
            # ("gemma3") -- config.model_type is "gemma3_text" for a
            # real text checkpoint and would silently miss that lookup.
            architecture="gemma3",
            hidden_size=hidden_size,
            vocab_size=int(getattr(config, "vocab_size", 0)),
            num_layers=num_layers,
            embedding_name="model.embed_tokens",
            decoder_layers=layers,
            final_norm=_rms_norm("model.norm", model.norm, hidden_size, eps),
            lm_head_name="lm_head",
            tied_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
            embedding_scale=float(hidden_size) ** 0.5,
        )

    # ------------------------------------------------------------------ layer
    def _build_decoder_layer(self, index, decoder, config, hidden_size, eps) -> DecoderLayerIR:
        prefix = f"model.layers.{index}"
        return DecoderLayerIR(
            index=index,
            input_norm=_rms_norm(f"{prefix}.input_layernorm", decoder.input_layernorm, hidden_size, eps),
            attention=self._build_attention(index, decoder.self_attn, config, hidden_size, eps),
            attn_post_norm=_rms_norm(
                f"{prefix}.post_attention_layernorm", decoder.post_attention_layernorm, hidden_size, eps,
            ),
            post_attention_norm=_rms_norm(
                f"{prefix}.pre_feedforward_layernorm", decoder.pre_feedforward_layernorm, hidden_size, eps,
            ),
            mlp=self._build_mlp(index, decoder.mlp, config),
            mlp_post_norm=_rms_norm(
                f"{prefix}.post_feedforward_layernorm", decoder.post_feedforward_layernorm, hidden_size, eps,
            ),
        )

    # -------------------------------------------------------------- attention
    def _build_attention(self, index, attn, config, hidden_size, eps) -> AttentionIR:
        prefix = f"model.layers.{index}.self_attn"
        num_heads = int(getattr(config, "num_attention_heads"))
        num_kv_heads = int(getattr(config, "num_key_value_heads", num_heads))
        head_dim = int(getattr(config, "head_dim", hidden_size // num_heads))

        # Read the already-resolved per-layer window/type straight off
        # the live module (Gemma3Attention.__init__ computes both from
        # config.layer_types[index] the same way nntrainer's hand-written
        # Gemma3Transformer does) rather than re-deriving the
        # sliding_window_pattern arithmetic a second time here.
        is_sliding = bool(getattr(attn, "is_sliding", False))
        layer_type = "sliding_attention" if is_sliding else "full_attention"
        sliding_window = getattr(attn, "sliding_window", None) if is_sliding else None

        q_norm = _rms_norm(f"{prefix}.q_norm", attn.q_norm, head_dim, eps, reshaped=True)
        k_norm = _rms_norm(f"{prefix}.k_norm", attn.k_norm, head_dim, eps, reshaped=True)

        return AttentionIR(
            source_name=prefix,
            q_proj=_linear_projection(f"{prefix}.q_proj", attn.q_proj),
            k_proj=_linear_projection(f"{prefix}.k_proj", attn.k_proj),
            v_proj=_linear_projection(f"{prefix}.v_proj", attn.v_proj),
            o_proj=_linear_projection(f"{prefix}.o_proj", attn.o_proj),
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_norm=q_norm,
            k_norm=k_norm,
            rope_theta=_layer_rope_theta(config, layer_type),
            max_position_embeddings=getattr(config, "max_position_embeddings", None),
            sliding_window=sliding_window,
            causal=True,
            use_kv_cache=True,
        )

    # -------------------------------------------------------------------- mlp
    def _build_mlp(self, index, mlp, config) -> MLPIR:
        prefix = f"model.layers.{index}.mlp"
        activation = str(getattr(config, "hidden_activation", "gelu_pytorch_tanh"))
        return MLPIR(
            source_name=prefix,
            up_proj=_linear_projection(f"{prefix}.up_proj", mlp.up_proj),
            down_proj=_linear_projection(f"{prefix}.down_proj", mlp.down_proj),
            activation=activation,
            gated=True,
            gate_proj=_linear_projection(f"{prefix}.gate_proj", mlp.gate_proj),
        )
