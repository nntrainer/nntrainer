"""
LlamaFamilyAdapter -- one adapter, several `model_type`s.

Llama, Mistral, Qwen2 and Qwen3 all share the same decoder shape
(RMSNorm -> attention -> residual -> RMSNorm -> gated MLP -> residual).
The only real differences are: whether Q/K get their own RMSNorm
(Qwen3), and whether attention uses a sliding window (Mistral). Both
are detected at runtime from the actual module tree / config rather
than hard-coded per subclass, so a new checkpoint that happens to
enable/disable one of these doesn't need a new adapter -- only a
genuinely different decoder shape does.

Subclasses exist so `select_adapter()` can report *which* architecture
matched (useful in logs and in the compatibility report), and so a
family member with a real structural difference can override the
relevant `_build_*` method without duplicating the rest.
"""
from __future__ import annotations

from typing import Any, Optional

from api.adapters.base import ArchitectureAdapter
from api.semantic.model import (
    AttentionIR, CausalLMIR, DecoderLayerIR, MLPIR, NormIR, ProjectionIR,
)


def _linear_projection(source_name: str, module) -> ProjectionIR:
    out_features = getattr(module, "out_features", None)
    in_features = getattr(module, "in_features", None)
    weight = getattr(module, "weight", None)
    if out_features is None and weight is not None:
        out_features = int(weight.shape[0])
    if in_features is None and weight is not None:
        in_features = int(weight.shape[1])
    return ProjectionIR(
        source_name=source_name,
        output_size=int(out_features) if out_features is not None else 0,
        input_size=int(in_features) if in_features is not None else None,
        bias=getattr(module, "bias", None) is not None,
        weight_name=f"{source_name}.weight",
    )


def _rms_norm(source_name: str, module, feature_size: int, default_eps: float, *, reshaped: bool = False) -> NormIR:
    return NormIR(
        source_name=source_name,
        norm_type="rms_norm",
        feature_size=int(feature_size),
        epsilon=float(getattr(module, "variance_epsilon", getattr(module, "eps", default_eps))),
        reshaped=reshaped,
    )


class LlamaFamilyAdapter(ArchitectureAdapter):
    model_types: tuple[str, ...] = ()

    def build_semantic_ir(self, config: Any, model: Any) -> CausalLMIR:
        hidden_size = int(config.hidden_size)
        num_layers = int(getattr(config, "num_hidden_layers", len(model.layers)))
        eps = float(getattr(config, "rms_norm_eps", 1e-6))

        layers = []
        for index, decoder in enumerate(model.layers):
            layers.append(self._build_decoder_layer(index, decoder, config, hidden_size, eps))

        return CausalLMIR(
            architecture=getattr(config, "model_type", self.model_types[0] if self.model_types else "llama"),
            hidden_size=hidden_size,
            vocab_size=int(getattr(config, "vocab_size", 0)),
            num_layers=num_layers,
            embedding_name="model.embed_tokens",
            decoder_layers=layers,
            final_norm=_rms_norm("model.norm", model.norm, hidden_size, eps),
            lm_head_name="lm_head",
            tied_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )

    # ------------------------------------------------------------------ layer
    def _build_decoder_layer(self, index, decoder, config, hidden_size, eps) -> DecoderLayerIR:
        prefix = f"model.layers.{index}"
        return DecoderLayerIR(
            index=index,
            input_norm=_rms_norm(
                f"{prefix}.input_layernorm", decoder.input_layernorm, hidden_size, eps,
            ),
            attention=self._build_attention(index, decoder.self_attn, config, hidden_size, eps),
            post_attention_norm=_rms_norm(
                f"{prefix}.post_attention_layernorm",
                decoder.post_attention_layernorm,
                hidden_size,
                eps,
            ),
            mlp=self._build_mlp(index, decoder.mlp, config),
        )

    # -------------------------------------------------------------- attention
    def _build_attention(self, index, attn, config, hidden_size, eps) -> AttentionIR:
        prefix = f"model.layers.{index}.self_attn"
        num_heads = int(getattr(config, "num_attention_heads"))
        num_kv_heads = int(getattr(config, "num_key_value_heads", num_heads))
        head_dim = int(getattr(config, "head_dim", hidden_size // num_heads))

        q_norm = k_norm = None
        # Detected at runtime (Qwen3-style per-head Q/K RMSNorm), not
        # hard-coded to a subclass -- see module docstring. reshaped=True
        # because this norm operates per-head (feature_size=head_dim, on
        # the reshaped [.., num_heads, head_dim] tensor) rather than over
        # the full hidden_size like the decoder-level input/post norms --
        # that distinction is exactly what tells the lowerer to emit
        # nntrainer's "reshaped_rms_norm" layer instead of plain "rms_norm".
        if getattr(attn, "q_norm", None) is not None:
            q_norm = _rms_norm(f"{prefix}.q_norm", attn.q_norm, head_dim, eps, reshaped=True)
        if getattr(attn, "k_norm", None) is not None:
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
            rope_theta=float(getattr(config, "rope_theta", 10000.0)),
            max_position_embeddings=getattr(config, "max_position_embeddings", None),
            sliding_window=getattr(config, "sliding_window", None),
            causal=True,
            use_kv_cache=True,
        )

    # -------------------------------------------------------------------- mlp
    def _build_mlp(self, index, mlp, config) -> MLPIR:
        prefix = f"model.layers.{index}.mlp"
        gate_proj = getattr(mlp, "gate_proj", None)
        activation = str(getattr(config, "hidden_act", "silu"))
        return MLPIR(
            source_name=prefix,
            up_proj=_linear_projection(f"{prefix}.up_proj", mlp.up_proj),
            down_proj=_linear_projection(f"{prefix}.down_proj", mlp.down_proj),
            activation=activation,
            gated=gate_proj is not None,
            gate_proj=_linear_projection(f"{prefix}.gate_proj", gate_proj) if gate_proj is not None else None,
        )
