"""
Semantic-level nntrainer capabilities, separate from the primitive
op_table (api.compatibility.op_table).

op_table answers "does nntrainer have a layer for this exact torch
module class". This answers a different question: "does nntrainer's
mha_core / reshaped_rms_norm / gated-MLP lowering path support this
architectural *feature*". A model can be 100% supported here despite
its raw module tree containing classes op_table has never heard of
(Qwen3RotaryEmbedding, Qwen3RMSNorm, ...) -- those get fused into
mha_core / reshaped_rms_norm by the lowering stage rather than mapped
1:1, so judging them against op_table would report a false gap.
"""
from api.semantic.model import CausalLMIR

NNTRAINER_SEMANTIC_CAPABILITIES = {
    "attention": {
        "mha": True,
        "gqa": True,
        "rope": True,
        "kv_cache": True,
        "sliding_window": True,
        "qk_norm": True,
    },
    "normalization": {
        "rms_norm": True,
        "reshaped_rms_norm": True,
        "layer_norm": True,
    },
    "mlp": {
        "swiglu": True,
        "gelu": True,
        "gated": True,
    },
}


def describe(model_ir: CausalLMIR) -> dict:
    """Human-readable per-feature support report for the compatibility
    panel, e.g. {"attention": {"gqa": "supported", "qk_norm": "custom
    registered layer", "kv_cache": "host managed"}, ...}."""
    if not model_ir.decoder_layers:
        return {}

    layer = model_ir.decoder_layers[0]
    attn = layer.attention
    mlp = layer.mlp

    report = {
        "attention": {
            "multi_head_attention": "supported by mha_core",
            "grouped_query_attention": (
                "supported (num_heads_kv != num_heads)" if attn.num_kv_heads != attn.num_heads
                else "not used by this model"
            ),
            "rope": "supported by mha_core" if attn.rope_theta else "not used by this model",
            "kv_cache": "host managed",
            "sliding_window": (
                f"supported (window={attn.sliding_window})" if attn.sliding_window else "not used by this model"
            ),
            "qk_norm": (
                "custom registered layer (reshaped_rms_norm)"
                if (attn.q_norm or attn.k_norm) else "not used by this model"
            ),
        },
        "normalization": {
            "input_norm": layer.input_norm.norm_type,
            "post_attention_norm": layer.post_attention_norm.norm_type,
        },
        "mlp": {
            "gated": "supported (SwiGLU-style gate * up)" if mlp.gated else "not gated",
            "activation": mlp.activation,
        },
    }
    return report
