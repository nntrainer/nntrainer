"""Attention pattern detection."""

from nntrainer_layers import LAYER_FC, LAYER_RMS_NORM, LAYER_LAYER_NORM, OP_SDPA
from .data_types import AttentionPattern


def detect_attention(block_idx, attn_scope, block_layers, all_layers, config):
    """Detect attention pattern within the given scope.

    Args:
        block_idx: Block number (0-based)
        attn_scope: HF module scope for attention
        block_layers: layers in this block
        all_layers: all layers (for RoPE detection)
        config: HF model config (optional)

    Returns:
        AttentionPattern
    """
    attn = AttentionPattern(block_idx=block_idx)

    attn_layers = [l for l in block_layers
                   if l.hf_module_name.startswith(attn_scope)]

    for layer in attn_layers:
        name_suffix = layer.hf_module_name[len(attn_scope):].lstrip(".")

        if layer.layer_type == LAYER_FC:
            _match_fc_projection(attn, layer, name_suffix, config)
        elif layer.layer_type in (LAYER_RMS_NORM, LAYER_LAYER_NORM):
            _match_norm(attn, layer, name_suffix)
        elif layer.layer_type == OP_SDPA:
            attn.sdpa = layer.name
            attn.layer_names.append(layer.name)

    # O projection fallback (BERT: attention.output.dense)
    if not attn.o_proj:
        _find_o_proj_fallback(attn, attn_scope, block_layers)

    # RoPE detection
    _detect_rope(attn, attn_scope, all_layers, config)

    # Attention type classification
    _classify_attention_type(attn, config)

    return attn


def _match_fc_projection(attn, layer, name_suffix, config):
    """Match FC layer to Q/K/V/O projection."""
    if name_suffix in ("q_proj", "q", "query", "q_proj.0"):
        attn.q_proj = layer.name
        attn.layer_names.append(layer.name)
        unit = int(layer.properties.get("unit", 0))
        if unit and config:
            attn.num_heads = getattr(config, "num_attention_heads", 0)
            attn.head_dim = getattr(config, "head_dim",
                                    unit // attn.num_heads
                                    if attn.num_heads else 0)
    elif name_suffix in ("k_proj", "k", "key", "k_proj.0"):
        attn.k_proj = layer.name
        attn.layer_names.append(layer.name)
        unit = int(layer.properties.get("unit", 0))
        if unit and attn.head_dim:
            attn.num_kv_heads = unit // attn.head_dim
    elif name_suffix in ("v_proj", "v", "value", "v_proj.0"):
        attn.v_proj = layer.name
        attn.layer_names.append(layer.name)
    elif name_suffix in ("o_proj", "out_proj", "o"):
        attn.o_proj = layer.name
        attn.layer_names.append(layer.name)


def _match_norm(attn, layer, name_suffix):
    """Match norm layer to Q/K norm."""
    if name_suffix in ("q_norm", "q_layernorm"):
        attn.q_norm = layer.name
        attn.has_qk_norm = True
        attn.layer_names.append(layer.name)
    elif name_suffix in ("k_norm", "k_layernorm"):
        attn.k_norm = layer.name
        attn.layer_names.append(layer.name)


def _find_o_proj_fallback(attn, attn_scope, block_layers):
    """Find O projection outside self-attention scope (BERT-style)."""
    parent_scope = attn_scope.rsplit(".", 1)[0] if "." in attn_scope else ""
    for layer in block_layers:
        if layer.layer_type != LAYER_FC:
            continue
        hf = layer.hf_module_name
        if parent_scope and hf.startswith(parent_scope):
            suffix = hf[len(parent_scope):].lstrip(".")
            if suffix in ("output.dense", "dense"):
                attn.o_proj = layer.name
                attn.layer_names.append(layer.name)
                break


def _detect_rope(attn, attn_scope, all_layers, config):
    """Detect RoPE from sin/cos ops or config."""
    for layer in all_layers:
        if layer.layer_type in ("sin", "cos"):
            name_lower = layer.name.lower()
            scope_lower = layer.hf_module_name.lower()
            if ("rotary" in name_lower or "rotary" in scope_lower
                or "rope" in name_lower or "rope" in scope_lower
                or scope_lower.startswith(attn_scope)):
                attn.has_rope = True
                break
    if not attn.has_rope and config:
        # Try top-level rope_theta first, then nested rope_parameters dict
        # (transformers >=5.x stores rope_theta inside rope_parameters)
        rope_theta = getattr(config, "rope_theta", 0)
        if not rope_theta:
            rope_params = getattr(config, "rope_parameters", None)
            if isinstance(rope_params, dict):
                rope_theta = rope_params.get("rope_theta", 0)
        if rope_theta and rope_theta > 0:
            attn.has_rope = True


def _classify_attention_type(attn, config):
    """Classify as MHA, GQA, or MQA."""
    if attn.num_heads and attn.num_kv_heads:
        if attn.num_kv_heads == 1:
            attn.attention_type = "mqa"
        elif attn.num_kv_heads < attn.num_heads:
            attn.attention_type = "gqa"
        else:
            attn.attention_type = "mha"

    if not attn.num_kv_heads and config:
        attn.num_kv_heads = getattr(config, "num_key_value_heads",
                                    attn.num_heads)
