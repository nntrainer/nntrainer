"""Block-level pattern detection: norms, residuals, and operator type."""

from nntrainer_layers import (
    LAYER_RMS_NORM, LAYER_LAYER_NORM, LAYER_ADDITION, LAYER_ADD,
)
from .data_types import TransformerBlockPattern
from .scope import (
    get_layers_in_scope, find_attention_scope,
    find_cross_attention_scope, find_ffn_scope, find_ssm_scope,
)
from .attention import detect_attention
from .ffn import detect_ffn
from .ssm import detect_ssm


def detect_block(block_idx, scope, layers, all_layers, config, by_name,
                 idx_by_name):
    """Detect a single transformer block's internal structure.

    Args:
        block_idx: Block number (0-based)
        scope: HF module scope (e.g. "model.layers.0")
        layers: all layers list
        all_layers: same as layers (for attention RoPE scan)
        config: HF model config
        by_name: dict name -> layer
        idx_by_name: dict name -> index

    Returns:
        TransformerBlockPattern or None
    """
    block_layers = get_layers_in_scope(layers, scope)
    if not block_layers:
        return None

    block = TransformerBlockPattern(block_idx=block_idx)

    # Detect attention
    attn_scope = find_attention_scope(scope, block_layers)
    if attn_scope:
        block.attention = detect_attention(
            block_idx, attn_scope, block_layers, all_layers, config)

    # Detect cross-attention
    cross_attn_scope = find_cross_attention_scope(scope, block_layers)
    if cross_attn_scope:
        block.cross_attention = detect_attention(
            block_idx, cross_attn_scope, block_layers, all_layers, config)

    # Detect SSM (Mamba mixer)
    ssm_scope = find_ssm_scope(scope, block_layers)
    if ssm_scope:
        block.ssm = detect_ssm(block_idx, ssm_scope, block_layers, config)

    # Detect FFN
    ffn_scope = find_ffn_scope(scope, block_layers)
    if ffn_scope:
        block.ffn = detect_ffn(block_idx, ffn_scope, block_layers, by_name)

    # Detect norms and residuals
    detect_norms_and_residuals(block, scope, block_layers, idx_by_name)

    # Auto-detect operator type
    detect_operator(block, scope, block_layers, layers)

    return block


def detect_norms_and_residuals(block, scope, block_layers, idx_by_name):
    """Detect normalization layers and residual connections."""
    norms = []
    residuals = []

    for layer in block_layers:
        if layer.layer_type in (LAYER_RMS_NORM, LAYER_LAYER_NORM):
            if block.attention and layer.name in (
                block.attention.q_norm, block.attention.k_norm):
                continue
            norms.append(layer)
        elif layer.layer_type in (LAYER_ADDITION, LAYER_ADD):
            residuals.append(layer)

    # Assign norms by name suffix
    norm_names_suffixes = []
    for n in norms:
        suffix = n.hf_module_name[len(scope):].lstrip(".")
        norm_names_suffixes.append((n, suffix))

    for norm, suffix in norm_names_suffixes:
        if any(kw in suffix for kw in ("input_layernorm",
                                       "layer.0.layer_norm",
                                       "pre_attention")):
            block.pre_attn_norm = norm.name
        elif "post_attention_layernorm" in suffix:
            block.post_attn_norm = norm.name
        elif "pre_feedforward_layernorm" in suffix:
            block.pre_ffn_norm = norm.name
        elif "post_feedforward_layernorm" in suffix:
            block.post_ffn_norm = norm.name
        elif any(kw in suffix for kw in ("pre_ffn", "final_layer_norm")):
            block.pre_ffn_norm = norm.name
        elif "layer.1.layer_norm" in suffix:
            if block.cross_attention:
                block.cross_attn_norm = norm.name
            else:
                block.pre_ffn_norm = norm.name
        elif "layer.2.layer_norm" in suffix:
            block.pre_ffn_norm = norm.name
        elif any(kw in suffix for kw in ("attention.output.LayerNorm",)):
            block.pre_attn_norm = norm.name
        elif any(kw in suffix for kw in ("output.LayerNorm",)):
            block.pre_ffn_norm = norm.name

    # post_attn_norm -> pre_ffn_norm promotion
    if block.post_attn_norm and not block.pre_ffn_norm:
        block.pre_ffn_norm = block.post_attn_norm
        block.post_attn_norm = ""

    # Positional fallback
    if not block.pre_attn_norm and not block.pre_ffn_norm and len(norms) >= 2:
        block.pre_attn_norm = norms[0].name
        block.pre_ffn_norm = norms[1].name
    elif not block.pre_attn_norm and len(norms) >= 1:
        block.pre_attn_norm = norms[0].name

    # Assign residuals
    if len(residuals) >= 2:
        block.attn_residual = residuals[0].name
        block.ffn_residual = residuals[-1].name
        if len(residuals) >= 3 and block.cross_attention:
            block.cross_attn_residual = residuals[1].name
    elif len(residuals) == 1:
        block.attn_residual = residuals[0].name

    # Norm type (pre vs post)
    if block.pre_attn_norm and block.attention and block.attention.q_proj:
        norm_idx = idx_by_name.get(block.pre_attn_norm, 0)
        q_idx = idx_by_name.get(block.attention.q_proj, 0)
        if norm_idx < q_idx:
            block.norm_type = "pre_norm"
        else:
            block.norm_type = "post_norm"
    elif block.pre_attn_norm and block.ssm and block.ssm.in_proj:
        # Mamba: norm before mixer is always pre_norm
        norm_idx = idx_by_name.get(block.pre_attn_norm, 0)
        proj_idx = idx_by_name.get(block.ssm.in_proj, 0)
        block.norm_type = "pre_norm" if norm_idx < proj_idx else "post_norm"


def detect_operator(block, scope, block_layers, all_layers):
    """Auto-detect the operator type and its learnable layers."""
    if block.attention:
        block.operator_type = "attention"
        block.operator_scope = find_attention_scope(
            scope, block_layers) or ""
        return

    if block.ssm:
        block.operator_type = "mixer"
        block.operator_scope = find_ssm_scope(scope, block_layers) or ""
        block.operator_layers = [
            l for l in block_layers
            if l.name in block.ssm.layer_names]
        return

    # Build set of known layer names
    known = set()
    for attr in ("pre_attn_norm", "post_attn_norm", "pre_ffn_norm",
                 "post_ffn_norm", "attn_residual", "ffn_residual"):
        val = getattr(block, attr, "")
        if val:
            known.add(val)
    if block.ffn:
        known.update(block.ffn.layer_names)

    operator_layers = [l for l in block_layers if l.name not in known]

    # Filter to learnable layers
    seen_hf = set()
    learnable = []
    for l in operator_layers:
        hf = l.hf_module_name
        if not hf:
            continue
        suffix = hf[len(scope):].lstrip(".")
        if "." not in suffix:
            continue
        if hf in seen_hf:
            continue
        parts = suffix.split(".")
        if len(parts) < 2:
            continue
        if l.layer_type in ("fully_connected", "conv1d", "conv2d",
                            "rms_norm", "layer_normalization",
                            "embedding"):
            seen_hf.add(hf)
            learnable.append(l)

    block.operator_layers = learnable

    # Detect operator type from sub-scope keyword
    op_scopes = set()
    for l in learnable:
        suffix = l.hf_module_name[len(scope):].lstrip(".")
        parts = suffix.split(".")
        if parts:
            op_scopes.add(parts[0])

    if len(op_scopes) == 1:
        block.operator_type = op_scopes.pop()
        block.operator_scope = scope + "." + block.operator_type
    elif op_scopes:
        from collections import Counter
        counts = Counter()
        for l in learnable:
            suffix = l.hf_module_name[len(scope):].lstrip(".")
            counts[suffix.split(".")[0]] += 1
        block.operator_type = counts.most_common(1)[0][0]
        block.operator_scope = scope + "." + block.operator_type
    else:
        block.operator_type = "unknown"
        block.operator_scope = scope
