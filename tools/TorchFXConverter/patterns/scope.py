"""Block, attention, and FFN scope discovery within the model."""

import re


def find_block_scopes(layers):
    """Find all transformer block scopes from layer names.

    Returns sorted list of block scope strings.
    """
    block_scopes = set()
    block_patterns = [
        r"(model\.layers\.\d+)",
        r"(layers\.\d+)",
        r"((?:bert\.)?encoder\.layer\.\d+)",
        r"((?:encoder|decoder)\.block\.\d+)",
        r"((?:transformer\.)?h\.\d+)",
        # Diffusion Transformer (DiT) patterns: FLUX, etc.
        r"(transformer_blocks\.\d+)",
        r"(single_transformer_blocks\.\d+)",
        # Conformer / Zipformer (speech/audio) patterns
        r"(conformer_layers\.\d+)",
        r"(encoders\.\d+\.(?:encoder\.)?layers\.\d+)",
    ]

    for layer in layers:
        name = layer.hf_module_name
        if not name:
            continue
        for pattern in block_patterns:
            m = re.match(pattern, name)
            if m:
                block_scopes.add(m.group(1))
                break

    def _sort_key(scope):
        parts = scope.split(".")
        key = []
        for p in parts:
            if p.isdigit():
                key.append((0, int(p)))
            else:
                key.append((1, p))
        return key

    return sorted(block_scopes, key=_sort_key)


def get_layers_in_scope(layers, scope):
    """Get all layers whose hf_module_name starts with scope."""
    sanitized = scope.replace(".", "_")
    result = []
    for l in layers:
        if (l.hf_module_name.startswith(scope + ".")
            or l.hf_module_name == scope
            or (not l.hf_module_name and l.name.startswith(sanitized + "_"))):
            result.append(l)
    return result


def find_attention_scope(block_scope, block_layers):
    """Find the self-attention scope within a block."""
    attn_keywords = [
        "self_attn", "attention.self", "SelfAttention",
        "self_attention", "attn",
    ]

    for layer in block_layers:
        name = layer.hf_module_name
        if not name.startswith(block_scope):
            continue
        remainder = name[len(block_scope):]
        for kw in attn_keywords:
            idx = remainder.find(kw)
            if idx >= 0:
                return block_scope + remainder[:idx + len(kw)]

    # Generic fallback
    for layer in block_layers:
        name = layer.hf_module_name
        if not name.startswith(block_scope):
            continue
        remainder = name[len(block_scope):]
        idx = remainder.find("attention")
        if idx >= 0:
            return block_scope + remainder[:idx + len("attention")]

    return None


def find_cross_attention_scope(block_scope, block_layers):
    """Find cross-attention scope (T5/mT5 encoder-decoder)."""
    cross_keywords = ["EncDecAttention", "crossattention",
                      "cross_attn", "encoder_attn"]
    for layer in block_layers:
        name = layer.hf_module_name
        if not name.startswith(block_scope):
            continue
        for kw in cross_keywords:
            if kw in name:
                idx = name.find(kw)
                return name[:idx + len(kw)]
    return None


def find_ssm_scope(block_scope, block_layers):
    """Find the SSM/Mamba mixer scope within a block.

    HuggingFace Mamba uses 'mixer' as the sub-module name:
        model.layers.0.mixer.in_proj
        model.layers.0.mixer.conv1d
        model.layers.0.mixer.x_proj
        model.layers.0.mixer.dt_proj
        model.layers.0.mixer.out_proj
    """
    ssm_keywords = ["mixer", "mamba", "ssm"]
    for kw in ssm_keywords:
        full = f"{block_scope}.{kw}"
        for layer in block_layers:
            name = layer.hf_module_name
            if name.startswith(full + ".") or name == full:
                return full
    return None


def find_ffn_scope(block_scope, block_layers):
    """Find the FFN/MLP scope within a block."""
    ffn_patterns = ["mlp", "shared_mlp", "feed_forward", "ffn", "ff",
                    "DenseReluDense"]
    for pat in ffn_patterns:
        full = f"{block_scope}.{pat}"
        for layer in block_layers:
            name = layer.hf_module_name
            if name.startswith(full + ".") or name == full:
                return full

    # T5-style
    for layer in block_layers:
        name = layer.hf_module_name
        if "DenseReluDense" in name:
            idx = name.find("DenseReluDense")
            return name[:idx + len("DenseReluDense")]

    # BERT-style
    has_intermediate = any(
        l.hf_module_name.startswith(f"{block_scope}.intermediate")
        for l in block_layers)
    if has_intermediate:
        return block_scope

    return None
