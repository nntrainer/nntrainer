"""C++ createAttention() method generation using symbolic Tensor graph."""

from .helpers import _cpp_tensor_layer, _class_name


def emit_attention_method(cname, block, arch_type="decoder_only",
                          external_kv_cache=False):
    """Generate createAttention() method body using Tensor flow."""
    attn = block.attention
    is_decoder = arch_type in ("decoder_only", "encoder_decoder")
    has_qk_norm = attn.has_qk_norm
    has_rope = attn.has_rope
    L = []

    L.append(f"Tensor")
    L.append(f"{cname}::createAttention(const int layer_id, int seq_len, "
             f"int n_heads,")
    L.append(f"                         int head_dim, "
             f"Tensor query,")
    L.append(f"                         Tensor key, "
             f"Tensor value) {{")
    L.append(f"")
    L.append(f"  using ml::train::createLayer;")
    L.append(f"")
    L.append(f'  auto V_name = "layer" + std::to_string(layer_id) + "_wv";')
    L.append(f'  auto K_name = "layer" + std::to_string(layer_id) + "_wk";')
    L.append(f'  auto Q_name = "layer" + std::to_string(layer_id) + "_wq";')
    if has_qk_norm:
        L.append(f'  auto K_norm_name = "layer" + std::to_string(layer_id) '
                 f'+ "_k_norm";')
        L.append(f'  auto Q_norm_name = "layer" + std::to_string(layer_id) '
                 f'+ "_q_norm";')
    L.append(f'  auto A_name = "layer" + std::to_string(layer_id) '
             f'+ "_attention";')
    L.append(f'  auto O_name = "layer" + std::to_string(layer_id) '
             f'+ "_attention_out";')
    L.append(f"")

    # V projection
    L.append(f"  // V projection")
    lines, v_out = _cpp_tensor_layer("v_proj", "fully_connected", [
        'withKey("name", V_name)',
        'withKey("unit", head_dim * n_heads / GQA_SIZE)',
        'withKey("disable_bias", "true")',
    ], "value")
    L.extend(lines)
    L.append(f"")

    # K projection
    L.append(f"  // K projection")
    lines, k_out = _cpp_tensor_layer("k_proj", "fully_connected", [
        'withKey("name", K_name)',
        'withKey("unit", head_dim * n_heads / GQA_SIZE)',
        'withKey("disable_bias", "true")',
    ], "key")
    L.extend(lines)
    L.append(f"")

    # Q projection
    L.append(f"  // Q projection")
    lines, q_out = _cpp_tensor_layer("q_proj", "fully_connected", [
        'withKey("name", Q_name)',
        'withKey("unit", head_dim * n_heads)',
        'withKey("disable_bias", "true")',
    ], "query")
    L.extend(lines)

    # Q/K norms (Qwen3-style)
    q_in = q_out
    k_in = k_out
    if has_qk_norm:
        L.append(f"")
        L.append(f"  // K norm (reshaped RMS norm)")
        lines, k_in = _cpp_tensor_layer("k_norm", "reshaped_rms_norm", [
            'withKey("name", K_norm_name)',
            'withKey("packed", "false")',
            'withKey("epsilon", NORM_EPS)',
            'withKey("feature_size", head_dim)',
        ], k_out)
        L.extend(lines)

        L.append(f"")
        L.append(f"  // Q norm (reshaped RMS norm)")
        lines, q_in = _cpp_tensor_layer("q_norm", "reshaped_rms_norm", [
            'withKey("name", Q_norm_name)',
            'withKey("packed", "false")',
            'withKey("epsilon", NORM_EPS)',
            'withKey("feature_size", head_dim)',
        ], q_out)
        L.extend(lines)

    # MHA core
    L.append(f"")
    L.append(f"  // Attention core layer")
    mha_props = [
        'withKey("name", A_name)',
        'withKey("num_heads", n_heads)',
        'withKey("num_heads_kv", n_heads / GQA_SIZE)',
    ]
    if is_decoder:
        mha_props.append(
            'withKey("max_timestep", std::to_string(INIT_SEQ_LEN + '
            'NUM_TO_GENERATE))')
        mha_props.append('withKey("sliding_window", SLIDING_WINDOW)')
    else:
        mha_props.append(
            'withKey("max_timestep", std::to_string(INIT_SEQ_LEN))')
    if has_rope:
        mha_props.append('withKey("rope_theta", ROPE_THETA)')
        mha_props.append(
            'withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS)')
    if is_decoder:
        mha_props.append('withKey("max_new_tokens", NUM_TO_GENERATE)')

    if external_kv_cache:
        mha_input = (f'{{{q_in}, {k_in}, {v_out}, '
                     f'key_cache_tensors[layer_id], '
                     f'val_cache_tensors[layer_id]}}')
    else:
        mha_input = f'{{{q_in}, {k_in}, {v_out}}}'

    lines, a_out = _cpp_tensor_layer("attn", "mha_core", mha_props, mha_input)
    L.extend(lines)

    # O projection
    L.append(f"")
    L.append(f"  // O projection")
    lines, o_out = _cpp_tensor_layer("o_proj", "fully_connected", [
        'withKey("name", O_name)',
        'withKey("unit", DIM)',
        'withKey("disable_bias", "true")',
    ], a_out)
    L.extend(lines)

    L.append(f"")
    L.append(f"  return {o_out};")
    L.append(f"}}")
    L.append(f"")
    return "\n".join(L)
