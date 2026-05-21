"""C++ createTransformerBlock() generation using symbolic Tensor graph."""

from .helpers import _cpp_tensor_layer, get_norm_type


def emit_block_method(cname, block_type, block, structure, is_encoder=None):
    """Emit a block method using symbolic Tensor graph pattern.

    Methods accept Tensor input and return Tensor output.
    Residuals use Tensor::add() instead of Addition layers.
    """
    norm_type = get_norm_type(structure.model_type)
    op_type = block.operator_type
    L = []

    # Method signature - returns Tensor, accepts Tensor
    if block_type in ("EncoderBlock",):
        L.append(f"Tensor")
        L.append(f"{cname}::create{block_type}("
                 f"const int layer_id,")
        L.append(f"  Tensor input) {{")
    elif block_type == "DecoderBlock" and is_encoder is False:
        L.append(f"Tensor")
        L.append(f"{cname}::create{block_type}("
                 f"const int layer_id,")
        L.append(f"  Tensor input, "
                 f"Tensor encoder_output) {{")
    else:
        op_label = (op_type.capitalize() if op_type != "attention"
                    else "Transformer")
        method_name = (f"createTransformer{block_type}"
                       if op_type == "attention"
                       else f"create{op_label}{block_type}")
        L.append(f"Tensor")
        L.append(f"{cname}::{method_name}("
                 f"const int layer_id,")
        L.append(f"  Tensor input) {{")

    L.append(f"")
    L.append(f"  using ml::train::createLayer;")
    L.append(f"")

    # Determine prefix for layer names
    if is_encoder is True:
        prefix_expr = '"enc_layer" + std::to_string(layer_id)'
    elif is_encoder is False:
        prefix_expr = '"dec_layer" + std::to_string(layer_id)'
    else:
        prefix_expr = '"layer" + std::to_string(layer_id)'

    L.append(f"  auto prefix = {prefix_expr};")
    L.append(f"")

    norm_suffix = ("_attention_norm" if op_type == "attention"
                   else f"_{op_type}_norm")

    # Track current tensor variable name
    current_input = "input"

    # Pre-operator norm
    if block.pre_attn_norm:
        norm_var = _emit_pre_norm(L, norm_type, norm_suffix, op_type,
                                  current_input)
        op_input = norm_var
    else:
        op_input = current_input

    # Operator
    if block.attention:
        L.append(f"  // Self attention")
        L.append(f"  Tensor att_out =")
        L.append(f"    createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, "
                 f"HEAD_DIM,")
        L.append(f"                    {op_input}, {op_input}, {op_input});")
        L.append(f"")
        op_output = "att_out"
    elif block.operator_layers:
        op_output = _emit_operator_layers(L, block, op_input, prefix_expr)
    else:
        op_output = op_input

    # Operator residual using Tensor::add()
    if block.attn_residual:
        L.append(f"  // {op_type.capitalize()} residual connection")
        L.append(f"  Tensor residual = {current_input}.add({op_output});")
        L.append(f"")
        last_residual = "residual"
    else:
        last_residual = op_output

    # Cross-attention (decoder blocks in encoder-decoder)
    if block.cross_attention and is_encoder is False:
        last_residual = _emit_cross_attention(L, block, norm_type,
                                              last_residual)

    # Pre-FFN norm
    if block.pre_ffn_norm:
        ffn_norm_var = _emit_pre_ffn_norm(L, norm_type, last_residual)
        ffn_input = ffn_norm_var
    else:
        ffn_input = last_residual

    # FFN
    if block.ffn:
        L.append(f"  // Feed forward")
        L.append(f"  Tensor ffn_out = createMlp(layer_id, DIM, "
                 f"INTERMEDIATE_SIZE, {ffn_input});")
        L.append(f"")

    # FFN residual using Tensor::add()
    if block.ffn_residual:
        L.append(f"  // FFN residual connection")
        L.append(f"  Tensor block_out = {last_residual}.add(ffn_out);")
        L.append(f"")
        final_output = "block_out"
    else:
        final_output = "ffn_out" if block.ffn else last_residual

    L.append(f"  return {final_output};")
    L.append(f"}}")
    L.append(f"")
    return "\n".join(L)


def _emit_pre_norm(L, norm_type, norm_suffix, op_type, input_var):
    """Emit pre-operator normalization. Returns output variable name."""
    L.append(f"  // Pre-{op_type} normalization")
    norm_props = [
        f'withKey("name", prefix + "{norm_suffix}")',
        'withKey("epsilon", NORM_EPS)',
    ]
    if norm_type == "rms_norm":
        norm_props.append('withKey("packed", "false")')
    elif norm_type == "layer_normalization":
        norm_props.append('withKey("axis", 3)')

    lines, out_var = _cpp_tensor_layer("att_norm", norm_type, norm_props,
                                       input_var)
    L.extend(lines)
    L.append(f"")
    return out_var


def _emit_pre_ffn_norm(L, norm_type, input_var):
    """Emit pre-FFN normalization. Returns output variable name."""
    L.append(f"  // Pre-FFN normalization")
    norm_props = [
        'withKey("name", prefix + "_ffn_norm")',
        'withKey("epsilon", NORM_EPS)',
    ]
    if norm_type == "rms_norm":
        norm_props.append('withKey("packed", "false")')
    elif norm_type == "layer_normalization":
        norm_props.append('withKey("axis", 3)')

    lines, out_var = _cpp_tensor_layer("ffn_norm", norm_type, norm_props,
                                       input_var)
    L.extend(lines)
    L.append(f"")
    return out_var


def _emit_cross_attention(L, block, norm_type, input_var):
    """Emit cross-attention section. Returns updated tensor variable name."""
    q_input = input_var

    if block.cross_attn_norm:
        L.append(f"  // Cross-attention normalization")
        cross_norm_props = [
            'withKey("name", prefix + "_cross_attn_norm")',
            'withKey("epsilon", NORM_EPS)',
        ]
        if norm_type == "rms_norm":
            cross_norm_props.append('withKey("packed", "false")')
        elif norm_type == "layer_normalization":
            cross_norm_props.append('withKey("axis", 3)')
        lines, q_input = _cpp_tensor_layer(
            "cross_norm", norm_type, cross_norm_props, input_var)
        L.extend(lines)
        L.append(f"")

    L.append(f"  // Cross-attention")
    L.append(f"  Tensor cross_att_out =")
    L.append(f"    createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, "
             f"HEAD_DIM,")
    L.append(f"                    {q_input}, encoder_output, "
             f"encoder_output);")
    L.append(f"")

    if block.cross_attn_residual:
        L.append(f"  // Cross-attention residual")
        L.append(f"  Tensor cross_residual = "
                 f"{input_var}.add(cross_att_out);")
        L.append(f"")
        return "cross_residual"
    return input_var


def _emit_operator_layers(L, block, input_var, prefix_expr):
    """Emit non-attention operator layers using Tensor flow.

    Returns the variable name of the last emitted tensor.
    """
    op_type = block.operator_type
    scope = block.operator_scope
    block_scope = scope.rsplit(".", 1)[0] if "." in scope else scope
    block_scope_san = block_scope.replace(".", "_")

    L.append(f"  // {op_type.capitalize()} operator (auto-generated)")
    prev_var = input_var

    for i, layer in enumerate(block.operator_layers):
        if layer.name.startswith(block_scope_san + "_"):
            suffix = layer.name[len(block_scope_san):]
        else:
            suffix = "_" + layer.name

        var_name = f"op_{i}"
        props = [f'withKey("name", prefix + "{suffix}")']
        for k, v in layer.properties.items():
            if isinstance(v, bool):
                props.append(f'withKey("{k}", '
                             f'"{str(v).lower()}")')
            elif isinstance(v, str):
                props.append(f'withKey("{k}", "{v}")')
            else:
                props.append(f'withKey("{k}", {v})')

        lines, prev_var = _cpp_tensor_layer(
            var_name, layer.layer_type, props, prev_var)
        L.extend(lines)
        L.append(f"")

    return prev_var
