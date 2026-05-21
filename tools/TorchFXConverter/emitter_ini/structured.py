"""Structured mode INI emission: pattern-based block output."""

from .helpers import norm_type_for_model, format_property


def emit_structured(layers, structure, batch_size):
    """Emit structured INI config using detected model patterns.

    Args:
        layers: List of NNTrainerLayerDef.
        structure: ModelStructure from pattern detection.
        batch_size: Batch size for [Model] section.

    Returns:
        str: Complete INI file content in structured mode.
    """
    s = structure
    sections = []

    # File header comment
    sections.append(f"# Auto-generated NNTrainer configuration")
    sections.append(f"# Model: {s.model_type} ({s.arch_type})")
    sections.append("")

    # [Model] section
    sections.append("[Model]")
    sections.append("Type = NeuralNetwork")
    sections.append(f"batch_size = {batch_size}")
    sections.append("")

    # Input layer
    sections.append("[input0]")
    sections.append("Type = input")
    sections.append(f"Input_Shape = 1:1:1")
    sections.append("")

    # Embedding
    if s.embedding:
        # Only use tie_word_embeddings if there's an LM head to tie to
        emb_type = ("tie_word_embeddings"
                    if s.tie_word_embeddings and s.lm_head
                    else "embedding_layer")
        sections.append("[embedding0]")
        sections.append(f"Type = {emb_type}")
        sections.append("input_layers = input0")
        sections.append(f"in_dim = {s.vocab_size}")
        sections.append(f"out_dim = {s.hidden_size}")
        sections.append("")

    # Transformer blocks
    first_input = "embedding0" if s.embedding else "input0"
    nt = norm_type_for_model(s.model_type)

    if s.arch_type == "encoder_decoder":
        _emit_encoder_decoder(sections, s, first_input, nt)
    else:
        _emit_single_stack(sections, s, first_input, nt)

    return "\n".join(sections)


# =========================================================================
# Encoder-decoder layout
# =========================================================================

def _emit_encoder_decoder(sections, s, first_input, nt):
    """Emit encoder + decoder blocks for enc-dec architectures."""
    # Encoder blocks
    enc_blocks = s.encoder_blocks
    enc_b0 = enc_blocks[0] if enc_blocks else None
    for i in range(s.num_encoder_layers):
        input_name = (first_input if i == 0
                      else f"enc_layer{i-1}_block_output")
        sections.extend(
            emit_block_ini(i, input_name, s,
                           block=enc_b0, prefix=f"enc_layer{i}",
                           norm_type=nt))
        sections.append("")

    # Encoder final norm
    enc_last = f"enc_layer{s.num_encoder_layers - 1}_block_output"
    sections.append("[encoder_output_norm]")
    sections.append(f"Type = {nt}")
    sections.append(f"input_layers = {enc_last}")
    sections.append(f"epsilon = {s.norm_eps}")
    if nt == "rms_norm":
        sections.append("packed = false")
    sections.append("")

    # Decoder blocks (with cross-attention)
    dec_blocks = s.decoder_blocks
    dec_b0 = dec_blocks[0] if dec_blocks else None
    for i in range(s.num_decoder_layers):
        input_name = (first_input if i == 0
                      else f"dec_layer{i-1}_block_output")
        sections.extend(
            emit_block_ini(
                i, input_name, s,
                block=dec_b0, prefix=f"dec_layer{i}",
                encoder_output="encoder_output_norm",
                norm_type=nt))
        sections.append("")

    # Decoder final norm
    if s.final_norm:
        dec_last = (f"dec_layer{s.num_decoder_layers - 1}"
                    f"_block_output")
        sections.append("[decoder_output_norm]")
        sections.append(f"Type = {nt}")
        sections.append(f"input_layers = {dec_last}")
        sections.append(f"epsilon = {s.norm_eps}")
        if nt == "rms_norm":
            sections.append("packed = false")
        sections.append("")

    # LM head
    if s.lm_head:
        lm_type = ("tie_word_embeddings" if s.tie_word_embeddings
                    else "fully_connected")
        sections.append("[lm_head]")
        sections.append(f"Type = {lm_type}")
        sections.append("input_layers = decoder_output_norm")
        sections.append(f"unit = {s.vocab_size}")
        sections.append("disable_bias = true")
        if s.tie_word_embeddings:
            sections.append("shared_from = embedding0")
        sections.append("")


# =========================================================================
# Single-stack layout (decoder_only / encoder_only)
# =========================================================================

def _emit_single_stack(sections, s, first_input, nt):
    """Emit blocks for single-stack (decoder-only/encoder-only) models."""
    for i in range(s.num_layers):
        input_name = (first_input if i == 0
                      else f"layer{i-1}_decoder_output")
        block_i = s.blocks[i] if i < len(s.blocks) else None
        sections.extend(
            emit_block_ini(i, input_name, s, block=block_i,
                           norm_type=nt))
        sections.append("")

    # Final norm
    if s.final_norm:
        last_block = f"layer{s.num_layers - 1}_decoder_output"
        sections.append("[output_norm]")
        sections.append(f"Type = {nt}")
        sections.append(f"input_layers = {last_block}")
        sections.append(f"epsilon = {s.norm_eps}")
        if nt == "rms_norm":
            sections.append("packed = false")
        sections.append("")

    # LM head
    if s.lm_head:
        lm_type = ("tie_word_embeddings" if s.tie_word_embeddings
                    else "fully_connected")
        sections.append("[lm_head]")
        sections.append(f"Type = {lm_type}")
        sections.append("input_layers = output_norm")
        sections.append(f"unit = {s.vocab_size}")
        sections.append("disable_bias = true")
        if s.tie_word_embeddings:
            sections.append("shared_from = embedding0")
        sections.append("")


# =========================================================================
# Block-level emission
# =========================================================================

def emit_block_ini(layer_id, input_name, s, block=None, prefix=None,
                   encoder_output=None, norm_type=None):
    """Emit a single block as INI sections.

    Handles both attention blocks and non-attention operator blocks
    (conv, ssm, etc.) generically using block.operator_layers.

    Args:
        layer_id: Block index.
        input_name: Name of the input layer.
        s: ModelStructure.
        block: TransformerBlockPattern to use (default: s.blocks[0]).
        prefix: Layer name prefix (default: "layer{layer_id}").
        encoder_output: Encoder output for cross-attention (decoder only).
        norm_type: Norm type string (e.g. "rms_norm").
    """
    lines = []
    if prefix is None:
        prefix = f"layer{layer_id}"
    b0 = block if block is not None else (s.blocks[0] if s.blocks else None)
    if norm_type is None:
        norm_type = norm_type_for_model(s.model_type)
    op_type = b0.operator_type if b0 else "attention"

    role = f" [{b0.block_role}]" if b0 and b0.block_role else ""
    op_label = op_type.capitalize() if op_type != "attention" else ""
    lines.append(f"# --- {op_label+' ' if op_label else ''}"
                 f"Block {layer_id}{role} ---")

    # Pre-operator norm
    norm_suffix = ("_attention_norm" if op_type == "attention"
                   else f"_{op_type}_norm")
    if b0 and b0.pre_attn_norm:
        norm_name = f"{prefix}{norm_suffix}"
        lines.append(f"[{norm_name}]")
        lines.append(f"Type = {norm_type}")
        lines.append(f"input_layers = {input_name}")
        lines.append(f"epsilon = {s.norm_eps}")
        if norm_type == "rms_norm":
            lines.append("packed = false")
        lines.append("")
        op_input = norm_name
    else:
        op_input = input_name

    # Operator layers
    if b0 and b0.attention:
        op_output = _emit_attention_layers(lines, b0, s, prefix,
                                           op_input, norm_type)
    elif b0 and b0.operator_layers:
        op_output = _emit_operator_layers(lines, b0, op_input, prefix)
    else:
        op_output = op_input

    # Operator residual
    is_enc_dec = (b0 and b0.block_role in ("encoder", "decoder"))
    block_out_name = "block_output" if is_enc_dec else "decoder_output"

    if b0 and b0.attn_residual:
        residual_suffix = (f"_{op_type}_add" if op_type != "attention"
                           else ("_self_attn_add" if is_enc_dec
                                 else "_decoder_add"))
        residual_name = f"{prefix}{residual_suffix}"
        lines.append(f"[{residual_name}]")
        lines.append("Type = addition")
        lines.append(f"input_layers = {input_name},{op_output}")
        lines.append("")
        last_residual = residual_name
    else:
        last_residual = op_output

    # Cross-attention (decoder blocks in encoder-decoder models)
    if b0 and b0.cross_attention and encoder_output:
        last_residual = _emit_cross_attention(
            lines, b0, s, prefix, last_residual, encoder_output,
            norm_type)

    ffn_norm_input = last_residual

    # Pre-FFN norm
    if b0 and b0.pre_ffn_norm:
        lines.append(f"[{prefix}_ffn_norm]")
        lines.append(f"Type = {norm_type}")
        lines.append(f"input_layers = {ffn_norm_input}")
        lines.append(f"epsilon = {s.norm_eps}")
        if norm_type == "rms_norm":
            lines.append("packed = false")
        lines.append("")
        ffn_input = f"{prefix}_ffn_norm"
    else:
        ffn_input = ffn_norm_input

    # FFN sub-layers
    if b0 and b0.ffn:
        _emit_ffn_layers(lines, b0, s, prefix, ffn_input)

    # FFN residual
    if b0 and b0.ffn_residual:
        lines.append(f"[{prefix}_{block_out_name}]")
        lines.append("Type = addition")
        lines.append(f"input_layers = {last_residual},{prefix}_ffn_down")
        lines.append("")

    return lines


# =========================================================================
# Sub-layer emission helpers
# =========================================================================

def _emit_attention_layers(lines, b0, s, prefix, op_input, norm_type):
    """Emit Q/K/V projections, optional Q/K norms, MHA core, O projection."""
    attn = b0.attention
    has_qk_norm = attn.has_qk_norm
    q_unit = s.head_dim * s.num_heads
    kv_unit = s.head_dim * s.num_kv_heads

    # Q projection
    lines.append(f"[{prefix}_wq]")
    lines.append("Type = fully_connected")
    lines.append(f"input_layers = {op_input}")
    lines.append(f"unit = {q_unit}")
    lines.append("disable_bias = true")
    lines.append("")

    # K projection
    lines.append(f"[{prefix}_wk]")
    lines.append("Type = fully_connected")
    lines.append(f"input_layers = {op_input}")
    lines.append(f"unit = {kv_unit}")
    lines.append("disable_bias = true")
    lines.append("")

    # V projection
    lines.append(f"[{prefix}_wv]")
    lines.append("Type = fully_connected")
    lines.append(f"input_layers = {op_input}")
    lines.append(f"unit = {kv_unit}")
    lines.append("disable_bias = true")
    lines.append("")

    # Q/K norms
    q_input = f"{prefix}_wq"
    k_input = f"{prefix}_wk"
    if has_qk_norm:
        lines.append(f"[{prefix}_q_norm]")
        lines.append("Type = reshaped_rms_norm")
        lines.append(f"input_layers = {prefix}_wq")
        lines.append("packed = false")
        lines.append(f"epsilon = {s.norm_eps}")
        lines.append(f"feature_size = {s.head_dim}")
        lines.append("")

        lines.append(f"[{prefix}_k_norm]")
        lines.append("Type = reshaped_rms_norm")
        lines.append(f"input_layers = {prefix}_wk")
        lines.append("packed = false")
        lines.append(f"epsilon = {s.norm_eps}")
        lines.append(f"feature_size = {s.head_dim}")
        lines.append("")
        q_input = f"{prefix}_q_norm"
        k_input = f"{prefix}_k_norm"

    # MHA core
    lines.append(f"[{prefix}_attention]")
    lines.append("Type = mha_core")
    lines.append(f"input_layers = {q_input},{k_input},{prefix}_wv")
    lines.append(f"num_heads = {s.num_heads}")
    lines.append(f"num_heads_kv = {s.num_kv_heads}")
    if attn.has_rope and s.rope_theta:
        lines.append(f"rope_theta = {int(s.rope_theta)}")
    lines.append("")

    # O projection
    lines.append(f"[{prefix}_attention_out]")
    lines.append("Type = fully_connected")
    lines.append(f"input_layers = {prefix}_attention")
    lines.append(f"unit = {s.hidden_size}")
    lines.append("disable_bias = true")
    lines.append("")

    return f"{prefix}_attention_out"


def _emit_cross_attention(lines, b0, s, prefix, last_residual,
                          encoder_output, norm_type):
    """Emit cross-attention layers for encoder-decoder models.

    Returns updated last_residual name.
    """
    # Cross-attention norm
    if b0.cross_attn_norm:
        lines.append(f"[{prefix}_cross_attn_norm]")
        lines.append(f"Type = {norm_type}")
        lines.append(f"input_layers = {last_residual}")
        lines.append(f"epsilon = {s.norm_eps}")
        if norm_type == "rms_norm":
            lines.append("packed = false")
        lines.append("")
        cross_q = f"{prefix}_cross_attn_norm"
    else:
        cross_q = last_residual

    # Cross-attention Q/K/V (Q from decoder, K/V from encoder)
    q_unit = s.head_dim * s.num_heads
    kv_unit = s.head_dim * s.num_kv_heads
    lines.append(f"[{prefix}_cross_wq]")
    lines.append("Type = fully_connected")
    lines.append(f"input_layers = {cross_q}")
    lines.append(f"unit = {q_unit}")
    lines.append("disable_bias = true")
    lines.append("")

    lines.append(f"[{prefix}_cross_wk]")
    lines.append("Type = fully_connected")
    lines.append(f"input_layers = {encoder_output}")
    lines.append(f"unit = {kv_unit}")
    lines.append("disable_bias = true")
    lines.append("")

    lines.append(f"[{prefix}_cross_wv]")
    lines.append("Type = fully_connected")
    lines.append(f"input_layers = {encoder_output}")
    lines.append(f"unit = {kv_unit}")
    lines.append("disable_bias = true")
    lines.append("")

    lines.append(f"[{prefix}_cross_attention]")
    lines.append("Type = mha_core")
    lines.append(f"input_layers = {prefix}_cross_wq,"
                 f"{prefix}_cross_wk,{prefix}_cross_wv")
    lines.append(f"num_heads = {s.num_heads}")
    lines.append(f"num_heads_kv = {s.num_kv_heads}")
    lines.append("")

    lines.append(f"[{prefix}_cross_attention_out]")
    lines.append("Type = fully_connected")
    lines.append(f"input_layers = {prefix}_cross_attention")
    lines.append(f"unit = {s.hidden_size}")
    lines.append("disable_bias = true")
    lines.append("")

    # Cross-attention residual
    if b0.cross_attn_residual:
        lines.append(f"[{prefix}_cross_attn_add]")
        lines.append("Type = addition")
        lines.append(f"input_layers = {last_residual},"
                     f"{prefix}_cross_attention_out")
        lines.append("")
        last_residual = f"{prefix}_cross_attn_add"

    return last_residual


def _emit_ffn_layers(lines, b0, s, prefix, ffn_input):
    """Emit FFN sub-layers (SwiGLU or standard/GELU)."""
    ffn = b0.ffn
    if ffn.ffn_type == "swiglu":
        lines.append(f"[{prefix}_ffn_up]")
        lines.append("Type = fully_connected")
        lines.append(f"input_layers = {ffn_input}")
        lines.append(f"unit = {s.intermediate_size}")
        lines.append("disable_bias = true")
        lines.append("")

        lines.append(f"[{prefix}_ffn_gate]")
        lines.append("Type = fully_connected")
        lines.append(f"input_layers = {ffn_input}")
        lines.append(f"unit = {s.intermediate_size}")
        lines.append("disable_bias = true")
        lines.append("")

        lines.append(f"[{prefix}_ffn_swiglu]")
        lines.append("Type = swiglu")
        lines.append(f"input_layers = {prefix}_ffn_up,"
                     f"{prefix}_ffn_gate")
        lines.append("")

        lines.append(f"[{prefix}_ffn_down]")
        lines.append("Type = fully_connected")
        lines.append(f"input_layers = {prefix}_ffn_swiglu")
        lines.append(f"unit = {s.hidden_size}")
        lines.append("disable_bias = true")
        lines.append("")
    else:
        act = "gelu" if ffn.ffn_type == "gelu_ffn" else "relu"
        lines.append(f"[{prefix}_ffn_fc1]")
        lines.append("Type = fully_connected")
        lines.append(f"input_layers = {ffn_input}")
        lines.append(f"unit = {s.intermediate_size}")
        lines.append("")

        lines.append(f"[{prefix}_ffn_act]")
        lines.append("Type = activation")
        lines.append(f"input_layers = {prefix}_ffn_fc1")
        lines.append(f"Activation = {act}")
        lines.append("")

        lines.append(f"[{prefix}_ffn_down]")
        lines.append("Type = fully_connected")
        lines.append(f"input_layers = {prefix}_ffn_act")
        lines.append(f"unit = {s.hidden_size}")
        lines.append("")


def _emit_operator_layers(lines, block, op_input, prefix):
    """Emit non-attention operator layers as INI sections.

    Takes learnable layers from block.operator_layers and emits them
    with parameterized names (replacing block index with prefix).

    Returns the name of the last emitted layer (for residual connection).
    """
    scope = block.operator_scope
    # Block-level scope (e.g. "model.layers.0")
    block_scope = scope.rsplit(".", 1)[0] if "." in scope else scope
    block_scope_san = block_scope.replace(".", "_")

    prev_name = op_input
    last_name = op_input

    for layer in block.operator_layers:
        # Compute layer name suffix relative to block scope
        if layer.name.startswith(block_scope_san + "_"):
            suffix = layer.name[len(block_scope_san):]
        else:
            suffix = "_" + layer.name
        layer_name = f"{prefix}{suffix}"

        lines.append(f"[{layer_name}]")
        lines.append(f"Type = {layer.layer_type}")
        lines.append(f"input_layers = {prev_name}")
        for k, v in layer.properties.items():
            lines.append(format_property(k, v))
        lines.append("")

        prev_name = layer_name
        last_name = layer_name

    return last_name
