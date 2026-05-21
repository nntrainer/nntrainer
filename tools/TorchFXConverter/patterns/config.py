"""Config metadata extraction and architecture type inference."""

from nntrainer_layers import LAYER_EMBEDDING, LAYER_TIE_WORD_EMBEDDINGS, LAYER_FC


def safe_cfg_int(cfg, *attrs, default=0):
    """Safely get an int config value, falling back through attrs."""
    for attr in attrs:
        val = getattr(cfg, attr, None)
        if val is not None and isinstance(val, (int, float)):
            return int(val)
    return default


def safe_cfg_float(cfg, *attrs, default=0.0):
    """Safely get a float config value, falling back through attrs."""
    for attr in attrs:
        val = getattr(cfg, attr, None)
        if val is not None and isinstance(val, (int, float)):
            return float(val)
    return default


def extract_config_metadata(structure, config):
    """Extract model metadata from HF config into structure."""
    if config is None:
        return

    model_type = getattr(config, "model_type", "")
    structure.model_type = model_type if isinstance(model_type, str) else str(model_type)
    structure.vocab_size = safe_cfg_int(config, "vocab_size")
    structure.hidden_size = safe_cfg_int(config, "hidden_size", "d_model")
    structure.num_heads = safe_cfg_int(config, "num_attention_heads", "num_heads")
    structure.num_kv_heads = safe_cfg_int(
        config, "num_key_value_heads", default=structure.num_heads)
    structure.head_dim = safe_cfg_int(
        config, "head_dim", "d_kv",
        default=(structure.hidden_size // structure.num_heads
                 if structure.num_heads else 0))
    structure.intermediate_size = safe_cfg_int(
        config, "intermediate_size", "d_ff")
    structure.rope_theta = safe_cfg_float(config, "rope_theta")
    if not structure.rope_theta:
        rope_params = getattr(config, "rope_parameters", None)
        if isinstance(rope_params, dict):
            rt = rope_params.get("rope_theta", rope_params.get("base", None))
            if isinstance(rt, (int, float)):
                structure.rope_theta = float(rt)
    structure.norm_eps = safe_cfg_float(
        config, "rms_norm_eps", "norm_eps", "layer_norm_eps",
        "layer_norm_epsilon")
    tie = getattr(config, "tie_word_embeddings", False)
    structure.tie_word_embeddings = bool(tie) if not callable(tie) else False
    structure.max_position_embeddings = safe_cfg_int(
        config, "max_position_embeddings", default=2048)
    structure.conv_l_cache = safe_cfg_int(
        config, "conv_L_cache", default=0)

    # Vision-language models (SigLIP, CLIP, BLIP) have nested sub-configs.
    # Extract vision_config metadata if main config has no hidden_size.
    if not structure.hidden_size:
        vision_cfg = getattr(config, "vision_config", None)
        if vision_cfg is not None:
            structure.hidden_size = safe_cfg_int(
                vision_cfg, "hidden_size")
            structure.num_heads = safe_cfg_int(
                vision_cfg, "num_attention_heads")
            structure.num_kv_heads = structure.num_heads
            structure.head_dim = (
                structure.hidden_size // structure.num_heads
                if structure.num_heads else 0)
            structure.intermediate_size = safe_cfg_int(
                vision_cfg, "intermediate_size")
            structure.norm_eps = safe_cfg_float(
                vision_cfg, "layer_norm_eps", "rms_norm_eps")

    # SSM / Mamba config
    structure.ssm_state_size = safe_cfg_int(
        config, "state_size", "ssm_state_size", default=0)
    structure.ssm_conv_kernel = safe_cfg_int(
        config, "conv_kernel", "d_conv", default=0)
    structure.ssm_expand = safe_cfg_int(
        config, "expand", "ssm_expand", default=0)
    # dt_rank can be "auto" in Mamba config
    dt_rank_val = getattr(config, "time_step_rank",
                  getattr(config, "dt_rank", 0))
    if isinstance(dt_rank_val, str) and dt_rank_val == "auto":
        hidden = structure.hidden_size
        dt_rank_val = max(1, hidden // 16) if hidden else 0
    structure.ssm_dt_rank = int(dt_rank_val) if isinstance(dt_rank_val, (int, float)) else 0


def detect_embedding_and_head(structure, layers):
    """Find embedding layer and LM head."""
    for layer in layers:
        if layer.layer_type == LAYER_EMBEDDING:
            structure.embedding = layer.name
            if not structure.vocab_size and layer.properties:
                structure.vocab_size = int(layer.properties.get(
                    "num_embeddings", 0))
            break

    for layer in reversed(layers):
        if layer.layer_type == LAYER_TIE_WORD_EMBEDDINGS:
            structure.lm_head = layer.name
            structure.tie_word_embeddings = True
            break
        if layer.layer_type == LAYER_FC:
            if "lm_head" in layer.hf_module_name:
                structure.lm_head = layer.name
                break


def infer_arch_type(structure, config, find_block_scopes_fn):
    """Infer architecture type from detected patterns."""
    has_cross_attn = any(b.cross_attention for b in structure.blocks)

    if config:
        model_type = getattr(config, "model_type", "")
        if model_type in ("bert", "roberta", "distilbert", "albert",
                          "electra", "camembert", "xlm-roberta"):
            structure.arch_type = "encoder_only"
            return
        elif model_type in ("t5", "mt5", "bart", "mbart",
                            "pegasus", "marian"):
            structure.arch_type = "encoder_decoder"
            return
        elif model_type in ("mamba", "mamba2"):
            structure.arch_type = "decoder_only"
            return
        elif model_type in ("flux",):
            structure.arch_type = "diffusion_transformer"
            return
        elif model_type in ("siglip", "clip", "blip"):
            structure.arch_type = "vision_language"
            return
        elif model_type in ("conformer", "zipformer"):
            structure.arch_type = "conformer"
            return

        architectures = getattr(config, "architectures", []) or []
        is_base_model = any(
            arch.endswith("Model") and not arch.endswith(("ForCausalLM",
                "ForConditionalGeneration", "ForSeq2SeqLM"))
            for arch in architectures
        )
        if is_base_model and not structure.lm_head:
            structure.arch_type = "embedding"
            return

    if has_cross_attn:
        structure.arch_type = "encoder_decoder"
    elif structure.lm_head or structure.tie_word_embeddings:
        structure.arch_type = "decoder_only"
    else:
        block_scopes = find_block_scopes_fn()
        has_encoder = any("encoder" in s for s in block_scopes)
        has_decoder = any("decoder" in s for s in block_scopes)
        if has_encoder and has_decoder:
            structure.arch_type = "encoder_decoder"
        elif has_encoder:
            structure.arch_type = "encoder_only"
        else:
            structure.arch_type = "decoder_only"
