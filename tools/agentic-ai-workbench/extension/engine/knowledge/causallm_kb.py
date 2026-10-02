"""
CausalLM Knowledge Bank — the closed menu handed to the code-generating LLM.

Why this file exists
--------------------
`nntrainer_kb.py` describes the *generic* nntrainer core API
(`model->addLayer(std::make_shared<FullyConnected>(...))`). CausalLM models do
not use that API at all -- they subclass `causallm::Transformer` /
`causallm::CausalLM` and build the graph with
`createLayer(type, {withKey(...)})`. Feeding the generic KB to the generator is
what produced the earlier broken output (wrong signatures, invented helpers).

Everything here is verified against the real sources under
`Applications/CausalLM/`. Type strings and property keys are transcribed from
the `.h` files, not guessed -- an LLM copying from this table is copying from
ground truth. That is the entire point: a weak model is good at *copying* and
bad at *recalling*, so we remove recall from the task.

Consumed by:
  * agents/cpp/prompts_causallm.py  -- renders the catalog into the prompt
  * agents/cpp/plan_validator.py    -- checks generated code against it
"""
from __future__ import annotations

from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Node-level keys. Consumed by LayerNode, NOT by the layer's setProperty.
# Always legal on any createLayer() call regardless of layer type.
# The validator must never flag these, and the prompt must not let the model
# confuse them with layer properties.
# ---------------------------------------------------------------------------
NODE_PROPS = frozenset({
    "name", "input_layers", "input_shape", "input_dtype", "weight_dtype",
    "packed", "shared_from", "trainable", "distribute", "activation",
    "flatten", "clip_grad_by_norm", "loss_scale_for_mixed", "compute_engine",
})

# ---------------------------------------------------------------------------
# Properties inherited from nntrainer::LayerImpl. ONLY layers whose entry sets
# layer_impl=True accept these. Plain nntrainer::Layer subclasses THROW
# std::invalid_argument if given one -- a very common failure mode, because the
# model sees `weight_initializer` on every fully_connected and copies it onto
# rms_norm / swiglu.
# ---------------------------------------------------------------------------
LAYER_IMPL_PROPS = frozenset({
    "weight_regularizer", "weight_regularizer_constant", "weight_initializer",
    "weight_decay", "bias_decay", "bias_initializer", "disable_bias",
    "print", "skip_prefill",
})

# Built-in nntrainer layer types used by CausalLM graphs. Always registered.
BUILTIN_TYPES: Dict[str, Dict] = {
    "fully_connected": {
        "props": {"unit": "uint", "disable_bias": "bool",
                  "weight_initializer": "enum"},
        "inputs": 1, "outputs": 1,
        "purpose": "Dense projection. The workhorse: q/k/v/o, ffn up/gate/down.",
    },
    "addition": {
        "props": {}, "inputs": 2, "outputs": 1,
        "purpose": "Elementwise add. Used for residual connections.",
    },
    "input": {
        "props": {"input_shape": "str", "input_dtype": "str"},
        "inputs": 0, "outputs": 1,
        "purpose": "Graph input placeholder. KV-cache placeholders are these.",
    },
    "multiply": {
        "props": {}, "inputs": 2, "outputs": 1,
        "purpose": "Elementwise multiply.",
    },
    "activation": {
        "props": {"activation": "enum"}, "inputs": 1, "outputs": 1,
        "purpose": "Standalone activation (relu/tanh/sigmoid/...).",
    },
}

# ---------------------------------------------------------------------------
# The CausalLM custom layer catalog.
#
# `type` (the dict key) is the EXACT string for createLayer(). Transcribed from
# `static constexpr const char *type` in each header.
#
# `registered_by` is load-bearing: if it is None the layer is compiled but no
# model calls registerFactory for it, so createLayer() throws at runtime unless
# the generated model registers it first.
# ---------------------------------------------------------------------------
LAYER_CATALOG: Dict[str, Dict] = {
    # -- normalisation --------------------------------------------------
    "rms_norm": {
        "cls": "causallm::RMSNormLayer",
        "header": "rms_norm.h",
        "inputs": 1, "outputs": 1,
        "layer_impl": False,
        "props": {"epsilon": "float", "gamma_initializer": "enum",
                  "skip_prefill": "bool"},
        "registered_by": "Transformer::registerCustomLayers",
        "purpose": "RMSNorm over the full width. Pre-attention / pre-FFN / final norm.",
        "usage": 'createLayer("rms_norm", {withKey("name", ...), '
                 'withKey("epsilon", std::to_string(NORM_EPS)), '
                 'withKey("packed", "false")})',
        "note": "Plain Layer: rejects weight_initializer/disable_bias. "
                "`packed` is a NODE prop and is always passed as \"false\".",
    },
    "reshaped_rms_norm": {
        "cls": "causallm::ReshapedRMSNormLayer",
        "header": "reshaped_rms_norm.h",
        "inputs": 1, "outputs": 1,
        "layer_impl": False,
        "props": {"epsilon": "float", "feature_size": "uint",
                  "gamma_initializer": "enum", "use_gamma": "bool",
                  "skip_prefill": "bool"},
        "registered_by": None,
        "purpose": "Per-head RMSNorm: normalises each `feature_size` block of the "
                   "width. This is q_norm / k_norm in Qwen3-style attention.",
        "usage": 'createLayer("reshaped_rms_norm", {withKey("name", ...), '
                 'withKey("packed", "false"), '
                 'withKey("epsilon", std::to_string(NORM_EPS)), '
                 'withKey("feature_size", std::to_string(head_dim))})',
        "note": "feature_size MUST divide the input width, and equals head_dim. "
                "NOT registered by the base -- the model's "
                "registerCustomLayers() must registerFactory it.",
    },
    "rms_reverse_norm": {
        "cls": "causallm::RMSReverseNormLayer",
        "header": "rms_reverse_norm.h",
        "inputs": 1, "outputs": 1,
        "layer_impl": False,
        "props": {"epsilon": "float", "weight_initializer": "enum",
                  "outscale_initializer": "enum", "skip_prefill": "bool"},
        "registered_by": None,
        "purpose": "input * weight -> normalize -> * out_scale.",
        "usage": 'createLayer("rms_reverse_norm", {withKey("name", ...), '
                 'withKey("epsilon", std::to_string(NORM_EPS))})',
        "note": "NOT registered anywhere and NO model file uses it. Must "
                "registerFactory it yourself. Prefer rms_norm unless the "
                "architecture genuinely needs reverse norm.",
    },

    # -- attention ------------------------------------------------------
    "mha_core": {
        "cls": "causallm::MHACoreLayer",
        "header": "mha_core.h",
        "inputs": 5, "outputs": 1,
        "layer_impl": True,
        "props": {
            "num_heads": "uint", "num_heads_kv": "uint",
            "max_timestep": "uint", "sliding_window": "uint",
            "max_new_tokens": "uint", "rope_theta": "uint",
            "use_rope": "bool", "max_position_embeddings": "uint",
            "use_sink": "bool", "rope_scaling_type": "str",
            "rope_scaling_factor": "float",
            "rope_partial_rotary_factor": "float",
            "rope_scaling_max_position_embeddings": "uint",
            "attn_logit_softcapping": "float", "is_causal": "bool",
            "dropout_rate": "float", "skip_prefill": "bool",
        },
        "registered_by": "Transformer::registerCustomLayers",
        "purpose": "Attention core: RoPE + scaled-dot-product + KV cache. Sits "
                   "between the q/k/v projections and the output projection.",
        "usage": 'Tensor a = mha({q, k, v, cache_k, cache_v});',
        "note": "INPUT ORDER IS FIXED: {q, k, v, cache_k, cache_v}. "
                "5 inputs = external cache mode (what CausalLM uses); 3-4 "
                "inputs = layer owns its cache. Get cache_k/cache_v from "
                "createKVCachePlaceholders(layer_id, n_heads) -- never build "
                "them by hand. head_dim is derived as q.width()/num_heads and "
                "must equal k.width()/num_heads_kv or finalize() throws. "
                "`cache_index` is a runtime pseudo-property; never emit it.",
    },
    "qkv_layer": {
        "cls": "causallm::QKVLayer",
        "header": "qkv_layer.h",
        "inputs": 1, "outputs": 3,
        "layer_impl": True,
        "props": {"q_unit": "uint", "k_unit": "uint", "v_unit": "uint"},
        "registered_by": None,
        "purpose": "Fused q/k/v projection: one input, three outputs.",
        "usage": "// no model file uses this yet -- see note",
        "note": "NOT registered and NO model uses it, so there is no verified "
                "example to copy. Use three separate fully_connected layers "
                "(wq/wk/wv) instead -- that is what every shipping model does. "
                "Only reach for qkv_layer if weights are genuinely fused.",
    },
    "deberta_attention": {
        "cls": "causallm::DebertaAttentionLayer",
        "header": "deberta_attention_layer.h",
        "inputs": 3, "outputs": 1,
        "layer_impl": True,
        "props": {"num_heads": "uint", "max_position_embeddings": "uint",
                  "max_relative_positions": "uint", "c2p": "bool",
                  "p2c": "bool", "share_att_key": "bool",
                  "relative_attention": "bool", "position_buckets": "int",
                  "input_len": "uint", "disable_bias": "bool"},
        "registered_by": "DebertaV2::registerCustomLayers",
        "purpose": "DeBERTa attention with c2p/p2c relative-position bias.",
        "usage": 'createLayer("deberta_attention", {...})',
        "note": "share_att_key must be true. Encoder-only architectures; not "
                "for causal LM decoders.",
    },

    # -- feed-forward ---------------------------------------------------
    "swiglu": {
        "cls": "causallm::SwiGLULayer",
        "header": "swiglu.h",
        "inputs": 2, "outputs": 1,
        "layer_impl": False,
        "props": {"skip_prefill": "bool"},
        "registered_by": "Transformer::registerCustomLayers",
        "purpose": "SwiGLU activation for the gated MLP.",
        "usage": "Tensor act = swiglu({up, gate}, {1, 0});",
        "note": "THE INDEX REMAP {1, 0} IS MANDATORY AND INTENTIONAL. The "
                "layer reads input[0] as gate and input[1] as up, but "
                "nntrainer binaries store MLP weights in up,gate order, so the "
                "layers are created up-then-gate and the remap fixes the "
                "wiring. Emitting swiglu({gate, up}) or dropping {1, 0} "
                "compiles and silently produces wrong numerics.",
    },
    "custom_multiply": {
        "cls": "causallm::CustomMultiplyLayer",
        "header": "custom_multiply.h",
        "inputs": 2, "outputs": 1,
        "layer_impl": False,
        "props": {"print": "bool", "inplace": "bool",
                  "inplace_direction": "str"},
        "registered_by": "LFM2::registerCustomLayers",
        "purpose": "Elementwise multiply with broadcasting + correct "
                   "incremental slicing. Use for gating.",
        "usage": 'createLayer("custom_multiply", {withKey("name", ...), '
                 'withKey("inplace", "true")})',
        "note": "inplace_direction is \"left\" or \"right\".",
    },
    "scalar_multiply": {
        "cls": "causallm::ScalarMultiplyLayer",
        "header": "scalar_multiply.h",
        "inputs": 1, "outputs": 1,
        "layer_impl": False,
        "props": {"multiplier": "float", "use_weight": "bool",
                  "skip_prefill": "bool"},
        "registered_by": "Gemma4::registerCustomLayers",
        "purpose": "Multiply by a scalar constant, or by a learned FP32 scalar.",
        "usage": 'createLayer("scalar_multiply", {withKey("name", ...), '
                 'withKey("multiplier", std::to_string(v))})',
        "note": "use_weight=true reads a scalar_multiplier weight instead of "
                "the multiplier property.",
    },

    # -- embedding ------------------------------------------------------
    "embedding_layer": {
        "cls": "causallm::EmbeddingLayer",
        "header": "embedding_layer.h",
        "inputs": 1, "outputs": 1,
        "layer_impl": True,
        "props": {"in_dim": "uint", "out_dim": "uint", "scale": "float",
                  "quantized_lut_path": "str",
                  "output_quant_scale": "float",
                  "output_quant_offset": "int"},
        "registered_by": "Transformer::registerCustomLayers",
        "purpose": "Token-id -> embedding vector lookup, with optional sidecar "
                   "quantized LUT.",
        "usage": "// prefer the base helper:\n"
                 "buildEmbeddingLayerProperties(name, in_dim, out_dim, "
                 "weight_dtype, scale, quantized_lut_path)",
        "note": "in_dim = vocab_size, out_dim = hidden_size. Use the base-class "
                "helper buildEmbeddingLayerProperties() rather than assembling "
                "props by hand -- it handles dtype and LUT wiring.",
    },
    "tie_word_embeddings": {
        "cls": "causallm::TieWordEmbedding",
        "header": "tie_word_embedding.h",
        "inputs": 1, "outputs": 1,
        "layer_impl": True,
        "props": {"in_dim": "uint", "out_dim": "uint", "unit": "uint",
                  "scale": "float"},
        "registered_by": "Transformer::registerCustomLayers",
        "purpose": "Dual-mode weight-tied embedding. Acts as the embedding when "
                   "`unit` is unset; acts as the LM head when `unit` IS set.",
        "usage": '// lm-head mode:\n'
                 'createLayer("tie_word_embeddings", '
                 '{withKey("name", "output_of_causallm"), '
                 'withKey("unit", NUM_VOCAB), '
                 'withKey("shared_from", "embedding0")})',
        "note": "TYPE STRING IS PLURAL: \"tie_word_embeddings\". The header is "
                "named tie_word_embedding.h (singular) -- do not copy the "
                "filename. Presence of `unit` is what selects head mode. "
                "Only used when config tie_word_embeddings == true.",
    },
    "embedding_normalize": {
        "cls": "causallm::EmbeddingNormalizeLayer",
        "header": "embedding_normalize_layer.h",
        "inputs": 1, "outputs": 1,
        "layer_impl": True,
        "props": {},
        "registered_by": "SentenceTransformer::registerCustomLayers",
        "purpose": "L2-normalise along width. sentence-transformers Normalize.",
        "usage": 'createLayer("embedding_normalize", {withKey("name", ...)})',
        "note": "Takes no layer properties. Embedding models only, not causal LM.",
    },
    "embedding_pooling": {
        "cls": "causallm::EmbeddingPoolingLayer",
        "header": "embedding_pooling_layer.h",
        "inputs": 1, "outputs": 1,
        "layer_impl": True,
        "props": {"word_embedding_dimension": "uint",
                  "pooling_mode_mean_tokens": "bool",
                  "pooling_mode_lasttoken": "bool",
                  "include_prompt": "bool"},
        "registered_by": "SentenceTransformer::registerCustomLayers",
        "purpose": "Sequence -> single sentence embedding. Output height = 1.",
        "usage": 'createLayer("embedding_pooling", '
                 '{withKey("pooling_mode_mean_tokens", "true")})',
        "note": "pooling_mode_cls_token / _max_tokens / _mean_sqrt_len_tokens / "
                "_weightedmean_tokens all THROW not_supported. Only mean_tokens "
                "and lasttoken work. Embedding models only.",
    },

    # -- head / output --------------------------------------------------
    "lm_head": {
        "cls": "causallm::LmHeadLayer",
        "header": "lm_head.h",
        "inputs": 1, "outputs": 1,
        "layer_impl": True,
        "props": {"unit": "uint"},
        "registered_by": "CausalLM::registerCustomLayers",
        "purpose": "Final vocab projection. Forces output height to 1.",
        "usage": 'createLayer("lm_head", '
                 '{withKey("name", "output_of_causallm"), '
                 'withKey("unit", NUM_VOCAB), '
                 'withKey("disable_bias", "true")})',
        "note": "CausalLM::constructModel() already adds this. Only emit it if "
                "you are overriding constructModel. Use tie_word_embeddings "
                "instead when config tie_word_embeddings == true.",
    },
    "logit_softcapping": {
        "cls": "causallm::LogitSoftCappingLayer",
        "header": "logit_softcapping.h",
        "inputs": 1, "outputs": 1,
        "layer_impl": False,
        "props": {"activation_type": "enum", "apply_rows": "uint",
                  "softcap_value": "float", "skip_prefill": "bool"},
        "registered_by": "Gemma4::registerCustomLayers",
        "purpose": "x/cap -> activation -> *cap, on the first `apply_rows` rows.",
        "usage": 'createLayer("logit_softcapping", '
                 '{withKey("activation_type", "tanh"), '
                 'withKey("apply_rows", "1"), '
                 'withKey("softcap_value", std::to_string(cap))})',
        "note": "activation_type is REQUIRED. softcap_value must be > 0.",
    },

    # -- misc / architecture-specific -----------------------------------
    "causal_conv1d": {
        "cls": "causallm::CausalConv1DLayer",
        "header": "causal_conv1d_layer.h",
        "inputs": 1, "outputs": 1,
        "layer_impl": True,
        "props": {},
        "registered_by": "LFM2::registerCustomLayers",
        "purpose": "Causal depthwise Conv1D (kernel fixed at 3) with a "
                   "conv_state cache for O(1) decode.",
        "usage": 'createLayer("causal_conv1d", {withKey("name", ...), '
                 'withKey("weight_dtype", "FP32")})',
        "note": "Kernel size is hard-coded to 3. Expects [B,1,T,W]; weight is "
                "[1,1,3,W] kernel-first. Hybrid conv/attention models (LFM2).",
    },
    "per_layer_slice": {
        "cls": "causallm::PerLayerSliceLayer",
        "header": "per_layer_slice.h",
        "inputs": 1, "outputs": 1,
        "layer_impl": False,
        "props": {"feature_size": "uint", "layer_index": "uint",
                  "skip_prefill": "bool"},
        "registered_by": "Gemma4::registerCustomLayers",
        "purpose": "Slice chunk `layer_index` of width `feature_size` out of a "
                   "packed per-layer embedding tensor.",
        "usage": 'createLayer("per_layer_slice", '
                 '{withKey("feature_size", std::to_string(fs)), '
                 'withKey("layer_index", std::to_string(i))})',
        "note": "feature_size is required and must be > 0.",
    },
    "shared_fully_connected": {
        "cls": "causallm::SharedFullyConnectedLayer",
        "header": "shared_fully_connected_layer.h",
        "inputs": 1, "outputs": 1,
        "layer_impl": True,
        "props": {"unit": "uint", "disable_bias": "bool",
                  "shared_mode": "bool", "full_input_range": "bool",
                  "weight_initializer": "enum", "bias_initializer": "enum"},
        "registered_by": "DebertaV2::registerCustomLayers",
        "purpose": "FC that does NOT read weights from file -- pair with the "
                   "node-level `shared_from` key to reuse another layer's weights.",
        "usage": 'createLayer("shared_fully_connected", '
                 '{withKey("shared_from", "<other layer name>"), '
                 'withKey("shared_mode", "true"), '
                 'withKey("full_input_range", "true")})',
        "note": "Transformer::constructModel special-cases this type when "
                "walking weights.",
    },
}


# ---------------------------------------------------------------------------
# Layer NAMES are load-bearing. Weight loading matches safetensors header keys
# against weight->getName(), and the .py weight converters emit exactly these
# nntrainer-style names (NOT HF dotted names). A wrong name compiles cleanly
# and then silently loads zero weights -- the worst possible failure mode, so
# this table is stated as a hard contract in the prompt.
# ---------------------------------------------------------------------------
LAYER_NAME_CONTRACT: List[Dict[str, str]] = [
    {"name": "input0", "what": "graph input (token ids)"},
    {"name": "embedding0", "what": "token embedding"},
    {"name": "layer<i>_attention_norm", "what": "pre-attention norm"},
    {"name": "layer<i>_wq", "what": "Q projection"},
    {"name": "layer<i>_q_norm", "what": "per-head Q norm (if architecture has it)"},
    {"name": "layer<i>_wk", "what": "K projection"},
    {"name": "layer<i>_k_norm", "what": "per-head K norm (if architecture has it)"},
    {"name": "layer<i>_wv", "what": "V projection"},
    {"name": "cache_k_l<i>", "what": "KV-cache K placeholder (from helper)"},
    {"name": "cache_v_l<i>", "what": "KV-cache V placeholder (from helper)"},
    {"name": "layer<i>_attention", "what": "mha_core"},
    {"name": "layer<i>_attention_out", "what": "O projection"},
    {"name": "layer<i>_decoder_add", "what": "post-attention residual add"},
    {"name": "layer<i>_ffn_norm", "what": "pre-FFN norm"},
    {"name": "layer<i>_ffn_up", "what": "FFN up projection"},
    {"name": "layer<i>_ffn_gate", "what": "FFN gate projection"},
    {"name": "layer<i>_ffn_swiglu", "what": "SwiGLU"},
    {"name": "layer<i>_ffn_down", "what": "FFN down projection"},
    {"name": "layer<i>_decoder_output", "what": "post-FFN residual add"},
    {"name": "output_norm", "what": "final norm"},
    {"name": "output_of_causallm", "what": "LM head"},
]

# Config members set for you by Transformer::setupParameters. The generated code
# must USE these, never re-derive them from the json. Every value a weak model
# would otherwise have to compute (and get wrong) is already a member.
CONFIG_MEMBERS: Dict[str, str] = {
    "NUM_VOCAB": "cfg.vocab_size",
    "DIM": "cfg.hidden_size",
    "INTERMEDIATE_SIZE": "cfg.intermediate_size",
    "NUM_LAYERS": "cfg.num_hidden_layers",
    "NUM_HEADS": "cfg.num_attention_heads",
    "HEAD_DIM": "cfg.head_dim, else DIM / NUM_HEADS",
    "NUM_KEY_VALUE_HEADS": "cfg.num_key_value_heads, else NUM_HEADS",
    "GQA_SIZE": "NUM_HEADS / NUM_KEY_VALUE_HEADS  (derived)",
    "SLIDING_WINDOW": "cfg.sliding_window, else UINT_MAX",
    "SLIDING_WINDOW_PATTERN": "cfg.sliding_window_pattern, else 1",
    "MAX_POSITION_EMBEDDINGS": "cfg.max_position_embeddings",
    "ROPE_THETA": "cfg.rope_theta | rope_parameters.rope_theta, else 10000",
    "TIE_WORD_EMBEDDINGS": "cfg.tie_word_embeddings, else false",
    "NORM_EPS": "cfg.rms_norm_eps, else 1e-5",
    "IS_CAUSAL": "cfg.is_causal / !use_bidirectional_attention",
    "BATCH_SIZE": "nntr_cfg.batch_size",
    "INIT_SEQ_LEN": "nntr_cfg.init_seq_len",
    "MAX_SEQ_LEN": "nntr_cfg.max_seq_len",
    "NUM_TO_GENERATE": "nntr_cfg.num_to_generate",
    "MODEL_TENSOR_TYPE": "nntr_cfg.model_tensor_type",
    "EMBEDDING_DTYPE": "nntr_cfg.embedding_dtype",
    "FC_LAYER_DTYPE": "nntr_cfg.fc_layer_dtype",
    "LMHEAD_DTYPE": "nntr_cfg.lmhead_dtype, else embedding_dtype",
}

# Base-class hooks that may be overridden, with exact signatures. Anything not
# in this dict does not exist -- the prompt states that explicitly so the model
# cannot invent createMLP / createDecoderLayer / applyRoPE, all of which
# appeared in earlier broken output.
OVERRIDABLE_HOOKS: Dict[str, Dict[str, str]] = {
    "createAttention": {
        "sig": "Tensor createAttention(const int layer_id, int seq_len, "
               "int n_heads, int head_dim, Tensor query, Tensor key, Tensor value)",
        "when": "Attention differs from plain llama: q/k norm, sinks, "
                "yarn scaling, fused qkv.",
        "default": "wq/wk/wv -> mha_core -> wo",
    },
    "createMlp": {
        "sig": "Tensor createMlp(const int layer_id, int dim, int hidden_dim, "
               "Tensor input)",
        "when": "MLP is not a standard SwiGLU gated FFN (e.g. MoE).",
        "default": "ffn_up + ffn_gate -> swiglu({up,gate},{1,0}) -> ffn_down",
    },
    "createTransformerDecoderBlock": {
        "sig": "Tensor createTransformerDecoderBlock(const int layer_id, Tensor input)",
        "when": "Block topology differs: extra norms, different residual "
                "routing, conv-instead-of-attention layers.",
        "default": "attn_norm -> createAttention -> add -> ffn_norm -> "
                   "createMlp -> add",
    },
    "constructModel": {
        "sig": "std::pair<Tensor, Tensor> constructModel()",
        "when": "Whole-graph shape differs: embedding scaling, extra "
                "pre/post layers, softcapping on logits.",
        "default": "input0 -> embedding0 -> N x decoder block -> output_norm, "
                   "then CausalLM appends the LM head",
    },
    "setupParameters": {
        "sig": "void setupParameters(json &cfg, json &generation_cfg, json &nntr_cfg)",
        "when": "The architecture has config fields the base does not read "
                "(num_experts, moe_intermediate_size, ...).",
        "default": "reads the standard HF fields listed in CONFIG_MEMBERS",
    },
    "registerCustomLayers": {
        "sig": "void registerCustomLayers()",
        "when": "ALWAYS, if you use any layer whose registered_by is None.",
        "default": "base registers swiglu, rms_norm, mha_core, "
                   "tie_word_embeddings, embedding_layer, lm_head",
    },
}

# Methods that do NOT exist. Earlier generator output invented every one of
# these. Listing them explicitly is far more effective on weak models than a
# generic "don't invent APIs" instruction.
FORBIDDEN_SYMBOLS: Dict[str, str] = {
    "createMLP": "Wrong case. The real hook is createMlp.",
    "createDecoderLayer": "Does not exist. Use createTransformerDecoderBlock.",
    "createFeedForward": "Does not exist. Use createMlp.",
    "applyRoPE": "Does not exist. mha_core applies RoPE internally.",
    "linear": "Does not exist. Use createLayer(\"fully_connected\", ...).",
    "matmul": "Does not exist. There is no tensor-level math in model files.",
    "softmax": "Does not exist. mha_core handles attention softmax.",
    "registerLayerType": "Does not exist. Use "
                         "app_context->registerFactory(nntrainer::createLayer<T>).",
    "util/tensor_util.h": "No such header.",
    "model->addLayer": "That is the generic nntrainer API. CausalLM models "
                       "build graphs functionally with createLayer + "
                       "LayerHandle operator().",
    "std::make_shared<FullyConnected>": "Generic nntrainer API. Use "
                                        "createLayer(\"fully_connected\", ...).",
}


# ---------------------------------------------------------------------------
# Rendering helpers. Kept here so the prompt module stays declarative and the
# validator and prompt cannot drift apart.
# ---------------------------------------------------------------------------
def all_known_types() -> frozenset:
    """Every type string legal in createLayer()."""
    return frozenset(LAYER_CATALOG) | frozenset(BUILTIN_TYPES)


def allowed_props(layer_type: str) -> frozenset:
    """Full set of property keys accepted by `layer_type`, node props included."""
    entry = LAYER_CATALOG.get(layer_type) or BUILTIN_TYPES.get(layer_type)
    if entry is None:
        return frozenset()
    props = frozenset(entry.get("props", {})) | NODE_PROPS
    if entry.get("layer_impl"):
        props |= LAYER_IMPL_PROPS
    return props


def unregistered_types() -> List[str]:
    """Types that need an explicit registerFactory before use."""
    return sorted(t for t, e in LAYER_CATALOG.items()
                  if e.get("registered_by") is None)


def render_catalog(only: Optional[List[str]] = None) -> str:
    """
    Render the catalog as the closed menu for the prompt.

    `only` restricts output to the layers the resolved plan actually needs --
    a much shorter menu is easier for a small model to hold, and it removes the
    chance of picking a plausible-but-wrong neighbour. Pass None for the full
    catalog (used by the chat/explain paths).
    """
    keys = sorted(only) if only else sorted(LAYER_CATALOG)
    out: List[str] = []
    for t in keys:
        e = LAYER_CATALOG.get(t) or BUILTIN_TYPES.get(t)
        if e is None:
            continue
        props = e.get("props", {})
        prop_str = ", ".join(f"{k}:{v}" for k, v in props.items()) or "(none)"
        if e.get("layer_impl"):
            prop_str += "  [+ weight_initializer, disable_bias, skip_prefill, ...]"
        lines = [
            f'### createLayer("{t}", ...)',
            f'  class     : {e.get("cls", "nntrainer built-in")}',
            f'  purpose   : {e["purpose"]}',
            f'  tensors   : {e.get("inputs")} in -> {e.get("outputs")} out',
            f'  properties: {prop_str}',
        ]
        if e.get("header"):
            lines.append(f'  include   : #include <{e["header"]}>')
        reg = e.get("registered_by")
        if "registered_by" in e:
            lines.append(
                f'  registered: {reg}' if reg
                else '  registered: *** NOT REGISTERED -- you must '
                     'registerFactory it in registerCustomLayers() ***'
            )
        if e.get("usage"):
            lines.append(f'  usage     : {e["usage"]}')
        if e.get("note"):
            lines.append(f'  !! NOTE   : {e["note"]}')
        out.append("\n".join(lines))
    return "\n\n".join(out)


def render_name_contract() -> str:
    """The load-bearing layer-name table."""
    return "\n".join(
        f'  {d["name"]:<32} {d["what"]}' for d in LAYER_NAME_CONTRACT
    )


def render_config_members() -> str:
    """Members already populated by setupParameters."""
    return "\n".join(f"  {k:<32} = {v}" for k, v in CONFIG_MEMBERS.items())


def render_forbidden() -> str:
    """Explicit do-not-emit list."""
    return "\n".join(f"  {k:<34} -> {v}" for k, v in FORBIDDEN_SYMBOLS.items())


def render_hooks(only: Optional[List[str]] = None) -> str:
    """Overridable hooks with exact signatures."""
    keys = only or list(OVERRIDABLE_HOOKS)
    out = []
    for k in keys:
        h = OVERRIDABLE_HOOKS.get(k)
        if not h:
            continue
        out.append(
            f"  {h['sig']}\n"
            f"      override when : {h['when']}\n"
            f"      base default  : {h['default']}"
        )
    return "\n\n".join(out)
