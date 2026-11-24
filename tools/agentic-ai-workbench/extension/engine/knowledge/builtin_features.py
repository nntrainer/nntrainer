# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file builtin_features.py
# @brief Built-in checklist features of the hand-written CausalLM models.
# @author prachi-t17 <prachi.t@samsung.com>

"""
Built-in checklist boxes: the architecture features the hand-written
Applications/CausalLM models implement, and how to recognise each one in a
Hugging Face checkpoint (config.json + tensor names/shapes).

Every box is plain data in the same format a learned box uses, so the code
bank can add new ones later (see checklist.py for the rule language).

Each rule was written from the code that implements the feature, and the
conditions include that code's limits -- e.g. GPT-OSS's YaRN parameters are
hard-coded, Gemma3's local-layer RoPE theta is fixed at 10000, and the
qwen_moe layer always re-normalises the top-k router weights. When a checkpoint
falls outside those limits the box does not tick, the config key or tensor
stays unexplained, and the checklist reports it as a new part instead of
silently reusing code that would compute something else.

Tensor names are matched after normalisation: the `model.` /
`language_model.model.` prefix is dropped and layer/expert indices become
`*` (`model.layers.3.mlp.experts.7.up_proj.weight` ->
`layers.*.mlp.experts.*.up_proj.weight`). `{a,b}` is alternation and `**`
matches anything.
"""

GEMMA_TYPES = ["gemma", "gemma2", "gemma3", "gemma3_text"]

# Hooks a feature changes. "setupParameters"/"registerCustomLayers" are
# composed from features rather than matched; "converter" is the weight
# converter; [] means the feature needs no code (it only explains keys).
BUILTIN_FEATURES = [
    # --------------------------------------------------------------- model
    {
        "id": "embedding",
        "label": "Token embedding",
        "hooks": ["constructModel"],
        "from": ["base"],
        "when": [{"tensor": "embed_tokens.weight"}],
        "claims": {"tensors": ["embed_tokens.weight"], "config": ["vocab_size", "hidden_size"]},
    },
    {
        "id": "decoder_stack",
        "label": "Stack of causal decoder layers",
        "hooks": ["constructModel"],
        "from": ["base"],
        "when": [
            {"config": "num_hidden_layers", "op": "exists"},
            {"tensor": "layers.*.**"},
            {"config": "is_causal", "op": "ne", "value": False},
            {"config": "use_bidirectional_attention", "op": "falsy"},
        ],
        "claims": {
            "config": [
                "num_hidden_layers",
                "max_position_embeddings",
                "is_causal",
                "use_bidirectional_attention",
            ]
        },
    },
    {
        "id": "final_norm_rms",
        "label": "Final RMSNorm",
        "hooks": ["constructModel"],
        "from": ["base"],
        "when": [{"tensor": "norm.weight"}],
        "claims": {"tensors": ["norm.weight"], "config": ["rms_norm_eps"]},
    },
    {
        "id": "lm_head_tied",
        "label": "LM head tied to the embedding",
        "hooks": ["constructModel"],
        "from": ["base"],
        "when": [{"config": "@tie_word_embeddings", "op": "eq", "value": True}],
        "claims": {"tensors": ["lm_head.weight"], "config": ["tie_word_embeddings"]},
        "note": (
            "nntrainer reuses embedding0's weight (tie_word_embeddings layer); a stored"
            " lm_head.weight is ignored."
        ),
    },
    {
        "id": "lm_head_untied",
        "label": "Separate LM head",
        "hooks": ["constructModel"],
        "from": ["base"],
        "when": [
            {"config": "@tie_word_embeddings", "op": "eq", "value": False},
            {"tensor": "lm_head.weight"},
        ],
        "claims": {"tensors": ["lm_head.weight"], "config": ["tie_word_embeddings"]},
    },
    # --------------------------------------------------------------- block
    {
        "id": "block_pre_norm",
        "label": "Pre-norm block (norm before attention and before MLP)",
        "hooks": ["createTransformerDecoderBlock"],
        "from": ["base"],
        "when": [
            {"tensor": "layers.*.input_layernorm.weight"},
            {"tensor": "layers.*.post_attention_layernorm.weight"},
            {"no_tensor": "layers.*.{pre,post}_feedforward_layernorm.weight"},
        ],
        "claims": {
            "tensors": [
                "layers.*.input_layernorm.weight",
                "layers.*.post_attention_layernorm.weight",
            ],
            "config": ["rms_norm_eps"],
        },
        "note": "In this layout HF's post_attention_layernorm is the norm before the MLP.",
    },
    {
        "id": "block_sandwich_norm",
        "label": "Norms before and after both attention and MLP",
        "hooks": ["createTransformerDecoderBlock"],
        "from": ["gemma3"],
        "when": [
            {"tensor": "layers.*.input_layernorm.weight"},
            {"tensor": "layers.*.post_attention_layernorm.weight"},
            {"tensor": "layers.*.pre_feedforward_layernorm.weight"},
            {"tensor": "layers.*.post_feedforward_layernorm.weight"},
        ],
        "claims": {
            "tensors": [
                "layers.*.input_layernorm.weight",
                "layers.*.post_attention_layernorm.weight",
                "layers.*.pre_feedforward_layernorm.weight",
                "layers.*.post_feedforward_layernorm.weight",
            ],
            "config": ["rms_norm_eps"],
        },
    },
    # ----------------------------------------------------------- attention
    {
        "id": "attn_gqa_rope",
        "label": "Multi-head / grouped-query attention with separate Q, K, V, O",
        "hooks": ["createAttention"],
        "from": ["base"],
        "when": [
            {"tensor": "layers.*.self_attn.q_proj.weight"},
            {"tensor": "layers.*.self_attn.k_proj.weight"},
            {"tensor": "layers.*.self_attn.v_proj.weight"},
            {"tensor": "layers.*.self_attn.o_proj.weight"},
            {"shape": "layers.*.self_attn.q_proj.weight", "dims": ["@q_width", "@hidden_size"]},
            {"shape": "layers.*.self_attn.k_proj.weight", "dims": ["@kv_width", "@hidden_size"]},
        ],
        "claims": {
            "tensors": ["layers.*.self_attn.{q,k,v,o}_proj.weight"],
            "config": ["num_attention_heads", "num_key_value_heads", "head_dim", "attention_bias"],
        },
    },
    {
        "id": "rope_default",
        "label": "Standard RoPE (no scaling)",
        "hooks": ["createAttention"],
        "from": ["base"],
        "when": [
            {"config": "@rope_types", "op": "all_in", "value": ["default"]},
            {"config": "partial_rotary_factor", "op": "in", "value": [None, 1, 1.0]},
        ],
        "claims": {
            "config": ["rope_theta", "rope_parameters", "rope_scaling", "partial_rotary_factor"]
        },
    },
    {
        "id": "rope_yarn",
        "label": "YaRN-scaled RoPE (GPT-OSS parameters)",
        "hooks": ["createAttention", "setupParameters"],
        "from": ["gpt_oss"],
        "when": [
            {"config": "@rope_types", "op": "all_in", "value": ["yarn"]},
            {"config": "@rope.original_max_position_embeddings", "op": "eq", "value": 4096},
            {"config": "@rope.beta_fast", "op": "in", "value": [None, 32, 32.0]},
            {"config": "@rope.beta_slow", "op": "in", "value": [None, 1, 1.0]},
        ],
        "claims": {"config": ["rope_theta", "rope_parameters", "rope_scaling"]},
        "note": (
            "GPT-OSS passes rope_scaling_max_position_embeddings=4096 and mha_core fixes"
            " beta_fast=32, beta_slow=1; only the factor is read from config."
        ),
    },
    {
        "id": "qkv_bias",
        "label": "Bias on Q, K and V projections",
        "hooks": ["createAttention"],
        "from": ["qwen2", "gpt_oss"],
        "when": [
            {"tensor": "layers.*.self_attn.q_proj.bias"},
            {"tensor": "layers.*.self_attn.k_proj.bias"},
            {"tensor": "layers.*.self_attn.v_proj.bias"},
        ],
        "claims": {
            "tensors": ["layers.*.self_attn.{q,k,v}_proj.bias"],
            "config": ["attention_bias"],
        },
    },
    {
        "id": "o_proj_bias",
        "label": "Bias on the output projection",
        "hooks": ["createAttention"],
        "from": ["gpt_oss"],
        "when": [{"tensor": "layers.*.self_attn.o_proj.bias"}],
        "claims": {"tensors": ["layers.*.self_attn.o_proj.bias"], "config": ["attention_bias"]},
    },
    {
        "id": "qk_norm_per_head",
        "label": "RMSNorm on Q and K, per head",
        "hooks": ["createAttention", "registerCustomLayers"],
        "from": ["qwen3", "gemma3"],
        "when": [
            {"tensor": "layers.*.self_attn.q_norm.weight"},
            {"tensor": "layers.*.self_attn.k_norm.weight"},
            {"shape": "layers.*.self_attn.{q,k}_norm.weight", "dims": ["@head_dim"]},
        ],
        "claims": {"tensors": ["layers.*.self_attn.{q,k}_norm.weight"]},
        "note": (
            "Uses reshaped_rms_norm over head_dim. A Q/K norm over the full projection width does"
            " not tick."
        ),
    },
    {
        "id": "attention_sinks",
        "label": "Attention sinks",
        "hooks": ["createAttention"],
        "from": ["gpt_oss"],
        "when": [{"tensor": "layers.*.self_attn.sinks"}],
        "claims": {"tensors": ["layers.*.self_attn.sinks"]},
    },
    {
        "id": "attn_logit_softcapping",
        "label": "Attention logit soft-capping",
        "hooks": ["createAttention", "setupParameters"],
        "from": ["gemma3"],
        "when": [{"config": "attn_logit_softcapping", "op": "not_null"}],
        "claims": {"config": ["attn_logit_softcapping"]},
        "note": (
            "mha_core implements it and Gemma3's attention passes it, but the value is only read in"
            " Gemma3Transformer::setupParameters, which never runs (constructors dispatch to the"
            " base setupParameters). No captured hook handles a non-null value, so this needs new"
            " code."
        ),
    },
    {
        "id": "query_scale_standard",
        "label": "Query scaled by 1/sqrt(head_dim)",
        "hooks": [],
        "from": ["base"],
        "when": [{"config": "query_pre_attn_scalar", "op": "eq", "value": "@head_dim"}],
        "claims": {"config": ["query_pre_attn_scalar"]},
        "note": (
            "mha_core always scales by head_dim; a different query_pre_attn_scalar does not tick."
        ),
    },
    # ------------------------------------------------------ sliding window
    {
        "id": "sliding_window_off",
        "label": "No sliding window",
        "hooks": [],
        "from": ["base"],
        "when": [
            {
                "any": [
                    {"config": "sliding_window", "op": "null_or_absent"},
                    {"config": "use_sliding_window", "op": "eq", "value": False},
                ]
            },
            {"config": "sliding_window_pattern", "op": "null_or_absent"},
            {"config": "@layer_kinds", "op": "all_in", "value": ["full_attention"]},
        ],
        "claims": {
            "config": ["sliding_window", "use_sliding_window", "max_window_layers", "layer_types"]
        },
        "note": (
            "If sliding_window is set but disabled, the hand-written attention still passes it to "
            "mha_core; that only matters past sliding_window tokens."
        ),
    },
    {
        "id": "sliding_window_uniform",
        "label": "Same sliding window on every layer",
        "hooks": ["createAttention"],
        "from": ["qwen2", "qwen3"],
        "when": [
            {"config": "sliding_window", "op": "not_null"},
            {"config": "use_sliding_window", "op": "ne", "value": False},
            {"config": "max_window_layers", "op": "null_or_absent"},
            {"config": "sliding_window_pattern", "op": "null_or_absent"},
            {"config": "@layer_kinds", "op": "all_in", "value": ["sliding_attention"]},
        ],
        "claims": {"config": ["sliding_window", "use_sliding_window", "layer_types"]},
    },
    {
        "id": "sliding_window_pattern",
        "label": "Sliding window on all but every N-th layer",
        "hooks": ["createAttention"],
        "from": ["base", "gemma3"],
        "when": [
            {"config": "sliding_window", "op": "not_null"},
            {"config": "sliding_window_pattern", "op": "not_null"},
            {"config": "layer_types", "op": "null_or_absent"},
        ],
        "claims": {"config": ["sliding_window", "sliding_window_pattern"]},
    },
    {
        "id": "sliding_window_layer_types",
        "label": "Per-layer sliding / full attention (layer_types)",
        "hooks": ["createAttention", "setupParameters"],
        "from": ["gemma3", "gpt_oss"],
        "when": [
            {"config": "@layer_kinds", "op": "contains", "value": "sliding_attention"},
            {
                "config": "@layer_kinds",
                "op": "all_in",
                "value": ["sliding_attention", "full_attention"],
            },
            {"config": "sliding_window", "op": "not_null"},
        ],
        "claims": {"config": ["layer_types", "sliding_window", "sliding_window_pattern"]},
    },
    {
        "id": "local_rope_theta_10000",
        "label": "Sliding layers use RoPE theta 10000 (Gemma3)",
        "hooks": ["createAttention"],
        "from": ["gemma3"],
        "when": [
            {
                "any": [
                    {"config": "model_type", "op": "in", "value": ["gemma3", "gemma3_text"]},
                    {"config": "rope_local_base_freq", "op": "exists"},
                ]
            },
            {"config": "rope_local_base_freq", "op": "in", "value": [None, 10000, 10000.0]},
            {
                "config": "@rope_by_layer_kind.sliding_attention.rope_theta",
                "op": "in",
                "value": [None, 10000, 10000.0],
            },
        ],
        "claims": {"config": ["rope_local_base_freq"]},
        "note": "Gemma3Transformer::createAttention hard-codes 10000 for sliding layers.",
    },
    # ----------------------------------------------------------------- MLP
    {
        "id": "mlp_gated_silu",
        "label": "Gated MLP with SiLU (SwiGLU)",
        "hooks": ["createMlp"],
        "from": ["base"],
        "when": [
            {"tensor": "layers.*.mlp.gate_proj.weight"},
            {"tensor": "layers.*.mlp.up_proj.weight"},
            {"tensor": "layers.*.mlp.down_proj.weight"},
            {"no_tensor": "layers.*.mlp.{gate,up,down}_proj.bias"},
            {"config": "hidden_act", "op": "in", "value": [None, "silu", "swish"]},
            {"config": "hidden_activation", "op": "in", "value": [None, "silu", "swish"]},
        ],
        "claims": {
            "tensors": ["layers.*.mlp.{gate,up,down}_proj.weight"],
            "config": ["hidden_act", "hidden_activation", "intermediate_size", "mlp_bias"],
        },
    },
    {
        "id": "mlp_gated_gelu_tanh",
        "label": "Gated MLP with tanh-GELU (GeGLU)",
        "hooks": ["createMlp"],
        "from": ["gemma3"],
        "when": [
            {"tensor": "layers.*.mlp.gate_proj.weight"},
            {"tensor": "layers.*.mlp.up_proj.weight"},
            {"tensor": "layers.*.mlp.down_proj.weight"},
            {"no_tensor": "layers.*.mlp.{gate,up,down}_proj.bias"},
            {
                "any": [
                    {"config": "hidden_activation", "op": "eq", "value": "gelu_pytorch_tanh"},
                    {
                        "all": [
                            {"config": "hidden_activation", "op": "null_or_absent"},
                            {"config": "hidden_act", "op": "eq", "value": "gelu_pytorch_tanh"},
                        ]
                    },
                ]
            },
        ],
        "claims": {
            "tensors": ["layers.*.mlp.{gate,up,down}_proj.weight"],
            "config": ["hidden_act", "hidden_activation", "intermediate_size", "mlp_bias"],
        },
    },
    {
        "id": "moe_qwen",
        "label": "Mixture of experts, softmax top-k router (Qwen3-MoE)",
        "hooks": ["createMlp", "setupParameters", "registerCustomLayers"],
        "from": ["qwen3_moe"],
        "when": [
            {"tensor": "layers.*.mlp.gate.weight"},
            {"tensor": "layers.*.mlp.experts.*.{gate,up,down}_proj.weight"},
            {"no_tensor": "layers.*.mlp.shared_expert**"},
            {"config": "num_experts", "op": "gt", "value": 0},
            {"config": "norm_topk_prob", "op": "eq", "value": True},
            {"config": "decoder_sparse_step", "op": "in", "value": [None, 1]},
            {"config": "mlp_only_layers", "op": "falsy"},
            {"config": "hidden_act", "op": "in", "value": [None, "silu", "swish"]},
        ],
        "claims": {
            "tensors": [
                "layers.*.mlp.gate.weight",
                "layers.*.mlp.experts.*.{gate,up,down}_proj.weight",
            ],
            "config": [
                "num_experts",
                "num_experts_per_tok",
                "moe_intermediate_size",
                "norm_topk_prob",
                "decoder_sparse_step",
                "mlp_only_layers",
                "router_aux_loss_coef",
                "output_router_logits",
                "hidden_act",
                "intermediate_size",
            ],
        },
        "note": (
            "qwen_moe re-normalises the top-k weights (norm_topk_prob=true); every layer must be"
            " MoE."
        ),
    },
    {
        "id": "moe_gpt_oss",
        "label": "Mixture of experts with router bias and fused gate/up (GPT-OSS)",
        "hooks": ["createMlp", "setupParameters", "registerCustomLayers"],
        "from": ["gpt_oss"],
        "when": [
            {"tensor": "layers.*.mlp.router.weight"},
            {"tensor": "layers.*.mlp.experts.gate_up_proj{,_blocks}"},
            {"tensor": "layers.*.mlp.experts.down_proj{,_blocks}"},
            {"config": "num_local_experts", "op": "gt", "value": 0},
            {"config": "swiglu_limit", "op": "in", "value": [None, 7, 7.0]},
        ],
        "claims": {
            "tensors": [
                "layers.*.mlp.router.{weight,bias}",
                "layers.*.mlp.experts.{gate_up_proj,down_proj}{,_blocks,_scales,_bias}",
            ],
            "config": [
                "num_local_experts",
                "num_experts_per_tok",
                "experts_per_token",
                "swiglu_limit",
                "quantization_config",
                "router_aux_loss_coef",
                "output_router_logits",
                "intermediate_size",
                "hidden_act",
            ],
        },
        "note": (
            "Checkpoints store experts as MXFP4 (_blocks/_scales); the converter dequantises them."
        ),
    },
    # ------------------------------------------------ Gemma-specific maths
    {
        "id": "embedding_scale_sqrt_hidden",
        "label": "Embeddings scaled by sqrt(hidden_size)",
        "hooks": ["setupParameters"],
        "from": ["gemma3"],
        "when": [{"config": "model_type", "op": "in", "value": GEMMA_TYPES}],
        "claims": {},
        "note": (
            "Gemma does this in its modelling code; no config key says so, so it is recognised by"
            " model_type."
        ),
    },
    {
        "id": "norm_one_plus_weight",
        "label": "RMSNorm uses (1 + weight)",
        "hooks": ["converter"],
        "from": ["gemma3"],
        "when": [{"config": "model_type", "op": "in", "value": GEMMA_TYPES}],
        "claims": {},
        "note": (
            "The Gemma3 converter adds 1.0 to every norm weight; nntrainer's rms_norm multiplies by"
            " weight."
        ),
    },
    # ----------------------------------------- whole-model (reused as a unit)
    {
        "id": "gemma4_model",
        "label": "Gemma4 (per-layer embeddings, shared KV) -- whole model",
        "hooks": [
            "constructModel",
            "createTransformerDecoderBlock",
            "createAttention",
            "createMlp",
        ],
        "from": ["gemma4"],
        "level": "model",
        "model_entry": "model:gemma4",
        "when": [
            {
                "any": [
                    {"config": "model_type", "op": "in", "value": ["gemma4", "gemma4_text"]},
                    {"config": "architectures", "op": "contains", "value": "Gemma4ForCausalLM"},
                ]
            }
        ],
        "claims": {"tensors": ["**"], "config": ["*"]},
        "claims_exclude": [
            "vision_tower.**",
            "audio_tower.**",
            "multi_modal_projector.**",
            "embed_vision.**",
            "embed_audio.**",
        ],
        "note": "Matched as a whole model; individual Gemma4 features are not modelled yet.",
    },
    {
        "id": "lfm2_model",
        "label": "LFM2 (convolution + attention hybrid) -- whole model",
        "hooks": [
            "constructModel",
            "createTransformerDecoderBlock",
            "createAttention",
            "createMlp",
        ],
        "from": ["lfm2"],
        "level": "model",
        "model_entry": "model:lfm2",
        "when": [
            {
                "any": [
                    {"config": "model_type", "op": "eq", "value": "lfm2"},
                    {"config": "architectures", "op": "contains", "value": "Lfm2ForCausalLM"},
                ]
            }
        ],
        "claims": {"tensors": ["**"], "config": ["*"]},
        "note": "Matched as a whole model; individual LFM2 features are not modelled yet.",
    },
]

# Config keys that never change the computed graph (metadata, training-only,
# tokenizer ids, generation defaults).
META_CONFIG_KEYS = {
    "architectures",
    "model_type",
    "transformers_version",
    "torch_dtype",
    "dtype",
    "_name_or_path",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "sep_token_id",
    "unk_token_id",
    "decoder_start_token_id",
    "use_cache",
    "initializer_range",
    "attention_dropout",
    "hidden_dropout",
    "hidden_dropout_prob",
    "attention_probs_dropout_prob",
    "resid_pdrop",
    "embd_pdrop",
    "attn_pdrop",
    "dropout",
    "output_attentions",
    "output_hidden_states",
    "return_dict",
    "use_return_dict",
    "torchscript",
    "id2label",
    "label2id",
    "problem_type",
    "task_specific_params",
    "auto_map",
    "chunk_size_feed_forward",
    "tie_encoder_decoder",
    "is_encoder_decoder",
    "is_decoder",
    "add_cross_attention",
    "cross_attention_hidden_size",
    "cache_implementation",
    "pretraining_tp",
    "max_length",
    "min_length",
    "do_sample",
    "temperature",
    "top_k",
    "top_p",
    "num_beams",
    "repetition_penalty",
    "length_penalty",
    "no_repeat_ngram_size",
    "bad_words_ids",
    "prefix",
    "tokenizer_class",
    "finetuning_task",
    "pruned_heads",
    "tf_legacy_loss",
    "use_bfloat16",
    "router_jitter_noise",
    "keys_to_ignore_at_inference",
    "base_model_tp_plan",
    "base_model_pp_plan",
    "text_config",
    "vision_config",
    "audio_config",
    "image_token_index",
    "boi_token_index",
    "eoi_token_index",
    "mm_tokens_per_image",
    "initializer_factor",
}
