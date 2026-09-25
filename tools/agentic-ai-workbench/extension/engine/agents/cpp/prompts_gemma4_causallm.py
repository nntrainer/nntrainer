"""
Gemma4-specific prompt templates for CausalLM model generation.

Gemma4 architecture patterns:
- Pre-attention RMSNorm with sandwich norms (post_attention_norm, post_ffn_norm)
- GeGLU MLP (gate -> tanh_gelu -> multiply with up -> down)
- Per-layer sliding window alternation (sliding_attention vs full_attention)
- QK normalization (reshaped_rms_norm on Q and K projections)
- Attention logit softcapping (attn_logit_softcapping)
- Final logit softcapping (final_logit_softcapping)
- Embedding output scaled by sqrt(hidden_size)
- KV cache sharing between layers (last N layers share KV from earlier layers)
- Per-layer input gating mechanism (per_layer_input_gate -> tanh_gelu -> multiply)
- Layer scalar multiplication at output
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from knowledge.causallm_kb import (
    render_catalog,
    render_config_members,
    render_forbidden,
    render_hooks,
    render_name_contract,
    unregistered_types,
)

# Re-use common utilities from base prompts
from .prompts_causallm import (
    GAP_MARKER_SPEC,
    SYSTEM_PROMPT,
    SELF_CHECK,
    _hard_constraints,
    correction_prompt,
    registration_prompt,
)

# ============================================================================
# GEMMA4 WORKED EXAMPLE - from gemma4_causallm.cpp (shipping)
# ============================================================================
GEMMA4_WORKED_EXAMPLE = '''\
This is `Gemma4Transformer::createAttention` from
Applications/CausalLM/models/gemma4/gemma4_causallm.cpp, verbatim and shipping.
Match its shape, spacing, and comment style.

```cpp
Tensor Gemma4Transformer::createAttention(const int layer_id, int seq_len,
                                          int n_heads, int head_dim,
                                          Tensor query, Tensor key,
                                          Tensor value) {
  (void)seq_len;
  (void)head_dim;

  const std::string Q = "layer" + std::to_string(layer_id) + "_wq";
  const std::string Q_norm = "layer" + std::to_string(layer_id) + "_q_norm";
  const std::string K = "layer" + std::to_string(layer_id) + "_wk";
  const std::string K_norm = "layer" + std::to_string(layer_id) + "_k_norm";
  const std::string V = "layer" + std::to_string(layer_id) + "_wv";
  const std::string V_norm = "layer" + std::to_string(layer_id) + "_v_norm";
  const std::string A = "layer" + std::to_string(layer_id) + "_attention";
  const std::string O = "layer" + std::to_string(layer_id) + "_attention_out";
  const std::string Q_scaled = "layer" + std::to_string(layer_id) + "_q_scaled";

  const bool is_sliding = isSlidingAttentionLayer(layer_id);
  const bool is_kv_shared_layer = isKVSharedLayer(layer_id);
  const int curr_head_dim = static_cast<int>(getAttentionHeadDim(layer_id));
  const int curr_kv_heads = static_cast<int>(getKVHeadCount(layer_id));

  // Q layer [B, S, H] -> [B, S, Nq*Dh]
  std::vector<std::string> q_params = {
    withKey("name", Q), withKey("unit", curr_head_dim * n_heads),
    withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  appendSkipPrefillIfNeeded(q_params, is_kv_shared_layer);
  LayerHandle wq(createLayer("fully_connected", q_params));
  Tensor q = wq(query);

  // K layer [B, S, H] -> [B, S, Nk*Dh]
  std::vector<std::string> k_params = {
    withKey("name", K), withKey("unit", curr_head_dim * curr_kv_heads),
    withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  appendSkipPrefillIfNeeded(k_params, is_kv_shared_layer);
  LayerHandle wk(createLayer("fully_connected", k_params));
  Tensor k = wk(key);

  // V layer [B, S, H] -> [B, S, Nk*Dh]
  std::vector<std::string> v_params = {
    withKey("name", V), withKey("unit", curr_head_dim * curr_kv_heads),
    withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  appendSkipPrefillIfNeeded(v_params, is_kv_shared_layer);
  LayerHandle wv(createLayer("fully_connected", v_params));
  Tensor v = wv(value);

  // q_norm on per-head projection [B, S, Nq*Dh]
  std::vector<std::string> q_norm_params = {
    withKey("name", Q_norm), withKey("packed", "false"),
    withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(curr_head_dim))};
  appendSkipPrefillIfNeeded(q_norm_params, is_kv_shared_layer);
  LayerHandle q_norm(createLayer("reshaped_rms_norm", q_norm_params));
  Tensor q_normed = q_norm(q);

  // Gemma4TextAttention uses scaling=1.0 after q_norm/k_norm.
  // mha_core backend applies 1/sqrt(head_dim) to QK, so pre-scale Q by
  // sqrt(head_dim) to preserve Gemma4 semantics.
  LayerHandle q_scale(createLayer(
    "scalar_multiply",
    {withKey("name", Q_scaled), withKey("packed", "false"),
     withKey("multiplier",
             std::to_string(std::sqrt(static_cast<float>(curr_head_dim))))}));
  Tensor q_scaled = q_scale(q_normed);

  // k_norm on per-head projection [B, S, Nk*Dh]
  std::vector<std::string> k_norm_params = {
    withKey("name", K_norm), withKey("packed", "false"),
    withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(curr_head_dim))};
  appendSkipPrefillIfNeeded(k_norm_params, is_kv_shared_layer);
  LayerHandle k_norm(createLayer("reshaped_rms_norm", k_norm_params));
  Tensor k_normed = k_norm(k);

  // v_norm on per-head projection [B, S, Nk*Dh] (no learned scale)
  std::vector<std::string> v_norm_params = {
    withKey("name", V_norm), withKey("packed", "false"),
    withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("feature_size", std::to_string(curr_head_dim))};
  v_norm_params.push_back(withKey("use_gamma", "false"));
  appendSkipPrefillIfNeeded(v_norm_params, is_kv_shared_layer);
  LayerHandle v_norm(createLayer("reshaped_rms_norm", v_norm_params));
  Tensor v_normed = v_norm(v);

  if (layer_id >= static_cast<int>(layer_k_norms.size())) {
    layer_k_norms.resize(layer_id + 1);
    layer_v_norms.resize(layer_id + 1);
  }
  layer_k_norms[layer_id] = k_normed;
  layer_v_norms[layer_id] = v_normed;

  unsigned int window_size = is_sliding ? SLIDING_WINDOW : UINT_MAX;
  unsigned int rope_theta =
    is_sliding ? SLIDING_ATTENTION_ROPE_THETA : FULL_ATTENTION_ROPE_THETA;
  const std::string &rope_type =
    is_sliding ? SLIDING_ATTENTION_ROPE_TYPE : FULL_ATTENTION_ROPE_TYPE;
  const float rope_partial_rotary_factor =
    is_sliding ? SLIDING_ATTENTION_ROPE_PARTIAL_ROTARY_FACTOR
               : FULL_ATTENTION_ROPE_PARTIAL_ROTARY_FACTOR;

  auto [cache_k, cache_v] =
    createGemma4KVCachePlaceholders(layer_id, getKVCacheWidth(layer_id));

  // Attention core receives [Q_norm, K_norm, V_norm].
  std::vector<std::string> a_params = {
    withKey("name", A),
    withKey("num_heads", n_heads),
    withKey("num_heads_kv", curr_kv_heads),
    withKey("max_timestep", std::to_string(MAX_SEQ_LEN)),
    withKey("max_position_embeddings", std::to_string(MAX_POSITION_EMBEDDINGS)),
    withKey("sliding_window", window_size),
    withKey("use_rope", "true"),
    withKey("rope_theta", std::to_string(rope_theta)),
    withKey("rope_scaling_type", rope_type),
    withKey("rope_partial_rotary_factor",
            std::to_string(rope_partial_rotary_factor)),
    withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
    withKey("attn_logit_softcapping", std::to_string(ATTN_LOGIT_SOFTCAPPING)),
    withKey("is_causal", IS_CAUSAL ? "true" : "false")};
  appendSkipPrefillIfNeeded(a_params, is_kv_shared_layer);
  LayerHandle mha(createLayer("mha_core", a_params));
  Tensor a = mha({q_scaled, k_normed, v_normed, cache_k, cache_v});

  // O layer [B, S, Nq*Dh] -> [B, S, H]
  std::vector<std::string> o_params = {withKey("name", O), withKey("unit", DIM),
                                       withKey("disable_bias", "true"),
                                       withKey("weight_initializer", "ones"),
                                       withKey("weight_dtype", FC_LAYER_DTYPE)};
  appendSkipPrefillIfNeeded(o_params, is_kv_shared_layer);
  LayerHandle wo(createLayer("fully_connected", o_params));

  return wo(a);
}
```

Note Gemma4-specific patterns:
  * `weight_dtype` property on fully_connected layers (FC_LAYER_DTYPE)
  * `attn_logit_softcapping` in mha_core
  * Per-layer `window_size` and `rope_theta` from isSlidingAttentionLayer()
  * QK normalization with `reshaped_rms_norm` and `feature_size`
  * Q scaling by sqrt(head_dim) before attention
  * V normalization with `use_gamma = "false"`
  * KV cache from `createGemma4KVCachePlaceholders()` (not base class)
  * `appendSkipPrefillIfNeeded()` for skip_prefill optimization
  * LayerHandle called then invoked: `Tensor q = wq(query);`
'''

# ============================================================================
# GEMMA4 MLP WORKED EXAMPLE (GeGLU)
# ============================================================================
GEMMA4_MLP_EXAMPLE = '''\
Gemma4 uses GeGLU (gate -> tanh_gelu -> multiply with up -> down)

```cpp
Tensor Gemma4Transformer::createMlp(const int layer_id, int dim, int hidden_dim,
                                    Tensor input) {
  const bool is_kv_shared_layer = isKVSharedLayer(layer_id);
  const int curr_hidden_dim =
    hidden_dim * ((USE_DOUBLE_WIDE_MLP && is_kv_shared_layer) ? 2 : 1);

  std::vector<std::string> ffn_gate_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate"),
    withKey("unit", curr_hidden_dim), withKey("disable_bias", "true"),
    withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  appendSkipPrefillIfNeeded(ffn_gate_props, is_kv_shared_layer);
  LayerHandle ffn_gate(createLayer("fully_connected", ffn_gate_props));
  Tensor gate = ffn_gate(input);

  std::vector<std::string> ffn_gate_gelu_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate_gelu"),
    withKey("activation", "tanh_gelu")};
  appendSkipPrefillIfNeeded(ffn_gate_gelu_props, is_kv_shared_layer);
  LayerHandle ffn_gate_gelu(createLayer("activation", ffn_gate_gelu_props));
  Tensor gate_gelu = ffn_gate_gelu(gate);

  std::vector<std::string> ffn_up_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_up"),
    withKey("unit", curr_hidden_dim), withKey("disable_bias", "true"),
    withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  appendSkipPrefillIfNeeded(ffn_up_props, is_kv_shared_layer);
  LayerHandle ffn_up(createLayer("fully_connected", ffn_up_props));
  Tensor up = ffn_up(input);

  std::vector<std::string> ffn_geglu_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_geglu")};
  appendSkipPrefillIfNeeded(ffn_geglu_props, is_kv_shared_layer);
  LayerHandle ffn_geglu(createLayer("multiply", ffn_geglu_props));
  Tensor geglu = ffn_geglu({gate_gelu, up});

  std::vector<std::string> ffn_down_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
    withKey("unit", dim), withKey("disable_bias", "true"),
    withKey("weight_initializer", "ones"),
    withKey("weight_dtype", FC_LAYER_DTYPE)};
  appendSkipPrefillIfNeeded(ffn_down_props, is_kv_shared_layer);
  LayerHandle ffn_down(createLayer("fully_connected", ffn_down_props));

  return ffn_down(geglu);
}
```

Key GeGLU pattern:
  * Uses `activation` layer with `tanh_gelu` (NOT swiglu, NOT gelu)
  * Uses `multiply` layer to combine gate_gelu * up
  * Order: gate -> tanh_gelu -> multiply with up -> down
  * `weight_dtype = FC_LAYER_DTYPE` on all fully_connected layers
  * `USE_DOUBLE_WIDE_MLP` may double hidden_dim for KV-shared layers
'''

# ============================================================================
# GEMMA4 DECODER BLOCK WORKED EXAMPLE (sandwich norms + per-layer input)
# ============================================================================
GEMMA4_BLOCK_EXAMPLE = '''\
Gemma4 decoder block has sandwich norms and per-layer input gating

```cpp
Tensor Gemma4Transformer::createTransformerDecoderBlock(const int layer_id,
                                                        Tensor input) {

  // Gemma4TextRMSNorm scales by `weight` (initialized to ones)
  const bool is_kv_shared_layer = isKVSharedLayer(layer_id);
  std::vector<std::string> attn_norm_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false")};
  appendSkipPrefillIfNeeded(attn_norm_props, is_kv_shared_layer);
  LayerHandle attn_norm(createLayer("rms_norm", attn_norm_props));
  Tensor normed = attn_norm(input);

  int shared_kv_layer_id = -1;
  // ... (KV sharing logic to find source layer)

  Tensor att_out;
  if (shared_kv_layer_id >= 0) {
    att_out = createSharedAttention(layer_id, shared_kv_layer_id, ...);
  } else {
    att_out = createAttention(layer_id, ...);
  }

  // Post-attention norm
  std::vector<std::string> post_attn_norm_props = {
    withKey("name",
            "layer" + std::to_string(layer_id) + "_post_attention_norm"),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false")};
  appendSkipPrefillIfNeeded(post_attn_norm_props, is_kv_shared_layer);
  LayerHandle post_attn_norm(createLayer("rms_norm", post_attn_norm_props));
  Tensor post_normed = post_attn_norm(att_out);

  // Post-attention residual
  std::vector<std::string> post_attention_add_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_post_attention")};
  appendSkipPrefillIfNeeded(post_attention_add_props, is_kv_shared_layer);
  LayerHandle post_attention_add(
    createLayer("addition", post_attention_add_props));
  Tensor post_attention = post_attention_add({input, post_normed});

  // Pre-FFN norm
  std::vector<std::string> pre_ffn_norm_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_pre_ffn_norm"),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false")};
  appendSkipPrefillIfNeeded(pre_ffn_norm_props, is_kv_shared_layer);
  LayerHandle pre_ffn_norm(createLayer("rms_norm", pre_ffn_norm_props));
  Tensor pre_ffn = pre_ffn_norm(post_attention);

  Tensor ffn_out = createMlp(layer_id, DIM, INTERMEDIATE_SIZE, pre_ffn);

  // Post-FFN norm
  std::vector<std::string> post_ffn_norm_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_post_ffn_norm"),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false")};
  appendSkipPrefillIfNeeded(post_ffn_norm_props, is_kv_shared_layer);
  LayerHandle post_ffn_norm(createLayer("rms_norm", post_ffn_norm_props));
  Tensor post_ffn = post_ffn_norm(ffn_out);

  // ... (per-layer input gating mechanism follows)
}
```

Note the sandwich norm pattern:
  1. Pre-attention norm -> attention -> Post-attention norm -> residual
  2. Pre-FFN norm -> MLP -> Post-FFN norm -> residual
  3. Uses `rms_norm` (not `layer_normalization`)
  4. `appendSkipPrefillIfNeeded()` on all layer properties
'''

# ============================================================================
# GEMMA4 SETUPPARAMETERS EXAMPLE
# ============================================================================
GEMMA4_SETUP_EXAMPLE = '''\
Gemma4 reads layer_types and softcapping values from config

```cpp
void Gemma4Transformer::setupParameters(json &cfg, json &generation_cfg,
                                        json &nntr_cfg) {
  Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);

  if (cfg.contains("layer_types")) {
    layer_types = cfg["layer_types"].get<std::vector<std::string>>();
  }

  if (cfg.contains("attn_logit_softcapping") &&
      !cfg["attn_logit_softcapping"].is_null()) {
    ATTN_LOGIT_SOFTCAPPING = cfg["attn_logit_softcapping"].get<float>();
  }
  if (cfg.contains("final_logit_softcapping") &&
      !cfg["final_logit_softcapping"].is_null()) {
    FINAL_LOGIT_SOFTCAPPING = cfg["final_logit_softcapping"].get<float>();
  }

  GLOBAL_HEAD_DIM =
    cfg.contains("global_head_dim") && !cfg["global_head_dim"].is_null()
      ? cfg["global_head_dim"].get<unsigned int>()
      : HEAD_DIM;

  NUM_GLOBAL_KEY_VALUE_HEADS =
    cfg.contains("num_global_key_value_heads") &&
        !cfg["num_global_key_value_heads"].is_null()
      ? cfg["num_global_key_value_heads"].get<unsigned int>()
      : NUM_KEY_VALUE_HEADS;

  ATTENTION_K_EQ_V =
    cfg.contains("attention_k_eq_v") && cfg["attention_k_eq_v"].get<bool>();

  // ... (more config reads)

  EMBEDDING_SCALE = std::sqrt(static_cast<float>(DIM));
  EMBEDDING_PER_LAYER_SCALE =
    std::sqrt(static_cast<float>(HIDDEN_SIZE_PER_LAYER_INPUT));
}
```
'''

# ============================================================================
# GEMMA4-SPECIFIC HARD CONSTRAINTS
# ============================================================================
def gemma4_hard_constraints(plan: Dict) -> str:
    """Gemma4-specific constraints on top of the common ones."""
    common = _hard_constraints(plan)
    return f"""\
{common}

G4. GEMMA4-SPECIFIC PATTERNS
    * MLP uses GeGLU (gate -> tanh_gelu -> multiply with up), NOT SwiGLU
    * Decoder block has sandwich norms: post_attention_norm AND post_ffn_norm
    * layer_types array determines per-layer sliding_window and rope_theta:
      - "sliding_attention": sliding_window = SLIDING_WINDOW
      - "full_attention": sliding_window = UINT_MAX
    * Attention logit softcapping via ATTN_LOGIT_SOFTCAPPING constant
    * Final logit softcapping via FINAL_LOGIT_SOFTCAPPING constant
    * Embedding scale: EMBEDDING_SCALE = sqrt(DIM), set in setupParameters
    * fully_connected layers have weight_dtype = FC_LAYER_DTYPE property
    * Q scaling by sqrt(head_dim) before attention via scalar_multiply
    * V normalization uses use_gamma = "false" (no learned scale)
    * KV cache from createGemma4KVCachePlaceholders() (not base class method)
    * appendSkipPrefillIfNeeded() on all layer properties for skip_prefill
    * isKVSharedLayer() determines KV cache sharing for last N layers

G5. GEMMA4 FORBIDDEN PATTERNS
    * Do NOT use swiglu layer - Gemma4 uses tanh_gelu + multiply
    * Do NOT use a single uniform sliding_window for all layers
    * Do NOT omit post_attention_norm or post_ffn_norm
    * Do NOT use silu/swish or gelu activation - must be tanh_gelu
    * Do NOT use HF dotted names like "model.layers.0.self_attn.q_proj"
    * Do NOT forget weight_dtype on fully_connected layers
    * Do NOT use base class createKVCachePlaceholders() - use Gemma4 version
    * Do NOT forget appendSkipPrefillIfNeeded() on layer properties
    * Do NOT use layer_normalization - use rms_norm
    * Do NOT forget Q scaling by sqrt(head_dim) before attention
"""

# ============================================================================
# GEMMA4 GENERATION PROMPT
# ============================================================================
def gemma4_generation_prompt(plan: Dict, skeleton_h: str, skeleton_cpp: str) -> str:
    """
    Gemma4-specific generation prompt.
    
    Uses Gemma4 worked examples and constraints.
    """
    holes = plan.get("holes", [])
    hole_block = "\n".join(
        f"  FILL:{h['id']}  {h['hook']}\n"
        f"           goal   : {h['goal']}\n"
        f"           layers : {', '.join(h.get('layers', [])) or '(see plan)'}"
        for h in holes
    ) or "  (none -- see skeleton markers)"

    hook_names = [h["hook"] for h in holes if h.get("hook")]

    return f"""\
Fill in {len(holes)} marked hole(s) in the C++ skeleton below. Nothing else.

Model      : {plan.get('model_id', '<unknown>')}
Architecture: {plan.get('architecture', '<unknown>')}
Class      : {plan.get('class_name', '<unknown>')}
Base       : {plan.get('base_class', 'causallm::CausalLM')}

This is a GEMMA4-family model. Match the Gemma4 architecture patterns:
  * GeGLU MLP (gate -> tanh_gelu -> multiply with up)
  * Sandwich norms (post_attention_norm, post_ffn_norm)
  * Per-layer sliding window alternation via layer_types
  * QK normalization with reshaped_rms_norm
  * Attention logit softcapping
  * Embedding scale = sqrt(DIM)
  * KV cache sharing between layers
  * Per-layer input gating mechanism

The base classes have ZERO pure virtual methods and already build the complete
llama-shaped graph. You are overriding ONLY where this architecture deviates.

================================ 1. THE PLAN =================================
{json.dumps(plan.get('resolved', plan), indent=2)}

HOLES TO FILL:
{hole_block}

============================== 2. THE SKELETON ===============================
--- {plan.get('file_stem', 'model')}_causallm.h ---
```cpp
{skeleton_h}
```

--- {plan.get('file_stem', 'model')}_causallm.cpp ---
```cpp
{skeleton_cpp}
```

============================== 3. THE LAYER MENU =============================
{render_catalog(only=plan.get('layer_types'))}

=========================== 4. HARD CONSTRAINTS ==============================
{gemma4_hard_constraints(plan)}

======================= 5. OVERRIDABLE HOOKS (exact) ========================
{render_hooks(only=hook_names or None)}

=========================== 6. WORKED EXAMPLES ===============================

{GEMMA4_WORKED_EXAMPLE}

{GEMMA4_MLP_EXAMPLE}

{GEMMA4_BLOCK_EXAMPLE}

{GEMMA4_SETUP_EXAMPLE}

============================= 7. GAP MARKER =================================
{GAP_MARKER_SPEC}

============================= 8. SELF-CHECK =================================
{SELF_CHECK}

============================ 9. OUTPUT FORMAT ================================
Emit exactly two fenced blocks, in this order:

```cpp
// FILE: {plan.get('file_stem', 'model')}_causallm.h
<complete header>
```

```cpp
// FILE: {plan.get('file_stem', 'model')}_causallm.cpp
<complete source>
```
"""
