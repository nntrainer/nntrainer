"""
Gemma3-specific prompt templates for CausalLM model generation.

Gemma3 architecture patterns:
- Pre-attention RMSNorm (before attention)
- Post-attention sandwich norm (re-normalize AFTER attention, before residual)
- Post-MLP sandwich norm (re-normalize AFTER MLP, before residual)
- GeGLU MLP (gate_proj -> tanh_gelu, up_proj -> multiply)
- Per-layer sliding window alternation (sliding_attention vs full_attention)
- QK normalization (reshaped_rms_norm on Q and K projections)
- Embedding output scaled by sqrt(hidden_size)
- Attention logit softcapping
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
# GEMMA3 WORKED EXAMPLE - from gemma3_causallm.cpp (shipping)
# ============================================================================
GEMMA3_WORKED_EXAMPLE = '''\
This is `Gemma3Transformer::createAttention` from
Applications/CausalLM/models/gemma3/gemma3_causallm.cpp, verbatim and shipping.
Match its shape, spacing, and comment style.

```cpp
Tensor Gemma3Transformer::createAttention(const int layer_id, int seq_len,
                                          int n_heads, int head_dim,
                                          Tensor query, Tensor key,
                                          Tensor value) {

  // Q layer
  LayerHandle wq(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wq"),
     withKey("unit", head_dim * n_heads), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  Tensor q = wq(query);

  // K layer
  LayerHandle wk(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wk"),
     withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  Tensor k = wk(key);

  // V layer
  LayerHandle wv(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wv"),
     withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "true"), withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  Tensor v = wv(value);

  // q_norm
  LayerHandle q_norm(createLayer(
    "reshaped_rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_q_norm"),
     withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("feature_size", std::to_string(head_dim))}));
  Tensor q_normed = q_norm(q);

  // k_norm
  LayerHandle k_norm(createLayer(
    "reshaped_rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_k_norm"),
     withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("feature_size", std::to_string(head_dim))}));
  Tensor k_normed = k_norm(k);

  // Attention core layer - handles sliding_window and rope_theta per-layer
  unsigned int window_size = UINT_MAX;
  if (!layer_types.empty()) {
    if (layer_id < layer_types.size()) {
      if (layer_types[layer_id] == "sliding_attention") {
        window_size = SLIDING_WINDOW;
      }
    }
  }

  float rope_theta = ROPE_THETA;
  if (!layer_types.empty() && layer_id < layer_types.size()) {
    if (layer_types[layer_id] == "sliding_attention") {
      rope_theta = 10000.0f;
    }
  }

  auto [cache_k, cache_v] = createKVCachePlaceholders(layer_id, n_heads);

  LayerHandle mha(createLayer(
    "mha_core",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention"),
     withKey("num_heads", n_heads), withKey("num_heads_kv", n_heads / GQA_SIZE),
     withKey("max_timestep", std::to_string(MAX_SEQ_LEN)),
     withKey("sliding_window", window_size),
     withKey("rope_theta", std::to_string(rope_theta)),
     withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
     withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
     withKey("attn_logit_softcapping", std::to_string(ATTN_LOGIT_SOFTCAPPING)),
     withKey("is_causal", IS_CAUSAL ? "true" : "false")}));
  Tensor a = mha({q_normed, k_normed, v, cache_k, cache_v});

  // O layer
  LayerHandle wo(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_out"),
     withKey("unit", DIM), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  return wo(a);
}
```

Note Gemma3-specific patterns:
  * `weight_dtype` property on fully_connected layers (FC_LAYER_DTYPE)
  * `attn_logit_softcapping` in mha_core
  * Per-layer `window_size` and `rope_theta` from `layer_types` array
  * QK norm uses `reshaped_rms_norm` with `feature_size`
  * LayerHandle called then invoked: `Tensor q = wq(query);`
'''

# ============================================================================
# GEMMA3 MLP WORKED EXAMPLE (GeGLU, not SwiGLU)
# ============================================================================
GEMMA3_MLP_EXAMPLE = '''\
Gemma3 uses GeGLU (gate -> tanh_gelu -> multiply with up), NOT SwiGLU

```cpp
Tensor Gemma3Transformer::createMlp(const int layer_id, int dim, int hidden_dim,
                                    Tensor input) {

  // Gate projection
  LayerHandle ffn_gate(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate"),
     withKey("unit", hidden_dim), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  Tensor gate = ffn_gate(input);

  // GeLU activation (tanh_gelu for Gemma3)
  LayerHandle gelu(createLayer(
    "activation",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate_gelu"),
     withKey("activation", "tanh_gelu")}));
  Tensor gate_gelu = gelu(gate);

  // Up projection
  LayerHandle ffn_up(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_up"),
     withKey("unit", hidden_dim), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  Tensor up = ffn_up(input);

  // Multiply (GeGLU = gate_gelu * up)
  LayerHandle mul(createLayer(
    "multiply",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_geglu")}));
  Tensor geglu = mul({gate_gelu, up});

  // Down projection
  LayerHandle ffn_down(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("unit", dim), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  return ffn_down(geglu);
}
```

Key differences from SwiGLU:
  * Uses `activation` layer with `tanh_gelu` (NOT `swiglu` layer)
  * Uses `multiply` layer to combine gate_gelu * up
  * Order: gate -> gelu -> multiply with up -> down
'''

# ============================================================================
# GEMMA3 DECODER BLOCK WORKED EXAMPLE (sandwich norms)
# ============================================================================
GEMMA3_BLOCK_EXAMPLE = '''\
Gemma3 decoder block has sandwich norms (post-attention and post-MLP norms)

```cpp
Tensor Gemma3Transformer::createTransformerDecoderBlock(const int layer_id,
                                                        Tensor input) {

  LayerHandle attn_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  Tensor normed = attn_norm(input);

  Tensor att_out = createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
                                   normed, normed, normed);

  LayerHandle post_attn_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_post_attention_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  Tensor post_normed = post_attn_norm(att_out);

  LayerHandle post_attn_add(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_post_attention")}));
  Tensor post_attn = post_attn_add({input, post_normed});

  LayerHandle pre_ffn_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_pre_ffn_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  Tensor pre_ffn = pre_ffn_norm(post_attn);

  Tensor ffn_out = createMlp(layer_id, DIM, INTERMEDIATE_SIZE, pre_ffn);

  LayerHandle post_ffn_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_post_ffn_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  Tensor post_ffn = post_ffn_norm(ffn_out);

  LayerHandle decoder_output(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_output")}));
  return decoder_output({post_attn, post_ffn});
}
```

Note the sandwich norm pattern:
  1. Pre-attention norm (attn_norm) -> attention -> Post-attention norm -> residual
  2. Pre-FFN norm (pre_ffn_norm) -> MLP -> Post-FFN norm -> residual
'''

# ============================================================================
# GEMMA3 SETUPPARAMETERS EXAMPLE
# ============================================================================
GEMMA3_SETUP_EXAMPLE = '''\
Gemma3 reads layer_types and attn_logit_softcapping from config

```cpp
void Gemma3Transformer::setupParameters(json &cfg, json &generation_cfg,
                                        json &nntr_cfg) {
  Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);
  EMBEDDING_SCALE = std::sqrt(static_cast<float>(DIM));
  if (cfg.contains("layer_types")) {
    layer_types = cfg["layer_types"].get<std::vector<std::string>>();
  }
  if (cfg.contains("attn_logit_softcapping") &&
      !cfg["attn_logit_softcapping"].is_null()) {
    ATTN_LOGIT_SOFTCAPPING = cfg["attn_logit_softcapping"].get<float>();
  }
}
```
'''

# ============================================================================
# GEMMA3-SPECIFIC HARD CONSTRAINTS
# ============================================================================
def gemma3_hard_constraints(plan: Dict) -> str:
    """Gemma3-specific constraints on top of the common ones."""
    common = _hard_constraints(plan)
    return f"""\
{common}

G3. GEMMA3-SPECIFIC PATTERNS
    * MLP uses GeGLU (gate -> tanh_gelu -> multiply with up), NOT SwiGLU
    * Decoder block has sandwich norms: post_attention_norm AND post_ffn_norm
    * layer_types array determines per-layer sliding_window and rope_theta:
      - "sliding_attention": sliding_window = SLIDING_WINDOW, rope_theta = 10000.0
      - "full_attention": sliding_window = UINT_MAX, rope_theta = ROPE_THETA
    * Attention logit softcapping via ATTN_LOGIT_SOFTCAPPING constant
    * Embedding scale: EMBEDDING_SCALE = sqrt(DIM), set in setupParameters
    * fully_connected layers have weight_dtype = FC_LAYER_DTYPE property
    * LayerHandle is called then invoked: `Tensor t = h(input);`

G4. GEMMA3 FORBIDDEN PATTERNS
    * Do NOT use swiglu layer - Gemma3 uses tanh_gelu + multiply
    * Do NOT use a single uniform sliding_window for all layers
    * Do NOT omit post_attention_norm or post_ffn_norm
    * Do NOT use silu/swish activation - must be tanh_gelu
    * Do NOT use HF dotted names like "model.layers.0.self_attn.q_proj"
"""

# ============================================================================
# GEMMA3 GENERATION PROMPT
# ============================================================================
def gemma3_generation_prompt(plan: Dict, skeleton_h: str, skeleton_cpp: str) -> str:
    """
    Gemma3-specific generation prompt.
    
    Uses Gemma3 worked examples and constraints instead of generic/Qwen3.
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

This is a GEMMA3-family model. Match the Gemma3 architecture patterns:
  * GeGLU MLP (gate -> tanh_gelu -> multiply with up)
  * Sandwich norms (post_attention_norm, post_ffn_norm)
  * Per-layer sliding window alternation via layer_types
  * QK normalization with reshaped_rms_norm
  * Attention logit softcapping
  * Embedding scale = sqrt(DIM)

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
{gemma3_hard_constraints(plan)}

======================= 5. OVERRIDABLE HOOKS (exact) ========================
{render_hooks(only=hook_names or None)}

=========================== 6. WORKED EXAMPLES ===============================

{GEMMA3_WORKED_EXAMPLE}

{GEMMA3_MLP_EXAMPLE}

{GEMMA3_BLOCK_EXAMPLE}

{GEMMA3_SETUP_EXAMPLE}

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
