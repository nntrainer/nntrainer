"""
Qwen2-specific prompt templates for CausalLM model generation.

Qwen2 architecture patterns:
- Pre-attention RMSNorm (before attention)
- Standard residual connections (no sandwich norms)
- SwiGLU MLP (gate_proj + up_proj -> swiglu -> down_proj)
- QK normalization NOT used (unlike Qwen3)
- Uniform sliding window (same for all layers, if present)
- RoPE with standard theta
- No attention logit softcapping
- Bias enabled on Q/K/V projections (unlike Qwen3)
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
# QWEN2 WORKED EXAMPLE - from qwen2_causallm.cpp (shipping)
# ============================================================================
QWEN2_WORKED_EXAMPLE = '''\
This is `Qwen2Transformer::createAttention` from
Applications/CausalLM/models/qwen2/qwen2_causallm.cpp, verbatim and shipping.
Match its shape, spacing, and comment style.

```cpp
Tensor Qwen2Transformer::createAttention(const int layer_id, int seq_len,
                                         int n_heads, int head_dim,
                                         Tensor query, Tensor key,
                                         Tensor value) {

  // Q layer (qwen2 uses bias on Q/K/V)
  LayerHandle wq(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wq"),
     withKey("unit", head_dim * n_heads), withKey("disable_bias", "false"),
     withKey("weight_initializer", "ones")}));
  Tensor q = wq(query);

  // K layer
  LayerHandle wk(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wk"),
     withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "false"), withKey("weight_initializer", "ones")}));
  Tensor k = wk(key);

  // V layer
  LayerHandle wv(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_wv"),
     withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "false"), withKey("weight_initializer", "ones")}));
  Tensor v = wv(value);

  // External KV cache placeholders (per-layer)
  auto [cache_k, cache_v] = createKVCachePlaceholders(layer_id, n_heads);

  // Attention core layer
  LayerHandle mha(createLayer(
    "mha_core",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention"),
     withKey("num_heads", n_heads), withKey("num_heads_kv", n_heads / GQA_SIZE),
     withKey("max_timestep", std::to_string(MAX_SEQ_LEN)),
     withKey("sliding_window", SLIDING_WINDOW),
     withKey("rope_theta", ROPE_THETA),
     withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
     withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
     withKey("is_causal", IS_CAUSAL ? "true" : "false")}));
  Tensor a = mha({q, k, v, cache_k, cache_v});

  // O layer
  LayerHandle wo(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_out"),
     withKey("unit", DIM), withKey("disable_bias", "true"),
     withKey("weight_initializer", "ones")}));
  return wo(a);
}
```

Note Qwen2-specific patterns:
  * Bias enabled on Q/K/V projections: `disable_bias = "false"`
  * No QK normalization (unlike Qwen3) - Q/K/V go directly to mha_core
  * No `attn_logit_softcapping` property
  * No `weight_dtype` property on fully_connected layers
  * LayerHandle called then invoked: `Tensor q = wq(query);`
'''

# ============================================================================
# QWEN2 MLP WORKED EXAMPLE (SwiGLU)
# ============================================================================
QWEN2_MLP_EXAMPLE = '''\
Qwen2 uses SwiGLU (gate + up -> swiglu with index remap -> down)

```cpp
Tensor Qwen2Transformer::createMlp(const int layer_id, int dim, int hidden_dim,
                                   Tensor input) {

  // Gate projection
  LayerHandle ffn_gate(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate"),
     withKey("unit", hidden_dim), withKey("disable_bias", "false"),
     withKey("weight_initializer", "ones")}));
  Tensor gate = ffn_gate(input);

  // Up projection
  LayerHandle ffn_up(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_up"),
     withKey("unit", hidden_dim), withKey("disable_bias", "false"),
     withKey("weight_initializer", "ones")}));
  Tensor up = ffn_up(input);

  // SwiGLU activation
  // IMPORTANT: {up, gate} order with {1, 0} index remap is mandatory
  LayerHandle swiglu_layer(createLayer(
    "swiglu",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_swiglu")}));
  Tensor act = swiglu_layer({up, gate}, {1, 0});

  // Down projection
  LayerHandle ffn_down(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("unit", dim), withKey("disable_bias", "false"),
     withKey("weight_initializer", "ones")}));
  return ffn_down(act);
}
```

Key SwiGLU pattern:
  * Uses `swiglu` layer (NOT activation + multiply)
  * Input order is `{up, gate}` with index remap `{1, 0}`
  * The remap is MANDATORY - layer reads input[0] as gate but weights
    are stored in up,gate order, so the remap fixes the wiring
  * Bias enabled on projections: `disable_bias = "false"`
'''

# ============================================================================
# QWEN2 DECODER BLOCK WORKED EXAMPLE (standard pre-norm)
# ============================================================================
QWEN2_BLOCK_EXAMPLE = '''\
Qwen2 decoder block uses standard pre-norm pattern (no sandwich norms)

```cpp
Tensor Qwen2Transformer::createTransformerDecoderBlock(const int layer_id,
                                                       Tensor input) {
  // Pre-attention norm
  LayerHandle attn_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  Tensor normed = attn_norm(input);

  Tensor att_out = createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
                                   normed, normed, normed);

  // Post-attention residual add
  LayerHandle decoder_add(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_add")}));
  Tensor post_attn = decoder_add({input, att_out});

  // Pre-FFN norm
  LayerHandle ffn_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  Tensor ffn_in = ffn_norm(post_attn);

  Tensor ffn_out = createMlp(layer_id, DIM, INTERMEDIATE_SIZE, ffn_in);

  // Post-FFN residual add
  LayerHandle decoder_output(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_output")}));
  return decoder_output({post_attn, ffn_out});
}
```

Note the standard pre-norm pattern:
  1. Pre-attention norm -> attention -> residual (no post-norm)
  2. Pre-FFN norm -> MLP -> residual (no post-norm)
'''

# ============================================================================
# QWEN2-SPECIFIC HARD CONSTRAINTS
# ============================================================================
def qwen2_hard_constraints(plan: Dict) -> str:
    """Qwen2-specific constraints on top of the common ones."""
    common = _hard_constraints(plan)
    return f"""\
{common}

Q2. QWEN2-SPECIFIC PATTERNS
    * MLP uses SwiGLU with mandatory index remap: `swiglu({{up, gate}}, {{1, 0}})`
    * Standard pre-norm decoder block (NO sandwich norms)
    * Uniform sliding_window for all layers (no per-layer variation)
    * NO QK normalization - Q/K/V projections go directly to mha_core
    * No attn_logit_softcapping in mha_core
    * No weight_dtype property on fully_connected layers
    * Bias ENABLED on Q/K/V projections: `disable_bias = "false"`
    * Bias DISABLED on O projection: `disable_bias = "true"`

Q3. QWEN2 FORBIDDEN PATTERNS
    * Do NOT use tanh_gelu + multiply - use swiglu layer with remap
    * Do NOT add post_attention_norm or post_ffn_norm
    * Do NOT use per-layer sliding_window or rope_theta variation
    * Do NOT use attn_logit_softcapping property
    * Do NOT use HF dotted names like "model.layers.0.self_attn.q_proj"
    * Do NOT forget the {{1, 0}} index remap on swiglu - it is mandatory
    * Do NOT use reshaped_rms_norm on Q/K - Qwen2 does NOT have QK normalization
    * Do NOT set disable_bias="true" on Q/K/V projections - must be "false"
"""

# ============================================================================
# QWEN2 GENERATION PROMPT
# ============================================================================
def qwen2_generation_prompt(plan: Dict, skeleton_h: str, skeleton_cpp: str) -> str:
    """
    Qwen2-specific generation prompt.
    
    Uses Qwen2 worked examples and constraints.
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

This is a QWEN2-family model. Match the Qwen2 architecture patterns:
  * SwiGLU MLP with mandatory index remap: swiglu({up, gate}, {1, 0})
  * Standard pre-norm decoder block (no sandwich norms)
  * Uniform sliding window for all layers
  * NO QK normalization (unlike Qwen3)
  * Bias enabled on Q/K/V projections

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
{qwen2_hard_constraints(plan)}

======================= 5. OVERRIDABLE HOOKS (exact) ========================
{render_hooks(only=hook_names or None)}

=========================== 6. WORKED EXAMPLES ===============================

{QWEN2_WORKED_EXAMPLE}

{QWEN2_MLP_EXAMPLE}

{QWEN2_BLOCK_EXAMPLE}

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
