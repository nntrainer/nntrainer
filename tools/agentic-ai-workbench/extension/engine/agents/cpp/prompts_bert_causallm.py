"""
BERT-specific prompt templates for CausalLM model generation.

BERT architecture patterns:
- Layer normalization (not RMSNorm)
- Position embeddings + token type embeddings
- Bidirectional attention (not causal)
- No RoPE (rope_theta = 0, use_rope = false)
- GeLU activation (not SwiGLU)
- Post-norm pattern (residual -> LayerNorm)
- Bias enabled on all projections
- No attention logit softcapping
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
# BERT SETUP HELPER - for precise epsilon values
# ============================================================================
BERT_SETUP_NOTE = '''\
BERT uses very small epsilon values (layer_norm_eps = 1e-12).
Use a helper function to convert floats to strings with enough precision:

```cpp
namespace {
std::string toStringPrecise(float v) {
  std::ostringstream oss;
  oss << std::setprecision(20) << v;
  return oss.str();
}
} // namespace
```

Then use: withKey("epsilon", toStringPrecise(NORM_EPS))
'''

# ============================================================================
# BERT WORKED EXAMPLE - from bert_transformer.cpp (shipping)
# ============================================================================
BERT_WORKED_EXAMPLE = '''\
This is `BertTransformer::createAttention` from
Applications/CausalLM/models/bert/bert_transformer.cpp, verbatim and shipping.
Match its shape, spacing, and comment style.

```cpp
Tensor BertTransformer::createAttention(const int layer_id, int seq_len,
                                        int n_heads, int head_dim, Tensor query,
                                        Tensor key, Tensor value) {

  auto Q = "layer" + std::to_string(layer_id) + "_wq";
  auto K = "layer" + std::to_string(layer_id) + "_wk";
  auto V = "layer" + std::to_string(layer_id) + "_wv";
  auto A = "layer" + std::to_string(layer_id) + "_attention";
  auto O = "layer" + std::to_string(layer_id) + "_attention_out";

  // Q layer (bias enabled for BERT)
  LayerHandle wq(createLayer(
    "fully_connected",
    {withKey("name", Q), withKey("unit", head_dim * n_heads),
     withKey("disable_bias", "false"), withKey("weight_initializer", "ones")}));
  Tensor q = wq(query);

  // K layer (bias enabled for BERT)
  LayerHandle wk(createLayer(
    "fully_connected",
    {withKey("name", K), withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "false"), withKey("weight_initializer", "ones")}));
  Tensor k = wk(key);

  // V layer (bias enabled for BERT)
  LayerHandle wv(createLayer(
    "fully_connected",
    {withKey("name", V), withKey("unit", head_dim * n_heads / GQA_SIZE),
     withKey("disable_bias", "false"), withKey("weight_initializer", "ones")}));
  Tensor v = wv(value);

  // Attention core layer (bidirectional, no RoPE)
  std::vector<std::string> a_params = {
    withKey("name", A),
    withKey("num_heads", n_heads),
    withKey("num_heads_kv", n_heads / GQA_SIZE),
    withKey("max_timestep", std::to_string(INIT_SEQ_LEN)),
    withKey("rope_theta", ROPE_THETA),
    withKey("use_rope", "false"),
    withKey("is_causal", "false")};
  LayerHandle mha(createLayer("mha_core", a_params));
  Tensor a = mha({q, k, v});

  // O layer (bias enabled for BERT)
  LayerHandle wo(
    createLayer("fully_connected", {withKey("name", O), withKey("unit", DIM),
                                    withKey("disable_bias", "false"),
                                    withKey("weight_initializer", "ones")}));

  return wo(a);
}
```

Note BERT-specific patterns:
  * Bias enabled on ALL projections: `disable_bias = "false"`
  * No RoPE: `use_rope = "false"`
  * Bidirectional attention: `is_causal = "false"`
  * No KV cache (encoder-only model) - mha receives only {q, k, v}
  * No attn_logit_softcapping property
  * Layer names stored in variables (Q, K, V, A, O) for clarity
'''

# ============================================================================
# BERT MLP WORKED EXAMPLE (GeLU)
# ============================================================================
BERT_MLP_EXAMPLE = '''\
BERT uses GeLU activation (gate -> gelu -> down), NOT SwiGLU

```cpp
Tensor BertTransformer::createMlp(const int layer_id, int dim, int hidden_dim,
                                  Tensor input) {
  LayerHandle fc1(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_fc1"),
     withKey("unit", hidden_dim), withKey("disable_bias", "false"),
     withKey("weight_initializer", "ones")}));
  Tensor fc1_out = fc1(input);

  LayerHandle act(createLayer(
    "activation",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_act"),
     withKey("activation", "gelu")}));
  Tensor activated = act(fc1_out);

  LayerHandle down(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("unit", dim), withKey("disable_bias", "false"),
     withKey("weight_initializer", "ones")}));

  return down(activated);
}
```

Key MLP pattern:
  * Uses `activation` layer with `gelu` (NOT swiglu, NOT tanh_gelu)
  * Single projection -> gelu -> down projection
  * Bias enabled on all projections: `disable_bias = "false"`
'''

# ============================================================================
# BERT DECODER BLOCK WORKED EXAMPLE (post-norm)
# ============================================================================
BERT_BLOCK_EXAMPLE = '''\
BERT decoder block uses post-norm pattern (residual -> LayerNorm)

```cpp
Tensor BertTransformer::createTransformerDecoderBlock(const int layer_id,
                                                      Tensor input) {

  // Self-attention sub-block
  Tensor att_out = createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
                                   input, input, input);

  // Residual (input + attention_out) + post LayerNorm
  LayerHandle attention_res(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_res")}));
  Tensor attention_residual = attention_res({input, att_out});

  LayerHandle attention_norm(createLayer(
    "layer_normalization",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
     withKey("epsilon", toStringPrecise(NORM_EPS)), withKey("axis", 3),
     withKey("packed", "false")}));
  Tensor attention_normed = attention_norm(attention_residual);

  // Feed-forward sub-block
  auto ffn_layers =
    createMlp(layer_id, DIM, INTERMEDIATE_SIZE, attention_normed);

  // Residual (normed + ffn_down) + post LayerNorm
  LayerHandle ffn_res(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_res")}));
  Tensor ffn_residual = ffn_res({attention_normed, ffn_layers});

  LayerHandle ffn_norm(createLayer(
    "layer_normalization",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_norm"),
     withKey("epsilon", toStringPrecise(NORM_EPS)), withKey("axis", 3),
     withKey("packed", "false")}));

  return ffn_norm(ffn_residual);
}
```

Note the post-norm pattern:
  1. Attention -> residual -> LayerNorm (not pre-norm)
  2. MLP -> residual -> LayerNorm (not pre-norm)
  3. Uses `layer_normalization` (not `rms_norm`)
  4. Epsilon requires high precision: toStringPrecise(NORM_EPS)
'''

# ============================================================================
# BERT EMBEDDING LAYER EXAMPLE
# ============================================================================
BERT_EMBEDDING_EXAMPLE = '''\
BERT has three embedding components: word + position + token_type

```cpp
// Token / Position / TokenType Embeddings
const std::string embedding_type =
  TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "embedding_layer";

LayerHandle word_embedding(createLayer(
  embedding_type,
  {withKey("name", "embedding0"), withKey("in_dim", NUM_VOCAB),
   withKey("weight_dtype", EMBEDDING_DTYPE), withKey("out_dim", DIM)}));
Tensor word = word_embedding(input);

LayerHandle position_embedding(
  createLayer("embedding_layer", {withKey("name", "position_embedding"),
                                  withKey("in_dim", MAX_POSITION_EMBEDDINGS),
                                  withKey("weight_dtype", EMBEDDING_DTYPE),
                                  withKey("out_dim", DIM)}));
Tensor position = position_embedding(position_ids);

LayerHandle token_type_embedding(
  createLayer("embedding_layer", {withKey("name", "token_type_embedding"),
                                  withKey("in_dim", TYPE_VOCAB_SIZE),
                                  withKey("weight_dtype", EMBEDDING_DTYPE),
                                  withKey("out_dim", DIM)}));
Tensor token_type = token_type_embedding(token_type_ids);

LayerHandle embedding_sum(
  createLayer("addition", {withKey("name", "embedding_sum")}));
Tensor h = embedding_sum({word, position, token_type});

LayerHandle embedding_norm(createLayer(
  "layer_normalization", {withKey("name", "embedding_norm"),
                          withKey("epsilon", toStringPrecise(NORM_EPS)),
                          withKey("axis", 3), withKey("packed", "false")}));
h = embedding_norm(h);
```

Note BERT embedding patterns:
  * Three separate embeddings: word, position, token_type
  * All three summed together before embedding_norm
  * Uses `layer_normalization` with axis=3
  * TYPE_VOCAB_SIZE is typically 2 (for sentence A/B)
'''

# ============================================================================
# BERT-SPECIFIC HARD CONSTRAINTS
# ============================================================================
def bert_hard_constraints(plan: Dict) -> str:
    """BERT-specific constraints on top of the common ones."""
    common = _hard_constraints(plan)
    return f"""\
{common}

B2. BERT-SPECIFIC PATTERNS
    * MLP uses GeLU activation: `activation = "gelu"` (NOT swiglu, NOT tanh_gelu)
    * Post-norm decoder block: residual -> layer_normalization
    * Uses `layer_normalization` (NOT `rms_norm`)
    * LayerNorm epsilon requires high precision: toStringPrecise(NORM_EPS)
    * LayerNorm axis = 3 (last dimension)
    * Bidirectional attention: `is_causal = "false"`, `use_rope = "false"`
    * No KV cache - encoder-only model, mha receives only {{q, k, v}}
    * No attn_logit_softcapping property
    * Bias ENABLED on ALL projections: `disable_bias = "false"`
    * Three embeddings: word + position + token_type, summed before norm

B3. BERT FORBIDDEN PATTERNS
    * Do NOT use swiglu or tanh_gelu - use gelu activation
    * Do NOT use pre-norm pattern - BERT uses post-norm
    * Do NOT use rms_norm - must use layer_normalization
    * Do NOT use is_causal = "true" - BERT is bidirectional
    * Do NOT use RoPE - set use_rope = "false"
    * Do NOT add KV cache placeholders - BERT is encoder-only
    * Do NOT use attn_logit_softcapping property
    * Do NOT use HF dotted names like "bert.encoder.layer.0.attention.self.query"
    * Do NOT set disable_bias="true" anywhere - BERT uses bias everywhere
    * Do NOT forget axis=3 on layer_normalization
"""

# ============================================================================
# BERT GENERATION PROMPT
# ============================================================================
def bert_generation_prompt(plan: Dict, skeleton_h: str, skeleton_cpp: str) -> str:
    """
    BERT-specific generation prompt.
    
    Uses BERT worked examples and constraints.
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

This is a BERT-family model. Match the BERT architecture patterns:
  * GeLU MLP (gate -> gelu -> down)
  * Post-norm decoder block (residual -> LayerNorm)
  * Bidirectional attention (no RoPE, no causal mask)
  * Layer normalization (not RMSNorm)
  * Three embeddings: word + position + token_type

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
{bert_hard_constraints(plan)}

======================= 5. OVERRIDABLE HOOKS (exact) ========================
{render_hooks(only=hook_names or None)}

=========================== 6. WORKED EXAMPLES ===============================

{BERT_SETUP_NOTE}

{BERT_WORKED_EXAMPLE}

{BERT_MLP_EXAMPLE}

{BERT_BLOCK_EXAMPLE}

{BERT_EMBEDDING_EXAMPLE}

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
