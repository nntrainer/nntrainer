// SPDX-License-Identifier: Apache-2.0
/** @file attention_config.h
 * @brief Shape and cache contract shared by attention and its execution state.
 * @author NNTrainer contributors
 * @bug No known bugs except for NYI items
 */
#ifndef NNTR_ATTENTION_CONFIG_H
#define NNTR_ATTENTION_CONFIG_H
#include <cstddef>
namespace nntrainer {
class ComputeOps;
struct AttentionConfig {
  unsigned int cache_index = 0;
  unsigned int cache_base = 0;
  size_t num_heads_Q = 0;
  size_t num_heads_KV = 0;
  size_t head_dim = 0;
  size_t local_window_size = 0;
  bool use_rope = true;
  float attn_logit_softcapping = 0.0f;
  bool is_causal = true;
  int kv_cache_quant_kind = -1;
  bool kv_cache_fixed_scales = true;
  bool attn_out_u16 = false;
  ComputeOps *compute_ops_ = nullptr;
};
} // namespace nntrainer
#endif
