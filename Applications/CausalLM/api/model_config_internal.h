// SPDX-License-Identifier: Apache-2.0
/**
 * @file   model_config_internal.h
 * @brief  Internal structures and registration for api.
 *         Self-contained — does NOT depend on upstream causal_lm_api headers.
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __QUICK_DOT_AI_MODEL_CONFIG_INTERNAL_H__
#define __QUICK_DOT_AI_MODEL_CONFIG_INTERNAL_H__

#include <stdbool.h>
#include <stddef.h>

/**
 * @brief Model Architecture Configuration (replaces config.json)
 */
typedef struct {
  unsigned int vocab_size;
  unsigned int hidden_size;
  unsigned int intermediate_size;
  unsigned int num_hidden_layers;
  unsigned int num_attention_heads;
  unsigned int head_dim;
  unsigned int num_key_value_heads;
  unsigned int max_position_embeddings;
  float rope_theta;
  float rms_norm_eps;
  bool tie_word_embeddings;
  unsigned int sliding_window;
  unsigned int sliding_window_pattern;

  unsigned int eos_token_ids[4];
  unsigned int num_eos_token_ids;
  unsigned int bos_token_id;

  char architecture[64];
} ModelArchConfig;

/**
 * @brief Model Runtime Configuration (replaces nntr_config.json)
 */
typedef struct {
  unsigned int batch_size;
  char model_type[32];
  char model_tensor_type[32];
  unsigned int init_seq_len;
  unsigned int max_seq_len;
  unsigned int num_to_generate;
  bool fsu;
  unsigned int fsu_lookahead;
  char embedding_dtype[32];
  char fc_layer_dtype[32];
  char model_file_name[256];
  char tokenizer_file[256];
  unsigned int bad_word_ids[16];
  unsigned int num_bad_word_ids;
  char lmhead_dtype[32];

  unsigned int top_k;
  float top_p;
  float temperature;
} ModelRuntimeConfig;

namespace quick_dot_ai {

/**
 * @brief Register a model architecture config (writes to g_arch_config_map)
 */
void register_arch(const char *arch_name, ModelArchConfig config);

/**
 * @brief Register a model runtime config (writes to g_model_registry)
 */
void register_model(const char *model_name, const char *arch_name,
                    ModelRuntimeConfig config);

/**
 * @brief Called from register_models() to register all built-in configs
 */
void register_builtin_configs();

} // namespace quick_dot_ai

#endif // __QUICK_DOT_AI_MODEL_CONFIG_INTERNAL_H__
