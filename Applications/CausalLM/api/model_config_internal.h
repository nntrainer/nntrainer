// SPDX-License-Identifier: Apache-2.0
/**
 * @file   model_config_internal.h
 * @brief  Internal Structures and Functions for Model Configuration
 *         This file should NOT be exposed to the public API users.
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#ifndef __MODEL_CONFIG_INTERNAL_H__
#define __MODEL_CONFIG_INTERNAL_H__

#include "causal_lm_api.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Model Architecture Configuration (config.json)
 */
typedef struct {
  // config.json parameters
  unsigned int vocab_size;
  unsigned int hidden_size;
  unsigned int intermediate_size;
  unsigned int num_hidden_layers;
  unsigned int num_attention_heads;
  unsigned int head_dim;
  unsigned int num_key_value_heads; // if 0, defaults to num_attention_heads
  unsigned int max_position_embeddings;
  float rope_theta;
  float rms_norm_eps;
  bool tie_word_embeddings;
  unsigned int sliding_window; // Use UINT_MAX for null
  unsigned int sliding_window_pattern;

  // generation_config.json (static model properties)
  unsigned int eos_token_ids[4];
  unsigned int num_eos_token_ids;
  unsigned int bos_token_id;

  // architecture identification
  char architecture[64]; // e.g., "Qwen3ForCausalLM"
} ModelArchConfig;

/**
 * @brief Model Runtime/Execution Configuration (nntr_config.json)
 */
typedef struct {
  // nntr_config.json parameters
  unsigned int batch_size;
  char model_type[32]; // e.g. "CausalLM"
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
  char embedding_file_name[256];
  char ple_file_name[256];
  unsigned int bad_word_ids[16];
  unsigned int num_bad_word_ids;
  char lmhead_dtype[32];

  // generation_config.json (runtime parameters)
  unsigned int top_k;
  float top_p;
  float temperature;
} ModelRuntimeConfig;

/**
 * @brief Register a model architecture configuration
 * @param arch_name Name of the architecture (e.g., "Qwen3-0.6B-Arch")
 * @param config Architecture configuration
 * @return ErrorCode
 */
ErrorCode registerModelArchitecture(const char *arch_name,
                                    ModelArchConfig config);

/**
 * @brief Register a full model configuration linking runtime config to an
 * architecture
 * @param model_name Name of the model to register (e.g., "Qwen3-0.6B")
 * @param arch_name Name of the registered architecture to use
 * @param config Runtime configuration
 * @return ErrorCode
 */
ErrorCode registerModel(const char *model_name, const char *arch_name,
                        ModelRuntimeConfig config);

/**
 * @brief Register built-in model configurations (e.g., Qwen3-0.6B)
 * @return 0 on success
 */
int register_builtin_model_configs();

#ifdef __cplusplus
}
#endif

#endif // __MODEL_CONFIG_INTERNAL_H__
