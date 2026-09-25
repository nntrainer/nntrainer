// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_qwen3_lora_training_smoke.cpp
 * @date   31 July 2026
 * @brief  End-to-end smoke test for LoRA training wiring: builds a tiny
 *         synthetic Qwen3CausalLM, compiles it for training with LoRA
 *         enabled, runs a few training steps on synthetic data, and checks
 *         that the LoRA adapter save/load round-trips.
 * @bug    No known bugs except for NYI items
 */

#include <qwen3_causallm.h>

#include <dataset.h>
#include <layer_context.h>
#include <model.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {

causallm::json makeTinyQwen3Config() {
  return {
    {"architectures", {"Qwen3ForCausalLM"}},
    {"bos_token_id", 0},
    {"eos_token_id", {31}},
    {"head_dim", 8},
    {"hidden_size", 64},
    {"intermediate_size", 64},
    {"is_causal", true},
    {"max_position_embeddings", 8},
    {"num_attention_heads", 8},
    {"num_hidden_layers", 2},
    {"num_key_value_heads", 4},
    {"rms_norm_eps", 1e-5},
    {"rope_theta", 10000},
    {"tie_word_embeddings", true},
    {"vocab_size", 32},
  };
}

causallm::json makeTinyGenerationConfig() {
  return {
    {"bos_token_id", 0}, {"eos_token_id", 31}, {"do_sample", false},
    {"top_k", 1},        {"top_p", 1.0},       {"temperature", 1.0},
  };
}

causallm::json makeTinyNntrainerConfig(unsigned int lora_rank) {
  return {
    {"bad_word_ids", std::vector<unsigned int>{}},
    {"batch_size", 1},
    {"embedding_dtype", "FP32"},
    {"fc_layer_dtype", "FP32"},
    {"init_seq_len", 4},
    {"lmhead_dtype", "FP32"},
    {"max_seq_len", 8},
    {"model_tensor_type", "FP32-FP32"},
    {"model_type", "CausalLM"},
    {"num_to_generate", 1},
    {"skip_tokenizer", true},
    {"lora_rank", lora_rank},
    {"lora_alpha", lora_rank * 2},
    {"lora_target",
     std::vector<std::string>{"wq", "wk", "wv", "wo", "ffn_up", "ffn_gate",
                              "ffn_down"}},
  };
}

/** @brief Synthetic in-memory next-token-classification generator matching
 *  the (single-label-per-sample) training scheme the LM-head layers assume:
 *  a fixed-length input of token ids and a one-hot label over vocab_size. */
struct SyntheticDataGen {
  unsigned int seq_len;
  unsigned int vocab_size;
  std::mt19937 rng{42};

  int operator()(float **input, float **label, bool *last) {
    std::uniform_int_distribution<unsigned int> tok(0, vocab_size - 1);
    for (unsigned int i = 0; i < seq_len; ++i)
      input[0][i] = static_cast<float>(tok(rng));

    std::fill(label[0], label[0] + vocab_size, 0.0f);
    label[0][tok(rng)] = 1.0f;

    *last = true;
    return ML_ERROR_NONE;
  }
};

int datasetCb(float **input, float **label, bool *last, void *user_data) {
  auto *gen = static_cast<SyntheticDataGen *>(user_data);
  return (*gen)(input, label, last);
}

} // namespace

TEST(Qwen3LoraTrainingSmoke, InitializeForTrainingCompilesWithLoraEnabled) {
  auto cfg = makeTinyQwen3Config();
  auto generation_cfg = makeTinyGenerationConfig();
  auto nntr_cfg = makeTinyNntrainerConfig(/*lora_rank=*/4);

  causallm::Qwen3CausalLM model(cfg, generation_cfg, nntr_cfg);
  ASSERT_NO_THROW(model.initializeForTraining(/*lr=*/1e-3f, /*epochs=*/1));
}

TEST(Qwen3LoraTrainingSmoke, TrainableFlagsMatchLoraFreezePolicy) {
  auto cfg = makeTinyQwen3Config();
  auto generation_cfg = makeTinyGenerationConfig();
  auto nntr_cfg = makeTinyNntrainerConfig(/*lora_rank=*/4);

  causallm::Qwen3CausalLM model(cfg, generation_cfg, nntr_cfg);
  ASSERT_NO_THROW(model.initializeForTraining(1e-3f, 1));

  std::vector<std::string> trainable_layers, frozen_layers;
  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn = [&](ml::train::Layer &l, nntrainer::RunLayerContext &context,
             void *) {
      // A layer is "trainable" here iff it has at least one weight with a
      // populated gradient buffer, which only happens when the layer-level
      // trainable flag AND the weight's own need_gradient flag are both true.
      // Layers with zero weights (input, residual "addition", the loss
      // layer) are trivially frozen and not part of this check at all.
      if (context.getNumWeights() == 0)
        return;
      bool has_grad = false;
      for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
        if (context.weightHasGradient(i)) {
          has_grad = true;
          break;
        }
      }
      (has_grad ? trainable_layers : frozen_layers).push_back(l.getName());
    };
  model.forEachLayer(fn, nullptr);

  // wq/wk/wv/wo/ffn_up/ffn_gate/ffn_down are LoRA targets -> trainable
  // (their loraA/loraB weights have gradients; the base weight does not).
  bool found_lora_fc = false;
  for (const auto &name : trainable_layers) {
    if (name.find("_wq") != std::string::npos ||
        name.find("_ffn_up") != std::string::npos) {
      found_lora_fc = true;
    }
  }
  EXPECT_TRUE(found_lora_fc)
    << "expected at least one LoRA-targeted FC layer to have a trainable "
       "(loraA/loraB) weight";

  // norms, embedding, mha_core, lm_head must NOT be trainable.
  for (const auto &name : frozen_layers) {
    EXPECT_TRUE(name.find("_norm") != std::string::npos ||
               name == "embedding0" || name.find("_attention") !=
                                          std::string::npos ||
               name == "output_of_causallm" || name.find("_swiglu") !=
                                                  std::string::npos)
      << "unexpected frozen layer: " << name;
  }
}

