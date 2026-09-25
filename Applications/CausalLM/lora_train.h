// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   lora_train.h
 * @date   31 July 2026
 * @brief  Text-file training data generator for Qwen3 LoRA fine-tuning.
 * @bug    No known bugs except for NYI items
 *
 * @details Each line of the input file is one training sample. A sample is
 *          tokenized, then split into (input, label) as: label = last
 *          token id (one-hot over vocab_size), input = every token before
 *          it, right-padded with 0 to seq_len. Because attention is causal,
 *          right-padding after the real content never corrupts the hidden
 *          state at the last real position - but the LM head's default read
 *          row is height-1 (a pad slot), so next() also points
 *          causallm::g_lm_head_read_row / g_tie_embedding_lm_head_read_row
 *          (see lm_head.h / tie_word_embedding.h) at the last *real* token
 *          before returning each sample, exactly as the model's forward
 *          pass for that sample is about to expect.
 *
 *          A line is treated as "chat format" (joined as a single sample
 *          rather than split into multiple lines) whenever the file
 *          contains a `<|im_start|>user` marker anywhere; otherwise each
 *          line is its own independent sample (e.g. one-sentence
 *          classification data).
 */
#ifndef __LORA_TRAIN_H__
#define __LORA_TRAIN_H__

#include <random>
#include <string>
#include <tokenizers_cpp.h>
#include <vector>

namespace causallm {

/**
 * @brief Generates (input, label) training samples from a text file for
 *        Qwen3 LoRA fine-tuning, matching the model's single-label-per-
 *        sample (last real token predicts the next token) training scheme.
 */
class TrainingDataGenerator {
public:
  /**
   * @param data_path path to a plain-text or chat-format training file
   * @param tokenizer tokenizer used to encode each sample (not owned)
   * @param seq_len fixed input width every sample is right-padded to
   * @param vocab_size width of the one-hot label
   * @param max_samples optional cap on the number of samples loaded (0 =
   *        no cap)
   * @param seed RNG seed for deterministic epoch shuffling
   */
  TrainingDataGenerator(const std::string &data_path,
                        tokenizers::Tokenizer *tokenizer, unsigned int seq_len,
                        unsigned int vocab_size, unsigned int max_samples = 0,
                        unsigned int seed = 42);

  /**
   * @brief ml::train GENERATOR dataset callback signature.
   * @details Fills one sample's (input, label), sets the LM-head read-row
   *          globals for that sample, and reshuffles at each epoch
   *          boundary. `*last` is set true after the final sample of an
   *          epoch (matching ml::train::DatasetGenCb semantics).
   */
  int next(float **input, float **label, bool *last);

  /** @brief Number of samples loaded from the training file. */
  size_t size() const { return samples_.size(); }

private:
  struct Sample {
    std::vector<int32_t> ids; /**< full tokenized sample, unpadded */
  };

  std::vector<Sample> samples_;
  std::vector<size_t> index_order_;
  size_t cursor_ = 0;
  unsigned int seq_len_;
  unsigned int vocab_size_;
  std::mt19937 rng_;

  void loadTextFile(const std::string &path, tokenizers::Tokenizer *tokenizer,
                    unsigned int max_samples);
  void reshuffle();
};

/**
 * @brief ml::train::createDataset(GENERATOR, ...)-compatible free-function
 *        callback; `user_data` must be a `TrainingDataGenerator*`.
 */
int trainingDataGenCb(float **input, float **label, bool *last,
                      void *user_data);

} // namespace causallm

#endif /* __LORA_TRAIN_H__ */
