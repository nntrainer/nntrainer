// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   lora_train.h
 * @date   01 Apr 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  LoRA training data pipeline and training loop for CausalLM
 */

#ifndef __LORA_TRAIN_H__
#define __LORA_TRAIN_H__

#include <functional>
#include <string>
#include <vector>

#include <dataset.h>
#include <model.h>
#include <tokenizers_cpp.h>

namespace causallm {

/**
 * @brief Training data sample for next-token prediction
 *
 * Given a sequence [t0, t1, t2, ..., tn]:
 *   input  = [t0, t1, ..., t(n-1)]
 *   label  = [t1, t2, ..., tn]
 *
 * For causal LM, the model learns to predict the next token at each position.
 */
class TrainingDataGenerator {
public:
  /**
   * @brief Construct a TrainingDataGenerator
   * @param tokenizer Tokenizer to use for encoding text
   * @param seq_len Sequence length for training samples
   * @param vocab_size Vocabulary size for one-hot representation
   */
  TrainingDataGenerator(tokenizers::Tokenizer *tokenizer, unsigned int seq_len,
                        unsigned int vocab_size);

  /**
   * @brief Load training text from a file
   * @param path Path to the text file
   */
  void loadTextFile(const std::string &path);

  /**
   * @brief Add pre-tokenized IDs directly
   * @param ids Vector of token IDs
   */
  void addTokenIds(const std::vector<int> &ids);

  /**
   * @brief Get the number of training samples
   */
  unsigned int getNumSamples() const;

  /**
   * @brief Limit the number of training samples
   * @param max_samples Maximum number of samples to keep
   */
  void limitSamples(unsigned int max_samples);

  /**
   * @brief Data generation callback for nntrainer's dataset API
   *
   * This is the callback signature expected by ml::train::createDataset
   * with DatasetType::GENERATOR.
   *
   * @param[out] input Pointer to input buffer (seq_len floats)
   * @param[out] label Pointer to label buffer (seq_len floats)
   * @param[out] last Set to true when epoch is complete
   * @param[in] user_data Pointer to TrainingDataGenerator instance
   * @return 0 on success
   */
  static int dataCb(float **input, float **label, bool *last, void *user_data);

private:
  tokenizers::Tokenizer *tokenizer_;
  unsigned int seq_len_;
  unsigned int vocab_size_;
  std::vector<std::vector<int>> samples_;
  unsigned int current_idx_;

  void reset();
};

} // namespace causallm

#endif
