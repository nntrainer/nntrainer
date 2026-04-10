// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   lora_train.cpp
 * @date   01 Apr 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  LoRA training data pipeline implementation
 */

#include "lora_train.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace causallm {

TrainingDataGenerator::TrainingDataGenerator(tokenizers::Tokenizer *tokenizer,
                                             unsigned int seq_len,
                                             unsigned int vocab_size) :
  tokenizer_(tokenizer),
  seq_len_(seq_len),
  vocab_size_(vocab_size),
  current_idx_(0) {}

void TrainingDataGenerator::loadTextFile(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open training data file: " + path);
  }

  std::string line;
  int count = 0;
  while (std::getline(file, line)) {
    if (line.empty())
      continue;
    auto ids = tokenizer_->Encode(line);
    samples_.push_back(ids);
    count++;
  }

  std::cout << "[TrainingData] Loaded " << path
            << " line by line, total: " << count << " samples." << std::endl;
}

void TrainingDataGenerator::addTokenIds(const std::vector<int> &ids) {
  samples_.push_back(ids);
}

unsigned int TrainingDataGenerator::getNumSamples() const {
  return static_cast<unsigned int>(samples_.size());
}

void TrainingDataGenerator::reset() { current_idx_ = 0; }

void TrainingDataGenerator::limitSamples(unsigned int max_samples) {
  if (max_samples < samples_.size()) {
    samples_.resize(max_samples);
  }
}

int TrainingDataGenerator::dataCb(float **input, float **label, bool *last,
                                  void *user_data) {
  auto *self = static_cast<TrainingDataGenerator *>(user_data);

  if (self->current_idx_ >= self->samples_.size()) {
    *last = true;
    self->reset();
    return 0;
  }

  const auto &ids = self->samples_[self->current_idx_];
  unsigned int available = ids.size();

  // Input: fill seq_len token IDs [t_0, t_1, ..., t_{L-1}, pad, ...]
  for (unsigned int j = 0; j < self->seq_len_; j++) {
    if (j < available) {
      input[0][j] = static_cast<float>(ids[j]);
    } else {
      input[0][j] = 0.0f; // pad
    }
  }

  // Label: single one-hot vector of size VOCAB_SIZE
  // The lm_head layer collapses the sequence to height=1 (last position),
  // so NNTrainer allocates the label buffer as [1, 1, 1, VOCAB_SIZE].
  // Target = next token after the last input position in the sequence.
  for (unsigned int v = 0; v < self->vocab_size_; ++v) {
    label[0][v] = 0.0f;
  }

  // Determine the target token: the token right after our input window
  unsigned int last_input_pos =
    std::min(available, (decltype(available))self->seq_len_);
  if (last_input_pos < available) {
    unsigned int target_id = ids[last_input_pos];
    if (target_id < self->vocab_size_) {
      label[0][target_id] = 1.0f;
    }
  } else {
    // No next token available (sequence ended), predict pad/eos (token 0)
    label[0][0] = 1.0f;
  }

  // Progress printing
  std::cout << "[DataGen] Sample " << self->current_idx_ << " / "
            << self->samples_.size() << std::endl;

  self->current_idx_++;
  *last = false;
  return 0;
}

} // namespace causallm
