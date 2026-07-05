// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   lora_train.cpp
 * @date   01 Apr 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @author Sumon Nath <sumon.nath@samsung.com>
 * @author Niket Agarwal <niket.a@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  LoRA training data pipeline implementation
 */

#include "lora_train.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include <lm_head.h>  // for g_lm_head_read_row

namespace causallm {

TrainingDataGenerator::TrainingDataGenerator(tokenizers::Tokenizer *tokenizer,
                                             unsigned int seq_len,
                                             unsigned int vocab_size,
                                             unsigned int seed) :
  tokenizer_(tokenizer),
  seq_len_(seq_len),
  vocab_size_(vocab_size),
  current_idx_(0),
  rng_(seed) {
  std::cout << "[TrainingData] shuffle seed=" << seed << std::endl;
}

void TrainingDataGenerator::loadTextFile(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open())
    throw std::runtime_error("Failed to open training data file: " + path);

  std::vector<std::string> all_lines;
  std::string line;
  while (std::getline(file, line))
    all_lines.push_back(line);

  // Detect chat format: any line starts with <|im_start|>user
  bool is_chat = false;
  for (const auto &l : all_lines) {
    if (l.rfind("<|im_start|>user", 0) == 0) {
      is_chat = true;
      break;
    }
  }

  int count = 0;
  if (is_chat) {
    // Each sample starts at a <|im_start|>user line and runs until the next
    // one. Lines are joined with '\n' so the tokenizer sees the real newlines.
    std::string current;
    for (const auto &l : all_lines) {
      if (l.rfind("<|im_start|>user", 0) == 0 && !current.empty()) {
        auto ids = tokenizer_->Encode(current);
        if (ids.size() >= 2) { samples_.push_back(ids); count++; }
        current.clear();
      }
      if (!current.empty()) current += '\n';
      current += l;
    }
    if (!current.empty()) {
      auto ids = tokenizer_->Encode(current);
      if (ids.size() >= 2) { samples_.push_back(ids); count++; }
    }
  } else {
    for (const auto &l : all_lines) {
      if (l.empty()) continue;
      auto ids = tokenizer_->Encode(l);
      samples_.push_back(ids);
      count++;
    }
  }

  std::cout << "[TrainingData] Loaded " << path << " — " << count
            << " samples." << std::endl;

  // Build initial shuffled index order after all samples are loaded.
  index_order_.resize(samples_.size());
  std::iota(index_order_.begin(), index_order_.end(), 0);
  std::shuffle(index_order_.begin(), index_order_.end(), rng_);
}

void TrainingDataGenerator::addTokenIds(const std::vector<int> &ids) {
  samples_.push_back(ids);
}

unsigned int TrainingDataGenerator::getNumSamples() const {
  return static_cast<unsigned int>(samples_.size());
}

void TrainingDataGenerator::reset() {
  current_idx_ = 0;
  std::shuffle(index_order_.begin(), index_order_.end(), rng_);
}

void TrainingDataGenerator::limitSamples(unsigned int max_samples) {
  if (max_samples < samples_.size()) {
    samples_.resize(max_samples);
    index_order_.resize(max_samples);
    std::iota(index_order_.begin(), index_order_.end(), 0);
    std::shuffle(index_order_.begin(), index_order_.end(), rng_);
  }
}

int TrainingDataGenerator::dataCb(float **input, float **label, bool *last,
                                  void *user_data) {
  auto *self = static_cast<TrainingDataGenerator *>(user_data);

  // Auto-reset at the start of each new epoch
  if (self->current_idx_ >= self->samples_.size())
    self->reset();

  const auto &ids = self->samples_[self->index_order_[self->current_idx_]];
  unsigned int available = static_cast<unsigned int>(ids.size());

  // Label = last token in the sequence (the rating digit).
  // Input = everything before it, RIGHT-padded to seq_len.
  //   Real tokens occupy positions 0..used-1; pads (token 0) at used..seq_len-1.
  //   Causal attention naturally excludes the future pad positions from position
  //   used-1, so no pad-token corruption reaches the final hidden state.
  //   g_lm_head_read_row is set to (used-1) so forwarding() and calcDerivative()
  //   operate on the correct row instead of the default height-1 (a pad).

  unsigned int label_token = 0;

  if (available >= 2) {
    label_token = static_cast<unsigned int>(ids[available - 1]);

    unsigned int input_len = available - 1;
    unsigned int start = (input_len > self->seq_len_) ? (input_len - self->seq_len_) : 0;
    unsigned int used = input_len - start;

    for (unsigned int j = 0; j < used; ++j)
      input[0][j] = static_cast<float>(ids[start + j]);
    for (unsigned int j = used; j < self->seq_len_; ++j)
      input[0][j] = 0.0f;

    // Tell lm_head which row is the final real token.
    g_lm_head_read_row = used - 1;
  } else {
    for (unsigned int j = 0; j < self->seq_len_; ++j)
      input[0][j] = 0.0f;
    g_lm_head_read_row = 0;
  }

  for (unsigned int v = 0; v < self->vocab_size_; ++v)
    label[0][v] = 0.0f;
  if (label_token < self->vocab_size_)
    label[0][label_token] = 1.0f;

  std::cout << "[DataGen] sample " << self->current_idx_ << " / "
            << self->samples_.size() << "\r" << std::flush;

  self->current_idx_++;
  *last = (self->current_idx_ >= self->samples_.size());
  return 0;
}

} // namespace causallm
