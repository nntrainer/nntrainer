// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   lora_train.cpp
 * @date   31 July 2026
 * @brief  Text-file training data generator for Qwen3 LoRA fine-tuning.
 * @bug    No known bugs except for NYI items
 */
#include <lora_train.h>

#include <nntrainer_error.h>

#include <algorithm>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace causallm {

TrainingDataGenerator::TrainingDataGenerator(const std::string &data_path,
                                             tokenizers::Tokenizer *tokenizer,
                                             unsigned int seq_len,
                                             unsigned int vocab_size,
                                             unsigned int max_samples,
                                             unsigned int seed) :
  seq_len_(seq_len), vocab_size_(vocab_size), rng_(seed) {
  loadTextFile(data_path, tokenizer, max_samples);
}

void TrainingDataGenerator::loadTextFile(const std::string &path,
                                         tokenizers::Tokenizer *tokenizer,
                                         unsigned int max_samples) {
  std::ifstream file(path);
  if (!file.is_open())
    throw std::runtime_error("Failed to open training data file: " + path);

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty())
      lines.push_back(line);
  }

  bool chat_format = false;
  for (const auto &l : lines) {
    if (l.find("<|im_start|>user") != std::string::npos) {
      chat_format = true;
      break;
    }
  }

  std::vector<std::string> raw_samples;
  if (chat_format) {
    std::string current;
    bool started = false;
    for (const auto &l : lines) {
      if (l.find("<|im_start|>user") != std::string::npos) {
        if (started && !current.empty())
          raw_samples.push_back(current);
        current = l;
        started = true;
      } else if (started) {
        current += "\n" + l;
      }
    }
    if (started && !current.empty())
      raw_samples.push_back(current);
  } else {
    raw_samples = lines;
  }

  for (const auto &s : raw_samples) {
    if (max_samples && samples_.size() >= max_samples)
      break;

    auto ids = tokenizer->Encode(s);
    // Need at least one input token plus the label token.
    if (ids.size() < 2)
      continue;

    // Input (everything but the last token) must fit within seq_len_; if
    // the tokenized sample is longer, keep the most recent seq_len_+1
    // tokens so the label (the true last token) is preserved.
    if (ids.size() > seq_len_ + 1)
      ids = std::vector<int32_t>(ids.end() - (seq_len_ + 1), ids.end());

    samples_.push_back({std::move(ids)});
  }

  if (samples_.empty())
    throw std::runtime_error("No usable training samples found in " + path);

  index_order_.resize(samples_.size());
  std::iota(index_order_.begin(), index_order_.end(), 0);
  reshuffle();
}

void TrainingDataGenerator::reshuffle() {
  std::shuffle(index_order_.begin(), index_order_.end(), rng_);
  cursor_ = 0;
}

int TrainingDataGenerator::next(float **input, float **label, bool *last) {
  const Sample &s = samples_[index_order_[cursor_]];
  const unsigned int used = static_cast<unsigned int>(s.ids.size() - 1);

  float *in = input[0];
  std::fill(in, in + seq_len_, 0.0f);
  for (unsigned int i = 0; i < used; ++i)
    in[i] = static_cast<float>(s.ids[i]);

  std::fill(label[0], label[0] + vocab_size_, 0.0f);
  const int32_t label_id = s.ids.back();
  if (label_id >= 0 && static_cast<unsigned int>(label_id) < vocab_size_)
    label[0][label_id] = 1.0f;

  /**
   * @note The lm head must read the row of the last *real* token
   * (used - 1), not row (seq_len_ - 1) which is padding. That row index is
   * deliberately NOT published from here: nntrainer runs this generator on
   * a separate producer thread that may prefetch ahead of the trainer, so
   * anything written here is both invisible to the training thread and
   * liable to describe the wrong sample by the time it is forwarded.
   * Instead the embedding layer re-derives it from the token ids during the
   * forward pass — see g_tie_embedding_lm_head_read_row in
   * tie_word_embedding.h. That derivation relies on padding being id 0 and
   * real tokens being placed strictly before it, which is what this
   * function guarantees.
   */
  ++cursor_;
  const bool epoch_done = cursor_ >= index_order_.size();
  if (epoch_done)
    reshuffle();
  *last = epoch_done;

  return ML_ERROR_NONE;
}

int trainingDataGenCb(float **input, float **label, bool *last,
                      void *user_data) {
  return static_cast<TrainingDataGenerator *>(user_data)->next(input, label,
                                                               last);
}

} // namespace causallm
