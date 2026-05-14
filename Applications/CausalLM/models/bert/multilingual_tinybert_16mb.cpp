// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   multilingual_tinybert_16mb.cpp
 * @date   21 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file defines multilingual-TinyBERT-16MB embedding model.
 */

#include "multilingual_tinybert_16mb.h"

#include <app_context.h>
#include <chrono>
#include <codecvt>
#include <engine.h>
#include <iomanip>
#include <iostream>
#include <llm_util.hpp>
#include <locale>
#include <model.h>
#include <sstream>
#include <stdexcept>

namespace causallm {

std::vector<float *> MultilingualTinyBert::encode(const WSTR prompt,
                                                  const WSTR system_prompt,
                                                  const WSTR tail_prompt) {
  if (!is_initialized) {
    throw std::runtime_error(
      "MultilingualTinyBert is not initialized. Please call "
      "initialize() before encode().");
  }

  std::string prompt_ = system_prompt + prompt + tail_prompt;
  auto tokenized = tokenizer->Encode(prompt_, true);

  unsigned int input_len =
    std::min(static_cast<unsigned int>(tokenized.size()), INIT_SEQ_LEN);

  float *input_sample =
    (float *)malloc(sizeof(float) * BATCH_SIZE * INIT_SEQ_LEN);
  float *position_ids =
    (float *)malloc(sizeof(float) * BATCH_SIZE * INIT_SEQ_LEN);
  float *token_type_ids =
    (float *)malloc(sizeof(float) * BATCH_SIZE * INIT_SEQ_LEN);

  if (!input_sample || !position_ids || !token_type_ids) {
    free(input_sample);
    free(position_ids);
    free(token_type_ids);
    throw std::runtime_error("Failed to allocate input buffers");
  }

  std::fill(input_sample, input_sample + BATCH_SIZE * INIT_SEQ_LEN, 0.0f);
  std::fill(position_ids, position_ids + BATCH_SIZE * INIT_SEQ_LEN, 0.0f);
  std::fill(token_type_ids, token_type_ids + BATCH_SIZE * INIT_SEQ_LEN, 0.0f);

  for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
    for (unsigned int i = 0; i < input_len; ++i) {
      input_sample[static_cast<size_t>(b) * INIT_SEQ_LEN + i] =
        static_cast<float>(tokenized[i]);
      position_ids[static_cast<size_t>(b) * INIT_SEQ_LEN + i] =
        static_cast<float>(i);
    }
  }

  std::vector<float *> input = {input_sample, position_ids, token_type_ids};
  std::vector<float *> label;

  auto start_prefill = std::chrono::high_resolution_clock::now();
  auto output = model->incremental_inference(BATCH_SIZE, input, label,
                                             input_len, 0, input_len, false);
  auto finish_prefill = std::chrono::high_resolution_clock::now();
  auto prefill_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    finish_prefill - start_prefill);

  std::cout << "prefill: " << input_len << " tokens, "
            << prefill_duration.count() << " ms, "
            << ((double)input_len / prefill_duration.count() * 1000)
            << " TPS\n";

  free(input_sample);
  free(position_ids);
  free(token_type_ids);

  return output;
}

} // namespace causallm
