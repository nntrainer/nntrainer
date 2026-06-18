// SPDX-License-Identifier: Apache-2.0
/**
 *
 * @file   llm_util.cpp
 * @brief  util functions for llm (refactored from main.cpp)
 * @date   21 August 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <algorithm>
#include <cmath>
#include <iterator>
#include <random>

#include <llm_util.hpp>

std::vector<unsigned int> generate_multi_tokens(
  float *logits, unsigned int NUM_VOCAB, unsigned int NUM_TARGET_TOKENS,
  float repetition_penalty, unsigned int *input_ids, unsigned int NUM_INPUT_IDS,
  unsigned int *bad_words_ids, unsigned int NUM_BAD_WORDS_IDS) {

  std::vector<unsigned int> outputs;

  // apply repetition penalty
  if (repetition_penalty != 1 && input_ids != nullptr && NUM_INPUT_IDS != 0) {
    applyRepetitionPenalty(logits, input_ids, NUM_INPUT_IDS,
                           repetition_penalty);
  }

  // apply bad words penalty
  if (bad_words_ids != nullptr && NUM_BAD_WORDS_IDS != 0)
    applyBadWordsPenalty(logits, bad_words_ids, NUM_BAD_WORDS_IDS);

  // Sort and generate multiple tokens
  std::vector<std::pair<unsigned int, float>> top_indices_and_logits;
  for (unsigned int i = 0; i < NUM_VOCAB; ++i) {
    top_indices_and_logits.push_back({i, logits[i]});
  }
  std::partial_sort(top_indices_and_logits.begin(),
                    top_indices_and_logits.begin() + NUM_TARGET_TOKENS,
                    top_indices_and_logits.end(),
                    [](auto &a, auto &b) { return a.second > b.second; });

  // add sampled words
  for (unsigned int i = 0; i < NUM_TARGET_TOKENS; ++i) {
    outputs.push_back(top_indices_and_logits[i].first);
  }

  return outputs;
}

void applyRepetitionPenalty(float *logits, unsigned int *input_ids,
                            unsigned int NUM_INPUT_IDS,
                            float repetition_penalty) {
  for (unsigned int i = 0; i < NUM_INPUT_IDS; ++i) {
    if (logits[input_ids[i]] < 0) {
      logits[input_ids[i]] *= repetition_penalty;
    } else {
      logits[input_ids[i]] /= repetition_penalty;
    }
  }
}

void applyBadWordsPenalty(float *logits, unsigned int *bad_words_ids,
                          unsigned int NUM_BAD_WORDS_IDS) {
  for (unsigned int i = 0; i < NUM_BAD_WORDS_IDS; ++i) {
    logits[bad_words_ids[i]] = -INFINITY;
  }
}

/**
 * @brief Apply temperature & top-k & top-p to logits, compute softmax, and
 * sample
 * @return Sampled token index
 */
unsigned int applyTKP(const float *logits, int len, float temperature,
                      unsigned int top_k, float top_p, std::mt19937 &rng) {

  // Apply temperature to scores (keep original logits unchanged)
  if (temperature <= 1e-5) {
    std::cerr << "[Warning] temperature is too small, using greedy strategy"
              << std::endl;
    auto max_it = std::max_element(logits, logits + len);
    return std::distance(logits, max_it);
  }

  std::vector<float> scores(len);
  for (int i = 0; i < len; ++i) {
    scores[i] = logits[i] / temperature;
  }

  // Partial sort scores with indices (top-k)
  std::vector<std::pair<int, float>> top_indices_and_scores;
  for (int i = 0; i < len; ++i) {
    top_indices_and_scores.push_back({i, scores[i]});
  }

  int sort_size = len;
  if (top_k > 0 && top_k < (unsigned int)len) {
    sort_size = top_k;
  }

  std::partial_sort(top_indices_and_scores.begin(),
                    top_indices_and_scores.begin() + sort_size,
                    top_indices_and_scores.end(),
                    [](auto &a, auto &b) { return a.second > b.second; });

  float max_score = top_indices_and_scores[0].second;

  // Calculate probabilities for top-k candidates
  std::vector<float> probs(sort_size);
  float sum_exp = 0.0f;

  for (int i = 0; i < sort_size; ++i) {
    float exp_val = std::exp(top_indices_and_scores[i].second - max_score);
    probs[i] = exp_val;
    sum_exp += exp_val;
  }

  // Normalize to get probabilities
  for (int i = 0; i < sort_size; ++i) {
    probs[i] /= sum_exp;
  }

  // Apply top-p (nucleus) filtering
  int filter_size = sort_size;
  if (top_p < 1.0f && top_p > 0.0f) {
    float cumsum = 0.0f;
    for (int i = 0; i < sort_size; ++i) {
      cumsum += probs[i];
      if (cumsum > top_p) {
        filter_size = i + 1;
        break;
      }
    }
    // Renormalize probabilities after filtering
    float sum_filtered = 0.0f;
    for (int i = 0; i < filter_size; ++i) {
      sum_filtered += probs[i];
    }
    for (int i = 0; i < filter_size; ++i) {
      probs[i] /= sum_filtered;
    }
  }

  // Sample from the filtered distribution
  std::discrete_distribution<unsigned int> dist(probs.begin(),
                                                probs.begin() + filter_size);
  unsigned int sampled_offset = dist(rng);

  return top_indices_and_scores[sampled_offset].first;
}
