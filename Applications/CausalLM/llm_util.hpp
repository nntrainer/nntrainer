// SPDX-License-Identifier: Apache-2.0
/**
 *
 * @file   llm_util.hpp
 * @brief  util functions for llm (refactored from main.cpp)
 * @date   21 August 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __LLM_UTIL_HPP__
#define __LLM_UTIL_HPP__ __LLM_UTIL_HPP__

#include <optional>

#include <base_properties.h>
#include <common.h>
#include <layer.h>
#include <model.h>
/***************** ALAIS *******************/
using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;
using ml::train::createLayer;

/****************** UTIL *******************/
/**
 * @brief util functio to make "key=value" from key and value
 *
 * @tparam T type of a value
 * @param key key
 * @param value value
 * @return std::string with "key=value"
 */
template <typename T>
static std::string withKey(const std::string &key, const T &value) {
  std::stringstream ss;
  ss << key << "=" << value;
  return ss.str();
}

/**
 * @brief util function to make "key=value1,value2, ..."  from key and value

 * @tparam T type of a value
 * @param key key
 * @param value list of value
 * @return std::string with "key=value1, value, ...."
 */
template <typename T>
static std::string withKey(const std::string &key,
                           std::initializer_list<T> value) {
  if (std::empty(value)) {
    throw std::invalid_argument("empty data cannot be converted");
  }

  std::stringstream ss;
  ss << key << "=";

  auto iter = value.begin();
  for (; iter != value.end() - 1; ++iter) {
    ss << *iter << ',';
  }
  ss << *iter;

  return ss.str();
}

/**
 * @brief Whether Q4_0 FC layers should be tagged engine=cdsp, so they
 * dispatch their GEMM through HexagonComputeOps (see
 * nntrainer/hexagon/hexagon_context.cpp) instead of the CPU NEON/AVX path.
 * Checked once per process via NNTR_USE_HEXAGON_CDSP; only meaningful on a
 * build with -Denable-hexagon-cdsp=true and libggml-hexagon.so reachable at
 * runtime (see nntr-htp-bridge.cpp in the ggml-hexagon repo) - otherwise
 * layer creation under the "cdsp" context, or the first accelerated GEMM
 * call, will fail.
 */
inline bool useHexagonCdsp() {
  static const bool use_cdsp = std::getenv("NNTR_USE_HEXAGON_CDSP") != nullptr;
  return use_cdsp;
}

/**
 * @brief Append engine=cdsp to a fully_connected layer's properties when
 * useHexagonCdsp() is set. Only meant for FC layers that hold Q4_0 weights
 * (the accel GEMM is Q4_0-specific - see
 * ComputeOps::supports_gemm_q4_0_accel_fp32) - tagging an FP32/Q6_K-weight
 * layer (e.g. embedding, lm_head) this way is harmless but pointless, since
 * the dtype check in float_tensor.cpp's dotQnK will just fall through to the
 * normal CPU path for it anyway.
 */
inline std::vector<std::string>
withHexagonEngine(std::vector<std::string> props) {
  if (useHexagonCdsp()) {
    props.push_back(withKey("engine", "cdsp"));
  }
  return props;
}

/**
 * @brief
 */
template <typename T>
T unwrap(std::optional<T> &&value, const std::string &error_msg) {
  if (value.has_value()) {
    return value.value();
  } else {
    throw std::runtime_error(error_msg);
  }
}

/**
 * @brief generate multi tokens from logits
 * @note This function apply repetition penalty, bad words penalty, and sort to
 * generate multiple tokens
 */
std::vector<unsigned int> generate_multi_tokens(
  float *logits, unsigned int NUM_VOCAB = 0, unsigned int NUM_TARGET_TOKENS = 1,
  float repetition_penalty = 1, unsigned int *input_ids = nullptr,
  unsigned int NUM_INPUT_IDS = 0, unsigned int *bad_words_ids = nullptr,
  unsigned int NUM_BAD_WORDS_IDS = 0);

/**
 * @brief Apply repetition penalty to logits
 */
void applyRepetitionPenalty(float *logits, unsigned int *input_ids,
                            unsigned int NUM_INPUT_IDS,
                            float repetition_penalty = 1);

/**
 * @brief Apply bad words penalty
 */
void applyBadWordsPenalty(float *logits, unsigned int *bad_words_ids,
                          unsigned int NUM_BAD_WORDS_IDS);

/**
 * @brief do sampling to logits with temperature, top-k, top-p
 * @return Sampled token index
 */
unsigned int applyTKP(const float *logits, int len, float temperature,
                      unsigned int top_k, float top_p, std::mt19937 &rng);

#endif // __LLM_UTIL_HPP__
