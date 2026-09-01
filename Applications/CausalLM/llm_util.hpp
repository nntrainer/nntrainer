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

#include <cstdlib> // getenv (causallm_engine); upstream dropped algorithm/math.h
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
 * @brief Compute engine for CausalLM model layers.
 * @return "gpu" by default (the v8c OpenCL inference path), or "cpu" when the
 *         NNTR_ENGINE=cpu env var is set. engine=cpu runs the model on the
 *         standard CPU layers + CpuComputeOps (e.g. Q4_0-FP32 host inference /
 *         CPU-only deployment), instead of the engine=gpu Cl layers whose
 *         ClComputeOps throws NI for plain CPU BLAS ops. An absent engine prop
 *         already defaults to CPU in LayerNode; this keeps the explicit GPU
 *         default for backward compatibility while making CPU one env away.
 */
// inline (not static): several sibling TUs include this header without ever
// calling causallm_engine(), which trips -Werror=unused-function on the
// static-with-internal-linkage form; inline exempts it from that warning and
// (correctly, since it's a pure function of the environment) gives the whole
// binary a single cached instance instead of one per TU.
inline std::string causallm_engine() {
  static const std::string eng = []() -> std::string {
    const char *e = std::getenv("NNTR_ENGINE");
    if (e != nullptr) {
      const std::string s(e);
      if (s == "cpu")
        return "cpu";
      if (s == "cuda") // additive NVIDIA CUDA backend (engine=cuda)
        return "cuda";
    }
#if defined(ENABLE_OPENCL)
    return "gpu";
#else
    // No OpenCL "gpu" Context is registered in this build (e.g. the FP32 CPU
    // reference / unittest build), so default to cpu instead of throwing
    // "[Engine] gpu Context is not registered" at model build.
    return "cpu";
#endif
  }();
  return eng;
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
