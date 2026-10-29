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
#include <env_compat.h>
#include <layer.h>
#include <model.h>
#include <runtime_config.h>
#include <token_sampler.h>
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
 * @brief Resolve a model builder's backend from the live context registry.
 *        Unset selects registered GPU, then CUDA, then CPU. An unavailable
 *        explicit backend follows Engine::parseComputeEngine's CPU fallback.
 *        The owning layer/facade snapshots this decision before execution.
 */
inline std::string causallm_engine() { return causallm::runtimeEngine(); }

/**
 * @brief Whether a model builder should set mha_core's use_gemm_attention.
 * @param use_flash the model's "use_flash_attention" (nntr_config, default
 *        true).
 * @return use_flash on the cpu engine (the host flash arm), and on a GPU
 *         engine when its device attention arm can run: NNTR_CUDA_ATTN on the
 *         cuda engine (defaulted on by the CUDA context), NNTR_MHA_GPU on the
 *         gpu engine. False otherwise.
 * @note On the cuda engine the activations can live in device-only memory
 *       (NNTR_CUDA_DEV_ACT), which the host per-row attention cannot read, so
 *       a model that leaves this off there cannot run. On the gpu engine
 *       without NNTR_MHA_GPU the per-row path is kept, as before.
 */
inline bool causallm_gemm_attention(bool use_flash) {
  if (!use_flash)
    return false;
  const std::string eng = causallm_engine();
  if (eng == "cpu")
    return true;
  if (eng == "cuda")
    return nntr_env_on("NNTR_CUDA_ATTN");
  return nntr_env_on("NNTR_MHA_GPU");
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
 * @details Thin wrapper over causallm::sampleToken() in token_sampler.h,
 *          which documents the tie and boundary rules.
 * @return Sampled token index
 */
unsigned int applyTKP(const float *logits, int len, float temperature,
                      unsigned int top_k, float top_p,
                      causallm::SamplingRng &rng);

#endif // __LLM_UTIL_HPP__
