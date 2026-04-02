// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   neon_setting.h
 * @date   18 Jan 2024
 * @see    https://github.com/nntrainer/nntrainer
 *         https://arxiv.org/abs/1706.03762
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file is for OpenMP setting
 *
 */

/// @note This variable should be optimized by user
/// @todo Must find a general solution to optimize the functionality of
/// multithreading : determining the combination of #threads and size of
/// (M x K) x (K x N) GEMM
/**
 * @brief Function for setting the number of threads to use for GEMM
 *
 * @return size_t& num_threads
 */
inline size_t &GEMM_NUM_THREADS() {
  static size_t num_threads = 1;
  return num_threads;
}
/**
 * @brief Set the gemm num threads
 *
 * @param n num_threads to set
 */
inline void set_gemm_num_threads(size_t n) { GEMM_NUM_THREADS() = n; }
/**
 * @brief Get the gemm num threads
 *
 * @return size_t num_threads
 */
inline size_t get_gemm_num_threads() { return GEMM_NUM_THREADS(); }
/**
 * @brief Function for setting the number of threads to use for GEMV
 *
 * @return size_t& num_threads
 */
inline size_t &GEMV_NUM_THREADS() {
  static size_t num_threads = 1;
  return num_threads;
}
/**
 * @brief Set the gemv num threads
 *
 * @param n num_threads to set
 */
inline void set_gemv_num_threads(size_t n) { GEMV_NUM_THREADS() = n; }
/**
 * @brief Get the gemv num threads
 *
 * @return size_t num_threads
 */
inline size_t get_gemv_num_threads() { return GEMV_NUM_THREADS(); }
