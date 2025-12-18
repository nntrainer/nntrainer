// SPDX-License-Identifier: Apache-2.0
/**
 * @file	q4_0_utils.h
 * @date	15 October 2025
 * @brief	This is Q4_0Utils class for utils for Q4_0 quantization format.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Maciej Nalewaj <m.nalewaj@samsung.com>
 * @bug		No known bugs
 */

#ifndef __NNTRAINER_Q4_0_UTILS_H__
#define __NNTRAINER_Q4_0_UTILS_H__

#include <algorithm>
#include <cstdint>
#include <vector>

#define QK4_0 32

template <int K> constexpr int QK_0() {
  if constexpr (K == 4) {
    return 32;
  }
  if constexpr (K == 8) {
    return 32;
  }
  return -1;
}

/**
 * @brief block_q4_0xN
 */
template <int K, int N> struct block {
  uint16_t d[N];                      // deltas for N qK_0 blocks
  int8_t qs[(QK_0<K>() * N * K) / 8]; // quants for N qK_0 blocks
};

using block_q4_0x4 = block<4, 4>;
using block_q4_0x8 = block<4, 8>;

/**
 * @brief block_q4_0
 */
typedef struct {
  uint16_t d;            // delta
  uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;

namespace nntrainer {

/**
 * @class Q4_0Utils class
 * @brief Q4_0Utils class with helpers for Q4_0 format calculation, quantization
 * and dequantization methods.
 */
class Q4_0Utils {
public:
  /**
   * @brief     Unpack one Q4_0x4 block to 4 Q4_0 blocks
   * @param[in] in block_q4_0x4* input Q4_0x4 block
   * @param[out] dst block_q4_0* output vector of 4 Q4_0 blocks
   */
  static void unpackOneBlockQ4_0x4(const block_q4_0x4 *in, block_q4_0 *dst);

  /**
   * @brief     Unpack Q4_0x4 blocks data to Q4_0 format
   * @param[in] src block_q4_0x4* input data in Q4_0x4 blocks format
   * @param[in] data_size number of Q4_0x4 blocks * sizeof(block_q4_0x4)
   * @param[in] nrow number of rows
   * @param[in] K number of columns
   * @param[out] dst block_q4_0* output data in Q4_0 blocks format
   */
  static void unpackBlocksQ4_0x4(const block_q4_0x4 *__restrict src,
                                 size_t data_size, size_t nrow, size_t K,
                                 block_q4_0 *__restrict dst);

  /**
   * @brief     Dequantize weights in block_q4_0x4 format to matrix of floats
   * @param[in] q4_weight_repacked void * input data in format block_q4_0x4
   * @param[in] N number of rows
   * @param[in] K number of columns
   * @param[out] dequantized_weights float * dequantized weights matrix
   */
  static void dequantizeQ4_0x4(const void *q4_weight_repacked, int N, int K,
                               float *dequantized_weights);

  /**
   * @brief     Unpack one Q4_0x8 block to 8 Q4_0 blocks
   * @param[in] in block_q4_0x8* input Q4_0x8 block
   * @param[out] dst block_q4_0* output vector of 8 Q4_0 blocks
   */
  static void unpackOneBlockQ4_0x8(const block_q4_0x8 *in, block_q4_0 *dst);

  /**
   * @brief     Unpack Q4_0x8 blocks data to Q4_0 format
   * @param[in] src block_q4_0x8* input data in Q4_0x8 blocks format
   * @param[in] data_size number of Q4_0x8 blocks * sizeof(block_q4_0x8)
   * @param[in] nrow number of rows
   * @param[in] K number of columns
   * @param[out] dst block_q4_0* output data in Q4_0 blocks format
   */
  static void unpackBlocksQ4_0x8(const block_q4_0x8 *__restrict src,
                                 size_t data_size, size_t nrow, size_t K,
                                 block_q4_0 *__restrict dst);

  /**
   * @brief     Dequantize weights in block_q4_0x8 format to matrix of floats
   * @param[in] q4_weight_repacked void * input data in format block_q4_0x8
   * @param[in] N number of rows
   * @param[in] K number of columns
   * @param[out] dequantized_weights float * dequantized weights matrix
   */
  static void dequantizeQ4_0x8(const void *q4_weight_repacked, int N, int K,
                               float *dequantized_weights);

  /**
   * @brief Transforms data from in-memory layout osv32_isv2 to block_q4_0x8 or
   * block_q4_0x4 in-memory layout.
   * @param N number of rows
   * @param K number of columns
   * @param osv32_weights uint8_t* data of weights in osv32_isv2 layout
   * @param osv32_scales fp16* scales
   * @param scale_group_size group size (32 or 64 or 128)
   * @param q4_0x_block_size output q4_0x block size - number of rows (4 or 8)
   * @param dst_q4_0x void * output data in block_q4_0x8 or block_q4_0x4 layout
   * depending on q4_0x_block_size
   */
  static void transformQ4_0x_FromInt4(size_t N, size_t K,
                                      const uint8_t *osv32_weights,
                                      const uint16_t *osv32_scales,
                                      size_t scale_group_size,
                                      int q4_0x_block_size, void *dst_q4_0x);

  /**
   * @brief     Print the Q4_0 block data
   * @param[in] block Pointer to the Q4_0 block
   */
  static void printBlockQ4_0(const block_q4_0 *block);
};
} // namespace nntrainer

#endif // __NNTRAINER_INT4_UTILS_H__
