// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   gemm_int4_cpu.cpp
 * @brief  CPU implementation of int4 GEMM operation
 * @author Samsung Electronics Co., Ltd.
 * @bug    No known bugs except for NYI items
 *
 */

#include "gemm_int4_cpu.h"
#include "fp16.h"
#include "int4_utils.h"
#include <cmath>
#include <iostream>
#include <vector>

namespace nntrainer {

void gemm_int4_cpu(const void *input, const void *weights, const void *scales,
                   const void *input_scales, float *output, unsigned int M,
                   unsigned int N, unsigned int K,
                   unsigned int quantization_group_size) {

  const int8_t *q_input = static_cast<const int8_t *>(input);
  const uint8_t *q_weights = static_cast<const uint8_t *>(weights);
  const uint16_t *w_scales = static_cast<const uint16_t *>(scales);
  const uint16_t *in_scales = static_cast<const uint16_t *>(input_scales);

  // Buffer to hold one dequantized row of weights (size K)
  // Since weights are packed as N rows x K columns (passed to quantize),
  // the n-th row of weights corresponds to the n-th column of the GEMM matrix
  // B.
  std::vector<float> dequantized_weight_row(K);

  // Input quantization layout parameters
  // Matches cpu_quantize_input_int8_pad logic
  unsigned int alignK = (K + quantization_group_size - 1) /
                        quantization_group_size * quantization_group_size;
  unsigned int groups_in_row = alignK / quantization_group_size;

  // Iterate over columns of output (N)
  for (unsigned int n = 0; n < N; ++n) {
    // 1. Dequantize the n-th row of weights (which is n-th col of B)
    Int4Utils::dequantizePackedRow(
      const_cast<uint8_t *>(q_weights), const_cast<uint16_t *>(w_scales), N, K,
      quantization_group_size, n, dequantized_weight_row.data());

    // 2. Compute dot product with each row of input (M)
    for (unsigned int m = 0; m < M; ++m) {
      float sum = 0.0f;

      for (unsigned int k = 0; k < K; ++k) {
        // Calculate index for quantized input
        unsigned int group_id_in_row = k / quantization_group_size;
        unsigned int global_group_id = m * groups_in_row + group_id_in_row;
        unsigned int offset_in_group = k % quantization_group_size;

        // Input is stored block-wise: group_id * group_size + offset
        unsigned int input_idx =
          global_group_id * quantization_group_size + offset_in_group;

        int8_t q_val = q_input[input_idx];

        // Get input scale
        // scales[group_id * 2] = scale, scales[group_id * 2 + 1] =
        // offset/unused
        float in_scale = compute_fp16_to_fp32(in_scales[global_group_id * 2]);

        float in_val = static_cast<float>(q_val) * in_scale;
        float w_val = dequantized_weight_row[k];

        sum += in_val * w_val;
      }

      output[m * N + n] = sum;
    }
  }
}

void gemm_int4_cpu_packed_block(const void *input, const void *weights,
                                const void *scales, const void *input_scales,
                                float *output, unsigned int M, unsigned int N,
                                unsigned int K,
                                unsigned int quantization_group_size) {
  const int8_t *q_input = static_cast<const int8_t *>(input);
  const uint8_t *q_weights = static_cast<const uint8_t *>(weights);
  const uint16_t *w_scales = static_cast<const uint16_t *>(scales);
  const uint16_t *in_scales = static_cast<const uint16_t *>(input_scales);

  // Initialize output to 0
  for (unsigned int i = 0; i < M * N; ++i) {
    output[i] = 0.0f;
  }

  // Input quantization layout parameters
  unsigned int alignK = (K + quantization_group_size - 1) /
                        quantization_group_size * quantization_group_size;
  unsigned int groups_in_row = alignK / quantization_group_size;

  // Outer loop: Height / 32 (N direction)
  // Blocks of 32 rows (output channels)
  for (unsigned int n_blk = 0; n_blk < N / 32; ++n_blk) {
    unsigned int n_start = n_blk * 32;

    // Inner loop: Width / 2 (K direction)
    // Blocks of 2 columns (input channels)
    for (unsigned int k_blk = 0; k_blk < K / 2; ++k_blk) {
      unsigned int k_start = k_blk * 2;

      // Pointer to the 32-byte block of weights
      // Block index = n_blk * (K / 2) + k_blk
      // Each block is 32 bytes
      const uint8_t *w_block = q_weights + (n_blk * (K / 2) + k_blk) * 32;

      // Process for all M rows of input
      for (unsigned int m = 0; m < M; ++m) {
        // Load input values and scale
        // k_start and k_start+1 are in the same group (assuming group_size >=
        // 2)
        unsigned int group_id_in_row = k_start / quantization_group_size;
        unsigned int global_group_id = m * groups_in_row + group_id_in_row;

        // Input scale
        float i_scale = compute_fp16_to_fp32(in_scales[global_group_id * 2]);

        // Input indices
        // Note: input is quantized with padding, so we use block addressing
        unsigned int offset_in_group = k_start % quantization_group_size;
        unsigned int input_idx =
          global_group_id * quantization_group_size + offset_in_group;

        float val0 = static_cast<float>(q_input[input_idx]) * i_scale;
        float val1 = static_cast<float>(q_input[input_idx + 1]) * i_scale;

        // Iterate over the 32 rows in the weight block
        for (unsigned int i = 0; i < 32; ++i) {
          unsigned int n = n_start + i;
          uint8_t w_byte = w_block[i];

          // Decode weights (2 per byte)
          // Low nibble is first weight (k_start), High nibble is second weight
          // (k_start+1)
          int8_t w0_int4 = (w_byte & 0x0F);
          int8_t w1_int4 = (w_byte >> 4);

          // Sign extension
          if (w0_int4 >= 8)
            w0_int4 -= 16;
          if (w1_int4 >= 8)
            w1_int4 -= 16;

          // Get weight scale
          // Scale depends on N (row) and K group
          unsigned int w_group_id = k_start / quantization_group_size;
          // Scales are stored as N x (K / group_size)
          // But wait, K passed to quantizeAndRepack was actually K (columns
          // count). So scales are N rows x (K/group_size) cols. Index = n * (K
          // / group_size) + w_group_id
          unsigned int scale_idx =
            n * (K / quantization_group_size) + w_group_id;
          float w_scale = compute_fp16_to_fp32(w_scales[scale_idx]);

          float w0 = static_cast<float>(w0_int4) * w_scale;
          float w1 = static_cast<float>(w1_int4) * w_scale;

          output[m * N + n] += val0 * w0 + val1 * w1;
        }
      }
    }
  }
}

} // namespace nntrainer
