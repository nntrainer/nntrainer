// SPDX-License-Identifier: Apache-2.0
/**
 * @file	int4_utils_cuda.cu
 * @date	19 December 2025
 * @brief	CUDA implementation for INT4 dequantization kernels
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Daekyoung Jung <dk11.jung@samsung.com>
 * @bug		No known bugs
 */
#include "dequantize_cuda.h"
#include <cuda_fp16.h>

namespace nntrainer {
namespace Int4UtilsCuda {

__global__ void
dequantize_rows_kernel(const uint8_t *weights, const uint16_t *scales,
                       unsigned int rows_count, unsigned int columns_count,
                       unsigned int group_size, const float *indices,
                       unsigned int num_indices, float *output) {
  unsigned int col_block_idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int idx_in_num_indices = blockIdx.y;

  // osv32_isv2 layout constants
  const unsigned int ROW_BLOCK_SIZE = 32;
  const unsigned int COLUMN_BLOCK_SIZE = 2;

  unsigned int rows_count_pad =
    ((rows_count + ROW_BLOCK_SIZE - 1) / ROW_BLOCK_SIZE) * ROW_BLOCK_SIZE;
  unsigned int columns_count_pad =
    ((columns_count + group_size - 1) / group_size) * group_size;
  unsigned int column_blocks_count =
    (columns_count_pad + COLUMN_BLOCK_SIZE - 1) / COLUMN_BLOCK_SIZE;

  if (col_block_idx >= column_blocks_count ||
      idx_in_num_indices >= num_indices) {
    return;
  }

  // Indices are stored as float, convert to unsigned int
  unsigned int row_index = (unsigned int)indices[idx_in_num_indices];

  // Safety check for out of bounds row_index
  if (row_index >= rows_count) {
    return;
  }

  // Addressing math for OS_IS_YX_OSV32_ISV2 layout
  unsigned int row_block_id = row_index / ROW_BLOCK_SIZE;
  unsigned int i_in_block = row_index % ROW_BLOCK_SIZE;
  unsigned int bytes_per_row_block_span = column_blocks_count * ROW_BLOCK_SIZE;
  unsigned int row_block_base =
    row_block_id * bytes_per_row_block_span + i_in_block;

  unsigned int weights_idx = row_block_base + col_block_idx * ROW_BLOCK_SIZE;
  uint8_t packed_byte = weights[weights_idx];

  // Signed 4-bit unpacking
  // Lower 4 bits
  int q_lo = (int8_t)((packed_byte & 0x0F) << 4) >> 4;
  // Upper 4 bits
  int q_hi = (int8_t)(packed_byte & 0xF0) >> 4;

  unsigned int col_lo = col_block_idx * COLUMN_BLOCK_SIZE;
  unsigned int col_hi = col_lo + 1;

  // Output pointer for this specific token/index
  float *out_row_ptr = output + idx_in_num_indices * columns_count;

  if (col_lo < columns_count) {
    unsigned int g_lo = col_lo / group_size;
    half s_lo_half = *(reinterpret_cast<const half *>(
      &scales[row_index + g_lo * rows_count_pad]));
    float s_lo = __half2float(s_lo_half);
    out_row_ptr[col_lo] = (float)q_lo * s_lo;
  }

  if (col_hi < columns_count) {
    unsigned int g_hi = col_hi / group_size;
    half s_hi_half = *(reinterpret_cast<const half *>(
      &scales[row_index + g_hi * rows_count_pad]));
    float s_hi = __half2float(s_hi_half);
    out_row_ptr[col_hi] = (float)q_hi * s_hi;
  }
}

void dequantize_rows_cuda(const uint8_t *weights, const uint16_t *scales,
                          unsigned int rows_count, unsigned int columns_count,
                          unsigned int group_size, const float *indices,
                          unsigned int num_indices, float *output,
                          cudaStream_t stream) {
  if (num_indices == 0)
    return;

  unsigned int group_size_local = group_size;
  unsigned int columns_count_pad =
    ((columns_count + group_size_local - 1) / group_size_local) *
    group_size_local;
  unsigned int column_blocks_count = (columns_count_pad + 1) / 2;

  const int threads_per_block = 256;
  dim3 block(threads_per_block);
  dim3 grid((column_blocks_count + threads_per_block - 1) / threads_per_block,
            num_indices);

  dequantize_rows_kernel<<<grid, block, 0, stream>>>(
    weights, scales, rows_count, columns_count, group_size, indices,
    num_indices, output);
}

} // namespace Int4UtilsCuda
} // namespace nntrainer
