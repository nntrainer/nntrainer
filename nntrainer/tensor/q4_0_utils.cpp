// SPDX-License-Identifier: Apache-2.0
/**
 * @file	q4_0_utils.cpp
 * @date	15 October 2025
 * @brief	This is Q4_0Utils class for utils for Q4_0 quantization format.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Maciej Nalewaj <m.nalewaj@samsung.com>
 * @bug		No known bugs
 */

#include <cassert>
#include <cmath>

#include "cpu_backend.h"
#include "fp16.h"
#include "int4_utils.h"
#include "nntrainer_error.h"
#include "q4_0_utils.h"
#include "util_func.h"

namespace nntrainer {

void Q4_0Utils::unpackOneBlockQ4_0x4(const block_q4_0x4 *in, block_q4_0 *dst) {
  unsigned int blck_size_interleave = 8;

  for (int i = 0; i < 4; i++) {
    dst[i].d = in->d[i];
  }

  const int end = QK4_0 * 2 / blck_size_interleave;
  const uint64_t xor_mask = 0x8888888888888888ULL;

  for (int i = 0; i < end; ++i) {
    int dst_id = i % 4;
    int dst_offset = (i / 4) * blck_size_interleave;
    int src_offset = i * blck_size_interleave;

    uint64_t elems;
    memcpy(&elems, &in->qs[src_offset], sizeof(uint64_t));
    elems ^= xor_mask;
    memcpy(&dst[dst_id].qs[dst_offset], &elems, sizeof(uint64_t));
  }
}

void Q4_0Utils::unpackBlocksQ4_0x4(const block_q4_0x4 *__restrict src,
                                   size_t data_size, size_t nrow, size_t K,
                                   block_q4_0 *__restrict dst) {
  int interleave_block = 4;

  const block_q4_0x4 *src_ = src;
  block_q4_0 *dst_ = (block_q4_0 *)dst;
  block_q4_0 dst_tmp[4];
  int nblocks = K / QK4_0;

  assert(data_size == (nrow / 4) * nblocks * sizeof(block_q4_0x4));

  for (size_t b = 0; b < nrow; b += interleave_block) {
    for (int64_t x = 0; x < nblocks; x++) {
      unpackOneBlockQ4_0x4(src_++, dst_tmp);

      for (size_t i = 0; i < interleave_block; i++) {
        dst_[x + i * nblocks] = dst_tmp[i];
      }
    }
    dst_ += interleave_block * nblocks;
  }
}

void Q4_0Utils::dequantizeQ4_0x4(const void *q4_weight_repacked, int N, int K,
                                 float *dequantized_weights) {
  assert(K % QK4_0 == 0);
  assert(N % 4 == 0);
  size_t data_size = (K / QK4_0) * (N / 4) * sizeof(block_q4_0x4);
  std::vector<uint8_t> q4_weight_out(data_size);
  unpackBlocksQ4_0x4((block_q4_0x4 *)q4_weight_repacked, data_size, N, K,
                     (block_q4_0 *)q4_weight_out.data());

  nntrainer::dequantize_row_q4_0((const void *)q4_weight_out.data(),
                                 dequantized_weights, K * N);
}

void Q4_0Utils::unpackOneBlockQ4_0x8(const block_q4_0x8 *in, block_q4_0 *dst) {
  unsigned int blck_size_interleave = 8;

  for (int i = 0; i < 8; i++) {
    dst[i].d = in->d[i];
  }

  const int end = QK4_0 * 4 / blck_size_interleave;
  const uint64_t xor_mask = 0x8888888888888888ULL;

  for (int i = 0; i < end; ++i) {
    int dst_id = i % 8;
    int dst_offset = (i / 8) * blck_size_interleave;
    int src_offset = i * blck_size_interleave;

    uint64_t elems;
    memcpy(&elems, &in->qs[src_offset], sizeof(uint64_t));
    elems ^= xor_mask;
    memcpy(&dst[dst_id].qs[dst_offset], &elems, sizeof(uint64_t));
  }
}

void Q4_0Utils::unpackBlocksQ4_0x8(const block_q4_0x8 *__restrict src,
                                   size_t data_size, size_t nrow, size_t K,
                                   block_q4_0 *__restrict dst) {
  int interleave_block = 8;

  const block_q4_0x8 *src_ = src;
  block_q4_0 *dst_ = (block_q4_0 *)dst;
  block_q4_0 dst_tmp[8];
  int nblocks = K / QK4_0;

  assert(data_size == (nrow / 8) * nblocks * sizeof(block_q4_0x8));

  for (size_t b = 0; b < nrow; b += interleave_block) {
    for (int64_t x = 0; x < nblocks; x++) {
      unpackOneBlockQ4_0x8(src_++, dst_tmp);

      for (size_t i = 0; i < interleave_block; i++) {
        dst_[x + i * nblocks] = dst_tmp[i];
      }
    }
    dst_ += interleave_block * nblocks;
  }
}

void Q4_0Utils::dequantizeQ4_0x8(const void *q4_weight_repacked, int N, int K,
                                 float *dequantized_weights) {
  assert(K % QK4_0 == 0);
  assert(N % 8 == 0);
  size_t data_size = (K / QK4_0) * (N / 8) * sizeof(block_q4_0x8);
  std::vector<uint8_t> q4_weight_out(data_size);
  unpackBlocksQ4_0x8((block_q4_0x8 *)q4_weight_repacked, data_size, N, K,
                     (block_q4_0 *)q4_weight_out.data());

  nntrainer::dequantize_row_q4_0((const void *)q4_weight_out.data(),
                                 dequantized_weights, K * N);
}

inline static void nntr_make_block_q4_0x4(const block_q4_0 *in,
                                          block_q4_0x4 *out) {
  constexpr size_t IN_CNT = 4;
  constexpr size_t HALF_SIZE = 8;

  for (int i = 0; i < IN_CNT; ++i) {
    out->d[i] = in[i].d;
  }

  for (int i = 0; i < IN_CNT; ++i) {
    memcpy(&out->qs[i * HALF_SIZE], &in[i].qs[0], HALF_SIZE);
  }
  for (int i = 0; i < IN_CNT; ++i) {
    memcpy(&out->qs[IN_CNT * HALF_SIZE + i * HALF_SIZE], &in[i].qs[8],
           HALF_SIZE);
  }
}

inline static void nntr_make_block_q4_0x8(const block_q4_0 *in,
                                          block_q4_0x8 *out) {
  constexpr size_t IN_CNT = 8;
  constexpr size_t HALF_SIZE = 8;

  for (int i = 0; i < IN_CNT; ++i) {
    out->d[i] = in[i].d;
  }

  for (int i = 0; i < IN_CNT; ++i) {
    memcpy(&out->qs[i * HALF_SIZE], &in[i].qs[0], HALF_SIZE);
  }
  for (int i = 0; i < IN_CNT; ++i) {
    memcpy(&out->qs[IN_CNT * HALF_SIZE + i * HALF_SIZE], &in[i].qs[8],
           HALF_SIZE);
  }
}

void Q4_0Utils::transformQ4_0x_FromInt4(size_t N, size_t K,
                                        const uint8_t *osv32_weights,
                                        const uint16_t *osv32_scales,
                                        size_t scale_group_size,
                                        int q4_0x_block_size, void *dst_q4_0x) {

  NNTR_THROW_IF((!(scale_group_size == 32 || scale_group_size == 64 ||
                   scale_group_size == 128)),
                std::invalid_argument)
    << "Scale group size must be 32/64/128";
  NNTR_THROW_IF(K % QK4_0 != 0, std::invalid_argument)
    << "K size must be divisable by QK4_0 (32)";
  NNTR_THROW_IF(N % 8 != 0, std::invalid_argument)
    << "N size must be divisable by 8";
  NNTR_THROW_IF((!(q4_0x_block_size == 4 || q4_0x_block_size == 8)),
                std::invalid_argument)
    << "q4_0x_block_size must be 4 or 8";

  static constexpr const size_t ROW_BLOCK_SIZE = 32;
  static constexpr const size_t COLUMN_BLOCK_SIZE = 2;

  uint8_t int4_weight[16];
  uint16_t scale;
  block_q4_0 dst_tmp[8];
  uint8_t *dst_ = reinterpret_cast<uint8_t *>(dst_q4_0x);

  // --- Layout ---
  const size_t rows_count_pad = align(N, ROW_BLOCK_SIZE);
  const size_t columns_count_pad = align(K, ROW_BLOCK_SIZE);
  const size_t column_blocks_count =
    columns_count_pad / COLUMN_BLOCK_SIZE; // COLUMN_BLOCK_SIZE == 2
  const size_t bytes_per_row_block_span = column_blocks_count * ROW_BLOCK_SIZE;

  for (size_t row_id = 0; row_id < N; row_id += q4_0x_block_size) {
    const size_t row_block_id = row_id / ROW_BLOCK_SIZE;
    size_t i_in_block = row_id % ROW_BLOCK_SIZE;
    for (int64_t column_idx = 0; column_idx < K; column_idx += QK4_0) {
      for (size_t i = 0; i < q4_0x_block_size; i++) {
        int row_idx = row_id + i;
        // Address the bytes for this row
        const size_t row_block_base =
          row_block_id * bytes_per_row_block_span + i_in_block + i;
        int index0 = row_block_base + (column_idx / 2) * ROW_BLOCK_SIZE;

        for (size_t column_block_id = 0; column_block_id < 16;
             ++column_block_id) {
          int4_weight[column_block_id] =
            osv32_weights[index0 + column_block_id * ROW_BLOCK_SIZE];
        }
        scale = osv32_scales[row_idx +
                             (column_idx / scale_group_size) * rows_count_pad];

        create_q4_0_weights(int4_weight, dst_tmp[i].qs);
        dst_tmp[i].d = scale;
      }
      // Repack Q4_0 data
      if (q4_0x_block_size == 4) {
        nntr_make_block_q4_0x4(dst_tmp, (block_q4_0x4 *)dst_);
      } else {
        nntr_make_block_q4_0x8(dst_tmp, (block_q4_0x8 *)dst_);
      }
      dst_ += q4_0x_block_size * sizeof(block_q4_0);
    }
  }
}

void Q4_0Utils::printBlockQ4_0(const block_q4_0 *block) {
  printf("Q4_0: ");
  for (int i = 0; i < 16; i++) {
    printf("%i %i ", block->qs[i] & 0x0F, (block->qs[i] >> 4) & 0x0F);
  }
  printf("| scale:%f\n", compute_fp16_to_fp32(block->d));
}

} // namespace nntrainer
