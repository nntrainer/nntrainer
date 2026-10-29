// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jaemin Shin <jaemin980311@gmail.com>
 *
 * @file   ggml_interface_q8_0.cpp
 * @date   20 July 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jaemin Shin <jaemin980311@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  Q8_0 weight GEMM/GEMV interface to the ggml kernels. Built for
 *         every thread backend, since it only uses ThreadManager::parallel_for.
 */

#include <ggml_interface.h>
#include <nntr_ggml_impl.h>
#include <nntr_ggml_impl_utils.h>
#include <thread_manager.h>

#include <algorithm>
#include <vector>

namespace nntrainer {

void __ggml_q8_0_4x4_q8_0_GEMM(const unsigned int M, const unsigned int N,
                               const unsigned int K, const float *A,
                               const unsigned int lda, const void *B,
                               const unsigned int ldb, float *C,
                               const unsigned int ldc) {
  auto &tm = ThreadManager::Global();

  // @todo optimize parameters
  if (M == 1) { // GEMV
    unsigned int B_step = sizeof(block_q8_0) * (K / QK8_0);
    unsigned int blocks_per_row = (K + QK8_0 - 1) / QK8_0;
    unsigned int qa_size = sizeof(block_q8_0) * blocks_per_row;
    std::vector<char> QA = std::vector<char>(qa_size);

    // online quantization for fp32 activation with no packing
    nntr_quantize_row_q8_0(A, QA.data(), K);

    unsigned int chunk_size = 16;
    unsigned int loop = (N + chunk_size - 1) / chunk_size;

    // compute multithreaded GEMV
    tm.parallel_for(0, loop, [=](size_t idx) {
      unsigned int N_step_start = chunk_size * idx;
      unsigned int N_step_end = std::min(chunk_size * (idx + 1), (size_t)N);

      nntr_gemv_q8_0_4x4_q8_0(K, (float *)((C) + N_step_start), N,
                              (void *)((char *)B + N_step_start * B_step),
                              QA.data(), M, N_step_end - N_step_start);
    });
  } else { // GEMM
    unsigned int blocks_per_4_rows = (K + QK8_0 - 1) / QK8_0;
    unsigned int qa_4_rows_size = sizeof(block_q8_0x4) * blocks_per_4_rows;
    const size_t qa_row_size =
      (sizeof(block_q8_0) * K) / QK8_0; // ignore remainder
    unsigned int M4 = M / 4;

    unsigned int qa_size =
      qa_4_rows_size * M4 + static_cast<unsigned int>(qa_row_size) * (M % 4);
    std::vector<char> QA = std::vector<char>(qa_size);
    char *qa_data = QA.data();

    // online quantization for M4 * 4 rows; parallelize over 8-group chunks
    // when there is enough work to amortize waking the pool
    unsigned int quant_chunk = 8;
    if (M4 >= 2 * quant_chunk) {
      tm.parallel_for(0, (M4 + quant_chunk - 1) / quant_chunk, [=](size_t t) {
        unsigned int q_end = std::min(quant_chunk * (t + 1), (size_t)M4);
        for (unsigned int i = quant_chunk * t; i < q_end; i++) {
          nntr_quantize_mat_q8_0_4x4(A + 4 * i * K,
                                     qa_data + i * qa_4_rows_size, K);
        }
      });
    } else {
      for (unsigned int i = 0; i < M4; i++) {
        nntr_quantize_mat_q8_0_4x4(A + 4 * i * K, qa_data + i * qa_4_rows_size,
                                   K);
      }
    }

    // online quantization for remainder
    for (unsigned int i = M4 * 4; i < M; i++) {
      nntr_quantize_row_q8_0(
        (float *)A + i * K,
        (QA.data() + (M4 * qa_4_rows_size) + (i - M4 * 4) * qa_row_size), K);
    }

    size_t row_chunk_size = 16;
    size_t row_loop = (M4 * 4 + row_chunk_size - 1) / row_chunk_size;
    size_t A_step = sizeof(block_q8_0) * (K / QK8_0);

    size_t col_chunk_size = 16;
    size_t col_loop = (N + col_chunk_size - 1) / col_chunk_size;
    size_t B_step = sizeof(block_q8_0) * (K / QK8_0);

    tm.parallel_for(0, col_loop * row_loop, [=](size_t i) {
      unsigned int r = i / col_loop;
      unsigned int c = i % col_loop;

      unsigned int r_start = r * row_chunk_size;
      unsigned int r_end = std::min((unsigned int)(row_chunk_size * (r + 1)),
                                    (unsigned int)(M4 * 4));

      unsigned int c_start = c * col_chunk_size;
      unsigned int c_end =
        std::min((unsigned int)(col_chunk_size * (c + 1)), N);

      nntr_gemm_q8_0_4x4_q8_0(K, (float *)(C + r_start * N + c_start), ldc,
                              (void *)((char *)B + c_start * B_step),
                              (void *)(QA.data() + r_start * A_step),
                              r_end - r_start, c_end - c_start);
    });

    // Compute leftover 1 ~ 3 rows with multithreaded GEMV
    for (unsigned int pb = M4 * 4; pb < M; pb++) {
      unsigned int chunk_size = 16;
      unsigned int loop = (N + chunk_size - 1) / chunk_size;

      tm.parallel_for(0, loop, [=](size_t idx) {
        unsigned int M_step_start = chunk_size * idx;
        unsigned int M_step_end = std::min(chunk_size * (idx + 1), (size_t)N);

        nntr_gemv_q8_0_4x4_q8_0(
          K, (float *)((C + ((pb - M4 * 4) * N) + (M4 * 4 * N)) + M_step_start),
          N, (void *)((char *)B + M_step_start * B_step),
          QA.data() + (M4 * qa_4_rows_size) + (pb - M4 * 4) * qa_row_size, 1,
          M_step_end - M_step_start);
      });
    }
  }
}

void __ggml_q8_0_4x8_q8_0_GEMM(const unsigned int M, const unsigned int N,
                               const unsigned int K, const float *A,
                               const unsigned int lda, const void *B,
                               const unsigned int ldb, float *C,
                               const unsigned int ldc) {
  auto &tm = ThreadManager::Global();

  // @todo optimize parameters
  if (M == 1) { // GEMV
    unsigned int B_step = sizeof(block_q8_0) * (K / QK8_0);
    unsigned int blocks_per_row = (K + QK8_0 - 1) / QK8_0;
    unsigned int qa_size = sizeof(block_q8_0) * blocks_per_row;
    std::vector<char> QA = std::vector<char>(qa_size);

    // online quantization for fp32 activation with no packing
    nntr_quantize_row_q8_0(A, QA.data(), K);

    unsigned int chunk_size = 16;
    unsigned int loop = (N + chunk_size - 1) / chunk_size;

    // compute multithreaded GEMV
    tm.parallel_for(0, loop, [=](size_t idx) {
      unsigned int N_step_start = chunk_size * idx;
      unsigned int N_step_end = std::min(chunk_size * (idx + 1), (size_t)N);

      nntr_gemv_q8_0_4x8_q8_0(K, (float *)((C) + N_step_start), N,
                              (void *)((char *)B + N_step_start * B_step),
                              QA.data(), M, N_step_end - N_step_start);
    });
  } else { // GEMM
    unsigned int blocks_per_4_rows = (K + QK8_0 - 1) / QK8_0;
    unsigned int qa_4_rows_size = sizeof(block_q8_0x4) * blocks_per_4_rows;
    const size_t qa_row_size =
      (sizeof(block_q8_0) * K) / QK8_0; // ignore remainder
    unsigned int M4 = M / 4;

    unsigned int qa_size =
      qa_4_rows_size * M4 + static_cast<unsigned int>(qa_row_size) * (M % 4);
    std::vector<char> QA = std::vector<char>(qa_size);
    char *qa_data = QA.data();

    // online quantization for M4 * 4 rows; parallelize over 8-group chunks
    // when there is enough work to amortize waking the pool
    unsigned int quant_chunk = 8;
    if (M4 >= 2 * quant_chunk) {
      tm.parallel_for(0, (M4 + quant_chunk - 1) / quant_chunk, [=](size_t t) {
        unsigned int q_end = std::min(quant_chunk * (t + 1), (size_t)M4);
        for (unsigned int i = quant_chunk * t; i < q_end; i++) {
          nntr_quantize_mat_q8_0_4x8(A + 4 * i * K,
                                     qa_data + i * qa_4_rows_size, K);
        }
      });
    } else {
      for (unsigned int i = 0; i < M4; i++) {
        nntr_quantize_mat_q8_0_4x8(A + 4 * i * K, qa_data + i * qa_4_rows_size,
                                   K);
      }
    }

    // online quantization for remainder
    for (unsigned int i = M4 * 4; i < M; i++) {
      nntr_quantize_row_q8_0(
        (float *)A + i * K,
        (QA.data() + (M4 * qa_4_rows_size) + (i - M4 * 4) * qa_row_size), K);
    }

    size_t row_chunk_size = 16;
    size_t row_loop = (M4 * 4 + row_chunk_size - 1) / row_chunk_size;
    size_t A_step = sizeof(block_q8_0) * (K / QK8_0);

    size_t col_chunk_size = 16;
    size_t col_loop = (N + col_chunk_size - 1) / col_chunk_size;
    size_t B_step = sizeof(block_q8_0) * (K / QK8_0);

    tm.parallel_for(0, col_loop * row_loop, [=](size_t i) {
      unsigned int r = i / col_loop;
      unsigned int c = i % col_loop;

      unsigned int r_start = r * row_chunk_size;
      unsigned int r_end = std::min((unsigned int)(row_chunk_size * (r + 1)),
                                    (unsigned int)(M4 * 4));

      unsigned int c_start = c * col_chunk_size;
      unsigned int c_end =
        std::min((unsigned int)(col_chunk_size * (c + 1)), N);

      nntr_gemm_q8_0_4x8_q8_0(K, (float *)(C + r_start * N + c_start), ldc,
                              (void *)((char *)B + c_start * B_step),
                              (void *)(QA.data() + r_start * A_step),
                              r_end - r_start, c_end - c_start);
    });

    // Compute leftover 1 ~ 3 rows with multithreaded GEMV
    for (unsigned int pb = M4 * 4; pb < M; pb++) {
      unsigned int chunk_size = 16;
      unsigned int loop = (N + chunk_size - 1) / chunk_size;

      tm.parallel_for(0, loop, [=](size_t idx) {
        unsigned int M_step_start = chunk_size * idx;
        unsigned int M_step_end = std::min(chunk_size * (idx + 1), (size_t)N);

        nntr_gemv_q8_0_4x8_q8_0(
          K, (float *)((C + ((pb - M4 * 4) * N) + (M4 * 4 * N)) + M_step_start),
          N, (void *)((char *)B + M_step_start * B_step),
          QA.data() + (M4 * qa_4_rows_size) + (pb - M4 * 4) * qa_row_size, 1,
          M_step_end - M_step_start);
      });
    }
  }
}

} // namespace nntrainer
