// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Michal Wlasiuk <testmailsmtp12345@gmail.com>
 * Copyright (C) 2025 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   ggml_interface_omp.cpp
 * @date   15 April 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Michal Wlasiuk <testmailsmtp12345@gmail.com>
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Function interface to use ggml lib from cpu_backend
 */

#include <ggml_interface.h>
#include <nntr_ggml_impl.h>
#include <nntr_ggml_impl_utils.h>
#include <thread_manager.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace nntrainer {

template <>
void __ggml_q4_0_4x8_q8_0_GEMM(const unsigned int M, const unsigned int N,
                               const unsigned int K, const float *A,
                               const unsigned int lda, const void *B,
                               const unsigned int ldb, float *C,
                               const unsigned int ldc) {
  auto &tm = ThreadManager::Global();

  if (M == 1) { // GEMV
    unsigned int B_step = sizeof(block_q4_0) * (K / QK4_0);
    unsigned int blocks_per_row = (K + QK8_0 - 1) / QK8_0;
    unsigned int qa_size = sizeof(block_q8_0) * blocks_per_row;
    std::vector<char> QA = std::vector<char>(qa_size);

    // online quantization for fp32 activation with no packing
    nntr_quantize_row_q8_0(A, QA.data(), K);

    unsigned int chunk_size = 16;
    unsigned int loop = (N + chunk_size - 1) / chunk_size;

    // compute multithreaded GEMV
    tm.parallel_for(0, loop, [=](size_t idx) {
      unsigned int M_step_start = chunk_size * idx;
      unsigned int M_step_end = std::min(chunk_size * (idx + 1), (size_t)N);

      nntr_gemv_q4_0_4x8_q8_0(K, (float *)((C) + M_step_start), N,
                              (void *)((char *)B + M_step_start * B_step),
                              QA.data(), M, M_step_end - M_step_start);
    });
  } else if (M % 4 != 0) {
    unsigned int blocks_per_4_rows = (K + QK8_0 - 1) / QK8_0;
    unsigned int qa_4_rows_size = sizeof(block_q8_0x4) * blocks_per_4_rows;
    const size_t qa_row_size =
      (sizeof(block_q8_0) * K) / QK8_0; // ignore remainder
    unsigned int M4 = M / 4;

    unsigned int qa_size = qa_4_rows_size * M4 + qa_row_size * (M % 4);
    std::vector<char> QA = std::vector<char>(qa_size);

    // online quantization for M4 * 4 rows
    for (unsigned int i = 0; i < M4; i++) {
      nntr_quantize_mat_q8_0_4x8(A + 4 * i * K, QA.data() + i * qa_4_rows_size,
                                 K);
    }

    // online quantization for remainder
    for (unsigned int i = M4 * 4; i < M; i++) {
      nntr_quantize_row_q8_0(
        (float *)A + i * K,
        (QA.data() + (M4 * qa_4_rows_size) + (i - M4 * 4) * qa_row_size), K);
    }

    // Compute 4-divisible-M row portion with multithreaded GEMM
    unsigned int row_chunk_size = 16;
    unsigned int row_loop = (M4 * 4 + row_chunk_size - 1) / row_chunk_size;
    unsigned int A_step = sizeof(block_q8_0) * (K / QK8_0);

    unsigned int col_chunk_size = 16;
    unsigned int col_loop = (N + col_chunk_size - 1) / col_chunk_size;
    unsigned int B_step = sizeof(block_q4_0) * (K / QK4_0);

    tm.parallel_for(0, col_loop * row_loop, [=](size_t i) {
      unsigned int r = i / col_loop;
      unsigned int c = i % col_loop;

      unsigned int r_start = r * row_chunk_size;
      unsigned int r_end = std::min(row_chunk_size * (r + 1), M4 * 4);

      unsigned int c_start = c * col_chunk_size;
      unsigned int c_end = std::min(col_chunk_size * (c + 1), N);

      nntr_gemm_q4_0_4x8_q8_0(K, (float *)(C + r_start * N + c_start), ldc,
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

        nntr_gemv_q4_0_4x8_q8_0(
          K, (float *)((C + ((pb - M4 * 4) * N) + (M4 * 4 * N)) + M_step_start),
          N, (void *)((char *)B + M_step_start * B_step),
          QA.data() + (M4 * qa_4_rows_size) + (pb - M4 * 4) * qa_row_size, 1,
          M_step_end - M_step_start);
      });
    }
  } else { // GEMM
    unsigned int blocks_per_4_rows = (K + QK8_0 - 1) / QK8_0;
    unsigned int qa_4_rows_size = sizeof(block_q8_0x4) * blocks_per_4_rows;
    unsigned int M4 = M / 4; // M % 4 == 0

    unsigned int qa_size = qa_4_rows_size * M4;
    std::vector<char> QA = std::vector<char>(qa_size);

    for (int i = 0; i < static_cast<int>(M4); i++) {
      nntr_quantize_mat_q8_0_4x8(A + 4 * i * K, QA.data() + i * qa_4_rows_size,
                                 K);
    }

    unsigned int row_chunk_size = 16;
    unsigned int row_loop = (M + row_chunk_size - 1) / row_chunk_size;
    unsigned int A_step = sizeof(block_q8_0) * (K / QK8_0);

    unsigned int col_chunk_size = 16;
    unsigned int col_loop = (N + col_chunk_size - 1) / col_chunk_size;
    unsigned int B_step = sizeof(block_q4_0) * (K / QK4_0);

    tm.parallel_for(0, col_loop * row_loop, [=](size_t i) {
      unsigned int r = i / col_loop;
      unsigned int c = i % col_loop;

      unsigned int r_start = r * row_chunk_size;
      unsigned int r_end = std::min(row_chunk_size * (r + 1), M);

      unsigned int c_start = c * col_chunk_size;
      unsigned int c_end = std::min(col_chunk_size * (c + 1), N);

      nntr_gemm_q4_0_4x8_q8_0(K, (float *)(C + r_start * N + c_start), ldc,
                              (void *)((char *)B + c_start * B_step),
                              (void *)(QA.data() + r_start * A_step),
                              r_end - r_start, c_end - c_start);
    });
  }
}

template <>
void __ggml_q4_0_4x8_q8_0_GEMM(const unsigned int M,
                               std::vector<unsigned int> Ns,
                               const unsigned int K, const float *A,
                               const unsigned int lda, std::vector<void *> Bs,
                               std::vector<unsigned int> ldbs,
                               std::vector<float *> Cs,
                               std::vector<unsigned int> ldcs) {
  int NB_COLS = 4;
  int B_step = sizeof(block_q4_0) * (K / QK4_0);
  int blocks_per_4_rows = (K + QK8_0 - 1) / QK8_0;

  auto &tm = ThreadManager::Global();
  unsigned int thread_num = tm.getComputeThreadCount();

  if (M == 1) {
    int qa_size = sizeof(block_q8_0) * blocks_per_4_rows;
    std::vector<char> QA = std::vector<char>(qa_size);
    auto qa_data = QA.data();
    nntr_quantize_row_q8_0(A, qa_data, K);
    if (std::all_of(Ns.begin(), Ns.end(),
                    [](unsigned int n) { return n <= 256; })) {
      for (unsigned int num_w = 0; num_w < Ns.size(); ++num_w) {
        unsigned int N = Ns[num_w];
        float *C = Cs[num_w];
        void *B = Bs[num_w];

        unsigned int M_step_start = 0;
        unsigned int M_step_end = N;
        M_step_start = (M_step_start % NB_COLS)
                         ? M_step_start + NB_COLS - (M_step_start % NB_COLS)
                         : M_step_start;
        M_step_end = (M_step_end % NB_COLS)
                       ? M_step_end + NB_COLS - (M_step_end % NB_COLS)
                       : M_step_end;

        nntr_gemv_q4_0_4x8_q8_0(K, (float *)(C + M_step_start), N,
                                (void *)((char *)B + M_step_start * B_step),
                                QA.data(), M, M_step_end - M_step_start);
      }
    } else {
      // Single-threaded (n_threads=1 in original)
      for (unsigned int num_w = 0; num_w < Ns.size(); ++num_w) {
        unsigned int N = Ns[num_w];
        float *C = Cs[num_w];
        void *B = Bs[num_w];
        unsigned int M_step_start = 0;
        unsigned int M_step_end = N;

        M_step_start = (M_step_start % NB_COLS)
                         ? M_step_start + NB_COLS - (M_step_start % NB_COLS)
                         : M_step_start;
        M_step_end = (M_step_end % NB_COLS)
                       ? M_step_end + NB_COLS - (M_step_end % NB_COLS)
                       : M_step_end;

        nntr_gemv_q4_0_4x8_q8_0(K, (float *)(C + M_step_start), N,
                                (void *)((char *)B + M_step_start * B_step),
                                QA.data(), M, M_step_end - M_step_start);
      }
    }
  } else {
    unsigned int qa_4_rows_size = sizeof(block_q8_0x4) * blocks_per_4_rows;
    const size_t qa_row_size = (sizeof(block_q8_0) * K) / QK8_0;

    unsigned int M4 = ((M - M % 4) / 4);
    unsigned int qa_size = qa_4_rows_size * (((M >> 2) << 2) / 4 + 1);

    std::vector<char> QA = std::vector<char>(qa_size);

    for (unsigned int i = 0; i < M4; i++) {
      nntr_quantize_mat_q8_0_4x8(A + 4 * i * K, QA.data() + i * qa_4_rows_size,
                                 K);
    }

    for (unsigned int i = M4 * 4; i < M; i++) {
      nntr_quantize_row_q8_0(
        (float *)A + i * K,
        (QA.data() + (M4 * qa_4_rows_size) + (i - M4 * 4) * qa_row_size), K);
    }

    tm.parallel_for_chunked(thread_num, [&](size_t i) {
      for (unsigned int num_w = 0; num_w < Ns.size(); ++num_w) {
        unsigned int N = Ns[num_w];
        unsigned int ldc = ldcs[num_w];

        float *C = Cs[num_w];
        void *B = Bs[num_w];

        unsigned int src0_start = (i * N) / thread_num;
        unsigned int src0_end = ((i + 1) * N) / thread_num;

        src0_start = (src0_start % NB_COLS)
                       ? src0_start + NB_COLS - (src0_start % NB_COLS)
                       : src0_start;

        src0_end = (src0_end % NB_COLS)
                     ? src0_end + NB_COLS - (src0_end % NB_COLS)
                     : src0_end;

        nntr_gemm_q4_0_4x8_q8_0(K, (float *)(C + src0_start), ldc,
                                (void *)((char *)B + src0_start * B_step),
                                QA.data(), M4 * 4, src0_end - src0_start);
      }
    });

    if (M4 * 4 != M) {
      tm.parallel_for_chunked(thread_num, [&](size_t thread_idx) {
        for (unsigned int num_w = 0; num_w < Ns.size(); ++num_w) {
          unsigned int N = Ns[num_w];
          unsigned int ldc = ldcs[num_w];
          float *C = Cs[num_w];
          void *B = Bs[num_w];

          for (int pb = M4 * 4; pb < static_cast<int>(M); pb++) {
            unsigned int M_step_start = (thread_idx * N) / thread_num;
            unsigned int M_step_end = ((thread_idx + 1) * N) / thread_num;
            M_step_start = (M_step_start % NB_COLS)
                             ? M_step_start + NB_COLS - (M_step_start % NB_COLS)
                             : M_step_start;
            M_step_end = (M_step_end % NB_COLS)
                           ? M_step_end + NB_COLS - (M_step_end % NB_COLS)
                           : M_step_end;

            nntr_gemv_q4_0_4x8_q8_0(
              K,
              (float *)((C + ((pb - M4 * 4) * N) + (M4 * 4 * N)) +
                        M_step_start),
              N, (void *)((char *)B + M_step_start * B_step),
              QA.data() + (M4 * qa_4_rows_size) + (pb - M4 * 4) * qa_row_size,
              1, M_step_end - M_step_start);
          }
        }
      });
    }
  }
}

void __ggml_q4_0_8x8_q8_0_GEMM(const unsigned int M, const unsigned int N,
                               const unsigned int K, const float *A,
                               const unsigned int lda, const void *B,
                               const unsigned int ldb, float *C,
                               const unsigned int ldc) {
  auto &tm = ThreadManager::Global();
  unsigned int thread_num = tm.getComputeThreadCount();

  if (M == 1) { // GEMV
    unsigned int B_step = sizeof(block_q4_0) * (K / QK4_0);
    unsigned int blocks_per_row = (K + QK8_0 - 1) / QK8_0;
    unsigned int qa_size = sizeof(block_q8_0) * blocks_per_row;
    std::vector<char> QA = std::vector<char>(qa_size);
    nntr_quantize_row_q8_0(A, QA.data(), K);

    tm.parallel_for_chunked(thread_num, [=](size_t thread_idx) {
      unsigned int M_step_start = (thread_idx * N) / thread_num;
      unsigned int M_step_end = ((thread_idx + 1) * N) / thread_num;

      M_step_start = (M_step_start % 8) ? M_step_start + 8 - (M_step_start % 8)
                                        : M_step_start;
      M_step_end =
        (M_step_end % 8) ? M_step_end + 8 - (M_step_end % 8) : M_step_end;

      nntr_gemv_q4_0_8x8_q8_0(K, (float *)((C) + M_step_start), N,
                              (void *)((char *)B + M_step_start * B_step),
                              QA.data(), M, M_step_end - M_step_start);
    });
  } else { // GEMM
    unsigned int blocks_per_4_rows = (K + QK8_0 - 1) / QK8_0;
    unsigned int qa_4_rows_size = sizeof(block_q8_0x4) * blocks_per_4_rows;
    const size_t qa_row_size = (sizeof(block_q8_0) * K) / QK8_0;
    unsigned int M4 = ((M - M % 4) / 4);
    int B_step = sizeof(block_q4_0) * (K / QK4_0);

    unsigned int qa_size = qa_4_rows_size * (((M >> 2) << 2) / 4 + 1);
    std::vector<char> QA = std::vector<char>(qa_size);

    // Quantize 4-divisible-M row portion with matrix-wise function
    for (unsigned int i = 0; i < M4; i++) {
      nntr_quantize_mat_q8_0_4x8(A + 4 * i * K, QA.data() + i * qa_4_rows_size,
                                 K);
    }
    // Quantize leftover 1 ~ 3 rows with row-wise function
    for (unsigned int i = M4 * 4; i < M; i++) {
      nntr_quantize_row_q8_0(
        (float *)A + i * K,
        (QA.data() + (M4 * qa_4_rows_size) + (i - M4 * 4) * qa_row_size), K);
    }

    // Compute 4-divisible-M row portion with multithreaded GEMM
    tm.parallel_for_chunked(thread_num, [=](size_t i) {
      unsigned int src0_start = (i * N) / thread_num;
      unsigned int src0_end = ((i + 1) * N) / thread_num;

      src0_start =
        (src0_start % 8) ? src0_start + 8 - (src0_start % 8) : src0_start;
      src0_end = (src0_end % 8) ? src0_end + 8 - (src0_end % 8) : src0_end;

      nntr_gemm_q4_0_8x8_q8_0(K, (float *)(C + src0_start), ldc,
                              (void *)((char *)B + src0_start * B_step),
                              QA.data(), M4 * 4, src0_end - src0_start);
    });

    // Compute leftover 1 ~ 3 rows with multithreaded GEMV
    for (unsigned int pb = M4 * 4; pb < M; pb++) {
      tm.parallel_for_chunked(thread_num, [=](size_t thread_idx) {
        unsigned int M_step_start = (thread_idx * N) / thread_num;
        unsigned int M_step_end = ((thread_idx + 1) * N) / thread_num;

        M_step_start = (M_step_start % 8)
                         ? M_step_start + 8 - (M_step_start % 8)
                         : M_step_start;
        M_step_end =
          (M_step_end % 8) ? M_step_end + 8 - (M_step_end % 8) : M_step_end;

        nntr_gemv_q4_0_8x8_q8_0(
          K, (float *)((C + ((pb - M4 * 4) * N) + (M4 * 4 * N)) + M_step_start),
          N, (void *)((char *)B + M_step_start * B_step),
          QA.data() + (M4 * qa_4_rows_size) + (pb - M4 * 4) * qa_row_size, 1,
          M_step_end - M_step_start);
      });
    }
  }
}

template <>
void __ggml_q4_0_8x8_q8_0_GEMM(const unsigned int M,
                               std::vector<unsigned int> Ns,
                               const unsigned int K, const float *A,
                               const unsigned int lda, std::vector<void *> Bs,
                               std::vector<unsigned int> ldbs,
                               std::vector<float *> C,
                               std::vector<unsigned int> ldcs) {
  throw std::runtime_error("nntrainer::__ggml_q4_0_8x8_q8_0_GEMM for "
                           "multi-weights is not implemented yet");
}

void __ggml_q4_K_8x8_q8_K_GEMM(const unsigned int M, const unsigned int N,
                               const unsigned int K, const float *A,
                               const unsigned int lda, const void *B,
                               const unsigned int ldb, float *C,
                               const unsigned int ldc) {
  auto &tm = ThreadManager::Global();
  unsigned int thread_num = tm.getComputeThreadCount();

  if (M == 1) { // GEMV
    unsigned int blocks_per_row = (K + QK_K - 1) / QK_K;
    unsigned int qa_size = sizeof(block_q8_K) * blocks_per_row;
    unsigned int B_step = sizeof(block_q4_K) * (K / QK_K);

    std::vector<char> QA = std::vector<char>(qa_size);

    nntr_quantize_row_q8_K(A, QA.data(), K);

    tm.parallel_for_chunked(thread_num, [=](size_t thread_idx) {
      unsigned int M_step_start = (thread_idx * N) / thread_num;
      unsigned int M_step_end = ((thread_idx + 1) * N) / thread_num;

      M_step_start = (M_step_start % 8) ? M_step_start + 8 - (M_step_start % 8)
                                        : M_step_start;
      M_step_end =
        (M_step_end % 8) ? M_step_end + 8 - (M_step_end % 8) : M_step_end;

      nntr_gemv_q4_K_8x8_q8_K(K, (float *)((C) + M_step_start), N,
                              (void *)((char *)B + M_step_start * B_step),
                              QA.data(), M, M_step_end - M_step_start);
    });
  } else {
    unsigned int blocks_per_4_rows = (K + QK_K - 1) / QK_K;
    unsigned int qa_4_rows_size = sizeof(block_q8_Kx4) * blocks_per_4_rows;
    const size_t qa_row_size = (sizeof(block_q8_K) * K) / QK_K;
    unsigned int M4 = ((M - M % 4) / 4);
    int B_step = sizeof(block_q4_K) * (K / QK_K);

    unsigned int qa_size = qa_4_rows_size * (((M >> 2) << 2) / 4 + 1);
    std::vector<char> QA = std::vector<char>(qa_size);

    for (unsigned int i = 0; i < M4; i++) {
      nntr_quantize_mat_q8_K_4x8(A + 4 * i * K, QA.data() + i * qa_4_rows_size,
                                 K);
    }
    for (unsigned int i = M4 * 4; i < M; i++) {
      nntr_quantize_row_q8_K(
        (float *)A + i * K,
        (QA.data() + (M4 * qa_4_rows_size) + (i - M4 * 4) * qa_row_size), K);
    }

    // Compute 4-divisible-M row portion with multithreaded GEMM
    tm.parallel_for_chunked(thread_num, [=](size_t i) {
      unsigned int src0_start = (i * N) / thread_num;
      unsigned int src0_end = ((i + 1) * N) / thread_num;

      src0_start =
        (src0_start % 8) ? src0_start + 8 - (src0_start % 8) : src0_start;
      src0_end = (src0_end % 8) ? src0_end + 8 - (src0_end % 8) : src0_end;

      nntr_gemm_q4_K_8x8_q8_K(K, (float *)(C + src0_start), ldc,
                              (void *)((char *)B + src0_start * B_step),
                              QA.data(), M4 * 4, src0_end - src0_start);
    });

    // Compute leftover 1 ~ 3 rows with multithreaded GEMV
    for (unsigned int pb = M4 * 4; pb < M; pb++) {
      tm.parallel_for_chunked(thread_num, [=](size_t thread_idx) {
        unsigned int M_step_start = (thread_idx * N) / thread_num;
        unsigned int M_step_end = ((thread_idx + 1) * N) / thread_num;

        M_step_start = (M_step_start % 8)
                         ? M_step_start + 8 - (M_step_start % 8)
                         : M_step_start;
        M_step_end =
          (M_step_end % 8) ? M_step_end + 8 - (M_step_end % 8) : M_step_end;

        nntr_gemv_q4_K_8x8_q8_K(
          K, (float *)((C + ((pb - M4 * 4) * N) + (M4 * 4 * N)) + M_step_start),
          N, (void *)((char *)B + M_step_start * B_step),
          QA.data() + (M4 * qa_4_rows_size) + (pb - M4 * 4) * qa_row_size, 1,
          M_step_end - M_step_start);
      });
    }
  }
}

void __ggml_q4_K_8x8_q8_K_GEMM(const unsigned int M,
                               std::vector<unsigned int> Ns,
                               const unsigned int K, const float *A,
                               const unsigned int lda, std::vector<void *> Bs,
                               std::vector<unsigned int> ldbs,
                               std::vector<float *> C,
                               std::vector<unsigned int> ldcs) {
  throw std::runtime_error("nntrainer::__ggml_q4_K_8x8_q8_K_GEMM for "
                           "multi-weights is not implemented yet");
}

template <>
void __ggml_gemm_q6_K(const unsigned int M, const unsigned int N,
                      const unsigned int K, const float *A,
                      const unsigned int lda, const void *B,
                      const unsigned int ldb, float *C,
                      const unsigned int ldc) {
  auto &tm = ThreadManager::Global();

  static constexpr const int32_t bs = 1;  // unused in ggml_vec_dot_q6_K_q8_K
  static constexpr const int32_t bx = 1;  // unused in ggml_vec_dot_q6_K_q8_K
  static constexpr const int32_t by = 1;  // unused in ggml_vec_dot_q6_K_q8_K
  static constexpr const int32_t nrc = 1; // unused in ggml_vec_dot_q6_K_q8_K

  const int32_t blocks_per_row = (K + QK_K - 1) / QK_K;
  const int32_t A_row_size = sizeof(block_q8_K) * blocks_per_row;
  const int32_t B_row_size = sizeof(block_q6_K) * blocks_per_row;

  // GEMV
  if (M == 1) {
    std::vector<char> quantized_A(A_row_size);
    nntr_quantize_row_q8_K(A, quantized_A.data(), K);

    const void *const quantized_A_data = quantized_A.data();

    tm.parallel_for(0, static_cast<size_t>(N), [&](size_t thread_job) {
      const int32_t B_row_data_offset = B_row_size * thread_job;

      const void *const B_data = (void *)((char *)B + B_row_data_offset);

      nntr_vec_dot_q6_K_q8_K(K, &C[thread_job], bs, B_data, bx,
                             quantized_A_data, by, nrc);
    });
  } else { // GEMM
    const int32_t A_total_size = A_row_size * M;
    std::vector<char> quantized_A(A_total_size);

    tm.parallel_for(0, static_cast<size_t>(M), [&](size_t thread_job) {
      const int32_t A_row_data_offset = A_row_size * thread_job;
      void *A_data = (void *)((char *)quantized_A.data() + A_row_data_offset);
      nntr_quantize_row_q8_K(A + thread_job * K, A_data, K);
    });

    tm.parallel_for(0, static_cast<size_t>(M), [&](size_t thread_job) {
      const int32_t A_row_data_offset = A_row_size * thread_job;
      void *A_data = (void *)((char *)quantized_A.data() + A_row_data_offset);

      for (uint32_t j = 0; j < N; j++) {
        const int32_t B_row_data_offset = B_row_size * j;
        const void *const B_data = (void *)((char *)B + B_row_data_offset);

        nntr_vec_dot_q6_K_q8_K(K, &C[thread_job * ldc + j], bs, B_data, bx,
                               A_data, by, nrc);
      }
    });
  }
}

} // namespace nntrainer
