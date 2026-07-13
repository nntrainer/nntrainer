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

#include <conv_indirect.h>
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

    unsigned int qa_size =
      qa_4_rows_size * M4 + static_cast<unsigned int>(qa_row_size) * (M % 4);
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
    size_t row_loop = (M4 * 4 + row_chunk_size - 1) / row_chunk_size;
    unsigned int A_step = sizeof(block_q8_0) * (K / QK8_0);

    unsigned int col_chunk_size = 16;
    size_t col_loop = (N + col_chunk_size - 1) / col_chunk_size;
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
    size_t row_loop = (M + row_chunk_size - 1) / row_chunk_size;
    unsigned int A_step = sizeof(block_q8_0) * (K / QK8_0);

    unsigned int col_chunk_size = 16;
    size_t col_loop = (N + col_chunk_size - 1) / col_chunk_size;
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

void __ggml_q4_0_4x8_q8_0_indirect_GEMM(const unsigned int M,
                                        const unsigned int N,
                                        const unsigned int K, const float *in,
                                        const ConvGatherParams &geom,
                                        const void *B, const unsigned int ldb,
                                        float *C, const unsigned int ldc) {
  /// Indirect (im2col-fused) variant of __ggml_q4_0_4x8_q8_0_GEMM: the
  /// activation matrix A = [M=OH*OW, K=CRS] is never materialized. Each Q8_0
  /// activation tile is gathered directly from the NCHW input `in` via `geom`
  /// (gather_conv_act_rows_fp32, byte-identical to im2col rows) and quantized
  /// on the fly, then fed to the SAME unchanged micro-kernels
  /// (nntr_gemm/gemv_q4_0_4x8_q8_0). The produced QA layout is byte-identical
  /// to the materialized path, so results are bit-identical. The fused gather +
  /// quantize is parallelized over row chunks (the materialized path quantized
  /// serially).
  auto &tm = ThreadManager::Global();

  const unsigned int blocks_per_4_rows = (K + QK8_0 - 1) / QK8_0;
  const unsigned int qa_4_rows_size = sizeof(block_q8_0x4) * blocks_per_4_rows;
  const size_t qa_row_size =
    (sizeof(block_q8_0) * K) / QK8_0; // ignore remainder
  const unsigned int M4 = M / 4;
  const unsigned int rem = M % 4;

  const unsigned int qa_size =
    qa_4_rows_size * M4 + static_cast<unsigned int>(qa_row_size) * rem;
  std::vector<char> QA(qa_size);
  char *QA_ptr = QA.data();

  /// Fused gather + Q8_0 quantize of the 4-row-divisible portion, parallel over
  /// chunks of QCHUNK rows. Within each chunk the gather feeds the quantizer
  /// one 4-row tile at a time through a small L1/L2-resident buffer (instead of
  /// gathering the whole chunk into a large FP32 staging buffer that spills
  /// cache before the quantizer reads it back). Same rows, same QA addressing
  /// -> bit-identical; each chunk writes a disjoint QA span (race-free).
  const unsigned int QCHUNK = 64; // multiple of 4
  if (M4 > 0) {
    const unsigned int rows4 = M4 * 4;
    const size_t qloops = (rows4 + QCHUNK - 1) / QCHUNK;
    tm.parallel_for(0, qloops, [=](size_t q) {
      const unsigned int r0 = static_cast<unsigned int>(q) * QCHUNK;
      const unsigned int r1 = std::min(r0 + QCHUNK, rows4);
      std::vector<float> tile((size_t)4 * K); // one quantize tile, reused
      for (unsigned int r = r0; r < r1; r += 4) {
        gather_conv_act_rows_fp32(tile.data(), in, geom, (int)r, 4);
        nntr_quantize_mat_q8_0_4x8(tile.data(),
                                   QA_ptr + (r / 4) * qa_4_rows_size, K);
      }
    });
  }
  /// Remainder rows (M % 4): single-row gather + quantize.
  for (unsigned int i = M4 * 4; i < M; ++i) {
    std::vector<float> staging((size_t)K);
    gather_conv_act_rows_fp32(staging.data(), in, geom, (int)i, 1);
    nntr_quantize_row_q8_0(
      staging.data(),
      QA_ptr + (M4 * qa_4_rows_size) + (i - M4 * 4) * qa_row_size, K);
  }

  /// GEMM over the 4-row-divisible rows + GEMV over the remainder. This block
  /// is a verbatim copy of __ggml_q4_0_4x8_q8_0_GEMM's (M % 4 != 0) path (same
  /// QA addressing, same micro-kernels) — only the activation source changed
  /// above.
  const unsigned int A_step = sizeof(block_q8_0) * (K / QK8_0);
  const unsigned int B_step = sizeof(block_q4_0) * (K / QK4_0);

  if (M4 > 0) {
    const unsigned int row_chunk_size = 16;
    const size_t row_loop = (M4 * 4 + row_chunk_size - 1) / row_chunk_size;
    const unsigned int col_chunk_size = 16;
    const size_t col_loop = (N + col_chunk_size - 1) / col_chunk_size;

    tm.parallel_for(0, col_loop * row_loop, [=](size_t i) {
      unsigned int r = i / col_loop;
      unsigned int c = i % col_loop;

      unsigned int r_start = r * row_chunk_size;
      unsigned int r_end = std::min(row_chunk_size * (r + 1), M4 * 4);

      unsigned int c_start = c * col_chunk_size;
      unsigned int c_end = std::min(col_chunk_size * (c + 1), N);

      nntr_gemm_q4_0_4x8_q8_0(K, (float *)(C + r_start * N + c_start), ldc,
                              (void *)((char *)B + c_start * B_step),
                              (void *)(QA_ptr + r_start * A_step),
                              r_end - r_start, c_end - c_start);
    });
  }

  for (unsigned int pb = M4 * 4; pb < M; pb++) {
    unsigned int chunk_size = 16;
    unsigned int loop = (N + chunk_size - 1) / chunk_size;

    tm.parallel_for(0, loop, [=](size_t idx) {
      unsigned int M_step_start = chunk_size * idx;
      unsigned int M_step_end = std::min(chunk_size * (idx + 1), (size_t)N);

      nntr_gemv_q4_0_4x8_q8_0(
        K, (float *)((C + ((pb - M4 * 4) * N) + (M4 * 4 * N)) + M_step_start),
        N, (void *)((char *)B + M_step_start * B_step),
        QA_ptr + (M4 * qa_4_rows_size) + (pb - M4 * 4) * qa_row_size, 1,
        M_step_end - M_step_start);
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

    tm.parallel_for(0, thread_num, [&](size_t i) {
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
      tm.parallel_for(0, thread_num, [&](size_t thread_idx) {
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

    tm.parallel_for(0, thread_num, [=](size_t thread_idx) {
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
    tm.parallel_for(0, thread_num, [=](size_t i) {
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
      tm.parallel_for(0, thread_num, [=](size_t thread_idx) {
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

    tm.parallel_for(0, thread_num, [=](size_t thread_idx) {
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
    tm.parallel_for(0, thread_num, [=](size_t i) {
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
      tm.parallel_for(0, thread_num, [=](size_t thread_idx) {
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

/**
 * @brief Q8_0 weights x FP32 activation GEMM/GEMV (plain block_q8_0 rows).
 *
 * Mirrors the Q4_0 GEMM threading in this file: the FP32 activation is
 * online-quantised to block_q8_0, then the output is tiled across
 * ThreadManager workers. The plain-block kernel computes an independent
 * [rows x cols] tile per call (weights are [N x K/QK8_0] blocks, one row per
 * output channel), so A/B/C slice with no synchronisation. A Q8_0x8
 * interleaved weight layout to match the Q4_0 8x8 micro-tile is a follow-up.
 */
void __ggml_q8_0_q8_0_GEMM(const unsigned int M, const unsigned int N,
                           const unsigned int K, const float *A,
                           const unsigned int lda, const void *B,
                           const unsigned int ldb, float *C,
                           const unsigned int ldc) {
  (void)lda;
  (void)ldb;

  auto &tm = ThreadManager::Global();

  const unsigned int blocks_per_row = (K + QK8_0 - 1) / QK8_0;
  const size_t row_bytes = sizeof(block_q8_0) * blocks_per_row;

  if (M == 1) { // GEMV
    std::vector<char> QA = std::vector<char>(row_bytes);
    nntr_quantize_row_q8_0(A, QA.data(), K);

    unsigned int chunk_size = 16;
    size_t loop = (N + chunk_size - 1) / chunk_size;

    // compute multithreaded GEMV over output-column chunks
    tm.parallel_for(0, loop, [=, &QA](size_t idx) {
      unsigned int c_start = chunk_size * idx;
      unsigned int c_end = std::min(chunk_size * (idx + 1), (size_t)N);

      nntr_gemm_q8_0_q8_0(K, C + c_start, ldc,
                          (void *)((char *)B + (size_t)c_start * row_bytes),
                          QA.data(), M, c_end - c_start);
    });
  } else { // GEMM
    // online quantization for all M activation rows (plain block layout)
    std::vector<char> QA = std::vector<char>((size_t)M * row_bytes);
    for (unsigned int i = 0; i < M; i++) {
      nntr_quantize_row_q8_0(A + (size_t)i * K, QA.data() + (size_t)i * row_bytes,
                             K);
    }

    // Compute with multithreaded GEMM over 2D row x column tiles
    unsigned int row_chunk_size = 16;
    size_t row_loop = (M + row_chunk_size - 1) / row_chunk_size;

    unsigned int col_chunk_size = 16;
    size_t col_loop = (N + col_chunk_size - 1) / col_chunk_size;

    tm.parallel_for(0, col_loop * row_loop, [=, &QA](size_t i) {
      unsigned int r = i / col_loop;
      unsigned int c = i % col_loop;

      unsigned int r_start = r * row_chunk_size;
      unsigned int r_end = std::min(row_chunk_size * (r + 1), M);

      unsigned int c_start = c * col_chunk_size;
      unsigned int c_end = std::min(col_chunk_size * (c + 1), N);

      nntr_gemm_q8_0_q8_0(K, C + (size_t)r_start * ldc + c_start, ldc,
                          (void *)((char *)B + (size_t)c_start * row_bytes),
                          QA.data() + (size_t)r_start * row_bytes,
                          r_end - r_start, c_end - c_start);
    });
  }
}

/**
 * @brief Q8_0x4-interleaved weights x FP32 activation GEMM/GEMV, FP32 output.
 *
 * Weights are the q8_0x4 interleaved layout produced at quantisation time by
 * __ggml_repack_q8_0_to_q8_0_4 (ISA::ARM target), the FC analogue of the
 * YOLO conv path's offline repack: single contiguous vld1q_s8 loads replace
 * the plain layout's scattered per-row loads. The activation is
 * online-quantised: packed q8_0x4 (nntr_quantize_mat_q8_0_4x8) for the
 * 4-row-aligned bulk, plain rows for M == 1 / the M %% 4 tail (GEMV kernel).
 * All column partitions are kept multiples of 4 so every kernel call sees
 * whole weight super-blocks (N %% 32 == 0 is guaranteed by the quantiser).
 */
void __ggml_q8_0_4x4_q8_0_GEMM(const unsigned int M, const unsigned int N,
                               const unsigned int K, const float *A,
                               const unsigned int lda, const void *B,
                               const unsigned int ldb, float *C,
                               const unsigned int ldc) {
  (void)lda;
  (void)ldb;
  assert(N % 4 == 0);

  auto &tm = ThreadManager::Global();

  const unsigned int nb = (K + QK8_0 - 1) / QK8_0;
  const size_t row_bytes = sizeof(block_q8_0) * nb;   // plain act row
  const size_t sb_bytes = sizeof(block_q8_0x4) * nb;  // 4-row/4-col super-row

  if (M == 1) { // GEMV
    std::vector<char> QA = std::vector<char>(row_bytes);
    nntr_quantize_row_q8_0(A, QA.data(), K);

    unsigned int chunk_size = 16; // multiple of 4: whole weight super-blocks
    size_t loop = (N + chunk_size - 1) / chunk_size;

    tm.parallel_for(0, loop, [=, &QA](size_t idx) {
      unsigned int c_start = chunk_size * idx;
      unsigned int c_end = std::min(chunk_size * (idx + 1), (size_t)N);

      nntr_gemv_q8_0x4_q8_0(
        K, C + c_start, ldc,
        (void *)((char *)B + (size_t)(c_start / 4) * sb_bytes), QA.data(),
        c_end - c_start);
    });
    return;
  }

  // GEMM: pack the 4-row-aligned bulk of the activation. Each super-row is an
  // independent [4 x K] quantize into its own QA region, so packing threads
  // over super-rows (the encoder packs 49 of them per FC call).
  const unsigned int M4 = M / 4;
  std::vector<char> QA = std::vector<char>((size_t)M4 * sb_bytes);
  tm.parallel_for(0, (size_t)M4, [=, &QA](size_t i) {
    nntr_quantize_mat_q8_0_4x8(A + 4 * i * K, QA.data() + i * sb_bytes, K);
  });

  // Multithreaded 2D row x column tiles (both chunk sizes multiples of 4).
  unsigned int row_chunk_size = 16;
  size_t row_loop = ((size_t)M4 * 4 + row_chunk_size - 1) / row_chunk_size;

  unsigned int col_chunk_size = 16;
  size_t col_loop = (N + col_chunk_size - 1) / col_chunk_size;

  tm.parallel_for(0, col_loop * row_loop, [=, &QA](size_t i) {
    unsigned int r = i / col_loop;
    unsigned int c = i % col_loop;

    unsigned int r_start = r * row_chunk_size;
    unsigned int r_end = std::min(row_chunk_size * (r + 1), M4 * 4);

    unsigned int c_start = c * col_chunk_size;
    unsigned int c_end = std::min(col_chunk_size * (c + 1), N);

    nntr_gemm_q8_0x4_q8_0x4(
      K, C + (size_t)r_start * ldc + c_start, ldc,
      (void *)((char *)B + (size_t)(c_start / 4) * sb_bytes),
      QA.data() + (size_t)(r_start / 4) * sb_bytes, r_end - r_start,
      c_end - c_start);
  });

  // M % 4 tail rows: plain-quantised row x interleaved weights GEMV.
  for (unsigned int m = M4 * 4; m < M; ++m) {
    std::vector<char> QR = std::vector<char>(row_bytes);
    nntr_quantize_row_q8_0(A + (size_t)m * K, QR.data(), K);

    unsigned int chunk_size = 16;
    size_t loop = (N + chunk_size - 1) / chunk_size;
    tm.parallel_for(0, loop, [=, &QR](size_t idx) {
      unsigned int c_start = chunk_size * idx;
      unsigned int c_end = std::min(chunk_size * (idx + 1), (size_t)N);
      nntr_gemv_q8_0x4_q8_0(
        K, C + (size_t)m * ldc + c_start, ldc,
        (void *)((char *)B + (size_t)(c_start / 4) * sb_bytes), QR.data(),
        c_end - c_start);
    });
  }
}

} // namespace nntrainer
