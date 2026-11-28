// SPDX-License-Identifier: Apache-2.0
/**
 * @file	tq2cx_kernels.cpp
 * @date	02 October 2026
 * @brief	Arch-neutral pieces of the TQ2CX (ternary) batch GEMM.
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs
 */

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include <thread_manager.h>
#include <tq2cx_kernels.h>

namespace nntrainer {

namespace {

/**
 * @brief Split (weight, row chunk, column chunk) tasks of a batched GEMM and
 *        run them: fixed 16-row chunks, column chunks in multiples of 8 sized
 *        for about 64 tasks per thread, every weight folded into one task
 *        space so the barrier is paid once. Each output element is written by
 *        exactly one task, so the result does not depend on the thread count.
 */
template <typename TaskFn>
void tq2cx_for_each_tile(unsigned int M, const std::vector<unsigned int> &Ns,
                         bool threaded, const TaskFn &task_fn) {
  const size_t num_w = Ns.size();
  if (M == 0 || num_w == 0)
    return;

  auto &tm = ThreadManager::Global();
  const size_t threads = threaded ? tm.getComputeThreadCount() : 1;

  // Same task shape as the q4_0 batch kernel: fixed row chunks, a column
  // chunk (multiple of 8 output channels) sized for ~64 tasks per thread, and
  // every weight folded into one task space so the barrier is paid once.
  const unsigned int row_chunk = 16;
  const size_t row_loop = (M + row_chunk - 1) / row_chunk;
  std::vector<size_t> col_chunks(num_w);
  std::vector<size_t> col_loops(num_w);
  std::vector<size_t> offsets(num_w + 1, 0);
  for (size_t w = 0; w < num_w; ++w) {
    col_chunks[w] = std::clamp<size_t>(
      (size_t)Ns[w] * row_loop / ((size_t)64 * threads) / 8 * 8, 16, 64);
    col_loops[w] = (Ns[w] + col_chunks[w] - 1) / col_chunks[w];
    offsets[w + 1] = offsets[w] + col_loops[w] * row_loop;
  }

  auto run_task = [&](size_t task) {
    size_t w = 0;
    while (w + 1 < num_w && task >= offsets[w + 1])
      ++w;
    const size_t local = task - offsets[w];
    const unsigned int r = static_cast<unsigned int>(local / col_loops[w]);
    const unsigned int c = static_cast<unsigned int>(local % col_loops[w]);
    const unsigned int m0 = r * row_chunk;
    const unsigned int m1 = std::min(M, m0 + row_chunk);
    const unsigned int n0 = static_cast<unsigned int>(c * col_chunks[w]);
    const unsigned int n1 = static_cast<unsigned int>(
      std::min<size_t>(Ns[w], (c + 1) * col_chunks[w]));
    task_fn(w, m0, m1, n0, n1);
  };

  if (threaded) {
    tm.parallel_for(0, offsets.back(), run_task);
  } else {
    for (size_t task = 0; task < offsets.back(); ++task)
      run_task(task);
  }
}

} // namespace

void tq2cx_tile_ref(const float *A, unsigned int lda, unsigned int m0,
                    unsigned int m1, const void *W, unsigned int N,
                    unsigned int K, unsigned int n0, unsigned int n1, float *C,
                    unsigned int ldc, const Tq2cxLayout &lay) {
  const size_t units = (K + 31) / 32;
  const uint32_t *masks = static_cast<const uint32_t *>(W);
  const uint8_t *scale_bytes =
    static_cast<const uint8_t *>(W) + tq2cxMaskBytes(lay, N, K);

  for (unsigned int n = n0; n < n1; ++n) {
    float s;
    std::memcpy(&s, scale_bytes + static_cast<size_t>(n) * sizeof(float),
                sizeof(float));
    for (unsigned int m = m0; m < m1; ++m) {
      const float *x = A + static_cast<size_t>(m) * lda;
      float acc = 0.0f;
      for (size_t u = 0; u < units; ++u) {
        const uint32_t *w = masks + 2 * tq2cxUnitIndex(lay, n, u, units);
        const uint32_t p = w[0];
        const uint32_t g = w[1];
        if ((p | g) == 0u)
          continue;
        const size_t base = u * 32;
        const size_t lim = std::min<size_t>(32, K - base);
        for (size_t b = 0; b < lim; ++b) {
          if ((p >> b) & 1u)
            acc += x[base + b];
          else if ((g >> b) & 1u)
            acc -= x[base + b];
        }
      }
      C[static_cast<size_t>(m) * ldc + n] = acc * s;
    }
  }
}

void tq2cx_gemm_batch(unsigned int M, const std::vector<unsigned int> &Ns,
                      unsigned int K, const float *A, unsigned int lda,
                      const std::vector<void *> &Ws,
                      const std::vector<float *> &Cs,
                      const std::vector<unsigned int> &ldcs, tq2cx_tile_fn tile,
                      bool threaded, Tq2cxLayout lay) {
  const size_t num_w = Ns.size();
  if (Ws.size() != num_w || Cs.size() != num_w || ldcs.size() != num_w)
    throw std::invalid_argument("tq2cx_gemm_batch: mismatched weight lists");

  tq2cx_for_each_tile(M, Ns, threaded,
                      [&](size_t w, unsigned int m0, unsigned int m1,
                          unsigned int n0, unsigned int n1) {
                        tile(A, lda, m0, m1, Ws[w], Ns[w], K, n0, n1, Cs[w],
                             ldcs[w], lay);
                      });
}

} // namespace nntrainer
