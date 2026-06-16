// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Yonghyeon Cho <dyddyd8574@gmail.com>
 *
 * @file   hgemm_blocked.cpp
 * @date   01 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho <dyddyd8574@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  C32 panel orchestration for x86 FP16 GEMM
 */

#include "hgemm_blocked.h"

#include "hgemm_common.h"
#include "hgemm_kernel/hgemm_kernel.h"
#include "hgemm_pack.h"
#include "hgemm_util.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread_manager.h>

namespace nntrainer::hgemm::internal {

namespace {

bool should_parallelize_blocked(unsigned int M, unsigned int N, unsigned int K,
                                unsigned int panel_count) {
  if (panel_count <= 1) {
    return false;
  }

  constexpr std::size_t min_parallel_work = 8u * 1024u * 1024u;
  const std::size_t work = static_cast<std::size_t>(M) * N * K;
  return work >= min_parallel_work;
}

template <typename AType, typename BType>
void hgemm_blocked_kernel(bool TransA, bool TransB, unsigned int M,
                          unsigned int N, unsigned int K, float alpha,
                          const AType *A, unsigned int a_stride, const BType *B,
                          unsigned int b_stride, float *C32,
                          unsigned int c32_stride, float *sa, float *sb,
                          const HgemmBlockSizes &block) {
  const unsigned int m_tile_block =
    std::max(1u, std::min<unsigned int>(block.m, block.c_m));
  const unsigned int raw_n_tile_block =
    std::max(1u, std::min<unsigned int>(block.n, block.c_n));
  const unsigned int n_tile_block = std::max<unsigned int>(
    X86_HGEMM_NR, (raw_n_tile_block / X86_HGEMM_NR) * X86_HGEMM_NR);
  const unsigned int k_block = std::max(1u, block.k);

  for (unsigned int ks = 0; ks < K; ks += k_block) {
    const unsigned int k_min = std::min<unsigned int>(K - ks, k_block);

    // Pack B once for this K block and the full outer N panel. The packed
    // buffer is indexed by absolute NR-column blocks inside the panel, so all
    // M tiles below can reuse it.
    for (unsigned int nn = 0; nn < N; nn += X86_HGEMM_NR) {
      const unsigned int n_act = std::min<unsigned int>(N - nn, X86_HGEMM_NR);
      float *pb = sb + (nn / X86_HGEMM_NR) * (k_min * X86_HGEMM_NR);
      if (TransB) {
        packing_B_N16_trans(k_min, n_act,
                            B + static_cast<std::size_t>(nn) * b_stride + ks,
                            b_stride, pb);
      } else {
        packing_B_N16(k_min, n_act,
                      B + static_cast<std::size_t>(ks) * b_stride + nn,
                      b_stride, pb);
      }
    }

    for (unsigned int mcs = 0; mcs < M; mcs += m_tile_block) {
      const unsigned int m_tile = std::min<unsigned int>(M - mcs, m_tile_block);

      // Pack A once for this M tile and K block, then reuse it across the
      // current outer N panel.
      for (unsigned int mm = 0; mm < m_tile; mm += X86_HGEMM_MR) {
        const unsigned int m_act =
          std::min<unsigned int>(m_tile - mm, X86_HGEMM_MR);
        float *pa = sa + (mm / X86_HGEMM_MR) * (k_min * X86_HGEMM_MR);
        if (TransA) {
          packing_A_M6_trans(m_act, k_min,
                             A + static_cast<std::size_t>(ks) * a_stride +
                               (mcs + mm),
                             a_stride, alpha, pa);
        } else {
          packing_A_M6(m_act, k_min,
                       A + static_cast<std::size_t>(mcs + mm) * a_stride + ks,
                       a_stride, alpha, pa);
        }
      }

      for (unsigned int ncs = 0; ncs < N; ncs += n_tile_block) {
        const unsigned int n_end =
          std::min<unsigned int>(N, ncs + n_tile_block);

        for (unsigned int mm = 0; mm < m_tile; mm += X86_HGEMM_MR) {
          const unsigned int m_act =
            std::min<unsigned int>(m_tile - mm, X86_HGEMM_MR);
          const float *pa = sa + (mm / X86_HGEMM_MR) * (k_min * X86_HGEMM_MR);
          for (unsigned int nn = ncs; nn < n_end; nn += X86_HGEMM_NR) {
            const unsigned int n_act =
              std::min<unsigned int>(n_end - nn, X86_HGEMM_NR);
            const float *pb = sb + (nn / X86_HGEMM_NR) * (k_min * X86_HGEMM_NR);
            float *c_tile =
              C32 + static_cast<std::size_t>(mcs + mm) * c32_stride + nn;
            hgemm_kernel_mxn(m_act, n_act, k_min, pa, pb, c_tile, c32_stride);
          }
        }
      }
    }
  }
}

template <typename AType, typename BType, typename CType>
void run_hgemm_panel(bool TransA, bool TransB, unsigned int M, unsigned int N,
                     unsigned int K, float alpha, const AType *A,
                     unsigned int a_stride, const BType *B,
                     unsigned int b_stride, float beta, CType *C,
                     unsigned int c_stride, HgemmWorkspace &workspace,
                     const HgemmBlockSizes &block) {
  const unsigned int k_block_size = std::max(1u, block.k);
  const unsigned int m_tile_block =
    std::max(1u, std::min<unsigned int>(block.m, block.c_m));
  const unsigned int max_k_block = std::min<unsigned int>(K, k_block_size);
  const unsigned int max_m_tile = std::min<unsigned int>(M, m_tile_block);

  const unsigned int c32_m_capacity = round_up(M, X86_HGEMM_MR);
  const unsigned int c32_n_capacity = round_up(N, X86_HGEMM_NR);
  float *C32 = workspace.ensure_c32(static_cast<std::size_t>(c32_m_capacity) *
                                    c32_n_capacity);

  const std::size_t sa_capacity =
    static_cast<std::size_t>(round_up(max_m_tile, X86_HGEMM_MR)) * max_k_block;
  const std::size_t sb_capacity =
    static_cast<std::size_t>(max_k_block) * c32_n_capacity;

  float *sa = workspace.ensure_pack_a(sa_capacity);
  float *sb = workspace.ensure_pack_b(sb_capacity);

  std::memset(C32, 0,
              static_cast<std::size_t>(c32_m_capacity) * c32_n_capacity *
                sizeof(float));
  if (std::fpclassify(beta) != FP_ZERO) {
    copy_C_to_C32<CType>(C, C32, M, N, c_stride, c32_n_capacity, beta);
  }

  hgemm_blocked_kernel<AType, BType>(TransA, TransB, M, N, K, alpha, A,
                                     a_stride, B, b_stride, C32, c32_n_capacity,
                                     sa, sb, block);

  copy_C32_to_C<CType>(C32, C, M, N, c32_n_capacity, c_stride);
}

} // namespace

template <typename AType, typename BType, typename CType>
void run_hgemm_blocked(bool TransA, bool TransB, unsigned int M, unsigned int N,
                       unsigned int K, float alpha, const AType *A,
                       unsigned int a_stride, const BType *B,
                       unsigned int b_stride, float beta, CType *C,
                       unsigned int c_stride, HgemmWorkspace &workspace) {
  const HgemmBlockSizes &block = get_hgemm_block_sizes();

  // Outer-panel sizes used purely to slice work for the thread pool. They start
  // at the cache-blocking sizes, but when a large GEMM produces fewer panels
  // than there are worker threads (e.g. M <= block.m with a modest N yields
  // only 1-2 panels), the panels are subdivided — N width first (NR-aligned),
  // then M height (MR-aligned) — until there is at least one panel per thread.
  // The inner cache tiling still uses `block`, so this only changes parallel
  // granularity, not the per-tile working set.
  unsigned int pm = std::max(1u, block.m);
  unsigned int pn = std::max(1u, block.n);
  {
    constexpr std::size_t min_parallel_work = 8u * 1024u * 1024u;
    const std::size_t work = static_cast<std::size_t>(M) * N * K;
    const unsigned int tc = ThreadManager::Global().getComputeThreadCount();
    if (tc > 1 && work >= min_parallel_work) {
      auto panels = [&](unsigned int a, unsigned int b) {
        return ((M + a - 1) / a) * ((N + b - 1) / b);
      };
      while (panels(pm, pn) < tc && pn > X86_HGEMM_NR) {
        pn = std::max<unsigned int>(X86_HGEMM_NR,
                                    (pn / 2 / X86_HGEMM_NR) * X86_HGEMM_NR);
      }
      while (panels(pm, pn) < tc && pm > X86_HGEMM_MR) {
        pm = std::max<unsigned int>(X86_HGEMM_MR,
                                    (pm / 2 / X86_HGEMM_MR) * X86_HGEMM_MR);
      }
    }
  }

  const unsigned int m_panel_count = (M + pm - 1) / pm;
  const unsigned int n_panel_count = (N + pn - 1) / pn;
  const unsigned int panel_count = m_panel_count * n_panel_count;

  auto run_panel = [&](unsigned int panel_idx,
                       HgemmWorkspace &local_workspace) {
    const unsigned int mi = panel_idx / n_panel_count;
    const unsigned int ni = panel_idx % n_panel_count;
    const unsigned int ms = mi * pm;
    const unsigned int ns = ni * pn;
    const unsigned int m_min = std::min<unsigned int>(M - ms, pm);
    const unsigned int n_min = std::min<unsigned int>(N - ns, pn);
    const AType *a_panel =
      TransA ? A + ms : A + static_cast<std::size_t>(ms) * a_stride;
    const BType *b_panel =
      TransB ? B + static_cast<std::size_t>(ns) * b_stride : B + ns;
    CType *c_panel = C + static_cast<std::size_t>(ms) * c_stride + ns;

    run_hgemm_panel<AType, BType, CType>(
      TransA, TransB, m_min, n_min, K, alpha, a_panel, a_stride, b_panel,
      b_stride, beta, c_panel, c_stride, local_workspace, block);
  };

  if (should_parallelize_blocked(M, N, K, panel_count)) {
    auto &tm = ThreadManager::Global();
    tm.parallel_for(
      0, static_cast<std::size_t>(panel_count), [&](std::size_t panel_idx) {
        run_panel(static_cast<unsigned int>(panel_idx), get_hgemm_workspace());
      });
    return;
  }

  for (unsigned int panel_idx = 0; panel_idx < panel_count; ++panel_idx) {
    run_panel(panel_idx, workspace);
  }
}

template void run_hgemm_blocked<_FP16, _FP16, _FP16>(
  bool, bool, unsigned int, unsigned int, unsigned int, float, const _FP16 *,
  unsigned int, const _FP16 *, unsigned int, float, _FP16 *, unsigned int,
  HgemmWorkspace &);
template void run_hgemm_blocked<float, _FP16, float>(
  bool, bool, unsigned int, unsigned int, unsigned int, float, const float *,
  unsigned int, const _FP16 *, unsigned int, float, float *, unsigned int,
  HgemmWorkspace &);
template void run_hgemm_blocked<_FP16, float, float>(
  bool, bool, unsigned int, unsigned int, unsigned int, float, const _FP16 *,
  unsigned int, const float *, unsigned int, float, float *, unsigned int,
  HgemmWorkspace &);

} /* namespace nntrainer::hgemm::internal */
