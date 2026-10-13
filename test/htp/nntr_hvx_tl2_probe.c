// SPDX-License-Identifier: Apache-2.0
/**
 * @file nntr_hvx_tl2_probe.c
 * @brief Native symmetric TL2 HVX GEMV device validation and timing.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#include "hexkl_dma_ring.h"
#include "hexkl_probe.h"
#include "hvx_gemv_u8tl2.h"
#include "hvx_worker_pool.h"
#include "nntr_hvx.h"
#include "nntr_hvx_session.h"
#include <AEEStdErr.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <remote.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  uint32_t K, N, threads, splits;
  const uint8_t *act, *weight;
  int16_t *tables;
  int32_t *out;
} tl2_probe_ctx;

static void tl2_prepare_worker(uint32_t threads, uint32_t id, void *v) {
  const tl2_probe_ctx *c = v;
  const uint32_t blocks = (c->K + 23u) / 24u;
  hvx_tl2_prepare(c->act, c->K, blocks * id / threads,
                  blocks * (id + 1u) / threads, c->tables);
}

static void tl2_gemv_worker(uint32_t threads, uint32_t id, void *v) {
  tl2_probe_ctx *c = v;
  const uint32_t panels = (c->N + 127u) / 128u;
  const uint32_t blocks = (c->K + 23u) / 24u;
  if (!id)
    c->threads = threads;
  for (uint32_t u = panels * c->splits * id / threads;
       u < panels * c->splits * (id + 1u) / threads; ++u) {
    const uint32_t p = u / c->splits, split = u % c->splits;
    const uint32_t lo = blocks * split / c->splits;
    const uint32_t hi = blocks * (split + 1u) / c->splits;
    hvx_gemv_u8tl2_panel(c->weight + ((size_t)p * blocks + lo) * 640u,
                         (hi - lo) * 24u, c->tables + (size_t)lo * 256u,
                         c->out + (split * panels + p) * 128u);
  }
}

int nntr_hvx_tl2_gemv_probe(remote_handle64 handle, uint32 K, uint32 N,
                            uint32 mode, uint32 workers, uint32 passes,
                            const uint8 *act, int actLen, const uint8 *weight,
                            int weightLen, int32 *out, int outLen,
                            uint32 *stats, int statsLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s || !s->vtcm_base || !K || !N || K > 8192u || N > 8192u ||
      (mode & ~771u) || !workers || workers > 8u || !passes || passes > 1000u ||
      !act || !weight || !out || !stats || actLen != (int)K ||
      outLen != (int)N || statsLen < 6 ||
      weightLen != (int)hvx_tl2_weight_bytes(K, N))
    return AEE_EBADPARM;
  const size_t bytes = hvx_tl2_weight_bytes(K, N);
  const size_t tab_bytes = hvx_tl2_table_bytes(K);
  const uint32_t npad = (N + 127u) & ~127u;
  const uint32_t splits =
    (mode & 3u) == 3u ? hvx_tl2_k_splits(K, N, workers) : 1u << (mode & 3u);
  if (bytes + tab_bytes + npad * 4u * splits > s->config_off)
    return AEE_ENOMEMORY;
  /* The benchmark validates canonical encodings before timing. */
  int8_t *decoded = malloc((size_t)K * N);
  if (!decoded)
    return AEE_ENOMEMORY;
  const int valid = hvx_tl2_unpack(weight, K, N, decoded);
  free(decoded);
  if (valid)
    return AEE_EBADPARM;
  uint8_t *a = memalign(128u, (K + 127u) & ~127u);
  if (!a)
    return AEE_ENOMEMORY;
  uint8_t *aligned_weight = NULL;
  if (mode & 512u) {
    aligned_weight = memalign(128u, bytes);
    if (!aligned_weight) {
      free(a);
      return AEE_ENOMEMORY;
    }
    memcpy(aligned_weight, weight, bytes);
    weight = aligned_weight;
  }
  memcpy(a, act, K);
  uint8_t *w = s->vtcm_base;
  int16_t *tables = (int16_t *)(w + bytes);
  int32_t *result = (int32_t *)((uint8_t *)tables + tab_bytes);
  memcpy(w, weight, bytes);
  hvx_worker_pool *pool = workers == 1u ? NULL : s->quant_pool;
  tl2_probe_ctx c = {K, N, 0u, splits, a, w, tables, result};
  memset(stats, 0, (size_t)statsLen * sizeof(*stats));
  const uint32_t panels = npad / 128u;
  for (uint32_t pass = 0; pass < passes + 3u; ++pass) {
    const uint64_t t0 = hexkl_probe_now();
    uint32_t dma_idx = 0;
    if (mode & 256u) {
      dma_idx = hexkl_dma_ring_next_idx();
      /* Native panels are contiguous: one linear burst, with no row gaps. */
      hexkl_dma_ring_push2d(w, weight, (uint32_t)bytes, (uint32_t)bytes,
                            (uint32_t)bytes, 1u, 0, 1);
    }
    const uint64_t t1 = hexkl_probe_now();
    hvx_worker_pool_run(pool, tl2_prepare_worker, &c, (K + 23u) / 24u);
    const uint64_t t2 = hexkl_probe_now();
    /** The activation LUT does not read weights. Hide its preparation under
     * the DMA into the disjoint weight region, then join before any lookup. */
    if (mode & 256u)
      hexkl_dma_ring_wait(dma_idx);
    const uint64_t ready = hexkl_probe_now();
    const uint32_t jobs = panels * splits;
    hvx_worker_pool_run(pool, tl2_gemv_worker, &c,
                        jobs < workers ? jobs : workers);
    for (uint32_t split = 1u; split < splits; ++split)
      for (uint32_t n = 0; n < npad; n += 32u)
        *(HVX_Vector *)(result + n) =
          Q6_Vw_vadd_VwVw(*(const HVX_Vector *)(result + n),
                          *(const HVX_Vector *)(result + split * npad + n));
    const uint64_t t3 = hexkl_probe_now();
    if (pass >= 3u) {
      stats[0] += (uint32_t)(t1 - t0 + ready - t2);
      stats[2] += (uint32_t)(t2 - t1);
      stats[3] += (uint32_t)(t3 - ready);
      stats[4] += (uint32_t)(t3 - t0);
    }
  }
  stats[5] = c.threads;
  memcpy(out, result, N * sizeof(*out));
  free(a);
  free(aligned_weight);
  return AEE_SUCCESS;
}
