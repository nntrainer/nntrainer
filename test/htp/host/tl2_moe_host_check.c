// SPDX-License-Identifier: Apache-2.0
/** @file tl2_moe_host_check.c
 * @brief Exact whole-MoE TL2 vs PR #4416 QS2, including DMA queue schedules.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker.
 */
#define main nntr_existing_moe_check_main
#include "moe_layer_host_check.c"
#undef main
#include "hvx_gemv_u8tl2.h"

static int check(int gu_magnitude, int dn_magnitude) {
  const uint32_t K = 2816u, I = 704u, N = 2816u, E = 8u;
  static uint8_t vtcm[8u << 20] __attribute__((aligned(128)));
  W weights[16];
  uint8_t *qs2[16], *tl2[16];
  uint32_t hg[8], hd[8], ri[8] = {0}, count[8];
  float rw[8], input[2816], ref[2816], got[2816];
  uint8_t lut[128];
  for (uint32_t h = 0; h < 2u * E; ++h) {
    const int magnitude = h < E ? gu_magnitude : dn_magnitude;
    const int8_t pal[4] = {-magnitude, -magnitude, 0, magnitude};
    hvx_expand_i2i4_table(pal, lut);
    make_weight(h, h < E ? K : I, h < E ? 2u * I : N, &weights[h]);
    hexkl_weight_u8i4 *w = &g_tbl.slots[h];
    size_t bytes = (size_t)w->K * w->N / 4u;
    qs2[h] = malloc(bytes);
    tl2[h] = malloc(hvx_tl2_weight_bytes(w->K, w->N));
    int8_t *dense = malloc(bytes * 4u);
    for (size_t i = 0; i < bytes; ++i)
      qs2[h][i] = rnd();
    hvx_expand_i2i4_scalar(qs2[h], bytes, lut, w->wh_bytes);
    for (uint32_t n = 0; n < w->N; ++n) {
      int sum = 0;
      for (uint32_t k = 0; k < w->K; ++k) {
        const uint32_t row = k % 32u;
        const size_t at = ((size_t)(k / 32u) * (w->N / 32u) + n / 32u) * 512u +
                          row / 8u * 128u + n % 32u * 4u + row % 4u;
        int val = (w->wh_bytes[at] >> (4u * (row / 4u % 2u))) & 15u;
        val = val >= 8 ? val - 16 : val;
        sum += val;
        dense[(size_t)k * w->N + n] = val / magnitude;
      }
      w->colsum_w[n] = sum;
    }
    if (hvx_tl2_pack(dense, w->K, w->N, tl2[h]))
      abort();
    free(dense);
    memcpy(w->pal, pal, 4u);
    w->bits = 2u;
    w->wh_bytes = qs2[h];
  }
  for (uint32_t e = 0; e < E; ++e) {
    hg[e] = e;
    hd[e] = E + e;
    count[e] = 1;
    rw[e] = 0.125f;
  }
  for (uint32_t k = 0; k < K; ++k)
    input[k] = rndf();
  hexkl_moe_scratch scratch = {0};
  int fail = 0;
  for (uint32_t active = 1; active <= E; ++active) {
    for (uint32_t e = 0; e < E; ++e)
      count[e] = e < active;
    for (uint32_t workers = 0; workers <= 7;
         workers = workers == 0 ? 3 : workers + 4) {
      g_workers = workers;
      for (uint32_t nq = 1; nq <= 4; nq *= 2) {
        const uint32_t flags = 0xf03e1u | HEXKL_MOE_FLAG_GELU_TANH;
        /** Default queues are encoded in bits 22..23; use the public helper's
         * scheme. */
        const uint32_t opts =
          (flags & ~(HEXKL_MOE_DMA_Q_BITS << HEXKL_MOE_DMA_Q_SHIFT)) |
          ((nq - 1u) << HEXKL_MOE_DMA_Q_SHIFT);
        for (uint32_t mode = 0; mode < 4; ++mode) {
          for (uint32_t h = 0; h < 2u * E; ++h) {
            /** Exercise either first expert in a mixed call: the shared
             * dense LUT must be ready before QS2 overwrites its AH buffer. */
            const int native = mode == 1u || (mode == 2u && h % E % 2u == 0u) ||
                               (mode == 3u && h % E % 2u == 1u);
            g_tbl.slots[h].wh_bytes = native ? tl2[h] : qs2[h];
            g_tbl.slots[h].tl2_bytes = native ? tl2[h] : NULL;
          }
          const int rc = hexkl_mm_u8i4_moe_layer_run(
            &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, 1u, K, I, N, E, hg, hd, ri,
            count, rw, input, mode ? got : ref, NULL, &scratch, opts);
          fail |= rc != 0 || (mode && memcmp(ref, got, sizeof ref));
          printf("TL2 MoE experts=%u workers=%u queues=%u mode=%u rc=%d\n",
                 active, workers, nq, mode, rc);
        }
        int bad = memcmp(ref, got, sizeof ref);
        printf("TL2 MoE exact=%d\n", bad == 0);
        fail |= bad != 0;
      }
    }
  }
  hexkl_moe_scratch_free(&scratch);
  for (uint32_t h = 0; h < 2u * E; ++h) {
    free(qs2[h]);
    free(tl2[h]);
    free(weights[h].nib);
    free(weights[h].ws);
    free(weights[h].cs);
    free(weights[h].bias);
  }
  return fail;
}

int main(void) { return check(1, 1) | check(7, 7) | check(1, 7) | check(7, 1); }
