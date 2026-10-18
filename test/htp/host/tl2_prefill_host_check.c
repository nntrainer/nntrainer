// SPDX-License-Identifier: Apache-2.0
/** @file tl2_prefill_host_check.c
 * @brief Native-only TL2 prefill versus independent int4 WH weights.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug See docs/gemma4-tl2-affine-handoff.md for validation limits.
 */
#define main nntr_existing_moe_check_main
#include "moe_layer_host_check.c"
#undef main
#include "hvx_gemv_u8tl2.h"
#include <assert.h>

static void check_dma(size_t bytes, uint32_t descriptors, int packed) {
  const hexkl_dma_trace *trace = hexkl_dma_trace_get();
  size_t transferred = 0;
  size_t first_bytes = 0;
  uint32_t count = 0;
  assert(trace->n_dropped == 0u);
  for (uint32_t i = 0; i < trace->n_push; ++i) {
    const hexkl_dma_trace_push_rec *p = &trace->push[i];
    if (p->kind < HEXKL_DMA_KIND_GATE || p->kind > HEXKL_DMA_KIND_DOWN)
      continue;
    ++count;
    transferred += p->bytes;
    if (p->expert == 0u && p->chunk == 0u && p->kind != HEXKL_DMA_KIND_DOWN)
      first_bytes += p->bytes;
    if (packed)
      assert(p->row_size == p->src_stride);
  }
  assert(transferred == bytes);
  if (count != descriptors)
    fprintf(stderr, "native DMA descriptors: got %u expected %u\n", count,
            descriptors);
  assert(count == descriptors);
  assert(hexkl_probe_us[HEXKL_PROBE_DMA_KB] > 0u);
  assert(hexkl_probe_us[HEXKL_PROBE_DMA_KB] <= bytes / 1024u);
  assert(hexkl_probe_us[HEXKL_PROBE_DMA_FIRST_KB] == first_bytes / 1024u);
}

static void check_shape(uint32_t K, uint32_t I, uint32_t N, uint32_t experts,
                        int8_t magnitude, uint32_t packed_desc,
                        uint32_t inplace_desc) {
  static uint8_t vtcm[(9u << 20) + 128u] __attribute__((aligned(128)));
  const uint32_t arena_bytes = sizeof vtcm - 128u;
  const int lfm2 = K == 2048u && I == 1536u && N == 2048u;
  const uint32_t nweights = 2u * experts;
  W weights[6];
  uint8_t *native[6];
  assert(experts <= 3u);
  memset(&g_tbl, 0, sizeof g_tbl);
  const int8_t palette[4] = {-magnitude, 0, magnitude, magnitude};
  uint8_t lut[128];
  hvx_expand_i2i4_table(palette, lut);
  for (uint32_t h = 0; h < nweights; ++h) {
    make_weight(h, h % 2u ? I : K, h % 2u ? N : 2u * I, &weights[h]);
    hexkl_weight_u8i4 *w = &g_tbl.slots[h];
    const size_t codes = (size_t)w->K * w->N / 4u;
    uint8_t *qs2 = malloc(codes);
    int8_t *dense = malloc(codes * 4u);
    for (size_t b = 0; b < codes; ++b)
      qs2[b] = rnd();
    hvx_expand_i2i4_scalar(qs2, codes, lut, w->wh_bytes);
    for (uint32_t n = 0; n < w->N; ++n) {
      int sum = 0;
      for (uint32_t k = 0; k < w->K; ++k) {
        const size_t at = ((size_t)(k / 32u) * (w->N / 32u) + n / 32u) * 512u +
                          k % 32u / 8u * 128u + n % 32u * 4u + k % 4u;
        int q = (w->wh_bytes[at] >> (4u * (k / 4u % 2u))) & 15u;
        q = q > 7 ? q - 16 : q;
        dense[(size_t)k * w->N + n] = q / magnitude;
        sum += q;
      }
      w->colsum_w[n] = sum;
    }
    native[h] = malloc(hvx_tl2_weight_bytes(w->K, w->N));
    assert(hvx_tl2_pack(dense, w->K, w->N, native[h]) == 0);
    g_tbl.slots[h + nweights] = *w;
    g_tbl.slots[h + nweights].wh_bytes = native[h];
    g_tbl.slots[h + nweights].tl2_bytes = native[h];
    g_tbl.slots[h + nweights].bits = 2u;
    memcpy(g_tbl.slots[h + nweights].pal, palette, 4u);
    free(qs2);
    free(dense);
  }
  float *input = malloc(65u * K * sizeof(float));
  float *expected = malloc(65u * N * sizeof(float));
  float *actual = malloc(65u * N * sizeof(float));
  assert(input && expected && actual);
  uint32_t rows[3u * 65u];
  float routing[3u * 65u];
  for (uint32_t r = 0; r < 65u; ++r) {
    for (uint32_t k = 0; k < K; ++k)
      input[r * K + k] = rndf();
  }
  hexkl_moe_layout layout;
  assert(hexkl_mm_u8i4_moe_layout(K, I, N, arena_bytes, &layout) == 0);
  const size_t packed_bytes =
    hvx_tl2_weight_bytes(K, 2u * I) + hvx_tl2_weight_bytes(I, N);
  assert(layout.total + packed_bytes < arena_bytes);
  /** Exact fit, one vector too small, and no spare bytes. Keep the layout
   * and accumulator ring unchanged so only native staging can differ. */
  const uint32_t arenas[] = {arena_bytes,
                             layout.total + packed_bytes,
                             layout.total + packed_bytes - 128u,
                             layout.total,
                             8u << 20,
                             layout.total + hvx_tl2_weight_bytes(K, 2u * I) -
                               128u};
  hexkl_moe_scratch scratch = {0};
  for (uint32_t M = 2u; M <= 65u; M = M == 2u ? 64u : M + 1u) {
    uint32_t gu[3], dn[3], counts[3];
    for (uint32_t e = 0; e < experts; ++e) {
      counts[e] = M;
      for (uint32_t r = 0; r < M; ++r) {
        rows[e * M + r] = (r + e * 17u) % M;
        routing[e * M + r] = 0.25f + 0.125f * (float)((r + e) % 5u);
      }
    }
    for (uint32_t workers = 0; workers <= 3u; workers += 3u) {
      g_workers = workers;
      for (uint32_t mode = 0; mode <= (lfm2 ? 6u : 4u); ++mode) {
        for (uint32_t e = 0; e < experts; ++e) {
          gu[e] = 2u * e + (mode ? nweights : 0u);
          dn[e] = gu[e] + 1u;
        }
        const uint32_t arena = arenas[mode ? mode - 1u : 0u];
        hexkl_moe_layout limited;
        assert(hexkl_mm_u8i4_moe_layout(K, I, N, arena, &limited) == 0);
        assert(limited.total == layout.total);
        memset(vtcm, 0xa5, sizeof vtcm);
        memset(hexkl_probe_us, 0, sizeof hexkl_probe_us);
        const uint64_t bypass_before = g_bypass_bytes;
        const int rc = hexkl_mm_u8i4_moe_layer_run(
          &g_tbl, vtcm, arena_bytes, arena, M, K, I, N, experts, gu, dn, rows,
          counts, routing, input, mode ? actual : expected, NULL, &scratch,
          (0xf03e1u & ~HEXKL_MOE_FLAG_GELU_TANH) |
            (lfm2 ? 0u : HEXKL_MOE_FLAG_GELU_TANH));
        assert(rc == 0);
        for (uint32_t b = arena; b < sizeof vtcm; ++b)
          assert(vtcm[b] == 0xa5);
        if (mode) {
          const int packed = mode <= 2u;
          assert(memcmp(expected, actual, (size_t)M * N * sizeof(float)) == 0);
          const int gu_only = lfm2 && (mode == 3u || mode == 5u);
          check_dma(experts * packed_bytes,
                    experts * (packed    ? packed_desc
                               : gu_only ? 38u
                                         : inplace_desc),
                    packed);
          assert(g_bypass_bytes - bypass_before == experts * packed_bytes);
          printf("Native-only TL2 prefill exact K=%u I=%u N=%u M=%u "
                 "experts=%u magnitude=%d workers=%u arena=%u packed=%d\n",
                 K, I, N, M, experts, magnitude, workers, arena, packed);
        }
      }
    }
    if (K > 1024u)
      break; // real Gemma shape: the K tail and the gate/up panel boundary
  }
  free(input);
  free(expected);
  free(actual);
  hexkl_moe_scratch_free(&scratch);
  for (uint32_t h = 0; h < nweights; ++h) {
    free(native[h]);
    free(weights[h].nib);
    free(weights[h].ws);
    free(weights[h].cs);
    free(weights[h].bias);
  }
}

int main(void) {
  check_shape(96u, 192u, 128u, 3u, 1, 3u, 8u);
  check_shape(128u, 576u, 256u, 2u, 7, 3u, 31u);
  check_shape(2816u, 704u, 2816u, 1u, 1, 7u, 99u);
  /* LFM2 keeps gate/up packed when both native sources exceed 8 MiB. */
  check_shape(2048u, 1536u, 2048u, 2u, 1, 8u, 104u);
  return 0;
}
