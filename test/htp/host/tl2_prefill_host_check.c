// SPDX-License-Identifier: Apache-2.0
/** @file tl2_prefill_host_check.c
 * @brief Native-only TL2 prefill versus independent int4 WH weights.
 */
#define main nntr_existing_moe_check_main
#include "moe_layer_host_check.c"
#undef main
#include "hvx_gemv_u8tl2.h"
#include <assert.h>

static void check_shape(uint32_t K, uint32_t I, uint32_t N) {
  static uint8_t vtcm[8u << 20] __attribute__((aligned(128)));
  W weights[2];
  uint8_t *native[2];
  const int8_t palette[4] = {-1, 0, 1, 1};
  uint8_t lut[128];
  hvx_expand_i2i4_table(palette, lut);
  for (uint32_t h = 0; h < 2u; ++h) {
    make_weight(h, h ? I : K, h ? N : 2u * I, &weights[h]);
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
        dense[(size_t)k * w->N + n] = q;
        sum += q;
      }
      w->colsum_w[n] = sum;
    }
    native[h] = malloc(hvx_tl2_weight_bytes(w->K, w->N));
    assert(hvx_tl2_pack(dense, w->K, w->N, native[h]) == 0);
    g_tbl.slots[h + 2u] = *w;
    g_tbl.slots[h + 2u].wh_bytes = native[h];
    g_tbl.slots[h + 2u].tl2_bytes = native[h];
    g_tbl.slots[h + 2u].bits = 2u;
    memcpy(g_tbl.slots[h + 2u].pal, palette, 4u);
    free(qs2);
    free(dense);
  }
  float *input = malloc(65u * K * sizeof(float));
  float *expected = malloc(65u * N * sizeof(float));
  float *actual = malloc(65u * N * sizeof(float));
  assert(input && expected && actual);
  uint32_t rows[65];
  float routing[65];
  for (uint32_t r = 0; r < 65u; ++r) {
    rows[r] = r;
    routing[r] = 1.0f;
    for (uint32_t k = 0; k < K; ++k)
      input[r * K + k] = rndf();
  }
  hexkl_moe_scratch scratch = {0};
  for (uint32_t M = 2u; M <= 65u; M = M == 2u ? 64u : M + 1u) {
    for (uint32_t workers = 0; workers <= 3u; workers += 3u) {
      g_workers = workers;
      for (uint32_t mode = 0; mode < 2u; ++mode) {
        uint32_t gu = mode ? 2u : 0u, dn = mode ? 3u : 1u;
        const int rc = hexkl_mm_u8i4_moe_layer_run(
          &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, K, I, N, 1u, &gu, &dn,
          rows, &M, routing, input, mode ? actual : expected, NULL, &scratch,
          0xf03e1u | HEXKL_MOE_FLAG_GELU_TANH);
        assert(rc == 0);
      }
      assert(memcmp(expected, actual, (size_t)M * N * sizeof(float)) == 0);
      printf("Native-only TL2 prefill exact K=%u I=%u N=%u M=%u workers=%u\n",
             K, I, N, M, workers);
    }
    if (K > 96u)
      break; // real Gemma shape: the K tail and the gate/up panel boundary
  }
  free(input);
  free(expected);
  free(actual);
  hexkl_moe_scratch_free(&scratch);
  for (uint32_t h = 0; h < 2u; ++h) {
    free(native[h]);
    free(weights[h].nib);
    free(weights[h].ws);
    free(weights[h].cs);
    free(weights[h].bias);
  }
}

int main(void) {
  check_shape(96u, 192u, 128u);
  check_shape(2816u, 704u, 2816u);
  return 0;
}
