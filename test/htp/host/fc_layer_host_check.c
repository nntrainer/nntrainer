/* Host harness for hexkl_mm_u8i4_layer_run (the FC / projection call) on
   the scalar stand-ins: several handles of different widths against one
   activation, as the attention q/k/v call makes it, checked row by row
   against the reference. What it checks is the batching added for the
   pooled epilogue -- that staged slot j carries n-tile nt0+j of the right
   handle and lands at its columns of the right output block, across row
   blocks, across handles, and through a partial last row block -- not the
   HMX's arithmetic, which the stubs define self-consistently. */
#include "hexkl_mm_opts.h"
#include "hexkl_mm_u8i4_dma.h"
#include "hexkl_probe.h"
#include "hvx_scalar_stubs.h"
#include <AEEStdErr.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void reference(const W *const *ws, uint32_t n_h, const float *x,
                      uint32_t M, uint32_t K, float *want, float base) {
  uint8_t *xq = (uint8_t *)malloc(K);
  size_t off = 0;
  for (uint32_t i = 0; i < n_h; ++i) {
    for (uint32_t t = 0; t < M; ++t) {
      float xs;
      int32_t xz;
      quant_row(x + (size_t)t * K, K, xq, &xs, &xz);
      ref_mm(ws[i], xq, xs, xz, want + off + (size_t)t * ws[i]->N);
      if (base != 0.f) {
        for (uint32_t c = 0; c < ws[i]->N; ++c)
          want[off + (size_t)t * ws[i]->N + c] += base;
      }
    }
    off += (size_t)M * ws[i]->N;
  }
  free(xq);
}

static int compare(const char *what, const float *got, const float *want,
                   size_t n) {
  double worst = 0.0;
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) {
    double d = fabs((double)got[i] - (double)want[i]);
    double s = fabs((double)want[i]) + 1e-6;
    if (d / s > 1e-5)
      ++bad;
    if (d / s > worst)
      worst = d / s;
  }
  printf("%-18s: mismatches=%zu of %zu   worst_rel=%g\n", what, bad, n, worst);
  return bad != 0;
}

int main(void) {
  /* Three row blocks with a short last one; the widest handle is 33
     n-tiles, one more than a staging batch, so the first handle takes two
     batches per row block and the parity crosses handles unevenly. */
  const uint32_t M = 150, K = 64;
  const uint32_t N[3] = {1056, 512, 512};
  static uint8_t vtcm[8u << 20];
  W w[3];
  const W *ws[3];
  uint32_t handles[3];
  size_t out_n = 0;
  for (uint32_t i = 0; i < 3; ++i) {
    make_weight(i, K, N[i], &w[i]);
    ws[i] = &w[i];
    handles[i] = i;
    out_n += (size_t)M * N[i];
  }
  float *x = (float *)malloc(sizeof(float) * M * K);
  for (uint32_t i = 0; i < M * K; ++i)
    x[i] = rndf();
  float *got = (float *)malloc(sizeof(float) * out_n);
  float *want = (float *)malloc(sizeof(float) * out_n);
  int fail = 0;

  int rc = hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, K,
                                   handles, 3, x, got, NULL);
  printf("run rc=%d\n", rc);
  if (rc)
    return 1;
  reference(ws, 3, x, M, K, want, 0.f);
  fail |= compare("three handles", got, want, out_n);

  /* One row: decode's shape, 63 padding rows never emitted. */
  rc = hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, 1, K,
                               handles, 3, x, got, NULL);
  reference(ws, 3, x, 1, K, want, 0.f);
  fail |= (rc != 0) | compare("M=1", got, want, N[0] + N[1] + N[2]);

  /* accumulate: the synchronous path, adding into what is there. */
  {
    hexkl_mm_opts o = {0};
    o.accumulate = 1;
    for (size_t i = 0; i < out_n; ++i)
      got[i] = 1.0f;
    rc = hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, K,
                                 handles, 3, x, got, &o);
    reference(ws, 3, x, M, K, want, 1.0f);
    fail |= (rc != 0) | compare("accumulate", got, want, out_n);
  }

  /* An arena with room for exactly two staging tiles still runs (one
     tile per batch); one tile short of that is refused. */
  {
    const uint32_t k_tiles = K / 32u, m_pad = 192u;
    const uint32_t act = (m_pad / 64u) * k_tiles * 2048u;
    const uint32_t wb = k_tiles * (N[0] / 32u) * 512u;
    const uint32_t result_off = act + 2u * wb;
    rc = hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, result_off + 2u * 8192u,
                                 result_off + 2u * 8192u, M, K, handles, 3, x,
                                 got, NULL);
    reference(ws, 3, x, M, K, want, 0.f);
    fail |= (rc != 0) | compare("2-tile arena", got, want, out_n);
    rc = hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, result_off + 8192u,
                                 result_off + 8192u, M, K, handles, 3, x, got,
                                 NULL);
    printf("1-tile arena      : rc=%d (want %d)\n", rc, AEE_ENOMEMORY);
    fail |= (rc != AEE_ENOMEMORY);
  }

  /* [#236] The chunker's shape: the qkv projection at P1024 (M = 1024,
     K = 2048, N 2048 / 512 / 512) as one call vs two 512-row calls whose
     per-handle blocks land at m0 * N[i] of each handle's whole block, as
     gemm_q4_0_batch_fp32's dsts put them -- byte-equal, because every row's
     quantization and epilogue are its own. The mutant (the second chunk
     reading the first chunk's rows) must differ, so the compare can fail. */
  {
    const uint32_t CM = 1024, CK = 2048, CS = 512;
    const uint32_t CN[3] = {2048, 512, 512};
    const uint32_t ch[3] = {3, 4, 5};
    W cw[3];
    size_t cn = 0;
    for (uint32_t i = 0; i < 3; ++i) {
      make_weight(ch[i], CK, CN[i], &cw[i]);
      cn += (size_t)CM * CN[i];
    }
    float *cx = (float *)malloc(sizeof(float) * CM * CK);
    for (size_t i = 0; i < (size_t)CM * CK; ++i)
      cx[i] = rndf();
    float *whole = (float *)malloc(sizeof(float) * cn);
    float *chunk = (float *)malloc(sizeof(float) * cn);
    float *part = (float *)malloc(sizeof(float) * (cn / (CM / CS)));
    int r = hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, CM,
                                    CK, ch, 3, cx, whole, NULL);
    int same = 0, mutant_same = 1;
    for (int mut = 0; mut < 2 && r == 0; ++mut) {
      for (uint32_t m0 = 0; m0 < CM && r == 0; m0 += CS) {
        const uint32_t src = mut ? 0 : m0;
        r =
          hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, CS,
                                  CK, ch, 3, cx + (size_t)src * CK, part, NULL);
        size_t off = 0, poff = 0;
        for (uint32_t i = 0; i < 3; ++i) {
          memcpy(chunk + off + (size_t)m0 * CN[i], part + poff,
                 sizeof(float) * CS * CN[i]);
          off += (size_t)CM * CN[i];
          poff += (size_t)CS * CN[i];
        }
      }
      const int eq = r == 0 && !memcmp(chunk, whole, sizeof(float) * cn);
      if (mut)
        mutant_same = eq;
      else
        same = eq;
    }
    printf("chunked 512+512   : %s\n",
           same ? "FC LAYER CHUNKED BIT-IDENTICAL" : "DIFFERS");
    printf("mutant (rows 0..) : %s\n",
           mutant_same ? "SAME (compare is blind)" : "differs, as it must");
    fail |= !same | mutant_same;
    free(cx);
    free(whole);
    free(chunk);
    free(part);
  }

  printf(fail ? "\nFAIL\n" : "\nALL CHECKS PASS\n");
  return fail;
}
