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
#include "hvx_quant_u8.h"
#include "hvx_rmsnorm_rows_f32.h"
#include "hvx_rope_rows_f32.h"
#include "hvx_scalar_stubs.h"
#include "standin/gelu_tab_det.h"
#include "swiglu_det.h"
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

/** a16 reference: row m's u16 codes q (given), then per column the two
    exact sweeps hi = sum (q >> 8) w, lo = sum (q & 255) w and the kernel's
    f32 combine (256 hi + lo - zp cs) s ws + b. */
static void ref16(const W *w, const uint16_t *q, float s, int32_t zp,
                  float *out) {
  const uint32_t kt_n = w->K / 32u, nt_n = w->N / 32u;
  for (uint32_t nt = 0; nt < nt_n; ++nt)
    for (uint32_t c = 0; c < 32; ++c) {
      int32_t hi = 0, lo = 0;
      for (uint32_t kt = 0; kt < kt_n; ++kt)
        for (uint32_t k = 0; k < 32; ++k) {
          const int wv = wh_value(
            (const uint8_t *)w->nib + (size_t)(kt * nt_n + nt) * 512u, k, c);
          hi += (int32_t)(q[kt * 32 + k] >> 8) * wv;
          lo += (int32_t)(q[kt * 32 + k] & 255u) * wv;
        }
      const uint32_t col = nt * 32 + c;
      out[col] =
        ((float)hi * 256.0f + (float)lo - (float)zp * (float)w->cs[col]) * s *
          w->ws[col] +
        w->bias[col];
    }
}

/* Row x's u16 codes over [0, 65535], the pack's parameter formula. */
static void quant16(const float *x, uint32_t k, uint16_t *q, float *s,
                    int32_t *zp) {
  float lo = 0.f, hi = 0.f;
  for (uint32_t j = 0; j < k; ++j) {
    lo = x[j] < lo ? x[j] : lo;
    hi = x[j] > hi ? x[j] : hi;
  }
  *s = lo != hi ? (hi - lo) / 65535.0f : 1.0f;
  long z = lo != hi ? (long)nearbyintf(-lo / *s) : 0;
  *zp = (int32_t)(z < 0 ? 0 : (z > 65535 ? 65535 : z));
  const float inv = 1.0f / *s;
  for (uint32_t j = 0; j < k; ++j) {
    long v = (long)nearbyintf(x[j] * inv) + *zp;
    q[j] = (uint16_t)(v < 0 ? 0 : (v > 65535 ? 65535 : v));
  }
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

  /* The per-head passes in the epilogue (hexkl_mm_opts.post_done): the
     norm of handles 0 and 2 and the rotation of handle 0 must give the
     bytes of the plain run with those passes made over it afterwards, and
     the kernel must say it did them. Then a table row off by one, which
     must not. */
  {
    const uint32_t HD = 32u, chunk[3] = {32u, 0u, 32u};
    float gm[64], *cs = (float *)malloc(sizeof(float) * (M + 1u) * 2u * HD);
    float *got2 = (float *)malloc(sizeof(float) * out_n);
    for (uint32_t j = 0; j < 64u; ++j)
      gm[j] = 0.5f + 0.01f * (float)j;
    for (uint32_t j = 0; j < (M + 1u) * 2u * HD; ++j)
      cs[j] = rndf();
    rc = hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, K,
                                 handles, 3, x, got, NULL);
    size_t off = 0;
    for (uint32_t i = 0; i < 3; ++i) {
      for (uint32_t m = 0; m < M; ++m) {
        float *row = got + off + (size_t)m * N[i];
        if (chunk[i])
          hvx_rmsnorm_row_chunks_f32(row, row, gm + (i ? 32 : 0), N[i],
                                     chunk[i], 1e-6f);
        if (i == 0)
          hvx_rope_row_f32(row, N[i], HD, cs + (size_t)m * 2u * HD);
      }
      off += (size_t)M * N[i];
    }
    int done = 0;
    hexkl_mm_opts o = {0};
    o.post_done = &done;
    o.post_chunk = chunk;
    o.post_gamma = gm;
    o.post_eps = 1e-6f;
    o.rope_hd = HD;
    o.rope_handles = 1u;
    o.rope_cs = cs;
    rc |= hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, K,
                                  handles, 3, x, got2, &o);
    const int same = !memcmp(got, got2, sizeof(float) * out_n);
    o.rope_cs = cs + 2u * HD;
    rc |= hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, K,
                                  handles, 3, x, got2, &o);
    const int blind = !memcmp(got, got2, sizeof(float) * out_n);
    printf("post in epilogue  : rc=%d done=%d %s%s\n", rc, done,
           same ? "bit-identical to the passes after" : "DIFFERS",
           blind ? " (compare is blind)" : "");
    fail |= (rc != 0) | !done | !same | blind;
    free(cs);
    free(got2);
  }

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
    /** the weight double buffer, then the byte-plane readout's three 1 KiB
        convert blocks rounded up to the 2 KiB activation alignment */
    const uint32_t result_off = act + 2u * wb + 4096u;
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

  /* The u16 activation (the attention's context for the o-proj): rows
     dequantized inside the quant pass, per head (scale, zp), must give
     the same bytes as the f32 call on the dequantized rows. A mutant
     encoding (the second head's scale) must differ. */
  {
    /* two heads of 64 (the pass takes a head per 64-lane vector) */
    const uint32_t UK = 128u, HD = 64u, UN = 512u;
    const uint32_t uh[1] = {6};
    W uw;
    make_weight(6, UK, UN, &uw);
    const W *uws[1] = {&uw};
    float enc[2 * 2] = {0.001f, 32768.0f, 0.0025f, 31000.0f};
    uint16_t *x16 = (uint16_t *)malloc(sizeof(uint16_t) * M * UK);
    float *xd = (float *)malloc(sizeof(float) * M * UK);
    float *scratch = (float *)malloc(sizeof(float) * M * UK);
    float *got16 = (float *)malloc(sizeof(float) * M * UN);
    float *gotf = (float *)malloc(sizeof(float) * M * UN);
    float *wantu = (float *)malloc(sizeof(float) * M * UN);
    for (uint32_t i = 0; i < M * UK; ++i) {
      x16[i] = (uint16_t)(20000u + (uint32_t)(i * 2654435761u) % 25000u);
      const uint32_t h = (i % UK) / HD;
      xd[i] = ((float)x16[i] - enc[2 * h + 1]) * enc[2 * h];
    }
    reference(uws, 1, xd, M, UK, wantu, 0.f);
    hexkl_mm_opts o = {0};
    o.pre_scratch = scratch;
    o.act_u16 = x16;
    o.act16_enc = enc;
    o.act16_hd = HD;
    rc = hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, UK,
                                 uh, 1, NULL, got16, &o);
    printf("u16 run rc=%d\n", rc);
    fail |= (rc != 0) | compare("u16 activation", got16, wantu, M * UN);
    rc = hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, UK,
                                 uh, 1, xd, gotf, NULL);
    const int same = rc == 0 && !memcmp(gotf, got16, sizeof(float) * M * UN);
    printf("u16 vs f32 rows   : %s\n",
           same ? "U16 ACTIVATION BIT-IDENTICAL" : "DIFFERENT");
    enc[2] = 0.003f;
    rc = hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, UK,
                                 uh, 1, NULL, got16, &o);
    const int mut_same =
      rc == 0 && !memcmp(gotf, got16, sizeof(float) * M * UN);
    printf("mutant enc        : %s\n",
           mut_same ? "SAME (compare is blind)" : "differs, as it must");
    fail |= (!same) | mut_same;
    free(x16);
    free(xd);
    free(scratch);
    free(got16);
    free(gotf);
    free(wantu);
  }

  /** a16 (hexkl_mm_opts.act16): two handles with nonzero colsums (the zero
      point term), 150 rows (a short last row block), K = 128 so the u16
      input can carry two heads. Three sources: f32 rows, u16 rows with one
      encoding (their codes used as they are) and u16 rows with a head each
      (requantized per row); then the flags it must refuse. */
  {
    const uint32_t AK = 128u, AN[2] = {1056u, 512u}, AH = 64u;
    W aw[2];
    uint32_t ah[2] = {3u, 4u};
    for (uint32_t i = 0; i < 2; ++i) {
      make_weight(ah[i], AK, AN[i], &aw[i]);
      for (uint32_t c = 0; c < AN[i]; ++c)
        aw[i].cs[c] = (int32_t)(rnd() % 2001u) - 1000;
    }
    const size_t an = (size_t)M * (AN[0] + AN[1]);
    float *ax = (float *)malloc(sizeof(float) * M * AK);
    uint16_t *a16 = (uint16_t *)malloc(sizeof(uint16_t) * M * AK);
    uint16_t *q = (uint16_t *)malloc(sizeof(uint16_t) * AK);
    float *ag = (float *)malloc(sizeof(float) * an);
    float *aw_ = (float *)malloc(sizeof(float) * an);
    float *dq = (float *)malloc(sizeof(float) * AK);
    float *asc = (float *)malloc(sizeof(float) * 8u * AK);
    for (uint32_t i = 0; i < M * AK; ++i) {
      ax[i] = rndf();
      a16[i] = (uint16_t)(rnd() & 0xffffu);
    }
    hexkl_mm_opts o = {0};
    o.act16 = 1;

    /* f32 rows */
    rc = hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, AK,
                                 ah, 2, ax, ag, &o);
    size_t off = 0;
    for (uint32_t i = 0; i < 2; ++i) {
      for (uint32_t t = 0; t < M; ++t) {
        float s;
        int32_t z;
        quant16(ax + (size_t)t * AK, AK, q, &s, &z);
        ref16(&aw[i], q, s, z, aw_ + off + (size_t)t * AN[i]);
      }
      off += (size_t)M * AN[i];
    }
    fail |= (rc != 0) | compare("a16 f32 rows", ag, aw_, an);

    /* u16 rows, one encoding: zp 30000.4 rounds to 30000 */
    const float enc1[2] = {0.0123f, 30000.4f};
    o.act_u16 = a16;
    o.act16_enc = enc1;
    o.act16_hd = AK;
    rc = hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, AK,
                                 ah, 2, NULL, ag, &o);
    off = 0;
    for (uint32_t i = 0; i < 2; ++i) {
      for (uint32_t t = 0; t < M; ++t)
        ref16(&aw[i], a16 + (size_t)t * AK, enc1[0], 30000,
              aw_ + off + (size_t)t * AN[i]);
      off += (size_t)M * AN[i];
    }
    fail |= (rc != 0) | compare("a16 u16 direct", ag, aw_, an);

    /**
     * the same rows prepacked (hexkl_mm_opts.act_ah_prepacked with act16):
     * both planes, low then high, and the fixed encoding as act_scale /
     * act_zp, must give the bytes of the direct run
     */
    {
      const uint32_t mp = (M + 63u) / 64u * 64u;
      uint8_t *pre = (uint8_t *)malloc(2u * (size_t)mp * AK);
      float *psc = (float *)malloc(sizeof(float) * mp);
      int32_t *pzp = (int32_t *)malloc(sizeof(int32_t) * mp);
      float *ag2 = (float *)malloc(sizeof(float) * an);
      int prc = hvx_quant_params_pack_u16_ah(NULL, a16, enc1, AK, NULL, M, mp,
                                             AK, NULL, 0.f, NULL, psc, pzp, pre,
                                             pre + (size_t)mp * AK, NULL);
      hexkl_mm_opts op = {0};
      op.act16 = 1;
      op.act_ah_prepacked = pre;
      op.act_scale = psc;
      op.act_zp = pzp;
      prc |= hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M,
                                     AK, ah, 2, NULL, ag2, &op);
      const int same = prc == 0 && !memcmp(ag, ag2, sizeof(float) * an);
      op.act_scale = NULL;
      op.act_zp = NULL;
      const int r0 = hexkl_mm_u8i4_layer_run(
        &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, AK, ah, 2, NULL, ag2, &op);
      printf("a16 prepacked     : rc=%d %s; without its encoding rc=%d\n", prc,
             same ? "bit-identical to the direct run" : "DIFFERS", r0);
      fail |= !same | (r0 != AEE_EBADPARM);
      free(pre);
      free(psc);
      free(pzp);
      free(ag2);
    }

    /* u16 rows, a (scale, zp) per head of 64 */
    const float enc2[4] = {0.002f, 31000.f, 0.0007f, 12000.f};
    o.act16_enc = enc2;
    o.act16_hd = AH;
    o.pre_scratch = asc;
    rc = hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, AK,
                                 ah, 2, NULL, ag, &o);
    off = 0;
    for (uint32_t i = 0; i < 2; ++i) {
      for (uint32_t t = 0; t < M; ++t) {
        for (uint32_t j = 0; j < AK; ++j)
          dq[j] = ((float)a16[(size_t)t * AK + j] - enc2[2u * (j / AH) + 1u]) *
                  enc2[2u * (j / AH)];
        float s;
        int32_t z;
        quant16(dq, AK, q, &s, &z);
        ref16(&aw[i], q, s, z, aw_ + off + (size_t)t * AN[i]);
      }
      off += (size_t)M * AN[i];
    }
    fail |= (rc != 0) | compare("a16 u16 per head", ag, aw_, an);

    /* accumulate: the same call again adds the same floats, v + v */
    o.accumulate = 1;
    memcpy(aw_, ag, sizeof(float) * an);
    const int r1 = hexkl_mm_u8i4_layer_run(
      &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, AK, ah, 2, NULL, ag, &o);
    size_t abad = 0;
    for (size_t i = 0; i < an; ++i) {
      const float v2 = aw_[i] + aw_[i];
      abad += memcmp(&v2, &ag[i], sizeof v2) != 0;
    }
    printf("a16 accumulate    : rc=%d, %zu of %zu not v + v\n", r1, abad, an);
    fail |= (r1 != 0) | (abad != 0);
    /* refused: per-head u16 without a scratch */
    o.accumulate = 0;
    o.pre_scratch = NULL;
    const int r2 = hexkl_mm_u8i4_layer_run(
      &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, AK, ah, 2, NULL, ag, &o);
    printf("a16 refusals      : no scratch rc=%d\n", r2);
    fail |= r2 != AEE_EBADPARM;

    /**
     * a16 from the handles (hexkl_weight_u8i4_set_act16): a plain f32 call
     * at their static encoding, rows RMS-normed on the way (pre_gamma), at
     * M rows on the HMX and 1..4 rows on the GEMV, which must give the
     * HMX rows' floats; mixed handles refused
     */
    {
      const float fs = 0.00071f, fz = 30461.f;
      float gamma[128];
      for (uint32_t j = 0; j < AK; ++j)
        gamma[j] = 0.5f + 0.01f * (float)(j % 17u);
      fail |= hexkl_weight_u8i4_set_act16(&g_tbl, ah[0], fs, fz) != 0;
      fail |= hexkl_weight_u8i4_set_act16(&g_tbl, ah[1], fs, fz) != 0;
      hexkl_mm_opts on = {0};
      on.pre_scratch = asc;
      on.pre_gamma = gamma;
      on.pre_eps = 1e-6f;
      rc = hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M,
                                   AK, ah, 2, ax, ag, &on);
      off = 0;
      for (uint32_t i = 0; i < 2; ++i) {
        for (uint32_t t = 0; t < M; ++t) {
          hvx_rmsnorm_row_f32(ax + (size_t)t * AK, dq, gamma, AK, 1e-6f);
          for (uint32_t j = 0; j < AK; ++j) {
            long v = (long)nearbyintf(dq[j] * (1.0f / fs)) + (long)fz;
            q[j] = (uint16_t)(v < 0 ? 0 : (v > 65535 ? 65535 : v));
          }
          ref16(&aw[i], q, fs, (int32_t)fz, aw_ + off + (size_t)t * AN[i]);
        }
        off += (size_t)M * AN[i];
      }
      fail |= (rc != 0) | compare("a16 static normed", ag, aw_, an);
      size_t gbad = 0;
      for (uint32_t m = 1; m <= 4u; ++m) {
        float *g = (float *)malloc(sizeof(float) * m * (AN[0] + AN[1]));
        rc |= hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, m,
                                      AK, ah, 2, ax, g, &on);
        for (uint32_t i = 0, o0 = 0, g0 = 0; i < 2; ++i) {
          gbad += memcmp(g + g0, ag + o0, sizeof(float) * m * AN[i]) != 0;
          o0 += M * AN[i];
          g0 += m * AN[i];
        }
        free(g);
      }
      printf("a16 static GEMV   : rc=%d, %zu of 8 row runs differ from HMX\n",
             rc, gbad);
      fail |= (rc != 0) | (gbad != 0);

      /**
       * the GLU epilogue (hexkl_mm_opts.glu_lo): two gate_up handles, 36
       * n-tiles (a batch of 32 splits its pairs 16 + 2) and 4, must give
       * the planes of the plain run, its GLU, then the fixed-encoding pack
       * -- pad rows and all, over planes that start as garbage; gelu is
       * the table's (gelu_tab_det.h)
       */
      {
        const uint32_t GN[2] = {1152u, 128u}, GK = 640u;
        const uint32_t mp = (M + 63u) / 64u * 64u;
        uint32_t gh[2] = {5u, 6u};
        W gw[2];
        for (uint32_t i = 0; i < 2; ++i) {
          make_weight(gh[i], AK, GN[i], &gw[i]);
          for (uint32_t c = 0; c < GN[i]; ++c)
            gw[i].cs[c] = (int32_t)(rnd() % 2001u) - 1000;
          fail |= hexkl_weight_u8i4_set_act16(&g_tbl, gh[i], fs, fz) != 0;
        }
        const float genc[2] = {0.0011f, 31000.f};
        float *gf = (float *)malloc(sizeof(float) * M * (GN[0] + GN[1]));
        float *gy = (float *)malloc(sizeof(float) * M * GK);
        uint8_t *gp = (uint8_t *)malloc(4u * (size_t)mp * GK);
        float *gsc = (float *)malloc(sizeof(float) * mp);
        int32_t *gzp = (int32_t *)malloc(sizeof(int32_t) * mp);
        static float gtab[2 * HVX_GELU_TAB_STRIDE];
        gelu_tab_det_build(gtab);
        for (uint32_t act = 0; act < 2u; ++act) {
          rc = hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm,
                                       M, AK, gh, 2, ax, gf, &on);
          for (uint32_t i = 0, o0 = 0, c0 = 0; i < 2; ++i) {
            const uint32_t in = GN[i] / 2u;
            for (uint32_t t = 0; t < M; ++t)
              for (uint32_t c = 0; c < in; ++c) {
                const float *r = gf + o0 + (size_t)t * GN[i];
                gy[(size_t)t * GK + c0 + c] =
                  act ? gelu_tab_det_one(gtab, r[c], r[in + c])
                      : swiglu_det_one(r[c], r[in + c]);
              }
            o0 += M * GN[i];
            c0 += in;
          }
          rc |= hvx_quant_params_pack_u16_ah(gy, NULL, NULL, 0u, genc, M, mp,
                                             GK, NULL, 0.f, NULL, gsc, gzp, gp,
                                             gp + (size_t)mp * GK, NULL);
          memset(gp + 2u * (size_t)mp * GK, 0xa5, 2u * (size_t)mp * GK);
          hexkl_mm_opts og = on;
          og.glu_lo = gp + 2u * (size_t)mp * GK;
          og.glu_hi = og.glu_lo + (size_t)mp * GK;
          og.glu_enc = genc;
          og.glu_k = GK;
          og.glu_act = act;
          rc |= hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm,
                                        M, AK, gh, 2, ax, NULL, &og);
          size_t bad = 0;
          for (size_t b = 0; b < 2u * (size_t)mp * GK; ++b)
            bad += gp[b] != og.glu_lo[b];
          og.glu_k = GK - 32u;
          const int r4 =
            hexkl_mm_u8i4_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M,
                                    AK, gh, 2, ax, NULL, &og);
          printf("a16 GLU planes %s: rc=%d, %zu of %zu bytes differ; "
                 "short glu_k rc=%d\n",
                 act ? "gelu" : "silu", rc, bad, 2u * (size_t)mp * GK, r4);
          fail |= (rc != 0) | (bad != 0) | (r4 != AEE_EBADPARM);
        }
        for (uint32_t i = 0; i < 2; ++i)
          hexkl_weight_u8i4_set_act16(&g_tbl, gh[i], 0.f, 0.f);
        free(gf);
        free(gy);
        free(gp);
        free(gsc);
        free(gzp);
      }
      fail |= hexkl_weight_u8i4_set_act16(&g_tbl, ah[1], 0.f, 0.f) != 0;
      const int r3 = hexkl_mm_u8i4_layer_run(
        &g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, AK, ah, 2, ax, ag, &on);
      printf("a16 mixed handles : rc=%d\n", r3);
      fail |= r3 != AEE_EBADPARM;
      hexkl_weight_u8i4_set_act16(&g_tbl, ah[0], 0.f, 0.f);
    }
    free(ax);
    free(a16);
    free(q);
    free(ag);
    free(aw_);
    free(dq);
    free(asc);
  }

  printf(fail ? "\nFAIL\n" : "\nALL CHECKS PASS\n");
  return fail;
}
