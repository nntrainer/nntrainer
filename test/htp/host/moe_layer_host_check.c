/* Host harness for hexkl_mm_u8i4_moe_layer_run: scalar stand-ins for every
   primitive it calls, then the kernel against a straightforward reference.
   The point is the loop structure -- which rows each expert gets, which
   weights it uses, whether the buffer reuse clobbers anything, whether the
   scatter lands on the right output row with the right routing weight --
   not HMX's arithmetic, which the stubs define self-consistently for both
   sides. */
#include "hexkl_acc_tile.h"
#include "hexkl_dma_ring.h"
#include "hexkl_mm_u8i4_moe.h"
#include "hexkl_probe.h"
#include <AEEStdErr.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- accumulator + acc layout ---- */
static int32_t g_acc[64][32];
static hexkl_acc_layout g_layout = {1, 1, 0,
                                    32}; /* probed, usable, base, stride */
const hexkl_acc_layout *hexkl_acc_layout_get(uint8_t *b, uint32_t off) {
  (void)b;
  (void)off;
  return &g_layout;
}
int hexkl_micro_hmx_acc_clear_int32(void) {
  memset(g_acc, 0, sizeof g_acc);
  return 0;
}

/* Weight tile: 512 bytes = 1024 int4 nibbles = 32k x 32n, k-major.
   Activation tile: 64 rows x 32 bytes u8. Both sides use this same model. */
int hexkl_micro_hmx_mm_u8i4(uint8_t *base, uint32_t act_off, uint32_t w_off) {
  const uint8_t *a = base + act_off;
  const uint8_t *w = base + w_off;
  for (int r = 0; r < 64; ++r)
    for (int c = 0; c < 32; ++c) {
      int32_t s = 0;
      for (int k = 0; k < 32; ++k) {
        int idx = k * 32 + c;
        int nib = (w[idx >> 1] >> ((idx & 1) ? 4 : 0)) & 0xF;
        s += (int32_t)a[r * 32 + k] * (int32_t)(nib - 8);
      }
      g_acc[r][c] += s;
    }
  return 0;
}
int hexkl_micro_hmx_acc_read_int32(uint8_t *base, uint32_t cfg, uint32_t off) {
  (void)cfg;
  memcpy(base + off, g_acc, sizeof g_acc);
  return 0;
}

/* ---- DMA ring: completes immediately, so a push issued while the
   destination is still live shows up as a wrong result. ---- */
void hexkl_dma_ring_reset(void) {}
void hexkl_dma_ring_drain(void) {}
void hexkl_dma_ring_push2d(void *dst, const void *src, uint32_t ds, uint32_t ss,
                           uint32_t rs, uint32_t nrows, int sv, int dv) {
  (void)sv;
  (void)dv;
  for (uint32_t r = 0; r < nrows; ++r)
    memcpy((uint8_t *)dst + (size_t)r * ds,
           (const uint8_t *)src + (size_t)r * ss, rs);
}

/* ---- quant / dequant / swiglu ---- */
void hvx_quant_rows_u8_params(const float *x, uint32_t m, uint32_t mp,
                              uint32_t k, float *scale, int32_t *zp,
                              hvx_worker_pool *p) {
  (void)p;
  for (uint32_t r = 0; r < mp; ++r) {
    float lo = 0.f, hi = 0.f;
    if (r < m)
      for (uint32_t j = 0; j < k; ++j) {
        float v = x[(size_t)r * k + j];
        if (v < lo)
          lo = v;
        if (v > hi)
          hi = v;
      }
    float s = (hi - lo) / 255.f;
    if (s <= 0.f)
      s = 1e-8f;
    scale[r] = s;
    long z = lrintf(-lo / s);
    if (z < 0)
      z = 0;
    if (z > 255)
      z = 255;
    zp[r] = (int32_t)z;
  }
}
int hvx_quant_pack_u8_ah(const float *x, uint32_t m, uint32_t mp, uint32_t k,
                         const float *scale, const int32_t *zp, uint8_t *out,
                         hvx_worker_pool *p) {
  (void)p;
  (void)mp;
  const uint32_t kt_n = k / 32u;
  memset(out, 0, (size_t)64 * k);
  for (uint32_t r = 0; r < m; ++r)
    for (uint32_t kt = 0; kt < kt_n; ++kt)
      for (uint32_t j = 0; j < 32; ++j) {
        long q = lrintf(x[(size_t)r * k + kt * 32 + j] / scale[r]) + zp[r];
        if (q < 0)
          q = 0;
        if (q > 255)
          q = 255;
        out[(size_t)kt * 2048 + r * 32 + j] = (uint8_t)q;
      }
  return 0;
}
void hvx_dequant_acc_tile_to_f32(const int32_t *tile, uint32_t stride,
                                 uint32_t m, const float *as, const int32_t *az,
                                 const int32_t *cs, const float *ws,
                                 const float *bias, float *out,
                                 uint32_t ostride, int accumulate) {
  for (uint32_t r = 0; r < m; ++r)
    for (uint32_t c = 0; c < 32; ++c) {
      float v = ((float)(tile[(size_t)r * stride + c] - az[r] * cs[c])) *
                  as[r] * ws[c] +
                bias[c];
      if (accumulate)
        out[(size_t)r * ostride + c] += v;
      else
        out[(size_t)r * ostride + c] = v;
    }
}
void hvx_swiglu_inplace_f32(float *gate, const float *up, uint32_t m,
                            uint32_t n, hvx_worker_pool *p) {
  (void)p;
  for (uint32_t r = 0; r < m; ++r)
    for (uint32_t j = 0; j < n; ++j) {
      float g = gate[(size_t)r * n + j];
      gate[(size_t)r * n + j] = g / (1.f + expf(-g)) * up[(size_t)r * n + j];
    }
}
uint64_t hexkl_probe_us[HEXKL_PROBE_N];

/* ---------------- reference: one expert, one row, at a time -------------- */
typedef struct {
  uint32_t K, N;
  int8_t *nib;
  float *ws;
  int32_t *cs;
  float *bias;
} W;

static void ref_mm(const W *w, const uint8_t *a_u8, float a_scale, int32_t a_zp,
                   float *out) {
  const uint32_t kt_n = w->K / 32u, nt_n = w->N / 32u;
  for (uint32_t nt = 0; nt < nt_n; ++nt)
    for (uint32_t c = 0; c < 32; ++c) {
      int32_t s = 0;
      for (uint32_t kt = 0; kt < kt_n; ++kt)
        for (uint32_t k = 0; k < 32; ++k) {
          int idx = (int)((kt * nt_n + nt) * 1024 + k * 32 + c);
          int nib = (w->nib[idx >> 1] >> ((idx & 1) ? 4 : 0)) & 0xF;
          s += (int32_t)a_u8[kt * 32 + k] * (int32_t)(nib - 8);
        }
      uint32_t col = nt * 32 + c;
      out[col] =
        ((float)(s - a_zp * w->cs[col])) * a_scale * w->ws[col] + w->bias[col];
    }
}
static void quant_row(const float *x, uint32_t k, uint8_t *q, float *scale,
                      int32_t *zp) {
  float lo = 0.f, hi = 0.f;
  for (uint32_t j = 0; j < k; ++j) {
    if (x[j] < lo)
      lo = x[j];
    if (x[j] > hi)
      hi = x[j];
  }
  float s = (hi - lo) / 255.f;
  if (s <= 0.f)
    s = 1e-8f;
  *scale = s;
  long z = lrintf(-lo / s);
  if (z < 0)
    z = 0;
  if (z > 255)
    z = 255;
  *zp = (int32_t)z;
  for (uint32_t j = 0; j < k; ++j) {
    long v = lrintf(x[j] / s) + *zp;
    if (v < 0)
      v = 0;
    if (v > 255)
      v = 255;
    q[j] = (uint8_t)v;
  }
}

/* ------------------------------- the test ------------------------------- */
static uint32_t rnd_state = 12345u;
static uint32_t rnd(void) {
  rnd_state = rnd_state * 1664525u + 1013904223u;
  return rnd_state >> 8;
}
static float rndf(void) { return (float)rnd() / 8388608.0f - 1.0f; }

static hexkl_weight_u8i4_table g_tbl;

static void make_weight(uint32_t slot, uint32_t K, uint32_t N, W *w) {
  const uint32_t tiles = (K / 32u) * (N / 32u);
  w->K = K;
  w->N = N;
  w->nib = (int8_t *)malloc(tiles * 512u);
  w->ws = (float *)malloc(sizeof(float) * N);
  w->cs = (int32_t *)malloc(sizeof(int32_t) * N);
  w->bias = (float *)malloc(sizeof(float) * N);
  for (uint32_t i = 0; i < tiles * 512u; ++i)
    w->nib[i] = (int8_t)(rnd() & 0xFF);
  for (uint32_t c = 0; c < N; ++c) {
    w->ws[c] = 0.01f + 0.001f * (float)(rnd() % 100u);
    w->cs[c] = 0; /* colsum folded into the model: kept 0 so both sides agree */
    w->bias[c] = 0.05f * rndf();
  }
  hexkl_weight_u8i4 *s = &g_tbl.slots[slot];
  s->in_use = 1;
  s->K = K;
  s->N = N;
  s->wh_bytes = (uint8_t *)w->nib;
  s->w_scale = w->ws;
  s->colsum_w = w->cs;
  s->bias = w->bias;
}

int main(void) {
  const uint32_t M = 37, K = 64, inter = 32, N_out = 64, NE = 5;
  static uint8_t vtcm[8u << 20];

  hexkl_moe_layout L;
  int rc = hexkl_mm_u8i4_moe_layout(K, inter, N_out, sizeof vtcm, &L);
  printf(
    "layout rc=%d total=%u (act %u gu %u dn %u gate %u up %u mid %u res %u)\n",
    rc, L.total, L.act_off, L.w_gu_off, L.w_dn_off, L.gate_off, L.up_off,
    L.mid_off, L.result_off);
  if (rc)
    return 1;

  W wg[8], wd[8];
  for (uint32_t e = 0; e < NE; ++e) {
    make_weight(e, K, 2 * inter, &wg[e]);
    make_weight(NE + e, inter, N_out, &wd[e]);
  }
  uint32_t hg[8], hd[8];
  for (uint32_t e = 0; e < NE; ++e) {
    hg[e] = e;
    hd[e] = NE + e;
  }

  float *act = (float *)malloc(sizeof(float) * M * K);
  for (uint32_t i = 0; i < M * K; ++i)
    act[i] = rndf();

  /* Routing: expert 0 gets many rows (multi-block), expert 2 gets none,
     rows repeat across experts the way top-k does. */
  uint32_t rc_[8] = {70, 12, 0, 25, 9};
  uint32_t n_rows = 0;
  for (uint32_t e = 0; e < NE; ++e)
    n_rows += rc_[e];
  uint32_t *ridx = (uint32_t *)malloc(sizeof(uint32_t) * n_rows);
  float *rw = (float *)malloc(sizeof(float) * n_rows);
  for (uint32_t i = 0; i < n_rows; ++i) {
    ridx[i] = rnd() % M;
    rw[i] = 0.1f + 0.9f * ((float)(rnd() % 100u) / 100.f);
  }

  float *got = (float *)malloc(sizeof(float) * M * N_out);
  rc = hexkl_mm_u8i4_moe_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, K,
                                   inter, N_out, NE, hg, hd, ridx, rc_, rw, act,
                                   got, NULL);
  printf("run rc=%d  n_rows=%u\n", rc, n_rows);
  if (rc)
    return 1;

  /* reference */
  float *want = (float *)calloc(M * N_out, sizeof(float));
  uint8_t *aq = (uint8_t *)malloc(K);
  uint8_t *mq = (uint8_t *)malloc(inter);
  float *gu = (float *)malloc(sizeof(float) * 2 * inter);
  float *dn = (float *)malloc(sizeof(float) * N_out);
  float *mid = (float *)malloc(sizeof(float) * inter);
  uint32_t base = 0;
  for (uint32_t e = 0; e < NE; ++e) {
    for (uint32_t i = 0; i < rc_[e]; ++i) {
      uint32_t row = ridx[base + i];
      float as;
      int32_t az;
      quant_row(act + (size_t)row * K, K, aq, &as, &az);
      ref_mm(&wg[e], aq, as, az, gu);
      for (uint32_t j = 0; j < inter; ++j)
        mid[j] = gu[j] / (1.f + expf(-gu[j])) * gu[inter + j];
      float ms;
      int32_t mz;
      quant_row(mid, inter, mq, &ms, &mz);
      ref_mm(&wd[e], mq, ms, mz, dn);
      for (uint32_t c = 0; c < N_out; ++c)
        want[(size_t)row * N_out + c] += dn[c] * rw[base + i];
    }
    base += rc_[e];
  }

  double worst = 0.0;
  uint32_t bad = 0;
  for (uint32_t i = 0; i < M * N_out; ++i) {
    double d = fabs((double)got[i] - (double)want[i]);
    double s = fabs((double)want[i]) + 1e-6;
    if (d / s > 1e-5) {
      ++bad;
    }
    if (d / s > worst)
      worst = d / s;
  }
  printf("mismatches=%u of %u   worst_rel=%g\n", bad, M * N_out, worst);
  printf(bad == 0 ? "MOE KERNEL MATCHES REFERENCE\n" : "MOE KERNEL DIFFERS\n");
  int fail = (bad != 0);

  /* --- edge cases the routing can actually produce --------------------- */
  {
    uint32_t z[8] = {0, 0, 0, 0, 0};
    memset(got, 0xA5, sizeof(float) * M * N_out);
    int r = hexkl_mm_u8i4_moe_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm,
                                        M, K, inter, N_out, NE, hg, hd, ridx, z,
                                        rw, act, got, NULL);
    int ok = (r == 0);
    for (uint32_t i = 0; i < M * N_out; ++i) {
      if (got[i] != 0.f) {
        ok = 0;
        break;
      }
    }
    printf("all-empty routing : %s\n", ok ? "zeroed, rc=0" : "WRONG");
    fail |= !ok;
  }
  {
    uint32_t c64[8] = {64, 0, 0, 0, 0};
    int r = hexkl_mm_u8i4_moe_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm,
                                        M, K, inter, N_out, NE, hg, hd, ridx,
                                        c64, rw, act, got, NULL);
    printf("exactly 64 rows   : rc=%d\n", r);
    fail |= (r != 0);
  }
  {
    uint32_t c1[8] = {1, 0, 0, 0, 0};
    uint32_t bad_row[1] = {M};
    int r = hexkl_mm_u8i4_moe_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm,
                                        M, K, inter, N_out, NE, hg, hd, bad_row,
                                        c1, rw, act, got, NULL);
    printf("row_index >= M    : rc=%d (want %d)\n", r, AEE_EBADPARM);
    fail |= (r != AEE_EBADPARM);
  }
  {
    hexkl_moe_layout R;
    int r = hexkl_mm_u8i4_moe_layout(2048, 1792, 2048, 8300u * 1024u, &R);
    printf("LFM2 shapes       : rc=%d total=%.2f MB (doc 46 says 6.37)\n", r,
           R.total / 1048576.0);
    fail |= (r != 0);
    r = hexkl_mm_u8i4_moe_layout(2048, 1792, 2048, 4u << 20, &R);
    printf("4 MB arena        : rc=%d (want %d)\n", r, AEE_ENOMEMORY);
    fail |= (r != AEE_ENOMEMORY);
  }
  printf(fail ? "\nFAIL\n" : "\nALL CHECKS PASS\n");
  return fail;
}
