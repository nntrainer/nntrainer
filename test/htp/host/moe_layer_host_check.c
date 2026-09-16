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
#include "hvx_dequant_i32.h"
#include "hvx_gather_ah_u8.h"
#include "hvx_scale_add_f32.h"
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
/* Indices are handed out and waited on for real shape, but every transfer
   has already completed by the time push2d returns, so a wait is a no-op.
   That is the point: a chunk consumed before its push would read stale
   bytes on device and read correct ones here, so what this harness checks
   is that every chunk IS pushed and that the offsets line up. */
static uint32_t g_stub_idx;
uint32_t hexkl_dma_ring_next_idx(void) { return g_stub_idx++; }
void hexkl_dma_ring_wait(uint32_t idx) { (void)idx; }
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
int hvx_quant_pack_u8_ah_mapped(const float *x, const uint32_t *map,
                                uint32_t m, uint32_t mp, uint32_t k,
                                const float *scale, const int32_t *zp,
                                uint8_t *out, hvx_worker_pool *p) {
  (void)p;
  /* Tiles run (row_block, inner_tile) at a 2048-byte stride, so a caller
     passing more than 64 rows writes several row blocks. The kernel now
     packs the whole activation in one call, so this can no longer assume
     one block the way it did. */
  const uint32_t kt_n = k / 32u;
  memset(out, 0, (size_t)mp * k);
  for (uint32_t r = 0; r < m; ++r)
    for (uint32_t kt = 0; kt < kt_n; ++kt)
      for (uint32_t j = 0; j < 32; ++j) {
        const size_t sr = map ? map[r] : r;
        long q = lrintf(x[sr * k + kt * 32 + j] / scale[r]) + zp[r];
        if (q < 0)
          q = 0;
        if (q > 255)
          q = 255;
        out[(size_t)(r / 64u) * kt_n * 2048u + (size_t)kt * 2048u +
            (size_t)(r % 64u) * 32u + j] = (uint8_t)q;
      }
  return 0;
}

int hvx_quant_pack_u8_ah(const float *x, uint32_t m, uint32_t mp, uint32_t k,
                         const float *scale, const int32_t *zp, uint8_t *out,
                         hvx_worker_pool *p) {
  return hvx_quant_pack_u8_ah_mapped(x, NULL, m, mp, k, scale, zp, out, p);
}
/* One block through the mapped stand-in: the pack of rows [rb*64, +64) is
   the same arithmetic whichever entry lands it, so the block entry is
   defined by the whole-buffer one rather than repeated -- a second copy of
   the formula would let the two drift and hide it. Offsets shift so the
   mapped stand-in sees a one-block problem. */
void hvx_quant_pack_u8_ah_block(const float *x, const uint32_t *map,
                                uint32_t rb, uint32_t k, const float *scale,
                                const int32_t *zp, uint8_t *out) {
  const uint32_t kt_n = k / 32u;
  hvx_quant_pack_u8_ah_mapped(x, map + (size_t)rb * 64u, 64u, 64u, k,
                              scale + (size_t)rb * 64u, zp + (size_t)rb * 64u,
                              out + (size_t)rb * kt_n * 2048u, NULL);
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
/* The pool runs everything on the caller, which is what its own NULL path
   does for n_units <= 1. Doing it here rather than passing NULL keeps the
   kernel's call sites exercised: the range arithmetic they hand the worker
   is part of what this check is for. */
void hvx_worker_pool_run(hvx_worker_pool *pool, hvx_worker_pool_func func,
                         void *ctx, uint32_t n_units) {
  (void)pool;
  /* One slice covering everything. Writing this the way the real function's
     degenerate branch used to be written -- func(n_units, 0, ctx) -- is what
     this check caught first time out: that form means "worker 0 of n_units"
     and does 1/n_units of the work. */
  func(1u, 0, ctx);
}
/* submit runs the job to completion on the spot and wait is a no-op: the
   harness cannot exercise the overlap, only that every job is submitted
   with the right buffer and retired before that buffer is reused -- which
   a job that ran late would show as a wrong result on device and cannot
   show here. The device tests are where the ordering is checked. */
void hvx_worker_pool_submit(hvx_worker_pool *pool, hvx_worker_pool_func func,
                            void *ctx, uint32_t n_units) {
  (void)pool;
  if (n_units != 0u)
    func(1u, 0, ctx);
}
void hvx_worker_pool_wait(hvx_worker_pool *pool) { (void)pool; }
/* The background lane, likewise: every unit runs at submit, in order, and
   the waits find them done. What this checks is that the kernel waits for
   the right block before it queues it -- a wait for too few units cannot
   show here, and a wait for too many only as a hang on device. */
void hvx_worker_pool_submit_bg(hvx_worker_pool *pool, hvx_worker_pool_func func,
                               void *ctx, uint32_t n_units, uint8_t *done) {
  (void)pool;
  for (uint32_t u = 0; u < n_units; ++u) {
    func(n_units, u, ctx);
    done[u] = 1;
  }
}
void hvx_worker_pool_wait_bg(hvx_worker_pool *pool, uint32_t n) {
  (void)pool;
  (void)n;
}

void hvx_copy_ah_block(uint8_t *dst, const uint8_t *src, uint32_t k,
                       hvx_worker_pool *pool) {
  (void)pool;
  memcpy(dst, src, (size_t)(k / 32u) * 2048u);
}

void hvx_scale_add_rows_f32(float *dst, const float *src, float scale,
                            uint32_t n) {
  /* Two operations, never one: HVX has no f32 fused multiply-add and the
     ARM path this has to agree with applies the routing weight with
     multiply_i and then accumulates with add_i. The volatile is what stops
     the host compiler contracting them and making this stub disagree with
     the kernel for a reason that is the stub's. */
  for (uint32_t i = 0; i < n; ++i) {
    volatile float p = src[i] * scale;
    dst[i] = dst[i] + p;
  }
}
/* Scalar stand-in for the fused gate/up dequant + SwiGLU job. Same policy
   as the batch dequant stand-in below: a loop over the per-tile stand-in,
   so what is checked is the kernel's pairing arithmetic -- that staged
   slot j is gate column g0+j and slot n_pairs+j the up column opposite it
   -- not the arithmetic of SwiGLU, which the device gates cover bit for
   bit. */
void hvx_dq_swiglu_worker(uint32_t n_threads, uint32_t i, void *vjob) {
  const hvx_dq_swiglu_job *c = (const hvx_dq_swiglu_job *)vjob;
  (void)n_threads;
  (void)i;
  float gt[64 * 32], ut[64 * 32];
  for (uint32_t j = 0; j < c->n_pairs; ++j) {
    const uint32_t cg = (c->g0 + j) * 32u, cu = c->inter + cg;
    hvx_dequant_acc_tile_to_f32(
      (const int32_t *)(c->tiles_base + (size_t)j * c->tile_stride),
      c->row_stride, c->m_count, c->act_scale, c->act_zp, c->colsum_w + cg,
      c->w_scale + cg, c->bias + cg, gt, 32u, 0);
    hvx_dequant_acc_tile_to_f32(
      (const int32_t *)(c->tiles_base +
                        (size_t)(c->n_pairs + j) * c->tile_stride),
      c->row_stride, c->m_count, c->act_scale, c->act_zp, c->colsum_w + cu,
      c->w_scale + cu, c->bias + cu, ut, 32u, 0);
    for (uint32_t r = 0; r < c->m_count; ++r)
      for (uint32_t k = 0; k < 32u; ++k) {
        const float g = gt[r * 32u + k];
        c->dst[(size_t)r * c->dst_stride + cg + k] =
          g / (1.f + expf(-g)) * ut[r * 32u + k];
      }
  }
}
/* Scalar stand-in for the pooled batch dequant job. Deliberately a loop
   over the per-tile stand-in above, exactly as the real one is a pooled
   loop over the real per-tile kernel: what this harness can check is the
   kernel's batching arithmetic -- which tile lands at which staged slot,
   which column it carries -- and a stand-in that recomputed the dequant
   itself would check the stand-in. */
void hvx_dq_tiles_worker(uint32_t n_threads, uint32_t i, void *vjob) {
  const hvx_dq_tiles_job *c = (const hvx_dq_tiles_job *)vjob;
  (void)n_threads;
  (void)i;
  for (uint32_t j = 0; j < c->n_tiles; ++j) {
    const uint32_t c0 = (c->nt0 + j) * 32u;
    const int32_t *tile =
      (const int32_t *)(c->tiles_base + (size_t)j * c->tile_stride);
    float *out =
      (c0 < c->split) ? (c->dst_a + c0) : (c->dst_b + (c0 - c->split));
    hvx_dequant_acc_tile_to_f32(tile, c->row_stride, c->m_count, c->act_scale,
                                c->act_zp, c->colsum_w + c0, c->w_scale + c0,
                                c->bias + c0, out, c->dst_stride, 0);
  }
}

uint64_t hexkl_probe_us[HEXKL_PROBE_N];
/* On, so the counting probes (blocks, DMA bytes) run here too -- this
   harness is where a miscounted block or a push that never happens shows up
   without a device. The timers read the stub clock and are not checked. */
int hexkl_probe_on = 1;

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
    "layout rc=%d total=%u (act %u gu %u dn %u gate %u mid %u stage %u res "
    "%u)\n",
    rc, L.total, L.act_off, L.w_gu_off, L.w_dn_off, L.gate_off, L.mid_off,
    L.result_off, L.res_f32_off);
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
  /* One scratch across every call below, the way the session holds it: the
     later, smaller calls must work out of the block the first one grew. */
  hexkl_moe_scratch scratch = {NULL, NULL, 0};
  rc = hexkl_mm_u8i4_moe_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm, M, K,
                                   inter, N_out, NE, hg, hd, ridx, rc_, rw, act,
                                   got, NULL, &scratch);
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
      for (uint32_t c = 0; c < N_out; ++c) {
        /* Two operations through a volatile, matching what the kernel and
           the ARM path both do -- see hvx_scale_add_rows_f32's stub. */
        volatile float p = dn[c] * rw[base + i];
        want[(size_t)row * N_out + c] = want[(size_t)row * N_out + c] + p;
      }
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

  /* Every active expert's gate_up and down must have been pushed. The DMA
     stub here completes immediately, so a missing push cannot show up as a
     wrong result the way a premature one does -- this counter is the only
     thing that catches it, and it is also what the device profile divides
     by to get GB/s, so a wrong count would quietly misreport the bandwidth
     the whole plan is gated on. */
  {
    uint32_t active = 0;
    for (uint32_t e = 0; e < NE; ++e) {
      if (rc_[e] != 0u)
        ++active;
    }
    const uint32_t gu_kb = ((K / 32u) * ((2u * inter) / 32u) * 512u) >> 10;
    const uint32_t dn_kb = ((inter / 32u) * (N_out / 32u) * 512u) >> 10;
    const uint64_t want = (uint64_t)active * (gu_kb + dn_kb);
    const uint64_t got_kb = hexkl_probe_us[HEXKL_PROBE_DMA_KB];
    printf("weight DMA        : %llu KB over %u experts (want %llu)\n",
           (unsigned long long)got_kb, active, (unsigned long long)want);
    if (got_kb != want)
      fail = 1;
    if (hexkl_probe_us[HEXKL_PROBE_BLOCKS] == 0u)
      fail = 1;
  }

  /* --- edge cases the routing can actually produce --------------------- */
  {
    uint32_t z[8] = {0, 0, 0, 0, 0};
    memset(got, 0xA5, sizeof(float) * M * N_out);
    int r = hexkl_mm_u8i4_moe_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm,
                                        M, K, inter, N_out, NE, hg, hd, ridx, z,
                                        rw, act, got, NULL, &scratch);
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
                                        c64, rw, act, got, NULL, &scratch);
    printf("exactly 64 rows   : rc=%d\n", r);
    fail |= (r != 0);
  }
  {
    uint32_t c1[8] = {1, 0, 0, 0, 0};
    uint32_t bad_row[1] = {M};
    int r = hexkl_mm_u8i4_moe_layer_run(&g_tbl, vtcm, sizeof vtcm, sizeof vtcm,
                                        M, K, inter, N_out, NE, hg, hd, bad_row,
                                        c1, rw, act, got, NULL, &scratch);
    printf("row_index >= M    : rc=%d (want %d)\n", r, AEE_EBADPARM);
    fail |= (r != AEE_EBADPARM);
  }
  {
    hexkl_moe_layout R;
    int r = hexkl_mm_u8i4_moe_layout(2048, 1792, 2048, 8300u * 1024u, &R);
    printf("LFM2 shapes       : rc=%d total=%.2f MB (doc 47 section 11 says "
           "6.92)\n",
           r, R.total / 1048576.0);
    fail |= (r != 0);
    r = hexkl_mm_u8i4_moe_layout(2048, 1792, 2048, 4u << 20, &R);
    printf("4 MB arena        : rc=%d (want %d)\n", r, AEE_ENOMEMORY);
    fail |= (r != AEE_ENOMEMORY);
  }
  printf(fail ? "\nFAIL\n" : "\nALL CHECKS PASS\n");
  hexkl_moe_scratch_free(&scratch);
  return fail;
}
