/* Scalar stand-ins shared by the block kernels' host checks. Pulled out of
   moe_layer_host_check.c unchanged when the conv block check arrived; the
   comments are that file's. */
#include "hvx_scalar_stubs.h"

#include "hvx_scalar.h"

#include "hexkl_acc_tile.h"
#include "hexkl_dma_ring.h"
#include "hexkl_mm_u8i8_dma.h"
#include "hexkl_probe.h"
#include "hvx_conv_gate_f32.h"
#include "hvx_gather_ah_u8.h"
#include "hvx_rmsnorm_rows_f32.h"
#include "hvx_rope_rows_f32.h"
#include "hvx_router_rows_f32.h"
#include "hvx_scale_add_f32.h"
#include <AEEStdErr.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- acc layout: the tile stand-in is 64 x 32 int32 at stride 32 ---- */
static hexkl_acc_layout g_layout = {1, 1, 0,
                                    32}; /* probed, usable, base, stride */
const hexkl_acc_layout *hexkl_acc_layout_get(uint8_t *b, uint32_t off) {
  (void)b;
  (void)off;
  return &g_layout;
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
int hexkl_dma_ring_is_done(uint32_t idx) {
  (void)idx;
  return 1;
}
void hexkl_dcache_inva(const void *p, uint32_t bytes) {
  (void)p;
  (void)bytes;
}
void hexkl_dma_ring_push2d(void *dst, const void *src, uint32_t ds, uint32_t ss,
                           uint32_t rs, uint32_t nrows, int sv, int dv) {
  (void)sv;
  (void)dv;
  for (uint32_t r = 0; r < nrows; ++r)
    memcpy((uint8_t *)dst + (size_t)r * ds,
           (const uint8_t *)src + (size_t)r * ss, rs);
}
/* #177's lane queue, likewise complete at once (the MoE file links here;
   moe_layer_host_check holds its schedule). */
void hexkl_dma_lane_push2d(hexkl_dma_desc2d *d, hexkl_dma_desc2d *prev,
                           void *dst, const void *src, uint32_t ds, uint32_t ss,
                           uint32_t rs, uint32_t nrows, int sv, int dv) {
  (void)d;
  (void)prev;
  hexkl_dma_ring_push2d(dst, src, ds, ss, rs, nrows, sv, dv);
}
int hexkl_dma_lane_wait(hexkl_dma_desc2d *d) {
  (void)d;
  return 0;
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
/* No workers: the MoE feed's GU(0) job (#185) takes its run fallback. */
uint32_t hvx_worker_pool_workers(const hvx_worker_pool *pool) {
  (void)pool;
  return 0u;
}
/* The background lane, likewise: every unit runs at submit, in order, and
   the waits find them done. What this checks is that the kernel waits for
   the right block before it queues it -- a wait for too few units cannot
   show here, and a wait for too many only as a hang on device. */
void hvx_worker_pool_submit_bg(hvx_worker_pool *pool, hvx_bg_job *job) {
  (void)pool;
  for (uint32_t u = 0; u < job->n_units; ++u) {
    job->func(job->n_units, u, job->ctx);
    job->done[u] = 1;
  }
}
void hvx_worker_pool_wait_bg(hvx_worker_pool *pool, hvx_bg_job *job,
                             uint32_t n) {
  (void)pool;
  (void)job;
  (void)n;
}

void hvx_copy_ah_block(uint8_t *dst, const uint8_t *src, uint32_t k,
                       hvx_worker_pool *pool) {
  (void)pool;
  memcpy(dst, src, (size_t)(k / 32u) * 2048u);
}

/* The row RMSNorm the *_norm layer entries fold in: plain f32, one row
   at a time, the kernel's own operation order (sum of squares, sqrtf,
   scale, gamma). What the checks that link the skel verify is the entry's
   plumbing -- which rows, which gamma, in place -- not HVX arithmetic,
   which rmsnorm_rows_host_check covers on the lane emulation. */
int hvx_rmsnorm_rows_f32(const float *x, float *y, uint32_t M, uint32_t n,
                         uint32_t chunk, const float *gamma, float eps,
                         hvx_worker_pool *pool) {
  (void)pool;
  if (!x || !y || M == 0u || chunk == 0u || chunk % 32u != 0u ||
      n % chunk != 0u) {
    return -1;
  }
  for (uint32_t r = 0; r < M; ++r) {
    for (uint32_t c = 0; c < n; c += chunk) {
      const float *xr = x + (size_t)r * n + c;
      float *yr = y + (size_t)r * n + c;
      float ss = 0.0f;
      for (uint32_t j = 0; j < chunk; ++j)
        ss += xr[j] * xr[j];
      const float rs = 1.0f / sqrtf(ss / (float)chunk + eps);
      for (uint32_t j = 0; j < chunk; ++j)
        yr[j] = xr[j] * rs * (gamma ? gamma[j] : 1.0f);
    }
  }
  return 0;
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
uint64_t hexkl_probe_us[HEXKL_PROBE_N];
/* On, so the counting probes (blocks, DMA bytes) run here too -- this
   harness is where a miscounted block or a push that never happens shows up
   without a device. The timers read the stub clock and are not checked. */
int hexkl_probe_on = 1;

/* Scalar stand-in for the conv gate: (w0*g[t] + w1*g[t-1]) + w2*g[t-2],
   each product and sum its own statement so the host compiler cannot
   contract them into the FMA the HVX does not have. */
void hvx_conv_gate_f32(float *z, uint32_t z_stride, const float *g,
                       uint32_t g_stride, uint32_t t0, uint32_t m_count,
                       uint32_t C, const float *conv_w, hvx_worker_pool *pool) {
  (void)pool;
  for (uint32_t r = 0; r < m_count; ++r) {
    const uint32_t t = t0 + r;
    for (uint32_t c = 0; c < C; ++c) {
      const float x0 = g[(size_t)t * g_stride + c];
      const float x1 = t >= 1u ? g[(size_t)(t - 1u) * g_stride + c] : 0.f;
      const float x2 = t >= 2u ? g[(size_t)(t - 2u) * g_stride + c] : 0.f;
      volatile float p0 = conv_w[c] * x0;
      volatile float p1 = conv_w[C + c] * x1;
      volatile float p2 = conv_w[2u * C + c] * x2;
      volatile float y = p0 + p1;
      y = y + p2;
      z[(size_t)r * z_stride + c] = z[(size_t)r * z_stride + c] * y;
    }
  }
}

/* ---------------- reference helpers -------------------------------------- */
void ref_mm(const W *w, const uint8_t *a_u8, float a_scale, int32_t a_zp,
            float *out) {
  const uint32_t kt_n = w->K / 32u, nt_n = w->N / 32u;
  for (uint32_t nt = 0; nt < nt_n; ++nt)
    for (uint32_t c = 0; c < 32; ++c) {
      int32_t s = 0;
      for (uint32_t kt = 0; kt < kt_n; ++kt)
        for (uint32_t k = 0; k < 32; ++k)
          s +=
            (int32_t)a_u8[kt * 32 + k] *
            wh_value((const uint8_t *)w->nib + (size_t)(kt * nt_n + nt) * 512u,
                     k, c);
      uint32_t col = nt * 32 + c;
      out[col] =
        ((float)(s - a_zp * w->cs[col])) * a_scale * w->ws[col] + w->bias[col];
    }
}
void quant_row(const float *x, uint32_t k, uint8_t *q, float *scale,
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
uint32_t rnd(void) {
  rnd_state = rnd_state * 1664525u + 1013904223u;
  return rnd_state >> 8;
}
float rndf(void) { return (float)rnd() / 8388608.0f - 1.0f; }

hexkl_weight_u8i4_table g_tbl;

void make_weight(uint32_t slot, uint32_t K, uint32_t N, W *w) {
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

/* Link-only stand-ins for the paths of hexkl_mm_u8i4_dma.c no host check
   takes: the registration bake (make_weight fills the table directly) and
   the DDR fallback's accumulator copy (the stub layout is always usable).
   Reaching one is a harness bug, not a kernel result. */
int hexkl_micro_hmx_rm_to_wh_i4(uint8_t *b, uint32_t off, const int8_t *rm,
                                uint32_t tr, uint32_t tc, uint32_t N) {
  (void)b, (void)off, (void)rm, (void)tr, (void)tc, (void)N;
  abort();
}
int hexkl_micro_hmx_copy_32b_to_submatrix(uint8_t *b, uint32_t off,
                                          int32_t *dst, uint32_t rb,
                                          uint32_t nt, uint32_t m_pad,
                                          uint32_t N) {
  (void)b, (void)off, (void)dst, (void)rb, (void)nt, (void)m_pad, (void)N;
  abort();
}

/* The router logits the router_logits_f32 entry runs: plain f32 dots,
   row by row. The entry's plumbing (the norm, which rows, where the
   logits land) is what a skel-linking check verifies; the HVX arithmetic
   is router_rows_host_check's on the lane emulation. */
int hvx_router_rows_f32(const float *x, const float *w, float *logits,
                        uint32_t M, uint32_t K, uint32_t E,
                        hvx_worker_pool *pool) {
  (void)pool;
  if (!x || !w || !logits || M == 0u || K == 0u || E == 0u || E % 32u != 0u ||
      E > 128u) {
    return -1;
  }
  for (uint32_t r = 0; r < M; ++r) {
    for (uint32_t e = 0; e < E; ++e) {
      float acc = 0.0f;
      for (uint32_t k = 0; k < K; ++k)
        acc += x[(size_t)r * K + k] * w[(size_t)k * E + e];
      logits[(size_t)r * E + e] = acc;
    }
  }
  return 0;
}

/* The fused-norm router entry: the scalar RMSNorm (hvx_rmsnorm_row_f32's
   formula), then the dots above. */
int hvx_router_rows_norm_f32(const float *x, const float *gamma, float eps,
                             const float *w, float *logits, uint32_t M,
                             uint32_t K, uint32_t E, hvx_worker_pool *pool) {
  if (!gamma)
    return hvx_router_rows_f32(x, w, logits, M, K, E, pool);
  if (!x || M == 0u || K == 0u || K % 32u != 0u)
    return -1;
  float *n = (float *)malloc(sizeof(float) * (size_t)M * K);
  if (!n)
    return -1;
  for (uint32_t r = 0; r < M; ++r) {
    const float *xr = x + (size_t)r * K;
    float ss = 0.0f;
    for (uint32_t k = 0; k < K; ++k)
      ss += xr[k] * xr[k];
    const float rs = 1.0f / sqrtf(ss / (float)K + eps);
    for (uint32_t k = 0; k < K; ++k)
      n[(size_t)r * K + k] = xr[k] * rs * gamma[k];
  }
  const int rc = hvx_router_rows_f32(n, w, logits, M, K, E, pool);
  free(n);
  return rc;
}

/* The router's selection (hvx_router_topk_rows_f32): plain f32 softmax
   and the same first-maximum scan; the HVX softmax is
   router_rows_host_check's. */
int hvx_router_topk_rows_f32(float *p, const float *scale, uint32_t *sel,
                             float *weight, uint32_t M, uint32_t E,
                             uint32_t top_k, uint32_t n_sel,
                             hvx_worker_pool *pool) {
  (void)pool;
  if (!p || !scale || !sel || !weight || M == 0u || E == 0u || E % 32u != 0u ||
      E > 128u || top_k == 0u || n_sel < top_k || n_sel > E) {
    return -1;
  }
  for (uint32_t r = 0; r < M; ++r) {
    float *pr = p + (size_t)r * E;
    float mx = pr[0], sum = 0.0f;
    for (uint32_t e = 1; e < E; ++e)
      mx = pr[e] > mx ? pr[e] : mx;
    for (uint32_t e = 0; e < E; ++e)
      sum += pr[e] = expf(pr[e] - mx);
    for (uint32_t e = 0; e < E; ++e)
      pr[e] /= sum;
    uint32_t taken[4] = {0u, 0u, 0u, 0u};
    for (uint32_t k = 0; k < n_sel; ++k) {
      uint32_t best = E;
      for (uint32_t e = 0; e < E; ++e)
        if (!(taken[e >> 5] & (1u << (e & 31u))) &&
            (best == E || pr[e] > pr[best]))
          best = e;
      taken[best >> 5] |= 1u << (best & 31u);
      sel[(size_t)r * n_sel + k] = best;
    }
    float wsum = 0.0f;
    for (uint32_t k = 0; k < top_k; ++k)
      wsum += pr[sel[(size_t)r * n_sel + k]];
    for (uint32_t k = 0; k < top_k; ++k) {
      const uint32_t e = sel[(size_t)r * n_sel + k];
      weight[(size_t)r * top_k + k] = pr[e] * (1.0f / wsum) * scale[e];
    }
  }
  return 0;
}

/* The RoPE the *_norm layer entry folds in: the CPU kernel's pair order,
   row by row; the HVX arithmetic is rope_rows_host_check's. */
int hvx_rope_rows_f32(float *x, uint32_t M, uint32_t n, uint32_t hd,
                      const float *cs, hvx_worker_pool *pool) {
  (void)pool;
  if (!x || !cs || M == 0u || hd == 0u || (hd / 2u) % 32u != 0u || n == 0u ||
      n % hd != 0u) {
    return -1;
  }
  const uint32_t half = hd / 2u;
  for (uint32_t r = 0; r < M; ++r) {
    const float *c = cs + (size_t)r * 2u * hd, *s = c + hd;
    for (uint32_t h = 0; h < n / hd; ++h) {
      float *head = x + (size_t)r * n + (size_t)h * hd;
      for (uint32_t j = 0; j < half; ++j) {
        const float a = head[j], b = head[j + half];
        head[j] = a * c[j] - b * s[j];
        head[j + half] = a * s[j] + b * c[j];
      }
    }
  }
  return 0;
}

/* [#260 r3] #4415 @ 2db6cad59's skel entries (row-strided norm / RoPE,
   the post-attention and post-FFN epilogues, the u8 x i8 router layer):
   plain f32 for the checks that link the skel; their HVX arithmetic is
   the rows host checks'. */
int hvx_rmsnorm_rows_ld_f32(const float *x, float *y, uint32_t M, uint32_t n,
                            uint32_t ld, uint32_t chunk, const float *gamma,
                            float eps, hvx_worker_pool *pool) {
  if (ld < n)
    return -1;
  for (uint32_t r = 0; r < M; ++r)
    if (hvx_rmsnorm_rows_f32(x + (size_t)r * ld, y + (size_t)r * ld, 1u, n,
                             chunk, gamma, eps, pool) != 0)
      return -1;
  return 0;
}

int hvx_rope_rows_ld_f32(float *x, uint32_t M, uint32_t n, uint32_t ld,
                         uint32_t hd, const float *cs, hvx_worker_pool *pool) {
  if (ld < n)
    return -1;
  for (uint32_t r = 0; r < M; ++r)
    if (hvx_rope_rows_f32(x + (size_t)r * ld, 1u, n, hd,
                          cs + (size_t)r * 2u * hd, pool) != 0)
      return -1;
  return 0;
}

/* out = scale * (res + rmsnorm(x [+ x2]) * gamma), one row at a time */
static void norm_add_row(float *out, const float *res, const float *x,
                         const float *x2, uint32_t n, const float *gamma,
                         float eps, float scale) {
  float ss = 0.0f;
  for (uint32_t j = 0; j < n; ++j) {
    const float v = x2 ? x[j] + x2[j] : x[j];
    ss += v * v;
  }
  const float rs = 1.0f / sqrtf(ss / (float)n + eps);
  for (uint32_t j = 0; j < n; ++j) {
    const float v = x2 ? x[j] + x2[j] : x[j];
    out[j] = scale * (res[j] + v * rs * (gamma ? gamma[j] : 1.0f));
  }
}

int hvx_rmsnorm_add_res_f32(float *out, const float *res, const float *x,
                            const float *x2, uint32_t M, uint32_t n,
                            const float *gamma, float eps, float scale,
                            hvx_worker_pool *pool) {
  (void)pool;
  if (!out || !x || M == 0u || n % 32u != 0u)
    return -1;
  for (uint32_t r = 0; r < M; ++r)
    norm_add_row(out + (size_t)r * n, (res ? res : out) + (size_t)r * n,
                 x + (size_t)r * n, x2 ? x2 + (size_t)r * n : NULL, n, gamma,
                 eps, scale);
  return 0;
}

int hvx_rmsnorm_add_blocks_f32(float *out, const float *res,
                               const float *const *xb, const uint32_t *cols,
                               uint32_t n_blk, uint32_t M, const float *gamma,
                               float eps, float scale, hvx_worker_pool *pool) {
  (void)pool;
  uint32_t n = 0;
  for (uint32_t b = 0; b < n_blk; ++b)
    n += cols[b];
  float *row = (float *)malloc(sizeof(float) * n);
  if (!row)
    return -1;
  for (uint32_t r = 0; r < M; ++r) {
    for (uint32_t b = 0, c0 = 0; b < n_blk; c0 += cols[b++])
      memcpy(row + c0, xb[b] + (size_t)r * cols[b], sizeof(float) * cols[b]);
    norm_add_row(out + (size_t)r * n, (res ? res : out) + (size_t)r * n, row,
                 NULL, n, gamma, eps, scale);
  }
  free(row);
  return 0;
}

int hexkl_mm_u8i8_layer_run(hexkl_weight_u8i8_table *tbl, uint8_t *vtcm_base,
                            uint32_t vtcm_size, uint32_t config_off, uint32_t M,
                            uint32_t K, const uint32_t *handles,
                            uint32_t n_handles, const float *act_f32,
                            float *out_cat, const hexkl_mm_opts *opts) {
  (void)tbl, (void)vtcm_base, (void)vtcm_size, (void)config_off, (void)M,
    (void)K, (void)handles, (void)n_handles, (void)act_f32, (void)out_cat,
    (void)opts;
  return AEE_EUNSUPPORTED;
}
