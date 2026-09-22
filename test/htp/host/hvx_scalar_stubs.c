/* Scalar stand-ins shared by the block kernels' host checks. Pulled out of
   moe_layer_host_check.c unchanged when the conv block check arrived; the
   comments are that file's. */
#include "hvx_scalar_stubs.h"

#include "hexkl_acc_tile.h"
#include "hexkl_dma_ring.h"
#include "hexkl_probe.h"
#include "hvx_conv_gate_f32.h"
#include "hvx_dequant_i32.h"
#include "hvx_gather_ah_u8.h"
#include "hvx_gemm_u8i4_wh.h"
#include "hvx_scale_add_f32.h"
#include "hvx_swiglu_f32.h"
#include <AEEStdErr.h>
#include <math.h>
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

/* Weight tile: 512 bytes = 32k x 32n int4 in the device's WH layout
   (htp_wh_layout.h: byte (k/8)*128 + c*4 + k%4, low nibble for k%8 < 4),
   two's complement nibbles. The HMX stub, the HVX GEMM stand-in and the
   reference all read a tile through this one function, so the layout is
   the real one and the three cannot disagree on it -- the HVX GEMM's
   whole claim is that it reads the same bytes as the HMX.
   Activation tile: 64 rows x 32 bytes u8. */
int wh_value(const uint8_t *tile, uint32_t k, uint32_t c) {
  const uint32_t byte = (k / 8u) * 128u + c * 4u + (k % 4u);
  const int nib = (tile[byte] >> (((k / 4u) % 2u) ? 4 : 0)) & 0xF;
  return nib >= 8 ? nib - 16 : nib;
}
int hexkl_micro_hmx_mm_u8i4(uint8_t *base, uint32_t act_off, uint32_t w_off) {
  const uint8_t *a = base + act_off;
  const uint8_t *w = base + w_off;
  for (int r = 0; r < 64; ++r)
    for (uint32_t c = 0; c < 32; ++c) {
      int32_t s = 0;
      for (uint32_t k = 0; k < 32; ++k)
        s += (int32_t)a[r * 32 + k] * wh_value(w, k, c);
      g_acc[r][c] += s;
    }
  return 0;
}
/* The HVX GEMM's stand-in: the same sum, over the tiles the kernel points
   it at, into the row-stride-32 tile the header promises. */
void hvx_gemm_u8i4_wh_col(const uint8_t *act_ah, uint32_t m, uint32_t k_tiles,
                          const uint8_t *wh, uint32_t n_col, uint32_t nt,
                          int32_t *out) {
  for (uint32_t r = 0; r < m; ++r)
    for (uint32_t c = 0; c < 32; ++c) {
      int32_t s = 0;
      for (uint32_t kt = 0; kt < k_tiles; ++kt) {
        const uint8_t *tile = wh + ((size_t)kt * n_col + nt) * 512u;
        const uint8_t *arow = act_ah + (size_t)kt * 2048u + r * 32u;
        for (uint32_t k = 0; k < 32; ++k)
          s += (int32_t)arow[k] * wh_value(tile, k, c);
      }
      out[r * 32u + c] = s;
    }
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
int hvx_quant_pack_u8_ah_mapped(const float *x, const uint32_t *map, uint32_t m,
                                uint32_t mp, uint32_t k, const float *scale,
                                const int32_t *zp, uint8_t *out,
                                hvx_worker_pool *p) {
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
/* A row range, same scalar formula as the mapped stand-in above so the
   two cannot drift: the kernel's units are 16-row quarters of a block. */
void hvx_quant_pack_u8_ah_rows(const float *x, const uint32_t *map, uint32_t m0,
                               uint32_t m1, uint32_t k, const float *scale,
                               const int32_t *zp, uint8_t *out) {
  const uint32_t kt_n = k / 32u;
  for (uint32_t r = m0; r < m1; ++r)
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
/* The tail path calls the pooled pair function directly (pool NULL, one
   pair); it is the job above run synchronously, as on device. */
void hvx_dequant_swiglu_acc_tiles_to_f32(
  const uint8_t *tiles_base, uint32_t tile_stride, uint32_t n_pairs,
  uint32_t g0, uint32_t row_stride, uint32_t m_count, const float *act_scale,
  const int32_t *act_zp, const int32_t *colsum_w, const float *w_scale,
  const float *bias, uint32_t inter, float *dst, uint32_t dst_stride,
  hvx_worker_pool *pool) {
  (void)pool;
  hvx_dq_swiglu_job jb;
  jb.tiles_base = tiles_base;
  jb.tile_stride = tile_stride;
  jb.n_pairs = n_pairs;
  jb.g0 = g0;
  jb.row_stride = row_stride;
  jb.m_count = m_count;
  jb.act_scale = act_scale;
  jb.act_zp = act_zp;
  jb.colsum_w = colsum_w;
  jb.w_scale = w_scale;
  jb.bias = bias;
  jb.inter = inter;
  jb.dst = dst;
  jb.dst_stride = dst_stride;
  hvx_dq_swiglu_worker(1u, 0u, &jb);
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

/* Scalar stand-in for the dequant + product pair job (the conv block's
   pre-conv gate): slot j is the first weight's column c0 + 32 j, slot
   n_pairs + j the second's -- the pairing arithmetic, like the SwiGLU
   one above. */
void hvx_dq_mul_worker(uint32_t n_threads, uint32_t i, void *vjob) {
  const hvx_dq_mul_job *c = (const hvx_dq_mul_job *)vjob;
  (void)n_threads;
  (void)i;
  float at[64 * 32], bt[64 * 32];
  for (uint32_t j = 0; j < c->n_pairs; ++j) {
    const uint32_t col = c->c0 + j * 32u;
    hvx_dequant_acc_tile_to_f32(
      (const int32_t *)(c->tiles_base + (size_t)j * c->tile_stride),
      c->row_stride, c->m_count, c->act_scale, c->act_zp, c->colsum_a + col,
      c->w_scale_a + col, c->bias_a + col, at, 32u, 0);
    hvx_dequant_acc_tile_to_f32(
      (const int32_t *)(c->tiles_base +
                        (size_t)(c->n_pairs + j) * c->tile_stride),
      c->row_stride, c->m_count, c->act_scale, c->act_zp, c->colsum_b + col,
      c->w_scale_b + col, c->bias_b + col, bt, 32u, 0);
    for (uint32_t r = 0; r < c->m_count; ++r)
      for (uint32_t k = 0; k < 32u; ++k)
        c->dst[(size_t)r * c->dst_stride + col + k] =
          at[r * 32u + k] * bt[r * 32u + k];
  }
}

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
   takes: the registration bake (make_weight fills the table directly),
   the DDR fallback when the accumulator layout is unusable (the stub
   layout always is), and the fused SwiGLU entries. Reaching one is a
   harness bug, not a kernel result. */
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
void hvx_dequant_i32_to_f32(const int32_t *acc, uint32_t m_valid,
                            uint32_t m_pad, uint32_t n, const float *act_scale,
                            const int32_t *act_zp, const int32_t *colsum_w,
                            const float *w_scale, const float *bias, float *out,
                            int accumulate) {
  (void)acc, (void)m_valid, (void)m_pad, (void)n, (void)act_scale, (void)act_zp;
  (void)colsum_w, (void)w_scale, (void)bias, (void)out, (void)accumulate;
  abort();
}
void hvx_swiglu_inplace_f32(float *gate, const float *up, uint32_t m_valid,
                            uint32_t n_out, hvx_worker_pool *pool) {
  (void)gate, (void)up, (void)m_valid, (void)n_out, (void)pool;
  abort();
}
