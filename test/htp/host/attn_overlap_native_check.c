// SPDX-License-Identifier: Apache-2.0
/**
 * @file attn_overlap_native_check.c
 * @author NNTrainer contributors
 * @bug No known bugs.
 */
/** @brief Real HVX score conversion, dispatch, bounds and error-drain checks.
 */
#include <assert.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "attn_overlap_under_test.h"

/** Original serial dequant_s from hexkl_attn_q.c at commit 0fcdfc1f9.
 * Source-file SHA256:
 * 67a1cc02396fd4115d5d243b3563fa1260402e06bf96739e3dbe3260c2de6d52.
 * Only the function name changes. Keep this reference independent of the
 * production dequant_s_at body so changed arithmetic cannot update both.
 */
static void reference_dequant_s(attn_ctx *c, uint32_t n, uint32_t kb,
                                uint32_t i, uint32_t col, uint32_t sbuf) {
  const HVX_Vector *restrict acc = acc_rows(c);
  const uint32_t rs = c->acc_layout->row_stride / 32u; /* vectors per row */
  const uint32_t k0 = kb * c->t->bc + 32u * col;
  const float *restrict qs =
    c->q_scale + (size_t)c->qi * c->t->g_br + HEXKL_ATTN_Q_ROWS * i;
  const int32_t *restrict qz =
    c->q_zp + (size_t)c->qi * c->t->g_br + HEXKL_ATTN_Q_ROWS * i;
  HVX_Vector sk, cs;
  if (k0 + 32u <= c->kv->max_rows) {
    sk = hvx_tile_load_u(c->kv->s_k + hexkl_kv_q_sk_index(c->kv, n, k0));
    cs = Q6_Vsf_equals_Vw(
      hvx_tile_load_u(c->kv->colsum_k + hexkl_kv_q_sk_index(c->kv, n, k0)));
  } else {
    // Past the registry: the tiles were zeroed, the lanes will be masked;
    // keep the arithmetic finite.
    sk = Q6_V_vzero();
    cs = Q6_V_vzero();
  }
  for (uint32_t h = 0; h < 2u; ++h) {
    HVX_Vector *restrict dst = s_tile(c, sbuf, 2u * i + h, col);
    const uint32_t r0 = 32u * h;
    for (uint32_t v = 0; v < TILE_VECS; v += 2u) {
      const uint32_t r = r0 + 2u * v;
      HVX_Vector q[4];
      for (uint32_t p = 0; p < 4u; ++p) {
        const HVX_Vector a = Q6_Vsf_equals_Vw(acc[(r + p) * rs]);
        const HVX_Vector zc =
          Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_equals_Vw(Q6_V_vsplat_R(qz[r + p])), cs);
        const HVX_Vector corr = Q6_Vsf_vsub_VsfVsf(a, zc);
        const HVX_Vector scaled =
          Q6_Vsf_vmpy_VsfVsf(corr, hvx_splat_sf(qs[r + p]));
        q[p] = Q6_Vqf32_vmpy_VsfVsf(scaled, sk);
      }
      dst[v] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(q[1], q[0]));
      dst[v + 1u] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(q[3], q[2]));
    }
  }
}

static attn_ctx *current;
static uint8_t *expected;
static uint32_t head, qblock, raw_kb, raw_tile, dot_calls, soft_sbuf;
static uint32_t submits, joins, stored, pv_calls, soft_calls;
static int fail_read, fail_pv;
static unsigned read_calls;
static _Atomic unsigned gate, done, soft_done;
static unsigned pending, expected_workers;
static hvx_worker_pool_func submitted_func;
static void *submitted_ctx;

static uint64_t now_us(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000u + (uint64_t)t.tv_nsec / 1000u;
}

static void *aligned_bytes(size_t n) {
  void *p = NULL;
  assert(posix_memalign(&p, 2048u, (n + 2047u) & ~(size_t)2047u) == 0);
  return p;
}

static void checked_worker(uint32_t threads, uint32_t id, void *unused) {
  (void)unused;
  // The actual worker is intentionally held until PV begins: this catches
  // a premature join and guarantees the accumulator gets overwritten.
  while (!atomic_load_explicit(&gate, memory_order_acquire)) {
  }
  submitted_func(threads, id, submitted_ctx);
  atomic_fetch_add_explicit(&done, 1u, memory_order_release);
}

static void checked_submit(hvx_worker_pool *pool, hvx_worker_pool_func func,
                           void *ctx, uint32_t units) {
  assert(!pending);
  assert(func == dequant_worker);
  assert(atomic_load(&soft_done) == hvx_worker_pool_workers(pool));
  submitted_func = func;
  submitted_ctx = ctx;
  expected_workers = hvx_worker_pool_workers(pool);
  if (expected_workers > units)
    expected_workers = units;
  atomic_store(&gate, 0u);
  atomic_store(&done, 0u);
  pending = 1u;
  ++submits;
  hvx_worker_pool_submit(pool, checked_worker, NULL, units);
}

static void checked_wait(hvx_worker_pool *pool) {
  hvx_worker_pool_wait(pool);
  if (pending) {
    assert(atomic_load_explicit(&done, memory_order_acquire) ==
           expected_workers);
    pending = 0u;
    ++joins;
  }
}

static void softmax_state_init(attn_ctx *c, uint32_t n, uint32_t qb) {
  assert(!pending);
  assert(c == current);
  head = n;
  qblock = qb;
}

static void dma_issue(attn_ctx *c, uint32_t n, uint32_t kb, uint32_t buf) {
  (void)c;
  (void)n;
  (void)kb;
  (void)buf;
}

static void dma_land(attn_ctx *c, uint32_t kb, uint32_t buf) {
  assert(c == current && buf == (kb & 1u));
  raw_kb = kb;
  raw_tile = 0u;
}

static void hexkl_micro_hmx_acc_clear_int32(void) { dot_calls = 0u; }

static void hexkl_hmx_mm_u8i8(const uint8_t *q, const uint8_t *k) {
  const uint32_t i = raw_tile / current->L.n_col_tiles;
  const uint32_t col = raw_tile % current->L.n_col_tiles;
  assert(q == q_tile(current, current->qi, i, dot_calls));
  assert(k == kt_tile(current, raw_kb & 1u, col, dot_calls));
  ++dot_calls;
}

static void hexkl_hmx_mm_u8i4(const uint8_t *q, const uint8_t *k) {
  hexkl_hmx_mm_u8i8(q, k);
}

static int hexkl_micro_hmx_acc_read_int32(uint8_t *vb, uint32_t cfg,
                                          uint32_t off) {
  attn_ctx *c = current;
  assert(vb == c->vb && cfg == c->cfg);
  assert(dot_calls == c->L.n_dot_tiles);
  if ((int)read_calls++ == fail_read)
    return AEE_EFAILED;
  const uint32_t i = raw_tile / c->L.n_col_tiles;
  const uint32_t col = raw_tile % c->L.n_col_tiles;
  const uint32_t want = qx_dequant_available(c) && raw_kb > 0u
                          ? c->L.qx_f32 + raw_tile * HEXKL_ATTN_Q_ACC_BYTES
                          : c->L.acc;
  assert(off == want && off % TB == 0u);
  assert((uint64_t)off + HEXKL_ATTN_Q_ACC_BYTES <= c->L.total);
  int32_t *p = (int32_t *)(vb + off);
  for (uint32_t r = 0; r < 64u; ++r)
    for (uint32_t d = 0; d < 32u; ++d) {
      uint32_t x = (head + 1u) * 1664525u + (raw_kb + 1u) * 1013904223u +
                   (qblock + 1u) * 12345u + raw_tile * 7649u + r * 433u + d;
      p[r * 32u + d] = (int32_t)(x % 4000001u) - 2000000;
    }
  attn_ctx ref = *c;
  ref.vb = expected;
  memcpy(expected + c->L.acc, vb + off, HEXKL_ATTN_Q_ACC_BYTES);
  reference_dequant_s(&ref, head, raw_kb, i, col, raw_kb & 1u);
  ++raw_tile;
  return AEE_SUCCESS;
}

static void fake_softmax_worker(uint32_t threads, uint32_t id, void *ctx) {
  attn_ctx *c = ctx;
  const size_t bytes = 2u * c->L.n_row_tiles * c->L.n_col_tiles * TB;
  uint32_t lo, hi;
  unit_range((uint32_t)bytes, threads, id, &lo, &hi);
  assert(memcmp(c->vb + c->L.s_hf[soft_sbuf] + lo,
                expected + c->L.s_hf[soft_sbuf] + lo, hi - lo) == 0);
  memset(c->vb + c->L.s_hf[soft_sbuf] + lo, 0x35, hi - lo);
  atomic_fetch_add_explicit(&soft_done, 1u, memory_order_release);
}

static void softmax_start(attn_ctx *c, uint32_t n, uint32_t qb, uint32_t kb,
                          uint32_t sbuf, int masked) {
  assert(!pending && n == head && qb == qblock && sbuf == (kb & 1u));
  (void)masked;
  soft_sbuf = sbuf;
  atomic_store(&soft_done, 0u);
  ++soft_calls;
  hvx_worker_pool_submit(c->pool, fake_softmax_worker, c, 256u);
}

static void softmax_finish_pquant(attn_ctx *c, uint32_t buf) {
  assert(buf == soft_sbuf);
  hvx_worker_pool_wait(c->pool);
  const unsigned workers = hvx_worker_pool_workers(c->pool);
  assert(atomic_load(&soft_done) == (workers ? workers : 1u));
  const size_t bytes = 2u * c->L.n_row_tiles * c->L.n_col_tiles * TB;
  for (size_t i = 0; i < bytes; ++i)
    assert(c->vb[c->L.s_hf[soft_sbuf] + i] == 0x35);
}

static int phase_pv(attn_ctx *c, uint32_t buf, int first) {
  (void)buf;
  assert(first == (pv_calls == 0u));
  memset(c->vb + c->L.acc, 0xad, HEXKL_ATTN_Q_ACC_BYTES);
  // Release only after PV has overwritten the original accumulator.
  atomic_store_explicit(&gate, 1u, memory_order_release);
  return (int)pv_calls++ == fail_pv ? AEE_EFAILED : AEE_SUCCESS;
}

static void phase_store(attn_ctx *c, uint32_t n, uint32_t qb) {
  assert(c == current && n == head && qb == qblock && !pending);
  ++stored;
}

static void check_guard(attn_ctx *c) {
  assert(qx_dequant_available(c));
  hexkl_attn_f16_shape s = *c->s;
  hexkl_attn_f16_tiling t = *c->t;
  hexkl_kv_q k = *c->kv;
  hexkl_attn_q_io io = *c->io;
  attn_ctx a = *c;
  a.s = &s;
  a.t = &t;
  a.kv = &k;
  a.io = &io;
  for (unsigned i = 0; i < 18u; ++i) {
    s = *c->s;
    t = *c->t;
    k = *c->kv;
    io = *c->io;
    a.L = c->L;
    a.cfg = c->cfg;
    a.pool = c->pool;
    switch (i) {
    case 0:
      s.n_q = 1024u;
      s.cache_to = s.n_q;
      break;
    case 1:
      s.n_q = 2049u;
      s.cache_to = s.n_q;
      break;
    case 2:
      s.cache_from = 128u;
      break;
    case 3:
      s.cache_to += 128u;
      break;
    case 4:
      s.n_head_q = 16u;
      break;
    case 5:
      s.n_head_kv = 4u;
      break;
    case 6:
      s.head_dim = 256u;
      break;
    case 7:
      s.window = 512u;
      break;
    case 8:
      s.softcap = 4.0f;
      break;
    case 9:
      k.kind = HEXKL_KV_Q4;
      break;
    case 10:
      k.fixed = 1;
      break;
    case 11:
      io.sinks = (const float *)&s;
      break;
    case 12:
      t.br = 16u;
      break;
    case 13:
      t.bc = 128u;
      break;
    case 14:
      a.L.n_qb_chunk = 1u;
      a.L.total = a.L.qx_f32 + 131072u;
      break;
    case 15:
      a.L.total = a.L.qx_f32 - 1u;
      break;
    case 16:
      a.cfg = a.L.total - 1u;
      break;
    case 17:
      a.pool = NULL;
      break;
    }
    assert(!qx_dequant_available(&a));
  }
}

static void check_case(uint32_t rows, uint32_t qb, uint32_t qi,
                       unsigned workers, int fixed, int error,
                       unsigned pattern) {
  hexkl_attn_f16_shape s = {rows, 0u, rows, 32u, 8u, 64u, 0u, 0.0f};
  hexkl_attn_f16_tiling t = {128u, 256u, 0u, 0u};
  assert(hexkl_attn_q_tiling_init(&s, &t) == HEXKL_ATTN_OK);
  uint32_t chunk = (rows + 127u) / 128u;
  if (chunk > 9u)
    chunk = 9u;
  hexkl_kv_q kv = {0};
  kv.kind = HEXKL_KV_Q8;
  kv.fixed = fixed;
  kv.max_rows = rows;
  kv.n_head_kv = 8u;
  kv.head_dim = 64u;
  kv.tile_bytes = 1024u;
  kv.n_col_tiles = (rows + 31u) / 32u;
  kv.s_k = aligned_bytes((size_t)rows * 8u * 4u + 128u);
  kv.colsum_k = aligned_bytes((size_t)rows * 8u * 4u + 128u);
  const float scales[] = {0.0f,   -0.0f,  0x1p-24f, 0x1p-12f,
                          0.125f, 1.001f, 0x1p12f,  1.0f};
  for (uint32_t i = 0; i < rows * 8u; ++i) {
    kv.s_k[i] = scales[(i + pattern) % 8u];
    kv.colsum_k[i] = (int32_t)((i * 1789u + pattern) % 16385u) - 8192;
  }
  hexkl_acc_layout acc = {1, 1, 0u, 32u};
  hexkl_attn_q_stats stats = {0};
  hexkl_attn_q_io io = {0};
  attn_ctx c = {0};
  c.s = &s;
  c.t = &t;
  c.kv = &kv;
  c.io = &io;
  c.st = &stats;
  c.acc_layout = &acc;
  c.cfg = 8u << 20;
  c.qi = qi;
  c.pool = workers ? hvx_worker_pool_create(workers) : NULL;
  assert(!workers || c.pool);
  assert(hexkl_attn_q_plan(&s, &t, c.cfg, 1024u, chunk, &c.L) == HEXKL_ATTN_OK);
  c.vb = aligned_bytes(c.L.total + 2048u);
  expected = aligned_bytes(c.L.total + 2048u);
  memset(c.vb, 0xa5, c.L.total + 2048u);
  memset(expected, 0xa5, c.L.total + 2048u);
  const uint32_t meta_rows = chunk * 512u;
  c.q_scale = aligned_bytes((size_t)meta_rows * 4u);
  c.q_zp = aligned_bytes((size_t)meta_rows * 4u);
  for (uint32_t i = 0; i < meta_rows; ++i) {
    c.q_scale[i] = scales[(i / 16u + pattern) % 8u];
    c.q_zp[i] = (int32_t)((i + pattern * 64u) & 255u);
  }
  const int enabled = rows >= 2048u && rows % 128u == 0u && workers && !fixed;
  assert(qx_dequant_available(&c) == enabled);
  if (enabled)
    check_guard(&c);
  current = &c;
  submits = joins = stored = pv_calls = soft_calls = read_calls = 0u;
  pending = 0u;
  fail_read = error == 1   ? 0
              : error == 2 ? 64
              : error == 5 ? 63
              : error == 6 ? 127
              : error == 7 ? 65
              : error == 8 ? 255
                           : -1;
  fail_pv = error == 3 ? 0 : error == 4 ? 1 : -1;
  const int rc = run_q_block(&c, pattern & 1u ? 7u : 0u, qb);
  assert(!pending && submits == joins);
  hvx_worker_pool_wait(c.pool);
  assert(rc == (error ? AEE_EFAILED : AEE_SUCCESS));
  assert(stored == (error ? 0u : 1u));
  if (!error) {
    uint32_t lo, hi;
    hexkl_attn_f16_kv_block_range(&s, &t, qb, &lo, &hi);
    assert(stats.n_blocks == hi - lo);
    assert(submits == (qx_dequant_available(&c) ? hi - lo - 1u : 0u));
  }
  // Raw readouts may only clobber Qx. Q tiles, KV tiles, P, O and the
  // trailing redzone must remain untouched by this scheduling fixture.
  for (uint32_t i = 0; i < c.L.acc; ++i)
    assert(c.vb[i] == 0xa5);
  for (uint32_t i = c.L.p_ah; i < c.L.qx_f32; ++i)
    assert(c.vb[i] == 0xa5);
  for (uint32_t i = c.L.total; i < c.L.total + 2048u; ++i)
    assert(c.vb[i] == 0xa5);
  for (uint32_t i = 0; i < meta_rows; ++i) {
    assert(memcmp(c.q_scale + i, scales + (i / 16u + pattern) % 8u, 4u) == 0);
    assert(c.q_zp[i] == (int32_t)((i + pattern * 64u) & 255u));
  }
  hvx_worker_pool_destroy(c.pool);
  free(c.vb);
  free(expected);
  free(c.q_scale);
  free(c.q_zp);
  free(kv.s_k);
  free(kv.colsum_k);
}

int main(void) {
  unsigned cases = 0u;
  for (unsigned workers = 0; workers <= 8u; ++workers)
    for (unsigned pattern = 0; pattern < 4u; ++pattern) {
      const uint32_t rows[] = {1024u, 1920u, 2048u, 2048u, 2048u,
                               2048u, 512u,  2176u, 2304u, 2432u};
      const uint32_t qb[] = {7u, 14u, 0u, 2u, 8u, 15u, 3u, 16u, 17u, 18u};
      const uint32_t qi[] = {7u, 5u, 0u, 2u, 8u, 6u, 3u, 7u, 8u, 0u};
      for (unsigned i = 0; i < sizeof(rows) / sizeof(rows[0]); ++i) {
        check_case(rows[i], qb[i], qi[i], workers, 0, 0, pattern);
        ++cases;
      }
      check_case(2048u, 7u, 7u, workers, 1, 0, pattern);
      ++cases;
    }
  for (unsigned workers = 1u; workers <= 8u; ++workers)
    for (int error = 1; error <= 8; ++error) {
      check_case(2048u, 7u, 7u, workers, 0, error, 2u);
      ++cases;
    }
  puts("Qx overlap: exact serial-HVX scores, async PV accumulator clobber,");
  printf("  %u pipeline/tail/error cases + repeated 18-guard sets PASS\n",
         cases);
  return 0;
}
