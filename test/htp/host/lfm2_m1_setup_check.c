// SPDX-License-Identifier: Apache-2.0
/** @file lfm2_m1_setup_check.c
 * @brief Actual whole-MoE setup parity with poisoned unused native M1 storage.
 * @author NNTrainer contributors
 * @bug DMA/pool and epilogues use the existing whole-layer host stand-ins.
 */
#define main nntr_existing_moe_check_main
#define hvx_worker_pool_run fixture_pool_run
#define hexkl_dma_lane_wait fixture_lane_wait
#include "moe_layer_host_check.c"
#undef main
#undef hvx_worker_pool_run
#undef hexkl_dma_lane_wait
#include "hvx_gemv_u8tl2.h"
#include <assert.h>

/** The existing stand-in treats every pool as its logical test pool. Actual
 * quantization uses NULL to run inline while GU(0)'s pool job is pending. */
void hvx_worker_pool_run(hvx_worker_pool *pool, hvx_worker_pool_func func,
                         void *ctx, uint32_t n_units) {
  if (!pool) {
    func(1u, 0u, ctx);
    return;
  }
  fixture_pool_run(pool, func, ctx, n_units);
}

/** Inject a lane timeout without enabling the old 4-bit GEMV scoreboard:
 * native TL2 kernels do not emit its per-column read callbacks. */
int hexkl_dma_lane_wait(hexkl_dma_desc2d *d) {
  const int rc = fixture_lane_wait(d);
  return g_lane_timeout && g_lane == 1u ? -1 : rc;
}

#if defined(__SANITIZE_ADDRESS__)
#include <sanitizer/asan_interface.h>
#define POISON(p, n) __asan_poison_memory_region((p), (n))
#define UNPOISON(p, n) __asan_unpoison_memory_region((p), (n))
#else
#define POISON(p, n) ((void)0)
#define UNPOISON(p, n) ((void)0)
#endif

extern int reference_hexkl_mm_u8i4_moe_layer_run(
  hexkl_weight_u8i4_table *, uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t,
  uint32_t, uint32_t, uint32_t, const uint32_t *, const uint32_t *,
  const uint32_t *, const uint32_t *, const float *, const float *, float *,
  hvx_worker_pool *, hexkl_moe_scratch *, uint32_t);

static int g_candidate, g_fast, g_observed, g_watch_padding;
static uint32_t expected_pending;
static const void *poison_ptr[32];
static size_t poison_size[32];
static uint32_t poison_count;
static float saved_scale;
static int32_t saved_zp;
static const unsigned char *saved_out;
static size_t saved_out_size;
static int refill_scratch = 1, use_pre_gamma;
static const unsigned char *scratch_base;
static unsigned char *scratch_before;
static size_t scratch_bytes;

static void expect_bytes(const void *p, size_t n, unsigned char expected) {
  const unsigned char *q = p;
  for (size_t i = 0; i < n; ++i)
    assert(q[i] == expected);
}

static void expect_unchanged(const void *p, size_t n) {
  const unsigned char *q = p;
  assert(q >= scratch_base && q + n <= scratch_base + scratch_bytes);
  assert(memcmp(q, scratch_before + (q - scratch_base), n) == 0);
}

static void poison_region(const void *p, size_t n) {
  assert(poison_count < 32u);
  poison_ptr[poison_count] = p;
  poison_size[poison_count++] = n;
  POISON(p, n);
}

/** This observer is inserted immediately after the actual kernel's M1 setup
 * by the runner, in both source versions. It changes no computed value. */
void lfm2_m1_setup_observe(uint32_t active, uint32_t slots, uint32_t K,
                           uint32_t N, const uint32_t *slot_of,
                           const uint32_t *slot_row, const float *slot_scale,
                           const int32_t *slot_zp, const float *scale_all,
                           const int32_t *zp_all, const uint8_t *pack_skip,
                           size_t pack_bytes, const float *out_c,
                           int rows_out) {
  (void)K;
  ++g_observed;
  assert(g_pending == expected_pending);
  assert(active >= 1u && slots == active * 64u);
  saved_scale = scale_all[0];
  saved_zp = zp_all[0];
  for (uint32_t e = 0; e < active; ++e) {
    const uint32_t sb = slot_of[e];
    assert(sb == e * 64u);
    assert(memcmp(slot_scale + sb, scale_all, sizeof(float)) == 0);
    assert(slot_zp[sb] == zp_all[0]);
    if (g_candidate && g_fast) {
      expect_unchanged(slot_scale + sb + 1u, 63u * sizeof(float));
      expect_unchanged(slot_zp + sb + 1u, 63u * sizeof(int32_t));
    } else {
      for (uint32_t j = 0; j < 64u; ++j) {
        assert(slot_row[sb + j] == 0u);
        assert(slot_scale[sb + j] == scale_all[0]);
        assert(slot_zp[sb + j] == zp_all[0]);
      }
    }
  }
  if (g_candidate && g_fast) {
    expect_unchanged(slot_row, slots * sizeof(uint32_t));
    expect_unchanged(pack_skip, pack_bytes);
    expect_unchanged(scale_all + 1u, 63u * sizeof(float));
    expect_unchanged(zp_all + 1u, 63u * sizeof(int32_t));
  } else {
    for (uint32_t j = 1u; j < 64u; ++j) {
      assert(scale_all[j] == 1.0f);
      assert(zp_all[j] == 0);
    }
  }
  saved_out = (const unsigned char *)out_c;
  saved_out_size = (size_t)N * sizeof(float);
  if (g_candidate && g_fast && rows_out)
    expect_unchanged(out_c, saved_out_size);
  else
    expect_bytes(out_c, saved_out_size, 0);
  if (g_watch_padding) {
    /** All-native M1 never reads padding even on the old path. Make any
     * accidental read/write fail under ASan after observing setup. */
    poison_region(slot_row, slots * sizeof(uint32_t));
    poison_region(pack_skip, pack_bytes);
    poison_region(scale_all + 1u, 63u * sizeof(float));
    poison_region(zp_all + 1u, 63u * sizeof(int32_t));
    for (uint32_t e = 0; e < active; ++e) {
      poison_region(slot_scale + slot_of[e] + 1u, 63u * sizeof(float));
      poison_region(slot_zp + slot_of[e] + 1u, 63u * sizeof(int32_t));
    }
    if (rows_out)
      poison_region(out_c, saved_out_size);
  }
}

static void unpoison_all(void) {
  for (uint32_t i = 0; i < poison_count; ++i)
    UNPOISON(poison_ptr[i], poison_size[i]);
  poison_count = 0u;
}

typedef struct {
  uint32_t K, I, N;
  W backing[8];
  uint8_t *native[8];
} setup_weights;

static void make_native(setup_weights *s, uint32_t K, uint32_t I, uint32_t N) {
  s->K = K;
  s->I = I;
  s->N = N;
  memset(&g_tbl, 0, sizeof(g_tbl));
  for (uint32_t h = 0; h < 8u; ++h) {
    make_weight(h, h < 4u ? K : I, h < 4u ? 2u * I : N, &s->backing[h]);
    hexkl_weight_u8i4 *w = &g_tbl.slots[h];
    int8_t *dense = malloc((size_t)w->K * w->N);
    s->native[h] = malloc(hvx_tl2_weight_bytes(w->K, w->N));
    assert(dense && s->native[h]);
    memset(w->colsum_w, 0, sizeof(int32_t) * w->N);
    for (uint32_t k = 0; k < w->K; ++k)
      for (uint32_t n = 0; n < w->N; ++n) {
        const int value = (int)(rnd() % 3u) - 1;
        dense[(size_t)k * w->N + n] = (int8_t)value;
        w->colsum_w[n] += value;
      }
    assert(hvx_tl2_pack(dense, w->K, w->N, s->native[h]) == 0);
    free(dense);
    const int8_t pal[4] = {-1, 0, 1, 1};
    memcpy(w->pal, pal, sizeof(pal));
    w->bits = 2u;
    w->wh_bytes = s->native[h];
    w->tl2_bytes = s->native[h];
    w->tl2_compact = 1;
  }
}

static void free_native(setup_weights *s) {
  for (uint32_t h = 0; h < 8u; ++h) {
    free(s->native[h]);
    free(s->backing[h].nib);
    free(s->backing[h].ws);
    free(s->backing[h].cs);
    free(s->backing[h].bias);
  }
}

static unsigned cases;
static hexkl_moe_scratch shared_scratch[2];
static void run_case(const setup_weights *s, uint32_t active, uint32_t E,
                     int rows_out, uint32_t workers, uint32_t queues, int gelu,
                     int timeout, int bad_row) {
  static uint8_t vtcm[8u << 20] __attribute__((aligned(128)));
  static unsigned pool_identity;
  hvx_worker_pool *pool = (hvx_worker_pool *)&pool_identity;
  uint32_t hg[64] = {0}, hd[64] = {0}, counts[64] = {0}, ri[4] = {0};
  const uint32_t ids[4] = {1u, 7u, 15u, E - 1u};
  const float weights[4] = {0.0625f, -0.3f, 0.33333334f, 1.25f};
  float input[2816] __attribute__((aligned(128)));
  float gamma[2816] __attribute__((aligned(128)));
  float output[2][4u * 2816u + 64u];
  float scales[2];
  int32_t zps[2];
  int rcs[2];
  const uint32_t out_words = rows_out ? active * s->N : s->N;
  uint32_t flags = (0xf03e1u & ~HEXKL_MOE_FLAG_GELU_TANH) |
                   (gelu ? HEXKL_MOE_FLAG_GELU_TANH : 0u);
  flags = (flags & ~(HEXKL_MOE_DMA_Q_BITS << HEXKL_MOE_DMA_Q_SHIFT)) |
          ((queues - 1u) << HEXKL_MOE_DMA_Q_SHIFT);
  if (rows_out)
    flags |= HEXKL_MOE_FLAG_ROWS_OUT;
  for (uint32_t i = 0; i < active; ++i) {
    hg[ids[i]] = i;
    hd[ids[i]] = 4u + i;
    counts[ids[i]] = 1u;
  }
  for (uint32_t k = 0; k < s->K; ++k) {
    input[k] = rndf();
    if (cases % 4u == 0u)
      input[k] = 0.0f;
    else if (cases % 4u == 1u)
      input[k] = (float)((int)(k % 255u) - 127) + 0.5f;
    else if (cases % 4u == 2u)
      input[k] *= 0x1p-20f;
    gamma[k] = 1.0f + (float)(k % 7u) * 0.03125f;
  }
  if (bad_row)
    ri[0] = 1u;
  g_workers = workers;
  g_lane_timeout = timeout;
  g_fast = s->K == 2048u && s->I == 1536u && s->N == 2048u && E == 64u &&
           !gelu && active != 0u;
  g_watch_padding = 1;
  expected_pending = queues > 1u && workers >= queues;
  for (int candidate = 0; candidate < 2; ++candidate) {
    hexkl_moe_scratch *scratch = &shared_scratch[candidate];
    /** Pre-sized only in this fixture. The reuse cases preserve bytes from
     * previous calls instead of refilling them. Snapshot the bytes that the
     * candidate is required to leave untouched before observing its setup. */
    assert(hexkl_moe_scratch_reserve(scratch, 32u << 20) == AEE_SUCCESS);
    if (refill_scratch)
      memset(scratch->base, 0xa5, scratch->cap);
    if (scratch_bytes < scratch->cap) {
      scratch_before = realloc(scratch_before, scratch->cap);
      assert(scratch_before);
      scratch_bytes = scratch->cap;
    }
    scratch_base = scratch->base;
    memcpy(scratch_before, scratch->base, scratch->cap);
    scratch->pre_gamma = use_pre_gamma ? gamma : NULL;
    scratch->pre_eps = 0.00001f;
    memset(vtcm, 0x96, sizeof(vtcm));
    memset(output[candidate], 0xff, sizeof(output[candidate]));
    g_candidate = candidate;
    g_observed = 0;
    saved_out = NULL;
    score_reset(0, vtcm, sizeof(vtcm));
    rcs[candidate] = (candidate ? hexkl_mm_u8i4_moe_layer_run
                                : reference_hexkl_mm_u8i4_moe_layer_run)(
      &g_tbl, vtcm, sizeof(vtcm), sizeof(vtcm), 1u, s->K, s->I, s->N, E, hg, hd,
      ri, counts, weights, input, output[candidate], pool, scratch, flags);
    unpoison_all();
    assert(!g_pending && !g_in_run);
    if (timeout)
      assert(g_lane_waits > 0u);
    if (!bad_row && active) {
      assert(g_observed == 1);
      scales[candidate] = saved_scale;
      zps[candidate] = saved_zp;
      if (rows_out) {
        if (candidate && g_fast)
          expect_unchanged(saved_out, saved_out_size);
        else
          expect_bytes(saved_out, saved_out_size, 0);
      }
    } else {
      assert(g_observed == 0);
      scales[candidate] = 0.0f;
      zps[candidate] = 0;
    }
    const uint32_t written = bad_row || (!active && rows_out) ? 0u : out_words;
    expect_bytes(output[candidate] + written,
                 sizeof(output[candidate]) - written * sizeof(float), 0xff);
    scratch->pre_gamma = NULL;
  }
  assert(rcs[0] == rcs[1]);
  if (bad_row)
    assert(rcs[0] == AEE_EBADPARM);
  else if (!active && rows_out)
    assert(rcs[0] == AEE_EUNSUPPORTED);
  else if (timeout)
    assert(rcs[0] == AEE_EFAILED);
  else
    assert(rcs[0] == AEE_SUCCESS);
  assert(memcmp(output[0], output[1], sizeof(output[0])) == 0);
  assert(memcmp(scales, scales + 1u, sizeof(float)) == 0 && zps[0] == zps[1]);
  ++cases;
  printf("M1 setup exact experts=%u E=%u rows_out=%d lanes=%u queues=%u "
         "fast=%d timeout=%d bad_row=%d refill=%d pre_gamma=%d\n",
         active, E, rows_out, workers + 1u, queues, g_fast, timeout, bad_row,
         refill_scratch, use_pre_gamma);
  g_lane_timeout = 0;
}

int main(void) {
  setvbuf(stdout, NULL, _IONBF, 0);
  setup_weights s;
  make_native(&s, 2048u, 1536u, 2048u);
  for (uint32_t n = 1u; n <= 4u; ++n)
    for (int rows = 0; rows <= 1; ++rows)
      for (uint32_t queues = 1u; queues <= 4u; queues *= 4u)
        run_case(&s, n, 64u, rows, queues == 1u ? 0u : 7u, queues, 0, 0, 0);
  run_case(&s, 2u, 32u, 1, 7u, 4u, 0, 0, 0);
  run_case(&s, 2u, 64u, 1, 7u, 4u, 1, 0, 0);
  run_case(&s, 1u, 64u, 1, 7u, 4u, 0, 0, 0);
  run_case(&s, 4u, 64u, 0, 7u, 4u, 0, 0, 0);
  run_case(&s, 2u, 64u, 1, 7u, 4u, 0, 1, 0);
  run_case(&s, 2u, 64u, 1, 7u, 4u, 0, 0, 1);
  run_case(&s, 0u, 64u, 0, 0u, 1u, 0, 0, 0);
  run_case(&s, 0u, 64u, 1, 0u, 1u, 0, 0, 0);
  /* Preserve actual previous-call contents across these transitions. */
  refill_scratch = 0;
  run_case(&s, 4u, 64u, 1, 7u, 4u, 0, 0, 0);
  run_case(&s, 2u, 32u, 1, 7u, 4u, 0, 0, 0);
  run_case(&s, 1u, 64u, 1, 7u, 4u, 0, 0, 0);
  run_case(&s, 4u, 64u, 0, 7u, 4u, 0, 0, 0);
  use_pre_gamma = 1;
  run_case(&s, 2u, 64u, 1, 7u, 4u, 0, 0, 0);
  use_pre_gamma = 0;
  refill_scratch = 1;
  free_native(&s);
  make_native(&s, 2816u, 704u, 2816u);
  run_case(&s, 2u, 64u, 0, 7u, 4u, 1, 0, 0);
  run_case(&s, 2u, 64u, 1, 7u, 4u, 1, 0, 0);
  free_native(&s);
  hexkl_moe_scratch_free(&shared_scratch[0]);
  hexkl_moe_scratch_free(&shared_scratch[1]);
  free(scratch_before);
  free(g_gemv_log);
  printf("M1 unused setup: %u whole-kernel parity/poison cases PASS\n", cases);
  return 0;
}
