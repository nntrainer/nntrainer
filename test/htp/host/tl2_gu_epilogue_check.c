// SPDX-License-Identifier: Apache-2.0
/**
 * @file tl2_gu_epilogue_check.c
 * @brief Actual native TL2 GU helper and requant byte parity on SDK libnative.
 * @author NNTrainer contributors
 * @bug Host instruction emulation does not establish silicon performance.
 */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hexkl_mm_u8i4_moe.h"
#include "hexkl_wh_affine.h"
#include "htp_tl2_opts.h"
#include "hvx_dequant_i32.h"
#include "hvx_expand_i2i4.h"
#include "hvx_gemv_u8tl2.h"
#include "hvx_quant_u8.h"

#if defined(__SANITIZE_ADDRESS__)
#include <sanitizer/asan_interface.h>
#define POISON(p, n) __asan_poison_memory_region(p, n)
#define UNPOISON(p, n) __asan_unpoison_memory_region(p, n)
#else
#define POISON(p, n) ((void)0)
#define UNPOISON(p, n) ((void)0)
#endif

#define ROUND_UP_U32(v, a) ((((v) + ((a)-1)) / (a)) * (a))
static unsigned reference_calls, candidate_calls, direct_calls;

/** Exact production NULL-pool branch. Explicit outer lane invocations below
 * cover each worker partition without emulating a thread pool or DMA. */
void hvx_worker_pool_run(hvx_worker_pool *pool, hvx_worker_pool_func fn,
                         void *ctx, uint32_t work_units) {
  assert(pool == NULL);
  (void)work_units;
  fn(1u, 0u, ctx);
}

#define SERIAL_WRAPPER(prefix)                                                 \
  static void prefix##_serial_swiglu(                                          \
    const uint8_t *base, uint32_t tile_stride, uint32_t pairs, uint32_t g0,    \
    uint32_t row_stride, uint32_t rows, const float *scale, const int32_t *zp, \
    const int32_t *cs, const float *ws, const float *bias, uint32_t inter,     \
    uint32_t act, float *dst, uint32_t dst_stride, hvx_worker_pool *pool) {    \
    ++prefix##_calls;                                                          \
    hvx_dequant_swiglu_acc_tiles_to_f32(                                       \
      base, tile_stride, pairs, g0, row_stride, rows, scale, zp, cs, ws, bias, \
      inter, act, dst, dst_stride, pool);                                      \
  }
SERIAL_WRAPPER(reference)
SERIAL_WRAPPER(candidate)
#undef SERIAL_WRAPPER

static void counted_direct_swiglu(uint32_t lanes, uint32_t id, void *job) {
  ++direct_calls;
  hvx_dq_swiglu_worker(lanes, id, job);
}

#include "candidate-unit.inc"
#include "reference-unit.inc"

enum {
  EXPERTS = 5,
  ACTIVE = 4,
  INTER = 1536,
  SPLITS = 4,
  PARAMS = 64,
  GUARD = 128
};
struct expert_output {
  float gate[GUARD + 4u * INTER + GUARD];
  uint8_t mid[GUARD + 64u * INTER + GUARD];
  float scale[PARAMS];
  int32_t zp[PARAMS];
  int16_t lut[GUARD + 64u * 256u + GUARD];
} __attribute__((aligned(128)));
static struct expert_output expected[EXPERTS], actual[EXPERTS];
static int32_t accum[EXPERTS][SPLITS * 2u * INTER]
  __attribute__((aligned(128)));
static int32_t input_snapshot[EXPERTS][SPLITS * 2u * INTER];
static int32_t sums[EXPERTS][2u * INTER] __attribute__((aligned(128)));
static float ws[EXPERTS][2u * INTER] __attribute__((aligned(128)));
static float bias[EXPERTS][2u * INTER] __attribute__((aligned(128)));
static float input_scale[EXPERTS][PARAMS] __attribute__((aligned(128)));
static int32_t input_zp[EXPERTS][PARAMS] __attribute__((aligned(128)));
static hexkl_weight_u8i4 gu[EXPERTS], dn[EXPERTS];
static reference_moe_m1_ctx reference_ctx;
static candidate_moe_m1_ctx candidate_ctx;
static uint32_t random_state = 713u;
static unsigned cases, lane_checks, fallback_cases, requant_cases, guard_cases;
static uint8_t dummy_weight;

static uint32_t next_u32(void) {
  random_state = random_state * 1664525u + 1013904223u;
  return random_state;
}

static void init(unsigned pattern, unsigned inter, unsigned splits,
                 unsigned magnitude, unsigned dn_magnitude) {
  memset(&reference_ctx, 0, sizeof reference_ctx);
  memset(&candidate_ctx, 0, sizeof candidate_ctx);
  memset(expected, 0xa5, sizeof expected);
  memset(actual, 0xa5, sizeof actual);
  memset(gu, 0, sizeof gu);
  memset(dn, 0, sizeof dn);
  for (unsigned e = 0; e < EXPERTS; ++e) {
    for (unsigned j = 0; j < SPLITS * 2u * INTER; ++j) {
      int32_t value = (int32_t)(next_u32() % 1044481u) - 522240;
      if (pattern == 1)
        value = 0;
      else if (pattern == 2)
        value = (j & 1u) ? 522240 : -522240;
      else if (pattern == 3)
        value = (int32_t)(j % 17u) - 8;
      accum[e][j] = value;
    }
    for (unsigned n = 0; n < 2u * INTER; ++n) {
      sums[e][n] = (int32_t)(next_u32() % 4097u) - 2048;
      ws[e][n] = (float)(1u + next_u32() % 32u) / 32768.0f;
      bias[e][n] = (float)((int32_t)(next_u32() % 129u) - 64) / 1024.0f;
      if (pattern == 1) {
        sums[e][n] = 0;
        bias[e][n] = (n & 1u) ? -0.0f : 0.0f;
      }
    }
    for (unsigned n = 0; n < PARAMS; ++n) {
      input_scale[e][n] = pattern == 4   ? 0.0f
                          : pattern == 3 ? 0.0001f
                                         : 0.013713f;
      input_zp[e][n] = pattern == 2 ? 255 : pattern == 5 ? 0 : 127;
    }
    gu[e].pal[0] = gu[e].pal[1] = (int8_t)-magnitude;
    gu[e].pal[3] = (int8_t)magnitude;
    dn[e].pal[0] = dn[e].pal[1] = (int8_t)-dn_magnitude;
    dn[e].pal[3] = (int8_t)dn_magnitude;
    gu[e].tl2_bytes = dn[e].tl2_bytes = &dummy_weight;
    gu[e].colsum_w = sums[e];
    gu[e].w_scale = ws[e];
    gu[e].bias = bias[e];
  }
#define INIT_CONTEXT(c, output)                                                \
  do {                                                                         \
    (c).n_active = ACTIVE;                                                     \
    (c).k_tiles = 64u;                                                         \
    (c).inter = inter;                                                         \
    (c).inter_ktiles = inter / HEXKL_HMX_INT8_BLOCK_N_INNER;                   \
    (c).inter_ntiles = inter / HEXKL_HMX_INT8_BLOCK_N_COL;                     \
    (c).gu_ntiles = 2u * (c).inter_ntiles;                                     \
    (c).dn_ntiles = 64u;                                                       \
    (c).N_out = 2048u;                                                         \
    (c).native_tl2 = 1u;                                                       \
    (c).tl2_splits = splits;                                                   \
    (c).act = HVX_GLU_SILU;                                                    \
    for (unsigned e = 0; e < EXPERTS; ++e) {                                   \
      (c).ex[e].g = &gu[e];                                                    \
      (c).ex[e].d = &dn[e];                                                    \
      (c).ex[e].act_scale = input_scale[e];                                    \
      (c).ex[e].act_zp = input_zp[e];                                          \
      (c).ex[e].acc_gu = accum[e];                                             \
      (c).ex[e].gate_f32 = (output)[e].gate + GUARD;                           \
      (c).ex[e].mid_ah = (output)[e].mid + GUARD;                              \
      (c).ex[e].rq_scale = (output)[e].scale;                                  \
      (c).ex[e].rq_zp = (output)[e].zp;                                        \
      (c).ex[e].tl2_down = (output)[e].lut + GUARD;                            \
      (c).ex[e].m = 1u;                                                        \
    }                                                                          \
  } while (0)
  INIT_CONTEXT(reference_ctx, expected);
  INIT_CONTEXT(candidate_ctx, actual);
#undef INIT_CONTEXT
  memcpy(input_snapshot, accum, sizeof accum);
  reference_calls = candidate_calls = direct_calls = 0;
}

static void poison_padding(unsigned inter, unsigned splits, int on) {
  const unsigned used = splits * ((2u * inter + 127u) & ~127u);
  for (unsigned e = 0; e < EXPERTS; ++e) {
    if (on) {
      POISON(input_scale[e] + 1, sizeof input_scale[e] - sizeof(float));
      POISON(input_zp[e] + 1, sizeof input_zp[e] - sizeof(int32_t));
      POISON(accum[e] + used, sizeof accum[e] - used * sizeof(int32_t));
      if (dn[e].tl2_bytes) {
        POISON(expected[e].gate + GUARD + inter,
               (4u * INTER - inter) * sizeof(float));
        POISON(actual[e].gate + GUARD + inter,
               (4u * INTER - inter) * sizeof(float));
      }
    } else {
      UNPOISON(input_scale[e], sizeof input_scale[e]);
      UNPOISON(input_zp[e], sizeof input_zp[e]);
      UNPOISON(accum[e], sizeof accum[e]);
      UNPOISON(expected[e].gate, sizeof expected[e].gate);
      UNPOISON(actual[e].gate, sizeof actual[e].gate);
    }
  }
}

static void same_outputs(void) {
  assert(memcmp(expected, actual, sizeof actual) == 0);
  assert(memcmp(accum, input_snapshot, sizeof accum) == 0);
}

/** Compare after each lane: a final-output-only test would miss a worker that
 * overwrites another lane with the same bytes. */
static void check_epilogue(unsigned lanes, unsigned expert, unsigned low,
                           unsigned high, int eligible) {
  const unsigned inter = reference_ctx.inter;
  reference_ctx.u_lo = candidate_ctx.u_lo = expert * (inter / 32u) + low;
  reference_ctx.u_hi = candidate_ctx.u_hi = expert * (inter / 32u) + high;
  for (unsigned step = 0; step < lanes; ++step) {
    unsigned lane = lanes - 1u - step;
    poison_padding(inter, reference_ctx.tl2_splits, 1);
    reference_moe_tl2_gate_epilogue_worker(lanes, lane, &reference_ctx);
    candidate_moe_tl2_gate_epilogue_worker(lanes, lane, &candidate_ctx);
    poison_padding(inter, reference_ctx.tl2_splits, 0);
    same_outputs();
    ++lane_checks;
  }
  if (eligible) {
    assert(direct_calls == lanes);
    assert(candidate_calls == 0u);
  } else {
    assert(direct_calls == 0u);
    assert(candidate_calls == high - low);
    ++fallback_cases;
  }
  assert(reference_calls == high - low);
  ++cases;
}

static void check_requant(unsigned lanes, unsigned active,
                          unsigned fast_experts) {
  reference_ctx.tl2_fused = candidate_ctx.tl2_fused = 1u;
  reference_ctx.n_active = candidate_ctx.n_active = active;
  for (unsigned step = 0; step < lanes; ++step) {
    unsigned lane = lanes - 1u - step;
    poison_padding(reference_ctx.inter, reference_ctx.tl2_splits, 1);
    reference_moe_m1_requant_worker(lanes, lane, &reference_ctx);
    candidate_moe_m1_requant_worker(lanes, lane, &candidate_ctx);
    poison_padding(reference_ctx.inter, reference_ctx.tl2_splits, 0);
    same_outputs();
    ++lane_checks;
  }
  assert(reference_calls == active * reference_ctx.inter_ntiles);
  assert(direct_calls == fast_experts);
  assert(candidate_calls ==
         (active - fast_experts) * reference_ctx.inter_ntiles);
  ++cases;
  ++requant_cases;
}

/** Mixed native/QS2 calls are accepted by the original kernel. The generic
 * expert's stage A has already produced gate_f32; it must only requantize here.
 * At four native experts the call fuses GU into stage B even with another QS2
 * expert. Below four, each native GU completes before the shared B run. */
static void check_mixed(unsigned lanes, unsigned native_count, int mag7) {
  init(0u, INTER, 1u, 1u, 1u);
  const unsigned active = native_count + 1u;
  gu[native_count].tl2_bytes = dn[native_count].tl2_bytes = NULL;
  for (unsigned j = 0; j < INTER; ++j)
    expected[native_count].gate[GUARD + j] =
      actual[native_count].gate[GUARD + j] =
        (float)((int)(j % 31u) - 15) / 8.0f;
  if (mag7) {
    gu[1].pal[0] = gu[1].pal[1] = -7;
    gu[1].pal[3] = 7;
  }
  reference_ctx.n_active = candidate_ctx.n_active = active;
  reference_ctx.tl2_fused = candidate_ctx.tl2_fused = native_count >= 4u;
  if (native_count < 4u) {
    for (unsigned e = 0; e < native_count; ++e) {
      reference_ctx.u_lo = candidate_ctx.u_lo = e * 48u;
      reference_ctx.u_hi = candidate_ctx.u_hi = (e + 1u) * 48u;
      for (unsigned lane = 0; lane < lanes; ++lane) {
        poison_padding(INTER, 1u, 1);
        reference_moe_tl2_gate_epilogue_worker(lanes, lane, &reference_ctx);
        candidate_moe_tl2_gate_epilogue_worker(lanes, lane, &candidate_ctx);
        poison_padding(INTER, 1u, 0);
        same_outputs();
        ++lane_checks;
      }
    }
  }
  for (unsigned lane = 0; lane < lanes; ++lane) {
    poison_padding(INTER, 1u, 1);
    reference_moe_m1_requant_worker(lanes, lane, &reference_ctx);
    candidate_moe_m1_requant_worker(lanes, lane, &candidate_ctx);
    poison_padding(INTER, 1u, 0);
    same_outputs();
    ++lane_checks;
  }
  assert(reference_calls == native_count * 48u);
  assert(candidate_calls == (mag7 ? 48u : 0u));
  assert(direct_calls ==
         (native_count - (unsigned)mag7) * (native_count >= 4u ? 1u : lanes));
  ++cases;
  ++requant_cases;
}

static void check_guard(unsigned which) {
  init(0u, INTER, 1u, 1u, 1u);
  switch (which) {
  case 0:
    candidate_ctx.native_tl2 = 0u;
    break;
  case 1:
    candidate_ctx.k_tiles = 88u;
    break;
  case 2:
    candidate_ctx.inter = 704u;
    break;
  case 3:
    candidate_ctx.inter_ntiles = 22u;
    break;
  case 4:
    candidate_ctx.N_out = 2816u;
    break;
  case 5:
    candidate_ctx.tl2_splits = 2u;
    break;
  case 6:
    candidate_ctx.act = HVX_GLU_GELU_TANH;
    break;
  case 7:
    candidate_ctx.ex[0].m = 2u;
    break;
  case 8:
    gu[0].tl2_bytes = NULL;
    break;
  case 9:
    dn[0].tl2_bytes = NULL;
    break;
  case 10:
    gu[0].pal[0] = gu[0].pal[1] = -7;
    gu[0].pal[3] = 7;
    break;
  case 11:
    dn[0].pal[0] = dn[0].pal[1] = -7;
    dn[0].pal[3] = 7;
    break;
  default:
    abort();
  }
  assert(candidate_moe_tl2_gate_direct(&candidate_ctx, 0u, 1u, 0u) == 0);
  assert(direct_calls == 0u);
  same_outputs();
  ++guard_cases;
}

int main(void) {
  for (unsigned pattern = 0; pattern < 6u; ++pattern)
    for (unsigned lanes = 1; lanes <= 8u; ++lanes)
      for (unsigned expert = 0; expert < ACTIVE; ++expert) {
        init(pattern, INTER, 1u, 1u, 1u);
        check_epilogue(lanes, expert, 0u, 48u, 1);
      }
  for (unsigned pattern = 0; pattern < 6u; ++pattern)
    for (unsigned lanes = 1; lanes <= 8u; ++lanes)
      for (unsigned active = 1; active <= ACTIVE; ++active) {
        init(pattern, INTER, 1u, 1u, 1u);
        check_requant(lanes, active, active);
      }
  for (unsigned lanes = 1; lanes <= 8u; ++lanes) {
    for (unsigned split = 2; split <= 4u; split *= 2u) {
      init(0u, INTER, split, 1u, 1u);
      check_epilogue(lanes, 2u, 0u, 48u, 0);
      init(0u, INTER, split, 1u, 1u);
      check_requant(lanes, ACTIVE, 0);
    }
    for (unsigned mag = 0; mag < 3u; ++mag) {
      init(0u, INTER, 1u, mag != 1u ? 7u : 1u, mag ? 7u : 1u);
      check_epilogue(lanes, 3u, 0u, 48u, 0);
      init(0u, INTER, 1u, mag != 1u ? 7u : 1u, mag ? 7u : 1u);
      check_requant(lanes, ACTIVE, 0);
    }
    init(0u, 704u, 4u, 1u, 1u);
    reference_ctx.k_tiles = candidate_ctx.k_tiles = 88u;
    reference_ctx.N_out = candidate_ctx.N_out = 2816u;
    reference_ctx.act = candidate_ctx.act = HVX_GLU_GELU_TANH;
    check_epilogue(lanes, 1u, 0u, 22u, 0);
    init(0u, INTER, 1u, 1u, 1u);
    reference_ctx.native_tl2 = candidate_ctx.native_tl2 = 0u;
    check_epilogue(lanes, 1u, 0u, 48u, 0);
    init(0u, INTER, 1u, 1u, 1u);
    check_epilogue(lanes, 0u, 1u, 49u, 0);
    init(0u, INTER, 1u, 1u, 1u);
    check_epilogue(lanes, 0u, 47u, 49u, 0);
    init(0u, INTER, 1u, 1u, 1u);
    check_epilogue(lanes, 0u, 0u, 96u, 0);
    init(0u, INTER, 1u, 1u, 1u);
    check_epilogue(lanes, 0u, 0u, 0u, 0);
  }
  for (unsigned lanes = 1; lanes <= 8u; ++lanes)
    for (unsigned n = 3u; n <= 4u; ++n)
      for (unsigned mag7 = 0; mag7 < 2u; ++mag7)
        check_mixed(lanes, n, (int)mag7);
  for (unsigned g = 0; g < 12u; ++g)
    check_guard(g);
  init(0u, INTER, 1u, 1u, 1u);
  candidate_ctx.inter_ntiles = 0u;
  candidate_moe_tl2_gate_epilogue_worker(1u, 0u, &candidate_ctx);
  same_outputs();
  assert(direct_calls == 0u && candidate_calls == 0u);
  ++guard_cases;
  printf("TL2_GU_EPILOGUE %u parity cases (%u requant, %u fallback), "
         "%u intermediate lane checks, %u unchanged-output guards PASS\n",
         cases, requant_cases, fallback_cases, lane_checks, guard_cases);
  return 0;
}
