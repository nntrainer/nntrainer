// SPDX-License-Identifier: Apache-2.0
/**
 * @file tl2_dn_dequant_check.c
 * @brief Native TL2 DN worker parity, dispatch guards and isolated epilogues.
 * @author NNTrainer contributors
 * @bug Host instruction emulation and simulator timing do not replace silicon.
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hvx_dequant_i32.h"

static unsigned calls4, calls1;
static void counted_dequant4(const int32_t *a, float s, int32_t z,
                             const int32_t *cs, const float *ws,
                             const float *bias, float *out) {
  ++calls4;
  hvx_dequant_acc_4tiles_m1_f32(a, s, z, cs, ws, bias, out);
}

static void counted_dequant1(const int32_t *a, uint32_t stride, uint32_t rows,
                             const float *s, const int32_t *z,
                             const int32_t *cs, const float *ws,
                             const float *bias, float *out, uint32_t out_stride,
                             int accumulate) {
  ++calls1;
  hvx_dequant_acc_tile_to_f32(a, stride, rows, s, z, cs, ws, bias, out,
                              out_stride, accumulate);
}

/** Compile the actual static worker. Unused production entry points are
 * discarded with -ffunction-sections -Wl,--gc-sections; the GEMV and both
 * dequantizers below are their production sources, not scalar stand-ins. */
#define hvx_dequant_acc_4tiles_m1_f32 counted_dequant4
#define hvx_dequant_acc_tile_to_f32 counted_dequant1
#include "hexkl_mm_u8i4_moe.c"
#undef hvx_dequant_acc_tile_to_f32
#undef hvx_dequant_acc_4tiles_m1_f32

#ifdef NNTR_TL2_DN_SIM
#include <hexagon_sim_timer.h>
#include <hexagon_standalone.h>
#endif

enum { MAX_K = 1536, MAX_N = 2816, GUARD = 32 };
static int8_t dense[MAX_K * MAX_N];
static uint8_t packed[64u * 22u * 640u] __attribute__((aligned(128)));
static uint8_t input[MAX_K] __attribute__((aligned(128)));
static int16_t tables[64u * 256u] __attribute__((aligned(128)));
static int32_t sums[MAX_N] __attribute__((aligned(128)));
static float scales[MAX_N] __attribute__((aligned(128)));
static float biases[MAX_N] __attribute__((aligned(128)));
static float got[MAX_N + 2u * GUARD] __attribute__((aligned(128)));
static float want[MAX_N + 2u * GUARD] __attribute__((aligned(128)));
static uint32_t rng = 13579u;

static uint32_t next_u32(void) {
  rng = rng * 1664525u + 1013904223u;
  return rng;
}

static void check_worker(uint32_t K, uint32_t inter, uint32_t N, uint32_t rows,
                         uint32_t native, int gu_mag, int dn_mag,
                         unsigned expect4) {
  hexkl_weight_u8i4 gu = {0}, dn = {0};
  moe_m1_ctx c = {0};
  int32_t acc[128] __attribute__((aligned(128)));
  float scale = 0.013713f;
  int32_t zp = 127;
  const uint32_t panels = (N + 127u) / 128u;
  for (uint32_t k = 0; k < inter; ++k)
    input[k] = (uint8_t)next_u32();
  for (uint32_t n = 0; n < N; ++n) {
    sums[n] = 0;
    scales[n] = (float)(next_u32() % 8193u) / 7919.f;
    biases[n] = (float)((int32_t)(next_u32() % 1025u) - 512) / 1009.f;
    for (uint32_t k = 0; k < inter; ++k) {
      const int8_t w = (int8_t)(next_u32() % 3u) - 1;
      dense[(size_t)k * N + n] = w;
      sums[n] += dn_mag * w;
    }
  }
  assert(hvx_tl2_pack(dense, inter, N, packed) == 0);
  hvx_tl2_prepare(input, inter, 0u, (inter + 23u) / 24u, tables);
  gu.pal[0] = gu.pal[1] = (int8_t)-gu_mag;
  gu.pal[3] = (int8_t)gu_mag;
  dn.pal[0] = dn.pal[1] = (int8_t)-dn_mag;
  dn.pal[3] = (int8_t)dn_mag;
  dn.colsum_w = sums;
  dn.w_scale = scales;
  dn.bias = biases;
  c.k_tiles = K / 32u;
  c.inter = inter;
  c.N_out = N;
  c.dn_ntiles = N / 32u;
  c.u_hi = c.dn_ntiles;
  c.native_tl2 = native;
  c.ex[0].g = &gu;
  c.ex[0].d = &dn;
  c.ex[0].m = rows;
  c.ex[0].dn_wh = packed;
  c.ex[0].tl2_down = tables;
  c.ex[0].rq_scale = &scale;
  c.ex[0].rq_zp = &zp;
  c.ex[0].res = got + GUARD;

  memset(want, 0xa5, sizeof want);
  for (uint32_t p = 0; p < panels; ++p) {
    hvx_gemv_u8tl2_panel(packed + (size_t)p * ((inter + 23u) / 24u) * 640u,
                         inter, tables, acc);
    for (uint32_t j = 0; j < 128u && p * 128u + j < N; j += 32u) {
      const uint32_t col = p * 128u + j;
      moe_tl2_restore_magnitude(acc + j, dn.pal);
      hvx_dequant_acc_tile_to_f32(acc + j, 32u, 1u, &scale, &zp, sums + col,
                                  scales + col, biases + col,
                                  want + GUARD + col, N, 0);
    }
  }
  for (uint32_t lanes = 1u; lanes <= 8u; lanes *= 2u) {
    memset(got, 0xa5, sizeof got);
    calls4 = calls1 = 0;
    for (uint32_t id = 0; id < lanes; ++id)
      moe_tl2_down_worker(lanes, id, &c);
    assert(memcmp(got, want, sizeof got) == 0);
    assert(calls4 == (expect4 ? panels : 0u));
    assert(calls1 == (expect4 ? 0u : N / 32u));
  }
  printf("TL2 DN exact K=%u I=%u N=%u rows=%u native=%u palette=%d/%d "
         "four_tile=%u lanes=1/2/4/8\n",
         (unsigned)K, (unsigned)inter, (unsigned)N, (unsigned)rows,
         (unsigned)native, gu_mag, dn_mag, expect4);
}

#ifdef NNTR_TL2_DN_SIM
/** Timings include only epilogues on identical hot, aligned arrays. They
 * exclude GEMV, DMA, pool dispatch and the production path's shape guards. */
static void benchmark(unsigned passes) {
  int32_t acc[128] __attribute__((aligned(128)));
  float output[128] __attribute__((aligned(128)));
  const float scale = 0.013713f;
  const int32_t zp = 127;
  for (unsigned j = 0; j < 128u; ++j)
    acc[j] = (int32_t)(next_u32() % 783361u) - 391680;
  for (unsigned round = 0; round < 6u; ++round) {
    unsigned long long elapsed[2] = {0, 0};
    for (unsigned order = 0; order < 2u; ++order) {
      const unsigned grouped = order ^ (round & 1u);
      const unsigned long long begin = hexagon_sim_read_pcycles();
      for (unsigned p = 0; p < passes; ++p) {
        if (grouped) {
          hvx_dequant_acc_4tiles_m1_f32(acc, scale, zp, sums, scales, biases,
                                        output);
        } else {
          for (unsigned j = 0; j < 128u; j += 32u)
            hvx_dequant_acc_tile_to_f32(acc + j, 32u, 1u, &scale, &zp, sums + j,
                                        scales + j, biases + j, output + j,
                                        2048u, 0);
        }
      }
      elapsed[grouped] = hexagon_sim_read_pcycles() - begin;
    }
    printf("TL2 DN SIM epilogue round=%u passes=%u old_pcycles=%llu "
           "four_tile_pcycles=%llu sample=%g\n",
           round, passes, elapsed[0], elapsed[1], (double)output[0]);
  }
}
#endif

int main(int argc, char **argv) {
#ifdef NNTR_TL2_DN_SIM
  SIM_ACQUIRE_HVX;
  SIM_SET_HVX_DOUBLE_MODE;
#else
  (void)argc;
  (void)argv;
#endif
  check_worker(2048u, 1536u, 2048u, 1u, 1u, 1, 1, 1u);
  check_worker(2816u, 704u, 2816u, 1u, 1u, 1, 1, 0u);
  check_worker(2048u, 1536u, 2048u, 1u, 1u, 7, 1, 0u);
  check_worker(2048u, 1536u, 2048u, 1u, 1u, 1, 7, 0u);
  check_worker(2048u, 1536u, 2016u, 1u, 1u, 1, 1, 0u);
  /** Synthetic guard cases: production selects native TL2 only at M=1. */
  check_worker(2048u, 1536u, 2048u, 2u, 1u, 1, 1, 0u);
  check_worker(2048u, 1536u, 2048u, 1u, 0u, 1, 1, 0u);
#ifdef NNTR_TL2_DN_SIM
  const unsigned passes =
    argc > 1 ? (unsigned)strtoul(argv[1], NULL, 10) : 2000u;
  assert(passes && passes <= 1000000u);
  benchmark(passes);
  SIM_RELEASE_HVX;
#endif
  return 0;
}
