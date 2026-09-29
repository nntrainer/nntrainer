// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   nntr_hvx_attn_m1_probe.c
 * @date   29 Sep 2026
 * @brief  DEBUG-ONLY FastRPC entry attn_m1_probe: hvx_attn_m1_hf.h's
 *         fp16-lane primitives on silicon, and the per-FMA cost and fetch
 *         rate of the kernel shapes plan 170 designs (step 1, sitting S1)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Plan 170 decides its kernel on two silicon facts the ISS cannot give:
 * whether the qf32 multiply-add narrowed once to hf is the CPU's fused
 * fp16 FMA (semantics ops, HvxAttnM1Probe.Semantics = G1), and what a
 * 64-lane FMA costs per lane count (cost ops, HvxAttnM1Probe.Cost). The
 * op codes and words are nntr_attn_m1_probe.h's. No model path calls this.
 *
 * The cost loops are the kernel's inner loops on L2-resident synthetic
 * data (a 32 KiB tile per lane, re-read for ATTN_M1_PROBE_L positions), so
 * they measure the vector pipe, not DDR; the FETCH pair streams a cold
 * 3 MiB slab (evicted by reading 4 MiB between runs, outside the timer)
 * with and without an l2fetch lead two 16 KiB blocks ahead. Each lane
 * reads its own data; each pool run is timed on the caller (pcycles and
 * qtimer), each lane times itself.
 *
 * ADDRESS BUDGET: the call allocates at most lanes * 48 KiB (compute ops)
 * or 7 MiB (the FETCH slab and evictor) on the DSP heap and frees it
 * before returning; nothing persists in the session.
 *
 * Errors: a bad op, length, lane or rep count is AEE_EINVALIDFORMAT, an
 * allocation failure AEE_ENOMEMORY; never AEE_EBADPARM (the stale-skel
 * symptom, rule 3).
 */

#include <AEEStdErr.h>
#include <HAP_perf.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <remote.h>
#include <stdlib.h>
#include <string.h>

#include "nntr_attn_m1_probe.h"
#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

#include "hvx_attn_m1_hf.h"
#include "hvx_convert.h"
#include "hvx_swiglu_det.h"
#include "hvx_worker_pool.h"

#define PL ATTN_M1_PROBE_L
/** @brief Per-lane bytes of the compute ops' data. */
#define PER_LANE (48u * 1024u)
/** @brief FETCH: 16 KiB blocks, the l2fetch box, two blocks ahead. */
#define FETCH_BLOCK_VEC 128u
#define FETCH_LEAD_VEC (2u * FETCH_BLOCK_VEC)
#define EVICT_BYTES (4u << 20)

/** @brief Hides a pointer's value from the optimiser, so a loop over the
 *         same tile is not hoisted out of the position loop. */
#define LAUNDER(p) __asm__ volatile("" : "+r"(p))

typedef struct {
  uint32_t op;
  uint8_t *buf;        /**< compute ops: PER_LANE bytes per lane */
  const uint8_t *slab; /**< FETCH: the slab */
  uint32_t chunk;      /**< FETCH: bytes per lane */
  uint64_t busy[ATTN_M1_PROBE_MAX_LANES];
  HVX_Vector sink[ATTN_M1_PROBE_MAX_LANES];
  uint32_t lanes;
} probe_job;

/* Per-lane buffer layout (all 128-byte aligned). */
#define OFF_TILE 0u         /* 8 KiB: Kt tile / V rows / the f32 tile */
#define OFF_Q (8u * 1024u)  /* 16 KiB: two heads' q splats, or q32 */
#define OFF_P (24u * 1024u) /* 8 KiB: four heads' probabilities (hf) */

/** @brief Today's score loop (hvx_attn_m1_f32.c): 32 f32 lanes, eight
 *         accumulators over d, hvx_fma16_sf. */
static HVX_Vector cost_fma16_sf(uint8_t *buf) {
  const float *q = (const float *)(buf + OFF_Q);
  HVX_Vector sink = Q6_V_vzero();
  for (uint32_t b = 0; b < PL / 32u; ++b) {
    const HVX_Vector *kt = (const HVX_Vector *)(buf + OFF_TILE);
    LAUNDER(kt);
    HVX_Vector acc[ATTN_M1_DET_ACC];
    for (uint32_t l = 0; l < ATTN_M1_DET_ACC; ++l) {
      acc[l] = Q6_V_vzero();
    }
    for (uint32_t d = 0; d < 64u; ++d) {
      acc[d % 8u] = hvx_fma16_sf(acc[d % 8u], hvx_splat_sf(q[d]), kt[d]);
    }
    for (uint32_t l = 0; l < ATTN_M1_DET_ACC; ++l) {
      sink = Q6_V_vor_VV(sink, acc[l]);
    }
  }
  return sink;
}

/** @brief One step of accumulator l for d = d0 + l. */
#define QFMA(A, Q, K) A = hvx_hf_fma(A, Q, K, one)

/** @brief hf scores over a tiled Kt ([64 d][64 positions] per tile), one q
 *         head: eight named accumulators over d = 8 blk + l (arrays spill
 *         on hexagon-clang 19 -O3), then the tree and * 0.125. */
static HVX_Vector cost_scores1(uint8_t *buf) {
  const HVX_Vector one = Q6_Vh_vsplat_R(HVX_HF_ONE);
  const HVX_Vector eighth = Q6_Vh_vsplat_R(0x3000);
  const HVX_Vector *qs = (const HVX_Vector *)(buf + OFF_Q);
  HVX_Vector sink = Q6_V_vzero();
  for (uint32_t b = 0; b < PL / 64u; ++b) {
    const HVX_Vector *kt = (const HVX_Vector *)(buf + OFF_TILE);
    LAUNDER(kt);
    HVX_Vector a0 = Q6_V_vzero(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0,
               a6 = a0, a7 = a0;
#pragma unroll 1
    for (uint32_t d = 0; d < 64u; d += 8u) {
      QFMA(a0, qs[d], kt[d]);
      QFMA(a1, qs[d + 1u], kt[d + 1u]);
      QFMA(a2, qs[d + 2u], kt[d + 2u]);
      QFMA(a3, qs[d + 3u], kt[d + 3u]);
      QFMA(a4, qs[d + 4u], kt[d + 4u]);
      QFMA(a5, qs[d + 5u], kt[d + 5u]);
      QFMA(a6, qs[d + 6u], kt[d + 6u]);
      QFMA(a7, qs[d + 7u], kt[d + 7u]);
    }
    const HVX_Vector acc[ATTN_M1_DET_ACC] = {a0, a1, a2, a3, a4, a5, a6, a7};
    sink = Q6_V_vor_VV(sink, hvx_hf_score(acc, eighth));
  }
  return sink;
}

/** @brief cost_scores1 for two q heads sharing each Kt load. */
static HVX_Vector cost_scores2(uint8_t *buf) {
  const HVX_Vector one = Q6_Vh_vsplat_R(HVX_HF_ONE);
  const HVX_Vector eighth = Q6_Vh_vsplat_R(0x3000);
  const HVX_Vector *qs = (const HVX_Vector *)(buf + OFF_Q);
  const HVX_Vector *qt = qs + 64;
  HVX_Vector sink = Q6_V_vzero();
  for (uint32_t b = 0; b < PL / 64u; ++b) {
    const HVX_Vector *kt = (const HVX_Vector *)(buf + OFF_TILE);
    LAUNDER(kt);
    HVX_Vector a0 = Q6_V_vzero(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0,
               a6 = a0, a7 = a0;
    HVX_Vector c0 = a0, c1 = a0, c2 = a0, c3 = a0, c4 = a0, c5 = a0, c6 = a0,
               c7 = a0;
#pragma unroll 1
    for (uint32_t d = 0; d < 64u; d += 8u) {
      HVX_Vector k;
      k = kt[d];
      QFMA(a0, qs[d], k);
      QFMA(c0, qt[d], k);
      k = kt[d + 1u];
      QFMA(a1, qs[d + 1u], k);
      QFMA(c1, qt[d + 1u], k);
      k = kt[d + 2u];
      QFMA(a2, qs[d + 2u], k);
      QFMA(c2, qt[d + 2u], k);
      k = kt[d + 3u];
      QFMA(a3, qs[d + 3u], k);
      QFMA(c3, qt[d + 3u], k);
      k = kt[d + 4u];
      QFMA(a4, qs[d + 4u], k);
      QFMA(c4, qt[d + 4u], k);
      k = kt[d + 5u];
      QFMA(a5, qs[d + 5u], k);
      QFMA(c5, qt[d + 5u], k);
      k = kt[d + 6u];
      QFMA(a6, qs[d + 6u], k);
      QFMA(c6, qt[d + 6u], k);
      k = kt[d + 7u];
      QFMA(a7, qs[d + 7u], k);
      QFMA(c7, qt[d + 7u], k);
    }
    const HVX_Vector acc[ATTN_M1_DET_ACC] = {a0, a1, a2, a3, a4, a5, a6, a7};
    const HVX_Vector acd[ATTN_M1_DET_ACC] = {c0, c1, c2, c3, c4, c5, c6, c7};
    sink = Q6_V_vor_VV(
      sink, Q6_V_vor_VV(hvx_hf_score(acc, eighth), hvx_hf_score(acd, eighth)));
  }
  return sink;
}

/** @brief hf PV: four q heads share each V row (lanes = d), p ascending. */
static HVX_Vector cost_pv4(uint8_t *buf) {
  const HVX_Vector one = Q6_Vh_vsplat_R(HVX_HF_ONE);
  const HVX_Vector *vr = (const HVX_Vector *)(buf + OFF_TILE);
  const uint16_t *pr = (const uint16_t *)(buf + OFF_P);
  HVX_Vector o0 = Q6_V_vzero(), o1 = o0, o2 = o0, o3 = o0;
  for (uint32_t p = 0; p < PL; ++p) {
    const HVX_Vector v = vr[p % 64u];
    o0 = hvx_hf_fma(o0, Q6_Vh_vsplat_R(pr[p]), v, one);
    o1 = hvx_hf_fma(o1, Q6_Vh_vsplat_R(pr[PL + p]), v, one);
    o2 = hvx_hf_fma(o2, Q6_Vh_vsplat_R(pr[2u * PL + p]), v, one);
    o3 = hvx_hf_fma(o3, Q6_Vh_vsplat_R(pr[3u * PL + p]), v, one);
  }
  return Q6_V_vor_VV(Q6_V_vor_VV(o0, o1), Q6_V_vor_VV(o2, o3));
}

/** @brief Every vector of @a bytes at @a src, optionally with an l2fetch
 *         of the block two ahead (Rtt: stride, width, height = 128 B x
 *         128 rows). */
static HVX_Vector cost_fetch(const uint8_t *src, uint32_t bytes, int l2f) {
  const HVX_Vector *v = (const HVX_Vector *)src;
  const uint32_t n = bytes / 128u;
  HVX_Vector acc = Q6_V_vzero();
  for (uint32_t i = 0; i < n; i += FETCH_BLOCK_VEC) {
    if (l2f && i + FETCH_LEAD_VEC < n) {
      Q6_l2fetch_AP((void *)(v + i + FETCH_LEAD_VEC),
                    (128ull << 32) | (128ull << 16) | FETCH_BLOCK_VEC);
    }
    for (uint32_t j = i; j < i + FETCH_BLOCK_VEC; j += 4u) {
      acc = Q6_V_vor_VV(acc, Q6_V_vor_VV(Q6_V_vor_VV(v[j], v[j + 1u]),
                                         Q6_V_vor_VV(v[j + 2u], v[j + 3u])));
    }
  }
  return acc;
}

static void probe_lane(uint32_t n, uint32_t i, void *arg) {
  probe_job *job = (probe_job *)arg;
  const uint64_t t0 = HAP_perf_get_pcycles();
  uint8_t *buf = job->buf ? job->buf + (size_t)i * PER_LANE : NULL;
  HVX_Vector r;
  switch (job->op) {
  case ATTN_M1_PROBE_FMA16_SF:
    r = cost_fma16_sf(buf);
    break;
  case ATTN_M1_PROBE_SCORES1:
    r = cost_scores1(buf);
    break;
  case ATTN_M1_PROBE_SCORES2:
    r = cost_scores2(buf);
    break;
  case ATTN_M1_PROBE_PV4:
    r = cost_pv4(buf);
    break;
  default:
    r = cost_fetch(job->slab + (size_t)i * job->chunk, job->chunk,
                   job->op == ATTN_M1_PROBE_FETCH_L2F);
    break;
  }
  job->busy[i] += HAP_perf_get_pcycles() - t0;
  job->sink[i] = Q6_V_vor_VV(job->sink[i], r);
  job->lanes = n;
}

/** @brief Fills each lane's data with finite fp16 values (or f32 on the
 *         fp16 grid for the sf loop): scores near 0, probabilities O(1/L). */
static void fill_lanes(uint8_t *buf, uint32_t lanes) {
  for (uint32_t i = 0; i < lanes; ++i) {
    uint8_t *b = buf + (size_t)i * PER_LANE;
    uint16_t *t16 = (uint16_t *)(b + OFF_TILE);
    for (uint32_t k = 0; k < 64u * 64u; ++k) {
      t16[k] = (uint16_t)(0x2000u + (k * 7u + i) % 1000u);
    }
    uint16_t *qh = (uint16_t *)(b + OFF_Q);
    for (uint32_t d = 0; d < 128u; ++d) {
      const uint16_t h = (uint16_t)(0x2C00u ^ (d * 37u % 512u));
      for (uint32_t l = 0; l < 64u; ++l) {
        qh[d * 64u + l] = h; /* one splat vector per (head, d) */
      }
    }
    uint16_t *pr = (uint16_t *)(b + OFF_P);
    for (uint32_t k = 0; k < 4u * PL; ++k) {
      pr[k] = (uint16_t)(0x1400u + (k * 13u) % 700u); /* ~ 2^-10 */
    }
  }
}

/** @brief The sf loop reads its tile and q as f32. */
static void fill_lanes_sf(uint8_t *buf, uint32_t lanes) {
  for (uint32_t i = 0; i < lanes; ++i) {
    float *t32 = (float *)(buf + (size_t)i * PER_LANE + OFF_TILE);
    float *q = (float *)(buf + (size_t)i * PER_LANE + OFF_Q);
    for (uint32_t k = 0; k < 64u * 32u; ++k) {
      t32[k] = (float)((int)((k * 7u + i) % 101u) - 50) / 1024.0f;
    }
    for (uint32_t d = 0; d < 64u; ++d) {
      q[d] = (float)((int)d - 31) / 128.0f;
    }
  }
}

static int run_semantics(uint32 op, const uint16 *a, const uint16 *b,
                         const uint16 *c, uint16 *y, uint32_t n) {
  const HVX_Vector one = Q6_Vh_vsplat_R(HVX_HF_ONE);
  for (uint32_t i = 0; i < n; i += 64u) {
    const HVX_Vector va = *(const HVX_UVector *)(a + i);
    const HVX_Vector vb = *(const HVX_UVector *)(b + i);
    const HVX_Vector vc = *(const HVX_UVector *)(c + i);
    HVX_Vector r;
    switch (op) {
    case ATTN_M1_PROBE_QFMA:
      r = hvx_hf_fma(vc, va, vb, one);
      break;
    case ATTN_M1_PROBE_ADD:
      r = Q6_Vhf_vadd_VhfVhf(va, vb);
      break;
    case ATTN_M1_PROBE_SUB:
      r = Q6_Vhf_vsub_VhfVhf(va, vb);
      break;
    case ATTN_M1_PROBE_MUL:
      r = Q6_Vhf_vmpy_VhfVhf(va, vb);
      break;
    case ATTN_M1_PROBE_MAX:
      r = Q6_Vhf_vmax_VhfVhf(va, vb);
      break;
    case ATTN_M1_PROBE_EIGHTH:
      r = Q6_Vhf_vmpy_VhfVhf(va, Q6_Vh_vsplat_R(0x3000));
      break;
    case ATTN_M1_PROBE_ZPLUS:
      r = Q6_Vhf_vadd_VhfVhf(Q6_V_vzero(), va);
      break;
    case ATTN_M1_PROBE_EXP16:
      r = hvx_hf_exp16(va, one);
      break;
    default: { /* DIV16 */
      const HVX_VectorPair l = hvx_hf_widen(vb, one);
      const HVX_VectorPair rc = Q6_W_vcombine_VV(
        hvx_recip_det_sf(Q6_V_hi_W(l)), hvx_recip_det_sf(Q6_V_lo_W(l)));
      r = hvx_hf_div16(va, l, rc, one);
      break;
    }
    }
    *(HVX_UVector *)(y + i) = r;
  }
  return AEE_SUCCESS;
}

static void put64(uint32 *prof, uint32_t w, uint64_t x) {
  prof[w] = (uint32_t)x;
  prof[w + 1u] = (uint32_t)(x >> 32);
}

int nntr_hvx_attn_m1_probe(remote_handle64 handle, uint32 op, uint32 lanes,
                           uint32 reps, const uint16 *a, int aLen,
                           const uint16 *b, int bLen, const uint16 *c, int cLen,
                           uint16 *y, int yLen, uint32 *prof, int profLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  if (op < ATTN_M1_PROBE_N_SEM) {
    if (aLen <= 0 || aLen % 64 != 0 || bLen != aLen || cLen != aLen ||
        yLen != aLen || (profLen != 0 && profLen != (int)ATTN_M1_PROBE_WORDS)) {
      return AEE_EINVALIDFORMAT;
    }
    const uint64_t t0 = HAP_perf_get_pcycles();
    const int rc = run_semantics(op, a, b, c, y, (uint32_t)aLen);
    if (profLen) {
      memset(prof, 0, ATTN_M1_PROBE_WORDS * sizeof(uint32));
      prof[ATTN_M1_PROBE_W_LANES] = 1u;
      prof[ATTN_M1_PROBE_W_REPS] = 1u;
      put64(prof, ATTN_M1_PROBE_W_WALL, HAP_perf_get_pcycles() - t0);
    }
    return rc;
  }
  if (op < ATTN_M1_PROBE_FMA16_SF || op >= ATTN_M1_PROBE_COST_END ||
      lanes == 0u || lanes > ATTN_M1_PROBE_MAX_LANES || reps == 0u ||
      reps > ATTN_M1_PROBE_MAX_REPS || aLen != 0 || bLen != 0 || cLen != 0 ||
      yLen != 0 || profLen != (int)ATTN_M1_PROBE_WORDS) {
    return AEE_EINVALIDFORMAT;
  }
  const int fetch = op >= ATTN_M1_PROBE_FETCH;
  probe_job job;
  memset(&job, 0, sizeof(job));
  job.op = op;
  uint8_t *mem = NULL, *evict = NULL;
  if (fetch) {
    mem = (uint8_t *)memalign(128, ATTN_M1_PROBE_SLAB_BYTES);
    evict = (uint8_t *)memalign(128, EVICT_BYTES);
    if (!mem || !evict) {
      free(mem);
      free(evict);
      return AEE_ENOMEMORY;
    }
    memset(mem, 0x11, ATTN_M1_PROBE_SLAB_BYTES);
    memset(evict, 0x22, EVICT_BYTES);
    job.slab = mem;
    /* whole 16 KiB blocks per lane */
    job.chunk =
      (ATTN_M1_PROBE_SLAB_BYTES / lanes) & ~(FETCH_BLOCK_VEC * 128u - 1u);
  } else {
    mem = (uint8_t *)memalign(128, (size_t)lanes * PER_LANE);
    if (!mem) {
      return AEE_ENOMEMORY;
    }
    memset(mem, 0, (size_t)lanes * PER_LANE);
    if (op == ATTN_M1_PROBE_FMA16_SF) {
      fill_lanes_sf(mem, lanes);
    } else {
      fill_lanes(mem, lanes);
    }
    job.buf = mem;
  }
  uint64_t wall = 0, qt = 0;
  HVX_Vector ev = Q6_V_vzero();
  for (uint32_t r = 0; r < reps; ++r) {
    if (fetch) { /* cold: read 4 MiB of other lines, outside the timer */
      ev = Q6_V_vor_VV(ev, cost_fetch(evict, EVICT_BYTES, 0));
    }
    const uint64_t t0 = HAP_perf_get_pcycles(),
                   q0 = HAP_perf_get_qtimer_count();
    hvx_worker_pool_run(s->quant_pool, probe_lane, &job, lanes);
    wall += HAP_perf_get_pcycles() - t0;
    qt += HAP_perf_get_qtimer_count() - q0;
  }
  memset(prof, 0, ATTN_M1_PROBE_WORDS * sizeof(uint32));
  prof[ATTN_M1_PROBE_W_LANES] = job.lanes;
  prof[ATTN_M1_PROBE_W_REPS] = reps;
  put64(prof, ATTN_M1_PROBE_W_WALL, wall);
  put64(prof, ATTN_M1_PROBE_W_QT, qt);
  uint64_t sum = 0, mx = 0;
  HVX_Vector sink = ev;
  for (uint32_t i = 0; i < job.lanes && i < ATTN_M1_PROBE_MAX_LANES; ++i) {
    sum += job.busy[i];
    mx = job.busy[i] > mx ? job.busy[i] : mx;
    sink = Q6_V_vor_VV(sink, job.sink[i]);
  }
  prof[ATTN_M1_PROBE_W_BUSY_MAX] = (uint32_t)mx;
  put64(prof, ATTN_M1_PROBE_W_BUSY_SUM, sum);
  prof[ATTN_M1_PROBE_W_FMA64] = op == ATTN_M1_PROBE_SCORES2 ? 2u * PL
                                : op == ATTN_M1_PROBE_PV4   ? 4u * PL
                                : fetch                     ? 0u
                                                            : PL;
  prof[ATTN_M1_PROBE_W_BYTES] = fetch ? job.chunk : 0u;
  prof[ATTN_M1_PROBE_W_SINK] = (uint32_t)Q6_R_vextract_VR(sink, 0);
  free(mem);
  free(evict);
  return AEE_SUCCESS;
}
