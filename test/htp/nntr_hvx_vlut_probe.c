// SPDX-License-Identifier: Apache-2.0
/**
 * @file nntr_hvx_vlut_probe.c
 * @brief Isolated VLUT issue-rate and dependency-chain measurements.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#include "nntr_hvx.h"
#include "nntr_hvx_session.h"
#include <AEEStdErr.h>
#include <HAP_perf.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

/** These barriers preserve every iteration without adding instructions. Inspect
 * the generated loops: eight VLUTs per iteration, with no loads or stores. */
#define KEEP8()                                                                \
  asm volatile(""                                                              \
               : "+v"(a0), "+v"(a1), "+v"(a2), "+v"(a3), "+v"(a4), "+v"(a5),   \
                 "+v"(a6), "+v"(a7))
#define LOOK32(v) Q6_Vb_vlut32_VbVbI(v, tab, 0)
#if __HEXAGON_ARCH__ >= 81
#define LOOK16(v) Q6_V_lo_W(Q6_Wh_vlut16_VbVhR_nomatch(v, tab, 0))
#else
#define LOOK16(v) Q6_V_lo_W(Q6_Wh_vlut16_VbVhI(v, tab, 0))
#endif
#define INDEPENDENT(LOOK)                                                      \
  for (uint32_t i = 0; i < n; ++i) {                                           \
    a0 = LOOK(a0);                                                             \
    a1 = LOOK(a1);                                                             \
    a2 = LOOK(a2);                                                             \
    a3 = LOOK(a3);                                                             \
    a4 = LOOK(a4);                                                             \
    a5 = LOOK(a5);                                                             \
    a6 = LOOK(a6);                                                             \
    a7 = LOOK(a7);                                                             \
    KEEP8();                                                                   \
  }
#define DEPENDENT(LOOK)                                                        \
  for (uint32_t i = 0; i < n; ++i) {                                           \
    a0 = LOOK(LOOK(LOOK(LOOK(LOOK(LOOK(LOOK(LOOK(a0))))))));                   \
    asm volatile("" : "+v"(a0));                                               \
  }

int nntr_hvx_vlut_probe(remote_handle64 handle, uint32 mode, uint32 n,
                        uint32 *stats, int statsLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s || !s->vtcm_base || !n || n > 1000000u || mode > 3u || !stats ||
      statsLen < 5)
    return AEE_EBADPARM;
  HVX_Vector a0 = Q6_V_vsplat_R(0), a1 = Q6_V_vsplat_R(1),
             a2 = Q6_V_vsplat_R(2), a3 = Q6_V_vsplat_R(3),
             a4 = Q6_V_vsplat_R(4), a5 = Q6_V_vsplat_R(5),
             a6 = Q6_V_vsplat_R(6), a7 = Q6_V_vsplat_R(7);
  HVX_Vector tab = Q6_V_vsplat_R(0x01020304);
  asm volatile("" : "+v"(tab));
  KEEP8();
  const uint64_t t0 = HAP_perf_get_time_us();
  const uint64_t c0 = HAP_perf_get_pcycles();
  switch (mode) {
  case 0:
    INDEPENDENT(LOOK32);
    break;
  case 1:
    INDEPENDENT(LOOK16);
    break;
  case 2:
    DEPENDENT(LOOK32);
    break;
  default:
    DEPENDENT(LOOK16);
    break;
  }
  KEEP8();
  const uint64_t cycles = HAP_perf_get_pcycles() - c0;
  const uint64_t us = HAP_perf_get_time_us() - t0;
  HVX_Vector sum = Q6_V_vxor_VV(a0, a1);
  sum = Q6_V_vxor_VV(sum, Q6_V_vxor_VV(a2, a3));
  sum = Q6_V_vxor_VV(sum, Q6_V_vxor_VV(a4, a5));
  sum = Q6_V_vxor_VV(sum, Q6_V_vxor_VV(a6, a7));
  *(HVX_Vector *)s->vtcm_base = sum;
  stats[0] = (uint32)cycles;
  stats[1] = (uint32)(cycles >> 32);
  stats[2] = (uint32)us;
  stats[3] = (uint32)(us >> 32);
  stats[4] = *(const uint32 *)s->vtcm_base;
  return AEE_SUCCESS;
}
