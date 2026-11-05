// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   nntr_hvx_attn_m1_probe.c
 * @date   29 Sep 2026
 * @brief  Diagnostic FastRPC entry for the fp16 primitives used by M1 attention
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Runs hvx_attn_m1_hf.h on host-supplied operands so the device test can
 * compare its fp16 arithmetic against the deterministic CPU specification.
 * Historical synthetic cost operations 16--26 are retired; their numeric
 * codes return AEE_EUNSUPPORTED. No model path calls this entry.
 */

#include <AEEStdErr.h>
#include <HAP_perf.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <remote.h>
#include <string.h>

#include "nntr_attn_m1_probe.h"
#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

#include "hvx_attn_m1_hf.h"
#include "hvx_swiglu_det.h"

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
  (void)lanes;
  (void)reps;
  if (op >= 16u && op <= 26u) {
    return AEE_EUNSUPPORTED;
  }
  if (op >= ATTN_M1_PROBE_N_SEM || aLen <= 0 || aLen % 64 != 0 ||
      bLen != aLen || cLen != aLen || yLen != aLen ||
      (profLen != 0 && profLen != (int)ATTN_M1_PROBE_WORDS)) {
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
