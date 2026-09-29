// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_hexagon_protos.h
 * @date   27 Sep 2026
 * @brief  Host emulation of the HVX intrinsics the M=1 small-op and
 *         decode-attention kernels use, one IEEE f32 operation per lane
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * THE PREMISE: one Vsf multiply / add / subtract is one IEEE-754 f32
 * operation, round to nearest even, subnormals kept. That is what
 * HvxSwigluDet.MatchesScalarBitExact confirmed on the device (rule 24), and
 * what HvxM1Ops.* re-checks for these kernels. Each lane below stores
 * through a volatile so the host compiler cannot contract or reassociate.
 * The integer and bitwise ops, the byte rotate, the predicate ops and the
 * mux move bits exactly; vmin / vmax on finite values are exact by definition,
 * and Vsf_equals_Vw is one exactly representable int -> f32 conversion for |w|
 * <= 2^24 (plan 81 section 3.3 adds these for the attention kernel). What this
 * emulation cannot see: an aligned-load fault (the small ops use HVX_UVector
 * everywhere; the attention kernel's cache loads are aligned
 * -- a review item), inf/NaN encodings, and timing.
 *
 * One scalar-core op is emulated too (#164): Q6_R_sfmpyacc_RR, the sffma
 * instruction, is fmaf -- one IEEE fused multiply-add, which the ISS
 * matched bit for bit on 784 RMSNORM rows and two subnormal-sum probes
 * (plan 164 section 0); HvxM1Ops.* re-checks it on silicon.
 *
 * vror follows the PRM: Vd.ub[i] = Vu.ub[(i + Rt) mod 128]. The reduction
 * that uses it gives the same bits in either direction (IEEE add
 * commutes), which m1_ops_det.h's comment spells out.
 */

#ifndef __NNTRAINER_HVX_EMU_HVX_HEXAGON_PROTOS_H__
#define __NNTRAINER_HVX_EMU_HVX_HEXAGON_PROTOS_H__

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "hexagon_types.h"

static inline float hvx_emu_f(int32_t w) {
  float f;
  memcpy(&f, &w, sizeof(f));
  return f;
}
static inline int32_t hvx_emu_w(float f) {
  int32_t w;
  memcpy(&w, &f, sizeof(w));
  return w;
}

/** @brief sffma: x + s * t with one rounding (the SDK's
 *         __builtin_HEXAGON_F2_sffma, accumulator first). */
static inline float Q6_R_sfmpyacc_RR(float x, float s, float t) {
  volatile float r = fmaf(s, t, x);
  return r;
}

static inline HVX_Vector Q6_V_vzero(void) {
  HVX_Vector r;
  memset(&r, 0, sizeof(r));
  return r;
}

static inline HVX_Vector Q6_V_vsplat_R(int32_t x) {
  HVX_Vector r;
  for (int i = 0; i < HVX_EMU_LANES; ++i) {
    r.w[i] = x;
  }
  return r;
}

#define HVX_EMU_SF_BINOP(name, op)                                             \
  static inline HVX_Vector name(HVX_Vector a, HVX_Vector b) {                  \
    HVX_Vector r;                                                              \
    for (int i = 0; i < HVX_EMU_LANES; ++i) {                                  \
      volatile float t = hvx_emu_f(a.w[i]) op hvx_emu_f(b.w[i]);               \
      r.w[i] = hvx_emu_w(t);                                                   \
    }                                                                          \
    return r;                                                                  \
  }
HVX_EMU_SF_BINOP(Q6_Vsf_vadd_VsfVsf, +)
HVX_EMU_SF_BINOP(Q6_Vsf_vsub_VsfVsf, -)
HVX_EMU_SF_BINOP(Q6_Vsf_vmpy_VsfVsf, *)
#undef HVX_EMU_SF_BINOP

/* max / min: exact on finite values (the kernels keep every lane finite;
   the mask fill is -FLT_MAX, not -inf). Equal values are the same bits
   except +-0, which attn_m1_det.h shows cannot both occur in a score. */
static inline HVX_Vector Q6_Vsf_vmax_VsfVsf(HVX_Vector a, HVX_Vector b) {
  HVX_Vector r;
  for (int i = 0; i < HVX_EMU_LANES; ++i) {
    const float x = hvx_emu_f(a.w[i]), y = hvx_emu_f(b.w[i]);
    r.w[i] = (x > y) ? a.w[i] : b.w[i];
  }
  return r;
}
static inline HVX_Vector Q6_Vsf_vmin_VsfVsf(HVX_Vector a, HVX_Vector b) {
  HVX_Vector r;
  for (int i = 0; i < HVX_EMU_LANES; ++i) {
    const float x = hvx_emu_f(a.w[i]), y = hvx_emu_f(b.w[i]);
    r.w[i] = (x < y) ? a.w[i] : b.w[i];
  }
  return r;
}

/* int32 -> f32, numeric (hvx_convert.h: NOT a reinterpret on v79). Exact
   for |w| <= 2^24; hvx_exp_det_sf's k is within [-127, 123]. */
static inline HVX_Vector Q6_Vsf_equals_Vw(HVX_Vector a) {
  HVX_Vector r;
  for (int i = 0; i < HVX_EMU_LANES; ++i) {
    r.w[i] = hvx_emu_w((float)a.w[i]);
  }
  return r;
}

#define HVX_EMU_W_BINOP(name, op)                                              \
  static inline HVX_Vector name(HVX_Vector a, HVX_Vector b) {                  \
    HVX_Vector r;                                                              \
    for (int i = 0; i < HVX_EMU_LANES; ++i) {                                  \
      r.w[i] = (int32_t)((uint32_t)a.w[i] op(uint32_t) b.w[i]);                \
    }                                                                          \
    return r;                                                                  \
  }
HVX_EMU_W_BINOP(Q6_Vw_vadd_VwVw, +)
HVX_EMU_W_BINOP(Q6_Vw_vsub_VwVw, -)
#undef HVX_EMU_W_BINOP

/* Bitwise ops and signed word max / min / arithmetic shift (#152: the
   fp16 rounding and the round-to-odd step); exact by definition. */
#define HVX_EMU_V_BITOP(name, op)                                              \
  static inline HVX_Vector name(HVX_Vector a, HVX_Vector b) {                  \
    HVX_Vector r;                                                              \
    for (int i = 0; i < HVX_EMU_LANES; ++i) {                                  \
      r.w[i] = a.w[i] op b.w[i];                                               \
    }                                                                          \
    return r;                                                                  \
  }
HVX_EMU_V_BITOP(Q6_V_vand_VV, &)
HVX_EMU_V_BITOP(Q6_V_vor_VV, |)
HVX_EMU_V_BITOP(Q6_V_vxor_VV, ^)
#undef HVX_EMU_V_BITOP

static inline HVX_Vector Q6_Vw_vmax_VwVw(HVX_Vector a, HVX_Vector b) {
  HVX_Vector r;
  for (int i = 0; i < HVX_EMU_LANES; ++i) {
    r.w[i] = a.w[i] > b.w[i] ? a.w[i] : b.w[i];
  }
  return r;
}
static inline HVX_Vector Q6_Vw_vmin_VwVw(HVX_Vector a, HVX_Vector b) {
  HVX_Vector r;
  for (int i = 0; i < HVX_EMU_LANES; ++i) {
    r.w[i] = a.w[i] < b.w[i] ? a.w[i] : b.w[i];
  }
  return r;
}

static inline HVX_Vector Q6_Vw_vasr_VwR(HVX_Vector a, int32_t n) {
  HVX_Vector r;
  for (int i = 0; i < HVX_EMU_LANES; ++i) {
    r.w[i] = a.w[i] >> (n & 31); /* arithmetic on every host compiler */
  }
  return r;
}

static inline HVX_Vector Q6_Vuw_vlsr_VuwR(HVX_Vector a, int32_t n) {
  HVX_Vector r;
  for (int i = 0; i < HVX_EMU_LANES; ++i) {
    r.w[i] = (int32_t)((uint32_t)a.w[i] >> (n & 31));
  }
  return r;
}

static inline HVX_Vector Q6_Vw_vasl_VwR(HVX_Vector a, int32_t n) {
  HVX_Vector r;
  for (int i = 0; i < HVX_EMU_LANES; ++i) {
    r.w[i] = (int32_t)((uint32_t)a.w[i] << (n & 31));
  }
  return r;
}

/* Vx += Vu << Rt, word lanes, wrapping. */
static inline HVX_Vector Q6_Vw_vaslacc_VwVwR(HVX_Vector x, HVX_Vector u,
                                             int32_t n) {
  HVX_Vector r;
  for (int i = 0; i < HVX_EMU_LANES; ++i) {
    r.w[i] = (int32_t)((uint32_t)x.w[i] + ((uint32_t)u.w[i] << (n & 31)));
  }
  return r;
}

/* Predicates: one flag per byte; the word compare sets a lane's 4 bytes. */
static inline HVX_VectorPred Q6_Q_vcmp_gt_VwVw(HVX_Vector a, HVX_Vector b) {
  HVX_VectorPred q;
  for (int i = 0; i < HVX_EMU_LANES; ++i) {
    const uint8_t f = (a.w[i] > b.w[i]) ? 1u : 0u;
    memset(q.q + 4 * i, f, 4);
  }
  return q;
}

static inline HVX_VectorPred Q6_Q_vcmp_eq_VwVw(HVX_Vector a, HVX_Vector b) {
  HVX_VectorPred q;
  for (int i = 0; i < HVX_EMU_LANES; ++i) {
    const uint8_t f = (a.w[i] == b.w[i]) ? 1u : 0u;
    memset(q.q + 4 * i, f, 4);
  }
  return q;
}

/* Predicate logic, per byte: Qs & Qt, Qs | Qt and Qs & ~Qt. */
static inline HVX_VectorPred Q6_Q_and_QQ(HVX_VectorPred a, HVX_VectorPred b) {
  HVX_VectorPred q;
  for (int i = 0; i < 4 * HVX_EMU_LANES; ++i) {
    q.q[i] = a.q[i] & b.q[i];
  }
  return q;
}
static inline HVX_VectorPred Q6_Q_or_QQ(HVX_VectorPred a, HVX_VectorPred b) {
  HVX_VectorPred q;
  for (int i = 0; i < 4 * HVX_EMU_LANES; ++i) {
    q.q[i] = a.q[i] | b.q[i];
  }
  return q;
}
static inline HVX_VectorPred Q6_Q_and_QQn(HVX_VectorPred a, HVX_VectorPred b) {
  HVX_VectorPred q;
  for (int i = 0; i < 4 * HVX_EMU_LANES; ++i) {
    q.q[i] = a.q[i] & (uint8_t)!b.q[i];
  }
  return q;
}

/* vsetq2(Rt): bytes 0 .. ((Rt - 1) & 127) set, so 128 (and 0) set all --
   unlike vsetq, whose Rt = 128 wraps to none. The kernels pass 4..128. */
static inline HVX_VectorPred Q6_Q_vsetq2_R(int32_t n) {
  HVX_VectorPred q;
  const int last = (n - 1) & (4 * HVX_EMU_LANES - 1);
  for (int i = 0; i < 4 * HVX_EMU_LANES; ++i) {
    q.q[i] = (i <= last) ? 1u : 0u;
  }
  return q;
}

/* Vd = Qt ? Vu : Vv, per byte. */
static inline HVX_Vector Q6_V_vmux_QVV(HVX_VectorPred q, HVX_Vector a,
                                       HVX_Vector b) {
  HVX_Vector r;
  const uint8_t *pa = (const uint8_t *)a.w, *pb = (const uint8_t *)b.w;
  uint8_t *pr = (uint8_t *)r.w;
  for (int i = 0; i < 4 * HVX_EMU_LANES; ++i) {
    pr[i] = q.q[i] ? pa[i] : pb[i];
  }
  return r;
}

static inline HVX_Vector Q6_V_vror_VR(HVX_Vector a, int32_t bytes) {
  HVX_Vector r;
  const uint8_t *src = (const uint8_t *)a.w;
  uint8_t *dst = (uint8_t *)r.w;
  for (int i = 0; i < 4 * HVX_EMU_LANES; ++i) {
    dst[i] = src[(i + bytes) & (4 * HVX_EMU_LANES - 1)];
  }
  return r;
}

static inline int32_t Q6_R_vextract_VR(HVX_Vector a, int32_t byte_off) {
  int32_t w;
  memcpy(&w, (const uint8_t *)a.w + (byte_off & (4 * HVX_EMU_LANES - 1)),
         sizeof(w));
  return w;
}

/* ==== fp16 lanes and qf32 (#170, plan 170 section 2) ====================
 *
 * An hf vector is 64 fp16 lanes: lane i is bytes 2i, 2i + 1, so lane 2j is
 * the low half of word j. The widening multiply puts the even lanes in the
 * pair's lo vector and the odd lanes in hi, and the narrowing conversion
 * reads them back the same way (the v79 ISS, plan 170 section 0).
 *
 * THE hf PREMISE: one Vhf add / sub / mpy is one IEEE fp16 operation,
 * round to nearest even, subnormals kept: here the f32 operation on the
 * exact operands rounded once to fp16 by the host's _Float16 conversion
 * (24 >= 2 * 11 + 2 makes the f32 step harmless). The v79 ISS matched
 * rne16(f32 op) bit for bit on every finite a x 2 random b; the device
 * gtest HvxAttnM1Probe.Semantics re-checks it on silicon. A result past
 * 65504 is outside every kernel's domain and reads inf here.
 *
 * qf32 IS NOT MODELLED AS A FORMAT. The hardware's qf32 is a non-IEEE
 * intermediate; what a kernel sees is what the one conversion after it
 * returns. Each op below keeps in its f32 lane the value that makes that
 * conversion right for the one consumer the kernels pair it with:
 *   Wqf32_vmpy_VhfVhf        the exact product (22 bits, >= 2^-48: an f32)
 *   Vqf32_vadd_Vqf32Vqf32    the exact sum rounded to ODD at 24 bits
 *                            (TwoSum and a sticky bit), so Vhf_equals_Wqf32's
 *                            one rounding to fp16 rounds the exact sum
 *                            (round to odd at p >= q + 2 bits, then round
 *                            to q bits, is rounding once): the fused step
 *                            of the CPU's fmla .8h, which hvx_attn_m1_hf.h's
 *                            hvx_hf_fma builds from these three ops
 *   Vqf32_vmpy/vadd/vsub_VsfVsf  RN24 of the exact result, what
 *                            Vsf_equals_Vqf32 then returns (rule 24's IEEE
 *                            sf op, which is how v79 lowers Vsf_vmpy / vadd)
 *   Vsf_equals_Vqf32         the lane as is
 *   Vhf_equals_Wqf32         each lane rounded once to fp16
 * So Vsf_equals_Vqf32 of a Vqf32Vqf32 sum would be WRONG here (the
 * round-to-odd value, where the ISS double-rounds): hvx_attn_m1_hf.h never
 * does that. Whether silicon's qf32 sum narrows like the exact one is plan
 * 170's S1 question; this file assumes it and cannot see otherwise.
 */

typedef _Float16 hvx_emu_f16;

static inline uint16_t hvx_emu_h(const HVX_Vector *v, int i) {
  uint16_t h;
  memcpy(&h, (const uint8_t *)v->w + 2 * i, sizeof(h));
  return h;
}
static inline void hvx_emu_set_h(HVX_Vector *v, int i, uint16_t h) {
  memcpy((uint8_t *)v->w + 2 * i, &h, sizeof(h));
}
static inline float hvx_emu_h2f(uint16_t h) {
  hvx_emu_f16 x;
  memcpy(&x, &h, sizeof(x));
  return (float)x;
}
/** @brief f32 -> fp16 bits, rounded once to nearest even. */
static inline uint16_t hvx_emu_f2h(float f) {
  volatile hvx_emu_f16 x = (hvx_emu_f16)f;
  const hvx_emu_f16 y = x;
  uint16_t h;
  memcpy(&h, &y, sizeof(h));
  return h;
}

static inline HVX_Vector Q6_Vh_vsplat_R(int32_t x) {
  HVX_Vector r;
  for (int i = 0; i < 2 * HVX_EMU_LANES; ++i) {
    hvx_emu_set_h(&r, i, (uint16_t)x);
  }
  return r;
}

#define HVX_EMU_HF_BINOP(name, op)                                             \
  static inline HVX_Vector name(HVX_Vector a, HVX_Vector b) {                  \
    HVX_Vector r;                                                              \
    for (int i = 0; i < 2 * HVX_EMU_LANES; ++i) {                              \
      volatile float t =                                                       \
        hvx_emu_h2f(hvx_emu_h(&a, i)) op hvx_emu_h2f(hvx_emu_h(&b, i));        \
      hvx_emu_set_h(&r, i, hvx_emu_f2h(t));                                    \
    }                                                                          \
    return r;                                                                  \
  }
HVX_EMU_HF_BINOP(Q6_Vhf_vadd_VhfVhf, +)
HVX_EMU_HF_BINOP(Q6_Vhf_vsub_VhfVhf, -)
HVX_EMU_HF_BINOP(Q6_Vhf_vmpy_VhfVhf, *)
#undef HVX_EMU_HF_BINOP

/* max on finite values: the larger value; of two equal values (+-0) the
   second operand, as the sf max above. */
static inline HVX_Vector Q6_Vhf_vmax_VhfVhf(HVX_Vector a, HVX_Vector b) {
  HVX_Vector r;
  for (int i = 0; i < 2 * HVX_EMU_LANES; ++i) {
    const uint16_t x = hvx_emu_h(&a, i), y = hvx_emu_h(&b, i);
    hvx_emu_set_h(&r, i, hvx_emu_h2f(x) > hvx_emu_h2f(y) ? x : y);
  }
  return r;
}

static inline HVX_Vector Q6_V_lo_W(HVX_VectorPair w) { return w.lo; }
static inline HVX_Vector Q6_V_hi_W(HVX_VectorPair w) { return w.hi; }
static inline HVX_VectorPair Q6_W_vcombine_VV(HVX_Vector hi, HVX_Vector lo) {
  HVX_VectorPair w;
  w.lo = lo;
  w.hi = hi;
  return w;
}

/* The exact product of two fp16 values: even lanes to lo, odd to hi. */
static inline HVX_VectorPair Q6_Wqf32_vmpy_VhfVhf(HVX_Vector a, HVX_Vector b) {
  HVX_VectorPair w;
  for (int j = 0; j < HVX_EMU_LANES; ++j) {
    volatile float lo =
      hvx_emu_h2f(hvx_emu_h(&a, 2 * j)) * hvx_emu_h2f(hvx_emu_h(&b, 2 * j));
    volatile float hi = hvx_emu_h2f(hvx_emu_h(&a, 2 * j + 1)) *
                        hvx_emu_h2f(hvx_emu_h(&b, 2 * j + 1));
    w.lo.w[j] = hvx_emu_w(lo);
    w.hi.w[j] = hvx_emu_w(hi);
  }
  return w;
}

/* The exact sum rounded to odd at 24 bits: TwoSum's error term is exact
   (Knuth; no overflow in any kernel's domain), and a nonzero one moves s
   one ulp toward the exact sum's side of it and sets the last bit. */
static inline HVX_Vector Q6_Vqf32_vadd_Vqf32Vqf32(HVX_Vector a, HVX_Vector b) {
  HVX_Vector r;
  for (int i = 0; i < HVX_EMU_LANES; ++i) {
    const float x = hvx_emu_f(a.w[i]), y = hvx_emu_f(b.w[i]);
    volatile float s = x + y;
    volatile float bv = s - x;
    volatile float e1 = x - (s - bv);
    volatile float e2 = y - bv;
    volatile float err = e1 + e2;
    uint32_t u = (uint32_t)hvx_emu_w(s);
    if (err != 0.0f) {
      if (((uint32_t)hvx_emu_w(err) ^ u) & 0x80000000u) {
        u -= 1u;
      }
      u |= 1u;
    }
    r.w[i] = (int32_t)u;
  }
  return r;
}

/* Each lane of the pair rounded once to fp16; lo to the even lanes. */
static inline HVX_Vector Q6_Vhf_equals_Wqf32(HVX_VectorPair w) {
  HVX_Vector r;
  for (int j = 0; j < HVX_EMU_LANES; ++j) {
    hvx_emu_set_h(&r, 2 * j, hvx_emu_f2h(hvx_emu_f(w.lo.w[j])));
    hvx_emu_set_h(&r, 2 * j + 1, hvx_emu_f2h(hvx_emu_f(w.hi.w[j])));
  }
  return r;
}

#define HVX_EMU_QF_BINOP(name, op)                                             \
  static inline HVX_Vector name(HVX_Vector a, HVX_Vector b) {                  \
    HVX_Vector r;                                                              \
    for (int i = 0; i < HVX_EMU_LANES; ++i) {                                  \
      volatile float t = hvx_emu_f(a.w[i]) op hvx_emu_f(b.w[i]);               \
      r.w[i] = hvx_emu_w(t);                                                   \
    }                                                                          \
    return r;                                                                  \
  }
HVX_EMU_QF_BINOP(Q6_Vqf32_vmpy_VsfVsf, *)
HVX_EMU_QF_BINOP(Q6_Vqf32_vadd_VsfVsf, +)
HVX_EMU_QF_BINOP(Q6_Vqf32_vsub_VsfVsf, -)
#undef HVX_EMU_QF_BINOP

static inline HVX_Vector Q6_Vsf_equals_Vqf32(HVX_Vector a) { return a; }

#endif /* __NNTRAINER_HVX_EMU_HVX_HEXAGON_PROTOS_H__ */
