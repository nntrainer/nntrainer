// SPDX-License-Identifier: Apache-2.0
/**
 * @file hvx_hadamard_f32.c
 * @brief Full normalized H44 x Sylvester rotations, without padding.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#include "hvx_hadamard_f32.h"
#include <math.h>
#include <stddef.h>
#include <string.h>

#if defined(__hexagon__)
#include "hvx_convert.h"
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#endif

enum {
  HADAMARD_MODE_SCALAR = 1u,
  HADAMARD_MODE_HVX = 2u,
  HADAMARD_MODE_PALEY_I = 4u
};

/** Independently generated Paley construction over F_43:
 * H[0,0]=1, H[0,j]=-1, H[i,0]=1, H[i,i]=1 and
 * H[i,j]=Legendre(i-j,43) otherwise. Four output signs per byte.
 * This equals SpinQuant get_had44; H is not symmetric, so the inverse
 * requires its transpose. The Sylvester butterflies are symmetric.
 */
static const unsigned char signs[11][44] = {
  {0, 9,  3,  5,  9,  3, 5, 11, 5,  9,  3, 7,  15, 13, 11,
   7, 15, 15, 15, 13, 9, 1, 3,  5,  11, 7, 15, 13, 9,  1,
   1, 1,  3,  5,  9,  1, 3, 7,  13, 11, 5, 11, 7,  13},
  {0,  5,  11, 6,  12, 9,  2,  4,  9,  2, 5, 10, 4,  9,  3,
   7,  14, 13, 11, 7,  15, 15, 14, 12, 8, 1, 2,  5,  11, 7,
   14, 12, 8,  0,  0,  1,  2,  4,  8,  1, 3, 6,  13, 10},
  {0,  3, 6,  13, 10, 5,  11, 6,  12, 9,  2,  4,  9,  2, 5,
   10, 4, 9,  3,  7,  14, 13, 11, 7,  15, 15, 14, 12, 8, 1,
   2,  5, 11, 7,  14, 12, 8,  0,  0,  1,  2,  4,  8,  1},
  {0,  2,  4, 8, 1,  3, 6,  13, 10, 5,  11, 6,  12, 9,  2,
   4,  9,  2, 5, 10, 4, 9,  3,  7,  14, 13, 11, 7,  15, 15,
   14, 12, 8, 1, 2,  5, 11, 7,  14, 12, 8,  0,  0,  1},
  {0,  8,  0,  0,  1,  2,  4, 8, 1,  3, 6,  13, 10, 5,  11,
   6,  12, 9,  2,  4,  9,  2, 5, 10, 4, 9,  3,  7,  14, 13,
   11, 7,  15, 15, 14, 12, 8, 1, 2,  5, 11, 7,  14, 12},
  {0,  11, 7,  14, 12, 8,  0,  0,  1,  2,  4, 8, 1,  3, 6,
   13, 10, 5,  11, 6,  12, 9,  2,  4,  9,  2, 5, 10, 4, 9,
   3,  7,  14, 13, 11, 7,  15, 15, 14, 12, 8, 1, 2,  5},
  {0, 8,  1, 2, 5,  11, 7,  14, 12, 8,  0,  0,  1,  2, 4,
   8, 1,  3, 6, 13, 10, 5,  11, 6,  12, 9,  2,  4,  9, 2,
   5, 10, 4, 9, 3,  7,  14, 13, 11, 7,  15, 15, 14, 12},
  {0, 15, 15, 14, 12, 8,  1, 2, 5,  11, 7,  14, 12, 8,  0,
   0, 1,  2,  4,  8,  1,  3, 6, 13, 10, 5,  11, 6,  12, 9,
   2, 4,  9,  2,  5,  10, 4, 9, 3,  7,  14, 13, 11, 7},
  {0,  14, 13, 11, 7, 15, 15, 14, 12, 8,  1, 2, 5,  11, 7,
   14, 12, 8,  0,  0, 1,  2,  4,  8,  1,  3, 6, 13, 10, 5,
   11, 6,  12, 9,  2, 4,  9,  2,  5,  10, 4, 9, 3,  7},
  {0, 4,  9,  3, 7,  14, 13, 11, 7, 15, 15, 14, 12, 8, 1,
   2, 5,  11, 7, 14, 12, 8,  0,  0, 1,  2,  4,  8,  1, 3,
   6, 13, 10, 5, 11, 6,  12, 9,  2, 4,  9,  2,  5,  10},
  {0,  9,  2, 5, 10, 4,  9,  3, 7,  14, 13, 11, 7, 15, 15,
   14, 12, 8, 1, 2,  5,  11, 7, 14, 12, 8,  0,  0, 1,  2,
   4,  8,  1, 3, 6,  13, 10, 5, 11, 6,  12, 9,  2, 4}};

int hvx_hadamard_supported(uint32_t n) {
  if (!n || n > 8192u)
    return 0;
  if ((n & (n - 1u)) == 0u)
    return 1;
  if (n == 1536u)
    return 1;
  return n % 44u == 0u && ((n / 44u) & (n / 44u - 1u)) == 0u;
}

static int hadamard_mode_supported(uint32_t n, uint32_t mode) {
  return hvx_hadamard_supported(n) &&
         (mode == HADAMARD_MODE_SCALAR || mode == HADAMARD_MODE_HVX ||
          (mode == HADAMARD_MODE_PALEY_I && (n == 2816u || n == 704u)));
}

/** LFM2 SR15's dephased H12. Rows of x multiply H12 on the right;
 * it is not symmetric. The output signs are supplied by the checkpoint. */
static const int8_t lfm_h12[12][12] = {
  {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
  {1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1},
  {1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1},
  {1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1},
  {1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1},
  {1, 1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1},
  {1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, -1},
  {1, -1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1},
  {1, -1, -1, 1, 1, 1, -1, 1, -1, -1, 1, -1},
  {1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, 1},
  {1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1},
  {1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1}};

static void hadamard_lfm1536(float *x) {
  for (uint32_t step = 1; step < 128; step *= 2)
    for (uint32_t base = 0; base < 1536; base += 2 * step)
      for (uint32_t j = 0; j < step; ++j) {
        const float a = x[base + j], b = x[base + j + step];
        x[base + j] = a + b;
        x[base + j + step] = a - b;
      }
  for (uint32_t k = 0; k < 128; ++k) {
    float v[12];
    for (uint32_t i = 0; i < 12; ++i)
      v[i] = x[i * 128 + k];
    for (uint32_t j = 0; j < 12; ++j) {
      float sum = 0;
      for (uint32_t i = 0; i < 12; ++i)
        sum += lfm_h12[i][j] * v[i];
      x[j * 128 + k] = sum * 0.025515519082546234f;
    }
  }
}

static void hadamard_scalar(float *x, uint32_t n, uint32_t p) {
  for (uint32_t step = 1u; step < p; step *= 2u)
    for (uint32_t base = 0u; base < n; base += 2u * step)
      for (uint32_t j = 0u; j < step; ++j) {
        const float a = x[base + j], b = x[base + j + step];
        x[base + j] = a + b;
        x[base + j + step] = a - b;
      }
  const float scale = 1.0f / sqrtf((float)n);
  if (p == n) {
    for (uint32_t i = 0u; i < n; ++i)
      x[i] *= scale;
    return;
  }
  for (uint32_t k = 0u; k < p; ++k) {
    float in[44];
    for (uint32_t j = 0u; j < 44u; ++j)
      in[j] = x[j * p + k];
    for (uint32_t i = 0u; i < 44u; ++i) {
      float sum = 0.0f;
      for (uint32_t j = 0u; j < 44u; ++j)
        sum += (signs[i / 4u][j] & (1u << (i % 4u))) ? -in[j] : in[j];
      x[i * p + k] = sum * scale;
    }
  }
}

#if defined(__hexagon__)
/** Reuse the eight signed sums of each four input blocks. H44 needs
 * 11 selections per output instead of 44 signed additions. Selectors
 * hold four output bytes: low three bits choose a sum, bit 3 its sign. */
static const uint32_t selectors[11][11] = {
  {0x01040207u, 0x0b020508u, 0x010b0608u, 0x0c0a0908u, 0x08080c08u, 0x0f040208u,
   0x09070c08u, 0x0f000008u, 0x02010f08u, 0x06030a08u, 0x05090308u},
  {0x02050603u, 0x0e0b0201u, 0x0b02050du, 0x010b0607u, 0x0c0a0907u, 0x08080c0eu,
   0x0f040205u, 0x09070c0eu, 0x0f000004u, 0x02010f04u, 0x06030a0du},
  {0x04060305u, 0x0d050903u, 0x0e0b0201u, 0x0b02050du, 0x010b0607u, 0x0c0a0907u,
   0x08080c0eu, 0x0f040205u, 0x09070c0eu, 0x0f000004u, 0x02010f04u},
  {0x04020100u, 0x0406030au, 0x0d050903u, 0x0e0b0201u, 0x0b02050du, 0x010b0607u,
   0x0c0a0907u, 0x08080c0eu, 0x0f040205u, 0x09070c0eu, 0x0f000004u},
  {0x01000000u, 0x0402010fu, 0x0406030au, 0x0d050903u, 0x0e0b0201u, 0x0b02050du,
   0x010b0607u, 0x0c0a0907u, 0x08080c0eu, 0x0f040205u, 0x09070c0eu},
  {0x05060703u, 0x0e0f0000u, 0x0402010fu, 0x0406030au, 0x0d050903u, 0x0e0b0201u,
   0x0b02050du, 0x010b0607u, 0x0c0a0907u, 0x08080c0eu, 0x0f040205u},
  {0x01000402u, 0x0509070cu, 0x0e0f0000u, 0x0402010fu, 0x0406030au, 0x0d050903u,
   0x0e0b0201u, 0x0b02050du, 0x010b0607u, 0x0c0a0907u, 0x08080c0eu},
  {0x07070703u, 0x0e0f0402u, 0x0509070cu, 0x0e0f0000u, 0x0402010fu, 0x0406030au,
   0x0d050903u, 0x0e0b0201u, 0x0b02050du, 0x010b0607u, 0x0c0a0907u},
  {0x07030506u, 0x0708080cu, 0x0e0f0402u, 0x0509070cu, 0x0e0f0000u, 0x0402010fu,
   0x0406030au, 0x0d050903u, 0x0e0b0201u, 0x0b02050du, 0x010b0607u},
  {0x02010406u, 0x070c0a09u, 0x0708080cu, 0x0e0f0402u, 0x0509070cu, 0x0e0f0000u,
   0x0402010fu, 0x0406030au, 0x0d050903u, 0x0e0b0201u, 0x0b02050du},
  {0x01040205u, 0x0d010b06u, 0x070c0a09u, 0x0708080cu, 0x0e0f0402u, 0x0509070cu,
   0x0e0f0000u, 0x0402010fu, 0x0406030au, 0x0d050903u, 0x0e0b0201u}};

#include "hvx_hadamard_h44.inc"

static inline HVX_Vector butterfly(HVX_Vector v, HVX_Vector lane,
                                   uint32_t step) {
  const HVX_Vector bit = Q6_V_vsplat_R((int)step);
  const HVX_VectorPred upper = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(lane, bit), bit);
  const HVX_Vector sign =
    Q6_V_vmux_QVV(upper, Q6_V_vsplat_R((int)0x80000000u), Q6_V_vzero());
  const HVX_Vector other =
    Q6_V_vdelta_VV(v, Q6_V_vsplat_R((int)(0x04040404u * step)));
  return Q6_Vsf_vadd_VsfVsf(Q6_V_vxor_VV(v, sign), other);
}

/** Keep eight butterfly vectors in registers. The low five stages are
 * lane shuffles, then three register stages; a second pass joins the eight
 * 256-wide blocks. Each output keeps exactly the scalar butterfly order. */
/** Extrema may be reduced in final-store order: min/max propagate NaNs,
 * while the caller's quantization clamps all NaN/zero extrema to +0. */
static inline __attribute__((always_inline)) void
hadamard_store_extrema(HVX_Vector vmin, HVX_Vector vmax, float *extrema) {
  for (uint32_t rot = 64u; rot >= 4u; rot >>= 1u) {
    vmin = Q6_Vsf_vmin_VsfVsf(vmin, Q6_V_vror_VR(vmin, (int)rot));
    vmax = Q6_Vsf_vmax_VsfVsf(vmax, Q6_V_vror_VR(vmax, (int)rot));
  }
  float mins[32], maxs[32];
  *(HVX_UVector *)mins = vmin;
  *(HVX_UVector *)maxs = vmax;
  extrema[0] = mins[0];
  extrema[1] = maxs[0];
}

static inline __attribute__((always_inline)) void
hadamard_2048_hvx_core(float *x, const uint32_t *mask, float *extrema) {
  static const int32_t lanes[32] __attribute__((aligned(128))) = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
  const HVX_Vector lane = *(const HVX_Vector *)lanes;
  for (uint32_t base = 0; base < 2048u; base += 256u) {
    HVX_Vector v[8];
#pragma clang loop unroll(full)
    for (uint32_t i = 0; i < 8u; ++i) {
      HVX_Vector a = *(const HVX_Vector *)(x + base + i * 32u);
      a = butterfly(a, lane, 1u);
      a = butterfly(a, lane, 2u);
      a = butterfly(a, lane, 4u);
      a = butterfly(a, lane, 8u);
      v[i] = butterfly(a, lane, 16u);
    }
#pragma clang loop unroll(full)
    for (uint32_t step = 1u; step < 8u; step *= 2u)
#pragma clang loop unroll(full)
      for (uint32_t i = 0; i < 8u; ++i)
        if (!(i & step)) {
          const HVX_Vector a = v[i], b = v[i + step];
          v[i] = Q6_Vsf_vadd_VsfVsf(a, b);
          v[i + step] = Q6_Vsf_vsub_VsfVsf(a, b);
        }
#pragma clang loop unroll(full)
    for (uint32_t i = 0; i < 8u; ++i)
      *(HVX_Vector *)(x + base + i * 32u) = v[i];
  }
  HVX_Vector vmin = Q6_V_vzero(), vmax = vmin;
  const HVX_Vector scale = hvx_splat_sf(0.022097086533904076f);
  for (uint32_t k = 0; k < 256u; k += 32u) {
    HVX_Vector v[8];
#pragma clang loop unroll(full)
    for (uint32_t i = 0; i < 8u; ++i)
      v[i] = *(const HVX_Vector *)(x + i * 256u + k);
#pragma clang loop unroll(full)
    for (uint32_t step = 1u; step < 8u; step *= 2u)
#pragma clang loop unroll(full)
      for (uint32_t i = 0; i < 8u; ++i)
        if (!(i & step)) {
          const HVX_Vector a = v[i], b = v[i + step];
          v[i] = Q6_Vsf_vadd_VsfVsf(a, b);
          v[i + step] = Q6_Vsf_vsub_VsfVsf(a, b);
        }
#pragma clang loop unroll(full)
    for (uint32_t i = 0; i < 8u; ++i) {
      HVX_Vector value = Q6_Vsf_vmpy_VsfVsf(v[i], scale);
      if (mask)
        value =
          Q6_V_vxor_VV(value, *(const HVX_UVector *)(mask + i * 256u + k));
      if (extrema) {
        vmin = Q6_Vsf_vmin_VsfVsf(vmin, value);
        vmax = Q6_Vsf_vmax_VsfVsf(vmax, value);
      }
      *(HVX_Vector *)(x + i * 256u + k) = value;
    }
  }
  if (extrema)
    hadamard_store_extrema(vmin, vmax, extrema);
}

static void hadamard_2048_hvx(float *x, const uint32_t *mask) {
  hadamard_2048_hvx_core(x, mask, NULL);
}

static inline __attribute__((always_inline)) void
hadamard_hvx(float *x, uint32_t n, uint32_t p, uint32_t pair,
             const uint32_t *mask, float *extrema) {
  static const int32_t lanes[32] __attribute__((aligned(128))) = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
  const HVX_Vector lane = *(const HVX_Vector *)lanes;
  if (n == 1536u) {
    /** Keep each H128 block in four registers through all seven stages.
     * This removes the two intermediate row-memory passes for stages32/64
     * while keeping each output's butterfly order unchanged. */
    for (uint32_t base = 0u; base < 1536u; base += 128u) {
      HVX_Vector v[4];
#pragma clang loop unroll(full)
      for (uint32_t i = 0; i < 4u; ++i) {
        HVX_Vector a = *(const HVX_Vector *)(x + base + i * 32u);
        a = butterfly(a, lane, 1u);
        a = butterfly(a, lane, 2u);
        a = butterfly(a, lane, 4u);
        a = butterfly(a, lane, 8u);
        v[i] = butterfly(a, lane, 16u);
      }
#pragma clang loop unroll(full)
      for (uint32_t step = 1u; step < 4u; step *= 2u)
#pragma clang loop unroll(full)
        for (uint32_t i = 0; i < 4u; ++i)
          if (!(i & step)) {
            const HVX_Vector a = v[i], b = v[i + step];
            v[i] = Q6_Vsf_vadd_VsfVsf(a, b);
            v[i + step] = Q6_Vsf_vsub_VsfVsf(a, b);
          }
#pragma clang loop unroll(full)
      for (uint32_t i = 0; i < 4u; ++i)
        *(HVX_Vector *)(x + base + i * 32u) = v[i];
    }
  } else {
    /** Each register contains one or two independent power-of-two blocks.
     * XOR shuffles implement butterfly partners; the upper member changes
     * sign before the add. There are no scalar lane loads/stores. */
    for (uint32_t base = 0u; base < n * (1u + pair); base += 32u) {
      HVX_Vector v = *(const HVX_Vector *)(x + base);
      /** p >= 16 on this path. Literal stages let the compiler hoist all
       * shuffle controls and sign masks out of the register loop. */
      v = butterfly(v, lane, 1u);
      v = butterfly(v, lane, 2u);
      v = butterfly(v, lane, 4u);
      v = butterfly(v, lane, 8u);
      if (p >= 32u)
        v = butterfly(v, lane, 16u);
      *(HVX_Vector *)(x + base) = v;
    }
    for (uint32_t step = 32u; step < p; step *= 2u)
      for (uint32_t base = 0u; base < n; base += 2u * step)
        for (uint32_t j = 0u; j < step; j += 32u) {
          const HVX_Vector a = *(const HVX_Vector *)(x + base + j);
          const HVX_Vector b = *(const HVX_Vector *)(x + base + j + step);
          *(HVX_Vector *)(x + base + j) = Q6_Vsf_vadd_VsfVsf(a, b);
          *(HVX_Vector *)(x + base + j + step) = Q6_Vsf_vsub_VsfVsf(a, b);
        }
  }
  const HVX_Vector scale =
    hvx_splat_sf(n == 1536u ? 0.025515519082546234f : 1.0f / sqrtf((float)n));
  if (p == n) {
    for (uint32_t i = 0u; i < n; i += 32u)
      *(HVX_Vector *)(x + i) =
        Q6_Vsf_vmpy_VsfVsf(*(const HVX_Vector *)(x + i), scale);
    return;
  }
  if (n == 1536u) {
    /** Twelve independent accumulators hide the add latency without
     * changing the order within any H12 dot. Reading one input at a time
     * also avoids materializing twelve vector inputs on the worker stack. */
    const HVX_Vector sign_bit = Q6_V_vsplat_R((int)0x80000000u);
    HVX_Vector vmin = Q6_V_vzero(), vmax = vmin;
    for (uint32_t k = 0; k < 128u; k += 32u) {
      HVX_Vector sums[12];
#pragma clang loop unroll(full)
      for (uint32_t j = 0; j < 12u; ++j)
        sums[j] = Q6_V_vzero();
#pragma clang loop unroll(full)
      for (uint32_t i = 0; i < 12u; ++i) {
        const HVX_Vector in = *(const HVX_Vector *)(x + i * 128u + k);
        const HVX_Vector neg = Q6_V_vxor_VV(in, sign_bit);
#pragma clang loop unroll(full)
        for (uint32_t j = 0; j < 12u; ++j)
          sums[j] = Q6_Vsf_vadd_VsfVsf(sums[j], lfm_h12[i][j] < 0 ? neg : in);
      }
#pragma clang loop unroll(full)
      for (uint32_t j = 0; j < 12u; ++j) {
        HVX_Vector value = Q6_Vsf_vmpy_VsfVsf(sums[j], scale);
        if (mask)
          value =
            Q6_V_vxor_VV(value, *(const HVX_UVector *)(mask + j * 128u + k));
        if (extrema) {
          vmin = Q6_Vsf_vmin_VsfVsf(vmin, value);
          vmax = Q6_Vsf_vmax_VsfVsf(vmax, value);
        }
        *(HVX_Vector *)(x + j * 128u + k) = value;
      }
    }
    if (extrema)
      hadamard_store_extrema(vmin, vmax, extrema);
    return;
  }
  if (n == 2816u || n == 704u) {
    HVX_Vector out[44];
    const uint32_t bits = Q6_R_vextract_VR(scale, 0);
    for (uint32_t k = 0; k < p; k += 32) {
      if (pair)
        had44_pair(x, out, bits);
      else if (n == 704u)
        had44_single(x, out, bits);
      else
        had44_32(x + k, out, bits);
      if (n == 704u) {
        const HVX_VectorPred low = Q6_Q_vsetq_R(64);
        for (uint32_t i = 0; i < 44; i += 2) {
          *(HVX_Vector *)(x + i * 16u) =
            Q6_V_vmux_QVV(low, out[i], Q6_V_vror_VR(out[i + 1], 64));
          if (pair)
            *(HVX_Vector *)(x + 704u + i * 16u) =
              Q6_V_vmux_QVV(low, Q6_V_vror_VR(out[i], 64), out[i + 1]);
        }
      } else {
        for (uint32_t i = 0; i < 44; ++i)
          *(HVX_Vector *)(x + i * p + k) = out[i];
      }
    }
    return;
  }
  for (uint32_t k = 0u; k < p; k += 32u) {
    HVX_Vector sums[88];
    for (uint32_t j = 0u; j < 44u; j += 4u) {
      HVX_Vector a, b, c, d;
      if (p == 16u) {
        a = *(const HVX_Vector *)(x + j * p);
        b = Q6_V_vror_VR(a, 64);
        c = *(const HVX_Vector *)(x + (j + 2u) * p);
        d = Q6_V_vror_VR(c, 64);
        if (pair) {
          const HVX_Vector ab = *(const HVX_Vector *)(x + n + j * p);
          const HVX_Vector cd = *(const HVX_Vector *)(x + n + (j + 2u) * p);
          const HVX_VectorPred low = Q6_Q_vsetq_R(64);
          a = Q6_V_vmux_QVV(low, a, Q6_V_vror_VR(ab, 64));
          b = Q6_V_vmux_QVV(low, b, ab);
          c = Q6_V_vmux_QVV(low, c, Q6_V_vror_VR(cd, 64));
          d = Q6_V_vmux_QVV(low, d, cd);
        }
      } else {
        a = *(const HVX_Vector *)(x + (j + 0u) * p + k);
        b = *(const HVX_Vector *)(x + (j + 1u) * p + k);
        c = *(const HVX_Vector *)(x + (j + 2u) * p + k);
        d = *(const HVX_Vector *)(x + (j + 3u) * p + k);
      }
      const HVX_Vector ab0 = Q6_Vsf_vadd_VsfVsf(a, b);
      const HVX_Vector ab1 = Q6_Vsf_vsub_VsfVsf(a, b);
      const HVX_Vector cd0 = Q6_Vsf_vadd_VsfVsf(c, d);
      const HVX_Vector cd1 = Q6_Vsf_vsub_VsfVsf(c, d);
      HVX_Vector *t = sums + 2u * j;
      t[0] = Q6_Vsf_vadd_VsfVsf(ab0, cd0);
      t[1] = Q6_Vsf_vadd_VsfVsf(ab1, cd0);
      t[2] = Q6_Vsf_vsub_VsfVsf(ab0, cd1);
      t[3] = Q6_Vsf_vsub_VsfVsf(ab1, cd1);
      t[4] = Q6_Vsf_vadd_VsfVsf(ab0, cd1);
      t[5] = Q6_Vsf_vadd_VsfVsf(ab1, cd1);
      t[6] = Q6_Vsf_vsub_VsfVsf(ab0, cd0);
      t[7] = Q6_Vsf_vsub_VsfVsf(ab1, cd0);
    }
#pragma clang loop unroll(full)
    for (uint32_t i = 0u; i < 44u; i += 4u) {
      HVX_Vector a = Q6_V_vzero(), b = a, c = a, d = a;
#pragma clang loop unroll(full)
      for (uint32_t j = 0u; j < 11u; ++j) {
        const uint32_t select = selectors[i / 4u][j];
        const uint32_t sa = select & 15u, sb = (select >> 8u) & 15u;
        const uint32_t sc = (select >> 16u) & 15u, sd = select >> 24u;
        const HVX_Vector *t = sums + j * 8u;
        a = (sa & 8u) ? Q6_Vsf_vsub_VsfVsf(a, t[sa & 7u])
                      : Q6_Vsf_vadd_VsfVsf(a, t[sa & 7u]);
        b = (sb & 8u) ? Q6_Vsf_vsub_VsfVsf(b, t[sb & 7u])
                      : Q6_Vsf_vadd_VsfVsf(b, t[sb & 7u]);
        c = (sc & 8u) ? Q6_Vsf_vsub_VsfVsf(c, t[sc & 7u])
                      : Q6_Vsf_vadd_VsfVsf(c, t[sc & 7u]);
        d = (sd & 8u) ? Q6_Vsf_vsub_VsfVsf(d, t[sd & 7u])
                      : Q6_Vsf_vadd_VsfVsf(d, t[sd & 7u]);
      }
      a = Q6_Vsf_vmpy_VsfVsf(a, scale);
      b = Q6_Vsf_vmpy_VsfVsf(b, scale);
      c = Q6_Vsf_vmpy_VsfVsf(c, scale);
      d = Q6_Vsf_vmpy_VsfVsf(d, scale);
      if (p == 16u) {
        const HVX_VectorPred low = Q6_Q_vsetq_R(64);
        *(HVX_Vector *)(x + i * p) = Q6_V_vmux_QVV(low, a, Q6_V_vror_VR(b, 64));
        *(HVX_Vector *)(x + (i + 2u) * p) =
          Q6_V_vmux_QVV(low, c, Q6_V_vror_VR(d, 64));
        if (pair) {
          *(HVX_Vector *)(x + n + i * p) =
            Q6_V_vmux_QVV(low, Q6_V_vror_VR(a, 64), b);
          *(HVX_Vector *)(x + n + (i + 2u) * p) =
            Q6_V_vmux_QVV(low, Q6_V_vror_VR(c, 64), d);
        }
      } else {
        *(HVX_Vector *)(x + (i + 0u) * p + k) = a;
        *(HVX_Vector *)(x + (i + 1u) * p + k) = b;
        *(HVX_Vector *)(x + (i + 2u) * p + k) = c;
        *(HVX_Vector *)(x + (i + 3u) * p + k) = d;
      }
    }
  }
}
#endif

/** Preserve the first H44 block and negate the remaining input blocks. */
static void hadamard_paley_prepare(float *x, uint32_t n) {
  uint32_t i = n / 44u;
#if defined(__hexagon__)
  if (!((uintptr_t)x & 127u)) {
    const HVX_Vector sign = Q6_V_vsplat_R((int32_t)0x80000000u);
    const uint32_t first = i & ~31u;
    if (i != first) {
      const HVX_VectorPred keep = Q6_Q_vsetq_R((int)((i - first) * 4u));
      const HVX_Vector mask = Q6_V_vmux_QVV(keep, Q6_V_vzero(), sign);
      *(HVX_Vector *)(x + first) =
        Q6_V_vxor_VV(*(const HVX_Vector *)(x + first), mask);
      i = first + 32u;
    }
    for (; n - i >= 32u; i += 32u)
      *(HVX_Vector *)(x + i) = Q6_V_vxor_VV(*(const HVX_Vector *)(x + i), sign);
  }
#endif
  for (; i < n; ++i)
    x[i] = -x[i];
}

int hvx_hadamard_f32(float *x, uint32_t n, uint32_t mode) {
  if (!x || !hadamard_mode_supported(n, mode))
    return -1;
  /* Paley-I H = diag(1,-1,...,-1) * SpinQuant H^T. */
  if (mode == HADAMARD_MODE_PALEY_I) {
    hadamard_paley_prepare(x, n);
    mode = HADAMARD_MODE_HVX;
  }
  if (n == 1536u) {
#if defined(__hexagon__)
    if (mode == HADAMARD_MODE_HVX && !((uintptr_t)x & 127u))
      hadamard_hvx(x, 1536u, 128u, 0u, NULL, NULL);
    else
#endif
      hadamard_lfm1536(x);
    return 0;
  }
  const uint32_t p = (n & (n - 1u)) == 0u ? n : n / 44u;
#if defined(__hexagon__)
  if (mode == HADAMARD_MODE_HVX && n % 32u == 0u && p >= 16u) {
    if ((uintptr_t)x & 127u) {
      hadamard_scalar(x, n, p);
    } else if (n == 2048u) {
      hadamard_2048_hvx(x, NULL);
    } else if (n == 704u) {
      hadamard_hvx(x, 704u, 16u, 0u, NULL, NULL);
    } else if (n == 2816u) {
      hadamard_hvx(x, 2816u, 64u, 0u, NULL, NULL);
    } else {
      hadamard_hvx(x, n, p, 0u, NULL, NULL);
    }
    return 0;
  }
#endif
  hadamard_scalar(x, n, p);
  return 0;
}

int hvx_hadamard_rows_f32(float *x, uint32_t rows, uint32_t n, uint32_t mode) {
  if (!x || !hadamard_mode_supported(n, mode) ||
      rows > SIZE_MAX / sizeof(float) / n)
    return -1;
  if (mode == HADAMARD_MODE_PALEY_I) {
    for (uint32_t r = 0; r < rows; ++r)
      hadamard_paley_prepare(x + (size_t)r * n, n);
    /* The prepared rows use the existing two-row H704 kernel unchanged. */
    mode = HADAMARD_MODE_HVX;
  }
  uint32_t m = 0u;
#if defined(__hexagon__)
  if (mode == HADAMARD_MODE_HVX && n == 704u && !((uintptr_t)x & 127u))
    for (; rows - m >= 2u; m += 2u)
      hadamard_hvx(x + (size_t)m * 704u, 704u, 16u, 1u, NULL, NULL);
#endif
  for (; m < rows; ++m)
    (void)hvx_hadamard_f32(x + (size_t)m * n, n, mode);
  return 0;
}

/** XOR is deliberately after normalization: multiplying a signed scale instead
 * would change exceptional values and their sign/payload behavior. */
static void hadamard_xor_row(float *x, const uint32_t *mask, uint32_t n) {
  uint32_t i = 0;
#if defined(__hexagon__)
  for (; n - i >= 32u; i += 32u)
    *(HVX_UVector *)(x + i) = Q6_V_vxor_VV(*(const HVX_UVector *)(x + i),
                                           *(const HVX_UVector *)(mask + i));
#endif
  for (; i < n; ++i) {
    uint32_t value;
    memcpy(&value, x + i, sizeof(value));
    value ^= mask[i];
    memcpy(x + i, &value, sizeof(value));
  }
}

int hvx_hadamard_signed_f32(float *x, uint32_t n, uint32_t mode,
                            const uint32_t *mask) {
  if (!mask)
    return hvx_hadamard_f32(x, n, mode);
#if defined(__hexagon__)
  if (x && mode == HADAMARD_MODE_HVX && !((uintptr_t)x & 127u)) {
    if (n == 2048u) {
      hadamard_2048_hvx(x, mask);
      return 0;
    }
    if (n == 1536u) {
      hadamard_hvx(x, 1536u, 128u, 0u, mask, NULL);
      return 0;
    }
  }
#endif
  const int result = hvx_hadamard_f32(x, n, mode);
  if (!result)
    hadamard_xor_row(x, mask, n);
  return result;
}

int hvx_hadamard_signed_rows_f32(float *x, uint32_t rows, uint32_t n,
                                 uint32_t mode, const uint32_t *mask) {
  if (!mask)
    return hvx_hadamard_rows_f32(x, rows, n, mode);
  if (!x || !hadamard_mode_supported(n, mode) ||
      rows > SIZE_MAX / sizeof(float) / n)
    return -1;
#if defined(__hexagon__)
  if (mode == HADAMARD_MODE_HVX && !((uintptr_t)x & 127u)) {
    if (n == 2048u) {
      for (uint32_t r = 0; r < rows; ++r)
        hadamard_2048_hvx(x + (size_t)r * n, mask);
      return 0;
    }
    if (n == 1536u) {
      for (uint32_t r = 0; r < rows; ++r)
        hadamard_hvx(x + (size_t)r * n, 1536u, 128u, 0u, mask, NULL);
      return 0;
    }
  }
#endif
  const int result = hvx_hadamard_rows_f32(x, rows, n, mode);
  if (!result)
    for (uint32_t r = 0; r < rows; ++r)
      hadamard_xor_row(x + (size_t)r * n, mask, n);
  return result;
}

#if defined(__hexagon__)
/** Keep each row's two extrema live only within its transform. Inlining
 * these into the row loop spills the extrema inside H12's four iterations. */
static __attribute__((noinline, nonnull(1, 3))) void
hadamard_extrema_1536_hvx(float *x, const uint32_t *mask, float *out) {
  hadamard_hvx(x, 1536u, 128u, 0u, mask, out);
}
static __attribute__((noinline, nonnull(1, 3))) void
hadamard_extrema_2048_hvx(float *x, const uint32_t *mask, float *out) {
  hadamard_2048_hvx_core(x, mask, out);
}
#endif
int hvx_hadamard_signed_rows_extrema_f32(float *x, uint32_t rows, uint32_t n,
                                         const uint32_t *mask, float *extrema) {
  if (!x || !extrema || (n != 1536u && n != 2048u) || ((uintptr_t)x & 127u) ||
      rows > SIZE_MAX / sizeof(float) / n)
    return -1;
  for (uint32_t r = 0; r < rows; ++r) {
    float *row = x + (size_t)r * n;
    float *out = extrema + (size_t)r * 2u;
#if defined(__hexagon__)
    if (n == 2048u)
      hadamard_extrema_2048_hvx(row, mask, out);
    else
      hadamard_extrema_1536_hvx(row, mask, out);
#else
    (void)hvx_hadamard_signed_f32(row, n, HADAMARD_MODE_HVX, mask);
    float vmin = 0.0f, vmax = 0.0f;
    for (uint32_t k = 0; k < n; ++k) {
      if (isnan(row[k])) {
        vmin = vmax = row[k];
        break;
      }
      if (row[k] < vmin)
        vmin = row[k];
      if (row[k] > vmax)
        vmax = row[k];
    }
    out[0] = vmin;
    out[1] = vmax;
#endif
  }
  return 0;
}
