// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (c) 2023-2024 The ggml authors
 *
 * Portions of this file are derived from llama.cpp
 * (https://github.com/ggml-org/llama.cpp), licensed under the MIT License.
 * Copyright (c) Contributors to llama.cpp
 *
 * Modified by Pawel Debski, 2025: Adapted for CPU backend integration
 *
 * @file   nntr_ggml_impl_quant.cpp
 * @date   20 August 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author  Pawel Debski <p.debski2@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Implementations of GGML quantization functions
 */

#include <nntr_ggml_impl.h>
#include <nntr_ggml_impl_utils.h>

#include <assert.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>

//
// ===================== Helper functions
//

static float make_qx_quants(int n, int nmax, const float *__restrict x,
                            int8_t *__restrict L, int rmse_type,
                            const float *__restrict qw) {
  float max = 0;
  float amax = 0;
  for (int i = 0; i < n; ++i) {
    float ax = fabsf(x[i]);
    if (ax > amax) {
      amax = ax;
      max = x[i];
    }
  }
  if (amax < GROUP_MAX_EPS) { // all zero
    for (int i = 0; i < n; ++i) {
      L[i] = 0;
    }
    return 0.f;
  }
  float iscale = -nmax / max;
  if (rmse_type == 0) {
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale * x[i]);
      L[i] = nmax + MAX(-nmax, MIN(nmax - 1, l));
    }
    return 1 / iscale;
  }
  bool return_early = false;
  if (rmse_type < 0) {
    rmse_type = -rmse_type;
    return_early = true;
  }
  float sumlx = 0;
  float suml2 = 0;
#ifdef HAVE_BUGGY_APPLE_LINKER
  // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
  for (volatile int i = 0; i < n; ++i) {
#else
  for (int i = 0; i < n; ++i) {
#endif
    int l = nearest_int(iscale * x[i]);
    l = MAX(-nmax, MIN(nmax - 1, l));
    L[i] = l + nmax;
    float w = qw               ? qw[i]
              : rmse_type == 1 ? x[i] * x[i]
              : rmse_type == 2 ? 1
              : rmse_type == 3 ? fabsf(x[i])
                               : sqrtf(fabsf(x[i]));
    sumlx += w * x[i] * l;
    suml2 += w * l * l;
  }
  float scale = suml2 ? sumlx / suml2 : 0.0f;
  if (return_early)
    return suml2 > 0 ? 0.5f * (scale + 1 / iscale) : 1 / iscale;
  float best = scale * sumlx;
  for (int is = -9; is <= 9; ++is) {
    if (is == 0) {
      continue;
    }
    iscale = -(nmax + 0.1f * is) / max;
    sumlx = suml2 = 0;
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale * x[i]);
      l = MAX(-nmax, MIN(nmax - 1, l));
      float w = qw               ? qw[i]
                : rmse_type == 1 ? x[i] * x[i]
                : rmse_type == 2 ? 1
                : rmse_type == 3 ? fabsf(x[i])
                                 : sqrtf(fabsf(x[i]));
      sumlx += w * x[i] * l;
      suml2 += w * l * l;
    }
    if (suml2 > 0 && sumlx * sumlx > best * suml2) {
      for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        L[i] = nmax + MAX(-nmax, MIN(nmax - 1, l));
      }
      scale = sumlx / suml2;
      best = scale * sumlx;
    }
  }
  return scale;
}

static float make_qkx2_quants(int n, int nmax, const float *__restrict x,
                              const float *__restrict weights,
                              uint8_t *__restrict L, float *__restrict the_min,
                              uint8_t *__restrict Laux, float rmin,
                              float rdelta, int nstep, bool use_mad) {
  float min = x[0];
  float max = x[0];
  float sum_w = weights[0];
  float sum_x = sum_w * x[0];
#ifdef HAVE_BUGGY_APPLE_LINKER
  // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
  for (volatile int i = 1; i < n; ++i) {
#else
  for (int i = 1; i < n; ++i) {
#endif
    if (x[i] < min)
      min = x[i];
    if (x[i] > max)
      max = x[i];
    float w = weights[i];
    sum_w += w;
    sum_x += w * x[i];
  }
  if (min > 0)
    min = 0;
  if (max == min) {
    for (int i = 0; i < n; ++i)
      L[i] = 0;
    *the_min = -min;
    return 0.f;
  }
  float iscale = nmax / (max - min);
  float scale = 1 / iscale;
  float best_mad = 0;
  for (int i = 0; i < n; ++i) {
    int l = nearest_int(iscale * (x[i] - min));
    L[i] = MAX(0, MIN(nmax, l));
    float diff = scale * L[i] + min - x[i];
    diff = use_mad ? fabsf(diff) : diff * diff;
    float w = weights[i];
    best_mad += w * diff;
  }
  if (nstep < 1) {
    *the_min = -min;
    return scale;
  }
  for (int is = 0; is <= nstep; ++is) {
    iscale = (rmin + rdelta * is + nmax) / (max - min);
    float sum_l = 0, sum_l2 = 0, sum_xl = 0;
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale * (x[i] - min));
      l = MAX(0, MIN(nmax, l));
      Laux[i] = l;
      float w = weights[i];
      sum_l += w * l;
      sum_l2 += w * l * l;
      sum_xl += w * l * x[i];
    }
    float D = sum_w * sum_l2 - sum_l * sum_l;
    if (D > 0) {
      float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
      float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
      if (this_min > 0) {
        this_min = 0;
        this_scale = sum_xl / sum_l2;
      }
      float mad = 0;
      for (int i = 0; i < n; ++i) {
        float diff = this_scale * Laux[i] + this_min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights[i];
        mad += w * diff;
      }
      if (mad < best_mad) {
        for (int i = 0; i < n; ++i) {
          L[i] = Laux[i];
        }
        best_mad = mad;
        scale = this_scale;
        min = this_min;
      }
    }
  }
  *the_min = -min;
  return scale;
}

static float make_qkx3_quants(int n, int nmax, const float *__restrict x,
                              const float *__restrict weights,
                              uint8_t *__restrict L, float *__restrict the_min,
                              uint8_t *__restrict Laux, float rmin,
                              float rdelta, int nstep, bool use_mad) {
  float min = x[0];
  float max = x[0];
  float sum_w = weights ? weights[0] : x[0] * x[0];
  float sum_x = sum_w * x[0];
#ifdef HAVE_BUGGY_APPLE_LINKER
  // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
  for (volatile int i = 1; i < n; ++i) {
#else
  for (int i = 1; i < n; ++i) {
#endif
    if (x[i] < min)
      min = x[i];
    if (x[i] > max)
      max = x[i];
    float w = weights ? weights[i] : x[i] * x[i];
    sum_w += w;
    sum_x += w * x[i];
  }
  if (min > 0) {
    min = 0;
  }
  if (max <= min) {
    memset(L, 0, n);
    *the_min = -min;
    return 0.f;
  }
  float iscale = nmax / (max - min);
  float scale = 1 / iscale;
  float best_mad = 0;
  for (int i = 0; i < n; ++i) {
    int l = nearest_int(iscale * (x[i] - min));
    L[i] = MAX(0, MIN(nmax, l));
    float diff = scale * L[i] + min - x[i];
    diff = use_mad ? fabsf(diff) : diff * diff;
    float w = weights ? weights[i] : x[i] * x[i];
    best_mad += w * diff;
  }
  if (nstep < 1) {
    *the_min = -min;
    return scale;
  }
  for (int is = 0; is <= nstep; ++is) {
    iscale = (rmin + rdelta * is + nmax) / (max - min);
    float sum_l = 0, sum_l2 = 0, sum_xl = 0;
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale * (x[i] - min));
      l = MAX(0, MIN(nmax, l));
      Laux[i] = l;
      float w = weights ? weights[i] : x[i] * x[i];
      sum_l += w * l;
      sum_l2 += w * l * l;
      sum_xl += w * l * x[i];
    }
    float D = sum_w * sum_l2 - sum_l * sum_l;
    if (D > 0) {
      float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
      float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
      if (this_min > 0) {
        this_min = 0;
        this_scale = sum_xl / sum_l2;
      }
      float mad = 0;
      for (int i = 0; i < n; ++i) {
        float diff = this_scale * Laux[i] + this_min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights ? weights[i] : x[i] * x[i];
        mad += w * diff;
      }
      if (mad < best_mad) {
        for (int i = 0; i < n; ++i) {
          L[i] = Laux[i];
        }
        best_mad = mad;
        scale = this_scale;
        min = this_min;
      }
    }
  }
  *the_min = -min;
  return scale;
}

static float make_qp_quants(int n, int nmax, const float *__restrict x,
                            uint8_t *__restrict L, const float *quant_weights) {
  float max = 0;
  for (int i = 0; i < n; ++i) {
    max = MAX(max, x[i]);
  }
  if (!max) { // all zero
    for (int i = 0; i < n; ++i) {
      L[i] = 0;
    }
    return 0.f;
  }
  float iscale = nmax / max;
  for (int i = 0; i < n; ++i) {
    L[i] = nearest_int(iscale * x[i]);
  }
  float scale = 1 / iscale;
  float best_mse = 0;
  for (int i = 0; i < n; ++i) {
    float diff = x[i] - scale * L[i];
    float w = quant_weights[i];
    best_mse += w * diff * diff;
  }
  for (int is = -4; is <= 4; ++is) {
    if (is == 0)
      continue;
    float iscale_is = (0.1f * is + nmax) / max;
    float scale_is = 1 / iscale_is;
    float mse = 0;
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale_is * x[i]);
      l = MIN(nmax, l);
      float diff = x[i] - scale_is * l;
      float w = quant_weights[i];
      mse += w * diff * diff;
    }
    if (mse < best_mse) {
      best_mse = mse;
      iscale = iscale_is;
    }
  }
  float sumlx = 0;
  float suml2 = 0;
  for (int i = 0; i < n; ++i) {
    int l = nearest_int(iscale * x[i]);
    l = MIN(nmax, l);
    L[i] = l;
    float w = quant_weights[i];
    sumlx += w * x[i] * l;
    suml2 += w * l * l;
  }
  for (int itry = 0; itry < 5; ++itry) {
    int n_changed = 0;
    for (int i = 0; i < n; ++i) {
      float w = quant_weights[i];
      float slx = sumlx - w * x[i] * L[i];
      float sl2 = suml2 - w * L[i] * L[i];
      if (slx > 0 && sl2 > 0) {
        int new_l = nearest_int(x[i] * sl2 / slx);
        new_l = MIN(nmax, new_l);
        if (new_l != L[i]) {
          slx += w * x[i] * new_l;
          sl2 += w * new_l * new_l;
          if (slx * slx * suml2 > sumlx * sumlx * sl2) {
            L[i] = new_l;
            sumlx = slx;
            suml2 = sl2;
            ++n_changed;
          }
        }
      }
    }
    if (!n_changed) {
      break;
    }
  }
  return sumlx / suml2;
}

static inline void get_scale_min_k4(int j, const uint8_t *__restrict q,
                                    uint8_t *__restrict d,
                                    uint8_t *__restrict m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

// -------- Q4_0 -----------------

void quantize_row_q4_0_ref(const float *__restrict x, block_q4_0 *__restrict y,
                           int64_t k) {
  static const int qk = QK4_0;

  assert(k % qk == 0);

  const int nb = k / qk;

  for (int i = 0; i < nb; i++) {
    float amax = 0.0f; // absolute max
    float max = 0.0f;

    for (int j = 0; j < qk; j++) {
      const float v = x[i * qk + j];
      if (amax < fabsf(v)) {
        amax = fabsf(v);
        max = v;
      }
    }

    const float d = max / -8;
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = nntr_fp32_to_fp16(d);

    for (int j = 0; j < qk / 2; ++j) {
      const float x0 = x[i * qk + 0 + j] * id;
      const float x1 = x[i * qk + qk / 2 + j] * id;

      const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
      const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

      y[i].qs[j] = xi0;
      y[i].qs[j] |= xi1 << 4;
    }
  }
}

static void quantize_row_q4_0_impl(const float *__restrict x,
                                   block_q4_0 *__restrict y, int64_t n_per_row,
                                   const float *quant_weights) {
  static_assert(QK4_0 == 32, "QK4_0 must be 32");

  if (!quant_weights) {
    quantize_row_q4_0_ref(x, y, n_per_row);
    return;
  }

  float weight[QK4_0];
  int8_t L[QK4_0];

  float sum_x2 = 0;
  for (int j = 0; j < n_per_row; ++j)
    sum_x2 += x[j] * x[j];
  float sigma2 = sum_x2 / n_per_row;

  const int64_t nb = n_per_row / QK4_0;
  for (int ib = 0; ib < nb; ++ib) {
    const float *xb = x + QK4_0 * ib;
    const float *qw = quant_weights + QK4_0 * ib;
    for (int j = 0; j < QK4_0; ++j)
      weight[j] = qw[j] * sqrtf(sigma2 + xb[j] * xb[j]);
    float d = make_qx_quants(QK4_0, 8, xb, L, 1, weight);
    y[ib].d = nntr_fp32_to_fp16(d);
    for (int j = 0; j < 16; ++j) {
      y[ib].qs[j] = L[j] | (L[j + 16] << 4);
    }
  }
}

size_t nntr_quantize_q4_0(const float *__restrict src, void *__restrict dst,
                          int64_t nrow, int64_t n_per_row,
                          const float *quant_weights) {
  if (!quant_weights) {
    quantize_row_q4_0_ref(src, (block_q4_0 *)dst, (int64_t)nrow * n_per_row);
    return nrow * ggml_row_size(GGML_TYPE_Q4_0, n_per_row);
  }
  size_t row_size = ggml_row_size(GGML_TYPE_Q4_0, n_per_row);
  char *qrow = (char *)dst;
  for (int64_t row = 0; row < nrow; ++row) {
    quantize_row_q4_0_impl(src, (block_q4_0 *)qrow, n_per_row, quant_weights);
    src += n_per_row;
    qrow += row_size;
  }
  return nrow * row_size;
}

void dequantize_row_q4_0_impl(const block_q4_0 *__restrict x,
                              float *__restrict y, int64_t k) {
  static const int qk = QK4_0;

  assert(k % qk == 0);

  const int nb = k / qk;

  for (int i = 0; i < nb; i++) {
    const float d = nntr_fp16_to_fp32(x[i].d);

    for (int j = 0; j < qk / 2; ++j) {
      const int x0 = (x[i].qs[j] & 0x0F) - 8;
      const int x1 = (x[i].qs[j] >> 4) - 8;

      y[i * qk + j + 0] = x0 * d;
      y[i * qk + j + qk / 2] = x1 * d;
    }
  }
}

void nntr_dequantize_row_q4_0(const void *__restrict x, float *__restrict y,
                              int64_t k) {
  dequantize_row_q4_0_impl((const block_q4_0 *)x, y, k);
}

// -------- Q8_0 -----------------

void quantize_row_q8_0_ref(const float *__restrict x, block_q8_0 *__restrict y,
                           int64_t k) {
  assert(k % QK8_0 == 0);
  const int nb = k / QK8_0;

  for (int i = 0; i < nb; i++) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
      const float v = x[i * QK8_0 + j];
      amax = MAX(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = nntr_fp32_to_fp16(d);

    for (int j = 0; j < QK8_0; ++j) {
      const float x0 = x[i * QK8_0 + j] * id;

      y[i].qs[j] = static_cast<int8_t>(roundf(x0));
    }
  }
}

size_t nntr_quantize_q8_0(const float *__restrict src, void *__restrict dst,
                          int64_t nrow, int64_t n_per_row,
                          const float *quant_weights) {
  (void)quant_weights; // not used
  const size_t row_size = ggml_row_size(GGML_TYPE_Q8_0, n_per_row);
  quantize_row_q8_0_ref(src, (block_q8_0 *)dst, (int64_t)nrow * n_per_row);
  return nrow * row_size;
}

void nntr_quantize_row_q8_0(const float *__restrict x, void *__restrict vy,
                            int64_t k) {
  assert(QK8_0 == 32);
  assert(k % QK8_0 == 0);
  const int nb = k / QK8_0;

  block_q8_0 *__restrict y = (block_q8_0 *)vy;

#if defined(__ARM_NEON)
  for (int i = 0; i < nb; i++) {
    float32x4_t srcv[8];
    float32x4_t asrcv[8];
    float32x4_t amaxv[8];

    for (int j = 0; j < 8; j++)
      srcv[j] = vld1q_f32(x + i * 32 + 4 * j);
    for (int j = 0; j < 8; j++)
      asrcv[j] = vabsq_f32(srcv[j]);

    for (int j = 0; j < 4; j++)
      amaxv[2 * j] = vmaxq_f32(asrcv[2 * j], asrcv[2 * j + 1]);
    for (int j = 0; j < 2; j++)
      amaxv[4 * j] = vmaxq_f32(amaxv[4 * j], amaxv[4 * j + 2]);
    for (int j = 0; j < 1; j++)
      amaxv[8 * j] = vmaxq_f32(amaxv[8 * j], amaxv[8 * j + 4]);

    const float amax = vmaxvq_f32(amaxv[0]);

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = nntr_fp32_to_fp16(d);

    for (int j = 0; j < 8; j++) {
      const float32x4_t v = vmulq_n_f32(srcv[j], id);
      const int32x4_t vi = vcvtnq_s32_f32(v);

      y[i].qs[4 * j + 0] = vgetq_lane_s32(vi, 0);
      y[i].qs[4 * j + 1] = vgetq_lane_s32(vi, 1);
      y[i].qs[4 * j + 2] = vgetq_lane_s32(vi, 2);
      y[i].qs[4 * j + 3] = vgetq_lane_s32(vi, 3);
    }
  }
#elif defined __wasm_simd128__
  for (int i = 0; i < nb; i++) {
    v128_t srcv[8];
    v128_t asrcv[8];
    v128_t amaxv[8];

    for (int j = 0; j < 8; j++)
      srcv[j] = wasm_v128_load(x + i * 32 + 4 * j);
    for (int j = 0; j < 8; j++)
      asrcv[j] = wasm_f32x4_abs(srcv[j]);

    for (int j = 0; j < 4; j++)
      amaxv[2 * j] = wasm_f32x4_max(asrcv[2 * j], asrcv[2 * j + 1]);
    for (int j = 0; j < 2; j++)
      amaxv[4 * j] = wasm_f32x4_max(amaxv[4 * j], amaxv[4 * j + 2]);
    for (int j = 0; j < 1; j++)
      amaxv[8 * j] = wasm_f32x4_max(amaxv[8 * j], amaxv[8 * j + 4]);

    const float amax = MAX(MAX(wasm_f32x4_extract_lane(amaxv[0], 0),
                               wasm_f32x4_extract_lane(amaxv[0], 1)),
                           MAX(wasm_f32x4_extract_lane(amaxv[0], 2),
                               wasm_f32x4_extract_lane(amaxv[0], 3)));

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = nntr_fp32_to_fp16(d);

    for (int j = 0; j < 8; j++) {
      const v128_t v = wasm_f32x4_mul(srcv[j], wasm_f32x4_splat(id));
      const v128_t vi = wasm_i32x4_trunc_sat_f32x4(v);

      y[i].qs[4 * j + 0] = wasm_i32x4_extract_lane(vi, 0);
      y[i].qs[4 * j + 1] = wasm_i32x4_extract_lane(vi, 1);
      y[i].qs[4 * j + 2] = wasm_i32x4_extract_lane(vi, 2);
      y[i].qs[4 * j + 3] = wasm_i32x4_extract_lane(vi, 3);
    }
  }
#elif defined(__AVX2__) || defined(__AVX__)
  for (int i = 0; i < nb; i++) {
    // Load elements into 4 AVX vectors
    __m256 v0 = _mm256_loadu_ps(x);
    __m256 v1 = _mm256_loadu_ps(x + 8);
    __m256 v2 = _mm256_loadu_ps(x + 16);
    __m256 v3 = _mm256_loadu_ps(x + 24);
    x += 32;

    // Compute max(abs(e)) for the block
    const __m256 signBit = _mm256_set1_ps(-0.0f);
    __m256 maxAbs = _mm256_andnot_ps(signBit, v0);
    maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v1));
    maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v2));
    maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v3));

    __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(maxAbs, 1),
                             _mm256_castps256_ps128(maxAbs));
    max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
    max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
    const float maxScalar = _mm_cvtss_f32(max4);

    // Quantize these floats
    const float d = maxScalar / 127.f;
    y[i].d = nntr_fp32_to_fp16(d);
    const float id = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
    const __m256 mul = _mm256_set1_ps(id);

    // Apply the multiplier
    v0 = _mm256_mul_ps(v0, mul);
    v1 = _mm256_mul_ps(v1, mul);
    v2 = _mm256_mul_ps(v2, mul);
    v3 = _mm256_mul_ps(v3, mul);

    // Round to nearest integer
    v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
    v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
    v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
    v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);

    // Convert floats to integers
    __m256i i0 = _mm256_cvtps_epi32(v0);
    __m256i i1 = _mm256_cvtps_epi32(v1);
    __m256i i2 = _mm256_cvtps_epi32(v2);
    __m256i i3 = _mm256_cvtps_epi32(v3);

#if defined(__AVX2__)
    // Convert int32 to int16
    i0 = _mm256_packs_epi32(
      i0, i1); // 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
    i2 = _mm256_packs_epi32(i2,
                            i3); // 16, 17, 18, 19,  24, 25, 26, 27,  20, 21,
                                 // 22, 23, 28, 29, 30, 31 Convert int16 to int8
    i0 = _mm256_packs_epi16(
      i0, i2); // 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,
               // 4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

    // We got our precious signed bytes, but the order is now wrong
    // These AVX2 pack instructions process 16-byte pieces independently
    // The following instruction is fixing the order
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    i0 = _mm256_permutevar8x32_epi32(i0, perm);

    _mm256_storeu_si256((__m256i *)y[i].qs, i0);
#else
    // Since we don't have in AVX some necessary functions,
    // we split the registers in half and call AVX2 analogs from SSE
    __m128i ni0 = _mm256_castsi256_si128(i0);
    __m128i ni1 = _mm256_extractf128_si256(i0, 1);
    __m128i ni2 = _mm256_castsi256_si128(i1);
    __m128i ni3 = _mm256_extractf128_si256(i1, 1);
    __m128i ni4 = _mm256_castsi256_si128(i2);
    __m128i ni5 = _mm256_extractf128_si256(i2, 1);
    __m128i ni6 = _mm256_castsi256_si128(i3);
    __m128i ni7 = _mm256_extractf128_si256(i3, 1);

    // Convert int32 to int16
    ni0 = _mm_packs_epi32(ni0, ni1);
    ni2 = _mm_packs_epi32(ni2, ni3);
    ni4 = _mm_packs_epi32(ni4, ni5);
    ni6 = _mm_packs_epi32(ni6, ni7);
    // Convert int16 to int8
    ni0 = _mm_packs_epi16(ni0, ni2);
    ni4 = _mm_packs_epi16(ni4, ni6);

    _mm_storeu_si128((__m128i *)(y[i].qs + 0), ni0);
    _mm_storeu_si128((__m128i *)(y[i].qs + 16), ni4);
#endif
  }
#elif defined(__riscv_v_intrinsic)

  size_t vl = QK8_0;

  for (int i = 0; i < nb; i++) {
    // load elements
    vfloat32m8_t v_x = __riscv_vle32_v_f32m8(x + i * QK8_0, vl);

    vfloat32m8_t vfabs = __riscv_vfabs_v_f32m8(v_x, vl);
    vfloat32m1_t tmp = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t vmax = __riscv_vfredmax_vs_f32m8_f32m1(vfabs, tmp, vl);
    float amax = __riscv_vfmv_f_s_f32m1_f32(vmax);

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = nntr_fp32_to_fp16(d);

    vfloat32m8_t x0 = __riscv_vfmul_vf_f32m8(v_x, id, vl);

    // convert to integer
    vint16m4_t vi = __riscv_vfncvt_x_f_w_i16m4(x0, vl);
    vint8m2_t vs = __riscv_vncvt_x_x_w_i8m2(vi, vl);

    // store result
    __riscv_vse8_v_i8m2(y[i].qs, vs, vl);
  }

#elif defined(__POWER9_VECTOR__)
  for (int i = 0; i < nb; i++) {
    vector float srcv[8];
    vector float asrcv[8];
    vector float amaxv[8];
    vector signed int vi[8];

    for (int j = 0; j < 8; j++)
      srcv[j] = vec_xl(0, x + i * 32 + 4 * j);
    for (int j = 0; j < 8; j++)
      asrcv[j] = vec_abs(srcv[j]);

    for (int j = 0; j < 4; j++)
      amaxv[2 * j] = vec_max(asrcv[2 * j], asrcv[2 * j + 1]);
    for (int j = 0; j < 2; j++)
      amaxv[4 * j] = vec_max(amaxv[4 * j], amaxv[4 * j + 2]);
    for (int j = 0; j < 1; j++)
      amaxv[8 * j] = vec_max(amaxv[8 * j], amaxv[8 * j + 4]);

    const float amax =
      MAX(MAX(vec_extract(amaxv[0], 0), vec_extract(amaxv[0], 1)),
          MAX(vec_extract(amaxv[0], 2), vec_extract(amaxv[0], 3)));

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;
    const vector float vid = vec_splats(id);

    y[i].d = nntr_fp32_to_fp16(d);

    for (int j = 0; j < 8; j++) {
      const vector float v = vec_round(vec_mul(srcv[j], vid));
      vi[j] = vec_cts(v, 0);
    }
    vec_xst(vec_pack(vec_pack(vi[0], vi[1]), vec_pack(vi[2], vi[3])), 0,
            &y[i].qs[0]);
    vec_xst(vec_pack(vec_pack(vi[4], vi[5]), vec_pack(vi[6], vi[7])), 16,
            &y[i].qs[0]);
  }

#elif defined(__loongarch_asx)
  for (int i = 0; i < nb; i++) {
    __m256 v0 = (__m256)__lasx_xvld(x, 0);
    __m256 v1 = (__m256)__lasx_xvld(x, 32);
    __m256 v2 = (__m256)__lasx_xvld(x, 64);
    __m256 v3 = (__m256)__lasx_xvld(x, 96);
    x += 32;

    // Compute max(abs(e)) for the block
    const __m256 sign_bit = __lasx_xvreplfr2vr_s(-0.0f);
    __m256 max_abs = (__m256)__lasx_xvandn_v((__m256i)sign_bit, (__m256i)v0);
    max_abs = __lasx_xvfmax_s(
      max_abs, (__m256)__lasx_xvandn_v((__m256i)sign_bit, (__m256i)v1));
    max_abs = __lasx_xvfmax_s(
      max_abs, (__m256)__lasx_xvandn_v((__m256i)sign_bit, (__m256i)v2));
    max_abs = __lasx_xvfmax_s(
      max_abs, (__m256)__lasx_xvandn_v((__m256i)sign_bit, (__m256i)v3));

    __m128 max4 =
      __lsx_vfmax_s(lasx_extractf128(max_abs, 1), lasx_extractf128(max_abs, 0));
    max4 = __lsx_vfmax_s(max4,
                         (__m128)__lsx_vpickod_d((__m128i)max4, (__m128i)max4));
    __m128 tmp = max4;
    max4 = __lsx_vfmax_s(
      max4, (__m128)__lsx_vinsgr2vr_w(tmp, __lsx_vpickve2gr_w(max4, 1), 0));
    const float max_scalar = ((v4f32)max4)[0];

    // Quantize these floats
    const float d = max_scalar / 127.f;
    y[i].d = nntr_fp32_to_fp16(d);
    const float id = (max_scalar != 0.0f) ? 127.f / max_scalar : 0.0f;
    const __m256 mul = (__m256)__lasx_xvreplfr2vr_s(id);

    // Apply the multiplier
    v0 = __lasx_xvfmul_s(v0, mul);
    v1 = __lasx_xvfmul_s(v1, mul);
    v2 = __lasx_xvfmul_s(v2, mul);
    v3 = __lasx_xvfmul_s(v3, mul);

    // Round to nearest integer
    __m256i i0 = __lasx_xvftintrne_w_s(v0);
    __m256i i1 = __lasx_xvftintrne_w_s(v1);
    __m256i i2 = __lasx_xvftintrne_w_s(v2);
    __m256i i3 = __lasx_xvftintrne_w_s(v3);

    __m128i ni0 = lasx_extracti128(i0, 0);
    __m128i ni1 = lasx_extracti128(i0, 1);
    __m128i ni2 = lasx_extracti128(i1, 0);
    __m128i ni3 = lasx_extracti128(i1, 1);
    __m128i ni4 = lasx_extracti128(i2, 0);
    __m128i ni5 = lasx_extracti128(i2, 1);
    __m128i ni6 = lasx_extracti128(i3, 0);
    __m128i ni7 = lasx_extracti128(i3, 1);

    // Convert int32 to int16
    ni0 = lsx_packs_w(ni0, ni1);
    ni2 = lsx_packs_w(ni2, ni3);
    ni4 = lsx_packs_w(ni4, ni5);
    ni6 = lsx_packs_w(ni6, ni7);
    // Convert int16 to int8
    ni0 = lsx_packs_h(ni0, ni2);
    ni4 = lsx_packs_h(ni4, ni6);

    __lsx_vst(ni0, (__m128i *)(y[i].qs + 0), 0);
    __lsx_vst(ni4, (__m128i *)(y[i].qs + 16), 0);
  }
#elif defined(__VXE__) || defined(__VXE2__)
  for (int i = 0; i < nb; i++) {
    __vector float srcv[8];
    __vector float asrcv[8];
    __vector float amaxv[8];

    for (int j = 0; j < 8; j++)
      srcv[j] = vec_xl(0, x + i * 32 + 4 * j);
    for (int j = 0; j < 8; j++)
      asrcv[j] = vec_abs(srcv[j]);
    for (int j = 0; j < 4; j++)
      amaxv[2 * j] = vec_max(asrcv[2 * j], asrcv[2 * j + 1]);
    for (int j = 0; j < 2; j++)
      amaxv[4 * j] = vec_max(amaxv[4 * j], amaxv[4 * j + 2]);
    for (int j = 0; j < 1; j++)
      amaxv[8 * j] = vec_max(amaxv[8 * j], amaxv[8 * j + 4]);

    const float amax =
      MAX(MAX(vec_extract(amaxv[0], 0), vec_extract(amaxv[0], 1)),
          MAX(vec_extract(amaxv[0], 2), vec_extract(amaxv[0], 3)));

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = nntr_fp32_to_fp16(d);

    for (int j = 0; j < 8; j++) {
      const __vector float v = vec_mul(srcv[j], vec_splats(id));
      const __vector int32_t vi = vec_signed(v);

      y[i].qs[4 * j + 0] = vec_extract(vi, 0);
      y[i].qs[4 * j + 1] = vec_extract(vi, 1);
      y[i].qs[4 * j + 2] = vec_extract(vi, 2);
      y[i].qs[4 * j + 3] = vec_extract(vi, 3);
    }
  }
#else
  // scalar
  quantize_row_q8_0_ref(x, y, k);
#endif
}

void dequantize_row_q8_0_impl(const block_q8_0 *__restrict x,
                              float *__restrict y, int64_t k) {
  static const int qk = QK8_0;

  assert(k % qk == 0);

  const int nb = k / qk;

  for (int i = 0; i < nb; i++) {
    const float d = nntr_fp16_to_fp32(x[i].d);

    for (int j = 0; j < qk; ++j) {
      y[i * qk + j] = x[i].qs[j] * d;
    }
  }
}

void nntr_dequantize_row_q8_0(const void *__restrict x, float *__restrict y,
                              int64_t k) {
  dequantize_row_q8_0_impl((const block_q8_0 *)x, y, k);
}

// -------- Q4_K -----------------

void quantize_row_q4_K_ref(const float *__restrict x, block_q4_K *__restrict y,
                           int64_t k) {
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  uint8_t L[QK_K];
  uint8_t Laux[32];
  float weights[32];
  float mins[QK_K / 32];
  float scales[QK_K / 32];

  for (int i = 0; i < nb; i++) {
    float max_scale =
      0; // as we are deducting the min, scales are always positive
    float max_min = 0;
    for (int j = 0; j < QK_K / 32; ++j) {
      // scales[j] = make_qkx1_quants(32, 15, x + 32*j, L + 32*j, &mins[j], 9,
      // 0.5f);
      float sum_x2 = 0;
      for (int l = 0; l < 32; ++l)
        sum_x2 += x[32 * j + l] * x[32 * j + l];
      float av_x = sqrtf(sum_x2 / 32);
      for (int l = 0; l < 32; ++l)
        weights[l] = av_x + fabsf(x[32 * j + l]);
      scales[j] = make_qkx2_quants(32, 15, x + 32 * j, weights, L + 32 * j,
                                   &mins[j], Laux, -1.f, 0.1f, 20, false);
      float scale = scales[j];
      if (scale > max_scale) {
        max_scale = scale;
      }
      float min = mins[j];
      if (min > max_min) {
        max_min = min;
      }
    }

    float inv_scale = max_scale > 0 ? 63.f / max_scale : 0.f;
    float inv_min = max_min > 0 ? 63.f / max_min : 0.f;
    for (int j = 0; j < QK_K / 32; ++j) {
      uint8_t ls = nearest_int(inv_scale * scales[j]);
      uint8_t lm = nearest_int(inv_min * mins[j]);
      ls = MIN(63, ls);
      lm = MIN(63, lm);
      if (j < 4) {
        y[i].scales[j] = ls;
        y[i].scales[j + 4] = lm;
      } else {
        y[i].scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
        y[i].scales[j - 4] |= ((ls >> 4) << 6);
        y[i].scales[j - 0] |= ((lm >> 4) << 6);
      }
    }
    y[i].data.data.d = nntr_fp32_to_fp16(max_scale / 63.f);
    y[i].data.data.dmin = nntr_fp32_to_fp16(max_min / 63.f);

    uint8_t sc, m;
    for (int j = 0; j < QK_K / 32; ++j) {
      get_scale_min_k4(j, y[i].scales, &sc, &m);
      const float d = nntr_fp16_to_fp32(y[i].data.data.d) * sc;
      if (!d)
        continue;
      const float dm = nntr_fp16_to_fp32(y[i].data.data.dmin) * m;
      for (int ii = 0; ii < 32; ++ii) {
        int l = nearest_int((x[32 * j + ii] + dm) / d);
        l = MAX(0, MIN(15, l));
        L[32 * j + ii] = l;
      }
    }

    uint8_t *q = y[i].qs;
    for (int j = 0; j < QK_K; j += 64) {
      for (int l = 0; l < 32; ++l)
        q[l] = L[j + l] | (L[j + l + 32] << 4);
      q += 32;
    }

    x += QK_K;
  }
}

static void quantize_row_q4_K_impl(const float *__restrict x,
                                   block_q4_K *__restrict y, int64_t n_per_row,
                                   const float *quant_weights) {
  assert(n_per_row % QK_K == 0);
  const int64_t nb = n_per_row / QK_K;

  uint8_t L[QK_K];
  uint8_t Laux[32];
  uint8_t Ls[QK_K / 32];
  uint8_t Lm[QK_K / 32];
  float weights[32];
  float sw[QK_K / 32];
  float mins[QK_K / 32];
  float scales[QK_K / 32];

  for (int i = 0; i < nb; i++) {

    float sum_x2 = 0;
    for (int l = 0; l < QK_K; ++l)
      sum_x2 += x[l] * x[l];
    float sigma2 = 2 * sum_x2 / QK_K;
    float av_x = sqrtf(sigma2);

    for (int j = 0; j < QK_K / 32; ++j) {
      if (quant_weights) {
        const float *qw = quant_weights + QK_K * i + 32 * j;
        for (int l = 0; l < 32; ++l)
          weights[l] = qw[l] * sqrtf(sigma2 + x[32 * j + l] * x[32 * j + l]);
      } else {
        for (int l = 0; l < 32; ++l)
          weights[l] = av_x + fabsf(x[32 * j + l]);
      }
      float sumw = 0;
      for (int l = 0; l < 32; ++l)
        sumw += weights[l];
      sw[j] = sumw;
      scales[j] = make_qkx3_quants(32, 15, x + 32 * j, weights, L + 32 * j,
                                   &mins[j], Laux, -0.9f, 0.05f, 36, false);
    }

    float d_block = make_qp_quants(QK_K / 32, 63, scales, Ls, sw);
    float m_block = make_qp_quants(QK_K / 32, 63, mins, Lm, sw);
    for (int j = 0; j < QK_K / 32; ++j) {
      uint8_t ls = Ls[j];
      uint8_t lm = Lm[j];
      if (j < 4) {
        y[i].scales[j] = ls;
        y[i].scales[j + 4] = lm;
      } else {
        y[i].scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
        y[i].scales[j - 4] |= ((ls >> 4) << 6);
        y[i].scales[j - 0] |= ((lm >> 4) << 6);
      }
    }
    y[i].data.data.d = nntr_fp32_to_fp16(d_block);
    y[i].data.data.dmin = nntr_fp32_to_fp16(m_block);

    uint8_t sc, m;
    for (int j = 0; j < QK_K / 32; ++j) {
      get_scale_min_k4(j, y[i].scales, &sc, &m);
      const float d = nntr_fp16_to_fp32(y[i].data.data.d) * sc;
      if (!d)
        continue;
      const float dm = nntr_fp16_to_fp32(y[i].data.data.dmin) * m;
      for (int ii = 0; ii < 32; ++ii) {
        int l = nearest_int((x[32 * j + ii] + dm) / d);
        l = MAX(0, MIN(15, l));
        L[32 * j + ii] = l;
      }
    }
    uint8_t *q = y[i].qs;
    for (int j = 0; j < QK_K; j += 64) {
      for (int l = 0; l < 32; ++l)
        q[l] = L[j + l] | (L[j + l + 32] << 4);
      q += 32;
    }

    x += QK_K;
  }
}

size_t nntr_quantize_q4_K(const float *__restrict src, void *__restrict dst,
                          int64_t nrow, int64_t n_per_row,
                          const float *quant_weights) {
  size_t row_size = ggml_row_size(GGML_TYPE_Q4_K, n_per_row);
  if (!quant_weights) {
    quantize_row_q4_K_ref(src, (block_q4_K *)dst, (int64_t)nrow * n_per_row);
  } else {
    char *qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
      quantize_row_q4_K_impl(src, (block_q4_K *)qrow, n_per_row, quant_weights);
      src += n_per_row;
      qrow += row_size;
    }
  }
  return nrow * row_size;
}

void dequantize_row_q4_K_impl(const block_q4_K *__restrict x,
                              float *__restrict y, int64_t k) {
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  for (int i = 0; i < nb; i++) {
    const uint8_t *q = x[i].qs;

    const float d = nntr_fp16_to_fp32(x[i].data.data.d);
    const float min = nntr_fp16_to_fp32(x[i].data.data.dmin);

    int is = 0;
    uint8_t sc, m;
    for (int j = 0; j < QK_K; j += 64) {
      get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
      const float d1 = d * sc;
      const float m1 = min * m;
      get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
      const float d2 = d * sc;
      const float m2 = min * m;
      for (int l = 0; l < 32; ++l)
        *y++ = d1 * (q[l] & 0xF) - m1;
      for (int l = 0; l < 32; ++l)
        *y++ = d2 * (q[l] >> 4) - m2;
      q += 32;
      is += 2;
    }
  }
}

void nntr_dequantize_row_q4_K(const void *__restrict x, float *__restrict y,
                              int64_t k) {
  dequantize_row_q4_K_impl((const block_q4_K *)x, y, k);
}

// --------- Q6_K --------------

void quantize_row_q6_K_ref(const float *__restrict x, block_q6_K *__restrict y,
                           int64_t k) {
  assert(k % QK_K == 0);
  const int64_t nb = k / QK_K;

  int8_t L[QK_K];
  float scales[QK_K / 16];

  for (int i = 0; i < nb; i++) {

    float max_scale = 0;
    float max_abs_scale = 0;

    for (int ib = 0; ib < QK_K / 16; ++ib) {

      const float scale =
        make_qx_quants(16, 32, x + 16 * ib, L + 16 * ib, 1, NULL);
      scales[ib] = scale;

      const float abs_scale = fabsf(scale);
      if (abs_scale > max_abs_scale) {
        max_abs_scale = abs_scale;
        max_scale = scale;
      }
    }

    if (max_abs_scale < GROUP_MAX_EPS) {
      memset(&y[i], 0, sizeof(block_q6_K));
      y[i].d = nntr_fp32_to_fp16(0.f);
      x += QK_K;
      continue;
    }

    float iscale = -128.f / max_scale;
    y[i].d = nntr_fp32_to_fp16(1 / iscale);
    for (int ib = 0; ib < QK_K / 16; ++ib) {
      y[i].scales[ib] = MIN(127, nearest_int(iscale * scales[ib]));
    }

    for (int j = 0; j < QK_K / 16; ++j) {
      float d = nntr_fp16_to_fp32(y[i].d) * y[i].scales[j];
      if (!d) {
        continue;
      }
      for (int ii = 0; ii < 16; ++ii) {
        int l = nearest_int(x[16 * j + ii] / d);
        l = MAX(-32, MIN(31, l));
        L[16 * j + ii] = l + 32;
      }
    }

    uint8_t *__restrict ql = y[i].ql;
    uint8_t *__restrict qh = y[i].qh;
    for (int j = 0; j < QK_K; j += 128) {
      for (int l = 0; l < 32; ++l) {
        const uint8_t q1 = L[j + l + 0] & 0xF;
        const uint8_t q2 = L[j + l + 32] & 0xF;
        const uint8_t q3 = L[j + l + 64] & 0xF;
        const uint8_t q4 = L[j + l + 96] & 0xF;
        ql[l + 0] = q1 | (q3 << 4);
        ql[l + 32] = q2 | (q4 << 4);
        qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) |
                ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
      }
      ql += 64;
      qh += 32;
    }

    x += QK_K;
  }
}

static void quantize_row_q6_K_impl(const float *__restrict x,
                                   block_q6_K *__restrict y, int64_t n_per_row,
                                   const float *quant_weights) {
  assert(n_per_row % QK_K == 0);
  const int64_t nb = n_per_row / QK_K;

  int8_t L[QK_K];
  float scales[QK_K / 16];
  // float   weights[16];

  for (int i = 0; i < nb; i++) {

    // float sum_x2 = 0;
    // for (int j = 0; j < QK_K; ++j) sum_x2 += x[j]*x[j];
    // float sigma2 = sum_x2/QK_K;

    float max_scale = 0;
    float max_abs_scale = 0;

    for (int ib = 0; ib < QK_K / 16; ++ib) {

      float scale;
      if (quant_weights) {
        const float *qw = quant_weights + QK_K * i + 16 * ib;
        // for (int j = 0; j < 16; ++j) weights[j] = qw[j] * sqrtf(sigma2 +
        // x[16*ib + j]*x[16*ib + j]); scale = make_qx_quants(16, 32, x +
        // 16*ib, L + 16*ib, 1, weights);
        scale = make_qx_quants(16, 32, x + 16 * ib, L + 16 * ib, 1, qw);
      } else {
        scale = make_qx_quants(16, 32, x + 16 * ib, L + 16 * ib, 1, NULL);
      }
      scales[ib] = scale;

      const float abs_scale = fabsf(scale);
      if (abs_scale > max_abs_scale) {
        max_abs_scale = abs_scale;
        max_scale = scale;
      }
    }

    if (max_abs_scale < GROUP_MAX_EPS) {
      memset(&y[i], 0, sizeof(block_q6_K));
      y[i].d = nntr_fp32_to_fp16(0.f);
      x += QK_K;
      continue;
    }

    float iscale = -128.f / max_scale;
    y[i].d = nntr_fp32_to_fp16(1 / iscale);
    for (int ib = 0; ib < QK_K / 16; ++ib) {
      y[i].scales[ib] = MIN(127, nearest_int(iscale * scales[ib]));
    }

    for (int j = 0; j < QK_K / 16; ++j) {
      float d = nntr_fp16_to_fp32(y[i].d) * y[i].scales[j];
      if (!d) {
        continue;
      }
      for (int ii = 0; ii < 16; ++ii) {
        int l = nearest_int(x[16 * j + ii] / d);
        l = MAX(-32, MIN(31, l));
        L[16 * j + ii] = l + 32;
      }
    }

    uint8_t *__restrict ql = y[i].ql;
    uint8_t *__restrict qh = y[i].qh;
    for (int j = 0; j < QK_K; j += 128) {
      for (int l = 0; l < 32; ++l) {
        const uint8_t q1 = L[j + l + 0] & 0xF;
        const uint8_t q2 = L[j + l + 32] & 0xF;
        const uint8_t q3 = L[j + l + 64] & 0xF;
        const uint8_t q4 = L[j + l + 96] & 0xF;
        ql[l + 0] = q1 | (q3 << 4);
        ql[l + 32] = q2 | (q4 << 4);
        qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) |
                ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
      }
      ql += 64;
      qh += 32;
    }

    x += QK_K;
  }
}

size_t nntr_quantize_q6_K(const float *__restrict src, void *__restrict dst,
                          int64_t nrow, int64_t n_per_row,
                          const float *quant_weights) {
  size_t row_size = ggml_row_size(GGML_TYPE_Q6_K, n_per_row);
  if (!quant_weights) {
    quantize_row_q6_K_ref(src, (block_q6_K *)dst, (int64_t)nrow * n_per_row);
  } else {
    char *qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
      quantize_row_q6_K_impl(src, (block_q6_K *)qrow, n_per_row, quant_weights);
      src += n_per_row;
      qrow += row_size;
    }
  }
  return nrow * row_size;
}

void dequantize_row_q6_K_impl(const block_q6_K *__restrict x,
                              float *__restrict y, int64_t k) {
  assert(k % QK_K == 0);
  const int64_t nb = k / QK_K;

  for (int i = 0; i < nb; i++) {
    const float d = nntr_compute_fp16_to_fp32(x[i].d);

    const uint8_t *__restrict ql = x[i].ql;
    const uint8_t *__restrict qh = x[i].qh;
    const int8_t *__restrict sc = x[i].scales;

    for (int n = 0; n < QK_K; n += 128) {
      for (int l = 0; l < 32; ++l) {
        int is = l / 16;
        const int8_t q1 =
          (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        const int8_t q2 =
          (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        const int8_t q3 =
          (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        const int8_t q4 =
          (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        y[l + 0] = d * sc[is + 0] * q1;
        y[l + 32] = d * sc[is + 2] * q2;
        y[l + 64] = d * sc[is + 4] * q3;
        y[l + 96] = d * sc[is + 6] * q4;
      }
      y += 128;
      ql += 64;
      qh += 32;
      sc += 8;
    }
  }
}

void nntr_dequantize_row_q6_K(const void *__restrict x, float *__restrict y,
                              int64_t k) {
  dequantize_row_q6_K_impl((const block_q6_K *)x, y, k);
}

// --------- Q8_K --------------

void quantize_row_q8_K_ref(const float *__restrict x, block_q8_K *__restrict y,
                           int64_t k) {
  assert(k % QK_K == 0);
  const int64_t nb = k / QK_K;

  for (int i = 0; i < nb; i++) {

    float max = 0;
    float amax = 0;
    for (int j = 0; j < QK_K; ++j) {
      float ax = fabsf(x[j]);
      if (ax > amax) {
        amax = ax;
        max = x[j];
      }
    }
    if (!amax) {
      y[i].d = 0;
      memset(y[i].qs, 0, QK_K);
      x += QK_K;
      continue;
    }
    // const float iscale = -128.f/max;
    //  We need this change for IQ2_XXS, else the AVX implementation becomes
    //  very awkward
    const float iscale = -127.f / max;
    for (int j = 0; j < QK_K; ++j) {
      int v = nearest_int(iscale * x[j]);
      y[i].qs[j] = MIN(127, v);
    }
    for (int j = 0; j < QK_K / 16; ++j) {
      int sum = 0;
      for (int ii = 0; ii < 16; ++ii) {
        sum += y[i].qs[j * 16 + ii];
      }
      y[i].bsums[j] = sum;
    }
    y[i].d = 1 / iscale;
    x += QK_K;
  }
}

void nntr_quantize_row_q8_K(const float *__restrict x, void *__restrict y,
                            int64_t k) {
#ifdef __wasm_simd128__
  assert(k % QK_K == 0);
  const int64_t nb = k / QK_K;
  block_q8_K *__restrict yc = y; // Cast to proper type

  for (int i = 0; i < nb; i++) {
    const float *x_block = x + i * QK_K;

    v128_t min_vec = wasm_v128_load(x_block);
    v128_t max_vec = min_vec;

    for (int j = 4; j < QK_K; j += 4) {
      v128_t x_vec = wasm_v128_load(x_block + j);
      max_vec = wasm_f32x4_pmax(max_vec, x_vec);
      min_vec = wasm_f32x4_pmin(min_vec, x_vec);
    }
    max_vec = wasm_f32x4_pmax(max_vec,
                              wasm_i32x4_shuffle(max_vec, max_vec, 2, 3, 0, 1));
    max_vec = wasm_f32x4_pmax(max_vec,
                              wasm_i32x4_shuffle(max_vec, max_vec, 1, 0, 3, 2));
    min_vec = wasm_f32x4_pmin(min_vec,
                              wasm_i32x4_shuffle(min_vec, min_vec, 2, 3, 0, 1));
    min_vec = wasm_f32x4_pmin(min_vec,
                              wasm_i32x4_shuffle(min_vec, min_vec, 1, 0, 3, 2));
    float max = wasm_f32x4_extract_lane(max_vec, 0);
    float min = wasm_f32x4_extract_lane(min_vec, 0);
    float amax = -min > max ? min : max;

    if (amax == 0.0f) {
      yc[i].d = 0.0f;
      const v128_t zero = wasm_i8x16_splat(0);
      for (int j = 0; j < QK_K; j += 16) {
        wasm_v128_store(yc[i].qs + j, zero);
      }
      continue;
    }

    const float iscale = -127.0f / amax;
    const v128_t scale_vec = wasm_f32x4_splat(iscale);

    // Process 16 elements per iteration
    for (int j = 0, jb = 0; j < QK_K; j += 16, jb++) {
      // Load and quantize 16 floats
      v128_t x0 = wasm_v128_load(x_block + j);
      v128_t x1 = wasm_v128_load(x_block + j + 4);
      v128_t x2 = wasm_v128_load(x_block + j + 8);
      v128_t x3 = wasm_v128_load(x_block + j + 12);

      v128_t q0 = wasm_f32x4_nearest(wasm_f32x4_mul(x0, scale_vec));
      v128_t q1 = wasm_f32x4_nearest(wasm_f32x4_mul(x1, scale_vec));
      v128_t q2 = wasm_f32x4_nearest(wasm_f32x4_mul(x2, scale_vec));
      v128_t q3 = wasm_f32x4_nearest(wasm_f32x4_mul(x3, scale_vec));

      // Convert to i32 with saturation
      v128_t i0 = wasm_i32x4_trunc_sat_f32x4(q0);
      v128_t i1 = wasm_i32x4_trunc_sat_f32x4(q1);
      v128_t i2 = wasm_i32x4_trunc_sat_f32x4(q2);
      v128_t i3 = wasm_i32x4_trunc_sat_f32x4(q3);

      // Pack into 16 i8 values
      v128_t i8 = wasm_i8x16_narrow_i16x8(wasm_i16x8_narrow_i32x4(i0, i1),
                                          wasm_i16x8_narrow_i32x4(i2, i3));
      wasm_v128_store(yc[i].qs + j, i8);

      // Calculate bsums using SIMD
      v128_t sum16 = wasm_i16x8_add(wasm_i16x8_extend_low_i8x16(i8),
                                    wasm_i16x8_extend_high_i8x16(i8));
      v128_t sum32 = wasm_i32x4_add(wasm_i32x4_extend_low_i16x8(sum16),
                                    wasm_i32x4_extend_high_i16x8(sum16));
      sum32 =
        wasm_i32x4_add(sum32, wasm_i32x4_shuffle(sum32, sum32, 2, 3, 0, 1));
      sum32 =
        wasm_i32x4_add(sum32, wasm_i32x4_shuffle(sum32, sum32, 1, 0, 3, 2));
      yc[i].bsums[jb] = wasm_i32x4_extract_lane(sum32, 0);
    }

    yc[i].d = 1.0f / iscale;
  }
#else
  quantize_row_q8_K_ref(x, (block_q8_K *)y, k);
#endif
}

void dequantize_row_q8_K_impl(const block_q8_K *__restrict x,
                              float *__restrict y, int64_t k) {
  assert(k % QK_K == 0);
  const int64_t nb = k / QK_K;

  for (int i = 0; i < nb; i++) {
    for (int j = 0; j < QK_K; ++j) {
      *y++ = x[i].d * x[i].qs[j];
    }
  }
}

void nntr_dequantize_row_q8_K(const void *__restrict x, float *__restrict y,
                              int64_t k) {
  dequantize_row_q8_K_impl((const block_q8_K *)x, y, k);
}
