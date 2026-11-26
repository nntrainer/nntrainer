// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file   lmhead_argmax.cl
 * @brief  FP16 LM head partial and final argmax kernels.
 * @see    https://github.com/nntrainer/nntrainer
 *
 * @details Extracted from blas_kernels.cpp.
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define ARGMAX_LS 64
#define ARGMAX_TAKE(bv, bi, ov, oi)                                            \
  if ((ov) > (bv) || ((ov) == (bv) && (oi) < (bi))) {                          \
    bv = (ov);                                                                 \
    bi = (oi);                                                                 \
  }

// Final logit softcapping, reproduced STEP FOR STEP as the host layer computes
// it. Softcapping is strictly monotone, so it cannot move an argmax on its own
// -- but it collapses near-equal logits into equal fp16 values, and the host
// tie rule then decides the token, so the rounding is what has to match, not
// the mathematical value. The host chain is not a tanh() call: it is
// ActiFunc's logistic form over fp16 tensors, which narrows to half at four
// separate points (util_func.h exp_util<_Float16>, acti_func.h sigmoid /
// tanhFloat, half_tensor.cpp multiply(float) -- which multiplies IN half):
//
//   h1 = half(x * half(1/cap))     HalfTensor::multiply(1/cap)
//   y  = 2 * h1                    tanhFloat: half(2)*x, exact
//   e  = half(exp(-y))             exp_util<_Float16>: exp, then narrow
//   s  = half(1 / (1 + e))         sigmoid
//   t  = half(2*s - 1)             tanhFloat
//   out= half(t * half(cap))       HalfTensor::multiply(cap)
//
// Approximating this with a single tanh() instead differs by ~1 half-ULP on
// most elements -- enough to flip a token within the first 40 of a 223-token
// generation, which is how this chain came to be written out.
inline float softcap_h(float x, float inv_cap_h, float cap_h) {
  const float h1 = (float)(half)(x * inv_cap_h);
  const float y = 2.0f * h1;
  const float e = (float)(half)exp(-y);
  const float s = (float)(half)(1.0f / (1.0f + e));
  const float t = (float)(half)(2.0f * s - 1.0f);
  return (float)(half)(t * cap_h);
}

__kernel void lmhead_argmax_f16_part(__global const half *logits,
                                     __global float *part_val,
                                     __global int *part_idx, const int N,
                                     const float inv_cap_h, const float cap_h) {
  const int lid = get_local_id(0);
  float bv = -MAXFLOAT;
  int bi = 0x7fffffff;
  for (int i = get_global_id(0); i < N; i += get_global_size(0)) {
    float v = (float)logits[i];
    if (cap_h > 0.0f)
      v = softcap_h(v, inv_cap_h, cap_h);
    ARGMAX_TAKE(bv, bi, v, i)
  }
  __local float lv[ARGMAX_LS];
  __local int li[ARGMAX_LS];
  lv[lid] = bv;
  li[lid] = bi;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int off = ARGMAX_LS / 2; off > 0; off >>= 1) {
    if (lid < off) {
      const float ov = lv[lid + off];
      const int oi = li[lid + off];
      ARGMAX_TAKE(lv[lid], li[lid], ov, oi)
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) {
    part_val[get_group_id(0)] = lv[0];
    part_idx[get_group_id(0)] = li[0];
  }
}

__kernel void lmhead_argmax_f16_final(__global const float *part_val,
                                      __global const int *part_idx,
                                      __global int *token, const int G) {
  const int lid = get_local_id(0);
  float bv = -MAXFLOAT;
  int bi = 0x7fffffff;
  for (int i = lid; i < G; i += ARGMAX_LS) {
    const float v = part_val[i];
    const int ix = part_idx[i];
    ARGMAX_TAKE(bv, bi, v, ix)
  }
  __local float lv[ARGMAX_LS];
  __local int li[ARGMAX_LS];
  lv[lid] = bv;
  li[lid] = bi;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int off = ARGMAX_LS / 2; off > 0; off >>= 1) {
    if (lid < off) {
      const float ov = lv[lid + off];
      const int oi = li[lid + off];
      ARGMAX_TAKE(lv[lid], li[lid], ov, oi)
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0)
    token[0] = li[0];
}
