// SPDX-License-Identifier: Apache-2.0
// Fused RMSNorm + v8c activation quant kernel (paper §3.6 fused-kernel idea
// applied to the smallest unit that eliminates a drift boundary).
//
// REPLACES: CPU/GPU RMSNorm + separate v8c_act_quant pass. The two are fused
//           so the normalized fp32 value never touches global memory; it is
//           computed, used for min/max, and then re-computed for quant in
//           registers. The quant input is therefore deterministic w.r.t.
//           the input fp32 row + gamma row (no rounding drift from an
//           intermediate fp32 readback/rewrite).
//
// PAPER:    Step 4 in TENSOR_VIRTUALIZATION_PLAN.md (also referenced as the
//           §3.6 #2 fused kernel). Tested hypothesis: collapsing the per-op
//           boundary between RMSNorm and the int8 quant of the next FC
//           eliminates the int8 bucket-flip cascade that turned coherent
//           output into garbage when either op was ported in isolation.
//
// MATH:     Identical to (CPU RMSNorm output * gamma) -> v8c_act_quant_f32.
//           - rms = sqrt(mean(x^2) + eps)
//           - normalized[k] = x[k] / rms * gamma[k]
//           - asymmetric int8 quant on normalized using KAI qai8dxp formula:
//               rmin = min(0, min(norm)), rmax = max(0, max(norm))
//               scale_q = 255 / (rmax - rmin)   ; recip = 1 / scale_q
//               zp = round(closer_of(qmin - rmin*scale_q, qmax - rmax*scale_q))
//               q[k] = clamp(round(norm[k] * scale_q) + zp, -128, 127)
//
// OUTPUTS:  (4 buffers) act_i8[M*K], scale_per_row[M], zp_per_row[M],
//           row_sum_act[M]. Memory layout is byte-identical to v8c's
//           act_quant outputs so the downstream v8c FC can consume them
//           directly (skipping its own quant pass).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define FUSED_RMSQ_LWS 64

// Local-memory cache size in floats. Must be >= the largest K we ever see
// in residual-stream RMSNorm. For Qwen3-0.6B the hidden dim is 1024;
// 2048 here gives headroom for 4B/7B-class hidden sizes (1024-2048).
#define FUSED_RMSQ_KMAX 2048

__attribute__((reqd_work_group_size(FUSED_RMSQ_LWS, 1, 1)))
__kernel void fused_rmsnorm_quant_f32_par(
    __global const float *in,           // [M, K] fp32 pre-norm activation
    __global const float *gamma,        // [K] fp32 per-channel scale
    __global       char  *out_i8,       // [M, K] int8 quantized normalized act
    __global       float *out_scale,    // [M] fp32 per-row recip scale
    __global       int   *out_zp,       // [M] int32 per-row zero point
    __global       int   *out_rs,       // [M] int32 per-row sum of int8 codes
    const float epsilon,
    const int M, const int K) {
  const int row = get_group_id(0);
  if (row >= M) return;
  const int tid = get_local_id(0);

  __local float lf_partial[FUSED_RMSQ_LWS];
  __local int   li_partial[FUSED_RMSQ_LWS];
  __local float l_inv_rms;
  __local float l_min;
  __local float l_max;
  __local float l_scale_q;
  __local int   l_zp;
  // Cache normalized fp32 values so pass 3 (quant) doesn't re-read input.
  // K must be <= FUSED_RMSQ_KMAX; host caller validates this precondition.
  __local float l_norm[FUSED_RMSQ_KMAX];

  const __global float *in_row = in + (long)row * K;

  // -------------------------------------------------------------------
  // Pass 1: sum of squares -> inv_rms
  // -------------------------------------------------------------------
  float psum = 0.0f;
  for (int k = tid; k < K; k += FUSED_RMSQ_LWS) {
    const float v = in_row[k];
    psum += v * v;
  }
  lf_partial[tid] = psum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = FUSED_RMSQ_LWS / 2; s > 0; s >>= 1) {
    if (tid < s) lf_partial[tid] += lf_partial[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
    const float mean_sq = lf_partial[0] / (float)K;
    l_inv_rms = rsqrt(mean_sq + epsilon);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float inv_rms = l_inv_rms;

  // -------------------------------------------------------------------
  // Pass 2: compute normalized values, cache to local memory, accumulate
  // per-WI min/max for the asymmetric range.
  // -------------------------------------------------------------------
  float pmin = 0.0f, pmax = 0.0f;
  for (int k = tid; k < K; k += FUSED_RMSQ_LWS) {
    const float norm = in_row[k] * inv_rms * gamma[k];
    l_norm[k] = norm;
    if (norm < pmin) pmin = norm;
    if (norm > pmax) pmax = norm;
  }
  // Reduce min.
  lf_partial[tid] = pmin;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = FUSED_RMSQ_LWS / 2; s > 0; s >>= 1) {
    if (tid < s) lf_partial[tid] = fmin(lf_partial[tid], lf_partial[tid + s]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) l_min = lf_partial[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  // Reduce max.
  lf_partial[tid] = pmax;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = FUSED_RMSQ_LWS / 2; s > 0; s >>= 1) {
    if (tid < s) lf_partial[tid] = fmax(lf_partial[tid], lf_partial[tid + s]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) l_max = lf_partial[0];
  barrier(CLK_LOCAL_MEM_FENCE);

  // -------------------------------------------------------------------
  // tid==0: compute scale + zp. Math is byte-identical to v8c_act_quant_f32
  // so downstream consumers see the same output as the unfused pipeline
  // would have produced IF the input were the CPU-RMSNorm output.
  // -------------------------------------------------------------------
  if (tid == 0) {
    const float fmn = l_min, fmx = l_max;
    const float rmin = fmn < 0.0f ? fmn : 0.0f;
    const float rmax = fmx > 0.0f ? fmx : 0.0f;
    const float qmin = -128.0f, qmax = 127.0f;
    const float range = rmax - rmin;
    const float scale_q = range > 0.0f ? 255.0f / range : 1.0f;
    const float recip   = range > 0.0f ? range / 255.0f : 1.0f;
    const float dmin = rmin * scale_q;
    const float dmax = rmax * scale_q;
    const float zp_lo = qmin - dmin;
    const float zp_hi = qmax - dmax;
    float zp_f = (qmin + dmin) + (qmax + dmax) > 0.0f ? zp_lo : zp_hi;
    if (zp_f < qmin) zp_f = qmin;
    if (zp_f > qmax) zp_f = qmax;
    out_scale[row] = recip;
    out_zp[row]    = (int)rint(zp_f);
    l_scale_q = scale_q;
    l_zp      = (int)rint(zp_f);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float scale_q = l_scale_q;
  const int   zp      = l_zp;

  // -------------------------------------------------------------------
  // Pass 3: quant from cached normalized values + row_sum reduce.
  // -------------------------------------------------------------------
  int prs = 0;
  __global char *out_row = out_i8 + (long)row * K;
  for (int k = tid; k < K; k += FUSED_RMSQ_LWS) {
    int q = (int)rint(l_norm[k] * scale_q) + zp;
    if (q < -128) q = -128;
    if (q > 127)  q = 127;
    out_row[k] = (char)q;
    prs += q;
  }
  li_partial[tid] = prs;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = FUSED_RMSQ_LWS / 2; s > 0; s >>= 1) {
    if (tid < s) li_partial[tid] += li_partial[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) out_rs[row] = li_partial[0];
}
