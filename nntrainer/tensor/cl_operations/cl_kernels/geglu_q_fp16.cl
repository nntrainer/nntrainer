#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// GeGLU (FP16) that also emits the int8 activation quantisation of the row it
// writes, so the decode step's GeGLU -> down projection pair costs one
// dispatch for the GeGLU instead of the GeGLU plus the down projection's own
// v8c_act_quant_f16_parx. The pattern of rmsnorm_cl_fp16_coop_q
// (rmsnorm_fp16.cl), with the norm half replaced by the GeGLU:
//
//  * The GeGLU half computes every element with the expression of
//    geglu_cl_fp16 / geglu_erf_cl_fp16 (geglu_fp16.cl) term for term and
//    stores the half result, so the row in memory is the row the plain kernel
//    writes.
//  * The quant half then re-READS that half row from global memory, behind a
//    global-memory barrier, exactly as the separate quantiser reads its input,
//    so the rounding to half happens in one place. It is the quant half of
//    rmsnorm_cl_fp16_coop_q verbatim: fmin/fmax and an integer sum fold the
//    per-lane partials in any order, so the result does not depend on the
//    work-group width.
//
// One work-group per row; out / act_int8 / scale / zp / row_sum are written
// for n_rows rows (the scratch of the v8c FC that consumes the handoff).
#define GEGLUQ_LWS_MAX 256

#define GEGLU_Q_QUANT_HALF                                                     \
  __global const half *q_in = out + base;                                      \
  float pmin = 0.0f, pmax = 0.0f;                                              \
  for (int k = tid; k < W; k += lsz) {                                         \
    const float v = (float)q_in[k];                                            \
    pmin = fmin(pmin, v);                                                      \
    pmax = fmax(pmax, v);                                                      \
  }                                                                            \
  lmin[tid] = pmin;                                                            \
  lmax[tid] = pmax;                                                            \
  barrier(CLK_LOCAL_MEM_FENCE);                                                \
  for (int s = lsz / 2; s > 0; s >>= 1) {                                      \
    if (tid < s) {                                                             \
      lmin[tid] = fmin(lmin[tid], lmin[tid + s]);                              \
      lmax[tid] = fmax(lmax[tid], lmax[tid + s]);                              \
    }                                                                          \
    barrier(CLK_LOCAL_MEM_FENCE);                                              \
  }                                                                            \
  if (tid == 0) {                                                              \
    const float fmn = lmin[0], fmx = lmax[0];                                  \
    const float rmin = fmn < 0.0f ? fmn : 0.0f;                                \
    const float rmax = fmx > 0.0f ? fmx : 0.0f;                                \
    const float qmin = -128.0f, qmax = 127.0f;                                 \
    const float range = rmax - rmin;                                           \
    const float scale_q = range > 0.0f ? 255.0f / range : 1.0f;                \
    const float recip = range > 0.0f ? range / 255.0f : 1.0f;                  \
    const float dmin = rmin * scale_q, dmax = rmax * scale_q;                  \
    const float zp_lo = qmin - dmin, zp_hi = qmax - dmax;                      \
    float zp_f = (qmin + dmin) + (qmax + dmax) > 0.0f ? zp_lo : zp_hi;         \
    if (zp_f < qmin)                                                           \
      zp_f = qmin;                                                             \
    if (zp_f > qmax)                                                           \
      zp_f = qmax;                                                             \
    l_scale_q = scale_q;                                                       \
    l_zp = (int)rint(zp_f);                                                    \
    scale_per_row[row] = recip;                                                \
    zp_per_row[row] = l_zp;                                                    \
  }                                                                            \
  barrier(CLK_LOCAL_MEM_FENCE);                                                \
  const float scale_q = l_scale_q;                                             \
  const int zp = l_zp;                                                         \
  int psum = 0;                                                                \
  for (int k = tid; k < W; k += lsz) {                                         \
    int q = (int)rint((float)q_in[k] * scale_q) + zp;                          \
    if (q < -128)                                                              \
      q = -128;                                                                \
    if (q > 127)                                                               \
      q = 127;                                                                 \
    act_int8[base + k] = (char)q;                                              \
    psum += q;                                                                 \
  }                                                                            \
  lsum[tid] = psum;                                                            \
  barrier(CLK_LOCAL_MEM_FENCE);                                                \
  for (int s = lsz / 2; s > 0; s >>= 1) {                                      \
    if (tid < s)                                                               \
      lsum[tid] += lsum[tid + s];                                              \
    barrier(CLK_LOCAL_MEM_FENCE);                                              \
  }                                                                            \
  if (tid == 0)                                                                \
    row_sum_act[row] = lsum[0];

#define GEGLU_Q_PROLOGUE                                                       \
  const int row = get_group_id(0);                                             \
  const int tid = get_local_id(0);                                             \
  const int lsz = (int)get_local_size(0);                                      \
  if (row >= n_rows)                                                           \
    return;                                                                    \
  const long base = (long)row * (long)W;                                       \
  __local float lmin[GEGLUQ_LWS_MAX];                                          \
  __local float lmax[GEGLUQ_LWS_MAX];                                          \
  __local int lsum[GEGLUQ_LWS_MAX];                                            \
  __local float l_scale_q;                                                     \
  __local int l_zp;

__kernel void geglu_cl_fp16_q(__global const half *in1,
                              __global const half *in2, __global half *out,
                              int n_rows, int W, __global char *act_int8,
                              __global float *scale_per_row,
                              __global int *zp_per_row,
                              __global int *row_sum_act) {
  GEGLU_Q_PROLOGUE
  /* ---- GeGLU half: geglu_cl_fp16 term for term ---- */
  for (int j = tid; j < W; j += lsz) {
    const long i = base + j;
    const float x = (float)in1[i];
    const float in2_val = (float)in2[i];

    const float k0 = 0.7978845608028654f; // sqrt(2/pi)
    const float k1 = 0.044715f;
    const float inner = k0 * (x + k1 * x * x * x);
    const float gelu = 0.5f * x * (1.0f + tanh(inner));

    out[i] = (half)(gelu * in2_val);
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  /* ---- quant half ---- */
  GEGLU_Q_QUANT_HALF
}

__kernel void geglu_erf_cl_fp16_q(__global const half *in1,
                                  __global const half *in2,
                                  __global half *out, int n_rows, int W,
                                  __global char *act_int8,
                                  __global float *scale_per_row,
                                  __global int *zp_per_row,
                                  __global int *row_sum_act) {
  GEGLU_Q_PROLOGUE
  /* ---- GeGLU half: geglu_erf_cl_fp16 term for term ---- */
  for (int j = tid; j < W; j += lsz) {
    const long i = base + j;
    const float x = (float)in1[i];
    const float gelu = 0.5f * x * (1.0f + erf(x * 0.7071067811865476f));
    out[i] = (half)(gelu * (float)in2[i]);
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  /* ---- quant half ---- */
  GEGLU_Q_QUANT_HALF
}
