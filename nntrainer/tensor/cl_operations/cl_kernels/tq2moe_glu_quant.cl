// MoE expert GeGLU fused with the int8 activation quantiser of the down
// projection's input: one launch where the expert path ran tq2moe_glu_f32
// (tq2cx_cl.cpp, kTq2MoeAux) and then v8c_act_quant_f32_par
// (int8_int4_gemm_v8c.cl) over the same rows.
//
// Built with the quantiser's options (tq2QuantOptions, plus -DV8C_AQ_PERM
// when the GEMV reads TQ2 order), so the quant half is that kernel's code
// under that kernel's options. The GeGLU half is tq2moe_glu_f32's expression
// term for term, under FP_CONTRACT OFF like the aux program it comes from.
// The GeGLU value is stored to act[] (the buffer the separate kernel wrote)
// and the quant half re-reads it, so both halves see exactly the float the
// separate pair passed through memory. Bit-identical to the pair is a
// measured property, not an assumption: every output byte was compared
// against the pair.
//
// One work-group of V8C_QUANT_LWS lanes per row; row r of the gate is
// gate[gate_off + r * K ...], of the up projection up[up_off + r * K ...].

#pragma OPENCL FP_CONTRACT OFF
inline float tq2moe_glu_value(const float x, const float u, const int exact) {
  float g;
  if (exact) {
    g = 0.5f * x * (1.0f + erf(x * 0.7071067811865476f));
  } else {
    const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    g = 0.5f * x * (1.0f + tanh(inner));
  }
  return g * u;
}
#pragma OPENCL FP_CONTRACT ON

#define V8C_QUANT_LWS 64

#ifdef V8C_AQ_PERM
#define V8C_AQ_IDX(k) (((k) & ~31) | (((k) & 7) << 2) | (((k) >> 3) & 3))
#else
#define V8C_AQ_IDX(k) (k)
#endif

__attribute__((reqd_work_group_size(V8C_QUANT_LWS, 1, 1)))
__kernel void tq2moe_glu_act_quant_f32(
    __global const float *gate, __global const float *up,
    __global       float *act_fp32,
    __global       char  *act_int8,
    __global       float *scale_per_row,
    __global       int   *zp_per_row,
    __global       int   *row_sum_act,
    const int M, const int K, const int exact,
    const int gate_off, const int up_off) {
  const int row = get_group_id(0);
  if (row >= M) return;
  const int tid = get_local_id(0);
  __local float lmin[V8C_QUANT_LWS];
  __local float lmax[V8C_QUANT_LWS];
  __local int   lsum[V8C_QUANT_LWS];
  __local float l_scale_q;
  __local int   l_zp;

  // pass 1: the GeGLU of this lane's elements, stored, and its min/max
  float pmin = 0.0f, pmax = 0.0f;
  for (int k = tid; k < K; k += V8C_QUANT_LWS) {
    const int i = row * K + k;
    const float v = tq2moe_glu_value(gate[gate_off + i], up[up_off + i], exact);
    act_fp32[(long)row * K + k] = v;
    pmin = fmin(pmin, v);
    pmax = fmax(pmax, v);
  }
  lmin[tid] = pmin;
  lmax[tid] = pmax;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = V8C_QUANT_LWS / 2; s > 0; s >>= 1) {
    if (tid < s) {
      lmin[tid] = fmin(lmin[tid], lmin[tid + s]);
      lmax[tid] = fmax(lmax[tid], lmax[tid + s]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid == 0) {
    const float fmn = lmin[0], fmx = lmax[0];
    const float rmin = fmn < 0.0f ? fmn : 0.0f;
    const float rmax = fmx > 0.0f ? fmx : 0.0f;
    const float qmin = -128.0f, qmax = 127.0f;
    const float range = rmax - rmin;
    const float scale_q = range > 0.0f ? 255.0f / range : 1.0f;
    const float recip = range > 0.0f ? range / 255.0f : 1.0f;
    const float dmin = rmin * scale_q, dmax = rmax * scale_q;
    const float zp_lo = qmin - dmin, zp_hi = qmax - dmax;
    float zp_f =
      (qmin + dmin) + (qmax + dmax) > 0.0f ? zp_lo : zp_hi;
    if (zp_f < qmin) zp_f = qmin;
    if (zp_f > qmax) zp_f = qmax;
    l_scale_q = scale_q;
    l_zp = (int)rint(zp_f);
    scale_per_row[row] = recip;
    zp_per_row[row] = l_zp;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // pass 2: quantise the stored values (this lane's own writes) + row sum
  const float scale_q = l_scale_q;
  const int zp = l_zp;
  int psum = 0;
  for (int k = tid; k < K; k += V8C_QUANT_LWS) {
    int q = (int)rint(act_fp32[(long)row * K + k] * scale_q) + zp;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    act_int8[(long)row * K + V8C_AQ_IDX(k)] = (char)q;
    psum += q;
  }
  lsum[tid] = psum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = V8C_QUANT_LWS / 2; s > 0; s >>= 1) {
    if (tid < s) lsum[tid] += lsum[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) row_sum_act[row] = lsum[0];
}
