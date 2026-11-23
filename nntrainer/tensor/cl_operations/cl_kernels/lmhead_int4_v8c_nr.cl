
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifndef NR
#define NR 8
#endif
__kernel void lmhead_int4_v8c_gemv_nr(__global const uchar *W,   // [N][K/2]
                                      __global const half *x,     // [K] fp16
                                      __global const float *scale,// [N] fp32
                                      __global half *logits,      // [N]
                                      const int N, const int K) {
  const int row0 = (int)get_group_id(0) * NR;
  const int t = get_local_id(0); // 0..63
  const int kblocks = K >> 5;
  const size_t rb = (size_t)(K >> 1);
  float sum[NR];
#pragma unroll
  for (int r = 0; r < NR; ++r) sum[r] = 0.0f;
  for (int kb = t; kb < kblocks; kb += 64) {
    __global const half *xb = x + (kb << 5);
    for (int c = 0; c < 4; ++c) {
      // The one load that NR rows now share.
      const float8 a = convert_float8(vload8(0, xb + (c << 3)));
#pragma unroll
      for (int r = 0; r < NR; ++r) {
        __global const uchar *blk = W + (size_t)(row0 + r) * rb + (kb << 4);
        const uchar4 by = vload4(0, blk + (c << 2));
        const float4 lo = convert_float4(convert_int4(by & (uchar4)0x0F) - 8);
        const float4 hi = convert_float4(convert_int4(by >> (uchar4)4) - 8);
        sum[r] += dot(a.lo, lo) + dot(a.hi, hi);
      }
    }
  }
  __local float red[NR][64];
#pragma unroll
  for (int r = 0; r < NR; ++r) red[r][t] = sum[r];
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int off = 32; off > 0; off >>= 1) {
    if (t < off) {
#pragma unroll
      for (int r = 0; r < NR; ++r) red[r][t] += red[r][t + off];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (t < NR) logits[row0 + t] = (half)(red[t][0] * scale[row0 + t]);
}
