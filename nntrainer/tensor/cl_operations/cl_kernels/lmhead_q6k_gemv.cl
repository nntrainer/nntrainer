
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void q6k_gemv_lmhead(__global const uchar *W,   // [V rows][H/256 blocks][210 B]
                              __constant float *x,        // [H] post-norm hidden
                              __global float *logits,     // [V]
                              const int V, const int H) {
  const int row = get_group_id(0);
  const int t = get_local_id(0); // 0..63
  const int nb = H >> 8;         // Q6_K blocks per row
  __global const uchar *rb = W + (size_t)row * (size_t)(nb * 210);
  const int bl = t >> 4;         // block lane 0..3
  const int u = t & 15;          // unit within block
  const int nh = u >> 3;         // n-half: 0 -> elems [0,128), 1 -> [128,256)
  const int q = u & 7;           // l-quad: l = 4q .. 4q+3
  float sum = 0.0f;
  for (int s = 0; s < nb; s += 4) {
    const int bi = s + bl;
    if (bi < nb) {
      __global const uchar *blk = rb + bi * 210;
      const uchar4 qlo = vload4(0, blk + (nh << 6) + (q << 2));
      const uchar4 qhi = vload4(0, blk + (nh << 6) + 32 + (q << 2));
      const uchar4 qh4 = vload4(0, blk + 128 + (nh << 5) + (q << 2));
      const float d = vload_half(0, (__global const half *)(blk + 208));
      const int is = q >> 2;     // (4q)/16 == (4q+3)/16 for q in 0..7
      const int sbase = 192 + (nh << 3);
      const float s0 = d * (float)((__global const char *)blk)[sbase + is];
      const float s2 = d * (float)((__global const char *)blk)[sbase + is + 2];
      const float s4 = d * (float)((__global const char *)blk)[sbase + is + 4];
      const float s6 = d * (float)((__global const char *)blk)[sbase + is + 6];
    const int yb = (bi << 8) + (nh << 7) + (q << 2);
    float4 a1, a2, a3, a4;
    a1.x = (float)((int)((qlo.x & 0xF) | (((qh4.x >> 0) & 3) << 4)) - 32);
    a1.y = (float)((int)((qlo.y & 0xF) | (((qh4.y >> 0) & 3) << 4)) - 32);
    a1.z = (float)((int)((qlo.z & 0xF) | (((qh4.z >> 0) & 3) << 4)) - 32);
    a1.w = (float)((int)((qlo.w & 0xF) | (((qh4.w >> 0) & 3) << 4)) - 32);
    a2.x = (float)((int)((qhi.x & 0xF) | (((qh4.x >> 2) & 3) << 4)) - 32);
    a2.y = (float)((int)((qhi.y & 0xF) | (((qh4.y >> 2) & 3) << 4)) - 32);
    a2.z = (float)((int)((qhi.z & 0xF) | (((qh4.z >> 2) & 3) << 4)) - 32);
    a2.w = (float)((int)((qhi.w & 0xF) | (((qh4.w >> 2) & 3) << 4)) - 32);
    a3.x = (float)((int)((qlo.x >> 4) | (((qh4.x >> 4) & 3) << 4)) - 32);
    a3.y = (float)((int)((qlo.y >> 4) | (((qh4.y >> 4) & 3) << 4)) - 32);
    a3.z = (float)((int)((qlo.z >> 4) | (((qh4.z >> 4) & 3) << 4)) - 32);
    a3.w = (float)((int)((qlo.w >> 4) | (((qh4.w >> 4) & 3) << 4)) - 32);
    a4.x = (float)((int)((qhi.x >> 4) | (((qh4.x >> 6) & 3) << 4)) - 32);
    a4.y = (float)((int)((qhi.y >> 4) | (((qh4.y >> 6) & 3) << 4)) - 32);
    a4.z = (float)((int)((qhi.z >> 4) | (((qh4.z >> 6) & 3) << 4)) - 32);
    a4.w = (float)((int)((qhi.w >> 4) | (((qh4.w >> 6) & 3) << 4)) - 32);
    const float4 x1 = (float4)(x[yb], x[yb + 1], x[yb + 2], x[yb + 3]);
    const float4 x2 = (float4)(x[yb + 32], x[yb + 33], x[yb + 34], x[yb + 35]);
    const float4 x3 = (float4)(x[yb + 64], x[yb + 65], x[yb + 66], x[yb + 67]);
    const float4 x4 = (float4)(x[yb + 96], x[yb + 97], x[yb + 98], x[yb + 99]);
    sum += s0 * dot(a1, x1) + s2 * dot(a2, x2) + s4 * dot(a3, x3) +
           s6 * dot(a4, x4);
    }
  }
  __local float red[64];
  red[t] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int off = 32; off > 0; off >>= 1) {
    if (t < off) red[t] += red[t + off];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (t == 0) logits[row] = red[0];
}
