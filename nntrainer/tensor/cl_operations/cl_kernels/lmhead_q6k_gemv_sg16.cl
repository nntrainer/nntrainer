
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void q6k_gemv_lmhead_sg16(__global const uchar *W,  // [V][H/256][212 B]
                                   __global const float *x,   // [H]
                                   __global float *logits,    // [V]
                                   const int V, const int H) {
  const int row =
    (int)(get_group_id(0) * get_num_sub_groups() + get_sub_group_id());
  if (row >= V) return;
  const int u = (int)get_sub_group_local_id(); // 0..15
  const int nb = H >> 8;
  // 212-byte blocks: ql[128] qh[64] scales[16] d[2] pad[2], word aligned.
  __global const uint *rb =
    (__global const uint *)(W + (size_t)row * (size_t)(nb * 212));
  const int nh = u >> 3;
  const int q = u & 7;
  const int is = q >> 2;
  float sum = 0.0f;
  for (int bi = 0; bi < nb; ++bi) {
    __global const uint *blk = rb + bi * 53;
    const int4 lo = convert_int4(as_uchar4(blk[(nh << 4) + q]));
    const int4 hi = convert_int4(as_uchar4(blk[(nh << 4) + 8 + q]));
    const int4 h = convert_int4(as_uchar4(blk[32 + (nh << 3) + q]));
    const char8 sc =
      as_char8((uint2)(blk[48 + (nh << 1)], blk[49 + (nh << 1)]));
    const float d = vload_half(0, (__global const half *)(blk + 52));
    const float s0 = d * (float)(is ? sc.s1 : sc.s0);
    const float s2 = d * (float)(is ? sc.s3 : sc.s2);
    const float s4 = d * (float)(is ? sc.s5 : sc.s4);
    const float s6 = d * (float)(is ? sc.s7 : sc.s6);
    const int yb = (bi << 8) + (nh << 7) + (q << 2);
    const float4 a1 = convert_float4(((lo & 0xF) | ((h & 3) << 4)) - 32);
    const float4 a2 = convert_float4(((hi & 0xF) | (((h >> 2) & 3) << 4)) - 32);
    const float4 a3 = convert_float4(((lo >> 4) | (((h >> 4) & 3) << 4)) - 32);
    const float4 a4 = convert_float4(((hi >> 4) | (((h >> 6) & 3) << 4)) - 32);
    const float4 x1 = vload4(0, x + yb);
    const float4 x2 = vload4(0, x + yb + 32);
    const float4 x3 = vload4(0, x + yb + 64);
    const float4 x4 = vload4(0, x + yb + 96);
    sum += s0 * dot(a1, x1) + s2 * dot(a2, x2) + s4 * dot(a3, x3) +
           s6 * dot(a4, x4);
  }
  sum = sub_group_reduce_add(sum);
  if (u == 0) logits[row] = sum;
}
