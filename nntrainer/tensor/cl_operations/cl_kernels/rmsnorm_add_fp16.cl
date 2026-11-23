
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void rmsnorm_cl_fp16_coop_add(__global const half *input,
                                       __global half *output,
                                       __global const half *alpha,
                                       float epsilon, int n_rows, int W,
                                       __global const half *residual) {
  const int row = get_group_id(0);
  const int tid = get_local_id(0);   // 0..63
  if (row >= n_rows)
    return;
  const long base = (long)row * (long)W;
  const int W8 = W >> 3;
  __global const half8 *in8 = (__global const half8 *)(input + base);
  float partial = 0.0f;
  for (int i = tid; i < W8; i += 64) {
    const float8 v = convert_float8(in8[i]);
    partial += dot(v.lo, v.lo) + dot(v.hi, v.hi);
  }
  __local float lsum[64];
  lsum[tid] = partial;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = 32; s > 0; s >>= 1) {
    if (tid < s)
      lsum[tid] += lsum[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float mean = lsum[0] / (float)W;
  const float scale = rsqrt(mean + (float)epsilon);
  __global half8 *out8 = (__global half8 *)(output + base);
  __global const half8 *r8 = (__global const half8 *)(residual + base);
  for (int i = tid; i < W8; i += 64) {
    // Large-gamma FP16 overflow guard (see rmsnorm_fp16.cl): apply gamma in
    // FP32 and clamp to the FP16-safe range before the residual add. gamma
    // (alpha) is a model-weight pointer with no 16-byte alignment guarantee,
    // so it is loaded per element -- a half8 vector load would read garbage.
    const float8 v = convert_float8(in8[i]) * scale;
    const int gi = i << 3;
    const float8 a = (float8)(
      (float)alpha[gi + 0], (float)alpha[gi + 1], (float)alpha[gi + 2],
      (float)alpha[gi + 3], (float)alpha[gi + 4], (float)alpha[gi + 5],
      (float)alpha[gi + 6], (float)alpha[gi + 7]);
    const float8 nv = clamp(v * a, -60000.0f, 60000.0f);
    out8[i] = convert_half8(nv) + r8[i];
  }
}
