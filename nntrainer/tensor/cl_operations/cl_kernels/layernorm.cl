#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#endif

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_32
#elif defined(ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif

__kernel void
layernorm_cl(__global const float *input, // Input tensor
             __global float *output,      // Output tensor
             __global const float *gamma, // Scale (one for each width)
             __global const float *beta,  // Shift (one for each width)
             float epsilon,
             int H, // Height of feature map (batch*channel*height rows)
             int W  // Width of feature map (normalized dimension)
) {
  // One workgroup normalizes one row (h); each work item strides the row and
  // the per-row reductions are collapsed with sub_group_reduce_add (mirrors
  // rmsnorm).
  int h = get_group_id(0);
  int index = h * W;
  const int W4 = W / 4;
  __global const float4 *in = (__global const float4 *)(input + index);

  // pass 1: mean = (1/W) * sum(x)
  float4 s4 = 0.0f;
  for (int i = get_local_id(0); i < W4; i += get_local_size(0)) {
    s4 += in[i];
  }
  float s = s4.x + s4.y + s4.z + s4.w;
  // scalar tail for W not a multiple of 4 (no-op when W % 4 == 0)
  for (int i = W4 * 4 + get_local_id(0); i < W; i += get_local_size(0)) {
    s += input[index + i];
  }
  s = sub_group_reduce_add(s);
  const float mean = s / W;

  // pass 2: variance = (1/W) * sum((x - mean)^2)
  float4 v4 = 0.0f;
  for (int i = get_local_id(0); i < W4; i += get_local_size(0)) {
    float4 d = in[i] - (float4)mean;
    v4 += d * d;
  }
  float v = v4.x + v4.y + v4.z + v4.w;
  for (int i = W4 * 4 + get_local_id(0); i < W; i += get_local_size(0)) {
    float d = input[index + i] - mean;
    v += d * d;
  }
  v = sub_group_reduce_add(v);
  const float scale = rsqrt(v / W + epsilon);

  // out = (x - mean) * scale * gamma + beta
  __global float4 *out = (__global float4 *)(output + index);
  __global const float4 *g = (__global const float4 *)gamma;
  __global const float4 *b = (__global const float4 *)beta;
  for (int i = get_local_id(0); i < W4; i += get_local_size(0)) {
    out[i] = (in[i] - (float4)mean) * scale * g[i] + b[i];
  }
  // scalar tail for the output (no-op when W % 4 == 0)
  for (int i = W4 * 4 + get_local_id(0); i < W; i += get_local_size(0)) {
    output[index + i] = (input[index + i] - mean) * scale * gamma[i] + beta[i];
  }
}
