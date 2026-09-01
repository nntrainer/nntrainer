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
rmsnorm_cl(__global const float *input, // Input tensor
           __global float *output,      // Output tensor
           __global const float *alpha, // Alpha values (one for each width)
           float epsilon,
           int H, // Height of feature map
           int W  // Width of feature map
) {
  // Compute the corresponding batch, height, and channel indices
  int h = get_group_id(0);
  int index = h * W;
  // Calculate RMS norm for the current channel, height, and batch
  __global const float4 *in = (__global const float4 *)(input + index);
  const int W4 = W / 4;
  float4 sum_squares_4 = 0.0f;
  for (int i = get_local_id(0); i < W4; i += get_local_size(0)) {
    sum_squares_4 += in[i] * in[i];
  }

  float sum_squares =
    sum_squares_4.x + sum_squares_4.y + sum_squares_4.z + sum_squares_4.w;
  // scalar tail for W not a multiple of 4 (no-op when W % 4 == 0, so the
  // float4 fast path is unchanged for hidden-size widths)
  for (int i = W4 * 4 + get_local_id(0); i < W; i += get_local_size(0)) {
    const float v = input[index + i];
    sum_squares += v * v;
  }
  sum_squares = sub_group_reduce_add(sum_squares);

  const float mean = sum_squares / W;
  const float scale = 1.0f / sqrt(mean + epsilon);

  __global float4 *out = (__global float4 *)(output + index);
  __global const float4 *a = (__global const float4 *)(alpha);
  for (int i = get_local_id(0); i < W4; i += get_local_size(0)) {
    out[i] = in[i] * scale * a[i];
  }
  // scalar tail for the output (no-op when W % 4 == 0)
  for (int i = W4 * 4 + get_local_id(0); i < W; i += get_local_size(0)) {
    output[index + i] = input[index + i] * scale * alpha[i];
  }
}
