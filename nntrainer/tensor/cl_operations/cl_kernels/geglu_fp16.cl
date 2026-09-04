#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void geglu_cl_fp16(__global const half *in1, __global const half *in2,
                            __global half *out) {
  const int i = get_global_id(0);

  // Compute the gelu in fp32 for accuracy, multiply back in fp16.
  const float x = (float)in1[i];
  const float in2_val = (float)in2[i];

  const float k0 = 0.7978845608028654f; // sqrt(2/pi)
  const float k1 = 0.044715f;
  const float inner = k0 * (x + k1 * x * x * x);
  const float gelu = 0.5f * x * (1.0f + tanh(inner));

  out[i] = (half)(gelu * in2_val);
}
