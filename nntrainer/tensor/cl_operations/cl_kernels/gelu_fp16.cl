#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void gelu_cl_fp16(__global const half *input, __global half *output,
                           int mode, int N) {
  int i = get_global_id(0);
  if (i >= N)
    return;
  // read half, compute in float, store half (matches the fp32 kernel math)
  float x = (float)input[i];
  float y;
  if (mode == 1) {
    // tanh approximation (gelu_pytorch_tanh / ACT_TANH_GELU)
    float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    y = 0.5f * x * (1.0f + tanh(inner));
  } else {
    // erf-based exact GELU (ACT_GELU)
    y = 0.5f * x * (1.0f + erf(x * 0.70710678118654752f));
  }
  output[i] = (half)y;
}
