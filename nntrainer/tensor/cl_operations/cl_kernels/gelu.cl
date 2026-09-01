__kernel void gelu_cl(__global const float *input, __global float *output,
                      int mode, int N) {
  int i = get_global_id(0);
  if (i >= N)
    return;
  float x = input[i];
  float y;
  if (mode == 1) {
    // tanh approximation (gelu_pytorch_tanh / ACT_TANH_GELU)
    float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    y = 0.5f * x * (1.0f + tanh(inner));
  } else {
    // erf-based exact GELU (ACT_GELU)
    y = 0.5f * x * (1.0f + erf(x * 0.70710678118654752f));
  }
  output[i] = y;
}
