__kernel void geglu_cl(__global const float *restrict in1,
                       __global const float *restrict in2,
                       __global float *restrict out) {
  const int i = get_global_id(0);

  const float x = in1[i];
  const float in2_val = in2[i];

  // gelu (tanh approximation, gelu_pytorch_tanh):
  //   0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
  const float k0 = 0.7978845608028654f; // sqrt(2/pi)
  const float k1 = 0.044715f;
  const float inner = k0 * (x + k1 * x * x * x);
  const float gelu = 0.5f * x * (1.0f + tanh(inner));

  out[i] = gelu * in2_val;
}
