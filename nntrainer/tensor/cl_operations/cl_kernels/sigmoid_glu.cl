__kernel void sigmoid_glu_cl(__global const float *restrict in1,
                             __global const float *restrict in2,
                             __global float *restrict out) {
  const int i = get_global_id(0);

  // sigmoid(x) = 1/(1+exp(-x)), spelled the same way as the CPU sigmoid_f()
  // helper and the CUDA ELTWISE_SRC kernel so the three backends agree token
  // for token. The form never overflows: x << 0 -> exp(-x) = inf -> 1/inf = 0.
  const float x = in1[i];
  const float s = 1.0f / (1.0f + exp(-x));

  out[i] = s * in2[i];
}
