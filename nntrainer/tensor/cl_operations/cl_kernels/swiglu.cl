__kernel void swiglu_cl(__global const float *restrict in1,
                        __global const float *restrict in2,
                        __global float *restrict out) {
  const int i = get_global_id(0);

  // Numerically stable SiLU: x/(1+exp(-x)) == x*sigmoid(x). The
  // x*exp(x)/(1+exp(x)) form overflows for x > ~88 (exp(x) = inf -> inf/inf
  // = NaN). The sigmoid form never overflows (x << 0 -> exp(-x) = inf ->
  // x/inf = 0).
  const float x = in1[i];
  const float swish = x / (1.0f + exp(-x));

  out[i] = swish * in2[i];
}
