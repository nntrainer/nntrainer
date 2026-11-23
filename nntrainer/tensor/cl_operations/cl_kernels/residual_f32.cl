
__kernel void residual_f32(__global const float *in, __global float *out,
                           const int N, const int accumulate) {
  const int i = get_global_id(0);
  if (i < N)
    out[i] = accumulate ? out[i] + in[i] : in[i];
}
