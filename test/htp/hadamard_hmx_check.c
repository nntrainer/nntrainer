// SPDX-License-Identifier: Apache-2.0
/**
 * @file hadamard_hmx_check.c
 * @brief Full-width FP16 HMX rotation: boundaries, precision and DSP timing.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#include "nntr_hvx.h"
#include <AEEStdErr.h>
#include <assert.h>
#include <math.h>
#include <remote.h>
#include <stdio.h>
#include <stdlib.h>

static void check(remote_handle64 session, unsigned rows, unsigned n,
                  unsigned pattern) {
  const unsigned len = rows * n;
  float *x = malloc(len * sizeof(float));
  float *ref = malloc(len * sizeof(float));
  float *storage = malloc((len + 2u) * sizeof(float));
  assert(x && ref && storage);
  float *out = storage + 1;
  storage[0] = storage[len + 1u] = 917.125f;
  for (unsigned i = 0; i < len; ++i) {
    const unsigned row = i / n, col = i % n;
    const float sign = row % 2u ? -1.0f : 1.0f;
    x[i] = pattern == 0u   ? sinf(i * .01713f) + .3f * cosf(i * .173f)
           : pattern == 1u ? (col == (row * 173u + n - 1u) % n ? sign : 0.0f)
           : pattern == 2u ? sign * (1.0f + row % 7u)
           : pattern == 3u
             ? (col == n / 2u ? sign * 1000.0f : .001f * (int)(col % 3u))
           : pattern == 4u ? sign * 1000.0f
                           : 0.0f;
  }
  uint32_t ticks[2];
  assert(nntr_hvx_hadamard_probe(session, rows, n, 1u, 1u, x, len, ref, len,
                                 ticks, 2) == AEE_SUCCESS);
  const unsigned passes = rows > 64u ? 2u : 8u;
  assert(nntr_hvx_hadamard_probe(session, rows, n, 3u, passes, x, len, out, len,
                                 ticks, 2) == AEE_SUCCESS);
  double err = 0, energy = 0, out_energy = 0, max_error = 0;
  for (unsigned i = 0; i < len; ++i) {
    assert(isfinite(out[i]));
    const double d = out[i] - ref[i];
    err += d * d;
    energy += (double)ref[i] * ref[i];
    out_energy += (double)out[i] * out[i];
    if (fabs(d) > max_error)
      max_error = fabs(d);
  }
  const double nrmse = sqrt(err / fmax(energy, 1e-30));
  const double energy_error = fabs(out_energy - energy) / fmax(energy, 1e-30);
  printf("HMX rows=%u width=%u pattern=%u us=%.6f nrmse=%.9g energy_error=%.9g "
         "max_abs=%.9g\n",
         rows, n, pattern,
         ((uint64_t)ticks[1] << 32 | ticks[0]) / (19.2 * passes), nrmse,
         energy_error, max_error);
  fflush(stdout);
  /** FP16 arithmetic guard, separate from the stricter FP32 reference check
     and from the quantized pipeline. This is not a model-quality assertion. */
  assert(nrmse <= 0.002 && energy_error <= 0.003);
  assert(storage[0] == 917.125f && storage[len + 1u] == 917.125f);
  free(x);
  free(ref);
  free(storage);
}

int main(void) {
  struct remote_rpc_control_unsigned_module u = {CDSP_DOMAIN_ID, 1};
  assert(remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &u, sizeof u) ==
         AEE_SUCCESS);
  struct remote_rpc_thread_params t = {CDSP_DOMAIN_ID, -1, 64 * 1024};
  assert(remote_session_control(FASTRPC_THREAD_PARAMS, &t, sizeof t) ==
         AEE_SUCCESS);
  remote_handle64 session;
  assert(nntr_hvx_open(nntr_hvx_URI "&_dom=cdsp", &session) == AEE_SUCCESS);
  const unsigned counts[] = {1u, 2u, 3u, 17u, 33u, 65u, 1024u};
  for (unsigned n = 704u; n <= 2816u; n *= 4u)
    for (unsigned c = 0; c < sizeof counts / sizeof counts[0]; ++c)
      for (unsigned p = 0; p < 6u; ++p)
        check(session, counts[c], n, p);
  float input[704] = {0}, output[704];
  for (unsigned i = 0; i < 704u; ++i)
    output[i] = 917.125f;
  uint32_t ticks[2];
  assert(nntr_hvx_hadamard_probe(session, 1u, 64u, 3u, 1u, input, 64, output,
                                 64, ticks, 2) != AEE_SUCCESS);
  assert(nntr_hvx_hadamard_probe(session, 1u, 704u, 4u, 1u, input, 704, output,
                                 704, ticks, 2) != AEE_SUCCESS);
  /** FastRPC rout buffers are unspecified on an error return; the DSP
     validates before writing. Guard checks above cover successful calls. */
  uint32_t applied;
  assert(nntr_hvx_moe_set_opts(session, 0x30000000u, &applied) != AEE_SUCCESS);
  assert(nntr_hvx_moe_set_opts(session, 0x20000000u, &applied) != AEE_SUCCESS);
  assert(nntr_hvx_close(session) == AEE_SUCCESS);
  puts("HMX HADAMARD CHECK PASSED");
  return 0;
}
