// SPDX-License-Identifier: Apache-2.0
/**
 * @file hadamard_check.c
 * @brief Independent dense-matrix reference and DSP rotation timing.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#include "hvx_hadamard_f32.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef NNTR_HADAMARD_DEVICE
#include "nntr_hvx.h"
#include <AEEStdErr.h>
#include <remote.h>
static remote_handle64 session;
#endif

/* Independent O(n^2) definition, without butterflies or the kernel's table. */
static int coefficient(uint32_t i, uint32_t j, uint32_t p) {
  const uint32_t a = i / p, b = j / p;
  int sign = 1;
  if (!a && b)
    sign = -1;
  else if (a && b && a != b) {
    const uint32_t d = (a + 43u - b) % 43u;
    sign = -1;
    for (uint32_t k = 1u; k < 43u; ++k)
      if (k * k % 43u == d) {
        sign = 1;
        break;
      }
  }
  uint32_t bits = (i % p) & (j % p);
  while (bits) {
    sign = -sign;
    bits &= bits - 1u;
  }
  return sign;
}

/** Dephased Paley H12 from quadratic residues in F_11. Output column
 * a receives input row b: the nonsymmetric basis must not be transposed. */
static int lfm12_coefficient(uint32_t a, uint32_t b) {
  if (!a || !b)
    return 1;
  if (a == b)
    return -1;
  const uint32_t d = (a + 11u - b) % 11u;
  for (uint32_t k = 1; k < 11u; ++k)
    if (k * k % 11u == d)
      return -1;
  return 1;
}

static void check(uint32_t n, uint32_t pattern) {
  float *storage = malloc((n + 2u) * sizeof(float));
  float *x = storage + 1, *input = malloc(n * sizeof(float));
  double *ref = calloc(n, sizeof(double));
  assert(storage && input && ref);
  storage[0] = storage[n + 1u] = 917.125f;
  const uint32_t p = n == 1536u ? 128u : ((n & (n - 1u)) == 0u ? n : n / 44u);
  for (uint32_t i = 0; i < n; ++i)
    input[i] = pattern == 0u   ? sinf((float)i * .7123f) + .3f * cosf((float)i)
               : pattern == 1u ? (i == n - 1u ? 1.0f : 0.0f)
               : pattern == 2u
                 ? 1.0f
                 : (i == n / 2u ? 100.0f : (float)((int)(i % 7u) - 3) * .001f);
  /* Cache the dense sign matrix's small H44 part to keep this test cheap. */
  int h[44][44];
  for (uint32_t a = 0; a < n / p; ++a)
    for (uint32_t b = 0; b < n / p; ++b)
      h[a][b] = n == 1536u ? lfm12_coefficient(a, b) : coefficient(a, b, 1u);
  const double norm = 1.0 / sqrt((double)n);
  for (uint32_t i = 0; i < n; ++i) {
    double sum = 0;
    for (uint32_t j = 0; j < n; ++j) {
      int v = h[i / p][j / p];
      uint32_t bits = (i % p) & (j % p);
      while (bits) {
        v = -v;
        bits &= bits - 1u;
      }
      sum += v * (double)input[j];
    }
    ref[i] = sum * norm;
  }
  for (uint32_t mode = 1; mode <= 2; ++mode) {
    memcpy(x, input, n * sizeof(float));
#ifdef NNTR_HADAMARD_DEVICE
    uint32_t ticks[2];
    assert(nntr_hvx_hadamard_probe(session, 1u, n, mode, 20u, input, (int)n, x,
                                   (int)n, ticks, 2) == AEE_SUCCESS);
    const double us = ((uint64_t)ticks[1] << 32 | ticks[0]) / (19.2 * 20.0);
#else
    assert(hvx_hadamard_f32(x, n, mode) == 0);
    const double us = 0;
#endif
    double max_abs = 0, in_energy = 0, out_energy = 0;
    for (uint32_t i = 0; i < n; ++i) {
      const double err = fabs(x[i] - ref[i]);
      if (err > max_abs)
        max_abs = err;
      in_energy += (double)input[i] * input[i];
      out_energy += (double)x[i] * x[i];
      if (!isfinite(x[i]) || err >= 3e-5 * (1.0 + fabs(ref[i])))
        fprintf(stderr,
                "rotation mismatch width=%u pattern=%u mode=%u i=%u "
                "got=%.9g want=%.9g error=%.9g\n",
                n, pattern, mode, i, x[i], ref[i], err);
      assert(isfinite(x[i]) && err < 3e-5 * (1.0 + fabs(ref[i])));
    }
    assert(fabs(in_energy - out_energy) < 3e-6 * in_energy + 1e-12);
    assert(storage[0] == 917.125f && storage[n + 1u] == 917.125f);
    printf("HADAMARD width=%u pattern=%u mode=%u max_abs=%.9g us=%.6f\n", n,
           pattern, mode, max_abs, us);
  }
  free(storage);
  free(input);
  free(ref);
}

/** Distinct adjacent rows catch accidental mixing of the two HVX halves.
 * Odd sizes exercise the paired path and its single-row remainder. */
static void check_rows(uint32_t rows) {
  const uint32_t n = 704u, len = rows * n;
  float *input = malloc(len * sizeof(float));
  float *storage = malloc((len + 2u) * sizeof(float));
  float *ref = malloc(len * sizeof(float));
  assert(input && storage && ref);
  float *out = storage + 1;
  storage[0] = storage[len + 1u] = 917.125f;
  for (uint32_t i = 0; i < len; ++i)
    input[i] = sinf((float)i * .093f) + (float)(i / n) * .7f;
  memcpy(ref, input, len * sizeof(float));
  for (uint32_t r = 0; r < rows; ++r)
    assert(hvx_hadamard_f32(ref + r * n, n, 1u) == 0);
  for (uint32_t mode = 1u; mode <= 2u; ++mode) {
#ifdef NNTR_HADAMARD_DEVICE
    uint32_t ticks[2];
    assert(nntr_hvx_hadamard_probe(session, rows, n, mode, 2u, input, (int)len,
                                   out, (int)len, ticks, 2) == AEE_SUCCESS);
#else
    memcpy(out, input, len * sizeof(float));
    assert(hvx_hadamard_rows_f32(out, rows, n, mode) == 0);
#endif
    for (uint32_t i = 0; i < len; ++i)
      assert(isfinite(out[i]) &&
             fabsf(out[i] - ref[i]) < 3e-5f * (1.0f + fabsf(ref[i])));
    assert(storage[0] == 917.125f && storage[len + 1u] == 917.125f);
  }
  printf("HADAMARD ROWS OK rows=%u\n", rows);
  free(input);
  free(storage);
  free(ref);
}

int main(void) {
#ifdef NNTR_HADAMARD_DEVICE
  struct remote_rpc_control_unsigned_module u = {CDSP_DOMAIN_ID, 1};
  assert(remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &u,
                                sizeof(u)) == AEE_SUCCESS);
  struct remote_rpc_thread_params thread_params = {CDSP_DOMAIN_ID, -1,
                                                   64 * 1024};
  assert(remote_session_control(FASTRPC_THREAD_PARAMS, &thread_params,
                                sizeof(thread_params)) == AEE_SUCCESS);
  assert(nntr_hvx_open(nntr_hvx_URI "&_dom=cdsp", &session) == AEE_SUCCESS);
#endif
  const uint32_t widths[] = {1, 16, 32, 64, 128, 176, 704, 1536, 2048, 2816};
  for (uint32_t w = 0; w < sizeof(widths) / sizeof(widths[0]); ++w)
    for (uint32_t pattern = 0; pattern < 4; ++pattern)
      check(widths[w], pattern);
  const uint32_t row_counts[] = {1, 2, 3, 7, 17};
  for (uint32_t i = 0; i < sizeof(row_counts) / sizeof(row_counts[0]); ++i)
    check_rows(row_counts[i]);
  float sentinel = 91.0f;
  assert(hvx_hadamard_f32(&sentinel, 0, 2) == -1);
  assert(hvx_hadamard_f32(&sentinel, 2112, 2) == -1);
  assert(hvx_hadamard_f32(&sentinel, 8193, 2) == -1);
  assert(hvx_hadamard_f32(&sentinel, 1, 3) == -1);
  assert(hvx_hadamard_f32(NULL, 1, 2) == -1);
  assert(hvx_hadamard_rows_f32(&sentinel, 1, 2112, 2) == -1);
  assert(hvx_hadamard_rows_f32(&sentinel, 0, 704, 2) == 0);
  assert(sentinel == 91.0f);
#ifdef NNTR_HADAMARD_DEVICE
  uint32_t ticks[2];
  assert(nntr_hvx_hadamard_probe(session, 1, 2112, 2, 1, &sentinel, 1,
                                 &sentinel, 1, ticks, 2) != AEE_SUCCESS);
  assert(sentinel == 91.0f);
  /* Batched worker path, compared to the independently verified scalar. */
  const uint32_t batch_widths[] = {704, 1536, 2048, 2816};
  for (uint32_t wi = 0; wi < 4; ++wi) {
    const uint32_t n = batch_widths[wi];
    const uint32_t rows = 1024;
    const size_t len = (size_t)n * rows;
    float *in = malloc(len * sizeof(float)), *out = malloc(len * sizeof(float));
    assert(in && out);
    for (size_t i = 0; i < len; ++i)
      in[i] = sinf((float)(i % 1009) * .11f);
    for (uint32_t mode = 1; mode <= 2; ++mode) {
      assert(nntr_hvx_hadamard_probe(session, rows, n, mode, 10, in, (int)len,
                                     out, (int)len, ticks, 2) == AEE_SUCCESS);
      for (uint32_t r = 0; r < rows; r += 127) {
        float ref[2816];
        memcpy(ref, in + (size_t)r * n, n * sizeof(float));
        assert(hvx_hadamard_f32(ref, n, 1) == 0);
        for (uint32_t k = 0; k < n; ++k)
          assert(fabsf(ref[k] - out[(size_t)r * n + k]) <
                 3e-5f * (1 + fabsf(ref[k])));
      }
      printf("HADAMARD_BATCH rows=%u width=%u mode=%u us=%.6f\n", rows, n, mode,
             ((uint64_t)ticks[1] << 32 | ticks[0]) / (19.2 * 10.0));
    }
    free(in);
    free(out);
  }
  assert(nntr_hvx_close(session) == AEE_SUCCESS);
#endif
  puts("HADAMARD CHECK OK");
  return 0;
}
