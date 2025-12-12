/**
 * @file ggml_quantize_cpu.cpp
 * @brief CPU implementation for quantizing GGML data
 * @author Samsung R&D Institute
 * @bug No known bugs
 */
#include "ggml_quantize_cpu.h"
#include "ggml_cuda_common.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

// Helper for float to half conversion on CPU
static inline ggml_half ggml_compute_fp32_to_fp16(float x) {
  uint16_t rh;
  // Simple implementation or use a library if available.
  // For now, let's use a basic implementation or rely on bit manipulation.
  // Since we don't want to depend on external libraries, we can implement a
  // minimal version or assume the user has a way to handle this. However, for
  // correctness, let's use a standard conversion logic.

  // Using a simplified version or just casting if strict accuracy isn't
  // critical for this test setup, but for Q8_1 we need reasonable accuracy.
  // Let's use a known conversion routine.

  // F16C intrinsic if available?
  // _mm_cvtps_ph

  // Fallback C implementation:
  uint32_t x_u;
  std::memcpy(&x_u, &x, sizeof(float));

  const uint32_t sign = (x_u >> 16) & 0x8000;
  const uint32_t exp = (x_u >> 23) & 0xFF;
  const uint32_t mant = x_u & 0x7FFFFF;

  if (exp == 0) {
    rh = sign; // Denormal or zero -> zero
  } else if (exp == 255) {
    rh = sign | 0x7C00 | (mant ? 0x200 : 0); // Inf or NaN
  } else {
    int new_exp = (int)exp - 127 + 15;
    if (new_exp < 0) {
      rh = sign; // Underflow -> zero
    } else if (new_exp >= 31) {
      rh = sign | 0x7C00; // Overflow -> Inf
    } else {
      rh = sign | (new_exp << 10) | (mant >> 13);
    }
  }
  return rh;
}

#define GGML_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)

void quantize_row_q8_1_host(const float *__restrict x, void *__restrict vy,
                            int64_t k) {
  assert(QK8_1 == 32);
  assert(k % QK8_1 == 0);
  const int nb = k / QK8_1;

  block_q8_1 *__restrict y = (block_q8_1 *)vy;

  for (int i = 0; i < nb; i++) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_1; j++) {
      const float v = x[i * QK8_1 + j];
      amax = std::max(amax, std::abs(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].GGML_COMMON_AGGR_S.d = GGML_FP32_TO_FP16(d);

    int sum = 0;

    for (int j = 0; j < QK8_1 / 2; ++j) {
      const float v0 = x[i * QK8_1 + j];
      const float v1 = x[i * QK8_1 + j + QK8_1 / 2];

      const int8_t q0 = roundf(v0 * id);
      const int8_t q1 = roundf(v1 * id);

      y[i].qs[j] = q0;
      y[i].qs[j + QK8_1 / 2] = q1;

      sum += q0 + q1;
    }

    y[i].GGML_COMMON_AGGR_S.s = GGML_FP32_TO_FP16(sum * d);
  }
}
