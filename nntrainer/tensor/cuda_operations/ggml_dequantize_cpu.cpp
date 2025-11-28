/**
 * @file ggml_dequantize_cpu.cpp
 * @brief CPU implementation for dequantizing GGML data
 * @author Samsung R&D Institute
 * @bug No known bugs
 */
#include "ggml_dequantize_cpu.h"
#include "ggml_cuda_common.h"

#include <cassert>
#include <cstring>

// Helper for half to float conversion on CPU
static inline float ggml_compute_fp16_to_fp32(ggml_half h) {
  uint16_t h_u = h;

  const uint32_t sign = (h_u >> 15) & 0x1;
  const uint32_t exp = (h_u >> 10) & 0x1F;
  const uint32_t mant = h_u & 0x3FF;

  uint32_t f_u;

  if (exp == 0) {
    if (mant == 0) {
      // Zero
      f_u = sign << 31;
    } else {
      // Denormal
      int e = -14;
      uint32_t m = mant;
      while ((m & 0x400) == 0) {
        m <<= 1;
        e--;
      }
      m &= 0x3FF;
      f_u = (sign << 31) | ((e + 127) << 23) | (m << 13);
    }
  } else if (exp == 31) {
    // Inf or NaN
    f_u = (sign << 31) | (0xFF << 23) | (mant << 13);
  } else {
    // Normal
    f_u = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
  }

  float result;
  std::memcpy(&result, &f_u, sizeof(float));
  return result;
}

#define GGML_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)

void dequantize_row_q8_1_host(const void *vx, float *y, int64_t k) {
  assert(QK8_1 == 32);
  assert(k % QK8_1 == 0);
  const int nb = k / QK8_1;

  const block_q8_1 *x = (const block_q8_1 *)vx;

  for (int i = 0; i < nb; i++) {
    const float d = GGML_FP16_TO_FP32(x[i].GGML_COMMON_AGGR_S.d);

    for (int j = 0; j < QK8_1; ++j) {
      y[i * QK8_1 + j] = x[i].qs[j] * d;
    }
  }
}
