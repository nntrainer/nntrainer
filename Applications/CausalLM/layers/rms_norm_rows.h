// SPDX-License-Identifier: Apache-2.0
/**
 * @file rms_norm_rows.h
 * @brief Parallel independent FP32 RMSNorm rows with the existing reduction.
 * @author NNTrainer contributors
 * @bug No known bugs.
 */
#ifndef __CAUSALLM_RMS_NORM_ROWS_H__
#define __CAUSALLM_RMS_NORM_ROWS_H__
#include <cpu_backend.h>
#include <cstdint>
#include <tensor.h>
#include <thread_manager.h>

namespace causallm {
/** Preserve each core reduction and apply gamma while its row is still hot. */
inline bool parallelRmsNormRows(const nntrainer::Tensor &src,
                                nntrainer::Tensor &dst,
                                const nntrainer::Tensor &gamma, float eps) {
  const auto fp32 = nntrainer::Tdatatype::FP32;
  const unsigned width = src.width();
  if (!width || src.size() / width < 128 || src.size() != dst.size() ||
      src.getDataType() != fp32 || dst.getDataType() != fp32 ||
      gamma.getDataType() != fp32 || gamma.size() != width ||
      !src.getContiguous() || !dst.getContiguous() || !gamma.getContiguous())
    return false;
  const float *input = src.getData<float>(), *scale = gamma.getData<float>();
  float *output = dst.getData<float>();
  const size_t bytes = src.size() * sizeof(float);
  const auto overlaps = [](const void *a, size_t an, const void *b, size_t bn) {
    const uintptr_t x = reinterpret_cast<uintptr_t>(a);
    const uintptr_t y = reinterpret_cast<uintptr_t>(b);
    return x <= y ? y - x < an : x - y < bn;
  };
  if (!input || !scale || !output ||
      (input != output && overlaps(input, bytes, output, bytes)) ||
      overlaps(scale, size_t(width) * sizeof(float), output, bytes))
    return false;
  nntrainer::ThreadManager::Global().parallel_for(
    0, src.size() / width, [&](size_t row) {
      float *out = output + row * width;
      nntrainer::rms_norm_wrt_width_fp32_intrinsic(input + row * width, out, 1,
                                                   width, eps);
      for (unsigned c = 0; c < width; ++c)
        out[c] *= scale[c];
    });
  return true;
}
} // namespace causallm
#endif
