// SPDX-License-Identifier: Apache-2.0
/**
 * @file lfm2_norm_rows_check.cpp
 * @brief Compare production parallel RMSNorm rows with the original core path.
 * @author NNTrainer contributors
 * @bug No known bugs.
 */
#include <cpu_backend.h>
#include <cstdio>
#include <cstring>
#include <nntrainer_error.h>
#include <stdexcept>
#include <tensor.h>
#include <thread_manager.h>

#include <rms_norm_rows.h>

int main() {
  nntrainer::ThreadManagerConfig threads;
  threads.compute_threads = 8;
  threads.enable_affinity = false;
  nntrainer::ThreadManager::setConfig(threads);
  unsigned cases = 0;
  for (unsigned width : {32u, 2048u})
    for (unsigned rows : {1u, 127u, 128u, 129u, 1024u})
      for (unsigned pattern = 0; pattern < 4; ++pattern)
        for (bool in_place : {false, true}) {
          nntrainer::TensorDim dim(1, 1, rows, width);
          nntrainer::Tensor input(dim, true), output(dim, true), ref(dim, true);
          nntrainer::Tensor gamma(nntrainer::TensorDim(1, 1, 1, width), true);
          for (size_t i = 0; i < input.size(); ++i)
            input.getData<float>()[i] =
              pattern == 1
                ? 0.0f
                : float(int((i * 193u + pattern * 17u) % 509u) - 254) / 191.0f;
          for (unsigned i = 0; i < width; ++i)
            gamma.getData<float>()[i] =
              float(int((i * 13u + pattern) % 71u) - 35) / 64.0f;
          nntrainer::rms_norm_wrt_width_fp32_intrinsic(
            input.getData<float>(), ref.getData<float>(), rows, width, 1e-6f);
          ref.multiply_i(gamma);
          auto &dst = in_place ? input : output;
          const bool parallel =
            causallm::parallelRmsNormRows(input, dst, gamma, 1e-6f);
          if (parallel != (rows >= 128))
            throw std::runtime_error("parallel RMSNorm guard differs");
          if (!parallel) {
            nntrainer::rms_norm_wrt_width_fp32_intrinsic(
              input.getData<float>(), dst.getData<float>(), rows, width, 1e-6f);
            dst.multiply_i(gamma);
          }
          if (std::memcmp(dst.getData(), ref.getData(), ref.size() * 4)) {
            std::fprintf(
              stderr,
              "NORM_ROWS mismatch rows=%u width=%u pattern=%u inplace=%d\n",
              rows, width, pattern, in_place);
            return 1;
          }
          ++cases;
        }
  std::printf("NORM_ROWS %u cases bit-exact PASS\n", cases);
}
