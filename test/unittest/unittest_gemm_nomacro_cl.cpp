#include "gtest/gtest.h"

#include "fallback_internal.h"
#include "int4_utils.h"
#include "nntrainer_test_util.h"
#include "q4_0_utils.h"
#include "swiglu_cl.h"
#include "tensor_dim.h"
#include "timer.h"
#include <blas_kernel_interface.h>
#include <blas_kernels.h>
#include <cl_context.h>
#include <cpu_backend.h>
#include <fp16.h>
#include <layer_context.h>
#include <tensor.h>

#include "unittest_util.h"

using namespace nntrainer;

static void run_int4_gemm_test_(const uint32_t M, const uint32_t K,
                                const uint32_t N, const int scale_group_size,
                                bool use_ones = false, bool print = false) {
  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  const int INT4_BLOCK_N_SIZE = 32;
  uint32_t alignN = align(N, INT4_BLOCK_N_SIZE);
  uint32_t alignK = align(K, scale_group_size);

  static constexpr uint32_t run_count = 200;

  uint32_t input_size = M * alignK;

  std::vector<float> input;
  std::vector<float> weight_fp32;
  if (use_ones) {
    float ones_ratio = 0.1f;
    if (M * N * K > 1000000) { // For large input/output decrease ones_ratio to
                               // decrese results error
      ones_ratio = 0.01f;
    }
    // input = generate_vector(input_size, 1.0f, 1.0f);
    input = generate_01_vector(input_size, ones_ratio);
    weight_fp32 = generate_01_vector(N * K, ones_ratio);
  } else {
    input = generate_random_vector<float, false>(input_size, -1.0, 1.0);
    weight_fp32 = generate_random_vector<float, false>(N * K, -1.0, 1.0);
    // input = generate_vector(input_size, -2.0f, 2.0f);
    // weight_fp32 = generate_vector(N * K, -2.0f, 2.0f);
  }

  // Print input weights ones
  if (print && use_ones) {
    for (int x = 0; x < K; x++) {
      for (int y = 0; y < N; y++) {
        if (y % 10 == 0) {
          printf("| ");
        }
        if (weight_fp32[y * K + x] > 0.1) {
          printf("1 ");
        } else {
          printf("0 ");
        }
      }
      printf("\n");
    }
  }

  // Reference SGEMM
  std::vector<float> ref_dst(M * N, 0.0f);
  nntrainer::sgemm(0, false, true, M, N, K, 1.F, input.data(), K,
                   weight_fp32.data(), K, 0.F, ref_dst.data(), N);

  // Reference Q4_0 GEMV
  if (K % Q4_0 == 0 && N % 8 == 0) {
    size_t q4_data_size = K * N / Q4_0 * sizeof(block_q4_0);
    std::vector<float> q4_output_fp32(M * N);
    std::vector<uint8_t> q4_weight(q4_data_size);
    std::vector<uint8_t> q4_weight_repack(q4_data_size);
    nntrainer::quantize_q4_0(weight_fp32.data(), q4_weight.data(), N, K,
                             nullptr);
    nntrainer::repack_q4_0(q4_weight_repack.data(), q4_weight.data(),
                           q4_data_size, N, K);
    nntrainer::gemm_q4_0(M, N, K, input.data(), K, q4_weight_repack.data(), N,
                         q4_output_fp32.data(), N);
    float mse_q4 = mse<float>(ref_dst.data(), q4_output_fp32.data(), M * N);
    std::cout << "MSE Q4_0: " << std::setprecision(10) << mse_q4 << std::endl;
  }

  // Int4 GEMM - THE MAIN TEST
  uint16_t *input_ptr = (uint16_t *)allocateSVM(input_size * sizeof(uint16_t));
  int8_t *weight_ptr = (int8_t *)allocateSVM(alignK * alignN / 2);
  uint16_t *scale_ptr = (uint16_t *)allocateSVM(ceilDiv(K, scale_group_size) *
                                                alignN * sizeof(uint16_t));
  uint16_t *output_ptr = (uint16_t *)allocateSVM(M * alignN * sizeof(uint16_t));

  blas_cc->command_queue_inst_.enqueueSVMMap(
    input_ptr, input_size * sizeof(uint16_t), false);
  blas_cc->command_queue_inst_.enqueueSVMMap(weight_ptr, alignK * alignN / 2,
                                             false);
  blas_cc->command_queue_inst_.enqueueSVMMap(
    scale_ptr, ceilDiv(K, scale_group_size) * alignN * sizeof(uint16_t), false);

  std::vector<uint8_t> quantized_weights;
  std::vector<uint16_t> quantized_scales;
  Int4Utils::quantizeAndRepack(weight_fp32.data(), N, K, scale_group_size,
                               quantized_weights, quantized_scales);

  for (unsigned int i = 0; i < input_size; ++i) {
    input_ptr[i] = compute_fp32_to_fp16((input.data())[i]);
  }

  for (unsigned int i = 0; i < ceilDiv(K, scale_group_size) * alignN; ++i) {
    scale_ptr[i] = quantized_scales[i];
  }

  for (unsigned int i = 0; i < alignN * align(K, scale_group_size) / 2; ++i) {
    weight_ptr[i] = quantized_weights[i];
  }

  blas_cc->command_queue_inst_.enqueueSVMUnmap(input_ptr);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(weight_ptr);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(scale_ptr);
  // GPU INT4 GEMM
  auto t3 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::openvino_gemm_cl_nomacro(input_ptr, weight_ptr, scale_ptr, output_ptr, M,
                                N, K, scale_group_size);
  }

  auto t4 = std::chrono::high_resolution_clock::now();
  auto gpu_dt =
    std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();

  std::vector<float> output_fp32(M * N);
  for (unsigned int i = 0; i < M * N; ++i) {
    output_fp32[i] = compute_fp16_to_fp32(output_ptr[i]);
  }

  uint32_t first_zero_index = UINT32_MAX;
  uint32_t first_nonzero_index = UINT32_MAX;
  int zeros = 0;
  int non_zeros = 0;
  int nans = 0;

  for (uint32_t i = 0; i < M * N; ++i) {
    if (compute_fp16_to_fp32(output_ptr[i]) == 0) {
      zeros++;
      if (first_zero_index == UINT32_MAX) {
        first_zero_index = i;
      }
    } else {
      non_zeros++;
      if (first_nonzero_index == UINT32_MAX) {
        first_nonzero_index = i;
      }
    }

    if (std::isnan(compute_fp16_to_fp32(output_ptr[i]))) {
      nans++;
    }
  }

  auto debug_print_beg_end = [M, K, N](const uint16_t *const data,
                                       const uint32_t count = 6) {
    std::cout << "[";
    for (unsigned int i = 0; i < count; ++i) {
      std::cout << compute_fp16_to_fp32(data[i]) << " ";
    }
    std::cout << "][";
    for (unsigned int i = M * N - count; i < M * N; ++i) {
      std::cout << compute_fp16_to_fp32(data[i]) << " ";
    }
    std::cout << "]";
  };

  auto debug_print_beg_end_float = [M, K, N](const float *const data,
                                             const uint32_t count = 6) {
    std::cout << "[";
    for (unsigned int i = 0; i < count; ++i) {
      std::cout << data[i] << " ";
    }
    std::cout << "][";
    for (unsigned int i = M * N - count; i < M * N; ++i) {
      std::cout << data[i] << " ";
    }
    std::cout << "]";
  };

  std::cout << "INT4 GEMM : M:" << M << " x K:" << K << " x N:" << N
            << std::endl;
  std::cout << " - time : GPU = " << gpu_dt / (run_count * 1.0f) << " ms"
            << std::endl;

  std::vector<float> diff(M * N);
  float maxDiff = 0;
  for (int i = 0; i < M * N; i++) {
    diff[i] = ref_dst[i] - output_fp32[i];
    if (abs(diff[i]) > maxDiff) {
      maxDiff = diff[i];
    }
  }

  if (print) {
    if (use_ones) {
      printMatrixI("REF ", ref_dst.data(), M, N);
      printMatrixI("INT4", output_fp32.data(), M, N);
      printMatrixI("DIFF", diff.data(), M, N);
    } else {
      std::cout << " - sample: REF = ";
      debug_print_beg_end_float(ref_dst.data(), 16);
      std::cout << std::endl;
      std::cout << " - sample : GPU = ";
      debug_print_beg_end(output_ptr, 16);
      std::cout << std::endl;

      std::cout << " - zeros : " << zeros << " / " << M * N << " [ "
                << zeros * 100.0f / float(M * N) << " %] - first at [ "
                << first_zero_index << " ]" << std::endl;
      std::cout << " - non zeros : " << non_zeros << " / " << M * N << " [ "
                << non_zeros * 100.0f / float(M * N) << " %] - first at [ "
                << first_nonzero_index << " ]" << std::endl;
      std::cout << " - nans : " << nans << " / " << M * N << " [ "
                << nans * 100.0f / float(M * N) << " %]" << std::endl;
    }
  }
  printf("maxDiff:%f\n", maxDiff);

  float mse_int4_err = mse<float>(ref_dst.data(), output_fp32.data(), M * N);
  std::cout << "MSE int4: " << mse_int4_err << std::endl;

  if (use_ones) {
    EXPECT_IN_RANGE(mse_int4_err, 0.0f, 0.0001f);
  }

  freeSVM(weight_ptr);
  freeSVM(scale_ptr);
  freeSVM(input_ptr);
  freeSVM(output_ptr);
}

#define DECLARE_int4_gemm_test_M_K_N(M, K, N, G)                               \
  TEST(nntrainer_blas_kernel, int4_gemm_nomacro_test_##M##_##K##_##N##_Group##G) {     \
    run_int4_gemm_test_(M, K, N, G);                                           \
  }

DECLARE_int4_gemm_test_M_K_N(28, 3072, 256, 32);
DECLARE_int4_gemm_test_M_K_N(28, 3072, 8192, 32);
DECLARE_int4_gemm_test_M_K_N(28, 8192, 3072, 32);
DECLARE_int4_gemm_test_M_K_N(28, 3072, 3072, 32);

DECLARE_int4_gemm_test_M_K_N(28, 3072, 256, 128);
DECLARE_int4_gemm_test_M_K_N(28, 3072, 8192, 128);
DECLARE_int4_gemm_test_M_K_N(28, 8192, 3072, 128);
DECLARE_int4_gemm_test_M_K_N(28, 3072, 3072, 128);

DECLARE_int4_gemm_test_M_K_N(28, 32, 3072, 32);
DECLARE_int4_gemm_test_M_K_N(28, 64, 3072, 32);

DECLARE_int4_gemm_test_M_K_N(28, 3000, 3072, 128);
DECLARE_int4_gemm_test_M_K_N(28, 2000, 3072, 128);

DECLARE_int4_gemm_test_M_K_N(4, 3060, 3072, 32);
DECLARE_int4_gemm_test_M_K_N(4, 3072, 3072, 32);


GTEST_API_ int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
