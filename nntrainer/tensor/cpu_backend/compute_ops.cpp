// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   compute_ops.cpp
 * @date   04 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  ComputeOps default-throw impls + global pointer + thread-safe init.
 *
 * `ensureComputeOps()` is the single canonical entry point that
 * guarantees `g_compute_ops` is initialized exactly once across
 * threads. It uses std::call_once, which gives us:
 *
 *   1. Mutual exclusion during init_backend() — non-idempotent setup
 *      (__ggml_init, __openblas_set_num_threads) can no longer race
 *      when getComputeOps() is hit concurrently from a cold cache.
 *
 *   2. Acquire/release synchronization — any thread returning from
 *      call_once observes the writes init_backend() made to
 *      g_compute_ops, even on the "already initialized" path. The
 *      inline getComputeOps() fast-path stays correct: the racy
 *      nullptr check only ever upgrades into ensureComputeOps(), and
 *      the read after returning is synchronized through the once_flag.
 *
 * The bulk of the file is the default-throw bodies for every virtual
 * method on ComputeOps. Concrete CPU/CL/QNN subclasses override what
 * they support; anything they don't override hits these throws,
 * tagged with the op name via throwNotImplemented().
 */

#include <compute_ops.h>

#include <mutex>
#include <stdexcept>
#include <string>

namespace nntrainer {

ComputeOps *g_compute_ops = nullptr;

namespace {
std::once_flag g_compute_ops_init_flag;
} // namespace

void ensureComputeOps() {
  std::call_once(g_compute_ops_init_flag, []() { init_backend(); });
}

[[noreturn]] void ComputeOps::throwNotImplemented(const char *op) {
  throw std::runtime_error(std::string("ComputeOps::") + op +
                           " not implemented by this backend");
}

// ----------------------------------------------------------------------------
// Default impls — all throw via throwNotImplemented(name).
// Out-of-line so the header doesn't expand 80 throw bodies in every TU.
// ----------------------------------------------------------------------------
#define NI(op) throwNotImplemented(#op)

void ComputeOps::sgemm_fp32(unsigned int, bool, bool, unsigned int,
                            unsigned int, unsigned int, float, const float *,
                            unsigned int, const float *, unsigned int, float,
                            float *, unsigned int) {
  NI(sgemm_fp32);
}
void ComputeOps::sgemv_fp32(unsigned int, bool, unsigned int, unsigned int,
                            float, const float *, unsigned int, const float *,
                            unsigned int, float, float *, unsigned int) {
  NI(sgemv_fp32);
}
float ComputeOps::sdot_fp32(unsigned int, const float *, unsigned int,
                            const float *, unsigned int) {
  NI(sdot_fp32);
}
void ComputeOps::saxpy_fp32(unsigned int, float, const float *, unsigned int,
                            float *, unsigned int) {
  NI(saxpy_fp32);
}
void ComputeOps::scopy_fp32(unsigned int, const float *, unsigned int, float *,
                            unsigned int) {
  NI(scopy_fp32);
}
void ComputeOps::sscal_fp32(unsigned int, float, float *, unsigned int) {
  NI(sscal_fp32);
}
float ComputeOps::snrm2_fp32(unsigned int, const float *, unsigned int) {
  NI(snrm2_fp32);
}
unsigned int ComputeOps::isamax_fp32(unsigned int, const float *,
                                     unsigned int) {
  NI(isamax_fp32);
}

void ComputeOps::ele_mul_fp32(unsigned int, const float *, const float *,
                              float *, float, float, unsigned int,
                              unsigned int) {
  NI(ele_mul_fp32);
}
void ComputeOps::ele_add_fp32(unsigned int, const float *, const float *,
                              float *, float, float, unsigned int,
                              unsigned int) {
  NI(ele_add_fp32);
}
void ComputeOps::ele_sub_fp32(unsigned int, const float *, const float *,
                              float *, float, float, unsigned int,
                              unsigned int) {
  NI(ele_sub_fp32);
}
void ComputeOps::ele_div_fp32(unsigned int, const float *, const float *,
                              float *, float, float, unsigned int,
                              unsigned int) {
  NI(ele_div_fp32);
}

void ComputeOps::swiglu_fp32(unsigned int, float *, float *, float *) {
  NI(swiglu_fp32);
}
void ComputeOps::swiglu_alpha_fp32(unsigned int, float *, float *, float *,
                                   float) {
  NI(swiglu_alpha_fp32);
}
void ComputeOps::tanh_gelu_fp32(unsigned int, const float *, float *) {
  NI(tanh_gelu_fp32);
}
void ComputeOps::gelu_v2_fp32(unsigned int, const float *, float *) {
  NI(gelu_v2_fp32);
}
void ComputeOps::tanh_gelu_v2_fp32(unsigned int, const float *, float *) {
  NI(tanh_gelu_v2_fp32);
}
void ComputeOps::tanh_gelu_mul_fp32(unsigned int, float *, float *, float *) {
  NI(tanh_gelu_mul_fp32);
}
void ComputeOps::tanh_gelu_v2_mul_fp32(unsigned int, float *, float *,
                                       float *) {
  NI(tanh_gelu_v2_mul_fp32);
}
float ComputeOps::max_val_fp32(unsigned int, float *) { NI(max_val_fp32); }
void ComputeOps::softmax_fp32(unsigned int, float *, float *) {
  NI(softmax_fp32);
}
bool ComputeOps::is_valid_fp32(unsigned int, const float *) {
  NI(is_valid_fp32);
}

void ComputeOps::transpose_matrix_fp32(unsigned int, unsigned int,
                                       const float *, unsigned int, float *,
                                       unsigned int) {
  NI(transpose_matrix_fp32);
}

void ComputeOps::scopy_u8(unsigned int, const uint8_t *, unsigned int,
                          uint8_t *, unsigned int) {
  NI(scopy_u8);
}
void ComputeOps::scopy_s8(unsigned int, const int8_t *, unsigned int, int8_t *,
                          unsigned int) {
  NI(scopy_s8);
}
void ComputeOps::scopy_int4_to_float32(unsigned int, const uint8_t *,
                                       unsigned int, float *, unsigned int) {
  NI(scopy_int4_to_float32);
}
void ComputeOps::copy_s16_fp32(unsigned int, const int16_t *, float *) {
  NI(copy_s16_fp32);
}
void ComputeOps::copy_u16_fp32(unsigned int, const uint16_t *, float *) {
  NI(copy_u16_fp32);
}
void ComputeOps::copy_fp32_u32(unsigned int, const float *, uint32_t *) {
  NI(copy_fp32_u32);
}
void ComputeOps::copy_fp32_u16(unsigned int, const float *, uint16_t *) {
  NI(copy_fp32_u16);
}
void ComputeOps::copy_fp32_u8(unsigned int, const float *, uint8_t *) {
  NI(copy_fp32_u8);
}
void ComputeOps::copy_fp32_s16(unsigned int, const float *, int16_t *) {
  NI(copy_fp32_s16);
}
void ComputeOps::copy_fp32_s8(unsigned int, const float *, int8_t *) {
  NI(copy_fp32_s8);
}

void ComputeOps::gemm_q4_0_fp32(unsigned int, unsigned int, unsigned int,
                                const float *, unsigned int, const void *,
                                unsigned int, float *, unsigned int) {
  NI(gemm_q4_0_fp32);
}
void ComputeOps::gemm_q4_K_fp32(unsigned int, unsigned int, unsigned int,
                                const float *, unsigned int, const void *,
                                unsigned int, float *, unsigned int) {
  NI(gemm_q4_K_fp32);
}
void ComputeOps::gemm_q6_K_fp32(unsigned int, unsigned int, unsigned int,
                                const float *, unsigned int, const void *,
                                unsigned int, float *, unsigned int) {
  NI(gemm_q6_K_fp32);
}

void ComputeOps::unpack_q4_0(const void *, void *, size_t, unsigned int,
                             unsigned int) {
  NI(unpack_q4_0);
}
void ComputeOps::unpack_q4_0x8_transpose16(const void *, uint16_t *, uint16_t *,
                                           int, int) {
  NI(unpack_q4_0x8_transpose16);
}
size_t ComputeOps::quantize_q4_0(const float *, void *, int64_t, int64_t,
                                 const float *) {
  NI(quantize_q4_0);
}
void ComputeOps::dequantize_row_q4_0(const void *, float *, int64_t) {
  NI(dequantize_row_q4_0);
}
void ComputeOps::repack_q4_0(void *, void *, size_t, unsigned int,
                             unsigned int) {
  NI(repack_q4_0);
}

void ComputeOps::clamp_fp32(const float *, float *, size_t, float, float) {
  NI(clamp_fp32);
}

void ComputeOps::scopy_int8_to_fp32_u(unsigned int, const uint8_t *,
                                      unsigned int, float *, unsigned int) {
  NI(scopy_int8_to_fp32_u);
}
void ComputeOps::scopy_int8_to_fp32_s(unsigned int, const int8_t *,
                                      unsigned int, float *, unsigned int) {
  NI(scopy_int8_to_fp32_s);
}

// Accelerator-only ops — default just throws; supports_*() lets caller skip.
void ComputeOps::gemm_q4_0_batch_fp32(std::vector<void *>, float *,
                                      std::vector<float *>, unsigned int,
                                      std::vector<unsigned int>, unsigned int) {
  NI(gemm_q4_0_batch_fp32);
}
void ComputeOps::gemm_q4_0_accel_fp32(void *, float *, float *, unsigned int,
                                      unsigned int, unsigned int) {
  NI(gemm_q4_0_accel_fp32);
}
void ComputeOps::gemv_int4_batch_fp32(std::vector<void *>,
                                      std::vector<uint16_t *>, float *,
                                      std::vector<float *>, unsigned int,
                                      std::vector<unsigned int>, unsigned int) {
  NI(gemv_int4_batch_fp32);
}
void ComputeOps::gemm_int4_batch_fp32(float *, std::vector<void *>,
                                      std::vector<uint16_t *>,
                                      std::vector<float *>, unsigned int,
                                      std::vector<unsigned int>, unsigned int,
                                      unsigned int) {
  NI(gemm_int4_batch_fp32);
}
void ComputeOps::gemv_int4_accel_fp32(char *, uint16_t *, float *, float *,
                                      unsigned int, unsigned int,
                                      unsigned int) {
  NI(gemv_int4_accel_fp32);
}
void ComputeOps::sgemm_int4_accel_fp32(float *, char *, uint16_t *, float *,
                                       unsigned int, unsigned int, unsigned int,
                                       unsigned int) {
  NI(sgemm_int4_accel_fp32);
}

#ifdef ENABLE_FP16
void ComputeOps::sgemm_fp16(unsigned int, bool, bool, unsigned int,
                            unsigned int, unsigned int, float, const _FP16 *,
                            unsigned int, const _FP16 *, unsigned int, float,
                            _FP16 *, unsigned int) {
  NI(sgemm_fp16);
}
void ComputeOps::sgemv_fp16(unsigned int, bool, unsigned int, unsigned int,
                            float, const _FP16 *, unsigned int, const _FP16 *,
                            unsigned int, float, _FP16 *, unsigned int) {
  NI(sgemv_fp16);
}
_FP16 ComputeOps::sdot_fp16(unsigned int, const _FP16 *, unsigned int,
                            const _FP16 *, unsigned int) {
  NI(sdot_fp16);
}
void ComputeOps::saxpy_fp16(unsigned int, float, const _FP16 *, unsigned int,
                            _FP16 *, unsigned int) {
  NI(saxpy_fp16);
}
void ComputeOps::scopy_fp16(unsigned int, const _FP16 *, unsigned int, _FP16 *,
                            unsigned int) {
  NI(scopy_fp16);
}
void ComputeOps::scopy_fp32_to_fp16(unsigned int, const float *, unsigned int,
                                    _FP16 *, unsigned int) {
  NI(scopy_fp32_to_fp16);
}
void ComputeOps::scopy_fp16_to_fp32(unsigned int, const _FP16 *, unsigned int,
                                    float *, unsigned int) {
  NI(scopy_fp16_to_fp32);
}
void ComputeOps::sscal_fp16(unsigned int, float, _FP16 *, unsigned int) {
  NI(sscal_fp16);
}
_FP16 ComputeOps::snrm2_fp16(unsigned int, const _FP16 *, unsigned int) {
  NI(snrm2_fp16);
}
unsigned int ComputeOps::isamax_fp16(unsigned int, const _FP16 *,
                                     unsigned int) {
  NI(isamax_fp16);
}

void ComputeOps::ele_mul_fp16(unsigned int, const _FP16 *, const _FP16 *,
                              _FP16 *, float, float, unsigned int,
                              unsigned int) {
  NI(ele_mul_fp16);
}
void ComputeOps::ele_add_fp16(unsigned int, const _FP16 *, const _FP16 *,
                              _FP16 *, float, float, unsigned int,
                              unsigned int) {
  NI(ele_add_fp16);
}
void ComputeOps::ele_sub_fp16(unsigned int, const _FP16 *, const _FP16 *,
                              _FP16 *, float, float, unsigned int,
                              unsigned int) {
  NI(ele_sub_fp16);
}
void ComputeOps::ele_div_fp16(unsigned int, const _FP16 *, const _FP16 *,
                              _FP16 *, float, float, unsigned int,
                              unsigned int) {
  NI(ele_div_fp16);
}

void ComputeOps::swiglu_fp16(unsigned int, _FP16 *, _FP16 *, _FP16 *) {
  NI(swiglu_fp16);
}
_FP16 ComputeOps::max_val_fp16(unsigned int, _FP16 *) { NI(max_val_fp16); }
void ComputeOps::softmax_fp16(unsigned int, _FP16 *, _FP16 *) {
  NI(softmax_fp16);
}
bool ComputeOps::is_valid_fp16(unsigned int, const _FP16 *) {
  NI(is_valid_fp16);
}
void ComputeOps::inv_sqrt_inplace_fp16(unsigned int, _FP16 *) {
  NI(inv_sqrt_inplace_fp16);
}

void ComputeOps::transpose_matrix_fp16(unsigned int, unsigned int,
                                       const _FP16 *, unsigned int, _FP16 *,
                                       unsigned int) {
  NI(transpose_matrix_fp16);
}

void ComputeOps::scopy_int4_to_float16(unsigned int, const uint8_t *,
                                       unsigned int, _FP16 *, unsigned int) {
  NI(scopy_int4_to_float16);
}
void ComputeOps::scopy_int8_to_float16_u(unsigned int, const uint8_t *,
                                         unsigned int, _FP16 *, unsigned int) {
  NI(scopy_int8_to_float16_u);
}
void ComputeOps::scopy_int8_to_float16_s(unsigned int, const int8_t *,
                                         unsigned int, _FP16 *, unsigned int) {
  NI(scopy_int8_to_float16_s);
}

void ComputeOps::shgemm(unsigned int, bool, bool, unsigned int, unsigned int,
                        unsigned int, float, const float *, unsigned int,
                        const _FP16 *, unsigned int, float, float *,
                        unsigned int) {
  NI(shgemm);
}
void ComputeOps::shgemv(unsigned int, bool, unsigned int, unsigned int, float,
                        const float *, unsigned int, const _FP16 *,
                        unsigned int, float, float *, unsigned int) {
  NI(shgemv);
}
void ComputeOps::hsgemm(unsigned int, bool, bool, unsigned int, unsigned int,
                        unsigned int, float, const _FP16 *, unsigned int,
                        const float *, unsigned int, float, float *,
                        unsigned int) {
  NI(hsgemm);
}
void ComputeOps::hsgemv(unsigned int, bool, unsigned int, unsigned int, float,
                        const _FP16 *, unsigned int, const float *,
                        unsigned int, float, float *, unsigned int) {
  NI(hsgemv);
}

void ComputeOps::gemm_q4_0_fp16(unsigned int, unsigned int, unsigned int,
                                const _FP16 *, unsigned int, const void *,
                                unsigned int, _FP16 *, unsigned int) {
  NI(gemm_q4_0_fp16);
}
void ComputeOps::gemm_q6_K_fp16(unsigned int, unsigned int, unsigned int,
                                const _FP16 *, unsigned int, const void *,
                                unsigned int, _FP16 *, unsigned int) {
  NI(gemm_q6_K_fp16);
}

void ComputeOps::compute_rotary_embedding_value(unsigned int, unsigned int,
                                                unsigned int, _FP16 *, _FP16 *,
                                                float *, float *) {
  NI(compute_rotary_embedding_value);
}
#endif // ENABLE_FP16

#undef NI

} // namespace nntrainer
