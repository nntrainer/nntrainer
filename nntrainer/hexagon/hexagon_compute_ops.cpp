// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   hexagon_compute_ops.cpp
 * @date   23 July 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @brief  Hexagon cDSP ComputeOps subclass.
 *
 * Stage 2: gemm_q4_0_accel_fp32() dlopen/dlsyms
 * nntr_htp_bridge_gemm_q4_0() out of libggml-hexagon.so (see
 * ggml/src/ggml-hexagon/nntr-htp-bridge.cpp in the ggml-hexagon repo) and
 * calls straight into it. That bridge function reuses ggml-hexagon's own
 * cDSP session machinery directly (ggml_hexagon_session::enqueue_op/flush)
 * without going through ggml's graph/backend scheduler, since nntrainer has
 * no ggml_cgraph of its own to hand it. matAdata is already q4x4x2-tiled by
 * repack_q4_0_to_htp_q4x4x2 at quantize time, so no repack happens here or
 * on the DSP side.
 *
 * UNVERIFIED: no Hexagon hardware was available while writing this, so
 * dlopen/dlsym wiring and the ggml-hexagon-side bridge are only
 * compile-checked (Android arm64 cross-build), not run. The M/N/K -> tensor
 * shape mapping in nntr-htp-bridge.cpp is confirmed against nntrainer's own
 * CPU reference test (unittest_nntrainer_cpu_backend.cpp), not against real
 * hardware or a reference DSP GEMM - see the UNVERIFIED note in that file
 * before trusting output values.
 *
 * Every op below except gemm_q4_0_accel_fp32 forwards to get_cpu_ops().
 * ComputeOps::setComputeOps() is set once at the *context* level
 * (HexagonContext::initialize(), mirroring ClContext), so every tensor
 * created through a layer registered under the "cdsp" context - not just
 * the Q4_0 weight tensor - gets this ComputeOps for every op it runs, not
 * only the one op actually accelerated. The base ComputeOps class throws
 * "not implemented" for anything a subclass doesn't override (by design -
 * see compute_ops.h) - originally this class only overrode the one
 * accelerated method, on the assumption a Q4_0 weight tensor would only
 * ever have gemm_q4_0_accel_fp32 called on it. That assumption was wrong in
 * practice: float_tensor.cpp's GEMV path (decode, M=1) calls sgemv_fp32 on
 * the *same* context, which was hitting ComputeOps's default throw and
 * surfacing as "sgemv_fp32" (not implemented) on-device. ClComputeOps has
 * the identical gap (cl_operations/cl_compute_ops.cpp overrides ~12 of the
 * ~60+ methods) - it was never noticed there because no layer in this
 * codebase was ever tagged engine=gpu before engine=cdsp was wired into
 * CausalLM's Qwen3 model.
 */

#include <compute_ops.h>
#include <hexagon_compute_ops.h>
#include <hexagon_repack.h>

#include <dlfcn.h>

#include <cstdint>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace nntrainer {

namespace {

using nntr_htp_bridge_gemm_q4_0_fn = int (*)(const void *, const float *,
                                              float *, unsigned int,
                                              unsigned int, unsigned int);
using nntr_htp_bridge_upload_fn = int (*)(const void *, const void *,
                                           unsigned int, unsigned int);

struct BridgeApi {
  nntr_htp_bridge_upload_fn upload = nullptr;
  nntr_htp_bridge_gemm_q4_0_fn gemm = nullptr;
};

/**
 * @brief Lazily dlopen libggml-hexagon.so and dlsym the bridge entry points.
 * Cached for the process lifetime - dlopen/dlsym cost is not worth paying
 * per GEMM call, and ggml-hexagon's own session (kept alive across calls on
 * its side) already assumes one long-lived library handle.
 */
const BridgeApi &get_bridge_api() {
  static BridgeApi api = [] {
    void *handle = dlopen("libggml-hexagon.so", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
      std::ostringstream oss;
      oss << "HexagonComputeOps: dlopen(libggml-hexagon.so) failed: "
          << dlerror();
      throw std::runtime_error(oss.str());
    }

    auto sym = [handle](const char *name) {
      void *s = dlsym(handle, name);
      if (!s) {
        std::ostringstream oss;
        oss << "HexagonComputeOps: dlsym(" << name << ") failed: " << dlerror();
        throw std::runtime_error(oss.str());
      }
      return s;
    };

    BridgeApi a;
    a.upload = reinterpret_cast<nntr_htp_bridge_upload_fn>(
      sym("nntr_htp_bridge_upload_weight_q4x4x2"));
    a.gemm = reinterpret_cast<nntr_htp_bridge_gemm_q4_0_fn>(
      sym("nntr_htp_bridge_gemm_q4_0"));
    return a;
  }();

  return api;
}

} // namespace

class HexagonComputeOps : public ComputeOps {
public:
  bool supports_gemm_q4_0_accel_fp32() const override { return true; }

  /**
   * @brief 32 rows, matching the DSP's own HMX gate.
   *
   * htp/matmul-ops.c computes m_hmx = M & ~31 and drops to the HVX kernel when
   * that is zero, so below M == 32 the systolic array never engages and the DSP
   * is doing wide-SIMD work no better than ARM's - while still paying a FastRPC
   * round trip per op. Measured on an 8 Elite (v79) with Qwen3-0.6B, this is not
   * marginal: ggml-hexagon's own mature backend (full graph scheduler, op
   * batching, weights resident from load) decodes at 34.6 tok/s against 158.9 on
   * 4 CPU threads - 4.6x slower. Decode is GEMV, i.e. bandwidth-bound, and the
   * DSP has no bandwidth advantage. That same backend is 1.41x faster than CPU
   * at a 90-token prefill and 3.58x faster at 512 tokens, because HMX scales
   * with M while CPU attention is O(n^2).
   *
   * So the split is: prefill on the DSP, decode on the CPU. Threshold 32 puts it
   * exactly where the hardware itself switches kernels.
   */
  unsigned int gemm_q4_0_accel_min_rows() const override { return 32; }

  void gemm_q4_0_accel_fp32(void *matAdata, float *matBdata, float *matCdata,
                            unsigned int M, unsigned int N,
                            unsigned int K) override {
    static std::mutex bridge_init_mutex;
    const BridgeApi *api;
    {
      // get_bridge_api()'s static-local init is already thread-safe (C++11
      // magic statics); the mutex here only serializes the dlopen/dlsym
      // path itself against concurrent first-callers so a failed attempt
      // from one thread can't race a second attempt from another.
      std::lock_guard<std::mutex> lock(bridge_init_mutex);
      api = &get_bridge_api();
    }

    // matAdata is in the *ARM* q4_0x4 layout, because the CPU kernels still read
    // these same bytes for every decode step (M < 32, see
    // gemm_q4_0_accel_min_rows). The DSP needs q4x4x2, so convert on first
    // sight only: q4_0x4 -> plain block_q4_0 -> q4x4x2, then hand the result to
    // the bridge, which copies it into a persistent rpcmem arena keyed on
    // matAdata. All three layouts are 9K/16 bytes per row, so the scratches are
    // exactly one weight each and get reused across weights - peak extra host
    // memory is 2x the largest FC weight, not 2x the model.
    //
    // This is what lets the model .bin stay in its ordinary ARM layout: the
    // q4x4x2 copy exists only in rpcmem, derived at load, and no
    // Hexagon-specific weight file is needed.
    if (uploaded_.find(matAdata) == uploaded_.end()) {
      const size_t nbytes = (size_t)N * ((size_t)K / 32) * 18;

      if (unpack_scratch_.size() < nbytes) {
        unpack_scratch_.resize(nbytes);
        htp_scratch_.resize(nbytes);
      }

      cpu_->unpack_q4_0(matAdata, unpack_scratch_.data(), nbytes, N, K);
      repack_q4_0_to_htp_q4x4x2(htp_scratch_.data(), unpack_scratch_.data(),
                                nbytes, N, K);

      if (api->upload(matAdata, htp_scratch_.data(), N, K) != 0) {
        throw std::runtime_error(
          "HexagonComputeOps::gemm_q4_0_accel_fp32: "
          "nntr_htp_bridge_upload_weight_q4x4x2 failed (see log for details)");
      }
      uploaded_.insert(matAdata);
    }

    int rc = api->gemm(matAdata, matBdata, matCdata, M, N, K);
    if (rc != 0) {
      throw std::runtime_error(
        "HexagonComputeOps::gemm_q4_0_accel_fp32: "
        "nntr_htp_bridge_gemm_q4_0 failed (see log for details)");
    }
  }

  // ===========================================================================
  // Everything below forwards to get_cpu_ops() - see the class-level comment
  // for why this is necessary (context-wide ComputeOps attachment, not
  // per-tensor/per-op).
  // ===========================================================================

  // --- FP32 BLAS ---
  void sgemm_fp32(const unsigned int TStorageOrder, bool TransA, bool TransB,
                  const unsigned int M, const unsigned int N,
                  const unsigned int K, const float alpha, const float *A,
                  const unsigned int lda, const float *B,
                  const unsigned int ldb, const float beta, float *C,
                  const unsigned int ldc) override {
    cpu_->sgemm_fp32(TStorageOrder, TransA, TransB, M, N, K, alpha, A, lda, B,
                     ldb, beta, C, ldc);
  }
  void sgemv_fp32(const unsigned int TStorageOrder, bool TransA,
                  const unsigned int M, const unsigned int N,
                  const float alpha, const float *A, const unsigned int lda,
                  const float *X, const unsigned int incX, const float beta,
                  float *Y, const unsigned int incY) override {
    cpu_->sgemv_fp32(TStorageOrder, TransA, M, N, alpha, A, lda, X, incX, beta,
                     Y, incY);
  }
  float sdot_fp32(const unsigned int N, const float *X,
                  const unsigned int incX, const float *Y,
                  const unsigned int incY) override {
    return cpu_->sdot_fp32(N, X, incX, Y, incY);
  }
  void saxpy_fp32(const unsigned int N, const float alpha, const float *X,
                  const unsigned int incX, float *Y,
                  const unsigned int incY) override {
    cpu_->saxpy_fp32(N, alpha, X, incX, Y, incY);
  }
  void scopy_fp32(const unsigned int N, const float *X,
                  const unsigned int incX, float *Y,
                  const unsigned int incY) override {
    cpu_->scopy_fp32(N, X, incX, Y, incY);
  }
  void sscal_fp32(const unsigned int N, const float alpha, float *X,
                  const unsigned int incX) override {
    cpu_->sscal_fp32(N, alpha, X, incX);
  }
  float snrm2_fp32(const unsigned int N, const float *X,
                   const unsigned int incX) override {
    return cpu_->snrm2_fp32(N, X, incX);
  }
  unsigned int isamax_fp32(const unsigned int N, const float *X,
                           const unsigned int incX) override {
    return cpu_->isamax_fp32(N, X, incX);
  }

  // --- FP32 Element-wise ---
  void ele_mul_fp32(const unsigned int N, const float *X, const float *Y,
                    float *Z, float alpha, float beta, unsigned int i_stride,
                    unsigned int o_stride) override {
    cpu_->ele_mul_fp32(N, X, Y, Z, alpha, beta, i_stride, o_stride);
  }
  void ele_add_fp32(const unsigned int N, const float *X, const float *Y,
                    float *Z, float alpha, float beta, unsigned int i_stride,
                    unsigned int o_stride) override {
    cpu_->ele_add_fp32(N, X, Y, Z, alpha, beta, i_stride, o_stride);
  }
  void ele_sub_fp32(const unsigned int N, const float *X, const float *Y,
                    float *Z, float alpha, float beta, unsigned int i_stride,
                    unsigned int o_stride) override {
    cpu_->ele_sub_fp32(N, X, Y, Z, alpha, beta, i_stride, o_stride);
  }
  void ele_div_fp32(const unsigned int N, const float *X, const float *Y,
                    float *Z, float alpha, float beta, unsigned int i_stride,
                    unsigned int o_stride) override {
    cpu_->ele_div_fp32(N, X, Y, Z, alpha, beta, i_stride, o_stride);
  }

  // --- FP32 Activation / Special ---
  void swiglu_fp32(const unsigned int N, float *X, float *Y,
                   float *Z) override {
    cpu_->swiglu_fp32(N, X, Y, Z);
  }
  void swiglu_alpha_fp32(const unsigned int N, float *X, float *Y, float *Z,
                         float alpha) override {
    cpu_->swiglu_alpha_fp32(N, X, Y, Z, alpha);
  }
  void tanh_gelu_fp32(const unsigned int N, const float *X,
                      float *Y) override {
    cpu_->tanh_gelu_fp32(N, X, Y);
  }
  void gelu_v2_fp32(const unsigned int N, const float *X, float *Y) override {
    cpu_->gelu_v2_fp32(N, X, Y);
  }
  void tanh_gelu_v2_fp32(const unsigned int N, const float *X,
                         float *Y) override {
    cpu_->tanh_gelu_v2_fp32(N, X, Y);
  }
  void tanh_gelu_mul_fp32(const unsigned int N, float *X, float *Y,
                          float *Z) override {
    cpu_->tanh_gelu_mul_fp32(N, X, Y, Z);
  }
  void tanh_gelu_v2_mul_fp32(const unsigned int N, float *X, float *Y,
                             float *Z) override {
    cpu_->tanh_gelu_v2_mul_fp32(N, X, Y, Z);
  }
  float max_val_fp32(const unsigned int N, float *X) override {
    return cpu_->max_val_fp32(N, X);
  }
  void softmax_fp32(const unsigned int N, float *X, float *Y) override {
    cpu_->softmax_fp32(N, X, Y);
  }
  bool is_valid_fp32(const unsigned int N, const float *X) override {
    return cpu_->is_valid_fp32(N, X);
  }

  // --- FP32 Matrix ops ---
  void transpose_matrix_fp32(const unsigned int M, const unsigned int N,
                             const float *src, unsigned int ld_src,
                             float *dst, unsigned int ld_dst) override {
    cpu_->transpose_matrix_fp32(M, N, src, ld_src, dst, ld_dst);
  }

  // --- FP32 Data conversion / Copy ---
  void scopy_u8(const unsigned int N, const uint8_t *X,
               const unsigned int incX, uint8_t *Y,
               const unsigned int incY) override {
    cpu_->scopy_u8(N, X, incX, Y, incY);
  }
  void scopy_s8(const unsigned int N, const int8_t *X,
               const unsigned int incX, int8_t *Y,
               const unsigned int incY) override {
    cpu_->scopy_s8(N, X, incX, Y, incY);
  }
  void scopy_int4_to_float32(const unsigned int N, const uint8_t *X,
                             const unsigned int incX, float *Y,
                             const unsigned int incY) override {
    cpu_->scopy_int4_to_float32(N, X, incX, Y, incY);
  }
  void copy_s16_fp32(const unsigned int N, const int16_t *X,
                     float *Y) override {
    cpu_->copy_s16_fp32(N, X, Y);
  }
  void copy_u16_fp32(const unsigned int N, const uint16_t *X,
                     float *Y) override {
    cpu_->copy_u16_fp32(N, X, Y);
  }
  void copy_fp32_u32(const unsigned int N, const float *X,
                     uint32_t *Y) override {
    cpu_->copy_fp32_u32(N, X, Y);
  }
  void copy_fp32_u16(const unsigned int N, const float *X,
                     uint16_t *Y) override {
    cpu_->copy_fp32_u16(N, X, Y);
  }
  void copy_fp32_u8(const unsigned int N, const float *X,
                    uint8_t *Y) override {
    cpu_->copy_fp32_u8(N, X, Y);
  }
  void copy_fp32_s16(const unsigned int N, const float *X,
                     int16_t *Y) override {
    cpu_->copy_fp32_s16(N, X, Y);
  }
  void copy_fp32_s8(const unsigned int N, const float *X,
                    int8_t *Y) override {
    cpu_->copy_fp32_s8(N, X, Y);
  }

  // --- Quantized GEMM (GGUF format) ---
  void gemm_q4_0_fp32(const unsigned int M, const unsigned int N,
                      const unsigned int K, const float *A,
                      const unsigned int lda, const void *B,
                      const unsigned int ldb, float *C,
                      const unsigned int ldc) override {
    cpu_->gemm_q4_0_fp32(M, N, K, A, lda, B, ldb, C, ldc);
  }
  void gemm_q4_K_fp32(const unsigned int M, const unsigned int N,
                      const unsigned int K, const float *A,
                      const unsigned int lda, const void *B,
                      const unsigned int ldb, float *C,
                      const unsigned int ldc) override {
    cpu_->gemm_q4_K_fp32(M, N, K, A, lda, B, ldb, C, ldc);
  }
  void gemm_q6_K_fp32(const unsigned int M, const unsigned int N,
                      const unsigned int K, const float *A,
                      const unsigned int lda, const void *B,
                      const unsigned int ldb, float *C,
                      const unsigned int ldc) override {
    cpu_->gemm_q6_K_fp32(M, N, K, A, lda, B, ldb, C, ldc);
  }

  // --- Quantized weight packing / quantization ---
  void unpack_q4_0(const void *in_q4_0x, void *out_q4_0, size_t data_size,
                   const unsigned int M, const unsigned int N) override {
    cpu_->unpack_q4_0(in_q4_0x, out_q4_0, data_size, M, N);
  }
  void unpack_q4_0x8_transpose16(const void *src, uint16_t *d_out,
                                 uint16_t *qs_out, int N, int K) override {
    cpu_->unpack_q4_0x8_transpose16(src, d_out, qs_out, N, K);
  }
  size_t quantize_q4_0(const float *src, void *dst, int64_t nrow,
                       int64_t n_per_row,
                       const float *quant_weights) override {
    return cpu_->quantize_q4_0(src, dst, nrow, n_per_row, quant_weights);
  }
  void dequantize_row_q4_0(const void *x, float *y, int64_t k) override {
    cpu_->dequantize_row_q4_0(x, y, k);
  }
  void repack_q4_0(void *dst, void *src, size_t data_size,
                   const unsigned int M, const unsigned int N) override {
    cpu_->repack_q4_0(dst, src, data_size, M, N);
  }

  // --- Clamp ---
  void clamp_fp32(const float *input, float *output, size_t length,
                  float lower_bound, float upper_bound) override {
    cpu_->clamp_fp32(input, output, length, lower_bound, upper_bound);
  }

  // --- Data conversion (int8 -> FP32) ---
  void scopy_int8_to_fp32_u(const unsigned int N, const uint8_t *X,
                            const unsigned int incX, float *Y,
                            const unsigned int incY) override {
    cpu_->scopy_int8_to_fp32_u(N, X, incX, Y, incY);
  }
  void scopy_int8_to_fp32_s(const unsigned int N, const int8_t *X,
                            const unsigned int incX, float *Y,
                            const unsigned int incY) override {
    cpu_->scopy_int8_to_fp32_s(N, X, incX, Y, incY);
  }

  // --- Other accelerator-only ops (no Hexagon accel yet - CPU fallback) ---
  void gemm_q4_0_batch_fp32(std::vector<void *> matAdata, float *matBdata,
                            std::vector<float *> matCdata, unsigned int M,
                            std::vector<unsigned int> N,
                            unsigned int K) override {
    cpu_->gemm_q4_0_batch_fp32(matAdata, matBdata, matCdata, M, N, K);
  }
  void gemv_int4_batch_fp32(std::vector<void *> weights,
                            std::vector<uint16_t *> scales, float *input,
                            std::vector<float *> outputs, unsigned int K,
                            std::vector<unsigned int> Ns,
                            unsigned int group_size) override {
    cpu_->gemv_int4_batch_fp32(weights, scales, input, outputs, K, Ns,
                               group_size);
  }
  void gemm_int4_batch_fp32(float *input, std::vector<void *> weights,
                            std::vector<uint16_t *> scales,
                            std::vector<float *> matCdata, unsigned int M,
                            std::vector<unsigned int> Ns, unsigned int K,
                            unsigned int group_size) override {
    cpu_->gemm_int4_batch_fp32(input, weights, scales, matCdata, M, Ns, K,
                               group_size);
  }
  void gemv_int4_accel_fp32(char *weight, uint16_t *scale, float *input,
                            float *output, unsigned int K, unsigned int N,
                            unsigned int group_size) override {
    cpu_->gemv_int4_accel_fp32(weight, scale, input, output, K, N,
                               group_size);
  }
  void sgemm_int4_accel_fp32(float *input, char *weight, uint16_t *scale,
                             float *output, unsigned int M, unsigned int N,
                             unsigned int K, unsigned int group_size) override {
    cpu_->sgemm_int4_accel_fp32(input, weight, scale, output, M, N, K,
                                group_size);
  }

#ifdef ENABLE_FP16
  // --- FP16 BLAS ---
  void sgemm_fp16(const unsigned int TStorageOrder, bool TransA, bool TransB,
                  const unsigned int M, const unsigned int N,
                  const unsigned int K, const float alpha, const _FP16 *A,
                  const unsigned int lda, const _FP16 *B,
                  const unsigned int ldb, const float beta, _FP16 *C,
                  const unsigned int ldc) override {
    cpu_->sgemm_fp16(TStorageOrder, TransA, TransB, M, N, K, alpha, A, lda, B,
                     ldb, beta, C, ldc);
  }
  void sgemv_fp16(const unsigned int TStorageOrder, bool TransA,
                  const unsigned int M, const unsigned int N,
                  const float alpha, const _FP16 *A, const unsigned int lda,
                  const _FP16 *X, const unsigned int incX, const float beta,
                  _FP16 *Y, const unsigned int incY) override {
    cpu_->sgemv_fp16(TStorageOrder, TransA, M, N, alpha, A, lda, X, incX, beta,
                     Y, incY);
  }
  _FP16 sdot_fp16(const unsigned int N, const _FP16 *X,
                  const unsigned int incX, const _FP16 *Y,
                  const unsigned int incY) override {
    return cpu_->sdot_fp16(N, X, incX, Y, incY);
  }
  void saxpy_fp16(const unsigned int N, const float alpha, const _FP16 *X,
                  const unsigned int incX, _FP16 *Y,
                  const unsigned int incY) override {
    cpu_->saxpy_fp16(N, alpha, X, incX, Y, incY);
  }
  void scopy_fp16(const unsigned int N, const _FP16 *X,
                  const unsigned int incX, _FP16 *Y,
                  const unsigned int incY) override {
    cpu_->scopy_fp16(N, X, incX, Y, incY);
  }
  void scopy_fp32_to_fp16(const unsigned int N, const float *X,
                          const unsigned int incX, _FP16 *Y,
                          const unsigned int incY) override {
    cpu_->scopy_fp32_to_fp16(N, X, incX, Y, incY);
  }
  void scopy_fp16_to_fp32(const unsigned int N, const _FP16 *X,
                          const unsigned int incX, float *Y,
                          const unsigned int incY) override {
    cpu_->scopy_fp16_to_fp32(N, X, incX, Y, incY);
  }
  void sscal_fp16(const unsigned int N, const float alpha, _FP16 *X,
                  const unsigned int incX) override {
    cpu_->sscal_fp16(N, alpha, X, incX);
  }
  _FP16 snrm2_fp16(const unsigned int N, const _FP16 *X,
                   const unsigned int incX) override {
    return cpu_->snrm2_fp16(N, X, incX);
  }
  unsigned int isamax_fp16(const unsigned int N, const _FP16 *X,
                           const unsigned int incX) override {
    return cpu_->isamax_fp16(N, X, incX);
  }

  // --- FP16 Element-wise ---
  void ele_mul_fp16(const unsigned int N, const _FP16 *X, const _FP16 *Y,
                    _FP16 *Z, float alpha, float beta, unsigned int i_stride,
                    unsigned int o_stride) override {
    cpu_->ele_mul_fp16(N, X, Y, Z, alpha, beta, i_stride, o_stride);
  }
  void ele_add_fp16(const unsigned int N, const _FP16 *X, const _FP16 *Y,
                    _FP16 *Z, float alpha, float beta, unsigned int i_stride,
                    unsigned int o_stride) override {
    cpu_->ele_add_fp16(N, X, Y, Z, alpha, beta, i_stride, o_stride);
  }
  void ele_sub_fp16(const unsigned int N, const _FP16 *X, const _FP16 *Y,
                    _FP16 *Z, float alpha, float beta, unsigned int i_stride,
                    unsigned int o_stride) override {
    cpu_->ele_sub_fp16(N, X, Y, Z, alpha, beta, i_stride, o_stride);
  }
  void ele_div_fp16(const unsigned int N, const _FP16 *X, const _FP16 *Y,
                    _FP16 *Z, float alpha, float beta, unsigned int i_stride,
                    unsigned int o_stride) override {
    cpu_->ele_div_fp16(N, X, Y, Z, alpha, beta, i_stride, o_stride);
  }

  // --- FP16 Activation / Special ---
  void swiglu_fp16(const unsigned int N, _FP16 *X, _FP16 *Y,
                   _FP16 *Z) override {
    cpu_->swiglu_fp16(N, X, Y, Z);
  }
  _FP16 max_val_fp16(const unsigned int N, _FP16 *X) override {
    return cpu_->max_val_fp16(N, X);
  }
  void softmax_fp16(const unsigned int N, _FP16 *X, _FP16 *Y) override {
    cpu_->softmax_fp16(N, X, Y);
  }
  bool is_valid_fp16(const unsigned int N, const _FP16 *X) override {
    return cpu_->is_valid_fp16(N, X);
  }
  void inv_sqrt_inplace_fp16(const unsigned int N, _FP16 *X) override {
    cpu_->inv_sqrt_inplace_fp16(N, X);
  }

  // --- FP16 Matrix ops ---
  void transpose_matrix_fp16(const unsigned int M, const unsigned int N,
                             const _FP16 *src, unsigned int ld_src,
                             _FP16 *dst, unsigned int ld_dst) override {
    cpu_->transpose_matrix_fp16(M, N, src, ld_src, dst, ld_dst);
  }

  // --- FP16 Data conversion ---
  void scopy_int4_to_float16(const unsigned int N, const uint8_t *X,
                             const unsigned int incX, _FP16 *Y,
                             const unsigned int incY) override {
    cpu_->scopy_int4_to_float16(N, X, incX, Y, incY);
  }
  void scopy_int8_to_float16_u(const unsigned int N, const uint8_t *X,
                               const unsigned int incX, _FP16 *Y,
                               const unsigned int incY) override {
    cpu_->scopy_int8_to_float16_u(N, X, incX, Y, incY);
  }
  void scopy_int8_to_float16_s(const unsigned int N, const int8_t *X,
                               const unsigned int incX, _FP16 *Y,
                               const unsigned int incY) override {
    cpu_->scopy_int8_to_float16_s(N, X, incX, Y, incY);
  }

  // --- Mixed precision BLAS ---
  void shgemm(const unsigned int TStorageOrder, bool TransA, bool TransB,
             const unsigned int M, const unsigned int N,
             const unsigned int K, const float alpha, const float *A,
             const unsigned int lda, const _FP16 *B, const unsigned int ldb,
             const float beta, float *C, const unsigned int ldc) override {
    cpu_->shgemm(TStorageOrder, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
                beta, C, ldc);
  }
  void shgemv(const unsigned int TStorageOrder, bool TransA,
             const unsigned int M, const unsigned int N, const float alpha,
             const float *A, const unsigned int lda, const _FP16 *X,
             const unsigned int incX, const float beta, float *Y,
             const unsigned int incY) override {
    cpu_->shgemv(TStorageOrder, TransA, M, N, alpha, A, lda, X, incX, beta, Y,
                incY);
  }
  void hsgemm(const unsigned int TStorageOrder, bool TransA, bool TransB,
             const unsigned int M, const unsigned int N,
             const unsigned int K, const float alpha, const _FP16 *A,
             const unsigned int lda, const float *B, const unsigned int ldb,
             const float beta, float *C, const unsigned int ldc) override {
    cpu_->hsgemm(TStorageOrder, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
                beta, C, ldc);
  }
  void hsgemv(const unsigned int TStorageOrder, bool TransA,
             const unsigned int M, const unsigned int N, const float alpha,
             const _FP16 *A, const unsigned int lda, const float *X,
             const unsigned int incX, const float beta, float *Y,
             const unsigned int incY) override {
    cpu_->hsgemv(TStorageOrder, TransA, M, N, alpha, A, lda, X, incX, beta, Y,
                incY);
  }

  // --- Quantized GEMM (FP16 variants) ---
  void gemm_q4_0_fp16(const unsigned int M, const unsigned int N,
                      const unsigned int K, const _FP16 *A,
                      const unsigned int lda, const void *B,
                      const unsigned int ldb, _FP16 *C,
                      const unsigned int ldc) override {
    cpu_->gemm_q4_0_fp16(M, N, K, A, lda, B, ldb, C, ldc);
  }
  void gemm_q6_K_fp16(const unsigned int M, const unsigned int N,
                      const unsigned int K, const _FP16 *A,
                      const unsigned int lda, const void *B,
                      const unsigned int ldb, _FP16 *C,
                      const unsigned int ldc) override {
    cpu_->gemm_q6_K_fp16(M, N, K, A, lda, B, ldb, C, ldc);
  }

  // --- Rotary embedding ---
  void compute_rotary_embedding_value(unsigned int dim, unsigned int half_,
                                      unsigned int w, _FP16 *in, _FP16 *out,
                                      float *cos_, float *sin_) override {
    cpu_->compute_rotary_embedding_value(dim, half_, w, in, out, cos_, sin_);
  }
#endif // ENABLE_FP16

private:
  ComputeOps *cpu_ = get_cpu_ops();

  // Weights already converted and uploaded to the DSP, keyed on the ARM-layout
  // weight pointer. nntrainer allocates its weight pool once with
  // MAX_LIFESPAN, so these addresses are stable for the process lifetime.
  // Mirrors the bridge's own map (which is the real source of truth) purely so
  // we can skip the unpack/repack work when it would be discarded anyway.
  std::unordered_set<const void *> uploaded_;

  // Reused across weights; sized to the largest seen so far.
  std::vector<uint8_t> unpack_scratch_;
  std::vector<uint8_t> htp_scratch_;
};

ComputeOps *get_hexagon_ops() {
  static HexagonComputeOps instance;
  return &instance;
}

} // namespace nntrainer
