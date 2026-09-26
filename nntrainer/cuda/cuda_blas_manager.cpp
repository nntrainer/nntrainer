// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_blas_manager.cpp
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   cuBLAS handle management implementation.
 */

#include "cuda_blas_manager.h"
#include "cuda_context_manager.h"
#include "cuda_stream_manager.h"

#include <nntrainer_log.h>

#include <cstdlib>
#include <cuda_runtime.h>

namespace nntrainer::cuda {

namespace {
// Contiguous repack scratch for the int8 GEMM K-chunking (igemmRowMajor).
signed char *g_i8_bchunk = nullptr;
size_t g_i8_bchunk_cap = 0;
} // namespace

void BlasManager::initialize() noexcept {
  ContextManager::Global().EnsureCurrent();
  if (cublasCreate(&handle_) != CUBLAS_STATUS_SUCCESS) {
    ml_loge("[CUDA] cublasCreate failed");
    handle_ = nullptr;
    return;
  }
  // bind to the backend stream so GEMMs order with the dequant / copy kernels.
  cublasSetStream(handle_, StreamManager::Global().GetStream());
  // Explicit cuBLAS workspace. The legacy cublasGemmEx int8 IMMA picks an
  // algorithm that needs scratch; with no workspace set, on Orin/sm_87 the
  // large-K down-proj (m=1536,n=992,k=6144) does an illegal memory access
  // (RTX/sm_89 happens to pick a no-workspace algo, so it never showed there).
  // A persistent 64MB workspace lets the workspace-using algo run. Tunable via
  // NNTR_CUBLAS_WS_MB.
  {
    size_t ws_mb = 64;
    if (const char *e = std::getenv("NNTR_CUBLAS_WS_MB"))
      ws_mb = (size_t)atoi(e);
    static void *ws = nullptr;
    static size_t ws_bytes = 0;
    const size_t want = ws_mb * 1024ull * 1024ull;
    if (want > 0 && ws_bytes < want) {
      if (ws)
        cudaFree(ws);
      ws = nullptr;
      ws_bytes = 0;
      if (cudaMalloc(&ws, want) == cudaSuccess)
        ws_bytes = want;
    }
    if (ws && ws_bytes > 0)
      cublasSetWorkspace(handle_, ws, ws_bytes);
  }
  ok_ = true;
}

bool BlasManager::sgemmRowMajor(int M, int N, int K, const float *X,
                                const float *W, float *Y) {
  if (!ok_)
    return false;
  // NNTR_DETERMINISTIC / NNTR_CUBLAS_PEDANTIC: pin the FP32 path's math
  // mode. Default cuBLAS math on Ampere+ may pick TF32 tensor-core kernels
  // whose split-K epilogues carry documented run-to-run ulp drift — the
  // int8 IMMA path is int32-exact and unaffected. One-time
  // per handle; no effect on quantized models (this path needs FP32 weights).
  {
    static const bool pedantic = []() {
      const char *d = std::getenv("NNTR_DETERMINISTIC");
      const char *p = std::getenv("NNTR_CUBLAS_PEDANTIC");
      return (d && d[0] == '1') || (p && p[0] == '1');
    }();
    static bool pinned = false;
    if (pedantic && !pinned) {
      cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH);
      pinned = true;
    }
  }
  const float alpha = 1.0f;
  const float beta = 0.0f;
  // Column-major C[N,M] = W_view[N,K] * X_view[K,M] = (X*W)^T, read back
  // row-major as Y[M,N] = X*W. (orientation validated vs CPU reference)
  cublasStatus_t s = cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                 &alpha, W, N, X, K, &beta, Y, N);
  if (s != CUBLAS_STATUS_SUCCESS) {
    ml_loge("[CUDA] cublasSgemm failed: %d", (int)s);
    return false;
  }
  return true;
}

bool BlasManager::igemmRowMajor(int M, int N, int K, const signed char *A,
                                const signed char *B, int *C) {
  static int dbg = -2;
  if (dbg == -2)
    dbg = (std::getenv("NNTR_IGEMM_DBG") != nullptr) ? 1 : 0;
  if (!ok_) {
    if (dbg)
      fprintf(stderr, "[IGEMM] ok_=false (BlasManager not initialized)\n");
    return false;
  }
  const int alpha = 1;
  // Mirror sgemmRowMajor's orientation: column-major C[N,M] = B_view[N,K] *
  // A_view[K,M] reads back row-major as C[M,N] = A*B. B is int8 weight [K,N]
  // (ld=N), A is int8 act [M,K] (ld=K). int8 in / int32 accumulate -> IMMA
  // Tensor Cores on sm_75+ (N,K are multiples of 32 for the e2b dims).
  //
  // K-chunking: the legacy cublasGemmEx int8 IMMA does an ILLEGAL MEMORY ACCESS
  // for large K on Orin/sm_87 (measured: k=6144 FFN down-proj faults; k<=2048
  // q/k/v/o/gate/up are fine; the identical call works on RTX/sm_89 -- it's an
  // sm_87 large-K algo-selection bug, NOT a caller error). Split K into
  // <=kchunk slices and accumulate in int32 (beta=1) so every cuBLAS call stays
  // in the working regime. K<=kchunk is a single call (no overhead for the
  // common FCs). A is act [K,M] col-major ld=K -> k-slice = A + k0 (rows). B is
  // weight [N,K] col-major ld=N -> k-slice = B + k0*N (cols). Tunable:
  // NNTR_CUBLAS_KCHUNK.
  static const int kchunk = []() {
    const char *e = std::getenv("NNTR_CUBLAS_KCHUNK");
    if (e) {
      int v = atoi(e);
      return v > 0 ? v : 2048;
    }
    // The K-chunk only works around an sm_87 (integrated Orin) large-K int8
    // IMMA algo-selection bug. Discrete GPUs (RTX) run the full-K call
    // correctly and a few % faster (no per-chunk repack + extra accumulating
    // GEMMs), so don't chunk there: a huge kchunk makes K>kchunk false -> one
    // full-K call.
    return ContextManager::Global().isIntegrated() ? 2048 : (1 << 28);
  }();
  // GEMM algo probe (NNTR_CUBLAS_ALGO): the int8 GEMM is the prefill critical
  // path at only ~20 TOPS. cublasGemmEx defaults to CUBLAS_GEMM_DEFAULT(-1);
  // sweep ALGO0..23 / DEFAULT_TENSOR_OP(99) / ALGO0..15_TENSOR_OP(100..115) to
  // see if a better IMMA kernel exists for these shapes. -1 = library default.
  static const cublasGemmAlgo_t igemm_algo = []() {
    const char *e = std::getenv("NNTR_CUBLAS_ALGO");
    return e ? (cublasGemmAlgo_t)atoi(e) : CUBLAS_GEMM_DEFAULT;
  }();
  const bool chunked = K > kchunk;
  // When chunking, cublas's B operand = the activation A viewed [K,M] col-major
  // ld=K -> a k-row-slice is STRIDED (ld=K>kc), which int8 IMMA rejects
  // (illegal access). Repack each slice into a CONTIGUOUS [kc,M] (ld=kc) buffer
  // via a strided D2D cudaMemcpy2D (no custom kernel) so every chunk runs in
  // the proven contiguous k<=kchunk regime. A is [M,K] row-major: row m, cols
  // [k0,k0+kc) -> dst pitch kc, src pitch K.
  cudaStream_t stream = StreamManager::Global().GetStream();
  for (int k0 = 0; k0 < K; k0 += kchunk) {
    const int kc = (K - k0 < kchunk) ? (K - k0) : kchunk;
    const int beta_c = (k0 == 0) ? 0 : 1;
    const signed char *actB = A + (size_t)k0;
    int ldB = K;
    if (chunked) {
      // +256B tail pad: int8 IMMA reads the operand in wide vectorized tiles
      // that can run past the last element; an exactly-sized contiguous slice
      // then faults. (kc is a multiple of 32 for the e2b dims, so no per-chunk
      // dimension padding is needed; the tail pad covers the vectorized read.)
      const size_t need = (size_t)kc * M + 256;
      if (g_i8_bchunk_cap < need) {
        if (g_i8_bchunk)
          cudaFree(g_i8_bchunk);
        g_i8_bchunk = nullptr;
        g_i8_bchunk_cap = 0;
        if (cudaMalloc((void **)&g_i8_bchunk, need) != cudaSuccess)
          return false;
        g_i8_bchunk_cap = need;
      }
      if (cudaMemcpy2DAsync(g_i8_bchunk, (size_t)kc, A + (size_t)k0, (size_t)K,
                            (size_t)kc, (size_t)M, cudaMemcpyDeviceToDevice,
                            stream) != cudaSuccess)
        return false;
      actB = g_i8_bchunk;
      ldB = kc;
    }
    cublasStatus_t s =
      cublasGemmEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N, N, M, kc, &alpha,
                   B + (size_t)k0 * N, CUDA_R_8I, N, actB, CUDA_R_8I, ldB,
                   &beta_c, C, CUDA_R_32I, N, CUBLAS_COMPUTE_32I, igemm_algo);
    if (s != CUBLAS_STATUS_SUCCESS) {
      if (dbg)
        fprintf(stderr,
                "[IGEMM] cublasGemmEx int8 status=%d (M=%d N=%d K=%d k0=%d "
                "kc=%d)\n",
                (int)s, M, N, K, k0, kc);
      ml_loge("[CUDA] cublasGemmEx int8 failed: %d", (int)s);
      return false;
    }
  }
  if (dbg) {
    static int once = 0;
    if (once++ < 3)
      fprintf(stderr, "[IGEMM] OK M=%d N=%d K=%d\n", M, N, K);
  }
  return true;
}

BlasManager::~BlasManager() {
  if (handle_)
    cublasDestroy(handle_);
}

BlasManager &BlasManager::Global() {
  // Out-of-line + intentionally leaked (see header note): never destroyed,
  // so ~BlasManager (cublasDestroy) never runs at process exit (2026-07-20
  // field crash fix, same convention as ClContext).
  static BlasManager *instance = new BlasManager();
  instance->initializeOnce();
  return *instance;
}

} // namespace nntrainer::cuda
