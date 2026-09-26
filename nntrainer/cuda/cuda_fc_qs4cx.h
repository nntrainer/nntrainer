// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_fc_qs4cx.h
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Fused QS4CX (per-channel signed int4) GEMM for the CUDA FC path:
 *          Y[M,N] = X[M,K] * dequant(W), where W is the plain QS4CX payload
 *          (packed signed nibbles plus one FP32 scale per output channel).
 *          The int4 weight is read and dequantized INLINE inside the kernel --
 *          there is never a dense FP32 weight buffer -- which is what makes a
 *          real-size model fit at all. Three routes share that contract: a
 *          dequant-GEMM floor, a dp4a int8xint4 fast path, and cuBLAS int8 on
 *          the Tensor Cores for wide (prefill-shaped) GEMMs.
 */

#ifndef __CUDA_FC_QS4CX_H__
#define __CUDA_FC_QS4CX_H__

namespace nntrainer::cuda {

/**
 * @brief Smallest output width N at which a decode (M == 1) QS4CX FC takes the
 *        fp-ACTIVATION int4 GEMV instead of the w4a8 dp4a route.
 *
 * The shape this separates is the untied lm_head (N = vocab, 262144 here) from
 * every ordinary projection (N ~ 1536-12288, which keeps dp4a). It is a
 * FIDELITY threshold, not a performance one: only at vocab width does the
 * int8-activation quant's per-logit noise (sigma 0.18-0.37 measured) exceed the
 * top1-top2 argmax margin (~0.117) often enough to change which token is
 * sampled. Mirrors the OpenCL lane, which routes the same weight to its own
 * fp-activation GEMV once N passes the OpenCL image height cap that makes the
 * other backends take the same route.
 */
constexpr unsigned int CUDA_FC_FPACT_MIN_N = 65536u;

/**
 * @brief Build (and cache) the N-entry UVM fp32 per-channel scale buffer --
 *        the fp16 buffer's sibling for the fp-activation lm_head GEMV, which
 *        keeps the tensor's full-precision scale because its whole purpose is
 *        argmax fidelity. @p out_sc receives the cached device pointer.
 * @param source_readable false when @p fp32_scales may no longer be
 *        dereferenced -- the QS4CX scale tail shares the plain payload's
 *        allocation, so a caller that releases the payload releases the scale
 *        with it and the bytes read back as zeros. A cache HIT ignores the
 *        flag; a MISS with it false is refused, so the caller falls back
 *        instead of computing against a zero scale. A caller that runs while
 *        the payload is still live (the load-time prewarm) leaves it at the
 *        default.
 * @return false if the buffer neither exists nor could be built.
 */
bool cuda_fc_qs4cx_scales_to_uvm_fp32(const float *fp32_scales, unsigned int N,
                                      const float **out_sc,
                                      bool source_readable = true);

/**
 * @brief fp-ACTIVATION int4 GEMV for the huge-N decode lm_head (M == 1).
 *
 * Reads the fp16 activation DIRECTLY -- no int8 activation quant -- against the
 * same derived weight cache the dp4a path builds (signed packed int4), fp32
 * accumulate, fp16 logits out. w4a8's per-row activation quant injects a
 * per-logit error (sigma 0.18-0.37 measured on this lm_head) well above the
 * top1-top2 argmax margin (~0.117), so at vocab width it changes the sampled
 * token often enough to walk a long greedy decode into a repetition loop; this
 * route removes that noise at no extra weight traffic. Same design decision the
 * OpenCL lane already makes for this weight (lmhead_int4_v8c_gemv).
 *
 * @param scales_fp32 the tensor's per-channel fp32 scales (dequant
 *        multipliers); the UVM copy is built/cached internally.
 * @return false -- and NOTHING dispatched, so the caller must fall through to
 *         dp4a -- when NNTR_CUDA_LMHEAD_FPACT=0, when the scale buffer is
 *         unavailable, or on any registration/dispatch failure.
 */
bool cuda_fc_qs4cx_fpact_gemv_fp16(const unsigned short *Xh,
                                   const unsigned char *plain_w,
                                   const float *scales_fp32, unsigned short *Yh,
                                   unsigned int N, unsigned int K);

/**
 * @brief Y[M,N] = X[M,K] * dequant(QS4CX W) where W is the PLAIN QS4CX payload
 *        (weight.getData(): row-major [N][(K+1)/2] nibble bytes, even k = low
 *        nibble, stored uint4 = int4+8) -- the same blob the OpenCL v8c kernel
 *        consumes, decoded inline; no dense FP32 weight buffer and no side
 *        copy of the nibbles is materialised. Per-output-channel fp16 scale
 *        (one per N), converted to fp32 in-kernel.
 *
 * @param X            [M,K] row-major FP32 activation (device-accessible)
 * @param plain_w      plain QS4CX nibble payload = weight.getData()
 * (device-acc)
 * @param scales_fp16  N fp16 per-channel scales (device-acc; see
 *                     cuda_fc_qs4cx_scales_to_uvm_fp16)
 * @param Y            [M,N] row-major FP32 output (device-accessible)
 * @param M,N,K        GEMM dims
 * @return true on success
 */
bool cuda_fc_qs4cx_gemm_fp32(const float *X, const unsigned char *plain_w,
                             const unsigned short *scales_fp16, float *Y,
                             unsigned int M, unsigned int N, unsigned int K);

/**
 * @brief Same as cuda_fc_qs4cx_gemm_fp32 but for HOST-resident inputs.
 *        This wrapper mirrors the plain weight into device memory ONCE (cached
 *        by the host weight pointer -- weights are constant) and stages the
 *        activation in / output out per call, then runs the device kernel. It
 *        is the CUDA analogue of the OpenCL cl_mem residency bridge; a future
 *        UVM-resident tensor pool would let the zero-copy path above be taken
 *        instead.
 *
 * @param host_X        [M,K] FP32 activation on the host heap
 * @param host_plain    plain QS4CX nibble payload on the host heap (cache key)
 * @param host_scales   N fp16 per-channel scales on the host heap
 * @param host_Y        [M,N] FP32 output on the host heap (written back)
 */
bool cuda_fc_qs4cx_gemm_fp32_resident(const float *host_X,
                                      const unsigned char *host_plain,
                                      const unsigned short *host_scales,
                                      float *host_Y, unsigned int M,
                                      unsigned int N, unsigned int K);

/**
 * @brief w4a8 dp4a fast path: Y[M,N] = X[M,K] * dequant(QS4CX W) using Ada
 *        __dp4a (4-way int8 dot-accumulate). The FP32 activation is quantized
 *        to int8 per-row; the plain QS4CX weight is repacked ONCE into signed
 *        packed int4 [N,(K+1)/2] (a byte-wise XOR 0x88 -- cached by the plain
 *        pointer, normally CPU-prewarmed at load); the GEMM accumulates int32
 *        via __dp4a and rescales by act_scale[m]*w_scale[n]. Much higher
 *        arithmetic throughput than the naive FP32 kernel, and no FP32 weight
 *        blow-up (stays int4 in memory). All pointers must be
 *        device-accessible (UVM). Requires N%4==0, K%32==0 (load invariant).
 *
 * @param X            [M,K] row-major FP32 activation (device-accessible)
 * @param plain_w      plain QS4CX nibble payload = weight.getData()
 * (device-acc)
 * @param scales_fp16  N fp16 per-channel scales (device-accessible)
 * @param Y            [M,N] row-major FP32 output (device-accessible)
 * @param M,N,K        GEMM dims
 * @return true on success
 */
bool cuda_fc_qs4cx_dp4a_gemm_fp32(const float *X, const unsigned char *plain_w,
                                  const unsigned short *scales_fp16, float *Y,
                                  unsigned int M, unsigned int N,
                                  unsigned int K);

/**
 * @brief FP16-activation variant of the dp4a fast path (the real CausalLM
 *        models use QS4CX-FP16: int4 weight + fp16 activations). The fp16
 *        activation is read directly into the int8 quantizer and the fp16
 *        output is produced by a final float->half conversion.
 *
 * @param Xh           [M,K] row-major fp16 activation (device-accessible)
 * @param plain_w      plain QS4CX nibble payload (device-accessible)
 * @param scales_fp16  N fp16 per-channel scales (device-accessible)
 * @param Yh           [M,N] row-major fp16 output (device-accessible)
 */
bool cuda_fc_qs4cx_dp4a_gemm_fp16(const unsigned short *Xh,
                                  const unsigned char *plain_w,
                                  const unsigned short *scales_fp16,
                                  unsigned short *Yh, unsigned int M,
                                  unsigned int N, unsigned int K);

/** @brief NNTR_CUDA_FUSED_NORMQ (default on, =0 opts out): whether the decode
 *  RMSNorm may fold in the int8 activation quant of the FC group it feeds. */
bool cuda_fc_qs4cx_fused_normq_enabled();

/**
 * @brief RMSNorm fused with the int8 activation quant its consumer FC needs.
 *
 * Writes the normed fp16 rows to @p y exactly as cuda_rmsnorm_fp16 would, and
 * in the same launch stages the per-row asymmetric int8 quant of those rows in
 * the dp4a activation scratch. The next FC on @p y then runs its GEMM without
 * a quant launch of its own -- and so do its siblings (q/k/v share one norm,
 * gate/up share another), which is where the decode launch count comes down.
 * The staging is published under a pointer + width + stream-dispatch-sequence
 * stamp, so an unrelated kernel writing a recycled buffer at the same address
 * cannot be mistaken for it.
 *
 * Bit-identical to the split rmsnorm_fp16 + act_quant_i8_h pair (identical
 * reduction order and identical rounding), so it needs no numerical waiver.
 *
 * @param x     [rows, width] fp16 input (device-accessible)
 * @param gamma [width] fp16 per-feature scale, or nullptr
 * @param y     [rows, width] fp16 output (device-accessible)
 * @param eps   epsilon added to the mean of squares
 * @param rows  row count (decode: 1)
 * @param width feature size (== K of the consuming FC)
 * @return false if the lever is off or the staging could not be prepared --
 *         the caller must then run the plain norm (nothing was published).
 */
bool cuda_fc_qs4cx_rmsnorm_prequant_fp16(const unsigned short *x,
                                         const unsigned short *gamma,
                                         unsigned short *y, float eps,
                                         unsigned int rows, unsigned int width);

/**
 * @brief w4a8 on the INT8 Tensor Cores via cuBLAS (prefill FC). Same quant
 *        scheme as the dp4a path (per-row asym int8 activation + symmetric int4
 *        weight, fp16 io), but the int8xint8->int32 matmul runs on IMMA Tensor
 *        Cores (cublasGemmEx CUDA_R_8I) instead of __dp4a on the int ALU --
 *        ~10x the GEMM throughput at prefill M. The int4 weight is unpacked to
 *        int8 [K,N] ONCE (cached by the plain pointer) so the unpack stays off
 *        the per-call path. Bit-identical to dp4a (exact int32 accumulate).
 *        Best for large M (prefill); decode M=1 stays on the dp4a GEMV
 *        (BW-bound, Tensor Cores do not help).
 */
bool cuda_fc_qs4cx_cublas_i8_gemm_fp16(const unsigned short *Xh,
                                       const unsigned char *plain_w,
                                       const unsigned short *scales_fp16,
                                       unsigned short *Yh, unsigned int M,
                                       unsigned int N, unsigned int K);

/**
 * @brief Prewarm the dp4a packed-int4 weight cache at LOAD on the CPU
 *        (nntrainer ThreadManager-parallel), so the first inference skips the
 *        one-time plain -> packed int4 repack and the GPU never runs it.
 *        Mirrors repack_plain_i4 + weight_rowsum bit-exactly and uploads to
 *        the device cache keyed by the plain pointer. Also prewarms the cuBLAS
 *        int8 [K,N] cache when NNTR_FC_CUDA_CUBLAS is on. Idempotent.
 *
 * @param plain_w plain QS4CX nibble payload = weight.getData() (host-readable)
 * @param N,K     FC weight dims (N output channels, K input)
 */
bool cuda_fc_qs4cx_prewarm(const unsigned char *plain_w, unsigned int N,
                           unsigned int K);

/**
 * @brief True when the dp4a derived cache exists for this
 *        plain pointer -- dispatch may then treat the pointer as a pure key
 *        (no device access, no staging needed).
 */
bool cuda_fc_qs4cx_has_cache(const unsigned char *plain_w);

/**
 * @brief Pre-grow ALL the static dp4a decode scratch buffers (g_dp4a_q8 /
 *        ascale / azp / xf / yf) to the model's max decode capacity at LOAD, so
 *        the M=1 dp4a decode FC path never cudaMallocs inside a CUDA-graph
 *        capture (NNTR_CUDA_GRAPH). A cudaMalloc/Free between
 *        cudaStreamBeginCapture..EndCapture invalidates the capture
 *        ("NvMapMemAllocInternalTagged failed: error 12"); warming here (before
 *        any capture) makes every captured ensure_buf a pure cap-hit. The
 *        in-path isCapturing() guard is the safety net if a model exceeds these
 *        bounds. Idempotent (cap check).
 *
 * @param maxM max decode token rows (1 for decode; larger is a harmless grow)
 * @param maxK max FC input dim (hidden size; covers every decode FC's K)
 * @param maxN max FC output dim (max(vocab, intermediate); covers lm_head +
 * FFN)
 */
bool cuda_fc_qs4cx_dp4a_prewarm(unsigned int maxM, unsigned int maxK,
                                unsigned int maxN);

/**
 * @brief Stage a HOST-resident [M,K] fp16 activation into a device buffer for
 *        the fp16 GPU qs4cx path. When the FC input pointer is host memory (the
 *        weight/output are still device-resident), the fp16 dp4a/cublas kernels
 *        cannot read it directly; this copies it H2D (async, on the backend
 *        stream so it is ordered before the kernels) into a reusable staging
 *        buffer and returns the device pointer. Returns nullptr if the buffer
 *        can't be obtained (OOM, or a graph capture before the buffer was
 *        prewarmed) so the caller falls back to the host path. Pre-grown by
 *        cuda_fc_qs4cx_dp4a_prewarm so the copy is a pure cap-hit under
 * capture.
 *
 * @param host_Xh [M,K] row-major fp16 activation on the host heap
 * @param M,K     activation dims
 * @return device pointer to the staged fp16 X, or nullptr on failure
 */
const unsigned short *
cuda_fc_qs4cx_stage_host_x_fp16(const unsigned short *host_Xh, unsigned int M,
                                unsigned int K);

/**
 * @brief Stage a HOST-resident QS4CX weight (plain payload + fp16 scales) to
 *        cached device buffers. A model-load race can leave a weight
 *        unregistered on the host heap; the dp4a repack reads it on the GPU,
 *        so without this the FC falls to the i8mm host dot (SIGILL on Orin).
 *        Cached by the host plain pointer (uploaded once, on a non-capture
 *        forward).
 * @param host_plain  host plain QS4CX nibble payload
 * @param host_scales host fp16 scales [N]
 * @param N,K         output / contraction dims
 * @param dev_w       out: device plain-payload pointer
 * @param dev_scales  out: device scales pointer
 * @return true if device pointers are available, false to fall back
 */
bool cuda_fc_qs4cx_stage_host_weight(const unsigned char *host_plain,
                                     const unsigned short *host_scales,
                                     unsigned int N, unsigned int K,
                                     const unsigned char **dev_w,
                                     const unsigned short **dev_scales);

/**
 * @brief Build the cached N-entry UVM fp16 copy of a QS4CX
 *        weight's per-channel fp32 scales. The GEMM/dequant kernels read the
 *        per-channel scale on device every call and want it fp16; the tensor
 *        stores fp32. This is the ONLY per-weight side allocation the CUDA FC
 *        path makes -- the nibble payload is consumed plain, in place (the
 *        derived dp4a/cuBLAS device caches are keyed by its pointer). Call at
 *        load (bails under graph capture). Cached by the fp32-scale pointer.
 * @param fp32_scales per-channel fp32 scales (getScale<float>())
 * @param N           output-channel count
 * @param out_sc      out: UVM fp16 scales pointer (host+device readable)
 * @return true on success
 */
bool cuda_fc_qs4cx_scales_to_uvm_fp16(const float *fp32_scales, unsigned int N,
                                      const unsigned short **out_sc);

/**
 * @brief High-accuracy fp16 path: FP32-precision activation (no int8 quant),
 *        naive plain-decode GEMM. Selected by NNTR_FC_CUDA_DP4A=0 for an
 *        fp16 activation (and as an accuracy reference vs the dp4a w4a8 path).
 */
bool cuda_fc_qs4cx_gemm_fp16_naive(const unsigned short *Xh,
                                   const unsigned char *plain_w,
                                   const unsigned short *scales_fp16,
                                   unsigned short *Yh, unsigned int M,
                                   unsigned int N, unsigned int K);

/**
 * @brief Q6_K lm_head GEMV on the GPU (port of OpenCL kernel_mul_mv_q6_K_f32).
 *        Reads the FP16 hidden + (managed) Q6_K weight on the device and writes
 *        FP16 logits to the device output -- no host bounce, so the Q6_K
 *        lm_head (gemma2/qwen3) works under a device-only activation pool
 *        (NNTR_CUDA_DEV_ACT) where the host GEMV would fault.
 * @param w_q6k_dev       Q6_K weight, [vocab, hidden] row-major, device/managed
 * @param hidden_fp16_dev FP16 hidden (single row), device-resident, len=hidden
 * @param logits_fp16_dev FP16 logits out, device-resident, len=vocab
 * @return true on success; false (caller falls back to host) if hidden%256!=0
 *         or the kernel could not be dispatched.
 */
bool lmhead_gemv_q6_k_cuda(const void *w_q6k_dev,
                           const unsigned short *hidden_fp16_dev,
                           unsigned short *logits_fp16_dev, int vocab,
                           int hidden);

/**
 * @brief Exempt one QS4CX weight from the eager cuBLAS int8 [K,N] cache build.
 *        The int8 copy only ever pays off for a GEMM wide enough to reach the
 *        Tensor-Core route, so a weight the model only ever hits at M=1 should
 *        not carry one. Call at load time, before the prewarm walk; the caller
 *        is whoever knows the model's shapes.
 * @param plain_w plain QS4CX nibble payload (the cache key)
 */
void cuda_fc_qs4cx_prewarm_exempt_i8(const void *plain_w);

/**
 * @brief Release every cuBLAS int8 weight cache. The int8 copies are only
 *        needed while a wide (prefill-shaped) GEMM is running, so a caller
 *        that knows prefill is over can hand the memory back.
 */
void cuda_fc_qs4cx_free_i8_caches();

} // namespace nntrainer::cuda

#endif // __CUDA_FC_QS4CX_H__
