// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file	blas_kernel_interface.h
 * @date	5 June 2024
 * @brief	Interface for blas OpenCL kernels
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __BLAS_KERNEL_INTERFACE_H__
#define __BLAS_KERNEL_INTERFACE_H__

#include <string>
#include <tensor.h>

namespace nntrainer {

class ClContext;

/**
 * @brief Process data and dimensions for OpenCL dot operation
 * @param[in] input Tensor
 * @param[in] m Tensor
 * @param[in] RunLayerContext reference
 * @param[in] trans bool
 * @param[in] trans_m bool
 */
Tensor dotCl(Tensor const &input, Tensor const &m, bool trans = false,
             bool trans_m = false);

/**
 * @brief Process data and dimensions for OpenCL dot operation
 * @param[in] input Tensor
 * @param[in] m Tensor
 * @param[in] result Tensor
 * @param[in] RunLayerContext reference
 * @param[in] trans bool
 * @param[in] trans_m bool
 */
void dotCl(Tensor const &input, Tensor const &m, Tensor &result,
           bool trans = false, bool trans_m = false);

/**
 * @brief Process data and dimensions for OpenCL dot operation
 * @param[in] input Tensor
 * @param[in] m Tensor
 * @param[in] result Tensor
 * @param[in] RunLayerContext reference
 * @param[in] trans bool
 * @param[in] trans_m bool
 */
void dotBatchedCl(Tensor const &input, Tensor const &m, Tensor &result,
                  bool trans = false, bool trans_m = false);

/**
 * @brief Multiply value element by element immediately
 * @param[in] input Tensor
 * @param[in] value multiplier
 * @param[in] RunLayerContext reference
 */
void multiplyCl(Tensor &input, float const &value);

/**
 * @brief Process data and dimensions for add operation
 * @param[in] result Tensor
 * @param[in] input Tensor
 */
void add_i_cl(Tensor &result, Tensor const &input);

/**
 * @brief FP16 elementwise residual copy (dst = src) or accumulate (dst += src)
 *        where dst/src each bind the plane their STATIC ResidencyClass picked
 *        at allocation: the planner cl_mem sub-buffer for GPU_CLMEM, the SVM
 *        pointer otherwise (mixed args valid). Returns false only when NEITHER
 *        side is cl_mem (caller keeps its SVM/host path); after the
 *        static-class commitment a failure throws (a silent SVM fallback would
 *        recreate the corrupting hybrid). No clFinish -- the in-order SVM-pool
 *        queue provides the ordering (gpu_native's coherence model).
 * @param[in,out] dst destination tensor (residual accumulator)
 * @param[in] src source tensor
 * @param[in] accumulate false: dst = src; true: dst += src
 */
bool clmem_residual_op_cl(Tensor &dst, const Tensor &src, bool accumulate);

/**
 * @brief Non-invasive value probe for the cl_mem residency bring-up
 *        (NNTR_CLMEM_PROBE=1). During the forward it only enqueues
 *        DEVICE-SIDE copies of the probed buffer (cl_mem source ->
 *        clEnqueueCopyBuffer, SVM source -> copy kernel) into dedicated debug
 *        buffers -- NO host sync, no mid-pipeline readback (a blocking
 *        readback corrupts the run on Adreno). When NNTR_CLMEM_PROBE_MAX
 *        captures accumulate, ONE clFinish drains the queue and every entry
 *        is read back and dumped to stderr as [probe] tag fnv=<hash> v0..v3,
 *        in capture (= execution) order; diff two runs to find the first
 *        divergent op.
 * @param tag entry label (e.g. "layer3pre_ffn_norm:out")
 * @param svm_ptr SVM/host pointer source (used when clmem == nullptr)
 * @param clmem cl_mem source handle (takes precedence)
 * @param bytes bytes to capture
 */
void clmem_probe_capture(const char *tag, const void *svm_ptr, void *clmem,
                         unsigned int bytes);

/**
 * @brief Explicit host->cl_mem RAISE for a boundary tensor (design §2.5 input
 *        boundary): a HOST producer (the embedding dequant loop) wrote the
 *        tensor's SVM shadow; upload the valid bytes into its planner cl_mem
 *        sub-buffer so GPU_CLMEM consumers read fresh device data instead of
 *        a coarse-SVM handoff (the measured visibility hazard). Non-blocking
 *        write on the in-order queue (ordered before all later consumers);
 *        the SVM source stays stable until the next forward (the lm_head
 *        blocking read drains first). No-op (returns false) when the tensor
 *        is not GPU_CLMEM-resident.
 * @param t boundary tensor (host-written, GPU_CLMEM class)
 * @param valid_bytes bytes to upload from the tensor base (0 = full tensor)
 */
bool clmem_raise_cl(const Tensor &t, unsigned int valid_bytes);

/**
 * @brief Explicit cl_mem->host LOWER for a boundary tensor (design §2.5
 *        output boundary): blocking clEnqueueReadBuffer from the tensor's
 *        planner cl_mem sub-buffer into its SVM shadow, so a HOST consumer
 *        (the lm_head dequant+dot) reads fresh data through ordinary host
 *        pointers. This replaces the coarse-SVM map protocol at the one
 *        genuine GPU->host boundary -- device kernels writing/reading
 *        host-mapped coarse SVM intermittently see ZEROS on this driver
 *        (measured), so the boundary must be an explicit copy. No-op
 *        (returns false) when the tensor is not GPU_CLMEM-resident.
 * @param t boundary tensor (GPU_CLMEM class, host-consumed)
 * @param valid_bytes bytes to read back from the tensor base (0 = full)
 */
bool clmem_lower_cl(const Tensor &t, unsigned int valid_bytes);

/**
 * @brief Process data and dimensions for transpose operation
 * @param[in] direction string
 * @param[in] input Tensor
 * @param[in] result Tensor
 */
void transposeCl(const std::string &direction, Tensor const &in,
                 Tensor &result);

/**
 * @brief Copy data from one tensor to another
 *
 * @param input Tensor
 * @param result Tensor
 */
void copyCl(const Tensor &input, Tensor &result);

/**
 * @brief nrm2 computation : Euclidean norm
 * @param input Tensor
 * @return Euclidean norm
 * @note This function is used to compute the Euclidean norm of a vector.
 */
float nrm2Cl(const Tensor &input);

/**
 * @brief Absolute sum computation
 *
 * @param input Tensor
 * @return float absolute sum of the elements
 */
float asumCl(const Tensor &input);

/**
 * @brief Absolute max computation
 *
 * @param input Tensor
 * @return int index of the maximum absolute value
 * @note Not necessarily the first if there are multiple maximums.
 */
int amaxCl(const Tensor &input);

/**
 * @brief Absolute min computation
 *
 * @param input Tensor
 * @return int index of the minimum absolute value
 * @note Not necessarily the first if there are multiple minimums.
 */
int aminCl(const Tensor &input);

/**
 * @brief v8c GPU path entry point — paper 8/4/4 (arXiv:2505.00232): int8
 *        activation × channel-wise QINT4 weight GEMM. Default-on for the GPU
 *        FC dispatch; NNTR_FC_INT8_GPU=0 disables. Caller falls back to the
 *        generic host path on false.
 * @param[in] input fp32 or fp16 activation tensor [M, K]
 * @param[in] weight channel-wise QINT4 (QS4CX) weight tensor [K, N]
 * @param[out] output fp32 or fp16 tensor [M, N] (preallocated)
 * @return true if the v8c path executed; false if not applicable
 *         (env disabled, weight not int4, shape misaligned).
 */
bool dotCl_v8c(const Tensor &input, const Tensor &weight, Tensor &output);

/**
 * @brief Why the last dotCl_v8c call on THIS thread returned false, as a static
 *        string ("none" if it never has). Set at every reject site; read by the
 *        FC divert tripwire (ClComputeOps::fc, NNTR_FC_DIVERT_TRACE=1) so a
 *        host bounce names its own cause. Only meaningful immediately after a
 *        false return.
 */
const char *v8c_last_reject_reason();

/**
 * @brief Eagerly build the v8c GPU weight entry (nibble permute + upload +
 *        image view) for a freshly READ int4 FC weight, so the first prefill
 *        does not pay the lazy per-weight build. Called by the CL FC layer
 *        after the base read. Returns false (no-op) off the v8c path (env
 *        unset / non-int4 / unsupported shape); the lazy build in dotCl_v8c
 *        still covers those.
 */
bool dotCl_v8c_prebuild_weight(const Tensor &weight);

/**
 * @brief Pre-build the v8c output-residency kernel PROGRAM (the file-local
 *        v8c_out_residency_kernels source hosting v8c_copy_h2h / v8c_add_h2h /
 *        v8c_cvt_h2f / v8c_copy_f2f) on the given context. Called from
 *        ClContext::initAttentionClKernels so its first-use program build
 *        (clprof: the rmsnorm->v8c_copy_h2h / gemm->v8c_copy_h2h one-time
 *        idle outliers, ~25ms each) lands at model load, not inside the
 *        first timed prefill. One kernel suffices: the program cache makes
 *        the sibling kernels of the same source free.
 */
void v8c_prewarm_programs(ClContext &cc);

#ifdef ENABLE_FP16
/**
 * @brief Segment A: GPU RMSNorm with TensorBacking output residency.
 *
 *        Paper §3.2 cross-layer residency: the output cl_mem is owned by
 *        the process-global TensorBackingPool keyed by `output_name` and
 *        also assigned to `output.setBacking()`. Host data of `output` is
 *        left untouched — downstream consumers MUST read via
 *        `getBacking()` (or pool lookup by name).
 *
 *        If `input.getBacking()` exists with FP16 encoding, the backing
 *        cl_mem is used directly (zero host transfer). Otherwise the
 *        input is uploaded from host (one transfer, same as today).
 *        Gamma is uploaded once per (gamma name) and cached.
 *
 *        Env-gated via NNTR_RESIDENT_RMSNORM=1. Returns false if env not
 *        set or any precondition fails; caller falls back to CPU path.
 *
 * @param[in]  input  FP16 activation [B, C, H, W]
 * @param[in]  gamma  FP16 per-channel scale [W]
 * @param[in]  epsilon  RMS epsilon
 * @param[in]  B, C, H, W  shape constants matching `input`
 * @param[in]  output_name  stable Tensor name (used as pool key)
 * @param[out] output Tensor; setBacking() is called on success
 * @return true if the GPU path ran; false otherwise
 */
bool rmsnorm_resident_fp16(const Tensor &input, const Tensor &gamma,
                           float epsilon, unsigned int B, unsigned int C,
                           unsigned int H, unsigned int W,
                           const std::string &output_name, Tensor &output);

/**
 * @brief FP32 variant of rmsnorm_resident. Same contract as the FP16
 *        version but uses rmsnorm_cl (subgroup-reduced kernel) for the
 *        FP32 residual stream Qwen3 currently uses. Encoding of the
 *        resulting TensorBacking is Encoding::FP32.
 */
bool rmsnorm_resident_fp32(const Tensor &input, const Tensor &gamma,
                           float epsilon, unsigned int H, unsigned int W,
                           const std::string &output_name, Tensor &output);

#endif // ENABLE_FP16

/**
 * @brief Publish an already-computed FP32 host buffer to a GPU
 *        TensorBacking under `output_name`. Used by the CPU-norm
 *        + GPU-residency-handoff path: CPU RMSNorm writes to the
 *        output Tensor's host data, then this helper uploads that
 *        host data into the backing's cl_mem and registers it in
 *        the pool. Downstream FC layers with NNTR_RESIDENT_FC=1
 *        consume the backing directly. Bit-exact w.r.t. CPU output
 *        because no GPU computation happens here.
 * @return true if the backing was created/updated and registered.
 */
bool publish_host_fp32_to_backing(const Tensor &output,
                                  const std::string &output_name);

/**
 * @brief Publish a GPU-resident activation (cl_mem residency overlay, Step 1).
 *        GPU-copies the producer's SVM output (FP16/FP32, n_elems) into a
 * cl_mem TensorBacking keyed `resact:`+name (the producer's graph-output name);
 *        a downstream CL layer that resolved this edge via resolveResidentEdge
 *        consumes the cl_mem instead of the SVM buffer. No host bounce.
 * @return true on success; false ⇒ caller keeps the plain SVM output path.
 */
bool publish_resident_act(const std::string &name, const void *svm_ptr,
                          unsigned int n_elems, bool fp16);

/**
 * @brief Create/reuse the `resact:`+name cl_mem backing (no data written) and
 *        return its cl_mem (as void* so this header stays free of CL types), so
 *        a producer can bind it as its kernel output and write the activation
 *        device-resident directly (no SVM intermediate).
 * @return cl_mem backing buffer (as void*), or nullptr on failure.
 */
void *get_or_create_resident_backing(const std::string &name,
                                     unsigned int n_elems, bool fp16);

/**
 * @brief Read the contents of a tensor's GPU TensorBacking back into the
 *        tensor's host buffer. Used by the chain-robustification rmsnorm
 *        path to keep host and GPU views in sync after a GPU kernel
 *        writes to the backing. Blocks on clFinish + clEnqueueReadBuffer.
 * @param[in,out] t Tensor whose host buffer is overwritten with the
 *                  contents of t.getBacking()'s cl_mem. Number of bytes
 *                  read = t.bytes(). Caller is responsible for sizing.
 * @return true if backing existed and the read completed; false otherwise.
 */
bool readback_backing_to_host(Tensor &t);

/**
 * @brief Fused RMSNorm + v8c activation quantization in a single GPU
 *        dispatch (paper §3.6 fused-kernel idea, smallest unit that
 *        eliminates the RMSNorm-output → FC-quant-input drift boundary
 *        documented in `chain-robustification-dead`).
 *
 *        Math is byte-identical to (CPU RMSNorm(input, gamma, eps)
 *        followed by v8c_act_quant_f32) — same KAI qai8dxp asymmetric
 *        formula on normalized = x * inv_rms * gamma. The point of the
 *        fusion is NOT a perf win on its own (kernel is bounded by the
 *        same global-memory bandwidth as the unfused path); it is that
 *        the intermediate normalized fp32 values never touch global
 *        memory, so they cannot drift from one CPU/GPU run to another.
 *
 *        Outputs land in four TensorBackings registered in the global
 *        TensorBackingPool under the names:
 *          <output_name>:fused_i8     INT8  [M*K]
 *          <output_name>:fused_scale  FP32  [M]
 *          <output_name>:fused_zp     FP32  [M] (4 bytes per entry; int32 data)
 *          <output_name>:fused_rs     FP32  [M] (4 bytes per entry; int32 data)
 *        The encoding tag on _zp / _rs is FP32 (no INT32 enum yet); only
 *        the byte count and offset matter to downstream consumers. When
 *        output_host_ptr is non-null, the same four backings are ALSO
 *        registered under ptr-keyed names ('ptr:<output_host_ptr>:fused_*')
 *        so a downstream consumer that only has the output Tensor's data
 *        pointer (not its name) can look them up.
 *
 *        Env-gated via NNTR_FUSED_RMSQ=1; without it, this is a no-op
 *        that returns false. NNTR_FUSED_RMSQ_CHECK=1 additionally runs
 *        the CPU reference path on a single probe row and prints the
 *        max bit difference + relL2 — used to validate the kernel
 *        before any callers depend on its outputs.
 *
 *        Precondition: K <= 2048 (the kernel uses a local-memory cache
 *        of normalized values that's sized at compile time).
 *
 * @param[in]  input          [M, K] fp32 pre-norm activation tensor
 * @param[in]  gamma          [K]    fp32 per-channel scale
 * @param[in]  epsilon        RMSNorm epsilon
 * @param[in]  M, K           shape
 * @param[in]  output_name    base name for the four pool entries
 * @param[in]  output_host_ptr optional output tensor data pointer; when
 *             non-null, the four pool entries are also registered under
 *             ptr-keyed names for pointer-based lookup by a consumer
 * @return true if the fused kernel ran; false if env not set or any
 *         precondition failed
 */
bool fused_rmsnorm_quant_resident_fp32(const Tensor &input, const Tensor &gamma,
                                       float epsilon, unsigned int M,
                                       unsigned int K,
                                       const std::string &output_name,
                                       const void *output_host_ptr = nullptr);

} // namespace nntrainer
#endif /* __BLAS_KERNEL_INTERFACE_H__ */
