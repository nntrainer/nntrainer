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
#include <vector>

namespace nntrainer {

/**
 * @brief Process data and dimensions for OpenCL dot operation
 * @param[in] input Tensor
 * @param[in] m Tensor
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
 * @param[in] trans bool
 * @param[in] trans_m bool
 */
void dotBatchedCl(Tensor const &input, Tensor const &m, Tensor &result,
                  bool trans = false, bool trans_m = false);

/**
 * @brief Multiply value element by element immediately
 * @param[in] input Tensor
 * @param[in] value multiplier
 */
void multiplyCl(Tensor &input, float const &value);

/**
 * @brief Process data and dimensions for add operation
 * @param[in] result Tensor
 * @param[in] input Tensor
 */
void add_i_cl(Tensor &result, Tensor const &input);

/**
 * @brief Process data and dimensions for transpose operation
 * @param[in] direction string
 * @param[in] in Tensor
 * @param[in] result Tensor
 */
void transposeCl(const std::string &direction, Tensor const &in,
                 Tensor &result);

/**
 * @brief Copy data from one tensor to another
 *
 * @param input Tensor
 * @param result Tensor
 * @throws exception::not_supported no OpenCL kernel backs this entry point
 */
void copyCl(const Tensor &input, Tensor &result);

/**
 * @brief nrm2 computation : Euclidean norm
 * @param input Tensor
 * @return Euclidean norm
 * @note This function is used to compute the Euclidean norm of a vector.
 * @throws exception::not_supported no OpenCL kernel backs this entry point
 */
float nrm2Cl(const Tensor &input);

/**
 * @brief Absolute sum computation
 *
 * @param input Tensor
 * @return float absolute sum of the elements
 * @throws exception::not_supported no OpenCL kernel backs this entry point
 */
float asumCl(const Tensor &input);

/**
 * @brief Absolute max computation
 *
 * @param input Tensor
 * @return int index of the maximum absolute value
 * @note Not necessarily the first if there are multiple maximums.
 * @throws exception::not_supported no OpenCL kernel backs this entry point
 */
int amaxCl(const Tensor &input);

/**
 * @brief Absolute min computation
 *
 * @param input Tensor
 * @return int index of the minimum absolute value
 * @note Not necessarily the first if there are multiple minimums.
 * @throws exception::not_supported no OpenCL kernel backs this entry point
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
 * @brief Eagerly build the v8c GPU weight entry (nibble permute + upload +
 *        image view) for a freshly READ int4 FC weight, so the first prefill
 *        does not pay the lazy per-weight build. Called by the CL FC layer
 *        after the base read. Returns false (no-op) off the v8c path (env
 *        unset / non-int4 / unsupported shape); the lazy build in dotCl_v8c
 *        still covers those.
 */
bool dotCl_v8c_prebuild_weight(const Tensor &weight);

/**
 * @brief Register the per-output-channel zero points of a per-channel
 *        ASYMMETRIC int4 FC weight on the v8c GPU path.
 *
 * @details An asymmetric weight w[n][k] = s[n] * (q - z[n]), q in [0, 15], is
 * carried in the QS4CX layout unchanged -- nibble q (= signed code q - 8) and
 * scale s[n] -- plus its integer zero points z[n]. With them registered, every
 * v8c FC kernel of that weight (decode GEMV, dp4a / XMX GEMM, imageless
 * lm_head GEMV) runs its -DV8C_ASYM build: the zero point folds into the int32
 * epilogue exactly,
 *   sum_k (a - zp_a)(q - z) = acc - z*row_sum_a - zp_a*(row_sum_w + (8-z)*K),
 * i.e. the constant 8 of the symmetric epilogue becomes z[n] -- no extra
 * launch, no extra pass over the activation (the quantiser already emits its
 * row sum), and the result is the exact dequantised GEMM against the int8
 * activation. This equals adding c[n] * sum_k x[m][k] with c[n] =
 * s[n]*(8-z[n]).
 *
 * Keyed by the weight NAME, so it may be called before or after the weight's
 * device pack is built (load-time prebuild or first dispatch) and survives a
 * rebuild; cl_fc_release_caches() drops it with the packs. A weight with no
 * registered zero points runs the unchanged kernels, bit-identically.
 *
 * @param weight the int4 FC weight tensor (must be named)
 * @param zp     fp32 z[N] holding integer values (|z| <= 255), or nullptr to
 *               remove registered zero points
 * @param N      length of @p zp; must equal weight.width()
 * @return false (nothing registered) on an unnamed weight, a length mismatch,
 *         a non-integer zero point or a device allocation failure
 */
bool dotCl_v8c_set_weight_zero_points(const Tensor &weight, const float *zp,
                                      unsigned int N);

/**
 * @brief Same registration from the per-channel correction c[n] =
 *        s[n] * (8 - z[n]) (the form an asymmetric-weight loader may keep):
 *        recovers the integer z[n] from c and the weight's own fp32 scales,
 *        then calls dotCl_v8c_set_weight_zero_points(). Needs the scales to
 *        be readable (call before the plain payload is released).
 * @param corr c[N], or nullptr to remove registered zero points
 * @return false on a length mismatch, unreadable scales or a correction that
 *         does not correspond to an integer zero point
 */
bool dotCl_v8c_set_zero_point_correction(const Tensor &weight,
                                         const float *corr, unsigned int N);

/**
 * @brief Recover integer zero points z[n] from c[n] = scale[n] * (8 - z[n]).
 * @return false when a scale is not positive or a c[n] is not of that form
 */
bool zeroPointsFromCorrection(const float *scale, const float *corr,
                              unsigned int N, std::vector<float> &zp);

/**
 * @brief Eager v8c weight build whose nibbles come from @p src_nibbles rather
 *        than from the tensor's own storage.
 *
 * @details The zero-copy weight load: the loader hands the weight file's
 * mapping straight to the device build, so the plain payload is never copied
 * into the tensor at all. The tensor is still the cache identity (its address
 * is the key) and still owns the per-channel scales, which the caller has
 * read; only the nibbles are read from @p src_nibbles.
 *
 * On success the tensor's payload holds nothing meaningful and must not be
 * read by a host consumer -- the same contract the post-build DROP_PLAIN
 * release establishes, which is why this path is tied to that lever. On
 * failure nothing has changed and the caller must read the payload.
 *
 * @param[in] weight the QS4CX weight tensor (identity + scales)
 * @param[in] src_nibbles N*ceil(K/2) plain nibble bytes for this weight
 * @return true when the device backing was built from @p src_nibbles
 */
bool dotCl_v8c_prebuild_weight_from(const Tensor &weight,
                                    const void *src_nibbles);

/**
 * @brief Upload a boundary tensor's host bytes into the device buffer that
 *        backs it (host -> cl_mem RAISE).
 *
 * A tensor the planner placed on the GPU_CLMEM residency class keeps its bytes
 * in a device buffer; its host mirror is only meaningful once one of these two
 * calls has moved them. Use this after a genuine host WRITE, so the kernels
 * that read the buffer next see what the host produced.
 *
 * Only offset-0 views are bridgeable: the sub-buffer covers the whole tensor,
 * so a nonzero-offset view would read from the wrong place. That case throws
 * rather than silently misreading.
 *
 * @param[in] t tensor to raise; a non-cl_mem tensor is a no-op
 * @param[in] valid_bytes bytes to move, or 0 for the whole tensor
 * @return true when bytes were moved
 */
bool clmem_raise_cl(const Tensor &t, unsigned int valid_bytes);

/**
 * @brief Read a boundary tensor's device buffer back into its host mirror
 *        (cl_mem -> host LOWER). The counterpart of clmem_raise_cl.
 *
 * Use before a genuine host READ. The read is blocking on the in-order queue,
 * so it also waits for every command already enqueued -- which is exactly the
 * ordering a host consumer needs.
 *
 * @param[in] t tensor to lower; a non-cl_mem tensor is a no-op
 * @param[in] valid_bytes bytes to move, or 0 for the whole tensor
 * @return true when bytes were moved
 */
bool clmem_lower_cl(const Tensor &t, unsigned int valid_bytes);

/**
 * @brief Read one byte range of a boundary tensor's device buffer back into
 *        the same range of its host mirror (a partial cl_mem -> host LOWER).
 *
 * clmem_lower_cl() for a consumer that reads only part of the tensor, such as
 * the lm_head, which reads the last row of the final norm's output. The read
 * is blocking on the in-order queue, so it lands after every command already
 * enqueued -- including the kernel that produces the range.
 *
 * @param[in] t tensor to lower (offset-0 view); a non-cl_mem tensor is a no-op
 * @param[in] byte_offset first byte of the range, from the start of @a t
 * @param[in] bytes length of the range
 * @return true when bytes were moved
 */
bool clmem_lower_range_cl(const Tensor &t, size_t byte_offset, size_t bytes);

/**
 * @brief Residual copy / accumulate for operands on the device plane.
 *
 * Both operands must be device-visible -- on the device plane, or SVM. Once
 * one of them is device-plane resident the other is bound as an SVM argument,
 * so a plain host operand is refused at argument binding rather than
 * dispatched: a host pointer is not a kernel operand.
 *
 * @param[in,out] dst destination tensor (the residual accumulator)
 * @param[in] src source tensor
 * @param[in] accumulate false: dst = src; true: dst += src
 * @return false when neither operand is device-plane resident, so the caller
 *         keeps its shared-plane path
 */
bool clmem_residual_op_cl(Tensor &dst, const Tensor &src, bool accumulate);

/**
 * @brief Both residual operands in one device dispatch: dst = a + b.
 * @param[out] dst destination (device plane)
 * @param[in] a first operand (device plane)
 * @param[in] b second operand (device plane)
 * @return false when any operand is not device-plane resident or the shapes
 *         disagree, so the caller keeps the per-operand path
 */
bool clmem_residual_add2_cl(Tensor &dst, const Tensor &a, const Tensor &b);

/**
 * @brief Reserve the v8c activation scratch a fused norm can quantise into.
 *
 * A norm that writes a device-plane row is, on every LLM graph this backend
 * runs, immediately read by a v8c FC that would quantise exactly that row.
 * Letting the norm emit the quantisation itself removes a dispatch, and on
 * this device a dispatch is worth three to four times its own GPU time. This
 * hands the norm the same per-fanout scratch the FC's own quantiser would
 * have filled, so the FC can then simply use it.
 *
 * Nothing is promised: when the fusion is off, the shapes do not fit, or the
 * scratch cannot be grown, this returns false and the caller keeps the plain
 * norm. A reservation that is never committed costs one slot of the ring.
 *
 * @param[in] site stable identity of the calling norm (its gamma pointer):
 *            a norm whose quantisation nothing ever claims stops being fused,
 *            so speculation is paid for only where it is collected
 * @param[in] rows rows the norm will write (= the FC's real M)
 * @param[in] K row width (= the FC's K)
 * @param[out] act_i8 int8 activation scratch (cl_mem)
 * @param[out] act_scale per-row scale scratch (cl_mem)
 * @param[out] act_zp per-row zero-point scratch (cl_mem)
 * @param[out] act_rs per-row sum scratch (cl_mem)
 * @return true when the four scratch handles are usable
 */
bool v8cNormQuantBegin(const void *site, unsigned int rows, unsigned int K,
                       void **act_i8, void **act_scale, void **act_zp,
                       void **act_rs);

/**
 * @brief True when NNTR_FUSE_NORM_QUANT=2 asks for the ungated speculation:
 *        every device-plane norm fuses, whatever its shape or consumer.
 */
bool v8cNormQuantUngated();

/**
 * @brief Publish the reservation after the fused norm has been enqueued.
 *
 * @param[in] src_clmem device buffer the norm wrote the fp16 row into -- the
 *            handle the consuming FC will present as its input
 * @param[in] rows rows written
 * @param[in] K row width
 */
void v8cNormQuantCommit(void *src_clmem, unsigned int rows, unsigned int K);

/**
 * @brief Drop a reservation whose dispatch did not happen.
 */
void v8cNormQuantAbort();

/**
 * @brief Claim the outstanding norm->FC handoff for a consumer that is not a
 *        v8c FC (the TQ2CX expert GEMV): the int8 row, scale and zero point
 *        the fused norm wrote, when it wrote them for exactly @a src_clmem
 *        with this shape and nothing has written that buffer since.
 *
 * The caller must quantise with the arithmetic the norm used (the v8c
 * quantiser under the v8c FC's build options, plain TQ2 order) -- the claim
 * only answers whether the bytes behind the handle are still the ones the
 * norm quantised.
 *
 * @param[in] src_clmem device buffer the consumer would quantise
 * @param[in] rows rows the consumer reads
 * @param[in] K row width
 * @param[out] act_i8 int8 rows (cl_mem)
 * @param[out] act_scale per-row scale (cl_mem)
 * @param[out] act_zp per-row zero point (cl_mem)
 * @param[out] act_rs per-row sum (cl_mem)
 * @return true when the four handles hold this input's quantisation
 */
bool v8cNormQuantClaim(void *src_clmem, unsigned int rows, unsigned int K,
                       void **act_i8, void **act_scale, void **act_zp,
                       void **act_rs);

/**
 * @brief Model-teardown hook: drop every process-lifetime cache the OpenCL FC
 *        path keys by a HOST pointer, plus the quantisation handoffs in its
 *        shared scratch.
 *
 * @details The v8c derived-weight packs, the norm-site strike counters and the
 * two activation-quantisation handoffs are all written as "build once, keep for
 * the process" -- correct for a one-shot CLI, wrong for an SDK consumer that
 * destroys a handle and loads another, because the next load's allocator hands
 * out the very addresses those caches are keyed by. Surviving weight packs are
 * a whole model's worth of device memory per load cycle; a surviving quant
 * handoff lets the new model's first FC of a fanout multiply its weights by the
 * previous model's activation. Pure reset -- each item is rebuilt lazily or by
 * the next load -- and no driver call when nothing was cached. Call only when
 * no run is in flight, and before v8c_release_aux_arena(), whose chunks these
 * packs carve.
 *
 * This is the single release path for that state, and it has two callers:
 * NeuralNetwork::deallocate(), so a consumer that only destroys a model gets
 * its device memory back, and CausalLM::releaseDeviceCaches(), which is where
 * the arena / lm_head / attention hooks are ordered after it. Idempotent:
 * whichever runs second finds an empty state.
 */
void cl_fc_release_caches();

} // namespace nntrainer
#endif /* __BLAS_KERNEL_INTERFACE_H__ */
