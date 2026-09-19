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
 * @brief Eagerly build the v8c GPU weight entry (nibble permute + upload +
 *        image view) for a freshly READ int4 FC weight, so the first prefill
 *        does not pay the lazy per-weight build. Called by the CL FC layer
 *        after the base read. Returns false (no-op) off the v8c path (env
 *        unset / non-int4 / unsupported shape); the lazy build in dotCl_v8c
 *        still covers those.
 */
bool dotCl_v8c_prebuild_weight(const Tensor &weight);

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

} // namespace nntrainer
#endif /* __BLAS_KERNEL_INTERFACE_H__ */
