// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   moe_route_cl.h
 * @date   02 October 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  OpenCL data movement for a mixture-of-experts layer: the per-expert
 *         row gather, the weighted scatter-add back into the layer output, and
 *         the expert-weight FC dispatch seam.
 *
 * @details A MoE layer on the gpu engine runs every selected expert as
 *
 *   gather(token rows of expert e) -> FC gate, FC up -> GLU -> FC down
 *     -> scatter-add(routing weight * row) into the output rows
 *
 * The FCs are the device FC kernels the dense layers use (ComputeOps::fc).
 * The two data-movement steps around them are the kernels here, so the
 * activation rows stay on the device for the whole expert loop instead of
 * crossing to the host twice per expert.
 *
 * The scatter-add is issued once per expert, in the order the caller visits
 * the experts, on the in-order queue. Each call adds to distinct output rows
 * (a token appears at most once in one expert's row list), so a call has no
 * write conflict, and the sum per output element is formed in the visiting
 * order on every run: the result is identical run to run.
 *
 * Operands are FP32 or FP16 tensors on the shared (SVM) plane. A tensor the
 * residency planner placed on the device plane, or a plain host tensor, is
 * declined (the functions return false and enqueue nothing), so the caller
 * keeps its host loop for it.
 */

#ifndef __MOE_ROUTE_CL_H__
#define __MOE_ROUTE_CL_H__

#include <compute_ops.h>
#include <tensor_dim.h>

namespace nntrainer {

class Tensor;

/**
 * @brief out[r, :] = in[src_rows[r], :] for r in [0, rows).
 * @param in source rows (the layer input), [.., N, W]
 * @param out destination rows, [.., >= rows, W]
 * @param src_rows source row index per destination row
 * @param rows number of rows to gather
 * @return false when the operands are not both shared-plane FP32/FP16 tensors
 *         of the same width and dtype (nothing is enqueued then)
 */
bool moe_gather_rows_cl(const Tensor &in, Tensor &out,
                        const unsigned int *src_rows, unsigned int rows);

/**
 * @brief out[dst_rows[r], :] += weights[r] * expert_out[r, :] for r in
 *        [0, rows). The destination rows must be distinct within one call.
 * @return false when the operands are not both shared-plane FP32/FP16 tensors
 *         of the same width and dtype (nothing is enqueued then)
 */
bool moe_scatter_add_rows_cl(const Tensor &expert_out, Tensor &out,
                             const unsigned int *dst_rows, const float *weights,
                             unsigned int rows);

/**
 * @brief out[0 .. rows, :] = 0 on the device.
 * @return false when @a out is not a shared-plane FP32/FP16 tensor
 */
bool moe_zero_rows_cl(Tensor &out, unsigned int rows);

/**
 * @brief Device FC for one expert projection, for a weight dtype that the
 *        dense FC (ComputeOps::fc) does not cover.
 * @param input activation rows [.., M, K]
 * @param weight the expert weight
 * @param output result rows [.., M, N]; produced, not accumulated into
 */
using MoeExpertFcFn = void (*)(Tensor &input, Tensor &weight, Tensor &output);

/**
 * @brief Plug a device expert FC in for one weight dtype.
 *
 * @details This is the seam for an expert-only weight format. The MoE layers
 * project every expert through ComputeOps::fc_expert; on the OpenCL table that
 * looks the weight dtype up here first and only falls back to the dense
 * ComputeOps::fc (the Q4_0 / QS4CX / FP32 / FP16 kernels) when nothing was
 * registered for it. A new expert format therefore adds its kernel and one
 * registration -- no MoE layer changes.
 *
 * @param dtype weight data type the function serves
 * @param fn the FC, or nullptr to remove a registration
 */
void registerMoeExpertFcCl(ml::train::TensorDim::DataType dtype,
                           MoeExpertFcFn fn);

/**
 * @brief The expert FC: the registered function for the weight dtype, else
 *        the dense FC.
 */
void moe_expert_fc_cl(Tensor &input, Tensor &weight, Tensor &output);

/**
 * @brief Runs every selected expert of a MoE call at once for one expert
 *        weight dtype.
 * @return false when it declines (nothing enqueued)
 */
using MoeExpertGroupFn = bool (*)(const MoeExpertGroup &group);

/**
 * @brief Plug a grouped expert runner in for one expert weight dtype (the
 *        dtype of the gate weights). TQ2CX is registered by default.
 */
void registerMoeExpertGroupCl(ml::train::TensorDim::DataType dtype,
                              MoeExpertGroupFn fn);

/**
 * @brief ComputeOps::moe_experts on the OpenCL table: the registered runner
 *        for the expert dtype, or false.
 */
bool moe_experts_cl(const MoeExpertGroup &group);

/**
 * @brief Build the device copy of a MoE layer's experts at load (see
 *        ComputeOps::moe_experts_prebuild). TQ2CX is served; anything else
 *        returns false.
 */
bool moe_experts_prebuild_cl(const MoeExpertGroup &group);

/**
 * @brief ComputeOps::moe_route on the OpenCL table: route router rows on the
 *        device (the deterministic router kernels), one blocking read of the
 *        ids and weights; false, having enqueued nothing, when not served.
 */
bool moe_route_cl(const MoeRouter &router, unsigned int tokens, uint32_t *ids,
                  float *weights);

/**
 * @brief Start the reads of a streamed MoE layer's next call ahead (see
 *        ComputeOps::moe_experts_prefetch). TQ2CX is served; anything else
 *        returns false.
 */
bool moe_experts_prefetch_cl(const MoeExpertGroup &group);

/**
 * @brief Wait for the off-thread part of a streamed MoE layer's earlier
 *        calls (see ComputeOps::moe_experts_join).
 */
void moe_experts_join_cl(const MoeExpertGroup &group);

/**
 * @brief Enqueue the deferred kernels that write @a t, if any (see
 *        ComputeOps::flush_deferred).
 */
void moe_experts_flush_deferred_cl(const Tensor &t);

/**
 * @brief Does this device run the grouped expert launches, with device-side
 *        routing of a decode row, for experts of @a expert_dtype? A MoE layer
 *        whose experts it serves never reads its rows on the host in decode,
 *        so the model may keep them in device-only memory.
 */
bool moe_experts_device_served_cl(ml::train::TensorDim::DataType expert_dtype);

} // namespace nntrainer

#endif // __MOE_ROUTE_CL_H__
