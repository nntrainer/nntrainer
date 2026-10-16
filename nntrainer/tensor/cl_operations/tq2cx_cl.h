// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   tq2cx_cl.h
 * @date   02 October 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  OpenCL FC for a TQ2CX (ternary) weight: one expert projection.
 *
 * @details The device FC behind ComputeOps::fc_expert for TQ2CX weights,
 * plugged in through registerMoeExpertFcCl. One call projects one weight:
 *
 *   activation rows -> int8 per row (asymmetric, scale + zero point, the
 *   v8c activation quantizer's scheme) -> tq2cx_gemv_row (M = 1) or
 *   tq2cx_gemm_dp4a (M > 1) -> FP32 rows
 *
 * The weight record (masks, then the per-channel scales) is uploaded to the
 * device once and kept there for the life of the process, keyed by the
 * weight's host address: the experts it serves are resident weights, whose
 * address does not change. The accumulation is integer, so the result does
 * not depend on the work split and is identical run to run.
 */

#ifndef __TQ2CX_CL_H__
#define __TQ2CX_CL_H__

#include <compute_ops.h>

namespace nntrainer {

class Tensor;

/**
 * @brief output = input x weight for a TQ2CX @a weight.
 * @param input FP32 activation rows [.., M, K] on the shared plane or host
 * @param weight TQ2CX weight, K = height(), N = width(), K % 32 == 0
 * @param output FP32 rows [.., M, N] on the shared plane; produced, not
 *        accumulated into
 */
void tq2cx_expert_fc_cl(Tensor &input, Tensor &weight, Tensor &output);

/**
 * @brief Every selected TQ2CX expert of a MoE call in a handful of launches.
 *
 * @details The layer's experts are uploaded once, on first use, as ONE device
 * buffer of back-to-back TQ2CX records (gate, up, down of every expert, each
 * 64-byte aligned), addressed by word offsets. Then:
 *
 *  - decode (one token): quantize the row (v8c activation quantizer) ->
 *    tq2cx_gemv_batch over the gate and up records of all selected experts
 *    (one launch, shared activation row) -> GeGLU -> quantize the k rows ->
 *    tq2cx_gemv_batch over their down records (row b per expert) ->
 *    weighted combine into the output row;
 *  - prefill: gather every expert's routed rows -> quantize ->
 *    tq2cx_gemm_xmx_grouped for gate and for up (one launch each, all
 *    experts) -> GeGLU -> quantize -> grouped down -> destination-owned
 *    weighted combine.
 *
 * All accumulation is integer and every combine visits the experts in
 * ascending id, so the output is identical run to run. Declines (returns
 * false) for shapes the kernels do not take (K % 32, the XMX block rules)
 * or when the device has no DPAS for the prefill GEMM.
 *
 * Streamed experts (group.fsu): the layer's experts live in its partition of
 * the expert slot arena (group.fsu->slots slots of one record each, one
 * buffer per layer) instead of a stack of all of them. Before a launch reads
 * them, the experts the layer's acquire hook reports as misses are read from
 * the weight file (ExpertReaderPool, page-aligned bounce, non-blocking write
 * on a separate I/O queue) and the compute queue waits on those writes; a
 * slot is rewritten only after the marker that follows its last reader. A
 * prefill call with more active experts than slots runs in chunks of at most
 * that many experts (same output). The device router serves such a layer
 * only once every expert is resident (group.fsu->complete).
 */
bool tq2cx_moe_experts_cl(const MoeExpertGroup &group);

/**
 * @brief Build the layer's stacked device expert records now, from
 *        group.records when given (the file mapping) or from the weights;
 *        with group.fsu, in the layer's partition of the expert slot arena,
 *        reading group.fsu->fill from the file (and waiting for it).
 * @return true when the stack exists (built now or before); false, having
 *         built nothing, when the grouped kernels do not serve this layer.
 */
bool tq2cx_moe_experts_prebuild_cl(const MoeExpertGroup &group);

/**
 * @brief Route @a tokens router rows on the device (ComputeOps::moe_route):
 *        the deterministic router of moe_route_det.h, four launches with no
 *        local memory (tq2moe_route_*_rows), then one blocking read of the
 *        selected ids and weights.
 * @param[out] ids     [tokens][topk] expert ids in rank order
 * @param[out] weights [tokens][topk] their routing weights
 * @return false, having enqueued nothing, when the device or the operands
 *         are not served (the caller routes on the host)
 */
bool tq2cx_moe_route_cl(const MoeRouter &rt, unsigned int tokens, uint32_t *ids,
                        float *weights);

/**
 * @brief Streamed experts (group.fsu) whose next call streams whole layers
 *        (group.tokens >= fsu->spec_min_tokens, fewer slots than experts):
 *        start reading this layer's block (when it was not read ahead) and
 *        the next layer's block into the partitions the arena lends to the
 *        stream. False when the arena cannot lend them (nothing started).
 */
bool tq2cx_moe_experts_prefetch_cl(const MoeExpertGroup &group);

/**
 * @brief Wait for the decode step of the streamed layer group.layer_key
 *        whose experts are made resident on reader threads
 *        (NNTR_G4MOE_FSU_ASYNC); rethrow its failure. Also enqueues any
 *        deferred step tail first. No-op for other layers.
 */
void tq2cx_moe_experts_join_cl(const MoeExpertGroup &group);

/**
 * @brief Enqueue the deferred tail (expert GEMVs and combine) of the
 *        streamed decode step whose output is @a t, if one is pending.
 *        Cheap when nothing is pending (one atomic load).
 */
void tq2cx_moe_experts_flush_deferred_cl(const Tensor &t);

/**
 * @brief The time of one poll of the streamed decode's gate kernel
 *        (NNTR_G4MOE_FSU_FLAGS) on the registered gpu context, in ns,
 *        measured now on scratch blocks (the flags protocol measures it once
 *        at its decision to turn the gates' spin account into ms).
 * @return 0 when it cannot run here (no gpu context, no SVM, no build)
 */
double tq2cx_fsu_gate_poll_ns_cl();

} // namespace nntrainer

#endif // __TQ2CX_CL_H__
