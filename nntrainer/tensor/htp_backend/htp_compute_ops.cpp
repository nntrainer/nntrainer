// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   htp_compute_ops.cpp
 * @date   18 Jun 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  HTP (Hexagon/HMX) ComputeOps entry point.
 *
 * Compiled only when ENABLE_HEXKL is defined.
 *
 * HtpComputeOps overrides exactly the ops it accelerates and inherits the
 * CPU implementation of everything else from CpuComputeOps.
 *
 * It deliberately does NOT derive from the abstract ComputeOps base the way
 * ClComputeOps does. A layer's engine covers every tensor in it, not just
 * the ones this backend has a kernel for: Lfm2MoELayer's router gate is
 * FP32 by construction (lfm2_moe_layer.cpp keeps gate and expert_bias
 * unquantized), so with engine=htp its dot() reaches sgemm_fp32, which has
 * no supports_* guard and no fallback of its own in FloatTensor::dot. Off
 * the abstract base that threw "ComputeOps::sgemm_fp32 not implemented by
 * this backend" on the first prefill. CpuComputeOps is stateless, so
 * inheriting it costs nothing and makes every unaccelerated op behave the
 * way it does with engine=cpu.
 *
 * gemm_q4_0_accel_fp32 is the first kernel: one FastRPC call per Q4_0 FC
 * dot(), single weight, single activation -- hexkl_mm_u8i4_layer_run with
 * n_handles=1 underneath. It is Tier 1 of docs/htp_attention/
 * 40_moe_ffn_htp_task.md section 3 and benefits every FC layer under
 * engine=htp, not just the LFM2 MoE FFN that motivated this task.
 *
 * gemm_q4_0_batch_fp32 is the same call with n_handles > 1: several
 * weights sharing one activation (LFM2-MoE decode's selected experts'
 * gate_up projections against the single routed token) go out as one
 * FastRPC call so hexkl_mm_u8i4_layer_run can prefetch the next handle's
 * weight while the current one computes.
 */

#ifdef ENABLE_HEXKL

#include <attn_m1_det.h>
#include <compute_ops.h>
#include <context_data.h>
#include <cpu_backend.h>
#include <cpu_ops_table.h>
#include <htp_act_quant.h>
#include <htp_backend.h>
#include <htp_context.h>
#include <htp_dspq_wire.h>
#include <htp_expert_prefetch.h>
#include <htp_expert_read_pool.h>
#include <htp_fixed_opts.h>
#include <htp_graph_desc.h>
#include <htp_moe_opts.h>
#include <htp_prepared_tl2.h>
#include <htp_q4_0_convert.h>
#include <htp_router_predict.h>
#include <htp_rpcmem.h>
#include <htp_token_completion.h>
#include <htp_wh_layout.h>
#include <htp_wh_palette.h>
#include <m1_ops_det.h>
#include <nntr_htp_decode_hook.h>
#include <nntrainer_log.h>
#include <q4_gemv_cpu_det.h>
#include <swiglu_det.h>
#include <thread_manager.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <weight_file_reader.h>

#if defined(__linux__)
#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <remote.h>

#include <hmx/hexkl_dma_trace.h>

#include <nntr_hvx.h>

namespace nntrainer {

namespace {

/** [#225] Q4_0 weights re-quantized by qs4cxFromModelQ4_0 in this process:
 *  the FC WH banner's requant=, 0 when the sidecar served every FC. */
std::atomic<unsigned> g_q4_requant{0};

/**
 * @brief htp_qs4cx_from_q4_0x4 on a model weight in the CPU's own repack.
 *
 * On the device that repack is q4_0x4, the converter's format, and the
 * bytes pass through. The in-process host build (#84) holds the host
 * ISA's repack (x8 on x86), which the converter would read as garbage
 * scales (NaN logits): it is re-laid as q4_0x4 first, so the prefill FC
 * kinds the engine keys route (#222) run on the host as on the device.
 */
void qs4cxFromModelQ4_0(const void *w, uint32_t K, uint32_t N, int8_t *q,
                        float *scale, int32_t *colsum) {
  ++g_q4_requant;
#if defined(__aarch64__)
  htp_qs4cx_from_q4_0x4(w, K, N, q, scale, colsum);
#else
  const size_t bytes = static_cast<size_t>(N) * (K / 32u) * Q4_CPU_BLOCK_BYTES;
  std::vector<char> canon(bytes), x4(bytes);
  unpack_q4_0(w, canon.data(), bytes, N, K);
  repack_q4_0(x4.data(), canon.data(), bytes, N, K, ml::train::ISA::ARM);
  htp_qs4cx_from_q4_0x4(x4.data(), K, N, q, scale, colsum);
#endif
}

/**
 * @brief Slots mm_u8i4_layer_timed fills.
 *
 * Restated here because the ARM side cannot include the DSP source that
 * declares them (nntr_hvx_mm_u8i4.c). The entry point rejects a stale count
 * with AEE_EBADPARM rather than truncating, so a drift shows up as a failed
 * call, not as wrong numbers.
 */
enum {
  HTP_T_DSP_TOTAL = 0,
  HTP_T_QUANT,
  HTP_T_DEQUANT,
  HTP_T_ACC_READ,
  HTP_T_ACC_COPY,
  HTP_T_DRAIN,
  HTP_T_ACC_STRIDE,
  HTP_N_STAGES
};

/**
 * @brief Slots mm_u8i4_layer_fused_timed fills.
 *
 * Restated for the same reason as HTP_T_* (the ARM side cannot include the
 * DSP source). NOT the HTP_T_* layout: the fused call runs a second QUANT
 * pass (the SwiGLU output's requant) and a SwiGLU stage the unfused call
 * does not have, so it carries its own slot list and its own Bucket
 * counter -- folding SwiGLU into an existing slot would put its time in a
 * column named after something else.
 */
enum {
  HTP_FU_T_DSP_TOTAL = 0,
  HTP_FU_T_QUANT,
  HTP_FU_T_SWIGLU,
  HTP_FU_T_DEQUANT,
  HTP_FU_T_ACC_READ,
  HTP_FU_T_ACC_COPY,
  HTP_FU_T_DRAIN,
  HTP_FU_T_ACC_STRIDE,
  HTP_FU_N_STAGES
};

/**
 * @brief Slots mm_u8i4_gate_up_swiglu_timed fills, in order -- test/htp/
 * nntr_hvx_mm_u8i4.c's GU_T_* enum restated (the ARM side cannot include
 * that DSP source). NOT the same layout as HTP_FU_T_*: this call has no
 * down-side accumulator at all, so there is no ACC_COPY slot -- reusing
 * HTP_FU_T_*'s indices against this array would silently read DRAIN's
 * value out of the slot ACC_STRIDE actually lives in.
 */
enum {
  HTP_GU_T_DSP_TOTAL = 0,
  HTP_GU_T_QUANT,
  HTP_GU_T_SWIGLU,
  HTP_GU_T_DEQUANT,
  HTP_GU_T_ACC_READ,
  HTP_GU_T_DRAIN,
  HTP_GU_T_ACC_STRIDE,
  HTP_GU_N_STAGES
};

/**
 * @brief Slots mm_u8i4_moe_layer_timed fills, in order -- test/htp/
 * nntr_hvx_mm_u8i4.c's MOE_T_* restated, for the same reason HTP_GU_T_* is.
 * One slot none of the others have: the routing multiply and scatter-add,
 * which live on the ARM side until this call takes them.
 */
enum {
  HTP_MOE_T_DSP_TOTAL = 0,
  HTP_MOE_T_QUANT,
  HTP_MOE_T_SWIGLU,
  HTP_MOE_T_DEQUANT,
  HTP_MOE_T_ACC_READ,
  HTP_MOE_T_DRAIN,
  HTP_MOE_T_SCATTER,
  HTP_MOE_T_GATHER,
  HTP_MOE_T_REQUANT,
  HTP_MOE_T_BLOCKS,
  HTP_MOE_T_MM,
  HTP_MOE_T_DMA_KB,
  HTP_MOE_T_DMA_FIRST,
  HTP_MOE_T_ALLOC, /**< the layer call's own malloc and free */
  HTP_MOE_T_DMA_FIRST_KB,
  HTP_MOE_T_DRAIN_DN,
  HTP_MOE_T_PUSH,
  HTP_MOE_T_STAGE,
  HTP_MOE_T_ACC_STRIDE,
  /** [#87] The DMA ring trace's per-call numbers, hexkl_probe.h's
      HEXKL_PROBE_DMA_* in the same order. Counts unless named _US. */
  HTP_MOE_T_DMA_DESC,
  HTP_MOE_T_DMA_WAITS,
  HTP_MOE_T_DMA_WAITS_BLOCKED,
  HTP_MOE_T_DMA_WAIT_US,
  HTP_MOE_T_DMA_WAIT_ACT_US,
  HTP_MOE_T_DMA_BUSY_LO_US,
  HTP_MOE_T_DMA_BUSY_HI_US,
  HTP_MOE_T_DMA_DEPTH_MAX,
  HTP_MOE_T_DMA_FIRST_READY_US,
  HTP_MOE_T_DMA_LAST_ISSUE_US,
  HTP_MOE_T_PATH,    /**< NOT us: 0 = HMX block loop, 1 = M=1 HVX GEMV */
  HTP_MOE_T_M1_FEED, /**< NOT us: the DMA queues the VTCM feed used (#117:
                          1; #177: 1..4), 0 = the GEMV read the arena */
  HTP_MOE_N_STAGES
};

/**
 * @brief Per-stage timing for the HTP path. Off unless NNTR_HTP_PROFILE is set.
 *
 * NNTR_HTP_PROFILE=1 times the three host-side stages a weight goes through:
 * the Q4_0/QS4CX -> HexKL-registry conversion, the weight_register FastRPC,
 * and the per-dot() layer call. =2 additionally routes the layer call through
 * mm_u8i4_layer_timed, so the DSP reports its own microseconds and the
 * FastRPC transport share becomes host_us - dsp_us rather than a guess
 * (mobile_e2e_run_guide.md section 6, evidence step 2).
 *
 * Registration is accounted separately from the layer call on purpose: it
 * runs once per weight pointer and therefore lands entirely in the first
 * forward pass, so "prefill got slower but decode did not" and "every call
 * got slower" are different findings and have to be separable.
 */
class HtpProfile {
public:
  static HtpProfile &global() {
    static HtpProfile instance;
    return instance;
  }

  int level() const { return level_; }

  static uint64_t nowUs() {
    return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch())
        .count());
  }

  /** [#85] The per-token entry's own line under the layer rows: how many
   *  forward calls, ops per call, the DSP time of the whole call and the
   *  sum of the per-op pcycle brackets (level >= 2, forward_debug). The
   *  MoE op's stages go to the M==1 bucket through addInvokeMoeLayer, so
   *  that row stays comparable to the per-layer path's. */
  void addInvokeForward(unsigned n_ops, uint64_t dsp_us, uint64_t op_pcyc) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++graph_calls_;
    graph_ops_ += n_ops;
    graph_dsp_us_ += dsp_us;
    graph_op_pcyc_ += op_pcyc;
  }
  /** [#130] One resident op's pcycle bracket, by kind (level >= 2). */
  void addGraphOp(unsigned kind, uint64_t pcyc) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (kind < HTP_OP_KIND_N) {
      ++graph_kind_n_[kind];
      graph_kind_pcyc_[kind] += pcyc;
    }
  }
  void setGraphResident(uint32_t mask) { graph_resident_ = mask; }

  void addRegister(uint64_t total_us, uint64_t convert_us, uint64_t rpc_us,
                   bool ion) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++reg_calls_;
    reg_total_us_ += total_us;
    convert_us_ += convert_us;
    rpc_us_ += rpc_us;
    if (ion)
      ++reg_ion_calls_;
  }

  /** @brief One expert brought in from the model file at forward time, or
   *  one released to make room (doc 52): read_us is the pread into the
   *  arena slot, rpc_us the register/release FastRPC calls. Held as
   *  "pending" and attributed to the next MoE layer call's row, which is
   *  the call that paid for it, so the row reads miss/call directly. */
  void addExpertLoad(uint64_t read_us, uint64_t rpc_us, uint64_t loads) {
    std::lock_guard<std::mutex> lock(mutex_);
    miss_pending_ += loads;
    miss_read_pending_us_ += read_us;
    miss_rpc_pending_us_ += rpc_us;
    miss_total_ += loads;
    miss_read_total_us_ += read_us;
    miss_rpc_total_us_ += rpc_us;
  }

  /** @brief Experts read while a layer call ran (doc 52 section 10.10):
   *  wait_us is only the read time the call did not cover. Kept apart from
   *  the misses so the miss columns keep meaning a synchronous miss. */
  void addPrefetch(uint64_t n, uint64_t done_on_entry, uint64_t wait_us,
                   uint64_t rpc_us, uint32_t caller_cpus,
                   uint32_t reader_cpus) {
    std::lock_guard<std::mutex> lock(mutex_);
    prefetch_n_ += n;
    prefetch_done_on_entry_ += done_on_entry;
    prefetch_wait_us_ += wait_us;
    prefetch_rpc_us_ += rpc_us;
    prefetch_caller_cpus_ |= caller_cpus;
    prefetch_reader_cpus_ |= reader_cpus;
  }

  /** @brief One expert a reader thread read, and how long it took (doc 52
   *  section 10.30). */
  void addPrefetchRead(uint64_t read_us) {
    std::lock_guard<std::mutex> lock(mutex_);
    prefetch_read_us_ += read_us;
    ++prefetch_reads_;
  }

  void addInvoke(unsigned M, unsigned K, unsigned N, uint64_t host_us,
                 const uint32_t *stage_us) {
    std::lock_guard<std::mutex> lock(mutex_);
    // kind 3: a plain FC call. It gets its own row because the attention
    // q/o projections have the MoE layer call's K and N_out (2048 x 2048),
    // and the first all-on run averaged 12 two-millisecond FC calls into
    // the 22 fifteen-millisecond MoE calls of one "M>1" row.
    Bucket &b = buckets_[std::make_tuple(K, N, M == 1, 3)];
    ++b.calls;
    b.rows += M;
    b.host_us += host_us;
    if (stage_us != nullptr) {
      b.dsp_us += stage_us[HTP_T_DSP_TOTAL];
      b.quant_us += stage_us[HTP_T_QUANT];
      b.dequant_us += stage_us[HTP_T_DEQUANT];
      b.acc_us += stage_us[HTP_T_ACC_READ] + stage_us[HTP_T_ACC_COPY];
      b.drain_us += stage_us[HTP_T_DRAIN];
    }
  }

  /** Same buckets as addInvoke -- the fused call has its own K/N shape (the
   * down matmul's output width, not gate_up's), so it lands in its own row
   * and the two-dot path's numbers stay directly comparable across builds. */
  void addInvokeFused(unsigned M, unsigned K, unsigned N, uint64_t host_us,
                      const uint32_t *stage_us) {
    std::lock_guard<std::mutex> lock(mutex_);
    Bucket &b = buckets_[std::make_tuple(K, N, M == 1, 0)];
    ++b.calls;
    b.rows += M;
    b.host_us += host_us;
    if (stage_us != nullptr) {
      b.dsp_us += stage_us[HTP_FU_T_DSP_TOTAL];
      b.quant_us += stage_us[HTP_FU_T_QUANT];
      b.swiglu_us += stage_us[HTP_FU_T_SWIGLU];
      b.dequant_us += stage_us[HTP_FU_T_DEQUANT];
      b.acc_us += stage_us[HTP_FU_T_ACC_READ] + stage_us[HTP_FU_T_ACC_COPY];
      b.drain_us += stage_us[HTP_FU_T_DRAIN];
    }
  }

  /** Same buckets again, this call's own (smaller) stage layout -- see
   * HTP_GU_T_*'s doc comment for why it is not HTP_FU_T_*. Bucketed under
   * (K, 2*inter, M==1) -- gate_up's own real shape -- not the down matmul's,
   * so this row is directly comparable to what a plain (unfused) gate_up
   * dot() would have shown for the same layer. */
  void addInvokeGateUpSwiglu(unsigned M, unsigned K, unsigned N_gate_up,
                             uint64_t host_us, const uint32_t *stage_us) {
    std::lock_guard<std::mutex> lock(mutex_);
    Bucket &b = buckets_[std::make_tuple(K, N_gate_up, M == 1, 0)];
    ++b.calls;
    b.rows += M;
    b.host_us += host_us;
    if (stage_us != nullptr) {
      b.dsp_us += stage_us[HTP_GU_T_DSP_TOTAL];
      b.quant_us += stage_us[HTP_GU_T_QUANT];
      b.swiglu_us += stage_us[HTP_GU_T_SWIGLU];
      b.dequant_us += stage_us[HTP_GU_T_DEQUANT];
      b.acc_us += stage_us[HTP_GU_T_ACC_READ];
      b.drain_us += stage_us[HTP_GU_T_DRAIN];
    }
  }

  /** One call now covers a whole layer, so this bucket's `calls` is 1 where
   *  the split-call path's was 64. Bucketed under (K, N_out) -- the layer's
   *  own shape -- which is deliberately NOT either of the two shapes the
   *  path it replaces reported, so a profile cannot be read as if the two
   *  were the same row. `rows` stays M, the tokens the layer saw, not the
   *  expert-slot count, so it means the same thing it always did.
   *
   *  The scatter and the FastRPC-buffer staging go into their own field,
   *  not into acc_us. Folding them in looked tidy and cost a measurement:
   *  the first run of this call put 104.8 ms in a bucket ACC_READ and the
   *  scatter shared, and the profile could not say which. */
  /** @param kind 0 the MoE layer; 1 the dense FFN routed through the MoE
   *  layer kernel (doc 51 section 1); 2 the conv block (doc 51 section 2).
   *  Each files in its own row: the dense FFN has the MoE layer's shapes
   *  and the conv block the MoE row's K and N_out, so the key alone could
   *  not tell them apart. All three fill the same stage_us columns. */
  void addInvokeMoeLayer(unsigned M, unsigned K, unsigned N_out,
                         uint64_t host_us, const uint32_t *stage_us,
                         const HtpRpcBuffer &act_stage,
                         const HtpRpcBuffer &out_stage, size_t in_arg_bytes,
                         int kind = 0, bool via_dspq = false) {
    std::lock_guard<std::mutex> lock(mutex_);
    Bucket &b = buckets_[std::make_tuple(K, N_out, M == 1, kind)];
    ++b.calls;
    b.dspq_calls += via_dspq ? 1 : 0;
    b.rows += M;
    b.host_us += host_us;
    // [#88] What this call staged through, for the staging: line. The
    // class sizes are the max over the bucket's calls (one class per shape
    // in practice), ion is the AND (one heap fallback voids the number).
    b.stage_act_bytes = std::max(b.stage_act_bytes, act_stage.size());
    b.stage_out_bytes = std::max(b.stage_out_bytes, out_stage.size());
    b.stage_ion = b.stage_ion && act_stage.isIon() && out_stage.isIon();
    b.in_arg_bytes = std::max<uint64_t>(b.in_arg_bytes, in_arg_bytes);
    b.swiglu_hidden = true;
    b.misses += miss_pending_;
    b.miss_read_us += miss_read_pending_us_;
    b.miss_rpc_us += miss_rpc_pending_us_;
    miss_pending_ = miss_read_pending_us_ = miss_rpc_pending_us_ = 0;
    if (stage_us != nullptr) {
      b.dsp_us += stage_us[HTP_MOE_T_DSP_TOTAL];
      b.quant_us += stage_us[HTP_MOE_T_QUANT];
      b.swiglu_us += stage_us[HTP_MOE_T_SWIGLU];
      b.dequant_us += stage_us[HTP_MOE_T_DEQUANT];
      b.acc_us += stage_us[HTP_MOE_T_ACC_READ];
      b.scatter_us += stage_us[HTP_MOE_T_SCATTER];
      b.stage_us += stage_us[HTP_MOE_T_STAGE];
      b.gather_us += stage_us[HTP_MOE_T_GATHER];
      b.requant_us += stage_us[HTP_MOE_T_REQUANT];
      b.blocks += stage_us[HTP_MOE_T_BLOCKS];
      b.mm_us += stage_us[HTP_MOE_T_MM];
      b.dma_kb += stage_us[HTP_MOE_T_DMA_KB];
      b.dma_first_us += stage_us[HTP_MOE_T_DMA_FIRST];
      b.drain_dn_us += stage_us[HTP_MOE_T_DRAIN_DN];
      b.dma_first_kb += stage_us[HTP_MOE_T_DMA_FIRST_KB];
      b.alloc_us += stage_us[HTP_MOE_T_ALLOC];
      b.push_us += stage_us[HTP_MOE_T_PUSH];
      b.drain_us += stage_us[HTP_MOE_T_DRAIN];
      b.dma_desc += stage_us[HTP_MOE_T_DMA_DESC];
      b.dma_waits += stage_us[HTP_MOE_T_DMA_WAITS];
      b.dma_waits_blocked += stage_us[HTP_MOE_T_DMA_WAITS_BLOCKED];
      b.dma_wait_us += stage_us[HTP_MOE_T_DMA_WAIT_US];
      b.dma_wait_act_us += stage_us[HTP_MOE_T_DMA_WAIT_ACT_US];
      b.dma_busy_lo_us += stage_us[HTP_MOE_T_DMA_BUSY_LO_US];
      b.dma_busy_hi_us += stage_us[HTP_MOE_T_DMA_BUSY_HI_US];
      if (stage_us[HTP_MOE_T_DMA_DEPTH_MAX] > b.dma_depth_max) {
        b.dma_depth_max = stage_us[HTP_MOE_T_DMA_DEPTH_MAX];
      }
      b.dma_first_ready_us += stage_us[HTP_MOE_T_DMA_FIRST_READY_US];
      b.dma_last_issue_us += stage_us[HTP_MOE_T_DMA_LAST_ISSUE_US];
      b.m1_calls += (stage_us[HTP_MOE_T_PATH] != 0u) ? 1u : 0u;
      b.m1_feed_calls += (stage_us[HTP_MOE_T_M1_FEED] != 0u) ? 1u : 0u;
      b.m1_feed_q_sum += stage_us[HTP_MOE_T_M1_FEED];
    }
  }

  /** @brief [#87] Whether this call's per-descriptor trace should be
   *  dumped: the first NNTR_HTP_DMA_TRACE (default 3) timed calls of each
   *  bucket. Returns the 1-based call ordinal to print, or 0. */
  unsigned dmaTraceOrdinal(unsigned K, unsigned N_out, unsigned M,
                           int kind = 0) {
    std::lock_guard<std::mutex> lock(mutex_);
    Bucket &b = buckets_[std::make_tuple(K, N_out, M == 1, kind)];
    if (b.dma_traces >= dma_trace_calls_) {
      return 0;
    }
    return static_cast<unsigned>(++b.dma_traces);
  }

  /**
   * @brief [#87] Prints one traced call, one line per push and per wait,
   *        as read back by moe_dma_trace_read. Ticks are 19.2 MHz.
   *
   * At level 3 the trace is the last of the repeats -- a warm call -- which
   * the header states as rep=; level 2 prints single-shot (cold) calls.
   * Both are wanted (plan 87 hypothesis g).
   */
  static void dumpDmaTrace(unsigned ordinal, unsigned M, int reps,
                           const uint32_t *w, uint32_t n_words) {
    static const char *const kKind[] = {"act", "gate", "up", "down", "copy"};
    static const char *const kSite[] = {"act", "gu", "dn", "copy_in",
                                        "copy_out"};
    auto us = [](uint32_t ticks) { return ticks / 19.2; };
    if (n_words < HEXKL_DMA_TRACE_HDR_WORDS) {
      std::fprintf(stderr, "[HTP-DMA] call=%u M=%u trace unavailable\n",
                   ordinal, M);
      return;
    }
    const uint32_t n_push = w[0], n_wait = w[1], pw = w[6], ww = w[7];
    if (pw != HEXKL_DMA_TRACE_PUSH_WORDS || ww != HEXKL_DMA_TRACE_WAIT_WORDS ||
        n_words < HEXKL_DMA_TRACE_HDR_WORDS + n_push * pw + n_wait * ww) {
      std::fprintf(stderr, "[HTP-DMA] call=%u M=%u trace layout mismatch\n",
                   ordinal, M);
      return;
    }
    std::fprintf(stderr,
                 "[HTP-DMA] call=%u M=%u rep=%d/%d desc=%u waits=%u blocked=%u "
                 "depth_max=%u dropped=%u end=%.1fus\n",
                 ordinal, M, reps, reps, n_push, n_wait, w[2], w[3], w[4],
                 us(w[5]));
    const uint32_t *p = w + HEXKL_DMA_TRACE_HDR_WORDS;
    for (uint32_t k = 0; k < n_push; ++k, p += pw) {
      std::fprintf(stderr,
                   "[HTP-DMA] call=%u push k=%u t=%.1f kind=%s e=%u c=%u "
                   "bytes=%u row=%u nrows=%u stride=%u idx=%u depth=%u "
                   "done=%.1f..%.1f\n",
                   ordinal, k, us(p[0]), p[1] < 5 ? kKind[p[1]] : "?", p[2],
                   p[3], p[4], p[5], p[6], p[7], p[8], p[9], us(p[10]),
                   us(p[11]));
    }
    for (uint32_t k = 0; k < n_wait; ++k, p += ww) {
      std::fprintf(stderr,
                   "[HTP-DMA] call=%u wait k=%u site=%s idx=%u t=%.1f..%.1f "
                   "blocked=%s\n",
                   ordinal, k, p[3] < 5 ? kSite[p[3]] : "?", p[2], us(p[0]),
                   us(p[1]), p[4] ? "y" : "n");
    }
  }

  /** ARM-side staging: the memcpy of the activation into the rpcmem buffer
   *  and of the result back out. It sits OUTSIDE every host_us window
   *  above (those start after the copy, so that transport = host - dsp
   *  stays a FastRPC number), which meant it was invisible -- and at this
   *  model's shapes it is ~0.9 MB per expert, 64 times per layer. Counted
   *  here so "HTP host time" stops understating what the path costs. */
  void addStaging(uint64_t us, uint64_t bytes, int line) {
    std::lock_guard<std::mutex> lock(mutex_);
    staging_us_ += us;
    staging_bytes_ += bytes;
    auto &l = staging_by_line_[line];
    l[0] += us;
    l[1] += bytes;
    l[2] += 1;
  }

  ~HtpProfile() {
    if (level_ != 0)
      dump();
  }

private:
  struct Bucket {
    uint64_t calls = 0;
    uint64_t rows = 0; /**< summed M, so prefill batching is visible */
    /** MoE layer calls that took the M=1 HVX GEMV path (#80). Beside
        blocks it is the per-call proof of which path a row's numbers came
        from: m1_gemv=calls/calls with blocks=0 is the GEMV, 0/calls with
        blocks=4*calls the HMX loop. */
    uint64_t m1_calls = 0;
    /** Of those, the calls whose GEMV read VTCM slabs the DMA engine fed
        (#117): feed=calls/calls with a DMA ring: line is the feed, 0/calls
        the arena read. */
    uint64_t m1_feed_calls = 0;
    /** Expert cache misses paid before this row's calls (doc 52), and
        what they cost outside the call: the file read and the FastRPC
        register/release round trips. Both sit OUTSIDE host_us. */
    uint64_t misses = 0;
    uint64_t miss_read_us = 0;
    uint64_t miss_rpc_us = 0;
    /** Their DMA queue counts, summed (#177): dmaq= is this over
        m1_feed_calls, the queues the feed really used. */
    uint64_t m1_feed_q_sum = 0;
    uint64_t host_us = 0;
    uint64_t dsp_us = 0;
    uint64_t quant_us = 0;
    uint64_t swiglu_us = 0; /**< fused calls only; 0 elsewhere */
    /** swiglu_us is WORKER time that ran under the HMX (the MoE layer
        kernel's pool jobs, the conv block's stage units) rather than a
        synchronous pass, so the residual must not subtract it. Set by the
        call type that fills the bucket. */
    bool swiglu_hidden = false;
    /** MoE layer call only: the routing multiply and scatter-add, plus
        copying the FastRPC buffers to and from cached heap. Both are work
        no other call type does, so they get their own column rather than
        being folded into one that means something else. */
    uint64_t scatter_us = 0;
    /** MoE layer call only, and separate from scatter_us for the same
        reason scatter_us is separate from acc_us: these are the copies to
        and from the FastRPC buffers, they are the one thing the DMA change
        touches, and merged into scatter there is no reading of the profile
        that says whether it worked. */
    uint64_t stage_us = 0;
    /** MoE layer call only. quant_us used to hold all three; see
        hexkl_probe.h's HEXKL_PROBE_GATHER for why they split. */
    uint64_t gather_us = 0;
    uint64_t requant_us = 0;
    /** NOT microseconds: 64-row HMX blocks issued, summed over experts and
        calls. Printed as a count beside the times because the matmul scales
        with it -- 1776 routed rows can cost 32 blocks or 48 depending only
        on how the router spread them. */
    uint64_t blocks = 0;
    /** MoE layer call only. mm_us is the HMX issue loop measured rather
        than left as the subtraction residual; what the residual still holds
        after subtracting it is the honest "unnamed" figure. dma_kb and
        dma_first_us together give the DDR-to-VTCM weight rate. */
    uint64_t mm_us = 0;
    uint64_t dma_kb = 0;
    uint64_t dma_first_us = 0;
    /** The down drain apart from gate_up's, and the push itself. One bucket
        held both drains and the push sat unnamed in the residual; see
        hexkl_probe.h for why that left the 4.9 ms unreadable. */
    uint64_t drain_dn_us = 0;
    uint64_t dma_first_kb = 0;
    /** The layer call's own malloc and free -- about 12.8 MB a call. Was
        unnamed in the residual, which read 1019 us on the bake path and
        23888 on the weight-cache path with the kernel unchanged. */
    uint64_t alloc_us = 0;
    uint64_t push_us = 0;
    uint64_t dequant_us = 0;
    uint64_t acc_us = 0;
    uint64_t drain_us = 0;
    /** [#87] The DMA ring trace's per-call numbers, summed over calls
        (depth_max is the max). Printed on the DMA ring: line under weight
        DMA:, never folded into the columns above. */
    uint64_t dma_desc = 0;
    uint64_t dma_waits = 0;
    uint64_t dma_waits_blocked = 0;
    uint64_t dma_wait_us = 0;
    uint64_t dma_wait_act_us = 0;
    uint64_t dma_busy_lo_us = 0;
    uint64_t dma_busy_hi_us = 0;
    uint64_t dma_depth_max = 0;
    uint64_t dma_first_ready_us = 0;
    uint64_t dma_last_issue_us = 0;
    uint64_t dma_traces = 0; /**< per-descriptor dumps printed so far */
    /** [#88] MoE layer call only: the ION staging class the call rode
        (stage() in HtpComputeOps), whether both landed on ION, and the
        bytes of the non-ION in-args the stub hands the driver (the 48-byte
        primitive block plus the five routing/handle sequences). Printed
        as the staging: line so the transport column has its inventory
        beside it instead of in a plan. */
    size_t stage_act_bytes = 0;
    size_t stage_out_bytes = 0;
    bool stage_ion = true;
    uint64_t in_arg_bytes = 0;
    /** [#141] Of calls, how many rode the dspqueue; then in_arg_bytes is
        the request message, not the stub's non-ION in-args. */
    uint64_t dspq_calls = 0;
  };

  HtpProfile() {
    const char *env = std::getenv("NNTR_HTP_PROFILE");
    level_ = (env != nullptr) ? std::atoi(env) : 0;
    const char *trace_env = std::getenv("NNTR_HTP_DMA_TRACE");
    dma_trace_calls_ = (trace_env != nullptr) ? std::atoi(trace_env) : 3;
    // Captured at construction, not at static-destruction time in dump():
    // HtpProfile::global() is always reached through a HtpBackend::global()
    // call first (every accelerated entry point fetches the session handle
    // before it ever touches the profile), so construction order is safe;
    // destruction order is not something to lean on for a second singleton.
    qos_mode_ = HtpBackend::global().qosMode();
  }

  static double ms(uint64_t us) { return static_cast<double>(us) / 1000.0; }

  void dump() {
    // stderr, not the logger: this runs at static destruction, and the run
    // is driven from an adb shell where stderr is what the operator sees.
    std::fprintf(
      stderr,
      "\n[HTP-PROFILE] level=%d qos_mode=%d (2=poll 1=PM "
      "0=interrupt-driven -- every transport number below is only "
      "comparable to 34_fc_measured.md's 326us/call at qos_mode=2)\n",
      level_, qos_mode_);

    uint64_t invoke_us = 0;
    for (const auto &entry : buckets_)
      invoke_us += entry.second.host_us;

    std::fprintf(stderr,
                 "[HTP-PROFILE] weight registration (once per weight, so all "
                 "of it lands in the first forward)\n");
    std::fprintf(stderr,
                 "[HTP-PROFILE]   weights registered : %llu\n"
                 "[HTP-PROFILE]   convert to registry: %10.1f ms  (%.2f "
                 "ms/weight)\n"
                 "[HTP-PROFILE]   register FastRPC   : %10.1f ms  (%.2f "
                 "ms/weight)\n"
                 "[HTP-PROFILE]   alloc + other      : %10.1f ms\n"
                 "[HTP-PROFILE]   registration total : %10.1f ms\n"
                 "[HTP-PROFILE]   rpcmem/ION buffer  : %llu/%llu weights "
                 "(the rest used plain heap -- pinned+mapped per call)\n",
                 (unsigned long long)reg_calls_, ms(convert_us_),
                 reg_calls_ ? ms(convert_us_) / reg_calls_ : 0.0, ms(rpc_us_),
                 reg_calls_ ? ms(rpc_us_) / reg_calls_ : 0.0,
                 ms(reg_total_us_ - convert_us_ - rpc_us_), ms(reg_total_us_),
                 (unsigned long long)reg_ion_calls_,
                 (unsigned long long)reg_calls_);

    // M==1 is decode's shape, but an expert that drew a single token during
    // prefill lands in the same row -- read the two as shapes, not phases.
    std::fprintf(stderr,
                 "[HTP-PROFILE] layer calls (M==1 is decode's shape)\n");
    for (const auto &entry : buckets_) {
      const unsigned k = std::get<0>(entry.first);
      const unsigned n = std::get<1>(entry.first);
      const bool decode = std::get<2>(entry.first);
      // The kind names the row on both shapes: at decode the attention
      // FC calls and the MoE layer share K=2048 N=2048, so "M==1" alone
      // would print two rows nobody could tell apart. The MoE row stays
      // the bare "M==1" every handoff greps for.
      static const char *const kind_name[] = {"M>1", "M>1 dense", "M>1 conv",
                                              "M>1 FC"};
      static const char *const kind_name_m1[] = {"M==1", "M==1 dense",
                                                 "M==1 conv", "M==1 FC"};
      const int kind = std::get<3>(entry.first);
      const Bucket &b = entry.second;
      std::fprintf(stderr,
                   "[HTP-PROFILE]   K=%-5u N=%-5u %-10s calls=%-7llu "
                   "rows=%-8llu host=%9.1f ms (%7.1f us/call)",
                   k, n, (decode ? kind_name_m1 : kind_name)[kind],
                   (unsigned long long)b.calls, (unsigned long long)b.rows,
                   ms(b.host_us),
                   b.calls ? static_cast<double>(b.host_us) / b.calls : 0.0);
      if (level_ >= 2 && b.calls != 0) {
        const double dsp_per = static_cast<double>(b.dsp_us) / b.calls;
        const double host_per = static_cast<double>(b.host_us) / b.calls;
        const double quant_per = static_cast<double>(b.quant_us) / b.calls;
        const double swiglu_per = static_cast<double>(b.swiglu_us) / b.calls;
        const double scatter_per = static_cast<double>(b.scatter_us) / b.calls;
        const double dequant_per = static_cast<double>(b.dequant_us) / b.calls;
        const double acc_per = static_cast<double>(b.acc_us) / b.calls;
        const double drain_per = static_cast<double>(b.drain_us) / b.calls;
        const double stage_per = static_cast<double>(b.stage_us) / b.calls;
        const double gather_per = static_cast<double>(b.gather_us) / b.calls;
        const double requant_per = static_cast<double>(b.requant_us) / b.calls;
        const double mm_meas_per = static_cast<double>(b.mm_us) / b.calls;
        const double drain_dn_per =
          static_cast<double>(b.drain_dn_us) / b.calls;
        const double push_per = static_cast<double>(b.push_us) / b.calls;
        const double alloc_per = static_cast<double>(b.alloc_us) / b.calls;
        // What the accelerator actually exists for, by subtraction: the DSP
        // clock minus every stage that is a format change or a wait. Nothing
        // on the DSP times the HMX issue loop directly, and adding a probe
        // inside it would perturb what it measures -- this residue is the
        // honest number, and it is the one to compare against the layer's
        // MAC count. It also absorbs whatever the probes do not name, so
        // treat it as an upper bound on the matmul, not an exact figure.
        /** mm is the residual, so every named stage has to be subtracted --
           scatter included, or the MoE layer call's scatter time would be
           reported as matmul. Except SWIGLU on the MoE layer kernel's rows
           (MoE, dense, conv; upstream doc 53 section 8.4) and on the M=1
           GEMV path (#102): there it is WORKER time inside pool jobs
           (hexkl_mm_u8i4_moe.c's moe_worker_probe_add, hexkl_conv_block.c's
           cb_stage_probe_add) or the sum of every lane's wall time inside
           the two GEMV stages -- work that ran under the HMX or beside the
           caller rather than instead of it, not a stage on this clock, so
           subtracting it made the residual negative (3.3 ms of hidden work
           against a 4.0 ms conv call). Left in the sum for the fused layer
           call, where SWIGLU is its own synchronous elementwise pass
           (htp_moe_opts.h). */
        const bool hidden = b.swiglu_hidden || b.m1_calls != 0;
        const double mm_per = htp_moe_row_rest_us(
          dsp_per,
          quant_per + dequant_per + acc_per + drain_per + scatter_per +
            stage_per + gather_per + requant_per + mm_meas_per + drain_dn_per +
            push_per + alloc_per,
          swiglu_per, hidden);
        std::fprintf(
          stderr,
          "  dsp=%7.1f us/call (%4.1f%%) transport=%7.1f us/call"
          "  [quant %.1f gather %.1f requant %.1f swiglu%s %.1f "
          "dequant %.1f acc %.1f drain %.1f+%.1f push %.1f "
          "scatter %.1f alloc %.1f "
          "stage %.1f mm %.1f | rest<=%.1f (%.1f%% of host) "
          "blocks=%llu m1_gemv=%llu/%llu feed=%llu/%llu dmaq=%.2f]",
          dsp_per, host_per > 0.0 ? 100.0 * dsp_per / host_per : 0.0,
          host_per - dsp_per, quant_per, gather_per, requant_per,
          hidden ? "(hidden)" : "", swiglu_per, dequant_per, acc_per, drain_per,
          drain_dn_per, push_per, scatter_per, alloc_per, stage_per,
          mm_meas_per, mm_per, host_per > 0.0 ? 100.0 * mm_per / host_per : 0.0,
          (unsigned long long)b.blocks, (unsigned long long)b.m1_calls,
          (unsigned long long)b.calls, (unsigned long long)b.m1_feed_calls,
          (unsigned long long)b.calls,
          b.m1_feed_calls != 0
            ? static_cast<double>(b.m1_feed_q_sum) / b.m1_feed_calls
            : 0.0);
      }
      if (level_ >= 2 && b.calls != 0 && b.dma_first_us != 0) {
        // The first weight wait happens with an empty ring, so it times a
        // transfer rather than a pipeline. The size it covers is reported
        // alongside rather than derived from the shapes: since the weight
        // started arriving in chunks that wait covers one chunk, and
        // dividing its time by the whole weight claimed 114.7 GB/s for a
        // link that does about 33.
        const double first_us = static_cast<double>(b.dma_first_us) / b.calls;
        const double first_kb = static_cast<double>(b.dma_first_kb) / b.calls;
        const double kb = static_cast<double>(b.dma_kb) / b.calls;
        const double dsp_us = static_cast<double>(b.dsp_us) / b.calls;
        std::fprintf(stderr,
                     "\n[HTP-PROFILE]     weight DMA: %.0f KB/call, first "
                     "%.0f KB took %.0f us = %.1f GB/s; averaged over the "
                     "call %.1f GB/s",
                     kb, first_kb, first_us,
                     first_us > 0.0 ? first_kb * 1.024 / first_us : 0.0,
                     dsp_us > 0.0 ? kb * 1.024 / dsp_us : 0.0);
      } else if (level_ >= 2 && b.calls != 0 && b.m1_calls == b.calls) {
        // [#102] The GEMV path reads the weights straight from the arena and
        // never waits on the ring, so there is no first-wait rate to print;
        // this line keeps the row block's line count and says what swiglu
        // means on this path (lanes busy = swiglu / mm, 6 lanes on v79).
        // ponytail: no GB/s here -- the bucket knows K, N_out and M but not
        // inter, so the rate stays hand arithmetic (weight bytes / mm).
        // Upgrade: count DMA_KB-style bytes on the GEMV path in the kernel
        // (a skel change), or pass the bytes to addInvokeMoeLayer.
        const double mm_us = static_cast<double>(b.mm_us) / b.calls;
        std::fprintf(stderr,
                     "\n[HTP-PROFILE]     weight DMA: n/a (direct arena read "
                     "inside mm, no ring; swiglu = lane-time, %.2f lanes busy "
                     "over mm)",
                     mm_us > 0.0
                       ? static_cast<double>(b.swiglu_us) / b.calls / mm_us
                       : 0.0);
      }
      if (level_ >= 2 && b.calls != 0 && b.dma_desc != 0) {
        // [#87] How the call used the ring. busy is a bracket, not a point
        // (hexkl_dma_trace.h), so the engine rate is a range: bytes over
        // busy_hi up to bytes over busy_lo. gu / dn restate the drain
        // columns above so the three wait sites read side by side.
        const double n = static_cast<double>(b.calls);
        const double kb = static_cast<double>(b.dma_kb) / n;
        const double busy_lo = static_cast<double>(b.dma_busy_lo_us) / n;
        const double busy_hi = static_cast<double>(b.dma_busy_hi_us) / n;
        std::fprintf(
          stderr,
          "\n[HTP-PROFILE]     DMA ring: desc=%.0f/call waits=%.0f (blocked "
          "%.1f) wait=%.1f us [act %.1f gu %.1f+dn %.1f]",
          static_cast<double>(b.dma_desc) / n,
          static_cast<double>(b.dma_waits) / n,
          static_cast<double>(b.dma_waits_blocked) / n,
          static_cast<double>(b.dma_wait_us) / n,
          static_cast<double>(b.dma_wait_act_us) / n,
          static_cast<double>(b.drain_us) / n,
          static_cast<double>(b.drain_dn_us) / n);
        if (busy_hi > 0.0) {
          std::fprintf(stderr, " busy=%.0f..%.0f us -> engine %.1f..%.1f GB/s",
                       busy_lo, busy_hi, kb * 1.024 / busy_hi,
                       busy_lo > 0.0 ? kb * 1.024 / busy_lo : 0.0);
        } else {
          // The skel refuses the union when a call pushed more than the
          // trace table holds (prefill can); the counts above still stand.
          std::fprintf(stderr, " busy=n/a (trace truncated past %u pushes)",
                       static_cast<unsigned>(HEXKL_DMA_TRACE_MAX_PUSH));
        }
        std::fprintf(stderr,
                     "  depth max=%llu  first expert ready at %.0f us  last "
                     "issue at %.0f us of %.0f",
                     (unsigned long long)b.dma_depth_max,
                     static_cast<double>(b.dma_first_ready_us) / n,
                     static_cast<double>(b.dma_last_issue_us) / n,
                     static_cast<double>(b.dsp_us) / n);
      }
      if (level_ >= 2 && b.stage_act_bytes != 0) {
        // [#88] The MoE call's per-call transport inventory. The driver's
        // cache maintenance covers the whole staging buffer, so the class
        // size is the number that explains a transport figure; rpc allocs
        // is process-wide and must not grow with calls; the in-args are the
        // driver-copied bytes outside ION (prim block + 5 sequences; the
        // sixth sequence and the rout are the two staging buffers).
        std::fprintf(stderr,
                     "\n[HTP-PROFILE]     staging: act %zu B out %zu B ion=%c  "
                     "rpc allocs=%u (session)  ",
                     b.stage_act_bytes, b.stage_out_bytes,
                     b.stage_ion ? 'y' : 'n', HtpRpcBuffer::allocCount());
        if (b.dspq_calls != 0) {
          // [#141] The request message is copied into the queue's shared
          // memory; the act/out classes above are the queue's buffers.
          std::fprintf(stderr, "via=dspq %llu/%llu calls msg=%llu B",
                       (unsigned long long)b.dspq_calls,
                       (unsigned long long)b.calls,
                       (unsigned long long)b.in_arg_bytes);
        } else {
          std::fprintf(stderr, "non-ION in-args=6/%llu B",
                       (unsigned long long)b.in_arg_bytes);
        }
      }
      if (b.misses != 0 || b.miss_rpc_us != 0) {
        std::fprintf(
          stderr,
          "\n[HTP-PROFILE]     expert misses: %llu (%.2f/call), "
          "file read %.1f ms (%.2f ms/call, %.2f ms/miss), "
          "swap rpc %.1f ms (%.2f ms/call) -- outside "
          "host= above",
          (unsigned long long)b.misses, static_cast<double>(b.misses) / b.calls,
          ms(b.miss_read_us), ms(b.miss_read_us) / static_cast<double>(b.calls),
          b.misses ? ms(b.miss_read_us) / b.misses : 0.0, ms(b.miss_rpc_us),
          ms(b.miss_rpc_us) / static_cast<double>(b.calls));
      }
      std::fprintf(stderr, "\n");
    }

    if (graph_calls_ != 0) {
      const double n = static_cast<double>(graph_calls_);
      char names[128];
      std::fprintf(stderr,
                   "[HTP-PROFILE]   graph: calls=%llu ops/call=%.2f "
                   "resident=%s dsp=%.1f us/call op_pcyc=%.0f/call pcyc/op:",
                   (unsigned long long)graph_calls_,
                   static_cast<double>(graph_ops_) / n,
                   htp_graph_kinds_str(graph_resident_, names, sizeof(names)),
                   static_cast<double>(graph_dsp_us_) / n,
                   static_cast<double>(graph_op_pcyc_) / n);
      for (unsigned k = 0; k < HTP_OP_KIND_N; ++k) {
        if (graph_kind_n_[k] != 0)
          std::fprintf(stderr, " %s=%.0f", htp_graph_kind_name(k),
                       static_cast<double>(graph_kind_pcyc_[k]) /
                         static_cast<double>(graph_kind_n_[k]));
      }
      std::fprintf(stderr, "\n");
    }
    std::fprintf(
      stderr,
      "[HTP-PROFILE] layer calls total : %10.1f ms\n"
      "[HTP-PROFILE] arm staging memcpy: %10.1f ms  (%.1f MB in+out, "
      "%.1f GB/s) -- outside every host= above\n"
      "[HTP-PROFILE] HTP host time     : %10.1f ms "
      "(registration + layer calls + staging)\n\n",
      ms(invoke_us), ms(staging_us_),
      static_cast<double>(staging_bytes_) / (1024.0 * 1024.0),
      staging_us_ ? static_cast<double>(staging_bytes_) / staging_us_ / 1000.0
                  : 0.0,
      ms(reg_total_us_ + invoke_us + staging_us_));
    for (const auto &l : staging_by_line_)
      if (l.second[0] >= 1000)
        std::fprintf(stderr,
                     "[HTP-PROFILE]   staging at htp_compute_ops.cpp:%d: "
                     "%8.1f ms  %8.1f MB  %llu copies\n",
                     l.first, ms(l.second[0]),
                     static_cast<double>(l.second[1]) / (1024.0 * 1024.0),
                     (unsigned long long)l.second[2]);
    if (miss_total_ != 0) {
      std::fprintf(stderr,
                   "[HTP-PROFILE] expert cache misses: %llu, file read "
                   "%.1f ms (%.2f ms/miss), swap rpc %.1f ms "
                   "(%.2f ms/miss)\n\n",
                   (unsigned long long)miss_total_, ms(miss_read_total_us_),
                   ms(miss_read_total_us_) / miss_total_,
                   ms(miss_rpc_total_us_),
                   ms(miss_rpc_total_us_) / miss_total_);
    }
    if (prefetch_n_ != 0) {
      std::fprintf(stderr,
                   "[HTP-PROFILE] expert prefetch: %llu experts read under "
                   "the layer calls, exposed wait %.1f ms (%.2f ms/expert), "
                   "register rpc %.1f ms (%.2f ms/expert)\n",
                   (unsigned long long)prefetch_n_, ms(prefetch_wait_us_),
                   ms(prefetch_wait_us_) / prefetch_n_, ms(prefetch_rpc_us_),
                   ms(prefetch_rpc_us_) / prefetch_n_);
      // [doc 52 sections 10.18, 10.20] How far ahead the readers were: the
      // experts already read when their layer asked for them, and the
      // cores each side ran on.
      auto cpus = [](uint32_t m) {
        std::string out;
        for (int c = 0; c < 32; ++c)
          if (m & (1u << c))
            out += (out.empty() ? "" : ",") + std::to_string(c);
        return out.empty() ? std::string("?") : out;
      };
      std::fprintf(stderr,
                   "[HTP-PROFILE] expert prefetch progress: %llu of %llu read "
                   "when their layer asked (%.0f%%), caller cpus {%s}, reader "
                   "cpus {%s}\n\n",
                   (unsigned long long)prefetch_done_on_entry_,
                   (unsigned long long)prefetch_n_,
                   100.0 * prefetch_done_on_entry_ / prefetch_n_,
                   cpus(prefetch_caller_cpus_).c_str(),
                   cpus(prefetch_reader_cpus_).c_str());
      // [doc 52 section 10.30] The per-expert read on a reader thread: page
      // cache 1.4 ms with 2 readers and 2.1 with 4 (they share the write
      // bandwidth), flash several times that.
      if (prefetch_reads_ != 0)
        std::fprintf(stderr,
                     "[HTP-PROFILE] expert prefetch readers: %.2f ms/expert on "
                     "a reader, %llu experts\n\n",
                     ms(prefetch_read_us_) / prefetch_reads_,
                     (unsigned long long)prefetch_reads_);
    }
  }

  int level_ = 0;
  int dma_trace_calls_ = 3;
  int qos_mode_ = 0;
  std::mutex mutex_;
  uint64_t reg_calls_ = 0;
  uint64_t reg_ion_calls_ = 0;
  uint64_t reg_total_us_ = 0;
  uint64_t staging_us_ = 0;
  uint64_t staging_bytes_ = 0;
  /** us, bytes, copies by the caller's line of stagedMemcpy. */
  std::map<int, std::array<uint64_t, 3>> staging_by_line_;
  uint64_t convert_us_ = 0;
  uint64_t rpc_us_ = 0;
  uint64_t graph_calls_ = 0; /**< [#85] addInvokeForward */
  uint64_t graph_ops_ = 0;
  uint64_t graph_dsp_us_ = 0;
  uint64_t graph_op_pcyc_ = 0;
  uint32_t graph_resident_ = 0; /**< [#130] the description's mask */
  uint64_t graph_kind_n_[HTP_OP_KIND_N] = {0};
  uint64_t graph_kind_pcyc_[HTP_OP_KIND_N] = {0};
  uint64_t miss_pending_ = 0, miss_read_pending_us_ = 0,
           miss_rpc_pending_us_ = 0;
  uint64_t miss_total_ = 0, miss_read_total_us_ = 0, miss_rpc_total_us_ = 0;
  uint64_t prefetch_n_ = 0, prefetch_wait_us_ = 0, prefetch_rpc_us_ = 0;
  uint64_t prefetch_done_on_entry_ = 0;
  uint64_t prefetch_read_us_ = 0, prefetch_reads_ = 0;
  uint32_t prefetch_caller_cpus_ = 0, prefetch_reader_cpus_ = 0;
  /** (K, N, M == 1, kind): kind 0 is every layer call, 1 the dense FFN
   *  through the MoE layer kernel (doc 51). */
  std::map<std::tuple<unsigned, unsigned, bool, int>, Bucket> buckets_;
};

/**
 * @brief memcpy that charges itself to HtpProfile's staging line.
 *
 * Every host_us window in this file starts AFTER the activation is copied
 * into the rpcmem buffer and ends BEFORE the result is copied back out, on
 * purpose: that is what keeps `transport = host - dsp` a FastRPC number
 * rather than a FastRPC-plus-memcpy number. The consequence was that the
 * copies appeared nowhere at all, and on this model's MoE they are not
 * small -- ~450 KB in and ~450 KB out per expert call, 64 calls per layer.
 * Same copy, one accumulator, so the profile's bottom line stops
 * understating what the path costs.
 */
/**
 * @brief NNTR_L2_CHECK: does the DSP still hand back non-finite values?
 *
 * The two L2 failures in docs/htp_attention/43_moe_ffn_measured_next_
 * levers.md section 7 were diagnosed as hvx_recip_qf32 returning NaN for
 * any gate at or below hvx_swiglu_row_f32's exp clamp. That diagnosis is
 * only worth what a device can confirm, and the model's own output cannot
 * confirm it: wrong text is equally consistent with a dozen other causes.
 * This is the discriminator. If it counts zero and the text is still
 * wrong, the SwiGLU clamp is NOT the (whole) bug and the search moves
 * elsewhere; if it counts non-zero, whatever skel is on the device does
 * not have the fix in it.
 *
 * Cheap where it matters most: a NaN SwiGLU lane lands in its row's
 * requantization scale (hvx_quant_rows_u8_params scans the whole row for
 * min/max), so scanning m_pad floats -- 64 of them -- catches it before
 * the value has spread anywhere. Off unless the env var is set.
 */
inline bool l2CheckEnabled() {
  static const bool on = std::getenv("NNTR_L2_CHECK") != nullptr;
  return on;
}

/** @brief NNTR_L2_DIFF: run both MoE FFN paths and report their SNR. See
 *         HtpComputeOps::l2Diff for what the number separates. */
inline bool l2DiffEnabled() {
  static const bool on = std::getenv("NNTR_L2_DIFF") != nullptr;
  return on;
}

/**
 * @brief NNTR_L2_SHADOW: run the fused path, then hand the model the
 *        REFERENCE result instead of the fused one.
 *
 * Splits "the fused path computes wrong values" from "the fused path has a
 * side effect" -- the two remaining stories, and no SNR number can tell
 * them apart. Under this flag both paths execute in full, touching every
 * buffer, taking every lock, leaving every piece of state exactly as a
 * normal fused run would; only the floats the layer goes on to use are
 * swapped for the reference's.
 *
 *   text becomes correct -> the values are the fault, and the 67-80 dB
 *                           calls are worth chasing;
 *   text still broken    -> the values are NOT the fault (they match the
 *                           working path at 142 dB on 84% of calls
 *                           anyway), and the bug is a side effect --
 *                           aliasing, a clobbered workspace, retained
 *                           state -- that differencing output values can
 *                           never surface.
 *
 * Implies NNTR_L2_DIFF: the reference it hands over is the one l2Diff
 * already computes.
 */
inline bool l2ShadowEnabled() {
  static const bool on = std::getenv("NNTR_L2_SHADOW") != nullptr;
  return on;
}

/** @brief Reports the first non-finite element of @a v, once per call. */
inline void l2CheckFinite(const char *what, const float *v, size_t n,
                          unsigned M, unsigned K, unsigned N) {
  size_t bad = 0;
  size_t first = 0;
  for (size_t i = 0; i < n; ++i) {
    if (!std::isfinite(v[i])) {
      if (bad == 0)
        first = i;
      ++bad;
    }
  }
  if (bad != 0) {
    std::fprintf(stderr,
                 "[L2-CHECK] %s: %llu/%llu non-finite (first idx %llu = %g) "
                 "M=%u K=%u N=%u\n",
                 what, (unsigned long long)bad, (unsigned long long)n,
                 (unsigned long long)first, static_cast<double>(v[first]), M, K,
                 N);
  }
}

inline void stagedMemcpy(void *dst, const void *src, size_t bytes,
                         int line = __builtin_LINE()) {
  HtpProfile &p = HtpProfile::global();
  if (p.level() == 0) {
    std::memcpy(dst, src, bytes);
    return;
  }
  const uint64_t t0 = HtpProfile::nowUs();
  std::memcpy(dst, src, bytes);
  p.addStaging(HtpProfile::nowUs() - t0, bytes, line);
}

} // namespace

// --- nntrainer/nntrainer#4343: fp16 and quantized KV-cache attention ---
// Not in the in-process host twin: see the ops at the end of HtpComputeOps.
#ifndef NNTR_HTP_INPROC
namespace {

/**
 * @brief NNTR_HTP_ATTN_TRACE=1 logs every attention call with its host-side
 *        wall time and the skel's stats, so a model run shows where a step
 *        spends its time: in the kernel, or around the FastRPC call.
 */
bool attn_trace_enabled() {
  static const bool on = [] {
    const char *e = std::getenv("NNTR_HTP_ATTN_TRACE");
    return e && *e && *e != '0';
  }();
  return on;
}

int64_t now_us() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

/**
 * @brief llama.cpp's HMX-eligibility rule, inverted: fewer than five query
 *        rows with head_dim <= 128 are better served by HVX than by
 *        padding one or two rows into a 32-row tile.
 */
constexpr unsigned int kDecodeMaxRows = 5;
constexpr unsigned int kDecodeMaxHeadDim = 128;

} // namespace
#endif /* NNTR_HTP_INPROC */

/** [plan 201 S4] set_moe_geglu's flag (HTP_MOE_FLAG_GEGLU) and whether
 *  sendMoeOptsOnce has sent the word it goes into. */
static std::atomic<bool> &moeGeglu() {
  static std::atomic<bool> on{false};
  return on;
}
static std::atomic<bool> &moeOptsSent() {
  static std::atomic<bool> sent{false};
  return sent;
}

class HtpComputeOps : public CpuComputeOps {
public:
  /** [#130] The per-token entry's call count at close, the one line the
   *  host E2E harness reads (plan 130 section 3.5): tokens are the calls
   *  that started at the list's first resident op. No RPC here: the
   *  session may already be closed. The expert prefetch readers (doc 52)
   *  are stopped first. */
  void stopWorkers() {
    if (pool_srv_) { // [plan 201 S1] idle between tokens: nothing in flight
      {
        std::lock_guard<std::mutex> lock(pool_srv_->mu);
        pool_srv_->stop = true;
      }
      pool_srv_->cv.notify_all();
      if (pool_srv_->th.joinable())
        pool_srv_->th.join();
    }
    if (cpu_predictor_)
      cpu_predictor_->drain();
    if (pool_prefetch_)
      pool_prefetch_->drain();
    cpu_predictor_.reset();
    pool_prefetch_.reset();
    demand_read_pool_.reset();
    {
      std::lock_guard<std::mutex> lock(prefetch_mutex_);
      prefetch_stop_ = true;
    }
    prefetch_cv_.notify_all();
    for (std::thread &t : prefetch_readers_)
      if (t.joinable())
        t.join();
    {
      std::lock_guard<std::mutex> lock(pf_mu_);
      pf_stop_ = true;
      pf_q_.clear();
    }
    pf_cv_.notify_all();
    for (auto &t : pf_threads_)
      if (t.joinable())
        t.join();
    if (advice_) {
      {
        std::lock_guard<std::mutex> lock(advice_->mu);
        advice_->stop = true;
      }
      advice_->cv.notify_all();
      if (advice_->worker.joinable())
        advice_->worker.join();
    }
  }

  ~HtpComputeOps() override {
    stopWorkers();
    if (fcwh_fd_ >= 0)
      ::close(fcwh_fd_);
    for (const auto &r : pin_ranges_)
      ::munmap(const_cast<uint8_t *>(r.p), r.len);
    if (fwd_calls_ != 0) {
      std::fprintf(stderr,
                   "[HTP] graph: forward calls=%llu tokens=%llu "
                   "calls/token=%.2f\n",
                   (unsigned long long)fwd_calls_,
                   (unsigned long long)fwd_tokens_,
                   fwd_tokens_ ? static_cast<double>(fwd_calls_) /
                                   static_cast<double>(fwd_tokens_)
                               : 0.0);
    }
    if (cpu_fc_skipped_ != 0) // [#132 Part B] the harness's second line
      std::fprintf(stderr, "[HTP] graph: cpu fc skipped=%llu\n",
                   (unsigned long long)cpu_fc_skipped_);
  }

  /** Start an independent prompt; the caller has joined its generation. */
  void beginPrompt() {
    if (cpu_predictor_)
      cpu_predictor_->drain();
    if (pool_prefetch_)
      pool_prefetch_->drain();
    if (attn_registered_) {
      const int rc = nntr_hvx_attn_m1_release(HtpBackend::global().handle());
      if (rc != AEE_SUCCESS)
        throw std::runtime_error("HTP prompt KV release failed: " +
                                 graphErr(rc));
      attn_registered_ = false;
    }
    std::fill(kv_len_.begin(), kv_len_.end(), 0u);
    // A conv op's DSP state belongs to the last prompt: re-seed it at the
    // next prompt's first decode row even if that row number matches.
    std::fill(conv_next_pos_.begin(), conv_next_pos_.end(), HTP_GRAPH_NO_OP);
    std::fill(std::begin(kind_next_), std::end(kind_next_), 0u);
    cur_pos_ = HTP_GRAPH_NO_OP;
    pending_op_ = HTP_GRAPH_NO_OP;
    seed_ordinal_ = HTP_GRAPH_NO_OP;
    row_bound_ = false;
    have_id_ = false;
    // Prefill staging may have been freed during decode. Tensor addresses
    // are reused by the next prompt, so no old ION alias may survive it.
    ion_alias_ = {};
    ion_alias_next_ = 0;
    attn_out16_ = {};
    for (auto *pool : {&act_pool_, &out_pool_, &rope_pool_, &x2_pool_,
                       &attn_q_pool_, &attn_out_pool_}) {
      pool->holds.clear();
      pool->copied_out.clear();
    }
    pool_dirty_ = true;
    t1_printed_ = false;
    prompt_tokens_ = 0;
    seed_rpc_us_ = seed_calls_ = seed_kvq_ = 0;
  }

  bool supports_gemm_q4_0_accel_fp32() const override { return true; }

  // ponytail: decode-shaped (M == 1) calls are declined for the MoE FFN,
  // not accelerated -- reversing 34_fc_measured.md section5.1's own
  // conclusion for the isolated single-FC comparison it was measured on.
  // That comparison is still correct in isolation (M=1 padding tax ~40us
  // against 113us of DSP-only work, so one call is cheap) -- what it does
  // not cover is 22 MoE layers' worth of M==1 calls stacked into one
  // decode token. Device-measured on this branch: gate_up decode's DMA is
  // fully exposed at M=1 (drain 419.2us for a 14.7MB weight group = 35
  // GB/s, no different from one handle -- cross-matmul prefetch has
  // nothing to hide behind at this width, unlike the FC benchmark's
  // narrower weights), so the real per-layer cost is ~2.6ms/token, and
  // 22 layers' worth is a ~57ms/token transport floor alone (17.5 TPS
  // ceiling) against a CPU decode baseline measured at ~25.9 TPS on the
  // same device. HTP cannot win decode until the call is fused across
  // projections or the DMA-exposure problem above is fixed -- ceiling:
  // revisit if 41_moe_ffn_e2e_and_perf_task.md's P1.3 (fused gate_up+
  // swiglu+down, one call per layer) or P2 (u8 in/out, cuts payload 4x)
  // lands, since either changes this arithmetic. Until then declining M=1
  // costs nothing on the FC path this predicate was written for (single
  // dot(), not a MoE stack) and saves the one MoE FFN caller from a
  // documented loss.
  bool accelerates_q4_0_at_m1() const override { return false; }
  // QS4CX weights live on the DSP in their own format and nowhere else
  // once the FCs are quantized to it offline (doc 57 section 5): decode's
  // row goes to the same kernel as prefill, 64-row pad tax and all (one
  // call per FC, a few ms a token), rather than to a CPU kernel on another
  // packing of the same bytes. [#260] Except under the one-PD token
  // (NNTR_HTP_E2E=1): a decode row is the token's, and the layers' hooks
  // that hand it over (qkv's input norm first) sit on their row path.
  // ponytail: process-wide, so a decode row the token does not take (a
  // moe_htp_layers subset) runs its QS4CX FCs on the CPU kernel; a
  // per-row answer needs the hook's verdict before qkv's fused call.
  bool accelerates_qs4cx_at_m1() const override {
    return !HtpBackend::e2eRequested();
  }

  /** @brief The handles one conv block is registered as: in_proj's column
   *  thirds a, b, c (get_or_register_fc's slices at this model's K) and
   *  out_proj. Declared up here because invokeConvBlock takes it by
   *  reference -- a parameter type has to be complete where the function
   *  is declared. */
  struct ConvHandles {
    std::vector<uint32_t> h_in; /**< a, b, c */
    uint32_t h_out = 0;
  };

  // matAdata: Q4_0x4-repacked weight bytes, identity-cached across calls --
  // the same pointer for the lifetime of a loaded model (inference does not
  // move weight tensors), which is what makes registering once and keying
  // the HexKL handle off it safe.
  void gemm_q4_0_accel_fp32(void *matAdata, float *matBdata, float *matCdata,
                            unsigned int M, unsigned int N,
                            unsigned int K) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    // One handle for a weight the layer kernel's VTCM double buffer holds,
    // several column slices for a wider one (doc 50 section 3): the kernel
    // takes them as one call either way, prefetching each slice behind the
    // previous one's matmul, and copyOut puts the slices' output blocks
    // back into the caller's [M x N].
    const FcHandles &fh = get_or_register_fc(matAdata, session, K, N);

    // ponytail: NOT invokeLayerU8In. Quantizing the activation on ARM/NEON
    // moved that work off HVX (already vectorized, already writes straight
    // into VTCM for HMX to read -- no DRAM detour ever existed for this
    // step) onto ARM scalar/NEON code paying its own cost outside every
    // number this file's profiler measures. Device-measured: DSP time did
    // drop 19-26%, but wall-clock barely moved because the ARM-side
    // quantize (300-420us unvectorized, ~200us with a magic-number round)
    // ate the transport savings it was supposed to create. Reverted to
    // the HVX-quantizes-into-VTCM path. invokeLayerU8In/htp_act_quant.*
    // stay in the tree, unused, in case a caller that is not this one
    // ever legitimately arrives with pre-quantized bytes already in hand.
    //
    // In row chunks when the activation would not fit VTCM beside the
    // slice double buffer (fcMaxRows): the dense FFN's down projection is
    // K = 7168, 3.1 MiB of u8 rows at M = 444 and 7 at M = 1024. Each
    // chunk is a whole call, so a chunk costs the FastRPC fixed ~0.4 ms;
    // one call covers every K = 2048 weight up to 1,920 rows.
    invokeFc(session, fh, matBdata, matCdata, M, N, K);
  }

  /** @brief Rows one layer call may carry at this K.
   *
   * hexkl_mm_u8i4_layer_run holds the whole activation (m_pad x K u8) in
   * VTCM next to the widest handle's double buffer -- 2 x 2 MiB once
   * fcSliceCols has sized the handles -- a result tile and the HMX config.
   * Budgeted at 7.75 MiB of the 8 MiB VTCM, so 3.75 MiB of rows: 1,920 at
   * K = 2048, 512 at K = 7168. Whole 64-row blocks, at least one.
   * ponytail: 7.75 MiB stands in for the session's real vtcm_size less
   * hexkl_micro_hmx_config_size(), which the host does not see; a kernel
   * that walked 64-row blocks like hexkl_mm_u8i4_moe.c would need no cap. */
  static unsigned int fcMaxRows(unsigned int K) {
    constexpr size_t kVtcmBudget = (size_t(8) << 20) - (size_t(256) << 10);
    // the widest handle's double buffer: fcSliceCols's slice, int4
    const size_t slice_double = size_t(2) * fcSliceCols(K) * K / 2;
    unsigned int rows =
      static_cast<unsigned int>((kVtcmBudget - slice_double) / K);
    rows -= rows % 64u;
    return rows < 64u ? 64u : rows;
  }

  // Several Q4_0 weights that share ONE activation -- LFM2-MoE decode's
  // top-K selected experts' gate_up projections, all against the single
  // token just routed -- go out as ONE FastRPC call across their handles
  // instead of one call per weight. hexkl_mm_u8i4_layer_run's own doc
  // (test/htp/nntr_hvx.idl mm_u8i4_layer) says what that call buys beyond
  // saving round trips: it double-buffers each handle's weight into VTCM
  // with the next handle prefetched while the current one computes, which
  // only happens with more than one handle in the call -- measured 1.7-2x
  // over one-call-per-weight (docs/htp_attention/34_fc_measured.md
  // section4 items C and E). Nothing here changes CPU: FloatTensor::dot's
  // vector overload falls back to the same per-weight loop this replaces
  // when supports_gemm_q4_0_batch_fp32() is false, so this is additive.
  //
  // NOTE: this override was written and syntax-checked in an earlier
  // session but never actually landed in a commit -- Lfm2MoELayer's
  // caller-side grouping shipped without it, so every "grouped" decode
  // call silently fell through Tensor::dot's un-accelerated per-weight
  // loop (ComputeOps::gemm_q4_0_fp32, plain CPU dequant+GEMM) instead of
  // reaching HTP at all. Confirmed by NNTR_HTP_PROFILE=2 output showing
  // zero calls of any shape for gate_up at M==1 after the caller-side
  // change landed. This commit is that missing override.
  bool supports_gemm_q4_0_batch_fp32() const override { return true; }

  void gemm_q4_0_batch_fp32(std::vector<void *> matAdata, float *matBdata,
                            std::vector<float *> matCdata, unsigned int M,
                            std::vector<unsigned int> N,
                            unsigned int K) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const size_t n = matAdata.size();
    std::vector<uint32_t> handles(n);
    unsigned int n_total = 0;
    for (size_t i = 0; i < n; ++i) {
      handles[i] = get_or_register(matAdata[i], session, K, N[i]);
      n_total += N[i];
    }

    // mm_u8i4_layer writes one contiguous [M x N_i] block per handle, in
    // call order, and copyOut hands each weight its block straight out of
    // the staging buffer. Two callers: decode's grouped expert gate_ups at
    // M == 1, and the attention q/k/v projections at prefill (qkv_layer,
    // doc 51 section 2.21) -- 444 rows, where a heap copy in between
    // would be 5 MB a call. Row chunks as gemm_q4_0_accel_fp32 makes them,
    // for the same VTCM reason.
    const unsigned int step = fcMaxRows(K);
    for (unsigned int m0 = 0; m0 < M; m0 += step) {
      const unsigned int m = std::min(step, M - m0);
      std::vector<float *> dsts(n);
      for (size_t i = 0; i < n; ++i)
        dsts[i] = matCdata[i] + static_cast<size_t>(m0) * N[i];
      invokeLayer(session, handles.data(), static_cast<int>(n),
                  matBdata + static_cast<size_t>(m0) * K, nullptr, m, n_total,
                  K, &N, &dsts);
    }
  }

  bool supports_gemm_q4_0_batch_norm_fp32() const override { return true; }

  void gemm_q4_0_batch_norm_fp32(
    std::vector<void *> matAdata, std::vector<float *> matAscale,
    float *matBdata, std::vector<float *> matCdata, unsigned int M,
    std::vector<unsigned int> N, unsigned int K, const float *pre_gamma,
    const std::vector<unsigned int> &post_chunk, const float *post_gamma,
    float eps, const float *rope_cs, unsigned int rope_hd,
    unsigned int rope_weights, bool out_to_host) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const size_t n = matAdata.size();
    if ((!post_chunk.empty() && post_chunk.size() != n) ||
        (!matAscale.empty() && matAscale.size() != n) ||
        (rope_hd != 0 && (rope_weights > n || rope_cs == nullptr))) {
      throw std::invalid_argument(
        "gemm_q4_0_batch_norm_fp32: one post_chunk / scale per weight, "
        "rope on a prefix of them");
    }
    // Every weight as the column slices gemm_q4_0_accel_fp32 would make
    // of it (a 2816 x 4096 weight does not fit the VTCM double buffer
    // whole), all of them one call: handle by handle, the slice's chunk
    // and its weight's gamma again, and where the slice's rows land.
    std::vector<uint32_t> handles;
    std::vector<unsigned int> chunks;
    std::vector<float> gammas;
    std::vector<OutSlice> slices;
    unsigned int n_total = 0;
    unsigned int rope_slices = 0;
    const float *g = post_gamma;
    for (size_t i = 0; i < n; ++i) {
      const FcHandles &fh =
        get_or_register_fc(matAdata[i], session, K, N[i],
                           matAscale.empty() ? nullptr : matAscale[i]);
      const unsigned int chunk = post_chunk.empty() ? 0u : post_chunk[i];
      unsigned int c0 = 0;
      for (size_t j = 0; j < fh.handles.size(); ++j) {
        const unsigned int cols = fh.cols[j];
        if ((chunk != 0u && cols % chunk != 0u) ||
            (rope_hd != 0u && i < rope_weights && cols % rope_hd != 0u)) {
          throw std::invalid_argument(
            "gemm_q4_0_batch_norm_fp32: a slice of " + std::to_string(cols) +
            " columns does not hold whole chunks of " + std::to_string(chunk) +
            " / heads of " + std::to_string(rope_hd));
        }
        if (rope_hd != 0u && i < rope_weights)
          ++rope_slices;
        handles.push_back(fh.handles[j]);
        chunks.push_back(chunk);
        if (chunk != 0u)
          gammas.insert(gammas.end(), g, g + chunk);
        slices.push_back({matCdata[i], N[i], c0, cols});
        c0 += cols;
        n_total += cols;
      }
      if (chunk != 0u)
        g += chunk;
    }
    FusedNorms norms;
    norms.pre_gamma = pre_gamma;
    norms.post_chunk = post_chunk.empty() ? nullptr : &chunks;
    norms.post_gamma = gammas.data();
    norms.eps = eps;
    norms.rope_hd = rope_hd;
    norms.rope_handles = rope_slices;
    // One call whatever M is: the skel walks the rows VTCM cannot hold at
    // once in chunks of its own (fc_rows_a_call), so the activation and
    // the outputs cross once, and the outputs may stay in that one buffer
    // (the reader takes them by host_view).
    // NNTR_HTP_QKV_TO_HOST=1: the qkv outputs copied to the host whole (a
    // reference for the kept path; never set in a tok/s run)
    static const bool qkv_to_host = [] {
      const char *v = std::getenv("NNTR_HTP_QKV_TO_HOST");
      return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    norms.to_host =
      out_to_host || qkv_to_host || HtpProfile::global().level() >= 2;
    // The table rows through ION like the activation: a plain host pointer
    // is pinned and mapped by the driver every call (~155 MB/s, doc 34
    // section 4 item F), 1-2 MB a prefill layer here. [P11] The table is
    // one per (head_dim, theta), shared by the layers (qkv_layer's
    // ropeTable), so the rows are the same pointer and bytes layer after
    // layer: stageInput keeps the copy and skips the 4-8 MiB memcpy, the
    // sliding and the global layers' tables in different size classes.
    norms.rope_cs = nullptr;
    if (rope_hd) {
      const size_t bytes = static_cast<size_t>(M) * 2 * rope_hd * sizeof(float);
      norms.rope_cs = reinterpret_cast<const float *>(
        stageInput(rope_pool_, rope_cs, bytes).data());
    }
    invokeLayer(session, handles.data(), static_cast<int>(handles.size()),
                matBdata, nullptr, M, n_total, K, nullptr, nullptr, &norms,
                &slices);
  }

  // The block epilogue (see the base declaration): the residual rides the
  // out staging buffer in and the result out, the addends the act buffer
  // (x then x2), gamma as a plain host pointer like the fused norms'.
  bool supports_rmsnorm_add_fp32() const override { return true; }

  void rmsnorm_add_fp32(unsigned int M, unsigned int N, const float *resid,
                        const float *x, const float *x2, const float *gamma,
                        float eps, float scale, float *out) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const size_t rows = static_cast<size_t>(M) * N;
    const size_t x_len = x2 ? 2 * rows : rows;
    std::lock_guard<std::mutex> lock(invoke_mutex_);
    float *act =
      reinterpret_cast<float *>(stage(act_pool_, x_len * sizeof(float)).data());
    float *res =
      reinterpret_cast<float *>(stage(out_pool_, rows * sizeof(float)).data());
    stagedMemcpy(act, x, rows * sizeof(float));
    if (x2)
      stagedMemcpy(act + rows, x2, rows * sizeof(float));
    stagedMemcpy(res, resid, rows * sizeof(float));
    const uint64_t t0 = HtpProfile::nowUs();
    const int err = nntr_hvx_rmsnorm_add_f32(
      session, M, N, eps, scale, gamma, gamma ? static_cast<int>(N) : 0, act,
      static_cast<int>(x_len), res, static_cast<int>(rows));
    const uint64_t elapsed = HtpProfile::nowUs() - t0;
    if (err != AEE_SUCCESS) {
      throw std::runtime_error(
        "nntr_hvx_rmsnorm_add_f32 failed: err=" + std::to_string(err) +
        " (M=" + std::to_string(M) + " N=" + std::to_string(N) + ")");
    }
    stagedMemcpy(out, res, rows * sizeof(float));
    HtpProfile &profile = HtpProfile::global();
    if (profile.level())
      profile.addInvoke(M, x2 ? 2 * N : N, N, elapsed, nullptr);
  }

  // The router logits (see the base declaration): the rows through the
  // act pool, the logits through the out pool, gamma as a plain host
  // pointer, and the K x E f32 gate weight (1.4 MB at 2816 x 128) from an
  // ION copy made on its first call and kept by the weight's address --
  // as a plain pointer it was pinned and mapped every call (doc 57
  // section 9.10: 9.7 ms a call against the CPU dot's 6).
  void release_prefill_scratch() override {
    std::lock_guard<std::mutex> lock(invoke_mutex_);
    // KEEP_OUT can leave only sampled windows in the host tensor. Publish
    // its complete value while the caller still owns the prefill tensors.
    for (StagingPool *pool : {&act_pool_, &out_pool_, &rope_pool_, &x2_pool_,
                              &attn_q_pool_, &attn_out_pool_}) {
      for (auto *records : {&pool->holds, &pool->copied_out}) {
        for (const auto &record : *records) {
          const Held &held = record.second;
          auto slot = pool->by_class.find(record.first);
          if (held.kept && slot != pool->by_class.end() && slot->second &&
              sampledSame(held.first, slot->second->data(), held.second))
            stagedMemcpy(const_cast<void *>(held.first), slot->second->data(),
                         held.second);
        }
      }
    }
    if (attn_out16_.host) {
      const auto &a = attn_out16_;
      float *dst = const_cast<float *>(a.host);
      for (unsigned r = 0; r < a.rows; ++r)
        for (unsigned h = 0; h < a.width / a.head_dim; ++h) {
          const size_t base = static_cast<size_t>(r) * a.width + h * a.head_dim;
          for (unsigned i = 0; i < a.head_dim; ++i)
            dst[base + i] =
              (static_cast<float>(a.ion[base + i]) - a.enc[2 * h + 1]) *
              a.enc[2 * h];
        }
    }
    releasePrefillStaging();
  }

  bool supports_router_logits_fp32() const override { return true; }

  const float *routerWeightShared(const float *w, size_t bytes) {
    auto it = router_w_.find(w);
    if (it == router_w_.end()) {
      auto buf = std::make_unique<HtpRpcBuffer>(bytes);
      std::memcpy(buf->data(), w, bytes);
      it = router_w_.emplace(w, std::move(buf)).first;
    }
    return reinterpret_cast<const float *>(it->second->data());
  }

  /** @brief The gate weight (K x E f32) as a u8 x i8 HMX weight, built and
   *  registered on first use and kept by the weight's address: symmetric
   *  int8 per expert column (scale = max |w| / 127), its column sums, no
   *  bias. UINT32_MAX when the backend refuses (the f32 kernel then runs). */
  uint32_t routerWeightU8i8(remote_handle64 session, const float *w,
                            unsigned int K, unsigned int E) {
    auto it = router_u8i8_.find(w);
    if (it != router_u8i8_.end())
      return it->second;
    std::vector<int8_t> q(static_cast<size_t>(K) * E);
    std::vector<float> sc(E), bias(E, 0.0f);
    std::vector<int32_t> colsum(E, 0);
    for (unsigned int e = 0; e < E; ++e) {
      float amax = 0.0f;
      for (unsigned int k = 0; k < K; ++k)
        amax = std::max(amax, std::fabs(w[static_cast<size_t>(k) * E + e]));
      sc[e] = amax > 0.0f ? amax / 127.0f : 1.0f;
      const float inv = 1.0f / sc[e];
      for (unsigned int k = 0; k < K; ++k) {
        const long v = std::lround(w[static_cast<size_t>(k) * E + e] * inv);
        const int8_t qi =
          static_cast<int8_t>(std::max(-127L, std::min(127L, v)));
        q[static_cast<size_t>(k) * E + e] = qi;
        colsum[e] += qi;
      }
    }
    uint32_t h = UINT32_MAX;
    const int err = nntr_hvx_weight_register_u8i8(
      session, K, E, q.data(), static_cast<int>(q.size()), sc.data(),
      static_cast<int>(E), colsum.data(), static_cast<int>(E), bias.data(),
      static_cast<int>(E), &h);
    if (err != AEE_SUCCESS) {
      std::fprintf(stderr,
                   "[HTP] router stays on the f32 kernel: "
                   "weight_register_u8i8 err=%d\n",
                   err);
      h = UINT32_MAX;
    }
    router_u8i8_[w] = h;
    return h;
  }

  void router_logits_fp32(unsigned int M, unsigned int K, unsigned int E,
                          const float *x, const float *gamma, float eps,
                          const float *w, float *logits, unsigned int top_k,
                          unsigned int n_sel, const float *scale,
                          unsigned int *sel, float *weight) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const size_t x_len = static_cast<size_t>(M) * K;
    const size_t out_len = static_cast<size_t>(M) * E;
    std::lock_guard<std::mutex> lock(invoke_mutex_);
    float *act = reinterpret_cast<float *>(
      stageInput(act_pool_, x, x_len * sizeof(float)).data());
    float *out = reinterpret_cast<float *>(
      stage(out_pool_, out_len * sizeof(float)).data());
    // NNTR_HTP_ROUTER_U8I8=1: the logits on the HMX (u8 x i8). Off by
    // default: on device its logits were up to 0.17-0.39 off the CPU's
    // (the f32 kernel: 2e-5), the per-row u8 step being coarse beside
    // Gemma-4's outlier activations, and the routing that noise changes
    // is not a model's own (doc 57 section 9.33).
    static const bool router_u8i8 =
      std::getenv("NNTR_HTP_ROUTER_U8I8") != nullptr;
    const uint32_t h8 =
      router_u8i8 ? routerWeightU8i8(session, w, K, E) : UINT32_MAX;
    const float *w_shared =
      h8 != UINT32_MAX
        ? nullptr
        : routerWeightShared(w, static_cast<size_t>(K) * E * sizeof(float));
    const uint64_t t0 = HtpProfile::nowUs();
    // the selection is small (M x 13 + M x 8 at the softmax router's
    // width) and goes straight to the caller's arrays
    static_assert(sizeof(unsigned int) == sizeof(uint32_t), "sel is u32");
    const int err =
      h8 != UINT32_MAX
        ? nntr_hvx_router_logits_u8i8_f32(
            session, M, K, E, eps, gamma, gamma ? static_cast<int>(K) : 0, h8,
            act, static_cast<int>(x_len), top_k, top_k ? n_sel : 0u, scale,
            top_k ? static_cast<int>(E) : 0, out, static_cast<int>(out_len),
            reinterpret_cast<uint32_t *>(sel),
            top_k ? static_cast<int>(M * n_sel) : 0, weight,
            top_k ? static_cast<int>(M * top_k) : 0)
        : nntr_hvx_router_logits_f32(
            session, M, K, E, eps, gamma, gamma ? static_cast<int>(K) : 0,
            w_shared, static_cast<int>(static_cast<size_t>(K) * E), act,
            static_cast<int>(x_len), top_k, top_k ? n_sel : 0u, scale,
            top_k ? static_cast<int>(E) : 0, out, static_cast<int>(out_len),
            reinterpret_cast<uint32_t *>(sel),
            top_k ? static_cast<int>(M * n_sel) : 0, weight,
            top_k ? static_cast<int>(M * top_k) : 0);
    const uint64_t elapsed = HtpProfile::nowUs() - t0;
    if (err != AEE_SUCCESS) {
      throw std::runtime_error(
        "nntr_hvx_router_logits_f32 failed: err=" + std::to_string(err) +
        " (M=" + std::to_string(M) + " K=" + std::to_string(K) +
        " E=" + std::to_string(E) + ")");
    }
    stagedMemcpy(logits, out, out_len * sizeof(float));
    HtpProfile &profile = HtpProfile::global();
    if (profile.level())
      profile.addInvoke(M, K, E, elapsed, nullptr);
  }

  // The lm_head (see the base declaration): the canonical Q4_0 rows as
  // Q4M1 slices of kLmHeadSliceRows -- the same nibbles and scales,
  // reordered (q4m1_from_q4_0), no requantization -- placed in the arena
  // on the first call and attached by offset, then one call per row runs
  // the norm, every slice's GEMV and the softcap. A tied weight is ~415 MB
  // at a 262144 vocabulary: past the DSP heap (it refused at ~100 MiB), so
  // the arena or nothing. A placement or attach refused leaves the
  // lm_head on the CPU for the run, said once on stderr.
  // ponytail: it is placed after every other weight, in the address space
  // they leave; a second session's arena (the decode graph's S2) is the
  // upgrade if the first one has no room on a device.
  bool lm_head_q4_0_fp32(const void *w, unsigned int K, unsigned int N,
                         const float *x, const float *gamma, float eps,
                         float softcap, float *y) override {
    if (w == nullptr || K == 0 || K % 64 != 0 || K > 8192 || N == 0 ||
        N % 32 != 0)
      return false;
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const std::vector<uint32_t> *h = lmHeadHandles(w, session, K, N);
    if (h == nullptr)
      return false;
    return invokeLmHead(session, *h, K, N, x, gamma, eps, softcap, y);
  }

  bool invokeLmHead(remote_handle64 session, const std::vector<uint32_t> &h,
                    unsigned K, unsigned N, const float *x, const float *gamma,
                    float eps, float softcap, float *y) {
    std::lock_guard<std::mutex> lock(invoke_mutex_);
    float *act =
      reinterpret_cast<float *>(stage(act_pool_, K * sizeof(float)).data());
    float *out = reinterpret_cast<float *>(
      stage(out_pool_, static_cast<size_t>(N) * sizeof(float)).data());
    stagedMemcpy(act, x, K * sizeof(float));
    const uint64_t t0 = HtpProfile::nowUs();
    const int err = nntr_hvx_lm_head_q4m1_f32(
      session, eps, gamma, gamma ? static_cast<int>(K) : 0, softcap, h.data(),
      static_cast<int>(h.size()), act, static_cast<int>(K), out,
      static_cast<int>(N));
    const uint64_t elapsed = HtpProfile::nowUs() - t0;
    if (err != AEE_SUCCESS) {
      throw std::runtime_error(
        "nntr_hvx_lm_head_q4m1_f32 failed: err=" + std::to_string(err) +
        " (K=" + std::to_string(K) + " N=" + std::to_string(N) + ")");
    }
    stagedMemcpy(y, out, static_cast<size_t>(N) * sizeof(float));
    HtpProfile &profile = HtpProfile::global();
    if (profile.level())
      profile.addInvoke(1, K, N, elapsed, nullptr);
    return true;
  }

  void share_decode_lm_head(const void *w) override {
    if (e2e_ && weightPageCachePolicy() >= 0)
      shared_lm_heads_.emplace(w, SharedLmHead{});
  }

  bool lm_head_shared_fp32(const void *w, unsigned K, unsigned N,
                           const float *x, float *y) override {
    auto it = shared_lm_heads_.find(w);
    if (it == shared_lm_heads_.end() || it->second.h.empty())
      return false;
    if (it->second.K != K || it->second.N != N)
      throw std::invalid_argument("Shared LM head shape changed");
    if (it->second.wh)
      return invokeLmHeadWh(e2e_st_->h1, it->second.h, K, N, x, y);
    return invokeLmHead(e2e_st_->h1, it->second.h, K, N, x, nullptr, 0, 0, y);
  }

  /** @brief The untied QS4CX head's one row on its WH slices: the graph's
   *  LM_HEAD kernel outside the token driver (the prefill's last row). */
  bool invokeLmHeadWh(remote_handle64 session, const std::vector<uint32_t> &h,
                      unsigned K, unsigned N, const float *x, float *y) {
    std::lock_guard<std::mutex> lock(invoke_mutex_);
    float *act =
      reinterpret_cast<float *>(stage(act_pool_, K * sizeof(float)).data());
    float *out = reinterpret_cast<float *>(
      stage(out_pool_, static_cast<size_t>(N) * sizeof(float)).data());
    stagedMemcpy(act, x, K * sizeof(float));
    const uint64_t t0 = HtpProfile::nowUs();
    const int err = nntr_hvx_lm_head_wh_f32(
      session, h.data(), static_cast<int>(h.size()), act, static_cast<int>(K),
      out, static_cast<int>(N));
    const uint64_t elapsed = HtpProfile::nowUs() - t0;
    if (err != AEE_SUCCESS) {
      throw std::runtime_error(
        "nntr_hvx_lm_head_wh_f32 failed: err=" + std::to_string(err) +
        " (K=" + std::to_string(K) + " N=" + std::to_string(N) + ")");
    }
    stagedMemcpy(y, out, static_cast<size_t>(N) * sizeof(float));
    HtpProfile &profile = HtpProfile::global();
    if (profile.level())
      profile.addInvoke(1, K, N, elapsed, nullptr);
    return true;
  }

  void set_lm_head_wh_cache(const char *path) override {
    lmhead_wh_path_ = path ? path : "";
  }

  /** @brief The cache file's header (set_lm_head_wh_cache): the slicing
   *  and a hash of the N scales, then per slice WH bytes | f32 scales |
   *  i32 column sums, slices in column order. ponytail: codes requantized
   *  to the same scales pass the match; delete the file by hand then. */
  struct LmHeadWhHeader {
    char magic[8];
    uint32_t version, K, N, cols;
    uint64_t scale_hash;
  };

  /** @brief The whole buffer at @a off, or false (errno set). */
  static bool pwriteAll(int fd, const void *src, size_t len, uint64_t off) {
    const auto *p = static_cast<const uint8_t *>(src);
    while (len != 0) {
      const ssize_t n = ::pwrite(fd, p, len, static_cast<off_t>(off));
      if (n < 0 && errno == EINTR)
        continue;
      if (n <= 0)
        return false;
      len -= static_cast<size_t>(n);
      off += static_cast<uint64_t>(n);
      p += n;
    }
    return true;
  }

  /** @brief The untied QS4CX head's WH slices, columns [c0, c0 + cols) each
   *  (a column is K / 2 contiguous bytes, so a slice is a pointer offset
   *  into the codes), placed in the arena and registered as an FC slice is.
   *  With a matching cache file each slice is one pread straight into its
   *  arena room; without one the slice is converted here
   *  (htp_qs4cx_from_packed, whPack) and the cache written beside the model
   *  for the next load (a temporary renamed at the end, so a half-written
   *  file never matches). Keyed by the slice's first byte so a re-bind
   *  finds the handles. */
  void bindLmHeadWh(remote_handle64 session, const void *codes,
                    const float *scale, uint32_t K, uint32_t N, uint32_t cols,
                    uint32_t *h, uint32_t *parts) {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    LmHeadWhHeader want{};
    std::memcpy(want.magic, "NNTRLMWH", 8);
    want.version = 1u;
    want.K = K;
    want.N = N;
    want.cols = cols;
    want.scale_hash = fcWhKey(scale, sizeof(float) * N);
    int rfd = -1, wfd = -1;
    std::string tmp;
    if (!lmhead_wh_path_.empty()) {
      rfd = ::open(lmhead_wh_path_.c_str(), O_RDONLY | O_CLOEXEC);
      LmHeadWhHeader have{};
      if (rfd >= 0 && (preadAll(rfd, &have, sizeof(have), 0) != 0 ||
                       std::memcmp(&have, &want, sizeof(want)) != 0)) {
        std::fprintf(stderr, "[HTP] lm_head WH cache %s: stale, rebuilt\n",
                     lmhead_wh_path_.c_str());
        ::close(rfd);
        rfd = -1;
      }
      if (rfd < 0) {
        tmp = lmhead_wh_path_ + ".tmp";
        wfd =
          ::open(tmp.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0644);
        if (wfd >= 0 && !pwriteAll(wfd, &want, sizeof(want), 0)) {
          ::close(wfd);
          ::unlink(tmp.c_str());
          wfd = -1;
        }
        if (wfd < 0)
          std::fprintf(stderr, "[HTP] lm_head WH cache %s: not written (%s)\n",
                       lmhead_wh_path_.c_str(), std::strerror(errno));
      }
    }
    uint64_t off = sizeof(want);
    for (uint32_t c0 = 0; c0 < N; c0 += cols) {
      const uint32_t n = std::min(cols, N - c0);
      const size_t len = whBytes(K, n);
      void *key = static_cast<uint8_t *>(const_cast<void *>(codes)) +
                  static_cast<size_t>(c0) * (K / 2u);
      auto it = handle_cache_.find(key);
      if (it != handle_cache_.end()) {
        h[(*parts)++] = it->second;
        off += len + 8u * n;
        continue;
      }
      const uint64_t t0 = HtpProfile::nowUs();
      ArenaEntry e;
      e.K = K;
      e.N = n;
      e.w_scale.resize(n);
      e.colsum_w.resize(n);
      e.bias.assign(n, 0.0f);
      uint32_t handle = kNoHandle;
      if (ensureArena(session) && place(session, static_cast<uint32_t>(len),
                                        size_t(256) << 20, &e.chunk, &e.off)) {
        uint8_t *dst = arena_chunks_[e.chunk].buf->data() + e.off;
        if (rfd >= 0) {
          throwPread(preadAll(rfd, dst, len, off), "lm_head wh cache");
          throwPread(
            preadAll(rfd, e.w_scale.data(), sizeof(float) * n, off + len),
            "lm_head wh cache scales");
          throwPread(preadAll(rfd, e.colsum_w.data(), sizeof(int32_t) * n,
                              off + len + sizeof(float) * n),
                     "lm_head wh cache column sums");
        } else {
          std::vector<int8_t> rm(static_cast<size_t>(K) * n);
          htp_qs4cx_from_packed(key, scale + c0, K, n, rm.data(),
                                e.w_scale.data(), e.colsum_w.data());
          whPack(rm.data(), K, n, dst);
          if (wfd >= 0 &&
              !(pwriteAll(wfd, dst, len, off) &&
                pwriteAll(wfd, e.w_scale.data(), sizeof(float) * n,
                          off + len) &&
                pwriteAll(wfd, e.colsum_w.data(), sizeof(int32_t) * n,
                          off + len + sizeof(float) * n))) {
            std::fprintf(stderr,
                         "[HTP] lm_head WH cache %s: write failed (%s), "
                         "dropped\n",
                         lmhead_wh_path_.c_str(), std::strerror(errno));
            ::close(wfd);
            ::unlink(tmp.c_str());
            wfd = -1;
          }
        }
        handle = registerFromArena(session, e, K, n, t0);
      }
      if (handle == kNoHandle) {
        if (rfd >= 0)
          ::close(rfd);
        if (wfd >= 0) {
          ::close(wfd);
          ::unlink(tmp.c_str());
        }
        throw std::runtime_error(
          "set_decode_graph_desc: the lm_head's WH slice at column " +
          std::to_string(c0) + " (" + std::to_string(K) + " x " +
          std::to_string(n) + ") found no arena room (" + arena_fail_ + ")");
      }
      handle_cache_.emplace(key, handle);
      h[(*parts)++] = handle;
      off += len + 8u * n;
    }
    if (rfd >= 0) {
      ::close(rfd);
      std::fprintf(stderr, "[HTP] lm_head WH cache %s: read\n",
                   lmhead_wh_path_.c_str());
    }
    if (wfd >= 0) {
      ::close(wfd);
      if (::rename(tmp.c_str(), lmhead_wh_path_.c_str()) != 0) {
        std::fprintf(stderr, "[HTP] lm_head WH cache %s: rename failed (%s)\n",
                     lmhead_wh_path_.c_str(), std::strerror(errno));
        ::unlink(tmp.c_str());
      } else {
        std::fprintf(stderr, "[HTP] lm_head WH cache %s: written, %.1f MiB\n",
                     lmhead_wh_path_.c_str(),
                     static_cast<double>(off) / (1u << 20));
      }
    }
  }

  bool lm_head_q4_0_prepare(const void *w, unsigned int K,
                            unsigned int N) override {
    if (w == nullptr || K == 0 || K % 64 != 0 || K > 8192 || N == 0 ||
        N % 32 != 0)
      return false;
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    return lmHeadHandles(w, session, K, N) != nullptr;
  }

  /** @brief The lm_head weight's Q4M1 slice handles, placed on first use;
   *  nullptr when they could not be (then for good, said once). */
  const std::vector<uint32_t> *lmHeadHandles(const void *w,
                                             remote_handle64 session,
                                             unsigned int K, unsigned int N) {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    auto it = lm_head_.find(w);
    if (it != lm_head_.end())
      return it->second.ok ? &it->second.h : nullptr;
    LmHeadSlices lh;
    const uint8_t *canonical = static_cast<const uint8_t *>(w);
    const size_t row_bytes = static_cast<size_t>(K / 32u) * Q4_CPU_BLOCK_BYTES;
    size_t left = 0;
    for (uint32_t r0 = 0; r0 < N; r0 += kLmHeadSliceRows)
      left +=
        q4m1_bytes(K, std::min<uint32_t>(kLmHeadSliceRows, N - r0)) + 4096u;
    const size_t total = left;
    const uint64_t t0 = HtpProfile::nowUs();
    std::string why;
    for (uint32_t r0 = 0; r0 < N; r0 += kLmHeadSliceRows) {
      const uint32_t rows = std::min<uint32_t>(kLmHeadSliceRows, N - r0);
      const uint32_t bytes = static_cast<uint32_t>(q4m1_bytes(K, rows));
      std::vector<uint8_t> q4m1(bytes);
      q4m1_from_q4_0(canonical + r0 * row_bytes, K, rows, q4m1.data());
      uint32_t handle = 0, chunk = 0, off = 0;
      int err;
      if (ensureArena(session) && place(session, bytes, left, &chunk, &off)) {
        std::memcpy(arena_chunks_[chunk].buf->data() + off, q4m1.data(), bytes);
        err = nntr_hvx_q4m1_attach(session, arena_chunks_[chunk].dsp_id, off, K,
                                   rows, &handle);
        if (err != AEE_SUCCESS)
          why = "q4m1_attach err=" + std::to_string(err);
      } else {
        err = nntr_hvx_q4m1_register(session, K, rows, q4m1.data(),
                                     static_cast<int>(bytes), &handle);
        if (err != AEE_SUCCESS)
          why = "no arena room (" + arena_fail_ +
                "), q4m1_register err=" + std::to_string(err);
      }
      if (err != AEE_SUCCESS)
        break;
      lh.h.push_back(handle);
      left = left > bytes + 4096u ? left - bytes - 4096u : 0u;
    }
    if (!why.empty()) {
      for (uint32_t handle : lh.h)
        nntr_hvx_q4m1_release(session, handle);
      std::fprintf(stderr,
                   "[HTP] lm_head stays on the CPU: slice %zu of %u rows "
                   "refused (%s)\n",
                   lh.h.size(), N, why.c_str());
      lm_head_[w] = LmHeadSlices{};
      return nullptr;
    }
    lh.ok = true;
    std::fprintf(stderr,
                 "[HTP] lm_head on the NPU: %u x %u Q4_0 as %zu Q4M1 slices, "
                 "%.1f MiB, placed in %.0f ms\n",
                 N, K, lh.h.size(), static_cast<double>(total) / (1u << 20),
                 static_cast<double>(HtpProfile::nowUs() - t0) / 1000.0);
    return &(lm_head_[w] = std::move(lh)).h;
  }

  // A QS4CX weight was quantized once, straight from FP32, and already
  // holds the int4 values this registry wants -- so the seam is
  // htp_qs4cx_from_packed's bit rearrangement plus a colsum, not a second
  // quantization. That is both more accurate (measured on this branch:
  // mean_abs_err 0.0333 vs 0.0451 through Q4_0, a 26% cut) and cheaper to
  // load (95.9 -> 59.9 ms for a 2048x3584 gate_up, 77.2 -> 22.2 ms for a
  // 1792x2048 down, x86 host). Prefer feeding the FFN QS4CX; the Q4_0
  // entry above stays for weights shared with the CPU path.
  bool supports_gemm_qs4cx_accel_fp32() const override { return true; }

  void gemm_qs4cx_accel_fp32(void *matAdata, float *matAscale, float *matBdata,
                             float *matCdata, unsigned int M, unsigned int N,
                             unsigned int K) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    // Column slices and row chunks as gemm_q4_0_accel_fp32 takes them: one
    // handle over the whole weight overflows VTCM once it is wider than
    // fcSliceCols (2816 x 4096 at M = 512 answered AEE_ENOMEMORY). A weight
    // that fits one slice keeps its handle_cache_ entry, so an expert that
    // register_qs4cx_weight registered is not registered twice.
    invokeFc(session, get_or_register_fc(matAdata, session, K, N, matAscale),
             matBdata, matCdata, M, N, K);
  }

  // The o-proj with the post-attention epilogue (see the base declaration):
  // the FC's row chunks as invokeFc takes them, each one call that also
  // norms, adds the residual and scales, so the FC output never comes back.
  const void *host_view(const void *host, size_t bytes) override {
    const void *ion = ionAlias(host, bytes);
    return ion ? ion : host;
  }

  void sync_to_host(void *host, size_t bytes) override {
    std::lock_guard<std::mutex> lock(invoke_mutex_);
    const void *ion = ionAlias(host, bytes);
    if (!ion)
      ion = trueSource(host, bytes);
    if (ion == host)
      throwIfLost(host, bytes, __LINE__);
    if (ion && ion != host)
      stagedMemcpy(host, ion, bytes);
  }

  bool gemm_qs4cx_res_add_fp32(void *w, float *w_scale, const float *act,
                               unsigned int M, unsigned int K, unsigned int N,
                               const float *res, const float *gamma, float eps,
                               float scale, float *out,
                               bool out_to_host) override {
    if (M <= 1 || w == nullptr || act == nullptr || res == nullptr)
      return false;
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const FcHandles &fh = get_or_register_fc(w, session, K, N, w_scale);
    // the attention's u16 context, still in its staging buffer: the DSP
    // dequantizes it inside the quant pass, nothing is copied or converted
    bool u16_in = attn_out16_.host == act && attn_out16_.rows == M &&
                  attn_out16_.width == K;
    // NNTR_HTP_OPROJ_U16=0: the u16 context dequantized here into the host
    // tensor and staged as f32, instead of the DSP's u16 input path (a
    // reference for that path; never set in a tok/s run)
    static const bool u16_off = [] {
      const char *v = std::getenv("NNTR_HTP_OPROJ_U16");
      return v != nullptr && std::strcmp(v, "0") == 0;
    }();
    if (u16_in && u16_off) {
      float *dst = const_cast<float *>(act);
      const unsigned int hd = attn_out16_.head_dim;
      for (unsigned int r = 0; r < M; ++r)
        for (unsigned int h = 0; h < K / hd; ++h) {
          const float sc = attn_out16_.enc[2 * h],
                      zp = attn_out16_.enc[2 * h + 1];
          const uint16_t *u =
            attn_out16_.ion + static_cast<size_t>(r) * K + h * hd;
          float *d = dst + static_cast<size_t>(r) * K + h * hd;
          for (unsigned int i = 0; i < hd; ++i)
            d[i] = (static_cast<float>(u[i]) - zp) * sc;
        }
      attn_out16_.host = nullptr;
      u16_in = false;
    }
    // The residual, the output (and an f32 activation) staged whole, and
    // one call: the skel walks the row chunks VTCM asks for.
    const size_t o_all = static_cast<size_t>(M) * N * sizeof(float);
    const size_t a_all = static_cast<size_t>(M) * K * sizeof(float);
    std::lock_guard<std::mutex> lock(invoke_mutex_);
    // the residual is the rows the qkv call staged as its activation:
    // swapped in from act_pool_ when it still holds them
    const float *res_all = reinterpret_cast<const float *>(
      stageInput(x2_pool_, res, o_all, &act_pool_).data());
    const float *act_all = nullptr;
    if (!u16_in) {
      act_all = static_cast<const float *>(ionAlias(act, a_all));
      if (!act_all)
        act_all = reinterpret_cast<const float *>(
          stageInput(act_pool_, act, a_all).data());
    }
    float *out_all = reinterpret_cast<float *>(stage(out_pool_, o_all).data());
    const uint64_t t0 = HtpProfile::nowUs();
    const int err =
      u16_in
        ? nntr_hvx_mm_u8i4_layer_res_add_u16(
            session, M, K, eps, scale, gamma, gamma ? static_cast<int>(N) : 0,
            fh.handles.data(), static_cast<int>(fh.handles.size()),
            attn_out16_.ion, static_cast<int>(M * K), attn_out16_.enc.data(),
            static_cast<int>(attn_out16_.enc.size()), attn_out16_.head_dim,
            res_all, static_cast<int>(M * N), out_all, static_cast<int>(M * N))
        : nntr_hvx_mm_u8i4_layer_res_add(
            session, M, K, eps, scale, gamma, gamma ? static_cast<int>(N) : 0,
            fh.handles.data(), static_cast<int>(fh.handles.size()), act_all,
            static_cast<int>(M * K), res_all, static_cast<int>(M * N), out_all,
            static_cast<int>(M * N));
    const uint64_t elapsed = HtpProfile::nowUs() - t0;
    if (err != AEE_SUCCESS)
      throw std::runtime_error("nntr_hvx_mm_u8i4_layer_res_add failed: err=" +
                               std::to_string(err));
    if (HtpProfile::global().level())
      HtpProfile::global().addInvoke(M, K, N, elapsed, nullptr);
    // remembered for the dense call's stageInput (its from): the
    // post-attention rows swap in instead of being copied again
    copyOutOf(out_pool_, out, o_all, out_to_host);
    if (u16_in)
      attn_out16_.host = nullptr;
    // NNTR_HTP_DUMP=<dir>: this call's act (the attention's context, as the
    // host has it), res and out as fc_<call>_{act,res,out}.f32; valid with
    // NNTR_HTP_KEEP_OUT=0 (and NNTR_HTP_OPROJ_U16=0 for the u16 context)
    if (const char *dir = std::getenv("NNTR_HTP_DUMP")) {
      static unsigned int call = 0;
      char name[32];
      std::snprintf(name, sizeof(name), "/fc_%05u_", call++);
      const std::string base = std::string(dir) + name;
      dumpAllFile(base + "act.f32", act, static_cast<size_t>(M) * K);
      dumpAllFile(base + "res.f32", res, static_cast<size_t>(M) * N);
      dumpAllFile(base + "out.f32", out, static_cast<size_t>(M) * N);
      FILE *m = std::fopen((std::string(dir) + "/fc_manifest.txt").c_str(),
                           call == 1 ? "w" : "a");
      if (m) {
        std::fprintf(m, "%s %u %u %u\n", name + 1, M, K, N);
        std::fclose(m);
      }
    }
    return true;
  }

  // Same grouping as gemm_q4_0_batch_fp32, for QS4CX weights -- see the
  // comment on the base declaration (compute_ops.h). Without this,
  // FloatTensor::dot's vector overload had no accelerated path for QS4CX at
  // all: every "grouped" decode call under --moe_dtype QS4CX silently fell
  // through to one dotQs4cx() -- one FastRPC call -- per expert, the same
  // shape of miss gemm_q4_0_batch_fp32's own commit fixed for Q4_0.
  bool supports_gemm_qs4cx_batch_fp32() const override { return true; }

  void gemm_qs4cx_batch_fp32(std::vector<void *> matAdata,
                             std::vector<float *> matAscale, float *matBdata,
                             std::vector<float *> matCdata, unsigned int M,
                             std::vector<unsigned int> N,
                             unsigned int K) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const size_t n = matAdata.size();
    std::vector<uint32_t> handles(n);
    unsigned int n_total = 0;
    for (size_t i = 0; i < n; ++i) {
      handles[i] =
        get_or_register_qs4cx(matAdata[i], matAscale[i], session, K, N[i]);
      n_total += N[i];
    }

    // Same stage-then-hand-out shape as gemm_q4_0_batch_fp32 -- see that
    // function's comment for why: mm_u8i4_layer returns one contiguous
    // block per handle, matCdata is one pointer per weight.
    std::vector<float> out_cat(static_cast<size_t>(M) * n_total);
    invokeLayer(session, handles.data(), static_cast<int>(n), matBdata,
                out_cat.data(), M, n_total, K);

    size_t off = 0;
    for (size_t i = 0; i < n; ++i) {
      const size_t block = static_cast<size_t>(M) * N[i];
      std::memcpy(matCdata[i], out_cat.data() + off, block * sizeof(float));
      off += block;
    }
  }

  // The fused expert FFN (doc 43 §[L2]): one FastRPC call per expert per
  // layer instead of two, with the M x 2I + M x I intermediates never
  // leaving the DSP. The caller (Lfm2MoELayer) gates on M > 1 itself --
  // decode's single token cannot amortize the fused call's 64-row pad
  // tax, the same reasoning as accelerates_q4_0_at_m1() being false.
  // [doc 46] The whole layer in one call. supports_* stays behind the
  // caller's own M > 1 gate for the same reason the fused one does: a
  // single decode token cannot amortize the 64-row pad tax.
  bool supports_gemm_qs4cx_moe_layer_fp32() const override { return true; }

  void gemm_qs4cx_moe_layer_fp32(const std::vector<void *> &gate_up_data,
                                 const std::vector<float *> &gate_up_scale,
                                 const std::vector<void *> &down_data,
                                 const std::vector<float *> &down_scale,
                                 const std::vector<unsigned int> &row_index,
                                 const std::vector<unsigned int> &row_count,
                                 const std::vector<float> &row_weight,
                                 const float *act, float *out, unsigned int M,
                                 unsigned int K, unsigned int inter,
                                 unsigned int N_out, unsigned int w_bits,
                                 bool gelu, const float *pre_gamma,
                                 const float *post_gamma, float eps,
                                 const float *add_x2, const float *add_gamma,
                                 float add_scale, bool out_to_host) override {
    switchTl2Phase(false);
    // #4415's lfm2_moe passes its bool weights_wh: true reads as 4 bits
    if (w_bits == 1u)
      w_bits = 4u;
    const size_t n_experts = gate_up_data.size();
    if (n_experts == 0 || gate_up_scale.size() != n_experts ||
        down_data.size() != n_experts || down_scale.size() != n_experts ||
        row_count.size() != n_experts ||
        row_weight.size() != row_index.size()) {
      throw std::invalid_argument(
        "gemm_qs4cx_moe_layer_fp32: per-expert arrays disagree");
    }

    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    sendMoeOptsOnce(session);
    std::vector<uint32_t> h_gu(n_experts), h_dn(n_experts);
    for (size_t e = 0; e < n_experts; ++e) {
      if (w_bits != 0u) {
        h_gu[e] = get_or_register_wh(gate_up_data[e], gate_up_scale[e], session,
                                     K, 2 * inter, w_bits);
        h_dn[e] = get_or_register_wh(down_data[e], down_scale[e], session,
                                     inter, N_out, w_bits);
      } else {
        h_gu[e] = get_or_register_qs4cx(gate_up_data[e], gate_up_scale[e],
                                        session, K, 2 * inter);
        h_dn[e] = get_or_register_qs4cx(down_data[e], down_scale[e], session,
                                        inter, N_out);
      }
    }
    // [#85] With NNTR_HTP_FORWARD=1 and a description from the model, every
    // MoE call binds its handles to the next MoE op in list order (the
    // first pass runs the layers in order), and once all are bound the
    // M == 1 calls go through the per-token entry instead. Errors throw
    // (contract section 2): a fallback to mm_u8i4_moe_layer would report
    // per-layer numbers as the graph's.
    if (forwardSwitch() && !graph_words_.empty()) {
      // [plan 201 section 2.3] a virtual expert (NNTR_MOE_CACHE_EXPERTS: a
      // null scale) is resident only until the ARM's LRU evicts it, and
      // the per-token entry would keep reading its handle after that
      if (std::find(gate_up_scale.begin(), gate_up_scale.end(), nullptr) !=
          gate_up_scale.end()) {
        // [plan 201 S1] with NNTR_HTP_E2E=1 the token driver serves the
        // misses (poolServe) and the tables follow the pool at the next
        // token (poolSync); these calls (the prefill's) stay per layer
        if (!e2e_)
          throw std::runtime_error(
            "NNTR_HTP_FORWARD with NNTR_MOE_CACHE_EXPERTS: the expert pool "
            "is served only by the E2E token (NNTR_HTP_E2E=1)");
        {
          std::lock_guard<std::mutex> lock(graph_mutex_);
          // the layers hand their experts at their first calls, in order:
          // once all have, decode's rows go to the token entry
          if (pool_descs_.size() == moe_ops_.size())
            moe_bound_ = moe_ops_.size();
          pool_dirty_ = true;
        }
        // [#260] with #4415's epilogue (GeGLU, the folded norms), as below
        invokeMoeLayer(session, h_gu, h_dn, row_index, row_count, row_weight,
                       act, out, M, K, inter, N_out, 0, gelu ? 1u : 0u,
                       pre_gamma, post_gamma, eps, add_x2, add_gamma, add_scale,
                       out_to_host);
        return;
      }
      const uint32_t op = bindMoeOp(h_gu, h_dn, K, inter, N_out);
      // [#132] A sole MOE stretch runs here; with ADD resident the stretch
      // is [MOE ADD RMSNORM] and the next layer's norm hook runs it, so a
      // row the hooks do not drive (row_bound_ false: a one-token prompt
      // binding as it goes) keeps the per-layer path.
      const bool sole = stretch_end_[op] == op + 1u;
      if (M == 1 && moe_bound_ == moe_ops_.size() && graphOp(op)->resident &&
          (sole || row_bound_)) {
        ensureGraphInit(session);
        // pos: no MoE stretch reads the position; the hooks' row does
        runStretchOp(session, op, sole ? 0u : cur_pos_, act, K, out, N_out,
                     &row_index, &row_count, &row_weight);
        return;
      }
      if (M == 1 && !graph_short_warned_ && moe_bound_ != moe_ops_.size()) {
        // moe_htp_layers naming a subset: the list's MoE ops can never all
        // be bound, so say so once instead of silently taking the
        // per-layer path under a measurement switch. The small-op hooks
        // read the same condition and return 0 (the CPU path). All bound
        // but not taken above -- a row the hooks do not drive, or MOE not
        // resident -- is not this case (#132).
        graph_short_warned_ = true;
        std::fprintf(stderr,
                     "[HTP] graph: %zu of %zu MoE ops bound at the first "
                     "M==1 call; the per-token entry is NOT used (MoE ops "
                     "and the RMSNORM / QK_NORM / ROPE / CONV1D_GATE / "
                     "ATTN_M1 hooks alike)\n",
                     moe_bound_, moe_ops_.size());
      }
    }
    invokeMoeLayer(session, h_gu, h_dn, row_index, row_count, row_weight, act,
                   out, M, K, inter, N_out, 0, gelu ? 1u : 0u, pre_gamma,
                   post_gamma, eps, add_x2, add_gamma, add_scale, out_to_host);
  }

  /** [#85] NNTR_HTP_FORWARD=1 routes decode's MoE calls through the
   *  per-token entry. Off by default: with only MOE resident it is 22
   *  calls per token either way (plan 85 section 0), so the default path
   *  stays byte-for-byte today's. */
  static bool forwardSwitch() {
    static const bool on = [] {
      const char *env = std::getenv("NNTR_HTP_FORWARD");
      // [#132 Part B E3] NNTR_HTP_E2E=1 is the per-token entry with every
      // kind resident
      return (env != nullptr && std::atoi(env) != 0) ||
             HtpBackend::e2eRequested();
    }();
    return on;
  }

  /** The graph entries' error, named: 0x8000040E from them means the
   *  skel predates the per-token entry (rule 3), not a list the validator
   *  above accepted. */
  static std::string graphErr(int err) {
    char hex[16];
    std::snprintf(hex, sizeof(hex), "%08x", static_cast<unsigned>(err));
    return std::string(htp_graph_err_name(static_cast<uint32_t>(err))) +
           " (0x" + hex + ")" +
           (static_cast<unsigned>(err) == 0x8000040Eu
              ? " -- AEE_EBADPARM: rebuild the DSP skel (test/htp/build.sh)"
              : "");
  }

  /** [#130] The kinds this ARM side can drive: the MoE call and the
   *  layer hooks (htp_decode_hook.h), with #132's residual add and router.
   *  The skel's own table is checked at graph_init (AEE_ECLASSNOTSUPPORT
   *  names a kind it lacks). */
  static constexpr uint32_t kArmKinds =
    HTP_GRAPH_KIND_BIT(HTP_OP_MOE) | HTP_GRAPH_KIND_BIT(HTP_OP_RMSNORM) |
    HTP_GRAPH_KIND_BIT(HTP_OP_QK_NORM) | HTP_GRAPH_KIND_BIT(HTP_OP_ROPE) |
    HTP_GRAPH_KIND_BIT(HTP_OP_CONV1D_GATE) |
    HTP_GRAPH_KIND_BIT(HTP_OP_ATTN_M1) | HTP_GRAPH_KIND_BIT(HTP_OP_ADD) |
    HTP_GRAPH_KIND_BIT(HTP_OP_ROUTER_TOPK) | HTP_GRAPH_KINDS_Q4M1;

  /** [#132 Part B] Rows per lm_head slice (plan 132 section 3.2: 8 slices
   *  of 16384 rows at LFM2.5, 18 MiB each). */
  static constexpr uint32_t kLmHeadSliceRows = 16384u;

  bool set_decode_graph_desc(const std::vector<uint32_t> &words) override {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    uint32_t n_ops = 0;
    // The same validator the skel runs, so a bad list fails here with the
    // same code and name before any FastRPC call is made.
    const uint32_t rc = htp_graph_validate(
      words.data(), static_cast<uint32_t>(words.size()), kArmKinds, &n_ops);
    if (rc != 0u) {
      throw std::invalid_argument(std::string("set_decode_graph_desc: ") +
                                  htp_graph_err_name(rc));
    }
    uint32_t mask = 0, present = 0;
    for (uint32_t i = 0; i < n_ops; ++i) {
      const htp_graph_op *op = htp_graph_op_cat(words.data(), i);
      present |= HTP_GRAPH_KIND_BIT(op->kind);
      if (op->resident)
        mask |= HTP_GRAPH_KIND_BIT(op->kind);
    }
    // The ARM's own rule: a resident QK_NORM or ROPE needs ATTN_M1 --
    // no CPU layer consumes their DSP output (the qkv hook writes
    // nothing; ROPE has no layer), so the stretch must end in the
    // attention hook. The DSP enforces only ATTN_M1 -> ROPE.
    if ((mask & (HTP_GRAPH_KIND_BIT(HTP_OP_QK_NORM) |
                 HTP_GRAPH_KIND_BIT(HTP_OP_ROPE))) != 0u &&
        (mask & HTP_GRAPH_KIND_BIT(HTP_OP_ATTN_M1)) == 0u) {
      throw std::invalid_argument("set_decode_graph_desc: QK_NORM / ROPE "
                                  "resident without ATTN_M1 (no CPU layer "
                                  "consumes their output)");
    }
    // [#132] A resident ROUTER_TOPK needs ADD resident: then the ffn norm
    // before it and the ADD after its MOE are resident too, so the MoE
    // layer's one hook is always mid-stretch and the layer skips its
    // router, top-k and experts. Without ADD the MoE call would start the
    // stretch, and it has no row of its own to start it with.
    if ((mask & HTP_GRAPH_KIND_BIT(HTP_OP_ROUTER_TOPK)) != 0u &&
        (mask & HTP_GRAPH_KIND_BIT(HTP_OP_ADD)) == 0u) {
      throw std::invalid_argument("set_decode_graph_desc: ROUTER_TOPK "
                                  "resident without ADD");
    }
    // [#132 Part B, E1] No CPU layer hooks an FC, the dense FFN or a
    // projection, so a Q4M1 kind is resident only when every kind the
    // list has is: the list is then one stretch per token, op 0's norm
    // hook keeps the row and the lm_head hook runs it (one call per
    // token). Its weights come from the model at load
    // (add_decode_graph_q4_0).
    if ((mask & HTP_GRAPH_KINDS_Q4M1) != 0u && mask != present) {
      throw std::invalid_argument(
        "set_decode_graph_desc: FC / DENSE_FFN / LM_HEAD resident without "
        "every other kind (one session runs them only as the whole token)");
    }
    // [#132 Part B] NNTR_HTP_FC_FEED=l2 reads the Q4M1 kinds' weights
    // through the L2 scratch even where VTCM would fit (the second
    // session's feed, #178); unset or vtcm: VTCM when it fits
    const char *feed_env = std::getenv("NNTR_HTP_FC_FEED");
    const std::string feed_s = feed_env ? feed_env : "vtcm";
    if (feed_s != "vtcm" && feed_s != "l2") {
      throw std::invalid_argument("set_decode_graph_desc: NNTR_HTP_FC_FEED=" +
                                  feed_s + " (want vtcm or l2)");
    }
    // [#132 Part B E3, #211] NNTR_HTP_E2E=1: one PD, every kind resident,
    // the FC set on its own arena chunks and the expert pool beside it.
    const bool e2e = HtpBackend::e2eRequested();
    if (e2e) {
      // [#211] a guard, not a switch: the runners pass _PDS=1, and a _PDS=2
      // that ran one PD would label a row with a variant that is gone.
      // ponytail: drop it with the next IDL change (LEDGER 3a)
      const char *pds = std::getenv("NNTR_HTP_E2E_PDS");
      if (pds != nullptr && std::strcmp(pds, "1") != 0)
        throw std::invalid_argument(
          "NNTR_HTP_E2E_PDS=" + std::string(pds) +
          ": the two-PD path was removed (#211); one PD is the only E2E "
          "entry, unset the variable");
      if (mask != present) {
        throw std::invalid_argument(
          "set_decode_graph_desc: NNTR_HTP_E2E=1 needs every kind of the "
          "list resident");
      }
      if (graph_inited_) {
        // ponytail: one description per process on the E2E path (the
        // model sets it once); a reload would re-map the FC arena
        throw std::runtime_error("set_decode_graph_desc: a second "
                                 "description with NNTR_HTP_E2E=1");
      }
    }
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    if (graph_inited_) {
      // A second description in one process (a reload): the skel still
      // holds the first graph and would refuse the next init.
      nntr_hvx_graph_release(session);
      if (attn_registered_)
        nntr_hvx_attn_m1_release(session);
    }
    releaseQ4m1(session);
    q4_pending_.clear();
    q4m1_bound_ = false;
    graph_words_ = words;
    e2e_ = e2e;
    if (e2e && !e2e_st_) {
      e2e_st_ = std::make_shared<E2eState>();
      // [plan 201 S1] every kind, the FC set on its own arena chunks and
      // the expert pool all in S1, one token packet (the FC feed takes
      // S1's VTCM in turn with the MoE)
      e2e_st_->h1 = session;
      std::shared_ptr<E2eState> st = e2e_st_;
      HtpBackend::global().atClose([st] { e2eTeardown(*st); });
    }
    // [#132 Part B E5f] NNTR_HTP_FC_LANES=<small>,<large>: the L2 feed's
    // lanes for K <= 2048 / larger K (1..8 each; unset: the DSP's 6 / 3)
    uint32_t fc_lanes = 0;
    if (const char *l = std::getenv("NNTR_HTP_FC_LANES")) {
      unsigned a = 0, b = 0;
      if (std::sscanf(l, "%u,%u", &a, &b) != 2 || a < 1 || a > 8 || b < 1 ||
          b > 8)
        throw std::invalid_argument("NNTR_HTP_FC_LANES=" + std::string(l) +
                                    " (want <small>,<large>, 1..8 each)");
      fc_lanes = (a << 8) | (b << 12);
    }
    // [#194, htp_moe_ppl] NNTR_HTP_PPL_LEVERS=<mask>: the non-bit-exact
    // levers of plan 194 section 3.1, bit n = lever Ln (0x2: L1, the native
    // FC / DENSE_FFN / LM_HEAD kernels). Unset or 0 = the CPU-exact kernels
    // (E0, bit-identical to htp_moe); a bit no lever owns yet is refused.
    uint32_t levers = 0;
    if (const char *l = std::getenv("NNTR_HTP_PPL_LEVERS")) {
      char *end = nullptr;
      const unsigned long v = std::strtoul(l, &end, 0);
      if (end == l || *end != '\0' || (v & ~0x2ul) != 0)
        throw std::invalid_argument("NNTR_HTP_PPL_LEVERS=" + std::string(l) +
                                    " (known: 0x2 = L1 native FC)");
      levers = static_cast<uint32_t>(v);
    }
    std::fprintf(stderr, "[HTP] ppl levers=0x%x L1=%s\n", levers,
                 (levers & 0x2u) ? "native_fc" : "exact");
    for (uint32_t i = 0; i < n_ops; ++i) {
      htp_graph_op *op = htp_graph_op_at(graph_words_.data(), i);
      if ((HTP_GRAPH_KINDS_Q4M1 & HTP_GRAPH_KIND_BIT(op->kind)) != 0u)
        op->feed = (feed_s == "l2" ? HTP_GRAPH_FEED_L2 : 0u) | fc_lanes |
                   ((levers & 0x2u) ? HTP_GRAPH_FEED_NATIVE : 0u);
    }
    resident_mask_ = mask;
    moe_ops_.clear();
    for (auto &v : kind_ops_)
      v.clear();
    stretch_start_.assign(n_ops, 0);
    stretch_end_.assign(n_ops, 0);
    attn_ordinal_.assign(n_ops, 0);
    param_bound_.assign(n_ops, 0);
    conv_next_pos_.assign(n_ops, HTP_GRAPH_NO_OP);
    first_resident_op_ = HTP_GRAPH_NO_OP;
    uint32_t n_attn = 0;
    for (uint32_t i = 0; i < n_ops; ++i) {
      const htp_graph_op *op = htp_graph_op_cat(words.data(), i);
      if (op->kind == HTP_OP_MOE)
        moe_ops_.push_back(i);
      if (op->kind == HTP_OP_ATTN_M1)
        attn_ordinal_[i] = n_attn++;
      if (!op->resident)
        continue;
      kind_ops_[op->kind].push_back(i);
      if (first_resident_op_ == HTP_GRAPH_NO_OP)
        first_resident_op_ = i;
      // a stretch is a maximal run of resident ops [s, e)
      stretch_start_[i] =
        (i > 0 && htp_graph_op_cat(words.data(), i - 1)->resident)
          ? stretch_start_[i - 1]
          : i;
    }
    for (uint32_t i = n_ops; i-- > 0;) {
      const htp_graph_op *op = htp_graph_op_cat(words.data(), i);
      if (!op->resident)
        continue;
      stretch_end_[i] =
        (i + 1 < n_ops && htp_graph_op_cat(words.data(), i + 1)->resident)
          ? stretch_end_[i + 1]
          : i + 1;
    }
    kv_len_.assign(n_attn, 0);
    // [plan 201 S4] the attention shapes, in list order (one cache each,
    // at most two: Gemma 4's sliding and full layers)
    attn_op_.assign(n_attn, 0);
    attn_cache_.assign(n_attn, 0);
    attn_cache_ord_.assign(n_attn, 0);
    {
      std::vector<const htp_graph_op *> shapes;
      std::vector<uint32_t> per_shape;
      for (uint32_t i = 0; i < n_ops; ++i) {
        const htp_graph_op *op = htp_graph_op_cat(words.data(), i);
        if (op->kind != HTP_OP_ATTN_M1)
          continue;
        size_t c = 0;
        while (c < shapes.size() &&
               (shapes[c]->n_kv != op->n_kv || shapes[c]->gqa != op->gqa ||
                shapes[c]->head_dim != op->head_dim))
          ++c;
        if (c == shapes.size()) {
          if (c == 2u)
            throw std::invalid_argument(
              "set_decode_graph_desc: a third attention shape (the session "
              "holds two caches)");
          shapes.push_back(op);
          per_shape.push_back(0u);
        }
        const uint32_t ord = attn_ordinal_[i];
        attn_op_[ord] = i;
        attn_cache_[ord] = static_cast<uint32_t>(c);
        attn_cache_ord_[ord] = per_shape[c]++;
      }
    }
    load_params_.clear();
    std::fill(std::begin(kind_next_), std::end(kind_next_), 0u);
    cur_pos_ = HTP_GRAPH_NO_OP;
    clearPending();
    seed_ordinal_ = HTP_GRAPH_NO_OP;
    attn_registered_ = false;
    moe_bound_ = 0;
    moe_op_by_handle_.clear();
    moe_tables_.assign(moe_ops_.size(), {});
    graph_inited_ = false;
    HtpProfile::global().setGraphResident(mask);
    char names[128];
    std::fprintf(stderr,
                 "[HTP] graph: description n_ops=%u resident=%s "
                 "moe_ops=%zu\n",
                 n_ops, htp_graph_kinds_str(mask, names, sizeof(names)),
                 moe_ops_.size());
    return true;
  }

  /** [#132 Part B] The model's Q4_0 weights of the decode list's FC,
   *  DENSE_FFN and LM_HEAD ops, in list order (lfm2_moe_causallm.cpp at
   *  load): kept as pointers here, converted and registered at graph init
   *  (bindQ4m1). False when no such kind is resident. */
  bool add_decode_graph_q4_0(const void *data, unsigned K, unsigned N,
                             bool canonical) override {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    if (graph_words_.empty() || (resident_mask_ & HTP_GRAPH_KINDS_Q4M1) == 0u)
      return false;
    if (graph_inited_) {
      throw std::runtime_error(
        "add_decode_graph_q4_0: the graph is already initialised");
    }
    q4_pending_.push_back({data, K, N, canonical});
    return true;
  }

  bool set_fc_wh_source(const void *data, unsigned K, unsigned N,
                        bool affine = false) override {
    if (!data || K == 0 || N == 0 || K % WH_TILE || N % WH_TILE)
      throw std::invalid_argument("QS4CX_WH FC requires complete WH tiles");
    std::lock_guard<std::mutex> lock(handle_mutex_);
    FcWhEntry entry{};
    entry.K = K;
    entry.N = N;
    entry.key = reinterpret_cast<uintptr_t>(data);
    entry.bytes = whBytes(K, N) + (affine ? 12u : 8u) * size_t(N);
    auto inserted = direct_wh_.emplace(data, entry);
    if (!inserted.second &&
        (inserted.first->second.K != K || inserted.first->second.N != N ||
         inserted.first->second.bytes != entry.bytes))
      throw std::invalid_argument("QS4CX_WH FC source shape changed");
    return true;
  }

  /** [#260] compute_ops.h: a QS4CX FC / DENSE_FFN weight of the decode
   *  list, bound (bindQ4m1) to the WH handles get_or_register_fc /
   *  get_or_register_dense make from its codes and scales -- the ones the
   *  fused prefill calls use. */
  bool add_decode_graph_qs4cx(const void *data, const float *scale, unsigned K,
                              unsigned N) override {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    if (graph_words_.empty() || (resident_mask_ & HTP_GRAPH_KINDS_Q4M1) == 0u)
      return false;
    if (graph_inited_) {
      throw std::runtime_error(
        "add_decode_graph_qs4cx: the graph is already initialised");
    }
    q4_pending_.push_back({data, K, N, false, scale});
    return true;
  }

  /** [plan 201 S4] compute_ops.h: one f32 parameter of op @a op handed at
   *  load by the weight's name. The length is checked against the record
   *  now (hexkl_graph.c's graph_param_len), the pointer kept and bound
   *  after graph_init; the op's hook then binds nothing. */
  bool set_decode_graph_param(unsigned op, unsigned which, const float *data,
                              unsigned n) override {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    if (graph_words_.empty())
      return false;
    if (graph_inited_)
      throw std::runtime_error(
        "set_decode_graph_param: the graph is already initialised");
    if (op >= param_bound_.size() || data == nullptr)
      throw std::invalid_argument("set_decode_graph_param: op " +
                                  std::to_string(op) + " of " +
                                  std::to_string(param_bound_.size()));
    const htp_graph_op *rec = graphOp(op);
    uint32_t want = 0;
    switch (which) {
    case HTP_GRAPH_PARAM_GAMMA:
      want = rec->kind == HTP_OP_RMSNORM   ? rec->K
             : rec->kind == HTP_OP_QK_NORM ? 2u * rec->head_dim
                                           : 0u;
      break;
    case HTP_GRAPH_PARAM_ROUTER_W:
      want = rec->kind == HTP_OP_ROUTER_TOPK ? rec->K * rec->n_experts : 0u;
      break;
    case HTP_GRAPH_PARAM_ROUTER_BIAS:
      want = rec->kind != HTP_OP_ROUTER_TOPK ? 0u
             : rec->eps_bits != 0u           ? rec->K + rec->n_experts
                                             : rec->n_experts;
      break;
    case HTP_GRAPH_PARAM_ROPE_TABLE:
      want = rec->kind == HTP_OP_ROPE ? graph_words_[6] * rec->head_dim : 0u;
      break;
    default:
      break;
    }
    if (want == 0u || n != want)
      throw std::invalid_argument(
        "set_decode_graph_param: op " + std::to_string(op) + " (" +
        htp_graph_kind_name(rec->kind) + ") parameter " +
        std::to_string(which) + " of " + std::to_string(n) + " floats, want " +
        std::to_string(want));
    load_params_.push_back({op, which, data, n});
    param_bound_[op] = 1;
    return true;
  }

  /** [plan 201 S4] compute_ops.h: the model's gated activation; read once
   *  by sendMoeOptsOnce, so it is set before the first MoE call. */
  void set_moe_geglu(bool on) override {
    if (moeOptsSent().load() && moeGeglu().load() != on)
      throw std::runtime_error("set_moe_geglu: the MoE options were already "
                               "sent with the other activation");
    moeGeglu().store(on);
  }

  /** [#132 Part B] One weight's rows [r0, r0 + rows) as a Q4M1 handle:
   *  canonical block_q4_0 (unpacked from the CPU's repack unless the
   *  model says it is canonical already, as the tied lm_head is), the
   *  Q4M1 reorder, q4m1_register. The same nibbles and scales the CPU
   *  reads (plan 132 section 2), so no format tag moves. */
  uint32_t registerQ4m1(remote_handle64 session, const uint8_t *canonical,
                        uint32_t K, uint32_t r0, uint32_t rows) {
    const size_t row_bytes = static_cast<size_t>(K / 32u) * Q4_CPU_BLOCK_BYTES;
    std::vector<uint8_t> q4m1(q4m1_bytes(K, rows));
    q4m1_from_q4_0(canonical + r0 * row_bytes, K, rows, q4m1.data());
    uint32_t h = 0;
    if (e2e_) {
      // [#132 Part B E3] into the FC set's own arena chunks (place /
      // tryChunk / arena_attach), written before the attach; the heap
      // holds no weight (the #178 heap rule)
      E2eState &e = *e2e_st_;
      const uint32_t bytes = static_cast<uint32_t>(q4m1.size());
      uint32_t chunk = 0, off = 0;
      if (!placeOn(e.h1, CDSP_DOMAIN_ID, e.arena, e.arena_cap, bytes,
                   q4m1_left_, &chunk, &off)) {
        // [#211] on the 8B model every expert resident (3696 MiB) leaves
        // no room for the FC set (448 MiB) under the PD's 3840
        size_t mapped = arenaBytes();
        for (const ArenaChunk &c : e.arena)
          mapped += c.buf->size();
        throw std::runtime_error(
          "NNTR_HTP_E2E=1: no room for the FC set beside the resident "
          "experts (mapped=" +
          std::to_string(mapped >> 20) + " MiB, " + arena_fail_ +
          "); set NNTR_MOE_CACHE_EXPERTS=<C> (28 on the S25 / S26)");
      }
      std::memcpy(e.arena[chunk].buf->data() + off, q4m1.data(), bytes);
      q4m1_left_ = q4m1_left_ > bytes ? q4m1_left_ - bytes : 0u;
      e.attach_bytes += bytes;
      const int err =
        nntr_hvx_q4m1_attach(e.h1, e.arena[chunk].dsp_id, off, K, rows, &h);
      if (err != AEE_SUCCESS) {
        throw std::runtime_error("nntr_hvx_q4m1_attach(K=" + std::to_string(K) +
                                 " N=" + std::to_string(rows) +
                                 " off=" + std::to_string(off) +
                                 ") failed: " + graphErr(err));
      }
      e.q4m1.push_back(h);
      return h;
    }
    const int err = nntr_hvx_q4m1_register(session, K, rows, q4m1.data(),
                                           static_cast<int>(q4m1.size()), &h);
    if (err != AEE_SUCCESS) {
      throw std::runtime_error("nntr_hvx_q4m1_register(K=" + std::to_string(K) +
                               " N=" + std::to_string(rows) +
                               ") failed: " + graphErr(err));
    }
    q4m1_handles_.push_back(h);
    return h;
  }

  /** [#132 Part B] Frees every Q4M1 handle registered for the graph (after
   *  graph_release, or when bindQ4m1 / graph_init failed half-way). */
  void releaseQ4m1(remote_handle64 session) {
    shared_lm_heads_.clear();
    for (uint32_t h : q4m1_handles_)
      nntr_hvx_q4m1_release(session, h);
    q4m1_handles_.clear();
    if (e2e_st_) { // [#132 Part B E3] the E2E slots (the arena stays mapped)
      for (uint32_t h : e2e_st_->q4m1)
        nntr_hvx_q4m1_release(e2e_st_->h1, h);
      e2e_st_->q4m1.clear();
    }
  }

  /** [#132 Part B] Binds q4_pending_ to the resident Q4M1 ops in list
   *  order before graph_init checks them: an FC takes weights of its K
   *  until their widths make its N (q | k | v are three), a DENSE_FFN up,
   *  gate and down, the LM_HEAD the tied table in kLmHeadSliceRows slices.
   *  Throws when the model's weights and the list disagree. */
  void bindQ4m1(remote_handle64 session) {
    size_t next = 0;
    std::vector<uint8_t> canon;
    // [#225] With the FC WH sidecar open, the FC and DENSE_FFN ops bind the
    // handles the prefill registered from it (HTP_GRAPH_FEED_WH): one copy
    // of those weights for both. The tied lm_head stays Q4M1.
    // [#234 P4] Lever L1 (HTP_GRAPH_FEED_NATIVE, NNTR_HTP_PPL_LEVERS=2)
    // asks for the native Q4 kernels on every Q4M1 kind: the sidecar's
    // handles then stay out of the decode list.
    bool native = false;
    for (uint32_t i = 0; i < static_cast<uint32_t>(stretch_start_.size()); ++i)
      native |= (graphOp(i)->feed & HTP_GRAPH_FEED_NATIVE) != 0u;
    const bool wh = !native && [this] {
      std::lock_guard<std::mutex> lock(handle_mutex_);
      return fcwh_fd_ >= 0;
    }();
    uint32_t wh_handles = 0;
    q4m1_left_ = 0; // [#132 Part B E3] what the FC arena chunks are sized for
    if (!wh) {
      for (const Q4Pending &p : q4_pending_)
        if (p.scale == nullptr) // [#260] a QS4CX weight binds WH handles
          q4m1_left_ += q4m1_bytes(p.K, p.N) + 4096u;
    } else { // [#225] the LM_HEAD's slices only (a QS4CX head binds WH)
      for (uint32_t i = 0; i < static_cast<uint32_t>(stretch_start_.size());
           ++i) {
        const htp_graph_op *op = graphOp(i);
        if (op->resident && op->kind == HTP_OP_LM_HEAD &&
            !(q4_pending_.size() && q4_pending_.back().scale != nullptr))
          q4m1_left_ +=
            q4m1_bytes(op->K, op->N) +
            4096u * ((op->N + kLmHeadSliceRows - 1u) / kLmHeadSliceRows);
      }
    }
    auto pending = [&](uint32_t op, uint32_t K,
                       uint32_t N) -> const Q4Pending & {
      if (next >= q4_pending_.size())
        throw std::runtime_error("set_decode_graph_desc: op " +
                                 std::to_string(op) + " (" +
                                 htp_graph_kind_name(graphOp(op)->kind) +
                                 ") has no Q4_0 weight left "
                                 "(the model handed " +
                                 std::to_string(q4_pending_.size()) + ")");
      Q4Pending &p = q4_pending_[next++];
      // The untied Q4_0 head still runs on the CPU during every prefill.
      // Decode residency alone does not make its host weight disposable. A
      // QS4CX head's prefill row runs on its WH slices (lm_head_shared_fp32).
      p.keep_host = graphOp(op)->kind == HTP_OP_LM_HEAD && p.scale == nullptr;
      if (p.K != K || (N != 0u && p.N != N))
        throw std::runtime_error(
          "set_decode_graph_desc: op " + std::to_string(op) + " wants a " +
          std::to_string(K) + " x " + (N ? std::to_string(N) : "*") +
          " Q4_0 weight, the model's next is " + std::to_string(p.K) + " x " +
          std::to_string(p.N));
      return p;
    };
    auto take = [&](uint32_t op, uint32_t K, uint32_t N) -> const uint8_t * {
      const Q4Pending &p = pending(op, K, N);
      if (p.scale != nullptr) // [#260] no Q4M1 from QS4CX (contract s. 2)
        throw std::runtime_error("set_decode_graph_desc: op " +
                                 std::to_string(op) +
                                 " takes Q4M1, its weight is QS4CX");
      const size_t bytes =
        static_cast<size_t>(p.N) * (p.K / 32u) * Q4_CPU_BLOCK_BYTES;
      if (p.canonical)
        return static_cast<const uint8_t *>(p.data);
      canon.resize(bytes);
      nntrainer::unpack_q4_0(p.data, canon.data(), bytes, p.N, p.K);
      return canon.data();
    };
    // [#225] the prefill's handles of a pending weight (registered at the
    // load's warm-ups, or here if its layer was not keyed)
    auto key = [](const Q4Pending &p) { return const_cast<void *>(p.data); };
    const uint32_t n_ops = static_cast<uint32_t>(stretch_start_.size());
    for (uint32_t i = 0; i < n_ops; ++i) {
      htp_graph_op *op = htp_graph_op_at(graph_words_.data(), i);
      if (!op->resident ||
          (HTP_GRAPH_KINDS_Q4M1 & HTP_GRAPH_KIND_BIT(op->kind)) == 0u)
        continue;
      op->feed &= ~HTP_GRAPH_FEED_WH; // [#234 P4] a re-bind sets it anew
      uint32_t parts = 0;
      // [#260] a QS4CX weight takes the WH path with or without a sidecar;
      // lever L1 asks for the native Q4 kernels, which QS4CX cannot feed
      const bool qs4cx_op =
        next < q4_pending_.size() && q4_pending_[next].scale != nullptr;
      if (qs4cx_op && native)
        throw std::runtime_error(
          "set_decode_graph_desc: lever L1 (NNTR_HTP_PPL_LEVERS=2) takes "
          "Q4_0 weights, op " +
          std::to_string(i) + "'s is QS4CX");
      const bool wh_op = wh || qs4cx_op;
      if (wh_op && op->kind == HTP_OP_DENSE_FFN) {
        const Q4Pending &u = pending(i, op->K, op->N);
        const Q4Pending &g = pending(i, op->K, op->N);
        const Q4Pending &d = pending(i, op->N, op->N_out);
        const DenseHandles &dh =
          get_or_register_dense(key(u), key(g), key(d), session, op->K, op->N,
                                op->N_out, u.scale, g.scale, d.scale);
        if (dh.h_gu.size() > HTP_GRAPH_WH_DENSE_MAX_CHUNKS)
          throw std::runtime_error(
            "set_decode_graph_desc: DENSE_FFN op " + std::to_string(i) +
            " has " + std::to_string(dh.h_gu.size()) +
            " chunks, the M=1 kernel takes " +
            std::to_string(HTP_GRAPH_WH_DENSE_MAX_CHUNKS));
        for (size_t c = 0; c < dh.h_gu.size(); ++c) {
          op->h_gu[parts] = dh.h_gu[c];
          op->h_dn[parts++] = dh.h_dn[c];
        }
        op->feed |= HTP_GRAPH_FEED_WH;
        wh_handles += 2u * parts;
      } else if (wh_op && op->kind == HTP_OP_FC) {
        uint32_t sum = 0;
        while (sum < op->N) {
          const Q4Pending &p = pending(i, op->K, 0u);
          const FcHandles &fh =
            get_or_register_fc(key(p), session, p.K, p.N, p.scale);
          for (uint32_t h : fh.handles) {
            if (parts == HTP_GRAPH_MAX_PARTS)
              throw std::runtime_error("set_decode_graph_desc: FC op " +
                                       std::to_string(i) + " has over " +
                                       std::to_string(HTP_GRAPH_MAX_PARTS) +
                                       " WH parts");
            op->h_gu[parts++] = h;
          }
          sum += p.N;
        }
        if (sum != op->N)
          throw std::runtime_error(
            "set_decode_graph_desc: FC op " + std::to_string(i) +
            " is N=" + std::to_string(op->N) + ", its weights sum to " +
            std::to_string(sum));
        op->feed |= HTP_GRAPH_FEED_WH;
        wh_handles += parts;
      } else if (op->kind == HTP_OP_DENSE_FFN) {
        op->h_gu[0] = registerQ4m1(session, take(i, op->K, op->N), op->K, 0u,
                                   op->N); // up
        op->h_gu[1] = registerQ4m1(session, take(i, op->K, op->N), op->K, 0u,
                                   op->N); // gate
        op->h_dn[0] = registerQ4m1(session, take(i, op->N, op->N_out), op->N,
                                   0u, op->N_out);
        parts = 3u;
      } else if (op->kind == HTP_OP_LM_HEAD && qs4cx_op) {
        // An untied QS4CX head (lmhead_dtype QS4CX): its columns in at most
        // HTP_GRAPH_MAX_PARTS WH slices (4448 columns at a 262144 vocabulary),
        // each converted and placed as an FC slice is (registerRm, the arena),
        // bound to the op on HTP_GRAPH_FEED_WH. The same handles serve the
        // prefill's last row (lm_head_shared_fp32), so the host copy is
        // released with the FC set below (keep_host false).
        const Q4Pending &p = pending(i, op->K, op->N);
        if (op->N % 32u != 0u)
          throw std::runtime_error("set_decode_graph_desc: LM_HEAD op " +
                                   std::to_string(i) +
                                   " is N=" + std::to_string(op->N) +
                                   ", not a multiple "
                                   "of 32");
        // A registered handle must fit VTCM (hexkl_weight_u8i4_check: its
        // bake stages there), so a slice is at most 6 MiB of WH bytes --
        // 4448 columns at K = 2816, 59 slices -- and HTP_GRAPH_MAX_PARTS
        // bounds the count. ponytail: 6 MiB assumes the 8 MiB VTCM of v79 /
        // v81; a smaller VTCM wants the session's size here.
        const uint32_t fit = std::max(
          32u,
          static_cast<uint32_t>((size_t(6) << 20) / (op->K / 2u)) / 32u * 32u);
        const uint32_t cols =
          std::min(fit, ((op->N / 32u + HTP_GRAPH_MAX_PARTS - 1u) /
                         HTP_GRAPH_MAX_PARTS) *
                          32u);
        if ((op->N + cols - 1u) / cols > HTP_GRAPH_MAX_PARTS)
          throw std::runtime_error(
            "set_decode_graph_desc: LM_HEAD op " + std::to_string(i) + " (" +
            std::to_string(op->K) + " x " + std::to_string(op->N) +
            ") needs more than " + std::to_string(HTP_GRAPH_MAX_PARTS) +
            " WH slices of " + std::to_string(cols) + " columns");
        const uint64_t t0 = HtpProfile::nowUs();
        if (const FcWhEntry *image = fcwhFind(p.data, op->K, op->N)) {
          std::lock_guard<std::mutex> lock(handle_mutex_);
          for (uint32_t c0 = 0; c0 < op->N; c0 += cols) {
            const uint32_t cn = std::min(cols, op->N - c0);
            op->h_gu[parts++] =
              registerFcWh(static_cast<char *>(key(p)) + c0, session,
                           {{image, c0, cn}}, 0, op->K);
          }
        } else {
          bindLmHeadWh(session, p.data, p.scale, op->K, op->N, cols, op->h_gu,
                       &parts);
        }
        op->feed |= HTP_GRAPH_FEED_WH;
        wh_handles += parts;
        SharedLmHead &sh = shared_lm_heads_[p.data];
        sh.K = op->K;
        sh.N = op->N;
        sh.wh = true;
        sh.h.assign(op->h_gu, op->h_gu + parts);
        std::fprintf(stderr,
                     "[HTP] lm_head on the NPU: %u x %u QS4CX as %u WH slices "
                     "of %u (prefill row and decode), %.1f MiB, placed in "
                     "%.0f ms\n",
                     op->N, op->K, parts, cols,
                     static_cast<double>(op->K) * op->N / 2 / (1u << 20),
                     static_cast<double>(HtpProfile::nowUs() - t0) / 1000.0);
      } else if (op->kind == HTP_OP_LM_HEAD) {
        const Q4Pending &p = pending(i, op->K, op->N);
        if (p.scale)
          throw std::invalid_argument("LM_HEAD requires Q4_0 weights");
        const size_t row_bytes =
          static_cast<size_t>(op->K / 32u) * Q4_CPU_BLOCK_BYTES;
        // Unpack one slice at a time; a full vocabulary-sized canonical copy
        // would coexist with both the CPU weight and the DSP arena.
        for (uint32_t r0 = 0; r0 < op->N; r0 += kLmHeadSliceRows) {
          const uint32_t rows = std::min(kLmHeadSliceRows, op->N - r0);
          const auto *w = static_cast<const uint8_t *>(p.data) + r0 * row_bytes;
          if (!p.canonical) {
            canon.resize(rows * row_bytes);
            nntrainer::unpack_q4_0(w, canon.data(), canon.size(), rows, op->K);
            w = canon.data();
          }
          op->h_gu[parts++] = registerQ4m1(session, w, op->K, 0u, rows);
        }
        auto shared = shared_lm_heads_.find(p.data);
        if (!p.canonical && shared != shared_lm_heads_.end()) {
          shared->second.K = op->K;
          shared->second.N = op->N;
          shared->second.h.assign(op->h_gu, op->h_gu + parts);
          releaseArmSource(const_cast<void *>(p.data), op->N * row_bytes);
          q4_pending_[next - 1].dropped = true;
          std::fprintf(stderr,
                       "[HTP] LM head shares %u DSP handles for "
                       "prefill/decode; released %zu MiB host source\n",
                       parts, (op->N * row_bytes) >> 20);
        }
      } else {
        uint32_t sum = 0;
        while (sum < op->N && parts < HTP_GRAPH_MAX_PARTS) {
          const uint8_t *w = take(i, op->K, 0u);
          const uint32_t n = q4_pending_[next - 1].N;
          op->h_gu[parts++] = registerQ4m1(session, w, op->K, 0u, n);
          sum += n;
        }
        if (sum != op->N)
          throw std::runtime_error(
            "set_decode_graph_desc: FC op " + std::to_string(i) +
            " is N=" + std::to_string(op->N) + ", its weights sum to " +
            std::to_string(sum));
      }
      op->n_experts = parts;
    }
    if (next != q4_pending_.size())
      throw std::runtime_error(
        "set_decode_graph_desc: the model handed " +
        std::to_string(q4_pending_.size()) + " Q4_0 weights, the list's " +
        "FC / DENSE_FFN / LM_HEAD ops took " + std::to_string(next));
    std::fprintf(stderr, "[HTP] graph: q4m1 weights=%zu handles=%zu feed=%s",
                 q4_pending_.size(),
                 e2e_ ? e2e_st_->q4m1.size() : q4m1_handles_.size(),
                 q4m1FeedName());
    // [#225] the FC / DENSE_FFN ops on the sidecar's WH handles (absent:
    // the line as before)
    if (wh_handles != 0u)
      std::fprintf(stderr, " wh_handles=%u", wh_handles);
    std::fputc('\n', stderr);
    // [#4416 mem] the QS4CX FC / dense copies are read by nothing after
    // this: the prefill runs them on the WH handles just registered and
    // the token on its arena. Handing them back here, not at the first
    // decode token, frees 0.8 GiB for the prefill (S26 Ultra p2048:
    // mem_available at the prefill's end 1.5 GiB) -- the page cache keeps
    // that much more of the expert file. The Q4_0 lm_head stays: the
    // prefill's CPU lm_head (lmhead_engine cpu) reads it.
    if (wh_handles != 0u)
      dropHostFc(/*qs4cx_only=*/true);
  }

  /** [#130] The op record of @a op in the description (the words are
   *  the model's; the resident bits and shapes do not change after
   *  set_decode_graph_desc). */
  const htp_graph_op *graphOp(uint32_t op) const {
    return htp_graph_op_cat(graph_words_.data(), op);
  }

  /** [#130] graph_set_param once per (op, which); throws on any error
   *  (contract section 2: no silent fallback). */
  void setParam(remote_handle64 session, uint32_t op, uint32_t which,
                const float *data, unsigned n, unsigned want,
                const char *what) {
    if (data == nullptr || n != want) {
      throw std::runtime_error(std::string("decode_op_fp32: ") + what +
                               " of op " + std::to_string(op) + ": " +
                               std::to_string(n) + " floats, want " +
                               std::to_string(want));
    }
    const int err =
      nntr_hvx_graph_set_param(session, op, which, data, static_cast<int>(n));
    if (err != AEE_SUCCESS) {
      throw std::runtime_error(std::string("nntr_hvx_graph_set_param(") + what +
                               ") failed at op " + std::to_string(op) + ": " +
                               graphErr(err));
    }
  }

  /** [#130] The per-token hook (compute_ops.h): the counter of its kind
   *  names the op, the stretch tables say what to run. Hooks fire only at
   *  one row, from the model's thread, in list order within a token; the
   *  counters reset when pos changes, so every token re-synchronises and
   *  a prefill (rows > 1, no hooks) never disturbs them. */
  int decode_op_fp32(unsigned kind, unsigned pos, const float *in,
                     unsigned in_len, float *out, unsigned out_len,
                     const float *param, unsigned param_len, const float *state,
                     unsigned state_len, float eps) override {
    if (!forwardSwitch() || graph_words_.empty() || kind >= HTP_OP_KIND_N ||
        (resident_mask_ & HTP_GRAPH_KIND_BIT(kind)) == 0u)
      return 0;
    if (pos != cur_pos_) {
      if (pending_op_ != HTP_GRAPH_NO_OP) {
        throw std::runtime_error(
          "decode_op_fp32: pos moved to " + std::to_string(pos) +
          " with the stretch input of op " + std::to_string(pending_op_) +
          " pending (its last hook never came)");
      }
      std::fill(std::begin(kind_next_), std::end(kind_next_), 0u);
      cur_pos_ = pos;
      // Decided once per row, so a row that starts before the MoE handles
      // are all bound (a one-token prompt binds them as it goes) runs on
      // the CPU end to end: a hook after the last MoE call would otherwise
      // resolve to the first op of its kind. The subset case
      // (moe_htp_layers) stays 0 for good; gemm_qs4cx_moe_layer_fp32
      // warned once.
      row_bound_ = moe_bound_ == moe_ops_.size();
    }
    if (!row_bound_)
      return 0;
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    ensureGraphInit(session);
    const std::vector<uint32_t> &ops = kind_ops_[kind];
    if (kind_next_[kind] >= ops.size()) {
      throw std::runtime_error(std::string("decode_op_fp32: more ") +
                               htp_graph_kind_name(kind) + " hooks at pos " +
                               std::to_string(pos) + " than the list's " +
                               std::to_string(ops.size()) + " ops");
    }
    const uint32_t op = ops[kind_next_[kind]++];
    const htp_graph_op *rec = graphOp(op);
    if (rec->kind == HTP_OP_RMSNORM || rec->kind == HTP_OP_QK_NORM) {
      uint32_t bits;
      std::memcpy(&bits, &eps, sizeof(bits));
      if (bits != rec->eps_bits) {
        throw std::runtime_error("decode_op_fp32: op " + std::to_string(op) +
                                 " epsilon differs from the description's");
      }
    }
    switch (kind) {
    case HTP_OP_RMSNORM:
      if (!param_bound_[op]) {
        setParam(session, op, HTP_GRAPH_PARAM_GAMMA, param, param_len, rec->K,
                 "gamma");
        param_bound_[op] = 1;
      }
      break;
    case HTP_OP_CONV1D_GATE:
      if (!param_bound_[op]) {
        setParam(session, op, HTP_GRAPH_PARAM_CONV_W, param, param_len,
                 3u * rec->N, "conv_w");
        param_bound_[op] = 1;
      }
      // The DSP advances the state itself; re-seed from the CPU's copy
      // after any jump (the first token after a prefill, which refreshed
      // it; plan 130 section 3.3).
      if (conv_next_pos_[op] != pos) {
        setParam(session, op, HTP_GRAPH_PARAM_CONV_STATE, state, state_len,
                 2u * rec->N, "conv state");
        if (dumpAllDir() != nullptr)
          dumpAllFile(std::string(dumpAllDir()) + "/convstate_" +
                        std::to_string(op) + ".f32",
                      state, state_len);
      }
      conv_next_pos_[op] = pos + 1u;
      break;
    case HTP_OP_QK_NORM:
      if (!param_bound_[op]) {
        setParam(session, op, HTP_GRAPH_PARAM_GAMMA, param, param_len,
                 2u * rec->head_dim, "q | k gamma");
        param_bound_[op] = 1;
      }
      if (in == nullptr || in_len != rec->K) {
        throw std::runtime_error("decode_op_fp32: QK_NORM row of " +
                                 std::to_string(in_len) + " floats, want " +
                                 std::to_string(rec->K));
      }
      break;
    case HTP_OP_ATTN_M1: {
      const uint32_t ord = attn_ordinal_[op];
      // [plan 201 S4] the layer's RoPE table goes to its own ROPE op,
      // max_seq x head_dim: Gemma 4's layers differ in theta and rotary
      // variant even at one head_dim; the DSP shares equal tables, so
      // LFM2's layers still hold one
      uint32_t r = op;
      while (r > 0u && graphOp(r)->kind != HTP_OP_ROPE)
        --r;
      if (graphOp(r)->kind != HTP_OP_ROPE || graphOp(r)->layer != rec->layer)
        throw std::runtime_error("decode_op_fp32: ATTN_M1 op " +
                                 std::to_string(op) + " has no ROPE op");
      if (!param_bound_[r]) {
        if (param == nullptr || param_len == 0u) {
          --kind_next_[kind]; // the layer builds its table and calls again
          return 3;
        }
        // [#4416 D6] a table equal to one a bound ROPE op of this head_dim
        // holds is shared by name: the bytes cross once a distinct table
        // (S26 Ultra, p1024: 30 binds were 439 ms of the first token)
        const uint32_t want = graph_words_[6] * rec->head_dim;
        uint32_t src = HTP_GRAPH_NO_OP;
        for (const auto &t : rope_tables_)
          if (t.second.size() == param_len && param_len == want &&
              std::memcmp(t.second.data(), param, param_len * sizeof(float)) ==
                0) {
            src = t.first;
            break;
          }
        if (src != HTP_GRAPH_NO_OP) {
          const int err = nntr_hvx_graph_share_param(
            session, r, HTP_GRAPH_PARAM_ROPE_TABLE, src);
          if (err != AEE_SUCCESS)
            throw std::runtime_error(
              "nntr_hvx_graph_share_param(RoPE table) failed at op " +
              std::to_string(r) + ": " + graphErr(err));
        } else {
          setParam(session, r, HTP_GRAPH_PARAM_ROPE_TABLE, param, param_len,
                   want, "RoPE table");
          rope_tables_.emplace_back(
            r, std::vector<float>(param, param + param_len));
        }
        param_bound_[r] = 1;
      }
      // The DSP cache must hold rows [0, pos): seed it from the layer's
      // copy after any jump (the first token after a prefill, a second
      // prompt). pos 0 needs nothing; the kernel rewinds.
      // (rows from the layer's base on: set_decode_kv_window)
      const uint32_t row = pos - rec->n_experts;
      if (pos != 0 && kv_len_[ord] != row) {
        seed_ordinal_ = ord;
        --kind_next_[kind]; // the layer calls again for this same op
        return 2;
      }
      kv_len_[ord] = row + 1u;
      break;
    }
    case HTP_OP_ADD:
    case HTP_OP_LM_HEAD: // [#132 Part B] weights bound at init
      break;
    case HTP_OP_ROUTER_TOPK:
      // [#132] the gate weight [K][E] and the expert bias, once
      if (!param_bound_[op]) {
        setParam(session, op, HTP_GRAPH_PARAM_ROUTER_W, param, param_len,
                 rec->K * rec->n_experts, "router weight");
        setParam(session, op, HTP_GRAPH_PARAM_ROUTER_BIAS, state, state_len,
                 rec->n_experts, "router bias");
        param_bound_[op] = 1;
      }
      break;
    default:
      return 0;
    }
    runStretchOp(session, op, pos, in, in_len, out, out_len, nullptr, nullptr,
                 nullptr);
    return 1;
  }

  /** [#132] Every hook's one dispatch over its op's stretch [s, e) (plan
   *  132 section 3.4): the first op keeps its input row (and, for a MOE
   *  op, the ARM's routing) pending and returns; a mid op checks that the
   *  row is there and returns; the last op -- or a sole one -- runs the
   *  stretch on the pending row, or on its own when it is also the first.
   *  The layer skips its kernel either way: only the last op's @a out is
   *  written, and the CPU tensors between are stale by design (the DSP
   *  holds the stream: slot 0 the residual, the routing in the router's
   *  record). */
  void runStretchOp(remote_handle64 session, uint32_t op, uint32_t pos,
                    const float *in, unsigned in_len, float *out,
                    unsigned out_len, const std::vector<unsigned int> *ri,
                    const std::vector<unsigned int> *rc,
                    const std::vector<float> *rw) {
    static const std::vector<unsigned int> no_u;
    static const std::vector<float> no_f;
    const uint32_t s = stretch_start_[op], e = stretch_end_[op];
    // The level-3 repeat, the MoE dump and the M==1 MoE profile row are
    // for the sole MOE stretch only: an ADD in a stretch moves slot 0, so
    // a repeat would add twice, and C (KINDS=MOE) stays comparable to A.
    const bool sole_moe = graphOp(s)->kind == HTP_OP_MOE && e == s + 1u;
    // ROPE has no layer hook: a [ROPE ATTN_M1] stretch starts at the
    // attention hook, whose row is ROPE's input
    uint32_t first = s;
    while (graphOp(first)->kind == HTP_OP_ROPE)
      ++first;
    if (first == op) {
      if (pending_op_ != HTP_GRAPH_NO_OP) {
        throw std::runtime_error(
          "decode_op_fp32: op " + std::to_string(op) +
          " starts a stretch while op " + std::to_string(pending_op_) +
          "'s row is pending (its last hook never came)");
      }
      if (e == op + 1u) {
        invokeForward(session, s, e, pos, ri ? *ri : no_u, rc ? *rc : no_u,
                      rw ? *rw : no_f, in, in_len, out, out_len, sole_moe);
        return;
      }
      if (in == nullptr) {
        throw std::runtime_error("decode_op_fp32: op " + std::to_string(op) +
                                 " starts a stretch with no row");
      }
      pending_in_.assign(in, in + in_len);
      pending_ri_ = ri ? *ri : no_u;
      pending_rc_ = rc ? *rc : no_u;
      pending_rw_ = rw ? *rw : no_f;
      pending_op_ = s;
      return;
    }
    if (pending_op_ != s) {
      throw std::runtime_error(
        "decode_op_fp32: " +
        std::string(htp_graph_kind_name(graphOp(op)->kind)) + " op " +
        std::to_string(op) + " without the row of its stretch's first op " +
        std::to_string(s) + " (its hook never came)");
    }
    if (e == op + 1u) {
      invokeForward(session, s, e, pos, pending_ri_, pending_rc_, pending_rw_,
                    pending_in_.data(),
                    static_cast<unsigned>(pending_in_.size()), out, out_len,
                    sole_moe);
      clearPending();
    }
  }

  void clearPending() {
    pending_op_ = HTP_GRAPH_NO_OP;
    pending_in_.clear();
    pending_ri_.clear();
    pending_rc_.clear();
    pending_rw_.clear();
  }

  /** [#132 Part B] compute_ops.h: with a Q4M1 kind resident every kind
   *  is (set_decode_graph_desc's rule), so the token is one stretch; once
   *  op 0's hook at @a pos has kept the row, the DSP owns the whole row
   *  and the CPU's FC outputs until the lm_head hook are never read. */
  bool decode_row_resident(unsigned pos) override {
    const bool on = (resident_mask_ & HTP_GRAPH_KINDS_Q4M1) != 0u &&
                    row_bound_ && pos == cur_pos_ &&
                    pending_op_ != HTP_GRAPH_NO_OP;
    cpu_fc_skipped_ += on;
    return on;
  }

  /** The sliding layers' caches as a window of the context: their DSP
   *  cache is registered at `rows` rows and their ATTN_M1 records carry the
   *  base (n_experts, unused by the kind), which graph_op_attn_m1 subtracts
   *  from the token's position. Before the graph's init, which sends the
   *  records and registers the caches. */
  void set_decode_kv_window(unsigned rows, unsigned base) override {
    if (graph_words_.empty())
      return;
    if (attn_registered_)
      throw std::runtime_error(
        "set_decode_kv_window: the decode attention caches are registered");
    for (uint32_t op : kind_ops_[HTP_OP_ATTN_M1]) {
      htp_graph_op *rec = htp_graph_op_at(graph_words_.data(), op);
      if (rec->top_k != 0u) // a sliding layer (top_k: its window)
        rec->n_experts = base;
    }
    attn_window_rows_ = (rows + 63u) / 64u * 64u;
  }

  bool decode_kv_seed_fp32(unsigned n_rows, const float *k_rows,
                           const float *v_rows) override {
    return seedRows(n_rows, k_rows, v_rows, nullptr, nullptr);
  }

  /** [#4416 D6] the rows as fp16 bits, widened on the DSP (S26 Ultra:
   *  the CPU's fp16 -> f32 pass was 145 ms of the p1024 first token, 288
   *  at p2048, and the staged bytes halve) */
  bool decode_kv_seed_f16(unsigned n_rows, const uint16_t *k_rows,
                          const uint16_t *v_rows) override {
    return seedRows(n_rows, nullptr, nullptr, k_rows, v_rows);
  }

  bool seedRows(unsigned n_rows, const float *k_rows, const float *v_rows,
                const uint16_t *k16, const uint16_t *v16) {
    if (seed_ordinal_ == HTP_GRAPH_NO_OP) {
      throw std::runtime_error(
        "decode_kv_seed_fp32: no attention hook asked for a seed");
    }
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    // [plan 201 S4] the layer's own shape and cache
    const htp_graph_op *rec = graphOp(attn_op_[seed_ordinal_]);
    const int n = static_cast<int>(n_rows * rec->n_kv * rec->head_dim);
    const uint32_t layer =
      attn_cache_ord_[seed_ordinal_] |
      (attn_cache_[seed_ordinal_] ? HTP_ATTN_KV_CACHE_B : 0u);
    const uint64_t t_seed = HtpProfile::nowUs(); // [#282 F]
    // Through two staging buffers: as heap arguments the driver copied the
    // rows into a buffer of its own every call (S26 Ultra, 2048 rows, 30
    // layers: 1,055-1,097 ms of seed RPCs as heap arguments, 973 staged).
    // ponytail: one more copy of the rows (2 ms a layer); the upgrade is
    // the caller converting its cache straight into these buffers.
    int err;
    {
      std::lock_guard<std::mutex> lock(invoke_mutex_);
      if (k16) {
        // [#4416 mem] the fp16 rows are the layer's CPU cache, an rpcmem
        // block (alloc_shared): FastRPC maps it by its fd, so they go as
        // they are -- no staging buffer, no copy (a heap source would be
        // copied by the driver, as the f32 rows below were before staging)
        err = nntr_hvx_attn_m1_kv_append_f16(session, layer, 0u, n_rows, k16, n,
                                             v16, n);
      } else {
        const size_t bytes = static_cast<size_t>(n) * sizeof(float);
        float *k_ion =
          reinterpret_cast<float *>(stage(act_pool_, bytes).data());
        float *v_ion = reinterpret_cast<float *>(stage(x2_pool_, bytes).data());
        std::memcpy(k_ion, k_rows, bytes);
        std::memcpy(v_ion, v_rows, bytes);
        err = nntr_hvx_attn_m1_kv_append(session, layer, 0u, n_rows, k_ion, n,
                                         v_ion, n);
      }
    }
    seed_rpc_us_ += HtpProfile::nowUs() - t_seed;
    ++seed_calls_;
    if (err != AEE_SUCCESS) {
      throw std::runtime_error("nntr_hvx_attn_m1_kv_append failed at layer " +
                               std::to_string(seed_ordinal_) + ", " +
                               std::to_string(n_rows) +
                               " rows: " + graphErr(err));
    }
    if (dumpAllDir() != nullptr && k_rows != nullptr) { /* f32 rows only */
      const std::string base =
        std::string(dumpAllDir()) + "/kvseed_" + std::to_string(seed_ordinal_);
      dumpAllFile(base + "_k.f32", k_rows, static_cast<size_t>(n));
      dumpAllFile(base + "_v.f32", v_rows, static_cast<size_t>(n));
    }
    kv_len_[seed_ordinal_] = n_rows;
    seed_ordinal_ = HTP_GRAPH_NO_OP;
    return true;
  }

  /** [#289] NNTR_HTP_KV_SEED_Q=1: attn_m1 is registered lazily and each
   *  layer is seeded from its quantized prefill cache on the DSP. */
  static bool kvSeedQ() {
    static const bool on = [] {
      const char *v = std::getenv("NNTR_HTP_KV_SEED_Q");
      return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
  }

  bool decode_kv_seed_kvq(unsigned n_rows, int kv_handle,
                          const float *k_rows) override {
    if (!kvSeedQ() || kv_handle < 0 || seed_ordinal_ == HTP_GRAPH_NO_OP)
      return false;
    {
      std::lock_guard<std::mutex> lock(kvq_mutex_);
      if (kvq_handles_.count(static_cast<uint32_t>(kv_handle)) == 0)
        return false;
    }
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const uint32_t layer =
      attn_cache_ord_[seed_ordinal_] |
      (attn_cache_[seed_ordinal_] ? HTP_ATTN_KV_CACHE_B : 0u);
    const uint64_t t_seed = HtpProfile::nowUs();
    const int err = nntr_hvx_attn_m1_seed_kvq(
      session, layer, static_cast<uint32_t>(kv_handle), n_rows, k_rows,
      k_rows
        ? static_cast<int>(n_rows * graphOp(attn_op_[seed_ordinal_])->n_kv *
                           graphOp(attn_op_[seed_ordinal_])->head_dim)
        : 0);
    seed_rpc_us_ += HtpProfile::nowUs() - t_seed;
    if (err != AEE_SUCCESS) {
      std::fprintf(stderr,
                   "[HTP] attn_m1_seed_kvq: layer %u not seeded on the DSP "
                   "(%s); the f32 seed takes it\n",
                   seed_ordinal_, graphErr(err).c_str());
      return false;
    }
    {
      std::lock_guard<std::mutex> lock(kvq_mutex_);
      kvq_handles_.erase(static_cast<uint32_t>(kv_handle)); // DSP released
    }
    ++seed_calls_;
    ++seed_kvq_;
    kv_len_[seed_ordinal_] = n_rows;
    seed_ordinal_ = HTP_GRAPH_NO_OP;
    return true;
  }

  /** Binds a layer's handles to the next unbound MoE op (call order =
   *  list order on the first pass) and returns the op that owns them. */
  uint32_t bindMoeOp(const std::vector<uint32_t> &h_gu,
                     const std::vector<uint32_t> &h_dn, unsigned K,
                     unsigned inter, unsigned N_out) {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    auto it = moe_op_by_handle_.find(h_gu[0]);
    if (it != moe_op_by_handle_.end())
      return it->second;
    if (moe_bound_ >= moe_ops_.size() || graph_inited_) {
      throw std::runtime_error(
        "set_decode_graph_desc: more MoE layers than the list's " +
        std::to_string(moe_ops_.size()) + " MoE ops");
    }
    const uint32_t idx = moe_ops_[moe_bound_];
    htp_graph_op *op = htp_graph_op_at(graph_words_.data(), idx);
    if (op->n_experts != h_gu.size() || op->K != K || op->N != inter ||
        op->N_out != N_out) {
      throw std::runtime_error(
        "set_decode_graph_desc: MoE op " + std::to_string(idx) +
        " expects experts=" + std::to_string(op->n_experts) +
        " K=" + std::to_string(op->K) + " inter=" + std::to_string(op->N) +
        " N_out=" + std::to_string(op->N_out) + ", the layer has " +
        std::to_string(h_gu.size()) + "/" + std::to_string(K) + "/" +
        std::to_string(inter) + "/" + std::to_string(N_out));
    }
    // [plan 201 S1] the handles go to the op's EXPERTS table, bound after
    // graph_init (ensureGraphInit), not into the record
    std::vector<float> &t = moe_tables_[moe_bound_];
    t.resize(2 * h_gu.size());
    std::memcpy(t.data(), h_gu.data(), h_gu.size() * sizeof(uint32_t));
    std::memcpy(t.data() + h_gu.size(), h_dn.data(),
                h_dn.size() * sizeof(uint32_t));
    moe_op_by_handle_.emplace(h_gu[0], idx);
    ++moe_bound_;
    return idx;
  }

  /** [plan 201 S1] compute_ops.h: a MoE layer's experts, layer order. */
  void set_decode_moe_experts(const std::vector<ExpertFileDesc> &all,
                              const ExpertPoolFn &pool) override {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    const uint32_t m = static_cast<uint32_t>(pool_descs_.size());
    for (uint32_t e = 0; e < all.size(); ++e)
      pool_where_[all[e].key_gu] = {m, e};
    pool_descs_.push_back(all);
    pool_fn_ = pool;
  }

  void set_decode_pool_grow(const std::function<void(size_t)> &grow) override {
    pool_grow_ = grow;
  }

  /** [#4416 mem] NNTR_MOE_CACHE_GROW=<n>: once the first token is done
   *  (the prefill's staging released, its quantized KV caches freed), the
   *  pool takes n more experts a layer; the slots are placed as the
   *  misses fill them (place), in the DSP address space those buffers
   *  held. At load the same C would not fit: S26 Ultra p2048, C 40 fails
   *  the prefill's FC call for DSP heap, C 44 the load. */
  void growPool() {
    if (!pool_grow_ || moe_ops_.empty() || pool_descs_.empty() ||
        pool_descs_[0].empty())
      return;
    unsigned per_layer;
    if (const char *v = std::getenv("NNTR_MOE_CACHE_GROW")) {
      per_layer = static_cast<unsigned>(std::atoi(v));
    } else {
      // Built in: towards htp_fixed::MOE_POOL_TARGET a layer, bounded by the
      // room left under the PD's mapping ceiling (a run past it dies with
      // fastrpc_mmap on a later miss, not here).
      const size_t layers = moe_ops_.size();
      const size_t have = expert_slots_wanted_ / layers;
      const unsigned want =
        have < htp_fixed::MOE_POOL_TARGET
          ? static_cast<unsigned>(htp_fixed::MOE_POOL_TARGET - have)
          : 0u;
      uint32_t info1[7] = {0};
      nntr_hvx_session_info(e2e_st_->h1, info1, 7);
      size_t mapped = arenaBytes() + (static_cast<size_t>(info1[4]) << 10);
      for (const ArenaChunk &c : e2e_st_->arena)
        mapped += c.buf->size();
      const size_t ceiling = size_t(htp_fixed::PD_MAP_CEILING_MIB) << 20,
                   margin = size_t(htp_fixed::PD_MAP_MARGIN_MIB) << 20;
      const size_t room =
        mapped + margin < ceiling ? ceiling - mapped - margin : 0;
      const size_t slot = expertSlotBytes(pool_descs_[0][0]);
      const size_t fit = room / (slot * layers);
      per_layer = static_cast<unsigned>(std::min<size_t>(want, fit));
      std::fprintf(stderr,
                   "[HTP] pool: %zu a layer, target %u, mapped %zu MiB, room "
                   "for %zu more a layer\n",
                   have, htp_fixed::MOE_POOL_TARGET, mapped >> 20, fit);
    }
    if (per_layer == 0u)
      return;
    const size_t extra = static_cast<size_t>(per_layer) * moe_ops_.size();
    pool_grow_(extra);
    {
      std::lock_guard<std::mutex> lock(handle_mutex_);
      // The final arena chunk only needs the remaining slots. Otherwise a
      // decode-only grow still allocates a full 256 MiB chunk at its tail.
      expert_slots_wanted_ += extra;
    }
    std::fprintf(stderr,
                 "[HTP] pool: +%u experts a layer from the second token\n",
                 per_layer);
  }

  /** [plan 201 S1] Re-sends S1's EXPERTS tables when a prefill touched the
   *  pool since the last token: each expert's pair as the ARM files it
   *  (handle_cache_), HTP_GRAPH_NO_HANDLE for one the pool does not hold. */
  /** Release prefill/seed temporaries between completed synchronous calls.
   * The public phase-boundary hook publishes kept outputs first. The token
   * driver calls here after the seed's last reader has completed. */
  void releasePrefillStaging() {
    constexpr size_t keep_bytes = size_t(64) << 10;
    size_t freed = 0;
    for (StagingPool *pool : {&act_pool_, &out_pool_, &rope_pool_, &x2_pool_,
                              &attn_q_pool_, &attn_out_pool_}) {
      for (auto it = pool->by_class.upper_bound(keep_bytes);
           it != pool->by_class.end();) {
        freed += it->second ? it->second->size() : 0;
        it = pool->by_class.erase(it);
      }
      pool->holds.clear();
      pool->copied_out.clear();
    }
    // These aliases can point inside the allocations just returned above.
    ion_alias_ = {};
    ion_alias_next_ = 0;
    lost_ = {};
    lost_next_ = 0;
    attn_out16_ = {};
    if (act_ah_buf_ && act_ah_buf_->size() > keep_bytes) {
      freed += act_ah_buf_->size();
      act_ah_buf_.reset();
      std::vector<float>().swap(act_scale_scratch_);
      std::vector<int32_t>().swap(act_zp_scratch_);
    }
    if (freed)
      std::fprintf(stderr, "[HTP-MEM] released_prefill_scratch_bytes=%zu\n",
                   freed);
  }

  void poolSync() {
    if (!pool_dirty_)
      return;
    const uint64_t t_ps = HtpProfile::nowUs(); // [#282 F]
    switchTl2Phase(true);
    initPoolPrefetch();
    pinExperts();
    dropHostFc();
    const uint64_t t_drop = HtpProfile::nowUs();
    E2eState &e = *e2e_st_;
    if (pool_descs_.size() != moe_ops_.size())
      throw std::runtime_error(
        "token driver: " + std::to_string(pool_descs_.size()) +
        " MoE layers gave their experts (set_decode_moe_experts), the list "
        "has " +
        std::to_string(moe_ops_.size()));
    std::vector<std::vector<float>> tabs(pool_descs_.size());
    // [#216] The decode misses' file pages are dropped here, once per
    // prefill, not as each is read: a drop beside the miss reads (advice
    // worker) cost 0.4-0.7 ms a miss round (S25). Every resident expert is
    // named; those dropped already (load, prefill) are a no-op.
    std::vector<Advice> drop;
    {
      std::lock_guard<std::mutex> lock(handle_mutex_);
      for (size_t m = 0; m < pool_descs_.size(); ++m) {
        const std::vector<ExpertFileDesc> &d = pool_descs_[m];
        for (size_t x = 0; fadviseKnob() != 0 && x < d.size(); ++x)
          if (experts_.count(d[x].key_gu) != 0)
            drop.emplace_back(d[x], false);
        if (d.size() != graphOp(moe_ops_[m])->n_experts)
          throw std::runtime_error(
            "token driver: MoE layer " + std::to_string(m) + " gave " +
            std::to_string(d.size()) + " experts, its op has " +
            std::to_string(graphOp(moe_ops_[m])->n_experts));
        tabs[m].resize(2 * d.size());
        for (size_t x = 0; x < d.size(); ++x) {
          auto g = handle_cache_.find(d[x].key_gu);
          auto h = handle_cache_.find(d[x].key_dn);
          const bool in = g != handle_cache_.end() && h != handle_cache_.end();
          const uint32_t hg = in ? g->second : kNoHandle;
          const uint32_t hd = in ? h->second : kNoHandle;
          std::memcpy(&tabs[m][x], &hg, 4);
          std::memcpy(&tabs[m][d.size() + x], &hd, 4);
        }
      }
    }
    for (size_t m = 0; m < tabs.size(); ++m)
      setParam(e.h1, moe_ops_[m], HTP_GRAPH_PARAM_EXPERTS, tabs[m].data(),
               static_cast<unsigned>(tabs[m].size()),
               static_cast<unsigned>(tabs[m].size()), "EXPERTS");
    pool_dirty_ = false;
    if (!t1_printed_) { // [#282 F]
      t1_printed_ = true;
      size_t left = 0;
      if (kvSeedQ() && htp_fixed::E2E_FREE_KVQ) {
        // [#289] a layer the DSP seed did not take keeps its cache till here
        std::lock_guard<std::mutex> lock(kvq_mutex_);
        left = kvq_handles_.size();
        for (uint32_t h : kvq_handles_)
          nntr_hvx_kv_release_q(
            static_cast<remote_handle64>(HtpBackend::global().handle()), h);
        kvq_handles_.clear();
      }
      std::fprintf(stderr,
                   "[HTP] t1: pool_sync_ms=%.1f (drop_host_fc+pin %.1f) "
                   "kv_seed_rpc_ms=%.1f seeds=%u (dsp %u, kvq freed late %zu) "
                   "rss_mib=%.1f mem_available_mib=%.1f\n",
                   (HtpProfile::nowUs() - t_ps) / 1000.0,
                   (t_drop - t_ps) / 1000.0, seed_rpc_us_ / 1000.0, seed_calls_,
                   seed_kvq_, left, rssMib(), memAvailMib());
    }
    // ponytail: a generation's decode loads stay cached until the next
    // prefill (<= misses x 5.3 MiB: 0.7 GiB at G = 1024); a drop per N
    // tokens off the miss path is the upgrade for long generations
    adviseLater(std::move(drop));
  }

  /** [plan 201 S1] Files the last answer's loads under the pairs S1 wrote
   *  back after its rebinds (called once S1 has moved past that answer:
   *  at its next request, or when the token is done). */
  void poolHarvest(uint8_t *page) {
    PoolServer &p = *pool_srv_;
    const volatile htp_miss_ans *a =
      reinterpret_cast<const volatile htp_miss_ans *>(page + HTP_MBOX_MISS_ANS);
    std::lock_guard<std::mutex> lock(handle_mutex_);
    std::exception_ptr first;
    for (size_t i = 0; i < p.pending.size(); ++i) {
      const uint32_t hg = a->load[i].h_gu, hd = a->load[i].h_dn;
      if (hg == kNoHandle || hd == kNoHandle) { // S1's rebind failed
        free_expert_slots_.push_back(p.pending[i].slot);
        if (!first)
          first = std::make_exception_ptr(
            std::runtime_error("token driver: S1 did not rebind a pool load"));
        continue;
      }
      fileRegistered(p.pending[i], hg, hd);
    }
    p.pending.clear();
    if (first)
      std::rethrow_exception(first);
  }

  /** [plan 201 S1] One miss request of S1 (protocol P-A): the layer's
   *  policy picks the victims (never a routed expert) and names the loads,
   *  each load is read into a free slot -- the victim's, with its pair --
   *  and the answer names both; S1 rebinds the pairs and fills them in. */
  /** @brief [#282 C] The ARM's view of the DSP's QTimer: cntvct_el0, the
   *  same 19.2 MHz system counter on these SoCs (0 elsewhere). */
  static uint64_t qtNow() {
#if defined(__aarch64__)
    uint64_t v;
    asm volatile("mrs %0, cntvct_el0" : "=r"(v));
    return v;
#else
    return 0;
#endif
  }

  void poolAnswer(uint8_t *page, const htp_miss_req &r) {
    PoolServer &p = *pool_srv_;
    { // [#282 C] post -> noticed here
      const uint64_t tp = uint64_t(r.t_post[0]) | (uint64_t(r.t_post[1]) << 32),
                     tn = qtNow();
      if (tp != 0 && tn >= tp)
        e2e_st_->rt_notice_us += (tn - tp) * 10 / 192;
    }
    htp_miss_ans a;
    std::memset(&a, 0, sizeof(a));
    a.rc = AEE_SUCCESS;
    const uint64_t t0 = HtpProfile::nowUs();
    try {
      poolHarvest(page);
      const auto it = std::find(moe_ops_.begin(), moe_ops_.end(), r.op);
      if (it == moe_ops_.end() || r.n_routed > HTP_MBOX_MISS_MAX ||
          r.n_miss > r.n_routed)
        throw std::runtime_error("token driver: a miss request for op " +
                                 std::to_string(r.op) + " is malformed");
      const std::vector<ExpertFileDesc> &d = pool_descs_[it - moe_ops_.begin()];
      {
        std::lock_guard<std::mutex> lock(handle_mutex_);
        pool_answered_.resize(pool_descs_.size());
        pool_answered_[it - moe_ops_.begin()] = 1;
        if (prediction_confidence_)
          prediction_confidence_->observe(it - moe_ops_.begin(),
                                          (uint64_t(p.tok) << 32) + r.op,
                                          r.routed, r.n_routed);
      }
      std::vector<const void *> need, loads;
      for (uint32_t i = 0; i < r.n_routed; ++i) {
        if (r.routed[i] >= d.size())
          throw std::runtime_error("token driver: routed expert out of range");
        need.push_back(d[r.routed[i]].key_gu);
      }
      pool_fn_(
        need, [&](const void *k) { loads.push_back(k); },
        [&](const void *k) {
          const std::pair<uint32_t, uint32_t> w = pool_where_.at(k);
          ExpertFileDesc v;
          if (a.n_evict == HTP_MBOX_MISS_MAX || !releaseExpert(k, v))
            throw std::runtime_error("token driver: the pool evicted an "
                                     "expert it does not hold");
          if (fadviseKnob() == 1) // [#216] asked back after the answer
            p.advice.emplace_back(v, true);
          a.evict[a.n_evict][0] = moe_ops_[w.first];
          a.evict[a.n_evict++][1] = w.second;
        });
      // the ARM's pool and S1's table must agree on what is missing
      for (uint32_t i = 0; i < r.n_miss; ++i)
        if (r.miss[i] >= d.size())
          throw std::runtime_error("token driver: missed expert out of range");
      bool same = loads.size() == r.n_miss;
      for (uint32_t i = 0; same && i < r.n_miss; ++i)
        same = std::find(loads.begin(), loads.end(), d[r.miss[i]].key_gu) !=
               loads.end();
      if (!same)
        throw std::runtime_error(
          "token driver: S1 misses " + std::to_string(r.n_miss) +
          " experts, the pool loads " + std::to_string(loads.size()));
      const remote_handle64 session =
        static_cast<remote_handle64>(HtpBackend::global().handle());
      // [#266] stage every miss, then read them together outside the
      // lock; a staged slot is in p.pending at once, so a failed read
      // gives it back in poolDisarm like any failed round
      {
        std::lock_guard<std::mutex> lock(handle_mutex_);
        for (uint32_t i = 0; i < r.n_miss; ++i)
          p.pending.push_back(stageExpert(session, d[r.miss[i]]));
      }
      if (pool_prefetch_) {
        // Transfer the retired pair to the prepared buffer. The old buffer
        // becomes a spare with no DSP handles. S1 rebinds that pair in its
        // existing miss answer, after evicting its former graph entry.
        const char *join = std::getenv("NNTR_MOE_PREFETCH_JOIN");
        for (auto &st : p.pending) {
          if (!join || std::strcmp(join, "0") != 0)
            st.prefetch_claimed = pool_prefetch_->claim(st.d.key_gu);
          else
            pool_prefetch_->take(st.d.key_gu, [&st](StagedExpert &ready) {
              adoptPrefetch(st, ready);
            });
        }
        pool_prefetch_->expireBefore((uint64_t(p.tok) << 32) + r.op + 1u);
      }
      for (uint32_t i = 0; i < r.n_miss; ++i) // [#282 B]
        if (pf_issued_keys_.erase(d[r.miss[i]].key_gu) != 0)
          ++pf_useful_;
      const uint64_t t_r0 = HtpProfile::nowUs(); // [#282 C]
      e2e_st_->rt_pre_us += t_r0 - t0;
      readMisses(p.pending);
      e2e_st_->rt_read_us += HtpProfile::nowUs() - t_r0;
      std::lock_guard<std::mutex> lock(handle_mutex_);
      for (uint32_t i = 0; i < r.n_miss; ++i) {
        const ExpertFileDesc &x = d[r.miss[i]];
        const StagedExpert &st = p.pending[i];
        throwPread(st.rc, "expert weight (miss)");
        // ponytail: the DSP rebinds a decode miss on its own from this
        // answer, which carries no offsets; an affine expert whose offsets
        // are all zero (the rotated ternary checkpoint's) is exact as a
        // symmetric one. Offsets in the answer are the upgrade.
        if (!st.gu.offsets.empty() || !st.dn.offsets.empty())
          throw std::runtime_error(
            "QS4CX_WH_AFFINE expert with non-zero offsets on the decode miss "
            "path: not supported");
        htp_miss_load &l = a.load[a.n_load++];
        l.e = r.miss[i];
        l.old_gu = st.slot.h_gu;
        l.old_dn = st.slot.h_dn;
        l.arena = arena_chunks_[st.slot.chunk].dsp_id;
        l.off_gu = st.gu.off;
        l.off_dn = st.dn.off;
        // [plan 229] QS2CX_WH: the width and the palettes ride the answer
        // Native files use compact slots; prepared QS2 companions retain
        // their padded layout when switching between prefill and decode.
        l.bits = x.w_bits == HTP_TL2_WEIGHT_BITS ? HTP_TL2_COMPACT_WEIGHT_BITS
                 : st.tl2                        ? HTP_TL2_WEIGHT_BITS
                                                 : x.w_bits;
        std::memset(l.pal_gu, 0, sizeof(l.pal_gu));
        std::memset(l.pal_dn, 0, sizeof(l.pal_dn));
        std::copy(st.gu.pal.begin(), st.gu.pal.end(), l.pal_gu);
        std::copy(st.dn.pal.begin(), st.dn.pal.end(), l.pal_dn);
        l.h_gu = l.h_dn = kNoHandle;
      }
    } catch (...) {
      if (!p.err)
        p.err = std::current_exception();
      a.rc = AEE_EBADSTATE; // S1 fails the token now, not after its window
      a.n_load = 0;
    }
    const uint64_t read_us = HtpProfile::nowUs() - t0;
    e2e_st_->pool_read_us += read_us;
    e2e_st_->pool_rounds += 1;
    if (HtpProfile::global().level() != 0 && a.n_load != 0)
      HtpProfile::global().addExpertLoad(read_us, 0, a.n_load);
    // the body, then seq last: S1 reads seq, then the rest
    a.seq2 = r.seq;
    { // [#282 C] the answer's write time, for S1's answer latency
      const uint64_t ta = qtNow();
      a.t_ans[0] = static_cast<uint32_t>(ta);
      a.t_ans[1] = static_cast<uint32_t>(ta >> 32);
    }
    uint8_t *dst = page + HTP_MBOX_MISS_ANS;
    std::memcpy(dst + 4, reinterpret_cast<const uint8_t *>(&a) + 4,
                sizeof(a) - 4);
    std::atomic_thread_fence(std::memory_order_release);
    *reinterpret_cast<volatile uint32_t *>(dst) = r.seq;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    adviseLater(std::move(p.advice));
    p.advice.clear();
  }

  /** @brief [#282 B] NNTR_MOE_PREFETCH=<threads> (0 / unset: off; needs
   *  NNTR_HTP_PREDICT=1 for S1 to post hints): the reader threads that
   *  pull a hinted expert the pool lacks into the page cache, so its miss
   *  a layer later preads DDR, not flash. The slots and S1's tables are
   *  untouched -- the ids cannot move. */
  /** @brief Every MoE layer's router is Gemma's softmax router (eps_bits
   *  set): what the ARM predictor needs. False without a graph description. */
  bool routersSoftmax() const {
    if (moe_ops_.empty())
      return false;
    for (uint32_t op : moe_ops_) {
      const htp_graph_op *r = op == 0u ? nullptr : graphOp(op - 1u);
      if (r == nullptr || r->kind != HTP_OP_ROUTER_TOPK || r->eps_bits == 0u)
        return false;
    }
    return true;
  }

  /** @brief NNTR_HTP_PREDICT: 0 off, 1 the DSP's guess, 2 the ARM's; unset
   *  is htp_fixed::PREDICT when the routers are Gemma's, else off. */
  uint32_t predictMode() const {
    const char *v = std::getenv("NNTR_HTP_PREDICT");
    if (v == nullptr)
      return routersSoftmax() ? htp_fixed::PREDICT : 0u;
    if (std::strcmp(v, "0") != 0 && std::strcmp(v, "1") != 0 &&
        std::strcmp(v, "2") != 0)
      throw std::runtime_error(std::string("NNTR_HTP_PREDICT=") + v +
                               ": want 0 (off), 1 (DSP), or 2 (ARM)");
    return static_cast<uint32_t>(v[0] - '0');
  }

  /** @brief NNTR_MOE_PREFETCH_DIRECT=1: the predicted experts read into
   *  spare DMA slots; unset follows the ARM prediction (built in). */
  bool directPrefetch() const {
    const char *v = std::getenv("NNTR_MOE_PREFETCH_DIRECT");
    return v != nullptr ? std::strcmp(v, "1") == 0 : predictMode() == 2u;
  }

  size_t prefetchThreads() const {
    if (weightPageCachePolicy() >= 0 && !directPrefetch())
      return 0; // Only DMA-slot prefetch uses the selected bounded reader.
    const char *v = std::getenv("NNTR_MOE_PREFETCH");
    if (v != nullptr)
      return std::strtoul(v, nullptr, 10);
    return directPrefetch() ? htp_fixed::PREFETCH_READERS : 0u;
  }

  /** @brief [#282 B] The hint slot, if S1 posted a newer one this token
   *  (one slot: older hints the poll did not see are gone): the guessed
   *  experts of the next MoE layer the pool does not hold go to the
   *  prefetch readers. @return the seq taken. */
  uint32_t takeHint(uint8_t *page, uint32_t tok, uint32_t last) {
    const volatile htp_pred_hint *hv =
      reinterpret_cast<const volatile htp_pred_hint *>(page + HTP_MBOX_HINT);
    const uint32_t seq = hv->seq;
    if (seq <= last || seq > tok * 256u + 255u)
      return last;
    std::atomic_thread_fence(std::memory_order_acquire);
    htp_pred_hint h;
    std::memcpy(&h, const_cast<const htp_pred_hint *>(hv), sizeof(h));
    if (h.seq2 != seq || h.n > HTP_MBOX_MISS_MAX)
      return last;
    if (hv->seq != seq)
      return last; // The writer may have replaced the hint while we copied it.
    ++pf_hints_;
    const auto it = std::upper_bound(moe_ops_.begin(), moe_ops_.end(), h.op);
    if (it == moe_ops_.end())
      return seq;
    const size_t layer = static_cast<size_t>(it - moe_ops_.begin());
    const std::vector<ExpertFileDesc> &d = pool_descs_[layer];
    std::vector<ExpertFileDesc> jobs;
    if (pool_prefetch_) {
      // A new hint targets L+1 while the real router of L is still running.
      // Keep L's guesses until its demand; retire only earlier layers.
      const size_t m = static_cast<size_t>(it - moe_ops_.begin());
      const uint32_t current = m ? moe_ops_[m - 1] : 0u;
      pool_prefetch_->expireBefore((uint64_t(tok) << 32) + current);
    }
    if (h.width != 0u) {
      if (!cpu_predictor_ || h.width != prediction_routers_[layer].hidden ||
          h.width > HTP_MBOX_PRED_MAX_WIDTH)
        return seq;
      std::vector<float> input(h.width);
      std::memcpy(input.data(), page + HTP_MBOX_PRED_INPUT,
                  h.width * sizeof(float));
      std::atomic_thread_fence(std::memory_order_acquire);
      if (hv->seq != seq)
        return last;
      const uint64_t epoch = (uint64_t(tok) << 32) + *it;
      // There is no reason to compute an older hint once this one arrives.
      cpu_predictor_->expireBefore(epoch);
      cpu_predictor_->submit(d[0].key_gu, epoch, [&](PredictionJob &job) {
        job.input = std::move(input);
        job.layer = layer;
        job.epoch = epoch;
        job.native = tl2_decode_;
      });
      return seq;
    }
    {
      std::lock_guard<std::mutex> lock(handle_mutex_);
      // NNTR_MOE_PREFETCH_TOP: only the guess's best n (best first)
      static const uint32_t top = [] {
        const char *v = std::getenv("NNTR_MOE_PREFETCH_TOP");
        return v != nullptr ? static_cast<uint32_t>(std::atoi(v)) : 16u;
      }();
      for (uint32_t i = 0; i < std::min(h.n, top); ++i)
        if (h.ids[i] < d.size() && experts_.count(d[h.ids[i]].key_gu) == 0 &&
            pf_issued_keys_.insert(d[h.ids[i]].key_gu).second)
          jobs.push_back(d[h.ids[i]]);
    }
    if (pool_prefetch_) {
      for (const auto &d : jobs)
        pf_issued_ +=
          queueDirectPrefetch(d, (uint64_t(tok) << 32) + *it, tl2_decode_);
    } else if (!jobs.empty()) {
      prefetchQueue(std::move(jobs));
    }
    return seq;
  }

  /** @brief [#282 B] Hands @a jobs to the prefetch readers (started on
   *  first use, left running at exit like the advice worker): each preads
   *  both weights into a scratch buffer, which leaves them in the page
   *  cache. */
  void prefetchQueue(std::vector<ExpertFileDesc> jobs) {
    std::lock_guard<std::mutex> lock(pf_mu_);
    if (pf_threads_.empty()) {
      for (size_t t = 0; t < prefetchThreads(); ++t)
        pf_threads_.emplace_back([this] {
          std::vector<uint8_t> buf;
          for (;;) {
            ExpertFileDesc d;
            {
              std::unique_lock<std::mutex> l(pf_mu_);
              pf_cv_.wait(l, [this] { return pf_stop_ || !pf_q_.empty(); });
              if (pf_stop_)
                return;
              d = pf_q_.front();
              pf_q_.pop_front();
            }
            const uint64_t pal = paletteBytes(d.w_bits);
            const size_t gu = codeBytes(d.K, 2 * d.inter, d.w_bits) + pal +
                              tailPerCol(d) * 2 * d.inter;
            const size_t dn = codeBytes(d.inter, d.N_out, d.w_bits) + pal +
                              tailPerCol(d) * d.N_out;
            buf.resize(std::max(gu, dn));
            (void)preadAll(d.fd, buf.data(), gu, d.off_gu);
            (void)preadAll(d.fd, buf.data(), dn, d.off_dn);
            ++pf_done_;
          }
        });
    }
    pf_issued_ += jobs.size();
    for (ExpertFileDesc &j : jobs)
      pf_q_.push_back(j);
    pf_cv_.notify_all();
  }

  /** @brief A completion hint is optional: older skeletons leave it zero. */
  static bool serverDoneEnabled() {
    static const bool enabled = [] {
      const char *v = std::getenv("NNTR_MOE_SERVER_DONE");
      return v && std::strcmp(v, "1") == 0;
    }();
    return enabled;
  }

  /** [plan 201 S1] Answer miss rounds until the DSP finishes the token.
   *  The optional completion hint parks the server before DSPQ reception,
   *  so a server on the caller's core cannot delay the caller's wakeup. */
  void poolServe() {
    PoolServer &p = *pool_srv_;
    uint8_t *page = e2e_st_->mbox->data();
    const volatile htp_miss_req *q =
      reinterpret_cast<const volatile htp_miss_req *>(page + HTP_MBOX_MISS_REQ);
    const volatile htp_token_done *done =
      reinterpret_cast<const volatile htp_token_done *>(page +
                                                        HTP_MBOX_TOKEN_DONE);
    const bool done_enabled = serverDoneEnabled();
    HtpTokenPollState poll_state;
    for (;;) {
      uint32_t tok;
      {
        std::unique_lock<std::mutex> lock(p.mu);
        p.idle = true;
        p.cv.notify_all();
        p.cv.wait(lock, [&p, &poll_state] {
          return p.stop || poll_state.canStart(p.active.load(), p.tok);
        });
        if (p.stop)
          return;
        p.idle = false;
        tok = p.tok;
        poll_state.begin(tok);
      }
      // The graph and prefetch configuration are fixed for this token.
      // Resolving them on every mailbox poll repeats getenv and walks all
      // router nodes while no request is available.
      const bool hints = prefetchThreads() != 0;
      uint32_t hint_seq = tok * 256u; // [#282 B] the last hint taken
      for (uint32_t k = 0; k < 255u;) {
        if (hints)
          hint_seq = takeHint(page, tok, hint_seq);
        const uint32_t seq = tok * 256u + k + 1u; // hexkl_token_seq
        if (q->seq == seq) {
          std::atomic_thread_fence(std::memory_order_acquire);
          htp_miss_req r;
          std::memcpy(&r, const_cast<const htp_miss_req *>(q), sizeof(r));
          if (r.seq2 == seq) {
            poolAnswer(page, r);
            ++k;
            continue;
          }
        }
        if (!p.active.load() ||
            (done_enabled && HtpTokenPollState::isDone(tok, [done](unsigned w) {
               return w == 0u ? done->seq : done->inverse;
             }))) {
          break;
        }
        // A pause, not a yield: with sched_yield the request was noticed
        // 86-150 us after the DSP posted it on the S26 Ultra (1 us on the
        // S25), 17 rounds a token.
#if defined(__aarch64__)
        __asm__ volatile("yield");
#else
        std::this_thread::yield();
#endif
      }
    }
  }

  /** [plan 201 S1] Around one token: arm the server, then disarm it once
   *  the DSP answered, file the last loads and refresh the pool's
   *  recency from the token's routed sets. */
  void poolArm(uint32_t tok) {
    if (!pool_srv_) {
      pool_srv_ = std::make_unique<PoolServer>();
      // Retain the caller's affinity, as the original server did. The
      // completion hint can park polling before DSPQ wakes the caller.
      pool_srv_->th = std::thread([this] { poolServe(); });
    }
    std::lock_guard<std::mutex> lock(pool_srv_->mu);
    pf_issued_keys_.clear(); // [#282 B] per token: the server is idle here
    // The mailbox is uncached on ARM. Clear before sending the next token;
    // no old completion, including a wrapped sequence, may stop its server.
    auto *done = reinterpret_cast<volatile htp_token_done *>(
      e2e_st_->mbox->data() + HTP_MBOX_TOKEN_DONE);
    done->seq = 0u;
    done->inverse = 0u;
    std::atomic_thread_fence(std::memory_order_release);
    pool_srv_->tok = tok;
    pool_srv_->active = true;
    pool_srv_->cv.notify_all();
  }
  void poolDisarm() {
    PoolServer &p = *pool_srv_;
    const uint64_t start = HtpProfile::nowUs();
    {
      std::unique_lock<std::mutex> lock(p.mu);
      p.active = false;
      p.cv.wait(lock, [&p] { return p.idle; });
    }
    const uint64_t idle = HtpProfile::nowUs();
    e2e_st_->pool_idle_us += idle - start;
    // the round's own error first: a failed answer leaves its loads unfiled
    std::exception_ptr err = p.err;
    p.err = nullptr;
    if (cpu_predictor_)
      cpu_predictor_->expireBefore((uint64_t(p.tok) + 1u) << 32);
    if (pool_prefetch_)
      pool_prefetch_->expireBefore((uint64_t(p.tok) + 1u) << 32);
    // No speculative I/O may outlive this token's model/file ownership.
    // The final projection normally leaves ample time for reads to finish.
    if (cpu_predictor_)
      cpu_predictor_->drain();
    const uint64_t predicted = HtpProfile::nowUs();
    e2e_st_->pool_predict_drain_us += predicted - idle;
    if (pool_prefetch_)
      pool_prefetch_->drain();
    e2e_st_->pool_prefetch_drain_us += HtpProfile::nowUs() - predicted;
    if (err) {
      std::lock_guard<std::mutex> lock(handle_mutex_);
      for (StagedExpert &st : p.pending)
        free_expert_slots_.push_back(st.slot);
      p.pending.clear();
      std::rethrow_exception(err);
    }
    poolHarvest(e2e_st_->mbox->data());
  }
  /** [#266] NNTR_HTP_ROUTE_LOG=<path>: each token's routed sets, one line
   *  per MoE op in NNTR_MOE_TRACE's format ("<op> 1 | <ids> |"), so
   *  tools/moe_expert_cache_sim.py replays the one-PD token's routing.
   *  [#266 S2] Under NNTR_HTP_PREDICT=1 the third field is the previous
   *  router op's guess at this op's ids, best first (the sim's --predict
   *  trace). */
  static void routeLog(const htp_dspq_token_resp &s1r) {
    static std::FILE *f = [] {
      const char *path = std::getenv("NNTR_HTP_ROUTE_LOG");
      return path != nullptr && *path != '\0' ? std::fopen(path, "w") : nullptr;
    }();
    if (f == nullptr)
      return;
    const uint32_t pred_n =
      std::min<uint32_t>(s1r.pred_n, HTP_DSPQ_TOKEN_ROUTE);
    // prev: the previous router op's entry (its guess at this op)
    uint32_t p_at = 0, prev = pred_n;
    for (uint32_t m = 0, at = 0; at < s1r.route_n; ++m) {
      const uint32_t n = s1r.route[at++];
      std::fprintf(f, "%u 1 |", m);
      for (uint32_t i = 0; i < n && at < s1r.route_n; ++i)
        std::fprintf(f, " %u", s1r.route[at++]);
      std::fputs(" |", f);
      for (uint32_t i = 1;
           prev < pred_n && i <= s1r.pred[prev] && prev + i < pred_n; ++i)
        std::fprintf(f, " %u", s1r.pred[prev + i]);
      std::fputc('\n', f);
      prev = p_at;
      if (p_at < pred_n)
        p_at += 1u + s1r.pred[p_at];
    }
    std::fflush(f);
  }

  void poolRefresh(const htp_dspq_token_resp &s1r) {
    // [#289] under LRFU a layer's routed set counts once a token: a layer
    // whose miss round already handed it to the pool is not handed again
    std::vector<uint8_t> answered;
    {
      std::lock_guard<std::mutex> lock(handle_mutex_);
      answered.swap(pool_answered_);
    }
    if (!lrfuOn())
      answered.clear();
    size_t at = 0;
    std::vector<const void *> need;
    for (size_t m = 0; m < pool_descs_.size() && at < s1r.route_n; ++m) {
      const uint32_t n = s1r.route[at++];
      need.clear();
      {
        // a later layer's miss may have evicted an earlier layer's routed
        // expert within the token: only the resident ones are touched
        std::lock_guard<std::mutex> lock(handle_mutex_);
        for (uint32_t i = 0; i < n && at < s1r.route_n; ++i) {
          const void *k = pool_descs_[m].at(s1r.route[at++]).key_gu;
          if (experts_.count(k) != 0)
            need.push_back(k);
        }
      }
      if (m < answered.size() && answered[m] != 0)
        continue;
      pool_fn_(
        need,
        [](const void *) {
          throw std::logic_error("token driver: a routed expert is not in "
                                 "the pool after its token");
        },
        [](const void *) {
          throw std::logic_error("token driver: the pool's refresh evicted");
        });
    }
  }

  /** [#4416 mem] one line of where the memory is: the S1 PD's heap, the
   *  staging pools' buffers, this process and the system (probe) */
  void memLine(const char *where) {
    uint32_t info1[7] = {0};
    if (e2e_st_)
      nntr_hvx_session_info(e2e_st_->h1, info1, 7);
    size_t pools = 0;
    for (const StagingPool *pl : {&act_pool_, &out_pool_, &rope_pool_,
                                  &x2_pool_, &attn_q_pool_, &attn_out_pool_})
      for (const auto &kv : pl->by_class)
        pools += kv.second ? kv.second->size() : 0;
    std::fprintf(stderr,
                 "[HTP] mem %s: s1_heap_mib=%u staging_pools_mib=%zu "
                 "rss_mib=%zu mem_available_mib=%zu\n",
                 where, info1[4] >> 10, pools >> 20,
                 static_cast<size_t>(rssKb() >> 10), memAvailableMib());
  }

  void ensureGraphInit(remote_handle64 session) {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    if (graph_inited_) {
      ensureAttentionCaches(session);
      return;
    }
    memLine("t1 before graph init (= prefill end)");
    // [#282 F] the first token's setup, timed (one line at the end)
    const uint64_t t_init = HtpProfile::nowUs();
    struct InitLine {
      uint64_t t0;
      ~InitLine() {
        std::fprintf(stderr, "[HTP] t1: graph_init_ms=%.1f\n",
                     (HtpProfile::nowUs() - t0) / 1000.0);
      }
    } init_line{t_init};
    if (e2e_ && !q4m1_bound_)
      e2ePlaceFc(); // a caller that never ran finish_decode_graph_q4_0
    if ((resident_mask_ & HTP_GRAPH_KINDS_Q4M1) != 0u && !q4m1_bound_) {
      try {
        bindQ4m1(session);
      } catch (...) {
        releaseQ4m1(session); // a retry registers them all again
        throw;
      }
      // [#132 Part B E3] the E2E FC arena is placed once: a retry keeps it
      q4m1_bound_ = e2e_;
    }
    uint32_t n_ops = 0;
    {
      const int err =
        nntr_hvx_graph_init(session, graph_words_.data(),
                            static_cast<int>(graph_words_.size()), &n_ops);
      if (err != AEE_SUCCESS) {
        releaseQ4m1(session);
        throw std::runtime_error("nntr_hvx_graph_init failed: " +
                                 graphErr(err));
      }
    }
    // [plan 201 S1] each resident MoE op's EXPERTS table; a refusal
    // releases the graph, so a retry inits again
    try {
      for (size_t m = 0; m < moe_ops_.size(); ++m) {
        const std::vector<float> &t = moe_tables_[m];
        if (!t.empty() && (e2e_ || graphOp(moe_ops_[m])->resident))
          setParam(session, moe_ops_[m], HTP_GRAPH_PARAM_EXPERTS, t.data(),
                   static_cast<unsigned>(t.size()),
                   2u * graphOp(moe_ops_[m])->n_experts, "EXPERTS");
      }
      // [plan 201 S4] the parameters the model handed at load, by name
      for (const LoadParam &p : load_params_)
        setParam(session, p.op, p.which, p.data, p.n, p.n, "load parameter");
      // [#266 S2] NNTR_HTP_PREDICT=1: the routers' next-layer guesses,
      // logged through NNTR_HTP_ROUTE_LOG (unset: the skel's default, off)
      if (const uint32_t on = predictMode()) {
        if (on == 2u && (!directPrefetch() || !prefetchThreads()))
          throw std::invalid_argument(
            "ARM prediction requires direct prefetch and readers");
        float word;
        std::memcpy(&word, &on, sizeof(word));
        setParam(session, HTP_GRAPH_NO_OP, HTP_GRAPH_PARAM_PREDICT, &word, 1u,
                 1u, "PREDICT");
      }
      // [#289] A miss round runs the layer's dense branch while the ARM
      // reads: the same ops on the same input, earlier, in time the DSP
      // spent waiting. On unless NNTR_HTP_DENSE_EARLY=0.
      {
        const char *v = std::getenv("NNTR_HTP_DENSE_EARLY");
        if (v != nullptr && std::strcmp(v, "0") != 0 &&
            std::strcmp(v, "1") != 0)
          throw std::runtime_error(std::string("NNTR_HTP_DENSE_EARLY=") + v +
                                   ": want 0 or 1");
        const uint32_t on = v == nullptr || v[0] == '1' ? 1u : 0u;
        float word;
        std::memcpy(&word, &on, sizeof(word));
        setParam(session, HTP_GRAPH_NO_OP, HTP_GRAPH_PARAM_DENSE_EARLY, &word,
                 1u, 1u, "DENSE_EARLY");
      }
    } catch (...) {
      nntr_hvx_graph_release(session);
      if (!e2e_)
        releaseQ4m1(session);
      throw;
    }
    graph_inited_ = true;
    memLine("t1 after graph init");
    char names[128];
    std::fprintf(stderr, "[HTP] graph: init n_ops=%u resident=%s moe_ops=%zu\n",
                 n_ops,
                 htp_graph_kinds_str(resident_mask_, names, sizeof(names)),
                 moe_ops_.size());
    ensureAttentionCaches(session);
    if (e2e_)
      e2eStart();
  }

  void ensureAttentionCaches(remote_handle64 session) {
    if (attn_registered_)
      return;
    // [#282 D] NNTR_HTP_E2E_FREE_KVQ=1: the prefill's quantized KV caches
    // (kv_cache_q, the int8 masters and WH tiles, as large as the fp16
    // attn_m1 cache below) go before that cache is registered: the one-PD
    // token never reads them. ponytail: a later prefill on a layer whose
    // handle is gone fails its append and takes mha_core's CPU attention
    // -- single-prompt runs; the upgrade is mha_core re-registering.
    if (e2e_ && htp_fixed::E2E_FREE_KVQ &&
        !kvSeedQ()) { // [#289] the seeds free them one layer at a time
      std::lock_guard<std::mutex> lock(kvq_mutex_);
      for (uint32_t h : kvq_handles_)
        nntr_hvx_kv_release_q(session, h);
      std::fprintf(stderr, "[HTP] e2e: freed %zu quantized KV caches\n",
                   kvq_handles_.size());
      kvq_handles_.clear();
    }
    // [#130] the session's KV cache for the resident ATTN_M1 ops: one
    // ordinal per attention layer, the shape from the first record;
    // [plan 201 S4] a second cache for a second shape (Gemma 4's full
    // layers), registered second so the DSP files it as attn_m1_b
    for (uint32_t c = 0; c < 2u; ++c) {
      const htp_graph_op *rec = nullptr;
      uint32_t n_attn = 0;
      for (uint32_t op : kind_ops_[HTP_OP_ATTN_M1]) {
        if (attn_cache_[attn_ordinal_[op]] != c)
          continue;
        rec = rec ? rec : graphOp(op);
        ++n_attn;
      }
      if (rec == nullptr)
        continue;
      // a windowed cache (set_decode_kv_window) holds that many rows
      const uint32_t max_seq = rec->top_k != 0u && attn_window_rows_ != 0u
                                 ? attn_window_rows_
                                 : graph_words_[6];
      // [#282 D] NNTR_HTP_ATTN_M1_Q8: the int8 cache (half the bytes;
      // fixed scales from the prefill seed), HVX_ATTN_M1_Q8 in max_seq;
      // 1 = the first cache only (Gemma's sliding layers, 400 of its 440
      // MiB at max_seq 2048), 2 = both (the full layers too)
      constexpr int q8_knob = htp_fixed::ATTN_M1_Q8; // built in: 2
      const bool q8 = q8_knob >= 2 || (q8_knob == 1 && c == 0u);
      // [#4416 F] NNTR_HTP_ATTN_M1_FAST=1: the integer forward over the
      // int8 cache (HVX_ATTN_M1_FAST; not bit-identical to the CPU)
      static const bool fast_knob = [] {
        const char *v = std::getenv("NNTR_HTP_ATTN_M1_FAST");
        return v != nullptr && std::atoi(v) != 0;
      }();
      const bool fast = q8 && fast_knob;
      // [#289] lazy: a layer's cache comes at its seed, after the layer
      // before it freed its quantized prefill cache (HVX_ATTN_M1_LAZY)
      const int rc = nntr_hvx_attn_m1_register(
        session, n_attn, rec->n_kv, rec->gqa, rec->head_dim,
        max_seq | (q8 ? 0x80000000u : 0u) | (kvSeedQ() ? 0x40000000u : 0u) |
          (fast ? 0x20000000u : 0u));
      if (rc != AEE_SUCCESS) {
        throw std::runtime_error("nntr_hvx_attn_m1_register failed: " +
                                 graphErr(rc));
      }
      attn_registered_ = true;
      std::fprintf(stderr,
                   "[HTP] attn_m1: registered layers=%u kv=%u gqa=%u "
                   "head_dim=%u max_seq=%u cache=%llu KiB%s\n",
                   n_attn, rec->n_kv, rec->gqa, rec->head_dim, max_seq,
                   (unsigned long long)n_attn * rec->n_kv * rec->head_dim *
                     ((max_seq + 63u) / 64u * 64u) * 2u *
                     (q8 ? 1u : sizeof(uint16_t)) / 1024u,
                   fast ? " int8 fast" : (q8 ? " int8" : ""));
    }
  }

  /** [#80] The M=1 GEMV switch, read once from NNTR_MOE_HTP_M1_GEMV and
   *  sent to the DSP once per session through moe_set_opts (the DSP decides
   *  per call on M). On by default since #101; NNTR_MOE_HTP_M1_GEMV=0 is the
   *  opt-out (htp_moe_opts_flags). With the switch on, an error or an echo
   *  that differs from what was sent throws rather than falling back: a
   *  silent fallback would let a run report the HMX loop's numbers as the
   *  GEMV's -- so a skel older than moe_set_opts now fails the first MoE call
   *  unless the opt-out is set. With it off, such a skel is exactly the HMX
   *  loop, so that case only logs. The stderr line (source=default|env) is
   *  the proof of which path a run took when no profile is on;
   *  [HTP-PROFILE]'s m1_gemv= and blocks= are the per-call proof. */
  void sendMoeOptsOnce(remote_handle64 session) {
    std::call_once(moe_opts_once_, [this, session]() {
      moeOptsSent().store(true);
      const char *env = std::getenv("NNTR_MOE_HTP_M1_GEMV");
      const char *lead_env = std::getenv("NNTR_MOE_HTP_GEMV_LEAD_KB");
      const char *rows1_env = std::getenv("NNTR_MOE_HTP_GEMV_ROWS1");
      const char *feed_env = std::getenv("NNTR_MOE_HTP_GEMV_FEED");
      // #177: the M=1 feed's DMA queue count. A value other than 1..4 is
      // a mistyped variant, and running it as the default would measure
      // the reference under the variant's name, so it throws.
      const char *queues_env = std::getenv("NNTR_MOE_DMA_QUEUES");
      uint32_t queue_bits = 0;
      static_assert(htp_fixed::MOE_DMA_QUEUES >= 1u &&
                      htp_fixed::MOE_DMA_QUEUES <= 4u,
                    "the feed splits a matrix over 1..4 queues");
      const char queues_fixed[] = {
        static_cast<char>('0' + htp_fixed::MOE_DMA_QUEUES), '\0'};
      if (htp_moe_opts_dma_queues(queues_env ? queues_env : queues_fixed,
                                  &queue_bits) != 0) {
        throw std::runtime_error(std::string("NNTR_MOE_DMA_QUEUES=") +
                                 queues_env +
                                 " is not one of 1, 2, 3, 4 (unset = 1)");
      }
      const char *had_env = std::getenv("NNTR_MOE_HADAMARD");
      const char *had_scope = std::getenv("NNTR_MOE_HADAMARD_SCOPE");
      uint32_t had_scope_bits = 0;
      if (htp_moe_opts_hadamard_scope(had_scope, &had_scope_bits) != 0)
        throw std::runtime_error(std::string("NNTR_MOE_HADAMARD_SCOPE=") +
                                 had_scope + " is not 2bit or all");
      uint32_t had_bits = 0;
      if (htp_moe_opts_hadamard(had_env, &had_bits) != 0) {
        throw std::runtime_error(
          std::string("NNTR_MOE_HADAMARD=") + had_env +
          " is not off, up, down, both, both_scalar, up_hmx, "
          "down_hmx, both_hmx");
      }
      if (had_bits)
        had_bits |= had_scope_bits;
      const uint32_t flags =
        htp_moe_opts_flags(env, lead_env, rows1_env, feed_env) |
        htp_moe_opts_dma_bypass(std::getenv("NNTR_MOE_DMA_BYPASS")) |
        queue_bits | had_bits | (moeGeglu().load() ? HTP_MOE_FLAG_GEGLU : 0u);
      const char *source = env != nullptr ? "env" : "default";
      uint32_t applied = 0;
      const int err = nntr_hvx_moe_set_opts(session, flags, &applied);
      if (htp_moe_opts_must_match(flags) == HTP_MOE_FLAG_M1_GEMV &&
          err != AEE_SUCCESS) {
        // The opt-out was asked for, and a skel that predates moe_set_opts
        // runs the HMX loop, which is what "off" means: say so and go on.
        std::fprintf(stderr,
                     "[HTP] moe m1 gemv: off (moe_set_opts err=0x%08x; the "
                     "skel predates it) source=%s\n",
                     static_cast<unsigned>(err), source);
        return;
      }
      // Off: only bit 0 (and #158's bypass bit, if asked for) has to be
      // echoed. The tune bits are always sent (htp_moe_opts_flags) but mean
      // nothing on the HMX loop, so a skel that predates them must not fail
      // an opt-out run.
      const uint32_t must_match = htp_moe_opts_must_match(flags);
      if (err != AEE_SUCCESS || ((applied ^ flags) & must_match) != 0u) {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "nntr_hvx_moe_set_opts failed: err=0x%08x sent=0x%x "
                      "applied=0x%x",
                      static_cast<unsigned>(err), flags, applied);
        throw std::runtime_error(
          std::string(buf) +
          " (DSP skel or hardware did not accept the requested options; "
          "rebuild/push libnntr_hvx_skel.so with test/htp/build.sh and "
          "check FP16 HMX support when requesting *_hmx)");
      }
      // The measured (loop, lead, feed) cell, in the same line. All three
      // knobs are always sent (htp_moe_opts_flags: the env's value or the
      // default), and the echo above confirmed them, so an unset run reads
      // lead=192KB rows1=1 feed=vtcm and applied=0x303e1 (#113's D192
      // under #117's VTCM feed); NNTR_MOE_HTP_GEMV_FEED=0 reads feed=arena
      // and 0x103e1. The level-2 M==1 row's feed=n/calls is the per-call
      // proof. source= still refers to the M1 GEMV switch alone (LEDGER 16).
      std::fprintf(
        stderr,
        "[HTP] moe m1 gemv: %s (applied=0x%x) lead=%uKB rows1=%u "
        "feed=%s dma_bypass=%u dma_q=%u source=%s\n",
        (flags & HTP_MOE_FLAG_M1_GEMV) != 0u ? "on" : "off", applied,
        ((flags >> HTP_MOE_GEMV_LEAD_SHIFT) & HTP_MOE_GEMV_LEAD_BITS) *
          HTP_MOE_GEMV_LEAD_KB_UNIT,
        (flags & HTP_MOE_FLAG_GEMV_ROWS1) != 0u ? 1u : 0u,
        htp_moe_opts_feed_name(flags),
        (applied & HTP_MOE_FLAG_DMA_BYPASS) != 0u ? 1u : 0u,
        htp_moe_opts_dma_q(applied), source);
      if (had_bits)
        std::fprintf(stderr,
                     "[HTP] online Hadamard=%s scope=%s applied=0x%x; matching "
                     "offline weight rotation required for valid outputs\n",
                     had_env, had_scope ? had_scope : "2bit",
                     applied & HTP_MOE_FLAG_HAD_MASK);
      moe_opts_sent_ = true;
      sendHadamardSigns(session);
    });
  }

  /** [PR 4428] The rotation signs to the session, once they and the MoE
   *  word are both known (set_moe_hadamard_signs, sendMoeOptsOnce). */
  void sendHadamardSigns(remote_handle64 session) {
    std::lock_guard<std::mutex> lock(had_signs_mutex_);
    if (had_signs_in_.empty() && had_signs_down_.empty())
      return;
    const int rc = nntr_hvx_moe_set_hadamard_signs(
      session, had_signs_in_.data(), static_cast<int>(had_signs_in_.size()),
      had_signs_down_.data(), static_cast<int>(had_signs_down_.size()));
    if (rc != AEE_SUCCESS) {
      char code[16];
      std::snprintf(code, sizeof(code), "0x%08x", static_cast<unsigned>(rc));
      throw std::runtime_error(
        std::string("nntr_hvx_moe_set_hadamard_signs failed: err=") + code +
        " (widths " + std::to_string(had_signs_in_.size()) + " / " +
        std::to_string(had_signs_down_.size()) +
        "; a skel older than test/htp/nntr_hvx.idl answers this call with an "
        "error: rebuild and push libnntr_hvx_skel.so)");
    }
    std::fprintf(stderr, "[HTP] Hadamard signs: in=%zu down=%zu\n",
                 had_signs_in_.size(), had_signs_down_.size());
  }

  bool set_moe_hadamard_signs(const std::vector<float> &in,
                              const std::vector<float> &down) override {
    {
      std::lock_guard<std::mutex> lock(had_signs_mutex_);
      had_signs_in_ = in;
      had_signs_down_ = down;
    }
    if (moe_opts_sent_)
      sendHadamardSigns(
        static_cast<remote_handle64>(HtpBackend::global().handle()));
    return true;
  }

  /** Same registration the layer call above does on first use, keyed by
   *  the same data pointer, so the forward-time call is a cache hit. */
  bool register_qs4cx_weight(void *data, const float *scale, unsigned int K,
                             unsigned int N, unsigned int w_bits) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    if (w_bits != 0u) {
      get_or_register_wh(data, scale, session, K, N, w_bits);
    } else {
      get_or_register_qs4cx(data, scale, session, K, N);
    }
    return true;
  }

  /** Same, for a Q4_0 FC weight: the conversion and registration that
   *  gemm_q4_0_accel_fp32 would otherwise do on its first call. */
  bool register_q4_0_weight(void *data, unsigned int K,
                            unsigned int N) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    get_or_register_fc(data, session, K, N);
    return true;
  }

  /** [#225] Opens the FC WH sidecar and indexes it by fcWhKey, replacing
   *  any earlier one (a reload). Every check that can fail on a wrong or
   *  stale file fails here, at load, with the path: the magic and version
   *  (the layout's tag), the index inside the file, each image's length
   *  against its shape, each image inside the file, and no key twice. */
  bool set_fc_wh_file(const char *path) override {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    if (fcwh_fd_ >= 0)
      ::close(fcwh_fd_);
    fcwh_fd_ = -1;
    fcwh_.clear();
    fcwh_arena_bytes_ = fcwh_heap_bytes_ = 0; // [#225] fcwhLeft's, the banner's
    const std::string where = std::string("fc_wh_file_name ") + path + ": ";
    const int fd = openWeightFile(path);
    if (fd < 0)
      throw std::runtime_error(where + std::strerror(errno));
    std::unordered_map<uint64_t, FcWhEntry> index;
    try {
      const off_t end = ::lseek(fd, 0, SEEK_END);
      const uint64_t size = end < 0 ? 0u : static_cast<uint64_t>(end);
      char magic[sizeof(FCWH_MAGIC)];
      uint32_t vc[2] = {0, 0}; // version, count
      throwPread(preadAll(fd, magic, sizeof(magic), 0), "fc wh header");
      throwPread(preadAll(fd, vc, sizeof(vc), sizeof(magic)), "fc wh header");
      if (std::memcmp(magic, FCWH_MAGIC, sizeof(magic)) != 0 ||
          vc[0] != FCWH_VERSION || fcWhHeaderBytes(vc[1]) > size) {
        throw std::runtime_error("not an FC WH sidecar of version " +
                                 std::to_string(FCWH_VERSION));
      }
      std::vector<FcWhEntry> entries(vc[1]);
      throwPread(
        preadAll(fd, entries.data(), entries.size() * sizeof(FcWhEntry), 16u),
        "fc wh index");
      for (const FcWhEntry &e : entries) {
        const uint64_t want = whBytes(e.K, e.N) + 2u * sizeof(float) * e.N;
        if (e.K % WH_TILE != 0 || e.N % WH_TILE != 0 || e.bytes != want ||
            e.off < fcWhHeaderBytes(vc[1]) || e.off > size ||
            e.bytes > size - e.off || !index.emplace(e.key, e).second) {
          throw std::runtime_error("bad index entry " +
                                   std::string(e.name, strnlen(e.name, 64)));
        }
      }
    } catch (const std::exception &err) {
      ::close(fd);
      throw std::runtime_error(where + err.what());
    }
    fcwh_ = std::move(index);
    fcwh_fd_ = fd;
    fcwh_name_ = path;
    return true;
  }

  /**
   * @brief [doc 52] One expert's two WH weights, read from the model file
   *        straight into an arena slot and registered from there.
   *
   * The slot is the unit the LRU above this trades in: gate_up then down,
   * each 4 KB aligned, so a release hands back exactly what the next miss
   * needs and the pool never fragments. pread into the arena rather than
   * activate()'s mmap + memcpy: the arena's CPU mapping is uncached, so the
   * copy the kernel does into it IS the write to DDR the DSP will read,
   * and there is no second copy and no page fault storm to pay. The scales
   * and column sums follow the nibbles in the file (QS4CX_WH_Tensor's
   * layout, written by nntr_quantize_stream), so nothing is computed here.
   *
   * Slot reuse means the DSP reads a slot, the host overwrites it, the DSP
   * reads it again -- a sequence the arena tests never covered.
   * HmxArenaSlotReuse.Uncached (unittest_hvx_mm_u8i4) is the gate for it.
   */
  bool register_qs4cx_wh_expert_file(const ExpertFileDesc &d,
                                     bool at_load) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    std::lock_guard<std::mutex> lock(handle_mutex_);
    if (experts_.count(d.key_gu) != 0)
      return true;
    const uint64_t t0 = HtpProfile::nowUs();
    StagedExpert st = stageExpert(session, d);
    st.rc = readExpert(st, /*use_pool=*/true);
    const uint64_t t_read = HtpProfile::nowUs();
    registerStaged(session, st); // throws, and frees the slot, on failure
    const uint64_t t_rpc = HtpProfile::nowUs();

    HtpProfile &profile = HtpProfile::global();
    if (profile.level() != 0) {
      if (at_load) // the file read stands where the convert used to
        profile.addRegister(t_rpc - t0, t_read - t0, t_rpc - t_read, true);
      else
        profile.addExpertLoad(t_read - t0, t_rpc - t_read, 1);
    }
    return true;
  }

  /** [doc 52 section 10.23] One call's misses: each read in turn, then all
   *  registered in one round trip instead of one each. */
  bool register_qs4cx_wh_expert_files(
    const std::vector<ExpertFileDesc> &ds) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    std::lock_guard<std::mutex> lock(handle_mutex_);
    const uint64_t t0 = HtpProfile::nowUs();
    std::vector<StagedExpert> sts;
    try {
      for (const ExpertFileDesc &d : ds) {
        if (experts_.count(d.key_gu) != 0)
          continue;
        sts.push_back(stageExpert(session, d));
        sts.back().rc = readExpert(sts.back(), /*use_pool=*/true);
      }
    } catch (...) {
      for (const StagedExpert &st : sts)
        free_expert_slots_.push_back(st.slot);
      throw;
    }
    const uint64_t t_read = HtpProfile::nowUs();
    registerStagedBatch(session, sts); // files what it can, then throws
    HtpProfile &profile = HtpProfile::global();
    if (profile.level() != 0 && !sts.empty())
      profile.addExpertLoad(t_read - t0, HtpProfile::nowUs() - t_read,
                            sts.size());
    return true;
  }

  /**
   * @brief [doc 52 sections 10.10, 10.20] Queues one batch of experts --
   *        one layer's -- to be read into fresh slots by the prefetch
   *        readers, and returns at once. Batches are read and handed back
   *        in the order they were queued, so a caller can keep several
   *        layers in flight and collect the oldest with
   *        prefetch_qs4cx_wh_experts_end.
   *
   * What overlaps is the file read only. The register calls go through the
   * same FastRPC session as the layer calls, and the DSP's weight table is
   * not written while a kernel reads it, so they wait for _end, which the
   * caller makes between calls. The readers are this backend's own
   * threads, not ThreadManager's: they have to keep reading while the
   * caller blocks in FastRPC and while ThreadManager's workers run the ARM
   * side between calls. They stay off the caller's core (section 10.19:
   * sharing it slowed each layer call by up to 7 ms).
   */
  bool prefetch_qs4cx_wh_experts_begin(
    const std::vector<ExpertFileDesc> &ds) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    switchTl2Phase(false);
    auto batch = std::make_unique<PrefetchBatch>();
    {
      std::lock_guard<std::mutex> lock(handle_mutex_);
      try {
        for (const ExpertFileDesc &d : ds) {
          if (experts_.count(d.key_gu) == 0)
            batch->experts.push_back(stageExpert(session, d));
        }
      } catch (...) {
        for (const StagedExpert &st : batch->experts)
          free_expert_slots_.push_back(st.slot);
        throw;
      }
    }
    if (batch->experts.empty())
      return false;
    startPrefetchReaders();
    std::lock_guard<std::mutex> lock(prefetch_mutex_);
    PrefetchBatch *b = batch.get();
    prefetch_batches_.push_back(std::move(batch));
    for (size_t i = 0; i < b->experts.size(); ++i)
      prefetch_jobs_.emplace_back(b, i);
    prefetch_cv_.notify_all();
    return true;
  }

  /** @brief Waits for the oldest queued batch, registers it, and returns
   *  its keys. Empty when nothing is queued. */
  std::vector<const void *> prefetch_qs4cx_wh_experts_end() override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const uint64_t t0 = HtpProfile::nowUs();
    std::unique_ptr<PrefetchBatch> batch;
    size_t done_on_entry = 0;
    {
      std::unique_lock<std::mutex> lock(prefetch_mutex_);
      if (prefetch_batches_.empty())
        return {};
      PrefetchBatch *b = prefetch_batches_.front().get();
      done_on_entry = b->done;
      prefetch_done_cv_.wait(lock,
                             [b] { return b->done == b->experts.size(); });
      batch = std::move(prefetch_batches_.front());
      prefetch_batches_.pop_front();
    }
    const uint64_t t_wait = HtpProfile::nowUs();
    std::lock_guard<std::mutex> lock(handle_mutex_);
    std::exception_ptr first;
    try {
      registerStagedBatch(session, batch->experts);
    } catch (...) {
      first = std::current_exception();
    }
    std::vector<const void *> keys;
    for (const StagedExpert &st : batch->experts)
      if (experts_.count(st.d.key_gu) != 0)
        keys.push_back(st.d.key_gu);
    HtpProfile &profile = HtpProfile::global();
    if (profile.level() != 0) // the read time is only what the calls left
      profile.addPrefetch(batch->experts.size(), done_on_entry, t_wait - t0,
                          HtpProfile::nowUs() - t_wait, prefetch_caller_cpus_,
                          prefetch_reader_cpus_.load());
    if (first)
      std::rethrow_exception(first);
    return keys;
  }

  void reserve_qs4cx_wh_expert_slots(
    size_t n, const ExpertFileDesc *shape = nullptr) override {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    expert_slots_wanted_ = n;
    if (!shape || expert_slots_placed_ >= n)
      return;
    const auto session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    // Keep existing free slots aside so takeExpertSlot must place new ones.
    // Restore every allocated slot on failure as well as on success.
    auto available = std::move(free_expert_slots_);
    free_expert_slots_.clear();
    try {
      available.reserve(available.size() + n - expert_slots_placed_);
      while (expert_slots_placed_ < n)
        available.push_back(takeExpertSlot(session, *shape));
    } catch (...) {
      free_expert_slots_ = std::move(available);
      throw;
    }
    free_expert_slots_ = std::move(available);
  }

  /**
   * @brief Retires one expert without a round trip (doc 52 section 10.12).
   *
   * The expert stops being a hit at once -- its keys leave handle_cache_ --
   * but its two DSP handles stay registered and travel with the slot into
   * the free list. The next expert read into that slot releases them in
   * the same weight_swap_u8i4_arena call that registers it. That is safe
   * because nothing can pass a retired handle to the DSP any more, and the
   * slot's bytes are only overwritten between calls, never under one that
   * reads them. Handles held this way count against the DSP's table, but
   * only up to the number of free slots, which a steady state keeps at
   * zero: every eviction is followed by the load that wanted its slot.
   */
  bool release_qs4cx_wh_expert(const void *key_gu) override {
    ExpertFileDesc d;
    if (!releaseExpert(key_gu, d))
      return false;
    if (fadviseKnob() == 1)
      adviseLater({{d, true}});
    return true;
  }

  /** @brief release_qs4cx_wh_expert without the advice; @a d gets the
   *  victim's file ranges. */
  bool releaseExpert(const void *key_gu, ExpertFileDesc &d) {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    auto it = experts_.find(key_gu);
    if (it == experts_.end())
      return false;
    d = it->second.d;
    ExpertSlot slot = it->second.slot;
    slot.h_gu = it->second.h_gu;
    slot.h_dn = it->second.h_dn;
    handle_cache_.erase(key_gu);
    handle_cache_.erase(it->second.key_dn);
    free_expert_slots_.push_back(slot);
    experts_.erase(it);
    return true;
  }

  bool supports_gemm_qs4cx_fused_swiglu_fp32() const override { return true; }

  void gemm_qs4cx_fused_swiglu_fp32(std::vector<void *> matAdata,
                                    std::vector<float *> matAscale,
                                    float *matBdata, float *matCdata,
                                    unsigned int M, std::vector<unsigned int> N,
                                    unsigned int K) override {
    if (matAdata.size() != 2 || matAscale.size() != 2 || N.size() != 2) {
      throw std::invalid_argument(
        "gemm_qs4cx_fused_swiglu_fp32 needs exactly [gate_up, down]");
    }
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const unsigned int inter = N[0] / 2;
    const uint32_t h_gu =
      get_or_register_qs4cx(matAdata[0], matAscale[0], session, K, N[0]);
    const uint32_t h_dn =
      get_or_register_qs4cx(matAdata[1], matAscale[1], session, inter, N[1]);
    // ponytail: split-call path (invokeGateUpSwiglu + invokeLayerU8In), NOT
    // invokeFused. The one-call fused kernel (still in the tree, dormant --
    // see invokeFused's own comment) produced wrong output on this model's
    // real weights; root cause never found despite passing its own
    // synthetic-data unit test. This is the smaller-surface alternative
    // doc 43 §7 describes: gate_up+SwiGLU+requant in one call (one weight,
    // one VTCM region set), feeding the requantized u8 intermediate into
    // the ALREADY-device-verified u8in path for down instead of
    // reimplementing the down matmul. Verified stage-by-stage on device
    // (unittest_hvx_mm_u8i4's GateUpSwiglu* tests) before being wired here.
    invokeGateUpSwiglu(session, h_gu, matBdata, M, K, inter);
    invokeLayerU8InRaw(session, &h_dn, 1, htp_act_m_pad(M), matCdata, M, N[1],
                       inter);
    if (l2DiffEnabled() || l2ShadowEnabled())
      l2Diff(session, h_gu, h_dn, matBdata, matCdata, M, K, inter, N[1]);
  }

  /**
   * @brief NNTR_L2_DIFF: the one control neither L2 attempt ever applied.
   *
   * Both attempts were gated on unittest_hvx_mm_u8i4's SNR against
   * fill_deterministic weights and activations, passed at 138-141 dB, and
   * still broke the real model -- so the variable nobody has held fixed is
   * the DATA, not the kernel. This runs the unit test's exact reference
   * (mm_u8i4_layer on gate_up, SwiGLU on the host with expf, mm_u8i4_layer
   * on down) against the split-call path, on THIS model's registered weight
   * bytes and THIS forward's real activation, and reports the SNR per call.
   *
   * It bisects the search in one run:
   *   high SNR -> the kernels agree on real data too, so the fault is on the
   *               ARM side of the call (what is fed in, what is done with
   *               what comes out), not inside the DSP;
   *   low SNR  -> the kernels genuinely disagree on real data, and the
   *               synthetic distribution was hiding it -- the dB number to
   *               chase, with the offending M printed next to it.
   *
   * Debug path: plain heap buffers, three extra FastRPC round trips per
   * expert. Never on unless the env var is.
   */
  void l2Diff(remote_handle64 session, uint32_t h_gu, uint32_t h_dn,
              const float *matBdata, float *got, unsigned int M, unsigned int K,
              unsigned int inter, unsigned int N_out) {
    std::vector<float> act(static_cast<size_t>(M) * K);
    std::memcpy(act.data(), matBdata, act.size() * sizeof(float));

    std::vector<float> gu_out(static_cast<size_t>(M) * 2 * inter, 0.0f);
    int err = nntr_hvx_mm_u8i4_layer(
      session, M, K, &h_gu, 1, act.data(), static_cast<int>(act.size()),
      gu_out.data(), static_cast<int>(gu_out.size()));
    if (err != AEE_SUCCESS) {
      std::fprintf(stderr, "[L2-DIFF] reference gate_up failed: %d\n", err);
      return;
    }

    // [A1] swiglu_det, the same specification Lfm2MoELayer's two-dot path
    // and the DSP's hvx_swiglu_det.h both run. This used std::exp, which
    // made total_flips measure the wrong thing: a nonzero count could mean
    // either that the fused kernel disagreed with the host path the model
    // actually uses, or merely that both disagreed with libm. Against this
    // reference, total_flips == 0 is exactly the property the fused path
    // needs, and anything else is a real divergence.
    std::vector<float> mid(static_cast<size_t>(M) * inter);
    for (unsigned int m = 0; m < M; ++m) {
      for (unsigned int j = 0; j < inter; ++j) {
        const float g = gu_out[static_cast<size_t>(m) * 2 * inter + j];
        const float u = gu_out[static_cast<size_t>(m) * 2 * inter + inter + j];
        mid[static_cast<size_t>(m) * inter + j] = swiglu_det_one(g, u);
      }
    }

    std::vector<float> ref(static_cast<size_t>(M) * N_out, 0.0f);
    err = nntr_hvx_mm_u8i4_layer(session, M, inter, &h_dn, 1, mid.data(),
                                 static_cast<int>(mid.size()), ref.data(),
                                 static_cast<int>(ref.size()));
    if (err != AEE_SUCCESS) {
      std::fprintf(stderr, "[L2-DIFF] reference down failed: %d\n", err);
      return;
    }

    double sig = 0.0, noise = 0.0, max_abs_err = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
      const double r = ref[i], d = static_cast<double>(got[i]) - r;
      sig += r * r;
      noise += d * d;
      if (std::fabs(d) > max_abs_err)
        max_abs_err = std::fabs(d);
    }
    const double snr = (noise == 0.0) ? 999.0 : 10.0 * std::log10(sig / noise);

    // Outlier-row hypothesis: hvx_quant_rows_u8_params sets ONE scale per
    // row from that row's own min/max (the same scan the NaN bug poisoned
    // -- doc 43 section 5). A real trained model's SwiGLU intermediate can
    // have an outlier value fill_deterministic's bounded uniform fill never
    // produces; if one lane dominates a row's range, the rest of that row's
    // 255 levels spread thinner and the row's requant noise rises without
    // ever going non-finite. `mid` is exactly that intermediate (pre-
    // quantization, host f32), already computed above for the reference --
    // this reuses it rather than adding a DSP round trip.
    //
    // Printed for EVERY call, not just suspect ones: the first round only
    // ever showed the bad calls' own spans (0.05-0.24), with nothing to
    // compare them against, so the hypothesis was untestable -- a narrow
    // span could mean "outlier-free" (hypothesis holds) or "this model's
    // rows are always this narrow" (hypothesis is irrelevant). This line
    // is the missing baseline.
    float call_max_span = 0.0f;
    size_t call_max_span_row = 0;
    for (unsigned int m = 0; m < M; ++m) {
      float row_min = mid[static_cast<size_t>(m) * inter];
      float row_max = row_min;
      for (unsigned int j = 0; j < inter; ++j) {
        const float v = mid[static_cast<size_t>(m) * inter + j];
        row_min = std::min(row_min, v);
        row_max = std::max(row_max, v);
      }
      if (row_max - row_min > call_max_span) {
        call_max_span = row_max - row_min;
        call_max_span_row = m;
      }
    }

    if (snr < 100.0) {
      size_t worst_row = 0;
      double worst_row_err = -1.0;
      for (unsigned int m = 0; m < M; ++m) {
        double row_err = 0.0;
        for (unsigned int n = 0; n < N_out; ++n) {
          const double d =
            static_cast<double>(got[static_cast<size_t>(m) * N_out + n]) -
            ref[static_cast<size_t>(m) * N_out + n];
          row_err += d * d;
        }
        if (row_err > worst_row_err) {
          worst_row_err = row_err;
          worst_row = m;
        }
      }
      float row_min = mid[worst_row * inter], row_max = row_min;
      for (unsigned int j = 0; j < inter; ++j) {
        const float v = mid[static_cast<size_t>(worst_row) * inter + j];
        row_min = std::min(row_min, v);
        row_max = std::max(row_max, v);
      }

      // Bin-flip test, take 1 (dev_scale held fixed): is this a rounding-
      // boundary sensitivity given a SHARED scale? hvx_quant_rows_u8_params/
      // hvx_quant_pack_u8_ah are the SAME function on both paths -- verified
      // by inspection, not assumed -- so a difference cannot come from the
      // quantizer having two implementations. What CAN differ is the float
      // value handed to it: the reference quantizes the host's exact
      // expf(); the device quantized its own hvx_exp_sf/hvx_recip_qf32
      // approximation (~1e-6 relative error, well within spec -- doc 43's
      // L2 fix candidate). act_ah_buf_/act_scale_scratch_/act_zp_scratch_
      // still hold the device's own gate_up_swiglu output for this exact
      // forward's worst row -- invokeLayerU8InRaw only reads them, never
      // clears them -- so this first pass re-quantizes `mid` with the
      // DEVICE's OWN scale/zp and counts per-element disagreements.
      //
      // Take 2 (below, scale_diff/total_flips): the first round's result
      // (0 flips on the two WORST calls, 1 flip on the three milder ones)
      // means take 1 tests the wrong thing when it comes back clean --
      // hvx_quant_rows_u8_params derives scale/zp from THIS row's own
      // min/max, so if the approximation nudges the row's extreme value
      // even slightly, the REFERENCE path (which quantizes `mid` with a
      // scale/zp it computes fresh from `mid`'s own min/max, independent of
      // the device's) uses a DIFFERENT scale than the device did -- not a
      // sparse per-element flip but a systematic per-row bias that shifts
      // every one of the row's `inter` dequantized values in the same
      // direction, which a down matmul's summation does not cancel out.
      // Reimplements hvx_quant_rows_u8_params' exact formula (source read,
      // not guessed) on `mid` alone to get that independent scale/zp, then
      // diffs against the device's actual bytes the same way take 1 did.
      constexpr uint32_t kTileRow = 64, kTileInner = 32, kActTileBytes = 2048;
      const uint32_t n_ktiles = inter / kTileInner;
      const uint32_t rb = static_cast<uint32_t>(worst_row) / kTileRow;
      const uint32_t r = static_cast<uint32_t>(worst_row) % kTileRow;
      const float dev_scale = act_scale_scratch_[worst_row];
      const int32_t dev_zp = act_zp_scratch_[worst_row];
      const uint8_t *ah = act_ah_buf_->data();

      const float rmin = std::min(row_min, 0.0f);
      const float rmax = std::max(row_max, 0.0f);
      const float host_scale = (rmax > rmin) ? (rmax - rmin) / 255.0f : 1.0f;
      const int32_t host_zp =
        (rmax > rmin)
          ? std::max(0, std::min(255, static_cast<int32_t>(
                                        std::nearbyint(-rmin / host_scale))))
          : 0;

      // Flip count over the WHOLE block, not just worst_row. The previous
      // round counted only the worst row and its result ("1 flip") was
      // reported as if it described the call -- it did not. This is the
      // number that can honestly be held against row_sq_err and the SNR.
      uint32_t block_flips = 0;
      for (uint32_t m = 0; m < M; ++m) {
        const float s_m = act_scale_scratch_[m];
        const int32_t z_m = act_zp_scratch_[m];
        if (s_m == 0.0f)
          continue;
        const uint32_t rb_m = m / kTileRow, r_m = m % kTileRow;
        for (uint32_t j = 0; j < inter; ++j) {
          const uint32_t kt = j / kTileInner, c = j % kTileInner;
          const size_t idx =
            (static_cast<size_t>(rb_m) * n_ktiles + kt) * kActTileBytes +
            r_m * kTileInner + c;
          int32_t hb = static_cast<int32_t>(std::nearbyint(
                         mid[static_cast<size_t>(m) * inter + j] / s_m)) +
                       z_m;
          hb = std::max(0, std::min(255, hb));
          if (static_cast<int32_t>(ah[idx]) != hb)
            ++block_flips;
        }
      }

      uint32_t flips = 0, total_flips = 0;
      int32_t max_flip = 0, max_total_flip = 0;
      for (uint32_t j = 0; j < inter; ++j) {
        const uint32_t kt = j / kTileInner, c = j % kTileInner;
        const size_t idx =
          (static_cast<size_t>(rb) * n_ktiles + kt) * kActTileBytes +
          r * kTileInner + c;
        const int32_t dev_byte = ah[idx];
        const float v = mid[static_cast<size_t>(worst_row) * inter + j];

        int32_t byte_dev_scale =
          dev_scale != 0.0f
            ? static_cast<int32_t>(std::nearbyint(v / dev_scale)) + dev_zp
            : dev_zp;
        byte_dev_scale = std::max(0, std::min(255, byte_dev_scale));
        const int32_t d1 = dev_byte - byte_dev_scale;
        if (d1 != 0) {
          ++flips;
          if (std::abs(d1) > std::abs(max_flip))
            max_flip = d1;
        }

        int32_t byte_host_scale =
          static_cast<int32_t>(std::nearbyint(v / host_scale)) + host_zp;
        byte_host_scale = std::max(0, std::min(255, byte_host_scale));
        const int32_t d2 = dev_byte - byte_host_scale;
        if (d2 != 0) {
          ++total_flips;
          if (std::abs(d2) > std::abs(max_total_flip))
            max_total_flip = d2;
        }
      }
      const double scale_diff_pct =
        host_scale != 0.0f
          ? 100.0 * (static_cast<double>(dev_scale) - host_scale) / host_scale
          : 0.0;

      std::fprintf(
        stderr,
        "[L2-DIFF] M=%-4u K=%u inter=%u N=%u  snr=%8.2f dB  "
        "max_abs_err=%g  worst_row=%zu row_sq_err=%g "
        "mid_range=[%.4f, %.4f] mid_span=%.4f  call_max_span=%.4f@row%zu  "
        "bin_flips=%u/%u max_flip=%d  dev_scale=%.9g host_scale=%.9g "
        "scale_diff=%.6f%%  total_flips=%u/%u max_total_flip=%d  "
        "block_flips=%u/%u\n",
        M, K, inter, N_out, snr, max_abs_err, worst_row, worst_row_err, row_min,
        row_max, row_max - row_min, call_max_span, call_max_span_row, flips,
        inter, max_flip, dev_scale, host_scale, scale_diff_pct, total_flips,
        inter, max_total_flip, block_flips, static_cast<uint32_t>(M) * inter);
      if (l2ShadowEnabled())
        std::memcpy(got, ref.data(), ref.size() * sizeof(float));
      return;
    }
    std::fprintf(stderr,
                 "[L2-DIFF] M=%-4u K=%u inter=%u N=%u  snr=%8.2f dB  "
                 "max_abs_err=%g  call_max_span=%.4f@row%zu\n",
                 M, K, inter, N_out, snr, max_abs_err, call_max_span,
                 call_max_span_row);
    if (l2ShadowEnabled())
      std::memcpy(got, ref.data(), ref.size() * sizeof(float));
  }

private:
  /** @brief Grows @a buf to at least @a bytes, reusing it otherwise.
   *
   *  One buffer for activation, one for output, reused across every call
   *  instead of one rpcmem alloc/free per matmul -- the buffers this
   *  replaces (the tensor pool's plain heap pointers, passed straight
   *  through) are pinned and mapped by the FastRPC driver on EVERY call
   *  (htp_rpcmem.h's own doc comment; measured at ~155 MB/s vs ION's
   *  44.6 GB/s, docs/htp_attention/34_fc_measured.md section4 item F). The
   *  memcpy this adds is the trade: cheap relative to a per-call pin+map,
   *  but that trade is exactly what NNTR_HTP_PROFILE's transport column is
   *  for confirming on device -- do not assume the win without it. */
  static void ensureCapacity(std::unique_ptr<HtpRpcBuffer> &buf, size_t bytes) {
    if (!buf || buf->size() < bytes) {
      buf = std::make_unique<HtpRpcBuffer>(bytes);
    }
  }

  /** @brief ION scratch by size class, one buffer per power of two from
   *  64 KiB up, so a call stages through a buffer near its own size.
   *
   * One pair grown to the largest shape (ensureCapacity) made every call
   * pay for that shape: the driver's cache maintenance on a cached ION
   * buffer covers the whole dma-buf, not the bytes passed, so the MoE
   * decode call's 8 KB rode a 3.6 MB pair at 553 us of transport, a
   * 10.9 MB out buffer at 814 and a 12.7 MB pair at 1633 (doc 50 section
   * 3.6: 36-59 us/MB, cache-flush speed), and prefill's calls paid the
   * same. With classes, decode stays on the 64 KiB pair. */
  /** @brief The host range a staging buffer stands for. kept: its call
   *  left only sampledSame's windows on the host (copyOutOf's to_host
   *  false), so a copy of any part of it must come from the buffer
   *  (trueSource); a model tensor, alive as long as the model. */
  struct Held {
    const void *first;
    size_t second;
    bool kept = false;
  };
  struct StagingPool {
    std::map<size_t, std::unique_ptr<HtpRpcBuffer>> by_class;
    /** What stageInput last copied into a class's buffer, cleared by any
     *  other stage() of it (doc 57 section 9.31). */
    std::map<size_t, Held> holds;
    /** The host tensor a class's buffer was last copied out to, whole
     *  (copyOutOf), cleared by any other stage() of it. */
    std::map<size_t, Held> copied_out;
  };
  static size_t stageClass(size_t bytes) {
    size_t cls = size_t(64) << 10;
    while (cls < bytes)
      cls <<= 1;
    return cls;
  }
  /** @brief @a slot made large enough for @a bytes. The class only names
   *  the slot: the buffer is as large as the largest tensor it has held
   *  (plus a page), not the class, because the driver's cache maintenance
   *  covers the whole dma-buf on every call it rides (11.5 MB of rows rode
   *  16 MiB, 23 MB rode 32 MiB). Growing drops the old buffer, so the
   *  callers do it only where the old bytes are no longer wanted. */
  void fit(std::unique_ptr<HtpRpcBuffer> &slot, size_t bytes) {
    if (slot && slot->size() >= bytes)
      return;
    if (slot) {
      const uint8_t *lo = slot->data(), *hi = lo + slot->size();
      for (IonAlias &a : ion_alias_)
        if (a.ion >= static_cast<const void *>(lo) &&
            a.ion < static_cast<const void *>(hi))
          a = IonAlias{};
    }
    slot = std::make_unique<HtpRpcBuffer>(
      ((bytes + 4095) & ~static_cast<size_t>(4095)) + 4096);
  }
  HtpRpcBuffer &stage(StagingPool &pool, size_t bytes) {
    const size_t cls = stageClass(bytes);
    auto &slot = pool.by_class[cls];
    // A tensor a call kept in this buffer (copyOutOf's to_host false) has
    // only its windows on the host; once the buffer is reused nobody can
    // find its bytes any more (trueSource / ionAlias see the new bytes),
    // and a later reader of the host tensor would take the stale rows. So
    // the host tensor is made whole first -- when its windows still match
    // the buffer, i.e. the host has not rewritten it since. Only a step
    // whose sizes collide in one class pays (a 441-row chunk after 2048-row
    // ones: 2048 rows put every tensor in its own class).
    for (auto *m : {&pool.copied_out, &pool.holds}) {
      auto it = m->find(cls);
      if (it != m->end() && it->second.kept && slot &&
          sampledSame(it->second.first, slot->data(), it->second.second))
        stagedMemcpy(const_cast<void *>(it->second.first), slot->data(),
                     it->second.second);
    }
    pool.holds.erase(cls);
    pool.copied_out.erase(cls);
    fit(slot, bytes);
    return *slot;
  }
  /** @brief Whether @a b still holds the copy of @a a: 32 spread 64-byte
   *  windows and the tail compared. ponytail: a rewrite that keeps every
   *  sampled window is missed; a full compare costs about what the copy
   *  does. */
  static bool sampledSame(const void *a_, const void *b_, size_t bytes) {
    const uint8_t *a = static_cast<const uint8_t *>(a_);
    const uint8_t *b = static_cast<const uint8_t *>(b_);
    bool same =
      bytes < 64 || std::memcmp(a + bytes - 64, b + bytes - 64, 64) == 0;
    for (size_t i = 0; same && i < 32 && bytes >= 64; ++i) {
      const size_t off = (bytes - 64) / 32 * i;
      same = std::memcmp(a + off, b + off, 64) == 0;
    }
    return same;
  }
  /** @brief stage() plus the copy of @a bytes from @a src, skipped when the
   *  buffer still holds exactly that copy: the router, dense and MoE calls
   *  of a Gemma-4 layer each stage the same 11.5 MB post-attention rows.
   *  Any other stage() of the buffer forgets the copy, and before skipping
   *  the copy is compared (sampledSame) with @a src, so a host tensor
   *  rewritten (or another one placed at the same address) is copied again.
   *
   *  With @a from, a buffer of that pool that @a src was copied out of
   *  whole (copyOutOf) and still matches is swapped in instead of copied:
   *  the dense output is the MoE call's x2 and the MoE output the next
   *  qkv call's input (doc 57 section 9.34). Stage this before the call's
   *  own output buffer, which may be the one swapped in. */
  HtpRpcBuffer &stageInput(StagingPool &pool, const void *src, size_t bytes,
                           StagingPool *from = nullptr,
                           int line = __builtin_LINE()) {
    const size_t cls = stageClass(bytes);
    auto &slot = pool.by_class[cls];
    auto it = pool.holds.find(cls);
    if (it != pool.holds.end() && slot && it->second.first == src &&
        it->second.second == bytes && sampledSame(src, slot->data(), bytes))
      return *slot;
    if (from) {
      // what from's buffer of this size is: a call's output (copied_out),
      // or an input staged there (holds: the qkv call's activation rows are
      // the o-proj call's residual)
      auto c = from->copied_out.find(cls);
      const Held *was = c != from->copied_out.end() ? &c->second : nullptr;
      if (!was) {
        auto h = from->holds.find(cls);
        was = h != from->holds.end() ? &h->second : nullptr;
      }
      auto &fslot = from->by_class[cls];
      if (was && was->first == src && was->second == bytes && fslot &&
          sampledSame(src, fslot->data(), bytes)) {
        const bool kept = was->kept;
        // the buffer handed to from: a kept tensor of pool's class goes
        // with it. In a layer's own order that tensor has had its last
        // reader (the previous dense output, the layer input, the o-proj
        // output: 3 a layer, 11.5 MB each at 1024 rows), so it is not
        // copied to the host; it is remembered as lost, and a later read of
        // it throws instead of taking the windows-only host bytes.
        for (auto *m : {&pool.copied_out, &pool.holds}) {
          auto k = m->find(cls);
          if (k != m->end() && k->second.kept && slot &&
              k->second.first != src &&
              sampledSame(k->second.first, slot->data(), k->second.second))
            noteLost(k->second.first, k->second.second);
        }
        std::swap(slot, fslot);
        from->holds.erase(cls);
        from->copied_out.erase(cls);
        pool.copied_out.erase(cls);
        pool.holds[cls] = {src, bytes, kept};
        return *slot;
      }
    }
    // A copy after all (a row chunk of a tensor, a size the pools hold no
    // buffer of): from where the bytes really are -- a tensor a call kept
    // (copyOutOf's to_host false) has only its windows on the host.
    const void *from_buf = trueSource(src, bytes);
    static const bool stage_debug = std::getenv("NNTR_HTP_STAGE_DEBUG");
    if (stage_debug && from_buf == src) {
      // a host range that a kept record covers, whose buffer no longer
      // matches: the host bytes are the stale windows-only copy
      for (StagingPool *pl : {&out_pool_, &act_pool_, &x2_pool_})
        for (auto *m : {&pl->copied_out, &pl->holds})
          for (const auto &rec : *m) {
            const char *h = static_cast<const char *>(rec.second.first);
            const char *sc = static_cast<const char *>(src);
            if (rec.second.kept && h && sc >= h &&
                sc + bytes <= h + rec.second.second)
              ml_logw("HTP stage: STALE host read of %zu B at line %d (kept "
                      "record pool=%p cls=%zu bytes=%zu)",
                      bytes, line, static_cast<void *>(pl), rec.first,
                      rec.second.second);
          }
    }
    if (from_buf == src)
      throwIfLost(src, bytes, line);
    // a source inside this very buffer means it is large enough already
    fit(slot, bytes);
    if (from_buf != slot->data())
      stagedMemcpy(slot->data(), from_buf, bytes, line);
    pool.holds[cls] = {src, bytes, false};
    return *slot;
  }
  /** @brief A kept host tensor whose staging buffer was handed on
   *  (stageInput's swap) without the copy to the host: its range and the
   *  windows sampledSame compares, as they were. */
  struct Lost {
    const void *host = nullptr;
    size_t bytes = 0;
    std::array<uint8_t, 33 * 64> win{};
  };
  std::array<Lost, 8> lost_{};
  size_t lost_next_ = 0;
  static void sampledPack(uint8_t *win, const void *src_, size_t bytes) {
    const uint8_t *s = static_cast<const uint8_t *>(src_);
    std::memcpy(win, s + bytes - 64, 64);
    for (size_t i = 0; i < 32; ++i)
      std::memcpy(win + 64 * (i + 1), s + (bytes - 64) / 32 * i, 64);
  }
  void noteLost(const void *host, size_t bytes) {
    if (bytes < 64)
      return;
    Lost &l = lost_[lost_next_++ % lost_.size()];
    l.host = host;
    l.bytes = bytes;
    sampledPack(l.win.data(), host, bytes);
  }
  /** @brief Throws when [src, src + bytes) is inside a lost tensor the host
   *  has not rewritten since: those host bytes are only the windows.
   *  ponytail: the last 8 lost tensors are remembered (3 are lost a layer);
   *  a reader further behind than that is not caught. */
  void throwIfLost(const void *src, size_t bytes, int line) const {
    const char *s = static_cast<const char *>(src);
    for (const Lost &l : lost_) {
      const char *h = static_cast<const char *>(l.host);
      if (!h || s < h || s + bytes > h + l.bytes)
        continue;
      std::array<uint8_t, 33 * 64> now;
      sampledPack(now.data(), l.host, l.bytes);
      if (now == l.win)
        throw std::runtime_error(
          "HTP stage: host read (line " + std::to_string(line) +
          ") of a kept tensor whose staging buffer was reused; "
          "NNTR_HTP_KEEP_OUT=0 copies every output to the host");
    }
  }
  /** @brief Where [src, src + bytes) really is: inside a staging buffer
   *  that holds (by sampledSame over its whole tensor) a kept host tensor
   *  containing the range, else @a src itself. Only kept ones: any other
   *  record's host range may be memory freed since. */
  const void *trueSource(const void *src, size_t bytes) {
    const char *s = static_cast<const char *>(src);
    for (StagingPool *pool : {&out_pool_, &act_pool_, &x2_pool_}) {
      for (auto *m : {&pool->copied_out, &pool->holds}) {
        for (const auto &rec : *m) {
          const char *h = static_cast<const char *>(rec.second.first);
          const size_t hb = rec.second.second;
          auto slot = pool->by_class.find(rec.first);
          if (!rec.second.kept || !h || s < h || s + bytes > h + hb ||
              slot == pool->by_class.end() || !slot->second)
            continue;
          if (sampledSame(h, slot->second->data(), hb))
            return slot->second->data() + (s - h);
        }
      }
    }
    return src;
  }
  /** @brief The windows sampledSame compares, copied: what makes a host
   *  tensor that was NOT copied out still recognizable as this buffer's. */
  static void sampledCopy(void *dst_, const void *src_, size_t bytes) {
    uint8_t *d = static_cast<uint8_t *>(dst_);
    const uint8_t *s = static_cast<const uint8_t *>(src_);
    if (bytes < 64) {
      std::memcpy(d, s, bytes);
      return;
    }
    std::memcpy(d + bytes - 64, s + bytes - 64, 64);
    for (size_t i = 0; i < 32; ++i) {
      const size_t off = (bytes - 64) / 32 * i;
      std::memcpy(d + off, s + off, 64);
    }
  }
  /** @brief The copy of a call's whole output out of @a pool's buffer for
   *  @a bytes, remembered for stageInput's @a from.
   *
   *  @a to_host false: the layer says only accelerator calls read this
   *  output at prefill, so only sampledSame's windows go to the host (2 KB
   *  of an 11.5 MB tensor): the calls that take it find the buffer by the
   *  same content check as ever and swap it in, and a different tensor a
   *  CPU layer writes at that address still fails the check and is copied.
   *  A host READER of this tensor sees stale bytes between the windows: the
   *  builder sets the flag only where every consumer runs here, the layers'
   *  CPU fallbacks call sync_to_host first, and NNTR_HTP_PROFILE >= 2 (its
   *  host norms read these tensors) copies in full. */
  static void copyOutOf(StagingPool &pool, void *dst, size_t bytes,
                        bool to_host = true, int line = __builtin_LINE()) {
    const size_t cls = stageClass(bytes);
    const void *buf = pool.by_class[cls]->data();
    // NNTR_HTP_KEEP_OUT=0: every output copied to the host whole (a
    // reference for the kept path; never set in a tok/s run)
    static const bool keep_off = [] {
      const char *v = std::getenv("NNTR_HTP_KEEP_OUT");
      return v != nullptr && std::strcmp(v, "0") == 0;
    }();
    bool kept = !to_host && !keep_off && HtpProfile::global().level() < 2;
    if (kept) {
      sampledCopy(dst, buf, bytes);
      // the windows must be what sampledSame compares, or the readers
      // would take the stale host tensor: checked, the full copy otherwise
      kept = sampledSame(dst, buf, bytes);
    }
    if (!kept)
      stagedMemcpy(dst, buf, bytes, line);
    pool.copied_out[cls] = {dst, bytes, kept};
  }

  /** @brief A host range last copied out of a staging buffer, so a later
   *  call can hand the DSP that buffer's bytes in place of a fresh copy:
   *  the qkv call's Q block is the attention's Q, the attention's output
   *  the o-proj's input. The staging buffers live as long as the ops, so
   *  the pointer stays valid; whether its bytes still are the host's is
   *  sampledSame's to say at use. */
  struct IonAlias {
    const void *host = nullptr;
    const void *ion = nullptr;
    size_t bytes = 0;
  };
  std::array<IonAlias, 16> ion_alias_{};
  size_t ion_alias_next_ = 0;
  void noteIonAlias(const void *host, const void *ion, size_t bytes) {
    ion_alias_[ion_alias_next_++ % ion_alias_.size()] = {host, ion, bytes};
  }
  /** @brief The staging bytes that hold [host, host + bytes), or null. */
  const void *ionAlias(const void *host, size_t bytes) const {
    const char *h = static_cast<const char *>(host);
    for (const IonAlias &a : ion_alias_) {
      const char *b = static_cast<const char *>(a.host);
      if (!b || h < b || h + bytes > b + a.bytes)
        continue;
      const void *ion = static_cast<const char *>(a.ion) + (h - b);
      if (sampledSame(host, ion, bytes))
        return ion;
    }
    return nullptr;
  }

  /** @brief The attention's u16 context left in its staging buffer for the
   *  o-proj (sdpa_q2_kvcache_f32 -> gemm_qs4cx_res_add_fp32): the host
   *  tensor it stands for, the buffer, its shape and the per-head
   *  encodings. One at a time: the graph runs a layer's attention, then its
   *  o-proj. ponytail: a CPU consumer of that f32 tensor, or another
   *  attention call before the o-proj, is not covered -- the builder sets
   *  mha_core's attn_out_u16 only for the folded o-proj on this engine. */
  struct AttnOut16 {
    const float *host = nullptr;
    const uint16_t *ion = nullptr;
    unsigned int rows = 0, width = 0, head_dim = 0;
    std::vector<float> enc;
  } attn_out16_;

  /** @brief The one FastRPC layer call both accelerated entries make.
   *
   *  Under NNTR_HTP_PROFILE >= 2 it goes through mm_u8i4_layer_timed so the
   *  DSP's own microseconds come back alongside the host wall clock; the
   *  difference between the two is the FastRPC transport. The production
   *  path (profile off) still calls the untimed entry, which is why the
   *  DSP probes cost nothing when nobody is measuring.
   *
   *  Not static any more: it now owns act_pool_/out_pool_, rpcmem-backed
   *  scratch buffers by size class reused across calls (stage above).
   *  invoke_mutex_ guards them -- required once they are shared
   *  mutable state, and consistent with the single-owner assumption
   *  test/htp/nntr_hvx_session.h already documents for the one HTP session
   *  a process opens (one VTCM arena, one HMX lock).
   */
  /** @brief Copies the kernel's output back into the caller's [M x N].
   *
   * hexkl_mm_u8i4_layer_run writes one contiguous [M x N_i] block per
   * handle, in call order (its out_off += M * h->N), not a row-major
   * [M x sum(N_i)]. One handle is therefore the whole [M x N] and copies
   * straight through; the slices of one weight (get_or_register_fc) are
   * interleaved back, row by row, into their column ranges; separate
   * weights (@a dsts, the batch call) each get their block whole. */
  static void copyOut(float *dst, const float *out_cat, unsigned int M,
                      unsigned int N, const std::vector<unsigned int> *blocks,
                      const std::vector<float *> *dsts = nullptr,
                      int line = __builtin_LINE()) {
    if (dsts) {
      size_t off = 0;
      for (size_t i = 0; i < dsts->size(); ++i) {
        const size_t block = static_cast<size_t>(M) * (*blocks)[i];
        stagedMemcpy((*dsts)[i], out_cat + off, block * sizeof(float), line);
        off += block;
      }
      return;
    }
    if (!blocks || blocks->size() < 2) {
      stagedMemcpy(dst, out_cat, static_cast<size_t>(M) * N * sizeof(float),
                   line);
      return;
    }
    size_t off = 0;
    unsigned int c0 = 0;
    for (unsigned int n : *blocks) {
      for (unsigned int r = 0; r < M; ++r) {
        stagedMemcpy(dst + static_cast<size_t>(r) * N + c0,
                     out_cat + off + static_cast<size_t>(r) * n,
                     static_cast<size_t>(n) * sizeof(float), line);
      }
      off += static_cast<size_t>(M) * n;
      c0 += n;
    }
  }

  /** @param blocks N of each handle in call order, when @a handles are the
   *  slices of one weight and matCdata is its whole [M x N], or separate
   *  weights whose [M x N_i] blocks go to @a dsts[i]; NULL when each
   *  handle's output stays a block of its own (or there is only one). */
  /** @brief Where one handle's [M x cols] output block lands: columns
   *  [c0, c0 + cols) of a [M x stride] destination. */
  struct OutSlice {
    float *dst;
    unsigned int stride, c0, cols;
  };

  /** @brief The RMSNorms folded into one layer call (mm_u8i4_layer_norm):
   *  nothing set = the plain call. */
  struct FusedNorms {
    const float *pre_gamma = nullptr;                      /**< K floats */
    const std::vector<unsigned int> *post_chunk = nullptr; /**< per handle */
    const float *post_gamma = nullptr; /**< the chunks, concatenated */
    float eps = 0.0f;
    const float *rope_cs = nullptr; /**< M rows of 2*rope_hd, or nullptr */
    unsigned int rope_hd = 0;       /**< 0: no RoPE */
    unsigned int rope_handles = 0;  /**< the handles rotated, a prefix */
    /** false: only sampledSame's windows of each weight's rows go to the
     *  host tensors (copyOutOf's to_host); host_view finds the rest */
    bool to_host = true;
    bool any() const { return pre_gamma || post_chunk || rope_hd; }
  };

  void invokeLayer(remote_handle64 session, const uint32_t *handles,
                   int num_handles, float *matBdata, float *matCdata,
                   unsigned int M, unsigned int N, unsigned int K,
                   const std::vector<unsigned int> *blocks = nullptr,
                   const std::vector<float *> *dsts = nullptr,
                   const FusedNorms *norms = nullptr,
                   const std::vector<OutSlice> *slices = nullptr) {
    const int act_len = static_cast<int>(M) * static_cast<int>(K);
    const int out_len = static_cast<int>(M) * static_cast<int>(N);
    const auto shape = [&]() {
      return " (M=" + std::to_string(M) + " K=" + std::to_string(K) +
             " N=" + std::to_string(N) +
             " handles=" + std::to_string(num_handles) + ")";
    };

    std::lock_guard<std::mutex> lock(invoke_mutex_);
    float *act_f32 = reinterpret_cast<float *>(
      stageInput(act_pool_, matBdata,
                 static_cast<size_t>(M) * K * sizeof(float), &out_pool_)
        .data());
    float *out_cat = reinterpret_cast<float *>(
      stage(out_pool_, static_cast<size_t>(out_len) * sizeof(float)).data());

    HtpProfile &profile = HtpProfile::global();
    if (norms && norms->any()) {
      // The norms ride the same call; the timed twin has no norm variant,
      // so this path counts as an untimed call in the profile.
      int post_gamma_len = 0;
      if (norms->post_chunk) {
        for (unsigned int c : *norms->post_chunk)
          post_gamma_len += static_cast<int>(c);
      }
      // With slices, each lands in its weight's row-major [M x N] on the
      // DSP (out_off / out_ld), so a weight comes out as one copy; the
      // slice-row copies this replaces were 132.8 ms of 4 KB memcpys a
      // 1024-token prefill (doc 57 section 9.34). A weight's slices come
      // in column order from c0 = 0 (gemm_q4_0_batch_norm_fp32).
      std::vector<uint32_t> out_off, out_ld;
      if (slices) {
        size_t base = 0, next = 0;
        for (const OutSlice &o : *slices) {
          if (o.c0 == 0) {
            base = next;
            next += static_cast<size_t>(M) * o.stride;
          }
          out_off.push_back(static_cast<uint32_t>(base + o.c0));
          out_ld.push_back(o.stride);
        }
      }
      const uint64_t t0 = HtpProfile::nowUs();
      const int pre_len = norms->pre_gamma ? static_cast<int>(K) : 0;
      const uint32_t *chunk =
        norms->post_chunk ? norms->post_chunk->data() : nullptr;
      const int chunk_len =
        norms->post_chunk ? static_cast<int>(norms->post_chunk->size()) : 0;
      const int cs_len =
        norms->rope_hd ? static_cast<int>(M * 2 * norms->rope_hd) : 0;
      const int err =
        slices ? nntr_hvx_mm_u8i4_layer_norm_ld(
                   session, M, K, norms->eps, norms->pre_gamma, pre_len, chunk,
                   chunk_len, norms->post_gamma, post_gamma_len, norms->rope_hd,
                   norms->rope_handles, norms->rope_cs, cs_len, handles,
                   num_handles, out_off.data(), num_handles, out_ld.data(),
                   num_handles, act_f32, act_len, out_cat, out_len)
               : nntr_hvx_mm_u8i4_layer_norm(
                   session, M, K, norms->eps, norms->pre_gamma, pre_len, chunk,
                   chunk_len, norms->post_gamma, post_gamma_len, norms->rope_hd,
                   norms->rope_handles, norms->rope_cs, cs_len, handles,
                   num_handles, act_f32, act_len, out_cat, out_len);
      const uint64_t elapsed = HtpProfile::nowUs() - t0;
      if (err != AEE_SUCCESS) {
        throw std::runtime_error("nntr_hvx_mm_u8i4_layer_norm failed: err=" +
                                 std::to_string(err) + shape());
      }
      if (slices) {
        for (size_t i = 0; i < slices->size(); ++i) { // whole tensors
          const OutSlice &o = (*slices)[i];
          if (o.c0 == 0) {
            const size_t b = static_cast<size_t>(M) * o.stride * sizeof(float);
            float *dst = o.dst;
            const float *src = out_cat + out_off[i];
            if (!norms->to_host)
              sampledCopy(dst, src, b);
            if (norms->to_host || !sampledSame(dst, src, b))
              stagedMemcpy(dst, src, b);
            noteIonAlias(dst, src, b);
          }
        }
      } else {
        copyOut(matCdata, out_cat, M, N, blocks, dsts);
      }
      if (profile.level())
        profile.addInvoke(M, K, N, elapsed, nullptr);
      return;
    }
    if (profile.level() == 0) {
      const int err =
        nntr_hvx_mm_u8i4_layer(session, M, K, handles, num_handles, act_f32,
                               act_len, out_cat, out_len);
      if (err != AEE_SUCCESS) {
        throw std::runtime_error("nntr_hvx_mm_u8i4_layer failed: err=" +
                                 std::to_string(err) + shape());
      }
      copyOut(matCdata, out_cat, M, N, blocks, dsts);
      return;
    }

    uint32_t stage_us[HTP_N_STAGES] = {0};
    const bool timed = profile.level() >= 2;
    const uint64_t t0 = HtpProfile::nowUs();
    const int err =
      timed ? nntr_hvx_mm_u8i4_layer_timed(session, M, K, handles, num_handles,
                                           act_f32, act_len, out_cat, out_len,
                                           stage_us, HTP_N_STAGES)
            : nntr_hvx_mm_u8i4_layer(session, M, K, handles, num_handles,
                                     act_f32, act_len, out_cat, out_len);
    const uint64_t elapsed = HtpProfile::nowUs() - t0;
    if (err != AEE_SUCCESS) {
      throw std::runtime_error(std::string(timed
                                             ? "nntr_hvx_mm_u8i4_layer_timed"
                                             : "nntr_hvx_mm_u8i4_layer") +
                               " failed: err=" + std::to_string(err) + shape());
    }
    copyOut(matCdata, out_cat, M, N, blocks, dsts);
    profile.addInvoke(M, K, N, elapsed, timed ? stage_us : nullptr);
  }

  /** @brief Same call as invokeLayer, but the activation is quantized and
   *  AH-tile-packed on the ARM side first (htp_act_quant.h) and sent as u8
   *  instead of f32 -- see mm_u8i4_layer_u8in's IDL doc for what this buys.
   *  Used by both accel entries, which the M > 1 gate means only ever run
   *  at prefill shapes (accelerates_q4_0_at_m1() is false), matching this
   *  path's scope: 41_moe_ffn_e2e_and_perf_task.md's P2. */
  void invokeLayerU8In(remote_handle64 session, const uint32_t *handles,
                       int num_handles, const float *matBdata, float *matCdata,
                       unsigned int M, unsigned int N, unsigned int K) {
    const uint32_t m_pad = htp_act_m_pad(M);
    const size_t act_ah_bytes = static_cast<size_t>(m_pad) * K;
    const int out_len = static_cast<int>(M) * static_cast<int>(N);

    std::lock_guard<std::mutex> lock(invoke_mutex_);
    ensureCapacity(act_ah_buf_, act_ah_bytes);
    uint8_t *act_ah = act_ah_buf_->data();
    float *out_cat = reinterpret_cast<float *>(
      stage(out_pool_, static_cast<size_t>(out_len) * sizeof(float)).data());

    if (act_scale_scratch_.size() < m_pad) {
      act_scale_scratch_.resize(m_pad);
      act_zp_scratch_.resize(m_pad);
    }
    htp_quant_pack_u8_ah(matBdata, M, K, act_ah, act_scale_scratch_.data(),
                         act_zp_scratch_.data());

    HtpProfile &profile = HtpProfile::global();
    if (profile.level() == 0) {
      const int err = nntr_hvx_mm_u8i4_layer_u8in(
        session, M, K, handles, num_handles, act_ah,
        static_cast<int>(act_ah_bytes), act_scale_scratch_.data(),
        static_cast<int>(m_pad), act_zp_scratch_.data(),
        static_cast<int>(m_pad), out_cat, out_len);
      if (err != AEE_SUCCESS) {
        throw std::runtime_error("nntr_hvx_mm_u8i4_layer_u8in failed: err=" +
                                 std::to_string(err));
      }
      stagedMemcpy(matCdata, out_cat,
                   static_cast<size_t>(out_len) * sizeof(float));
      return;
    }

    uint32_t stage_us[HTP_N_STAGES] = {0};
    const bool timed = profile.level() >= 2;
    const uint64_t t0 = HtpProfile::nowUs();
    const int err =
      timed
        ? nntr_hvx_mm_u8i4_layer_u8in_timed(
            session, M, K, handles, num_handles, act_ah,
            static_cast<int>(act_ah_bytes), act_scale_scratch_.data(),
            static_cast<int>(m_pad), act_zp_scratch_.data(),
            static_cast<int>(m_pad), out_cat, out_len, stage_us, HTP_N_STAGES)
        : nntr_hvx_mm_u8i4_layer_u8in(
            session, M, K, handles, num_handles, act_ah,
            static_cast<int>(act_ah_bytes), act_scale_scratch_.data(),
            static_cast<int>(m_pad), act_zp_scratch_.data(),
            static_cast<int>(m_pad), out_cat, out_len);
    const uint64_t elapsed = HtpProfile::nowUs() - t0;
    if (err != AEE_SUCCESS) {
      throw std::runtime_error(
        std::string(timed ? "nntr_hvx_mm_u8i4_layer_u8in_timed"
                          : "nntr_hvx_mm_u8i4_layer_u8in") +
        " failed: err=" + std::to_string(err));
    }
    stagedMemcpy(matCdata, out_cat,
                 static_cast<size_t>(out_len) * sizeof(float));
    profile.addInvoke(M, K, N, elapsed, timed ? stage_us : nullptr);
  }

  /** @brief The fused MoE expert FFN call: gate_up -> SwiGLU -> down in one
   *  FastRPC round trip (doc 43 §[L2]). Same scratch-buffer reuse and
   *  timed-entry split as invokeLayer; the SwiGLU stage gets its own
   *  profile bucket via addInvokeFused. */
  void invokeFused(remote_handle64 session, const uint32_t *handles,
                   float *matBdata, float *matCdata, unsigned int M,
                   unsigned int N, unsigned int K) {
    const int act_len = static_cast<int>(M) * static_cast<int>(K);
    const int out_len = static_cast<int>(M) * static_cast<int>(N);

    std::lock_guard<std::mutex> lock(invoke_mutex_);
    float *act_f32 = reinterpret_cast<float *>(
      stage(act_pool_, static_cast<size_t>(act_len) * sizeof(float)).data());
    float *out_f32 = reinterpret_cast<float *>(
      stage(out_pool_, static_cast<size_t>(out_len) * sizeof(float)).data());
    stagedMemcpy(act_f32, matBdata,
                 static_cast<size_t>(act_len) * sizeof(float));

    HtpProfile &profile = HtpProfile::global();
    if (profile.level() == 0) {
      const int err = nntr_hvx_mm_u8i4_layer_fused(
        session, M, K, handles, 2, act_f32, act_len, out_f32, out_len);
      if (err != AEE_SUCCESS) {
        throw std::runtime_error("nntr_hvx_mm_u8i4_layer_fused failed: err=" +
                                 std::to_string(err));
      }
      stagedMemcpy(matCdata, out_f32,
                   static_cast<size_t>(out_len) * sizeof(float));
      return;
    }

    uint32_t stage_us[HTP_FU_N_STAGES] = {0};
    const bool timed = profile.level() >= 2;
    const uint64_t t0 = HtpProfile::nowUs();
    const int err =
      timed
        ? nntr_hvx_mm_u8i4_layer_fused_timed(session, M, K, handles, 2, act_f32,
                                             act_len, out_f32, out_len,
                                             stage_us, HTP_FU_N_STAGES)
        : nntr_hvx_mm_u8i4_layer_fused(session, M, K, handles, 2, act_f32,
                                       act_len, out_f32, out_len);
    const uint64_t elapsed = HtpProfile::nowUs() - t0;
    if (err != AEE_SUCCESS) {
      throw std::runtime_error(
        std::string(timed ? "nntr_hvx_mm_u8i4_layer_fused_timed"
                          : "nntr_hvx_mm_u8i4_layer_fused") +
        " failed: err=" + std::to_string(err));
    }
    stagedMemcpy(matCdata, out_f32,
                 static_cast<size_t>(out_len) * sizeof(float));
    profile.addInvokeFused(M, K, N, elapsed, timed ? stage_us : nullptr);
  }

  /** @brief [#84] NNTR_HTP_DUMP=<dir>: every MoE call's input (M x K f32)
   *  and output (M x N_out f32) as <dir>/moe_<call>_in.f32 / _out.f32, in
   *  call order, plus one line per call in <dir>/manifest.txt
   *  (name entry M K inter N_out kind r=<row_count per expert>, the
   *  routing the ARM handed the call -- [#136] so the comparator can
   *  tell a top-k flip at a near-tie from a fault). Read once, off unless the
   *  variable is set: with it unset this is one static load and a branch
   *  per call. tools/htp/htp_dump_eval.py compares two such directories;
   *  the first file that differs names the call, which is where a _det is
   *  missing (plan 84 section 3.1). Never in a tok/s run: 16 KiB of file
   *  I/O per call. Callers hold invoke_mutex_, so the counter is plain. */
  static void dumpMoeCall(const char *entry, const float *act, size_t act_n,
                          const float *out, size_t out_n, unsigned int M,
                          unsigned int K, unsigned int inter,
                          unsigned int N_out, int kind,
                          const std::vector<unsigned int> &row_count) {
    static const char *dir = std::getenv("NNTR_HTP_DUMP");
    if (dir == nullptr)
      return;
    static unsigned int call = 0;
    char name[32];
    std::snprintf(name, sizeof(name), "moe_%05u", call);
    const std::string base = std::string(dir) + "/" + name;
    const auto write = [&](const std::string &path, const float *p, size_t n) {
      FILE *f = std::fopen(path.c_str(), "wb");
      if (f == nullptr || std::fwrite(p, sizeof(float), n, f) != n)
        throw std::runtime_error("NNTR_HTP_DUMP: cannot write " + path);
      std::fclose(f);
    };
    write(base + "_in.f32", act, act_n);
    write(base + "_out.f32", out, out_n);
    // Truncated by this process's first call: a reused directory would
    // otherwise list a previous run's calls, and its stale files with them.
    FILE *m = std::fopen((std::string(dir) + "/manifest.txt").c_str(),
                         call == 0 ? "w" : "a");
    if (m == nullptr)
      throw std::runtime_error("NNTR_HTP_DUMP: cannot write the manifest");
    ++call;
    std::fprintf(m, "%s %s %u %u %u %u %d r=", name, entry, M, K, inter, N_out,
                 kind);
    for (size_t e = 0; e < row_count.size(); ++e)
      std::fprintf(m, "%s%u", e ? "," : "", row_count[e]);
    std::fprintf(m, "\n");
    std::fclose(m);
  }

  /** @brief [#136] NNTR_HTP_DUMP_ALL=1 (next to NNTR_HTP_DUMP=<dir>): what
   *  the ARM side hands the DSP between and inside the per-token
   *  stretches, all of it -- every stretch's input / output as
   *  <dir>/fwd_<call>_<kinds>_{in,out}.f32 with one line per call in
   *  <dir>/forward_manifest.txt (name op resume pos K N_out kinds), the KV
   *  seed rows as kvseed_<ordinal>_{k,v}.f32 and the conv state seed as
   *  convstate_<op>.f32. Its own manifest, so the MoE goldens
   *  (manifest.txt) stay byte-equal. The device handoff of #136 turns it
   *  on for one run; the workstation replays the stretch inputs through
   *  graph_host_check's emulation and names the first stretch whose
   *  device output leaves it (plan 136 section 3.2). Never in a tok/s run. */
  static const char *dumpAllDir() {
    static const char *dir = std::getenv("NNTR_HTP_DUMP_ALL") != nullptr
                               ? std::getenv("NNTR_HTP_DUMP")
                               : nullptr;
    return dir;
  }
  static void dumpAllFile(const std::string &path, const float *p, size_t n) {
    FILE *f = std::fopen(path.c_str(), "wb");
    if (f == nullptr || std::fwrite(p, sizeof(float), n, f) != n)
      throw std::runtime_error("NNTR_HTP_DUMP_ALL: cannot write " + path);
    std::fclose(f);
  }
  void dumpForwardCall(uint32_t op, uint32_t resume, uint32_t pos,
                       const float *act, size_t K, const float *out,
                       size_t N_out) {
    const char *dir = dumpAllDir();
    if (dir == nullptr)
      return;
    static unsigned int call = 0;
    std::string kinds;
    for (uint32_t i = op; i < resume; ++i)
      kinds +=
        std::string(i == op ? "" : "-") + htp_graph_kind_name(graphOp(i)->kind);
    char name[64];
    std::snprintf(name, sizeof(name), "fwd_%05u_%s", call, kinds.c_str());
    const std::string base = std::string(dir) + "/" + name;
    dumpAllFile(base + "_in.f32", act, K);
    dumpAllFile(base + "_out.f32", out, N_out);
    FILE *m = std::fopen((std::string(dir) + "/forward_manifest.txt").c_str(),
                         call == 0 ? "w" : "a");
    if (m == nullptr)
      throw std::runtime_error(
        "NNTR_HTP_DUMP_ALL: cannot write the forward manifest");
    ++call;
    std::fprintf(m, "%s %u %u %u %zu %zu %s\n", name, op, resume, pos, K, N_out,
                 kinds.c_str());
    std::fclose(m);
  }

  /** @brief [#85] The per-token entry over one resident stretch [op,
   *  expected_resume): the same staging pools as invokeMoeLayer, and for
   *  the MoE stretch (@a moe) the same profile bucket, so the M==1 row
   *  reads the same across the two paths. The DSP must stop exactly at
   *  expected_resume (the ARM's stretch table, plan 130 section 3.2);
   *  anything else throws. */
  void invokeForward(remote_handle64 session, uint32_t op,
                     uint32_t expected_resume, uint32_t pos,
                     const std::vector<unsigned int> &row_index,
                     const std::vector<unsigned int> &row_count,
                     const std::vector<float> &row_weight, const float *act,
                     unsigned int K, float *out, unsigned int N_out, bool moe) {
    if (e2e_) {
      // [#132 Part B E3] with every kind resident the token is the one
      // stretch [0, n_ops), run by the token driver
      if (op != 0u || expected_resume != stretch_start_.size() ||
          act == nullptr || out == nullptr)
        throw std::runtime_error(
          "token driver: stretch [" + std::to_string(op) + ", " +
          std::to_string(expected_resume) + ") is not the whole token");
      tokenForward(pos, act, K, out, N_out);
      return;
    }
    const int act_len = static_cast<int>(K);
    const int out_len = static_cast<int>(N_out);
    const uint32_t n_limit = expected_resume - op;
    if (act == nullptr || out == nullptr || expected_resume <= op) {
      throw std::runtime_error("nntr_hvx_forward: bad stretch [" +
                               std::to_string(op) + ", " +
                               std::to_string(expected_resume) + ")");
    }

    std::lock_guard<std::mutex> lock(invoke_mutex_);
    HtpRpcBuffer &act_stage =
      stage(act_pool_, static_cast<size_t>(act_len) * sizeof(float));
    HtpRpcBuffer &out_stage =
      stage(out_pool_, static_cast<size_t>(out_len) * sizeof(float));
    float *act_f32 = reinterpret_cast<float *>(act_stage.data());
    float *out_f32 = reinterpret_cast<float *>(out_stage.data());
    stagedMemcpy(act_f32, act, static_cast<size_t>(act_len) * sizeof(float));

    HtpProfile &profile = HtpProfile::global();
    uint32_t stage_us[HTP_MOE_N_STAGES] = {0};
    std::vector<uint32_t> op_pcyc(n_limit, 0u);
    const bool timed = profile.level() >= 2;
    // Level 3's 5x repeat is for the stateless MoE stretch only: a
    // CONV1D_GATE op advances the DSP conv state on every call, an
    // ATTN_M1 op appends to the cache (a rewind, harmless, but not free).
    const int reps = (profile.level() >= 3 && moe) ? 5 : 1;
    uint64_t best_elapsed = UINT64_MAX;
    uint32_t resume_at = 0;
    int err = AEE_SUCCESS;
    for (int rep = 0; rep < reps && err == AEE_SUCCESS; ++rep) {
      uint32_t rep_stage[HTP_MOE_N_STAGES] = {0};
      std::vector<uint32_t> rep_pcyc(n_limit, 0u);
      uint32_t rep_resume = 0;
      const uint64_t t0 = profile.level() ? HtpProfile::nowUs() : 0;
      err = timed ? nntr_hvx_forward_debug(
                      session, op, n_limit, pos, row_index.data(),
                      static_cast<int>(row_index.size()), row_count.data(),
                      static_cast<int>(row_count.size()), row_weight.data(),
                      static_cast<int>(row_weight.size()), act_f32, act_len,
                      out_f32, out_len, &rep_resume, rep_pcyc.data(),
                      static_cast<int>(n_limit), rep_stage, HTP_MOE_N_STAGES)
                  : nntr_hvx_forward(
                      session, op, pos, row_index.data(),
                      static_cast<int>(row_index.size()), row_count.data(),
                      static_cast<int>(row_count.size()), row_weight.data(),
                      static_cast<int>(row_weight.size()), act_f32, act_len,
                      out_f32, out_len, &rep_resume);
      const uint64_t elapsed = profile.level() ? HtpProfile::nowUs() - t0 : 0;
      if (err == AEE_SUCCESS && elapsed < best_elapsed) {
        best_elapsed = elapsed;
        std::memcpy(stage_us, rep_stage, sizeof(stage_us));
        op_pcyc.swap(rep_pcyc);
        resume_at = rep_resume;
      }
    }
    const uint64_t elapsed = (best_elapsed == UINT64_MAX) ? 0 : best_elapsed;
    if (err != AEE_SUCCESS) {
      throw std::runtime_error(
        std::string(timed ? "nntr_hvx_forward_debug" : "nntr_hvx_forward") +
        " failed at op " + std::to_string(op) + " pos " + std::to_string(pos) +
        ": " + graphErr(err));
    }
    if (resume_at != expected_resume) {
      throw std::runtime_error("nntr_hvx_forward: resume_at " +
                               std::to_string(resume_at) + " from op " +
                               std::to_string(op) + ", the stretch ends at " +
                               std::to_string(expected_resume));
    }
    stagedMemcpy(out, out_f32, static_cast<size_t>(out_len) * sizeof(float));
    ++fwd_calls_;
    if (op == first_resident_op_)
      ++fwd_tokens_;
    if (moe)
      dumpMoeCall("forward", act, static_cast<size_t>(act_len), out,
                  static_cast<size_t>(out_len), 1u, K, 0u, N_out, 0, row_count);
    dumpForwardCall(op, expected_resume, pos, act, K, out, N_out);
    if (profile.level()) {
      uint64_t pcyc_sum = 0;
      for (uint32_t i = 0; i < n_limit; ++i) {
        pcyc_sum += op_pcyc[i];
        if (timed)
          profile.addGraphOp(graphOp(op + i)->kind, op_pcyc[i]);
      }
      if (moe) {
        // The prim block (start_op, pos, six lengths) and the three routing
        // sequences; the 64 handles no longer travel (plan 83 section 2).
        const size_t in_arg_bytes =
          40 + sizeof(uint32_t) *
                 (row_index.size() + row_count.size() + row_weight.size());
        profile.addInvokeMoeLayer(1, K, N_out, elapsed,
                                  timed ? stage_us : nullptr, act_stage,
                                  out_stage, in_arg_bytes);
      }
      profile.addInvokeForward(
        resume_at - op, timed ? stage_us[HTP_MOE_T_DSP_TOTAL] : 0, pcyc_sum);
    }
  }

  /** @brief [#141] The M==1 MoE call's dspqueue (plan 141-dspq-moe.md
   *  section 3.4): the queue, its two ION buffers mapped once, and the
   *  counters. Shared with the HtpBackend close hook, which tears it down
   *  before the session closes: the two singletons' destruction order is
   *  not fixed, so the hook must own what it touches. */
  struct DspqMoe {
    enum State { UNTRIED, ON, OFF } state = UNTRIED;
    bool broken = false; /**< a transport failure after ON: every later
                              call throws, never FastRPC (section 3.5) */
    const HtpDspqApi *api = nullptr;
    dspqueue_t q = nullptr;
    remote_handle64 session = 0;
    int domain = CDSP_DOMAIN_ID; /**< [#132 Part B E3] the effective one */
    const char *tag = "dspq";    /**< log prefix */
    std::unique_ptr<HtpRpcBuffer> act, out;
    bool act_mapped = false, out_mapped = false;
    uint32_t seq = 0;
    uint64_t calls = 0;
    uint32_t arm_spin_us = 0;
    std::atomic<int> cb_err{0}; /**< the queue's error callback */
    std::vector<uint32_t> msg;  /**< the request message, reused */
  };

  static constexpr uint32_t kDspqTimeoutUs = 5000000; /**< 5 s, never a hang */

  static void dspqErrorCb(dspqueue_t, AEEResult err, void *ctx) {
    static_cast<DspqMoe *>(ctx)->cb_err.store(err == 0 ? -1 : err);
  }

  /** @brief Undoes whatever creation got through, in reverse. */
  static uint32_t dspqRelease(DspqMoe &st) {
    const HtpRpcMemApi &mem = HtpRpcMemApi::get();
    uint32_t fail = 0; // [#132 Part B] fastrpc_munmap refusals
    if (st.act_mapped)
      fail += mem.munmap(st.domain, st.act->fd(), st.act->data(),
                         st.act->size()) != 0;
    if (st.out_mapped)
      fail += mem.munmap(st.domain, st.out->fd(), st.out->data(),
                         st.out->size()) != 0;
    st.act_mapped = st.out_mapped = false;
    if (st.q != nullptr)
      st.api->close(st.q);
    st.q = nullptr;
    return fail;
  }

  /** @brief The close hook: QUIT, stop the DSP thread, close, unmap. */
  static void dspqTeardown(DspqMoe &st) {
    if (dspqStop(st))
      dspqRelease(st);
  }

  /** @brief QUIT and the DSP thread joined, nothing unmapped yet (the
   *  E2E teardown unmaps in its own order). @return whether the
   *  queue was on. */
  static bool dspqStop(DspqMoe &st) {
    if (st.state != DspqMoe::ON)
      return false;
    st.state = DspqMoe::OFF;
    const uint32_t quit[2] = {HTP_DSPQ_OP_QUIT, 0};
    st.api->write(st.q, 0, 0, nullptr, sizeof(quit),
                  reinterpret_cast<const uint8_t *>(quit), kDspqTimeoutUs);
    uint32_t res[4] = {0, 0, 0, 0};
    const int err = nntr_hvx_dspq_stop(st.session, res, 4);
    std::fprintf(stderr,
                 "[HTP] %s: close calls=%llu served=%u bad=%u "
                 "empty_polls=%u dsp_spin_us=%u stop_err=0x%x\n",
                 st.tag, (unsigned long long)st.calls, res[0], res[1], res[2],
                 res[3], static_cast<unsigned>(err));
    return true;
  }

  /** @brief Creation, once, at the first M==1 MoE call (section 3.4) or
   *  at the E2E graph init; any failure prints one "dspq: off" line and is
   *  never retried. [#211] With NNTR_HTP_E2E=1 the token packets ride this
   *  queue whichever call creates it (a one-row prefill chunk comes first):
   *  its out buffer holds the logits and it spins the E2E way. */
  void dspqCreate(remote_handle64 session) {
    const char *spin_env = std::getenv("NNTR_HTP_DSPQ_SPIN_US");
    const size_t logits_bytes =
      e2e_ ? static_cast<size_t>(graph_words_[5]) * 4u : 0u;
    dspq_ = dspqMake(
      session, CDSP_DOMAIN_ID, "dspq",
      std::max<size_t>(HTP_DSPQ_BUF_BYTES, logits_bytes),
      e2e_       ? e2eSpinUs()
      : spin_env ? static_cast<uint32_t>(std::strtoul(spin_env, nullptr, 10))
                 : 1000u);
  }

  /** [#132 Part B E3] NNTR_HTP_E2E_SPIN_US (default 0): how long the DSP
   *  spins (with a pause) before it sleeps -- at each miss wait and on the
   *  dspqueue after a token. On two PDs (E5b) 1000 us kept one of the six
   *  hardware threads busy through the other session's compute; E5d read
   *  20.8 / 20.2 / 16.5 tok/s at 0 / 20 / 1000. */
  static uint32_t e2eSpinUs() {
    static const uint32_t us = [] {
      const char *e = std::getenv("NNTR_HTP_E2E_SPIN_US");
      return e ? static_cast<uint32_t>(std::strtoul(e, nullptr, 10)) : 0u;
    }();
    return us;
  }

  /** @brief A queue on @a session in @a domain with an out buffer of
   *  @a out_bytes (HTP_DSPQ_BUF_BYTES; [#132 Part B E3] the E2E one holds
   *  the logits too). Its teardown is an HtpBackend close hook. */
  std::shared_ptr<DspqMoe> dspqMake(remote_handle64 session, int domain,
                                    const char *tag, size_t out_bytes,
                                    uint32_t dsp_spin_us) {
    auto st = std::make_shared<DspqMoe>();
    st->state = DspqMoe::OFF;
    st->session = session;
    st->domain = domain;
    st->tag = tag;
    char why[96] = "";
    int err = 0;
    auto fail = [&](const char *step, int e) {
      std::snprintf(why, sizeof(why), "%s 0x%x", step,
                    static_cast<unsigned>(e));
      err = e != 0 ? e : -1;
    };
    const HtpDspqApi &api = HtpDspqApi::get();
    st->api = &api;
    const HtpRpcMemApi &mem = HtpRpcMemApi::get();
    if (api.missing != nullptr) {
      std::snprintf(why, sizeof(why), "dlsym %s", api.missing);
      err = -1;
    }
    if (err == 0) {
      const int e = api.create(domain, 0, HTP_DSPQ_REQ_QUEUE_BYTES,
                               HTP_DSPQ_RESP_QUEUE_BYTES, nullptr, dspqErrorCb,
                               st.get(), &st->q);
      if (e != AEE_SUCCESS) {
        st->q = nullptr;
        fail("create", e);
      }
    }
    uint64_t id = 0;
    if (err == 0) {
      const int e = api.export_(st->q, &id);
      if (e != AEE_SUCCESS)
        fail("export", e);
    }
    if (err == 0) {
      st->act = std::make_unique<HtpRpcBuffer>(HTP_DSPQ_BUF_BYTES);
      st->out = std::make_unique<HtpRpcBuffer>(out_bytes);
      if (!st->act->isIon() || !st->out->isIon() || st->act->fd() < 0 ||
          st->out->fd() < 0 || mem.mmap == nullptr) {
        fail("ion", 0);
      }
    }
    if (err == 0) {
      int e = mem.mmap(domain, st->act->fd(), st->act->data(), 0,
                       st->act->size(), FASTRPC_MAP_FD);
      st->act_mapped = (e == 0);
      if (e == 0) {
        e = mem.mmap(domain, st->out->fd(), st->out->data(), 0, st->out->size(),
                     FASTRPC_MAP_FD);
        st->out_mapped = (e == 0);
      }
      if (e != 0)
        fail("fastrpc_mmap", e);
    }
    if (err == 0) {
      const int e = nntr_hvx_dspq_start(session, id, dsp_spin_us);
      if (e != AEE_SUCCESS)
        fail(static_cast<unsigned>(e) == 0x8000040Eu
               ? "dspq_start (stale skel: rebuild test/htp/build.sh)"
               : "dspq_start",
             e);
    }
    if (err != 0) {
      dspqRelease(*st);
      std::fprintf(stderr, "[HTP] %s: off (%s) -- MoE calls stay on FastRPC\n",
                   tag, why);
      return st;
    }
    st->state = DspqMoe::ON;
    st->arm_spin_us = HtpBackend::global().pollUs();
    if (out_bytes == HTP_DSPQ_BUF_BYTES)
      std::fprintf(stderr,
                   "[HTP] %s: on queue=0x%llx dsp_spin_us=%u arm_spin_us=%u "
                   "buffers=2x%u ion=y\n",
                   tag, (unsigned long long)id, dsp_spin_us, st->arm_spin_us,
                   static_cast<unsigned>(HTP_DSPQ_BUF_BYTES));
    else
      std::fprintf(stderr,
                   "[HTP] %s: on queue=0x%llx dsp_spin_us=%u arm_spin_us=%u "
                   "buffers=%u+%zu ion=y domain=%d\n",
                   tag, (unsigned long long)id, dsp_spin_us, st->arm_spin_us,
                   static_cast<unsigned>(HTP_DSPQ_BUF_BYTES), out_bytes,
                   domain);
    HtpBackend::global().atClose([st] { dspqTeardown(*st); });
    return st;
  }

  /** @brief Whether this call can ride the queue; creates it on first use.
   *  Caller holds invoke_mutex_. */
  bool dspqReady(remote_handle64 session, size_t act_bytes, size_t out_bytes,
                 uint64_t msg_bytes) {
    // On by default since the #141 sitting (user, 2026-09-28): bit-identical
    // MoE dumps, decode +7.1 / +5.0 %. NNTR_HTP_DSPQ=0 keeps FastRPC.
    static const bool enabled = [] {
      const char *e = std::getenv("NNTR_HTP_DSPQ");
      return e == nullptr || std::atoi(e) != 0;
    }();
    if (!enabled)
      return false;
    if (!dspq_)
      dspqCreate(session);
    // The DSP may still hold the failed packet and its buffers: neither a
    // retry on the queue nor a FastRPC fallback is safe.
    if (dspq_->broken)
      throw std::runtime_error("dspq: the queue failed on an earlier call");
    // ponytail: a call larger than the packet or the 64 KiB buffers takes
    // FastRPC; every MoE layer of this model has one shape that fits.
    return dspq_->state == DspqMoe::ON && dspq_->session == session &&
           act_bytes <= HTP_DSPQ_BUF_BYTES && out_bytes <= HTP_DSPQ_BUF_BYTES &&
           msg_bytes <= HTP_DSPQ_MAX_MSG;
  }

  /** @brief One request/response pair (section 3.1); the activation is
   *  already in the queue's act buffer and the output lands in its out
   *  buffer. Transport failures throw: the DSP may still hold the packet,
   *  so a FastRPC retry could run on the same buffers at once. Caller
   *  holds invoke_mutex_.
   *  @return the kernel's rc */
  int dspqCall(unsigned M, unsigned K, unsigned inter, unsigned N_out,
               const std::vector<uint32_t> &h_gu,
               const std::vector<uint32_t> &h_dn,
               const std::vector<unsigned int> &row_index,
               const std::vector<unsigned int> &row_count,
               const std::vector<float> &row_weight, size_t act_bytes,
               size_t out_bytes, uint32_t glu, uint32_t *stage_us) {
    static_assert(HTP_MOE_N_STAGES == HTP_DSPQ_STAGES,
                  "htp_dspq_wire.h's stage count is the timed call's");
    DspqMoe &st = *dspq_;
    const uint32_t ne = static_cast<uint32_t>(h_gu.size());
    const uint32_t nr = static_cast<uint32_t>(row_index.size());
    const uint32_t seq = ++st.seq;
    const htp_dspq_req_hdr hdr = {HTP_DSPQ_OP_MOE,
                                  seq,
                                  (stage_us ? HTP_DSPQ_FLAG_TIMED : 0u) |
                                    (glu ? HTP_DSPQ_FLAG_GELU : 0u),
                                  M,
                                  K,
                                  inter,
                                  N_out,
                                  ne,
                                  nr};
    const size_t words = htp_dspq_req_bytes(ne, nr) / 4;
    st.msg.resize(words);
    uint32_t *w = st.msg.data();
    std::memcpy(w, &hdr, sizeof(hdr));
    w += sizeof(hdr) / 4;
    std::memcpy(w, h_gu.data(), 4u * ne);
    std::memcpy(w + ne, h_dn.data(), 4u * ne);
    std::memcpy(w + 2 * ne, row_count.data(), 4u * ne);
    std::memcpy(w + 3 * ne, row_index.data(), 4u * nr);
    std::memcpy(w + 3 * ne + nr, row_weight.data(), 4u * nr);

    struct dspqueue_buffer bufs[2] = {};
    bufs[0].fd = static_cast<uint32_t>(st.act->fd());
    bufs[0].size = static_cast<uint32_t>(act_bytes);
    bufs[0].flags = DSPQUEUE_BUFFER_FLAG_REF |
                    DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |
                    DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
    bufs[0].ptr = st.act->data();
    bufs[1].fd = static_cast<uint32_t>(st.out->fd());
    bufs[1].size = static_cast<uint32_t>(out_bytes);
    bufs[1].flags = DSPQUEUE_BUFFER_FLAG_REF;
    bufs[1].ptr = st.out->data();

    const HtpDspqApi &api = *st.api;
    int err = api.write(st.q, 0, 2, bufs, static_cast<uint32_t>(words * 4),
                        reinterpret_cast<const uint8_t *>(st.msg.data()),
                        kDspqTimeoutUs);
    htp_dspq_resp resp;
    uint32_t flags = 0, rnb = 0, len = 0;
    struct dspqueue_buffer rbufs[2] = {};
    if (err == AEE_SUCCESS) {
      // Spin for the poll-QoS window, as the FastRPC call does, then block.
      const uint64_t t0 = HtpProfile::nowUs();
      for (uint32_t spins = 1;; ++spins) {
        err = api.read_noblock(st.q, &flags, 2, &rnb, rbufs, sizeof(resp), &len,
                               reinterpret_cast<uint8_t *>(&resp));
        if (err != AEE_EWOULDBLOCK)
          break;
        if ((spins & 255u) == 0 && HtpProfile::nowUs() - t0 >= st.arm_spin_us) {
          err = api.read(st.q, &flags, 2, &rnb, rbufs, sizeof(resp), &len,
                         reinterpret_cast<uint8_t *>(&resp), kDspqTimeoutUs);
          break;
        }
      }
    }
    const int cb = st.cb_err.load();
    if (err != AEE_SUCCESS || cb != 0 || len < HTP_DSPQ_RESP_BASE_BYTES ||
        resp.seq != seq || rnb != 2) {
      st.broken = true;
      char msg[160];
      std::snprintf(msg, sizeof(msg),
                    "dspq: MoE call %u failed: err=0x%x cb_err=0x%x len=%u "
                    "seq=%u nb=%u",
                    seq, static_cast<unsigned>(err), static_cast<unsigned>(cb),
                    len, len >= 4 ? resp.seq : 0u, rnb);
      throw std::runtime_error(msg);
    }
    ++st.calls;
    if (stage_us != nullptr && resp.rc == AEE_SUCCESS) {
      if (len != HTP_DSPQ_RESP_BASE_BYTES + 4u * HTP_DSPQ_STAGES)
        throw std::runtime_error("dspq: timed response without stage slots");
      std::memcpy(stage_us, resp.stage_us, sizeof(resp.stage_us));
    }
    return resp.rc;
  }

  struct E2eState; // [#132 Part B E3] defined with the members below

  /** @brief [#132 Part B E3, #211] The one-PD decode's ARM side: the token
   *  driver on S1 over one mailbox page (the expert pool's miss lines), the
   *  token packets on S1's dspqueue. Created at graph init; the HtpBackend
   *  close hook e2eTeardown undoes it. */
  void e2eStart() {
    E2eState &e = *e2e_st_;
    const HtpRpcMemApi &mem = HtpRpcMemApi::get();
    const size_t logits_bytes = static_cast<size_t>(graph_words_[5]) * 4u;
    if (!dspq_)
      dspqCreate(e.h1);
    if (dspq_->state != DspqMoe::ON || dspq_->session != e.h1) {
      throw std::runtime_error("NNTR_HTP_E2E=1: S1's dspqueue is off (see "
                               "the dspq line; NNTR_HTP_DSPQ=0?): the token "
                               "packets ride it");
    }
    e.q1 = dspq_;
    e.mbox = std::make_unique<HtpRpcBuffer>(kMboxBytes, HTP_RPC_FLAGS_UNCACHED);
    if (!e.mbox->isIon() || e.mbox->fd() < 0 || mem.mmap == nullptr) {
      throw std::runtime_error("NNTR_HTP_E2E=1: no ION page for the mailbox");
    }
    // zeroed before the DSP maps it, as the #178 probe's page: no sequence
    // word a first read could take for an answer
    std::memset(e.mbox->data(), 0, e.mbox->size());
    const int fd = e.mbox->fd();
    int rc = mem.mmap(CDSP_DOMAIN_ID, fd, e.mbox->data(), 0, e.mbox->size(),
                      FASTRPC_MAP_FD);
    e.mbox1 = rc == 0;
    if (rc != 0) {
      throw std::runtime_error("NNTR_HTP_E2E=1: fastrpc_mmap of the mailbox "
                               "page failed: " +
                               std::to_string(rc));
    }
    const uint32_t spin_us = e2eSpinUs();
    // role 0, the only one since #211 (the IDL keeps the argument)
    rc = nntr_hvx_token_driver_start(
      e.h1, fd, static_cast<uint32_t>(e.mbox->size()), 0u, spin_us);
    e.drv1 = rc == AEE_SUCCESS;
    if (rc != AEE_SUCCESS) {
      throw std::runtime_error("nntr_hvx_token_driver_start failed: " +
                               graphErr(rc));
    }
    e.moe_ops = static_cast<uint32_t>(moe_ops_.size());
    e.spin_us = spin_us;
    e.pgpgin0 = vmstatPgpginKib();
    std::fprintf(stderr,
                 "[HTP] token driver: on mbox=%zu spin_us=%u moe_ops=%u "
                 "logits_buf=%zu fadvise=%d\n",
                 e.mbox->size(), spin_us, e.moe_ops, logits_bytes,
                 fadviseKnob());
  }

  /** @brief [#132 Part B E3, #211] The close hook, before the session
   *  closes: the queue, the driver, the mailbox, then the graph, the FC
   *  slots and their arena chunks, each buffer fastrpc_munmap'd before
   *  nntr_hvx_close (#178's rule). The MoE arena stays as on the hybrid
   *  path. */
  static void e2eTeardown(E2eState &e) {
    static_assert(HTP_OP_KIND_N == HTP_DSPQ_TOKEN_KINDS,
                  "htp_dspq_wire.h's kind count is the graph's");
    static_assert(HTP_DSPQ_ATTN_PROF_WORDS == ATTN_M1_PROF_WORDS,
                  "htp_dspq_wire.h's phase words are attn_m1_det.h's");
    const HtpRpcMemApi &mem = HtpRpcMemApi::get();
    // [#132 Part B E5g] every DSP thread stopped before any munmap: the
    // queue (QUIT, joined), then the token driver (its HAP_mmap_put); then
    // the mappings. The close line counts what was mapped, what
    // fastrpc_munmap / arena_detach refused, and the DSP heap (E5f: one E
    // run in ~12 left the next process 256 MiB short of S1's 3840).
    if (e.q1)
      dspqStop(*e.q1);
    uint32_t r1[5] = {0, 0, 0, 0, 0};
    const int s1 = e.drv1 ? nntr_hvx_token_driver_stop(e.h1, r1, 5) : 0;
    if (e.drv1) {
      const double n = e.tokens ? static_cast<double>(e.tokens) : 1.0;
      std::fprintf(stderr,
                   "[HTP] graph: tokens=%llu pcyc/token=%.0f "
                   "wait_us/token=%.1f\n",
                   (unsigned long long)e.tokens,
                   static_cast<double>(e.pcyc) / n,
                   static_cast<double>(e.wait_us) / n);
      // the clock (pcycles over the token's wall time) and the op pcycles
      // per kind per token, with the ms they are at that clock
      const double wus = static_cast<double>(e.wall_us);
      const double mhz = wus > 0 ? static_cast<double>(e.wall_pcyc) / wus : 0.0;
      std::string line;
      char buf[96];
      for (uint32_t kd = 0; kd < HTP_OP_KIND_N; ++kd) {
        if (e.kind[kd] == 0)
          continue;
        const double pc = static_cast<double>(e.kind[kd]) / n;
        std::snprintf(buf, sizeof(buf), " %s=%.0f(%.3fms)",
                      htp_graph_kind_name(kd), pc,
                      mhz > 0 ? pc / mhz / 1000.0 : 0.0);
        line += buf;
      }
      std::fprintf(stderr,
                   "[HTP] graph per-kind pcyc/token:%s | wall_ms/token=%.3f "
                   "mhz=%.0f spin_us=%u\n",
                   line.c_str(), wus / n / 1000.0, mhz, e.spin_us);
      // [#267 L0] the same kinds in wall us at whatever clock each ran, the
      // MOE net of its miss waits, and the clock outside the waits
      line.clear();
      uint64_t sum_us = 0;
      for (uint32_t kd = 0; kd < HTP_OP_KIND_N; ++kd) {
        sum_us += e.kind_us[kd];
        if (e.kind_us[kd] == 0)
          continue;
        std::snprintf(buf, sizeof(buf), " %s=%.0f", htp_graph_kind_name(kd),
                      static_cast<double>(e.kind_us[kd]) / n);
        line += buf;
        if (kd == HTP_OP_MOE) {
          std::snprintf(buf, sizeof(buf), "(net of miss wait %.0f)",
                        (static_cast<double>(e.kind_us[kd]) -
                         static_cast<double>(e.miss_us)) /
                          n);
          line += buf;
        }
      }
      const double cus = wus - static_cast<double>(e.miss_us);
      std::fprintf(
        stderr,
        "[HTP] graph per-kind us/token:%s | ops/wall=%.4f compute_mhz=%.0f\n",
        line.c_str(), wus > 0 ? static_cast<double>(sum_us) / wus : 0.0,
        cus > 0 ? (static_cast<double>(e.wall_pcyc) -
                   static_cast<double>(e.miss_pcyc)) /
                    cus
                : 0.0);
      // [#261] ATTN_M1's phase words per token and KV cache (attn_m1_det.h:
      // the lane-summed pieces, the pool's span and busiest lane in k
      // pcycles; call_us the calls' wall)
      for (uint32_t c = 0; c < 2u && e.attn_tokens != 0; ++c) {
        if (e.attn_calls[c] == 0)
          continue;
        const double t = static_cast<double>(e.attn_tokens);
        auto k = [&](uint32_t w) {
          return static_cast<double>(e.attn_prof[c][w]) / t / 1000.0;
        };
        std::fprintf(
          stderr,
          "[HTP] graph attn_m1 cache=%u calls/token=%.2f kpcyc/token: "
          "append=%.1f scores=%.1f softmax=%.1f (exp=%.1f et=%.1f max=%.1f "
          "sum=%.1f div=%.1f) pv=%.1f pool=%.1f busy_max=%.1f lanes=%.2f "
          "call_us/token=%.1f\n",
          c, static_cast<double>(e.attn_calls[c]) / t, k(ATTN_M1_PROF_APPEND),
          k(ATTN_M1_PROF_SCORES), k(ATTN_M1_PROF_SOFTMAX), k(ATTN_M1_PROF_EXP),
          k(ATTN_M1_PROF_ET), k(ATTN_M1_PROF_MAX), k(ATTN_M1_PROF_SUM),
          k(ATTN_M1_PROF_DIV), k(ATTN_M1_PROF_PV), k(ATTN_M1_PROF_POOL),
          k(ATTN_M1_PROF_BUSY_MAX),
          static_cast<double>(e.attn_prof[c][ATTN_M1_PROF_LANES]) /
            static_cast<double>(e.attn_calls[c]),
          static_cast<double>(e.attn_prof[c][ATTN_M1_PROF_CALL_QT]) / t / 19.2);
      }
      if (e.moe_ops != 0 && e.kind[HTP_OP_MOE] != 0) {
        const double moe = static_cast<double>(e.kind[HTP_OP_MOE]) / n /
                           static_cast<double>(e.moe_ops);
        std::fprintf(
          stderr,
          "[HTP] graph moe pcyc/op=%.0f (%.3f ms) router pcyc/op=%.0f; "
          "fc+dense_ffn+lm_head ms/token=%.3f; arm token_ms=%.3f\n",
          moe, mhz > 0 ? moe / mhz / 1000.0 : 0.0,
          static_cast<double>(e.kind[HTP_OP_ROUTER_TOPK]) / n /
            static_cast<double>(e.moe_ops),
          mhz > 0
            ? static_cast<double>(e.kind[HTP_OP_FC] + e.kind[HTP_OP_DENSE_FFN] +
                                  e.kind[HTP_OP_LM_HEAD]) /
                n / mhz / 1000.0
            : 0.0,
          static_cast<double>(e.token_us) / n / 1000.0);
      }
      // [#194 L0] where the token's time goes outside the kernels: the
      // ARM's round trip (rt) against the DSP's own wall (wake = the
      // dspqueue transport and the ARM read's wake-up), tokenForward on the
      // ARM (arm_fwd, rt plus its copies) and the ARM between two
      // tokenForward calls (arm_us: the layer walk, sampler, tokenizer,
      // print). Wake split: disp = the ARM's post -> the DSP thread's read,
      // pkt = the DSP's handling outside its token wall, ret = the DSP's
      // write -> the ARM's read; rt = disp + pkt + wall + ret when the two
      // clocks are one counter (else these read nonsense and clk_resid
      // says by how much)
      const double an = e.arm_n ? static_cast<double>(e.arm_n) : 1.0;
      std::fprintf(
        stderr,
        "[HTP] token driver: L0 wake us/token disp=%.1f pkt=%.1f ret=%.1f "
        "clk_resid=%.1f\n",
        static_cast<double>(e.disp_us) / n,
        (static_cast<double>(e.inout_us) - static_cast<double>(e.wall_us)) / n,
        static_cast<double>(e.ret_us) / n,
        (static_cast<double>(e.token_us) -
         static_cast<double>(e.disp_us + e.inout_us + e.ret_us)) /
          n);
      std::fprintf(
        stderr,
        "[HTP] token driver: L0 us/token rt=%.1f dsp_wall=%.1f wake=%.1f "
        "hop_us=%.1f arm_fwd=%.1f arm_us=%.1f arm_n=%llu\n",
        static_cast<double>(e.token_us) / n, static_cast<double>(e.wall_us) / n,
        (static_cast<double>(e.token_us) - static_cast<double>(e.wall_us)) / n,
        static_cast<double>(e.hop_us) / n, static_cast<double>(e.fwd_us) / n,
        static_cast<double>(e.arm_us) / an, (unsigned long long)e.arm_n);
      std::fprintf(stderr,
                   "[HTP] token driver: pool finish us/token idle=%.1f "
                   "predictor=%.1f prefetch=%.1f\n",
                   static_cast<double>(e.pool_idle_us) / n,
                   static_cast<double>(e.pool_predict_drain_us) / n,
                   static_cast<double>(e.pool_prefetch_drain_us) / n);
      if (e.pool_rounds != 0 || e.misses != 0)
        std::fprintf(
          stderr,
          "[HTP] token driver: pool misses=%llu misses/token=%.2f "
          "miss_wait_us/token=%.1f rounds=%llu arm_ms/round=%.3f "
          "pgpgin_mib=%.1f rss_mib=%.1f mem_available_mib=%.1f\n",
          (unsigned long long)e.misses, static_cast<double>(e.misses) / n,
          static_cast<double>(e.miss_us) / n, (unsigned long long)e.pool_rounds,
          e.pool_rounds
            ? static_cast<double>(e.pool_read_us) / 1000.0 / e.pool_rounds
            : 0.0,
          static_cast<double>(vmstatPgpginKib() - e.pgpgin0) / 1024.0, rssMib(),
          memAvailMib());
      if (e.pool_rounds != 0) {
        const double rr = static_cast<double>(e.pool_rounds);
        std::fprintf(stderr,
                     "[HTP] token driver: round us: notice=%.0f pre=%.0f "
                     "read=%.0f answer=%.0f (arm total) | dsp overlap=%.0f "
                     "wait=%.0f anslat=%.0f rebind=%.0f\n",
                     e.rt_notice_us / rr, e.rt_pre_us / rr, e.rt_read_us / rr,
                     static_cast<double>(e.pool_read_us) / rr,
                     e.rt_overlap_us / rr, static_cast<double>(e.miss_us) / rr,
                     e.rt_anslat_us / rr, e.rt_rebind_us / rr);
      }
      if (e.pool_pair_reads)
        std::fprintf(stderr, "[HTP] token driver: expert_pair_reads=%llu\n",
                     (unsigned long long)e.pool_pair_reads);
      if (pf_hints_ != 0 || pf_issued_ != 0) // the hint path ran
        std::fprintf(stderr,
                     "[HTP] token driver: prefetch hints/token=%.2f "
                     "issued/token=%.2f done=%llu useful/token=%.2f\n",
                     static_cast<double>(pf_hints_) / n,
                     static_cast<double>(pf_issued_) / n,
                     (unsigned long long)pf_done_.load(),
                     static_cast<double>(pf_useful_) / n);
      if (e.moe_calls != 0)
        std::fprintf(stderr,
                     "[HTP] token driver: moe calls/token=%.2f "
                     "(one-at-a-time %.2f)\n",
                     static_cast<double>(e.moe_calls) / n,
                     static_cast<double>(e.moe_calls_1x) / n);
      std::fprintf(
        stderr,
        "[HTP] token driver: close tokens=%llu hops/token=%.2f "
        "served=%u timeouts=%u stale=%u id_checked=%llu "
        "id_mismatch=%llu stop_err=0x%x\n",
        (unsigned long long)e.tokens, static_cast<double>(e.hops) / n, r1[0],
        r1[2], r1[3], (unsigned long long)e.id_checked,
        (unsigned long long)e.id_mismatch, static_cast<unsigned>(s1));
    }
    e.drv1 = false;
    uint32_t info1[7] = {0};
    const int i1 = e.h1 ? nntr_hvx_session_info(e.h1, info1, 7) : -1;
    size_t mapped = 0;
    uint32_t unmap_fail = 0, detach_fail = 0;
    if (e.mbox1) {
      unmap_fail += mem.munmap(CDSP_DOMAIN_ID, e.mbox->fd(), e.mbox->data(),
                               e.mbox->size()) != 0;
      mapped += e.mbox->size();
      e.mbox1 = false;
    }
    if (e.q1 && e.q1->q != nullptr) {
      mapped += (e.q1->act_mapped ? e.q1->act->size() : 0u) +
                (e.q1->out_mapped ? e.q1->out->size() : 0u);
      unmap_fail += dspqRelease(*e.q1);
    }
    // [plan 201 S1] the graph names the FC slots below, so it goes first
    // (the graph's other close finds none left); slots exist once the FC
    // set was placed, which every graph init follows
    if (!e.q4m1.empty())
      nntr_hvx_graph_release(e.h1);
    for (uint32_t h : e.q4m1)
      nntr_hvx_q4m1_release(e.h1, h);
    e.q4m1.clear();
    for (auto c = e.arena.rbegin(); c != e.arena.rend(); ++c) {
      detach_fail += nntr_hvx_arena_detach(e.h1, c->dsp_id) != AEE_SUCCESS;
      if (mem.munmap != nullptr)
        unmap_fail += mem.munmap(CDSP_DOMAIN_ID, c->buf->fd(), c->buf->data(),
                                 c->buf->size()) != 0;
      mapped += c->buf->size();
    }
    e.arena.clear();
    std::fprintf(stderr,
                 "[HTP] e2e: close mapped_mib=%.2f unmap_fail=%u "
                 "detach_fail=%u heap_used_kib=%u (info rc 0x%x)\n",
                 static_cast<double>(mapped) / (1024.0 * 1024.0), unmap_fail,
                 detach_fail, info1[4], static_cast<unsigned>(i1));
  }

  /** @brief [#194 L0] The system counter in us (low 32 bits): the ARM's
   *  cntvct_el0, the counter the DSP's QTimer reads; off aarch64 (the
   *  in-process build, whose DSP clock is CLOCK_MONOTONIC) steady_clock. */
  static uint32_t sysCounterUs() {
#if defined(__aarch64__)
    uint64_t c, f;
    asm volatile("mrs %0, cntvct_el0" : "=r"(c));
    asm volatile("mrs %0, cntfrq_el0" : "=r"(f));
    return f ? static_cast<uint32_t>(static_cast<unsigned __int128>(c) *
                                     1000000u / f)
             : 0u;
#else
    return static_cast<uint32_t>(HtpProfile::nowUs());
#endif
  }

  /** @brief [#132 Part B E3, #211] One decode token on the one PD: the
   *  embedding row into the queue's act buffer, OP_TOKEN (op 0 to the
   *  argmax, the MOE ops' miss rounds on the page), the response. Only the
   *  id comes back unless set_decode_logits asked for the logits (the
   *  NNTR_PPL_DECODE / sampling / bad-words runs), which the DSP writes
   *  into the out buffer. Any failure throws; a transport failure marks
   *  the queue broken (the DSP may still hold a packet). */
  void tokenForward(uint32_t pos, const float *act, unsigned K, float *out,
                    unsigned N_out) {
    E2eState &e = *e2e_st_;
    const uint64_t entry_us = HtpProfile::nowUs();
    if (e.last_exit_us != 0) {
      e.arm_us += entry_us - e.last_exit_us;
      ++e.arm_n;
    }
    if (!e.drv1 || !dspq_)
      throw std::runtime_error("token driver: not started (its start threw "
                               "at graph init)");
    DspqMoe &q = *dspq_;
    if (q.broken)
      throw std::runtime_error("token driver: the queue failed on an earlier "
                               "token");
    const size_t act_bytes = static_cast<size_t>(K) * sizeof(float);
    const size_t out_bytes = static_cast<size_t>(N_out) * sizeof(float);
    if (act_bytes > q.act->size() || out_bytes > q.out->size())
      throw std::runtime_error("token driver: row " + std::to_string(K) +
                               " / logits " + std::to_string(N_out) +
                               " do not fit the queue's buffers");
    if (have_id_)
      throw std::runtime_error("token driver: the previous token's id was "
                               "never taken (take_decode_token_id)");
    // [#132 Part B E3] the bad-word ids the DSP's argmax skips (LM_BAN),
    // sent when they change; more than the DSP holds, or none after some
    // (an empty sequence may reach the skel as NULL, the stale-skel code)
    // -> the logits come back and the CPU picks
    const bool logits = want_logits_ || ban_.size() > HTP_GRAPH_MAX_BAN ||
                        (ban_.empty() && !ban_sent_.empty());
    if (!logits && ban_sent_ != ban_) {
      std::vector<float> words(ban_.size());
      std::memcpy(words.data(), ban_.data(), ban_.size() * sizeof(uint32_t));
      setParam(e.h1, kind_ops_[HTP_OP_LM_HEAD][0], HTP_GRAPH_PARAM_LM_BAN,
               words.data(), static_cast<unsigned>(ban_.size()),
               static_cast<unsigned>(ban_.size()), "LM_BAN");
      ban_sent_ = ban_;
    }
    if (!pool_descs_.empty())
      poolSync(); // [plan 201 S1]
    std::lock_guard<std::mutex> lock(invoke_mutex_);
    std::memcpy(q.act->data(), act, act_bytes);
    const uint32_t tok = e.tok++;
    if (!pool_descs_.empty())
      poolArm(tok);
    const uint32_t attn_prof =
      HtpProfile::global().level() ? HTP_DSPQ_TOKEN_ATTN_PROF : 0u; // [#261]
    const htp_dspq_token_req req = {
      HTP_DSPQ_OP_TOKEN, tok, (logits ? HTP_DSPQ_TOKEN_LOGITS : 0u) | attn_prof,
      pos};
    struct dspqueue_buffer b[2] = {};
    b[0].fd = static_cast<uint32_t>(q.act->fd());
    b[0].size = static_cast<uint32_t>(act_bytes);
    b[0].flags = DSPQUEUE_BUFFER_FLAG_REF | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |
                 DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
    b[0].ptr = q.act->data();
    b[1].fd = static_cast<uint32_t>(q.out->fd());
    b[1].size = static_cast<uint32_t>(out_bytes);
    b[1].flags = DSPQUEUE_BUFFER_FLAG_REF;
    b[1].ptr = q.out->data();
    const uint32_t nb = logits ? 2u : 1u;
    const uint32_t c0 = sysCounterUs();
    const uint64_t t0 = HtpProfile::nowUs();
    int err =
      q.api->write(q.q, 0, nb, b, sizeof(req),
                   reinterpret_cast<const uint8_t *>(&req), kDspqTimeoutUs);
    htp_dspq_token_resp r = {};
    uint32_t len = 0, rnb = 0;
    if (err == AEE_SUCCESS) {
      // a blocking read (the token is ~20 ms)
      uint32_t flags = 0;
      struct dspqueue_buffer rb[2] = {};
      err = q.api->read(q.q, &flags, 2, &rnb, rb, sizeof(r), &len,
                        reinterpret_cast<uint8_t *>(&r), kDspqTimeoutUs);
      r.route_n = std::min<uint32_t>(r.route_n, HTP_DSPQ_TOKEN_ROUTE);
      if (err == AEE_SUCCESS && len == sizeof(r))
        routeLog(r);
    }
    const uint64_t us = HtpProfile::nowUs() - t0;
    const uint32_t c2 = sysCounterUs();
    if (!pool_descs_.empty()) {
      // the DSP answered (or the token failed): no round in flight. A
      // failed round leaves the ARM's pool and the DSP's tables apart, so
      // the driver stops here for good, as after a transport failure.
      try {
        poolDisarm();
      } catch (...) {
        q.broken = true;
        throw;
      }
      if (r.rc != AEE_SUCCESS)
        q.broken = true;
    }
    const int cb = q.cb_err.load();
    if (err != AEE_SUCCESS || cb != 0 || len != sizeof(r) || r.seq != tok ||
        rnb != nb) {
      q.broken = true;
      char msg[200];
      std::snprintf(msg, sizeof(msg),
                    "token driver: token %u pos %u transport failed: "
                    "err=0x%x cb_err=0x%x len=%u seq=%u nb=%u",
                    tok, pos, static_cast<unsigned>(err),
                    static_cast<unsigned>(cb), len, r.seq, rnb);
      throw std::runtime_error(msg);
    }
    ++q.calls;
    if (r.rc != AEE_SUCCESS) {
      throw std::runtime_error("token driver: token " + std::to_string(tok) +
                               " pos " + std::to_string(pos) +
                               " failed: " + graphErr(r.rc) +
                               " (AEE_EEXPIRED: a miss round timed out)");
    }
    if (logits) {
      std::memcpy(out, q.out->data(), out_bytes);
      // the DSP's argmax against the logits it returned, the ids LM_BAN
      // held at the time masked as the DSP masked them
      std::vector<float> keep(ban_sent_.size());
      for (size_t i = 0; i < ban_sent_.size(); ++i) {
        keep[i] = out[ban_sent_[i]];
        out[ban_sent_[i]] = -INFINITY;
      }
      const uint32_t pick = m1_argmax_first(out, N_out);
      for (size_t i = ban_sent_.size(); i-- > 0;)
        out[ban_sent_[i]] = keep[i];
      ++e.id_checked;
      e.id_mismatch += pick != r.id;
    } else {
      pending_id_ = r.id;
      have_id_ = true;
    }
    ++e.tokens;
    e.misses += r.misses;
    e.miss_us += r.miss_us;
    e.rt_overlap_us += r.miss_overlap_us; // [#282 C]
    e.rt_anslat_us += r.miss_anslat_us;
    e.rt_rebind_us += r.miss_rebind_us;
    if (!pool_descs_.empty())
      poolRefresh(r);
    e.hops += r.hops;
    e.wait_us += r.wait_us;
    e.pcyc += r.pcycles;
    e.wall_us += r.wall_us;
    e.wall_pcyc += r.wall_pcyc;
    for (uint32_t k = 0; k < HTP_OP_KIND_N; ++k) {
      e.kind[k] += r.kind_pcyc[k];
      e.kind_us[k] += r.kind_us[k];
    }
    e.miss_pcyc += r.miss_pcyc;
    if (attn_prof) {
      ++e.attn_tokens;
      for (uint32_t c = 0; c < 2u; ++c) {
        e.attn_calls[c] += r.attn_calls[c];
        for (uint32_t w = 0; w < HTP_DSPQ_ATTN_PROF_WORDS; ++w)
          e.attn_prof[c][w] += r.attn_prof[c][w];
      }
    }
    e.moe_calls += r.moe_calls;
    e.moe_calls_1x += r.moe_calls_1x;
    e.token_us += us;
    e.hop_us += r.hop_us;
    e.disp_us += static_cast<int32_t>(r.t_in_us - c0);
    e.ret_us += static_cast<int32_t>(c2 - r.t_out_us);
    e.inout_us += static_cast<int32_t>(r.t_out_us - r.t_in_us);
    ++fwd_calls_; // one ARM -> DSP packet per token: calls/token = 1.00
    ++fwd_tokens_;
    if (++prompt_tokens_ == 1) {
      releasePrefillStaging(); // [#4416 mem] the seed was the last big row
    }
    if (e.tokens == 1)
      growPool();
    if (e.tokens == 1)
      std::fprintf(stderr,
                   "[HTP] token driver: first token pos=%u id=%u us=%llu "
                   "logits=%d\n",
                   pos, r.id, (unsigned long long)us, logits);
    HtpProfile &profile = HtpProfile::global();
    if (profile.level())
      profile.addInvokeForward(static_cast<unsigned>(stretch_start_.size()), 0,
                               r.pcycles);
    e.last_exit_us = HtpProfile::nowUs();
    e.fwd_us += e.last_exit_us - entry_us;
  }

  /** [#132 Part B E3] compute_ops.h: whether the next decode tokens must
   *  bring their logits back (else only the id travels). */
  void set_decode_logits(bool want) override { want_logits_ = want; }

  /** [#132 Part B E3] compute_ops.h: the bad-word ids the caller's pick
   *  sets to -inf; the DSP's argmax skips them (LM_BAN) on the id-only
   *  path. */
  void set_decode_ban(const unsigned *ids, unsigned n) override {
    ban_.assign(ids, ids + n);
  }

  /** [#132 Part B E3] compute_ops.h: the id the DSP's argmax picked for the
   *  last decode token, once, when its logits did not come back. */
  bool take_decode_token_id(unsigned *id) override {
    if (!have_id_)
      return false;
    *id = pending_id_;
    have_id_ = false;
    return true;
  }

  /** @brief A completed id-only row has no CPU logits to postprocess.
   *  Peeking leaves the token for take_decode_token_id. */
  bool has_decode_token_id(unsigned pos) const override {
    return have_id_ && pos == cur_pos_ && pending_op_ == HTP_GRAPH_NO_OP;
  }

  /** [#132 Part B E3] compute_ops.h: the model handed its last Q4_0
   *  weight. On the E2E path the FC set goes into its arena chunks now,
   *  at load (the startup cell), not at the first decode token. */
  bool finish_decode_graph_q4_0() override {
    // [#225] Called once after the load's registrations: where the FC
    // prefill weights went. heap_kib != 0 is the overflow plan 225 section
    // 3.3 c expects on the full-residency hybrid; requant != 0 is a model
    // without a sidecar.
    if (std::lock_guard<std::mutex> lock(handle_mutex_);
        fcwh_fd_ >= 0 || g_q4_requant.load() != 0) {
      std::fprintf(stderr,
                   "[HTP] fc wh: file=%s handles=%zu arena_kib=%zu "
                   "heap_kib=%zu requant=%u\n",
                   fcwh_fd_ >= 0 ? fcwh_name_.c_str() : "none", fcwhHandles(),
                   fcwh_arena_bytes_ >> 10, fcwh_heap_bytes_ >> 10,
                   g_q4_requant.load());
    }
    std::lock_guard<std::mutex> lock(graph_mutex_);
    if (!e2e_ || q4_pending_.empty() || q4m1_bound_ || graph_inited_)
      return false;
    e2ePlaceFc();
    return true;
  }

  /** [#234 P4] compute_ops.h: the E2E list's Q4_0 weights are still to
   *  bind, so the sidecar can serve its FC / DENSE_FFN ops (bindQ4m1). */
  bool has_decode_graph_q4_0() override {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    return e2e_ && !q4_pending_.empty() && !q4m1_bound_;
  }

  /** [#132 Part B E3, #211] Places the Q4M1 set on S1's own new arena
   *  chunks: at load after the MoE arena (the model calls
   *  finish_decode_graph_q4_0 after repack_weight), else at graph init.
   *  Caller holds graph_mutex_. The banner's s1_arena_mib is the MoE arena
   *  mapped before it. */
  void e2ePlaceFc() {
    E2eState &e = *e2e_st_;
    const size_t s1_mib = arenaBytes() >> 20;
    uint32_t info1[7] = {0}; // res[4]: S1's heap in use, KiB
    nntr_hvx_session_info(e.h1, info1, 7);
    const uint64_t t0 = HtpProfile::nowUs();
    try {
      bindQ4m1(e.h1);
    } catch (...) {
      releaseQ4m1(e.h1);
      throw;
    }
    q4m1_bound_ = true;
    const double ms = static_cast<double>(HtpProfile::nowUs() - t0) / 1000.0;
    size_t mapped = 0;
    for (const ArenaChunk &c : e.arena)
      mapped += c.buf->size();
    std::fprintf(stderr,
                 "[HTP] e2e: fc arena weights=%zu handles=%zu attach_mib=%.1f "
                 "chunks=%zu mapped_mib=%zu feed=%s load_ms=%.1f "
                 "lanes=%s s1_arena_mib=%zu s1_heap_kib=%u\n",
                 q4_pending_.size(), e.q4m1.size(),
                 static_cast<double>(e.attach_bytes) / (1024.0 * 1024.0),
                 e.arena.size(), mapped >> 20, q4m1FeedName(), ms,
                 std::getenv("NNTR_HTP_FC_LANES")
                   ? std::getenv("NNTR_HTP_FC_LANES")
                   : "6,3",
                 s1_mib, info1[4]);
  }

  /** [#132 Part B E3] The feed the FC kinds take: the op's l2, else VTCM. */
  const char *q4m1FeedName() const {
    return !kind_ops_[HTP_OP_FC].empty() &&
               (graphOp(kind_ops_[HTP_OP_FC][0])->feed & HTP_GRAPH_FEED_L2) !=
                 0u
             ? "l2"
             : "vtcm";
  }

  /** @brief [doc 46] One call for the whole layer.
   *
   *  The activation goes over once instead of once per expert, and the
   *  output comes back once: with 32 experts at top-4 the old path shipped
   *  the same token rows four times each in both directions. The routing
   *  table rides along as three small arrays -- 1776 uint32 + 1776 float +
   *  32 uint32 for this model, about 14 KB against the 3.6 MB activation.
   */
  void invokeMoeLayer(remote_handle64 session,
                      const std::vector<uint32_t> &h_gu,
                      const std::vector<uint32_t> &h_dn,
                      const std::vector<unsigned int> &row_index,
                      const std::vector<unsigned int> &row_count,
                      const std::vector<float> &row_weight, const float *act,
                      float *out, unsigned int M, unsigned int K,
                      unsigned int inter, unsigned int N_out, int kind = 0,
                      uint32_t glu = 0, const float *pre_gamma = nullptr,
                      const float *post_gamma = nullptr, float eps = 0.0f,
                      const float *add_x2 = nullptr,
                      const float *add_gamma = nullptr, float add_scale = 1.0f,
                      bool out_to_host = true) {
    const int act_len = static_cast<int>(M) * static_cast<int>(K);
    const int out_len = static_cast<int>(M) * static_cast<int>(N_out);
    // [PR 4428] act bit 1: the dense FFN's chunks were never rotated, so
    // the session's online rotation (NNTR_MOE_HADAMARD_SCOPE=all) skips
    // this call; the experts' calls keep it.
    if (kind == 1)
      glu |= 2u;
    const bool with_norms = pre_gamma != nullptr || post_gamma != nullptr;

    const size_t act_bytes = static_cast<size_t>(act_len) * sizeof(float);
    const size_t out_bytes = static_cast<size_t>(out_len) * sizeof(float);
    const uint64_t msg_bytes =
      htp_dspq_req_bytes(static_cast<uint32_t>(h_gu.size()),
                         static_cast<uint32_t>(row_index.size()));

    std::lock_guard<std::mutex> lock(invoke_mutex_);
    // [#141] The M==1 MoE call rides the dspqueue (default since the #141
    // sitting; NNTR_HTP_DSPQ=0 keeps FastRPC): the same DSP function, the same
    // argument bytes, only the transport differs. Anything the packet cannot
    // carry stays on FastRPC.
    const bool via_dspq = M == 1 && kind == 0 && !with_norms &&
                          h_dn.size() == h_gu.size() &&
                          row_count.size() == h_gu.size() &&
                          row_weight.size() == row_index.size() &&
                          dspqReady(session, act_bytes, out_bytes, msg_bytes);
    // NNTR_HTP_HOST_NORMS=1: the MoE call's norms and epilogue on the host
    // around the plain entry, as NNTR_HTP_PROFILE=2 does without its timing;
    // NNTR_HTP_MOE_HOST=<mask> picks them one at a time (1 pre-norm, 2
    // post-norm, 4 epilogue; the DSP's fused epilogue needs its pre-norm, so
    // a host pre-norm takes the epilogue with it). References for the fused
    // entry; never set in a tok/s run. Both need NNTR_HTP_KEEP_OUT=0: the
    // host reads act whole.
    static const unsigned host_mask = [] {
      const char *v = std::getenv("NNTR_HTP_HOST_NORMS");
      if (v != nullptr && std::strcmp(v, "1") == 0)
        return 7u;
      const char *m = std::getenv("NNTR_HTP_MOE_HOST");
      return m != nullptr ? static_cast<unsigned>(std::atoi(m)) : 0u;
    }();
    const unsigned host_bits =
      HtpProfile::global().level() >= 2 ? 7u : host_mask;
    const bool host_prenorm =
      (host_bits & 1u) && with_norms && !via_dspq && pre_gamma != nullptr;
    const bool host_post_norm =
      (host_bits & 2u) && with_norms && !via_dspq && post_gamma != nullptr;
    const bool host_epilogue =
      add_x2 != nullptr && !via_dspq &&
      ((host_bits & 4u) || host_prenorm || host_post_norm);
    // from out_pool_: the post-attention rows the res_add call left there
    // (the dense call), or the dense output the MoE call takes as act
    HtpRpcBuffer &act_stage =
      via_dspq       ? *dspq_->act
      : host_prenorm ? stage(act_pool_, act_bytes)
                     : stageInput(act_pool_, act, act_bytes, &out_pool_);
    // The post-FFN epilogue's second addend rides its own staging buffer,
    // staged before out_stage: it may be swapped in from out_pool_.
    float *x2_f32 = nullptr;
    if (add_x2 != nullptr) {
      if (pre_gamma == nullptr || K != N_out || via_dspq)
        throw std::invalid_argument(
          "gemm_qs4cx_moe_layer_fp32: the epilogue needs the raw rows "
          "(pre_gamma) and K == N_out");
      x2_f32 = reinterpret_cast<float *>(
        stageInput(x2_pool_, add_x2, out_bytes, &out_pool_).data());
    }
    HtpRpcBuffer &out_stage =
      via_dspq ? *dspq_->out : stage(out_pool_, out_bytes);
    float *act_f32 = reinterpret_cast<float *>(act_stage.data());
    float *out_f32 = reinterpret_cast<float *>(out_stage.data());
    // NNTR_HTP_MOE_DEBUG=1 (with NNTR_HTP_KEEP_OUT=0 so the host tensors
    // are whole): the staged act / x2 against the host's bytes
    static const bool moe_dbg = std::getenv("NNTR_HTP_MOE_DEBUG") != nullptr;
    if (moe_dbg && !via_dspq) {
      static unsigned call_no = 0;
      ++call_no;
      const bool a_ok =
        host_prenorm || std::memcmp(act_f32, act, act_bytes) == 0;
      const bool x_ok =
        x2_f32 == nullptr || std::memcmp(x2_f32, add_x2, out_bytes) == 0;
      if (!a_ok || !x_ok)
        ml_logw("HTP moe staging #%u M=%u: act %s x2 %s", call_no, M,
                a_ok ? "ok" : "DIFFERS", x_ok ? "ok" : "DIFFERS");
    }
    // NNTR_HTP_PROFILE>=2 on a call with folded norms: the timed entry
    // (stage breakdown) and the norms on the host around it, so the prefill
    // MoE and dense calls, which always carry them, have stages too (doc 57
    // section 9.26), the post-FFN epilogue too (section 9.34). Measurement
    // only; the norms' and the epilogue's time is outside the call.
    const float *pre_gamma_dsp = host_prenorm ? nullptr : pre_gamma;
    const float *post_gamma_dsp = host_post_norm ? nullptr : post_gamma;
    if (host_prenorm) {
      nntrainer::rms_norm_wrt_width_fp32_intrinsic(act, act_f32, M, K, eps);
      for (unsigned int r = 0; r < M; ++r)
        for (unsigned int k = 0; k < K; ++k)
          act_f32[static_cast<size_t>(r) * K + k] *= pre_gamma[k];
    } else if (via_dspq) {
      stagedMemcpy(act_f32, act, act_bytes);
    }

    // The five small sequences (handles, routing) go from the heap. Tried
    // from one rpcmem buffer (doc 51 section 2.26): transport 627 -> 605
    // us, noise. The gap to the FC calls' ~400 is the call's length, not
    // its arguments -- a call longer than the poll window (htp_backend.cpp,
    // 5 ms) ends in the interrupt-driven wait and pays its wake-up.

    HtpProfile &profile = HtpProfile::global();
    uint32_t stage_us[HTP_MOE_N_STAGES] = {0};
    const bool timed = profile.level() >= 2;
    const bool norm_entry =
      pre_gamma_dsp != nullptr || post_gamma_dsp != nullptr;
    // NNTR_HTP_PROFILE=3 runs the call several times on the same input and
    // keeps the fastest. Two runs with no functional change between them
    // differed by 4.5 ms of DSP time (doc 46 section 23.2) -- the
    // compute-bound stage identical, the DDR-touching ones all moving
    // together, so the device's memory system, not the code. Several of the
    // changes on the list are worth 2 ms, which that noise floor swallows.
    // Repeating inside one call puts the comparison at the same temperature
    // and the same contention, and the output is unchanged because the
    // input is.
    const int reps = (profile.level() >= 3) ? 5 : 1;
    uint64_t best_elapsed = UINT64_MAX;
    int err = AEE_SUCCESS;
    for (int rep = 0; rep < reps && err == AEE_SUCCESS; ++rep) {
      uint32_t rep_stage[HTP_MOE_N_STAGES] = {0};
      const uint64_t t0 = profile.level() ? HtpProfile::nowUs() : 0;
      err = via_dspq ? dspqCall(M, K, inter, N_out, h_gu, h_dn, row_index,
                                row_count, row_weight, act_bytes, out_bytes,
                                glu, timed ? rep_stage : nullptr)
            : add_x2 != nullptr && !host_epilogue
              ? nntr_hvx_mm_u8i4_moe_layer_norm_add(
                  session, M, K, inter, N_out, glu, eps, add_scale, pre_gamma,
                  static_cast<int>(K), post_gamma,
                  post_gamma ? static_cast<int>(N_out) : 0, add_gamma,
                  add_gamma ? static_cast<int>(N_out) : 0, h_gu.data(),
                  static_cast<int>(h_gu.size()), h_dn.data(),
                  static_cast<int>(h_dn.size()), row_index.data(),
                  static_cast<int>(row_index.size()), row_count.data(),
                  static_cast<int>(row_count.size()), row_weight.data(),
                  static_cast<int>(row_weight.size()), act_f32, act_len, x2_f32,
                  out_len, out_f32, out_len)
            : norm_entry
              ? nntr_hvx_mm_u8i4_moe_layer_norm(
                  session, M, K, inter, N_out, glu, eps, pre_gamma_dsp,
                  pre_gamma_dsp ? static_cast<int>(K) : 0, post_gamma_dsp,
                  post_gamma_dsp ? static_cast<int>(N_out) : 0, h_gu.data(),
                  static_cast<int>(h_gu.size()), h_dn.data(),
                  static_cast<int>(h_dn.size()), row_index.data(),
                  static_cast<int>(row_index.size()), row_count.data(),
                  static_cast<int>(row_count.size()), row_weight.data(),
                  static_cast<int>(row_weight.size()), act_f32, act_len,
                  out_f32, out_len)
            : timed ? nntr_hvx_mm_u8i4_moe_layer_timed(
                        session, M, K, inter, N_out, glu, h_gu.data(),
                        static_cast<int>(h_gu.size()), h_dn.data(),
                        static_cast<int>(h_dn.size()), row_index.data(),
                        static_cast<int>(row_index.size()), row_count.data(),
                        static_cast<int>(row_count.size()), row_weight.data(),
                        static_cast<int>(row_weight.size()), act_f32, act_len,
                        out_f32, out_len, rep_stage, HTP_MOE_N_STAGES)
                    : nntr_hvx_mm_u8i4_moe_layer(
                        session, M, K, inter, N_out, glu, h_gu.data(),
                        static_cast<int>(h_gu.size()), h_dn.data(),
                        static_cast<int>(h_dn.size()), row_index.data(),
                        static_cast<int>(row_index.size()), row_count.data(),
                        static_cast<int>(row_count.size()), row_weight.data(),
                        static_cast<int>(row_weight.size()), act_f32, act_len,
                        out_f32, out_len);
      const uint64_t elapsed = profile.level() ? HtpProfile::nowUs() - t0 : 0;
      // Fastest wins, stages and all, so the breakdown describes one real call
      // rather than a mix of a fast one and a slow one.
      if (err == AEE_SUCCESS && elapsed < best_elapsed) {
        best_elapsed = elapsed;
        std::memcpy(stage_us, rep_stage, sizeof(stage_us));
      }
    }
    const uint64_t elapsed = (best_elapsed == UINT64_MAX) ? 0 : best_elapsed;
    if (err != AEE_SUCCESS) {
      // 0x8000040E is AEE_EBADPARM, and every length this call passes is
      // derived from the same shapes the kernel checks against -- so by far
      // the likeliest cause is that libnntr_hvx_skel.so on the device is
      // older than this binary and the two disagree on how many stage_us
      // slots there are. build_android.sh does not rebuild the skel; only
      // test/htp/build.sh does, and forgetting that has cost three
      // measurement cycles.
      std::string hint;
      if (static_cast<unsigned>(err) == 0x8000040Eu) {
        hint =
          " (AEE_EBADPARM -- if the shapes are right, rebuild the DSP skel: "
          "test/htp/build.sh, then push libnntr_hvx_skel.so. This binary "
          "expects " +
          std::to_string(static_cast<int>(HTP_MOE_N_STAGES)) +
          " stage_us slots.)";
      }
      throw std::runtime_error(
        std::string(timed ? "nntr_hvx_mm_u8i4_moe_layer_timed"
                          : "nntr_hvx_mm_u8i4_moe_layer") +
        " failed: err=" + std::to_string(err) + hint +
        (via_dspq ? " (via dspq)" : ""));
    }
    if (via_dspq)
      stagedMemcpy(out, out_f32, out_bytes);
    else
      copyOutOf(out_pool_, out, out_bytes, out_to_host);
    if (host_post_norm) {
      std::vector<float> raw(out, out + static_cast<size_t>(out_len));
      nntrainer::rms_norm_wrt_width_fp32_intrinsic(raw.data(), out, M, N_out,
                                                   eps);
      for (unsigned int r = 0; r < M; ++r)
        for (unsigned int n = 0; n < N_out; ++n)
          out[static_cast<size_t>(r) * N_out + n] *= post_gamma[n];
    }
    if (host_epilogue) {
      // out = scale * (act + rmsnorm(out + x2) * gamma), as the epilogue.
      for (unsigned int r = 0; r < M; ++r) {
        float *o = out + static_cast<size_t>(r) * N_out;
        const float *x2 = add_x2 + static_cast<size_t>(r) * N_out;
        const float *res = act + static_cast<size_t>(r) * K;
        float ss = 0.0f;
        for (unsigned int n = 0; n < N_out; ++n)
          ss += (o[n] + x2[n]) * (o[n] + x2[n]);
        const float rs = 1.0f / std::sqrt(ss / N_out + eps);
        for (unsigned int n = 0; n < N_out; ++n)
          o[n] = add_scale * (res[n] + (o[n] + x2[n]) * rs *
                                         (add_gamma ? add_gamma[n] : 1.0f));
      }
    }
    dumpMoeCall("moe_layer", act, static_cast<size_t>(act_len), out,
                static_cast<size_t>(out_len), M, K, inter, N_out, kind,
                row_count);
    if (profile.level()) {
      // [#88] The bytes the stub hands the driver outside ION: the 48-byte
      // primitive block (_primIn[12] in generated/nntr_hvx_stub.c) and the
      // five uint32/float sequences that are not the staged activation.
      const size_t in_arg_bytes =
        via_dspq ? static_cast<size_t>(msg_bytes)
                 : 48 + sizeof(uint32_t) *
                          (h_gu.size() + h_dn.size() + row_index.size() +
                           row_count.size() + row_weight.size());
      profile.addInvokeMoeLayer(M, K, N_out, elapsed,
                                timed ? stage_us : nullptr, act_stage,
                                out_stage, in_arg_bytes, kind, via_dspq);
    }
    if (timed) {
      // [#87] The per-descriptor trace of the last repeat, for the first
      // NNTR_HTP_DMA_TRACE calls of this bucket. Read now, while the skel's
      // static tables still hold this call; printed now, so the lines sit
      // next to the token they came from.
      const unsigned ordinal = profile.dmaTraceOrdinal(K, N_out, M, kind);
      if (ordinal != 0) {
        std::vector<uint32_t> words(HEXKL_DMA_TRACE_MAX_WORDS);
        uint32_t n_words = 0;
        const int terr = nntr_hvx_moe_dma_trace_read(
          session, words.data(), static_cast<int>(words.size()), &n_words);
        if (terr == AEE_SUCCESS) {
          HtpProfile::dumpDmaTrace(ordinal, M, reps, words.data(), n_words);
        } else {
          std::fprintf(stderr,
                       "[HTP-DMA] call=%u M=%u moe_dma_trace_read err=%d "
                       "(older skel?)\n",
                       ordinal, M, terr);
        }
      }
    }
  }

  /** @brief [doc 51 section 2] One call for a whole conv block.
   *
   *  The activation goes over once and the output comes back once; the
   *  four M x C intermediates the block has stay on the DSP. conv_w (24 KB)
   *  rides along by value each call rather than being registered: it is
   *  under a hundredth of the activation. The state (2 x C) comes back in
   *  its own small buffer, not a size class of out_pool_: at a short M the
   *  output would land in the same 64 KiB class and the two would alias. */
  void invokeConvBlock(remote_handle64 session, const ConvHandles &ch,
                       const float *conv_w, const float *act, float *out,
                       float *state, unsigned int M, unsigned int K,
                       unsigned int C, unsigned int N_out,
                       const float *hist = nullptr) {
    const int act_len = static_cast<int>(M) * static_cast<int>(K);
    const int out_len = static_cast<int>(M) * static_cast<int>(N_out);
    // [#225] A chunk after the first carries the conv's two history rows
    // after the taps, [5 x C] (hexkl_conv_block_run's hist): no IDL change
    const int conv_len = (hist != nullptr ? 5 : 3) * static_cast<int>(C);
    const int state_len = 2 * static_cast<int>(C);

    std::lock_guard<std::mutex> lock(invoke_mutex_);
    HtpRpcBuffer &act_stage =
      stage(act_pool_, static_cast<size_t>(act_len) * sizeof(float));
    HtpRpcBuffer &out_stage =
      stage(out_pool_, static_cast<size_t>(out_len) * sizeof(float));
    float *act_f32 = reinterpret_cast<float *>(act_stage.data());
    float *out_f32 = reinterpret_cast<float *>(out_stage.data());
    const size_t small_bytes = static_cast<size_t>(7) * C * sizeof(float);
    if (!conv_buf_ || conv_buf_->size() < small_bytes)
      conv_buf_ = std::make_unique<HtpRpcBuffer>(small_bytes);
    float *conv_f32 = reinterpret_cast<float *>(conv_buf_->data());
    float *state_f32 = conv_f32 + 5 * C;
    stagedMemcpy(act_f32, act, static_cast<size_t>(act_len) * sizeof(float));
    std::memcpy(conv_f32, conv_w, static_cast<size_t>(3) * C * sizeof(float));
    if (hist != nullptr)
      std::memcpy(conv_f32 + 3 * C, hist,
                  static_cast<size_t>(state_len) * sizeof(float));

    HtpProfile &profile = HtpProfile::global();
    uint32_t stage_us[HTP_MOE_N_STAGES] = {0};
    const bool timed = profile.level() >= 2;
    const uint64_t t0 = profile.level() ? HtpProfile::nowUs() : 0;
    const int err =
      timed ? nntr_hvx_mm_u8i4_conv_block_timed(
                session, M, K, C, N_out, ch.h_in.data(),
                static_cast<int>(ch.h_in.size()), ch.h_out, conv_f32, conv_len,
                act_f32, act_len, out_f32, out_len, state_f32, state_len,
                stage_us, HTP_MOE_N_STAGES)
            : nntr_hvx_mm_u8i4_conv_block(
                session, M, K, C, N_out, ch.h_in.data(),
                static_cast<int>(ch.h_in.size()), ch.h_out, conv_f32, conv_len,
                act_f32, act_len, out_f32, out_len, state_f32, state_len);
    const uint64_t elapsed = profile.level() ? HtpProfile::nowUs() - t0 : 0;
    if (err != AEE_SUCCESS) {
      std::string hint;
      if (static_cast<unsigned>(err) == 0x8000040Eu) {
        hint = std::string(" (AEE_EBADPARM -- if the shapes are right, the "
                           "DSP skel on the device predates ") +
               (hist != nullptr ? "the chunked conv block (conv_wLen 5 C, "
                                  "#225)"
                                : "mm_u8i4_conv_block") +
               ": rebuild it with test/htp/build.sh and push "
               "libnntr_hvx_skel.so)";
      }
      throw std::runtime_error(
        std::string(timed ? "nntr_hvx_mm_u8i4_conv_block_timed"
                          : "nntr_hvx_mm_u8i4_conv_block") +
        " failed: err=" + std::to_string(err) + " M=" + std::to_string(M) +
        " K=" + std::to_string(K) + " C=" + std::to_string(C) +
        " N=" + std::to_string(N_out) + hint);
    }
    stagedMemcpy(out, out_f32, static_cast<size_t>(out_len) * sizeof(float));
    std::memcpy(state, state_f32,
                static_cast<size_t>(state_len) * sizeof(float));
    if (profile.level()) {
      // [#88] Outside ION: the primitive block and the three handles;
      // conv_w and the state ride in conv_buf_ (ION).
      const size_t in_arg_bytes = 48 + sizeof(uint32_t) * ch.h_in.size();
      profile.addInvokeMoeLayer(M, K, N_out, elapsed,
                                timed ? stage_us : nullptr, act_stage,
                                out_stage, in_arg_bytes, /*kind=*/2);
    }
  }

  /** @brief [L2, split-call variant] gate_up matmul -> SwiGLU -> requantize
   *  to u8 AH, ONE weight -- doc 43 §7's smaller-surface alternative to
   *  invokeFused. Leaves its result in act_ah_buf_/act_scale_scratch_/
   *  act_zp_scratch_ for invokeLayerU8InRaw (below) to consume immediately
   *  after -- both run under gemm_qs4cx_fused_swiglu_fp32's call, so there
   *  is no other caller between them to race with. */
  void invokeGateUpSwiglu(remote_handle64 session, uint32_t handle_gate_up,
                          const float *matBdata, unsigned int M, unsigned int K,
                          unsigned int inter) {
    const int act_len = static_cast<int>(M) * static_cast<int>(K);
    const uint32_t m_pad = htp_act_m_pad(M);
    const size_t out_ah_bytes = static_cast<size_t>(m_pad) * inter;

    std::lock_guard<std::mutex> lock(invoke_mutex_);
    ensureCapacity(act_ah_buf_, out_ah_bytes);
    if (act_scale_scratch_.size() < m_pad) {
      act_scale_scratch_.resize(m_pad);
      act_zp_scratch_.resize(m_pad);
    }
    float *act_f32 = reinterpret_cast<float *>(
      stage(act_pool_, static_cast<size_t>(act_len) * sizeof(float)).data());
    uint8_t *out_ah = act_ah_buf_->data();
    stagedMemcpy(act_f32, matBdata,
                 static_cast<size_t>(act_len) * sizeof(float));

    HtpProfile &profile = HtpProfile::global();
    if (profile.level() == 0) {
      const int err = nntr_hvx_mm_u8i4_gate_up_swiglu(
        session, M, K, handle_gate_up, act_f32, act_len, out_ah,
        static_cast<int>(out_ah_bytes), act_scale_scratch_.data(),
        static_cast<int>(m_pad), act_zp_scratch_.data(),
        static_cast<int>(m_pad));
      if (err != AEE_SUCCESS) {
        throw std::runtime_error(
          "nntr_hvx_mm_u8i4_gate_up_swiglu failed: err=" + std::to_string(err));
      }
      if (l2CheckEnabled())
        l2CheckFinite("gate_up_swiglu out_scale", act_scale_scratch_.data(),
                      m_pad, M, K, 2 * inter);
      return;
    }

    uint32_t stage_us[HTP_GU_N_STAGES] = {0};
    const bool timed = profile.level() >= 2;
    const uint64_t t0 = HtpProfile::nowUs();
    const int err =
      timed ? nntr_hvx_mm_u8i4_gate_up_swiglu_timed(
                session, M, K, handle_gate_up, act_f32, act_len, out_ah,
                static_cast<int>(out_ah_bytes), act_scale_scratch_.data(),
                static_cast<int>(m_pad), act_zp_scratch_.data(),
                static_cast<int>(m_pad), stage_us, HTP_GU_N_STAGES)
            : nntr_hvx_mm_u8i4_gate_up_swiglu(
                session, M, K, handle_gate_up, act_f32, act_len, out_ah,
                static_cast<int>(out_ah_bytes), act_scale_scratch_.data(),
                static_cast<int>(m_pad), act_zp_scratch_.data(),
                static_cast<int>(m_pad));
    const uint64_t elapsed = HtpProfile::nowUs() - t0;
    if (err != AEE_SUCCESS) {
      throw std::runtime_error(
        std::string(timed ? "nntr_hvx_mm_u8i4_gate_up_swiglu_timed"
                          : "nntr_hvx_mm_u8i4_gate_up_swiglu") +
        " failed: err=" + std::to_string(err));
    }
    if (l2CheckEnabled())
      l2CheckFinite("gate_up_swiglu out_scale", act_scale_scratch_.data(),
                    m_pad, M, K, 2 * inter);
    profile.addInvokeGateUpSwiglu(M, K, 2 * inter, elapsed,
                                  timed ? stage_us : nullptr);
  }

  /** @brief Down matmul via the u8in path, fed act_ah/scale/zp that are
   *  ALREADY quantized (by invokeGateUpSwiglu, just above) -- unlike
   *  invokeLayerU8In, this does not call htp_quant_pack_u8_ah itself; there
   *  is nothing f32 left to quantize, the DSP produced the bytes directly. */
  void invokeLayerU8InRaw(remote_handle64 session, const uint32_t *handles,
                          int num_handles, unsigned int m_pad_in,
                          float *matCdata, unsigned int M, unsigned int N,
                          unsigned int K) {
    const size_t act_ah_bytes = static_cast<size_t>(m_pad_in) * K;
    const int out_len = static_cast<int>(M) * static_cast<int>(N);

    std::lock_guard<std::mutex> lock(invoke_mutex_);
    const uint8_t *act_ah = act_ah_buf_->data();
    float *out_cat = reinterpret_cast<float *>(
      stage(out_pool_, static_cast<size_t>(out_len) * sizeof(float)).data());

    HtpProfile &profile = HtpProfile::global();
    if (profile.level() == 0) {
      const int err = nntr_hvx_mm_u8i4_layer_u8in(
        session, M, K, handles, num_handles, act_ah,
        static_cast<int>(act_ah_bytes), act_scale_scratch_.data(),
        static_cast<int>(m_pad_in), act_zp_scratch_.data(),
        static_cast<int>(m_pad_in), out_cat, out_len);
      if (err != AEE_SUCCESS) {
        throw std::runtime_error("nntr_hvx_mm_u8i4_layer_u8in failed: err=" +
                                 std::to_string(err));
      }
      if (l2CheckEnabled())
        l2CheckFinite("down out", out_cat, static_cast<size_t>(out_len), M, K,
                      N);
      stagedMemcpy(matCdata, out_cat,
                   static_cast<size_t>(out_len) * sizeof(float));
      return;
    }

    uint32_t stage_us[HTP_N_STAGES] = {0};
    const bool timed = profile.level() >= 2;
    const uint64_t t0 = HtpProfile::nowUs();
    const int err =
      timed ? nntr_hvx_mm_u8i4_layer_u8in_timed(
                session, M, K, handles, num_handles, act_ah,
                static_cast<int>(act_ah_bytes), act_scale_scratch_.data(),
                static_cast<int>(m_pad_in), act_zp_scratch_.data(),
                static_cast<int>(m_pad_in), out_cat, out_len, stage_us,
                HTP_N_STAGES)
            : nntr_hvx_mm_u8i4_layer_u8in(
                session, M, K, handles, num_handles, act_ah,
                static_cast<int>(act_ah_bytes), act_scale_scratch_.data(),
                static_cast<int>(m_pad_in), act_zp_scratch_.data(),
                static_cast<int>(m_pad_in), out_cat, out_len);
    const uint64_t elapsed = HtpProfile::nowUs() - t0;
    if (err != AEE_SUCCESS) {
      throw std::runtime_error(
        std::string(timed ? "nntr_hvx_mm_u8i4_layer_u8in_timed"
                          : "nntr_hvx_mm_u8i4_layer_u8in") +
        " failed: err=" + std::to_string(err));
    }
    if (l2CheckEnabled())
      l2CheckFinite("down out", out_cat, static_cast<size_t>(out_len), M, K, N);
    stagedMemcpy(matCdata, out_cat,
                 static_cast<size_t>(out_len) * sizeof(float));
    profile.addInvoke(M, K, N, elapsed, timed ? stage_us : nullptr);
  }

  uint32_t get_or_register(void *matAdata, remote_handle64 session, uint32_t K,
                           uint32_t N) {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    return get_or_register_unlocked(matAdata, session, K, N);
  }

  /** @note Call with handle_mutex_ already held. */
  uint32_t get_or_register_unlocked(void *matAdata, remote_handle64 session,
                                    uint32_t K, uint32_t N) {
    auto it = handle_cache_.find(matAdata);
    if (it != handle_cache_.end())
      return it->second;

    const uint64_t t_begin = HtpProfile::nowUs();
    HtpRpcBuffer q_w4_i8(static_cast<size_t>(K) * N);
    std::vector<float> w_scale(N);
    std::vector<int32_t> colsum_w(N);
    const uint64_t t_convert = HtpProfile::nowUs();
    qs4cxFromModelQ4_0(matAdata, K, N,
                       reinterpret_cast<int8_t *>(q_w4_i8.data()),
                       w_scale.data(), colsum_w.data());
    const uint64_t convert_us = HtpProfile::nowUs() - t_convert;

    return register_locked(matAdata, session, K, N, q_w4_i8, w_scale, colsum_w,
                           t_begin, convert_us);
  }

  /** @brief The handles one FC weight is registered as, in column order. */
  struct FcHandles {
    std::vector<uint32_t> handles;
    std::vector<unsigned int> cols; /**< N of each handle */
  };

  /** @brief One FC weight's layer calls, in fcMaxRows row chunks. */
  void invokeFc(remote_handle64 session, const FcHandles &fh, float *matBdata,
                float *matCdata, unsigned int M, unsigned int N,
                unsigned int K) {
    const unsigned int step = fcMaxRows(K);
    for (unsigned int m0 = 0; m0 < M; m0 += step) {
      const unsigned int m = std::min(step, M - m0);
      invokeLayer(session, fh.handles.data(),
                  static_cast<int>(fh.handles.size()),
                  matBdata + static_cast<size_t>(m0) * K,
                  matCdata + static_cast<size_t>(m0) * N, m, N, K, &fh.cols);
    }
  }

  /** @brief Columns of a K-deep weight per registered handle.
   *
   * hexkl_mm_u8i4_layer_run lays VTCM out as the whole activation
   * (m_pad x K) | the WIDEST handle's WH bytes twice (its double buffer) |
   * a result tile, and returns AEE_ENOMEMORY when that exceeds the 8 MiB
   * (doc 50 section 3: conv in_proj, 2048 x 6144 = 6 MiB, wanted 12). A
   * 2 MiB slice keeps the double buffer at 4 MiB, which leaves the
   * activation room up to m_pad ~ 1,900 rows at K = 2048.
   * ponytail: the cap is the activation's ceiling, not the kernel's; a
   * prompt past it needs the kernel to walk 64-row blocks the way
   * hexkl_mm_u8i4_moe.c does. */
  static unsigned int fcSliceCols(unsigned int K) {
    // 2 MiB slices, 1 MiB from K = 4096 up. The slice's double buffer is
    // what the activation rows share VTCM with (fcMaxRows): at 2 MiB the
    // o-proj's 4096-deep weight left 896 rows a call, two calls for a
    // 1024-row prefill; at 1 MiB it is 1,472 and one call (the same bytes
    // stream through VTCM in narrower slices).
    // ponytail: K = 2816 at 1 MiB would make the qkv call one for 2048
    // rows too (2,368 a call), but the MoE call after it then fails with
    // AEE_ENOMEMORY on the S26 Ultra at C = 32 (DSP memory, not yet found
    // which); the qkv call stays two calls there.
    const unsigned int kSliceBytes = K >= 4096u ? 1u << 20 : 2u << 20;
    const unsigned int k_tiles = K / 32u;
    const unsigned int n_tiles =
      k_tiles == 0 ? 0 : (kSliceBytes / 512u) / k_tiles;
    const unsigned int cols = n_tiles < 1u ? 32u : n_tiles * 32u;
    // A slice holds whole heads, so a per-head norm folded into the call
    // (mm_u8i4_layer_norm's post chunks) never straddles two handles: 512
    // columns is the widest head dim here and divides every smaller one.
    // ponytail: a head dim that does not divide 512 needs its own
    // alignment passed down, not this constant.
    return cols >= 512u ? (cols / 512u) * 512u : cols;
  }

  /** @brief get_or_register for an FC weight of any width: one handle when
   *  it fits fcSliceCols, else one per column slice, converted once and
   *  registered slice by slice under keys inside the weight (its data
   *  pointer plus the slice's first column -- distinct, and valid as long
   *  as the weight is). Cached by the weight pointer like every handle.
   *  @param qs4cx_scale non-null: @a matAdata is a QS4CX weight with these
   *         per-channel scales (quantized offline, no Q4_0 detour); null:
   *         Q4_0x4 bytes. */
  const FcHandles &get_or_register_fc(void *matAdata, remote_handle64 session,
                                      uint32_t K, uint32_t N,
                                      const float *qs4cx_scale = nullptr) {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    auto it = fc_cache_.find(matAdata);
    if (it != fc_cache_.end())
      return it->second;

    FcHandles fh;
    const unsigned int cap = fcSliceCols(K);
    if (const FcWhEntry *w = fcwhFind(matAdata, K, N)) {
      // [#225] LFM's FC WH sidecar ahead of #4415's order; no-op without
      // fc_wh_file_name (Gemma sets none)
      for (uint32_t c0 = 0; c0 < N; c0 += cap) {
        const uint32_t n = std::min<uint32_t>(cap, N - c0);
        fh.handles.push_back(registerFcWh(static_cast<char *>(matAdata) + c0,
                                          session, {{w, c0, n}}, 0, K));
        fh.cols.push_back(n);
      }
    } else if (N <= cap) {
      fh.handles.push_back(
        qs4cx_scale
          ? get_or_register_qs4cx_unlocked(matAdata, qs4cx_scale, session, K, N)
          : get_or_register_unlocked(matAdata, session, K, N));
      fh.cols.push_back(N);
    } else {
      std::vector<int8_t> full(static_cast<size_t>(K) * N);
      std::vector<float> w_scale(N);
      std::vector<int32_t> colsum_w(N);
      // The first slice's profile entry carries the conversion, so its
      // clock starts before it; the later slices' start with their copy.
      uint64_t t_begin = HtpProfile::nowUs();
      if (qs4cx_scale)
        htp_qs4cx_from_packed(matAdata, qs4cx_scale, K, N, full.data(),
                              w_scale.data(), colsum_w.data());
      else
        qs4cxFromModelQ4_0(matAdata, K, N, full.data(), w_scale.data(),
                           colsum_w.data());
      uint64_t convert_us = HtpProfile::nowUs() - t_begin;
      for (uint32_t c0 = 0; c0 < N; c0 += cap) {
        const uint32_t n = std::min<uint32_t>(cap, N - c0);
        if (c0 != 0)
          t_begin = HtpProfile::nowUs();
        std::vector<int8_t> rm(static_cast<size_t>(K) * n);
        for (uint32_t k = 0; k < K; ++k) {
          std::memcpy(rm.data() + static_cast<size_t>(k) * n,
                      full.data() + static_cast<size_t>(k) * N + c0, n);
        }
        std::vector<float> ws(w_scale.begin() + c0, w_scale.begin() + c0 + n);
        std::vector<int32_t> cs(colsum_w.begin() + c0,
                                colsum_w.begin() + c0 + n);
        fh.handles.push_back(registerRm(static_cast<char *>(matAdata) + c0,
                                        session, rm.data(), K, n, ws, cs,
                                        t_begin, convert_us));
        fh.cols.push_back(n);
        convert_us = 0; // counted once, on the first slice
      }
    }
    return fc_cache_.emplace(matAdata, std::move(fh)).first->second;
  }

  /** @brief [#225] The sidecar image of the Q4_0 weight at @a q4, or
   *  nullptr when the model has no sidecar. A sidecar that lacks the weight
   *  or has it at another shape is a main file re-quantized without its
   *  sidecar, refused rather than re-quantized around. */
  const FcWhEntry *fcwhFind(const void *q4, uint32_t K, uint32_t N) const {
    const auto direct = direct_wh_.find(q4);
    if (direct != direct_wh_.end()) {
      if (direct->second.K != K || direct->second.N != N)
        throw std::invalid_argument("QS4CX_WH FC source shape mismatch");
      return &direct->second;
    }
    if (fcwh_fd_ < 0)
      return nullptr;
    const size_t len = static_cast<size_t>(N) * (K / 32u) * Q4_CPU_BLOCK_BYTES;
    auto it = fcwh_.find(fcWhKey(q4, len));
    if (it == fcwh_.end() || it->second.K != K || it->second.N != N) {
      throw std::runtime_error(
        "fc_wh_file_name " + fcwh_name_ + ": no image for a " +
        std::to_string(K) + "x" + std::to_string(N) +
        " Q4_0 weight of the model file -- the sidecar is not the one "
        "nntr_quantize_stream --fc_wh_sidecar wrote with this model file");
    }
    return &it->second;
  }

  void fcwhRead(const FcWhEntry *entry, void *dst, size_t bytes,
                uint64_t offset) const {
    const auto direct = direct_wh_.find(
      reinterpret_cast<const void *>(static_cast<uintptr_t>(entry->key)));
    if (direct != direct_wh_.end() && &direct->second == entry) {
      if (offset > entry->bytes || bytes > entry->bytes - offset)
        throw std::out_of_range("QS4CX_WH FC source read outside record");
      std::memcpy(dst, static_cast<const uint8_t *>(direct->first) + offset,
                  bytes);
    } else {
      throwPread(preadAll(fcwh_fd_, dst, bytes, offset), "fc wh image");
    }
  }

  /** @brief [#225] Columns [c0, c0 + cn) of one sidecar image. */
  struct WhPart {
    const FcWhEntry *w;
    uint32_t c0, cn;
  };

  /** @brief [#225] Registers rows [k0, k0 + kn) of the column parts
   *  @a parts, side by side, as one [kn x sum(cn)] weight, from the sidecar.
   *
   * WH tiles are k-major (htp_wh_layout.h), so a part's kt-row is one
   * contiguous run of cn / 32 tiles: the bytes are preads, straight into
   * the arena when it has room -- no host copy, no conversion. The scales
   * are the parts' own; the column sums too when the rows are all of K,
   * else (the dense down's row chunks) summed from the values. No room:
   * the DSP heap through weight_register_u8i4, from whUnpack's row-major
   * values, which the bake packs back into the same tiles -- registerRm's
   * overflow without its conversion. NNTR_HTP_FC_WH_HEAP=1 sends every
   * handle that way (the host check of that path).
   * ponytail: 4-bit only -- the sidecar is QS4CX_WH/1 and the ArenaEntry
   * leaves pal empty; plan 229 S2 (2-bit FCs) bumps FCWH_FORMAT and sizes
   * the image by its bits here (whBytes2 + the palette, as the 2-bit
   * experts do).
   * @note Call with handle_mutex_ already held. */
  uint32_t registerFcWh(void *key, remote_handle64 session,
                        const std::vector<WhPart> &parts, uint32_t k0,
                        uint32_t kn) {
    const uint64_t t_begin = HtpProfile::nowUs();
    uint32_t cn = 0;
    for (const WhPart &p : parts)
      cn += p.cn;
    const size_t len = whBytes(kn, cn);
    const size_t row = static_cast<size_t>(cn / WH_TILE) * WH_TILE_BYTES;
    auto read_tiles = [&](uint8_t *dst) {
      if (parts.size() == 1 && parts[0].cn == parts[0].w->N) {
        // every column: the k-tile rows are one contiguous run
        fcwhRead(parts[0].w, dst, len,
                 parts[0].w->off + static_cast<uint64_t>(k0) / WH_TILE *
                                     (cn / WH_TILE) * WH_TILE_BYTES);
        return;
      }
      for (uint32_t kt = 0; kt < kn / WH_TILE; ++kt) {
        size_t at = static_cast<size_t>(kt) * row;
        for (const WhPart &p : parts) {
          const uint64_t n_tiles = p.w->N / WH_TILE;
          const uint64_t src =
            p.w->off +
            ((k0 / WH_TILE + kt) * n_tiles + p.c0 / WH_TILE) * WH_TILE_BYTES;
          const size_t bytes =
            static_cast<size_t>(p.cn / WH_TILE) * WH_TILE_BYTES;
          fcwhRead(p.w, dst + at, bytes, src);
          at += bytes;
        }
      }
    };
    std::vector<float> ws(cn), cs_f(cn);
    std::vector<int32_t> cs(cn);
    bool affine = false;
    for (const auto &p : parts)
      affine |= p.w->bytes == whBytes(p.w->K, p.w->N) + 12u * size_t(p.w->N);
    std::vector<float> bias(affine ? 2u * cn : cn, 0.0f);
    for (size_t i = 0, at = 0; i < parts.size(); at += parts[i++].cn) {
      const WhPart &p = parts[i];
      const uint64_t tail = p.w->off + whBytes(p.w->K, p.w->N);
      fcwhRead(p.w, ws.data() + at, sizeof(float) * p.cn,
               tail + sizeof(float) * p.c0);
      fcwhRead(p.w, cs_f.data() + at, sizeof(float) * p.cn,
               tail + sizeof(float) * (p.w->N + p.c0));
      if (p.w->bytes == whBytes(p.w->K, p.w->N) + 12u * size_t(p.w->N)) {
        std::vector<int32_t> offsets(p.cn);
        fcwhRead(p.w, offsets.data(), sizeof(int32_t) * p.cn,
                 tail + sizeof(float) * 2u * p.w->N + sizeof(int32_t) * p.c0);
        for (uint32_t c = 0; c < p.cn; ++c) {
          if (offsets[c] < -15 || offsets[c] > 15)
            throw std::invalid_argument("Invalid WH affine offset");
          bias[cn + at + c] = static_cast<float>(offsets[c]);
        }
      }
    }
    std::vector<uint8_t> host;
    auto load_host = [&] {
      if (host.empty()) {
        host.resize(len);
        read_tiles(host.data());
      }
    };
    if (kn == parts[0].w->K) {
      for (uint32_t i = 0; i < cn; ++i)
        cs[i] = static_cast<int32_t>(cs_f[i]);
    } else {
      load_host();
      std::vector<int8_t> rm(static_cast<size_t>(kn) * cn);
      whUnpack(host.data(), kn, cn, rm.data());
      for (uint32_t r = 0; r < kn; ++r)
        for (uint32_t i = 0; i < cn; ++i)
          cs[i] += rm[static_cast<size_t>(r) * cn + i];
    }
    static const bool heap_only = [] {
      const char *v = std::getenv("NNTR_HTP_FC_WH_HEAP");
      return v != nullptr && std::atoi(v) != 0;
    }();
    // [#225 PR 2] On the E2E path these images are also the decode FC set,
    // and the expert pool's chunks leave them no room (plan 225 section
    // 3.5: pool C=28 3328 + FC WH 216 + lm_head Q4M1 147 = 3691 of the
    // PD's 3840 MiB), so they map their own chunk, sized for every image
    // still to come. The hybrid keeps PR 1's rule: mapped room or the heap.
    uint32_t chunk = 0, off = 0;
    const bool e2e = HtpBackend::e2eRequested();
    if (!heap_only && ensureArena(session) &&
        (e2e ? place(session, static_cast<uint32_t>(len), fcwhLeft(), &chunk,
                     &off)
             : placeExisting(static_cast<uint32_t>(len), &chunk, &off))) {
      uint8_t *dst = arena_chunks_[chunk].buf->data() + off;
      if (host.empty())
        read_tiles(dst);
      else
        std::memcpy(dst, host.data(), len);
      ArenaEntry e;
      e.chunk = chunk;
      e.off = off;
      e.K = kn;
      e.N = cn;
      e.w_scale = ws;
      e.colsum_w = cs;
      e.bias = bias;
      const uint32_t handle = registerFromArena(session, e, kn, cn, t_begin);
      if (handle != kNoHandle) {
        handle_cache_.emplace(key, handle);
        fcwh_arena_bytes_ += len;
        return handle;
      }
    }
    // Unpacked on the host and copied in whole: buf may be an uncached
    // rpcmem mapping, where whUnpack's scattered stores would crawl.
    load_host();
    std::vector<int8_t> rm(static_cast<size_t>(kn) * cn);
    whUnpack(host.data(), kn, cn, rm.data());
    HtpRpcBuffer buf(rm.size());
    std::memcpy(buf.data(), rm.data(), rm.size());
    fcwh_heap_bytes_ += len;
    return register_locked(key, session, kn, cn, buf, ws, cs, t_begin, 0,
                           &bias);
  }

  /** @brief Registers one row-major int8 [K x N] weight (values in [-8, 7],
   *  the form htp_qs4cx_from_* produce) with its scales and column sums.
   *
   * [doc 50 section 3.3] Into a mapped arena chunk's free room first: the
   * 3840 MiB mapped hold 3696 of MoE weights, and the DSP heap gave the
   * loaded app ~100 MiB before AEE_ENOMEMORY. Only room that is already
   * mapped -- a new chunk would take the address space the heap
   * registrations after this one need. The arena wants WH bytes: packed
   * on the host into a cached buffer and copied in whole, since whPack's
   * read-modify-write into the uncached chunk would crawl. When no mapped
   * room is left, the DSP heap, as a whole weight would have gone. A
   * refusal by the DSP leaves the arena bytes unused, once.
   * @param key what handle_cache_ files the handle under: an address inside
   *            the weight this is a slice of, distinct per slice
   * @note  Call with handle_mutex_ already held. */
  uint32_t registerRm(void *key, remote_handle64 session, const int8_t *rm,
                      uint32_t K, uint32_t N, std::vector<float> &ws,
                      std::vector<int32_t> &cs, uint64_t t_begin,
                      uint64_t convert_us) {
    const uint32_t wh_len = static_cast<uint32_t>(whBytes(K, N));
    uint32_t chunk = 0, off = 0;
    // NNTR_HTP_FC_IN_ARENA=1: new 256 MiB chunks when the mapped room runs
    // out, so the FC set (~880 MiB on Gemma 4) stays off the DSP heap. The
    // heap's growth stops near 1.7 GiB on the S25, and a 4k prompt's
    // quantized KV caches (~950 MiB) are heap allocations.
    constexpr bool fc_in_arena = htp_fixed::FC_IN_ARENA; // built in
    if (ensureArena(session) &&
        (fc_in_arena ? place(session, wh_len, size_t(256) << 20, &chunk, &off)
                     : placeExisting(wh_len, &chunk, &off))) {
      std::vector<uint8_t> wh(wh_len);
      whPack(rm, K, N, wh.data());
      std::memcpy(arena_chunks_[chunk].buf->data() + off, wh.data(), wh_len);
      ArenaEntry e;
      e.chunk = chunk;
      e.off = off;
      e.K = K;
      e.N = N;
      e.w_scale = ws;
      e.colsum_w = cs;
      e.bias.assign(N, 0.0f);
      const uint32_t handle = registerFromArena(session, e, K, N, t_begin);
      if (handle != kNoHandle) {
        handle_cache_.emplace(key, handle);
        return handle;
      }
    }
    HtpRpcBuffer buf(static_cast<size_t>(K) * N);
    std::memcpy(buf.data(), rm, static_cast<size_t>(K) * N);
    return register_locked(key, session, K, N, buf, ws, cs, t_begin,
                           convert_us);
  }

  /** @brief The handles one dense FFN is registered as: chunk c of the
   *  intermediate dimension is the expert pair (gate_up [K x 2w] with the
   *  gate columns first, down [w x N]) of the MoE layer kernel. */
  struct DenseHandles {
    std::vector<uint32_t> h_gu, h_dn;
    unsigned int w = 0; /**< columns of I per chunk */
  };

  /** @brief Columns of the intermediate dimension per chunk: the largest
   *  divisor of I that is a multiple of 32 and at most 1792 -- this
   *  model's expert width, so the MoE layer kernel lays VTCM out exactly
   *  as it does for the experts (gate_up 3.5 MiB, down 1.75). 0 if none. */
  static unsigned int denseChunkCols(unsigned int I) {
    for (unsigned int w = std::min(I, 1792u); w >= 32u; w -= 32u) {
      if (I % w == 0)
        return w;
    }
    return 0;
  }

  /** @brief Converts and registers a dense FFN's three Q4_0x4 weights as
   *  I / w expert pairs (doc 51). The gate_up chunk takes column slices of
   *  gate and up, so its column sums are the full ones; the down chunk
   *  takes a row slice, so its column sums are recomputed over those rows
   *  -- the kernel's zero-point correction is per chunk. Cached by the up
   *  weight's pointer. */
  const DenseHandles &get_or_register_dense(void *up, void *gate, void *down,
                                            remote_handle64 session, uint32_t K,
                                            uint32_t I, uint32_t N,
                                            const float *up_s = nullptr,
                                            const float *gate_s = nullptr,
                                            const float *down_s = nullptr) {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    auto it = dense_cache_.find(up);
    if (it != dense_cache_.end())
      return it->second;

    const unsigned int w = denseChunkCols(I);
    if (w == 0) {
      throw std::invalid_argument(
        "gemm_q4_0_dense_ffn_fp32: intermediate size " + std::to_string(I) +
        " has no chunk width that is a multiple of 32");
    }
    DenseHandles dh;
    dh.w = w;
    if (const FcWhEntry *wu = fcwhFind(up, K, I)) {
      // [#225] the same chunks from the sidecar: gate | up column slices
      // side by side, the down's row slices with their own column sums
      const FcWhEntry *wg = fcwhFind(gate, K, I), *wd = fcwhFind(down, I, N);
      for (uint32_t c0 = 0; c0 < I; c0 += w) {
        dh.h_gu.push_back(registerFcWh(static_cast<char *>(gate) + c0, session,
                                       {{wg, c0, w}, {wu, c0, w}}, 0, K));
        dh.h_dn.push_back(registerFcWh(static_cast<char *>(down) + c0, session,
                                       {{wd, 0, N}}, c0, w));
      }
      return dense_cache_.emplace(up, std::move(dh)).first->second;
    }
    uint64_t t_begin = HtpProfile::nowUs();
    std::vector<int8_t> up_rm(static_cast<size_t>(K) * I),
      gate_rm(static_cast<size_t>(K) * I), down_rm(static_cast<size_t>(I) * N);
    std::vector<float> up_s_v(I), gate_s_v(I), down_s_v(N);
    std::vector<int32_t> up_c(I), gate_c(I), down_c(N);
    // QS4CX weights (scales given) are a bit rearrangement, as the FC
    // path's; Q4_0 ones a requantization.
    if (up_s) {
      htp_qs4cx_from_packed(up, up_s, K, I, up_rm.data(), up_s_v.data(),
                            up_c.data());
      htp_qs4cx_from_packed(gate, gate_s, K, I, gate_rm.data(), gate_s_v.data(),
                            gate_c.data());
      htp_qs4cx_from_packed(down, down_s, I, N, down_rm.data(), down_s_v.data(),
                            down_c.data());
    } else {
      qs4cxFromModelQ4_0(up, K, I, up_rm.data(), up_s_v.data(), up_c.data());
      qs4cxFromModelQ4_0(gate, K, I, gate_rm.data(), gate_s_v.data(),
                         gate_c.data());
      qs4cxFromModelQ4_0(down, I, N, down_rm.data(), down_s_v.data(),
                         down_c.data());
    }
    uint64_t convert_us = HtpProfile::nowUs() - t_begin;

    for (uint32_t c0 = 0; c0 < I; c0 += w) {
      // gate_up chunk: [K x 2w], gate columns then up columns -- the pair
      // layout hexkl_mm_u8i4_moe.c's epilogue reads (gate j with up w+j).
      std::vector<int8_t> gu(static_cast<size_t>(K) * 2 * w);
      for (uint32_t k = 0; k < K; ++k) {
        std::memcpy(gu.data() + static_cast<size_t>(k) * 2 * w,
                    gate_rm.data() + static_cast<size_t>(k) * I + c0, w);
        std::memcpy(gu.data() + static_cast<size_t>(k) * 2 * w + w,
                    up_rm.data() + static_cast<size_t>(k) * I + c0, w);
      }
      std::vector<float> gus(gate_s_v.begin() + c0, gate_s_v.begin() + c0 + w);
      gus.insert(gus.end(), up_s_v.begin() + c0, up_s_v.begin() + c0 + w);
      std::vector<int32_t> guc(gate_c.begin() + c0, gate_c.begin() + c0 + w);
      guc.insert(guc.end(), up_c.begin() + c0, up_c.begin() + c0 + w);
      if (c0 != 0)
        t_begin = HtpProfile::nowUs();
      dh.h_gu.push_back(registerRm(static_cast<char *>(gate) + c0, session,
                                   gu.data(), K, 2 * w, gus, guc, t_begin,
                                   convert_us));
      convert_us = 0;

      // down chunk: rows c0 .. c0 + w, contiguous in the row-major weight.
      const int8_t *dn = down_rm.data() + static_cast<size_t>(c0) * N;
      std::vector<int32_t> dnc(N, 0);
      for (uint32_t r = 0; r < w; ++r) {
        for (uint32_t n = 0; n < N; ++n)
          dnc[n] += dn[static_cast<size_t>(r) * N + n];
      }
      std::vector<float> dns(down_s_v);
      dh.h_dn.push_back(registerRm(static_cast<char *>(down) + c0, session, dn,
                                   w, N, dns, dnc, HtpProfile::nowUs(), 0));
    }
    return dense_cache_.emplace(up, std::move(dh)).first->second;
  }

  bool supports_gemm_q4_0_dense_ffn_fp32() const override { return true; }

  void gemm_q4_0_dense_ffn_fp32(void *up, void *gate, void *down,
                                const float *act, float *out, unsigned int M,
                                unsigned int K, unsigned int I, unsigned int N,
                                bool gelu, const float *pre_gamma,
                                const float *post_gamma, float eps,
                                const float *up_scale, const float *gate_scale,
                                const float *down_scale,
                                bool out_to_host) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const DenseHandles &dh = get_or_register_dense(
      up, gate, down, session, K, I, N, up_scale, gate_scale, down_scale);
    // Every chunk is an "expert" that takes every row with weight 1: the
    // kernel zero-fills out and scatter-adds each chunk's down result into
    // it, which is exactly the sum over the intermediate dimension.
    const size_t chunks = dh.h_gu.size();
    std::vector<unsigned int> row_index(chunks * M), row_count(chunks, M);
    std::vector<float> row_weight(chunks * M, 1.0f);
    for (size_t c = 0; c < chunks; ++c) {
      for (unsigned int r = 0; r < M; ++r)
        row_index[c * M + r] = r;
    }
    invokeMoeLayer(session, dh.h_gu, dh.h_dn, row_index, row_count, row_weight,
                   act, out, M, K, dh.w, N, /*kind=*/1, gelu ? 1u : 0u,
                   pre_gamma, post_gamma, eps, nullptr, nullptr, 1.0f,
                   out_to_host);
  }

  bool register_q4_0_dense_ffn(void *up, void *gate, void *down, unsigned int K,
                               unsigned int I, unsigned int N,
                               const float *up_scale, const float *gate_scale,
                               const float *down_scale) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    get_or_register_dense(up, gate, down, session, K, I, N, up_scale,
                          gate_scale, down_scale);
    return true;
  }

  /** @brief Registers a conv block's two Q4_0x4 weights (doc 51 section
   *  2). in_proj goes through get_or_register_fc, whose column slices at
   *  K = 2048 are exactly C = 2048 wide, so its three handles ARE a, b
   *  and c and the kernel's split is free; a shape where the slices do
   *  not fall on the thirds is refused rather than re-sliced.
   *  ponytail: fcSliceCols(K) == C is this model's coincidence. Another
   *  shape needs its own slicing here (three registerRm calls at c0 = 0,
   *  C, 2C), not a change to the kernel. Cached by in_proj's pointer. */
  const ConvHandles &get_or_register_conv(void *in_proj, void *out_proj,
                                          remote_handle64 session, uint32_t K,
                                          uint32_t C, uint32_t N) {
    {
      std::lock_guard<std::mutex> lock(handle_mutex_);
      auto it = conv_cache_.find(in_proj);
      if (it != conv_cache_.end())
        return it->second;
    }
    const FcHandles &in = get_or_register_fc(in_proj, session, K, 3 * C);
    const FcHandles &out = get_or_register_fc(out_proj, session, C, N);
    if (in.handles.size() != 3 || in.cols[0] != C || in.cols[1] != C ||
        in.cols[2] != C || out.handles.size() != 1) {
      throw std::invalid_argument(
        "gemm_q4_0_conv_block_fp32: in_proj [" + std::to_string(K) + " x " +
        std::to_string(3 * C) + "] slices into " +
        std::to_string(in.handles.size()) + " handles of " +
        std::to_string(fcSliceCols(K)) + " columns, not the three of " +
        std::to_string(C) + " the conv block kernel takes");
    }
    std::lock_guard<std::mutex> lock(handle_mutex_);
    ConvHandles ch;
    ch.h_in = in.handles;
    ch.h_out = out.handles[0];
    return conv_cache_.emplace(in_proj, std::move(ch)).first->second;
  }

  bool supports_gemm_q4_0_conv_block_fp32() const override { return true; }

  void gemm_q4_0_conv_block_fp32(void *in_proj, const float *conv_w,
                                 void *out_proj, const float *act, float *out,
                                 float *state, unsigned int M, unsigned int K,
                                 unsigned int C, unsigned int N) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const ConvHandles &ch =
      get_or_register_conv(in_proj, out_proj, session, K, C, N);
    // [#225] In row chunks like invokeMoeLayer's, each after the first
    // continuing from the state the one before handed back
    const unsigned int step = prefillRows() != 0 ? prefillRows() : M;
    for (unsigned int m0 = 0; m0 < M; m0 += step) {
      invokeConvBlock(session, ch, conv_w, act + static_cast<size_t>(m0) * K,
                      out + static_cast<size_t>(m0) * N, state,
                      std::min(step, M - m0), K, C, N,
                      m0 != 0 ? state : nullptr);
    }
  }

  /** @brief [#225] Rows per call of the conv block's prefill, whose session
   *  scratch grows with M (LFM2 only; since #260 #4415's MoE layer, FC and
   *  dense FFN calls take the whole prefill in one call): 512, the
   *  prefill the DSP heap held beside the FC weights in #222's sitting
   *  where 1024 failed with AEE_ENOMEMORY. NNTR_HTP_PREFILL_ROWS=<n> sets
   *  another, 0 one call per prefill (the host check of the chunking). */
  static unsigned int prefillRows() {
    static const unsigned int rows = [] {
      const char *v = std::getenv("NNTR_HTP_PREFILL_ROWS");
      return v != nullptr ? static_cast<unsigned int>(std::atoi(v)) : 512u;
    }();
    return rows;
  }

  bool register_q4_0_conv_block(void *in_proj, void *out_proj, unsigned int K,
                                unsigned int C, unsigned int N) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    get_or_register_conv(in_proj, out_proj, session, K, C, N);
    return true;
  }

  /** Declared here rather than beside arena_chunks_ at the bottom of the
     class: registerFromArena takes an ArenaEntry by reference, and a
     parameter type has to be complete where the function is declared, unlike
     a member a function BODY refers to. Only an HTP build compiles this
     file, so a host build cannot catch it. */
  /** @brief One uncached ION buffer the DSP has mapped, filled front to
   *  back. Never reused or rewritten: a weight placed here keeps its bytes
   *  for the process lifetime, which is also why the DSP can borrow them
   *  and why no cache line in either direction can go stale. */
  struct ArenaChunk {
    std::unique_ptr<HtpRpcBuffer> buf;
    uint32_t dsp_id; /**< what nntr_hvx_arena_attach called it */
    size_t used;     /**< bump pointer */
  };

  /** @brief Where one weight sits, and its three small arrays when they
   *  are passed to the DSP by value (the resident path). An expert slot
   *  keeps them in the arena after the WH bytes instead (readWeight) and
   *  leaves these empty. */
  struct ArenaEntry {
    uint32_t chunk;
    uint32_t off;
    uint32_t K, N;
    std::vector<float> w_scale;
    std::vector<int32_t> colsum_w;
    std::vector<float> bias;
    /** Four int4 codes when this is a QS2CX_WH weight, empty at 4 bits. */
    std::vector<int8_t> pal;
    /** [PR 4428] A QS4CX_WH_AFFINE expert's per-column integer offsets,
     *  kept only when one of them is not zero: all-zero offsets register
     *  symmetric, which the kernel computes bit-identically and faster. */
    std::vector<int32_t> offsets;
  };

  /** @brief One expert's place in the arena: gate_up at off, down at
   *  off + its 4 KB-rounded length (register_qs4cx_wh_expert_file). */
  struct ExpertSlot {
    uint32_t chunk = 0;
    uint32_t off = 0;
    /** Its span (expertSlotBytes): the pool holds one size per expert
     *  dtype, 1.33 MB native TL2 beside 3.0 MB 4-bit on the rotated
     *  ternary Gemma4, and a free slot is reused only by its own size. */
    uint32_t bytes = 0;
    /** The retired expert's handles, still registered on the DSP until the
     *  next load into this slot swaps them out; kNoHandle when none. */
    uint32_t h_gu = kNoHandle, h_dn = kNoHandle;
  };
  struct ExpertResident {
    ExpertSlot slot;
    const void *key_dn;
    uint32_t h_gu, h_dn;
    ExpertFileDesc d; /**< [#216] its file ranges, for the WILLNEED on evict */
    bool tl2 = false;
  };

  /** @brief Read a complete range: 0 or errno (ENODATA at end of file).
   *  Does not throw, so a worker thread can call it. */
  static int preadAll(int fd, void *dst, size_t len, uint64_t off,
                      void *tail = nullptr, size_t tail_bytes = 0) {
    try {
      return weightPreadSplit(fd, dst, len, tail, tail_bytes, off);
    } catch (const std::bad_alloc &) {
      return ENOMEM;
    } catch (const std::invalid_argument &) {
      return EINVAL;
    } catch (...) {
      return EIO;
    }
  }

  /** @brief A short read into a weight is a wrong matmul, never a warning. */
  static void throwPread(int rc, const char *what) {
    if (rc == 0)
      return;
    if (rc < 0) {
      throw std::runtime_error(std::string("pread(") + what +
                               ") hit end of file: the model file is "
                               "shorter than its weights");
    }
    throw std::runtime_error(std::string("pread(") + what +
                             ") failed: " + std::strerror(rc));
  }

  /** @brief One expert on its way into a slot: where it comes from, where
   *  it goes, what the file said about it, and how the read went. */
  struct StagedExpert {
    ExpertFileDesc d;
    ExpertSlot slot;
    /** The slot's address, resolved when the slot was taken, so a reader
     *  thread never indexes arena_chunks_ while it might grow. */
    uint8_t *base;
    ArenaEntry gu, dn;
    int rc; /**< preadAll's result for the whole expert */
    bool tl2 = false;
    bool prefetched = false;
    bool prefetch_claimed = false;
  };

  /** Reserve DMA-visible spare slots before any token is in flight. The
   * resident cache capacity and replacement policy remain unchanged.
   * Workers only read; arena attachment happens once here, and the regular
   * DSP miss answer rebinds a consumed slot without a registration RPC.
   */
  void initPoolPrefetch() {
    if (!directPrefetch() || pool_prefetch_)
      return;
    const uint32_t predict = predictMode();
    if (predict == 0u || !prefetchThreads())
      throw std::invalid_argument(
        "NNTR_MOE_PREFETCH_DIRECT requires NNTR_HTP_PREDICT=1 or 2 and "
        "NNTR_MOE_PREFETCH > 0");
    if (pool_descs_.empty() || pool_descs_[0].empty()) {
      if (std::getenv("NNTR_MOE_PREFETCH_DIRECT") != nullptr)
        throw std::invalid_argument("Direct prefetch requires virtual experts");
      return; // built-in default on a model without the expert pool
    }
    // Keep decode worker tuning independent of prefill lookahead.
    size_t readers = prefetchThreads();
    if (std::getenv("NNTR_MOE_PREFETCH") == nullptr &&
        weightPageCachePolicy() == 0 &&
        pool_descs_[0][0].w_bits == HTP_TL2_WEIGHT_BITS)
      readers = htp_fixed::TL2_DIRECT_PREFETCH_READERS;
    if (const char *v = std::getenv("NNTR_MOE_PREFETCH_READERS"))
      readers =
        std::max<size_t>(1, std::min<size_t>(8, std::strtoul(v, nullptr, 10)));
    size_t count = htp_fixed::PREFETCH_SLOTS;
    if (const char *v = std::getenv("NNTR_MOE_PREFETCH_SLOTS"))
      count =
        std::max<size_t>(1, std::min<size_t>(32, std::strtoul(v, nullptr, 10)));
    std::vector<StagedExpert> slots;
    const auto session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    {
      std::lock_guard<std::mutex> lock(handle_mutex_);
      const auto &d = pool_descs_[0][0];
      const size_t bytes = expertSlotBytes(d);
      // The spare slots are the first layer's size; a layer whose experts
      // are another size (the rotated ternary's 4-bit tail) is not
      // prefetched (queueDirectPrefetch) and misses as before.
      for (const auto &layer : pool_descs_)
        for (const auto &x : layer)
          if (x.K != d.K || x.inter != d.inter || x.N_out != d.N_out)
            throw std::invalid_argument(
              "Direct prefetch requires uniform expert shapes");
      prefetch_slot_bytes_ = static_cast<uint32_t>(bytes);
      for (size_t i = 0; i < count; ++i) {
        ExpertSlot slot;
        slot.bytes = static_cast<uint32_t>(bytes);
        if (!place(session, bytes, count * bytes, &slot.chunk, &slot.off))
          throw std::runtime_error("Cannot allocate direct prefetch slots: " +
                                   arena_fail_);
        slots.push_back({d,
                         slot,
                         arena_chunks_[slot.chunk].buf->data() + slot.off,
                         {},
                         {},
                         0,
                         false});
      }
      std::fprintf(stderr,
                   "[HTP-PREFETCH] direct slots=%zu bytes=%zu readers=%zu\n",
                   count, count * bytes, readers);
    }
    auto cpus = backgroundCpus(cpuBit());
    using Prefetch = HtpExpertPrefetch<StagedExpert>;
    auto queue = std::make_shared<Prefetch>(
      std::move(slots), readers,
      [this, cpus](StagedExpert &st, const Prefetch::CancelCheck &cancelled) {
        // ThreadManager pins the caller; readers must not inherit that
        // single core and delay the mailbox server on every expert read.
        thread_local bool pinned = false;
        if (!pinned) {
          pinToCpus(cpus);
          pinned = true;
        }
        st.rc = readExpertWeight(st, false, false);
        if (st.rc != 0)
          return Prefetch::ReadResult::Failed;
        // A real miss claims its read before expiry, so only a retired guess
        // can skip down. The worker still owns its slot until it returns.
        if (st.d.w_bits == HTP_TL2_WEIGHT_BITS &&
            weightPageCachePolicy() == 0 && cancelStaleExpertReads() &&
            cancelled())
          return Prefetch::ReadResult::Cancelled;
        st.rc = readExpertWeight(st, true, false);
        return st.rc == 0 ? Prefetch::ReadResult::Complete
                          : Prefetch::ReadResult::Failed;
      });
    pool_prefetch_ = queue;
    if (predict == 2u)
      initCpuPredictor(cpus);
    auto predictor = cpu_predictor_;
    // atClose runs before atCloseLast unmaps the arena. Keep the queue alive
    // independently of singleton destruction order; no reader outlives DMA.
    HtpBackend::global().atClose([queue, predictor] {
      if (predictor) {
        predictor->drain();
        const auto s = predictor->stats();
        std::fprintf(stderr,
                     "[HTP-PREDICT] arm submitted=%llu completed=%llu "
                     "failed=%llu full=%llu\n",
                     (unsigned long long)s.submitted,
                     (unsigned long long)s.completed,
                     (unsigned long long)s.failed, (unsigned long long)s.full);
      }
      queue->drain();
      const auto s = queue->stats();
      std::fprintf(
        stderr,
        "[HTP-PREFETCH] submitted=%llu completed=%llu used=%llu late=%llu "
        "discarded=%llu failed=%llu full=%llu joined=%llu cancelled=%llu "
        "registration_rpc=0\n",
        (unsigned long long)s.submitted, (unsigned long long)s.completed,
        (unsigned long long)s.used, (unsigned long long)s.late,
        (unsigned long long)s.discarded, (unsigned long long)s.failed,
        (unsigned long long)s.full, (unsigned long long)s.joined,
        (unsigned long long)s.cancelled);
    });
  }

  struct PredictionJob {
    std::vector<float> input;
    size_t layer = 0;
    uint64_t epoch = 0;
    bool native = false;
  };
  struct PredictionRouter {
    unsigned hidden = 0, experts = 0, top = 0;
    const float *weights = nullptr;
    const float *gamma = nullptr;
  };

  static unsigned prefetchTop() {
    const char *v = std::getenv("NNTR_MOE_PREFETCH_TOP");
    return v ? std::min(16u,
                        static_cast<unsigned>(std::strtoul(v, nullptr, 10)))
             : 8u;
  }

  bool queueDirectPrefetch(const ExpertFileDesc &d, uint64_t epoch,
                           bool native) {
    if (expertSlotBytes(d) != prefetch_slot_bytes_)
      return false;
    return pool_prefetch_->submit(d.key_gu, epoch, [&](StagedExpert &st) {
      st.d = d;
      st.gu = {};
      st.dn = {};
      st.rc = 0;
      st.tl2 = d.w_bits == HTP_TL2_WEIGHT_BITS || (native && d.w_bits == 2u);
      st.prefetched = false;
      st.prefetch_claimed = false;
    });
  }

  /** Approximate lookahead only. Borrow the model's gate and gamma instead of
   * retaining another full gate copy. poolDisarm drains predictor jobs before
   * inference returns, so none can outlive the model's parameter storage.
   * RMS's positive scalar and softmax do not change the top-k order. Actual
   * routing and mixture weights always come from the NPU router.
   */
  void initCpuPredictor(const std::vector<int> &cpus) {
    prediction_routers_.resize(moe_ops_.size());
    for (size_t m = 0; m < moe_ops_.size(); ++m) {
      const uint32_t router = moe_ops_[m] - 1u;
      const auto *op = graphOp(router);
      if (op->kind != HTP_OP_ROUTER_TOPK || !op->eps_bits ||
          op->K > HTP_MBOX_PRED_MAX_WIDTH)
        throw std::invalid_argument(
          "ARM prefetch prediction requires Gemma softmax routers");
      const float *w = nullptr, *gamma = nullptr;
      for (const auto &p : load_params_)
        if (p.op == router) {
          if (p.which == HTP_GRAPH_PARAM_ROUTER_W)
            w = p.data;
          if (p.which == HTP_GRAPH_PARAM_ROUTER_BIAS)
            gamma = p.data;
        }
      if (!w || !gamma)
        throw std::invalid_argument("Missing ARM prediction router weights");
      auto &r = prediction_routers_[m];
      r.hidden = op->K;
      r.experts = op->n_experts;
      r.top = std::min(op->top_k, prefetchTop());
      r.weights = w;
      r.gamma = gamma;
    }
    std::vector<PredictionJob> jobs(2);
    for (auto &j : jobs)
      j.input.resize(prediction_routers_[0].hidden);
    if (const char *v = std::getenv("NNTR_MOE_PREFETCH_CONFIDENCE")) {
      const float threshold = std::strtof(v, nullptr);
      if (!std::isfinite(threshold) || threshold < 0 || threshold > 1)
        throw std::invalid_argument("Prefetch confidence must be in [0, 1]");
      if (threshold > 0)
        prediction_confidence_.reset(
          new HtpPrefetchConfidence(prediction_routers_.size(), threshold));
    }
    const char *blend_env = std::getenv("NNTR_MOE_PREFETCH_RESIDUAL");
    const float blend = blend_env ? std::strtof(blend_env, nullptr)
                                  : htp_fixed::PREFETCH_RESIDUAL;
    if (!std::isfinite(blend) || blend < 0 || blend > 1)
      throw std::invalid_argument("Prefetch residual must be in [0, 1]");
    std::shared_ptr<FILE> trace;
    if (const char *path = std::getenv("NNTR_HTP_PREDICT_LOG")) {
      FILE *file = std::fopen(path, "w");
      if (!file)
        throw std::runtime_error("Cannot open prediction trace");
      trace.reset(file, std::fclose);
    }
    auto residual = std::make_shared<HtpRouterResidual>(
      prediction_routers_.size(), prediction_routers_[0].hidden);
    cpu_predictor_ = std::make_shared<HtpExpertPrefetch<PredictionJob>>(
      std::move(jobs), 1,
      [this, cpus, blend, residual, trace](PredictionJob &job) {
        thread_local bool pinned = false;
        if (!pinned) {
          pinToCpus(cpus);
          pinned = true;
        }
        const auto &r = prediction_routers_[job.layer];
        const uint64_t token = job.epoch >> 32;
        std::vector<float> corrected(r.hidden);
        residual->observe(job.layer - 1u, token, job.input.data());
        if (trace) {
          std::fprintf(trace.get(), "%llu %zu", (unsigned long long)token,
                       job.layer);
          for (float candidate : {0.0f, 0.5f, 1.0f}) {
            residual->apply(job.layer, token, job.input.data(), candidate,
                            corrected.data());
            for (unsigned h = 0; h < r.hidden; ++h)
              corrected[h] *= r.gamma[h];
            const auto ranked = htpPredictExperts(corrected.data(), r.weights,
                                                  r.hidden, r.experts, 8u);
            std::fputs(" |", trace.get());
            for (unsigned e : ranked)
              std::fprintf(trace.get(), " %u", e);
          }
          std::fputc('\n', trace.get());
        }
        residual->apply(job.layer, token, job.input.data(), blend,
                        corrected.data());
        for (unsigned h = 0; h < r.hidden; ++h)
          corrected[h] *= r.gamma[h];
        const auto ids = htpPredictExperts(corrected.data(), r.weights,
                                           r.hidden, r.experts, r.top);
        std::lock_guard<std::mutex> lock(handle_mutex_);
        std::vector<bool> absent;
        for (size_t k = 0; k < ids.size(); ++k) {
          const auto &d = pool_descs_[job.layer][ids[k]];
          absent.push_back(experts_.count(d.key_gu) == 0);
          if (absent.back() && (!prediction_confidence_ ||
                                prediction_confidence_->allow(job.layer, k)))
            queueDirectPrefetch(d, job.epoch, job.native);
        }
        if (prediction_confidence_)
          prediction_confidence_->record(job.layer, job.epoch, ids, absent);
        return true;
      });
  }

  /** @note Call with handle_mutex_ held. */
  StagedExpert stageExpert(remote_handle64 session, const ExpertFileDesc &d) {
    if (!prepared_tl2_ && d.w_bits == 2u) {
      const char *path = std::getenv("NNTR_HTP_TL2_PREPARED_FILE");
      if (path && *path) {
        if (!e2e_)
          throw std::runtime_error("prepared TL2 requires NNTR_HTP_E2E=1");
        if (weightPageCachePolicy() >= 0)
          throw std::invalid_argument(
            "NNTR_HTP_PAGE_CACHE does not support prepared TL2 artifacts; "
            "unset NNTR_HTP_TL2_PREPARED_FILE");
        prepared_tl2_.reset(new PreparedTl2(path, d.fd));
        std::fprintf(stderr, "[HTP-TL2] prepared=%s receipt=%d\n", path,
                     prepared_tl2_->validatedReceipt());
      }
    }
    const ExpertSlot slot = takeExpertSlot(session, d);
    return StagedExpert{d,
                        slot,
                        arena_chunks_[slot.chunk].buf->data() + slot.off,
                        {},
                        {},
                        0,
                        d.w_bits == HTP_TL2_WEIGHT_BITS ||
                          (tl2_decode_ && d.w_bits == 2u)};
  }

  /** @brief A free slot of this expert's size, or a new one bumped into a
   *  chunk (the load, and the first prefill before the pool is full). The
   *  free list holds every size the model's experts come in; the LRU is
   *  shared across them, so a layer whose experts are larger than its
   *  reserved share grows the arena by the difference.
   *  ponytail: with one LRU over two slot sizes, the 4-bit layers can hold
   *  more experts than were reserved for them while same-count smaller
   *  slots idle; bounded by those two layers' expert count. A per-size LRU
   *  is the upgrade if the arena growth shows in the memory numbers.
   *  @note Call with handle_mutex_ held. */
  ExpertSlot takeExpertSlot(remote_handle64 session, const ExpertFileDesc &d) {
    if (d.fd < 0) {
      throw std::runtime_error("register_qs4cx_wh_expert_file: the virtual "
                               "expert weight carries no model file fd");
    }
    if (!ensureArena(session)) {
      throw std::runtime_error(
        "QS4CX_WH weights need the DSP arena, which this device did not "
        "provide (no rpcmem_to_fd or no fastrpc_mmap).");
    }
    const uint32_t slot_bytes = expertSlotBytes(d);
    ExpertSlot slot;
    slot.bytes = slot_bytes;
    // A new chunk is sized to the slots still to come when the caller said
    // how many there will be (reserve_qs4cx_wh_expert_slots), else a full
    // chunk as before. newChunk rounds up to its 64 MiB grain.
    const size_t left = expert_slots_wanted_ > expert_slots_placed_
                          ? expert_slots_wanted_ - expert_slots_placed_
                          : 0;
    const size_t want =
      left != 0 ? std::min(kArenaChunkMax, left * slot_bytes) : kArenaChunkMax;
    const auto free_it = std::find_if(
      free_expert_slots_.rbegin(), free_expert_slots_.rend(),
      [slot_bytes](const ExpertSlot &f) { return f.bytes == slot_bytes; });
    if (free_it != free_expert_slots_.rend()) {
      slot = *free_it;
      free_expert_slots_.erase(std::next(free_it).base());
    } else if (place(session, slot_bytes, want, &slot.chunk, &slot.off)) {
      ++expert_slots_placed_;
    } else {
      throw std::runtime_error(
        "HTP arena: cannot place an expert slot (" +
        std::to_string(slot_bytes >> 20) + " MiB). " +
        (arena_fail_.empty() ? "no chunk was attempted" : arena_fail_) +
        ". mapped=" + std::to_string(arenaBytes() >> 20) + " MiB in " +
        std::to_string(arena_chunks_.size()) +
        " chunks, RSS=" + std::to_string(rssKb() >> 10) + " MB");
    }
    return slot;
  }

  /** @brief One weight's span in an expert slot: its WH bytes, then its N
   *  f32 scales and N i32 column sums (doc 52 section 10.30: the DSP takes
   *  them from here, so a swap carries offsets and nothing else), rounded
   *  to a page. +44 KB on a 5.25 MB expert. */
  static uint32_t expertStride(uint32_t K, uint32_t N, uint32_t w_bits) {
    const auto capacity = codeBytes(K, N, w_bits);
    return (static_cast<uint32_t>(capacity) + 8u * N + 4095u) & ~4095u;
  }

  /** @brief One expert's slot: gate_up's stride then down's. An affine
   *  expert's offsets travel by RPC (weight_set_offsets_batch), not in
   *  the slot, so its stride is the symmetric 4-bit one. */
  static uint32_t expertSlotBytes(const ExpertFileDesc &d) {
    return expertStride(d.K, 2 * d.inter, d.w_bits) +
           expertStride(d.inter, d.N_out, d.w_bits);
  }

  /** @brief Bytes a column adds after the codes in the file: f32 scale and
   *  f32 column sum, plus an i32 offset for QS4CX_WH_AFFINE. */
  static uint64_t tailPerCol(const ExpertFileDesc &d) {
    return d.affine ? 12u : 8u;
  }

  /** @brief [plan 229] A weight's WH code bytes: whBytes at four bits,
   *  whBytes2 at two (QS2CX_WH). */
  static size_t codeBytes(uint32_t K, uint32_t N, uint32_t w_bits) {
    if (w_bits == HTP_TL2_WEIGHT_BITS)
      return htp_tl2_native_bytes(K, N);
    return w_bits == 2u ? whBytes2(K, N) : whBytes(K, N);
  }

  /** @brief [plan 229] The palette between a QS2CX_WH weight's codes and
   *  its scales in the file (QS2CX_WH_Tensor); none at four bits. */
  static uint32_t paletteBytes(uint32_t w_bits) {
    return w_bits == 2u ? WH_PALETTE_LEVELS : 0u;
  }

  /** @brief Reads both weights of @a st into its slot. No lock and no
   *  throw -- the slot is this expert's alone until registerStaged -- so a
   *  background thread can run it. @return 0, errno, or -1 at EOF. */
  int readExpert(StagedExpert &st, bool use_pool, bool advise = true) {
    int rc = readExpertWeight(st, false, use_pool);
    if (rc == 0)
      rc = readExpertWeight(st, true, use_pool);
    if (rc == 0 && advise && fadviseKnob() != 0) // [#216] the slot holds it
      adviseLater({{st.d, false}});
    return rc;
  }

  /** @brief readExpert's half: gate_up, or down when @a dn. Touches only
   *  that half's ArenaEntry, so the two can run on two threads. */
  int readExpertWeight(StagedExpert &st, bool dn, bool use_pool) {
    const ExpertFileDesc &d = st.d;
    const uint32_t gu_stride = expertStride(d.K, 2 * d.inter, d.w_bits);
    ArenaEntry &e = dn ? st.dn : st.gu;
    e.chunk = st.slot.chunk;
    e.off = st.slot.off + (dn ? gu_stride : 0u);
    if (prepared_tl2_ && d.w_bits == 2u) {
      try {
        const uint32_t K = dn ? d.inter : d.K, N = dn ? d.N_out : 2u * d.inter;
        uint8_t *dst = st.base + (dn ? gu_stride : 0u);
        std::array<int8_t, 4> pal;
        prepared_tl2_->read(d.fd, dn ? d.off_dn : d.off_gu, K, N, st.tl2, dst,
                            pal.data());
        e.pal.assign(pal.begin(), pal.end());
        e.K = K;
        e.N = N;
        return 0;
      } catch (const std::exception &err) {
        std::fprintf(stderr, "[HTP-TL2] %s\n", err.what());
        return EINVAL;
      }
    }
    return dn ? readWeight(d.fd, d.off_dn, d.inter, d.N_out, d.w_bits, d.affine,
                           st.base + gu_stride, e, use_pool)
              : readWeight(d.fd, d.off_gu, d.K, 2 * d.inter, d.w_bits, d.affine,
                           st.base, e, use_pool);
  }

  /** Diagnostic opt-outs keep both changes independently measurable. */
  static bool coalesceExpertReads() {
    static const bool enabled = [] {
      const char *v = std::getenv("NNTR_MOE_COALESCE_READ");
      return !v || std::strcmp(v, "0") != 0;
    }();
    return enabled;
  }

  static bool cancelStaleExpertReads() {
    static const bool enabled = [] {
      const char *v = std::getenv("NNTR_MOE_CANCEL_STALE_READ");
      return !v || std::strcmp(v, "0") != 0;
    }();
    return enabled;
  }

  static bool validNativeTl2Shape(uint32_t K, uint32_t N) {
    return K && N && K <= 8192u && N <= 8192u && K % 32u == 0 &&
           N % 128u == 0 && htp_tl2_native_bytes(K, N) <= whBytes2(K, N);
  }

  /** Only adjacent native TL2 records that fit one aligned direct read.
   * Invalid shapes fall through to the regular reader's error handling. */
  static bool canReadExpertTogether(const ExpertFileDesc &d) {
    if (!coalesceExpertReads() || weightPageCachePolicy() != 0 ||
        d.w_bits != HTP_TL2_WEIGHT_BITS || !d.inter || d.inter > 4096u ||
        !validNativeTl2Shape(d.K, 2u * d.inter) ||
        !validNativeTl2Shape(d.inter, d.N_out))
      return false;
    const size_t gu_bytes =
      htp_tl2_native_bytes(d.K, 2u * d.inter) + 8u * (2u * d.inter);
    const size_t dn_bytes =
      htp_tl2_native_bytes(d.inter, d.N_out) + 8u * d.N_out;
    if (d.off_dn < d.off_gu || d.off_dn - d.off_gu != gu_bytes ||
        d.off_dn > UINT64_MAX - dn_bytes)
      return false;
    return gu_bytes + dn_bytes <=
           WEIGHT_DIRECT_WINDOW_BYTES - d.off_gu % WEIGHT_DIRECT_ALIGNMENT;
  }

  /** Demand only: read the two payloads and metadata tails through the same
   * bounded scratch, without an intermediate full expert copy. */
  int readExpertTogether(StagedExpert &st) {
    const ExpertFileDesc &d = st.d;
    const uint32_t gu_n = 2u * d.inter;
    const uint32_t gu_stride = expertStride(d.K, gu_n, d.w_bits);
    std::vector<float> gu_tail(2u * gu_n), dn_tail(2u * d.N_out);
    const WeightReadSlice parts[] = {
      {st.base, htp_tl2_native_bytes(d.K, gu_n)},
      {gu_tail.data(), gu_tail.size() * sizeof(float)},
      {st.base + gu_stride, htp_tl2_native_bytes(d.inter, d.N_out)},
      {dn_tail.data(), dn_tail.size() * sizeof(float)}};
    const int rc = weightPreadScatter(d.fd, parts, 4, d.off_gu);
    if (rc)
      return rc;
    st.gu.chunk = st.dn.chunk = st.slot.chunk;
    st.gu.off = st.slot.off;
    st.dn.off = st.slot.off + gu_stride;
    const int gu_rc =
      finishTl2Weight(d.K, gu_n, gu_tail.data(), st.base, st.gu);
    return gu_rc ? gu_rc
                 : finishTl2Weight(d.inter, d.N_out, dn_tail.data(),
                                   st.base + gu_stride, st.dn);
  }

  /** Move a filled spare into a retired DSP pair without a copy or RPC. */
  static void adoptPrefetch(StagedExpert &st, StagedExpert &ready) {
    ready.slot.h_gu = st.slot.h_gu;
    ready.slot.h_dn = st.slot.h_dn;
    st.slot.h_gu = st.slot.h_dn = kNoHandle;
    std::swap(st, ready);
    st.prefetched = true;
    st.prefetch_claimed = false;
  }

  /** @brief [#266] A miss round's reads, all at once: one job per weight,
   *  or per adjacent native TL2 pair that fits one direct read.
   *  NNTR_MOE_MISS_READERS threads of the ThreadManager
   *  taking them in turn, each job one whole-range pread plus its tail.
   *  The S25 probe (docs/measurements/266-miss-readers.md): rounds of 3
   *  experts into ION read 3.1 GiB/s this way at 4 readers, 1.8 GiB/s as
   *  each weight split in 8 slices read one expert after the other (the
   *  old miss path: the slices share one readahead window). Each
   *  StagedExpert's rc is set; no lock, no throw. The readers are the
   *  pool server and ThreadManager's workers, so at most
   *  getComputeThreadCount() (NNTR_NUM_THREADS) run at once. */

  void readMisses(std::vector<StagedExpert> &st) {
    std::vector<size_t> jobs;
    std::vector<bool> together(st.size(), false);
    size_t pair_reads = 0;
    for (size_t i = 0; i < st.size(); ++i)
      if (!st[i].prefetched && !st[i].prefetch_claimed) {
        together[i] = canReadExpertTogether(st[i].d);
        jobs.push_back(2 * i);
        if (together[i])
          ++pair_reads;
        else
          jobs.push_back(2 * i + 1);
      }
    const size_t n_jobs = jobs.size();
    std::vector<int> rc(2 * st.size(), 0);
    if (n_jobs) {
      const auto read = [&](size_t j) {
        const size_t part = jobs[j];
        rc[part] = together[part / 2]
                     ? readExpertTogether(st[part / 2])
                     : readExpertWeight(st[part / 2], part % 2 != 0, false);
      };
      // A small miss batch need not wake every compute worker and leave
      // them spinning between I/O rounds. Zero restores the compute pool.
      static const bool dedicated = [] {
        const char *v = std::getenv("NNTR_MOE_DEMAND_POOL");
        return !v || std::strcmp(v, "0") != 0;
      }();
      if (dedicated) {
        if (!demand_read_pool_) {
          const size_t readers = std::min<size_t>(
            missReaders(), ThreadManager::Global().getComputeThreadCount());
          auto cpus = backgroundCpus(cpuBit());
          demand_read_pool_.reset(
            new HtpExpertReadPool(readers, [cpus] { pinToCpus(cpus); }));
          std::fprintf(stderr, "[HTP] demand read pool: readers=%zu\n",
                       readers);
        }
        demand_read_pool_->run(n_jobs, read);
      } else {
        std::atomic<size_t> next{0};
        ThreadManager::Global().parallel_for(
          0, std::min(missReaders(), n_jobs), [&](size_t) {
            for (size_t j; (j = next.fetch_add(1)) < n_jobs;)
              read(j);
          });
      }
    }
    for (size_t i = 0; i < st.size(); ++i) {
      auto &expert = st[i];
      expert.rc = rc[2 * i] != 0 ? rc[2 * i] : rc[2 * i + 1];
      if (expert.prefetch_claimed) {
        const bool ok = pool_prefetch_->takeClaimed(
          expert.d.key_gu,
          [&expert](StagedExpert &ready) { adoptPrefetch(expert, ready); });
        expert.prefetch_claimed = false;
        if (!ok) {
          if (canReadExpertTogether(expert.d)) {
            ++pair_reads;
            expert.rc = readExpertTogether(expert);
          } else {
            expert.rc = readExpert(expert, false, false);
          }
        }
      }
    }
    if (e2e_st_)
      e2e_st_->pool_pair_reads += pair_reads;
  }

  /** @brief [#266] NNTR_MOE_MISS_READERS: the miss round's reader count
   *  (4, the S25 probe's best for whole-weight reads; 1 reads the round's
   *  weights one after another on the pool server). */
  static size_t missReaders() {
    static const size_t n = [] {
      const char *v = std::getenv("NNTR_MOE_MISS_READERS");
      return v != nullptr ? std::max<size_t>(1, std::strtoul(v, nullptr, 10))
                          : size_t(4);
    }();
    return n;
  }

  /** @brief [#216] NNTR_MOE_FADVISE: unset / 0 = no advice (the bytes and
   *  the reads are the same either way); 1 = an expert's file pages are
   *  dropped from the page cache once its bytes are in the arena and asked
   *  back when it leaves the arena, so the cache holds about the arena's
   *  complement instead of the whole file; 2 = the drop only (diagnostic:
   *  every miss then reads storage). */
  /** @brief [#289] The process's resident set now, MiB (0 off Linux). */
  static double rssMib() {
    double mib = 0.0;
#if defined(__linux__)
    if (FILE *f = std::fopen("/proc/self/statm", "r")) {
      unsigned long size = 0, res = 0;
      if (std::fscanf(f, "%lu %lu", &size, &res) == 2)
        mib = static_cast<double>(res) * sysconf(_SC_PAGESIZE) / 1048576.0;
      std::fclose(f);
    }
#endif
    return mib;
  }

  /** @brief [#289] /proc/meminfo's MemAvailable, MiB (0 off Linux): the
   *  dma-buf blocks (the KV cache, the arena) are not in the RSS. */
  static double memAvailMib() {
    double mib = 0.0;
#if defined(__linux__)
    if (FILE *f = std::fopen("/proc/meminfo", "r")) {
      char key[64];
      unsigned long kib = 0;
      while (std::fscanf(f, "%63s %lu kB", key, &kib) == 2)
        if (std::strcmp(key, "MemAvailable:") == 0) {
          mib = static_cast<double>(kib) / 1024.0;
          break;
        }
      std::fclose(f);
    }
#endif
    return mib;
  }

  /** @brief [#289] NNTR_MOE_LRFU unset or a half-life other than 0: the
   *  pool's policy (causallm::ExpertLru, whose default this restates) is
   *  LRFU, which parses the value. */
  static bool lrfuOn() {
    static const bool on = [] {
      const char *v = std::getenv("NNTR_MOE_LRFU");
      return v == nullptr || *v == '\0' || std::strtod(v, nullptr) > 0.0;
    }();
    return on;
  }

  static int fadviseKnob() {
    if (weightPageCachePolicy() >= 0)
      return 0;
    static const int k = [] {
      const char *v = std::getenv("NNTR_MOE_FADVISE");
      return v != nullptr ? std::atoi(v) : 0;
    }();
    return k;
  }

  /** @brief [#216] posix_fadvise DONTNEED (or WILLNEED) on both weight
   *  ranges of @a d as readWeight reads them: nibbles, N scales, N sums.
   *  Advice only -- a failure changes nothing a matmul reads, so it is not
   *  checked. */
  static void fadviseExpert(const ExpertFileDesc &d, bool willneed) {
    if (weightPageCachePolicy() >= 0)
      return;
#if defined(__linux__)
    // One WILLNEED reads at most the device's read-ahead window (measured:
    // 1 MiB on the S25's 6.6 kernel, 128 KiB on the workstation) of its
    // range, so it goes in 128 KiB pieces; DONTNEED takes the whole range.
    const uint64_t piece = willneed ? (uint64_t(128) << 10) : ~uint64_t(0);
    const auto advise = [&](uint64_t off, uint64_t len) {
      for (uint64_t at = 0; at < len; at += std::min(piece, len - at))
        (void)posix_fadvise(d.fd, static_cast<off_t>(off + at),
                            static_cast<off_t>(std::min(piece, len - at)),
                            willneed ? POSIX_FADV_WILLNEED
                                     : POSIX_FADV_DONTNEED);
    };
    const uint32_t n_gu = 2 * d.inter;
    const uint64_t pal = paletteBytes(d.w_bits);
    advise(d.off_gu,
           codeBytes(d.K, n_gu, d.w_bits) + pal + tailPerCol(d) * n_gu);
    advise(d.off_dn, codeBytes(d.inter, d.N_out, d.w_bits) + pal +
                       tailPerCol(d) * d.N_out);
#else
    (void)d;
    (void)willneed;
#endif
  }

  /** @brief [#216] fadviseExpert on each (expert, willneed) of @a v, on
   *  one background thread: on the S25 a WILLNEED reads before it returns
   *  (3-8 ms an expert) and a DONTNEED took about 2 ms on a miss's path;
   *  a thread per call inherited its caller's core pin (the pool server's)
   *  and stretched S1's miss wait. The unpinned worker drains and joins at
   *  model teardown. Each queued advice owns a dup of the file descriptor. */
  using Advice = std::pair<ExpertFileDesc, bool>;
  struct AdviceQueue {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<Advice> q;
    bool stop = false;
    std::thread worker;
  };
  std::unique_ptr<AdviceQueue> advice_;

  void adviseLater(std::vector<Advice> v) {
    if (v.empty())
      return;
    if (!advice_) {
      advice_ = std::make_unique<AdviceQueue>();
      auto *n = advice_.get();
      n->worker = std::thread([n] {
#if defined(__linux__)
        cpu_set_t all;
        CPU_ZERO(&all);
        for (long c = 0; c < sysconf(_SC_NPROCESSORS_CONF) && c < CPU_SETSIZE;
             ++c)
          CPU_SET(c, &all);
        sched_setaffinity(0, sizeof(all), &all);
#endif
        for (;;) {
          std::unique_lock<std::mutex> lock(n->mu);
          n->cv.wait(lock, [n] { return n->stop || !n->q.empty(); });
          if (n->q.empty() && n->stop)
            return;
          const Advice a = n->q.front();
          n->q.pop_front();
          lock.unlock();
          fadviseExpert(a.first, a.second);
          ::close(a.first.fd);
        }
      });
    }
    auto *q = advice_.get();
    {
      std::lock_guard<std::mutex> lock(q->mu);
      for (Advice &a : v)
        if ((a.first.fd = ::dup(a.first.fd)) >= 0)
          q->q.push_back(a);
    }
    q->cv.notify_one();
  }

  /** @brief [#216] /proc/vmstat pgpgin (KiB read from block devices,
   *  system-wide), 0 where there is no such file. */
  static uint64_t vmstatPgpginKib() {
    uint64_t v = 0;
    if (FILE *f = std::fopen("/proc/vmstat", "r")) {
      char k[64];
      unsigned long long n = 0;
      while (std::fscanf(f, "%63s %llu", k, &n) == 2)
        if (std::strcmp(k, "pgpgin") == 0) {
          v = n;
          break;
        }
      std::fclose(f);
    }
    return v;
  }

  /** @brief Registers a read expert and files it. On any failure the slot
   *  goes back to the free list and this throws.
   *  @note Call with handle_mutex_ held. */
  void registerStaged(remote_handle64 session, StagedExpert &st) {
    if (st.d.w_bits == 2u || st.d.w_bits == HTP_TL2_WEIGHT_BITS) {
      std::vector<StagedExpert> one{st};
      registerStagedBatch(session, one);
      return;
    }
    const ExpertFileDesc &d = st.d;
    uint32_t h_gu = kNoHandle, h_dn = kNoHandle;
    int err = AEE_SUCCESS;
    try {
      throwPread(st.rc, "expert weight");
      // One round trip: register the pair just read and release the pair
      // retired from this slot, if any (release_qs4cx_wh_expert). On error
      // the DSP has changed nothing, so the slot goes back to the free list
      // still holding its retired pair.
      err = nntr_hvx_weight_swap_u8i4_arena(
        session, st.slot.h_gu, st.slot.h_dn, d.K, d.inter, d.N_out,
        arena_chunks_[st.slot.chunk].dsp_id, st.gu.off, st.dn.off,
        st.gu.w_scale.data(), static_cast<int>(st.gu.w_scale.size()),
        st.gu.colsum_w.data(), static_cast<int>(st.gu.colsum_w.size()),
        st.dn.w_scale.data(), static_cast<int>(st.dn.w_scale.size()),
        st.dn.colsum_w.data(), static_cast<int>(st.dn.colsum_w.size()), &h_gu,
        &h_dn);
      if (err != AEE_SUCCESS) {
        char code[16];
        std::snprintf(code, sizeof(code), "0x%08x", static_cast<unsigned>(err));
        throw std::runtime_error(
          std::string("nntr_hvx_weight_swap_u8i4_arena failed: err=") + code +
          " (a skel older than test/htp/nntr_hvx.idl answers this call with "
          "an error: rebuild and push libnntr_hvx_skel.so)");
      }
    } catch (...) {
      free_expert_slots_.push_back(st.slot);
      throw;
    }
    fileRegistered(st, h_gu, h_dn);
  }

  /** @brief Files an expert the DSP has just registered.
   *  @note Call with handle_mutex_ held. */
  void fileRegistered(StagedExpert &st, uint32_t h_gu, uint32_t h_dn) {
    const ExpertFileDesc &d = st.d;
    st.slot.h_gu = st.slot.h_dn = kNoHandle; // released by the swap
    handle_cache_[d.key_gu] = h_gu;
    handle_cache_[d.key_dn] = h_dn;
    experts_.emplace(d.key_gu,
                     ExpertResident{st.slot, d.key_dn, h_gu, h_dn, d, st.tl2});
  }

  /**
   * @brief registerStaged for several experts in ONE round trip (doc 52
   *        section 10.23): a prefill layer's read-ahead batch, or one call's
   *        misses. The experts the DSP swapped are filed; one whose read
   *        failed, and every one from the first the DSP refused on, goes
   *        back to the free list with its retired pair, which the DSP still
   *        holds. The first failure is thrown after that.
   * @note Call with handle_mutex_ held.
   */
  void registerStagedBatch(remote_handle64 session,
                           std::vector<StagedExpert> &sts) {
    std::exception_ptr first;
    std::vector<StagedExpert *> ok;
    for (StagedExpert &st : sts) {
      if (st.rc == 0) {
        ok.push_back(&st);
        continue;
      }
      free_expert_slots_.push_back(st.slot);
      if (!first) {
        try {
          throwPread(st.rc, "expert weight");
        } catch (...) {
          first = std::current_exception();
        }
      }
    }
    if (!ok.empty()) {
      const ExpertFileDesc &d0 = ok[0]->d;
      const size_t n = ok.size();
      std::vector<uint32_t> og(n), od(n), ar(n), ofg(n), ofd(n);
      std::vector<uint32_t> hg(n, kNoHandle), hd(n, kNoHandle);
      std::vector<float> gs, ds;
      std::vector<int32_t> gc, dc;
      std::vector<int8_t> pg, pd; // [plan 229] QS2CX_WH palettes, 4 each
      for (size_t i = 0; i < n; ++i) {
        const StagedExpert &st = *ok[i];
        pg.insert(pg.end(), st.gu.pal.begin(), st.gu.pal.end());
        pd.insert(pd.end(), st.dn.pal.begin(), st.dn.pal.end());
        og[i] = st.slot.h_gu;
        od[i] = st.slot.h_dn;
        ar[i] = arena_chunks_[st.slot.chunk].dsp_id;
        ofg[i] = st.gu.off;
        ofd[i] = st.dn.off;
        gs.insert(gs.end(), st.gu.w_scale.begin(), st.gu.w_scale.end());
        gc.insert(gc.end(), st.gu.colsum_w.begin(), st.gu.colsum_w.end());
        ds.insert(ds.end(), st.dn.w_scale.begin(), st.dn.w_scale.end());
        dc.insert(dc.end(), st.dn.colsum_w.begin(), st.dn.colsum_w.end());
      }
      const int ni = static_cast<int>(n);
      uint32_t done = 0;
      int32_t err = AEE_SUCCESS;
      const bool two = d0.w_bits == 2u || d0.w_bits == HTP_TL2_WEIGHT_BITS;
      const bool compact_tl2 = d0.w_bits == HTP_TL2_WEIGHT_BITS;
      const int rc =
        compact_tl2
          ? nntr_hvx_weight_swap_batch_tl2_compact_arena(
              session, d0.K, d0.inter, d0.N_out, og.data(), ni, od.data(), ni,
              ar.data(), ni, ofg.data(), ni, ofd.data(), ni, hg.data(), ni,
              hd.data(), ni, &done, &err)
        : two ? nntr_hvx_weight_swap_batch_u2i4_arena(
                  session, d0.K, d0.inter, d0.N_out, og.data(), ni, od.data(),
                  ni, ar.data(), ni, ofg.data(), ni, ofd.data(), ni, pg.data(),
                  static_cast<int>(pg.size()), pd.data(),
                  static_cast<int>(pd.size()), hg.data(), ni, hd.data(), ni,
                  &done, &err)
              : nntr_hvx_weight_swap_batch_u8i4_arena(
                  session, d0.K, d0.inter, d0.N_out, og.data(), ni, od.data(),
                  ni, ar.data(), ni, ofg.data(), ni, ofd.data(), ni, gs.data(),
                  static_cast<int>(gs.size()), gc.data(),
                  static_cast<int>(gc.size()), ds.data(),
                  static_cast<int>(ds.size()), dc.data(),
                  static_cast<int>(dc.size()), hg.data(), ni, hd.data(), ni,
                  &done, &err);
      if (rc != AEE_SUCCESS) { // refused whole: the DSP changed nothing
        done = 0;
        err = rc;
      }
      if (done && ok[0]->tl2 && !compact_tl2) {
        std::vector<uint32_t> handles;
        for (uint32_t i = 0; i < done; ++i) {
          handles.push_back(hg[i]);
          handles.push_back(hd[i]);
        }
        if (nntr_hvx_weights_tl2(session, handles.data(), handles.size(), 1u))
          throw std::runtime_error("TL2 registration failed");
      }
      for (size_t i = 0; i < n; ++i) {
        if (i < done)
          fileRegistered(*ok[i], hg[i], hd[i]);
        else
          free_expert_slots_.push_back(ok[i]->slot);
      }
      // [PR 4428] The affine experts' non-zero offsets, one call for the
      // batch: the swap registered them symmetric, and a handle whose
      // offsets are all zero stays that way (bit-identical, faster).
      std::vector<uint32_t> oh, on;
      std::vector<float> ov;
      for (size_t i = 0; i < done; ++i)
        for (const auto &[h, e] :
             {std::pair<uint32_t, const ArenaEntry *>{hg[i], &ok[i]->gu},
              std::pair<uint32_t, const ArenaEntry *>{hd[i], &ok[i]->dn}})
          if (!e->offsets.empty()) {
            oh.push_back(h);
            on.push_back(e->N);
            ov.insert(ov.end(), e->offsets.begin(), e->offsets.end());
          }
      if (!oh.empty()) {
        const int rc_off = nntr_hvx_weight_set_offsets_batch(
          session, oh.data(), static_cast<int>(oh.size()), on.data(),
          static_cast<int>(on.size()), ov.data(), static_cast<int>(ov.size()));
        if (rc_off != AEE_SUCCESS) {
          char code[16];
          std::snprintf(code, sizeof(code), "0x%08x",
                        static_cast<unsigned>(rc_off));
          throw std::runtime_error(
            std::string("nntr_hvx_weight_set_offsets_batch failed: err=") +
            code +
            " (an affine expert's offsets did not reach the DSP; a "
            "skel older than test/htp/nntr_hvx.idl answers this call "
            "with an error: rebuild and push libnntr_hvx_skel.so)");
        }
      }
      if (done < n && !first) {
        char code[16];
        std::snprintf(code, sizeof(code), "0x%08x", static_cast<unsigned>(err));
        first = std::make_exception_ptr(std::runtime_error(
          std::string(compact_tl2
                        ? "nntr_hvx_weight_swap_batch_tl2_compact_arena"
                      : two ? "nntr_hvx_weight_swap_batch_u2i4_arena"
                            : "nntr_hvx_weight_swap_batch_u8i4_arena") +
          " failed: err=" + code + " at expert " + std::to_string(done) +
          " of " + std::to_string(n) +
          " (a skel older than test/htp/nntr_hvx.idl answers this call with "
          "an error: rebuild and push libnntr_hvx_skel.so)"));
      }
    }
    if (first)
      std::rethrow_exception(first);
  }

  /** @brief Starts the readers once, pinned to NNTR_MOE_PREFETCH_CPUS or
   *  else to every core but the caller's. */
  void startPrefetchReaders() {
    if (!prefetch_readers_.empty())
      return;
    const PrefetchKnobs &knobs = prefetchKnobs();
    prefetch_caller_cpus_ = cpuBit();
    std::vector<int> cpus = knobs.cpus;
    if (cpus.empty() && prefetch_caller_cpus_ != 0) {
      const long n = sysconf(_SC_NPROCESSORS_CONF);
      for (int c = 0; c < n && c < 32; ++c)
        if ((prefetch_caller_cpus_ & (1u << c)) == 0)
          cpus.push_back(c);
    }
    for (size_t t = 0; t < knobs.readers; ++t)
      prefetch_readers_.emplace_back([this, cpus] {
        pinToCpus(cpus);
        prefetchReaderLoop();
      });
  }

  void prefetchReaderLoop() {
    for (;;) {
      std::pair<PrefetchBatch *, size_t> job;
      {
        std::unique_lock<std::mutex> lock(prefetch_mutex_);
        prefetch_cv_.wait(
          lock, [this] { return prefetch_stop_ || !prefetch_jobs_.empty(); });
        if (prefetch_stop_)
          return;
        job = prefetch_jobs_.front();
        prefetch_jobs_.pop_front();
      }
      prefetch_reader_cpus_.fetch_or(cpuBit());
      const uint64_t t0 = HtpProfile::nowUs();
      StagedExpert &st = job.first->experts[job.second];
      st.rc = readExpert(st, /*use_pool=*/false);
      HtpProfile &profile = HtpProfile::global();
      if (profile.level() != 0)
        profile.addPrefetchRead(HtpProfile::nowUs() - t0);
      {
        std::lock_guard<std::mutex> lock(prefetch_mutex_);
        ++job.first->done;
      }
      prefetch_done_cv_.notify_all();
    }
  }

  /** @brief Reads one QS4CX_WH weight from the model file into the arena
   *  at @a arena_dst: the nibbles, then the N scales and N column sums
   *  that follow them in the file, laid out as the DSP registry expects
   *  (f32 scales, then i32 column sums -- the file keeps the sums as f32,
   *  so they pass through a small buffer). @a e gets the shape only; its
   *  arrays stay empty, which is what tells the swap call the arrays are
   *  in the arena. whBytes(K, N) is the nibble half exactly:
   *  QS4CX_Tensor::size() counts N * ceil(K / 2) and K is a multiple of
   *  32 here. [plan 229] At @a w_bits 2 (QS2CX_WH) the codes are
   *  whBytes2(K, N) and four palette bytes sit between them and the
   *  scales: those go to @a e.pal, not to the arena, which keeps the
   *  slot's [codes][scales][sums] shape. @return 0, errno, or -1 at end
   *  of file. */
  /** @brief [#282] One layer's expert bytes held in RAM: file range
   *  [off, off + len) at @a p. */
  struct PinRange {
    int fd;
    uint64_t off, len;
    const uint8_t *p;
  };

  /** @brief [#282] NNTR_MOE_PIN: 0 / unset = misses pread the file (as
   *  before); 1 = each MoE layer's expert range is mmapped MAP_POPULATE
   *  (+ mlock, logged, not required) and a miss copies from that mapping;
   *  2 = the range is read into anonymous memory (not evictable as page
   *  cache is; swap aside). Once, at the first token's poolSync, so the
   *  cost lands in the first decode token and is logged as "pin:". */
  static int pinKnob() {
    if (weightPageCachePolicy() >= 0)
      return 0; // Pinning retains a second copy of every expert.
    static const int k = [] {
      const char *v = std::getenv("NNTR_MOE_PIN");
      return v != nullptr ? std::atoi(v) : 0;
    }();
    return k;
  }

  void pinExperts() {
    if (pinKnob() == 0 || pinned_)
      return;
    pinned_ = true;
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t total = 0;
    int lock_fail = 0;
    for (const std::vector<ExpertFileDesc> &d : pool_descs_) {
      if (d.empty())
        continue;
      uint64_t lo = ~uint64_t(0), hi = 0;
      for (const ExpertFileDesc &x : d) {
        const uint64_t pal = paletteBytes(x.w_bits);
        lo = std::min<uint64_t>(lo, std::min(x.off_gu, x.off_dn));
        hi = std::max<uint64_t>(
          hi,
          std::max<uint64_t>(x.off_gu + codeBytes(x.K, 2 * x.inter, x.w_bits) +
                               pal + tailPerCol(x) * 2 * x.inter,
                             x.off_dn + codeBytes(x.inter, x.N_out, x.w_bits) +
                               pal + tailPerCol(x) * x.N_out));
      }
      const uint64_t page = 4096, a = lo & ~(page - 1), len = hi - a;
      // Mode 2 on the S25 with all 30 layers (5.6 GB anonymous beside the
      // app's 3.3 GB) rebooted the phone: stop while MemAvailable keeps
      // NNTR_MOE_PIN_FLOOR_MIB (default 2048) above this layer's bytes.
      if (pinKnob() == 2 && memAvailableMib() < pinFloorMib() + (len >> 20)) {
        std::fprintf(stderr,
                     "[HTP] pin: stop at layer %zu: MemAvailable %zu MiB < "
                     "floor %zu + %llu\n",
                     pin_ranges_.size(), memAvailableMib(), pinFloorMib(),
                     (unsigned long long)(len >> 20));
        break;
      }
      void *m = MAP_FAILED;
      if (pinKnob() == 1) {
        m = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE | MAP_POPULATE, d[0].fd,
                   static_cast<off_t>(a));
        if (m != MAP_FAILED && ::mlock(m, len) != 0)
          ++lock_fail;
      } else {
        m = ::mmap(nullptr, len, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
        if (m != MAP_FAILED) {
          // four readers, 64 MiB pieces
          const uint64_t piece = uint64_t(64) << 20;
          std::atomic<uint64_t> next{0};
          std::atomic<int> bad{0};
          ThreadManager::Global().parallel_for(0, 4, [&](size_t) {
            for (uint64_t at; (at = next.fetch_add(piece)) < len;)
              if (preadAll(d[0].fd, static_cast<uint8_t *>(m) + at,
                           std::min(piece, len - at), a + at) != 0)
                bad = 1;
          });
          if (bad.load() != 0) {
            ::munmap(m, len);
            m = MAP_FAILED;
          }
        }
      }
      if (m == MAP_FAILED) {
        std::fprintf(stderr,
                     "[HTP] pin: layer range of %llu MiB failed (errno %d); "
                     "%zu layers pinned, the rest read the file\n",
                     (unsigned long long)(len >> 20), errno,
                     pin_ranges_.size());
        break;
      }
      pin_ranges_.push_back({d[0].fd, a, len, static_cast<const uint8_t *>(m)});
      total += len;
    }
    const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    std::fprintf(stderr,
                 "[HTP] pin: mode=%d layers=%zu mib=%llu ms=%.0f mlock_fail=%d "
                 "rss_mib=%zu mem_avail_mib=%zu\n",
                 pinKnob(), pin_ranges_.size(),
                 (unsigned long long)(total >> 20), ms, lock_fail,
                 static_cast<size_t>(rssKb() >> 10), memAvailableMib());
  }

  /** @brief [#282] /proc/meminfo's MemAvailable, MiB (0 if unreadable). */
  static size_t memAvailableMib() {
    FILE *f = std::fopen("/proc/meminfo", "r");
    if (f == nullptr)
      return 0;
    char line[128];
    size_t kb = 0;
    while (std::fgets(line, sizeof(line), f) != nullptr)
      if (std::sscanf(line, "MemAvailable: %zu kB", &kb) == 1)
        break;
    std::fclose(f);
    return kb >> 10;
  }

  /** @brief [#282] NNTR_MOE_PIN_FLOOR_MIB (default 2048). */
  static size_t pinFloorMib() {
    static const size_t v = [] {
      const char *e = std::getenv("NNTR_MOE_PIN_FLOOR_MIB");
      return e != nullptr ? std::strtoul(e, nullptr, 10) : size_t(2048);
    }();
    return v;
  }

  /** @brief [#282 B] NNTR_HTP_DROP_HOST_FC=1: at the first decode token,
   *  hand the OS back the CPU copies of the decode list's FC / DENSE_FFN /
   *  LM_HEAD weights (q4_pending_). The one-PD token reads its own arena
   *  copies (bindQ4m1, lmHeadHandles) and the prefill FCs their WH handles,
   *  registered by then; the CPU layers skip their GEMVs on a resident row.
   *  Keep the LM_HEAD for later prefills; unlike the other accelerated FCs,
   *  an untied head can still run on the CPU. A CPU fallback for a dropped
   *  FC would require re-reading its weight from the model file. */
  void dropHostFc(bool qs4cx_only = false) {
    // The memory profile releases sources individually after registration and
    // an alias check in Transformer. Do not override a retained CPU consumer.
    if (weightPageCachePolicy() >= 0)
      return;
    if (!htp_fixed::DROP_HOST_FC || host_fc_dropped_ || !e2e_)
      return;
    host_fc_dropped_ = !qs4cx_only;
    const size_t before = memAvailableMib();
    size_t bytes = 0;
    for (Q4Pending &p : q4_pending_) {
      if (p.canonical || p.keep_host)
        continue;
      if (p.dropped || (qs4cx_only && p.scale == nullptr))
        continue;
      p.dropped = true;
      const size_t codes =
        p.scale != nullptr
          ? static_cast<size_t>(p.K) * p.N / 2
          : static_cast<size_t>(p.K) / 32 * p.N * Q4_CPU_BLOCK_BYTES;
      releaseArmSource(const_cast<void *>(p.data), codes);
      bytes += codes;
      if (p.scale != nullptr) {
        releaseArmSource(const_cast<float *>(p.scale), sizeof(float) * p.N);
        bytes += sizeof(float) * p.N;
      }
    }
    std::fprintf(stderr,
                 "[HTP] drop host fc%s: weights=%zu mib=%zu mem_avail_mib "
                 "%zu -> %zu rss_mib=%zu\n",
                 qs4cx_only ? " (QS4CX, at load)" : "", q4_pending_.size(),
                 bytes >> 20, before, memAvailableMib(),
                 static_cast<size_t>(rssKb() >> 10));
    memLine("t1 after drop host fc");
  }

  /** @brief [#282] The pinned bytes of file range [off, off + len), or
   *  nullptr when they are not pinned. */
  const uint8_t *pinnedAt(int fd, uint64_t off, uint64_t len) const {
    for (const PinRange &r : pin_ranges_)
      if (r.fd == fd && off >= r.off && off + len <= r.off + r.len)
        return r.p + (off - r.off);
    return nullptr;
  }

  /** Native file tails are FP32; compact arena metadata follows native codes.
   */
  static int finishTl2Weight(uint32_t K, uint32_t N, const float *tail,
                             uint8_t *arena_dst, ArenaEntry &e) {
    const size_t capacity = htp_tl2_native_bytes(K, N);
    std::vector<int32_t> sums(N);
    for (uint32_t n = 0; n < N; ++n) {
      if (!std::isfinite(tail[n]) || tail[n] < 0.0f ||
          !std::isfinite(tail[N + n]) ||
          std::abs(tail[N + n]) > static_cast<float>(K))
        return EINVAL;
      sums[n] = static_cast<int32_t>(tail[N + n]);
    }
    std::memcpy(arena_dst + capacity, tail, N * sizeof(float));
    std::memcpy(arena_dst + capacity + N * sizeof(float), sums.data(),
                N * sizeof(int32_t));
    e.pal = {-1, 0, 1, 1};
    e.K = K;
    e.N = N;
    return 0;
  }

  /** @brief The record's tail after the codes and palette: N f32 scales, N
   *  f32 column sums and, for QS4CX_WH_AFFINE, N i32 offsets. The scales
   *  and the sums (as i32) go after the codes in the slot, where the DSP
   *  takes them from; offsets that are not all zero stay in @a e for the
   *  weight_set_offsets_batch call after the swap. */
  static void finishWhTail(const uint8_t *tail, uint32_t N, bool affine,
                           uint8_t *arena_dst, size_t nib, ArenaEntry &e) {
    const float *f = reinterpret_cast<const float *>(tail);
    std::vector<int32_t> colsum(N);
    for (uint32_t i = 0; i < N; ++i)
      colsum[i] = static_cast<int32_t>(f[N + i]);
    // Two sequential stores into the uncached mapping, never a read back.
    std::memcpy(arena_dst + nib, f, sizeof(float) * N);
    std::memcpy(arena_dst + nib + sizeof(float) * N, colsum.data(),
                sizeof(int32_t) * N);
    e.offsets.clear();
    if (affine) {
      const int32_t *o = reinterpret_cast<const int32_t *>(f + 2 * N);
      for (uint32_t i = 0; i < N; ++i)
        if (o[i] != 0) {
          e.offsets.assign(o, o + N);
          break;
        }
    }
  }

  int readWeight(int fd, uint64_t off, uint32_t K, uint32_t N, uint32_t w_bits,
                 bool affine, uint8_t *arena_dst, ArenaEntry &e,
                 bool use_pool) {
    const size_t nib = codeBytes(K, N, w_bits);
    const size_t tail_bytes = (affine ? 12u : 8u) * size_t(N);
    if (w_bits == HTP_TL2_WEIGHT_BITS) {
      if (!validNativeTl2Shape(K, N))
        return EINVAL;
      std::vector<float> tail(2u * N);
      const int rc = preadAll(fd, arena_dst, nib, off, tail.data(),
                              tail.size() * sizeof(float));
      if (rc)
        return rc;
      return finishTl2Weight(K, N, tail.data(), arena_dst, e);
    }
    // [#282] a pinned expert is a DDR copy, never a file read
    const size_t pal0 = paletteBytes(w_bits);
    if (const uint8_t *pin = pinnedAt(fd, off, nib + pal0 + tail_bytes)) {
      std::memcpy(arena_dst, pin, nib);
      e.pal.assign(pin + nib, pin + nib + pal0);
      finishWhTail(pin + nib + pal0, N, affine, arena_dst, nib, e);
      e.K = K;
      e.N = N;
      return 0;
    }
    // [doc 52 sections 10.7, 10.9] The nibble read is 82% of a miss and
    // capped near 4.9 GB/s by the uncached mapping whatever the thread
    // count: 8 slices bought 18% warm. Kept for the load and prefill
    // registration; the miss round (readMisses, #266: slices of one weight
    // share its readahead window cold) and the prefetch readers read
    // whole weights. Page-sized
    // slices, so no two threads write the same page; a worker cannot throw
    // across parallel_for, so each slice keeps its result.
    std::atomic<int> first_rc{0};
    const size_t pal = paletteBytes(w_bits);
    std::vector<uint8_t> after(pal + tail_bytes);
    auto slice_read = [&](uint8_t *dst, size_t len, uint64_t at) {
      const int rc = preadAll(fd, dst, len, at);
      if (rc == 0)
        return;
      int expected = 0;
      first_rc.compare_exchange_strong(expected, rc);
    };
    if (use_pool) {
      auto &tm = ThreadManager::Global();
      const size_t n_slices =
        std::min<size_t>(tm.getComputeThreadCount(), kExpertReadSlicesMax);
      const size_t slice =
        (((nib + n_slices - 1) / n_slices) + 4095u) & ~size_t(4095u);
      tm.parallel_for(0, n_slices, [&](size_t i) {
        const size_t b = i * slice;
        if (b < nib)
          slice_read(arena_dst + b, std::min(slice, nib - b), off + b);
      });
    } else {
      // Code and palette/scales are adjacent in the file. One IO avoids a
      // second small direct read on every miss, using the same bounded scratch
      // and separate destinations for uncached DMA and CPU metadata.
      first_rc = preadAll(fd, arena_dst, nib, off, after.data(), after.size());
    }
    if (first_rc.load() != 0)
      return first_rc.load();
    // The palette (none at four bits) and the tail in one read
    if (use_pool) {
      const int rc = preadAll(fd, after.data(), after.size(), off + nib);
      if (rc != 0)
        return rc;
    }
    e.pal.assign(after.begin(), after.begin() + pal);
    finishWhTail(after.data() + pal, N, affine, arena_dst, nib, e);
    e.K = K;
    e.N = N;
    return 0;
  }

  bool uses_device_weight(const void *data) const override {
    return e2e_ && std::any_of(q4_pending_.begin(), q4_pending_.end(),
                               [data](const Q4Pending &p) {
                                 return p.data == data && p.scale;
                               });
  }

  void discard_weight_source(void *data, size_t bytes) override {
    releaseArmSource(data, bytes);
    for (auto &p : q4_pending_)
      if (p.data == data)
        p.dropped = true;
  }

  /**
   * @brief Hands the OS back the pages of a weight that is now in the arena.
   *
   * The arena copy doubles this model's 3.9 GB of expert weights, and the
   * device refuses the fourth 1 GiB ION chunk long before the last layer
   * registers (measured: 1170 of 1408 weights, doc 46 section 36.3). The ARM
   * copy is dead the moment the memcpy above returns -- the matmul reads the
   * arena -- so this drops it. MADV_DONTNEED rather than free() because every
   * weight is a slice of ONE pool allocation that nobody may free piecemeal;
   * see whSourcePageRange for why the range is rounded inward and why the
   * pointer stays a valid handle_cache_ key afterwards.
   *
   * A read of these bytes after this point returns zeros, which would be a
   * silently wrong matmul. Nothing reads them: the only other reader is the
   * layer's per-expert CPU loop, and that path throws on a QS4CX_WH weight
   * (FloatTensor::dot) rather than computing with zeros. Set
   * NNTR_HTP_KEEP_ARM_WEIGHTS=1 to keep them anyway when bisecting a wrong
   * answer -- the model then needs both copies resident.
   */
  void releaseArmSource(void *src, size_t len) {
#if defined(__linux__)
    static const bool keep =
      std::getenv("NNTR_HTP_KEEP_ARM_WEIGHTS") != nullptr;
    if (keep)
      return;
    const long ps = sysconf(_SC_PAGESIZE);
    if (ps <= 0)
      return;
    uintptr_t begin = 0;
    size_t span = 0;
    whSourcePageRange(src, len, static_cast<size_t>(ps), &begin, &span);
    if (span != 0)
      madvise(reinterpret_cast<void *>(begin), span, MADV_DONTNEED);
#else
    (void)src;
    (void)len;
#endif
  }

  /**
   * @brief Registers a weight the offline quantizer already put in WH
   *        layout, by copying it into the arena as-is.
   *
   * No conversion, no DSP bake, no cache file: the bytes on disk are the
   * bytes the matmul reads (doc 46 section 35), and the DSP borrows them out
   * of the arena rather than keeping a heap copy. This is the whole of what
   * registration costs for a QS4CX_WH model.
   *
   * @param matAdata  whBytes(K, N) of WH nibbles
   * @param matAscale N scales, with N column sums immediately after them --
   *                  QS4CX_WH_Tensor's layout, which is why no separate
   *                  pointer is passed
   */
  uint32_t get_or_register_wh(void *matAdata, const float *matAscale,
                              remote_handle64 session, uint32_t K, uint32_t N,
                              uint32_t w_bits) {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    auto it = handle_cache_.find(matAdata);
    if (it != handle_cache_.end())
      return it->second;
    // A null scale is the layer's way of saying "this is a virtual expert's
    // key, the bytes are in your slot pool" (register_qs4cx_wh_expert_file).
    // Not finding it is a bookkeeping bug -- the caller's LRU let an expert
    // reach the call without acquiring it -- and reading the key as bytes
    // would compute a wrong answer quietly.
    if (matAscale == nullptr) {
      throw std::runtime_error(
        "QS4CX_WH expert weight (" + std::to_string(K) + "x" +
        std::to_string(N) +
        ") is not resident: register_qs4cx_wh_expert_file must precede the "
        "call for a virtual expert");
    }

    const uint64_t t_begin = HtpProfile::nowUs();
    // There is no other way to register these. The DSP-heap path bakes its
    // input, which would rearrange bytes that are already arranged, so a
    // model quantized to QS4CX_WH needs the arena and saying so plainly
    // beats computing a wrong answer quietly.
    if (!ensureArena(session)) {
      throw std::runtime_error(
        "QS4CX_WH weights need the DSP arena, which this device did not "
        "provide (no rpcmem_to_fd or no fastrpc_mmap). Quantize the model as "
        "QS4CX to use the conversion path instead.");
    }

    /** QS2CX_WH is half the bytes, with the four palette codes right after
       them -- QS2CX_WH_Tensor's layout, which is why the scale pointer the
       caller passed already skips them. */
    const uint32_t wh_len = static_cast<uint32_t>(codeBytes(K, N, w_bits));
    uint32_t chunk = 0, off = 0;
    // want = kArenaChunkMax, not wh_len: these weights arrive one at a time
    // with no total to size a chunk from, so every chunk is made full size
    // and filled by bump before the next. 3696 MiB of weights is 15 chunks,
    // against the 3840 MiB a clean process can map (doc 46 section 41) and
    // NNTR_HVX_MAX_ARENAS slots.
    if (!place(session, wh_len, kArenaChunkMax, &chunk, &off)) {
      // Everything the next step needs, because getting it costs a device
      // round: which of the four calls refused, how much was mapped when it
      // did, and this process's RSS -- which is how a run says whether
      // releaseArmSource gave the ARM copies back. Host RAM and DSP address
      // space fail the same way here and have nothing in common as fixes.
      throw std::runtime_error(
        "HTP arena: cannot register a " + std::to_string(K) + "x" +
        std::to_string(N) + " weight (" + std::to_string(wh_len >> 20) +
        " MiB). " +
        (arena_fail_.empty() ? "no chunk was attempted" : arena_fail_) +
        ". mapped=" + std::to_string(arenaBytes() >> 20) + " MiB in " +
        std::to_string(arena_chunks_.size()) +
        " chunks, RSS=" + std::to_string(rssKb() >> 10) + " MB");
    }
    std::memcpy(arena_chunks_[chunk].buf->data() + off, matAdata, wh_len);

    ArenaEntry e;
    e.chunk = chunk;
    e.off = off;
    e.K = K;
    e.N = N;
    e.w_scale.assign(matAscale, matAscale + N);
    e.colsum_w.resize(N);
    e.bias.assign(N, 0.0f); // FC weights carry no bias tensor
    // The column sums are the N floats after the scales, written there by
    // the quantizer. They are whole numbers small enough to be exact in
    // f32 -- a sum of at most K values in [-8, 7] -- so this conversion is
    // lossless, and the registry wants int32.
    const float *colsum_f = matAscale + N;
    for (uint32_t i = 0; i < N; ++i) {
      e.colsum_w[i] = static_cast<int32_t>(colsum_f[i]);
    }

    if (w_bits == 2u) {
      const int8_t *pal = reinterpret_cast<const int8_t *>(matAdata) + wh_len;
      e.pal.assign(pal, pal + paletteBytes(w_bits));
    }
    const uint32_t handle = registerFromArena(session, e, K, N, t_begin);
    if (handle == kNoHandle) {
      throw std::runtime_error("weight_register_u8i4_arena rejected a " +
                               std::to_string(K) + "x" + std::to_string(N) +
                               " WH weight");
    }
    // Only now: e.w_scale and e.colsum_w are already copies, so the source
    // buffer -- whose scales sit just past the nibbles -- has no reader left.
    releaseArmSource(matAdata, wh_len);
    // Still a usable key after releaseArmSource: the pages went, the mapping
    // did not, so no later allocation can take this address and turn a miss
    // into a hit on somebody else's weight.
    handle_cache_.emplace(matAdata, handle);
    return handle;
  }

  uint32_t get_or_register_qs4cx(void *matAdata, const float *matAscale,
                                 remote_handle64 session, uint32_t K,
                                 uint32_t N) {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    return get_or_register_qs4cx_unlocked(matAdata, matAscale, session, K, N);
  }

  /** @note Call with handle_mutex_ already held. */
  uint32_t get_or_register_qs4cx_unlocked(void *matAdata,
                                          const float *matAscale,
                                          remote_handle64 session, uint32_t K,
                                          uint32_t N) {
    auto it = handle_cache_.find(matAdata);
    if (it != handle_cache_.end())
      return it->second;

    const uint64_t t_begin = HtpProfile::nowUs();

    // Convert on the host, bake on the DSP, keep the baked bytes on the DSP
    // heap: the 1.89 GB ceiling Gate 0 measured, and the reason QS4CX_WH
    // exists. A QS4CX model that fits under it still runs this way; one that
    // does not has to be quantized as QS4CX_WH, which needs no conversion,
    // no bake, and no DSP heap at all (doc 46 section 35).
    HtpRpcBuffer q_w4_i8(static_cast<size_t>(K) * N);
    std::vector<float> w_scale(N);
    std::vector<int32_t> colsum_w(N);
    const uint64_t t_convert = HtpProfile::nowUs();
    htp_qs4cx_from_packed(matAdata, matAscale, K, N,
                          reinterpret_cast<int8_t *>(q_w4_i8.data()),
                          w_scale.data(), colsum_w.data());
    const uint64_t convert_us = HtpProfile::nowUs() - t_convert;

    return register_locked(matAdata, session, K, N, q_w4_i8, w_scale, colsum_w,
                           t_begin, convert_us);
  }

  /** @brief Sentinel for "the cache did not have it"; a real handle is a
   *  table slot index, and 0 is a valid one. */
  static constexpr uint32_t kNoHandle = 0xFFFFFFFFu;

  /**
   * @brief Decides, once, whether weights can be registered out of the arena.
   *
   * Nothing is mapped here: the first weight's place() makes the first chunk.
   * All this answers is whether this device gives out the two calls an arena
   * is made of -- an fd for a host buffer and a way to attach it.
   *
   * @return true if weights can be registered out of the arena. false means
   *         the ordinary DSP-heap path, which still works for QS4CX -- it
   *         just has the 1.89 GB ceiling Gate 0 measured -- and which
   *         get_or_register_wh refuses, because WH bytes have no other home.
   * @note   Call with handle_mutex_ already held.
   */
  bool ensureArena(remote_handle64 session) {
    (void)session;
    if (arena_state_ != ARENA_UNTRIED)
      return arena_state_ == ARENA_ON;

    const HtpRpcMemApi &api = HtpRpcMemApi::get();
    arena_state_ =
      (api.to_fd != nullptr && api.mmap != nullptr) ? ARENA_ON : ARENA_OFF;
    return arena_state_ == ARENA_ON;
  }

  /**
   * @brief Finds room for @a bytes in the arena, making a chunk if needed.
   *
   * @param want how much more is expected to follow, so a new chunk is
   *             sized for the run rather than for this one weight
   * @note  Call with handle_mutex_ already held.
   */
  bool place(remote_handle64 session, uint32_t bytes, size_t want,
             uint32_t *chunk, uint32_t *off) {
    return placeOn(session, CDSP_DOMAIN_ID, arena_chunks_, chunk_cap_, bytes,
                   want, chunk, off);
  }

  /** @brief place() on any chunk list: [#132 Part B E3] the E2E FC arena
   *  is the same path on its own chunks. */
  bool placeOn(remote_handle64 session, int domain,
               std::vector<ArenaChunk> &chunks, size_t &cap, uint32_t bytes,
               size_t want, uint32_t *chunk, uint32_t *off) {
    if (placeExistingOn(chunks, bytes, chunk, off))
      return true;
    if (!newChunkOn(session, domain, chunks, cap, bytes, want))
      return false;
    chunks.back().used = bytes;
    *chunk = static_cast<uint32_t>(chunks.size() - 1);
    *off = 0;
    return true;
  }

  /** @brief place() without the new chunk: room in a mapped chunk or
   *  nothing. What the FC slices use (get_or_register_fc), so that they
   *  never map address space the remaining heap registrations need. */
  bool placeExisting(uint32_t bytes, uint32_t *chunk, uint32_t *off) {
    return placeExistingOn(arena_chunks_, bytes, chunk, off);
  }
  bool placeExistingOn(std::vector<ArenaChunk> &chunks, uint32_t bytes,
                       uint32_t *chunk, uint32_t *off) {
    for (size_t c = 0; c < chunks.size(); ++c) {
      // 4 KB rather than the 512 the DSP requires: a weight that starts on
      // a page boundary is one the DMA never splits across a page for
      // alignment reasons alone, and the waste is under a part in 400.
      const size_t at = (chunks[c].used + 4095u) & ~size_t(4095u);
      if (at + bytes <= chunks[c].buf->size()) {
        chunks[c].used = at + bytes;
        *chunk = static_cast<uint32_t>(c);
        *off = static_cast<uint32_t>(at);
        return true;
      }
    }
    return false;
  }

  /** @brief Allocates one uncached ION buffer and attaches it to the DSP.
   *  @note  Call with handle_mutex_ already held. */
  /**
   * @brief Largest arena chunk to ask the DSP for.
   *
   * 256 MiB, and the size is the whole fix for the memory wall (doc 46
   * section 41). A clean process maps 3840 MiB in 256 MiB steps; the model
   * with 1 GiB chunks stopped at 3072 and was then refused at every size
   * down to 64 MiB. A flat byte budget cannot produce both. A bump
   * allocator that aligns each mapping to its own size can: 1 GiB chunks
   * land at 1G, 2G, 3G and push the cursor to the 4 GB end of the PD's
   * address space, so the 768 MiB below the first one is never reachable
   * again. 256 MiB chunks waste none of it.
   *
   * The model needs 3696 MiB of weights. That leaves ~144 MiB of margin,
   * which place() protects by scanning every chunk for room -- a 1.75 MiB
   * down weight fits tails a 3.5 MiB gate_up cannot.
   *
   * ponytail: if a bigger model needs more, the next thing to take is the
   * DSP heap, which answered AEE_ENOMEMORY after 182 MiB with 3840 mapped
   * -- heap and mappings share the one 4 GB space, so there is no second
   * budget to find, only this one to stop wasting.
   */
  static constexpr size_t kArenaChunkMax = size_t(256) << 20;
  /** @brief This process's resident set, in KB, or 0 if it cannot be read.
   *  The one number that says whether releaseArmSource actually gave the
   *  pages back -- MADV_DONTNEED never reports failure for a range it simply
   *  did not reclaim. */
  static unsigned long long rssKb() {
#if defined(__linux__)
    std::FILE *f = std::fopen("/proc/self/statm", "r");
    if (f == nullptr)
      return 0;
    unsigned long long total = 0, resident = 0;
    const int n = std::fscanf(f, "%llu %llu", &total, &resident);
    std::fclose(f);
    if (n != 2)
      return 0;
    return resident * 4u; // statm counts pages; 4 KB on every target here
#else
    return 0;
#endif
  }

  /** @brief Total bytes the arena has mapped to the DSP so far. */
  size_t arenaBytes() const {
    size_t total = 0;
    for (const ArenaChunk &c : arena_chunks_)
      total += c.buf->size();
    return total;
  }

  /**
   * @brief Attaches one arena chunk of exactly @a size bytes.
   *
   * @return true on success. On failure arena_fail_ says which of the four
   *         calls refused and with what, because they mean four unrelated
   *         things: rpcmem_alloc is the host's ION heap, fastrpc_mmap is the
   *         DSP's address space, arena_attach is the DSP's own arena table.
   */
  bool tryChunk(remote_handle64 session, size_t size) {
    return tryChunkOn(session, CDSP_DOMAIN_ID, arena_chunks_, size);
  }
  bool tryChunkOn(remote_handle64 session, int domain,
                  std::vector<ArenaChunk> &chunks, size_t size) {
    const unsigned long long rss_before = rssKb();
    // Uncached, because the DSP maps this once and the host keeps writing
    // into it afterwards -- with an uncached CPU mapping those writes reach
    // DDR with no flush to remember. The host never reads it back, so the
    // cost is on the write side only. The probe that passed Gate 0c wrote
    // BEFORE attaching and used a cached buffer, so this ordering is the
    // one thing section 34 rests on that the probe did not show; the
    // ArenaUncachedWriteAfterMap test is what answers it.
    auto buf = std::make_unique<HtpRpcBuffer>(size, HTP_RPC_FLAGS_UNCACHED);
    // [#132 Part B E5g] the buffer of a refused mapping lives until this
    // one exists, so the retry gets another fd: E5g's retries at 128 and 64
    // MiB after a refused 256 were refused AEE_EALREADY on the freed
    // buffer's fd number, which the driver still held
    last_failed_.reset();
    if (!buf->isIon()) {
      arena_fail_ = "rpcmem_alloc(" + std::to_string(size >> 20) +
                    " MiB) failed -- the HOST ION heap is out, so the ARM "
                    "copies are what to shrink";
      return false;
    }
    const int fd = buf->fd();
    if (fd < 0) {
      arena_fail_ = "rpcmem_to_fd returned no fd for an ION buffer";
      return false;
    }

    const HtpRpcMemApi &api = HtpRpcMemApi::get();
    // FASTRPC_MAP_FD, not 0: 0 is FASTRPC_MAP_STATIC, which is the driver's
    // mapping for a buffer passed as a call argument and is not tagged with
    // the fd, so HAP_mmap_get on the DSP refuses it. Three device runs went
    // on that (doc 46 section 32.10).
    const int merr = api.mmap(domain, fd, buf->data(), 0, size, FASTRPC_MAP_FD);
    if (merr != 0) {
      // the driver may keep the fd's entry after a refused map: drop it
      if (api.munmap != nullptr)
        api.munmap(domain, fd, buf->data(), size);
      last_failed_ = std::move(buf);
      arena_fail_ =
        "fastrpc_mmap(" + std::to_string(size >> 20) +
        " MiB) failed: err=" + std::to_string(merr) +
        " -- the host buffer exists, so this is the DSP side (its address "
        "space or the driver's mapping table), not host memory";
      return false;
    }

    uint32_t dsp_id = 0;
    const int aerr =
      nntr_hvx_arena_attach(session, fd, static_cast<uint32_t>(size), &dsp_id);
    if (aerr != AEE_SUCCESS) {
      // Hex as well as decimal: AEEStdErr offsets every code by 0x80000400
      // under __hexagon__, so 0x8000040d is the one that reads as a plain
      // AEE_EBADSTATE and a decimal print hides that.
      char err[80];
      std::snprintf(err, sizeof(err), "%zu MiB) failed: err=%d (0x%08x)",
                    size >> 20, aerr, static_cast<unsigned>(aerr));
      arena_fail_ = "nntr_hvx_arena_attach(" + std::string(err) +
                    " -- mapped, but the DSP refused it; NNTR_HVX_MAX_ARENAS "
                    "or HAP_mmap_get";
      if (api.munmap != nullptr)
        api.munmap(domain, fd, buf->data(), size);
      return false;
    }
    chunks.push_back(ArenaChunk{std::move(buf), dsp_id, 0});
    if (&chunks == &arena_chunks_ && chunks.size() == 1) {
      std::shared_ptr<std::vector<ArenaChunk>> a = s1_arena_;
      HtpBackend::global().atCloseLast(
        [a, session] { s1ArenaTeardown(session, *a); });
    }
    if (HtpProfile::global().level() != 0) {
      size_t total = 0;
      for (const ArenaChunk &c : chunks)
        total += c.buf->size();
      std::printf("[HTP] arena chunk %zu: %zu MiB, dsp_id=%u, mapped total "
                  "%zu MiB, RSS %llu -> %llu MB%s\n",
                  chunks.size() - 1, size >> 20, dsp_id, total >> 20,
                  rss_before >> 10, rssKb() >> 10,
                  domain == CDSP_DOMAIN_ID ? "" : " (S2)");
    }
    return true;
  }

  /**
   * @brief [#132 Part B E5i] The S1 arena's close hook, after the queues and
   *        the token drivers (atCloseLast: after every atClose hook): the
   *        skel releases the graph, the weights and every arena
   *        (arenas_release), then each chunk is fastrpc_munmap'd, every rc
   *        checked, before the session closes -- the S1Ceiling probe's
   *        order. E5f-E5h: about one app run in 20, A or E, left the next
   *        process one 256 MiB window short when the arena was left mapped
   *        to process exit (and its buffers rpcmem_free'd, still mapped, by
   *        whichever singleton died first).
   */
  static void s1ArenaTeardown(remote_handle64 session,
                              std::vector<ArenaChunk> &chunks) {
    if (chunks.empty())
      return;
    const HtpRpcMemApi &mem = HtpRpcMemApi::get();
    uint32_t put = 0, unmapped = 0;
    size_t bytes = 0;
    const int rc = nntr_hvx_arenas_release(session, &put);
    for (auto c = chunks.rbegin(); c != chunks.rend(); ++c) {
      bytes += c->buf->size();
      const int u = mem.munmap != nullptr
                      ? mem.munmap(CDSP_DOMAIN_ID, c->buf->fd(), c->buf->data(),
                                   c->buf->size())
                      : -1;
      if (u == 0)
        ++unmapped;
      else
        std::fprintf(stderr,
                     "[HTP] arena: fastrpc_munmap(chunk dsp_id=%u, %zu MiB) "
                     "failed: 0x%x\n",
                     c->dsp_id, c->buf->size() >> 20, static_cast<unsigned>(u));
    }
    std::fprintf(stderr,
                 "[HTP] arena: chunks unmapped %u/%zu mib=%zu (release rc=0x%x "
                 "put=%u)\n",
                 unmapped, chunks.size(), bytes >> 20,
                 static_cast<unsigned>(rc), put);
    chunks.clear(); // rpcmem_free, each buffer unmapped first
  }

  /**
   * @brief Adds a chunk, halving the request until one is accepted.
   *
   * The DSP refused a fourth 1 GiB mapping after three were accepted (doc 46
   * section 39), so this ceiling is measured in whole GiB and the last GiB
   * before it is left on the floor. Halving turns a ceiling that happens to
   * fall between two chunk sizes into one the arena can fill up to, and the
   * size the run settles on is itself the measurement -- there is no other
   * way to learn where the DSP's address space actually ends.
   *
   * chunk_cap_ remembers what was refused, so a size known not to fit is not
   * asked for again on the next weight. A failure to map is not free: the
   * host buffer is allocated and released each time.
   */
  bool newChunk(remote_handle64 session, uint32_t bytes, size_t want) {
    return newChunkOn(session, CDSP_DOMAIN_ID, arena_chunks_, chunk_cap_, bytes,
                      want);
  }
  bool newChunkOn(remote_handle64 session, int domain,
                  std::vector<ArenaChunk> &chunks, size_t &cap, uint32_t bytes,
                  size_t want) {
    static constexpr size_t kGrain = size_t(64) << 20;
    /** Below this a chunk holds too few weights to be worth an arena slot,
     *  and the DSP's table is not unbounded. [#132 Part B E5g] 16 MiB (was
     *  64): the tail of the MoE arena, ~112 MiB above 14 x 256, may have
     *  to go in pieces on a boot whose last 256 MiB window is taken. */
    static constexpr size_t kFloor = size_t(16) << 20;

    size_t size = (std::max(want, size_t(bytes)) + kGrain - 1) & ~(kGrain - 1);
    size = std::min(std::max(size, kGrain), cap);
    if (size < bytes) {
      arena_fail_ = "a weight is larger than a whole chunk";
      return false; // not this model
    }

    while (true) {
      arena_fail_.clear();
      if (tryChunkOn(session, domain, chunks, size))
        return true;
      const size_t half = size / 2;
      if (half < kFloor || half < bytes)
        return false; // arena_fail_ holds the last refusal, which is the one
      if (HtpProfile::global().level() != 0) {
        std::printf("[HTP] arena: %zu MiB refused, retrying at %zu MiB (%s)\n",
                    size >> 20, half >> 20, arena_fail_.c_str());
      }
      size = half;
      cap = size;
    }
  }

  /** @brief Registers a weight whose bytes are already in the arena.
   *  @return the handle, or kNoHandle if the DSP refused it.
   *  @param  t_begin caller's start time, or 0 on the miss path -- there
   *          register_locked already counted this weight and counting it
   *          again would report two registrations for one weight.
   *  @note   Call with handle_mutex_ already held. */
  uint32_t registerFromArena(remote_handle64 session, const ArenaEntry &e,
                             uint32_t K, uint32_t N, uint64_t t_begin) {
    // The file named a shape; the caller wants one. A directory holding
    // another model's files would otherwise register the wrong weight,
    // since the path is keyed on a hash of bytes this process never read.
    if (e.K != K || e.N != N)
      return kNoHandle;

    uint32_t handle = 0;
    const uint64_t t_rpc = HtpProfile::nowUs();
    const int err =
      e.pal.empty()
        ? nntr_hvx_weight_register_u8i4_arena(
            session, K, N, arena_chunks_[e.chunk].dsp_id, e.off,
            e.w_scale.data(), static_cast<int>(N), e.colsum_w.data(),
            static_cast<int>(N), e.bias.data(), static_cast<int>(e.bias.size()),
            &handle)
        : nntr_hvx_weight_register_u2i4_arena(
            session, K, N, arena_chunks_[e.chunk].dsp_id, e.off, e.pal.data(),
            4, e.w_scale.data(), static_cast<int>(N), e.colsum_w.data(),
            static_cast<int>(N), e.bias.data(), static_cast<int>(N), &handle);
    const uint64_t rpc_us = HtpProfile::nowUs() - t_rpc;
    if (err != AEE_SUCCESS)
      return kNoHandle;

    HtpProfile &profile = HtpProfile::global();
    if (profile.level() != 0 && t_begin != 0)
      profile.addRegister(HtpProfile::nowUs() - t_begin, /*convert_us=*/0,
                          rpc_us, /*ion=*/true);
    return handle;
  }

  /** @brief Register a converted weight and cache its handle.
   *  @note  Call with handle_mutex_ already held.
   *  @param t_begin   caller's start timestamp, for the profile's
   *                   registration total (unused when profiling is off)
   *  @param convert_us microseconds the caller spent in htp_qs4cx_from_* */
  uint32_t register_locked(void *key, remote_handle64 session, uint32_t K,
                           uint32_t N, HtpRpcBuffer &q_w4_i8,
                           std::vector<float> &w_scale,
                           std::vector<int32_t> &colsum_w, uint64_t t_begin,
                           uint64_t convert_us,
                           const std::vector<float> *bias_override = nullptr) {
    std::vector<float> bias =
      bias_override ? *bias_override : std::vector<float>(N, 0.0f);

    uint32_t handle = 0;
    const uint64_t t_rpc = HtpProfile::nowUs();
    const int err = nntr_hvx_weight_register_u8i4(
      session, K, N, reinterpret_cast<int8_t *>(q_w4_i8.data()),
      static_cast<int>(q_w4_i8.size()), w_scale.data(), static_cast<int>(N),
      colsum_w.data(), static_cast<int>(N), bias.data(),
      static_cast<int>(bias.size()), &handle);
    const uint64_t rpc_us = HtpProfile::nowUs() - t_rpc;
    if (err != AEE_SUCCESS) {
      throw std::runtime_error("nntr_hvx_weight_register_u8i4 failed: err=" +
                               std::to_string(err));
    }
    HtpProfile &profile = HtpProfile::global();
    if (profile.level() != 0) {
      profile.addRegister(HtpProfile::nowUs() - t_begin, convert_us, rpc_us,
                          q_w4_i8.isIon());
    }

    // Kept resident for the process lifetime -- Stage 6 (residency, see
    // 40_moe_ffn_htp_task.md) is what has to bound this table's size and
    // add release for the full-checkpoint case; not needed yet.
    handle_cache_.emplace(key, handle);
    return handle;
  }

  std::mutex handle_mutex_;
  std::unordered_map<const void *, uint32_t> handle_cache_;
  /** FC weights by data pointer -> their handle(s); see get_or_register_fc.
   *  Values are never erased or moved, so the references it hands out stay
   *  valid (std::unordered_map keeps node addresses across rehash). */
  std::unordered_map<const void *, FcHandles> fc_cache_;
  /** Dense FFNs by their up weight's pointer; see get_or_register_dense. */
  std::unordered_map<const void *, DenseHandles> dense_cache_;
  /** Conv blocks by their in_proj's pointer; see get_or_register_conv. */
  std::unordered_map<const void *, ConvHandles> conv_cache_;
  /** [#225] The FC WH sidecar (set_fc_wh_file): its fd, path and index by
   *  fcWhKey, and the bytes its images took in the arena and on the heap. */
  int fcwh_fd_ = -1;
  std::string fcwh_name_;
  std::unordered_map<uint64_t, FcWhEntry> fcwh_;
  /** Main-file WH records keyed by their tensor data pointer. */
  std::unordered_map<const void *, FcWhEntry> direct_wh_;
  size_t fcwh_arena_bytes_ = 0, fcwh_heap_bytes_ = 0;
  /** [#225 PR 2] WH bytes of the sidecar's images not registered yet, plus
   *  placeExistingOn's 4 KiB alignment for four handles an image (its most:
   *  a dense FFN's chunks). */
  size_t fcwhLeft() const {
    size_t all = 0;
    for (const auto &e : fcwh_)
      all += whBytes(e.second.K, e.second.N) + 4u * 4096u;
    for (const auto &e : direct_wh_)
      all +=
        whBytes(e.second.K, e.second.N) +
        ((e.second.N + fcSliceCols(e.second.K) - 1u) / fcSliceCols(e.second.K) +
         4u) *
          4096u;
    const size_t done = fcwh_arena_bytes_ + fcwh_heap_bytes_;
    return all > done ? all - done : 0u;
  }
  /** Handles registered from the sidecar: the FC slices plus the dense
   *  chunks' two each. */
  size_t fcwhHandles() const {
    if (fcwh_fd_ < 0)
      return 0;
    size_t n = 0;
    for (const auto &f : fc_cache_)
      n += f.second.handles.size();
    for (const auto &d : dense_cache_)
      n += d.second.h_gu.size() + d.second.h_dn.size();
    return n;
  }

  /** S1's arena, shared with its close hook (s1ArenaTeardown): the two
   *  singletons' destruction order is not fixed. */
  std::shared_ptr<std::vector<ArenaChunk>> s1_arena_ =
    std::make_shared<std::vector<ArenaChunk>>();
  std::vector<ArenaChunk> &arena_chunks_ = *s1_arena_;
  /** Expert slots given back by release_qs4cx_wh_expert, each of its
   *  expert's size; a miss takes one of its size before it bumps a chunk,
   *  so after the load the chunk count moves only when a size runs out. */
  std::vector<ExpertSlot> free_expert_slots_;
  std::unordered_map<const void *, ExpertResident> experts_;
  /** The direct prefetch's spare slots' size (initPoolPrefetch). */
  uint32_t prefetch_slot_bytes_ = 0;
  /** reserve_qs4cx_wh_expert_slots's count, and slots placed so far. */
  size_t expert_slots_wanted_ = 0, expert_slots_placed_ = 0;
  /** Slices of one expert weight's file read: 3.5 MiB over 8 is 448 KiB a
   *  pread, past which the per-call cost stops paying for the split. */
  static constexpr size_t kExpertReadSlicesMax = 8;
  /** Experts read in the background between prefetch _begin and _end, and
   *  the threads reading them -- one expert per thread at a time. Four:
   *  the write rate into the arena stops scaling well before that (doc 52
   *  section 10.9), and the ARM side has its own work after the call. */
  std::vector<StagedExpert> prefetch_;
  /** One _begin's experts; done counts the ones read. */
  struct PrefetchBatch {
    std::vector<StagedExpert> experts;
    size_t done = 0;
  };
  /** Queued batches, oldest first, and their unread experts, all under
   *  prefetch_mutex_. A batch's address is stable until _end takes it. */
  std::mutex prefetch_mutex_;
  std::condition_variable prefetch_cv_, prefetch_done_cv_;
  std::deque<std::unique_ptr<PrefetchBatch>> prefetch_batches_;
  std::deque<std::pair<PrefetchBatch *, size_t>> prefetch_jobs_;
  bool prefetch_stop_ = false;
  std::vector<std::thread> prefetch_readers_;
  std::atomic<uint32_t> prefetch_reader_cpus_{0};
  uint32_t prefetch_caller_cpus_ = 0;

  /**
   * @brief [doc 52 section 10.18] Measurement switches for the prefetch
   *        readers, read once. Section 10.13 found the readers make almost
   *        no progress while the layer call runs and the call's host side
   *        grows by 7 ms; section 10.19 traced that to the readers
   *        sharing the caller's core. Not defaults.
   *  - NNTR_MOE_PREFETCH_READERS=<n>: reader threads (6)
   *  - NNTR_MOE_PREFETCH_CPUS=<a,b,..>: pin the readers to these cores
   *    (default: every core but the caller's)
   */
  struct PrefetchKnobs {
    size_t readers = 6;
    std::vector<int> cpus;
  };
  static const PrefetchKnobs &prefetchKnobs() {
    static const PrefetchKnobs k = [] {
      PrefetchKnobs r;
      if (const char *v = std::getenv("NNTR_MOE_PREFETCH_READERS"))
        r.readers = std::max<size_t>(1, std::strtoul(v, nullptr, 10));
      if (const char *v = std::getenv("NNTR_MOE_PREFETCH_CPUS")) {
        for (const char *p = v; *p != '\0';) {
          char *end = nullptr;
          const long c = std::strtol(p, &end, 10);
          if (end == p)
            break;
          if (c >= 0 && c < 32)
            r.cpus.push_back(static_cast<int>(c));
          p = (*end == ',') ? end + 1 : end;
        }
      }
      return r;
    }();
    return k;
  }
  /** Match the prefetch worker policy without inheriting a caller pinned
   *  to a single compute core. An explicit prefetch CPU list is retained. */
  static std::vector<int> backgroundCpus(uint32_t caller) {
    auto cpus = prefetchKnobs().cpus;
#if defined(__linux__)
    if (cpus.empty() && caller) {
      const long n = sysconf(_SC_NPROCESSORS_CONF);
      for (int c = 0; c < n && c < 32; ++c)
        if ((caller & (1u << c)) == 0)
          cpus.push_back(c);
    }
#else
    (void)caller;
#endif
    return cpus;
  }
  /** @brief This thread's core as a bit, 0 where that is not known. */
  static uint32_t cpuBit() {
#if defined(__linux__)
    const int c = sched_getcpu();
    return (c >= 0 && c < 32) ? (1u << c) : 0u;
#else
    return 0u;
#endif
  }
  static void pinToCpus(const std::vector<int> &cpus) {
#if defined(__linux__)
    if (cpus.empty())
      return;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : cpus)
      CPU_SET(c, &set);
    sched_setaffinity(0, sizeof(set), &set);
#else
    (void)cpus;
#endif
  }
  enum ArenaState { ARENA_UNTRIED, ARENA_ON, ARENA_OFF };
  ArenaState arena_state_ = ARENA_UNTRIED;
  /** Why the last newChunk refused, in words, for the throw that follows. */
  std::string arena_fail_;
  /** Largest chunk still worth asking for; only ever shrinks. */
  size_t chunk_cap_ = kArenaChunkMax;
  /** [#132 Part B E5g] The last refused chunk's buffer, freed after the
   *  next one is allocated (tryChunkOn). */
  std::unique_ptr<HtpRpcBuffer> last_failed_;

  // [#132 Part B E5i] Unmapped by the close hook s1ArenaTeardown (it was
  // left mapped to process exit until E5h). ponytail: a process that loads
  // and unloads models still keeps every arena to its end; a reload would
  // need the same hook at model teardown.

  // ION-backed activation/output scratch, reused across calls, one buffer
  // per size class (stage) -- see invokeLayer's comment. Guarded by the
  // same mutex that serializes every call into the one HTP session.
  std::mutex invoke_mutex_;
  std::once_flag moe_opts_once_; /**< sendMoeOptsOnce */
  std::atomic<bool> moe_opts_sent_{false};
  /** [PR 4428] set_moe_hadamard_signs's vectors until the session has them */
  std::mutex had_signs_mutex_;
  std::vector<float> had_signs_in_, had_signs_down_;
  // [#85] The decode op list from set_decode_graph_desc, its MoE ops in
  // list order, how many have their handles bound (first-pass call order),
  // the op each layer's first gate_up handle belongs to, and whether
  // graph_init has run on the session.
  std::mutex graph_mutex_;
  std::vector<uint32_t> graph_words_;
  std::vector<uint32_t> moe_ops_;
  size_t moe_bound_ = 0;
  std::unordered_map<uint32_t, uint32_t> moe_op_by_handle_;
  /** [plan 201 S1] per MoE op (moe_ops_ order) its EXPERTS table as the
   *  f32 words graph_set_param carries: h_gu[0..E) then h_dn[0..E) */
  std::vector<std::vector<float>> moe_tables_;
  /** [plan 201 S1] The expert pool behind the per-token entry
   *  (set_decode_moe_experts): per MoE layer its experts, where each
   *  gate_up key sits (layer, expert), the layer's policy, and whether a
   *  prefill touched the pool since the last token (poolSync). */
  std::vector<std::vector<ExpertFileDesc>> pool_descs_;
  std::vector<PinRange> pin_ranges_; /**< [#282] NNTR_MOE_PIN */
  bool pinned_ = false;
  bool host_fc_dropped_ = false; /**< [#282 B] NNTR_HTP_DROP_HOST_FC */
  /** [#282 F] the first token's setup timers */
  uint64_t seed_rpc_us_ = 0;
  unsigned seed_calls_ = 0;
  unsigned seed_kvq_ = 0; /**< [#289] of them, seeded on the DSP from kvq */
  bool t1_printed_ = false;
  /** [#282 D] the kv_cache_q handles registered and not released */
  std::mutex kvq_mutex_;
  std::set<uint32_t> kvq_handles_;
  /** [#282 B] the prefetch readers (NNTR_MOE_PREFETCH): their queue, and
   *  the counts the close line prints. The issued keys are the pool
   *  server's alone: a later miss of one counts as useful. */
  std::mutex pf_mu_;
  std::condition_variable pf_cv_;
  std::deque<ExpertFileDesc> pf_q_;
  std::vector<std::thread> pf_threads_;
  bool pf_stop_ = false;
  std::unordered_set<const void *> pf_issued_keys_;
  // static: e2eTeardown (static) prints them; one backend per process
  static inline uint64_t pf_hints_ = 0, pf_issued_ = 0, pf_useful_ = 0;
  static inline std::atomic<uint64_t> pf_done_{0};
  std::unordered_map<const void *, std::pair<uint32_t, uint32_t>> pool_where_;
  ExpertPoolFn pool_fn_;
  std::function<void(size_t)> pool_grow_; /**< [#4416] set_decode_pool_grow */
  std::vector<uint8_t> pool_answered_;    /**< [#289] layers a miss round served
                                             this token (handle_mutex_) */
  bool pool_dirty_ = true;
  std::shared_ptr<HtpExpertPrefetch<StagedExpert>> pool_prefetch_;
  std::unique_ptr<HtpExpertReadPool> demand_read_pool_;
  std::shared_ptr<HtpExpertPrefetch<PredictionJob>> cpu_predictor_;
  std::vector<PredictionRouter> prediction_routers_;
  std::unique_ptr<HtpPrefetchConfidence> prediction_confidence_;
  std::unique_ptr<PreparedTl2> prepared_tl2_;
  bool tl2_decode_ = false;

  /** Decode replaces QS2 residents lazily on misses; prefill restores any
   * TL2 residents before their HMX use. Prefetch is joined at the boundary.
   * StagedExpert captures each read's format, independently of this flag. */
  void switchTl2Phase(bool native) {
    // A matched QS2 control reads the same validated artifact and uses the
    // same host reader, but keeps its prefill representation for decode.
    const char *decode = std::getenv("NNTR_HTP_TL2_DECODE");
    if (decode && std::strcmp(decode, "0") == 0)
      native = false;
    if (!prepared_tl2_ || tl2_decode_ == native)
      return;
    if (cpu_predictor_)
      cpu_predictor_->drain();
    if (pool_prefetch_)
      pool_prefetch_->drain();
    while (!prefetch_qs4cx_wh_experts_end().empty()) {
    }
    const uint64_t start = HtpProfile::nowUs();
    prepared_tl2_->discardAsync(!native);
    if (native) {
      // Existing QS2 residents are already symmetric and remain usable.
      // Decode misses replace them with TL2 as the LRFU cache turns over;
      // avoid a synchronous reread of the whole 960-expert pool at token 1.
      tl2_decode_ = true;
      std::fprintf(stderr, "[HTP-TL2] phase=decode lazy=1 ms=%.1f\n",
                   (HtpProfile::nowUs() - start) / 1000.0);
      return;
    }
    const auto session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    std::lock_guard<std::mutex> lock(handle_mutex_);
    std::vector<StagedExpert> staged;
    std::vector<uint32_t> handles;
    for (const auto &item : experts_) {
      const ExpertResident &r = item.second;
      if (r.d.w_bits != 2u || !r.tl2)
        continue;
      staged.push_back({r.d,
                        r.slot,
                        arena_chunks_[r.slot.chunk].buf->data() + r.slot.off,
                        {},
                        {},
                        0,
                        native});
      handles.push_back(r.h_gu);
      handles.push_back(r.h_dn);
    }
    readMisses(staged);
    for (const auto &st : staged)
      throwPread(st.rc, "TL2 phase handoff");
    if (nntr_hvx_weights_tl2(session, handles.data(), handles.size(), native))
      throw std::runtime_error("TL2 phase registration failed");
    for (auto &item : experts_)
      item.second.tl2 = false;
    tl2_decode_ = native;
    std::fprintf(stderr, "[HTP-TL2] phase=%s experts=%zu ms=%.1f\n",
                 native ? "decode" : "prefill", staged.size(),
                 (HtpProfile::nowUs() - start) / 1000.0);
  }
  bool graph_inited_ = false;
  bool graph_short_warned_ = false;
  // [#130] The stretch tables of section 3.2: per op the maximal resident
  // run it belongs to, per kind the resident ops in list order (what the
  // hooks' counters index), per op whether its parameter is bound and,
  // for CONV1D_GATE, the position its DSP state is valid for; per
  // attention ordinal the DSP cache's length (the kernel's kv_len,
  // mirrored). pending_in_ is the row of the stretch's first op (QK_NORM,
  // ADD, or a MOE op with its routing) until the stretch's last hook runs
  // it (#132's role dispatch). Touched by the model's thread only.
  uint32_t resident_mask_ = 0;
  std::vector<uint32_t> stretch_start_, stretch_end_, attn_ordinal_;
  std::vector<uint32_t> kind_ops_[HTP_OP_KIND_N];
  uint32_t kind_next_[HTP_OP_KIND_N] = {0};
  uint32_t cur_pos_ = HTP_GRAPH_NO_OP;
  bool row_bound_ = false; /**< every MoE op bound when this row began */
  std::vector<uint8_t> param_bound_;
  /** [#4416 D6] the distinct RoPE tables bound: (ROPE op, its bytes) */
  std::vector<std::pair<uint32_t, std::vector<float>>> rope_tables_;
  std::vector<uint32_t> conv_next_pos_;
  std::vector<uint32_t> kv_len_;
  bool attn_registered_ = false;
  uint32_t attn_window_rows_ = 0; /**< set_decode_kv_window, 0: none */
  /** [plan 201 S4] per attention ordinal its op, its session cache (0, or
   *  1 for the second attn_m1_register: another (n_kv, gqa, head_dim)) and
   *  its ordinal within that cache (hexkl_graph_init's) */
  std::vector<uint32_t> attn_op_, attn_cache_, attn_cache_ord_;
  /** [plan 201 S4] the f32 parameters the model handed at load
   *  (set_decode_graph_param), bound after graph_init */
  struct LoadParam {
    uint32_t op, which;
    const float *data;
    uint32_t n;
  };
  std::vector<LoadParam> load_params_;
  uint32_t seed_ordinal_ = HTP_GRAPH_NO_OP;
  uint32_t pending_op_ = HTP_GRAPH_NO_OP;
  std::vector<float> pending_in_;
  // [#132] the routing a MOE op that starts a stretch was handed
  std::vector<unsigned int> pending_ri_, pending_rc_;
  std::vector<float> pending_rw_;
  // [#132 Part B] the model's Q4_0 weights for the Q4M1 kinds, in list
  // order, until bindQ4m1; the handles registered for them
  struct Q4Pending {
    const void *data;
    uint32_t K, N;
    bool canonical;
    const float *scale = nullptr; /**< [#260] QS4CX: N scales; Q4_0: null */
    bool dropped = false;         /**< [#4416 mem] CPU copy handed back */
    bool keep_host = false;       /**< CPU prefill still reads this weight */
  };
  std::vector<Q4Pending> q4_pending_;

  std::vector<uint32_t> q4m1_handles_;
  /** @brief The lm_head's slices by weight (lmHeadHandles); ok false: it
   *  could not be placed and stays on the CPU. */
  struct LmHeadSlices {
    bool ok = false;
    std::vector<uint32_t> h;
  };
  std::map<const void *, LmHeadSlices> lm_head_;
  struct SharedLmHead {
    unsigned K = 0, N = 0;
    bool wh = false; /**< h are u8i4 WH handles (a QS4CX head), else Q4M1 */
    std::vector<uint32_t>
      h; // Borrowed from e2e_st_->q4m1, never released twice.
  };
  std::map<const void *, SharedLmHead> shared_lm_heads_;
  std::string lmhead_wh_path_; /**< set_lm_head_wh_cache; empty: no cache */
  uint32_t first_resident_op_ = HTP_GRAPH_NO_OP;
  uint64_t fwd_calls_ = 0;      /**< nntr_hvx_forward* calls */
  uint64_t fwd_tokens_ = 0;     /**< of them at the first resident op */
  uint64_t cpu_fc_skipped_ = 0; /**< [#132 Part B] decode_row_resident's
                                     true answers: CPU FC calls skipped */
  StagingPool act_pool_;
  StagingPool out_pool_;
  StagingPool rope_pool_;     /**< the fused projection call's RoPE rows */
  StagingPool x2_pool_;       /**< the MoE call's post-FFN second addend */
  StagingPool attn_q_pool_;   /**< the row-blocked attention's f32 Q */
  StagingPool attn_out_pool_; /**< and its f32 output */
  /** @brief ION copies of the MoE routers' gate weights, by address. */
  std::map<const float *, std::unique_ptr<HtpRpcBuffer>> router_w_;
  std::map<const float *, uint32_t> router_u8i8_; /**< gate -> u8i8 handle */
  /** [#141] The M==1 MoE call's dspqueue; null until the first such call
      unless NNTR_HTP_DSPQ=0. */
  std::shared_ptr<DspqMoe> dspq_;
  /** [#132 Part B E3, #211] The E2E path's state, shared with its
   *  HtpBackend close hook (which runs after this object is gone). */
  struct E2eState {
    remote_handle64 h1 = 0;
    std::vector<ArenaChunk> arena; /**< the FC set's chunks (Q4M1, attached) */
    size_t arena_cap = kArenaChunkMax;
    size_t attach_bytes = 0;
    std::vector<uint32_t> q4m1;         /**< the FC set's q4m1_attach handles */
    std::unique_ptr<HtpRpcBuffer> mbox; /**< the page S1 maps (miss lines) */
    bool mbox1 = false, drv1 = false;
    std::shared_ptr<DspqMoe> q1; /**< S1's queue (dspq_) */
    uint32_t tok = 0;
    uint64_t tokens = 0, hops = 0, wait_us = 0, pcyc = 0, id_checked = 0,
             id_mismatch = 0;
    uint64_t wall_us = 0, wall_pcyc = 0,
             token_us = 0; /**< the DSP's; token_us: the ARM's round trip */
    /** [#194 L0] the hops' own latency (0 since #211, the wire keeps it),
     *  the ARM's time inside tokenForward, and between one tokenForward's
     *  end and the next one's start (the layer walk, sampler, tokenizer,
     *  print; arm_n intervals) */
    uint64_t hop_us = 0, fwd_us = 0, arm_us = 0, arm_n = 0, last_exit_us = 0;
    uint64_t pool_idle_us = 0, pool_predict_drain_us = 0,
             pool_prefetch_drain_us = 0;
    /** [#194 L0] wake split on the shared system counter: the ARM's post to
     *  the DSP thread's read, the DSP's write to the ARM's read, the DSP's
     *  packet handling around its token (out - in) */
    int64_t disp_us = 0, ret_us = 0, inout_us = 0;
    uint64_t kind[HTP_OP_KIND_N] = {0};
    /** [#267 L0] the ops' wall us per kind, the pcycles over the waits */
    uint64_t kind_us[HTP_OP_KIND_N] = {0}, miss_pcyc = 0;
    /** [#261] NNTR_HTP_PROFILE: the ATTN_M1 phase words per KV cache,
     *  summed over attn_tokens tokens */
    uint64_t attn_calls[2] = {0, 0}, attn_tokens = 0;
    uint64_t attn_prof[2][HTP_DSPQ_ATTN_PROF_WORDS] = {};
    /** [#267 L3] MOE kernel calls, and the one-at-a-time form's count */
    uint64_t moe_calls = 0, moe_calls_1x = 0;
    uint32_t moe_ops = 0, spin_us = 0;
    /** [plan 201 S1] the pool: experts S1 loaded and its waits, the miss
     *  rounds served and the ARM's time on them */
    uint64_t misses = 0, miss_us = 0, pool_rounds = 0, pool_read_us = 0;
    uint64_t pool_pair_reads =
      0; /**< adjacent TL2 demand pairs read together */
    /** [#282 C] a miss round's parts, us summed: post -> the server
     *  noticed it, its work before the reads, the reads, S1's present
     *  experts computed meanwhile, the answer written -> S1 saw it, S1's
     *  rebinds */
    uint64_t rt_notice_us = 0, rt_pre_us = 0, rt_read_us = 0, rt_overlap_us = 0,
             rt_anslat_us = 0, rt_rebind_us = 0;
    uint64_t pgpgin0 = 0; /**< [#216] vmstatPgpginKib() at driver on */
  };
  /** @brief Mailbox allocation, including the optional prediction row. */
  static constexpr size_t kMboxBytes = size_t(64) << 10;
  static_assert(HTP_MBOX_PRED_INPUT + HTP_MBOX_PRED_BYTES <= kMboxBytes,
                "Prediction input must fit the shared mailbox");
  /** [plan 201 S1] poolServe's thread and its handshake with tokenForward:
   *  active while a token is in flight, idle once the thread stopped
   *  polling; the loads of the last answer until S1 fills their pairs. */
  struct PoolServer {
    std::thread th;
    std::mutex mu;
    std::condition_variable cv;
    std::atomic<bool> active{false};
    bool stop = false, idle = false;
    uint32_t tok = 0;
    std::vector<StagedExpert> pending;
    /** [#216] this answer's victims, asked back after it */
    std::vector<std::pair<ExpertFileDesc, bool>> advice;
    std::exception_ptr err;
  };
  std::unique_ptr<PoolServer> pool_srv_;
  bool e2e_ = false; /**< NNTR_HTP_E2E=1 and a description set */
  std::shared_ptr<E2eState> e2e_st_;
  bool q4m1_bound_ = false; /**< the Q4M1 weights registered at load (E3) */
  size_t q4m1_left_ = 0;    /**< E2E FC arena bytes still to place */
  bool want_logits_ = true; /**< set_decode_logits */
  std::vector<uint32_t> ban_, ban_sent_; /**< set_decode_ban, the LM_BAN */
  bool have_id_ = false;                 /**< take_decode_token_id */
  uint64_t prompt_tokens_ = 0;           /**< staging lifetime per prompt */
  uint32_t pending_id_ = 0;
  /** invokeConvBlock's conv_w in and state out, one small ION buffer. */
  std::unique_ptr<HtpRpcBuffer> conv_buf_;

  // invokeLayerU8In's scratch: the AH-packed activation (ION-backed, same
  // reasoning as act_pool_/out_pool_) and the small per-row scale/zp arrays
  // it produces alongside it (plain heap -- a few KB at most, not worth
  // ION's pin/map bookkeeping).
  std::unique_ptr<HtpRpcBuffer> act_ah_buf_;
  std::vector<float> act_scale_scratch_;
  std::vector<int32_t> act_zp_scratch_;

#ifndef NNTR_HTP_INPROC
  // The attention entries below have no in-process stand-in (their
  // kernels need HMX fp16 and intrinsics hvx_emu does not emulate), so
  // the host twin leaves them on the CPU defaults.
public:
  bool supports_sdpa_fp16_kvcache() const override {
    return HtpBackend::global().enabled();
  }

  // The KV cache the op below reads: in rpcmem, FastRPC hands the used
  // range to the DSP by mapping, not by copying it every call.
  void *alloc_shared(size_t bytes) override {
    return HtpBackend::global().alloc_shared(bytes);
  }
  void free_shared(void *block) override {
    HtpBackend::global().free_shared(block);
  }
  void drop_shared(void *block) override {
    HtpBackend::global().drop_shared(block);
  }

  // --- quantized (int8 / int4) KV cache resident on the DSP ---

  bool supports_kv_cache_q() const override {
    return HtpBackend::global().enabled();
  }

  int kv_cache_q_register(unsigned int kind, unsigned int max_rows,
                          unsigned int n_head_kv,
                          unsigned int head_dim) override {
    HtpBackend &hb = HtpBackend::global();
    // head_dim up to 512 is the registry's limit; the column-blocked
    // kernel below checks its own 256 at call time.
    if (!hb.enabled() || kind > 1 || max_rows == 0 || max_rows > 0xFFFFu ||
        n_head_kv == 0 || head_dim == 0 || (head_dim % 32) != 0 ||
        head_dim > 512) {
      return -1;
    }
    uint32_t h = 0;
    const int err =
      nntr_hvx_kv_register_q(static_cast<remote_handle64>(hb.handle()), kind,
                             max_rows, n_head_kv, head_dim, &h);
    if (err != AEE_SUCCESS) {
      ml_logw("HTP quantized KV cache: register failed: 0x%x", err);
      return -1;
    }
    {
      std::lock_guard<std::mutex> lock(kvq_mutex_);
      kvq_handles_.insert(h); // [#282 D]
    }
    return static_cast<int>(h);
  }

  bool kv_cache_q_append(int handle, unsigned int row0, unsigned int n_rows,
                         unsigned int kv_stride, const uint16_t *k_rows,
                         const uint16_t *v_rows) override {
    HtpBackend &hb = HtpBackend::global();
    if (!hb.enabled() || handle < 0 || n_rows == 0 || kv_stride == 0) {
      return false;
    }
    const int len = static_cast<int>(n_rows * kv_stride);
    const int err = nntr_hvx_kv_append_q(
      static_cast<remote_handle64>(hb.handle()), static_cast<uint32_t>(handle),
      row0, k_rows, len, v_rows, len);
    if (err != AEE_SUCCESS) {
      ml_logw("HTP quantized KV cache: append failed: 0x%x", err);
      return false;
    }
    return true;
  }

  void kv_cache_q_release(int handle) override {
    HtpBackend &hb = HtpBackend::global();
    {
      std::lock_guard<std::mutex> lock(kvq_mutex_);
      if (handle < 0 || kvq_handles_.erase(static_cast<uint32_t>(handle)) == 0)
        return; // [#282 D] released already (NNTR_HTP_E2E_FREE_KVQ)
    }
    if (hb.enabled() && handle >= 0) {
      nntr_hvx_kv_release_q(static_cast<remote_handle64>(hb.handle()),
                            static_cast<uint32_t>(handle));
    }
  }

  bool sdpa_q_kvcache(int handle, unsigned int append_row0,
                      unsigned int append_rows, unsigned int kv_stride,
                      const uint16_t *k_rows, const uint16_t *v_rows,
                      const float *q, unsigned int q_stride, unsigned int n_q,
                      unsigned int cache_from, unsigned int cache_to,
                      unsigned int n_head_q, unsigned int n_head_kv,
                      unsigned int head_dim, unsigned int window, float softcap,
                      const float *sinks, float *out,
                      unsigned int out_stride) override {
    HtpBackend &hb = HtpBackend::global();
    if (!hb.enabled() || handle < 0 || n_q == 0 || n_head_kv == 0 ||
        (n_head_q % n_head_kv) != 0 || head_dim == 0 || (head_dim % 32) != 0 ||
        head_dim > 256 || cache_to < cache_from + n_q || cache_to > 0xFFFFu ||
        q_stride != n_head_q * head_dim || out_stride != n_head_q * head_dim ||
        (append_rows != 0 &&
         (kv_stride != n_head_kv * head_dim || !k_rows || !v_rows))) {
      return false;
    }
    const remote_handle64 h = static_cast<remote_handle64>(hb.handle());
    const int q_len = static_cast<int>(n_q * n_head_q * head_dim);
    const int sinks_len = sinks ? static_cast<int>(n_head_q) : 0;
    const int rows_len = static_cast<int>(append_rows * kv_stride);
    uint32_t stats[8] = {0};
    const int64_t t0 = now_us();
    const int err = nntr_hvx_attn_q_step(
      h, static_cast<uint32_t>(handle), append_row0, k_rows, rows_len, v_rows,
      rows_len, n_q, cache_from, cache_to, n_head_q, window, softcap, q, q_len,
      sinks, sinks_len, out, q_len, stats, 8);
    if (attn_trace_enabled()) {
      ml_logi("HTP attn trace q: handle=%d n_q=%u cache=%u..%u append=%u "
              "err=0x%x wall_us=%lld dsp_us=%u append_us=%u attn_us=%u "
              "quant_us=%u stage_us=%u bake_us=%u",
              handle, n_q, cache_from, cache_to, append_rows, err,
              static_cast<long long>(now_us() - t0), stats[3], stats[0],
              stats[1], stats[4], stats[5], stats[6]);
    }
    if (err != AEE_SUCCESS) {
      ml_logw("HTP quantized attention step failed: 0x%x; CPU fallback", err);
      return false;
    }
    return true;
  }

  // --- the row-blocked kernel over a fixed-scale int8 cache ---

  bool supports_kv_cache_q2() const override {
    return HtpBackend::global().enabled();
  }

  bool kv_cache_q_set_fixed_scales(int handle, unsigned int n_head_kv,
                                   unsigned int head_dim, const float *s_k,
                                   const float *s_v) override {
    HtpBackend &hb = HtpBackend::global();
    if (!hb.enabled() || handle < 0 || !s_k || !s_v || n_head_kv == 0 ||
        head_dim == 0) {
      return false;
    }
    const int err = nntr_hvx_kv_set_fixed_scales_q(
      static_cast<remote_handle64>(hb.handle()), static_cast<uint32_t>(handle),
      s_k, static_cast<int>(n_head_kv), s_v,
      static_cast<int>(n_head_kv * head_dim));
    if (err != AEE_SUCCESS) {
      ml_logw("HTP quantized KV cache: fixed scales failed: 0x%x", err);
      return false;
    }
    return true;
  }

  bool sdpa_q2_kvcache(int handle, unsigned int append_row0,
                       unsigned int append_rows, unsigned int kv_stride,
                       const uint16_t *k_rows, const uint16_t *v_rows,
                       const uint16_t *q, const float *q_enc,
                       unsigned int q_stride, unsigned int n_q,
                       unsigned int cache_from, unsigned int cache_to,
                       unsigned int n_head_q, unsigned int n_head_kv,
                       unsigned int head_dim, unsigned int window,
                       uint16_t *out, const float *out_enc,
                       unsigned int out_stride) override {
    HtpBackend &hb = HtpBackend::global();
    if (!hb.enabled() || handle < 0 || n_q == 0 || n_head_kv == 0 ||
        (n_head_q % n_head_kv) != 0 || head_dim == 0 || (head_dim % 32) != 0 ||
        head_dim > 512 || cache_to < cache_from + n_q || cache_to > 0xFFFFu ||
        q_stride != n_head_q * head_dim || out_stride != n_head_q * head_dim ||
        !q || !q_enc || !out || !out_enc ||
        (append_rows != 0 &&
         (kv_stride != n_head_kv * head_dim || !k_rows || !v_rows))) {
      return false;
    }
    const remote_handle64 h = static_cast<remote_handle64>(hb.handle());
    const int q_len = static_cast<int>(n_q * n_head_q * head_dim);
    const int rows_len = static_cast<int>(append_rows * kv_stride);
    const int enc_len = static_cast<int>(2 * n_head_q);
    uint32_t stats[12] = {0};
    const int64_t t0 = now_us();
    // Q and the output through ION, as the FC path stages its rows: from
    // plain heap FastRPC moved 2 x 16.8 MB a 1024-row layer, 7-12 ms of an
    // 18-24 ms call around 11.5 ms on the DSP (doc 57 section 9.17). A
    // memcpy each way here is ~1 ms (u16 since PR 4343's a16 path: half).
    const size_t q_bytes = static_cast<size_t>(q_len) * sizeof(uint16_t);
    uint16_t *q_ion =
      reinterpret_cast<uint16_t *>(stage(attn_q_pool_, q_bytes).data());
    uint16_t *out_ion =
      reinterpret_cast<uint16_t *>(stage(attn_out_pool_, q_bytes).data());
    stagedMemcpy(q_ion, q, q_bytes);
    const int err = nntr_hvx_attn_q2_step(
      h, static_cast<uint32_t>(handle), append_row0, k_rows, rows_len, v_rows,
      rows_len, n_q, cache_from, cache_to, n_head_q, window, q_ion, q_len,
      q_enc, enc_len, out_enc, enc_len, out_ion, q_len, stats, 12);
    if (err == AEE_SUCCESS) {
      stagedMemcpy(out, out_ion, q_bytes);
    }
    if (attn_trace_enabled()) {
      // stats (nntr_hvx_attn_q2_step): append, kernel, -, total, then the
      // append's quant/stage/bake, rows appended, the kernel's qk, softmax,
      // pv and worker wait, in microseconds. wall - total is transport.
      ml_logi("HTP attn trace q2: n_q=%u cache=%u..%u err=0x%x wall_us=%lld "
              "dsp_us: append=%u (quant=%u stage=%u bake=%u rows=%u) "
              "kernel=%u (qk=%u softmax=%u pv=%u wait=%u) total=%u",
              n_q, cache_from, cache_to, err,
              static_cast<long long>(now_us() - t0), stats[0], stats[4],
              stats[5], stats[6], stats[7], stats[1], stats[8], stats[9],
              stats[10], stats[11], stats[3]);
    }
    if (err != AEE_SUCCESS) {
      ml_logw("HTP row-blocked attention step failed: 0x%x; CPU fallback", err);
      return false;
    }
    return true;
  }

  bool sdpa_q2_kvcache_f32(int handle, unsigned int append_row0,
                           unsigned int append_rows, unsigned int kv_stride,
                           const uint16_t *k_rows, const uint16_t *v_rows,
                           const float *q, const float *q_inv,
                           const float *q_enc, unsigned int q_stride,
                           unsigned int n_q, unsigned int cache_from,
                           unsigned int cache_to, unsigned int n_head_q,
                           unsigned int n_head_kv, unsigned int head_dim,
                           unsigned int window, float *out,
                           uint16_t *out_u16_copy, const float *out_enc,
                           unsigned int out_stride) override {
    HtpBackend &hb = HtpBackend::global();
    if (!hb.enabled() || handle < 0 || n_q == 0 || n_head_kv == 0 ||
        (n_head_q % n_head_kv) != 0 || head_dim == 0 || (head_dim % 64) != 0 ||
        head_dim > 512 || cache_to < cache_from + n_q || cache_to > 0xFFFFu ||
        q_stride != n_head_q * head_dim || out_stride != n_head_q * head_dim ||
        !q || !q_inv || !q_enc || !out || !out_enc ||
        (append_rows != 0 &&
         (kv_stride != n_head_kv * head_dim || !k_rows || !v_rows))) {
      return false;
    }
    const remote_handle64 h = static_cast<remote_handle64>(hb.handle());
    const int q_len = static_cast<int>(n_q * n_head_q * head_dim);
    const int rows_len = static_cast<int>(append_rows * kv_stride);
    const int enc_len = static_cast<int>(2 * n_head_q);
    uint32_t stats[12] = {0};
    const int64_t t0 = now_us();
    // Q as the qkv call left it (f32, straight from that call's staging
    // buffer when it still holds it; the DSP's pool quantizes), the u16
    // context left in its buffer for the o-proj: no ARM pass either way.
    const size_t q_bytes = static_cast<size_t>(q_len) * sizeof(float);
    const float *q_ion = static_cast<const float *>(ionAlias(q, q_bytes));
    if (!q_ion) {
      float *b = reinterpret_cast<float *>(stage(attn_q_pool_, q_bytes).data());
      stagedMemcpy(b, q, q_bytes);
      q_ion = b;
    }
    const size_t out_bytes = static_cast<size_t>(q_len) * sizeof(uint16_t);
    uint16_t *out_ion =
      reinterpret_cast<uint16_t *>(stage(attn_out_pool_, out_bytes).data());
    if (attn_out16_.host) {
      static bool warned = false;
      if (!warned) {
        warned = true;
        ml_logw("HTP attention: the previous u16 context was never taken by "
                "an o-proj on the accelerator (its f32 tensor was not "
                "written); check attn_out_u16 against the o-proj's engine");
      }
    }
    const int err = nntr_hvx_attn_q2_step_fq(
      h, static_cast<uint32_t>(handle), append_row0, k_rows, rows_len, v_rows,
      rows_len, n_q, cache_from, cache_to, n_head_q, window, q_ion, q_len,
      q_inv, static_cast<int>(n_head_q), q_enc, enc_len, out_enc, enc_len,
      out_ion, q_len, stats, 12);
    if (err == AEE_SUCCESS) {
      attn_out16_.host = out;
      attn_out16_.ion = out_ion;
      attn_out16_.rows = n_q;
      attn_out16_.width = out_stride;
      attn_out16_.head_dim = head_dim;
      attn_out16_.enc.assign(out_enc, out_enc + enc_len);
      if (out_u16_copy)
        stagedMemcpy(out_u16_copy, out_ion, out_bytes);
    }
    if (attn_trace_enabled()) {
      ml_logi("HTP attn trace q2f: n_q=%u cache=%u..%u err=0x%x wall_us=%lld "
              "dsp_us: append=%u (quant=%u stage=%u bake=%u rows=%u) q16=%u "
              "kernel=%u (qk=%u softmax=%u pv=%u wait=%u) total=%u",
              n_q, cache_from, cache_to, err,
              static_cast<long long>(now_us() - t0), stats[0], stats[4],
              stats[5], stats[6], stats[7], stats[2], stats[1], stats[8],
              stats[9], stats[10], stats[11], stats[3]);
    }
    if (err != AEE_SUCCESS) {
      // a skel without the entry answers with an error: the caller converts
      static bool warned = false;
      if (!warned) {
        warned = true;
        ml_logw("HTP f32-Q attention step failed: 0x%x; the u16 path", err);
      }
      return false;
    }
    return true;
  }

  bool sdpa_q2_kvcache_f32_calib(
    int handle, unsigned int append_row0, unsigned int append_rows,
    unsigned int kv_stride, const uint16_t *k_rows, const uint16_t *v_rows,
    const float *q, unsigned int q_stride, unsigned int n_q,
    unsigned int cache_from, unsigned int cache_to, unsigned int n_head_q,
    unsigned int n_head_kv, unsigned int head_dim, unsigned int window,
    float *out, float *s_k, float *s_v, float *q_enc, float *out_enc,
    unsigned int out_stride) override {
    HtpBackend &hb = HtpBackend::global();
    if (!hb.enabled() || handle < 0 || n_q == 0 || n_head_kv == 0 ||
        (n_head_q % n_head_kv) != 0 || head_dim == 0 || (head_dim % 64) != 0 ||
        head_dim > 512 || cache_to < cache_from + n_q || cache_to > 0xFFFFu ||
        q_stride != n_head_q * head_dim || out_stride != n_head_q * head_dim ||
        !q || !out || !s_k || !s_v || !q_enc || !out_enc ||
        kv_stride != n_head_kv * head_dim ||
        (append_rows != 0 && (!k_rows || !v_rows))) {
      return false;
    }
    const remote_handle64 h = static_cast<remote_handle64>(hb.handle());
    const int q_len = static_cast<int>(n_q * n_head_q * head_dim);
    const int rows_len = static_cast<int>(append_rows * kv_stride);
    const int enc_len = static_cast<int>(2 * n_head_q);
    uint32_t stats[12] = {0};
    const int64_t t0 = now_us();
    const size_t q_bytes = static_cast<size_t>(q_len) * sizeof(float);
    const float *q_ion = static_cast<const float *>(ionAlias(q, q_bytes));
    if (!q_ion) {
      float *b = reinterpret_cast<float *>(stage(attn_q_pool_, q_bytes).data());
      stagedMemcpy(b, q, q_bytes);
      q_ion = b;
    }
    const size_t out_bytes = static_cast<size_t>(q_len) * sizeof(uint16_t);
    uint16_t *out_ion =
      reinterpret_cast<uint16_t *>(stage(attn_out_pool_, out_bytes).data());
    std::vector<float> calib(n_head_kv + kv_stride + 4 * n_head_q);
    const int err = nntr_hvx_attn_q2_step_fqc(
      h, static_cast<uint32_t>(handle), append_row0, k_rows, rows_len, v_rows,
      rows_len, n_q, cache_from, cache_to, n_head_q, window, q_ion, q_len,
      calib.data(), static_cast<int>(calib.size()), out_ion, q_len, stats, 12);
    if (err == AEE_SUCCESS) {
      const float *p = calib.data();
      std::copy(p, p + n_head_kv, s_k);
      p += n_head_kv;
      std::copy(p, p + kv_stride, s_v);
      p += kv_stride;
      std::copy(p, p + enc_len, q_enc);
      p += enc_len;
      std::copy(p, p + enc_len, out_enc);
      attn_out16_.host = out;
      attn_out16_.ion = out_ion;
      attn_out16_.rows = n_q;
      attn_out16_.width = out_stride;
      attn_out16_.head_dim = head_dim;
      attn_out16_.enc.assign(out_enc, out_enc + enc_len);
    }
    if (attn_trace_enabled()) {
      ml_logi("HTP attn trace q2fc: n_q=%u cache=%u..%u err=0x%x wall_us=%lld "
              "dsp_us: append=%u (quant=%u stage=%u bake=%u rows=%u) calib=%u "
              "kernel=%u (qk=%u softmax=%u pv=%u wait=%u) total=%u",
              n_q, cache_from, cache_to, err,
              static_cast<long long>(now_us() - t0), stats[0], stats[4],
              stats[5], stats[6], stats[7], stats[2], stats[1], stats[8],
              stats[9], stats[10], stats[11], stats[3]);
    }
    if (err != AEE_SUCCESS) {
      static bool warned = false;
      if (!warned) {
        warned = true;
        ml_logw("HTP calibrating f32-Q attention step failed: 0x%x; the "
                "layer calibrates",
                err);
      }
      return false;
    }
    return true;
  }

  bool sdpa_fp16_kvcache(const float *q, unsigned int q_stride,
                         const uint16_t *k_cache, const uint16_t *v_cache,
                         unsigned int kv_stride, unsigned int n_q,
                         unsigned int cache_from, unsigned int cache_to,
                         unsigned int n_head_q, unsigned int n_head_kv,
                         unsigned int head_dim, unsigned int window,
                         float softcap, const float *sinks, float *out,
                         unsigned int out_stride) override {
    HtpBackend &hb = HtpBackend::global();
    if (!hb.enabled() || n_q == 0 || n_head_kv == 0 ||
        (n_head_q % n_head_kv) != 0 || head_dim == 0 || (head_dim % 32) != 0 ||
        head_dim > 512 || cache_to < cache_from + n_q || cache_to > 0xFFFFu) {
      return false;
    }
    // The IDL takes dense [n_q][n_head_q*head_dim] and
    // [cache_to][n_head_kv*head_dim]; anything strided differently is the
    // CPU's.
    if (q_stride != n_head_q * head_dim || out_stride != n_head_q * head_dim ||
        kv_stride != n_head_kv * head_dim) {
      return false;
    }
    const remote_handle64 h = static_cast<remote_handle64>(hb.handle());
    const int q_len = static_cast<int>(n_q * n_head_q * head_dim);
    const int kv_len = static_cast<int>(cache_to * n_head_kv * head_dim);
    const int sinks_len = sinks ? static_cast<int>(n_head_q) : 0;
    uint32_t stats[8] = {0};

    int err;
    const int64_t t0 = now_us();
    if (n_q < kDecodeMaxRows && head_dim <= kDecodeMaxHeadDim) {
      err = nntr_hvx_attn_f16_decode(h, n_q, cache_from, cache_to, n_head_q,
                                     n_head_kv, head_dim, window, softcap, q,
                                     q_len, k_cache, kv_len, v_cache, kv_len,
                                     sinks, sinks_len, out, q_len, stats, 8);
    } else {
      err = nntr_hvx_attn_f16_prefill(
        h, n_q, cache_from, cache_to, n_head_q, n_head_kv, head_dim, window,
        /*br=*/0, /*bc=*/0, softcap, q, q_len, k_cache, kv_len, v_cache, kv_len,
        sinks, sinks_len, out, q_len, stats, 8);
    }
    if (attn_trace_enabled()) {
      // stats: qprep, dma, tile, qk, softmax, pv, store microseconds and the
      // block count (nntr_hvx_attn_f16.c).
      ml_logi("HTP attn trace f16: n_q=%u cache=%u..%u err=0x%x wall_us=%lld "
              "dsp_us: qprep=%u dma=%u tile=%u qk=%u softmax=%u pv=%u "
              "store=%u blocks=%u",
              n_q, cache_from, cache_to, err,
              static_cast<long long>(now_us() - t0), stats[0], stats[1],
              stats[2], stats[3], stats[4], stats[5], stats[6], stats[7]);
    }
    if (err != AEE_SUCCESS) {
      // AEE_EUNSUPPORTED is the skel saying this part has no fp16 HMX;
      // anything else is a transport or shape failure. Either way the
      // caller recomputes on the CPU, so this is a warning, not an error.
      ml_logw("HTP attention (%s) failed: 0x%x; CPU fallback for this call",
              n_q < kDecodeMaxRows ? "decode" : "prefill", err);
      return false;
    }
    return true;
  }
#endif /* NNTR_HTP_INPROC */
};

static std::unique_ptr<HtpComputeOps> &htpOpsInstance() {
  static std::unique_ptr<HtpComputeOps> instance;
  return instance;
}

static thread_local bool htp_decode_active = false;

bool htpDecodeActive() noexcept { return htp_decode_active; }

bool htpSetDecodeActive(bool active) noexcept {
  const bool previous = htp_decode_active;
  htp_decode_active = active;
  return previous;
}

ComputeOps *get_htp_ops() {
  static std::mutex creation_mutex;
  std::lock_guard<std::mutex> lock(creation_mutex);
  auto &instance = htpOpsInstance();
  if (!instance)
    instance = std::make_unique<HtpComputeOps>();
  return instance.get();
}

void htp_prepare_runtime() {
  auto &backend = HtpBackend::global();
  backend.open();
  if (!backend.enabled())
    throw std::runtime_error("HTP session could not be initialized");
  HtpContext::Global().getContextData()->setComputeOps(get_htp_ops());
}

void htp_begin_prompt() {
  if (htpOpsInstance())
    htpOpsInstance()->beginPrompt();
}

void htp_stop_workers() {
  if (htpOpsInstance())
    htpOpsInstance()->stopWorkers();
}

void htp_release_runtime() {
  // No layers may retain this ops table. The API drops them before release.
  HtpContext::Global().getContextData()->setComputeOps(get_cpu_ops());
  htpOpsInstance().reset();
  HtpBackend::global().close();
  moeOptsSent().store(false);
  moeGeglu().store(false);
}

} // namespace nntrainer

#endif // ENABLE_HEXKL
