// SPDX-License-Identifier: Apache-2.0
/**
 * @file htp_profile.h
 * @brief HTP profiling counters and reporting, independent of dispatch.
 * @author NNTrainer contributors
 * @bug No known bugs except for NYI items
 */
#ifndef NNTR_HTP_PROFILE_H
#define NNTR_HTP_PROFILE_H
#include "htp_runtime_config.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <hmx/hexkl_dma_trace.h>
#include <htp_graph_desc.h>
#include <htp_profile_wire.h>
#include <htp_rpcmem.h>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>

namespace nntrainer {
/**
 * @brief Per-stage timing for the HTP path. Off unless htp.profile is set.
 *
 * htp.profile=1 times the three host-side stages a weight goes through:
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
    auto &storage = instance();
    auto *profile = storage.current.load(std::memory_order_acquire);
    if (!profile) {
      std::lock_guard<std::mutex> lock(storage.mutex);
      profile = storage.current.load(std::memory_order_relaxed);
      if (!profile) {
        storage.owner.reset(new HtpProfile());
        profile = storage.owner.get();
        storage.current.store(profile, std::memory_order_release);
      }
    }
    return *profile;
  }

  /** All runtime workers must have stopped before resetting the report. */
  static void reset() {
    auto &storage = instance();
    std::lock_guard<std::mutex> lock(storage.mutex);
    storage.current.store(nullptr, std::memory_order_release);
    storage.owner.reset();
  }

  int level() const { return level_; }

  static uint64_t nowUs() {
    return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch())
        .count());
  }

  /** The per-token entry's own line under the layer rows: how many
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
  /** One resident op's pcycle bracket, by kind (level >= 2). */
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

  /** Count calls by whole layer shape (K, N_out); rows counts input tokens,
   *  not expert slots. Keep scatter and RPC staging separate from acc_us
   *  so profiling distinguishes those costs. */
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
    // What this call staged through, for the staging: line. The
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

  /** @brief Whether this call's per-descriptor trace should be
   *  dumped: the first htp.dma_trace_calls (default 3) timed calls of each
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
   * @brief Prints one traced call, one line per push and per wait,
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
  struct Storage {
    std::mutex mutex;
    std::unique_ptr<HtpProfile> owner;
    std::atomic<HtpProfile *> current{nullptr};
  };
  static Storage &instance() {
    static Storage storage;
    return storage;
  }

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
    /** The DMA ring trace's per-call numbers, summed over calls
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
    /** MoE layer call only: the ION staging class the call rode
        (stage() in HtpComputeOps), whether both landed on ION, and the
        bytes of the non-ION in-args the stub hands the driver (the 48-byte
        primitive block plus the five routing/handle sequences). Printed
        as the staging: line so the transport column has its inventory
        beside it instead of in a plan. */
    size_t stage_act_bytes = 0;
    size_t stage_out_bytes = 0;
    bool stage_ion = true;
    uint64_t in_arg_bytes = 0;
    /** Of calls, how many rode the dspqueue; then in_arg_bytes is
        the request message, not the stub's non-ION in-args. */
    uint64_t dspq_calls = 0;
  };

  HtpProfile() {
    level_ = static_cast<int>(htpRuntimeConfig().htp.profile);
    dma_trace_calls_ = static_cast<int>(htpRuntimeConfig().htp.dma_trace_calls);
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
        // GEMV reads the arena directly, so there is no ring wait rate.
        // swiglu is summed lane time; swiglu / mm estimates busy lanes.
        // Weight bandwidth is unavailable because this bucket lacks inter.
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
        // How the call used the ring. busy is a bracket, not a point
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
        // The MoE call's per-call transport inventory. The driver's
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
          // The request message is copied into the queue's shared
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
      // How far ahead the readers were: the
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
      // The per-expert read on a reader thread: page
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
  uint64_t graph_calls_ = 0; /**< addInvokeForward */
  uint64_t graph_ops_ = 0;
  uint64_t graph_dsp_us_ = 0;
  uint64_t graph_op_pcyc_ = 0;
  uint32_t graph_resident_ = 0; /**< the description's mask */
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

} // namespace nntrainer
#endif
