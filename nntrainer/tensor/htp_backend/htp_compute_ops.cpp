// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   htp_compute_ops.cpp
 * @date   18 Jun 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  HTP (Hexagon/HMX) ComputeOps: the CPU table with the attention
 *         op routed to the DSP.
 *
 * Compiled only when ENABLE_HEXKL is defined.
 *
 * HtpComputeOps is CpuComputeOps plus the accelerator ops the DSP skel
 * implements. Today that is sdpa_fp16_kvcache, dispatched over the FastRPC
 * session HtpBackend owns to attn_f16_decode (a few query rows, pure HVX)
 * or attn_f16_prefill (HMX flash attention), the same gate llama.cpp's
 * HTP backend uses. Every other op is inherited unchanged, so an
 * engine="htp" model runs correctly with exactly this one op accelerated.
 */

#ifdef ENABLE_HEXKL

#include <compute_ops.h>
#include <cpu_ops_table.h>
#include <htp_backend.h>
#include <nntrainer_log.h>

#include <chrono>
#include <cstdlib>
#include <vector>

#include <AEEStdErr.h>
#include <remote.h>

#include <nntr_hvx.h>

namespace nntrainer {

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

class HtpComputeOps : public CpuComputeOps {
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

  // --- quantized (int8 / int4) KV cache resident on the DSP ---

  bool supports_kv_cache_q() const override {
    return HtpBackend::global().enabled();
  }

  int kv_cache_q_register(unsigned int kind, unsigned int max_rows,
                          unsigned int n_head_kv,
                          unsigned int head_dim) override {
    HtpBackend &hb = HtpBackend::global();
    if (!hb.enabled() || kind > 1 || max_rows == 0 || max_rows > 0xFFFFu ||
        n_head_kv == 0 || head_dim == 0 || (head_dim % 32) != 0 ||
        head_dim > 256) {
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
        head_dim > 256 || cache_to < cache_from + n_q || cache_to > 0xFFFFu) {
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
};

} // namespace

ComputeOps *get_htp_ops() {
  static HtpComputeOps ops;
  return &ops;
}

} // namespace nntrainer

#endif // ENABLE_HEXKL
