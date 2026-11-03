// SPDX-License-Identifier: Apache-2.0
/** @file htp_runtime_config.h
 * @brief Typed configuration for one HTP runtime and its expert policy.
 * @author NNTrainer contributors
 * @bug No known bugs
 */
#ifndef NNTR_HTP_RUNTIME_CONFIG_H
#define NNTR_HTP_RUNTIME_CONFIG_H

#include <optional>
#include <string>
#include <vector>

namespace nntrainer {

using HtpOptionalBool = std::optional<bool>;
using HtpOptionalInt = std::optional<int>;
using HtpOptionalUnsigned = std::optional<unsigned>;
using HtpOptionalString = std::optional<std::string>;
using HtpCpuList = std::vector<int>;

// Shared field lists keep parsing, defaults and equality on the same schema.
#define NNTR_HTP_CONFIG_FIELDS(X)                                              \
  X(unsigned, poll_us, 5000u)                                                  \
  X(unsigned, profile, 0u)                                                     \
  X(unsigned, dma_trace_calls, 3u)                                             \
  X(bool, dsp_queue, true)                                                     \
  X(unsigned, dsp_queue_spin_us, 1000u)                                        \
  X(unsigned, e2e_spin_us, 0u)                                                 \
  X(int, page_cache, -1)                                                       \
  X(HtpOptionalBool, load_prefetch, {})                                        \
  X(unsigned, arena_grain_mib, 64u)                                            \
  X(unsigned, prefill_rows, 512u)                                              \
  X(bool, release_prefill_scratch, true)                                       \
  X(bool, release_prefill_readers, false)                                      \
  X(bool, purge_prefill, false)                                                \
  X(bool, keep_out, true)                                                      \
  X(bool, keep_arm_weights, false)                                             \
  X(bool, stage_debug, false)                                                  \
  X(bool, prefill_io_trace, false)                                             \
  X(bool, init_trace, false)                                                   \
  X(std::string, token_trace, {})                                              \
  X(std::string, static_linear, {})                                            \
  X(std::string, dump_dir, {})                                                 \
  X(bool, dump_all, false)                                                     \
  X(std::string, fc_feed, "vtcm")                                              \
  X(HtpOptionalUnsigned, fc_lanes_small, {})                                   \
  X(HtpOptionalUnsigned, fc_lanes_large, {})                                   \
  X(bool, native_fc, false)                                                    \
  X(bool, fc_whole_prefill, false)                                             \
  X(bool, fc_wh_heap, false)                                                   \
  X(bool, dense_early, true)                                                   \
  X(bool, qkv_to_host, false)                                                  \
  X(bool, oproj_u16, true)                                                     \
  X(bool, kv_seed_q, false)                                                    \
  X(bool, kv_session, false)                                                   \
  X(bool, kv_request_reserve, false)                                           \
  X(bool, attn_f16, true)                                                      \
  X(bool, attn_q2, true)                                                       \
  X(bool, attn_dsp_calib, true)                                                \
  X(bool, attn_f32, true)                                                      \
  X(bool, attn_trace, false)                                                   \
  X(bool, attn_f16_check, false)                                               \
  X(HtpOptionalInt, attn_f32_check, {})                                        \
  X(float, q2_margin, 1.0f)                                                    \
  X(bool, attn_m1_fast, false)                                                 \
  X(std::string, attn_m1_fast_scope, "all")                                    \
  X(unsigned, attn_m1_q_bits, 16u)                                             \
  X(bool, l2_check, false)                                                     \
  X(bool, l2_diff, false)                                                      \
  X(bool, l2_shadow, false)                                                    \
  X(bool, lfm2_default_preset, true)                                           \
  X(bool, lfm2_router_input_handoff, false)                                    \
  X(bool, lfm2_prefill_views, false)                                           \
  X(bool, lfm2_parallel_stage, false)                                          \
  X(bool, lfm2_attn_ion_handoff, false)                                        \
  X(bool, lfm2_conv_retain_out, false)                                         \
  X(bool, lfm2_conv_channel_tile, false)                                       \
  X(bool, lfm2_conv_state_only, false)                                         \
  X(unsigned, lfm2_conv_stripe_rows, 64u)                                      \
  X(bool, lfm2_conv_whole_prefill, false)                                      \
  X(bool, lfm2_parallel_add, false)                                            \
  X(bool, lfm2_parallel_norm, false)

#define NNTR_MOE_CONFIG_FIELDS(X)                                              \
  X(unsigned, cache_experts, 16u)                                              \
  X(unsigned, preload_experts, 128u)                                           \
  X(HtpOptionalUnsigned, cache_grow, {})                                       \
  X(double, lrfu_half_life, 32.0)                                              \
  X(unsigned, prefill_lookahead, 1u << 20)                                     \
  X(unsigned, prefill_readers, 6u)                                             \
  X(HtpCpuList, worker_cpus, {})                                               \
  X(std::string, prediction, "auto")                                           \
  X(HtpOptionalUnsigned, decode_prefetch_readers, {})                          \
  X(unsigned, prefetch_slots, 32u)                                             \
  X(HtpOptionalUnsigned, prefetch_top, {})                                     \
  X(float, prefetch_residual, 0.5f)                                            \
  X(float, prefetch_confidence, 0.0f)                                          \
  X(bool, prefetch_join, true)                                                 \
  X(unsigned, miss_readers, 4u)                                                \
  X(bool, demand_pool, true)                                                   \
  X(bool, coalesce_read, true)                                                 \
  X(bool, cancel_stale_read, true)                                             \
  X(bool, server_done, false)                                                  \
  X(unsigned, fadvise, 0u)                                                     \
  X(unsigned, pin, 0u)                                                         \
  X(unsigned, pin_floor_mib, 2048u)                                            \
  X(bool, m1_gemv, true)                                                       \
  X(unsigned, gemv_lead_kb, 192u)                                              \
  X(bool, gemv_rows1, true)                                                    \
  X(bool, gemv_feed, true)                                                     \
  X(unsigned, dma_queues, 4u)                                                  \
  X(bool, dma_bypass, true)                                                    \
  X(HtpOptionalString, hadamard, {})                                           \
  X(HtpOptionalString, hadamard_scope, {})                                     \
  X(unsigned, host_mask, 0u)                                                   \
  X(bool, debug, false)                                                        \
  X(bool, profile, false)                                                      \
  X(bool, descriptor_trace, false)                                             \
  X(std::string, trace, {})                                                    \
  X(std::string, route_log, {})                                                \
  X(std::string, predict_log, {})                                              \
  X(bool, router_u8i8, false)                                                  \
  X(std::string, tl2_prepared_file, {})                                        \
  X(bool, tl2_decode, true)                                                    \
  X(bool, tl2_direct_read, true)                                               \
  X(bool, tl2_stable_slots, false)                                             \
  X(bool, lfm2_router_arm, false)                                              \
  X(bool, lfm2_prefill_early, false)                                           \
  X(bool, lfm2_prefill_prune, false)                                           \
  X(bool, lfm2_compact_recency, false)                                         \
  X(bool, lfm2_decode_pair_read, false)                                        \
  X(bool, lfm2_decode_demand_handoff, false)                                   \
  X(bool, lfm2_prefetch_gamma, false)

#define NNTR_RUNTIME_CONFIG_DECLARE(type, name, value) type name = value;
struct HtpConfig {
  NNTR_HTP_CONFIG_FIELDS(NNTR_RUNTIME_CONFIG_DECLARE)
  bool operator==(const HtpConfig &other) const;
};
struct MoeConfig {
  NNTR_MOE_CONFIG_FIELDS(NNTR_RUNTIME_CONFIG_DECLARE)
  bool operator==(const MoeConfig &other) const;
};
#undef NNTR_RUNTIME_CONFIG_DECLARE

struct HtpRuntimeConfig {
  HtpConfig htp;
  MoeConfig moe;
  bool operator==(const HtpRuntimeConfig &other) const {
    return htp == other.htp && moe == other.moe;
  }
};

/** The active configuration is immutable while a session is open. */
const HtpRuntimeConfig &htpRuntimeConfig();
/** Validate the complete candidate before replacing any setting. */
void validateHtpRuntimeConfig(const HtpRuntimeConfig &config);
void configureHtpRuntime(const HtpRuntimeConfig &config);
void lockHtpRuntimeConfig();
/** Call only after workers, compute ops and the DSP session are destroyed. */
void resetHtpRuntimeConfig();

} // namespace nntrainer
#endif // NNTR_HTP_RUNTIME_CONFIG_H
