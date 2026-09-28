// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   unittest_hvx_fc.cpp
 * @date   08 Aug 2026
 * @brief  Device benchmark for one fully connected layer on HMX (u8 x i8)
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The counterpart to unittest_hvx_attn.cpp's FusedForwardPerLayerCost, for the
 * other half of the QNN comparison table: one fully connected layer, at
 * Qwen3-0.6B's q_proj shape (K = hidden_size 1024, N = heads 16 x head_dim
 * 128 = 2048), uint8 activation against int8 weight.
 *
 * Reported the way the QNN net-run numbers it gets compared against are
 * reported -- 10 iterations, average over ALL TEN, min, max, and a separate
 * init time for the one-off setup a graph prepare corresponds to.
 *
 * CORRECTNESS for this kernel is NOT here: unittest_hvx_mm_u8i4.cpp's
 * HmxMmU8I8Layer fixture already gates mm_u8i8_layer against the integer
 * reference, and re-deriving that at q_proj's shape would prove nothing new.
 * What is new is mm_u8i8_layer_timed, so that is what this file gates -- the
 * instrumented entry point must return the SAME BYTES as the production one,
 * or every breakdown below describes a different computation than the headline.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <AEEStdErr.h>
#include <remote.h>

#include "htp_rpc_bench.h"
#include "nntr_hvx.h"

namespace {

/**
 * @brief Qwen3-0.6B's q_proj.
 *
 * hidden_size 1024, num_attention_heads 16, head_dim 128 --
 * register_qwen3_0_6b() in Applications/CausalLM/api/model_config.cpp. The
 * other projections in that model are one more entry each in kShapes below
 * (k/v_proj 1024x1024, o_proj 2048x1024, gate/up 1024x3072, down 3072x1024);
 * only q_proj is run because only q_proj is being compared right now, and
 * every extra shape is another 13 device calls per M.
 */
struct FcShape {
  const char *name;
  uint32_t K, N;
};
const FcShape kShapes[] = {{"qproj", 1024u, 2048u}};

/**
 * @brief Sequence lengths, i.e. the matmul's M.
 *
 * 1 is decode. 64 is the row count hexkl_micro_fc_bench.c measured at, and it
 * is here to make that comparison exact rather than argued: the accumulator is
 * 64 rows wide, so m_pad rounds 1 up to 64 and the HMX matmul, the weight DMA
 * and the accumulator read cost the same at both -- but quant and dequant walk
 * the REAL rows, so they do not, and neither does the FastRPC payload. 512 and
 * 1024 are the lengths the QNN table quotes; 1024 is also this model's
 * init_seq_len.
 */
const uint32_t kSeqLens[] = {1u, 64u, 512u, 1024u};

/**
 * @brief Handles per call: one, then a cross-matmul group.
 *
 * The cross-matmul prefetch in hexkl_mm_u8i8_dma.c only fires when a call
 * carries more than one handle -- the push that stages weight i+1 while i
 * computes is guarded by `i + 1 < n_handles`, and the first weight is always
 * drained synchronously before any multiply. So a ONE-handle call measures
 * the path with its weight DMA fully exposed (51 us of 130 at M=1), and
 * doc13 3a's 1.7-2x cross win is not merely absent from the number, it is
 * unreachable by construction. Both are worth publishing: one handle is what
 * an isolated op costs, a group is what a real q/k/v or gate/up set costs.
 *
 * IDENTICAL q_proj shapes rather than a real q/k/v group, so that
 * us_per_matmul stays directly comparable to the one-handle row beside it.
 * A real group has mixed N (2048/1024/1024 here) and would need its own
 * normalisation before it could be compared to anything.
 *
 * Six because that is what hexkl_micro_fc_bench.c's run_pipelined_layer used
 * (KMM = 6): with three handles the per-matmul cost did not reach that
 * bench's number and the shortfall was larger than the drain it amortises,
 * so the group size is measured rather than argued about.
 */
const uint32_t kGroups[] = {1u, 3u, 6u};
constexpr uint32_t kGroupMax = 6u;

/**
 * @brief Weight width. Both, because the QNN op this is compared against is
 *        i4 and only the i8 half was ever measured here.
 *
 * The two paths share the activation quantizer, the int32 dequant and the
 * tile geometry (64x32 activation x 32x32 weight -> 64x32 int32 accumulator);
 * the whole difference is the baked WH bytes per tile, 512 against 1024. So
 * the weight DMA halves and everything else should not move -- which is a
 * prediction this table either confirms or refutes.
 */
enum Width { W_I4 = 4, W_I8 = 8 };
const Width kWidths[] = {W_I4, W_I8};

/** @brief Suffix for the reported path and label. */
const char *width_tag(Width w) { return w == W_I4 ? "i4" : "i8"; }

constexpr int kIters = 10;    /**< the QNN convention: 10 runs, all counted */
constexpr int kProbeRuns = 3; /**< MHA_HTP_PLAN.md 9.7: >= 3, all printed */

/**
 * @brief Slots mm_u8i8_layer_timed fills.
 *
 * Mirrors the enum in test/htp/nntr_hvx_mm_u8i8.c. The DSP header is not
 * includable from the ARM side, so the skel rejects a stale count with
 * AEE_EBADPARM rather than silently truncating.
 */
const char *kStage[] = {"dsp_total_us", "quant_us", "dequant_us", "acc_read_us",
                        "acc_copy_us",  "drain_us", "acc_stride"};
constexpr int kNStages = (int)(sizeof(kStage) / sizeof(kStage[0]));
constexpr int kStageDspTotal = 0;

/** @brief Deterministic pseudo-random fill in [-1, 1). */
void fill_deterministic(float *v, size_t n, uint32_t seed) {
  uint32_t s = seed;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    v[i] = static_cast<float>(static_cast<int32_t>(s >> 8)) /
             static_cast<float>(1 << 23) -
           1.0f;
  }
}

/**
 * @brief Per-channel weight quantization, at either width.
 *
 * The same two quantizers unittest_hvx_mm_u8i4.cpp uses -- qs4cx's
 * 15/(rmax-rmin) into [-8, 7] and the int8 form's 255/(rmax-rmin) into
 * [-128, 127] -- restated here rather than shared, and parametrised over the
 * width rather than copied twice: this file measures time, that one measures
 * error, and the day one of them needs a different quantizer neither should
 * have to be re-verified because the other moved.
 *
 * @param[in]  w_f32  weights, K rows by N columns, row-major
 * @param[in]  w      width, which fixes the span and the clamp
 * @param[out] q_w    quantized values in the width's range, K by N, row-major
 * @param[out] d      per-channel dequantization multiplier, N entries
 * @param[out] colsum per-channel sum of q_w over K, N entries
 */
void quantize_weights_symmetric(const std::vector<float> &w_f32, uint32_t K,
                                uint32_t N, Width w, std::vector<int8_t> &q_w,
                                std::vector<float> &d,
                                std::vector<int32_t> &colsum) {
  const float span = (w == W_I4) ? 15.0f : 255.0f;
  const int qlo = (w == W_I4) ? -8 : -128;
  const int qhi = (w == W_I4) ? 7 : 127;
  q_w.assign(static_cast<size_t>(K) * N, 0);
  d.assign(N, 0.0f);
  colsum.assign(N, 0);

  for (uint32_t n = 0; n < N; ++n) {
    float min0 = w_f32[n];
    float max0 = min0;
    for (uint32_t k = 0; k < K; ++k) {
      const float v = w_f32[static_cast<size_t>(k) * N + n];
      min0 = std::min(min0, v);
      max0 = std::max(max0, v);
    }
    const float rmin = std::min(0.0f, min0);
    const float rmax = std::max(0.0f, max0);
    const float scale = (rmin == rmax) ? 1.0f : span / (rmax - rmin);

    int32_t sum = 0;
    for (uint32_t k = 0; k < K; ++k) {
      int32_t q = static_cast<int32_t>(
        std::round(w_f32[static_cast<size_t>(k) * N + n] * scale));
      q = std::max(qlo, std::min(qhi, q));
      q_w[static_cast<size_t>(k) * N + n] = static_cast<int8_t>(q);
      sum += q;
    }
    d[n] = 1.0f / scale;
    colsum[n] = sum;
  }
}

/**
 * @brief Opens an unsigned-PD CDSP session with the same transport controls
 *        the attention benchmark runs under (htp_rpc_bench.h).
 *
 * A failure here is a hard FAIL rather than a skip: proving the DSP comes up
 * is part of what this measures.
 */
class HtpFc : public ::testing::Test {
protected:
  void SetUp() override {
    int err = htp_enable_unsigned_pd();
    ASSERT_EQ(err, AEE_SUCCESS) << "enabling unsigned PD failed: " << hex(err);

    const std::string uri = std::string(nntr_hvx_URI) + "&_dom=cdsp";
    err = nntr_hvx_open(uri.c_str(), &handle_);
    ASSERT_EQ(err, AEE_SUCCESS)
      << "nntr_hvx_open failed: " << hex(err)
      << " -- is libnntr_hvx_skel.so on ADSP_LIBRARY_PATH?";

    std::cout << "FC_FIELD path=env field=rpc_poll_qos value="
              << htp_set_latency_qos(handle_) << std::endl;
  }

  void TearDown() override {
    if (handle_) {
      nntr_hvx_close(handle_);
    }
  }

  remote_handle64 handle_ = 0;
};

} // namespace

/**
 * @brief End-to-end cost of one fully connected layer, in microseconds.
 *
 * Timed on the ARM side around the FastRPC call, so the headline INCLUDES
 * transport. For a fully connected layer that is a much bigger share than it
 * is for attention -- the interface carries M x K floats in and M x N floats
 * out, 12 MB per call at M=1024, against attention's 1 MB -- which is exactly
 * why the DSP-internal time is measured alongside rather than assumed. A QNN
 * per-op number has no host transfer in it; dsp_total_us is the term to put
 * beside it, and wall is the term to put beside a real caller's experience.
 *
 * Rules this follows, each one a bug this project already shipped and found
 * (MHA_HTP_PLAN.md 9.7):
 *   - the consistency check is NOT in the timed region;
 *   - more than three runs, and the spread is published, not just the mean;
 *   - field=... value=... marker lines with the unit in the field name;
 *   - nothing is asserted about the timings. The reviewer compares.
 */
TEST_F(HtpFc, LayerCostQwen3) {
  std::cout
    << "\nQwen3-0.6B fully connected layer, u8 activation x i8 weight, "
       "one call end to end\n"
    << "10 iterations; avg/min/max are over ALL 10, run 1 included, which is "
       "the\n"
    << "convention the QNN net-run numbers this gets compared against use. "
       "init is\n"
    << "weight_register -- baking K x N to WH layout, the one-time setup a "
       "graph\n"
    << "prepare corresponds to. dsp is the same work measured inside the DSP, "
       "so\n"
    << "wall - dsp is the FastRPC transport for M*K + M*N floats.\n"
    << "shape           M     nh      avg1-10      min      max      dsp"
       "     init   us/mm  dsp/mm\n";

  for (const FcShape &s : kShapes) {
    for (Width W : kWidths) {
      const std::string sname = std::string(s.name) + "_" + width_tag(W);
      /** One weight, registered once per shape and reused across every M: the
       * bake is expensive and M does not enter it, so re-registering per M
       * would only measure the same bake three times. init_us below is that
       * one measurement, reported against the shape it belongs to. */
      std::vector<float> w_f32((size_t)s.K * s.N);
      fill_deterministic(w_f32.data(), w_f32.size(), 0xFC000001u);
      std::vector<int8_t> q_w;
      std::vector<float> d;
      std::vector<int32_t> colsum;
      quantize_weights_symmetric(w_f32, s.K, s.N, W, q_w, d, colsum);
      std::vector<float> bias(s.N);
      fill_deterministic(bias.data(), bias.size(), 0xFC000002u);

      /** kGroupMax registrations of the same bytes, so a group call has distinct
       * handles to prefetch across. init_us is the FIRST bake only: it is the
       * per-weight one-time cost a graph prepare corresponds to, and averaging
       * three of them would only measure the same work three times. */
      uint32_t handles[kGroupMax];
      double init_us = 0.0;
      for (uint32_t g = 0; g < kGroupMax; ++g) {
        handles[g] = 0xFFFFFFFFu;
        const auto init_t0 = std::chrono::steady_clock::now();
        const int rc_reg =
          (W == W_I4)
            ? nntr_hvx_weight_register_u8i4(
                handle_, s.K, s.N, q_w.data(), (int)q_w.size(), d.data(),
                (int)d.size(), colsum.data(), (int)colsum.size(), bias.data(),
                (int)bias.size(), &handles[g])
            : nntr_hvx_weight_register_u8i8(
                handle_, s.K, s.N, q_w.data(), (int)q_w.size(), d.data(),
                (int)d.size(), colsum.data(), (int)colsum.size(), bias.data(),
                (int)bias.size(), &handles[g]);
        const auto init_t1 = std::chrono::steady_clock::now();
        ASSERT_EQ(rc_reg, AEE_SUCCESS)
          << "weight_register_u8" << width_tag(W) << ": " << hex(rc_reg);
        if (g == 0) {
          init_us = std::chrono::duration<double, std::micro>(init_t1 - init_t0)
                      .count();
        }
      }

      for (uint32_t M : kSeqLens) {
        for (uint32_t G : kGroups) {
          const size_t an = (size_t)M * s.K;
          /* out_cat concatenates one M x N block per handle, in call order. */
          const size_t on = (size_t)G * M * s.N;
          /** Both ride in every timed call. See RpcBuf for why they are
           * ION-backed: plain heap is re-pinned and re-mapped on EVERY call,
           * and at 12 MB that mapping is a large part of what is being
           * measured. */
          RpcBuf abuf(an * sizeof(float));
          RpcBuf obuf(on * sizeof(float));
          ASSERT_NE(abuf.p, nullptr);
          ASSERT_NE(obuf.p, nullptr);
          float *act = (float *)abuf.p;
          float *out = (float *)obuf.p;
          fill_deterministic(act, an, 0x5EED0000u + M);
          std::memset(out, 0, on * sizeof(float));
          std::cout << "FC_FIELD path=env field=ion_buffers value="
                    << ((abuf.ion && obuf.ion) ? 1 : 0) << std::endl;

          /** PRODUCTION path: mm_u8i8_layer leaves the DSP's in-loop probes off,
           * which is what a real caller pays and therefore what the headline
           * reports. The instrumented runs below are for the breakdown, and the
           * gap between them is published rather than folded in. */
          double iter[kIters];
          for (int r = 0; r < kIters; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            const int err =
              (W == W_I4) ? nntr_hvx_mm_u8i4_layer(handle_, M, s.K, handles, G,
                                                   act, (int)an, out, (int)on)
                          : nntr_hvx_mm_u8i8_layer(handle_, M, s.K, handles, G,
                                                   act, (int)an, out, (int)on);
            const auto t1 = std::chrono::steady_clock::now();
            ASSERT_EQ(err, AEE_SUCCESS)
              << "mm_u8" << width_tag(W) << "_layer: " << hex(err)
              << " at M=" << M
              << " -- ENOMEMORY here means the activation plus the "
                 "double-buffered weight outgrew the VTCM arena";
            iter[r] =
              std::chrono::duration<double, std::micro>(t1 - t0).count();
          }
          double sum = 0.0, lo = iter[0], hi = iter[0];
          for (int r = 0; r < kIters; ++r) {
            sum += iter[r];
            lo = std::min(lo, iter[r]);
            hi = std::max(hi, iter[r]);
          }
          const double avg = sum / (double)kIters;

          /** Keep the production result so the instrumented entry point can be
           * held to it byte for byte, below and outside every timed region. */
          std::vector<float> want(out, out + on);

          double probed[kProbeRuns];
          std::vector<uint32_t> stage_of[kProbeRuns];
          for (int r = 0; r < kProbeRuns; ++r) {
            std::vector<uint32_t> stage(kNStages, 0u);
            const auto t0 = std::chrono::steady_clock::now();
            const int err =
              (W == W_I4)
                ? nntr_hvx_mm_u8i4_layer_timed(handle_, M, s.K, handles, G, act,
                                               (int)an, out, (int)on,
                                               stage.data(), (int)stage.size())
                : nntr_hvx_mm_u8i8_layer_timed(handle_, M, s.K, handles, G, act,
                                               (int)an, out, (int)on,
                                               stage.data(), (int)stage.size());
            const auto t1 = std::chrono::steady_clock::now();
            /* Checked after the clock stops, deliberately. */
            ASSERT_EQ(err, AEE_SUCCESS)
              << "mm_u8" << width_tag(W) << "_layer_timed: " << hex(err);
            probed[r] =
              std::chrono::duration<double, std::micro>(t1 - t0).count();
            stage_of[r] = stage;
          }

          /** The gate this file carries: instrumentation must not change the
           * answer. Bitwise, because both paths run identical integer
           * arithmetic on identical inputs -- a tolerance here would hide a
           * probe that perturbs the kernel. */
          const int diff = std::memcmp(out, want.data(), on * sizeof(float));
          EXPECT_EQ(diff, 0) << "mm_u8" << width_tag(W)
                             << "_layer_timed returned different bytes than "
                                "the production entry at M="
                             << M;
          const int timed_ok = (diff == 0) ? 1 : 0;

          /** MEDIAN of the probe runs, not the min: wall clock on this device is
           * bimodal at roughly +-25% run to run (DVFS residency, not our code),
           * so the min publishes whichever run landed in the fast state. It
           * stays a single measured run rather than a mean, so the stage
           * breakdown beside it comes from that same run and sums to its own
           * dsp_total. */
          int ord[kProbeRuns];
          for (int r = 0; r < kProbeRuns; ++r) {
            ord[r] = r;
          }
          std::sort(ord, ord + kProbeRuns,
                    [&probed](int a, int b) { return probed[a] < probed[b]; });
          const double med_probed = probed[ord[kProbeRuns / 2]];
          const std::vector<uint32_t> &med_stage =
            stage_of[ord[kProbeRuns / 2]];
          const double dsp = (double)med_stage[kStageDspTotal];

          /** The compute bucket -- dsp minus everything that only moves data --
           * computed PER RUN and published as a range, not derived once from
           * the median stage set.
           *
           * It has to be a range because the probes are HAP_perf_get_time_us,
           * whose resolution is 1 us, while one acc_read is ~0.38 us: at M=1
           * there are only 64 of them per matmul, so the sum is dominated by
           * which samples happened to cross a microsecond boundary. Measured
           * proof that this is resolution and not a real effect: at M=1024,
           * where the same probe covers 1024 calls, i4 and i8 agree to 3%
           * (389 against 378 us) as they must -- the accumulator readout
           * cannot depend on the weight width. At M=1 they came out 35 and 22.
           * dsp_total is one timestamp pair around the whole call and does not
           * have this problem; only the split does. */
          double comp[kProbeRuns];
          for (int r = 0; r < kProbeRuns; ++r) {
            const std::vector<uint32_t> &st = stage_of[r];
            comp[r] = (double)st[kStageDspTotal] -
                      (double)(st[1] + st[2] + st[3] + st[4]);
          }
          std::sort(comp, comp + kProbeRuns);

          std::cout << std::left << std::setw(14) << sname << std::setw(6) << M
                    << std::right << std::setw(3) << G << std::fixed
                    << std::setprecision(1) << std::setw(13) << avg
                    << std::setw(9) << lo << std::setw(9) << hi << std::setw(9)
                    << dsp << std::setw(9) << init_us << std::setw(8)
                    << (avg / (double)G) << std::setw(8) << (dsp / (double)G)
                    << "\n"
                    << std::left;

          std::string path =
            std::string("fc_") + sname +
            (G > 1u ? ("_x" + std::to_string(G)) : std::string()) + "_m" +
            std::to_string(M);
          static const char *kRun[] = {"us_avg_1_10", "us_min", "us_max",
                                       "us_first", "init_us"};
          const double run_v[] = {avg, lo, hi, iter[0], init_us};
          for (size_t f = 0; f < sizeof(run_v) / sizeof(run_v[0]); ++f) {
            std::cout << "FC_FIELD path=" << path << " field=" << kRun[f]
                      << " value=" << run_v[f] << std::endl;
          }
          /** Shape travels with the numbers so the report can derive MAC counts
           * and transported bytes without a second copy of kShapes to drift. */
          std::cout << "FC_FIELD path=" << path << " field=M value=" << M
                    << std::endl;
          std::cout << "FC_FIELD path=" << path << " field=K value=" << s.K
                    << std::endl;
          std::cout << "FC_FIELD path=" << path << " field=N value=" << s.N
                    << std::endl;
          /** Published, not just asserted: a report that only shows timings
           * should be able to say on its own face whether the breakdown beside
           * them describes the same computation as the headline. */
          std::cout << "FC_FIELD path=" << path
                    << " field=timed_matches_production value=" << timed_ok
                    << std::endl;

          for (int st = 0; st < kNStages; ++st) {
            std::cout << "FC_STAGE path=" << path << " field=" << kStage[st]
                      << " value=" << med_stage[st] << std::endl;
          }
          /** dsp_total came from the INSTRUMENTED run, so subtract it from THAT
           * run's wall; mixing it with the production wall would fold the
           * instrumentation into the transport estimate. */
          std::cout << "FC_STAGE path=" << path
                    << " field=transport_us value=" << (med_probed - dsp)
                    << std::endl;
          std::cout << "FC_STAGE path=" << path
                    << " field=probe_overhead_us value=" << (med_probed - avg)
                    << std::endl;
          /** The two numbers the cross row exists to publish. Per MATMUL,
           * because that is the unit doc13 3a's 1.7-2x is quoted in and the
           * unit the one-handle row beside it reports. dsp_per_matmul is the
           * comparable one: wall_per_matmul also carries transport, which
           * scales with the G x M x N output bytes and so gets worse with G no
           * matter how well the weight staging hides. */
          std::cout << "FC_FIELD path=" << path
                    << " field=n_handles value=" << G << std::endl;
          std::cout << "FC_FIELD path=" << path
                    << " field=us_per_matmul value=" << (avg / (double)G)
                    << std::endl;
          std::cout << "FC_STAGE path=" << path
                    << " field=dsp_per_matmul_us value=" << (dsp / (double)G)
                    << std::endl;
          static const char *kComp[] = {"compute_us_lo", "compute_us_med",
                                        "compute_us_hi"};
          const double comp_v[] = {comp[0], comp[kProbeRuns / 2],
                                   comp[kProbeRuns - 1]};
          for (size_t f = 0; f < 3; ++f) {
            std::cout << "FC_STAGE path=" << path << " field=" << kComp[f]
                      << " value=" << comp_v[f] << std::endl;
          }
        }
      }

      for (uint32_t g = 0; g < kGroupMax; ++g) {
        EXPECT_EQ((W == W_I4)
                    ? nntr_hvx_weight_release_u8i4(handle_, handles[g])
                    : nntr_hvx_weight_release_u8i8(handle_, handles[g]),
                  AEE_SUCCESS);
      }
    }
  }
}

int main(int argc, char **argv) {
  int result = -1;
  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }
  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }
  return result;
}
