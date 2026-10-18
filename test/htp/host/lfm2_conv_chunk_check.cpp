// SPDX-License-Identifier: Apache-2.0
/**
 * @file lfm2_conv_chunk_check.cpp
 * @brief lfm2 conv chunk check validation.
 * @author NNTrainer contributors
 * @bug No known bugs.
 */
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#if defined(__aarch64__)
#include <arm_neon.h>
#else
#define CONV_SIMDE_FIXTURE 1
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/arm/neon.h>
// Set only after system/SIMDe headers select the real host architecture.
#define __aarch64__ 1
#endif
#include <htp_router_handoff.h>
#define ml_logw(...) ((void)0)

static void check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
static unsigned cases = 0;
static size_t copied_bytes = 0;
static void stagedMemcpy(void *d, const void *s, size_t n, int = 0) {
  std::memcpy(d, s, n);
  copied_bytes += n;
}
namespace nntrainer {
struct ThreadManager {
  inline static unsigned lanes = 1;
  static ThreadManager &Global() {
    static ThreadManager value;
    return value;
  }
  template <typename F> void parallel_for(size_t begin, size_t end, F fn) {
    if (lanes == 1) {
      for (size_t i = begin; i < end; ++i)
        fn(i);
      return;
    }
    std::vector<std::thread> workers;
    for (unsigned lane = 0; lane < lanes; ++lane)
      workers.emplace_back([=] {
        for (size_t r = begin + lane; r < end; r += lanes)
          fn(r);
      });
    for (auto &w : workers)
      w.join();
  }
};
struct HtpProfile {
  static HtpProfile &global() {
    static HtpProfile p;
    return p;
  }
  unsigned level() const { return 0; }
};
struct HtpRpcBuffer {
  inline static size_t live = 0, peak = 0, allocations = 0;
  size_t count;
  std::unique_ptr<uint8_t[]> ptr;
  explicit HtpRpcBuffer(size_t n) : count(n), ptr(new uint8_t[n]) {
    live += n;
    peak = std::max(peak, live);
    ++allocations;
  }
  ~HtpRpcBuffer() { live -= count; }
  uint8_t *data() { return ptr.get(); }
  size_t size() const { return count; }
};
struct ViewBase {
  virtual const void *host_view(const void *, size_t) = 0;
  virtual void sync_to_host(void *, size_t) = 0;
  virtual ~ViewBase() = default;
};
struct ComputeOps : ViewBase {
#include "staging.inc"
#include "views.inc"
  StagingPool act_pool_, out_pool_, x2_pool_;
  htp_router_handoff router_handoff_{};
  std::mutex invoke_mutex_;
  enum View {
    Separate,
    Host,
    Null,
    OverlapInput,
    OverlapGated,
    OverlapOutput,
    OverlapState
  };
  View mode = Separate;
  bool supported = true, poison_old = true;
  unsigned output_projections = 0;
  unsigned calls = 0, syncs = 0, views = 0, fail_rpc = 0, fail_sync = 0;
  const float *input_base = nullptr, *truth = nullptr;
  float *gated = nullptr, *output = nullptr, *state = nullptr;
  const float *expected_gated = nullptr, *expected_output = nullptr;
  std::vector<unsigned> row_calls;
  std::vector<std::array<size_t, 4>> requests;
  void gemm_qs4cx_accel_fp32(const char *w, const float *scales,
                             const float *in, float *out, unsigned rows,
                             unsigned n, unsigned k) {
    check(w && scales && in == output && out == output && rows == 2048 &&
            n == 2048 && k == 2048,
          "chunked output projection must use the layer's packed HTP path");
    ++output_projections;
  }
  bool supports_gemm_q4_0_batch_norm_fp32() const { return supported; }
  const void *view(const void *host, size_t n) {
    ++views;
    if (mode == Null)
      return nullptr;
    if (mode == OverlapInput)
      return input_base;
    if (mode == OverlapGated)
      return gated;
    if (mode == OverlapOutput)
      return output;
    if (mode == OverlapState)
      return state;
    return host_view(host, n);
  }
  void sync(void *host, size_t n) {
    ++syncs;
    if (fail_sync == calls)
      throw std::runtime_error("injected sync failure");
    sync_to_host(host, n);
  }
  void project(float *input, float *dst, unsigned rows, bool retained) {
    ++calls;
    row_calls.push_back(rows);
    const size_t first = size_t(input - input_base) / 2048u;
    check(first == 0 || first == 1920, "unexpected input row offset");
    if (first == 1920 && expected_gated) {
      check(std::memcmp(gated, expected_gated, 1920u * 2048u * 4u) == 0,
            "first gated chunk not completed before next RPC");
      check(std::memcmp(output, expected_output, 1920u * 2048u * 4u) == 0,
            "first conv chunk not completed before next RPC");
      if (poison_old) {
        auto &old = out_pool_.by_class[size_t(64) << 20];
        check(bool(old), "first ION missing");
        std::memset(old->data(), 0xff, old->size());
      }
    }
    if (fail_rpc == calls)
      throw std::runtime_error("injected RPC failure");
    const size_t in_bytes = size_t(rows) * 2048 * 4,
                 out_bytes = size_t(rows) * 6144 * 4;
    auto &act = stageInput(act_pool_, input, in_bytes, &out_pool_);
    auto &out = stage(out_pool_, out_bytes);
    check(std::memcmp(act.data(), input, in_bytes) == 0,
          "input changed or staging stale");
    std::memcpy(out.data(), truth + first * 6144, out_bytes);
    if (!retained || mode == Host)
      stagedMemcpy(dst, out.data(), out_bytes);
    else
      sampledCopy(dst, out.data(), out_bytes);
    noteIonAlias(dst, out.data(), out_bytes);
    requests.push_back({in_bytes, out_bytes, act.size(), out.size()});
  }
  void
  gemm_q4_0_batch_norm_fp32(std::vector<void *> w, std::vector<float *> scales,
                            float *in, std::vector<float *> dst, unsigned rows,
                            std::vector<unsigned> widths, unsigned k,
                            const float *pre, const std::vector<unsigned> &post,
                            const float *gamma, float eps, const float *rope,
                            unsigned hd, unsigned rope_weights, bool to_host) {
    check(w.size() == 1 && w[0] && scales.size() == 1 && scales[0] &&
            dst.size() == 1 && widths == std::vector<unsigned>{6144} &&
            k == 2048 && !pre && post == std::vector<unsigned>{0} && !gamma &&
            eps == 0 && !rope && !hd && !rope_weights && !to_host,
          "retained FC contract changed");
    project(in, dst[0], rows, true);
  }
};
} // namespace nntrainer

namespace baseline {
#include "baseline_helpers.inc"
}
// Route host-view faults through virtual test hooks while retaining the actual
// host_view/sync_to_host implementations above for normal ownership checks.
#define host_view view
#define sync_to_host sync
namespace candidate {
#include "candidate_helpers.inc"
}
#undef host_view
#undef sync_to_host

constexpr unsigned C = 2048, M = 2048, N = 6144;
static void fill(std::vector<float> &x, uint32_t seed, unsigned pattern) {
  const uint32_t special[] = {0,          0x80000000, 1,          0x80000001,
                              0x00800000, 0x80800000, 0x3f800000, 0xbf800000,
                              0x7f800000, 0xff800000, 0x7fc00021, 0xffc00042};
  for (size_t i = 0; i < x.size(); ++i) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    uint32_t bits = (seed & 0x807fffff) | ((120u + unsigned(i % 12)) << 23);
    if (pattern == 1)
      bits = special[(i + seed) % 12];
    if (pattern == 2)
      bits = ((i / 2048 == 1918 || i / 2048 == 1919) ? 0x3f800000 : 0);
    std::memcpy(&x[i], &bits, 4);
  }
}
struct Fixture {
  std::vector<float> input, truth, proj, g, out, state, weights, refg, refout,
    refstate;
  nntrainer::ComputeOps ops;
  Fixture() :
    input(size_t(M) * C),
    truth(size_t(M) * N),
    proj(truth.size(), -9876),
    g(input.size(), -9876),
    out(input.size(), -9876),
    state(2 * C, -77),
    weights(3 * C),
    refg(g.size()),
    refout(out.size()),
    refstate(state.size()) {
    ops.input_base = input.data();
    ops.truth = truth.data();
    ops.gated = g.data();
    ops.output = out.data();
    ops.state = state.data();
  }
  void prepare(unsigned pattern, unsigned generation) {
    fill(input, 0x741243u + generation, 0);
    fill(truth, 0x53181u + generation, pattern);
    fill(weights, 0x7813u + generation, pattern);
    for (size_t r = 0; r < M; ++r)
      for (unsigned c = 0; c < C; ++c)
        refg[r * C + c] = truth[r * N + c] * truth[r * N + 2 * C + c];
    check(baseline::lfmConvPostGatePrefill(refg.data(), weights.data(),
                                           truth.data(), refout.data(), M, C),
          "baseline rejected");
    std::memcpy(refstate.data(), refg.data() + size_t(M - 2) * C, 2 * C * 4);
    std::fill(proj.begin(), proj.end(), -9876);
    std::fill(g.begin(), g.end(), -9876);
    std::fill(out.begin(), out.end(), -9876);
    std::fill(state.begin(), state.end(), -77);
    ops.expected_gated = refg.data();
    ops.expected_output = refout.data();
    ops.calls = ops.syncs = ops.views = 0;
    ops.row_calls.clear();
    ops.requests.clear();
  }
  bool run(unsigned rows = M, unsigned channels = C) {
    return candidate::lfmConvChunkedPrefill(
      &ops, truth.data(), weights.data(), input.data(), proj.data(), g.data(),
      weights.data(), out.data(), state.data(), rows, channels);
  }
  void exact() {
    check(std::memcmp(g.data(), refg.data(), g.size() * 4) == 0,
          "gated byte mismatch");
    check(std::memcmp(out.data(), refout.data(), out.size() * 4) == 0,
          "output byte mismatch");
    check(std::memcmp(state.data(), refstate.data(), state.size() * 4) == 0,
          "state byte mismatch");
  }
};

static void numeric() {
  for (unsigned lanes : {1u, 4u})
    for (unsigned pattern : {0u, 1u, 2u}) {
      nntrainer::ThreadManager::lanes = lanes;
      Fixture f;
      for (unsigned generation = 0; generation < 2; ++generation) {
        f.prepare(pattern, generation);
        check(f.run(), "chunk path declined");
        f.exact();
        check(f.ops.row_calls == std::vector<unsigned>({1920, 128}) &&
                f.ops.syncs == 0 && f.ops.views == 2,
              "dispatch/view counts");
        check(std::memcmp(f.proj.data(), f.truth.data(), f.proj.size() * 4) !=
                0,
              "poisoned host projection unexpectedly materialized");
        ++cases;
      }
    }
  nntrainer::ThreadManager::lanes = 1;
  for (auto mode : {nntrainer::ComputeOps::Host, nntrainer::ComputeOps::Null,
                    nntrainer::ComputeOps::OverlapInput,
                    nntrainer::ComputeOps::OverlapGated,
                    nntrainer::ComputeOps::OverlapOutput,
                    nntrainer::ComputeOps::OverlapState}) {
    Fixture f;
    f.prepare(0, 0);
    f.ops.mode = mode;
    check(f.run(), "fallback view declined");
    f.exact();
    check(f.ops.syncs == unsigned(mode == nntrainer::ComputeOps::Host ? 0 : 2),
          "bad fallback sync count");
    ++cases;
  }
}
static void capacity() {
  Fixture f;
  f.prepare(0, 0);
  f.ops.expected_gated = nullptr;
  f.ops.expected_output = nullptr;
  copied_bytes = 0;
  f.ops.project(f.input.data(), f.proj.data(), 1920, false);
  f.ops.project(f.input.data() + 1920u * C, f.proj.data() + 1920u * N, 128,
                false);
  const auto before = f.ops.requests;
  const size_t copies = copied_bytes,
               allocs = nntrainer::HtpRpcBuffer::allocations;
  f.prepare(0, 1);
  copied_bytes = 0;
  check(f.run(), "capacity run declined");
  f.exact();
  check(f.ops.requests == before, "staging request/capacity changed");
  check(nntrainer::HtpRpcBuffer::allocations == allocs,
        "staging allocation grew on same requests");
  const std::vector<std::array<size_t, 4>> expected = {
    {{15728640, 47185920, 15732736, 47190016}},
    {{1048576, 3145728, 1052672, 3149824}}};
  check(before == expected, "actual fit capacity differs");
  check(copies == 67108864 && copied_bytes == 16777216,
        "copy byte removal differs");
  std::printf("CAPACITY requests_same=1 allocated_same=1 act=15732736+1052672 "
              "out=47190016+3149824 copy_bytes=%zu->%zu\n",
              copies, copied_bytes);
  ++cases;
}
static void guards() {
  Fixture f;
  f.prepare(0, 0);
  for (unsigned reason = 0; reason < 23; ++reason) {
    auto *input = f.input.data(), *p = f.proj.data(), *g = f.g.data(),
         *y = f.out.data(), *state = f.state.data();
    const float *w = f.weights.data();
    unsigned rows = M, channels = C;
    if (reason == 0)
      rows = 1024;
    if (reason == 1)
      rows = 1920;
    if (reason == 2)
      rows = 2049;
    if (reason == 3)
      channels = 2816;
    if (reason == 4)
      g = input;
    if (reason == 5)
      g = input + 1;
    if (reason == 6)
      y = input;
    if (reason == 7)
      y = input + 1;
    if (reason == 8)
      p = input;
    if (reason == 9)
      g = p;
    if (reason == 10)
      y = p + 1;
    if (reason == 11)
      state = p;
    if (reason == 12)
      y = g;
    if (reason == 13)
      state = g;
    if (reason == 14)
      state = y;
    if (reason == 15)
      y = const_cast<float *>(w);
    if (reason == 16)
      g = const_cast<float *>(w);
    if (reason == 17)
      p = const_cast<float *>(w);
    if (reason == 18)
      state = input;
    if (reason == 19)
      input = nullptr;
    if (reason == 20)
      g = nullptr;
    if (reason == 21)
      y = nullptr;
    if (reason == 22)
      f.ops.supported = false;
    check(!candidate::lfmConvChunkedPrefill(&f.ops, f.truth.data(),
                                            f.weights.data(), input, p, g, w, y,
                                            state, rows, channels),
          "guard failed");
    check(f.ops.calls == 0, "rejected case started backend work");
    ++cases;
  }
  f.ops.supported = true;
  for (auto bounds : std::vector<std::array<unsigned, 2>>{
         {2048, 2048}, {2049, 2050}, {1920, 2049}, {2048, 0}}) {
    check(!candidate::lfmConvPostGatePrefillRange(f.g.data(), f.weights.data(),
                                                  f.truth.data(), f.out.data(),
                                                  M, C, bounds[0], bounds[1]),
          "range guard failed");
    ++cases;
  }
}
static void failure() {
  for (unsigned which = 1; which <= 2; ++which)
    for (bool sync : {false, true}) {
      Fixture f;
      f.prepare(0, 0);
      if (sync) {
        f.ops.mode = nntrainer::ComputeOps::Null;
        f.ops.fail_sync = which;
      } else
        f.ops.fail_rpc = which;
      bool threw = false;
      try {
        f.run();
      } catch (const std::runtime_error &) {
        threw = true;
      }
      check(threw && std::all_of(f.state.begin(), f.state.end(),
                                 [](float x) { return x == -77; }),
            "failed RPC/sync updated state or swallowed error");
      f.ops.fail_rpc = f.ops.fail_sync = 0;
      f.prepare(0, 1);
      check(f.run(), "retry declined");
      f.exact();
      ++cases;
    }
}

namespace nntrainer {
enum class Tdatatype { FP32, FP16, QS4CX_WH_AFFINE, Q4_0 };
}
struct Tensor {
  float *data;
  nntrainer::Tdatatype dtype = nntrainer::Tdatatype::FP32;
  bool contiguous = true;
  unsigned dots = 0;
  void *getData() const { return data; }
  template <typename T> T *getData() const {
    return reinterpret_cast<T *>(data);
  }
  template <typename T> T *getScale() const {
    return reinterpret_cast<T *>(data);
  }
  nntrainer::Tdatatype getDataType() const { return dtype; }
  bool getContiguous() const { return contiguous; }
  void dot(const Tensor &, Tensor &, bool, bool) { ++dots; }
};
static bool htpDecodeRowResident(unsigned) { return false; }
namespace candidate {
static void appRoute(Tensor &in_step, Tensor &in_w, Tensor &out_w, Tensor &proj,
                     Tensor &gated, Tensor &conv_out, Tensor &out_step,
                     float *state, const float *w_ptr,
                     nntrainer::ComputeOps *ops, unsigned rows, unsigned K,
                     unsigned C, unsigned N, unsigned from, bool compare,
                     bool htp_owns_state_, bool &fellthrough) {
  const auto affine = nntrainer::Tdatatype::QS4CX_WH_AFFINE;
#include "app_route.inc"
  (void)p;
  (void)row_resident;
  (void)projection_disjoint;
  fellthrough = true;
}
} // namespace candidate
static void app_guards() {
  Fixture f;
  for (unsigned reason = 0; reason < 25; ++reason) {
    f.prepare(0, reason);
    f.ops.output_projections = 0;
    Tensor in{f.input.data()},
      iw{f.truth.data(), nntrainer::Tdatatype::QS4CX_WH_AFFINE};
    Tensor ow = iw, p{f.proj.data()}, g{f.g.data()}, y{f.out.data()},
           o{f.out.data()};
    unsigned rows = M, K = C, channels = C, Nout = C, from = 0;
    bool compare = false, owns = false, fellthrough = false;
    auto *ops = &f.ops;
    if (reason == 1)
      from =
        2048; // A continuation must preserve its incoming convolution state.
    if (reason == 2)
      owns = true; // A fresh prompt starts at0 even after decode.
    if (reason == 3) {
      owns = true;
      from = 2048;
    }
    if (reason == 4)
      rows = 1024;
    if (reason == 5)
      rows = 1920;
    if (reason == 6)
      K = 2816;
    if (reason == 7)
      Nout = 2816;
    if (reason == 8)
      channels = 2816;
    if (reason == 9)
      iw.dtype = nntrainer::Tdatatype::Q4_0;
    if (reason == 10)
      ow.dtype = nntrainer::Tdatatype::Q4_0;
    if (reason == 11)
      in.dtype = nntrainer::Tdatatype::FP16;
    if (reason == 12)
      p.dtype = nntrainer::Tdatatype::FP16;
    if (reason == 13)
      g.dtype = nntrainer::Tdatatype::FP16;
    if (reason == 14)
      y.dtype = nntrainer::Tdatatype::FP16;
    if (reason == 15)
      in.contiguous = false;
    if (reason == 16)
      p.contiguous = false;
    if (reason == 17)
      g.contiguous = false;
    if (reason == 18)
      y.contiguous = false;
    if (reason == 19)
      compare = true;
    if (reason == 20)
      ops = nullptr;
    if (reason == 21)
      f.ops.supported = false;
    if (reason == 22)
      g.data = in.data;
    if (reason == 23)
      y.data = in.data + 1;
    if (reason == 24)
      p.data = g.data;
    candidate::appRoute(in, iw, ow, p, g, y, o, f.state.data(),
                        f.weights.data(), ops, rows, K, channels, Nout, from,
                        compare, owns, fellthrough);
    const bool accepted = reason == 0 || reason == 2;
    check(fellthrough != accepted && f.ops.calls == (accepted ? 2u : 0u) &&
            y.dots == 0 && f.ops.output_projections == unsigned(accepted),
          "actual app guard/fallback path differs");
    if (accepted)
      f.exact();
    f.ops.supported = true;
    ++cases;
  }
}
static void unchanged_and_ranges() {
  Fixture f;
  f.prepare(1, 0);
  for (unsigned rows : {16u, 17u, 31u, 64u, 1024u, 2048u}) {
    std::fill(f.out.begin(), f.out.end(), -9876);
    std::fill(f.refout.begin(), f.refout.end(), -9876);
    check(baseline::lfmConvPostGatePrefill(f.refg.data(), f.weights.data(),
                                           f.truth.data(), f.refout.data(),
                                           rows, C),
          "old helper guard failed");
    check(candidate::lfmConvPostGatePrefill(f.refg.data(), f.weights.data(),
                                            f.truth.data(), f.out.data(), rows,
                                            C),
          "new default helper guard failed");
    check(std::memcmp(f.out.data(), f.refout.data(), f.out.size() * 4) == 0,
          "old default path output/canary differs");
    ++cases;
  }
  std::fill(f.out.begin(), f.out.end(), -9876);
  for (auto range : std::vector<std::array<unsigned, 2>>{
         {0, 16}, {16, 2047}, {2047, 2048}}) {
    check(candidate::lfmConvPostGatePrefillRange(
            f.refg.data(), f.weights.data(),
            f.truth.data() + size_t(range[0]) * N, f.out.data(), M, C, range[0],
            range[1]),
          "valid ranged helper declined");
  }
  check(std::memcmp(f.out.data(), f.refout.data(), f.out.size() * 4) == 0,
        "range partition loses global history or gate offset");
  ++cases;
}

static void fused_gate_stripes() {
  for (unsigned lanes : {1u, 3u, 8u})
    for (unsigned pattern : {0u, 1u, 2u, 3u}) {
      nntrainer::ThreadManager::lanes = lanes;
      Fixture f;
      f.prepare(pattern == 1 ? 1 : 0, pattern);
      if (pattern == 2) {
        std::fill(f.truth.begin(), f.truth.end(), 0.0f);
        for (unsigned r : {0u, 1u, 62u, 63u, 64u, 65u, 126u, 127u, 128u})
          for (unsigned c = 0; c < N; ++c)
            f.truth[size_t(r) * N + c] = float(int(c % 13) - 6) / 7.0f;
      }
      if (pattern == 3) {
        // SR15's decoded convolution weights lie on the BF16 grid.
        for (float &v : f.weights) {
          uint32_t bits;
          std::memcpy(&bits, &v, 4);
          bits &= 0xffff0000u;
          std::memcpy(&v, &bits, 4);
        }
      }
      for (unsigned rows : {16u, 63u, 64u, 65u, 127u, 128u, 129u, 1024u}) {
        std::fill(f.refg.begin(), f.refg.end(), -9876);
        std::fill(f.refout.begin(), f.refout.end(), -9876);
        std::fill(f.g.begin(), f.g.end(), -9876);
        std::fill(f.out.begin(), f.out.end(), -9876);
        for (unsigned r = 0; r < rows; ++r)
          for (unsigned c = 0; c < C; ++c)
            f.refg[size_t(r) * C + c] =
              f.truth[size_t(r) * N + c] * f.truth[size_t(r) * N + 2u * C + c];
        check(baseline::lfmConvPostGatePrefill(f.refg.data(), f.weights.data(),
                                               f.truth.data(), f.refout.data(),
                                               rows, C),
              "fused reference rejected");
        check(candidate::lfmConvFusedGatePrefill(f.truth.data(),
                                                 f.weights.data(), f.g.data(),
                                                 f.out.data(), rows, C),
              "fused stripe rejected");
        check(std::memcmp(f.g.data(), f.refg.data(), f.g.size() * 4) == 0,
              "fused pre-gate/history/canary differs");
        for (size_t i = 0; i < f.out.size(); ++i) {
          uint32_t got, want;
          std::memcpy(&got, &f.out[i], 4);
          std::memcpy(&want, &f.refout[i], 4);
#ifdef CONV_SIMDE_FIXTURE
          // x86 FMA lowering may choose another input NaN's payload. Keep
          // every finite/infinite/zero bit exact; native ARM checks all bits.
          const bool nan_pair = (got & 0x7fffffffu) > 0x7f800000u &&
                                (want & 0x7fffffffu) > 0x7f800000u;
          if (nan_pair)
            continue;
#endif
          check(got == want, "fused stripe output/canary differs");
        }
        ++cases;
      }
      for (unsigned reason = 0; reason < 8; ++reason) {
        const float *p = f.truth.data(), *w = f.weights.data();
        float *g = f.g.data(), *y = f.out.data();
        unsigned rows = 128, channels = C;
        if (reason == 0)
          p = nullptr;
        if (reason == 1)
          w = nullptr;
        if (reason == 2)
          g = nullptr;
        if (reason == 3)
          y = g;
        if (reason == 4)
          g = f.truth.data() + 1;
        if (reason == 5)
          y = f.weights.data();
        if (reason == 6)
          rows = 15;
        if (reason == 7)
          channels = 1024;
        check(!candidate::lfmConvFusedGatePrefill(p, w, g, y, rows, channels),
              "fused overlap/shape guard accepted");
        ++cases;
      }
    }
}

int main() {
  try {
    numeric();
    capacity();
    guards();
    failure();
    app_guards();
    unchanged_and_ranges();
    fused_gate_stripes();
    std::printf("CONV_CHUNK %u cases PASS\n", cases);
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "CONV_CHUNK FAIL: %s\n", e.what());
    return 1;
  }
}
