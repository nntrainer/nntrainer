// SPDX-License-Identifier: Apache-2.0
/**
 * @file lfm2_conv_ion_route_check.cpp
 * @brief Validate LFM retained projection dispatch and fallback ownership.
 * @author NNTrainer contributors
 * @bug Uses a mock backend; DSP arithmetic is checked by the device fixture.
 */
// Compile with route.inc extracted verbatim from the candidate's projection
// dispatch. The mock records the output contract and guards; no DSP is used.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static void check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}

namespace nntrainer {
enum class Tdatatype { FP32, FP16, QS4CX_WH_AFFINE, Q4_0 };
static unsigned cpu_calls;
template <typename T>
void gemm_q4_0(unsigned, unsigned, unsigned, float *, unsigned, char *,
               unsigned, float *, unsigned) {
  ++cpu_calls;
}
} // namespace nntrainer

static bool resident;
static bool htpDecodeRowResident(unsigned) { return resident; }

struct MockOps {
  enum View { Separate, Host, Null, Overlap, FailRpc, FailSync };
  View mode = Separate;
  bool supported = true;
  unsigned calls = 0, views = 0, syncs = 0;
  std::vector<float> truth;
  const float *overlap = nullptr;
  float *destination = nullptr;
  bool supports_gemm_q4_0_batch_norm_fp32() const { return supported; }
  void gemm_q4_0_batch_norm_fp32(
    std::vector<void *> weight, std::vector<float *> scale, float *input,
    std::vector<float *> output, unsigned rows, std::vector<unsigned> widths,
    unsigned k, const float *pre, const std::vector<unsigned> &post,
    const float *gamma, float eps, const float *rope, unsigned hd,
    unsigned rope_weights, bool to_host) {
    ++calls;
    check(weight.size() == 1 && scale.size() == 1 && weight[0] && scale[0] &&
            input && output.size() == 1 && output[0] && widths.size() == 1 &&
            widths[0] == 6144 && k == 2048 && pre == nullptr &&
            post == std::vector<unsigned>{0} && gamma == nullptr &&
            eps == 0.0f && rope == nullptr && hd == 0 && rope_weights == 0 &&
            !to_host,
          "FC contract changed");
    if (mode == FailRpc)
      throw std::runtime_error("injected RPC failure");
    destination = output[0];
    truth.resize(size_t(rows) * widths[0]);
    for (size_t i = 0; i < truth.size(); ++i)
      truth[i] = float(int(i % 257) - 128) * 0.125f;
    const size_t bytes = truth.size() * sizeof(float);
    if (mode == Host) {
      std::memcpy(destination, truth.data(), bytes);
      return;
    }
    auto *dst = reinterpret_cast<unsigned char *>(destination);
    const auto *src = reinterpret_cast<const unsigned char *>(truth.data());
    std::memcpy(dst + bytes - 64, src + bytes - 64, 64);
    for (unsigned i = 0; i < 32; ++i) {
      const size_t at = (bytes - 64) / 32 * i;
      std::memcpy(dst + at, src + at, 64);
    }
  }
  const void *host_view(const void *host, size_t bytes) {
    ++views;
    check(host == destination && bytes == truth.size() * sizeof(float),
          "host_view range");
    if (mode == Host)
      return host;
    if (mode == Null || mode == FailSync)
      return nullptr;
    return mode == Overlap ? overlap : truth.data();
  }
  void sync_to_host(void *host, size_t bytes) {
    ++syncs;
    if (mode == FailSync)
      throw std::runtime_error("injected sync failure");
    check(host == destination && bytes == truth.size() * sizeof(float),
          "sync range");
    std::memcpy(host, truth.data(), bytes);
  }
};

struct Tensor {
  float *data;
  nntrainer::Tdatatype type = nntrainer::Tdatatype::FP32;
  bool contiguous = true;
  unsigned dots = 0;
  float *replacement = nullptr;
  template <typename T> T *getData() const {
    return reinterpret_cast<T *>(data);
  }
  template <typename T> T *getScale() const {
    return reinterpret_cast<T *>(data);
  }
  nntrainer::Tdatatype getDataType() const { return type; }
  bool getContiguous() const { return contiguous; }
  void dot(const Tensor &, Tensor &output, bool, bool) {
    ++dots;
    if (replacement)
      output.data = replacement;
  }
};

struct Result {
  const float *p;
  bool retained, lfm;
};

static Result route(Tensor &in_step, Tensor &in_w, Tensor &out_w, Tensor &proj,
                    Tensor &gated, Tensor &conv_out, float *state, MockOps *ops,
                    unsigned rows, unsigned K, unsigned C, unsigned N,
                    bool compare = false) {
  const unsigned from = 0;
#include "route.inc"
  return {p, retain_projection, lfm_prefill};
}

struct Fixture {
  static constexpr unsigned C = 2048;
  unsigned rows;
  std::vector<float> storage, alternate;
  Tensor input, in_w, out_w, proj, gated, output;
  float *state;
  MockOps ops;
  explicit Fixture(unsigned m) :
    rows(m),
    storage(10u * size_t(std::max(m, 1u)) * C + 3u * C, -99.f),
    alternate(3u * size_t(std::max(m, 1u)) * C),
    input{storage.data()},
    in_w{storage.data() + storage.size() - C,
         nntrainer::Tdatatype::QS4CX_WH_AFFINE},
    out_w{in_w.data, nntrainer::Tdatatype::QS4CX_WH_AFFINE},
    proj{storage.data() + size_t(m) * C},
    gated{storage.data() + 4u * size_t(m) * C},
    output{storage.data() + 5u * size_t(m) * C},
    state(storage.data() + 8u * size_t(m) * C) {}
  Result run(unsigned K = C, unsigned N = C, unsigned channels = C,
             bool compare = false, bool null_ops = false) {
    return route(input, in_w, out_w, proj, gated, output, state,
                 null_ops ? nullptr : &ops, rows, K, channels, N, compare);
  }
};

int main() {
  unsigned cases = 0;
  for (unsigned rows : {16u, 17u, 64u, 1024u, 1920u}) {
    for (auto mode :
         {MockOps::Separate, MockOps::Host, MockOps::Null, MockOps::Overlap}) {
      Fixture f(rows);
      f.ops.mode = mode;
      f.ops.overlap = f.gated.data;
      const auto r = f.run();
      check(r.retained && r.lfm && f.ops.calls == 1 && f.ops.views == 1 &&
              f.input.dots == 0,
            "retained dispatch");
      check(f.ops.syncs ==
              unsigned(mode == MockOps::Null || mode == MockOps::Overlap),
            "fallback sync count");
      check(std::memcmp(r.p, f.ops.truth.data(),
                        f.ops.truth.size() * sizeof(float)) == 0,
            "view/fallback did not provide complete exact data");
      ++cases;
    }
  }
  for (unsigned reason = 0; reason < 23; ++reason) {
    Fixture f(reason == 0   ? 1
              : reason == 1 ? 15
              : reason == 2 ? 1921
              : reason == 3 ? 2048
                            : 16);
    unsigned K = 2048, N = 2048, C = 2048;
    if (reason == 4)
      K = 2816;
    if (reason == 5)
      N = 2816;
    if (reason == 6)
      C = 2816;
    if (reason == 7)
      f.in_w.type = nntrainer::Tdatatype::Q4_0;
    if (reason == 8)
      f.out_w.type = nntrainer::Tdatatype::Q4_0;
    if (reason == 9)
      f.input.type = nntrainer::Tdatatype::FP16;
    if (reason == 10)
      f.proj.type = nntrainer::Tdatatype::FP16;
    if (reason == 11)
      f.gated.type = nntrainer::Tdatatype::FP16;
    if (reason == 12)
      f.output.type = nntrainer::Tdatatype::FP16;
    if (reason == 13)
      f.input.contiguous = false;
    if (reason == 14)
      f.proj.contiguous = false;
    if (reason == 15)
      f.gated.contiguous = false;
    if (reason == 16)
      f.output.contiguous = false;
    if (reason == 17)
      f.ops.supported = false;
    if (reason == 18)
      f.gated.data = f.proj.data;
    if (reason == 19)
      f.output.data = f.proj.data + 1;
    if (reason == 20)
      f.state = f.proj.data;
    if (reason == 21)
      f.input.data = f.proj.data;
    const auto r = f.run(K, N, C, false, reason == 22);
    check(!r.retained && !f.ops.calls && f.input.dots == 1,
          "guard failed to take original projection");
    if (reason == 3)
      check(r.lfm, "2048 must retain conv-postgate optimization");
    ++cases;
  }
  {
    Fixture f(16);
    f.input.replacement = f.alternate.data();
    f.ops.supported = false;
    check(f.run().p == f.alternate.data(), "fallback dot may change output");
    ++cases;
  }
  {
    Fixture f(16);
    check(!f.run(2048, 2048, 2048, true).retained &&
            nntrainer::cpu_calls == 1 && !f.ops.calls && !f.input.dots,
          "compare path changed");
    ++cases;
  }
  {
    Fixture f(1);
    resident = true;
    check(!f.run().retained && !f.ops.calls && !f.input.dots,
          "resident decode changed");
    resident = false;
    ++cases;
  }
  for (auto mode : {MockOps::FailRpc, MockOps::FailSync}) {
    Fixture f(16);
    f.ops.mode = mode;
    bool threw = false;
    try {
      (void)f.run();
    } catch (const std::runtime_error &) {
      threw = true;
    }
    check(threw && f.input.dots == 0, "backend failure must propagate");
    ++cases;
  }
  std::printf("CONV_ION_ROUTE %u cases PASS\n", cases);
}
