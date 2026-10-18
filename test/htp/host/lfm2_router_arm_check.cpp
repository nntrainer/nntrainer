// SPDX-License-Identifier: Apache-2.0
/**
 * @file lfm2_router_arm_check.cpp
 * @brief lfm2 router arm check validation.
 * @author NNTrainer contributors
 * @bug No known bugs.
 */
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <vector>
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/arm/neon.h>
// Only the extracted application guards use the target architecture macro.
#define __aarch64__ 1
#define ENABLE_HEXKL 1

static void check(bool ok, const char *why) {
  if (!ok)
    throw std::runtime_error(why);
}
static unsigned cases;
static bool enabled() {
  const char *s = std::getenv("NNTR_LFM2_ROUTER_ARM");
  return s && std::strcmp(s, "1") == 0 && !std::getenv("NNTR_HTP_ROUTER_U8I8");
}
extern "C" {
typedef struct hvx_worker_pool_s hvx_worker_pool;
typedef void (*hvx_worker_pool_func)(uint32_t, uint32_t, void *);
int hvx_router_rows_f32(const float *, const float *, float *, uint32_t,
                        uint32_t, uint32_t, hvx_worker_pool *);
int hvx_router_rows_norm_f32(const float *, const float *, float, const float *,
                             float *, float *, uint32_t, uint32_t, uint32_t,
                             hvx_worker_pool *);
void hvx_worker_pool_run(hvx_worker_pool *, hvx_worker_pool_func function,
                         void *ctx, uint32_t units) {
  unsigned lanes = std::min(3u, units);
  for (unsigned i = 0; i < lanes; ++i)
    function(lanes, i, ctx);
}
}
#include "lfm2_router_arm.h"
namespace nntrainer {
enum class Tdatatype { FP32, FP16, TL2CX_WH, QS4CX_WH };
enum class Tformat { NCHW, NHWC };
struct ContextData {};
struct ComputeOps {
  bool supported = true, router = true;
  unsigned calls = 0;
  bool fail = false, numerical = false;
  bool supports_gemm_qs4cx_moe_layer_fp32() const { return supported; }
  bool supports_router_logits_fp32() const { return router; }
  void router_logits_fp32(unsigned m, unsigned k, unsigned e, const float *x,
                          const float *gamma, float eps, const float *w,
                          float *out, unsigned, unsigned, const float *,
                          unsigned *, float *) {
    ++calls;
    if (fail)
      throw std::runtime_error("injected RPC failure");
    if (numerical)
      check(hvx_router_rows_norm_f32(x, gamma, eps, w, nullptr, out, m, k, e,
                                     nullptr) == 0,
            "HVX lane reference failed");
    else
      std::fill(out, out + size_t(m) * e, -8192.25f);
  }
};
static ComputeOps htp, other;
static ComputeOps *get_htp_ops() { return &htp; }
struct Tensor {
  std::array<unsigned, 4> dim{1, 1, 1, 2048};
  Tdatatype type = Tdatatype::FP32;
  Tformat format = Tformat::NCHW;
  bool contiguous = true;
  float *data = nullptr;
  ComputeOps *ops = &htp;
  std::shared_ptr<ContextData> context;
  unsigned batch() const { return dim[0]; }
  unsigned channel() const { return dim[1]; }
  unsigned height() const { return dim[2]; }
  unsigned width() const { return dim[3]; }
  size_t size() const { return size_t(dim[0]) * dim[1] * dim[2] * dim[3]; }
  auto getDim() const { return std::make_tuple(dim, type, format); }
  Tdatatype getDataType() const { return type; }
  Tformat getFormat() const { return format; }
  bool getContiguous() const { return contiguous; }
  ComputeOps *getOps() const { return ops; }
  const std::shared_ptr<ContextData> &getContextData() const { return context; }
  template <typename T> T *getData() const {
    return reinterpret_cast<T *>(data);
  }
};
struct RunLayerContext {
  unsigned inputs = 1;
  unsigned getNumInputs() const { return inputs; }
  std::vector<Tensor> weights;
  Tensor &getWeight(unsigned i) { return weights.at(i); }
};
namespace props {
struct Unit {
  unsigned value;
  unsigned get() const { return value; }
};
} // namespace props
struct ThreadManager {
  inline static unsigned lanes = 1, order = 0, dispatches = 0;
  inline static size_t last_jobs = 0;
  static ThreadManager &Global() {
    static ThreadManager tm;
    return tm;
  }
  template <typename F> void parallel_for(size_t begin, size_t end, F fn) {
    ++dispatches;
    last_jobs = end - begin;
    std::vector<size_t> jobs(end - begin);
    std::iota(jobs.begin(), jobs.end(), begin);
    if (order == 1)
      std::reverse(jobs.begin(), jobs.end());
    if (order == 2)
      std::shuffle(jobs.begin(), jobs.end(), std::mt19937(789));
    std::atomic<size_t> next{0};
    std::exception_ptr error;
    std::mutex lock;
    auto work = [&] {
      try {
        size_t i;
        while ((i = next.fetch_add(1)) < jobs.size()) {
          if ((jobs[i] & 1) == 0)
            std::this_thread::yield();
          fn(jobs[i]);
        }
      } catch (...) {
        std::lock_guard<std::mutex> guard(lock);
        error = std::current_exception();
      }
    };
    std::vector<std::thread> workers;
    for (unsigned i = 1; i < lanes; ++i)
      workers.emplace_back(work);
    work();
    for (auto &w : workers)
      w.join();
    if (error)
      std::rethrow_exception(error);
  }
};
} // namespace nntrainer
namespace ml {
namespace train {
struct TensorDim {
  using DataType = nntrainer::Tdatatype;
};
} // namespace train
} // namespace ml
namespace causallm {
struct AcceleratedRoute {
  unsigned top_k = 0, n_sel = 0;
  const float *scale = nullptr;
  std::vector<unsigned> sel;
  std::vector<float> weight;
};
static void normRows(const nntrainer::Tensor &, nntrainer::Tensor &,
                     const nntrainer::Tensor &, float) {
  throw std::runtime_error("unexpected norm");
}
#include "accelerator.inc"
#include "helper.inc"
struct Policy {
  bool in_norm = true, norm_on_accel = false, out_add = false;
  bool router_norm = false, gelu_act = false, softmax_router = false;
  unsigned num_experts = 64, inter = 1536;
};
static bool actualCaller(
  nntrainer::Tensor &router_in, const nntrainer::Tensor &gate_weights,
  nntrainer::Tensor &router_logits, nntrainer::RunLayerContext &context,
  const std::vector<unsigned> &expert_gate_up_proj_indices,
  const std::vector<unsigned> &expert_down_proj_indices, const Policy &p) {
  auto &raw = router_in;
  const bool in_norm = p.in_norm, norm_on_accel = p.norm_on_accel,
             out_add = p.out_add;
  const bool router_norm = p.router_norm, gelu_act = p.gelu_act,
             softmax_router = p.softmax_router;
  const unsigned num_experts = p.num_experts, total_tokens = router_in.height(),
                 hidden_size = router_in.width();
  const unsigned router_gamma_idx = 128;
  const float norm_eps = 1e-5f;
  const auto moe_props = std::make_tuple(nntrainer::props::Unit{p.inter});
  AcceleratedRoute route;
#include "caller.inc"
  check(router_ready, "router not ready");
  return router_on_arm;
}
} // namespace causallm
#include "lfm2_router_arm_data.h"

struct Data {
  std::vector<float> x, w, y, ref, gamma;
  nntrainer::Tensor input, weight, output;
  nntrainer::RunLayerContext context;
  std::vector<unsigned> gu, dn;
  explicit Data(unsigned rows) :
    x(size_t(rows) * 2048 + 32),
    w(2048 * 64 + 32),
    y(size_t(rows) * 64 + 32),
    ref(y.size()),
    gamma(2048, 1.0f) {
    input.dim = {1, 1, rows, 2048};
    weight.dim = {1, 1, 2048, 64};
    output.dim = {rows, 1, 1, 64};
    input.data = x.data() + 16;
    weight.data = w.data() + 16;
    output.data = y.data() + 16;
    input.context = weight.context = output.context =
      std::make_shared<nntrainer::ContextData>();
    context.weights.resize(129);
    for (unsigned i = 0; i < 64; ++i) {
      gu.push_back(i);
      dn.push_back(i + 64);
      context.weights[i].type = context.weights[i + 64].type =
        nntrainer::Tdatatype::TL2CX_WH;
    }
    context.weights[128].data = gamma.data();
  }
  void reset(unsigned mode = 0) {
    fill(input.data, input.size(), mode, false);
    fill(weight.data, weight.size(), mode, true);
    std::fill(y.begin(), y.end(), -1234.5f);
    std::fill(ref.begin(), ref.end(), -1234.5f);
    nntrainer::ThreadManager::dispatches = 0;
    nntrainer::ThreadManager::last_jobs = 0;
    nntrainer::htp.calls = 0;
    nntrainer::htp.fail = false;
    nntrainer::htp.supported = true;
    nntrainer::htp.numerical = false;
    nntrainer::htp.router = true;
  }
  bool helper() {
    return causallm::lfmRouterOnArm(input, weight, output, context, gu, dn);
  }
  bool caller(const causallm::Policy &p = {}) {
    return causallm::actualCaller(input, weight, output, context, gu, dn, p);
  }
};

int main() {
  try {
    // Actual strict kernel against the real HVX router C TU on lane emulation.
    // Hardware special/subnormal behavior is independently checked by native
    // DSP.
    for (unsigned mode = 0; mode < 40; ++mode) {
      Data d(4);
      d.reset(mode);
      bool accepted = causallm::lfm2RouterArmWeightSupported(d.weight.data);
      if (accepted)
        accepted = causallm::lfm2RouterArmBlock(d.input.data, d.weight.data,
                                                d.output.data);
      const bool expected = mode != 1 && mode != 2 && mode != 3 && mode != 4 &&
                            mode != 5 && !(mode >= 23 && mode <= 29) &&
                            mode < 36;
      check(accepted == expected, "primitive special qualification");
      if (accepted) {
        check(hvx_router_rows_f32(d.input.data, d.weight.data,
                                  d.ref.data() + 16, 4, 2048, 64, nullptr) == 0,
              "reference failure");
        check(std::memcmp(d.ref.data(), d.y.data(), d.ref.size() * 4) == 0,
              "primitive bits/canary mismatch");
      }
      ++cases;
    }
    for (unsigned rows : {1024u, 2048u}) {
      Data d(rows);
      d.reset();
      hvx_router_rows_f32(d.input.data, d.weight.data, d.ref.data() + 16, rows,
                          2048, 64, nullptr);
      const auto reference = d.ref, original_x = d.x, original_w = d.w;
      for (unsigned lanes : {1u, 8u})
        for (unsigned order : {0u, 1u, 2u}) {
          d.reset();
          nntrainer::ThreadManager::lanes = lanes;
          nntrainer::ThreadManager::order = order;
          nntrainer::htp.numerical = true;
          const bool used = d.caller();
          check(used == enabled(), "actual caller qualification");
          check(nntrainer::htp.calls == unsigned(!enabled()),
                "actual caller RPC count");
          check(nntrainer::ThreadManager::dispatches == unsigned(enabled()),
                "actual caller jobs");
          if (enabled())
            check(nntrainer::ThreadManager::last_jobs == rows / 4,
                  "task coverage");
          check(std::memcmp(d.y.data(), reference.data(), d.y.size() * 4) == 0,
                "whole caller numeric/canary bits");
          check(std::memcmp(d.x.data(), original_x.data(), d.x.size() * 4) == 0,
                "input write");
          check(std::memcmp(d.w.data(), original_w.data(), d.w.size() * 4) == 0,
                "weight write");
          ++cases;
        }
      // Early task refusal or a post-arithmetic failure must join all tasks and
      // let the actual caller's original accelerator function overwrite all
      // rows.
      for (unsigned mode : {3u, 5u, 36u, 37u, 38u, 39u}) {
        d.reset(mode);
        nntrainer::ThreadManager::lanes = 8;
        nntrainer::ThreadManager::order = 2;
        check(!d.caller(), "special caller used ARM");
        check(nntrainer::htp.calls == 1, "special RPC count");
        for (size_t i = 0; i < d.output.size(); ++i)
          check(bits(d.output.data[i]) == bits(-8192.25f),
                "partial fallback overwrite");
        nntrainer::htp.fail = true;
        bool threw = false;
        try {
          d.caller();
        } catch (const std::runtime_error &) {
          threw = true;
        }
        check(threw && nntrainer::htp.calls == 2, "failed RPC propagation");
        nntrainer::htp.fail = false;
        ++cases;
      }
      for (unsigned policy = 0; policy < 8; ++policy) {
        d.reset();
        causallm::Policy p;
        if (policy == 0)
          p.in_norm = false;
        if (policy == 1)
          p.norm_on_accel = true;
        if (policy == 2)
          p.out_add = true;
        if (policy == 3)
          p.router_norm = true;
        if (policy == 4)
          p.gelu_act = true;
        if (policy == 5)
          p.softmax_router = true;
        if (policy == 6)
          p.inter = 704;
        if (policy == 7)
          d.context.inputs = 2;
        check(!d.caller(p), "unsupported caller policy");
        check(nntrainer::htp.calls == 1 &&
                nntrainer::ThreadManager::dispatches == 0,
              "policy changed original route");
        ++cases;
      }
    }
    // Every metadata rejection precedes any dispatch or output write. Alias
    // cases use valid storage in both directions rather than forged addresses.
    for (unsigned bad = 0; bad < 40; ++bad) {
      Data d(1024);
      d.reset();
      using nntrainer::Tdatatype;
      using nntrainer::Tformat;
      switch (bad) {
      case 0:
        d.input.dim[2] = 1;
        break;
      case 1:
        d.input.dim[2] = 512;
        break;
      case 2:
        d.input.dim[2] = 2047;
        break;
      case 3:
        d.input.dim[0] = 2;
        break;
      case 4:
        d.input.dim[1] = 2;
        break;
      case 5:
        d.input.dim[3] = 2816;
        break;
      case 6:
        d.input.format = Tformat::NHWC;
        break;
      case 7:
        d.input.type = Tdatatype::FP16;
        break;
      case 8:
        d.weight.dim[0] = 2;
        break;
      case 9:
        d.weight.dim[1] = 2;
        break;
      case 10:
        d.weight.dim[2] = 2047;
        break;
      case 11:
        d.weight.dim[3] = 63;
        break;
      case 12:
        d.weight.type = Tdatatype::FP16;
        break;
      case 13:
        d.weight.format = Tformat::NHWC;
        break;
      case 14:
        d.output.dim[0] = 1023;
        break;
      case 15:
        d.output.dim[1] = 2;
        break;
      case 16:
        d.output.dim[2] = 2;
        break;
      case 17:
        d.output.dim[3] = 63;
        break;
      case 18:
        d.output.type = Tdatatype::FP16;
        break;
      case 19:
        d.output.format = Tformat::NHWC;
        break;
      case 20:
        d.input.contiguous = false;
        break;
      case 21:
        d.weight.contiguous = false;
        break;
      case 22:
        d.output.contiguous = false;
        break;
      case 23:
        d.gu.pop_back();
        break;
      case 24:
        d.dn.push_back(1);
        break;
      case 25:
        d.input.ops = nullptr;
        break;
      case 26:
        d.output.ops = &nntrainer::other;
        break;
      case 27:
        d.input.ops = d.output.ops = &nntrainer::other;
        break;
      case 28:
        nntrainer::htp.supported = false;
        break;
      case 29:
        nntrainer::htp.router = false;
        break;
      case 30:
        d.weight.context = std::make_shared<nntrainer::ContextData>();
        break;
      case 31:
        d.output.context = std::make_shared<nntrainer::ContextData>();
        break;
      case 32:
        d.context.weights[63].type = Tdatatype::QS4CX_WH;
        break;
      case 33:
        d.context.weights[127].type = Tdatatype::QS4CX_WH;
        break;
      case 34:
        d.input.data = nullptr;
        break;
      case 35:
        d.output.data = nullptr;
        break;
      case 36:
        d.weight.data = nullptr;
        break;
      case 37:
        d.output.data = d.input.data + 1;
        break;
      case 38:
        d.output.data = d.weight.data - 1;
        break;
      case 39:
        d.output.data = d.input.data - 1;
        break;
      }
      const auto x = d.x, w = d.w, y = d.y;
      check(!d.helper(), "metadata guard accepted");
      check(nntrainer::ThreadManager::dispatches == 0,
            "rejected helper dispatched");
      check(d.x == x && d.w == w && d.y == y, "rejected helper wrote memory");
      ++cases;
    }
    std::printf("ROUTER_ARM_HOST %u cases env=%u PASS\n", cases,
                unsigned(enabled()));
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "ROUTER_ARM_HOST FAIL %s\n", e.what());
    return 1;
  }
}
