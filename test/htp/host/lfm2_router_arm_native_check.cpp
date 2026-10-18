// SPDX-License-Identifier: Apache-2.0
/**
 * @file lfm2_router_arm_native_check.cpp
 * @brief lfm2 router arm native check validation.
 * @author NNTrainer contributors
 * @bug No known bugs.
 */
// Actual extracted app helper/caller, real core Tensor/HTP/ThreadManager and
// separately compiled production arithmetic. The RPC interposer forwards to
// the original shared-library stub and counts actual calls to that stub.
#include "lfm2_router_arm.h"
#include <atomic>
#include <compute_ops.h>
#include <cpu_backend.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <htp_backend.h>
#include <layer_context.h>
#include <memory>
#include <nntrainer_error.h>
#include <stdexcept>
#include <tensor.h>
#include <thread_manager.h>
#include <tuple>
#include <vector>
#include <weight.h>

static void check(bool ok, const char *why) {
  if (!ok)
    throw std::runtime_error(why);
}
static unsigned rpc_calls;
static bool fail_rpc;
extern "C" __attribute__((visibility("default"))) int
nntr_hvx_router_logits_f32(uint64_t session, uint32_t m, uint32_t k, uint32_t e,
                           float eps, const float *gamma, int gamma_len,
                           const float *w, int w_len, const float *x, int x_len,
                           uint32_t topk, uint32_t selected, const float *scale,
                           int scale_len, float *out, int out_len,
                           uint32_t *sel, int sel_len, float *weight,
                           int weight_len) {
  using Fn = decltype(&nntr_hvx_router_logits_f32);
  static Fn real =
    reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "nntr_hvx_router_logits_f32"));
  check(real && real != &nntr_hvx_router_logits_f32,
        "RPC interposer resolution");
  ++rpc_calls;
  if (fail_rpc)
    return 1;
  return real(session, m, k, e, eps, gamma, gamma_len, w, w_len, x, x_len, topk,
              selected, scale, scale_len, out, out_len, sel, sel_len, weight,
              weight_len);
}

namespace causallm {
struct AcceleratedRoute {
  unsigned int top_k = 0, n_sel = 0;
  const float *scale = nullptr;
  std::vector<unsigned int> sel;
  std::vector<float> weight;
};
#include "accelerator.inc"
#include "baseline.inc"
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
  const bool in_norm = p.in_norm, norm_on_accel = p.norm_on_accel;
  const bool out_add = p.out_add, router_norm = p.router_norm;
  const bool gelu_act = p.gelu_act, softmax_router = p.softmax_router;
  const unsigned num_experts = p.num_experts;
  const unsigned total_tokens = router_in.height(),
                 hidden_size = router_in.width();
  const unsigned router_gamma_idx = 128;
  const float norm_eps = 1e-5f;
  std::tuple<nntrainer::props::Unit> moe_props;
  std::get<nntrainer::props::Unit>(moe_props).set(p.inter);
  AcceleratedRoute route;
#include "caller.inc"
  check(router_ready, "native prefill router unexpectedly unavailable");
  return router_on_arm;
}
} // namespace causallm
#include "lfm2_router_arm_data.h"

static bool expectedArm(unsigned mode, bool enabled) {
  return enabled && mode != 1 && mode != 2 && mode != 3 && mode != 4 &&
         mode != 5 && !(mode >= 23 && mode <= 29) && mode < 36;
}
int main(int argc, char **argv) {
  try {
    const char *option = std::getenv("NNTR_LFM2_ROUTER_ARM");
    const bool enabled = option && std::strcmp(option, "1") == 0 &&
                         !std::getenv("NNTR_HTP_ROUTER_U8I8");
    nntrainer::ThreadManagerConfig config;
    config.compute_threads = argc == 2 ? std::stoul(argv[1]) : 8;
    nntrainer::ThreadManager::setConfig(config);
    check(nntrainer::HtpBackend::global().enabled(), "HTP unavailable");
    auto *ops = nntrainer::get_htp_ops();
    auto ct = std::make_shared<nntrainer::ContextData>();
    ct->setComputeOps(ops);
    std::vector<std::unique_ptr<nntrainer::Tensor>> retained_weights;
    unsigned cases = 0;
    for (unsigned rows : {1024u, 2048u}) {
      nntrainer::Tensor input(1, 1, rows, 2048), out(rows, 1, 1, 64);
      nntrainer::Tensor ref(rows, 1, 1, 64), gamma(1, 1, 1, 2048), empty;
      for (auto *t : {&input, &out, &ref, &gamma})
        t->setContextData(ct);
      for (size_t i = 0; i < gamma.size(); ++i)
        gamma.getData<float>()[i] = 1.0f;
      std::vector<std::unique_ptr<nntrainer::Weight>> storage;
      std::vector<nntrainer::Weight *> weights;
      std::vector<unsigned> gu, dn;
      for (unsigned i = 0; i < 128; ++i) {
        nntrainer::TensorDim dim(1, 1, i < 64 ? 2048 : 1536,
                                 i < 64 ? 3072 : 2048, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::TL2CX_WH);
        storage.emplace_back(
          new nntrainer::Weight(dim, nntrainer::Initializer::ZEROS,
                                nntrainer::WeightRegularizer::NONE, 0, 0, 0,
                                false, false, "metadata" + std::to_string(i)));
        check(!storage.back()->getVariableRef().isAllocated(),
              "metadata allocation");
        weights.push_back(storage.back().get());
      }
      storage.emplace_back(new nntrainer::Weight(gamma, empty, empty, "gamma"));
      weights.push_back(storage.back().get());
      for (unsigned i = 0; i < 64; ++i) {
        gu.push_back(i);
        dn.push_back(i + 64);
      }
      nntrainer::Var_Grad in(input, empty, "input"),
        output(out, empty, "output");
      nntrainer::RunLayerContext context("router", false, 0, false, 1, ct,
                                         false, weights, {&in}, {&output}, {});
      for (unsigned mode = 0; mode < 40; ++mode) {
        retained_weights.emplace_back(new nntrainer::Tensor(1, 1, 2048, 64));
        auto &w = *retained_weights.back();
        w.setContextData(ct);
        fill(input.getData<float>(), input.size(), mode, false);
        fill(w.getData<float>(), w.size(), mode, true);
        const std::vector<float> original(
          input.getData<float>(), input.getData<float>() + input.size());
        const std::vector<float> original_w(w.getData<float>(),
                                            w.getData<float>() + w.size());
        unsigned before = rpc_calls;
        check(causallm::routerLogitsOnAccelerator(input, w, nullptr, 1e-5f,
                                                  rows, 2048, 64, ref, nullptr),
              "reference unavailable");
        check(rpc_calls == before + 1, "reference must execute real DSP RPC");
        before = rpc_calls;
        const bool used =
          causallm::actualCaller(input, w, out, context, gu, dn, {});
        const bool expect = expectedArm(mode, enabled);
        check(used == expect, "actual caller path mismatch");
        check(rpc_calls == before + unsigned(!expect),
              "fallback actual RPC count");
        check(std::memcmp(ref.getData<float>(), out.getData<float>(),
                          out.size() * 4) == 0,
              "actual caller/DSP output bits");
        check(std::memcmp(original.data(), input.getData<float>(),
                          input.size() * 4) == 0,
              "input changed");
        check(
          std::memcmp(original_w.data(), w.getData<float>(), w.size() * 4) == 0,
          "weights changed");
        std::printf("EXACT M=%u mode=%u arm=%u rpc=%u PASS\n", rows, mode,
                    unsigned(used), rpc_calls - before);
        ++cases;
      }
      auto &w = *retained_weights.back();
      // Use a fresh immutable finite weight allocation: the HTP cache correctly
      // retains an earlier weight by pointer, so changing it in place is
      // invalid.
      retained_weights.emplace_back(new nntrainer::Tensor(1, 1, 2048, 64));
      auto &finite_w = *retained_weights.back();
      finite_w.setContextData(ct);
      fill(input.getData<float>(), input.size(), 0, false);
      fill(finite_w.getData<float>(), finite_w.size(), 0, true);
      for (unsigned policy = 0; policy < 6; ++policy) {
        causallm::Policy p;
        if (policy == 0)
          p.in_norm = false;
        if (policy == 1)
          p.norm_on_accel = true;
        if (policy == 2)
          p.out_add = true;
        if (policy == 3)
          p.gelu_act = true;
        if (policy == 4)
          p.inter = 704;
        if (policy == 5)
          p.router_norm = true;
        causallm::routerLogitsOnAccelerator(
          input, finite_w, p.router_norm ? gamma.getData<float>() : nullptr,
          1e-5f, rows, 2048, 64, ref, nullptr);
        unsigned before = rpc_calls;
        check(!causallm::actualCaller(input, finite_w, out, context, gu, dn, p),
              "policy accepted");
        check(rpc_calls == before + 1, "policy fallback RPC count");
        check(std::memcmp(ref.getData<float>(), out.getData<float>(),
                          out.size() * 4) == 0,
              "policy fallback bytes");
        ++cases;
      }
      // Special-value fallback failure must propagate after every worker has
      // joined. A subsequent finite call reuses the same pool successfully.
      unsigned before = rpc_calls;
      fail_rpc = true;
      bool threw = false;
      try {
        causallm::actualCaller(input, w, out, context, gu, dn, {});
      } catch (const std::runtime_error &) {
        threw = true;
      }
      fail_rpc = false;
      check(threw && rpc_calls == before + 1,
            "failed fallback did not propagate");
      check(causallm::actualCaller(input, finite_w, out, context, gu, dn, {}) ==
              enabled,
            "pool reuse after failed RPC");
      ++cases;
    }
    std::printf("ROUTER_ARM_NATIVE %u cases threads=%u env=%u RPC=%u PASS\n",
                cases, config.compute_threads, unsigned(enabled), rpc_calls);
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "ROUTER_ARM_NATIVE FAIL %s\n", e.what());
    return 1;
  }
}
