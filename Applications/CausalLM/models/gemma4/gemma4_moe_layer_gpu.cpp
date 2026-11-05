// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   gemma4_moe_layer_gpu.cpp
 * @date   02 October 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Gemma4 sparse MoE layer for the OpenCL (gpu) engine.
 */

#include <gemma4_moe_layer_gpu.h>

// Built only with the OpenCL backend; a build list that names this file
// unconditionally (the Android app library) gets an empty unit otherwise.
#if defined(ENABLE_OPENCL)

#include <blas_kernel_interface.h> // clmem_lower_cl / clmem_raise_cl
#include <blas_kernels.h>
#include <compute_ops.h>
#include <cpu_backend.h>
#include <fp16.h>
#include <layer_prof.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>

#include <qs4cx_tensor.h>   // the released-payload registry
#include <ternary_tensor.h> // tq2cxSingleCopyOn

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <unordered_set>
#if !defined(_WIN32)
#include <sys/mman.h>
#endif
#include <stdexcept>
#include <vector>

namespace causallm {

namespace {

/**
 * @brief Holds a shared-plane tensor for a host read.
 * @details A kernel still in the queue may write the region (its producer):
 * map it, which blocks until the queue has run up to it, and give it back to
 * the device afterwards. Nothing to do when the queue last drained (every
 * kernel output is then complete and mapped back for the host) or for a
 * tensor that is not on the shared plane.
 */
class HostRead {
public:
  explicit HostRead(const nntrainer::Tensor &t) {
    const auto md = t.getMemoryData();
    if (!md || !md->isSVM() || t.isClMem())
      return;
    if (!nntrainer::cl_queue_may_be_busy())
      return;
    ptr_ = const_cast<void *>(static_cast<const void *>(t.getData<uint8_t>()));
    nntrainer::cl_svm_map_force(ptr_, t.bytes(), /*read_only=*/true);
  }
  HostRead(const HostRead &) = delete;
  HostRead &operator=(const HostRead &) = delete;
  ~HostRead() {
    if (ptr_ != nullptr)
      nntrainer::cl_svm_unmap_force(ptr_);
  }

private:
  void *ptr_ = nullptr;
};

bool isHalf(const nntrainer::Tensor &t) {
  return t.getDataType() == nntrainer::TensorDim::DataType::FP16;
}

/**
 * @brief Host fallback of moe_gather_rows, for an input on host memory.
 */
void hostGather(const nntrainer::Tensor &in, nntrainer::Tensor &out,
                const std::vector<unsigned int> &rows, unsigned int width) {
  HostRead hold(in);
  const size_t elem = isHalf(in) ? sizeof(uint16_t) : sizeof(float);
  const size_t bytes = (size_t)rows.size() * width * elem;
  uint8_t *dst = out.getData<uint8_t>();
  const bool svm = out.getMemoryData() && out.getMemoryData()->isSVM();
  if (svm)
    nntrainer::cl_svm_host_write_begin(dst, bytes);
  const uint8_t *src = in.getData<uint8_t>();
  for (size_t r = 0; r < rows.size(); ++r)
    std::memcpy(dst + r * width * elem, src + (size_t)rows[r] * width * elem,
                width * elem);
  if (svm)
    nntrainer::cl_svm_host_write_end(dst, bytes);
}

/**
 * @brief Host fallback of moe_scatter_add_rows, for an output on host memory.
 */
void hostScatterAdd(const nntrainer::Tensor &expert_out, nntrainer::Tensor &out,
                    const std::vector<unsigned int> &rows,
                    const std::vector<float> &weights, unsigned int width) {
  HostRead hold(expert_out);
  if (isHalf(expert_out)) {
    const uint16_t *src = expert_out.getData<uint16_t>();
    uint16_t *dst = out.getData<uint16_t>();
    for (size_t r = 0; r < rows.size(); ++r)
      for (unsigned int j = 0; j < width; ++j) {
        uint16_t &d = dst[(size_t)rows[r] * width + j];
        const float v =
          nntrainer::compute_fp16_to_fp32(d) +
          weights[r] * nntrainer::compute_fp16_to_fp32(src[r * width + j]);
        d = nntrainer::compute_fp32_to_fp16(
          std::min(65504.0f, std::max(-65504.0f, v)));
      }
    return;
  }
  const float *src = expert_out.getData<float>();
  float *dst = out.getData<float>();
  for (size_t r = 0; r < rows.size(); ++r)
    nntrainer::saxpy(width, weights[r], src + r * width, 1,
                     dst + (size_t)rows[r] * width, 1);
}

/**
 * @brief Leave the layer output with the device when the expert half ends.
 * @details The row ops map their output back for the host after each launch
 * (the shared-memory protocol of this backend). The MoE output's consumer
 * is the next device op (the post-expert norm); measured on Intel Xe3, that
 * norm read stale rows unless the last map-back was undone or made blocking,
 * so the output is handed back to the device here -- one enqueue, no drain.
 * A host reader maps it itself (it does whenever kernels may be in flight).
 */
void handOutputToDevice(const nntrainer::Tensor &output) {
  const auto md = output.getMemoryData();
  if (md && md->isSVM() && !output.isClMem())
    nntrainer::cl_svm_unmap_force(
      const_cast<uint8_t *>(output.getData<uint8_t>()));
}

/**
 * @brief Route a decode row on the device (NNTR_G4MOE_DEVICE_ROUTER=0 routes
 *        every step on the host).
 * @details Default on: the device kernels and the host router are the same
 * deterministic routine (moe_route_det.h), so they select the same experts
 * with the same weights, and a decode step on the device never waits for
 * the host. The host router serves prefill and any step the backend declines.
 */
bool deviceRouterOn() {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_G4MOE_DEVICE_ROUTER");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}

} // namespace

Gemma4MoELayerGpu::Gemma4MoELayerGpu() :
  Gemma4MoELayer(),
  gathered_idx(0),
  gate_out_idx(0),
  up_out_idx(0),
  act_out_idx(0),
  down_out_idx(0),
  prebuilt(false),
  exact_gelu(false) {}

Gemma4MoELayerGpu::~Gemma4MoELayerGpu() = default;

void Gemma4MoELayerGpu::finalize(nntrainer::InitLayerContext &context) {
  Gemma4MoELayer::finalize(context);

  NNTR_THROW_IF(cache_size > 0, std::invalid_argument)
    << "Gemma4 MoE on the gpu engine keeps every expert resident; "
       "moe_cache_size must be 0 (the virtual expert cache runs on the cpu "
       "engine only)";
  const auto act = std::get<props::MoEActivation>(moe_props).get();
  NNTR_THROW_IF(act != nntrainer::ActivationType::ACT_TANH_GELU &&
                  act != nntrainer::ActivationType::ACT_GELU,
                std::invalid_argument)
    << "Gemma4 MoE on the gpu engine supports the tanh_gelu and gelu expert "
       "gates only";
  exact_gelu = act == nntrainer::ActivationType::ACT_GELU;

  const auto &expert_dim = context.getInputDimensions()[0];
  const unsigned int rows = expert_dim.batch() * expert_dim.height();
  const unsigned int hidden_size = expert_dim.width();
  const unsigned int intermediate_size =
    std::get<nntrainer::props::Unit>(moe_props).get();

  // Workspace of the expert loop. Requested from the context, not built on
  // the heap: a context tensor of a gpu-engine layer is on the shared plane
  // and resolves the gpu op table, which the gather / FC / scatter kernels
  // need; a heap tensor would silently dispatch to the host.
  // in the data type of the rows it moves (the activation type)
  const nntrainer::TensorDim::TensorType act_type(context.getFormat(),
                                                  expert_dim.getDataType());
  auto request = [&](unsigned int width, const char *name) {
    return context.requestTensor(
      nntrainer::TensorDim({1, 1, rows, width}, act_type), name,
      nntrainer::Initializer::NONE, false,
      nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  };
  gathered_idx = request(hidden_size, "expert_gathered");
  gate_out_idx = request(intermediate_size, "expert_gate_out");
  up_out_idx = request(intermediate_size, "expert_up_out");
  act_out_idx = request(intermediate_size, "expert_act_out");
  down_out_idx = request(hidden_size, "expert_down_out");
}

void Gemma4MoELayerGpu::prebuildExpertWeights(
  nntrainer::RunLayerContext &context) {
  if (prebuilt)
    return;
  auto *ops = context.getWeight(expert_gate_proj_indices[0]).getOps();
  for (unsigned int e = 0; e < num_experts; ++e) {
    ops->fc_prebuild_weight(context.getWeight(expert_gate_proj_indices[e]));
    ops->fc_prebuild_weight(context.getWeight(expert_up_proj_indices[e]));
    ops->fc_prebuild_weight(context.getWeight(expert_down_proj_indices[e]));
  }
  prebuilt = true;
}

nntrainer::MoeExpertGroup
Gemma4MoELayerGpu::expertWeightGroup(nntrainer::RunLayerContext &context) {
  nntrainer::MoeExpertGroup group;
  group.layer_key = this;
  group.gate.resize(num_experts);
  group.up.resize(num_experts);
  group.down.resize(num_experts);
  for (unsigned int e = 0; e < num_experts; ++e) {
    group.gate[e] = &context.getWeight(expert_gate_proj_indices[e]);
    group.up[e] = &context.getWeight(expert_up_proj_indices[e]);
    group.down[e] = &context.getWeight(expert_down_proj_indices[e]);
  }
  group.exact_gelu = exact_gelu;
  return group;
}

bool Gemma4MoELayerGpu::buildDeviceExperts(
  nntrainer::RunLayerContext &context,
  const std::vector<const void *> *records) {
  if (!nntrainer::tq2cxSingleCopyOn() || num_experts == 0)
    return false;
  nntrainer::MoeExpertGroup group = expertWeightGroup(context);
  if (group.gate[0]->getDataType() != nntrainer::TensorDim::DataType::TQ2CX)
    return false;
  group.records = records;
  // The layer's op table: a self-owned (heap) weight does not name one.
  nntrainer::ComputeOps *ops = context.getComputeOps();
  if (ops == nullptr || !ops->moe_experts_prebuild(group)) {
    ml_logw("Gemma4 MoE %s: the backend keeps no device copy of the experts "
            "(%s); they stay in host memory",
            context.getName().c_str(),
            ops == nullptr ? "no op table" : "declined");
    return false;
  }

  // The device stack owns the experts now. A filled payload is a heap block
  // of its own (the manager's single-copy arm): give its pages back, INWARD
  // page aligned. Either way record the range, so a host reader refuses it.
  auto release = [&](nntrainer::Tensor *w) {
    uint8_t *p = w->getData<uint8_t>();
    const size_t bytes = w->getMemoryBytes();
    if (records == nullptr) {
#if !defined(_WIN32)
      const uintptr_t page = 4096;
      const uintptr_t lo = ((uintptr_t)p + page - 1) & ~(page - 1);
      const uintptr_t hi = ((uintptr_t)p + bytes) & ~(page - 1);
      if (hi > lo)
        (void)::madvise((void *)lo, (size_t)(hi - lo), MADV_DONTNEED);
#endif
    }
    nntrainer::markQs4cxPayloadDropped(p, bytes);
  };
  for (unsigned int e = 0; e < num_experts; ++e) {
    release(group.gate[e]);
    release(group.up[e]);
    release(group.down[e]);
  }
  return true;
}

void Gemma4MoELayerGpu::read(
  nntrainer::ReadSource src, nntrainer::RunLayerContext &run_context,
  bool opt_var, ml::train::ExecutionMode mode, bool trainable,
  nntrainer::TensorDim::DataType defineWeightDataType, bool fsu,
  size_t start_offset, bool read_from_offset, int file_fd) {
  const char *const *mapping = std::get_if<const char *>(&src);
  const bool eligible = !opt_var && !fsu && nntrainer::tq2cxSingleCopyOn();
  if (eligible && read_from_offset && mapping != nullptr &&
      *mapping != nullptr) {
    // Upload the expert records straight from the mapping, before anything
    // is copied: the host payloads are then never filled.
    std::vector<const void *> records;
    records.reserve(3 * num_experts);
    for (unsigned int e = 0; e < num_experts; ++e)
      for (unsigned int idx :
           {expert_gate_proj_indices[e], expert_up_proj_indices[e],
            expert_down_proj_indices[e]})
        records.push_back(*mapping +
                          run_context.getWeight(idx).getFileOffset());
    if (buildDeviceExperts(run_context, &records)) {
      std::unordered_set<unsigned int> experts;
      for (unsigned int e = 0; e < num_experts; ++e) {
        experts.insert(expert_gate_proj_indices[e]);
        experts.insert(expert_up_proj_indices[e]);
        experts.insert(expert_down_proj_indices[e]);
      }
      for (unsigned int i = 0; i < run_context.getNumWeights(); ++i)
        if (!experts.count(i) && run_context.isGradientFirstAccess(i))
          run_context.getWeight(i).read(src, start_offset, read_from_offset,
                                        file_fd);
      return;
    }
  }
  Gemma4MoELayer::read(src, run_context, opt_var, mode, trainable,
                       defineWeightDataType, fsu, start_offset,
                       read_from_offset, file_fd);
  if (eligible)
    buildDeviceExperts(run_context, nullptr);
}

void Gemma4MoELayerGpu::read(
  std::ifstream &file, nntrainer::RunLayerContext &run_context, bool opt_var,
  ml::train::ExecutionMode mode, bool trainable,
  nntrainer::TensorDim::DataType defineWeightDataType, bool fsu,
  size_t start_offset, bool read_from_offset, int file_fd) {
  Gemma4MoELayer::read(file, run_context, opt_var, mode, trainable,
                       defineWeightDataType, fsu, start_offset,
                       read_from_offset, file_fd);
  if (!opt_var && !fsu)
    buildDeviceExperts(run_context, nullptr);
}

void Gemma4MoELayerGpu::forwardTensors(nntrainer::RunLayerContext &context,
                                       nntrainer::Tensor &expert_input,
                                       nntrainer::Tensor &router_input,
                                       nntrainer::Tensor &output,
                                       nntrainer::Tensor &router_input_scaled,
                                       nntrainer::Tensor &router_logits) {
  // The router runs on host scratch (below), not on these: they are gpu
  // context tensors whose op table has no host GEMM.
  (void)router_input_scaled;
  (void)router_logits;

  const unsigned int total_tokens =
    expert_input.batch() * expert_input.height();
  const unsigned int hidden_size = expert_input.width();
  expert_input.reshape({total_tokens, 1, 1, hidden_size});
  router_input.reshape({total_tokens, 1, 1, hidden_size});
  output.reshape({total_tokens, 1, 1, hidden_size});

  // NNTR_LAYER_PROFILE scopes: the host read of the router rows waits for
  // the queue to reach them; the routing itself is host compute.
  const bool decode_step = total_tokens == 1;
  nntrainer::ComputeOps *ops = expert_input.getOps();

  // ---- decode: route on the device -------------------------------------
  // The whole step stays on the queue: no host read of the router row, no
  // drain. A backend that cannot route on the device declines having
  // enqueued nothing, and the step takes the host router below.
  if (decode_step && deviceRouterOn()) {
    nntrainer::MoeRouter rt;
    rt.input = &router_input;
    rt.weight = &context.getWeight(router_idx);
    rt.scale = &context.getWeight(router_scale_idx);
    rt.norm = router_norm_idx != std::numeric_limits<unsigned int>::max()
                ? &context.getWeight(router_norm_idx)
                : nullptr;
    rt.per_expert_scale = &context.getWeight(per_expert_scale_idx);
    rt.epsilon = epsilon;
    rt.topk = topk;
    nntrainer::MoeExpertGroup group = expertWeightGroup(context);
    group.input = &expert_input;
    group.output = &output;
    group.tokens = 1;
    group.router = &rt;
    bool ran = false;
    {
      nntrainer::LayerProfScope prof_exp("g4moe_experts_enqueue", true);
      ran = ops->moe_experts(group);
    }
    if (ran) {
      handOutputToDevice(output);
      return;
    }
  }

  // ---- routing, on the host --------------------------------------------
  // The same code as the cpu layer on host tensors, so the selected experts,
  // their order and their weights are identical to the cpu engine's.
  Routing routing;
  hostRoute(context, router_input, total_tokens, hidden_size, routing);

  // ---- experts, on the device ------------------------------------------
  // An expert format whose kernels batch several experts per launch runs the
  // whole expert half in one call (see ComputeOps::moe_experts); the others
  // take the per-expert loop below.
  {
    nntrainer::MoeExpertGroup group = expertWeightGroup(context);
    group.input = &expert_input;
    group.output = &output;
    group.tokens = total_tokens;
    group.active = routing.active;
    group.assignments = &routing.assignments;
    nntrainer::LayerProfScope prof_exp("g4moe_experts_enqueue", decode_step);
    if (ops->moe_experts(group)) {
      handOutputToDevice(output);
      return;
    }
  }

  // The per-expert loop works on the shared plane: bring a device-only
  // input back to it, and the output to the device at the end.
  nntrainer::clmem_lower_cl(expert_input, 0);
  forwardExpertLoop(context, expert_input, output, routing, total_tokens,
                    hidden_size);
  nntrainer::clmem_raise_cl(output, 0);
}

void Gemma4MoELayerGpu::hostRoute(nntrainer::RunLayerContext &context,
                                  nntrainer::Tensor &router_input,
                                  unsigned int total_tokens,
                                  unsigned int hidden_size, Routing &routing) {
  const bool decode_step = total_tokens == 1;
  if (host_router_scaled.empty() ||
      host_router_scaled.batch() != total_tokens) {
    host_router_scaled = nntrainer::Tensor(total_tokens, 1, 1, hidden_size);
    host_router_logits = nntrainer::Tensor(total_tokens, 1, 1, num_experts);
  }
  // a device-only router row is read back first (a blocking read)
  nntrainer::clmem_lower_cl(router_input, 0);
  if (isHalf(router_input)) {
    // FP16 rows: widen them into host FP32 scratch, the router's input type
    if (host_router_input.empty() || host_router_input.batch() != total_tokens)
      host_router_input = nntrainer::Tensor(total_tokens, 1, 1, hidden_size);
    {
      nntrainer::LayerProfScope prof_wait("g4moe_router_wait", decode_step);
      HostRead hold(router_input);
      const uint16_t *src = router_input.getData<uint16_t>();
      float *dst = host_router_input.getData<float>();
      const size_t n = (size_t)total_tokens * hidden_size;
      for (size_t i = 0; i < n; ++i)
        dst[i] = nntrainer::compute_fp16_to_fp32(src[i]);
    }
    nntrainer::LayerProfScope prof_route("g4moe_router_host", decode_step);
    routeTokens(context, host_router_input, host_router_scaled,
                host_router_logits, total_tokens, hidden_size, routing);
  } else {
    HostRead hold(router_input);
    routeTokens(context, router_input, host_router_scaled, host_router_logits,
                total_tokens, hidden_size, routing);
  }
}

void Gemma4MoELayerGpu::forwardExpertLoop(nntrainer::RunLayerContext &context,
                                          nntrainer::Tensor &expert_input,
                                          nntrainer::Tensor &output,
                                          const Routing &routing,
                                          unsigned int total_tokens,
                                          unsigned int hidden_size) {
  nntrainer::ComputeOps *ops = expert_input.getOps();
  prebuildExpertWeights(context);

  nntrainer::Tensor &gathered_all = context.getTensor(gathered_idx);
  nntrainer::Tensor &gate_all = context.getTensor(gate_out_idx);
  nntrainer::Tensor &up_all = context.getTensor(up_out_idx);
  nntrainer::Tensor &act_all = context.getTensor(act_out_idx);
  nntrainer::Tensor &down_all = context.getTensor(down_out_idx);
  const auto tensor_type = gathered_all.getTensorType();
  const unsigned int intermediate_size = gate_all.width();

  // Device row movement when the planes allow it. An operand on host memory
  // (a host producer upstream, or a host consumer downstream) keeps the host
  // loop for that one step; the order of the additions is the same either way.
  const bool device_output = ops->moe_zero_rows(output, total_tokens);
  if (!device_output)
    output.setZero();

  std::vector<unsigned int> rows;
  std::vector<float> weights;
  for (const unsigned int expert : routing.active) {
    const auto &assigned = routing.assignments[expert];
    const unsigned int n = static_cast<unsigned int>(assigned.size());
    rows.resize(n);
    weights.resize(n);
    for (unsigned int i = 0; i < n; ++i) {
      rows[i] = assigned[i].first;
      weights[i] = assigned[i].second;
    }

    // Views over the workspace base: the FC reads M from the view height,
    // and a device plane can only be bound from its base.
    nntrainer::TensorDim hidden_dim({1, 1, n, hidden_size}, tensor_type);
    nntrainer::TensorDim inter_dim({1, 1, n, intermediate_size}, tensor_type);
    nntrainer::Tensor gathered =
      gathered_all.getSharedDataTensor(hidden_dim, 0, true);
    nntrainer::Tensor gate_out =
      gate_all.getSharedDataTensor(inter_dim, 0, true);
    nntrainer::Tensor up_out = up_all.getSharedDataTensor(inter_dim, 0, true);
    nntrainer::Tensor act_out = act_all.getSharedDataTensor(inter_dim, 0, true);
    nntrainer::Tensor down_out =
      down_all.getSharedDataTensor(hidden_dim, 0, true);

    if (!ops->moe_gather_rows(expert_input, gathered, rows.data(), n))
      hostGather(expert_input, gathered, rows, hidden_size);

    ops->fc_expert(
      gathered, context.getWeight(expert_gate_proj_indices[expert]), gate_out);
    ops->fc_expert(gathered, context.getWeight(expert_up_proj_indices[expert]),
                   up_out);
    if (exact_gelu)
      ops->geglu_erf(gate_out, up_out, act_out, n, 0);
    else
      ops->geglu(gate_out, up_out, act_out, n, 0);
    ops->fc_expert(act_out, context.getWeight(expert_down_proj_indices[expert]),
                   down_out);

    // Ascending expert id, one call per expert on the in-order queue: every
    // output element sums its contributions in the cpu layer's order.
    if (!device_output || !ops->moe_scatter_add_rows(
                            down_out, output, rows.data(), weights.data(), n))
      hostScatterAdd(down_out, output, rows, weights, hidden_size);
  }
  if (device_output)
    handOutputToDevice(output);
}

} // namespace causallm

#endif // ENABLE_OPENCL
