// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   htp_weight_preparation.cpp
 * @date   10 July 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Shared HTP weight preparation from the existing layer contexts.
 */

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gemma4_htp_static_quant.h>
#include <mutex>
#include <set>

#include <app_context.h>
#include <engine.h>
#include <model.h>

#include <htp_weight_preparation.h>
#include <llm_util.hpp>
#include <tokenizers_cpp.h>

#include <dense_ffn_layer.h>
#include <embedding_layer.h>
#include <htp_backend/htp_tl2_opts.h>
#include <htp_fixed_opts.h>
#include <htp_mha_core.h>
#include <htp_tie_word_embedding.h>
#include <htp_wh_layout.h>
#include <lfm2_moe_layer.h>
#include <lm_head.h>
#include <neuralnet.h>
#include <qs4cx_tensor.h>
#include <rms_norm.h>
#include <swiglu_layer.h>
#include <weight_file_reader.h>

namespace causallm {
namespace {
bool isMoeLayer(const ml::train::Layer &l) {
  return l.getType() == Lfm2MoELayer::type;
}

void packCpuWeight(nntrainer::Tensor &weight) {
  // Only ARM's KleidiAI backend consumes this derived layout. Other CPU
  // backends and HTP read the original QS4CX payload directly.
#if defined(__aarch64__) || defined(__arm__)
  weight.pack();
#else
  (void)weight;
#endif
}
} // namespace

void prepareHtpWeights(ml::train::Model &model,
                       const HtpWeightPreparation &options) {
  // Register FC weights after all MoE arena chunks are mapped. DSP chunks
  // require 256 MiB alignment in a 32-bit address space; interleaved heap
  // allocations strand the space up to the next chunk boundary.
  std::map<std::string, HtpStaticEncoding> static_weights;
  std::string static_path = options.static_activation_file;
  if (const char *path = std::getenv("NNTR_HTP_STATIC_LINEAR"))
    static_path = path;
  if (!static_path.empty()) {
    std::filesystem::path path(static_path);
    if (path.is_relative())
      path = std::filesystem::path(options.weight_dir) / path;
    std::ifstream stream(path);
    if (!stream)
      throw std::runtime_error("Cannot open static activation file: " +
                               path.string());
    static_weights =
      parseHtpStaticLinear(nlohmann::json::parse(stream), options.num_experts);
    if (!options.all_linear_htp)
      throw std::invalid_argument(
        "Static Gemma4 activation encodings require all linear ops on HTP with "
        "WH FC/head weights");
  }
  size_t static_bound = 0;
  struct PendingFc {
    nntrainer::ComputeOps *ops;
    void *data;
    const float *scale; ///< QS4CX per-channel scales; null for Q4_0
    unsigned int K, N;
  };
  std::vector<PendingFc> fc_pending;
  std::set<const void *> cpu_weight_sources;
  std::set<const void *> shared_heads;
  const bool discard_sources =
    nntrainer::weightPageCachePolicy() >= 0 && nntrainer::htp_fixed::E2E;
  struct PendingDense {
    nntrainer::ComputeOps *ops;
    void *up, *gate, *down;
    const float *up_s, *gate_s, *down_s; /**< QS4CX scales; null for Q4_0 */
    unsigned int K, I, N;
  };
  std::vector<PendingDense> dense_pending;
  struct PendingConv {
    nntrainer::ComputeOps *ops;
    void *in_proj, *out_proj;
    const float *conv_w;
    unsigned int K, C, N;
  };
  std::vector<PendingConv> conv_pending;
  // Defer warm-up until every MoE weight is mapped: its scratch grows the
  // DSP heap and would otherwise fragment the arena mappings above.
  struct PendingMoeWarm {
    nntrainer::ComputeOps *ops = nullptr;
    std::vector<void *> gu_data, dn_data;
    std::vector<float *> gu_scale, dn_scale;
    unsigned int K = 0, inter = 0, N_out = 0;
    unsigned int w_bits = 0u; /**< 0 QS4CX, 4 QS4CX_WH, 2 QS2CX_WH */
  } moe_warm;

  // Prepare the tied lm_head after all other weights, either in the
  // accelerator arena or as a blocked CPU copy. Its node-owned context
  // outlives the walk.
  struct PendingTie {
    nntrainer::HtpTieWordEmbedding *layer = nullptr;
    nntrainer::RunLayerContext *context = nullptr;
  } tie_pending;
  // The attention layers' quantized KV caches, registered after the
  // lm_head for the same reason it waits: they are DSP heap.
  std::vector<std::pair<HtpMHACoreLayer *, nntrainer::RunLayerContext *>>
    mha_pending;

  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn = [&fc_pending, &dense_pending, &conv_pending, &moe_warm, &tie_pending,
          &mha_pending, &cpu_weight_sources, &shared_heads, &static_weights,
          &static_bound, discard_sources](
           ml::train::Layer &l, nntrainer::RunLayerContext &context, void *) {
      // forEachLayer hands out LayerNodes.
      auto *consumer = context.getComputeOps();
      for (auto *w : context.getWeights()) {
        auto &tensor = w->getVariableRef();
        auto encoding = static_weights.find(w->getName());
        if (!static_weights.empty() && htpStaticLinearWeight(w->getName()) &&
            encoding == static_weights.end())
          throw std::runtime_error("Missing static input encoding: " +
                                   w->getName());
        if (encoding != static_weights.end()) {
          using DType = ml::train::TensorDim::DataType;
          const auto dtype = tensor.getDataType();
          const bool router =
            w->getName().find("_sparse_moe:gate") != std::string::npos;
          if (!(router && dtype == DType::FP32) && dtype != DType::QS4CX_WH &&
              dtype != DType::QS4CX_WH_AFFINE && dtype != DType::QS2CX_WH &&
              dtype != DType::TL2CX_WH)
            throw std::invalid_argument(
              "Static input requires WH weights or an FP32 router: " +
              w->getName());
          const auto &q = encoding->second;
          const void *key = tensor.isVirtual()
                              ? static_cast<const void *>(&tensor)
                              : tensor.getData();
          if (!consumer ||
              !consumer->set_weight_static_quant(key, q.scale, q.zero_point))
            throw std::runtime_error("Cannot attach static encoding: " +
                                     w->getName());
          ++static_bound;
        }
        if (tensor.isVirtual())
          continue;
        // Only the bias-free untied head opts into this. Another CPU consumer
        // of the same tensor still prevents release, irrespective of walk
        // order.
        if (discard_sources && l.getType() == nntrainer::LmHeadLayer::type &&
            context.getWeights().size() == 1 &&
            (tensor.getDataType() == ml::train::TensorDim::DataType::Q4_0 ||
             tensor.getDataType() == ml::train::TensorDim::DataType::QS4CX ||
             tensor.getDataType() == ml::train::TensorDim::DataType::QS4CX_WH ||
             tensor.getDataType() ==
               ml::train::TensorDim::DataType::QS4CX_WH_AFFINE)) {
          shared_heads.insert(tensor.getData());
          continue;
        }
        const bool accelerated =
          consumer && ((l.getType() == "dense_ffn" &&
                        consumer->supports_gemm_q4_0_dense_ffn_fp32()) ||
                       (consumer->supports_gemm_qs4cx_accel_fp32() &&
                        (consumer->accelerates_qs4cx_at_m1() ||
                         consumer->uses_device_weight(tensor.getData()))));
        if (!accelerated)
          cpu_weight_sources.insert(tensor.getData());
      }
      if (l.getType() == HtpMHACoreLayer::type) {
        if (auto *m = dynamic_cast<HtpMHACoreLayer *>(
              static_cast<nntrainer::LayerNode &>(l).getLayer()))
          mha_pending.push_back({m, &context});
        return;
      }
      if (l.getType() == nntrainer::HtpTieWordEmbedding::type) {
        auto *tw = dynamic_cast<nntrainer::HtpTieWordEmbedding *>(
          static_cast<nntrainer::LayerNode &>(l).getLayer());
        if (tw) {
          tie_pending.layer = tw;
          tie_pending.context = &context;
        }
        return;
      }
      // repack FC and MoE FFN layers -- both can hold QS4CX weights.
      // FloatTensor::dotQs4cx only needs a weight packed when it falls
      // through to the CPU KleidiAI path (getPackedData()); the HTP path
      // (gemm_qs4cx_accel_fp32) reads getData()/getScale() directly and
      // never touches packed_data. Avoid retaining a CPU packed copy when
      // both prefill and decode use the registered device weight. CPU and
      // partially accelerated models still require their CPU packing.
      // A dense_ffn layer's three Q4_0 weights are registered as one
      // fused set (doc 51), after the walk like the FCs below -- not as
      // three FC weights, which the loop below would otherwise do.
      if (l.getType() == "dense_ffn") {
        // By name: with its norms folded in (in_norm / out_norm) the layer
        // also holds two FP32 gammas, and gate_first swaps the first two.
        // Matched on the weight order of a size()==3 test, the folded
        // layer skipped this and registered inside the first prefill: a
        // whole scalar Q4_0 -> QS4CX conversion per layer under the
        // prefill timer (doc 57 section 9.10).
        nntrainer::Tensor *up_p = nullptr, *gate_p = nullptr, *down_p = nullptr;
        for (auto *w : context.getWeights()) {
          const std::string &name = w->getName();
          auto ends_with = [&name](const char *tail) {
            const size_t n = std::strlen(tail);
            return name.size() >= n &&
                   name.compare(name.size() - n, n, tail) == 0;
          };
          if (ends_with(":up"))
            up_p = &w->getVariableRef();
          else if (ends_with(":gate"))
            gate_p = &w->getVariableRef();
          else if (ends_with(":down"))
            down_p = &w->getVariableRef();
        }
        // Only CPU execution needs a KleidiAI copy. The accelerated fused
        // path owns its registered weights for both prefill and decode.
        const auto *dense_ops = context.getComputeOps();
        const bool cpu_dense =
          !dense_ops || !dense_ops->supports_gemm_q4_0_dense_ffn_fp32();
        for (auto *p : {up_p, gate_p, down_p})
          if (cpu_dense && p &&
              p->getDataType() == ml::train::TensorDim::DataType::QS4CX &&
              !p->isVirtual())
            packCpuWeight(*p);
        if (up_p && gate_p && down_p && context.getComputeOps()) {
          auto &up = *up_p;
          auto &gate = *gate_p;
          auto &down = *down_p;
          const auto wtype = up.getDataType();
          const bool wh =
            (wtype == ml::train::TensorDim::DataType::QS4CX_WH ||
             wtype == ml::train::TensorDim::DataType::QS4CX_WH_AFFINE);
          const bool qs4cx =
            wh || wtype == ml::train::TensorDim::DataType::QS4CX;
          if ((wtype == ml::train::TensorDim::DataType::Q4_0 || qs4cx) &&
              gate.getDataType() == wtype && down.getDataType() == wtype) {
            if (wh)
              for (auto *tensor : {&up, &gate, &down})
                if (!context.getComputeOps()->set_fc_wh_source(
                      tensor->getData(), tensor->height(), tensor->width(),
                      wtype == ml::train::TensorDim::DataType::QS4CX_WH_AFFINE))
                  throw std::runtime_error("WH dense FFN requires HTP");
            dense_pending.push_back({context.getComputeOps(),
                                     up.getData<char>(), gate.getData<char>(),
                                     down.getData<char>(),
                                     qs4cx ? up.getScale<float>() : nullptr,
                                     qs4cx ? gate.getScale<float>() : nullptr,
                                     qs4cx ? down.getScale<float>() : nullptr,
                                     static_cast<unsigned int>(up.height()),
                                     static_cast<unsigned int>(up.width()),
                                     static_cast<unsigned int>(down.width())});
          }
        }
        return;
      }
      // A conv_block layer's in_proj and out_proj, likewise (doc 51
      // section 2): the conv weight is FP32 and rides with each call.
      if (l.getType() == "conv_block") {
        auto weights = context.getWeights();
        if (weights.size() == 3 && context.getComputeOps() &&
            weights[0]->getVariableRef().getDataType() ==
              ml::train::TensorDim::DataType::QS4CX_WH_AFFINE) {
          auto *ops = context.getComputeOps();
          for (auto index : {0u, 2u}) {
            auto &t = weights[index]->getVariableRef();
            if (t.getDataType() !=
                  ml::train::TensorDim::DataType::QS4CX_WH_AFFINE ||
                !ops->set_fc_wh_source(t.getData(), t.height(), t.width(),
                                       true))
              throw std::runtime_error(
                "Affine WH conv projections require HTP");
            fc_pending.push_back({ops, t.getData(), t.getScale<float>(),
                                  static_cast<unsigned>(t.height()),
                                  static_cast<unsigned>(t.width())});
          }
          return;
        }
        if (weights.size() == 3 && context.getComputeOps()) {
          auto &in_proj = weights[0]->getVariableRef();
          auto &conv = weights[1]->getVariableRef();
          auto &out_proj = weights[2]->getVariableRef();
          if (in_proj.getDataType() == ml::train::TensorDim::DataType::Q4_0 &&
              out_proj.getDataType() == ml::train::TensorDim::DataType::Q4_0) {
            conv_pending.push_back(
              {context.getComputeOps(), in_proj.getData<char>(),
               out_proj.getData<char>(), conv.getData<float>(),
               static_cast<unsigned int>(in_proj.height()),
               static_cast<unsigned int>(conv.width()),
               static_cast<unsigned int>(out_proj.width())});
          }
        }
        return;
      }
      // residual_add: its proj weight, when it holds the o-proj (doc 57
      // section 9.32); its FP32 gamma / scalar are skipped below.
      if (l.getType() != "fully_connected" &&
          l.getType() != "shared_fully_connected" &&
          l.getType() != "htp_qkv_layer" && !isMoeLayer(l) &&
          l.getType() != nntrainer::LmHeadLayer::type &&
          l.getType() != "residual_add")
        return;

      // Register accelerator MoE weights at load to avoid first-token work.
      // Expert tensors use the kernel's [K, N] layout.
      auto *ops = isMoeLayer(l) ? context.getComputeOps() : nullptr;
      // Defer FC registration until after MoE mappings; CPU backends decline
      // registration. FC weights use dot()'s [K, N] layout.
      auto *fc_ops = isMoeLayer(l) ? nullptr : context.getComputeOps();

      auto weights = context.getWeights();
      std::vector<void *> gu_data, dn_data;
      std::vector<float *> gu_scale, dn_scale;
      unsigned int gu_h = 0, gu_w = 0, dn_h = 0, dn_w = 0;
      unsigned int w_bits = 0u;
      for (auto &w : weights) {
        auto &t = w->getVariableRef();
        const auto dtype = t.getDataType();
        if (ops && dtype == ml::train::TensorDim::DataType::QS4CX_WH_AFFINE &&
            w->getName().size() >= 5 &&
            w->getName().compare(w->getName().size() - 5, 5, ":gate") == 0) {
          if (!ops->set_fc_wh_source(t.getData(), t.height(), t.width(), true))
            throw std::runtime_error("INT4 router requires the HTP backend");
          fc_pending.push_back({ops, t.getData<char>(), t.getScale<float>(),
                                static_cast<unsigned>(t.height()),
                                static_cast<unsigned>(t.width())});
          // The graph borrows this packed record until its first invocation.
          cpu_weight_sources.insert(t.getData());
          continue;
        }
        auto *weight_ops = context.getComputeOps();
        const bool cpu_pack = !weight_ops ||
                              !weight_ops->supports_gemm_qs4cx_accel_fp32() ||
                              (!weight_ops->accelerates_qs4cx_at_m1() &&
                               !weight_ops->uses_device_weight(t.getData()));
        if (cpu_pack && dtype == ml::train::TensorDim::DataType::QS4CX &&
            !t.isVirtual()) {
          packCpuWeight(t);
        }
        // A QS4CX FC weight (quantized offline, no Q4_0 detour) is
        // registered here too, through the key gemm_qs4cx_accel_fp32 looks
        // up: the weight's own data pointer and scales.
        if (fc_ops &&
            (dtype == ml::train::TensorDim::DataType::Q4_0 ||
             ((dtype == ml::train::TensorDim::DataType::QS4CX ||
               dtype == ml::train::TensorDim::DataType::QS4CX_WH ||
               dtype == ml::train::TensorDim::DataType::QS4CX_WH_AFFINE) &&
              !t.isVirtual()))) {
          const bool wh =
            dtype == ml::train::TensorDim::DataType::QS4CX_WH ||
            dtype == ml::train::TensorDim::DataType::QS4CX_WH_AFFINE;
          const bool qs4cx =
            wh || dtype == ml::train::TensorDim::DataType::QS4CX;
          if (wh && !fc_ops->set_fc_wh_source(
                      t.getData(), t.height(), t.width(),
                      dtype == ml::train::TensorDim::DataType::QS4CX_WH_AFFINE))
            throw std::runtime_error("WH FC/head requires HTP: " +
                                     w->getName());
          // The head binds its wider vocabulary slices with the decode
          // list; the same handles serve the prefill's last row.
          if (l.getType() != nntrainer::LmHeadLayer::type)
            fc_pending.push_back({fc_ops, t.getData<char>(),
                                  qs4cx ? t.getScale<float>() : nullptr,
                                  static_cast<unsigned int>(t.height()),
                                  static_cast<unsigned int>(t.width())});
        }
        if (ops && (dtype == ml::train::TensorDim::DataType::QS4CX ||
                    dtype == ml::train::TensorDim::DataType::QS4CX_WH ||
                    dtype == ml::train::TensorDim::DataType::QS2CX_WH ||
                    dtype == ml::train::TensorDim::DataType::TL2CX_WH ||
                    dtype == ml::train::TensorDim::DataType::QS4CX_WH_AFFINE)) {
          const auto h = static_cast<unsigned int>(t.height());
          const auto wd = static_cast<unsigned int>(t.width());
          w_bits = dtype == ml::train::TensorDim::DataType::TL2CX_WH
                     ? HTP_TL2_WEIGHT_BITS
                   : (dtype == ml::train::TensorDim::DataType::QS4CX_WH ||
                      dtype == ml::train::TensorDim::DataType::QS4CX_WH_AFFINE)
                     ? 4u
                   : dtype == ml::train::TensorDim::DataType::QS2CX_WH ? 2u
                                                                       : 0u;
          // A virtual expert has no bytes here: preloadExperts
          // below reads it from the model file into the accelerator's slot
          // pool, keyed by the tensor itself, and the warm-up passes that
          // key with a null scale the way the layer's forward does.
          void *key =
            t.isVirtual() ? static_cast<void *>(&t) : t.getData<char>();
          float *scale = t.isVirtual() ? nullptr : t.getScale<float>();
          if (!t.isVirtual())
            ops->register_qs4cx_weight(key, scale, h, wd, w_bits);
          // The expert weights come in two shapes: gate_up is [K, 2*inter]
          // and down [inter, N_out]. Sorted here for the warm-up below by
          // the identity that tells them apart, gate_up.width == 2 *
          // down.height, checked once both are seen.
          if (gu_h == 0 || (h == gu_h && wd == gu_w)) {
            if (gu_h == 0) {
              gu_h = h;
              gu_w = wd;
            }
            gu_data.push_back(key);
            gu_scale.push_back(scale);
          } else {
            dn_h = h;
            dn_w = wd;
            dn_data.push_back(key);
            dn_scale.push_back(scale);
          }
        }
      }
      // Virtual experts into the shared pool now, in layer order
      // until it is full, so the first prefill maps no arena chunk. A
      // layer whose experts did not all fit cannot host the warm-up call.
      bool moe_all_resident = true;
      if (ops && isMoeLayer(l)) {
        auto *layer = static_cast<nntrainer::LayerNode &>(l).getLayer();
        if (auto *moe = dynamic_cast<Lfm2MoELayer *>(layer))
          moe_all_resident = moe->preloadExperts(context);
      }
      // One valid expert pair warms the session's scratch, FastRPC buffers
      // and DSP setup at load. Use zero input and discard the output.
      if (ops && moe_warm.ops == nullptr && moe_all_resident && isMoeLayer(l) &&
          ops->supports_gemm_qs4cx_moe_layer_fp32() && !gu_data.empty() &&
          gu_data.size() == dn_data.size() && gu_w == 2 * dn_h &&
          gu_h == dn_w) {
        moe_warm.ops = ops;
        moe_warm.gu_data = gu_data;
        moe_warm.gu_scale = gu_scale;
        moe_warm.dn_data = dn_data;
        moe_warm.dn_scale = dn_scale;
        moe_warm.K = gu_h;
        moe_warm.inter = dn_h;
        moe_warm.N_out = dn_w;
        moe_warm.w_bits = w_bits;
      }
    };
  try {
    model.forEachLayer(fn, nullptr);
#ifdef ENABLE_HEXKL
    for (const void *head : shared_heads)
      if (!cpu_weight_sources.count(head))
        nntrainer::get_htp_ops()->share_decode_lm_head(head);
#endif
    // Set the FC WH sidecar before registering pending accelerator weights.
    // CPU backends decline it without opening the file.
    if (!options.fc_wh_file.empty()) {
      NNTR_THROW_IF(options.fc_wh_format != nntrainer::FCWH_FORMAT,
                    std::runtime_error)
        << "fc_wh_format \"" << options.fc_wh_format
        << "\" in nntr_config.json, this "
        << "build reads \"" << nntrainer::FCWH_FORMAT
        << "\": re-run nntr_quantize_stream --fc_wh_sidecar";
      const std::filesystem::path p(options.fc_wh_file);
      const std::string path =
        p.is_absolute()
          ? p.string()
          : (std::filesystem::path(options.weight_dir) / p).string();
      std::set<nntrainer::ComputeOps *> backends;
      for (const auto &f : fc_pending)
        backends.insert(f.ops);
      for (const auto &d : dense_pending)
        backends.insert(d.ops);
      for (const auto &c : conv_pending)
        backends.insert(c.ops);
#ifdef ENABLE_HEXKL
      // The decode graph also binds FC / DENSE_FFN images from the sidecar.
      if (nntrainer::get_htp_ops()->has_decode_graph_q4_0())
        backends.insert(nntrainer::get_htp_ops());
#endif
      for (auto *b : backends)
        b->set_fc_wh_file(path.c_str());
    }
    // Warm the MoE session after all arena mappings (see PendingMoeWarm).
    if (moe_warm.ops != nullptr) {
      // Size scratch and staging for the configured prefill to avoid
      // growing and first-touching those buffers on the first call.
      const unsigned int M = std::max(512u, options.prefill_rows),
                         K = moe_warm.K, N_out = moe_warm.N_out;
      const unsigned int E = static_cast<unsigned int>(moe_warm.gu_data.size());
      const unsigned int per = 64; // one 64-row block per expert
      std::vector<unsigned int> row_index, row_count(E, per);
      std::vector<float> row_weight;
      for (unsigned int e = 0; e < E; ++e) {
        for (unsigned int r = 0; r < per; ++r) {
          row_index.push_back((e * per + r) % M);
          row_weight.push_back(0.0f);
        }
      }
      std::vector<float> act(static_cast<size_t>(M) * K, 0.0f);
      std::vector<float> out(static_cast<size_t>(M) * N_out, 0.0f);
      moe_warm.ops->gemm_qs4cx_moe_layer_fp32(
        moe_warm.gu_data, moe_warm.gu_scale, moe_warm.dn_data,
        moe_warm.dn_scale, row_index, row_count, row_weight, act.data(),
        out.data(), M, K, moe_warm.inter, N_out, moe_warm.w_bits);
      ml_logd("MoE HTP kernel warmed up at load (M=%u, %u experts)", M, E);
    }
    // The deferred FC registrations (see fc_pending above). A CPU-engine
    // layer's ops answer false and cost nothing. One warm-up call per
    // session after the first accelerated one, like the MoE one: the first
    // FC call otherwise grows the staging buffers to the prefill size and
    // touches every page of them inside the first prefill. In graph order
    // that first weight is a conv in_proj (the widest N) whenever those are
    // routed, so one call at M=512 covers every later shape's buffers.
    // A QS4CX weight has no register-only entry that slices it the way
    // the FC call does (register_qs4cx_weight is the one-handle expert
    // form), so its call registers it: 64 rows, one row block, for every
    // weight but the warm-up one.
    if (!static_weights.empty()) {
      if (static_bound != static_weights.size())
        throw std::runtime_error("Static weight names did not all match: " +
                                 std::to_string(static_bound) + "/" +
                                 std::to_string(static_weights.size()));
      std::fprintf(stderr, "[HTP-STATIC] bound %zu weight inputs\n",
                   static_bound);
    }
    bool fc_warmed = false;
    for (const auto &p : fc_pending) {
      const bool qs4cx = p.scale != nullptr;
      if (qs4cx ? !p.ops->supports_gemm_qs4cx_accel_fp32()
                : !p.ops->register_q4_0_weight(p.data, p.K, p.N))
        continue;
      if (fc_warmed && !qs4cx)
        continue;
      if (!qs4cx && !p.ops->supports_gemm_q4_0_accel_fp32())
        continue;
      const unsigned int M = p.N > 65536 ? 1 : (fc_warmed ? 64 : 512);
      fc_warmed = true;
      std::vector<float> act(static_cast<size_t>(M) * p.K, 0.0f);
      std::vector<float> out(static_cast<size_t>(M) * p.N, 0.0f);
      if (qs4cx)
        p.ops->gemm_qs4cx_accel_fp32(p.data, const_cast<float *>(p.scale),
                                     act.data(), out.data(), M, p.N, p.K);
      else
        p.ops->gemm_q4_0_accel_fp32(p.data, act.data(), out.data(), M, p.N,
                                    p.K);
      ml_logd("FC HTP kernel warmed up at load (M=%u, K=%u, N=%u)", M, p.K,
              p.N);
      if (qs4cx && discard_sources && !cpu_weight_sources.count(p.data))
        p.ops->discard_weight_source(
          p.data, static_cast<size_t>(p.K) * p.N / 2 + sizeof(float) * p.N);
    }
    // Same for the dense FFNs: the registration converts and packs three
    // weights into I / w expert pairs, and the one warm-up call grows the
    // MoE layer kernel's scratch to this shape's row count.
    bool dense_warmed = false;
    for (const auto &p : dense_pending) {
      if (!p.ops->register_q4_0_dense_ffn(p.up, p.gate, p.down, p.K, p.I, p.N,
                                          p.up_s, p.gate_s, p.down_s))
        continue;
      if (p.up_s && discard_sources) {
        for (auto weight :
             {std::make_pair(p.up, static_cast<size_t>(p.K) * p.I / 2 +
                                     sizeof(float) * p.I),
              std::make_pair(p.gate, static_cast<size_t>(p.K) * p.I / 2 +
                                       sizeof(float) * p.I),
              std::make_pair(p.down, static_cast<size_t>(p.I) * p.N / 2 +
                                       sizeof(float) * p.N)})
          if (!cpu_weight_sources.count(weight.first))
            p.ops->discard_weight_source(weight.first, weight.second);
      }
      if (dense_warmed || !p.ops->supports_gemm_q4_0_dense_ffn_fp32())
        continue;
      dense_warmed = true;
      const unsigned int M = 512;
      std::vector<float> act(static_cast<size_t>(M) * p.K, 0.0f);
      std::vector<float> out(static_cast<size_t>(M) * p.N, 0.0f);
      p.ops->gemm_q4_0_dense_ffn_fp32(p.up, p.gate, p.down, act.data(),
                                      out.data(), M, p.K, p.I, p.N,
                                      !static_weights.empty(), nullptr, nullptr,
                                      0.0f, p.up_s, p.gate_s, p.down_s);
      ml_logd("dense FFN HTP kernel warmed up at load (M=%u, K=%u, I=%u, N=%u)",
              M, p.K, p.I, p.N);
    }
    // And the conv blocks: in_proj's three slices and out_proj registered,
    // one warm-up call so the first prefill finds the kernel's scratch and
    // the staging buffers already grown to this shape.
    bool conv_warmed = false;
    for (const auto &p : conv_pending) {
      if (!p.ops->register_q4_0_conv_block(p.in_proj, p.out_proj, p.K, p.C,
                                           p.N))
        continue;
      if (conv_warmed || !p.ops->supports_gemm_q4_0_conv_block_fp32())
        continue;
      conv_warmed = true;
      const unsigned int M = 512;
      std::vector<float> act(static_cast<size_t>(M) * p.K, 0.0f);
      std::vector<float> out(static_cast<size_t>(M) * p.N, 0.0f);
      std::vector<float> state(static_cast<size_t>(2) * p.C, 0.0f);
      p.ops->gemm_q4_0_conv_block_fp32(p.in_proj, p.conv_w, p.out_proj,
                                       act.data(), out.data(), state.data(), M,
                                       p.K, p.C, p.N);
      ml_logd(
        "conv block HTP kernel warmed up at load (M=%u, K=%u, C=%u, N=%u)", M,
        p.K, p.C, p.N);
    }
    if (tie_pending.layer != nullptr &&
        !tie_pending.layer->placeLmheadOnAccelerator(*tie_pending.context))
      tie_pending.layer->prepareLmhead(*tie_pending.context);
    for (auto &p : mha_pending)
      p.first->registerQuantizedCache(*p.second);
    ml_logd("QS4CX weights repacked successfully");
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to repack weights: " +
                             std::string(e.what()));
  }
};

/**
 * @brief Run a transformer model for a prompt.
 */
} // namespace causallm
