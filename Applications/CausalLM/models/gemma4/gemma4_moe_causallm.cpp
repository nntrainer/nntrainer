// SPDX-License-Identifier: Apache-2.0
/**
 * @file   gemma4_moe_causallm.cpp
 * @brief  Gemma4 MoE causal language model implementation.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs
 */

#include <app_context.h>
#include <engine.h>
#include <gemma4_moe_causallm.h>
#include <gemma4_moe_layer.h>
#if defined(ENABLE_OPENCL)
#include <gemma4_moe_layer_gpu.h>
#endif
#include <llm_util.hpp>
#include <model.h>
#include <nntrainer_log.h>
#include <residency_policy.h>
#include <safetensors_util.h>
#include <tq2cx_layout.h>
#if defined(ENABLE_OPENCL)
#include <moe_route_cl.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <iostream>

namespace causallm {

namespace {

/**
 * @brief Set the process's TQ2CX stored layout from the pack: the safetensors
 *        header's __metadata__ "nntr_tq2cx_layout", else the nntr_config key
 *        "tq2cx_layout", else row-major. Before the model is initialized, so
 *        the weights are sized and the kernels built for it; the loader then
 *        refuses a file whose header names another one.
 */
void setTq2cxLayoutFromPack(const json &nntr_cfg) {
  std::string from_header, from_config;
  bool has_header = false;
  if (nntr_cfg.contains("model_file_name") &&
      nntr_cfg["model_file_name"].is_string()) {
    const std::string path = nntr_cfg["model_file_name"].get<std::string>();
    if (path.size() > 12 &&
        path.compare(path.size() - 12, 12, ".safetensors") == 0) {
      const auto meta = nntrainer::safetensors::readMetadata(path);
      auto it = meta.find(nntrainer::TQ2CX_LAYOUT_META_KEY);
      if (it != meta.end()) {
        from_header = it->second;
        has_header = true;
      }
    }
  }
  const bool has_config =
    nntr_cfg.contains(nntrainer::TQ2CX_LAYOUT_CONFIG_KEY) &&
    nntr_cfg[nntrainer::TQ2CX_LAYOUT_CONFIG_KEY].is_string();
  if (has_config)
    from_config =
      nntr_cfg[nntrainer::TQ2CX_LAYOUT_CONFIG_KEY].get<std::string>();
  NNTR_THROW_IF(has_header && has_config && from_header != from_config,
                std::invalid_argument)
    << "[Gemma4MoE] the weight file names TQ2CX layout '" << from_header
    << "' but nntr_config says '" << from_config << "'";
  const std::string name =
    has_header ? from_header : (has_config ? from_config : "rm");
  nntrainer::Tq2cxLayout lay;
  NNTR_THROW_IF(!nntrainer::tq2cxLayoutFromName(name, lay),
                std::invalid_argument)
    << "[Gemma4MoE] TQ2CX layout '" << name
    << "' is not implemented (rm, t2r16v2)";
  nntrainer::setTq2cxStoredLayout(lay);
}

} // namespace

void Gemma4MoECausalLM::setupParameters(json &cfg, json &generation_cfg,
                                        json &nntr_cfg) {
  Gemma4CausalLM::setupParameters(cfg, generation_cfg, nntr_cfg);

  NNTR_THROW_IF(!cfg.contains("num_experts") || cfg["num_experts"].is_null() ||
                  !cfg.contains("top_k_experts") ||
                  cfg["top_k_experts"].is_null() ||
                  !cfg.contains("moe_intermediate_size") ||
                  cfg["moe_intermediate_size"].is_null(),
                std::invalid_argument)
    << "[Gemma4MoE] num_experts, top_k_experts, and moe_intermediate_size "
       "must be provided";

  num_experts = cfg["num_experts"].get<unsigned int>();
  top_k_experts = cfg["top_k_experts"].get<unsigned int>();
  moe_intermediate_size = cfg["moe_intermediate_size"].get<unsigned int>();
  moe_cache_size =
    nntr_cfg.contains("moe_cache_size") && !nntr_cfg["moe_cache_size"].is_null()
      ? nntr_cfg["moe_cache_size"].get<unsigned int>()
      : 0;
  // moe_fsu (gpu engine): moe_cache_size is then the device slots per layer
  const bool fsu_in_cfg =
    nntr_cfg.contains("moe_fsu") && !nntr_cfg["moe_fsu"].is_null();
  moe_fsu = fsu_in_cfg && nntr_cfg["moe_fsu"].get<bool>();

  moe_expert_dtype = nntr_cfg.contains("moe_expert_dtype") &&
                         !nntr_cfg["moe_expert_dtype"].is_null()
                       ? nntr_cfg["moe_expert_dtype"].get<std::string>()
                       : FC_LAYER_DTYPE;
  if (moe_expert_dtype == "TQ2CX")
    setTq2cxLayoutFromPack(nntr_cfg);

#if defined(__ANDROID__) && defined(ENABLE_OPENCL)
  // A phone cannot keep every expert of a TQ2CX pack resident (~7.5 GB on
  // the gpu engine), so a pack that does not say otherwise streams its
  // experts through 32 device slots per layer (C32: same tokens as resident,
  // ~3.4 GB). An explicit "moe_fsu" key in nntr_config wins, and
  // NNTR_G4MOE_FSU / NNTR_MOE_CACHE_EXPERTS still override at finalize.
  if (!fsu_in_cfg && causallm_engine() == "gpu" &&
      moe_expert_dtype == "TQ2CX") {
    const char *e = std::getenv("NNTR_G4MOE_FSU");
    const bool env_off = e != nullptr && e[0] != '\0' && e[0] != '1';
    if (!env_off) {
      moe_fsu = true;
      if (moe_cache_size == 0)
        moe_cache_size = 32;
      std::fprintf(stderr, "[G4MOE] fsu default on Android: cache_experts=%u\n",
                   moe_cache_size);
      ml_logi("[G4MOE] fsu default on Android: cache_experts=%u",
              moe_cache_size);
    }
  }
#endif

  moe_router_norm_weight = nntr_cfg.contains("moe_router_norm_weight") &&
                           nntr_cfg["moe_router_norm_weight"].get<bool>();

  // Norm + residual add in one kernel (rmsnorm_cl_fp16_coop_add): only the
  // gpu engine's FP16 activation arm has it, and there it rounds exactly like
  // the separate norm and add. Any other arm keeps the addition layers.
  fuse_norm_add = false;
#if defined(ENABLE_OPENCL) && defined(ENABLE_FP16)
  {
    const bool f16_act =
      MODEL_TENSOR_TYPE.size() >= 5 &&
      MODEL_TENSOR_TYPE.compare(MODEL_TENSOR_TYPE.size() - 5, 5, "-FP16") == 0;
    const char *e = std::getenv("NNTR_G4MOE_NORM_ADD");
    const bool opt_out = e != nullptr && e[0] == '0';
    fuse_norm_add = causallm_engine() == "gpu" && f16_act && !opt_out;
  }
#endif

  NNTR_THROW_IF(num_experts == 0 || top_k_experts == 0 ||
                  top_k_experts > num_experts || moe_intermediate_size == 0,
                std::invalid_argument)
    << "[Gemma4MoE] invalid expert configuration";
  NNTR_THROW_IF(NUM_KV_SHARED_LAYERS > 0, std::invalid_argument)
    << "[Gemma4MoE] shared KV layers are not supported";

  // The MoE layer reads its router rows on the host and binds its expert
  // rows and its output as shared memory, so those three tensors stay on the
  // shared plane even when every producer and consumer is a device layer
  // (FP16 activations would otherwise put them in device-only memory) --
  // unless the device serves these experts end to end (grouped launches,
  // decode routed on the device), where the boundary tensors stay on the
  // device plane and a decode step never drains the queue for them.
#if defined(ENABLE_OPENCL)
  if (causallm_engine() == "gpu" && moe_expert_dtype == "TQ2CX" &&
      nntrainer::moe_experts_device_served_cl(
        ml::train::TensorDim::DataType::TQ2CX))
    return;
#endif
  // With the fused norm + add the residual stream after attention is the
  // post-attention norm's output (there is no "_post_attention" layer).
  auto &rp = nntrainer::ResidencyPolicy::global();
  const char *post_attention_out =
    fuse_norm_add ? "_post_attention_norm:out0" : "_post_attention:out0";
  for (const char *pat :
       {"_pre_ffn_norm_2:out0", post_attention_out, "_sparse_moe:out0"}) {
    if (rp.exclude_patterns.find(pat) != std::string::npos)
      continue;
    if (!rp.exclude_patterns.empty())
      rp.exclude_patterns += ",";
    rp.exclude_patterns += pat;
  }
}

Tensor Gemma4MoECausalLM::createFeedForwardBlock(const int layer_id,
                                                 Tensor post_attention,
                                                 bool is_kv_shared_layer) {
  std::vector<std::string> pre_ffn_norm_props = {
    withKey("name", "layer" + std::to_string(layer_id) + "_pre_ffn_norm"),
    withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false"),
    withKey("engine", causallm_engine())};
  appendSkipPrefillIfNeeded(pre_ffn_norm_props, is_kv_shared_layer);
  LayerHandle pre_ffn_norm(createLayer("rms_norm", pre_ffn_norm_props));
  Tensor dense_input = pre_ffn_norm(post_attention);
  Tensor dense_output =
    createMlp(layer_id, DIM, INTERMEDIATE_SIZE, dense_input);

  LayerHandle post_dense_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_post_ffn_norm_1"),
     withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false"),
     withKey("engine", causallm_engine())}));
  Tensor post_dense = post_dense_norm(dense_output);

  LayerHandle pre_sparse_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_pre_ffn_norm_2"),
     withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false"),
     withKey("engine", causallm_engine())}));
  Tensor sparse_input = pre_sparse_norm(post_attention);
  LayerHandle sparse_moe(createLayer(
    "gemma4_moe",
    {withKey("name", "layer" + std::to_string(layer_id) + "_sparse_moe"),
     withKey("unit", std::to_string(moe_intermediate_size)),
     withKey("num_experts", std::to_string(num_experts)),
     withKey("num_experts_per_token", std::to_string(top_k_experts)),
     withKey("moe_cache_size", std::to_string(moe_cache_size)),
     withKey("moe_fsu", moe_fsu ? "true" : "false"),
     withKey("moe_activation", HIDDEN_ACTIVATION),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("router_norm_weight", moe_router_norm_weight ? "true" : "false"),
     withKey("weight_dtype", moe_expert_dtype),
     withKey("engine", causallm_engine())}));
  Tensor sparse_output = sparse_moe({sparse_input, post_attention});

  LayerHandle post_sparse_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_post_ffn_norm_2"),
     withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false"),
     withKey("engine", causallm_engine())}));
  // Fused form (fuseNormResidualAdd()): post_ffn_norm_2 adds the dense
  // branch in its kernel, and post_ffn_norm adds post_attention, so the block
  // returns post_attention + FFN and the decoder block adds nothing more.
  //
  // Schedule: the graph runs (and orders the weights of) the layers in the
  // depth-first order of these inputs. The fused form names the sparse
  // branch first, so the router and the experts run before the dense MLP of
  // the same layer and the dense MLP covers the expert half's wait (its
  // router read, an expert fill); the two branches are independent until
  // they are summed, so only the order of their kernels changes, not a
  // value. The unfused form keeps the dense branch first: a .bin weight file
  // is read in graph order, so this is the order existing .bin files and the
  // cpu engine were written in.
  Tensor combined_ffn;
  if (fuse_norm_add) {
    combined_ffn = post_sparse_norm({sparse_output, post_dense});
  } else {
    Tensor post_sparse = post_sparse_norm(sparse_output);
    LayerHandle combine_ffn(createLayer(
      "addition",
      {withKey("name", "layer" + std::to_string(layer_id) + "_combine_ffn"),
       withKey("engine", causallm_engine())}));
    combined_ffn = combine_ffn({post_dense, post_sparse});
  }

  LayerHandle post_combined_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_post_ffn_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false"),
     withKey("engine", causallm_engine())}));
  if (fuse_norm_add)
    return post_combined_norm({combined_ffn, post_attention});
  return post_combined_norm(combined_ffn);
}

void Gemma4MoECausalLM::registerCustomLayers() {
  Gemma4CausalLM::registerCustomLayers();
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));
  try {
    app_context->registerFactory(nntrainer::createLayer<Gemma4MoELayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }

#if defined(ENABLE_OPENCL)
  // The same layer type on the gpu context, with its experts on the device.
  // Inert when there is no gpu context (CPU-only run) or it is registered
  // already -- both benign.
  try {
    ct_engine.registerLayerFactory("gpu",
                                   nntrainer::createLayer<Gemma4MoELayerGpu>);
  } catch (std::invalid_argument &e) {
  }
#endif
}

} // namespace causallm
