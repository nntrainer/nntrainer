// SPDX-License-Identifier: Apache-2.0
/**
 * @file   gemma4_moe_causallm.cpp
 * @brief  Gemma4 MoE causal language model implementation.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs
 */

#include <app_context.h>
#include <engine.h>
#include <gemma4_htp_moe_causallm.h>
#include <htp_model_config.h>
#include <layer_context.h>
#include <llm_util.hpp>
#include <model.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>

#ifdef ENABLE_HEXKL
#include <compute_ops.h>
#include <htp_backend.h>
#include <htp_graph_binding.h>
#include <htp_graph_desc.h>
#include <htp_mha_core.h>
#include <htp_model_binding.h>
#include <layer_node.h>
#endif

namespace causallm {

void Gemma4HtpMoECausalLM::setupParameters(json &cfg, json &generation_cfg,
                                           json &nntr_cfg) {
  const auto runtime_config = normalizeHtpRuntimeConfig(nntr_cfg);
  if (nntr_cfg.value("gemma4_weight_format", std::string()) != "htp_fused_v1")
    throw std::invalid_argument(
      "Gemma4 htp_fused requires gemma4_weight_format=htp_fused_v1");
  Gemma4HtpCausalLM::setupParameters(cfg, generation_cfg, nntr_cfg);
  if (!ENABLE_MOE_BLOCK || BATCH_SIZE != 1 || DIM == 0 || NUM_EXPERTS == 0 ||
      NUM_EXPERTS_PER_TOK == 0 || NUM_EXPERTS_PER_TOK > NUM_EXPERTS ||
      INIT_SEQ_LEN == 0 || INIT_SEQ_LEN > MAX_SEQ_LEN ||
      nntr_cfg.value("kv_window_ring", false))
    throw std::invalid_argument(
      "Gemma4 htp_fused requires a batch-one MoE graph and full KV cache");
  for (const auto &eng :
       {MOE_ENGINE, ATTN_PROJ_ENGINE, FFN_ENGINE, LMHEAD_ENGINE})
    if (eng != "cpu" && eng != "htp")
      throw std::invalid_argument(
        "Gemma4 htp_fused supports cpu and htp engines only");
  if (!ATTENTION_ENGINE.empty() && ATTENTION_ENGINE != "cpu" &&
      ATTENTION_ENGINE != "htp")
    throw std::invalid_argument(
      "Gemma4 htp_fused has an unsupported attention_engine");
  std::set<std::string> moe_dtypes = {MOE_LAYER_DTYPE};
  for (const auto &[layer, dtype] : MOE_LAYER_DTYPE_PER_LAYER) {
    if (layer < 0 || layer >= static_cast<int>(NUM_LAYERS))
      throw std::invalid_argument(
        "moe_layer_dtype_per_layer names a layer outside the model");
    moe_dtypes.insert(dtype);
  }
  for (const auto &dtype : moe_dtypes) {
    if ((dtype == "QS2CX_WH" || dtype == "QS4CX_WH" || dtype == "TL2CX_WH" ||
         dtype == "QS4CX_WH_AFFINE") &&
        (MOE_ENGINE != "htp" || !MOE_HTP_LAYERS.empty()))
      throw std::invalid_argument(
        "HTP WH expert weights require every expert layer on HTP");
    if (dtype != "FP32" && dtype != "Q4_0" && dtype != "QS4CX" &&
        dtype != "QS2CX_WH" && dtype != "QS4CX_WH" && dtype != "TL2CX_WH" &&
        dtype != "QS4CX_WH_AFFINE")
      throw std::invalid_argument("Unsupported Gemma4 htp_fused expert dtype");
  }
  for (const auto &dtype : {EMBEDDING_DTYPE, FC_LAYER_DTYPE, LMHEAD_DTYPE})
    if (dtype != "FP32" && dtype != "Q4_0" && dtype != "QS4CX" &&
        dtype != "QS4CX_WH" && dtype != "QS4CX_WH_AFFINE")
      throw std::invalid_argument("Unsupported Gemma4 htp_fused weight dtype");
  if (((FC_LAYER_DTYPE == "QS4CX_WH" || FC_LAYER_DTYPE == "QS4CX_WH_AFFINE") &&
       (ATTN_PROJ_ENGINE != "htp" || FFN_ENGINE != "htp")) ||
      ((LMHEAD_DTYPE == "QS4CX_WH" || LMHEAD_DTYPE == "QS4CX_WH_AFFINE") &&
       LMHEAD_ENGINE != "htp") ||
      (EMBEDDING_DTYPE == "QS4CX_WH" || EMBEDDING_DTYPE == "QS4CX_WH_AFFINE"))
    throw std::invalid_argument(
      "Gemma4 WH FC/head weights require HTP; embedding uses Q4_0 or FP32");
  const auto tensor_type =
    nntr_cfg.value("model_tensor_type", std::string("FP32-FP32"));
  if (tensor_type.substr(tensor_type.find_last_of('-') + 1) != "FP32")
    throw std::invalid_argument("Gemma4 htp_fused requires FP32 activations");
  htp_execution_profile_ = true;
#ifndef ENABLE_HEXKL
  if (MOE_ENGINE == "htp" || ATTN_PROJ_ENGINE == "htp" || FFN_ENGINE == "htp" ||
      ATTENTION_ENGINE == "htp" || LMHEAD_ENGINE == "htp")
    throw std::invalid_argument(
      "Gemma4 htp_fused HTP engines require an HTP build");
#endif
#ifdef ENABLE_HEXKL
  // Build the full decode graph after loading its weights.
  htp_e2e = MOE_ENGINE == "htp";
  htp_session_enabled_ = runtime_config.htp.kv_session;
  if (htp_session_enabled_ &&
      (!htp_e2e || ATTENTION_ENGINE != "htp" || ATTN_PROJ_ENGINE != "htp" ||
       FFN_ENGINE != "htp" || LMHEAD_ENGINE != "htp"))
    throw std::invalid_argument(
      "Gemma4 KV sessions require a fully resident HTP graph");
  NNTR_THROW_IF(htp_e2e && (HIDDEN_SIZE_PER_LAYER_INPUT > 0 ||
                            ATTN_LOGIT_SOFTCAPPING > 0.0f),
                std::invalid_argument)
    << "[Gemma4MoE] the HTP decode list has no per-layer input (PLE) and no "
       "attention logit soft-cap";
  // Set GeGLU for the decode graph's MOE and DENSE_FFN ops as well as
  // the prefill calls, which supply it per call.
  if (MOE_ENGINE == "htp" || ATTN_PROJ_ENGINE == "htp" || FFN_ENGINE == "htp" ||
      ATTENTION_ENGINE == "htp" || LMHEAD_ENGINE == "htp")
    nntrainer::configureHtpRuntime(runtime_config);
  if (MOE_ENGINE == "htp") {
    auto *ops = nntrainer::get_htp_ops();
    ops->set_moe_geglu(true);
    ops->set_moe_hadamard(MOE_HADAMARD, MOE_HADAMARD_SCOPE);
  }
#endif
}

void Gemma4HtpMoECausalLM::load_weight(const std::string &weight_path) {
  if (weights_prepared_ && (MOE_ENGINE == "htp" || ATTENTION_ENGINE == "htp" ||
                            ATTN_PROJ_ENGINE == "htp" || FFN_ENGINE == "htp" ||
                            LMHEAD_ENGINE == "htp"))
    throw std::logic_error(
      "Gemma4 HTP weight reload requires a new model handle");
  WEIGHT_DIR = std::filesystem::path(weight_path).parent_path().string();
#ifdef ENABLE_HEXKL
  // [PR 4428] The rotated checkpoint's signs (nntr_config moe_hadamard_signs,
  // a JSON beside the model: "in" K and "down" inter values of +-1), to the
  // HTP backend before the first MoE call.
  if (MOE_ENGINE == "htp" && !MOE_HADAMARD_SIGNS_FILE.empty()) {
    std::filesystem::path p(MOE_HADAMARD_SIGNS_FILE);
    if (p.is_relative())
      p = std::filesystem::path(WEIGHT_DIR) / p;
    std::ifstream f(p);
    if (!f)
      throw std::runtime_error("[Gemma4MoE] moe_hadamard_signs: cannot open " +
                               p.string());
    const auto j = nlohmann::json::parse(f);
    if (j.value("format", std::string()) != "nntr_hadamard_signs_v1" ||
        !j.contains("in") || !j.contains("down"))
      throw std::runtime_error(
        "[Gemma4MoE] moe_hadamard_signs: not an nntr_hadamard_signs_v1 file "
        "with \"in\" and \"down\": " +
        p.string());
    const auto in = j.at("in").get<std::vector<float>>();
    const auto down = j.at("down").get<std::vector<float>>();
    if (in.size() != DIM || down.size() != MOE_INTERMEDIATE_SIZE)
      throw std::runtime_error(
        "[Gemma4MoE] moe_hadamard_signs: " + std::to_string(in.size()) + " / " +
        std::to_string(down.size()) + " signs for widths " +
        std::to_string(DIM) + " / " + std::to_string(MOE_INTERMEDIATE_SIZE));
    if (!nntrainer::get_htp_ops()->set_moe_hadamard_signs(in, down))
      throw std::runtime_error(
        "[Gemma4MoE] moe_hadamard_signs: the HTP backend took no signs");
  }
#endif
  weights_prepared_ = false;
  loading_weights_ = true;
  try {
    Gemma4HtpCausalLM::load_weight(weight_path);
  } catch (...) {
    loading_weights_ = false;
    throw;
  }
  loading_weights_ = false;
#ifdef ENABLE_HEXKL
  if (!htp_e2e) {
    repack_weight();
    return;
  }
  // The weights by name, "<layer>:<weight>" (#4415's fused layers hold
  // several each: qkv's gammas and projections, dense_ffn's gate / up /
  // down and its two gammas, residual_add's gamma and the block's scalar),
  // never by index. Keyed under the owning layer too: a shared_from layer
  // (the tied head) names its own weights after the layer it shares with.
  const HtpModelBinding binding(*model);

  // The list, now that layer_scalar (a checkpoint weight) is in memory.
  const uint32_t n_layers = static_cast<uint32_t>(NUM_LAYERS);
  std::vector<uint8_t> is_full(n_layers);
  std::vector<float> scalar(n_layers);
  for (uint32_t l = 0; l < n_layers; ++l) {
    is_full[l] = !isSlidingAttentionLayer(static_cast<int>(l));
    const std::string p = "layer" + std::to_string(l);
    scalar[l] =
      binding.f32(binding.choose(p + "_sparse_moe:add_scalar",
                                 p + "_post_ffn_norm:scalar_multiplier"),
                  1)[0];
    // The list's ADD stores 0.0f as "no multiplier".
    if (scalar[l] == 0.0f)
      throw std::runtime_error("[Gemma4MoE] HTP decode: layer " +
                               std::to_string(l) +
                               "'s layer_scalar is 0, which the decode list "
                               "cannot carry");
  }
  htp_graph_gemma_shape shape{};
  shape.n_layers = n_layers;
  shape.hidden = static_cast<uint32_t>(DIM);
  shape.inter_dense = static_cast<uint32_t>(INTERMEDIATE_SIZE);
  shape.inter_moe = MOE_INTERMEDIATE_SIZE;
  shape.n_experts = NUM_EXPERTS;
  shape.top_k = NUM_EXPERTS_PER_TOK;
  shape.n_heads = static_cast<uint32_t>(NUM_HEADS);
  shape.n_kv = static_cast<uint32_t>(NUM_KEY_VALUE_HEADS);
  shape.head_dim = static_cast<uint32_t>(HEAD_DIM);
  shape.n_kv_full = ATTENTION_K_EQ_V
                      ? NUM_GLOBAL_KEY_VALUE_HEADS
                      : static_cast<uint32_t>(NUM_KEY_VALUE_HEADS);
  shape.head_dim_full = GLOBAL_HEAD_DIM;
  shape.k_eq_v = ATTENTION_K_EQ_V ? 1u : 0u;
  shape.window = SLIDING_WINDOW;
  shape.vocab = NUM_VOCAB;
  shape.max_seq = MAX_SEQ_LEN;
  shape.eps = NORM_EPS;
  // a tied head folds the final norm and the softcap (#4415's
  // FOLD_OUTPUT_NORM), so the DSP caps; an untied one keeps the CPU's
  // logit_softcapping layer after the head, which caps the logits
  shape.softcap = TIE_WORD_EMBEDDINGS ? FINAL_LOGIT_SOFTCAPPING : 0.0f;
  std::vector<uint32_t> words(htp_graph_words_for(n_layers, HTP_GRAPH_MAX_OPS));
  const uint32_t n_words = htp_graph_gemma_build(
    words.data(), static_cast<uint32_t>(words.size()), &shape, is_full.data(),
    scalar.data(), HTP_GRAPH_KINDS_ALL);
  if (n_words == 0u)
    throw std::runtime_error("[Gemma4MoE] HTP decode: the decode op list "
                             "does not fit HTP_GRAPH_MAX_OPS");
  words.resize(n_words);
  auto *ops = nntrainer::get_htp_ops();
  if (!TIE_WORD_EMBEDDINGS && LMHEAD_DTYPE == "QS4CX")
    ops->set_lm_head_wh_cache((weight_path + ".lmhead_wh").c_str());
  if (!ops->set_decode_graph_desc(words))
    throw std::runtime_error(
      "[Gemma4MoE] HTP decode: the HTP backend has no per-token entry");

  // Bind semantic roles to graph-owned weights; wire operation order is
  // unchanged.
  const uint32_t n_ops = words[3], H = shape.hidden, E = NUM_EXPERTS;
  auto hand = [ops](uint32_t op, uint32_t which, const float *d, size_t n) {
    if (!ops->set_decode_graph_param(op, which, d, static_cast<unsigned>(n)))
      throw std::runtime_error(
        "[Gemma4MoE] HTP decode: the backend took no parameter");
  };
  htp_params.clear();
  for (uint32_t i = 0; i < n_ops; ++i) {
    const htp_graph_op *op = htp_graph_op_cat(words.data(), i);
    const uint32_t layer = op->layer;
    const std::string p = "layer" + std::to_string(layer);
    if (op->kind == HTP_OP_RMSNORM) {
      using Role = nntrainer::HtpBindingRole;
      std::string name;
      switch (nntrainer::htpBindingRole(words, i)) {
      case Role::InputNorm:
        name = p + "_qkv:in_norm_gamma";
        break;
      case Role::AttentionOutputNorm:
        name = p + "_post_attention_norm:gamma";
        break;
      case Role::MoeInputNorm:
        name = p + "_sparse_moe:in_norm_gamma";
        break;
      case Role::MoeOutputNorm:
        name = p + "_sparse_moe:out_norm_gamma";
        break;
      case Role::DenseInputNorm:
        name = p + "_ffn:in_norm_gamma";
        break;
      case Role::DenseOutputNorm:
        name = p + "_ffn:out_norm_gamma";
        break;
      case Role::FfnOutputNorm:
        name = binding.choose(p + "_sparse_moe:add_gamma",
                              p + "_post_ffn_norm:gamma");
        break;
      case Role::OutputNorm:
        name = TIE_WORD_EMBEDDINGS ? "output_of_causallm:gamma"
                                   : "output_norm:gamma";
        break;
      default:
        throw std::invalid_argument("Gemma HTP: unsupported norm role");
      }
      hand(i, HTP_GRAPH_PARAM_GAMMA, binding.f32(name, H), H);
    } else if (op->kind == HTP_OP_QK_NORM) {
      const uint32_t hd = op->head_dim;
      std::vector<float> g(2u * hd);
      std::copy_n(binding.f32(p + "_qkv:q_norm_gamma", hd), hd, g.begin());
      std::copy_n(binding.f32(p + "_qkv:k_norm_gamma", hd), hd, g.begin() + hd);
      htp_params.push_back(std::move(g));
      hand(i, HTP_GRAPH_PARAM_GAMMA, htp_params.back().data(), 2u * hd);
    } else if (op->kind == HTP_OP_ROUTER_TOPK) {
      // #4415's lfm2_moe (router_type softmax_scale): router [H][E], its
      // norm's gamma [H] with router.scale * H^-0.5 already folded in the
      // file, the per-expert scale [E]; ROUTER_BIAS is gamma | scale, as
      // the file has them
      const std::string m = p + "_sparse_moe:";
      hand(i, HTP_GRAPH_PARAM_ROUTER_W, binding.f32(m + "gate", size_t(H) * E),
           size_t(H) * E);
      std::vector<float> b(H + E);
      std::copy_n(binding.f32(m + "router_norm_gamma", H), H, b.begin());
      std::copy_n(binding.f32(m + "expert_bias", E), E, b.begin() + H);
      htp_params.push_back(std::move(b));
      hand(i, HTP_GRAPH_PARAM_ROUTER_BIAS, htp_params.back().data(), H + E);
    }
  }

  // The Q4_0 weights of the FC / DENSE_FFN / LM_HEAD ops, in list order:
  // q | k (| v), o; up, gate, down; the tied table.
  // An FC / dense weight may be QS4CX (fc_layer_dtype QS4CX): it
  // binds the WH handles the prefill registers from the same bytes
  // The LM_HEAD: Q4_0 as Q4M1 slices (tied: the embedding's own bytes), or
  // an untied QS4CX head (lmhead_dtype QS4CX) as WH slices like an FC, the
  // same handles serving the prefill's last row (lm_head_shared_fp32).
  auto q4 = [&binding, ops](const std::string &name, bool tied,
                            bool lm_head = false) {
    nntrainer::Tensor &t = binding.weight(name);
    const auto dt = t.getDataType();
    const bool wh = dt == ml::train::TensorDim::DataType::QS4CX_WH ||
                    dt == ml::train::TensorDim::DataType::QS4CX_WH_AFFINE;
    const bool qs4cx = wh || dt == ml::train::TensorDim::DataType::QS4CX;
    if (dt != ml::train::TensorDim::DataType::Q4_0 && !qs4cx)
      throw std::runtime_error("[Gemma4MoE] HTP decode: " + name + " is not " +
                               (lm_head
                                  ? "Q4_0 or QS4CX_WH, the LM_HEAD types"
                                  : "Q4_0, QS4CX or QS4CX_WH, the FC types"));
    const unsigned K = tied ? t.width() : t.height();
    const unsigned N = tied ? t.height() : t.width();
    if (wh &&
        (tied || !ops->set_fc_wh_source(
                   t.getData(), K, N,
                   dt == ml::train::TensorDim::DataType::QS4CX_WH_AFFINE)))
      throw std::runtime_error("Gemma4: WH decode weight registration failed");
    if (qs4cx ? !ops->add_decode_graph_qs4cx(t.getData<char>(),
                                             t.getScale<float>(), K, N)
              : !ops->add_decode_graph_q4_0(t.getData<char>(), K, N, tied))
      throw std::runtime_error(
        "[Gemma4MoE] HTP decode: the backend took no weight " + name);
  };
  for (uint32_t i = 0; i < n_ops; ++i) {
    const htp_graph_op *op = htp_graph_op_cat(words.data(), i);
    const uint32_t layer = op->layer;
    const std::string p = "layer" + std::to_string(layer);
    if (op->kind == HTP_OP_FC && nntrainer::htpBindingRole(words, i) ==
                                   nntrainer::HtpBindingRole::AttentionInput) {
      q4(p + "_qkv:qweight", false);
      q4(p + "_qkv:kweight", false);
      if (binding.contains(p + "_qkv:vweight")) // absent under k = v
        q4(p + "_qkv:vweight", false);
    } else if (op->kind == HTP_OP_FC) {
      if (nntrainer::htpBindingRole(words, i) !=
          nntrainer::HtpBindingRole::AttentionOutput)
        throw std::invalid_argument("Gemma HTP: unsupported FC role");
      q4(binding.choose(p + "_post_attention_norm:weight",
                        p + "_attention_out:weight"),
         false);
    } else if (op->kind == HTP_OP_DENSE_FFN) {
      // the list's order up, gate, down; the file's is gate-first
      q4(p + "_ffn:up", false);
      q4(p + "_ffn:gate", false);
      q4(p + "_ffn:down", false);
    } else if (op->kind == HTP_OP_LM_HEAD) {
      q4(TIE_WORD_EMBEDDINGS ? "embedding0:Embedding"
                             : "output_of_causallm:weight",
         TIE_WORD_EMBEDDINGS, true);
    }
  }
  std::fprintf(stderr,
               "[HTP] gemma: list n_ops=%u layers=%u params=%zu by name\n",
               n_ops, n_layers, htp_params.size());
#endif
  repack_weight();
}

void Gemma4HtpMoECausalLM::run(const WSTR prompt, bool do_sample,
                               const WSTR system_prompt, const WSTR tail_prompt,
                               bool log_output) {
  if (MOE_ENGINE == "htp" || ATTENTION_ENGINE == "htp" ||
      ATTN_PROJ_ENGINE == "htp" || FFN_ENGINE == "htp" ||
      LMHEAD_ENGINE == "htp") {
#ifdef ENABLE_HEXKL
    if (!htp_session_enabled_ || getSessionTokenPosition() == 0u)
      beginIndependentHtpPrompt(htp_run_started_);
    else if (htp_session_failed_)
      throw std::logic_error(
        "HTP KV session failed; reset its token position to zero");
#endif
    htp_run_started_ = true;
  }
  try {
    CausalLM::run(prompt, do_sample, system_prompt, tail_prompt, log_output);
  } catch (...) {
    htp_session_failed_ = true;
    throw;
  }
}

void Gemma4HtpMoECausalLM::repack_weight() {
  if (loading_weights_ || weights_prepared_)
    return;
  Gemma4HtpCausalLM::repack_weight();
#ifdef ENABLE_HEXKL
  // Finalize the decode FC arena after registering the experts.
  if (htp_e2e)
    nntrainer::get_htp_ops()->finish_decode_graph_q4_0();
#endif
  weights_prepared_ = true;
  weight_repacked_ = true;
}

} // namespace causallm
