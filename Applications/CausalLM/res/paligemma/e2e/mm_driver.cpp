// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   mm_driver.cpp
 * @brief  Standalone e2e verification driver for the PaliGemma port.
 * @date   12 Nov 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Loads the converted SiglipVisionModel (vision dir) and PaliGemmaCausalLM
 * (text dir), feeds the HF processor's pixel_values verbatim (no
 * re-preprocessing, so bicubic-vs-bilinear never diverges), composes the
 * merged embedding plane following input_ids.json (image rows = raw
 * projector output, text rows = lookupEmbedding() scaled by sqrt(hidden)),
 * and drives the prefix-LM orchestration: the leading token_type_ids==0
 * block (image tokens, and bos when the processor folds it in) is prefilled
 * in ONE bidirectional forward, the text suffix is fed one row at a time
 * (causal), and generation continues on the decode path. Generated tokens
 * are compared against the HF greedy reference (mm_tokens.json) dumped by
 * golden_dump.py.
 *
 * usage: nntr_paligemma_e2e <text_model_dir> <vision_model_dir> <golden_dir>
 */

#include <cmath>
#include <factory.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../e2e_validation.h"
#include "json.hpp"
#include "paligemma/paligemma_causallm.h"
#include "paligemma/siglip_vision.h"

namespace {

using json = nlohmann::json;

json loadJson(const std::string &path) {
  std::ifstream f(path);
  if (!f.is_open())
    throw std::runtime_error("failed to open " + path);
  return json::parse(f);
}

std::vector<int> loadIds(const std::string &path) {
  return loadJson(path).get<std::vector<int>>();
}

std::string rebasePath(const std::string &dir, const json &cfg,
                       const char *key) {
  if (!cfg.contains(key) || !cfg[key].is_string())
    return "";
  std::filesystem::path p = cfg[key].get<std::string>();
  if (p.empty() || p.is_absolute())
    return p.string();
  return (std::filesystem::path(dir) / p).string();
}

} // namespace

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "usage: " << argv[0]
              << " <text_model_dir> <vision_model_dir> <golden_dir>\n";
    return 1;
  }
  const std::string text_dir = argv[1];
  const std::string vision_dir = argv[2];
  std::string golden_dir = argv[3];
  const char *alternate = std::getenv("NNTR_VALIDATE_ALTERNATE_REFERENCE");

  const auto load_start = e2e::Clock::now();
  try {
    // This driver registers the two factories it creates directly; Factory
    // shares one process-wide instance (see factory.h).
    causallm::Factory::Instance().registerModel(
      "PaliGemmaForCausalLM",
      [](json &cfg, json &generation_cfg, json &nntr_cfg) {
        return std::make_unique<causallm::PaliGemmaCausalLM>(
          cfg, generation_cfg, nntr_cfg);
      });
    causallm::Factory::Instance().registerModel(
      "SiglipVisionModel", [](json &cfg, json &generation_cfg, json &nntr_cfg) {
        return std::make_unique<causallm::SiglipVisionModel>(
          cfg, generation_cfg, nntr_cfg);
      });

    // ---- vision producer ----
    json vcfg = loadJson(vision_dir + "/config.json");
    json vgen = json::object();
    json vnntr = loadJson(vision_dir + "/nntr_config.json");
    auto vision = causallm::Factory::Instance().create("SiglipVisionModel",
                                                       vcfg, vgen, vnntr);
    if (vision == nullptr)
      throw std::runtime_error("failed to create SiglipVisionModel");
    if (argc > 4 && std::string(argv[4]) == "--preprocess-only") {
      const auto decoded =
        dynamic_cast<causallm::SiglipVisionModel *>(vision.get())
          ->preprocessImage(golden_dir + "/sample.png");
      if (const char *dump = std::getenv("NNTR_VALIDATE_PIXEL_DUMP")) {
        std::ofstream f(dump, std::ios::binary);
        f.write(reinterpret_cast<const char *>(decoded.data()),
                decoded.size() * sizeof(float));
      }
      return e2e::compare("file_preprocessing", decoded,
                          e2e::readFloats(golden_dir + "/pixel_values.bin"),
                          2e-7f)
               ? 0
               : 2;
    }
    vision->initialize();
    vision->load_weight(rebasePath(vision_dir, vnntr, "model_file_name"));
    vision->repack_weight();
    std::cout << "[driver] vision model loaded\n";

    // ---- text consumer ----
    json tcfg = loadJson(text_dir + "/config.json");
    json tgen = loadJson(text_dir + "/generation_config.json");
    json tnntr = loadJson(text_dir + "/nntr_config.json");
    // Rebase relative paths like main.cpp does — the bins and the tokenizer
    // live next to nntr_config.json, not in the driver's working directory.
    // A missed tokenizer_file leaves the tokenizer unloaded and
    // registerOutputs() dereferences it unconditionally.
    for (const char *key :
         {"embedding_bin_path", "tokenizer_file", "model_file_name"}) {
      if (tnntr.contains(key) && tnntr[key].is_string()) {
        tnntr[key] = rebasePath(text_dir, tnntr, key);
      }
    }
    tnntr["num_to_generate"] = loadIds(golden_dir + "/mm_tokens.json").size();
    if (alternate)
      tnntr["num_to_generate"] =
        std::max(tnntr["num_to_generate"].get<size_t>(),
                 loadIds(std::string(alternate) + "/mm_tokens.json").size());
    auto llm = causallm::Factory::Instance().create("PaliGemmaForCausalLM",
                                                    tcfg, tgen, tnntr);
    if (llm == nullptr)
      throw std::runtime_error("failed to create PaliGemmaCausalLM");
    llm->initialize();
    llm->load_weight(rebasePath(text_dir, tnntr, "model_file_name"));
    llm->repack_weight();
    std::cout << "[driver] text model loaded\n";
    // Transformer is a VIRTUAL base of the diamond — static_cast down from
    // it is ill-formed; dynamic_cast is the only legal downcast here.
    auto *mm = dynamic_cast<causallm::PaliGemmaCausalLM *>(llm.get());
    if (mm == nullptr)
      throw std::runtime_error("llm is not a PaliGemmaCausalLM");

    const double load_s = e2e::seconds(load_start);
    const char *repeat_env = std::getenv("NNTR_BENCH_REPEATS");
    const int repeats = repeat_env ? std::stoi(repeat_env) : 1;
    if (repeats < 1 || repeats > 100)
      throw std::invalid_argument("NNTR_BENCH_REPEATS must be in [1,100]");
    for (int request = 0; request < repeats; ++request) {
      golden_dir = alternate && request % 2 ? alternate : argv[3];
      std::cout << "[request] index=" << request << " reference=" << golden_dir
                << "\n";
      mm->setNumToGenerate(loadIds(golden_dir + "/mm_tokens.json").size());

      // ---- golden inputs ----
      const std::vector<int> input_ids =
        loadIds(golden_dir + "/input_ids.json");
      const std::vector<int> token_type_ids =
        loadIds(golden_dir + "/token_type_ids.json");
      const int image_token_index = tcfg["image_token_index"].get<int>();
      const unsigned int dim =
        tcfg["text_config"]["hidden_size"].get<unsigned int>();

      // Feed the HF processor's pixel_values verbatim.
      std::ifstream pf(golden_dir + "/pixel_values.bin", std::ios::binary);
      if (!pf.is_open())
        throw std::runtime_error("failed to open pixel_values.bin");
      pf.seekg(0, std::ios::end);
      const size_t pixel_bytes = static_cast<size_t>(pf.tellg());
      pf.seekg(0, std::ios::beg);
      if (pixel_bytes % sizeof(float) != 0)
        throw std::runtime_error("pixel_values.bin size is not float-aligned");
      std::vector<float> pixels(pixel_bytes / sizeof(float));
      pf.read(reinterpret_cast<char *>(pixels.data()),
              static_cast<std::streamsize>(pixel_bytes));
      if (pixels.size() % 3 != 0)
        throw std::runtime_error("pixel_values.bin is not a CHW image");
      const unsigned int img_side = static_cast<unsigned int>(
        std::lround(std::sqrt(static_cast<double>(pixels.size() / 3))));

      auto *siglip = dynamic_cast<causallm::SiglipVisionModel *>(vision.get());
      if (std::filesystem::exists(golden_dir + "/sample.png")) {
        const auto decoded =
          siglip->preprocessImage(golden_dir + "/sample.png");
        if (!e2e::compare("file_preprocessing", decoded, pixels, 2e-7f))
          throw std::runtime_error(
            "file preprocessing differs from HF processor");
      }
      const auto vision_start = e2e::Clock::now();
      auto image_out = vision->run_image(
        "", {pixels.data(), pixels.size() * sizeof(float)},
        static_cast<int>(img_side), static_cast<int>(img_side));
      if (image_out.first == nullptr)
        throw std::runtime_error("vision model produced no embeddings");
      std::unique_ptr<float, void (*)(float *)> img_guard(
        static_cast<float *>(image_out.first), [](float *p) { delete[] p; });
      const size_t n_image_rows = image_out.second / (dim * sizeof(float));
      std::cout << "[driver] image embeddings: " << n_image_rows << " rows of "
                << dim << "\n";

      const double vision_s = e2e::seconds(vision_start);
      bool numeric_pass = true;
      if (std::filesystem::exists(golden_dir + "/image_emb.bin")) {
        const auto ref = e2e::readFloats(golden_dir + "/image_emb.bin");
        const auto *begin = static_cast<const float *>(image_out.first);
        numeric_pass &= e2e::compare(
          "vision", {begin, begin + image_out.second / sizeof(float)}, ref,
          0.03f, 0.002f);
      }
      // ---- compose the merged embedding plane ----
      const size_t n_total = input_ids.size();
      size_t img_cursor = 0;
      size_t prefix_len = n_total;
      for (size_t i = 0; i < token_type_ids.size() && i < n_total; ++i) {
        if (token_type_ids[i] != 0) {
          prefix_len = i; // first text row: everything before it is the prefix
          break;
        }
      }
      std::vector<float> plane(n_total * dim);
      for (size_t i = 0; i < n_total; ++i) {
        float *dst = plane.data() + i * dim;
        if (input_ids[i] == image_token_index) {
          if (img_cursor >= n_image_rows)
            throw std::runtime_error("not enough image embeddings for the "
                                     "image-token positions");
          const float *src =
            static_cast<const float *>(image_out.first) + img_cursor * dim;
          std::copy(src, src + dim, dst); // raw — HF injects it unscaled
          ++img_cursor;
        } else {
          std::vector<float> row =
            mm->lookupEmbedding(static_cast<unsigned int>(input_ids[i]));
          std::copy(row.begin(), row.end(), dst); // scaled by sqrt(hidden)
        }
      }
      std::cout << "[driver] prefix_len=" << prefix_len
                << " n_total=" << n_total << "\n";

      const auto inference_start = e2e::Clock::now();
      mm->setSessionTokenPosition(0);
      mm->run_with_embeddings(plane.data(), n_total, prefix_len, input_ids,
                              false, true);

      const double inference_s = e2e::seconds(inference_start);
      const auto reference_logits =
        loadJson(golden_dir + "/mm_logits.json").get<std::vector<float>>();
      numeric_pass &=
        e2e::compare("prefill_logits", mm->prefillLogits(), reference_logits,
                     tnntr["model_tensor_type"].get<std::string>().find(
                       "-FP16") != std::string::npos
                       ? 0.1f
                       : 0.01f,
                     0.001f);
      std::cout << "[timing] load_s=" << load_s << " vision_s=" << vision_s
                << " inference_s=" << inference_s
                << " total_inference_s=" << vision_s + inference_s << "\n";
      const auto &generated = mm->getGeneratedIds();
      const auto reference = loadIds(golden_dir + "/mm_tokens.json");
      size_t match = 0;
      for (; match < generated.size() && match < reference.size(); ++match) {
        if (static_cast<int>(generated[match]) != reference[match])
          break;
      }
      std::cout << "[driver] generated " << generated.size() << " tokens, HF "
                << reference.size() << ", prefix match " << match << "\n";
      std::cout << "[driver] nntr tokens:";
      for (auto t : generated)
        std::cout << " " << t;
      std::cout << "\n[driver]  HF tokens:";
      for (auto t : reference)
        std::cout << " " << t;
      std::cout << "\n";

      const bool pass = match == reference.size() &&
                        generated.size() == reference.size() &&
                        !generated.empty() && numeric_pass;
      std::cout << "[driver] " << (pass ? "PASS" : "MISMATCH") << "\n";
      if (!pass)
        return 2;
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "[driver] error: " << e.what() << "\n";
    return 1;
  }
}
