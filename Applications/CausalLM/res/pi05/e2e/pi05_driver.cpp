// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   pi05_driver.cpp
 * @brief  Standalone e2e verification driver for the pi0.5 port.
 * @date   12 Nov 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Loads the converted SiglipVisionModel (vision dir) and PI05Model (text
 * dir), feeds the openpi golden's pixel_values verbatim, and drives the
 * full inference pipeline: vision -> projector rows -> PI05Model::
 * sampleActions (backbone prefix pass + K/V transplant + expert Euler
 * loop). The final action chunk is compared against the openpi reference
 * (actions.bin / meta.json dumped by golden_dump.py).
 *
 * usage: nntr_pi05_e2e <text_model_dir> <vision_model_dir> <golden_dir>
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
#include "paligemma/siglip_vision.h"
#include "pi05/pi05_model.h"

namespace {

using json = nlohmann::json;

json loadJson(const std::string &path) {
  std::ifstream f(path);
  if (!f.is_open())
    throw std::runtime_error("failed to open " + path);
  return json::parse(f);
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

std::vector<float> loadBin(const std::string &path, size_t expected_floats,
                           const char *what) {
  // openpi saves [1, horizon, dim] — a leading batch dim of 1 changes the
  // FLOAT COUNT not at all, so the flat count is the contract here.
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f.is_open())
    throw std::runtime_error(std::string("failed to open ") + what);
  const std::streamsize bytes = f.tellg();
  f.seekg(0);
  if (static_cast<size_t>(bytes) % sizeof(float) != 0)
    throw std::runtime_error(std::string(what) + " is not float-aligned");
  std::vector<float> v(static_cast<size_t>(bytes) / sizeof(float));
  f.read(reinterpret_cast<char *>(v.data()), bytes);
  if (!expected_floats || v.size() == expected_floats)
    return v;
  throw std::runtime_error(std::string(what) + ": expected " +
                           std::to_string(expected_floats) + " floats, got " +
                           std::to_string(v.size()));
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
    causallm::Factory::Instance().registerModel(
      "PI05Model", [](json &cfg, json &generation_cfg, json &nntr_cfg) {
        return std::make_unique<causallm::PI05Model>(cfg, generation_cfg,
                                                     nntr_cfg);
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
    vision->initialize();
    vision->load_weight(rebasePath(vision_dir, vnntr, "model_file_name"));
    vision->repack_weight();
    std::cout << "[driver] vision model loaded\n";

    // ---- pi0.5 consumer ----
    json tcfg = loadJson(text_dir + "/config.json");
    json tgen = loadJson(text_dir + "/generation_config.json");
    json tnntr = loadJson(text_dir + "/nntr_config.json");
    for (const char *key : {"embedding_bin_path", "tokenizer_file",
                            "model_file_name", "expert_file_name"}) {
      if (tnntr.contains(key) && tnntr[key].is_string())
        tnntr[key] = rebasePath(text_dir, tnntr, key);
    }
    auto model =
      causallm::Factory::Instance().create("PI05Model", tcfg, tgen, tnntr);
    if (model == nullptr)
      throw std::runtime_error("failed to create PI05Model");
    model->initialize();
    model->load_weight(rebasePath(text_dir, tnntr, "model_file_name"));
    model->repack_weight();
    std::cout << "[driver] pi05 model loaded\n";
    auto *pi05 = dynamic_cast<causallm::PI05Model *>(model.get());
    if (pi05 == nullptr)
      throw std::runtime_error("model is not a PI05Model");

    const double load_s = e2e::seconds(load_start);
    const char *repeat_env = std::getenv("NNTR_BENCH_REPEATS");
    const int repeats = repeat_env ? std::stoi(repeat_env) : 1;
    if (repeats < 1 || repeats > 100)
      throw std::invalid_argument("NNTR_BENCH_REPEATS must be in [1,100]");
    for (int request = 0; request < repeats; ++request) {
      golden_dir = alternate && request % 2 ? alternate : argv[3];
      std::cout << "[request] index=" << request << " reference=" << golden_dir
                << "\n";

      // ---- golden inputs ----
      const auto meta = loadJson(golden_dir + "/meta.json");
      const unsigned int dim =
        tcfg["text_config"]["hidden_size"].get<unsigned int>();
      const unsigned int horizon = meta["action_horizon"].get<unsigned int>();
      const unsigned int action_dim = meta["action_dim"].get<unsigned int>();

      std::ifstream pf(golden_dir + "/pixel_values.bin", std::ios::binary);
      if (!pf.is_open())
        throw std::runtime_error("failed to open pixel_values.bin");
      pf.seekg(0, std::ios::end);
      const size_t pixel_bytes = static_cast<size_t>(pf.tellg());
      pf.seekg(0, std::ios::beg);
      std::vector<float> pixels(pixel_bytes / sizeof(float));
      pf.read(reinterpret_cast<char *>(pixels.data()),
              static_cast<std::streamsize>(pixel_bytes));
      const unsigned int img_side = static_cast<unsigned int>(
        std::lround(std::sqrt(static_cast<double>(pixels.size() / 3))));

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
      const auto ref_vision = e2e::readFloats(golden_dir + "/image_emb.bin");
      const auto *vision_data = static_cast<const float *>(image_out.first);
      bool numeric_pass =
        e2e::compare("vision", {vision_data, vision_data + n_image_rows * dim},
                     ref_vision, 0.03f, 0.002f);
      // ---- prompt ids: images first (openpi order), then bos + text ----
      std::vector<float> vision_rows(
        static_cast<const float *>(image_out.first),
        static_cast<const float *>(image_out.first) + n_image_rows * dim);
      pi05->setVisionEmbeddings(vision_rows,
                                static_cast<unsigned int>(n_image_rows));

      std::vector<unsigned int> prompt_ids;
      const unsigned int image_token = 257152u; // pi05 image_token_index
      for (size_t i = 0; i < n_image_rows; ++i)
        prompt_ids.push_back(image_token);
      for (int id : meta["lang_ids_unpadded"]) {
        if (id != 0) // strip the max_token_len padding
          prompt_ids.push_back(static_cast<unsigned int>(id));
        else
          break; // padding is contiguous
      }

      // ---- noise (batch dim stripped) ----
      std::vector<float> noise =
        loadBin(golden_dir + "/noise.bin",
                static_cast<size_t>(horizon) * action_dim, "noise.bin");

      // ---- cross-check the transplant + cond against the openpi dump ----
      if (std::getenv("NNTR_PI05_DEBUG")) {
        auto dump4 = [](const std::string &path, const char *what) {
          std::vector<float> v = loadBin(path, 0, what); // size-agnostic read
          std::cout << "[driver] " << what << "[0..3]: ";
          for (size_t i = 0; i < 4 && i < v.size(); ++i)
            std::cout << v[i] << " ";
          std::cout << "\n";
          return v;
        };
        std::vector<float> gk =
          dump4(golden_dir + "/bb0_k.bin", "golden bb0_k");
        (void)gk;
        std::vector<float> gc =
          dump4(golden_dir + "/cond0.bin", "golden cond0");
        (void)gc;
        std::vector<float> gimg =
          dump4(golden_dir + "/image_emb.bin", "golden image_emb");
        // also print OUR vision rows for the same positions
        std::cout << "[driver]  ours  image_emb[0..3]: ";
        for (size_t i = 0; i < 4 && i < vision_rows.size(); ++i)
          std::cout << vision_rows[i] << " ";
        std::cout << "\n";
        (void)gimg;
      }

      // ---- run the flow loop ----
      const auto inference_start = e2e::Clock::now();
      const std::vector<float> actions = pi05->sampleActions(
        prompt_ids, noise, meta["num_steps"].get<unsigned int>());

      const double inference_s = e2e::seconds(inference_start);
      const auto ref_velocity = e2e::readFloats(golden_dir + "/v_t.bin");
      std::vector<float> velocities;
      for (const auto &step : pi05->perStepVelocities())
        velocities.insert(velocities.end(), step.begin(), step.end());
      numeric_pass &= e2e::compare("per_step_velocity", velocities,
                                   ref_velocity, meta["atol_v_t"].get<float>());
      std::cout << "[timing] load_s=" << load_s << " vision_s=" << vision_s
                << " inference_s=" << inference_s
                << " total_inference_s=" << vision_s + inference_s << "\n";
      // ---- step-0 v_t cross-check ----
      if (std::getenv("NNTR_PI05_DEBUG")) {
        std::vector<float> golden_vt =
          loadBin(golden_dir + "/v_t.bin", 0, "v_t.bin");
        std::cout << "[driver] golden step0 v_t[0..4]: ";
        for (size_t i = 0; i < 5 && i < golden_vt.size(); ++i)
          std::cout << golden_vt[i] << " ";
        std::cout << "\n";
      }

      // ---- compare against the openpi final actions ----
      std::vector<float> golden_actions =
        loadBin(golden_dir + "/actions.bin",
                static_cast<size_t>(horizon) * action_dim, "actions.bin");

      const float atol = meta["atol_actions"].get<float>();
      size_t worst = 0;
      float worst_diff = 0.0f;
      size_t n_bad = 0;
      for (size_t i = 0; i < golden_actions.size(); ++i) {
        const float diff = std::abs(actions[i] - golden_actions[i]);
        if (diff > worst_diff) {
          worst_diff = diff;
          worst = i;
        }
        if (!(diff <= atol))
          ++n_bad;
      }
      std::cout << "[driver] actions: " << actions.size()
                << " elems, max|diff|=" << worst_diff << " at " << worst
                << " (atol " << atol << "), out-of-tol: " << n_bad << "\n";
      std::cout << "[driver] nntr actions[0..4]: ";
      for (size_t i = 0; i < 5 && i < actions.size(); ++i)
        std::cout << actions[i] << " ";
      std::cout << "\n[driver]  HF actions[0..4]: ";
      for (size_t i = 0; i < 5 && i < golden_actions.size(); ++i)
        std::cout << golden_actions[i] << " ";
      std::cout << "\n";

      const bool pass = n_bad == 0 && numeric_pass;
      std::cout << (pass ? "[driver] PASS" : "[driver] FAIL") << "\n";
      if (!pass)
        return 1;
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "[driver] ERROR: " << e.what() << "\n";
    return 1;
  }
}
