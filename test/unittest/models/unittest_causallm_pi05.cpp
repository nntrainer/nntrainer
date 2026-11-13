// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   unittest_causallm_pi05.cpp
 * @brief  Differential test for the pi0.5 flow-matching model against the
 *         pure-torch tiny golden (pi05_tiny fixture).
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The fixture records the openpi inference pipeline at tiny scale:
 *   - backbone-ALONE bidirectional prefix pass, post-rope K/V per layer
 *     (bbN_k / bbN_v records — the C++ test transplants them into the
 *     shared cache directly, mirroring the backbone-graph capture),
 *   - 10-step expert-ONLY Euler loop from t=1 to 0 with adaRMS cond,
 *   - per-step v_t and the final action chunk in meta.json.
 *
 * The test drives PI05Model::sampleActions with the recorded noise and
 * golden vision embeddings (vision_out record) and diffs per-step v_t and
 * the final actions against the golden.
 */

#include <causallm_test_utils.h>
#include <gtest/gtest.h>

#include <pi05/pi05_model.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

#include "json.hpp"

using json = nlohmann::json;

namespace {

using namespace causallm_test;

/**
 * @brief locate the pi05_tiny fixture directory
 */
static std::filesystem::path pi05FixtureDir() {
  if (const char *env = std::getenv("NNTRAINER_CAUSALLM_FIXTURE_DIR"))
    return std::filesystem::path(env);
  std::filesystem::path base =
    std::filesystem::path(__FILE__).parent_path() / "causallm_reference";
  auto candidate = base / "pi05_tiny";
  if (std::filesystem::exists(candidate))
    return candidate;
  return candidate;
}

/**
 * @brief read one named record (extension-less, as the generator writes)
 */
static std::vector<float> readRecord(const std::filesystem::path &dir,
                                     const std::string &name) {
  std::ifstream f(dir / name, std::ios::binary | std::ios::ate);
  if (!f.is_open()) {
    ADD_FAILURE() << "cannot open record " << (dir / name).string();
    return {};
  }
  std::streamsize bytes = f.tellg();
  f.seekg(0);
  std::vector<float> v(static_cast<size_t>(bytes) / sizeof(float));
  f.read(reinterpret_cast<char *>(v.data()), bytes);
  return v;
}

/**
 * @brief parse a json file (the fixture meta + configs)
 */
static nlohmann::json readJson(const std::filesystem::path &p) {
  std::ifstream f(p);
  nlohmann::json j;
  f >> j;
  return j;
}

TEST(PI05TinyDifferentialTest, MatchesGoldenEulerLoop) {
  const auto dir = pi05FixtureDir();
  if (!std::filesystem::exists(dir / "meta.json")) {
    GTEST_SKIP() << "pi05_tiny fixture not generated";
  }
  const auto meta = readJson(dir / "meta.json");
  const auto cfg = readJson(dir / "config.json");
  const auto nntr_cfg = readJson(dir / "nntr_config.json");

  const unsigned int D = meta["hidden_size"].get<unsigned int>();
  const unsigned int DEPTH = meta["depth"].get<unsigned int>();
  const unsigned int KVW = meta["kv_width"].get<unsigned int>();
  const unsigned int Tb = meta["prefix_len"].get<unsigned int>();
  const unsigned int HORIZON = meta["horizon"].get<unsigned int>();
  const float ATOL = meta["atol_fp32"].get<float>();

  // ---- build the model (config jsons copied from the fixture) ----
  json model_cfg = cfg;
  json gen_cfg = readJson(dir / "generation_config.json");
  json n_cfg = nntr_cfg;
  // rebase file names onto the fixture dir
  n_cfg["model_file_name"] = (dir / "nntr_pi05_backbone.bin").string();
  n_cfg["embedding_bin_path"] = (dir / "nntr_pi05_emb.bin").string();
  n_cfg["expert_file_name"] = (dir / "nntr_pi05_expert.bin").string();

  auto model = std::make_unique<causallm::PI05Model>(model_cfg, gen_cfg, n_cfg);
  model->initialize();
  model->load_weight(n_cfg["model_file_name"].get<std::string>());
  model->repack_weight();

  // ---- golden vision embeddings (the tiny test injects; no vision graph) --
  std::vector<float> vision_out =
    readRecord(dir, "vision_out"); // [num_patches, D]
  model->setVisionEmbeddings(vision_out,
                             static_cast<unsigned int>(vision_out.size() / D));

  // ---- prompt ids: bos + num_patches image tokens + text ids ----
  // openpi embed_prefix order: image rows FIRST, then language (bos+text)
  std::vector<unsigned int> prompt_ids;
  for (unsigned int i = 0; i < vision_out.size() / D; ++i)
    prompt_ids.push_back(meta["image_token_index"].get<unsigned int>());
  // meta["prompt_ids"] starts with bos (2) already — do not push another
  for (int id : meta["prompt_ids"])
    prompt_ids.push_back(static_cast<unsigned int>(id));

  // golden transplant sanity (compare against [pi05-dbg] transplanted line)
  if (std::getenv("NNTR_PI05_DEBUG")) {
    std::vector<float> gk = readRecord(dir, "bb0_k");
    std::fprintf(stderr,
                 "[pi05-dbg] golden     bb0_k[0..3]=%.6f,%.6f,%.6f,%.6f\n",
                 gk[0], gk[1], gk[2], gk[3]);
  }

  // ---- run the Euler loop with the recorded noise ----
  std::vector<float> noise;
  for (const auto &row : meta["noise"])
    for (const auto &v : row)
      noise.push_back(v.get<float>());

  std::vector<float> actions = model->sampleActions(
    prompt_ids, noise, meta["num_steps"].get<unsigned int>());

  EXPECT_THROW(model->sampleActions(prompt_ids, noise, 0),
               std::invalid_argument);
  const auto repeated = model->sampleActions(
    prompt_ids, noise, meta["num_steps"].get<unsigned int>());
  ASSERT_EQ(actions.size(), repeated.size());
  for (size_t i = 0; i < actions.size(); ++i)
    EXPECT_NEAR(actions[i], repeated[i], 1e-5f) << "repeated inference " << i;

  // ---- diff final actions ----
  std::vector<float> golden;
  for (const auto &row : meta["final_actions"])
    for (const auto &v : row)
      golden.push_back(v.get<float>());
  ASSERT_EQ(actions.size(), golden.size());
  for (size_t i = 0; i < golden.size(); ++i)
    EXPECT_NEAR(actions[i], golden[i], ATOL) << "action[" << i << "]";

  // ---- diff per-step v_t ----
  const auto &steps = model->perStepVelocities();
  ASSERT_EQ(steps.size(), meta["per_step_v_t"].size());
  for (size_t s = 0; s < steps.size(); ++s) {
    std::vector<float> gstep;
    for (const auto &row : meta["per_step_v_t"][s])
      for (const auto &v : row)
        gstep.push_back(v.get<float>());
    ASSERT_EQ(steps[s].size(), gstep.size());
    for (size_t i = 0; i < gstep.size(); ++i)
      EXPECT_NEAR(steps[s][i], gstep[i], ATOL)
        << "v_t step " << s << " element " << i;
  }

  // Modulation may be reused across observations, but its Euler schedule
  // must change with num_steps and be invalidated when weights are loaded.
  auto varied_noise = noise;
  varied_noise.at(0) += 0.125f;
  const unsigned int changed_steps = meta["num_steps"].get<unsigned int>() + 1;
  const auto varied =
    model->sampleActions(prompt_ids, varied_noise, changed_steps);
  auto fresh = std::make_unique<causallm::PI05Model>(model_cfg, gen_cfg, n_cfg);
  fresh->initialize();
  fresh->load_weight(n_cfg["model_file_name"].get<std::string>());
  fresh->setVisionEmbeddings(vision_out,
                             static_cast<unsigned int>(vision_out.size() / D));
  const auto fresh_varied =
    fresh->sampleActions(prompt_ids, varied_noise, changed_steps);
  ASSERT_EQ(varied.size(), fresh_varied.size());
  for (size_t i = 0; i < varied.size(); ++i)
    EXPECT_NEAR(varied[i], fresh_varied[i], ATOL) << "changed schedule " << i;
  model->load_weight(n_cfg["model_file_name"].get<std::string>());
  const auto reloaded = model->sampleActions(
    prompt_ids, noise, meta["num_steps"].get<unsigned int>());
  ASSERT_EQ(reloaded.size(), golden.size());
  for (size_t i = 0; i < golden.size(); ++i)
    EXPECT_NEAR(reloaded[i], golden[i], ATOL) << "weights reloaded " << i;
}

} // namespace
