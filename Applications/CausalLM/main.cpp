/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	main.cpp
 * @date	23 July 2025
 * @brief	This is a main file for CausalLM application
 * @see		https://github.com/nnstreamer/
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#include <algorithm>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "json.hpp"
#include <app_context.h>
#include <factory.h>

#include "causal_lm.h"
#include "chat_template.h"
#include "deberta_v2.h"
#include "embedding_gemma.h"
#include "gemma3_causallm.h"
#include "gemma4_causallm.h"
#if !defined(_WIN32)
#include "gptoss_cached_slim_causallm.h"
#endif
#include "gptoss_causallm.h"
#if !defined(_WIN32) && !defined(__ANDROID__)
#include "multilingual_tinybert_16mb.h"
#endif
#include "qwen2_causallm.h"
#include "qwen2_embedding.h"
#if !defined(_WIN32)
#include "qwen3_cached_slim_moe_causallm.h"
#endif
#include "lfm2_causallm.h"
#include "lfm2-vl/lfm2_vl_model.h"
#include "qwen3_causallm.h"
#include "qwen3_embedding.h"
#include "qwen3_moe_causallm.h"
#include "qwen3_slim_moe_causallm.h"
#include "timm_vit/timm_vit_transformer.h"
#include "vjepa2_vit/vjepa2_vit.h"
#include "vjepa_lfm2_vl/vjepa_lfm2_vl.h"
#include <models/gemma3/function.h>
#if !defined(_WIN32)
#include <sys/resource.h>
#endif

#include <atomic>
#include <chrono>
#include <filesystem>
#include <thread>

using json = nlohmann::json;

std::atomic<size_t> peak_rss_kb{0};
std::atomic<bool> tracking_enabled{true};

/**
 * @brief Print the maximum resident set size for the current process.
 */
void printMemoryUsage() {
#if defined(_WIN32)
  std::cout << "Max Resident Set Size: unavailable on Windows" << std::endl;
#else
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  std::cout << "Max Resident Set Size: " << usage.ru_maxrss << " KB"
            << std::endl;
#endif
}

/**
 * @brief Read the current process resident set size on Linux.
 */
size_t read_vm_rss_kb() {
#if defined(_WIN32)
  return 0;
#else
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.find("VmRSS:") == 0) {
      size_t kb = 0;
      sscanf(line.c_str(), "VmRSS: %zu kB", &kb);
      return kb;
    }
  }
  return 0;
#endif
}

/**
 * @brief Read private resident memory from smaps_rollup on Linux.
 */
size_t read_private_rss_kb() {
#if defined(_WIN32)
  return 0;
#else
  std::ifstream smaps("/proc/self/smaps_rollup");
  std::string line;
  size_t total = 0;
  while (std::getline(smaps, line)) {
    if (line.find("Private_Clean:") == 0 || line.find("Private_Dirty:") == 0) {
      size_t kb;
      sscanf(line.c_str(), "%*s %zu", &kb);
      total += kb;
    }
  }
  return total;
#endif
}

/**
 * @brief Start a background sampler for peak private RSS.
 */
void start_peak_tracker() {
  std::thread([] {
    while (tracking_enabled.load()) {
      size_t current = read_private_rss_kb();
      size_t prev = peak_rss_kb.load();
      if (current > prev) {
        peak_rss_kb.store(current);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }).detach();
}

/**
 * @brief Stop the memory sampler and print the observed peak.
 */
void stop_and_print_peak() {
  tracking_enabled.store(false);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  std::cout << "Peak memory usage (VmRSS): " << peak_rss_kb.load() << " KB"
            << std::endl;
}

/**
 * @brief Resolve config architecture names to registered model factory names.
 */
std::string resolve_architecture(std::string model_type,
                                 const std::string &architecture) {
  std::transform(model_type.begin(), model_type.end(), model_type.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (model_type == "embedding") {
    if (architecture == "Qwen3ForCausalLM") {
      return "Qwen3Embedding";
    } else if (architecture == "Gemma3ForCausalLM" ||
               architecture == "Gemma3TextModel") {
      return "EmbeddingGemma";
    } else if (architecture == "Qwen2Model") {
      return "Qwen2Embedding";
    } else if (architecture == "BertForMaskedLM") {
      return "MultilingualTinyBert";
    } else if (architecture == "TimmViT" ||
               architecture == "vit_base_patch16_siglip_224") {
      return "TimmViT";
    } else if (architecture == "deberta-v2" ||
               architecture == "DebertaV2Model" ||
               architecture == "DebertaV2ForMaskedLM") {
      return "DebertaV2";
    } else {
      throw std::invalid_argument(
        "Unsupported architecture for embedding model: " + architecture);
    }
  }

  if (architecture == "TimmViT" ||
      architecture == "vit_base_patch16_siglip_224") {
    return "TimmViT";
  }

  if (architecture == "VJEPA2ViT" || architecture == "vjepa2_1_vit_base_384") {
    return "VJEPA2ViT";
  }

  if (architecture == "Lfm2VLVJepa21BModel" ||
      architecture == "vora_lfm2_vl_vjepa2_1_b") {
    return "Lfm2VLVJepa21BModel";
  }

  if (architecture == "Gemma4ForConditionalGeneration") {
    return "Gemma4ForCausalLM";
  }

  return architecture;
}

/**
 * @brief Entry point for loading, initializing, and running a CausalLM model.
 */
int main(int argc, char *argv[]) {

  auto start_time = std::chrono::high_resolution_clock::now();

  /** Register all runnable causallm models to factory */
  causallm::Factory::Instance().registerModel(
    "LlamaForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::CausalLM>(cfg, generation_cfg,
                                                  nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Qwen2ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen2CausalLM>(cfg, generation_cfg,
                                                       nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Qwen2Embedding", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen2Embedding>(cfg, generation_cfg,
                                                        nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Qwen3ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3CausalLM>(cfg, generation_cfg,
                                                       nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Qwen3MoeForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3MoECausalLM>(cfg, generation_cfg,
                                                          nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Qwen3SlimMoeForCausalLM",
    [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3SlimMoECausalLM>(
        cfg, generation_cfg, nntr_cfg);
    });
#if !defined(_WIN32)
  causallm::Factory::Instance().registerModel(
    "Qwen3CachedSlimMoeForCausalLM",
    [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3CachedSlimMoECausalLM>(
        cfg, generation_cfg, nntr_cfg);
    });
#endif
  causallm::Factory::Instance().registerModel(
    "Qwen3Embedding", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3Embedding>(cfg, generation_cfg,
                                                        nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "GptOssForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::GptOssForCausalLM>(cfg, generation_cfg,
                                                           nntr_cfg);
    });
#if !defined(_WIN32)
  causallm::Factory::Instance().registerModel(
    "GptOssCachedSlimCausalLM",
    [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::GptOssCachedSlimCausalLM>(
        cfg, generation_cfg, nntr_cfg);
    });
#endif
  causallm::Factory::Instance().registerModel(
    "Gemma3ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Gemma3CausalLM>(cfg, generation_cfg,
                                                        nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Gemma4ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Gemma4CausalLM>(cfg, generation_cfg,
                                                        nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "EmbeddingGemma", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::EmbeddingGemma>(cfg, generation_cfg,
                                                        nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "DebertaV2", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::DebertaV2>(cfg, generation_cfg,
                                                   nntr_cfg);
    });
#if !defined(_WIN32) && !defined(__ANDROID__)
  causallm::Factory::Instance().registerModel(
    "MultilingualTinyBert", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::MultilingualTinyBert>(
        cfg, generation_cfg, nntr_cfg);
    });
#endif
  causallm::Factory::Instance().registerModel(
    "TimmViT", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::TimmViTTransformer>(cfg, generation_cfg,
                                                            nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "VJEPA2ViT", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::VJEPA2ViT>(cfg, generation_cfg,
                                                   nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Lfm2ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Lfm2CausalLM>(cfg, generation_cfg,
                                                      nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Lfm2VLVJepa21BModel", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::VjepaLfm2ForConditionalGeneration>(
        cfg, generation_cfg, nntr_cfg);
    });

  // Validate arguments
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model_path> [input_prompt]\n"
              << "  <model_path>   : Path to model directory\n"
              << "  [input_prompt] : Optional input text (uses sample_input or "
                 "chat_input if omitted)\n";
    return EXIT_FAILURE;
  }

  const std::string model_path = argv[1];
  std::string input_text;
  std::string system_head_prompt = "";
  std::string system_tail_prompt = "";

  std::cout << model_path << std::endl;

  try {
    // Load configuration files
    json cfg = causallm::LoadJsonFile(model_path + "/config.json");
    json generation_cfg = json::object();
    std::string generation_config_path = model_path + "/generation_config.json";
    if (std::filesystem::exists(generation_config_path)) {
      generation_cfg = causallm::LoadJsonFile(generation_config_path);
    }
    json nntr_cfg = causallm::LoadJsonFile(model_path + "/nntr_config.json");
    // Resolve relative paths in nntr_cfg against model_path so that
    // config files with bare filenames work when run as:
    //   nntr_causallm <model_dir>
    // Absolute paths (e.g. for CI override) are left unchanged.
    {
      auto resolve_path = [&](const std::string &key) {
        if (!nntr_cfg.contains(key))
          return;
        const auto val = nntr_cfg[key];
        if (!val.is_string())
          return;
        const std::string s = val.get<std::string>();
        if (s.empty())
          return;
        if (!std::filesystem::path(s).is_absolute())
          nntr_cfg[key] = model_path + "/" + s;
      };
      resolve_path("tokenizer_file");
      resolve_path("embedding_bin_path");
      resolve_path("image_path");
      resolve_path("video_path");
    }

    if (nntr_cfg.contains("system_prompt")) {
      system_head_prompt =
        nntr_cfg["system_prompt"]["head_prompt"].get<std::string>();
      system_tail_prompt =
        nntr_cfg["system_prompt"]["tail_prompt"].get<std::string>();
    }

    // Construct weight file path
    const std::string weight_file =
      model_path + "/" + nntr_cfg["model_file_name"].get<std::string>();

    std::cout << weight_file << std::endl;

    // Initialize and run model
    std::string architecture;
    if (cfg.contains("architectures") && cfg["architectures"].is_array() &&
        !cfg["architectures"].empty()) {
      architecture = cfg["architectures"].get<std::vector<std::string>>()[0];
    } else if (cfg.contains("architecture") &&
               cfg["architecture"].is_string()) {
      architecture = cfg["architecture"].get<std::string>();
    } else if (cfg.contains("model_type") && cfg["model_type"].is_string()) {
      architecture = cfg["model_type"].get<std::string>();
    } else {
      throw std::invalid_argument(
        "config.json must contain 'architectures', 'architecture', or "
        "'model_type'.");
    }

    if (nntr_cfg.contains("model_type")) {
      std::string model_type = nntr_cfg["model_type"].get<std::string>();
      architecture = resolve_architecture(model_type, architecture);
    }

    // Load chat template from tokenizer_config.json or jinja (if available)
    std::optional<causallm::ChatTemplate> chat_template;
    if (causallm::ChatTemplate::Exists(model_path)) {
      chat_template.emplace(causallm::ChatTemplate::Load(model_path));
    }

    // Determine input text
    if (argc >= 3) {
      input_text = argv[2];
    } else {
      if (nntr_cfg.contains("chat_input")) {
        if (chat_template.has_value()) {
          input_text = chat_template->apply(nntr_cfg["chat_input"]);
          system_head_prompt.clear();
          system_tail_prompt.clear();
        } else {
          std::cerr << "[Warning] 'chat_input' is set but support for model "
                       "architecture '"
                    << architecture
                    << "' is not implemented. Falling back to 'sample_input'."
                    << std::endl;
          input_text = nntr_cfg["sample_input"].get<std::string>();
        }
      } else {
        input_text = nntr_cfg["sample_input"].get<std::string>();
      }
    }

    bool do_sample = generation_cfg.value("do_sample", false);

    if (architecture == "Lfm2VlForConditionalGeneration") {
      // LFM2-VL multimodal path
      causallm::Lfm2VlForConditionalGeneration vl_model(cfg, generation_cfg,
                                                        nntr_cfg);
      vl_model.initialize();
      vl_model.load_weight(model_path);

      if (!nntr_cfg.contains("image_path") ||
          nntr_cfg["image_path"].get<std::string>().empty()) {
        throw std::invalid_argument(
          "nntr_config.json must contain a non-empty 'image_path' key "
          "pointing to a valid image file (JPEG/PNG/BMP).");
      }
      const std::string image_path = nntr_cfg["image_path"].get<std::string>();
#ifdef PROFILE
      start_peak_tracker();
#endif
      vl_model.run(image_path, input_text, do_sample, true);
#ifdef PROFILE
      stop_and_print_peak();
#endif
    } else if (architecture == "Lfm2VLVJepa21BModel") {
      causallm::VjepaLfm2ForConditionalGeneration vl_model(
        cfg, generation_cfg, nntr_cfg);
      vl_model.initialize();
      vl_model.load_weight(model_path);

      if (!nntr_cfg.contains("video_path") ||
          !nntr_cfg["video_path"].is_string() ||
          nntr_cfg["video_path"].get<std::string>().empty()) {
        throw std::invalid_argument(
          "nntr_config.json must contain a non-empty 'video_path' key "
          "pointing to a raw float32 [C,T,H,W] video tensor file.");
      }

      const json vision_cfg =
        cfg.contains("vision_config") ? cfg["vision_config"] : json::object();
      const unsigned int num_frames =
        vision_cfg.value("num_frames", 16u);
      const unsigned int frame_height =
        vision_cfg.contains("image_height")
          ? vision_cfg["image_height"].get<unsigned int>()
          : (vision_cfg.contains("image_size")
               ? vision_cfg["image_size"].get<unsigned int>()
               : vision_cfg.value("img_size", 256u));
      const unsigned int frame_width =
        vision_cfg.contains("image_width")
          ? vision_cfg["image_width"].get<unsigned int>()
          : frame_height;
      const std::string video_path = nntr_cfg["video_path"].get<std::string>();

#ifdef PROFILE
      start_peak_tracker();
#endif
      vl_model.run_video_bin(video_path, static_cast<int>(num_frames),
                             static_cast<int>(frame_height),
                             static_cast<int>(frame_width), input_text,
                             do_sample, true);
#ifdef PROFILE
      stop_and_print_peak();
#endif
    } else if (architecture == "VJEPA2ViT") {
      // V-JEPA2 ViT encoder path — load video and run encoder
      causallm::VJEPA2ViT vjepa_model(cfg, generation_cfg, nntr_cfg);
      vjepa_model.initialize();
      vjepa_model.load_weight(weight_file);

      // Determine video source:
      //   1) "video_path" in nntr_config.json → directory of image frames
      //   2) "sample_input" in nntr_config.json → raw binary tensor file
      //      (backward-compatible with existing config files)
      std::string video_dir;
      std::string video_bin_path;

      if (nntr_cfg.contains("video_path") &&
          nntr_cfg["video_path"].is_string() &&
          !nntr_cfg["video_path"].get<std::string>().empty()) {
        video_dir = nntr_cfg["video_path"].get<std::string>();
        if (std::filesystem::is_directory(video_dir)) {
          // Directory of image frames → use loadAndPreprocessVideo
          std::cout << "Video source: image frames directory (" << video_dir
                    << ")" << std::endl;
        } else {
          // It's a file — determine type by extension
          std::string ext = std::filesystem::path(video_dir).extension().string();
          std::transform(ext.begin(), ext.end(), ext.begin(),
                         [](unsigned char c) { return std::tolower(c); });
          if (ext == ".bin" || ext == ".raw" || ext == ".dat" ||
              ext == ".fp32") {
            video_bin_path = video_dir;
            video_dir.clear();
            std::cout << "Video source: binary tensor file (" << video_bin_path
                      << ")" << std::endl;
          } else {
            // Video file (.mp4, .avi, .mkv, ...) — needs frame extraction
            std::cerr << "[Error] Video file '" << video_dir
                      << "' is a compressed video format (" << ext << ").\n"
                      << "  VJEPA2ViT requires either:\n"
                      << "    1) A directory of image frames (JPEG/PNG/BMP), "
                         "or\n"
                      << "    2) A raw float32 binary file [C,T,H,W].\n"
                      << "  Extract frames first, e.g.:\n"
                      << "    ffmpeg -i " << video_dir
                      << " -q:v 2 frames/%05d.jpg\n"
                      << "  Then set 'video_path' to the 'frames/' directory."
                      << std::endl;
            return EXIT_FAILURE;
          }
        }
      } else {
        // Fall back to sample_input (raw binary tensor path)
        video_bin_path = input_text;
        std::cout << "Video source: sample_input binary tensor (" << video_bin_path
                  << ")" << std::endl;
      }

#ifdef PROFILE
      start_peak_tracker();
#endif
      vjepa_model.run_with_video(video_dir, video_bin_path);
#ifdef PROFILE
      stop_and_print_peak();
#endif
    } else {
    auto model = causallm::Factory::Instance().create(architecture, cfg,
                                                      generation_cfg, nntr_cfg);
    if (!model) {
      std::cerr << "Unknown architecture: " << architecture << std::endl;
      std::cerr << "Registered architectures:";
      causallm::Factory::Instance().printRegistered(std::cerr);
      std::cerr << std::endl;
      return EXIT_FAILURE;
    }
    model->initialize();
    model->load_weight(weight_file);

#ifdef PROFILE
    start_peak_tracker();
#endif
#if defined(_WIN32)
    model->run(input_text.c_str(), do_sample, system_head_prompt.c_str(),
               system_tail_prompt.c_str());
#else
    if (architecture.find("Visual") != std::string::npos) {
      // Temp code for testing multimodal input
      int my_image_height = 1024;
      int my_image_width = 1024;
      int my_image_size = 5 * 512 * 512 * 3 * sizeof(uint16_t);
      void *my_image = malloc(my_image_size);
      causallm::multimodal_pointer image =
        std::make_pair(my_image, my_image_size);
      auto output =
        model->run_image(input_text, image, my_image_height, my_image_width,
                         do_sample, system_head_prompt, system_tail_prompt);
      free(my_image);
      std::cout << output.second; // To avoid unused variable warning
    } else {
      model->run(input_text, do_sample, system_head_prompt, system_tail_prompt);
    }
#endif
#ifdef PROFILE
    stop_and_print_peak();
#endif
    }
    auto finish_time = std::chrono::high_resolution_clock::now();
    auto e2e_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      finish_time - start_time);
    std::cout << "[e2e time]: " << e2e_duration.count() << " ms \n";
    printMemoryUsage();

  } catch (const std::exception &e) {
    std::cerr << "\n[!] FATAL ERROR: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
