// SPDX-License-Identifier: Apache-2.0
/** @file htp_continuation_device_check.cpp
 * @brief Compare resident continuation against single-row feeding on a device.
 * Run each mode in a fresh process with the same pack, then compare the emitted
 * FP32 files at positions 8, 9, 12, 13, 15 and 16 byte for byte.
 * @author NNTrainer contributors
 * @bug Requires an HTP model pack and device; this is not a host arithmetic
 * test.
 */
#ifndef ENABLE_HEXKL
#error "The continuation device check requires an HTP-enabled engine"
#endif
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <gemma4_htp_moe_causallm.h>
#include <htp_decode_hook.h>
#include <htp_model_config.h>
#include <iostream>
#include <lfm2_htp_moe_causallm.h>
#include <lfm2_runtime_defaults.h>
#include <neuralnet.h>

using causallm::json;

template <typename Base> class ContinuationModel : public Base {
public:
  ContinuationModel(json &cfg, json &gen, json &nntr) :
    causallm::Transformer(causallm::normalizeHtpModelConfig(cfg),
                          causallm::normalizeHtpGenerationConfig(gen, cfg),
                          nntr, causallm::ModelType::CAUSALLM),
    Base(cfg, gen, nntr) {}

  void step(unsigned from, unsigned count, const std::string &output) {
    this->allocateAndBindKVCache();
    std::vector<float> ids(this->MAX_SEQ_LEN, 0.0f);
    for (unsigned i = 0; i < count; ++i)
      ids[i] = float(32 + (from + i) % 64);
    auto *nn = static_cast<nntrainer::NeuralNetwork *>(this->model.get());
    std::vector<float *> inputs(nn->getInputDimension().size(), ids.data());
    this->beforePrefillBlock(from, from + count);
    causallm::htpDecodeWantLogits(true);
    auto logits =
      this->incrementalInference(1, inputs, count, from, from + count);
    if (logits.size() != 1)
      throw std::runtime_error("expected one vocabulary output");
    std::ofstream file(output, std::ios::binary);
    file.write(reinterpret_cast<const char *>(logits[0]),
               this->NUM_VOCAB * sizeof(float));
    if (!file)
      throw std::runtime_error("could not write logits");
    const auto id =
      std::max_element(logits[0], logits[0] + this->NUM_VOCAB) - logits[0];
    std::cout << "CONTINUATION position=" << from + count << " argmax=" << id
              << '\n';
    this->releaseLogitsBuf(logits[0]);
  }
};

template <typename Base>
void run(json &cfg, json &gen, json &nntr, bool chunked,
         const std::string &prefix) {
  ContinuationModel<Base> model(cfg, gen, nntr);
  model.initialize();
  model.load_weight(nntr.at("model_file_name").get<std::string>());
  model.repack_weight();
  model.step(0, 8, prefix + "-8.f32");
  unsigned pos = 8;
  for (unsigned chunk : {1u, 3u, 1u, 2u, 1u}) {
    const unsigned end = pos + chunk;
    do {
      const unsigned count = chunked ? end - pos : 1u;
      model.step(pos, count,
                 prefix + "-" + std::to_string(pos + count) + ".f32");
      pos += count;
    } while (pos < end);
  }
}

int main(int argc, char **argv) {
  if (argc != 5 ||
      (std::string(argv[2]) != "gemma" && std::string(argv[2]) != "lfm") ||
      (std::string(argv[3]) != "rows" && std::string(argv[3]) != "chunks")) {
    std::cerr << "usage: " << argv[0]
              << " PACK gemma|lfm rows|chunks OUTPUT_PREFIX\n";
    return 2;
  }
  try {
    const std::filesystem::path pack(argv[1]);
    auto load = [&](const char *name) {
      std::ifstream file(pack / name);
      return json::parse(file);
    };
    json cfg = load("config.json"), gen = load("generation_config.json"),
         nntr = load("nntr_config.json");
    causallm::configureLfm2HtpDefaults(
      cfg.value("model_type", std::string()),
      nntr.value("lfm2_weight_format", std::string()),
      nntr.value("lfm2_execution_profile", std::string()));
    for (auto key : {"tokenizer_file", "model_file_name", "embedding_file_name",
                     "kv_quant_file", "static_activation_file"}) {
      if (nntr.contains(key) && nntr[key].is_string()) {
        std::filesystem::path path(nntr[key].get<std::string>());
        if (!path.empty() && path.is_relative())
          nntr[key] = (pack / path).string();
      }
    }
    nntr["init_seq_len"] = 64;
    nntr["max_seq_len"] = 128;
    nntr["num_to_generate"] = 16;
    nntr["skip_prefill"] = false;
    const bool chunks = std::string(argv[3]) == "chunks";
    if (std::string(argv[2]) == "gemma")
      run<causallm::Gemma4HtpMoECausalLM>(cfg, gen, nntr, chunks, argv[4]);
    else
      run<causallm::Lfm2HtpMoECausalLM>(cfg, gen, nntr, chunks, argv[4]);
  } catch (const std::exception &e) {
    std::cerr << "CONTINUATION FAILED: " << e.what() << '\n';
    return 1;
  }
}
