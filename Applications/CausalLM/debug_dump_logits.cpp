// SPDX-License-Identifier: Apache-2.0
// Temporary diagnostic tool: dump raw prefill logits + top-k for a given
// token id sequence, to compare against an independent HF forward pass.
// Usage: debug_dump_logits <model_dir> <comma-separated-token-ids>

#include <algorithm>
#include <causal_lm.h>
#include <cmath>
#include <cstdlib>
#include <causallm_test_utils.h>
#include <fstream>
#include <iostream>
#include <lfm2_moe_causallm.h>
#include <sstream>
#include <transformer.h>
#include <vector>

using json = nlohmann::json;

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <model_dir> <ids csv>\n";
    return 1;
  }
  std::string model_path = argv[1];
  std::vector<unsigned int> ids;
  {
    std::stringstream ss(argv[2]);
    std::string tok;
    while (std::getline(ss, tok, ','))
      ids.push_back(static_cast<unsigned int>(std::stoul(tok)));
  }

  json cfg = causallm::LoadJsonFile(model_path + "/config.json");
  json generation_cfg =
    causallm::LoadJsonFile(model_path + "/generation_config.json");
  json nntr_cfg = causallm::LoadJsonFile(model_path + "/nntr_config.json");

  auto resolve = [&](const char *key) {
    if (nntr_cfg.contains(key) && !nntr_cfg[key].is_null())
      nntr_cfg[key] =
        (std::filesystem::path(model_path) / nntr_cfg[key].get<std::string>())
          .string();
  };
  resolve("tokenizer_file");
  resolve("embedding_bin_path");

  causallm_test::CausalLMTestAdapter<causallm::Lfm2MoeCausalLM> model(
    cfg, generation_cfg, nntr_cfg);
  model.initializeModel();

  std::string weight_path =
    model_path + "/" + nntr_cfg["model_file_name"].get<std::string>();
  model.loadWeight(weight_path);

  auto logits = model.prefillLogitsFromIds(ids);
  std::cout << "prefillLogitsFromIds returned, size=" << logits.size()
            << "\n";
  std::cout.flush();

  // Scan every layer's output for the first NaN/Inf appearance.
  if (!std::getenv("SKIP_LAYER_SCAN"))
  model.forEachLayer([](ml::train::Layer &layer,
                        nntrainer::RunLayerContext &context, void *) {
    try {
    for (unsigned int o = 0; o < context.getNumOutputs(); ++o) {
      auto &out = context.getOutput(o);
      if (!out.isAllocated() || out.empty())
        continue;
      if (out.getDataType() != ml::train::TensorDim::DataType::FP32)
        continue;
      const float *data = out.getData<float>();
      if (data == nullptr)
        continue;
      size_t n = out.size();
      size_t nan_count = 0, inf_count = 0;
      float vmin = 1e30f, vmax = -1e30f;
      for (size_t i = 0; i < n; ++i) {
        float v = data[i];
        if (std::isnan(v))
          ++nan_count;
        else if (std::isinf(v))
          ++inf_count;
        else {
          vmin = std::min(vmin, v);
          vmax = std::max(vmax, v);
        }
      }
      if (nan_count || inf_count) {
        std::cout << "LAYER " << layer.getName() << " (" << layer.getType()
                  << ") output[" << o << "] n=" << n
                  << " nan=" << nan_count << " inf=" << inf_count
                  << " min=" << vmin << " max=" << vmax << "\n";
        std::cout.flush();
      } else {
        std::cout << "ok    " << layer.getName() << " (" << layer.getType()
                  << ") n=" << n << " min=" << vmin << " max=" << vmax
                  << "\n";
        std::cout.flush();
      }
    }
    } catch (const std::exception &e) {
      std::cout << "EXC layer=" << layer.getName() << " what=" << e.what()
                << "\n";
      std::cout.flush();
    } catch (...) {
      std::cout << "EXC(unknown) layer=" << layer.getName() << "\n";
      std::cout.flush();
    }
  });

  std::vector<int> idx(logits.size());
  for (size_t i = 0; i < idx.size(); ++i)
    idx[i] = static_cast<int>(i);
  std::partial_sort(idx.begin(), idx.begin() + 10, idx.end(),
                    [&](int a, int b) { return logits[a] > logits[b]; });
  std::cout << "top10:\n";
  for (int i = 0; i < 10; ++i)
    std::cout << "  id=" << idx[i] << " logit=" << logits[idx[i]] << "\n";

  std::ofstream out("nntr_logits.bin", std::ios::binary);
  out.write(reinterpret_cast<char *>(logits.data()),
           static_cast<std::streamsize>(logits.size() * sizeof(float)));
  std::cout << "wrote nntr_logits.bin (" << logits.size() << " floats)\n";
  return 0;
}
