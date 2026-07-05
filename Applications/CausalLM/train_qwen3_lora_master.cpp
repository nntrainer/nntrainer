// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   train_qwen3_lora_master.cpp
 * @date   01 Apr 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @author Sumon Nath <sumon.nath@samsung.com>
 * @author Niket Agarwal <niket.a@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Entry point for Qwen3-0.6B LoRA fine-tuning
 *
 * Usage:
 *   train_qwen3_lora_master <model_dir> <train_data.txt>
 *       [--lr <float>] [--epochs <int>]
 *       [--output <path>] [--lora_path <path>]
 *       [--max_samples <int>] [--skip_weights]
 *       [--lora_qat]   enable Q6_K fake-quant QAT (saves .q6k.bin)
 *       [--lora_q4]    enable Q4_0 fake-quant QAT (saves .q4.bin); rank defaults to 32
 */

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <sys/resource.h>

#include <causal_lm.h>
#include <dataset.h>
#include <lora_train.h>
#include <model.h>
#include <transformer.h>

#include "json.hpp"
#include "qwen3_causallm.h"

using json = nlohmann::json;

static size_t readVmRSS_KB() {
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("VmRSS:", 0) == 0) {
      size_t kb = 0;
      sscanf(line.c_str(), "VmRSS: %zu kB", &kb);
      return kb;
    }
  }
  struct rusage u;
  getrusage(RUSAGE_SELF, &u);
#ifdef __APPLE__
  return static_cast<size_t>(u.ru_maxrss) / 1024;
#else
  return static_cast<size_t>(u.ru_maxrss);
#endif
}

static void printMemRow(const char *label, size_t kb) {
  std::cout << "  [MEM] " << label << ": " << kb / 1024 << " MB  ("
            << kb << " KB)\n";
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <model_dir> <train_data.txt>"
                 " [--lr <float>] [--epochs <int>]"
                 " [--output <path>] [--lora_path <path>]"
                 " [--max_samples <int>] [--skip_weights]"
                 " [--lora_qat] [--lora_q4] [--seed <int>]\n";
    return 1;
  }

  std::string model_dir       = argv[1];
  std::string train_data_path = argv[2];
  float lr                    = 1e-4f;
  unsigned int epochs         = 1;
  std::string output_path     = "lora_weights.bin";
  std::string lora_path;
  int max_samples   = -1;
  bool skip_weights = false;
  unsigned int patience = 5;
  bool lora_qat = false;
  bool lora_q4  = false;  // Q4_0 LoRA: implies lora_qat, uses Q4_0 fake-quant range
  unsigned int seed = 42;

  for (int i = 3; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--lr" && i + 1 < argc)
      lr = std::atof(argv[++i]);
    else if (arg == "--epochs" && i + 1 < argc)
      epochs = static_cast<unsigned int>(std::atoi(argv[++i]));
    else if (arg == "--output" && i + 1 < argc)
      output_path = argv[++i];
    else if (arg == "--lora_path" && i + 1 < argc)
      lora_path = argv[++i];
    else if (arg == "--max_samples" && i + 1 < argc)
      max_samples = std::atoi(argv[++i]);
    else if (arg == "--skip_weights")
      skip_weights = true;
    else if (arg == "--patience" && i + 1 < argc)
      patience = static_cast<unsigned int>(std::atoi(argv[++i]));
    else if (arg == "--lora_qat")
      lora_qat = true;
    else if (arg == "--lora_q4")
      lora_q4 = true;  // save as Q4_0 PTQ at each checkpoint; no QAT needed
    else if (arg == "--seed" && i + 1 < argc)
      seed = static_cast<unsigned int>(std::atoi(argv[++i]));
  }

  try {
    std::string config_path    = model_dir + "/config.json";
    std::string gen_config_path = model_dir + "/generation_config.json";
    std::string nntr_config_path = model_dir + "/nntr_config.json";

    auto cfg      = causallm::LoadJsonFile(config_path);
    auto gen_cfg  = causallm::LoadJsonFile(gen_config_path);
    auto nntr_cfg = causallm::LoadJsonFile(nntr_config_path);

    size_t mem_baseline = readVmRSS_KB();

    std::cout << "=== Qwen3 LoRA Training ===\n";
    std::cout << "Model dir : " << model_dir << "\n";
    std::cout << "Train data: " << train_data_path << "\n";
    std::cout << "LR=" << lr << "  epochs=" << epochs
              << "  patience=" << patience
              << "  lora_qat=" << (lora_qat ? "true" : "false")
              << "  lora_q4=" << (lora_q4 ? "true" : "false") << "\n\n";

    // Inject LoRA config into nntr_cfg (override JSON in memory)
    if (!nntr_cfg.contains("lora_rank") || nntr_cfg["lora_rank"] == 0) {
      std::cout << "[LoRA] Injecting default LoRA config (rank=8, alpha=16).\n";
      nntr_cfg["lora_rank"]  = 8;
      nntr_cfg["lora_alpha"] = 16;
      nntr_cfg["lora_target"] =
        json::array({"wq", "wk", "wv", "wo", "ffn_up", "ffn_down", "ffn_gate"});
    }
    // Q4_0 requires rank % 32 == 0; force rank=32 regardless of config.
    if (lora_q4) {
      nntr_cfg["lora_rank"]  = 32;
      nntr_cfg["lora_alpha"] = 64;
      if (!nntr_cfg.contains("lora_target") || nntr_cfg["lora_target"].empty())
        nntr_cfg["lora_target"] =
          json::array({"wq", "wk", "wv", "wo", "ffn_up", "ffn_down", "ffn_gate"});
      std::cout << "[LoRA] Q4_0 mode: forcing rank=32, alpha=64.\n";
    }
    if (lora_qat)
      nntr_cfg["lora_qat"] = true;
    // lora_weight_q4 is inference-only; do not inject during training.
    std::cout << "[LoRA] rank=" << nntr_cfg["lora_rank"]
              << "  alpha=" << nntr_cfg["lora_alpha"]
              << "  targets=" << nntr_cfg["lora_target"].dump() << "\n\n";

    auto model = std::make_unique<causallm::Qwen3CausalLM>(cfg, gen_cfg, nntr_cfg);
    model->initializeForTraining(lr, epochs);
    size_t mem_after_init = readVmRSS_KB();

    if (!skip_weights && nntr_cfg.contains("model_file_name")) {
      std::string weight_path =
        model_dir + "/" + nntr_cfg["model_file_name"].get<std::string>();
      std::cout << "Loading weights: " << weight_path << "\n";
      if (!lora_path.empty()) {
        std::cout << "Loading LoRA overlay: " << lora_path << "\n";
        model->load_weight_lora(weight_path, lora_path);
      } else {
        model->load_weight(weight_path);
      }
    } else {
      std::cout << "Skipping weight load (random init).\n";
    }
    size_t mem_after_weights = readVmRSS_KB();

    // Build tokenizer
    std::string tokenizer_path = model_dir + "/tokenizer.json";
    if (nntr_cfg.contains("tokenizer_file"))
      tokenizer_path = nntr_cfg["tokenizer_file"].get<std::string>();
    auto blob      = causallm::LoadBytesFromFile(tokenizer_path);
    auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(blob);

    unsigned int seq_len   = nntr_cfg["init_seq_len"].get<unsigned int>();
    unsigned int vocab_size = cfg["vocab_size"].get<unsigned int>();

    causallm::TrainingDataGenerator data_gen(tokenizer.get(), seq_len, vocab_size, seed);
    data_gen.loadTextFile(train_data_path);

    if (max_samples > 0 &&
        static_cast<unsigned int>(max_samples) < data_gen.getNumSamples()) {
      std::cout << "Limiting to " << max_samples << " samples.\n";
      data_gen.limitSamples(static_cast<unsigned int>(max_samples));
    }
    if (data_gen.getNumSamples() == 0) {
      std::cerr << "Error: no training samples loaded.\n";
      return 1;
    }
    std::cout << "Training samples: " << data_gen.getNumSamples() << "\n\n";

    auto dataset = std::shared_ptr<ml::train::Dataset>(
      ml::train::createDataset(ml::train::DatasetType::GENERATOR,
                               causallm::TrainingDataGenerator::dataCb,
                               &data_gen));
    model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset);
    // Same data used for accuracy tracking after each epoch
    model->setDataset(ml::train::DatasetModeType::MODE_VALID, dataset);

    // Epoch callback: cumulative loss + perplexity + early stopping.
    // Accuracy is always ~0% for next-token top-1 with vocab_size=151936;
    // perplexity (exp(loss)) is the meaningful language-model metric.
    struct CumStats {
      causallm::Qwen3CausalLM *mdl;
      unsigned int epoch_count  = 0;
      float cumulative_loss     = 0.0f;
      // early stopping
      unsigned int patience;
      unsigned int patience_left;
      float best_val_loss       = std::numeric_limits<float>::max();
      unsigned int best_epoch   = 0;
      bool stop_flag            = false;
      std::string output_path;
      bool lora_qat             = false;
      bool lora_q4              = false;
      // memory tracking
      size_t peak_train_mem_kb  = 0;
      size_t mem_epoch1_kb      = 0;   // snapshot after first epoch ends
    };
    CumStats cum{model.get()};
    cum.patience      = patience;
    cum.patience_left = patience;
    cum.output_path   = output_path;
    cum.lora_qat      = lora_qat;
    cum.lora_q4       = lora_q4;

    auto epoch_cb = [](void *ud) {
      auto *c = static_cast<CumStats *>(ud);
      c->epoch_count++;
      size_t cur_mem = readVmRSS_KB();
      if (cur_mem > c->peak_train_mem_kb) c->peak_train_mem_kb = cur_mem;
      if (c->epoch_count == 1) c->mem_epoch1_kb = cur_mem;
      auto ts = c->mdl->getTrainingStats();
      auto vs = c->mdl->getValidStats();
      c->cumulative_loss += ts.loss;
      float avg     = c->cumulative_loss / static_cast<float>(c->epoch_count);
      float ppl     = std::exp(ts.loss);
      float cum_ppl = std::exp(avg);
      std::cout << "  Cumulative | AvgLoss: " << avg
                << "  CumPPL: " << cum_ppl
                << "  EpochPPL: " << ppl
                << "  Mem: " << cur_mem / 1024 << " MB\n";
      if (c->lora_qat)
        c->mdl->printLoRAQATStats();

      // Early stopping: track best validation loss
      if (vs.loss < c->best_val_loss) {
        c->best_val_loss   = vs.loss;
        c->best_epoch      = c->epoch_count;
        c->patience_left   = c->patience;
        // Always save FP32 weights; also save Q4_0 (PTQ) when --lora_q4.
        c->mdl->save_weight_lora(c->output_path);
        if (c->lora_q4) {
          std::string q4_path = c->output_path;
          auto dot = q4_path.rfind('.');
          if (dot != std::string::npos)
            q4_path.insert(dot, ".q4");
          else
            q4_path += ".q4.bin";
          c->mdl->save_weight_lora_q4(q4_path);
        }
        std::cout << "  [Best] val_loss=" << vs.loss
                  << " at epoch " << c->epoch_count
                  << " -> checkpoint saved\n";
      } else {
        c->patience_left--;
        std::cout << "  [EarlyStopping] No improvement. patience="
                  << c->patience_left << "/" << c->patience << "\n";
        if (c->patience_left == 0)
          c->stop_flag = true;
      }
    };

    auto stop_cb = [](void *ud) -> bool {
      return static_cast<CumStats *>(ud)->stop_flag;
    };

    size_t mem_pre_train = readVmRSS_KB();
    std::cout << "\n=== Starting training ===\n";
    auto t0 = std::chrono::steady_clock::now();
    model->train(epoch_cb, &cum, stop_cb, &cum);
    double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    size_t mem_after_training = readVmRSS_KB();

    std::cout << "\nTraining done in " << elapsed << " s.\n";
    std::cout << "Best checkpoint: epoch " << cum.best_epoch
              << "  val_loss=" << cum.best_val_loss
              << "  val_PPL=" << std::exp(cum.best_val_loss) << "\n";
    std::cout << "LoRA weights saved to: " << output_path << "\n";

    // --- Analytical LoRA weight size ---
    // loraA(rank×K) + loraB(N×rank) in FP32 per layer per target
    unsigned int lora_rank  = nntr_cfg["lora_rank"].get<unsigned int>();
    unsigned int hidden     = cfg.value("hidden_size", 1024u);
    unsigned int inter      = cfg.value("intermediate_size", 2816u);
    unsigned int num_heads  = cfg.value("num_attention_heads", 16u);
    unsigned int kv_heads   = cfg.value("num_key_value_heads", num_heads);
    unsigned int num_layers = cfg.value("num_hidden_layers", 28u);
    unsigned int head_dim   = hidden / num_heads;
    unsigned int kv_dim     = kv_heads * head_dim;

    // Elements per layer across all 7 LoRA targets
    size_t elems_per_layer =
      2 * lora_rank * hidden  +            // wq: A+B
      lora_rank * hidden + kv_dim * lora_rank +  // wk
      lora_rank * hidden + kv_dim * lora_rank +  // wv
      2 * lora_rank * hidden  +            // wo
      lora_rank * hidden + inter * lora_rank +   // ffn_up
      lora_rank * hidden + inter * lora_rank +   // ffn_gate
      inter * lora_rank + hidden * lora_rank;    // ffn_down (K=inter, N=hidden)
    size_t lora_weight_kb = elems_per_layer * num_layers * sizeof(float) / 1024;
    // QAT: a_fq + b_fq tensors = same size as LoRA weights
    size_t qat_fq_kb      = lora_qat ? lora_weight_kb : 0;
    // Gradients for loraA+loraB (same size as weights)
    size_t lora_grad_kb   = lora_weight_kb;
    // Adam optimizer: 2 moment vectors per param
    size_t lora_optim_kb  = 2 * lora_weight_kb;

    auto safeDelta = [](size_t a, size_t b) -> size_t {
      return (a > b) ? (a - b) : 0;
    };

    size_t base_weights_kb  = safeDelta(mem_after_weights, mem_after_init);
    size_t train_peak_delta = safeDelta(cum.peak_train_mem_kb, mem_pre_train);

    std::cout << "\n=== Memory Usage Summary ===\n";
    std::cout << "--- Snapshots ---\n";
    printMemRow("Process baseline        ", mem_baseline);
    printMemRow("After model graph init  ", mem_after_init);
    printMemRow("After base weights load ", mem_after_weights);
    printMemRow("Pre-train (ready)       ", mem_pre_train);
    if (cum.mem_epoch1_kb)
      printMemRow("After epoch 1           ", cum.mem_epoch1_kb);
    printMemRow("Peak during training    ", cum.peak_train_mem_kb);
    printMemRow("After training done     ", mem_after_training);

    std::cout << "--- Deltas ---\n";
    std::cout << "  [MEM] Model graph alloc      : "
              << safeDelta(mem_after_init, mem_baseline) / 1024 << " MB"
              << "  (init - baseline)\n";
    std::cout << "  [MEM] Base model weights (Q4): "
              << base_weights_kb / 1024 << " MB"
              << "  (post-load - post-init)\n";
    std::cout << "  [MEM] Forward+Backward peak  : "
              << train_peak_delta / 1024 << " MB"
              << "  (peak - pre-train; activations+grads+optimizer)\n";

    std::cout << "--- Analytical LoRA breakdown (rank=" << lora_rank
              << ", layers=" << num_layers << ") ---\n";
    std::cout << "  [MEM] LoRA weights (FP32)    : "
              << lora_weight_kb / 1024 << " MB"
              << "  (loraA+loraB all layers)\n";
    std::cout << "  [MEM] LoRA gradients         : "
              << lora_grad_kb / 1024 << " MB"
              << "  (dL/dA + dL/dB)\n";
    std::cout << "  [MEM] Adam optimizer states  : "
              << lora_optim_kb / 1024 << " MB"
              << "  (m + v per param)\n";
    if (lora_qat)
      std::cout << "  [MEM] QAT fq tensors (a_fq+b_fq): "
                << qat_fq_kb / 1024 << " MB"
                << "  (FP32 fake-quant copies)\n";
    std::cout << "  [MEM] LoRA total (weights+grad+optim"
              << (lora_qat ? "+fq" : "") << "): "
              << (lora_weight_kb + lora_grad_kb + lora_optim_kb + qat_fq_kb) / 1024
              << " MB\n";

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
