// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   ced_transformer.cpp
 * @date   24 August 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This ced_transformer.cpp constructs a class for the CED audio
 * tagging model (https://huggingface.co/mispeech/ced-tiny).
 */

#include "ced_transformer.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <factory.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace causallm {

/**
 * @brief Map the HuggingFace CED config onto the generic ViT parameters.
 *
 * Every value below comes from the upstream config.json of the CED checkpoint
 * (https://huggingface.co/mispeech/ced-tiny/blob/main/config.json); the two
 * epsilons are the PyTorch defaults implied by modeling_ced.py rather than
 * config entries, and are annotated as such.
 */
void CedTransformer::setupParameters(json &cfg, json &generation_cfg,
                                     json &nntr_cfg) {
  const unsigned int embed_dim = cfg.value("embed_dim", 192u);
  const unsigned int num_heads = cfg.value("num_heads", 3u);
  const float mlp_ratio = cfg.value("mlp_ratio", 4.0f);
  const unsigned int patch_size = cfg.value("patch_size", 16u);
  const std::string pooling = cfg.value("pooling", std::string("mean"));

  if (pooling != "mean" && pooling != "logit" && pooling != "xi") {
    throw std::invalid_argument(
      "CED pooling mode '" + pooling +
      "' is not supported yet (only 'mean', 'logit' and 'xi')");
  }

  json vit_cfg = cfg;

  // -- encoder trunk -----------------------------------------------------
  vit_cfg["hidden_size"] = embed_dim;
  vit_cfg["num_hidden_layers"] = cfg.value("depth", 12u);
  vit_cfg["num_attention_heads"] = num_heads;
  vit_cfg["num_key_value_heads"] = num_heads;
  vit_cfg["head_dim"] = embed_dim / num_heads;
  vit_cfg["intermediate_size"] =
    static_cast<unsigned int>(embed_dim * mlp_ratio);
  // modeling_ced.py: norm_layer = partial(nn.LayerNorm, eps=1e-6)
  vit_cfg["norm_eps"] = 1e-6;
  vit_cfg["is_causal"] = false;
  vit_cfg["rope_theta"] = 0;
  vit_cfg["tie_word_embeddings"] = false;
  vit_cfg["vocab_size"] = 1;

  // -- input geometry ----------------------------------------------------
  vit_cfg["input_height"] = cfg.value("n_mels", 64u);
  vit_cfg["input_width"] = cfg.value("target_length", 1012u);
  vit_cfg["in_chans"] = 1;
  vit_cfg["patch_size"] = patch_size;
  vit_cfg["patch_stride"] = cfg.value("patch_stride", patch_size);
  vit_cfg.erase("num_patches"); // derive from the grid, never guess

  // -- init_bn -----------------------------------------------------------
  vit_cfg["use_input_norm"] = true;
  // nn.BatchNorm2d default eps; modeling_ced.py only overrides momentum.
  vit_cfg["input_norm_eps"] = 1e-5;

  // -- classification head ----------------------------------------------
  // outputlayer = Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim,
  // outputdim)); forward_head applies sigmoid for every pooling mode except
  // "logit".
  vit_cfg["num_classes"] = cfg.value("outputdim", 527u);
  vit_cfg["pooling"] = pooling;
  // forward_head() sigmoids only the plain-average pooling modes; "logit" and
  // the xi-vector posterior pooling return raw logits and leave the squashing
  // to the caller, so the graph does too and run() applies it.
  vit_cfg["head_sigmoid"] = (pooling == "mean");
  vit_cfg["head_norm_eps"] = 1e-5; // plain nn.LayerNorm default

  TimmViTTransformer::setupParameters(vit_cfg, generation_cfg, nntr_cfg);

  // Class names, for a readable top-k report.
  LABELS.clear();
  if (cfg.contains("id2label") && cfg["id2label"].is_object()) {
    LABELS.assign(NUM_CLASSES, std::string());
    for (auto it = cfg["id2label"].begin(); it != cfg["id2label"].end(); ++it) {
      const unsigned long idx = std::strtoul(it.key().c_str(), nullptr, 10);
      if (idx < LABELS.size() && it.value().is_string()) {
        LABELS[idx] = it.value().get<std::string>();
      }
    }
  }

  // Optional wav front-end. Without it the runner only accepts a raw mel
  // spectrogram, which is the boundary the upstream HuggingFace CED uses.
  HAS_FRONT_END = cfg.contains("front_end") && cfg["front_end"].is_object();
  if (HAS_FRONT_END) {
    const json &fe = cfg["front_end"];
    FRONT_END.sample_rate = fe.value("sample_rate", 16000u);
    FRONT_END.n_fft = fe.value("n_fft", 512u);
    FRONT_END.win_size = fe.value("win_size", FRONT_END.n_fft);
    FRONT_END.hop_size = fe.value("hop_size", 160u);
    FRONT_END.center = fe.value("center", true);
    FRONT_END.f_min = fe.value("f_min", 0.0f);
    FRONT_END.f_max = fe.value("f_max", FRONT_END.sample_rate / 2.0f);
    FRONT_END.n_mels = INPUT_HEIGHT;
    FRONT_END.top_db = fe.value("top_db", 80.0f);
    FRONT_END.amin = fe.value("amin", 1e-10f);
    FRONT_END.log_eps = fe.value("log_eps", 1e-5f);
    // "db" is torchaudio's AmplitudeToDB; "natural" is ln(max(x, log_eps)),
    // which is what a graph with the mel baked in may have been trained on.
    const std::string log_mode = fe.value("log_mode", std::string("db"));
    if (log_mode == "natural")
      FRONT_END.log_mode = audio::FrontEndConfig::LogMode::NATURAL;
    else if (log_mode == "db")
      FRONT_END.log_mode = audio::FrontEndConfig::LogMode::DB;
    else
      throw std::invalid_argument("Unknown front_end log_mode: " + log_mode);
    FRONT_END.power = fe.value("power", 2.0f);
    WINDOW_SAMPLES = fe.value("window_samples", FRONT_END.sample_rate);
    STRIDE_SAMPLES = fe.value("stride_samples", WINDOW_SAMPLES);
    NORMALIZE_DIVISOR = fe.value("normalize_divisor", 32768.0f);
  }
  TOP_K = cfg.value("top_k", 3u);

  // Per-class detection thresholds, keyed by label so the config stays
  // readable; classes without an entry can never fire.
  THRESHOLDS.assign(NUM_CLASSES, 2.0f);
  if (cfg.contains("thresholds") && cfg["thresholds"].is_object()) {
    for (auto it = cfg["thresholds"].begin(); it != cfg["thresholds"].end();
         ++it) {
      const auto pos = std::find(LABELS.begin(), LABELS.end(), it.key());
      if (pos != LABELS.end())
        THRESHOLDS[pos - LABELS.begin()] = it.value().get<float>();
    }
  }

  std::cout << "[CED] input " << IMG_CHANNELS << "x" << INPUT_HEIGHT << "x"
            << INPUT_WIDTH << ", patch " << PATCH_SIZE << "/" << PATCH_STRIDE
            << " -> grid " << GRID_H << "x" << GRID_W << " = " << NUM_PATCHES
            << " tokens, dim " << DIM << ", layers " << NUM_LAYERS << ", heads "
            << NUM_HEADS << ", classes " << NUM_CLASSES
            << (HEAD_SIGMOID ? " (sigmoid)" : " (logit)") << std::endl;
}

/**
 * @brief Read a raw FP32 mel spectrogram of exactly `count` values.
 */
static std::vector<float> loadMelSpectrogram(const std::string &path,
                                             size_t count) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    throw std::runtime_error("Failed to open mel spectrogram: " + path);
  }
  const size_t got = static_cast<size_t>(f.tellg()) / sizeof(float);
  if (got != count) {
    throw std::runtime_error(
      "Mel spectrogram " + path + " holds " + std::to_string(got) +
      " floats, expected " + std::to_string(count) +
      " (n_mels * target_length). The file must be FP32 [n_mels, frames], "
      "frame-major rows, i.e. the HuggingFace `input_values` tensor.");
  }
  f.seekg(0);
  std::vector<float> data(count);
  f.read(reinterpret_cast<char *>(data.data()),
         static_cast<std::streamsize>(count * sizeof(float)));
  return data;
}

/**
 * @brief Run CED inference on a mel-spectrogram file.
 */
void CedTransformer::run(const WSTR prompt, bool do_sample,
                         const WSTR system_prompt, const WSTR tail_prompt,
                         bool log_output) {
  (void)do_sample;
  (void)system_prompt;
  (void)tail_prompt;
  (void)log_output;

  if (!is_initialized) {
    throw std::runtime_error("CED model is not initialized. Please call "
                             "initialize() before run().");
  }

  const std::string in_path(prompt);
  if (HAS_FRONT_END && in_path.size() > 4 &&
      in_path.compare(in_path.size() - 4, 4, ".wav") == 0) {
    runAudioFile(in_path);
    return;
  }

  const std::string mel_path(in_path);
  const size_t mel_count =
    static_cast<size_t>(INPUT_HEIGHT) * static_cast<size_t>(INPUT_WIDTH);
  std::vector<float> mel = loadMelSpectrogram(mel_path, mel_count);

  std::vector<float *> input{mel.data()};
  std::vector<float *> label;

  std::vector<float *> output = model->incremental_inference(
    BATCH_SIZE, input, label, NUM_PATCHES, 0, NUM_PATCHES, false);

  const size_t out_count = NUM_CLASSES;

  // Top-5 report.
  std::vector<size_t> order(out_count);
  std::iota(order.begin(), order.end(), 0);
  const size_t topk = std::min<size_t>(5, out_count);
  std::partial_sort(
    order.begin(), order.begin() + topk, order.end(),
    [&](size_t a, size_t b) { return output[0][a] > output[0][b]; });
  std::cout << std::setprecision(6) << std::fixed;
  for (size_t i = 0; i < topk; ++i) {
    const size_t idx = order[i];
    std::cout << "[top" << (i + 1) << "] " << idx << " "
              << (idx < LABELS.size() ? LABELS[idx] : std::string("?")) << " = "
              << output[0][idx] << std::endl;
  }
  std::cout.unsetf(std::ios::floatfield);

  if (const char *dump_path = std::getenv("CED_OUT_BIN")) {
    std::ofstream dump(dump_path, std::ios::binary);
    if (!dump) {
      std::cerr << "[CED_OUT_BIN] cannot open " << dump_path << std::endl;
    } else {
      dump.write(reinterpret_cast<const char *>(output[0]),
                 static_cast<std::streamsize>(out_count * sizeof(float)));
      std::cout << "[CED_OUT_BIN] wrote " << out_count << " floats to "
                << dump_path << std::endl;
    }
  }

  if (const char *ref_path = std::getenv("CED_REF_BIN")) {
    std::ifstream ref(ref_path, std::ios::binary | std::ios::ate);
    if (!ref) {
      std::cerr << "[CED_REF_BIN] cannot open " << ref_path << std::endl;
      return;
    }
    const size_t ref_count = static_cast<size_t>(ref.tellg()) / sizeof(float);
    if (ref_count != out_count) {
      std::cerr << "[CED_REF_BIN] size mismatch: reference has " << ref_count
                << " floats, output has " << out_count << std::endl;
      return;
    }
    ref.seekg(0);
    std::vector<float> ref_data(ref_count);
    ref.read(reinterpret_cast<char *>(ref_data.data()),
             static_cast<std::streamsize>(ref_count * sizeof(float)));

    double max_abs_diff = 0.0;
    double sum_sq_diff = 0.0;
    size_t max_idx = 0;
    size_t nan_count = 0;
    for (size_t i = 0; i < out_count; ++i) {
      const float got = output[0][i];
      // Bit-pattern NaN test: -ffast-math folds std::isnan() to false.
      uint32_t bits;
      std::memcpy(&bits, &got, sizeof(bits));
      if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u) {
        ++nan_count;
        continue;
      }
      const double diff = std::fabs(static_cast<double>(got) - ref_data[i]);
      sum_sq_diff += diff * diff;
      if (diff > max_abs_diff) {
        max_abs_diff = diff;
        max_idx = i;
      }
    }
    std::cout << "[CED_REF_BIN] elements=" << out_count << " nan=" << nan_count
              << " max_abs_diff=" << max_abs_diff << " (at class " << max_idx
              << ") rms_diff=" << std::sqrt(sum_sq_diff / out_count)
              << std::endl;
    // Default tolerance is the FP32-parity level; a quantized run should pass
    // its own budget via CED_REF_TOL (e.g. 0.05 for W8A8 sigmoid outputs).
    double tol = 1e-4;
    if (const char *tol_env = std::getenv("CED_REF_TOL")) {
      tol = std::strtod(tol_env, nullptr);
    }
    std::cout << "[CED_REF_BIN] tol=" << tol << " "
              << ((nan_count == 0 && max_abs_diff < tol) ? "PASS" : "FAIL")
              << std::endl;
  }
}

/**
 * @brief Run a batch of mel windows through the graph.
 *
 * @param mels  windows laid out contiguously, [windows][n_mels * frames]
 * @param count number of windows in the buffer
 * @return raw head output, [count][NUM_CLASSES]
 *
 * All windows go through in a single call on purpose. incremental_inference
 * re-runs allocateTensors() whenever `from` is 0, so calling it once per window
 * re-plans the tensor pool on every window and aborts on the second one; one
 * batched call plans once. Memory therefore scales with the clip length, which
 * is fine for the clip-at-a-time usage this runner targets.
 */
std::vector<float> CedTransformer::inferWindows(std::vector<float> &mels,
                                                unsigned int count) {
  std::vector<float> all(static_cast<size_t>(count) * NUM_CLASSES);
  const size_t mel_len =
    static_cast<size_t>(INPUT_HEIGHT) * static_cast<size_t>(INPUT_WIDTH);

  for (unsigned int i = 0; i < count; ++i) {
    // Each window is an independent clip, so the attention caches must start
    // from position 0 again; without this they keep advancing across windows
    // and run past max_timestep.
    resetAttentionCache();

    std::vector<float *> input{mels.data() + static_cast<size_t>(i) * mel_len};
    std::vector<float *> label;
    std::vector<float *> output = model->incremental_inference(
      BATCH_SIZE, input, label, NUM_PATCHES, 0, NUM_PATCHES, false);
    std::copy(output[0], output[0] + NUM_CLASSES,
              all.begin() + static_cast<size_t>(i) * NUM_CLASSES);
    delete[] output[0];
  }
  return all;
}

/**
 * @brief Rewind every attention layer's KV cache write position to 0.
 */
void CedTransformer::resetAttentionCache() {
  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn = [](ml::train::Layer &l, nntrainer::RunLayerContext &, void *) {
      if (l.getType() == "mha_core")
        l.setProperty({"cache_index=0"});
    };
  model->forEachLayer(fn, nullptr);
}

/**
 * @brief Slide over a wav file, classify every window and report detections.
 *
 * Mirrors the reference pipeline end to end: int16 samples scaled by 1/32768,
 * fixed-length windows at a fixed stride, the log-mel front-end per window,
 * then a sigmoid over the head output and a per-class threshold test. The
 * printed table is byte-identical in layout to the reference so the two can be
 * diffed directly.
 */
void CedTransformer::runAudioFile(const std::string &path) {
  const audio::Waveform wav = audio::readWav16(path, NORMALIZE_DIVISOR);
  if (wav.sample_rate != FRONT_END.sample_rate) {
    throw std::runtime_error(
      "wav sample rate is " + std::to_string(wav.sample_rate) +
      " but the model expects " + std::to_string(FRONT_END.sample_rate) +
      "; resampling is not part of this port, please resample beforehand");
  }
  if (WINDOW_SAMPLES == 0 || STRIDE_SAMPLES == 0)
    throw std::runtime_error("front_end window/stride are not configured");

  const size_t slash = path.find_last_of('/');
  const std::string name =
    slash == std::string::npos ? path : path.substr(slash + 1);

  // Optional per-window comparison against the reference dumps.
  const char *ref_dir = std::getenv("AD_REF_DIR");
  double tol = 1e-3;
  if (const char *tol_env = std::getenv("AD_REF_TOL"))
    tol = std::strtod(tol_env, nullptr);
  double worst_diff = 0.0;
  size_t worst_window = 0;
  size_t nan_count = 0;
  size_t compared = 0;

  // Window count is known up front, so the header can be printed before the
  // first inference and each row streamed as it is produced.
  const unsigned int total =
    wav.samples.size() >= WINDOW_SAMPLES
      ? static_cast<unsigned int>(
          (wav.samples.size() - WINDOW_SAMPLES) / STRIDE_SAMPLES + 1)
      : 0u;
  // AD_JSON switches to the reference runtime's machine-readable stream; the
  // human-readable table stays the default.
  const bool json_out = std::getenv("AD_JSON") != nullptr;
  if (!json_out)
    std::cout << "=== " << name << " (" << total
              << " windows) ===" << std::endl;

  // Stage timings, so the front-end and the graph can be compared against
  // another runtime separately from model load.
  using clock = std::chrono::steady_clock;
  double frontend_ms = 0.0;
  double infer_ms = 0.0;

  // Front-end first for every window, then one batched inference.
  const size_t mel_len =
    static_cast<size_t>(INPUT_HEIGHT) * static_cast<size_t>(INPUT_WIDTH);
  std::vector<float> mels(static_cast<size_t>(total) * mel_len);
  const auto fe_start = clock::now();
  for (unsigned int i = 0; i < total; ++i) {
    std::vector<float> mel;
    const unsigned int frames = audio::logMelSpectrogram(
      FRONT_END, wav.samples.data() + static_cast<size_t>(i) * STRIDE_SAMPLES,
      WINDOW_SAMPLES, mel);
    if (frames != INPUT_WIDTH) {
      throw std::runtime_error("front-end produced " + std::to_string(frames) +
                               " frames but the graph input width is " +
                               std::to_string(INPUT_WIDTH));
    }
    std::copy(mel.begin(), mel.end(), mels.begin() + i * mel_len);
  }
  frontend_ms =
    std::chrono::duration<double, std::milli>(clock::now() - fe_start).count();

  const auto inf_start = clock::now();
  std::vector<float> scores_all = inferWindows(mels, total);
  infer_ms =
    std::chrono::duration<double, std::milli>(clock::now() - inf_start).count();

  // The graph applies the sigmoid only for the plain-average pooling modes;
  // otherwise it belongs to post-processing, as in the reference.
  if (!HEAD_SIGMOID) {
    for (float &v : scores_all)
      v = 1.0f / (1.0f + std::exp(-v));
  }

  unsigned int index = 0;
  for (index = 0; index < total; ++index) {
    const float *scores =
      scores_all.data() + static_cast<size_t>(index) * NUM_CLASSES;

    for (unsigned int c = 0; c < NUM_CLASSES; ++c) {
      uint32_t bits;
      std::memcpy(&bits, &scores[c], sizeof(bits));
      if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u)
        ++nan_count;
    }

    if (ref_dir) {
      const std::string ref_path =
        std::string(ref_dir) + "/window" + std::to_string(index) + "_probs.bin";
      std::ifstream ref(ref_path, std::ios::binary | std::ios::ate);
      if (ref) {
        const size_t n = static_cast<size_t>(ref.tellg()) / sizeof(float);
        if (n == NUM_CLASSES) {
          ref.seekg(0);
          std::vector<float> ref_data(n);
          ref.read(reinterpret_cast<char *>(ref_data.data()),
                   static_cast<std::streamsize>(n * sizeof(float)));
          for (size_t i = 0; i < n; ++i) {
            const double d =
              std::fabs(static_cast<double>(scores[i]) - ref_data[i]);
            if (d > worst_diff) {
              worst_diff = d;
              worst_window = index;
            }
          }
          ++compared;
        }
      }
    }

    std::vector<size_t> order(NUM_CLASSES);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t a, size_t b) { return scores[a] > scores[b]; });

    // The window label spans the actual window length, which is not the stride:
    // a 3 s window advancing 1 s at a time covers [i, i + 3).
    const unsigned int win_s = WINDOW_SAMPLES / FRONT_END.sample_rate;
    const unsigned int step_s = STRIDE_SAMPLES / FRONT_END.sample_rate;
    const unsigned int begin_s = index * (step_s ? step_s : 1);
    std::ostringstream row;
    row << "  [" << std::setw(4) << begin_s << "-" << std::setw(4)
        << (begin_s + (win_s ? win_s : 1)) << "s] ";
    const size_t k = std::min<size_t>(TOP_K, order.size());
    for (size_t i = 0; i < k; ++i) {
      if (i)
        row << ", ";
      const size_t c = order[i];
      row << (c < LABELS.size() ? LABELS[c] : std::to_string(c)) << "="
          << std::fixed << std::setprecision(3) << scores[c];
      row.unsetf(std::ios::floatfield);
    }

    std::vector<std::string> detected;
    for (const size_t c : order) {
      if (c < THRESHOLDS.size() && scores[c] >= THRESHOLDS[c])
        detected.push_back(c < LABELS.size() ? LABELS[c] : std::to_string(c));
    }
    if (!detected.empty()) {
      row << "  <-- DETECTED: [";
      for (size_t i = 0; i < detected.size(); ++i)
        row << (i ? ", " : "") << "'" << detected[i] << "'";
      row << "]";
    }
    if (json_out) {
      // Same field order and precision as the reference runtime's --json, so
      // the two streams can be diffed line by line.
      const int64_t t0 = (int64_t)begin_s * 1000000;
      const int64_t t1 = t0 + (int64_t)(win_s ? win_s : 1) * 1000000;
      std::ostringstream js;
      js << std::fixed << std::setprecision(6);
      js << "{\"window\":" << index << ",\"timestamp_begin_us\":" << t0
         << ",\"timestamp_end_us\":" << t1
         << ",\"has_detection\":" << (detected.empty() ? "false" : "true")
         << ",\"detected_events\":[";
      for (size_t i = 0; i < detected.size(); ++i) {
        const size_t pos = static_cast<size_t>(
          std::find(LABELS.begin(), LABELS.end(), detected[i]) -
          LABELS.begin());
        if (i)
          js << ",";
        js << "{\"label\":\"" << detected[i] << "\",\"score\":" << scores[pos]
           << ",\"threshold\":" << THRESHOLDS[pos] << "}";
      }
      js << "],\"all_scores\":{";
      for (unsigned int c = 0; c < NUM_CLASSES; ++c)
        js << (c ? "," : "") << "\"" << LABELS[c] << "\":" << scores[c];
      js << "},\"thresholds\":{";
      for (unsigned int c = 0; c < NUM_CLASSES; ++c)
        js << (c ? "," : "") << "\"" << LABELS[c] << "\":" << THRESHOLDS[c];
      js << "}}";
      std::cout << js.str() << std::endl;
    } else {
      std::cout << row.str() << std::endl;
    }
  }

  const double audio_ms =
    1000.0 * static_cast<double>(wav.samples.size()) / FRONT_END.sample_rate;
  std::cout << std::fixed << std::setprecision(2)
            << "[AD_TIME] windows=" << total << " frontend=" << frontend_ms
            << "ms infer=" << infer_ms
            << "ms total=" << (frontend_ms + infer_ms)
            << "ms per_window=" << (frontend_ms + infer_ms) / total
            << "ms audio=" << audio_ms / 1000.0
            << "s realtime=" << std::setprecision(4)
            << (frontend_ms + infer_ms) / audio_ms << "x" << std::endl;
  std::cout.unsetf(std::ios::floatfield);

  if (ref_dir) {
    std::cout << "[AD_REF] windows=" << compared << "/" << index
              << " nan=" << nan_count << " worst_max_abs_diff=" << worst_diff
              << " (window " << worst_window << ") tol=" << tol << " "
              << ((compared == index && compared > 0 && nan_count == 0 &&
                   worst_diff < tol)
                    ? "PASS"
                    : "FAIL")
              << std::endl;
  }
}

} // namespace causallm
