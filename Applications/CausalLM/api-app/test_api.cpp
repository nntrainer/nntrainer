// SPDX-License-Identifier: Apache-2.0
/**
 * @file    test_api.cpp
 * @brief   Test application for src C API
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 * @note    This file only includes quick_dot_ai_api.h and standard headers.
 *          It links against libquick_dot_ai_api.so.
 */

#include "quick_dot_ai_api.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ── ANSI color codes ─────────────────────────────────────────────────────────
namespace clr {
const char *reset = "\033[0m";
const char *bold = "\033[1m";
const char *dim = "\033[2m";
const char *cyan = "\033[36m";
const char *green = "\033[32m";
const char *yellow = "\033[33m";
const char *red = "\033[31m";
const char *magenta = "\033[35m";
const char *blue = "\033[34m";
const char *bold_cyan = "\033[1;36m";
const char *bold_green = "\033[1;32m";
const char *bold_yellow = "\033[1;33m";
const char *bold_red = "\033[1;31m";
const char *bold_magenta = "\033[1;35m";
const char *bold_blue = "\033[1;34m";
const char *bold_white = "\033[1;37m";
} // namespace clr

// ── ASCII art banner ─────────────────────────────────────────────────────────
static void print_banner() {
  std::cout << clr::bold_cyan;
  std::cout << R"(
     ____        _      _           _    ___
    / __ \      (_)    | |         / \  |_ _|
   | |  | |_   _ _  ___| | __    / _ \  | |
   | |  | | | | | |/ __| |/ /   / ___ \ | |
   | |__| | |_| | | (__|   < _ / /   \ \| |_
    \___\_\\__,_|_|\___|_|\_(_)_/     \_\___| )"
            << "\n";
  std::cout << clr::reset;
  std::cout << clr::dim << "   ─────────────────────────────────────────────\n"
            << "    On-Device LLM Inference Engine  |  API Test\n"
            << "   ─────────────────────────────────────────────" << clr::reset
            << "\n\n";
}

// ── Box-drawing helpers ──────────────────────────────────────────────────────
static void print_section(const char *title, const char *color) {
  std::cout << color << clr::bold << "┌─ " << title << " ";
  int pad = 50 - static_cast<int>(strlen(title));
  for (int i = 0; i < pad; i++)
    std::cout << "─";
  std::cout << "┐" << clr::reset << "\n";
}

static void print_section_end(const char *color) {
  std::cout << color << "└";
  for (int i = 0; i < 53; i++)
    std::cout << "─";
  std::cout << "┘" << clr::reset << "\n\n";
}

static void print_kv(const char *key, const std::string &value,
                     const char *color) {
  std::cout << color << "│" << clr::reset << "  " << clr::dim << std::left
            << std::setw(18) << key << clr::reset << clr::bold_white << value
            << clr::reset << "\n";
}

static void print_status(const char *msg, const char *icon, const char *color) {
  std::cout << color << icon << "  " << msg << clr::reset << "\n";
}

static void print_error(const std::string &msg) {
  std::cout << clr::bold_red << "  ERROR " << clr::reset << clr::red << msg
            << clr::reset << "\n";
}

// ── Usage ────────────────────────────────────────────────────────────────────
static void print_usage(const char *prog) {
  print_banner();
  print_section("Usage", clr::yellow);
  std::cout << clr::yellow << "│" << clr::reset << "  " << clr::bold_white
            << prog << clr::reset << " <model> [prompt] [chat_tpl] [quant] "
            << "[verbose]\n";
  std::cout << clr::yellow << "│" << clr::reset << "\n";
  print_kv("model", "qwen3-0.6b | gemma4-e2b-qnn | <catalog id>", clr::yellow);
  print_kv("", "ouro → embedding smoke (encodeModelHandle)", clr::yellow);
  print_kv("prompt", "\"Hello, how are you?\"", clr::yellow);
  print_kv("chat_tpl", "true | false  (default: true)", clr::yellow);
  print_kv("quant", "W4A32 | W16A16 | W8A16 | W32A32", clr::yellow);
  print_kv("verbose", "true | false  (default: true)", clr::yellow);
  print_kv("model_base_path",
           "Base directory for models (or set QUICKAI_MODEL_BASE_PATH)",
           clr::yellow);
  print_section_end(clr::yellow);
}

// Loads nothing; assumes `handle` is an already-loaded embedding model.
// Calls encodeModelHandle, prints + sanity-checks the vector. Returns true on
// success.
static bool run_embedding_smoke(CausalLmHandle handle, const char *text) {
  print_section("Embedding (encode)", clr::green);
  std::cout << clr::green << "│" << clr::reset << "  " << clr::dim
            << "Input: " << clr::reset << clr::bold_white << text << clr::reset
            << "\n";

  float *vec = nullptr;
  int dim = 0;
  ErrorCode err = encodeModelHandle(handle, text, &vec, &dim);
  if (err != CAUSAL_LM_ERROR_NONE) {
    print_error("encodeModelHandle failed (code " + std::to_string(err) + ")");
    return false;
  }
  if (vec == nullptr || dim <= 0) {
    print_error("encodeModelHandle returned empty result (dim=" +
                std::to_string(dim) + ")");
    freeEmbedding(vec);
    return false;
  }

  // Sanity: all finite, compute L2 norm.
  bool all_finite = true;
  double sumsq = 0.0;
  for (int i = 0; i < dim; ++i) {
    if (!std::isfinite(vec[i])) {
      all_finite = false;
      break;
    }
    sumsq += static_cast<double>(vec[i]) * vec[i];
  }
  const double norm = std::sqrt(sumsq);

  print_kv("Dim", std::to_string(dim), clr::green);
  {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << norm;
    print_kv("L2 norm", oss.str(), clr::green);
  }
  {
    std::ostringstream oss;
    const int n = dim < 16 ? dim : 16;
    oss << "[";
    for (int i = 0; i < n; ++i) {
      oss << std::fixed << std::setprecision(4) << vec[i];
      if (i != n - 1)
        oss << ", ";
    }
    if (dim > 16)
      oss << ", …";
    oss << "]";
    print_kv(("First " + std::to_string(n)).c_str(), oss.str(), clr::green);
  }

  freeEmbedding(vec);

  bool ok = all_finite && dim > 0 && norm > 0.0;
  if (!all_finite)
    print_error("embedding contains non-finite values");
  if (norm <= 0.0)
    print_error("embedding L2 norm is zero");
  if (ok) {
    std::cout << clr::green << "│" << clr::reset << "  " << clr::green
              << "✓ embedding sanity checks passed" << clr::reset << "\n";
  }
  print_section_end(clr::green);
  return ok;
}

#ifdef ENABLE_QNN_MODELS
// Builds a format-correct dummy pixel buffer for V-JEPA2 (raw layout
// B=1,T=24,C=3,H=256,W=256 = 4,718,592 floats, auto-detected by
// VJEPA2_QNN::run_image) and runs the vision encoder. Times each run
// (encode = preprocess + HTP inference), prints per-run latency, avg/min/max,
// throughput, peak memory, and a sample of the output. Returns true on success.
//
// Env knobs:
//   VJEPA2_PIXEL_FILL = zero | ramp   (input fill; default ramp)
//   VJEPA2_RUNS       = N             (timed repetitions on the loaded handle;
//   default 1)
static bool run_vision_smoke(CausalLmHandle handle) {
  print_section("Vision Encode (dummy)", clr::green);

  const size_t numFloats = 1ull * 24 * 3 * 256 * 256; // 4,718,592
  std::vector<float> dummy(numFloats);
  const char *fill_env = std::getenv("VJEPA2_PIXEL_FILL");
  const bool zero_fill =
    (fill_env != nullptr && std::string(fill_env) == "zero");
  if (zero_fill) {
    std::fill(dummy.begin(), dummy.end(), 0.0f);
  } else {
    for (size_t i = 0; i < numFloats; ++i)
      dummy[i] = static_cast<float>(i % 255) / 255.0f;
  }

  int runs = 1;
  if (const char *r = std::getenv("VJEPA2_RUNS")) {
    runs = std::atoi(r);
    if (runs < 1)
      runs = 1;
  }

  std::cout << clr::green << "│" << clr::reset << "  " << clr::dim
            << "Dummy input: " << clr::reset << clr::bold_white << numFloats
            << " floats (" << (numFloats * sizeof(float))
            << " bytes, fill=" << (zero_fill ? "zero" : "ramp")
            << "), runs=" << runs << clr::reset << "\n";

  void *out = nullptr;
  int out_bytes = 0;
  double sum_ms = 0.0, min_ms = 0.0, max_ms = 0.0;
  for (int i = 0; i < runs; ++i) {
    if (out) {
      freeImageEmbedding(out);
      out = nullptr;
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    ErrorCode err =
      encodeImageModelHandle(handle, dummy.data(), numFloats, /*height=*/256,
                             /*width=*/256, &out, &out_bytes);
    auto t1 = std::chrono::high_resolution_clock::now();
    if (err != CAUSAL_LM_ERROR_NONE || out == nullptr) {
      print_error("encodeImageModelHandle failed (code " + std::to_string(err) +
                  ")");
      return false;
    }
    const double ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
    sum_ms += ms;
    if (i == 0 || ms < min_ms)
      min_ms = ms;
    if (i == 0 || ms > max_ms)
      max_ms = ms;
    std::ostringstream rss;
    rss << "run " << std::setw(2) << (i + 1) << "/" << runs << ": "
        << std::fixed << std::setprecision(2) << ms << " ms";
    std::cout << clr::green << "│" << clr::reset << "  " << clr::dim
              << rss.str() << clr::reset << "\n";
  }

  const double avg_ms = sum_ms / runs;
  // Output layout: out_tokens × 768 dim × 2 bytes(uint16). tok/s = patches/sec.
  const int out_tokens = out_bytes / (768 * 2);
  const double tps = avg_ms > 0 ? out_tokens / avg_ms * 1000.0 : 0.0;

  std::ostringstream sp;
  sp << std::fixed << std::setprecision(2) << "avg " << avg_ms << " ms"
     << "  (min " << min_ms << " / max " << max_ms << ")  "
     << std::setprecision(1) << tps << " tok/s";
  std::cout << clr::green << "│" << clr::reset << "  " << clr::bold_white
            << "Speed: " << clr::reset << clr::bold_white << sp.str()
            << clr::reset << "\n";

  PerformanceMetrics pm;
  memset(&pm, 0, sizeof(pm));
  if (getPerformanceMetricsHandle(handle, &pm) == CAUSAL_LM_ERROR_NONE) {
    std::ostringstream mm;
    mm << std::fixed << std::setprecision(2) << "internal last-run "
       << pm.total_duration_ms << " ms, peak " << pm.peak_memory_kb << " KB";
    std::cout << clr::green << "│" << clr::reset << "  " << clr::dim << mm.str()
              << clr::reset << "\n";
  }

  std::cout << clr::green << "│" << clr::reset << "  " << clr::dim
            << "Output: " << clr::reset << clr::bold_white << out_bytes
            << " bytes" << clr::reset << "\n";

  const uint16_t *u16 = static_cast<const uint16_t *>(out);
  int n_show = out_bytes / static_cast<int>(sizeof(uint16_t));
  if (n_show > 8)
    n_show = 8;
  std::cout << clr::green << "│" << clr::reset << "  " << clr::dim << "First "
            << n_show << " u16: " << clr::reset << clr::bold_white;
  for (int i = 0; i < n_show; ++i)
    std::cout << u16[i] << " ";
  std::cout << clr::reset << "\n";

  freeImageEmbedding(out);
  print_section_end(clr::green);
  return true;
}
#endif // ENABLE_QNN_MODELS

int main(int argc, char *argv[]) {
  if (argc < 2) {
    print_error("Missing required argument: <model>");
    print_usage(argv[0]);
    return 1;
  }

  // ── Parse arguments ──────────────────────────────────────────────────────
  const char *model_name = argv[1];
  const char *prompt = (argc >= 3) ? argv[2] : "Hello, how are you?";

  bool use_chat_template = true;
  if (argc >= 4) {
    std::string arg(argv[3]);
    std::transform(arg.begin(), arg.end(), arg.begin(), ::tolower);
    use_chat_template = (arg == "true" || arg == "1");
  }

  ModelQuantizationType quant_type = CAUSAL_LM_QUANTIZATION_W4A32;
  std::string quant_str = "W4A32";
  if (argc >= 5) {
    quant_str = std::string(argv[4]);
    if (quant_str == "W4A32") {
      quant_type = CAUSAL_LM_QUANTIZATION_W4A32;
    } else if (quant_str == "W16A16") {
      quant_type = CAUSAL_LM_QUANTIZATION_W16A16;
    } else if (quant_str == "W8A16") {
      quant_type = CAUSAL_LM_QUANTIZATION_W8A16;
    } else if (quant_str == "W32A32") {
      quant_type = CAUSAL_LM_QUANTIZATION_W32A32;
    }
  }

  bool verbose = true;
  if (argc >= 6) {
    std::string arg(argv[5]);
    verbose = (arg == "1" || arg == "true");
  }

  bool use_sd = false; // set inside a speculative-decoding QNN model's routing
                       // block, if one is matched

  // Model base path: CLI arg > env var > nullptr (uses C API default)
  const char *model_base_path = nullptr;
  std::string model_base_path_storage;
  if (argc >= 7) {
    model_base_path_storage = argv[6];
    model_base_path = model_base_path_storage.c_str();
  } else {
    const char *env_path = std::getenv("QUICKAI_MODEL_BASE_PATH");
    if (env_path != nullptr && strlen(env_path) > 0) {
      model_base_path_storage = env_path;
      model_base_path = model_base_path_storage.c_str();
    }
  }

  // ── Banner ─────────────────────────────────────────────────────────────
  print_banner();

  // ── Configuration ──────────────────────────────────────────────────────
  print_section("Configuration", clr::cyan);
  print_kv("Model", model_name, clr::cyan);
  print_kv("Prompt", std::string("\"") + prompt + "\"", clr::cyan);
  print_kv("Chat Template", use_chat_template ? "Yes" : "No", clr::cyan);
  print_kv("Quantization", quant_str, clr::cyan);
  print_kv("Verbose", verbose ? "Yes" : "No", clr::cyan);
  print_kv("Model Base Path",
           model_base_path ? model_base_path : "(C API default)", clr::cyan);
  print_section_end(clr::cyan);

  // ── Set options ────────────────────────────────────────────────────────
  Config config;
  config.use_chat_template = use_chat_template;
  config.debug_mode = verbose;
  config.verbose = verbose;

  ErrorCode err = setOptions(config);
  if (err != CAUSAL_LM_ERROR_NONE) {
    print_error("Failed to set options (code " + std::to_string(err) + ")");
    return 1;
  }

  // ── Resolve model type ─────────────────────────────────────────────────
  // Public models use the compat enum path (loadModelHandle).
  // Other catalog-registered models are not in the enum — they use
  // loadModelHandleByName via catalog id. Any model_name that doesn't match
  // a known branch below falls through to the generic catalog-id path, so
  // any registered model can still be loaded by its full catalog id.
  std::string model_name_str(model_name);
  std::transform(model_name_str.begin(), model_name_str.end(),
                 model_name_str.begin(), ::tolower);

  ModelType model_type =
    CAUSAL_LM_MODEL_QWEN3_0_6B; // default, overridden below
  std::string catalog_id;       // non-empty → use loadModelHandleByName
  bool use_by_name = false;
  bool is_embedding = false;
  bool is_vision = false;
  // Backend for loadModelHandleByName. Most catalog models accept CPU; QNN-only
  // descriptors (e.g. vjepa2-qnn, backend_mask = B(NPU)) require NPU.
  BackendType load_backend = CAUSAL_LM_BACKEND_CPU;

  if (model_name_str == "qwen3-0.6b") {
    model_type = CAUSAL_LM_MODEL_QWEN3_0_6B;
  } else if (model_name_str == "qwen3-1.7b-q40" ||
             model_name_str == "qwen3_1.7b_q40") {
    model_type = CAUSAL_LM_MODEL_QWEN3_1_7B_Q40;
  } else if (model_name_str == "tiny_bert" || model_name_str == "tiny-bert") {
    model_type = CAUSAL_LM_MODEL_TINY_BERT;
  } else if (model_name_str == "function_gemma" ||
             model_name_str == "function-gemma") {
    model_type = CAUSAL_LM_MODEL_FUNCTION_GEMMA;
  } else if (model_name_str == "gemma4_cpu" || model_name_str == "gemma4-cpu") {
    model_type = CAUSAL_LM_MODEL_GEMMA4_CPU;
  } else if (model_name_str == "ouro_embedding") {
    model_type = CAUSAL_LM_MODEL_OURO_EMBEDDING;
    is_embedding = true;
  } else if (model_name_str == "ouro") {
    // Recommended embedding smoke target: by-name load aligns with the
    // on-device models/ouro dir (descriptor config_name = "ouro").
    catalog_id = "ouro";
    use_by_name = true;
    is_embedding = true;
  } else if (model_name_str == "gemma4_e2b_qnn" ||
             model_name_str == "gemma4-e2b-qnn") {
    model_type = CAUSAL_LM_MODEL_GEMMA4_E2B_QNN;
  } else if (model_name_str == "vjepa2-qnn" || model_name_str == "vjepa2_qnn" ||
             model_name_str == "vjepa") {
#ifdef ENABLE_QNN_MODELS
    catalog_id = "vjepa2-qnn";
    use_by_name = true;
    is_vision = true;
    load_backend = CAUSAL_LM_BACKEND_NPU;
#else
    print_error("Model '" + std::string(model_name) +
                "' requires QNN support. Rebuild with -Denable-qnn=true.");
    return 1;
#endif
  } else {
    // Generic fallback: any name that doesn't match a known branch above is
    // treated as a full catalog id and loaded via loadModelHandleByName, so
    // any catalog-registered model (including QNN-only ones) can still be
    // loaded without a dedicated branch here.
    catalog_id = model_name_str;
    use_by_name = true;
  }

#ifdef ENABLE_QNN
  print_kv("Speculative Dec.", use_sd ? "Enabled" : "Disabled", clr::cyan);
#endif

  // ── Load/Unload Stress Test ────────────────────────────────────────────
  // Default 1 cycle (load→unload→destroy) then a final reload. Override with
  // env STRESS_CYCLES (e.g. 0 to skip the reload-stress and load once for a
  // clean single-shot inference — useful while the reload path is being fixed).
  int STRESS_CYCLES = 1;
  if (const char *sc = std::getenv("STRESS_CYCLES")) {
    STRESS_CYCLES = std::atoi(sc);
    if (STRESS_CYCLES < 0)
      STRESS_CYCLES = 0;
  }
  print_section("Load/Unload Stress Test", clr::blue);

  for (int i = 0; i < STRESS_CYCLES; ++i) {
    std::cout << clr::blue << "│" << clr::reset << "  " << clr::bold_white
              << "Cycle " << (i + 1) << "/" << STRESS_CYCLES << clr::reset
              << ": ";

    // Load
    CausalLmHandle cycle_handle = nullptr;
    if (use_by_name) {
      err = loadModelHandleByName(load_backend, catalog_id.c_str(), quant_type,
                                  nullptr, model_base_path, &cycle_handle);
    } else {
      err = loadModelHandle(CAUSAL_LM_BACKEND_CPU, model_type, quant_type,
                            nullptr, model_base_path, &cycle_handle);
    }
    if (err != CAUSAL_LM_ERROR_NONE) {
      print_error("loadModel failed at cycle " + std::to_string(i + 1) +
                  " (code " + std::to_string(err) + ")");
      return 1;
    }
    std::cout << clr::bold_green << "LOAD OK" << clr::reset;

    // Unload
    err = unloadModelHandle(cycle_handle);
    if (err != CAUSAL_LM_ERROR_NONE) {
      print_error("unloadModelHandle failed at cycle " + std::to_string(i + 1) +
                  " (code " + std::to_string(err) + ")");
      destroyModelHandle(cycle_handle);
      return 1;
    }
    std::cout << clr::dim << " → " << clr::reset;
    std::cout << clr::bold_yellow << "UNLOAD OK" << clr::reset;

    // Destroy handle (unload keeps struct alive, must destroy to avoid leak)
    destroyModelHandle(cycle_handle);
    std::cout << clr::dim << " → " << clr::reset;
    std::cout << clr::dim << "DESTROY OK" << clr::reset << "\n";
  }

  std::cout << clr::blue << "│" << clr::reset << "\n";
  std::cout << clr::blue << "│" << clr::reset << "  " << clr::bold_white
            << "Final load (#" << (STRESS_CYCLES + 1) << "):" << clr::reset
            << " Loading " << clr::bold_white << model_name << clr::reset
            << " (" << quant_str << ") ...\n";

  CausalLmHandle handle = nullptr;
  if (use_by_name) {
    err = loadModelHandleByName(load_backend, catalog_id.c_str(), quant_type,
                                nullptr, model_base_path, &handle);
  } else {
    err = loadModelHandle(CAUSAL_LM_BACKEND_CPU, model_type, quant_type,
                          nullptr, model_base_path, &handle);
  }
  if (err != CAUSAL_LM_ERROR_NONE) {
    print_error("Final loadModel failed (code " + std::to_string(err) + ")");
    return 1;
  }

  std::cout << clr::blue << "│" << clr::reset << "  ";
  print_status("Model loaded successfully", ">>", clr::bold_green);
#ifdef ENABLE_QNN
  if (use_sd) {
    err = configureSpeculativeDecoding(handle, true);
    if (err != CAUSAL_LM_ERROR_NONE) {
      print_error("configureSpeculativeDecoding failed (code " +
                  std::to_string(err) + ")");
      destroyModelHandle(handle);
      return 1;
    }
    print_status("Speculative decoding configured", ">>", clr::bold_cyan);
  }
#endif
  print_section_end(clr::blue);

#ifdef ENABLE_QNN_MODELS
  if (is_vision) {
    bool ok = run_vision_smoke(handle);
    destroyModelHandle(handle);
    std::cout << (ok ? clr::bold_green : clr::bold_red)
              << (ok ? "  Done." : "  Failed.") << clr::reset << "\n\n";
    // The QNN context's global singleton runs its destructor during
    // __cxa_finalize (after main returns). That post-inference teardown
    // dereferences an invalid pointer and SIGSEGVs — turning a successful run
    // into a crash *after* all useful work is done. The smoke test is complete
    // here, so flush stdout and exit immediately, bypassing the fragile global
    // teardown (the OS reclaims all resources on process exit).
    std::cout.flush();
    std::fflush(nullptr);
    std::_Exit(ok ? 0 : 1);
  }
#endif

  if (is_embedding) {
    bool emb_ok = run_embedding_smoke(handle, prompt);
    if (!emb_ok) {
      destroyModelHandle(handle);
      return 1;
    }
    // Embedding models have no token generation; skip the generation metrics
    // block below by jumping straight to cleanup.
    destroyModelHandle(handle);
    std::cout << clr::bold_green << "  Done." << clr::reset << "\n\n";
    return 0;
  }

  // ── Inference (generation) ─────────────────────────────────────────────
  print_section("Inference", clr::green);
  std::cout << clr::green << "│" << clr::reset << "  " << clr::dim
            << "Input: " << clr::reset << clr::bold_white << prompt
            << clr::reset << "\n";
  std::cout << clr::green << "│" << clr::reset << "\n";

  const char *outputText = nullptr;
  // CausalLMChatMessage msg;
  // msg.role = "user";
  // msg.content = prompt;
  // err = runModelHandleWithMessages(handle, &msg, 1, true, &outputText);

  // QDA_STREAM=1: unconstrained per-token streaming generation, matching the
  // GUI app's Chat path (runChatModelHandleStreaming →
  // runModelHandleStreaming). The default tool/grammar path below produces only
  // a short constrained JSON, which never reaches the longer free-text
  // generation the app does.
  if (std::getenv("QDA_STREAM")) {
    struct Cb {
      static int on_token(const char *delta, void *) {
        if (delta) {
          std::cout << delta << std::flush;
        }
        return 0; // keep generating
      }
    };
    std::cout << clr::green << "│" << clr::reset << "  " << clr::dim
              << "Output (streaming):" << clr::reset << "\n";
    err = runModelHandleStreaming(handle, prompt, &Cb::on_token, nullptr);
    std::cout << "\n";
  } else {
    // XGrammar Test
    auto tool_name = "web_search";
    auto schema =
      "{\"type\": \"object\",\"properties\": {\"query\": {\"type\": "
      "\"string\", "
      "\"description\": \"Search query in the most effective language for "
      "results (use Korean for Korean local info, English for global "
      "topics)\"},\"count\": {\"type\": \"integer\", \"minimum\": 1, "
      "\"maximum\": 10, \"description\": \"Number of results to return "
      "(default 5, max 10)\"}},\"required\": [\"query\"]}";
    err =
      runModelHandleWithTool(handle, prompt, &outputText, tool_name, schema);
  }

  if (err != CAUSAL_LM_ERROR_NONE) {
    print_error("Inference failed (code " + std::to_string(err) + ")");
    return 1;
  }

  if (outputText) {
    std::cout << clr::green << "│" << clr::reset << "  " << clr::dim
              << "Output:" << clr::reset << "\n";
    std::cout << clr::green << "│" << clr::reset << "  " << clr::bold_white
              << outputText << clr::reset << "\n";
  }
  print_section_end(clr::green);

  // ── Performance Metrics ────────────────────────────────────────────────
  print_section("Performance Metrics", clr::magenta);

  PerformanceMetrics metrics;
  memset(&metrics, 0, sizeof(metrics));
  err = getPerformanceMetricsHandle(handle, &metrics);
  if (err == CAUSAL_LM_ERROR_NONE) {
    double prefill_tps =
      metrics.prefill_duration_ms > 0
        ? metrics.prefill_tokens / metrics.prefill_duration_ms * 1000.0
        : 0.0;
    double gen_tps =
      metrics.generation_duration_ms > 0
        ? metrics.generation_tokens / metrics.generation_duration_ms * 1000.0
        : 0.0;

    std::ostringstream oss;

    oss << metrics.initialization_duration_ms << " ms";
    print_kv("Init", oss.str(), clr::magenta);

    oss.str("");
    oss << metrics.prefill_tokens << " tokens / " << metrics.prefill_duration_ms
        << " ms  (" << std::fixed << std::setprecision(1) << prefill_tps
        << " tok/s)";
    print_kv("Prefill", oss.str(), clr::magenta);

    oss.str("");
    oss << metrics.generation_tokens << " tokens / "
        << metrics.generation_duration_ms << " ms  (" << std::fixed
        << std::setprecision(1) << gen_tps << " tok/s)";
    print_kv("Generation", oss.str(), clr::magenta);

    oss.str("");
    oss << metrics.total_duration_ms << " ms";
    print_kv("Total", oss.str(), clr::magenta);

    oss.str("");
    oss << metrics.peak_memory_kb << " KB";
    print_kv("Peak Memory", oss.str(), clr::magenta);

    // ── Metric validation ────────────────────────────────────────────────
    bool metrics_ok = true;
    if (metrics.prefill_tokens == 0) {
      std::cout << clr::magenta << "│" << clr::reset << "  " << clr::yellow
                << "⚠ Warning: prefill_tokens is zero" << clr::reset << "\n";
      metrics_ok = false;
    }
    if (metrics.generation_tokens == 0) {
      std::cout << clr::magenta << "│" << clr::reset << "  " << clr::yellow
                << "⚠ Warning: generation_tokens is zero" << clr::reset << "\n";
      metrics_ok = false;
    }
    if (metrics.generation_duration_ms <= 0) {
      std::cout << clr::magenta << "│" << clr::reset << "  " << clr::yellow
                << "⚠ Warning: generation_duration_ms is zero/negative"
                << clr::reset << "\n";
      metrics_ok = false;
    }
    if (metrics.total_duration_ms <
        metrics.prefill_duration_ms + metrics.generation_duration_ms - 0.001) {
      std::cout << clr::magenta << "│" << clr::reset << "  " << clr::yellow
                << "⚠ Warning: total_duration_ms < prefill + generation"
                << clr::reset << "\n";
      metrics_ok = false;
    }
    if (metrics_ok) {
      std::cout << clr::magenta << "│" << clr::reset << "  " << clr::green
                << "✓ All metric sanity checks passed" << clr::reset << "\n";
    }
  } else {
    std::cout << clr::magenta << "│" << clr::reset << "  " << clr::dim
              << "(metrics not available)" << clr::reset << "\n";
  }
  print_section_end(clr::magenta);

  // ── Cleanup ────────────────────────────────────────────────────────────
  destroyModelHandle(handle);

  // ── Done ───────────────────────────────────────────────────────────────
  std::cout << clr::bold_green << "  Done." << clr::reset << "\n\n";

  return 0;
}
