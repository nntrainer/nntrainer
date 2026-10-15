// SPDX-License-Identifier: Apache-2.0
/** @file gemma4_prefill_device_check.cpp
 * @brief Measure Gemma4 prefill through the public API without decode.
 * Arguments: model base directory, native library directory, prompt file.
 * The base must contain gemma4-26b-a4b-htp-w4a32/ with the desired config.
 * Use skip_prefill=false to include all prompt tokens and the WH head.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug See docs/gemma4-tl2-affine-handoff.md for validation limits.
 */
#include "quick_dot_ai_api.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>

static void check(ErrorCode rc, const char *operation) {
  if (rc) {
    std::fprintf(stderr, "%s failed %d: %s\n", operation, int(rc),
                 getLastModelLoadError());
    std::exit(1);
  }
}

static int first_token(const char *delta, void *data) {
  ++*static_cast<unsigned *>(data);
  std::printf("PREFILL_TOKEN %s\n", delta);
  std::fflush(stdout);
  return 1;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s MODEL_BASE LIB_DIR PROMPT_FILE\n", argv[0]);
    return 2;
  }
  std::ifstream file(argv[3]);
  const std::string prompt((std::istreambuf_iterator<char>(file)), {});
  if (!file || prompt.empty())
    return 2;
  Config options{};
  options.use_chat_template = false;
  check(setOptions(options), "options");
  CausalLmHandle handle = nullptr;
  check(loadModelHandleByName(CAUSAL_LM_BACKEND_NPU, "gemma4-26b-a4b-htp",
                              CAUSAL_LM_QUANTIZATION_W4A32, argv[2], argv[1],
                              &handle),
        "load");
  /** Cancellation is checked by the generation loop, not the prefill
   * forward pass. Keep it set even if the first token has no text delta. */
  std::atomic<bool> done{false};
  std::thread stop([&] {
    while (!done.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      if (!done.load())
        cancelModelHandle(handle);
    }
  });
  unsigned callbacks = 0;
  const ErrorCode result =
    runModelHandleStreaming(handle, prompt.c_str(), first_token, &callbacks);
  done.store(true);
  stop.join();
  check(result, "prefill");
  PerformanceMetrics metrics{};
  check(getPerformanceMetricsHandle(handle, &metrics), "metrics");
  std::printf("PREFILL_RESULT tokens=%u ms=%.3f tps=%.3f ttft_ms=%.3f "
              "decode_tokens=%u callbacks=%u\n",
              metrics.prefill_tokens, metrics.prefill_duration_ms,
              metrics.prefill_tokens * 1000.0 / metrics.prefill_duration_ms,
              metrics.ttft_ms, metrics.generation_tokens, callbacks);
  check(destroyModelHandle(handle), "destroy");
  return metrics.generation_tokens != 0;
}
