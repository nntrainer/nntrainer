// SPDX-License-Identifier: Apache-2.0
/**
 * @file   vjepa2_qnn.h
 * @brief  QNN model implementation for V-JEPA2 video encoder.
 * @note   This class is not to be executed alone.
 *
 */

#ifndef __VJEPA2_QNN_H__
#define __VJEPA2_QNN_H__

#include "nntrainer_error.h"
#include "quick_dot_ai_qnn.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "QuickAI"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#define LOGD(fmt, ...) fprintf(stdout, fmt "\n", ##__VA_ARGS__)
#define LOGE(fmt, ...) fprintf(stderr, fmt "\n", ##__VA_ARGS__)
#endif

namespace causallm {

/**
 * @brief VJEPA2_QNN class
 * @note  This class runs V-JEPA2 video encoder QNN class and that's all.
 *
 */
class VJEPA2_QNN : public Quick_Dot_AI_QNN {

public:
  static constexpr const char *architectures = "VJEPA2_QNN";

  VJEPA2_QNN(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Quick_Dot_AI_QNN(cfg, generation_cfg, nntr_cfg) {}

  ~VJEPA2_QNN() override;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;
  void initialize() override;

  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = "", const WSTR tail_prompt = "",
           bool log_output = true) override;

  multimodal_pointer run_image(const WSTR prompt, multimodal_pointer image,
                               int image_height, int image_width,
                               bool do_sample = false,
                               const WSTR system_prompt = "",
                               const WSTR tail_prompt = "",
                               bool log_output = true) override;
  void set_quant_param(float scale, int offset) override;

private:
  // QNN input tensor pointers (cached in initialize())
  uint8_t *rotation_matrix_input_ = nullptr; // UF8  [64,64]
  uint16_t *pixel_values_input_ = nullptr;   // UF16 [12,256,256,6]
  uint16_t *rope_cos_input_ = nullptr;       // UF16 [1,1,3072,64]
  uint16_t *rope_sin_input_ = nullptr;       // UF16 [1,1,3072,64]

  // mmap state for rotation_matrix binary file
  void *rotation_matrix_mmap_ptr_ = nullptr;
  size_t rotation_matrix_mmap_size_ = 0;
  std::string rotation_matrix_path_;
  std::string rope_cos_path_;
  std::string rope_sin_path_;

  // LLM quant params for output requantization
  bool llm_quant_param_given_ = false;
  float llm_scale_ = 1.0f;
  int llm_offset_ = 0;

  TensorInfo get_input_info();  // Returns pixel_values_video info
  TensorInfo get_output_info(); // Returns image_embeds info

  void requantEmbedding(void *from, void *to, size_t length);
  void loadRotationMatrix(); // mmap + memcpy from rotation_matrix_path_
  void loadTensorFromFile(const std::string &path, void *dest,
                          size_t expected_size, const std::string &name);

  // -------------------------------------------------------------------
  // Video preprocessing
  // -------------------------------------------------------------------
  int tubelet_size_ = 2;              // Read from nntr_cfg at setup time
  std::string input_format_ = "auto"; // "auto", "raw", "preprocessed"

  /** In-place reshape + quantize into the QNN graph's NHWC buffer. */
  void preprocessToQnnInput(const float *raw_nchw, int B, int T, int C, int H,
                            int W, uint16_t *qnn_hwc_dest, float scale,
                            int offset);

  /** Fixed-shape fast path: (B=1,T=24,C=3,H=256,W=256). */
  void preprocessToQnnInput_FixedShape(const float *__restrict raw,
                                       uint16_t *__restrict dst, float scale,
                                       int offset);

  /** Standalone helper for unit tests / external callers. */
  static std::vector<uint8_t> frameToDepth(const float *raw_nchw, int B, int T,
                                           int C, int H, int W,
                                           int tubelet_size, float scale,
                                           int offset);

  /** Fixed-shape standalone helper. */
  static std::vector<uint8_t> frameToDepth_FixedShape(const float *raw,
                                                      float scale, int offset);
};

} // namespace causallm

#endif /* __VJEPA2_QNN_H__ */
