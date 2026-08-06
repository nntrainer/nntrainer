// SPDX-License-Identifier: Apache-2.0
/**
 * @file   vjepa2_qnn.cpp
 * @brief  QNN model implementation for V-JEPA2 video encoder.
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 * @note   This class is not to be executed alone.
 *
 */

#include "vjepa2_qnn.h"
#include "factory.h"
#include "nntrainer_error.h"
#include <model_descriptor.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

__attribute__((constructor)) static void register_vjepa2_qnn() {
  causallm::Factory::Instance().registerModel(
    "VJEPA2_QNN", [](causallm::json cfg, causallm::json generation_cfg,
                     causallm::json nntr_cfg) {
      return std::make_unique<causallm::VJEPA2_QNN>(cfg, generation_cfg,
                                                    nntr_cfg);
    });

  static const ModelDescriptor d = {"vjepa2-qnn",
                                    "vjepa",
                                    "V-JEPA 2 (QNN)",
                                    QDA_RUNTIME_NATIVE,
                                    (1u << 2),
                                    QDA_CAP_MULTIMODAL | QDA_CAP_MESSAGES_API |
                                      QDA_CAP_MULTI_IMAGE,
                                    "VJEPA2-QNN",
                                    "VJEPA2_QNN"};
  quick_dot_ai::register_model_descriptor(&d);
}

// ---------------------------------------------------------------------------
// Local path helpers (mirrors the copies in other QNN model cpp files).
// ---------------------------------------------------------------------------
static bool is_absolute_path_(const std::string &path) {
  return !path.empty() && path[0] == '/';
}

static std::string dirname_(const std::string &path) {
  auto pos = path.find_last_of('/');
  return (pos == std::string::npos) ? std::string() : path.substr(0, pos);
}

static std::string
rebase_relative_to_model_file(const std::string &path,
                              const std::string &model_file) {
  if (path.empty() || is_absolute_path_(path))
    return path;
  auto base = dirname_(model_file);
  if (base.empty())
    return path;
  return base + "/" + path;
}

causallm::VJEPA2_QNN::~VJEPA2_QNN() {
  if (rotation_matrix_mmap_ptr_ != nullptr &&
      rotation_matrix_mmap_ptr_ != MAP_FAILED &&
      rotation_matrix_mmap_ptr_ != reinterpret_cast<void *>(1)) {
    ::munmap(rotation_matrix_mmap_ptr_, rotation_matrix_mmap_size_);
    rotation_matrix_mmap_ptr_ = nullptr;
    rotation_matrix_mmap_size_ = 0;
  }
}

void causallm::VJEPA2_QNN::setupParameters(json &cfg, json &generation_cfg,
                                           json &nntr_cfg) {
  Quick_Dot_AI_QNN::setupParameters(cfg, generation_cfg, nntr_cfg);

  if (nntr_cfg.contains("vjepa2_tubelet_size"))
    tubelet_size_ = nntr_cfg["vjepa2_tubelet_size"].get<int>();

  if (nntr_cfg.contains("vjepa2_input_format"))
    input_format_ = nntr_cfg["vjepa2_input_format"].get<std::string>();

  if (nntr_cfg.contains("rotation_matrix_path")) {
    rotation_matrix_path_ = nntr_cfg["rotation_matrix_path"].get<std::string>();
    rotation_matrix_path_ =
      rebase_relative_to_model_file(rotation_matrix_path_, model_file_name);
  }
  if (nntr_cfg.contains("rope_cos_path")) {
    rope_cos_path_ = nntr_cfg["rope_cos_path"].get<std::string>();
    rope_cos_path_ =
      rebase_relative_to_model_file(rope_cos_path_, model_file_name);
  }
  if (nntr_cfg.contains("rope_sin_path")) {
    rope_sin_path_ = nntr_cfg["rope_sin_path"].get<std::string>();
    rope_sin_path_ =
      rebase_relative_to_model_file(rope_sin_path_, model_file_name);
  }
}

TensorInfo causallm::VJEPA2_QNN::get_input_info() {
  std::string graph_name = graphs_to_use[0];
  auto &[model_info, model, model_input] = models[graph_name];
  (void)model;
  (void)model_input;
  int idx =
    GraphParser::find_tensor_index(model_info.raw_inputs, "pixel_values_video");
  NNTR_THROW_IF(idx < 0, std::invalid_argument)
    << "pixel_values_video not found in graph inputs";
  return model_info.raw_inputs[idx];
}

TensorInfo causallm::VJEPA2_QNN::get_output_info() {
  std::string graph_name = graphs_to_use[0];
  auto &[model_info, model, model_input] = models[graph_name];
  (void)model;
  (void)model_input;
  return model_info.raw_outputs[0];
}

void causallm::VJEPA2_QNN::loadTensorFromFile(const std::string &path,
                                              void *dest, size_t expected_size,
                                              const std::string &name) {
  int fd = ::open(path.c_str(), O_RDONLY);
  NNTR_THROW_IF(fd == -1, std::invalid_argument)
    << "Cannot open " << name << " file: " << path;

  struct stat st {};
  if (::fstat(fd, &st) == -1) {
    ::close(fd);
    throw std::invalid_argument("Cannot fstat " + name + " file: " + path);
  }

  size_t file_size = static_cast<size_t>(st.st_size);
  if (file_size != expected_size) {
    ::close(fd);
    throw std::invalid_argument(name + " file size " +
                                std::to_string(file_size) + " != expected " +
                                std::to_string(expected_size));
  }

  void *ptr =
    ::mmap(nullptr, file_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  ::close(fd);
  NNTR_THROW_IF(ptr == MAP_FAILED, std::runtime_error)
    << "mmap failed for " << name << ": " << path;

  std::memcpy(dest, ptr, file_size);
  ::munmap(ptr, file_size);
  LOGD("%s loaded (%zu bytes)", name.c_str(), file_size);
}

void causallm::VJEPA2_QNN::loadRotationMatrix() {
  if (rotation_matrix_path_.empty())
    return;

  std::string graph_name = graphs_to_use[0];
  auto &[model_info, model, model_input] = models[graph_name];
  (void)model;
  int idx =
    GraphParser::find_tensor_index(model_info.raw_inputs, "rotation_matrix");
  NNTR_THROW_IF(idx < 0, std::invalid_argument)
    << "rotation_matrix not found in graph inputs";

  size_t expected_size =
    GraphParser::get_tensor_size(model_info.raw_inputs[idx]);
  loadTensorFromFile(rotation_matrix_path_,
                     std::get<uint8_t *>(model_input[idx]), expected_size,
                     "rotation_matrix");
  rotation_matrix_mmap_ptr_ = reinterpret_cast<void *>(1); // loaded sentinel
}

void causallm::VJEPA2_QNN::initialize() {
  Quick_Dot_AI_QNN::initialize();

  std::string graph_name = graphs_to_use[0];
  auto &[model_info, model, model_input] = models[graph_name];
  (void)model;

  // Cache input pointers by name
  int idx_rot =
    GraphParser::find_tensor_index(model_info.raw_inputs, "rotation_matrix");
  int idx_pix =
    GraphParser::find_tensor_index(model_info.raw_inputs, "pixel_values_video");
  int idx_cos =
    GraphParser::find_tensor_index(model_info.raw_inputs, "rope_cos");
  int idx_sin =
    GraphParser::find_tensor_index(model_info.raw_inputs, "rope_sin");

  // pixel_values_video is the only mandatory input for V-JEPA2 inference.
  NNTR_THROW_IF(idx_pix < 0, std::invalid_argument)
    << "pixel_values_video not found in graph inputs";

  if (idx_rot >= 0)
    rotation_matrix_input_ = std::get<uint8_t *>(model_input[idx_rot]);
  pixel_values_input_ = std::get<uint16_t *>(model_input[idx_pix]);
  if (idx_cos >= 0)
    rope_cos_input_ = std::get<uint16_t *>(model_input[idx_cos]);
  if (idx_sin >= 0)
    rope_sin_input_ = std::get<uint16_t *>(model_input[idx_sin]);

  // Load rotation matrix from external file if provided.
  if (idx_rot >= 0 && !rotation_matrix_path_.empty())
    loadRotationMatrix();

  // Detect missing RoPE-related inputs and populate with dummy zero data.
  // The serialized model may not supply these constant tensors.
  if (idx_rot >= 0 && rotation_matrix_mmap_ptr_ == nullptr) {
    size_t rot_byte_size =
      GraphParser::get_tensor_size(model_info.raw_inputs[idx_rot]);
    std::memset(rotation_matrix_input_, 0, rot_byte_size);
    LOGD("VJEPA2_QNN: WARNING rotation_matrix missing --- zero-filled dummy "
         "(%zu bytes)",
         rot_byte_size);
  }
  if (idx_cos >= 0) {
    size_t cos_byte_size =
      GraphParser::get_tensor_size(model_info.raw_inputs[idx_cos]);
    if (!rope_cos_path_.empty()) {
      loadTensorFromFile(rope_cos_path_, rope_cos_input_, cos_byte_size,
                         "rope_cos");
    } else {
      std::memset(rope_cos_input_, 0, cos_byte_size);
      LOGD("VJEPA2_QNN: WARNING rope_cos missing --- zero-filled dummy"
           " (%zu bytes)",
           cos_byte_size);
    }
  }
  if (idx_sin >= 0) {
    size_t sin_byte_size =
      GraphParser::get_tensor_size(model_info.raw_inputs[idx_sin]);
    if (!rope_sin_path_.empty()) {
      loadTensorFromFile(rope_sin_path_, rope_sin_input_, sin_byte_size,
                         "rope_sin");
    } else {
      std::memset(rope_sin_input_, 0, sin_byte_size);
      LOGD("VJEPA2_QNN: WARNING rope_sin missing --- zero-filled dummy"
           " (%zu bytes)",
           sin_byte_size);
    }
  }
}

void causallm::VJEPA2_QNN::run(const WSTR prompt, bool do_sample,
                               const WSTR system_prompt, const WSTR tail_prompt,
                               bool log_output) {
  // Unimplemented — vision-only encoder
}

void causallm::VJEPA2_QNN::set_quant_param(float scale, int offset) {
  llm_quant_param_given_ = true;
  llm_scale_ = scale;
  llm_offset_ = offset;
}

void causallm::VJEPA2_QNN::requantEmbedding(void *from, void *to,
                                            size_t length) {
  auto output_info = get_output_info();
  std::string encoderOutputDataType = output_info.data_type;
  std::string modelInputDataType = "QNN_DATATYPE_UFIXED_POINT_16";
  NNTR_THROW_IF(!llm_quant_param_given_, std::runtime_error)
    << "Please give LLM quant param!";

  double requant_scale = output_info.scale / llm_scale_;
  double requant_offset = requant_scale * output_info.offset - llm_offset_;

  LOGD("%zu : %s, %s, %f, %f", length, encoderOutputDataType.c_str(),
       modelInputDataType.c_str(), requant_scale, requant_offset);

  for (size_t i = 0; i < length; i++) {
    if (encoderOutputDataType == "QNN_DATATYPE_SFIXED_POINT_8" &&
        modelInputDataType == "QNN_DATATYPE_SFIXED_POINT_8") {
      static_cast<int8_t *>(to)[i] = static_cast<int8_t>(
        requant_scale * static_cast<int8_t *>(from)[i] + requant_offset);
    } else if (encoderOutputDataType == "QNN_DATATYPE_SFIXED_POINT_8" &&
               modelInputDataType == "QNN_DATATYPE_SFIXED_POINT_16") {
      static_cast<int16_t *>(to)[i] = static_cast<int16_t>(
        requant_scale * static_cast<int8_t *>(from)[i] + requant_offset);
    } else if (encoderOutputDataType == "QNN_DATATYPE_UFIXED_POINT_8" &&
               modelInputDataType == "QNN_DATATYPE_UFIXED_POINT_8") {
      static_cast<uint8_t *>(to)[i] = static_cast<uint8_t>(
        requant_scale * static_cast<uint8_t *>(from)[i] + requant_offset);
    } else if (encoderOutputDataType == "QNN_DATATYPE_UFIXED_POINT_8" &&
               modelInputDataType == "QNN_DATATYPE_UFIXED_POINT_16") {
      static_cast<uint16_t *>(to)[i] = static_cast<uint16_t>(
        requant_scale * static_cast<uint8_t *>(from)[i] + requant_offset);
    } else if (encoderOutputDataType == "QNN_DATATYPE_SFIXED_POINT_16" &&
               modelInputDataType == "QNN_DATATYPE_SFIXED_POINT_8") {
      static_cast<int8_t *>(to)[i] = static_cast<int8_t>(
        requant_scale * static_cast<int16_t *>(from)[i] + requant_offset);
    } else if (encoderOutputDataType == "QNN_DATATYPE_SFIXED_POINT_16" &&
               modelInputDataType == "QNN_DATATYPE_SFIXED_POINT_16") {
      static_cast<int16_t *>(to)[i] = static_cast<int16_t>(
        requant_scale * static_cast<int16_t *>(from)[i] + requant_offset);
    } else if (encoderOutputDataType == "QNN_DATATYPE_UFIXED_POINT_16" &&
               modelInputDataType == "QNN_DATATYPE_UFIXED_POINT_8") {
      static_cast<uint8_t *>(to)[i] = static_cast<uint8_t>(
        requant_scale * static_cast<uint16_t *>(from)[i] + requant_offset);
    } else if (encoderOutputDataType == "QNN_DATATYPE_UFIXED_POINT_16" &&
               modelInputDataType == "QNN_DATATYPE_UFIXED_POINT_16") {
      static_cast<uint16_t *>(to)[i] = static_cast<uint16_t>(
        requant_scale * static_cast<uint16_t *>(from)[i] + requant_offset);
    }
  }
}

void causallm::VJEPA2_QNN::preprocessToQnnInput(const float *raw_nchw, int B,
                                                int T, int C, int H, int W,
                                                uint16_t *qnn_hwc_dest,
                                                float scale, int offset) {

  int Dp = T / tubelet_size_;
  NNTR_THROW_IF(T % tubelet_size_ != 0, std::invalid_argument)
    << "preprocessToQnnInput: T (" << T
    << ") must be divisible by tubelet_size (" << tubelet_size_ << ")";

  for (int b = 0; b < B; ++b) {
    for (int d = 0; d < Dp; ++d) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          for (int tau = 0; tau < tubelet_size_; ++tau) {
            for (int c = 0; c < C; ++c) {
              size_t src_idx = static_cast<size_t>(b) * T * C * H * W +
                               (d * tubelet_size_ + tau) * C * H * W +
                               c * H * W + h * W + w;
              size_t dst_idx =
                static_cast<size_t>(b * Dp + d) * H * W * tubelet_size_ * C +
                h * W * tubelet_size_ * C + w * tubelet_size_ * C + tau * C + c;

              float val = raw_nchw[src_idx];
              if (std::isfinite(val)) {
                float quantized = val / scale - offset;
                qnn_hwc_dest[dst_idx] = static_cast<uint16_t>(
                  std::max(0.0f, std::min(65535.0f, quantized)));
              } else {
                qnn_hwc_dest[dst_idx] = 0;
              }
            }
          }
        }
      }
    }
  }
}

std::vector<uint8_t>
causallm::VJEPA2_QNN::frameToDepth(const float *raw_nchw, int B, int T, int C,
                                   int H, int W, int tubelet_size, float scale,
                                   int offset) {

  int Dp = T / tubelet_size;
  NNTR_THROW_IF(T % tubelet_size != 0, std::invalid_argument)
    << "frameToDepth: T must be divisible by tubelet_size";

  size_t num_elements = static_cast<size_t>(B * Dp * H * W * tubelet_size * C);
  std::vector<uint8_t> buffer(num_elements * sizeof(uint16_t));
  uint16_t *dest = reinterpret_cast<uint16_t *>(buffer.data());

  for (int b = 0; b < B; ++b) {
    for (int d = 0; d < Dp; ++d) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          for (int tau = 0; tau < tubelet_size; ++tau) {
            for (int c = 0; c < C; ++c) {
              size_t src_idx = static_cast<size_t>(b) * T * C * H * W +
                               (d * tubelet_size + tau) * C * H * W +
                               c * H * W + h * W + w;
              size_t dst_idx =
                static_cast<size_t>(b * Dp + d) * H * W * tubelet_size * C +
                h * W * tubelet_size * C + w * tubelet_size * C + tau * C + c;

              float val = raw_nchw[src_idx];
              if (std::isfinite(val)) {
                float quantized = val / scale - offset;
                dest[dst_idx] = static_cast<uint16_t>(
                  std::max(0.0f, std::min(65535.0f, quantized)));
              } else {
                dest[dst_idx] = 0;
              }
            }
          }
        }
      }
    }
  }

  return buffer;
}

// -------------------------------------------------------------------
// Fixed-shape fast path  (B=1, T=24, C=3, H=256, W=256, tubelet=2)
// -------------------------------------------------------------------
static inline void
preprocessToQnnInput_FixedShape_Helper(const float *__restrict raw,
                                       uint16_t *__restrict dst, float scale,
                                       int offset) {

  constexpr int B = 1, T = 24, C = 3, H = 256, W = 256, t = 2;
  constexpr int Dp = T / t;                 // 12
  constexpr int PLANE = H * W;              // 65536
  constexpr int SRC_STRIDE_TAU = C * PLANE; // 196608
  constexpr int DST_HWC = W * t * C;        // 1536
  constexpr int DST_SLICE = H * DST_HWC;    // 393216

  const float inv_scale = 1.0f / scale;
  const float offset_f = static_cast<float>(offset);

#ifndef NDEBUG
  // One-shot NaN/Inf guard before the hot loop.
  constexpr size_t TOTAL_FLOATS = static_cast<size_t>(B) * T * C * H * W;
  for (size_t i = 0; i < TOTAL_FLOATS; ++i) {
    if (__builtin_expect(!std::isfinite(raw[i]), 0)) {
      std::cerr << "NaN/Inf detected at index " << i
                << " in preprocessToQnnInput_FixedShape" << std::endl;
      std::abort();
    }
  }
#endif

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int d = 0; d < Dp; ++d) {
    const float *raw_d = raw + d * t * SRC_STRIDE_TAU;
    uint16_t *dst_d = dst + d * DST_SLICE;

    for (int h = 0; h < H; ++h) {
      int w = 0;

#ifdef __ARM_NEON
      const float32x4_t v_inv_scale = vdupq_n_f32(inv_scale);
      const float32x4_t v_offset = vdupq_n_f32(offset_f);
      const float32x4_t v_zero = vdupq_n_f32(0.0f);
      const float32x4_t v_max = vdupq_n_f32(65535.0f);

      // Vectorised quantise path: 4 pixels at once per (tau, c).
      // The destination is strided (t*C = 6), so we scalar-store the
      // 4 results, but the FMA / clamp / cvtt is fully vectorised.
      for (; w + 4 <= W; w += 4) {
        for (int tau = 0; tau < t; ++tau) {
          for (int c = 0; c < C; ++c) {
            const float *src =
              raw_d + tau * SRC_STRIDE_TAU + c * PLANE + h * W + w;
            float32x4_t v = vld1q_f32(src);

            float32x4_t q = vmulq_f32(v, v_inv_scale);
            q = vsubq_f32(q, v_offset);
            q = vmaxq_f32(q, v_zero);
            q = vminq_f32(q, v_max);
            uint32x4_t u32 = vcvtq_u32_f32(q);
            uint16x4_t u16 = vmovn_u32(u32);

            uint16_t tmp[4];
            vst1_u16(tmp, u16);

            uint16_t *dst_ptr = dst_d + h * DST_HWC + w * t * C + tau * C + c;
            dst_ptr[0 * t * C] = tmp[0];
            dst_ptr[1 * t * C] = tmp[1];
            dst_ptr[2 * t * C] = tmp[2];
            dst_ptr[3 * t * C] = tmp[3];
          }
        }
      }
#endif

      // Scalar tail + fallback.
      for (; w < W; ++w) {
        for (int tau = 0; tau < t; ++tau) {
          for (int c = 0; c < C; ++c) {
            size_t src_idx = static_cast<size_t>(d) * t * SRC_STRIDE_TAU +
                             tau * SRC_STRIDE_TAU + c * PLANE + h * W + w;
            size_t dst_idx = static_cast<size_t>(d) * DST_SLICE + h * DST_HWC +
                             w * t * C + tau * C + c;

            float val = raw[src_idx];
            float quantized = val * inv_scale - offset_f;
            dst[dst_idx] = static_cast<uint16_t>(
              std::max(0.0f, std::min(65535.0f, quantized)));
          }
        }
      }
    }
  }
}

void causallm::VJEPA2_QNN::preprocessToQnnInput_FixedShape(
  const float *__restrict raw, uint16_t *__restrict dst, float scale,
  int offset) {
  preprocessToQnnInput_FixedShape_Helper(raw, dst, scale, offset);
}

std::vector<uint8_t>
causallm::VJEPA2_QNN::frameToDepth_FixedShape(const float *raw, float scale,
                                              int offset) {
  constexpr size_t num_elements =
    static_cast<size_t>(1) * 12 * 256 * 256 * 2 * 3;
  std::vector<uint8_t> buffer(num_elements * sizeof(uint16_t));
  uint16_t *dest = reinterpret_cast<uint16_t *>(buffer.data());
  preprocessToQnnInput_FixedShape_Helper(raw, dest, scale, offset);
  return buffer;
}

causallm::multimodal_pointer
causallm::VJEPA2_QNN::run_image(const WSTR prompt, multimodal_pointer image,
                                int image_height, int image_width,
                                bool do_sample, const WSTR system_prompt,
                                const WSTR tail_prompt, bool log_output) {

  auto start_total = std::chrono::high_resolution_clock::now();

  std::string graph_name = graphs_to_use[0];
  auto &[model_info, model, model_input] = models[graph_name];

  auto input_info = get_input_info();
  auto output_info = get_output_info();

  int num_bytes_per_inference = GraphParser::get_tensor_size(input_info);
  int num_elements_per_inference = GraphParser::get_tensor_count(input_info);

  size_t raw_size = image.second;
  constexpr size_t PREPROCESSED_FLOAT_BYTES =
    12 * 256 * 256 * 6 * sizeof(float);
  constexpr size_t RAW_FLOAT_BYTES = 1 * 24 * 3 * 256 * 256 * sizeof(float);

  // For the standard V-JEPA2 graph both constants evaluate to the same value.
  // When they differ (different H/W or tubelet config) auto-sniff works as
  // intended.  Users can also force a path via vjepa2_input_format.
  bool use_preprocessed =
    (input_format_ == "preprocessed") ||
    (input_format_ == "auto" && raw_size == PREPROCESSED_FLOAT_BYTES);
  bool use_raw = (input_format_ == "raw") ||
                 (input_format_ == "auto" && raw_size == RAW_FLOAT_BYTES);

  if (!use_preprocessed && !use_raw) {
    NNTR_THROW_IF(true, std::invalid_argument)
      << "Unexpected video buffer size " << raw_size << ". Expected "
      << PREPROCESSED_FLOAT_BYTES << " (preprocessed) or " << RAW_FLOAT_BYTES
      << " (raw frames).";
  }

  int num_inference = image.second / (num_bytes_per_inference *
                                      (sizeof(float) / sizeof(uint16_t)));

  LOGD("image.second %zu, num_bytes_per_inference : %d, num_inference: %d",
       image.second, num_bytes_per_inference, num_inference);

  int out_embedding_size = output_info.dimensions[2]; // 768
  int output_bw = GraphParser::get_tensor_bit_width(output_info);
  int num_tokens = output_info.dimensions[1]; // 3072
  int total_embedding_size = num_tokens * out_embedding_size * output_bw;

  void *my_output = malloc(total_embedding_size * num_inference);

  for (int i = 0; i < num_inference; i++) {
    auto src = ((const float *)image.first) + i * num_elements_per_inference;
    if (use_preprocessed) {
      quantize_uint16_memcpy(const_cast<float *>(src), pixel_values_input_,
                             num_elements_per_inference, input_info.scale,
                             input_info.offset);
    } else {
      if (image_height == 256 && image_width == 256) {
        preprocessToQnnInput_FixedShape(src, pixel_values_input_,
                                        input_info.scale, input_info.offset);
      } else {
        preprocessToQnnInput(src,
                             /*B=*/1, /*T=*/24, /*C=*/3,
                             /*H=*/image_height, /*W=*/image_width,
                             pixel_values_input_, input_info.scale,
                             input_info.offset);
      }
    }
    auto qnn_output = model->inference(1, model_input)[0];
    void *vision_encoder_output = std::visit(
      [](auto *p) -> void * { return static_cast<void *>(p); }, qnn_output);

    int vision_encoder_output_size = GraphParser::get_tensor_size(output_info);

    void *dest =
      static_cast<uint8_t *>(my_output) + i * vision_encoder_output_size;
    if (llm_quant_param_given_) {
      requantEmbedding(vision_encoder_output, dest,
                       vision_encoder_output_size / output_bw);
    } else {
      std::memcpy(dest, vision_encoder_output, vision_encoder_output_size);
    }
  }

  auto end_total = std::chrono::high_resolution_clock::now();
  auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_total - start_total)
                    .count();

  performance_metrics.prefill_tokens = static_cast<unsigned int>(num_tokens);
  performance_metrics.prefill_duration_ms = static_cast<double>(total_ms);
  performance_metrics.generation_tokens = 0;
  performance_metrics.generation_duration_ms = 0.0;
  performance_metrics.total_duration_ms = static_cast<double>(total_ms);
  performance_metrics.peak_memory_kb = getPeakMemoryKb();
  has_run_ = true;

  std::cout << "run_image done!" << std::endl;
  return std::make_pair(my_output, total_embedding_size * num_inference);
}
