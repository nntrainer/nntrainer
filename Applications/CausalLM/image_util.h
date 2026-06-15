// SPDX-License-Identifier: Apache-2.0
/**
 * @file   image_util.h
 * @date   04 June 2026
 * @brief  Shared image-loading and preprocessing utilities for CausalLM apps.
 *
 * Provides resizeImage() and loadAndPreprocessImage() for use by
 * TimmViTTransformer and Lfm2VlVisionTransformer without duplicating code.
 *
 * IMPORTANT: Define STB_IMAGE_IMPLEMENTATION exactly once in a .cpp file
 * before including stb_image.inc; do NOT define it here.
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __CAUSALLM_IMAGE_UTIL_H__
#define __CAUSALLM_IMAGE_UTIL_H__

#include "stb_image.inc"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace causallm {

/** Normalization mode for loadAndPreprocessImage. */
enum NormalizeMode {
  NORM_NONE = 0,     ///< No normalization, raw [0, 255]
  NORM_NEG1_TO_1 = 1, ///< (val/255 - 0.5) / 0.5 -> [-1, 1] (SigLIP2 style)
  NORM_IMAGENET = 2,  ///< (val/255 - mean[c]) / std[c] (ImageNet, V-JEPA 2.1)
};

/**
 * @brief Resize an interleaved (HWC) image buffer with bilinear interpolation.
 *
 * @param src      Source pixel buffer (H*W*channels bytes).
 * @param src_w    Source width in pixels.
 * @param src_h    Source height in pixels.
 * @param channels Number of channels (1, 3, or 4).
 * @param dst_w    Target width in pixels.
 * @param dst_h    Target height in pixels.
 * @return Resized pixel buffer (dst_h*dst_w*channels bytes).
 */
inline std::vector<unsigned char> resizeImage(const unsigned char *src,
                                              int src_w, int src_h,
                                              int channels, int dst_w,
                                              int dst_h) {
  std::vector<unsigned char> dst(dst_w * dst_h * channels);
  float x_ratio = static_cast<float>(src_w) / static_cast<float>(dst_w);
  float y_ratio = static_cast<float>(src_h) / static_cast<float>(dst_h);

  for (int y = 0; y < dst_h; ++y) {
    for (int x = 0; x < dst_w; ++x) {
      float px = x * x_ratio;
      float py = y * y_ratio;
      int x0 = static_cast<int>(std::floor(px));
      int y0 = static_cast<int>(std::floor(py));
      int x1 = std::min(x0 + 1, src_w - 1);
      int y1 = std::min(y0 + 1, src_h - 1);
      float fx = px - x0;
      float fy = py - y0;

      for (int c = 0; c < channels; ++c) {
        float v00 = src[(y0 * src_w + x0) * channels + c];
        float v10 = src[(y0 * src_w + x1) * channels + c];
        float v01 = src[(y1 * src_w + x0) * channels + c];
        float v11 = src[(y1 * src_w + x1) * channels + c];
        float v0 = v00 * (1.0f - fx) + v10 * fx;
        float v1 = v01 * (1.0f - fx) + v11 * fx;
        dst[(y * dst_w + x) * channels + c] =
          static_cast<unsigned char>(std::round(v0 * (1.0f - fy) + v1 * fy));
      }
    }
  }

  return dst;
}

/**
 * @brief Load an image file, resize to target dimensions, convert to 3-channel
 *        CHW float, and optionally normalize with mean=std=0.5 (SigLIP2 style).
 *
 * Supports JPEG, PNG, BMP, and any format handled by stb_image.
 * Output layout: channel-major (C,H,W) as a flat float vector of size
 * 3 * target_height * target_width.
 *
 * @param filepath      Path to the image file (jpg, png, bmp, ??.
 * @param target_width  Output width in pixels.
 * @param target_height Output height in pixels.
 * @param normalize     If true, apply (val/255 - 0.5) / 0.5 per channel;
 *                      otherwise output raw [0,255] floats.
 * @return CHW float buffer: [3 * target_height * target_width].
 * @throws std::runtime_error if the file cannot be loaded.
 */
inline std::vector<float> loadAndPreprocessImage(const std::string &filepath,
                                                 int target_width,
                                                 int target_height,
                                                 bool normalize) {
  int width, height, channels;
  unsigned char *image =
    stbi_load(filepath.c_str(), &width, &height, &channels, STBI_default);
  if (!image) {
    throw std::runtime_error("Failed to load image: " + filepath);
  }

  unsigned char *data = image;
  std::vector<unsigned char> resized_data;
  if (width != target_width || height != target_height) {
    resized_data =
      resizeImage(image, width, height, channels, target_width, target_height);
    data = resized_data.data();
  }

  std::vector<unsigned char> rgb_data;
  unsigned char *rgb = data;
  if (channels == 1) {
    rgb_data.resize(target_width * target_height * 3);
    for (int i = 0; i < target_width * target_height; ++i) {
      rgb_data[i * 3] = data[i];
      rgb_data[i * 3 + 1] = data[i];
      rgb_data[i * 3 + 2] = data[i];
    }
    rgb = rgb_data.data();
  } else if (channels == 4) {
    rgb_data.resize(target_width * target_height * 3);
    for (int i = 0; i < target_width * target_height; ++i) {
      rgb_data[i * 3] = data[i * 4];
      rgb_data[i * 3 + 1] = data[i * 4 + 1];
      rgb_data[i * 3 + 2] = data[i * 4 + 2];
    }
    rgb = rgb_data.data();
  } else if (channels != 3) {
    stbi_image_free(image);
    throw std::runtime_error("Unsupported number of channels: " +
                             std::to_string(channels));
  }

  std::vector<float> output(3 * target_height * target_width);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < target_height; ++y) {
      for (int x = 0; x < target_width; ++x) {
        unsigned char val = rgb[y * target_width * 3 + x * 3 + c];
        float pixel =
          normalize ? (val / 255.0f - 0.5f) / 0.5f : static_cast<float>(val);
        output[c * target_height * target_width + y * target_width + x] = pixel;
      }
    }
  }

  stbi_image_free(image);
  return output;
}

/**
 * @brief Load an image file, resize, convert to CHW float with configurable
 *        normalization.
 *
 * @param filepath      Path to the image file.
 * @param target_width  Output width in pixels.
 * @param target_height Output height in pixels.
 * @param norm_mode     Normalization mode (NORM_NONE, NORM_NEG1_TO_1,
 *                      NORM_IMAGENET).
 * @return CHW float buffer: [3 * target_height * target_width].
 */
inline std::vector<float> loadAndPreprocessImage(const std::string &filepath,
                                                 int target_width,
                                                 int target_height,
                                                 NormalizeMode norm_mode) {
  // ImageNet per-channel mean and std (RGB order)
  static constexpr float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f};
  static constexpr float IMAGENET_STD[3]  = {0.229f, 0.224f, 0.225f};

  int width, height, channels;
  unsigned char *image =
    stbi_load(filepath.c_str(), &width, &height, &channels, STBI_default);
  if (!image) {
    throw std::runtime_error("Failed to load image: " + filepath);
  }

  unsigned char *data = image;
  std::vector<unsigned char> resized_data;
  if (width != target_width || height != target_height) {
    resized_data =
      resizeImage(image, width, height, channels, target_width, target_height);
    data = resized_data.data();
  }

  std::vector<unsigned char> rgb_data;
  unsigned char *rgb = data;
  if (channels == 1) {
    rgb_data.resize(target_width * target_height * 3);
    for (int i = 0; i < target_width * target_height; ++i) {
      rgb_data[i * 3] = data[i];
      rgb_data[i * 3 + 1] = data[i];
      rgb_data[i * 3 + 2] = data[i];
    }
    rgb = rgb_data.data();
  } else if (channels == 4) {
    rgb_data.resize(target_width * target_height * 3);
    for (int i = 0; i < target_width * target_height; ++i) {
      rgb_data[i * 3] = data[i * 4];
      rgb_data[i * 3 + 1] = data[i * 4 + 1];
      rgb_data[i * 3 + 2] = data[i * 4 + 2];
    }
    rgb = rgb_data.data();
  } else if (channels != 3) {
    stbi_image_free(image);
    throw std::runtime_error("Unsupported number of channels: " +
                             std::to_string(channels));
  }

  std::vector<float> output(3 * target_height * target_width);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < target_height; ++y) {
      for (int x = 0; x < target_width; ++x) {
        unsigned char val = rgb[y * target_width * 3 + x * 3 + c];
        float pixel;
        switch (norm_mode) {
        case NORM_NEG1_TO_1:
          pixel = (val / 255.0f - 0.5f) / 0.5f;
          break;
        case NORM_IMAGENET:
          pixel = (val / 255.0f - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
          break;
        case NORM_NONE:
        default:
          pixel = static_cast<float>(val);
          break;
        }
        output[c * target_height * target_width + y * target_width + x] = pixel;
      }
    }
  }

  stbi_image_free(image);
  return output;
}

} // namespace causallm

#endif /* __CAUSALLM_IMAGE_UTIL_H__ */
