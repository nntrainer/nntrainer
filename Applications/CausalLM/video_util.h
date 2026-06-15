// SPDX-License-Identifier: Apache-2.0
/**
 * @file   video_util.h
 * @date   10 June 2026
 * @brief  Shared video-loading and preprocessing utilities for CausalLM apps.
 *
 * Provides loadAndPreprocessVideo() for loading a video from a directory of
 * image frames (JPEG/PNG/BMP) and producing a [C,T,H,W] float32 buffer
 * suitable for VJEPA2ViT.
 *
 * IMPORTANT: Define STB_IMAGE_IMPLEMENTATION exactly once in a .cpp file
 * before including stb_image.inc; do NOT define it here.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __CAUSALLM_VIDEO_UTIL_H__
#define __CAUSALLM_VIDEO_UTIL_H__

#include "image_util.h"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#error "No filesystem header available"
#endif

namespace causallm {

/**
 * @brief Load a video from a directory of image frames, resize to target
 *        dimensions, convert to 3-channel [C,T,H,W] float, and normalize.
 *
 * Expects the directory to contain image files (JPEG/PNG/BMP) named with
 * zero-padded sequential numbers (e.g., 00001.jpg, 00002.jpg, ...) or
 * any names that sort lexicographically into the correct temporal order.
 * Exactly @p num_frames frames will be loaded; if the directory contains
 * fewer frames, an error is raised. If it contains more, only the first
 * @p num_frames are used (uniformly sampled).
 *
 * Output layout: [C, T, H, W] as a flat float vector of size
 * 3 * num_frames * target_height * target_width.
 *
 * @param video_dir      Path to directory containing image frames.
 * @param target_width   Output width in pixels per frame.
 * @param target_height  Output height in pixels per frame.
 * @param num_frames     Number of frames to load.
 * @param normalize      If true, apply (val/255 - 0.5) / 0.5 per channel;
 *                       otherwise output raw [0,255] floats.
 *                       For V-JEPA 2.1, use the NormalizeMode overload instead.
 * @return [C,T,H,W] float buffer: [3 * num_frames * target_height *
 *         target_width].
 * @throws std::runtime_error if the directory cannot be read or too few
 *         frames are available.
 */
inline std::vector<float>
loadAndPreprocessVideo(const std::string &video_dir, int target_width,
                       int target_height, unsigned int num_frames,
                       bool normalize = true) {
  // Collect and sort image file paths
  std::vector<fs::path> frame_paths;
  for (const auto &entry : fs::directory_iterator(video_dir)) {
    if (!entry.is_regular_file())
      continue;
    std::string ext = entry.path().extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" ||
        ext == ".webp") {
      frame_paths.push_back(entry.path());
    }
  }

  if (frame_paths.empty()) {
    throw std::runtime_error("No image frames found in directory: " +
                             video_dir);
  }

  // Sort lexicographically to get temporal order
  std::sort(frame_paths.begin(), frame_paths.end());

  if (frame_paths.size() < num_frames) {
    throw std::runtime_error(
      "Not enough frames in directory: found " +
      std::to_string(frame_paths.size()) + ", need " +
      std::to_string(num_frames) + " (directory: " + video_dir + ")");
  }

  // Uniformly sample num_frames indices from the available frames
  std::vector<size_t> selected_indices(num_frames);
  if (num_frames == 1) {
    selected_indices[0] = 0;
  } else {
    for (unsigned int i = 0; i < num_frames; ++i) {
      selected_indices[i] = static_cast<size_t>(
        (static_cast<double>(i) * (frame_paths.size() - 1)) /
        (num_frames - 1));
    }
  }

  // Allocate output: [C=3, T=num_frames, H=target_height, W=target_width]
  const size_t frame_size =
    static_cast<size_t>(3) * target_height * target_width;
  std::vector<float> video(3 * num_frames * target_height * target_width);

  for (unsigned int t = 0; t < num_frames; ++t) {
    const std::string frame_path =
      frame_paths[selected_indices[t]].string();
    std::vector<float> frame_data = loadAndPreprocessImage(
      frame_path, target_width, target_height, normalize);

    // frame_data is [C=3, H, W] CHW, copy into [C, T, H, W] at offset t
    const size_t dst_offset = static_cast<size_t>(t) * target_height *
                              target_width;
    for (int c = 0; c < 3; ++c) {
      const size_t src_offset =
        static_cast<size_t>(c) * target_height * target_width;
      const size_t channel_offset =
        static_cast<size_t>(c) * num_frames * target_height * target_width;
      std::copy_n(frame_data.data() + src_offset,
                  target_height * target_width,
                  video.data() + channel_offset + dst_offset);
    }
  }

  return video;
}

/**
 * @brief Load a video from a directory of image frames with configurable
 *        normalization mode (supports ImageNet mean/std for V-JEPA 2.1).
 */
inline std::vector<float>
loadAndPreprocessVideo(const std::string &video_dir, int target_width,
                       int target_height, unsigned int num_frames,
                       NormalizeMode norm_mode) {
  // Collect and sort image file paths
  std::vector<fs::path> frame_paths;
  for (const auto &entry : fs::directory_iterator(video_dir)) {
    if (!entry.is_regular_file())
      continue;
    std::string ext = entry.path().extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" ||
        ext == ".webp") {
      frame_paths.push_back(entry.path());
    }
  }

  if (frame_paths.empty()) {
    throw std::runtime_error("No image frames found in directory: " +
                             video_dir);
  }

  std::sort(frame_paths.begin(), frame_paths.end());

  if (frame_paths.size() < num_frames) {
    throw std::runtime_error(
      "Not enough frames in directory: found " +
      std::to_string(frame_paths.size()) + ", need " +
      std::to_string(num_frames) + " (directory: " + video_dir + ")");
  }

  std::vector<size_t> selected_indices(num_frames);
  if (num_frames == 1) {
    selected_indices[0] = 0;
  } else {
    for (unsigned int i = 0; i < num_frames; ++i) {
      selected_indices[i] = static_cast<size_t>(
        (static_cast<double>(i) * (frame_paths.size() - 1)) /
        (num_frames - 1));
    }
  }

  const size_t frame_size =
    static_cast<size_t>(3) * target_height * target_width;
  std::vector<float> video(3 * num_frames * target_height * target_width);

  for (unsigned int t = 0; t < num_frames; ++t) {
    const std::string frame_path =
      frame_paths[selected_indices[t]].string();
    std::vector<float> frame_data = loadAndPreprocessImage(
      frame_path, target_width, target_height, norm_mode);

    const size_t dst_offset = static_cast<size_t>(t) * target_height *
                              target_width;
    for (int c = 0; c < 3; ++c) {
      const size_t src_offset =
        static_cast<size_t>(c) * target_height * target_width;
      const size_t channel_offset =
        static_cast<size_t>(c) * num_frames * target_height * target_width;
      std::copy_n(frame_data.data() + src_offset,
                  target_height * target_width,
                  video.data() + channel_offset + dst_offset);
    }
  }

  return video;
}

/**
 * @brief Load a video from a raw float32 binary file in [C,T,H,W] layout.
 *
 * This is a direct binary read — no resizing or normalization is applied.
 * The file must contain exactly C * T * H * W float32 values.
 *
 * @param bin_path    Path to the raw float32 binary file.
 * @param channels    Number of channels (typically 3).
 * @param num_frames  Number of frames.
 * @param height      Frame height.
 * @param width       Frame width.
 * @return [C,T,H,W] float buffer.
 * @throws std::runtime_error if the file cannot be read or size mismatches.
 */
inline std::vector<float>
loadVideoFromBin(const std::string &bin_path, unsigned int channels,
                 unsigned int num_frames, unsigned int height,
                 unsigned int width) {
  const size_t expected =
    static_cast<size_t>(channels) * num_frames * height * width;

  std::ifstream f(bin_path, std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error("Failed to open video binary file: " + bin_path);
  }

  std::vector<float> video(expected);
  f.read(reinterpret_cast<char *>(video.data()), expected * sizeof(float));
  if (static_cast<size_t>(f.gcount()) != expected * sizeof(float)) {
    throw std::runtime_error(
      "Video binary file size mismatch: expected " +
      std::to_string(expected) + " float32 values ([C,T,H,W] = [" +
      std::to_string(channels) + "," + std::to_string(num_frames) + "," +
      std::to_string(height) + "," + std::to_string(width) + "]) in " +
      bin_path);
  }

  return video;
}

} // namespace causallm

#endif /* __CAUSALLM_VIDEO_UTIL_H__ */
