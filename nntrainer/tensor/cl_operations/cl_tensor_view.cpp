// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cl_tensor_view.cpp
 * @date    20 May 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   TensorBacking implementation. Single cl_mem + cached zero-copy
 *          image2d_from_buffer views.
 */
#include "cl_tensor_view.h"

#include <opencl_loader.h>

#include <stdexcept>
#include <string>

namespace nntrainer::tv {

TensorBacking::TensorBacking(cl_context ctx, cl_mem buf, Encoding enc,
                             Layout lay, size_t bytes, bool owned) :
  ctx_(ctx), buf_(buf), enc_(enc), lay_(lay), bytes_(bytes), owned_(owned) {}

TensorBacking::~TensorBacking() {
  for (auto &kv : image_cache_) {
    if (kv.second)
      opencl::clReleaseMemObject(kv.second);
  }
  if (owned_ && buf_)
    opencl::clReleaseMemObject(buf_);
}

cl_mem TensorBacking::imageView(const ViewSpec &spec) {
  if (spec.kind == ViewKind::BUFFER)
    return buf_;

  auto it = image_cache_.find(spec);
  if (it != image_cache_.end())
    return it->second;

  cl_image_format fmt{};
  fmt.image_channel_order = spec.image_channel_order;
  fmt.image_channel_data_type = spec.image_channel_type;

  cl_image_desc desc{};
  switch (spec.kind) {
  case ViewKind::IMAGE_2D:
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = spec.width;
    desc.image_height = spec.height;
    desc.image_row_pitch = spec.row_pitch_bytes;
    break;
  case ViewKind::IMAGE_1D:
    desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
    desc.image_width = spec.width;
    break;
  case ViewKind::IMAGE_3D:
    desc.image_type = CL_MEM_OBJECT_IMAGE3D;
    desc.image_width = spec.width;
    desc.image_height = spec.height;
    desc.image_depth = spec.depth;
    desc.image_row_pitch = spec.row_pitch_bytes;
    desc.image_slice_pitch = spec.slice_pitch_bytes;
    break;
  case ViewKind::BUFFER:
    break; // handled above
  }
  desc.buffer = buf_; // zero-copy view over the same cl_mem

  cl_int err = CL_SUCCESS;
  cl_mem img =
    opencl::clCreateImage(ctx_, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &err);
  if (err != CL_SUCCESS || img == nullptr) {
    throw std::runtime_error("TensorBacking::imageView clCreateImage failed: " +
                             std::to_string(err));
  }
  image_cache_.emplace(spec, img);
  return img;
}

// =============================================================================
// ViewSpec factories. These compute the right (width, height, row_pitch,
// channel_order, channel_type) for the named layout patterns so call sites
// don't repeat the bookkeeping.
// =============================================================================

ViewSpec make_image2d_rgba_uint32(size_t byte_width, size_t byte_height) {
  ViewSpec s;
  s.kind = ViewKind::IMAGE_2D;
  s.image_channel_order = CL_RGBA;
  s.image_channel_type = CL_UNSIGNED_INT32;
  s.width = byte_width / 16; // 16 bytes per RGBA UINT32 texel
  s.height = byte_height;
  s.row_pitch_bytes = byte_width;
  return s;
}

// =============================================================================
// Device specialization (paper §3.4): query image2d storage capabilities so
// packed image views can be validated up front instead of failing opaquely.
// =============================================================================
DeviceImageCaps queryDeviceImageCaps(cl_device_id device) {
  DeviceImageCaps caps;
  if (device == nullptr)
    return caps;

  cl_bool img_support = CL_FALSE;
  if (opencl::clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT,
                              sizeof(img_support), &img_support,
                              nullptr) != CL_SUCCESS)
    return caps;
  caps.image_support = (img_support == CL_TRUE);
  if (!caps.image_support) {
    caps.queried = true;
    return caps;
  }

  // These three may legitimately be absent on some stacks; treat a failed
  // query as "unknown" (0) rather than aborting — image2dViewFits ignores a
  // zero bound.
  opencl::clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t),
                          &caps.max_width, nullptr);
  opencl::clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t),
                          &caps.max_height, nullptr);
  opencl::clGetDeviceInfo(device, CL_DEVICE_IMAGE_PITCH_ALIGNMENT,
                          sizeof(cl_uint), &caps.pitch_align, nullptr);
  opencl::clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),
                          &caps.max_work_group_size, nullptr);
  caps.queried = true;
  return caps;
}

bool image2dViewFits(const ViewSpec &spec, const DeviceImageCaps &caps) {
  if (spec.kind == ViewKind::BUFFER)
    return true;
  if (!caps.queried)
    return true; // never queried → assume OK (legacy behavior preserved)
  if (!caps.image_support)
    return false;
  if (caps.max_width && spec.width > caps.max_width)
    return false;
  if (caps.max_height && spec.height > caps.max_height)
    return false;
  return true;
}

bool select2dLws(size_t gws_x, size_t gws_y, size_t pref_x, size_t pref_y,
                 const DeviceImageCaps &caps, size_t *out_x, size_t *out_y) {
  *out_x = 0;
  *out_y = 0;
  if (!pref_x || !pref_y || !gws_x || !gws_y)
    return false;
  // Device max work-group size: 0 = unknown → don't cap (legacy behavior).
  const size_t max_wg =
    caps.max_work_group_size ? caps.max_work_group_size : (size_t)-1;

  size_t lx = pref_x, ly = pref_y;
  // Shrink the preferred LWS until lx*ly fits the device's max work-group size,
  // halving the larger axis first. (Halving keeps power-of-two divisibility,
  // which is what the v8c sweet spot 4x16 uses.)
  while (lx * ly > max_wg) {
    if (ly >= lx && ly > 1)
      ly >>= 1;
    else if (lx > 1)
      lx >>= 1;
    else
      break; // 1x1 still over max_wg (pathological) → give up
  }
  // The kernel has no cross-WI sync, so any LWS dividing the global size on
  // both axes is valid. If the (shrunk) preferred LWS does not divide gws,
  // report failure so the caller passes NULL (driver picks).
  if (lx * ly > max_wg)
    return false;
  if (gws_x % lx != 0 || gws_y % ly != 0)
    return false;
  *out_x = lx;
  *out_y = ly;
  return true;
}

} // namespace nntrainer::tv
