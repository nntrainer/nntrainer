// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   yolo_model.h
 * @brief  Reusable YOLOv11 detector pieces: layer registration, letterboxing,
 *         and post-processing (DFL decode + dist2bbox + NMS).
 *
 * The graph builders live in yolov11_graph.h; everything needed to actually
 * *run* a detection and turn its three raw scale outputs into boxes lives
 * here. These used to sit inside the yolov11_infer main.cpp, which made them
 * unusable from any other binary -- an embedder such as the Quick.AI ScreenAI
 * VTT pipeline (YOLO detect -> crop -> caption, all in one process) needs the
 * same decode/NMS as the standalone tool, and duplicating it would let the two
 * drift apart.
 *
 * @see    yolov11_graph.h
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __YOLO_MODEL_H__
#define __YOLO_MODEL_H__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "c2psa_layer.h"
#include "yolov11_graph.h"
#include <app_context.h>
#include <layer.h>

namespace yolov11 {

// ===== Post-processing (DFL decode + dist2bbox + NMS) =====
// ---------------------------------------------------------------------------
// Anchor generation (ultralytics tal.py: make_anchors)
//
// For each scale with grid (H, W) and stride s:
//   sx = [0.5, 1.5, ..., W-0.5]
//   sy = [0.5, 1.5, ..., H-0.5]
//   row-major meshgrid: y outer, x inner → [H*W, 2] (x,y) in grid units
// Concat P3 → P4 → P5.
// ---------------------------------------------------------------------------
struct ScaleInfo {
  int H;
  int W;
  float stride;
};

/**
 * @brief Generate anchor points and stride tensor.
 * @param scales  Vector of (H, W, stride) for each detection scale.
 * @param[out] anchors     [num_anchors, 2] (x, y) in grid units.
 * @param[out] strides_out [num_anchors] stride per anchor.
 */
inline void makeAnchors(const std::vector<ScaleInfo> &scales,
                        std::vector<float> &anchors,
                        std::vector<float> &strides_out) {
  size_t total = 0;
  for (const auto &s : scales) {
    total += static_cast<size_t>(s.H) * s.W;
  }
  anchors.resize(total * 2);
  strides_out.resize(total);

  size_t off = 0;
  for (const auto &s : scales) {
    for (int iy = 0; iy < s.H; ++iy) {
      for (int ix = 0; ix < s.W; ++ix) {
        anchors[off * 2 + 0] = static_cast<float>(ix) + 0.5f;
        anchors[off * 2 + 1] = static_cast<float>(iy) + 0.5f;
        strides_out[off] = s.stride;
        ++off;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// DFL (Distribution Focal Loss) decode
// ---------------------------------------------------------------------------
inline void dfl(const float *raw_box, int reg_max, int N,
                std::vector<float> &dist) {
  dist.resize(4 * N);
  for (int c = 0; c < 4; ++c) {
    for (int a = 0; a < N; ++a) {
      float logits[16];
      for (int k = 0; k < reg_max; ++k) {
        logits[k] = raw_box[(c * reg_max + k) * N + a];
      }
      float max_logit = *std::max_element(logits, logits + reg_max);
      float sum = 0.0f;
      float exp_v[16];
      for (int k = 0; k < reg_max; ++k) {
        exp_v[k] = std::exp(logits[k] - max_logit);
        sum += exp_v[k];
      }
      float val = 0.0f;
      for (int k = 0; k < reg_max; ++k) {
        val += (exp_v[k] / sum) * static_cast<float>(k);
      }
      dist[c * N + a] = val;
    }
  }
}

// ---------------------------------------------------------------------------
// dist2bbox + stride multiply → XYWH pixels at 832-scale
// ---------------------------------------------------------------------------
inline void dist2bbox(const std::vector<float> &dist, int N,
                      const std::vector<float> &anchors,
                      const std::vector<float> &strides,
                      std::vector<float> &decoded_box) {
  const float *lt_x = dist.data() + 0 * N;
  const float *lt_y = dist.data() + 1 * N;
  const float *rb_x = dist.data() + 2 * N;
  const float *rb_y = dist.data() + 3 * N;

  decoded_box.resize(4 * N);
  float *cx = decoded_box.data() + 0 * N;
  float *cy = decoded_box.data() + 1 * N;
  float *w = decoded_box.data() + 2 * N;
  float *h = decoded_box.data() + 3 * N;

  for (int a = 0; a < N; ++a) {
    float ax = anchors[a * 2 + 0];
    float ay = anchors[a * 2 + 1];
    float s = strides[a];

    float x1 = ax - lt_x[a];
    float y1 = ay - lt_y[a];
    float x2 = ax + rb_x[a];
    float y2 = ay + rb_y[a];

    cx[a] = (x1 + x2) * 0.5f * s;
    cy[a] = (y1 + y2) * 0.5f * s;
    w[a] = (x2 - x1) * s;
    h[a] = (y2 - y1) * s;
  }
}

// ---------------------------------------------------------------------------
// Full post-processing pipeline for one scale
// ---------------------------------------------------------------------------
inline void decodeOneScale(const float *raw, int H, int W, float stride,
                           const std::vector<float> &anchors,
                           const std::vector<float> &strides_vec,
                           int anchor_off, int N_total, int box_ch, int nc,
                           std::vector<float> &decoded) {
  const int N = H * W;
  const int reg_max = box_ch / 4;

  const float *raw_box = raw;
  const float *raw_cls = raw + box_ch * N;

  std::vector<float> dist;
  dfl(raw_box, reg_max, N, dist);

  std::vector<float> scale_anchors(N * 2);
  std::vector<float> scale_strides(N);
  for (int a = 0; a < N; ++a) {
    scale_anchors[a * 2 + 0] = anchors[(anchor_off + a) * 2 + 0];
    scale_anchors[a * 2 + 1] = anchors[(anchor_off + a) * 2 + 1];
    scale_strides[a] = strides_vec[anchor_off + a];
  }

  std::vector<float> decoded_box;
  dist2bbox(dist, N, scale_anchors, scale_strides, decoded_box);

  for (int c = 0; c < 4; ++c) {
    for (int a = 0; a < N; ++a) {
      decoded[c * N_total + anchor_off + a] = decoded_box[c * N + a];
    }
  }
  // Per-class sigmoid scores: decoded[(4 + cls) * N_total + anchor_off + a]
  for (int cls = 0; cls < nc; ++cls) {
    for (int a = 0; a < N; ++a) {
      decoded[(4 + cls) * N_total + anchor_off + a] =
        1.0f / (1.0f + std::exp(-raw_cls[cls * N + a]));
    }
  }
}

// ---------------------------------------------------------------------------
// NMS
// ---------------------------------------------------------------------------
struct Detection {
  float x1, y1, x2, y2, conf;
  int cls;
};

inline float iou(const Detection &a, const Detection &b) {
  float ix1 = std::max(a.x1, b.x1);
  float iy1 = std::max(a.y1, b.y1);
  float ix2 = std::min(a.x2, b.x2);
  float iy2 = std::min(a.y2, b.y2);
  float inter_w = std::max(0.0f, ix2 - ix1);
  float inter_h = std::max(0.0f, iy2 - iy1);
  float inter = inter_w * inter_h;
  if (inter == 0.0f)
    return 0.0f;
  float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
  float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (area_a + area_b - inter);
}

/**
 * @brief Run per-class NMS on decoded predictions.
 * @param decoded    [(4+nc), N_total] (cx, cy, w, h, cls0_score, cls1_score,
 * ...)
 * @param N_total    total anchors
 * @param nc         number of classes
 * @param conf_thres confidence threshold
 * @param iou_thres  IoU threshold
 * @param max_det    max detections to keep
 * @return sorted (descending conf) list of Detection
 */
inline std::vector<Detection> nms(const std::vector<float> &decoded,
                                  int N_total, int nc, float conf_thres,
                                  float iou_thres, int max_det) {
  const float *cx_ptr = decoded.data() + 0 * N_total;
  const float *cy_ptr = decoded.data() + 1 * N_total;
  const float *w_ptr = decoded.data() + 2 * N_total;
  const float *h_ptr = decoded.data() + 3 * N_total;

  // Build candidate list: for each anchor, find the best class score;
  // if it exceeds conf_thres, emit a Detection with that class.
  std::vector<Detection> candidates;
  candidates.reserve(512);
  for (int a = 0; a < N_total; ++a) {
    float best_score = 0.0f;
    int best_cls = 0;
    for (int cls = 0; cls < nc; ++cls) {
      float score = decoded[(4 + cls) * N_total + a];
      if (score > best_score) {
        best_score = score;
        best_cls = cls;
      }
    }
    if (best_score <= conf_thres)
      continue;
    float cx = cx_ptr[a];
    float cy = cy_ptr[a];
    float bw = w_ptr[a];
    float bh = h_ptr[a];
    float x1 = cx - bw * 0.5f;
    float y1 = cy - bh * 0.5f;
    float x2 = cx + bw * 0.5f;
    float y2 = cy + bh * 0.5f;
    candidates.push_back({x1, y1, x2, y2, best_score, best_cls});
  }

  // Sort descending by conf
  std::sort(
    candidates.begin(), candidates.end(),
    [](const Detection &a, const Detection &b) { return a.conf > b.conf; });

  // Greedy per-class NMS: only suppress within the same class
  std::vector<bool> suppressed(candidates.size(), false);
  std::vector<Detection> result;
  result.reserve(max_det);

  for (size_t i = 0; i < candidates.size(); ++i) {
    if (suppressed[i])
      continue;
    result.push_back(candidates[i]);
    if (static_cast<int>(result.size()) >= max_det)
      break;
    for (size_t j = i + 1; j < candidates.size(); ++j) {
      if (suppressed[j] || candidates[j].cls != candidates[i].cls)
        continue;
      if (iou(candidates[i], candidates[j]) > iou_thres) {
        suppressed[j] = true;
      }
    }
  }

  return result;
}

/** @brief Register the YOLOv11 custom layers with the global AppContext.
 *  Only C2PSA is custom; the Detect head uses standard nntrainer layers.
 *  Idempotent in practice: an embedder may build several detectors. */
inline void registerCustomLayers() {
  auto &app_ctx = nntrainer::AppContext::Global();
  app_ctx.registerFactory(nntrainer::createLayer<yolov11::PSAAttentionLayer>);
}

/**
 * @brief Letterbox an in-memory RGB buffer to [1,3,size,size] float32 CHW.
 *
 * Same transform as the file-based path in yolov11_infer (ultralytics
 * letterbox: aspect-preserving bilinear resize with half-pixel centers, pad
 * the remainder with @p pad, scale to [0,1]) but operating on pixels the
 * caller already holds. @p scale / @p pad_l / @p pad_t are returned so
 * detections can be mapped back to original-image coordinates.
 *
 * @param[in]  rgb    interleaved RGB8 source, @p W * @p H * 3 bytes
 * @param[in]  W, H   source dimensions
 * @param[in]  size   square letterbox side (YOLOv11 default 832)
 * @param[out] scale  resize ratio applied to the source
 * @param[out] pad_l  left padding in letterboxed pixels
 * @param[out] pad_t  top padding in letterboxed pixels
 * @param[in]  pad    fill value for the padded border (ultralytics uses 114)
 * @return [1,3,size,size] CHW float32 in [0,1]
 */
inline std::vector<float> letterboxFromRGB(const unsigned char *rgb, int W,
                                           int H, int size, float &scale,
                                           int &pad_l, int &pad_t,
                                           float pad = 114.0f) {
  if (!rgb || W <= 0 || H <= 0)
    throw std::invalid_argument("letterboxFromRGB: empty source image");

  const float r = std::min((float)size / H, (float)size / W);
  const int nh = (int)std::round(H * r), nw = (int)std::round(W * r);
  scale = r;
  pad_t = (size - nh) / 2;
  pad_l = (size - nw) / 2;

  std::vector<float> out(3UL * size * size, pad / 255.0f);
  for (int oy = 0; oy < nh; ++oy) {
    float sy = (oy + 0.5f) * H / nh - 0.5f; // cv2 INTER_LINEAR mapping
    int y0 = (int)std::floor(sy);
    float fy = sy - y0;
    int y0c = std::min(std::max(y0, 0), H - 1), y1c = std::min(y0 + 1, H - 1);
    for (int ox = 0; ox < nw; ++ox) {
      float sx = (ox + 0.5f) * W / nw - 0.5f;
      int x0 = (int)std::floor(sx);
      float fx = sx - x0;
      int x0c = std::min(std::max(x0, 0), W - 1), x1c = std::min(x0 + 1, W - 1);
      for (int ch = 0; ch < 3; ++ch) {
        auto px = [&](int yy, int xx) {
          return (float)rgb[((size_t)yy * W + xx) * 3 + ch];
        };
        float v = px(y0c, x0c) * (1 - fx) * (1 - fy) +
                  px(y0c, x1c) * fx * (1 - fy) + px(y1c, x0c) * (1 - fx) * fy +
                  px(y1c, x1c) * fx * fy;
        out[(size_t)ch * size * size + (size_t)(pad_t + oy) * size +
            (pad_l + ox)] = v / 255.0f;
      }
    }
  }
  return out;
}

} // namespace yolov11

#endif // __YOLO_MODEL_H__
