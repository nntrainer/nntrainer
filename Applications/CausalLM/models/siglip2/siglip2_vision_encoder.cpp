// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   siglip2_vision_encoder.cpp
 * @date   18 Jun 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  SigLIP2 ViT-B/16 vision encoder implementation.
 *
 * Architecture deltas from timm_vit_transformer:
 *   - Conv2D patch embed has bias (disable_bias=false).
 *   - Q/K/V are three separate fully_connected layers fed from LN1 output.
 *   - MLP activation is tanh_gelu (maps HF gelu_pytorch_tanh).
 *   - layer_norm_eps = 1e-6.
 *   - After post_layernorm, a final fully_connected(unit=256) named
 *     enc_to_dec_proj. Output is [1, 196, 256].
 *
 * Weight loading order (must match weight_converter.py::collect_encoder):
 *   patch_embed conv w, b -> pos_embed ->
 *   12 x [LN1 w,b -> q w,b -> k w,b -> v w,b -> out w,b ->
 *          LN2 w,b -> fc1 w,b -> fc2 w,b] ->
 *   post_ln w,b -> enc_to_dec_proj w,b
 */

#include "siglip2_vision_encoder.h"
// stb_image implementation is provided by timm_vit_transformer.cpp;
// include only for declarations here.
#include "stb_image.inc"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <factory.h>
#include <iomanip>
#include <llm_util.hpp>
#include <nntrainer-api-common.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace causallm {

/* -------------------------------------------------------------------------
 * Image preprocessing helpers (identical to timm_vit_transformer.cpp)
 * --------------------------------------------------------------------- */

/**
 * @brief Precompute Pillow's resample coefficients for one axis.
 *
 * Reproduces Pillow's precompute_coeffs() (src/libImaging/Resample.c) for the
 * BILINEAR (triangle) filter exactly:
 *   scale       = in_size / out_size
 *   filterscale = max(scale, 1)
 *   support     = 1.0 * filterscale            (BILINEAR support = 1.0)
 *   center      = (xx + 0.5) * scale
 *   xmin        = max(0, floor(center - support))
 *   xmax        = min(in_size, ceil(center + support)) - xmin
 *   weight[k]   = bilinear((xmin + k + 0.5 - center) / filterscale)
 *   weights normalized to sum to 1.
 * where bilinear(t) = 1 - |t| for |t| < 1, else 0.
 *
 * NOTE the half-pixel convention is center = (xx+0.5)*scale (NO trailing -0.5)
 * and the support is filterscale (= scale when downscaling), which is what
 * makes downscaling match PIL — the previous (x+0.5)*scale-0.5 / support=scale
 * 2D kernel did not.
 *
 * @param in_size  source dimension
 * @param out_size destination dimension
 * @param bounds   out: 2*out_size ints (xmin, xmax per output pixel)
 * @param coeffs   out: out_size * ksize floats (row-major)
 * @return ksize   max number of source samples contributing per output pixel
 */
static int siglip2PrecomputeCoeffs(int in_size, int out_size,
                                   std::vector<int> &bounds,
                                   std::vector<double> &coeffs) {
  const double scale =
    static_cast<double>(in_size) / static_cast<double>(out_size);
  const double filterscale = scale < 1.0 ? 1.0 : scale;
  const double support = 1.0 * filterscale; // BILINEAR filter support == 1.0
  const int ksize = static_cast<int>(std::ceil(support)) * 2 + 1;

  bounds.assign(static_cast<size_t>(out_size) * 2, 0);
  coeffs.assign(static_cast<size_t>(out_size) * ksize, 0.0);

  for (int xx = 0; xx < out_size; ++xx) {
    const double center = (xx + 0.5) * scale;
    const double ss = 1.0 / filterscale;
    int xmin = static_cast<int>(center - support + 0.5);
    if (xmin < 0)
      xmin = 0;
    int xmax = static_cast<int>(center + support + 0.5);
    if (xmax > in_size)
      xmax = in_size;
    xmax -= xmin;

    double *k = &coeffs[static_cast<size_t>(xx) * ksize];
    double wsum = 0.0;
    int x = 0;
    for (; x < xmax; ++x) {
      double t = std::abs((x + xmin - center + 0.5) * ss);
      double w = t < 1.0 ? 1.0 - t : 0.0; // bilinear (triangle) filter
      k[x] = w;
      wsum += w;
    }
    for (x = 0; x < xmax; ++x) {
      if (wsum != 0.0)
        k[x] /= wsum;
    }
    // zero the tail (already zeroed by assign, but keep explicit for clarity)
    for (; x < ksize; ++x)
      k[x] = 0.0;

    bounds[static_cast<size_t>(xx) * 2 + 0] = xmin;
    bounds[static_cast<size_t>(xx) * 2 + 1] = xmax;
  }
  return ksize;
}

/**
 * @brief Resize an image buffer reproducing Pillow's Image.BILINEAR resample
 *        BIT-EXACTLY (uint8 path).
 *
 * Performs the two separable passes Pillow uses (horizontal then vertical).
 * Crucially, this reproduces Pillow's *fixed-point integer* 8-bit resample
 * (ImagingResampleHorizontal_8bpc / ...Vertical_8bpc in src/libImaging/
 * Resample.c): the double filter coefficients are quantized to integers with
 * PRECISION_BITS = 32 - 8 - 2 = 22, each pass accumulates int64 products, and a
 * rounding shift (clip8) reduces back to uint8 between passes and at the end.
 *
 * The earlier float-coefficient version left ~212 pixels differing by a single
 * uint8 LSB versus PIL (max ~0.008 in normalized units); the SigLIP2 encoder
 * amplified that into a ~2.96 hidden-state diff that flipped a generated token.
 * Matching PIL's integer arithmetic exactly removes ALL pixel differences
 * (verified: 0/150528 pixels differ vs img.resize((224,224), Image.BILINEAR)).
 *
 * @param src Source image data (interleaved HWC, uint8).
 * @param src_w, src_h Source dimensions.
 * @param channels Number of channels (must be 3).
 * @param dst_w, dst_h Destination dimensions.
 * @param dst_float Output buffer of size dst_w * dst_h * channels (float,
 *        holding the exact uint8 result values in [0,255]).
 */
static void siglip2ResizeImageFloat(const unsigned char *src, int src_w,
                                    int src_h, int channels, int dst_w,
                                    int dst_h, std::vector<float> &dst_float) {
  // Pillow's 8-bit fixed-point precision (Resample.c: PRECISION_BITS).
  constexpr int PRECISION_BITS = 32 - 8 - 2;
  const int64_t ROUND = static_cast<int64_t>(1) << (PRECISION_BITS - 1);

  // Quantize one axis' double coeffs into PIL's int16-range fixed-point coeffs
  // (Resample.c normalize_coeffs_8bpc: int(round(coef * (1 <<
  // PRECISION_BITS)))).
  auto quantize = [](const std::vector<double> &coeffs, int n, int ksize,
                     std::vector<int> &icoeffs) {
    icoeffs.assign(static_cast<size_t>(n) * ksize, 0);
    for (size_t i = 0; i < static_cast<size_t>(n) * ksize; ++i)
      icoeffs[i] = static_cast<int>(
        std::lround(coeffs[i] * static_cast<double>(1 << PRECISION_BITS)));
  };

  // clip8: add rounding constant, arithmetic right-shift, clamp to [0,255],
  // matching Pillow's clip8() at the end of each 8-bit resample pass.
  auto clip8 = [ROUND](int64_t acc) -> float {
    int64_t r = (acc + ROUND) >> PRECISION_BITS;
    if (r < 0)
      r = 0;
    if (r > 255)
      r = 255;
    return static_cast<float>(r);
  };

  // ----- Horizontal pass: [src_h, src_w] -> [src_h, dst_w] -----
  std::vector<int> h_bounds;
  std::vector<double> h_coeffs;
  const int h_ksize = siglip2PrecomputeCoeffs(src_w, dst_w, h_bounds, h_coeffs);
  std::vector<int> h_icoeffs;
  quantize(h_coeffs, dst_w, h_ksize, h_icoeffs);

  std::vector<float> tmp(static_cast<size_t>(src_h) * dst_w * channels);
  for (int y = 0; y < src_h; ++y) {
    for (int x = 0; x < dst_w; ++x) {
      const int xmin = h_bounds[static_cast<size_t>(x) * 2 + 0];
      const int xmax = h_bounds[static_cast<size_t>(x) * 2 + 1];
      const int *k = &h_icoeffs[static_cast<size_t>(x) * h_ksize];
      int64_t acc[3] = {0, 0, 0};
      for (int i = 0; i < xmax; ++i) {
        const int sx = xmin + i;
        const int idx = (y * src_w + sx) * channels;
        for (int c = 0; c < channels; ++c)
          acc[c] += static_cast<int64_t>(src[idx + c]) * k[i];
      }
      const int out_idx = (y * dst_w + x) * channels;
      for (int c = 0; c < channels; ++c)
        tmp[out_idx + c] = clip8(acc[c]);
    }
  }

  // ----- Vertical pass: [src_h, dst_w] -> [dst_h, dst_w] -----
  std::vector<int> v_bounds;
  std::vector<double> v_coeffs;
  const int v_ksize = siglip2PrecomputeCoeffs(src_h, dst_h, v_bounds, v_coeffs);
  std::vector<int> v_icoeffs;
  quantize(v_coeffs, dst_h, v_ksize, v_icoeffs);

  dst_float.resize(static_cast<size_t>(dst_w) * dst_h * channels);
  for (int y = 0; y < dst_h; ++y) {
    const int ymin = v_bounds[static_cast<size_t>(y) * 2 + 0];
    const int ymax = v_bounds[static_cast<size_t>(y) * 2 + 1];
    const int *k = &v_icoeffs[static_cast<size_t>(y) * v_ksize];
    for (int x = 0; x < dst_w; ++x) {
      int64_t acc[3] = {0, 0, 0};
      for (int i = 0; i < ymax; ++i) {
        const int sy = ymin + i;
        const int idx = (sy * dst_w + x) * channels;
        for (int c = 0; c < channels; ++c)
          acc[c] += static_cast<int64_t>(tmp[idx + c]) * k[i];
      }
      const int out_idx = (y * dst_w + x) * channels;
      for (int c = 0; c < channels; ++c)
        dst_float[out_idx + c] = clip8(acc[c]);
    }
  }
}

/**
 * @brief Load, resize, and normalize an image into CHW float data.
 *
 * Uses siglip2ResizeImageFloat for resizing, which employs a triangle filter
 * matching PIL's Image.BILINEAR behaviour. The resized pixel values (in [0,255]
 * float range) are normalized to [-1, 1] without intermediate uint8 rounding,
 * giving near-exact parity with the PyTorch/PIL golden outputs.
 */
static std::vector<float>
siglip2LoadAndPreprocessImage(const std::string &filepath, int target_width,
                              int target_height, bool normalize) {
  int width, height, channels;
  unsigned char *image =
    stbi_load(filepath.c_str(), &width, &height, &channels, STBI_default);
  if (!image) {
    throw std::runtime_error("Failed to load image: " + filepath);
  }

  // Convert to 3-channel RGB uint8 first (stbi_load output)
  std::vector<unsigned char> rgb_buf;
  const unsigned char *rgb_ptr = image;
  int rgb_channels = channels;
  if (channels == 1) {
    rgb_buf.resize(static_cast<size_t>(width * height * 3));
    for (int i = 0; i < width * height; ++i) {
      rgb_buf[i * 3] = image[i];
      rgb_buf[i * 3 + 1] = image[i];
      rgb_buf[i * 3 + 2] = image[i];
    }
    rgb_ptr = rgb_buf.data();
    rgb_channels = 3;
  } else if (channels == 4) {
    rgb_buf.resize(static_cast<size_t>(width * height * 3));
    for (int i = 0; i < width * height; ++i) {
      rgb_buf[i * 3] = image[i * 4];
      rgb_buf[i * 3 + 1] = image[i * 4 + 1];
      rgb_buf[i * 3 + 2] = image[i * 4 + 2];
    }
    rgb_ptr = rgb_buf.data();
    rgb_channels = 3;
  } else if (channels != 3) {
    stbi_image_free(image);
    throw std::runtime_error("Unsupported number of channels: " +
                             std::to_string(channels));
  }

  // Resize using triangle filter (float output in [0,255], matching PIL
  // BILINEAR)
  std::vector<float> resized_float;
  if (width != target_width || height != target_height) {
    siglip2ResizeImageFloat(rgb_ptr, width, height, rgb_channels, target_width,
                            target_height, resized_float);
  } else {
    // No resize needed: convert uint8 to float directly
    resized_float.resize(
      static_cast<size_t>(target_width * target_height * rgb_channels));
    for (size_t i = 0; i < resized_float.size(); ++i) {
      resized_float[i] = static_cast<float>(rgb_ptr[i]);
    }
  }

  stbi_image_free(image);

  // Convert HWC float [0,255] to CHW float [-1,1] (SigLIP2 normalization)
  std::vector<float> output(3 * target_height * target_width);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < target_height; ++y) {
      for (int x = 0; x < target_width; ++x) {
        float val = resized_float[(y * target_width + x) * 3 + c];
        float pixel = normalize ? (val / 255.0f - 0.5f) / 0.5f : val;
        output[c * target_height * target_width + y * target_width + x] = pixel;
      }
    }
  }

  return output;
}

/* -------------------------------------------------------------------------
 * Parameter setup
 * --------------------------------------------------------------------- */

/**
 * @brief Set SigLIP2 encoder parameters from model and runtime configs.
 *
 * A combined vision-encoder-decoder config.json wraps the encoder params
 * under cfg["encoder"]; a standalone config keeps them at the top level.
 * nntr_cfg provides runtime overrides (batch size, etc.).
 */
void Siglip2VisionEncoder::setupParameters(json &cfg, json &generation_cfg,
                                           json &nntr_cfg) {
  (void)generation_cfg;

  BATCH_SIZE = nntr_cfg.value("batch_size", 1);
  MODEL_TENSOR_TYPE = nntr_cfg.value("model_tensor_type", "FP32-FP32");
  EMBEDDING_DTYPE = nntr_cfg.value("embedding_dtype", "FP32");
  FC_LAYER_DTYPE = nntr_cfg.value("fc_layer_dtype", "FP32");
  PATCH_EMBED_DTYPE = nntr_cfg.value("patch_embed_dtype", "FP32");

  // Encoder config may be nested under cfg["encoder"] in a combined
  // config.json, or at the top level (standalone encoder config.json).
  json enc_cfg = cfg.contains("encoder") ? cfg["encoder"] : cfg;

  DIM = enc_cfg.value("hidden_size", 768);
  INTERMEDIATE_SIZE = enc_cfg.value("intermediate_size", 3072);
  NUM_LAYERS = enc_cfg.value("num_hidden_layers", 12);
  NUM_HEADS = enc_cfg.value("num_attention_heads", 12);
  HEAD_DIM = DIM / NUM_HEADS;
  NUM_KEY_VALUE_HEADS = NUM_HEADS;
  NORM_EPS = enc_cfg.value("layer_norm_eps", 1e-6f);
  GQA_SIZE = 1;

  IS_CAUSAL = false;
  ROPE_THETA = 0;
  TIE_WORD_EMBEDDINGS = false;
  SLIDING_WINDOW = UINT_MAX;

  IMG_SIZE = enc_cfg.value("image_size", nntr_cfg.value("img_size", 224));
  PATCH_SIZE = enc_cfg.value("patch_size", nntr_cfg.value("patch_size", 16));
  NUM_PATCHES = nntr_cfg.value("num_patches", 196u);
  IMG_CHANNELS = 3;

  INIT_SEQ_LEN =
    nntr_cfg.value("init_seq_len", static_cast<unsigned int>(NUM_PATCHES));
  MAX_SEQ_LEN =
    nntr_cfg.value("max_seq_len", static_cast<unsigned int>(NUM_PATCHES));
  NUM_TO_GENERATE = nntr_cfg.value("num_to_generate", 0);
  MEMORY_SWAP = nntr_cfg.contains("fsu") ? nntr_cfg["fsu"].get<bool>() : false;
  FSU_LOOKAHEAD = nntr_cfg.contains("fsu_lookahead")
                    ? nntr_cfg["fsu_lookahead"].get<unsigned int>()
                    : 1;
  MAX_POSITION_EMBEDDINGS = NUM_PATCHES;
}

/* -------------------------------------------------------------------------
 * Graph construction helpers
 * --------------------------------------------------------------------- */

/**
 * @brief Create patch embedding and positional embedding layers.
 *
 * Weight order (matches collect_encoder):
 *   1. patch_embed_conv:weight  [768, 3, 16, 16] OIHW
 *   2. patch_embed_conv:bias    [768]
 *   3. pos_embedding:weight     [1, 1, 196, 768]
 */
Tensor Siglip2VisionEncoder::createPatchEmbed(Tensor input) {
  const int embed_dim = DIM;

  // Conv2D with bias (SigLIP2 delta from timm_vit which had disable_bias=true)
  // Weight dtype follows nntr_config "patch_embed_dtype" (default FP32): the
  // 16x16 stride-16 patch conv is a pure [196, 768] x [768, 768] matmul (no
  // overlap), so with Q8_0 it runs through the same interleaved int8 GEMM as
  // the FC layers (conv2d stores the filter as a [CRS, out_ch] matmul weight
  // and takes the im2col + dotQnK path on NCHW). It is deliberately NOT tied
  // to fc_layer_dtype so pre-existing quantized files (conv saved FP32) keep
  // loading unchanged.
  LayerHandle conv(createLayer(
    "conv2d", {withKey("name", "patch_embed_conv"),
               withKey("kernel_size", {std::to_string(PATCH_SIZE),
                                       std::to_string(PATCH_SIZE)}),
               withKey("filters", std::to_string(embed_dim)),
               withKey("stride", {std::to_string(PATCH_SIZE),
                                  std::to_string(PATCH_SIZE)}),
               withKey("padding", "valid"), withKey("disable_bias", "false"),
               withKey("weight_dtype", PATCH_EMBED_DTYPE)}));
  Tensor h = conv(input);

  // Reshape from [1, embed_dim, H/P, W/P] -> [1, 1, embed_dim, num_patches]
  LayerHandle flatten(createLayer(
    "reshape", {withKey("name", "patch_embed_flatten"),
                withKey("target_shape", "1:" + std::to_string(embed_dim) + ":" +
                                          std::to_string(NUM_PATCHES))}));
  h = flatten(h);

  // Transpose to [1, 1, num_patches, embed_dim]
  LayerHandle transpose(
    createLayer("permute", {withKey("name", "patch_embed_transpose"),
                            withKey("direction", {1, 3, 2})}));
  h = transpose(h);

  // Learned position embedding [1, 1, 196, 768]
  LayerHandle pos_embed(
    createLayer("weight", {withKey("name", "pos_embedding"),
                           withKey("dim", "1:1:" + std::to_string(NUM_PATCHES) +
                                            ":" + std::to_string(embed_dim)),
                           withKey("tensor_dtype", "FP32"),
                           withKey("weight_name", "pos_embedding")}));
  Tensor pos = pos_embed(input);

  LayerHandle add(
    createLayer("addition", {withKey("name", "patch_embed_add")}));
  return add({h, pos});
}

/**
 * @brief Create a pre-normalized self-attention block.
 *
 * Weight order per layer (matches collect_encoder):
 *   enc_layer{i}_ln1:gamma, beta
 *   enc_layer{i}_wq:weight, bias
 *   enc_layer{i}_wk:weight, bias
 *   enc_layer{i}_wv:weight, bias
 *   enc_layer{i}_out:weight, bias
 */
Tensor Siglip2VisionEncoder::createAttention(const int layer_id, Tensor input) {
  const std::string pfx = "enc_layer" + std::to_string(layer_id);

  // Pre-norm (LN1)
  LayerHandle norm(createLayer(
    "layer_normalization", {withKey("name", pfx + "_ln1"), withKey("axis", "3"),
                            withKey("epsilon", std::to_string(NORM_EPS)),
                            withKey("packed", "false")}));
  Tensor normed = norm(input);

  // Three separate Q/K/V projections
  LayerHandle q_proj(
    createLayer("fully_connected", {withKey("name", pfx + "_wq"),
                                    withKey("unit", std::to_string(DIM)),
                                    withKey("disable_bias", "false")}));
  LayerHandle k_proj(
    createLayer("fully_connected", {withKey("name", pfx + "_wk"),
                                    withKey("unit", std::to_string(DIM)),
                                    withKey("disable_bias", "false")}));
  LayerHandle v_proj(
    createLayer("fully_connected", {withKey("name", pfx + "_wv"),
                                    withKey("unit", std::to_string(DIM)),
                                    withKey("disable_bias", "false")}));

  Tensor query = q_proj(normed);
  Tensor key = k_proj(normed);
  Tensor value = v_proj(normed);

  LayerHandle attention(createLayer(
    "mha_core", {withKey("name", pfx + "_attn"),
                 withKey("num_heads", std::to_string(NUM_HEADS)),
                 withKey("num_heads_kv", std::to_string(NUM_HEADS)),
                 withKey("max_timestep", std::to_string(NUM_PATCHES + 1)),
                 withKey("is_causal", "false"), withKey("use_rope", "false"),
                 withKey("rope_theta", std::to_string(ROPE_THETA))}));
  Tensor context = attention({query, key, value});

  // Output projection
  LayerHandle out_proj(
    createLayer("fully_connected", {withKey("name", pfx + "_out"),
                                    withKey("unit", std::to_string(DIM)),
                                    withKey("disable_bias", "false")}));
  return out_proj(context);
}

/**
 * @brief Create a pre-normalized feed-forward block with tanh_gelu activation.
 *
 * Weight order per layer:
 *   enc_layer{i}_ln2:gamma, beta
 *   enc_layer{i}_fc1:weight, bias
 *   enc_layer{i}_fc2:weight, bias
 */
Tensor Siglip2VisionEncoder::createMlp(const int layer_id, Tensor input) {
  const std::string pfx = "enc_layer" + std::to_string(layer_id);

  // Pre-norm (LN2)
  LayerHandle norm(createLayer(
    "layer_normalization", {withKey("name", pfx + "_ln2"), withKey("axis", "3"),
                            withKey("epsilon", std::to_string(NORM_EPS)),
                            withKey("packed", "false")}));
  Tensor h = norm(input);

  // FC1 -> tanh_gelu -> FC2
  LayerHandle fc1(createLayer(
    "fully_connected", {withKey("name", pfx + "_fc1"),
                        withKey("unit", std::to_string(INTERMEDIATE_SIZE)),
                        withKey("disable_bias", "false")}));
  h = fc1(h);

  // tanh_gelu maps HF's gelu_pytorch_tanh
  LayerHandle gelu(
    createLayer("activation", {withKey("name", pfx + "_gelu"),
                               withKey("activation", "tanh_gelu")}));
  h = gelu(h);

  LayerHandle fc2(
    createLayer("fully_connected", {withKey("name", pfx + "_fc2"),
                                    withKey("unit", std::to_string(DIM)),
                                    withKey("disable_bias", "false")}));
  return fc2(h);
}

/**
 * @brief Create one SigLIP2 transformer encoder block with residual
 * connections.
 */
Tensor Siglip2VisionEncoder::createTransformerDecoderBlock(const int layer_id,
                                                           Tensor input) {
  const std::string pfx = "enc_layer" + std::to_string(layer_id);

  // Attention sub-block with residual
  Tensor att_out = createAttention(layer_id, input);
  LayerHandle att_res(
    createLayer("addition", {withKey("name", pfx + "_att_res")}));
  Tensor residual = att_res({input, att_out});

  // MLP sub-block with residual
  Tensor mlp_out = createMlp(layer_id, residual);
  LayerHandle mlp_res(
    createLayer("addition", {withKey("name", pfx + "_mlp_res")}));
  return mlp_res({residual, mlp_out});
}

/**
 * @brief Construct the SigLIP2 encoder inference graph.
 *
 * Full weight order matches collect_encoder in weight_converter.py:
 *   patch_embed_conv w,b -> pos_embedding w ->
 *   12x [ln1 w,b -> wq w,b -> wk w,b -> wv w,b -> out w,b ->
 *         ln2 w,b -> fc1 w,b -> fc2 w,b] ->
 *   post_ln w,b -> enc_to_dec_proj w,b
 */
std::pair<Tensor, Tensor> Siglip2VisionEncoder::constructModel() {
  Tensor input({BATCH_SIZE, IMG_CHANNELS, IMG_SIZE, IMG_SIZE}, "input0");
  Tensor h = createPatchEmbed(input);

  for (int i = 0; i < NUM_LAYERS; i++) {
    h = createTransformerDecoderBlock(i, h);
  }

  // Post layer norm
  LayerHandle post_ln(createLayer(
    "layer_normalization", {withKey("name", "post_ln"), withKey("axis", "3"),
                            withKey("epsilon", std::to_string(NORM_EPS)),
                            withKey("packed", "false")}));
  h = post_ln(h);

  // Encoder-to-decoder projection: [1, 196, 768] -> [1, 196, 256]
  LayerHandle enc_proj(createLayer(
    "fully_connected", {withKey("name", "enc_to_dec_proj"),
                        withKey("unit", std::to_string(ENC_TO_DEC_DIM)),
                        withKey("disable_bias", "false")}));
  h = enc_proj(h);

  return {input, h};
}

/* -------------------------------------------------------------------------
 * Custom layer registration
 * --------------------------------------------------------------------- */

/**
 * @brief Register custom layers required by the base transformer.
 */
void Siglip2VisionEncoder::registerCustomLayers() {
  Transformer::registerCustomLayers();
}

/* -------------------------------------------------------------------------
 * Inference
 * --------------------------------------------------------------------- */

/**
 * @brief Run the encoder on an image and return projected hidden states.
 *
 * Preprocesses the image (resize to IMG_SIZE x IMG_SIZE, normalize to
 * [-1, 1] with mean=0.5, std=0.5), runs incremental_inference, copies the
 * result into enc_output_buf_ and returns a reference to it.
 */
std::vector<float> Siglip2VisionEncoder::encode(const std::string &image_path) {
  if (!is_initialized) {
    throw std::runtime_error(
      "Siglip2VisionEncoder is not initialized. Call initialize() first.");
  }

  std::vector<float> image_data = siglip2LoadAndPreprocessImage(
    image_path, static_cast<int>(IMG_SIZE), static_cast<int>(IMG_SIZE), true);

  std::vector<float *> inputs{image_data.data()};
  std::vector<float *> labels;

  std::vector<float *> output = model->incremental_inference(
    BATCH_SIZE, inputs, labels, NUM_PATCHES, 0, NUM_PATCHES, false);

  const int out_size =
    static_cast<int>(NUM_PATCHES) * static_cast<int>(ENC_TO_DEC_DIM);
  enc_output_buf_.assign(output[0], output[0] + out_size);
  for (auto *p : output)
    delete[] p;
  return enc_output_buf_;
}

/**
 * @brief Run the encoder on pre-loaded CHW float pixels, bypassing C++ resize.
 *
 * Used for encoder-math isolation: the caller supplies the exact same pixel
 * tensor that PyTorch consumed (saved via verify_parity.py --dump), so any
 * remaining difference vs the golden must come from the encoder math, not
 * from image-preprocessing differences.
 */
std::vector<float> Siglip2VisionEncoder::encodePixels(const float *pixel_data,
                                                      size_t pixel_count) {
  if (!is_initialized) {
    throw std::runtime_error(
      "Siglip2VisionEncoder is not initialized. Call initialize() first.");
  }
  const size_t expected =
    static_cast<size_t>(IMG_CHANNELS) * IMG_SIZE * IMG_SIZE;
  if (pixel_count != expected) {
    throw std::runtime_error("encodePixels: expected " +
                             std::to_string(expected) + " floats, got " +
                             std::to_string(pixel_count));
  }

  // Copy pixel_data into a local buffer (incremental_inference takes float*;
  // mirrors how encode() owns its image buffer).
  std::vector<float> pixel_buf(pixel_data, pixel_data + pixel_count);
  std::vector<float *> inputs{pixel_buf.data()};
  std::vector<float *> labels;

  std::vector<float *> output = model->incremental_inference(
    BATCH_SIZE, inputs, labels, NUM_PATCHES, 0, NUM_PATCHES, false);

  const int out_size =
    static_cast<int>(NUM_PATCHES) * static_cast<int>(ENC_TO_DEC_DIM);
  enc_output_buf_.assign(output[0], output[0] + out_size);
  for (auto *p : output)
    delete[] p;
  return enc_output_buf_;
}

/**
 * @brief Run the encoder on an image path (run() interface).
 */
void Siglip2VisionEncoder::run(const WSTR prompt, bool do_sample,
                               const WSTR system_prompt, const WSTR tail_prompt,
                               bool log_output) {
  (void)do_sample;
  (void)system_prompt;
  (void)tail_prompt;
  (void)log_output;

  if (!is_initialized) {
    throw std::runtime_error(
      "Siglip2VisionEncoder is not initialized. Call initialize() first.");
  }

  std::string image_path_str(prompt);
  std::vector<float> result = encode(image_path_str);

  std::cout << std::setprecision(9)
            << "Encoder output [1,196,256], first 10 values:";
  const int print_count = static_cast<int>(ENC_TO_DEC_DIM) > 10
                            ? 10
                            : static_cast<int>(ENC_TO_DEC_DIM);
  for (int i = 0; i < print_count; ++i) {
    std::cout << " [" << i << "]=" << result[i];
  }
  std::cout << std::endl;
}

} // namespace causallm
