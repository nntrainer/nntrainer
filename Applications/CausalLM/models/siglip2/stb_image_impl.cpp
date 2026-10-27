// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   stb_image_impl.cpp
 * @date   25 Jun 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Provides the stb_image implementation for the SigLIP2 encoder's
 *         image loading (siglip2_vision_encoder.cpp includes stb_image.inc for
 *         declarations only). Link this into targets that do NOT already pull
 *         in another STB_IMAGE_IMPLEMENTATION translation unit (e.g.
 *         nntr_quantize). Do NOT add it to libcausallm / nntr_causallm, which
 *         already get the implementation from timm_vit_transformer.cpp.
 */
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.inc"
