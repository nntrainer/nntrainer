// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jungwon-Lee <jungone.lee@samsung.com>
 *
 * @file   tokenizer_loader.h
 * @date   07 Apr 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Tokenizer loading helpers for Quick.AI models
 */
#ifndef __TOKENIZER_LOADER_H__
#define __TOKENIZER_LOADER_H__

#include <memory>

#include "json.hpp"
#include <tokenizers_cpp.h>

namespace causallm {

std::unique_ptr<tokenizers::Tokenizer> LoadTokenizer(nlohmann::json &nntr_cfg);

} // namespace causallm

#endif // __TOKENIZER_LOADER_H__
