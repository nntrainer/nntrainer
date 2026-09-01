// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	model_registry.h
 * @date	14 July 2026
 * @brief	Shared model-registration for the CausalLM Factory
 * @see		https://github.com/nnstreamer/
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __MODEL_REGISTRY_H__
#define __MODEL_REGISTRY_H__

namespace causallm {

/**
 * @brief Register every runnable model on Factory::Instance(); idempotent
 * (internal std::call_once).
 */
void registerAllModels();

} // namespace causallm

#endif // __MODEL_REGISTRY_H__
