// SPDX-License-Identifier: Apache-2.0
/**
 * @file   model_registry.h
 * @brief  One-call registration of every runnable CausalLM model with the
 *         causallm::Factory. Moved out of main.cpp so alternative entry
 *         points (tests, SDK wrappers) register the same model set without
 *         duplicating the list.
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __CAUSALLM_MODEL_REGISTRY_H__
#define __CAUSALLM_MODEL_REGISTRY_H__

namespace causallm {

/**
 * @brief Register all runnable causallm models with the Factory.
 *        Idempotent (std::call_once inside).
 */
void registerAllModels();

} // namespace causallm

#endif // __CAUSALLM_MODEL_REGISTRY_H__
