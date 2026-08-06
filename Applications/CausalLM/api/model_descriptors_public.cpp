// SPDX-License-Identifier: Apache-2.0
/**
 * @file   model_descriptors_public.cpp
 * @brief  Public model descriptor self-registration. Zero proprietary-model
 *         literals — proprietary/extension models register themselves in
 *         their own TUs.
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * config_name values verified against get_model_name_from_type() in
 * quick_dot_ai_api.cpp.  arch_string values verified against
 * register_models() Factory registrations in the same file.
 */
#include "model_descriptor.h"

using namespace quick_dot_ai;

#define B(x) (1u << (unsigned)(x)) /* BackendType: CPU=0, GPU=1, NPU=2 */

__attribute__((constructor)) static void register_public_descriptors() {
  static const ModelDescriptor kPublic[] = {
    {"qwen3-0.6b", "qwen3-0.6b", "Qwen3 0.6B", QDA_RUNTIME_NATIVE, B(0) | B(1),
     QDA_CAP_STREAMING | QDA_CAP_TOOL_USE,
     "QWEN3-0.6B", /* get_model_name_from_type(CAUSAL_LM_MODEL_QWEN3_0_6B) */
     "Qwen3ForCausalLM"},
    {"qwen3-1.7b-q40", "qwen3-1.7b", "Qwen3 1.7B (Q40)", QDA_RUNTIME_NATIVE,
     B(0) | B(1), QDA_CAP_STREAMING | QDA_CAP_TOOL_USE,
     "QWEN3-1.7B-Q40", /* get_model_name_from_type(CAUSAL_LM_MODEL_QWEN3_1_7B_Q40)
                        */
     "Qwen3ForCausalLM"},
    {"tiny-bert", "tiny-bert", "Tiny BERT", QDA_RUNTIME_NATIVE, B(0),
     QDA_CAP_EMBEDDING,
     "TINY_BERT", /* get_model_name_from_type(CAUSAL_LM_MODEL_TINY_BERT) */
     "MultilingualTinyBert"},
    {"function-gemma", "function-gemma", "Function Gemma", QDA_RUNTIME_NATIVE,
     B(0) | B(1), QDA_CAP_TOOL_USE,
     "FUNCTION_GEMMA", /* get_model_name_from_type(CAUSAL_LM_MODEL_FUNCTION_GEMMA)
                        */
     "Gemma3ForCausalLM"},
    {"gemma4-cpu", "gemma4", "Gemma4 (CPU)", QDA_RUNTIME_NATIVE, B(0),
     QDA_CAP_STREAMING,
     "GEMMA4_CPU", /* get_model_name_from_type(CAUSAL_LM_MODEL_GEMMA4_CPU) */
     "Gemma4ForCausalLM" /* Factory registration pending */},
    {"vjepa-lfm2", "vjepa", "V-JEPA 2 + LFM2 (CPU)", QDA_RUNTIME_NATIVE, B(0),
     QDA_CAP_STREAMING | QDA_CAP_MULTIMODAL | QDA_CAP_MESSAGES_API |
       QDA_CAP_MULTI_IMAGE,
     "vjepa-lfm2", /* config_name = on-device model directory name */
     "Lfm2VLVJepa21BModel"},
#ifdef ENABLE_QNN_MODELS
    {"gemma4-e2b-qnn", "gemma4", "Gemma4 E2B (QNN)", QDA_RUNTIME_NATIVE, B(2),
     QDA_CAP_MESSAGES_API,
     "GEMMA4-E2B-QNN", /* get_model_name_from_type(CAUSAL_LM_MODEL_GEMMA4_E2B_QNN)
                        */
     "Gemma4_E2B_QNN"},
    {"vjepa2-qnn", "vjepa", "V-JEPA 2 (QNN)", QDA_RUNTIME_NATIVE, B(2),
     QDA_CAP_MULTIMODAL | QDA_CAP_MESSAGES_API | QDA_CAP_MULTI_IMAGE,
     "VJEPA2-QNN", "VJEPA2_QNN"},
#endif
  };
  for (const auto &d : kPublic)
    register_model_descriptor(&d);
}
