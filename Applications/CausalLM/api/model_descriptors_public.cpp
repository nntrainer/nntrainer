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
  /* GPU (OpenCL/Adreno) arm of the same architecture. A separate entry
     rather than B(0)|B(1) on the one above, because the two backends read
     DIFFERENT on-device packs -- the GPU arm is the QS4CX + fp16-sidecar
     pack with its pre-packed .v8cpack -- and config_name is what selects the
     directory (g_model_path_map: GEMMA4_GPU -> gemma4_gpu). Present only on
     an OpenCL build; advertising GPU on a CPU-only library would be an empty
     claim. */
#ifdef ENABLE_OPENCL
    {"gemma4-gpu", "gemma4", "Gemma4 (GPU)", QDA_RUNTIME_NATIVE, B(1),
     QDA_CAP_STREAMING, "GEMMA4_GPU", "Gemma4ForCausalLM"},
    /* Gemma4 26B-A4B MoE (TQ2CX experts + QS4ACX dense) on the GPU, experts
       streamed from the pack (moe_fsu). Its own directory
       (GEMMA4_MOE_GPU -> gemma4_26b_moe_gpu) so it does not collide with the
       E2B pack; the MoE class is picked from config.json
       (text_config.enable_moe_block) by the Gemma4ForCausalLM factory. */
    {"gemma4-26b-moe-gpu", "gemma4", "Gemma4 26B-A4B MoE (GPU)",
     QDA_RUNTIME_NATIVE, B(1), QDA_CAP_STREAMING, "GEMMA4_MOE_GPU",
     "Gemma4ForCausalLM"},
#endif
#ifdef ENABLE_HEXKL
    {"lfm2-24ba2b-htp", "lfm2", "LFM2 24B A2B MoE (HTP)", QDA_RUNTIME_NATIVE,
     B(2), QDA_CAP_STREAMING | QDA_CAP_MESSAGES_API, "LFM2-24BA2B-HTP",
     "Lfm2MoeForCausalLM"},
    {"gemma4-26b-a4b-htp", "gemma4-26b-a4b", "Gemma4 26B A4B MoE (HTP)",
     QDA_RUNTIME_NATIVE, B(2), QDA_CAP_STREAMING | QDA_CAP_MESSAGES_API,
     "GEMMA4-26B-A4B-HTP", "Gemma4ForCausalLM"},
    /* Same weights, a 30k-context pack for the agent KV-prefix demo. */
    {"gemma4-26b-a4b-htp-long", "gemma4-26b-a4b",
     "Gemma4 26B A4B MoE (HTP, 30k demo)", QDA_RUNTIME_NATIVE, B(2),
     QDA_CAP_STREAMING | QDA_CAP_MESSAGES_API, "GEMMA4-26B-A4B-HTP-LONG",
     "Gemma4ForCausalLM"},
    /* The rotated checkpoint converted to TL2 / affine WH (2-bit experts,
       4-bit tail layers 28-29, online Hadamard). Short context. */
    /* The sample resolves one model per family/runtime/backend tuple. */
    {"gemma4-26b-a4b-htp-2bit-4bit", "gemma4-26b-a4b-2bit-4bit",
     "Gemma4 26B A4B MoE (HTP, rotated 2/4-bit)", QDA_RUNTIME_NATIVE, B(2),
     QDA_CAP_STREAMING | QDA_CAP_MESSAGES_API, "GEMMA4-26B-A4B-HTP-2BIT-4BIT",
     "Gemma4ForCausalLM"},
#endif
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
