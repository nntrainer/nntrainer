// SPDX-License-Identifier: Apache-2.0
/**
 * @file  Examples.kt
 * @brief Default prompts, model folder names and label rules carried
 *        over from the View-based sample.
 */
package com.example.quickdotaiapp

import com.example.quickdotai.BackendType
import com.example.quickdotai.Capability
import com.example.quickdotai.ModelDescriptor
import com.example.quickdotai.ModelIds
import com.example.quickdotai.RuntimeKind

const val DEFAULT_BASE_PATH = "/sdcard/Download/aistudio-mobile/models"

class Examples {
    fun rawJsonFor(descriptor: ModelDescriptor?): String {
        if (descriptor == null) return messagesApiExample
        exampleByModelId[descriptor.id]?.let { return it }
        return if (Capability.MESSAGES_API in descriptor.capabilities) messagesApiExample else toolExample
    }

    companion object {
        val modelPathById: Map<String, String> = mapOf(
            ModelIds.GEMMA4 to "gemma-4-E2B-it/gemma-4-E2B-it.litertlm",
            ModelIds.QWEN3_0_6B to "qwen3-0.6b",
            ModelIds.QWEN3_1_7B_Q40 to "qwen3-1.7b-q40-arm",
            ModelIds.TINY_BERT to "tiny-bert",
            ModelIds.FUNCTION_GEMMA to "function_gemma",
            ModelIds.GEMMA4_CPU to "gemma4_cpu",
            ModelIds.GEMMA4_E2B_QNN to "gemma-4-e2b-qnn",
            ModelIds.VJEPA_QNN to "vjepa2-qnn",
        )

        fun modelPath(descriptor: ModelDescriptor, basePath: String): String =
            "${basePath.trimEnd('/')}/${modelPathById[descriptor.id] ?: descriptor.id}"

        /** What must exist before a load replaces the current engine: the folder's
         *  nntr_config.json for Quick.AI models (the C API refuses a folder without
         *  it, and a half-copied folder must not cost the loaded model), the
         *  .litertlm file for LiteRT. */
        fun requiredPath(descriptor: ModelDescriptor, basePath: String): String {
            val rel = modelPathById[descriptor.id] ?: descriptor.id
            val base = basePath.trimEnd('/')
            return if (descriptor.runtime == RuntimeKind.LITERT) "$base/$rel"
            else "$base/${rel.substringBefore('/')}/nntr_config.json"
        }

        fun runtimeLabel(runtime: RuntimeKind): String = when (runtime) {
            RuntimeKind.NATIVE -> "Quick.AI"
            RuntimeKind.LITERT -> "LiteRT"
        }

        fun isSdCapable(descriptor: ModelDescriptor?): Boolean =
            descriptor?.runtime == RuntimeKind.NATIVE && Capability.SPECULATIVE in descriptor.capabilities

        fun badge(descriptor: ModelDescriptor): String = when {
            Capability.MULTIMODAL in descriptor.capabilities -> "MULTIMODAL"
            Capability.EMBEDDING in descriptor.capabilities -> "EMBEDDING"
            descriptor.backends == setOf(BackendType.NPU) -> "NPU"
            else -> "TEXT"
        }

        private val exampleByModelId: Map<String, String> = mapOf(
            ModelIds.GEMMA4 to """[
  {"role": "system", "content": "You are a concise vision assistant."},
  {"role": "user", "content": [{"type": "text", "text": "Describe this image."}, {"type": "image_url", "image_url": {"url": "quickdotaiapp://selected-image"}}]}
]""",
            ModelIds.FUNCTION_GEMMA to """{
  "messages": [
    {"role": "system", "content": "You can call tools when they are useful."},
    {"role": "user", "content": "01012345678 번호로 상담 예약 확인 문자를 보내줘."}
  ],
  "tools": [{"type": "function", "function": {"name": "send_sms", "description": "Send a text message to a phone number.", "parameters": {"type": "object", "properties": {"phone_number": {"type": "string"}, "message": {"type": "string"}}, "required": ["phone_number", "message"]}}}]
}""",
            ModelIds.TINY_BERT to """{
  "messages": [{"role": "user", "content": "Explain what text embeddings are in one sentence."}]
}""",
        )
        private const val messagesApiExample = """[
  {"role": "system", "content": "You are a helpful assistant. Answer briefly."},
  {"role": "user", "content": "Summarize why on-device language models are useful."}
]"""
        private const val toolExample = """{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a short checklist for testing an Android API wrapper."}
  ]
}"""
    }
}
