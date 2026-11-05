// SPDX-License-Identifier: Apache-2.0
/**
 * @file  PowerPrompts.kt
 * @brief The bundled fixed prompts of the Power tab: four English texts
 *        sized by four characters per token (res/raw/power_prompt_*.txt),
 *        each ending with the same instruction. The engine's own prompt
 *        token count is what the result shows; these sizes are nominal.
 */
package com.example.quickdotaiapp.power

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/** What one request to a native pack may use: prompt tokens and answer tokens (nntr_config.json). */
data class PackLimits(val promptTokens: Int, val decodeTokens: Int)

object PowerPrompts {
    val sizes: List<Int> = listOf(256, 1024, 2048, 4096)
    const val INSTRUCTION = "Summarize the text above in three sentences."

    fun resourceName(tokens: Int): String = when (tokens) {
        256 -> "power_prompt_256"
        1024 -> "power_prompt_1k"
        2048 -> "power_prompt_2k"
        4096 -> "power_prompt_4k"
        else -> throw IllegalArgumentException("no bundled prompt of $tokens tokens")
    }

    /** The HTP summarization prompt: the first user message of an OpenAI messages array; null for anything else. */
    fun firstUserContent(messagesJson: String): String? = runCatching {
        Json.parseToJsonElement(messagesJson).jsonArray
            .firstOrNull { it.jsonObject["role"]?.jsonPrimitive?.content == "user" }
            ?.jsonObject?.get("content")?.jsonPrimitive?.content
    }.getOrNull()

    /** Nominal size in characters: four per token. */
    fun charsFor(tokens: Int): Int = tokens * 4

    /** The Metrics benchmark's choices, in nominal tokens. */
    val benchPrefill: List<Int> = listOf(128, 256, 512, 1024, 2048, 4096)
    val benchDecode: List<Int> = listOf(32, 64, 128, 256)
    /** Asks for an answer longer than any decode choice, so the window, not the end of turn, ends the run. */
    const val BENCH_INSTRUCTION = "Retell the text above in full detail, section by section, in your own words."

    /** [source] (a bundled prompt) without its instruction, cut at a word to [tokens] nominal tokens, plus [BENCH_INSTRUCTION]. */
    fun bench(source: String, tokens: Int): String {
        val body = source.trimEnd().removeSuffix(INSTRUCTION).trimEnd()
        val room = (charsFor(tokens) - BENCH_INSTRUCTION.length - 2).coerceAtLeast(1)
        val cut = if (body.length <= room) body
            else body.substring(0, room).let { s -> s.substring(0, s.lastIndexOf(' ').takeIf { it > 0 } ?: s.length) }
        return cut + "\n\n" + BENCH_INSTRUCTION
    }

    /** The engine's ceiling (causal_lm.cpp): a prompt gets at most max_seq_len - num_to_generate tokens, the
     *  answer num_to_generate. Null when the file lacks one of the two keys.
     *  ponytail: a model that does not chunk its prefill also stops at init_seq_len, and whether it chunks is
     *  decided in C++ (prefillDriveChunk), so the app cannot tell; such a prompt is cut and the run's prompt
     *  token count shows it. Exposing the budget through the API (nntrainer #4474) is the upgrade path. */
    fun packLimits(nntrConfigJson: String): PackLimits? = runCatching {
        val o = Json.parseToJsonElement(nntrConfigJson).jsonObject
        fun int(key: String) = o[key]?.jsonPrimitive?.intOrNull
        val max = int("max_seq_len") ?: return null
        val gen = int("num_to_generate") ?: return null
        PackLimits(promptTokens = (max - gen).coerceAtLeast(0), decodeTokens = gen)
    }.getOrNull()
}
