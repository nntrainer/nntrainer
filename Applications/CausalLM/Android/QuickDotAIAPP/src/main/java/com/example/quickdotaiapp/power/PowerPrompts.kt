// SPDX-License-Identifier: Apache-2.0
/**
 * @file  PowerPrompts.kt
 * @brief The bundled fixed prompts of the Power tab: four English texts
 *        sized by four characters per token (res/raw/power_prompt_*.txt),
 *        each ending with the same instruction. The engine's own prompt
 *        token count is what the result shows; these sizes are nominal.
 */
package com.example.quickdotaiapp.power

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

    /** Nominal size in characters: four per token. */
    fun charsFor(tokens: Int): Int = tokens * 4
}
