// SPDX-License-Identifier: Apache-2.0
/**
 * @file  RawRequest.kt
 * @brief OpenAI-style request text as the JSON composer and the .json
 *        attachment see it: detection, message conversion, image attach.
 */
package com.example.quickdotaiapp

import com.example.quickdotai.PromptPart
import com.example.quickdotai.QuickAiChatMessage
import com.example.quickdotai.QuickAiChatRole
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonPrimitive

object RawRequest {
    sealed interface Parsed {
        /** A top-level messages array, or an object with a "messages" array. */
        data class Messages(val messages: List<QuickAiChatMessage>, val hasTools: Boolean) : Parsed
        /** Valid JSON that is not an OpenAI request. */
        data object Other : Parsed
        data class Invalid(val reason: String) : Parsed
    }

    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    /** Never throws: the composer calls this on every keystroke. */
    fun parse(text: String): Parsed {
        val element = try {
            json.parseToJsonElement(text)
        } catch (t: Throwable) {
            return Parsed.Invalid(t.message ?: "invalid JSON")
        }
        val array: JsonArray
        val hasTools: Boolean
        when {
            element is JsonArray -> { array = element; hasTools = false }
            element is JsonObject && element["messages"] is JsonArray -> {
                array = element.getValue("messages").jsonArray
                hasTools = element["tools"] is JsonArray || element["functions"] is JsonArray
            }
            else -> return Parsed.Other
        }
        val messages = try {
            array.mapNotNull { item ->
                val obj = item as? JsonObject ?: return@mapNotNull null
                val role = when (obj["role"]?.jsonPrimitive?.contentOrNull?.lowercase()) {
                    "system" -> QuickAiChatRole.SYSTEM
                    "user" -> QuickAiChatRole.USER
                    "assistant" -> QuickAiChatRole.ASSISTANT
                    else -> return@mapNotNull null
                }
                QuickAiChatMessage(role, contentParts(obj["content"]))
            }
        } catch (t: IllegalArgumentException) {
            // jsonPrimitive on an object or array: a shape no chat template accepts.
            return Parsed.Invalid(t.message ?: "unexpected message shape")
        }
        return Parsed.Messages(messages, hasTools)
    }

    private fun contentParts(content: JsonElement?): List<PromptPart> {
        if (content == null || content is JsonNull) return listOf(PromptPart.Text(""))
        if (content is JsonArray) {
            val parts = content.mapNotNull { p ->
                val o = p as? JsonObject ?: return@mapNotNull null
                when (o["type"]?.jsonPrimitive?.contentOrNull?.lowercase()) {
                    "text", "input_text" ->
                        o["text"]?.jsonPrimitive?.contentOrNull?.takeIf { it.isNotEmpty() }?.let { PromptPart.Text(it) }
                    else -> null // image parts: the app attaches picked images itself
                }
            }
            return parts.ifEmpty { listOf(PromptPart.Text("")) }
        }
        return listOf(PromptPart.Text(content.jsonPrimitive.content))
    }

    /** Puts [images] in front of the last user message's parts, once. */
    fun withImages(messages: List<QuickAiChatMessage>, images: List<ByteArray>): List<QuickAiChatMessage> {
        if (images.isEmpty()) return messages
        val i = messages.indexOfLast { it.role == QuickAiChatRole.USER }
        if (i < 0) return messages
        val m = messages[i]
        if (m.parts.any { it is PromptPart.ImageBytes || it is PromptPart.ImageFile }) return messages
        return messages.toMutableList().also {
            it[i] = m.copy(parts = images.map { b -> PromptPart.ImageBytes(b) } + m.parts)
        }
    }
}
