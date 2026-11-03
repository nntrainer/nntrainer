// SPDX-License-Identifier: Apache-2.0
/**
 * @file  PowerChat.kt
 * @brief What the Power runner needs from the chat: the moments of a
 *        generation and a way to send one prompt. ChatController
 *        implements it; the tests use a fake.
 */
package com.example.quickdotaiapp.power

/** Moments of one generation on the controller's clock. onToken / onFirstToken run on the engine thread. */
interface GenerationObserver {
    fun onSend(tMs: Long)
    fun onFirstToken(tMs: Long)
    fun onToken(count: Int, tMs: Long)
    fun onDone(tMs: Long, tokens: Int, promptTokens: Int?, stopped: Boolean, error: String?)
}

interface PowerChat {
    val isLiteRt: Boolean
    val isLoaded: Boolean
    /** Identifies the loaded model (null when nothing is loaded); a session ends when it changes. */
    val loadedKey: String?
    var observer: GenerationObserver?
    fun composerText(): String
    fun clearConversation()
    /** Sends [text] as a user turn; false (with a status) when blank, busy or nothing is loaded. */
    fun sendPrompt(text: String): Boolean
    fun cancel()
}
