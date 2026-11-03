// SPDX-License-Identifier: Apache-2.0
package com.example.quickdotaiapp.power

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/** A chat that streams [tokens] tokens on the test scope's virtual clock and honours cancel(). */
class FakePowerChat(private val scope: CoroutineScope, private val now: () -> Long) : PowerChat {
    override var isLiteRt = false
    override var isLoaded = true
    override var loadedKey: String? = "m1"
    /** Returns true from sendPrompt but never emits an event (a generation that died before onSend). */
    var silent = false
    override var observer: GenerationObserver? = null
    var composer = "composer text"
    var tokens = 100
    var ttftMs = 500L
    var perTokenMs = 50L
    var promptTokens: Int? = 1000
    var error: String? = null
    /** True between the first token and done: the sampler reads it to draw more current. */
    @Volatile var generating = false
    var sends = 0
    var cancels = 0
    var clears = 0
    @Volatile private var cancelled = false
    private var job: Job? = null

    override fun composerText(): String = composer
    override fun clearConversation() { clears++ }
    override fun cancel() { cancels++; cancelled = true }

    override fun sendPrompt(text: String): Boolean {
        sends++
        cancelled = false
        if (silent) return true
        val obs = observer ?: return true
        obs.onSend(now())
        job = scope.launch {
            delay(ttftMs)
            generating = true
            var produced = 0
            while (produced < tokens && !cancelled) {
                produced++
                if (produced == 1) obs.onFirstToken(now())
                obs.onToken(produced, now())
                if (produced < tokens && !cancelled) delay(perTokenMs)
            }
            generating = false
            obs.onDone(now(), produced, promptTokens, cancelled, error)
        }
        return true
    }
}
