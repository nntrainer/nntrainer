// SPDX-License-Identifier: Apache-2.0
package com.example.quickdotaiapp

import com.example.quickdotai.QuickAiError
import com.example.quickdotai.StreamSink
import org.junit.Assert.assertEquals
import org.junit.Test

class NativeStreamSinkTest {
    private class RecordingSink : StreamSink {
        val events = mutableListOf<String>()
        override fun onDelta(text: String) { events.add("text:$text") }
        override fun onReasoningDelta(text: String) { events.add("reasoning:$text") }
        override fun onDone() { events.add("done") }
        override fun onError(error: QuickAiError, message: String?) { events.add("error:${error.name}:$message") }
    }

    @Test fun successHasOneTerminalAndDropsLateCallbacks() {
        val recorded = RecordingSink()
        val sink = NativeWire.TerminalSink(recorded)
        sink.onReasoningDelta("plan"); sink.onDelta("answer"); sink.onDone()
        sink.onDone(); sink.onDelta("late"); sink.onError(QuickAiError.UNKNOWN, "late error")
        assertEquals(listOf("reasoning:plan", "text:answer", "done"), recorded.events)
    }

    @Test fun workerFailureHasOneErrorAndDropsLateSuccess() {
        val recorded = RecordingSink()
        val sink = NativeWire.TerminalSink(recorded)
        sink.onDelta("partial"); sink.onError(QuickAiError.UNKNOWN, "worker exited")
        sink.onReasoningDelta("late"); sink.onDone(); sink.onError(QuickAiError.UNKNOWN, "duplicate")
        assertEquals(listOf("text:partial", "error:UNKNOWN:worker exited"), recorded.events)
    }
}
