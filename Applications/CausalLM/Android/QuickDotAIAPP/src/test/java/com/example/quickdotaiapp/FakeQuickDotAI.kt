// SPDX-License-Identifier: Apache-2.0
package com.example.quickdotaiapp

import com.example.quickdotai.BackendResult
import com.example.quickdotai.LoadModelRequest
import com.example.quickdotai.PerformanceMetrics
import com.example.quickdotai.PromptPart
import com.example.quickdotai.QuickAiChatMessage
import com.example.quickdotai.QuickAiChatResult
import com.example.quickdotai.QuickAiChatSessionConfig
import com.example.quickdotai.QuickDotAI
import com.example.quickdotai.StreamSink
import java.util.concurrent.CountDownLatch

/** Scripted engine: streams [deltas], records every call, can block mid-stream. */
class FakeQuickDotAI(
    override val kind: String = "native",
    var deltas: List<String> = listOf("Hello", " world", "<turn|>"),
    var loadResult: BackendResult<Unit> = BackendResult.Ok(Unit),
) : QuickDotAI {
    override var architecture: String? = "fake-arch"
    val calls: MutableList<String> = java.util.Collections.synchronizedList(mutableListOf())
    @Volatile var cancelled = false
    @Volatile var closed = false
    @Volatile var sessionId: String? = null
    @Volatile var lastMessages: List<QuickAiChatMessage>? = null
    @Volatile var lastJson: String? = null
    @Volatile var lastSessionConfig: QuickAiChatSessionConfig? = null
    var metricsToReturn = PerformanceMetrics(
        prefillTokens = 10, prefillDurationMs = 100.0, generationTokens = 3,
        generationDurationMs = 300.0, totalDurationMs = 400.0, ttftMs = 120.0,
    )
    /** When set, streaming blocks after the first delta until released (or cancelled). */
    @Volatile var gate: CountDownLatch? = null
    /** Released when the stream is parked on [gate]. */
    @Volatile var parked: CountDownLatch? = null

    override val chatSessionId: String? get() = sessionId
    override fun load(req: LoadModelRequest): BackendResult<Unit> { calls += "load:${req.modelId}"; return loadResult }
    override fun unload(): BackendResult<Unit> { calls += "unload"; return BackendResult.Ok(Unit) }
    override fun metrics(): BackendResult<PerformanceMetrics> = BackendResult.Ok(metricsToReturn)
    /** A test that asserts on the state between cancel() and the engine's reaction keeps the gate closed. */
    @Volatile var releaseGateOnCancel = true
    override fun cancel() { cancelled = true; if (releaseGateOnCancel) gate?.countDown() }
    override fun chatCancel() = cancel()
    override fun close() { closed = true; calls += "close" }
    override fun openChatSession(config: QuickAiChatSessionConfig?): BackendResult<String> {
        lastSessionConfig = config; sessionId = "s1"; calls += "open"; return BackendResult.Ok("s1")
    }
    override fun closeChatSession(): BackendResult<Unit> { sessionId = null; calls += "closeSession"; return BackendResult.Ok(Unit) }
    override fun chatRebuild(messages: List<QuickAiChatMessage>): BackendResult<Unit> { calls += "rebuild"; return BackendResult.Ok(Unit) }
    override fun runChatModelHandleStreaming(text: String, sink: StreamSink): BackendResult<QuickAiChatResult> {
        calls += "chat:$text"; stream(sink); return BackendResult.Ok(QuickAiChatResult(content = ""))
    }
    override fun runChatMultimodalHandleStreaming(parts: List<PromptPart>, sink: StreamSink): BackendResult<QuickAiChatResult> {
        calls += "chatMultimodal:${parts.size}"; stream(sink); return BackendResult.Ok(QuickAiChatResult(content = ""))
    }
    override fun runModelHandleWithMessagesStreaming(messages: List<QuickAiChatMessage>, sink: StreamSink): BackendResult<Unit> {
        lastMessages = messages; calls += "messages:${messages.size}"; stream(sink); return BackendResult.Ok(Unit)
    }
    override fun runMultimodalHandleWithMessagesStreaming(messages: List<QuickAiChatMessage>, sink: StreamSink): BackendResult<Unit> {
        lastMessages = messages; calls += "multimodal:${messages.size}"; stream(sink); return BackendResult.Ok(Unit)
    }
    override fun runModelHandleWithJsonStreaming(jsonRequest: String, sink: StreamSink): BackendResult<Unit> {
        lastJson = jsonRequest; calls += "json"; stream(sink); return BackendResult.Ok(Unit)
    }

    private fun stream(sink: StreamSink) {
        cancelled = false
        for ((i, d) in deltas.withIndex()) {
            if (cancelled) break
            sink.onDelta(d)
            if (i == 0) gate?.let { g -> parked?.countDown(); g.await() }
        }
        sink.onDone()
    }
}
