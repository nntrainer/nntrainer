// SPDX-License-Identifier: Apache-2.0
package com.example.quickdotaiapp

import com.example.quickdotai.BackendResult
import com.example.quickdotai.BackendType
import com.example.quickdotai.Capability
import com.example.quickdotai.ModelDescriptor
import com.example.quickdotai.ModelIds
import com.example.quickdotai.PromptPart
import com.example.quickdotai.QuickAiChatRole
import com.example.quickdotai.QuickAiError
import com.example.quickdotai.RuntimeKind
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import com.example.quickdotaiapp.power.GenerationObserver
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

class ChatControllerTest {
    val npu = ModelDescriptor("fake-npu", "fake-npu", "Fake NPU",
        RuntimeKind.NATIVE, setOf(BackendType.NPU), setOf(Capability.STREAMING, Capability.MESSAGES_API))
    val qwen = ModelDescriptor(ModelIds.QWEN3_0_6B, "qwen3-0.6b", "Qwen3 0.6B",
        RuntimeKind.NATIVE, setOf(BackendType.CPU, BackendType.GPU), setOf(Capability.STREAMING, Capability.TOOL_USE))
    val litert = ModelDescriptor(ModelIds.GEMMA4, "gemma4", "Gemma4 (LiteRT)",
        RuntimeKind.LITERT, setOf(BackendType.GPU), setOf(Capability.MULTIMODAL, Capability.MESSAGES_API, Capability.STREAMING))
    val htp = ModelDescriptor(ModelIds.GEMMA4_26B_A4B_HTP, "gemma4-26b-a4b", "Gemma4 26B A4B (HTP)",
        RuntimeKind.NATIVE, setOf(BackendType.NPU), setOf(Capability.STREAMING, Capability.MESSAGES_API))
    var catalog = listOf(qwen, npu, litert)

    private val executor = Executors.newSingleThreadExecutor()
    val engines: MutableList<FakeQuickDotAI> = java.util.Collections.synchronizedList(mutableListOf())
    var nextEngine: () -> FakeQuickDotAI = { FakeQuickDotAI() }
    val existing = mutableSetOf("/m/fake-npu/nntr_config.json", "/m/qwen3-0.6b/nntr_config.json", "/m/gemma-4-E2B-it/gemma-4-E2B-it.litertlm")
    @Volatile var clockMs = 0L
    var publishInterval = 0L

    fun controller(): ChatController = ChatController(
        scope = CoroutineScope(Dispatchers.Unconfined),
        engineDispatcher = executor.asCoroutineDispatcher(),
        catalogProvider = { catalog },
        engineFactory = { _, _ -> nextEngine().also { engines += it } },
        nativeLibDir = "/data/app/lib",
        examples = Examples(),
        pathExists = { it in existing },
        now = { clockMs },
        publishIntervalMs = publishInterval,
    ).also { c -> await("catalog") { c.state.value.catalog.isNotEmpty() } }

    fun await(label: String, timeoutMs: Long = 5000, cond: () -> Boolean) {
        val end = System.currentTimeMillis() + timeoutMs
        while (!cond()) { if (System.currentTimeMillis() > end) fail("timeout: $label"); Thread.sleep(5) }
    }
    fun ChatController.awaitIdle() {
        // busy is briefly false between a finished load and the generation it
        // hands off to, so drain the engine queue and re-check until it stays idle.
        do {
            await("idle") { !state.value.busy }
            executor.submit {}.get()                               // queued engine work finished too
        } while (state.value.busy)
    }
    fun ChatController.selectNpu() { selectFamily("fake-npu"); setBasePath("/m") }

    @After fun tearDown() { executor.shutdownNow() }

    @Test
    fun catalogSelectsTheFirstDescriptorAndSeedsTheRawExample() {
        val c = controller()
        val s = c.state.value
        assertEquals(qwen, s.descriptor)
        assertEquals("", s.draft)
        assertTrue(s.rawJson.trimStart().startsWith("{"))             // qwen: full request object
        c.setDraft("keep me")
        c.selectFamily("fake-npu")
        assertEquals(npu, c.state.value.descriptor)
        assertEquals("keep me", c.state.value.draft)                   // a model pick does not wipe the draft
        assertTrue(c.state.value.rawJson.trimStart().startsWith("[")) // MESSAGES_API: messages array
        c.selectFamily("qwen3-0.6b")
        c.selectBackend(BackendType.GPU)
        assertEquals(BackendType.GPU, c.state.value.selection.backend)
    }

    @Test
    fun loadReportsAndKeepsEngineOnMissingPath() {
        val c = controller(); c.selectNpu()
        c.load(); c.awaitIdle()
        assertTrue(c.state.value.status, c.state.value.status.startsWith("Loaded fake-npu"))
        assertEquals(listOf("load:fake-npu"), engines.single().calls)
        c.setBasePath("/m/missing-model")
        c.load(); c.awaitIdle()
        assertEquals("Load failed: not found: /m/missing-model/fake-npu/nntr_config.json", c.state.value.status)
        assertTrue(c.state.value.load is LoadState.Loaded)        // the old engine stayed
        assertFalse(engines.single().closed)
        assertEquals(1, engines.size)
    }

    @Test
    fun loadFailureReportsNativeDiagnostic() {
        nextEngine = { FakeQuickDotAI(loadResult = BackendResult.Err(QuickAiError.MODEL_LOAD_FAILED, "Failed to open file x")) }
        val c = controller(); c.selectNpu()
        c.load(); c.awaitIdle()
        assertEquals("Load failed: [MODEL_LOAD_FAILED] Failed to open file x", c.state.value.status)
        assertEquals(LoadState.Idle, c.state.value.load)
        assertTrue(engines.single().closed)
    }

    @Test
    fun reloadReplacesEngineAndUnloadReleasesIt() {
        val c = controller(); c.selectNpu()
        c.load(); c.awaitIdle()
        c.load(); c.awaitIdle()                                   // same request: no new engine
        assertEquals(1, engines.size)
        c.load(force = true); c.awaitIdle()
        assertEquals(2, engines.size)
        assertTrue(engines[0].closed)
        assertTrue(c.state.value.status.startsWith("Loaded "))
        c.unload(); c.awaitIdle()
        assertEquals("Unloaded.", c.state.value.status)
        assertEquals(LoadState.Idle, c.state.value.load)
        assertTrue(engines[1].calls.containsAll(listOf("unload", "close")))
        c.unload(); c.awaitIdle()
        assertEquals("Nothing to unload.", c.state.value.status)
    }

    @Test
    fun selectionChangeAfterLoadMakesNextLoadReplace() {
        val c = controller(); c.selectNpu()
        c.load(); c.awaitIdle()
        c.selectFamily("qwen3-0.6b")
        c.load(); c.awaitIdle()
        assertEquals(2, engines.size)
        assertEquals("load:qwen3-0.6b", engines[1].calls.first())
        assertTrue(engines[0].closed)
    }

    fun ChatController.loadedNpu(): ChatController { selectNpu(); load(); awaitIdle(); return this }

    @Test
    fun sendBuildsMessagesFromHistoryForNative() {
        val c = controller().loadedNpu()
        c.setDraft("first"); c.send(); c.awaitIdle()
        val e = engines.single()
        assertEquals(listOf("first"), e.lastMessages!!.map { (it.parts.single() as PromptPart.Text).text })
        assertEquals("Hello world<turn|>", c.state.value.lastAssistantText)
        assertEquals("Done.", c.state.value.status)
        c.setDraft("second"); c.send(); c.awaitIdle()
        val history = e.lastMessages!!.map { it.role to (it.parts.single() as PromptPart.Text).text }
        assertEquals(listOf(QuickAiChatRole.USER to "first", QuickAiChatRole.ASSISTANT to "Hello world",
            QuickAiChatRole.USER to "second"), history)   // the end-of-turn marker is not echoed back
        assertEquals(4, c.state.value.turns.size)
        assertEquals("", c.state.value.draft)
    }

    @Test
    fun sendWithImagesUsesTheMultimodalPath() {
        val c = controller().loadedNpu()
        c.setDraft("look"); c.addDraftImages(listOf(byteArrayOf(9)))
        c.send(); c.awaitIdle()
        assertEquals("multimodal:1", engines.single().calls.last { it.startsWith("multimodal") || it.startsWith("messages") })
        assertTrue(engines.single().lastMessages!!.single().parts.first() is PromptPart.ImageBytes)
        assertTrue(c.state.value.draftImages.isEmpty())
    }

    @Test
    fun sendUsesSessionForLiteRt() {
        val c = controller(); c.selectFamily("gemma4"); c.setBasePath("/m")
        c.setSession(SessionSettings(systemPrompt = "Be brief", temperature = "0.5", thinking = ThinkingChoice.OFF))
        c.setDraft("hi"); c.send(); c.awaitIdle()
        val e = engines.single()
        assertEquals(listOf("load:gemma4", "open", "chat:hi"), e.calls.toList())
        assertEquals("Be brief", e.lastSessionConfig!!.systemInstruction)
        assertEquals(0.5, e.lastSessionConfig!!.sampling!!.temperature!!, 1e-9)
        assertEquals(false, e.lastSessionConfig!!.chatTemplateKwargs!!.enableThinking)
        c.setDraft("again"); c.send(); c.awaitIdle()
        assertEquals("chat:again", e.calls.last())                 // the session is reused
        c.setSession(SessionSettings(systemPrompt = "Other")); c.awaitIdle()
        assertEquals("closeSession", e.calls.last())               // settings change closes it
        c.setDraft("x"); c.send(); c.awaitIdle()
        assertEquals(listOf("open", "chat:x"), e.calls.takeLast(2))
        c.clearConversation(); c.awaitIdle()
        assertEquals("rebuild", e.calls.last())
        assertTrue(c.state.value.turns.isEmpty())
    }

    @Test
    fun rawRoutesByCapability() {
        val c = controller(); c.selectFamily("qwen3-0.6b"); c.setBasePath("/m")
        c.setRawJson("""{"messages":[{"role":"user","content":"u"}],"tools":[]}""")
        c.sendRaw(); c.awaitIdle()
        assertEquals("json", engines[0].calls.last { it != "load:qwen3-0.6b" })
        assertTrue(c.state.value.turns.first().raw)
        assertEquals(Screen.CHAT, c.state.value.screen)
        c.selectFamily("fake-npu")
        c.setRawJson("""[{"role":"user","content":"u"}]""")
        c.sendRaw(); c.awaitIdle()
        assertEquals("messages:1", engines[1].calls.last())
        c.setRawJson("not json"); c.sendRaw()
        assertTrue(c.state.value.status.startsWith("Invalid JSON"))
    }

    @Test
    fun rawWithNoUserMessageStillRuns() {
        val c = controller().loadedNpu()
        c.setRawJson("""[{"role":"system","content":"only"}]""")
        c.sendRaw(); c.awaitIdle()
        assertEquals("messages:1", engines.single().calls.last())
        assertEquals("Done.", c.state.value.status)
    }

    @Test
    fun rawTurnsStayOutOfChatHistory() {
        val c = controller().loadedNpu()
        c.setRawJson("""[{"role":"user","content":"raw"}]"""); c.sendRaw(); c.awaitIdle()
        c.setDraft("chat"); c.send(); c.awaitIdle()
        assertEquals(listOf("chat"), engines.single().lastMessages!!.map { (it.parts.single() as PromptPart.Text).text })
    }

    @Test
    fun cancelOnLiteRtSaysTheEngineFinishesFirst() {
        val c = controller(); c.selectFamily("gemma4"); c.setBasePath("/m"); c.load(); c.awaitIdle()
        val e = engines.single()
        e.gate = CountDownLatch(1); e.parked = CountDownLatch(1)
        c.setDraft("go"); c.send()
        assertTrue(e.parked!!.await(2, TimeUnit.SECONDS))
        assertEquals("s1", e.chatSessionId)                  // LiteRT runs inside its chat session
        e.releaseGateOnCancel = false                         // the status is read before the engine can finish
        c.cancel()
        assertEquals("Stopping… LiteRT finishes the current answer first", c.state.value.status)
        e.gate!!.countDown(); c.awaitIdle()
        assertTrue(c.state.value.turns.last().stopped)
    }

    @Test
    fun cancelMarksTurnStopped() {
        val c = controller().loadedNpu()
        val e = engines.single()
        e.gate = CountDownLatch(1); e.parked = CountDownLatch(1)
        c.setDraft("go"); c.send()
        assertTrue(e.parked!!.await(2, TimeUnit.SECONDS))
        assertTrue(c.state.value.streaming)
        c.cancel()
        c.awaitIdle()
        val turn = c.state.value.turns.last()
        assertTrue(turn.stopped); assertFalse(turn.streaming)
        assertEquals("Hello", turn.text)
        assertEquals("Done.", c.state.value.status)
    }

    @Test
    fun sendWhileBusyIsIgnored() {
        val c = controller().loadedNpu()
        val e = engines.single()
        e.gate = CountDownLatch(1); e.parked = CountDownLatch(1)
        c.setDraft("one"); c.send()
        assertTrue(e.parked!!.await(2, TimeUnit.SECONDS))
        c.setDraft("two"); c.send()
        assertEquals("Busy.", c.state.value.status)
        e.gate!!.countDown(); c.awaitIdle()
        assertEquals(1, e.calls.count { it.startsWith("messages") })
        assertEquals("two", c.state.value.draft)                   // nothing was consumed
    }

    @Test
    fun unloadDuringStreamCancelsThenReleases() {
        val c = controller().loadedNpu()
        val e = engines.single()
        e.gate = CountDownLatch(1); e.parked = CountDownLatch(1)
        c.setDraft("go"); c.send()
        assertTrue(e.parked!!.await(2, TimeUnit.SECONDS))
        c.unload(); c.awaitIdle()
        await("unloaded") { c.state.value.status == "Unloaded." }
        assertTrue(e.cancelled)
        assertTrue(c.state.value.turns.last().stopped)
        assertEquals("Hello", c.state.value.turns.last().text)
        assertTrue(e.calls.indexOf("unload") > e.calls.indexOfFirst { it.startsWith("messages") })
        c.load(); c.awaitIdle()
        assertTrue(c.state.value.status.startsWith("Loaded "))
    }

    @Test
    fun streamingPublishesInBatches() {
        publishInterval = 50
        val c = controller().loadedNpu()
        val e = engines.single()
        e.gate = CountDownLatch(1); e.parked = CountDownLatch(1)
        c.setDraft("go"); c.send()
        assertTrue(e.parked!!.await(2, TimeUnit.SECONDS))
        assertEquals("", c.state.value.lastAssistantText)         // 0 ms elapsed: not published yet
        clockMs = 60
        e.gate!!.countDown(); c.awaitIdle()
        assertEquals("Hello world<turn|>", c.state.value.lastAssistantText)
        assertEquals(3, c.state.value.liveTokens)
    }

    @Test
    fun metricsFillTurnAndRuns() {
        val c = controller().loadedNpu()
        c.setDraft("go"); c.send(); c.awaitIdle()
        val m = c.state.value.turns.last().metrics!!
        assertEquals(120.0, m.ttftMs, 1e-9)
        assertEquals(10.0, m.tokensPerSec, 1e-9)
        assertEquals(3, m.generationTokens)
        assertEquals("Quick.AI", m.engineLabel)
        val run = c.state.value.runs.single()
        assertEquals("chat", run.mode); assertEquals(10, run.prefillTokens); assertEquals("Hello world<turn|>", run.output)
        assertEquals(120.0, c.state.value.lastMetrics!!.ttftMs, 1e-9)
    }

    @Test
    fun insertTextFileKeepsNewlinesAndStripsBom() {
        val c = controller()
        c.setDraft("")
        c.insertTextFile("a.txt", "\uFEFFline1\r\nline2\n")
        assertEquals("line1\r\nline2\n", c.state.value.draft)
        assertEquals("a.txt inserted into the message (13 chars)", c.state.value.notice)
        c.insertTextFile("b.txt", "more")
        assertEquals("line1\r\nline2\nmore", c.state.value.draft)
    }

    @Test
    fun rawModeIsAComposerToggle() {
        val c = controller()
        assertFalse(c.state.value.rawMode)
        c.setRawMode(true)
        assertTrue(c.state.value.rawMode)
        assertEquals(Screen.CHAT, c.state.value.screen)
        c.setRawMode(false)
        assertFalse(c.state.value.rawMode)
    }

    @Test
    fun loadJsonFileRoutes() {
        val c = controller()
        c.setScreen(Screen.METRICS)
        c.loadJsonFile("req.json", """{"messages":[{"role":"user","content":"u"}]}""")
        // A messages file lands in the JSON composer on the Chat screen.
        assertTrue(c.state.value.rawMode)
        assertEquals(Screen.CHAT, c.state.value.screen)
        assertTrue(c.state.value.rawJson.contains("\"messages\""))
        assertEquals("req.json loaded as JSON (1 messages)", c.state.value.notice)
        c.setDraft("")
        c.loadJsonFile("other.json", """{"a":1}""")
        assertEquals("""{"a":1}""", c.state.value.draft)
        c.loadJsonFile("bad.json", "{")
        assertTrue(c.state.value.notice!!.startsWith("bad.json is not valid JSON"))
    }

    @Test
    fun clearRunsEmptiesHistoryAndLastMetrics() {
        val c = controller().loadedNpu()
        c.setDraft("go"); c.send(); c.awaitIdle()
        assertEquals(1, c.state.value.runs.size)
        c.clearRuns()
        assertTrue(c.state.value.runs.isEmpty())
        assertEquals(null, c.state.value.lastMetrics)
        c.setDraft("again"); c.send(); c.awaitIdle()
        assertEquals(2, c.state.value.runs.single().index)        // indices stay unique after a clear
    }

    // ----- final-review findings -------------------------------------------

    @Test
    fun rawAfterLiteRtChatClosesTheSessionFirst() {
        val c = controller(); c.selectFamily("gemma4"); c.setBasePath("/m")
        c.setDraft("hi"); c.send(); c.awaitIdle()
        c.setRawJson("""[{"role":"user","content":"u"}]"""); c.sendRaw(); c.awaitIdle()
        val calls = engines.single().calls.toList()
        val close = calls.indexOf("closeSession")
        assertTrue("session must close before the flat messages call: $calls",
            close >= 0 && close < calls.indexOf("messages:1"))
    }

    @Test
    fun loadFailureKeepsTheDraftAndImages() {
        existing.remove("/m/fake-npu/nntr_config.json")
        val c = controller(); c.selectNpu()
        c.setDraft("hello"); c.addDraftImages(listOf(byteArrayOf(1)))
        c.send(); c.awaitIdle()
        assertTrue(c.state.value.status, c.state.value.status.startsWith("Load failed:"))
        assertEquals("hello", c.state.value.draft)
        assertEquals(1, c.state.value.draftImages.size)
        assertTrue(c.state.value.turns.isEmpty())
    }

    @Test
    fun reloadWhileStreamingCancelsAndKeepsThePartialTurn() {
        val c = controller().loadedNpu()
        val e = engines.single()
        e.gate = CountDownLatch(1); e.parked = CountDownLatch(1)
        c.setDraft("go"); c.send()
        assertTrue(e.parked!!.await(2, TimeUnit.SECONDS))
        c.load(force = true); c.awaitIdle()
        assertTrue(e.cancelled)
        val partial = c.state.value.turns.single { it.role == TurnRole.ASSISTANT }
        assertTrue(partial.stopped)
        assertEquals("Hello", partial.text)
        assertTrue(c.state.value.status, c.state.value.status.startsWith("Loaded "))
        assertEquals(2, engines.size)
    }

    @Test
    fun nativeHistorySendsOnlyTheLastTurnsImages() {
        val c = controller().loadedNpu()
        val x = byteArrayOf(1); val y = byteArrayOf(2)
        c.setDraft("a"); c.addDraftImages(listOf(x)); c.send(); c.awaitIdle()
        c.setDraft("b"); c.addDraftImages(listOf(y)); c.send(); c.awaitIdle()
        val msgs = engines.single().lastMessages!!
        assertEquals(1, msgs.sumOf { m -> m.parts.count { it is PromptPart.ImageBytes } })
        assertEquals(PromptPart.ImageBytes(y), msgs.last().parts.first())
    }

    /** The composables keep their own text and resync only when these move (typing must not move them). */
    @Test
    fun textRevisionsMoveOnExternalWritesOnly() {
        val c = controller().loadedNpu()
        val d0 = c.state.value.draftRevision
        c.setDraft("typed")
        assertEquals(d0, c.state.value.draftRevision)
        c.insertTextFile("a.txt", "x")
        val d1 = c.state.value.draftRevision
        assertTrue(d1 > d0)
        c.send(); c.awaitIdle()
        assertEquals("", c.state.value.draft)
        assertTrue(c.state.value.draftRevision > d1)
        val r0 = c.state.value.rawRevision
        c.setRawJson("[]")
        assertEquals(r0, c.state.value.rawRevision)
        c.resetRawJson()
        assertTrue(c.state.value.rawRevision > r0)
    }

    @Test
    fun sendPromptReportsTheGenerationEvents() {
        val c = controller(); c.selectFamily("fake-npu"); c.setBasePath("/m"); c.load(); c.awaitIdle()
        val events = mutableListOf<String>()
        c.observer = object : GenerationObserver {
            override fun onSend(tMs: Long) { events += "send" }
            override fun onFirstToken(tMs: Long) { events += "first" }
            override fun onToken(count: Int, tMs: Long) { events += "tok$count" }
            override fun onDone(tMs: Long, tokens: Int, promptTokens: Int?, stopped: Boolean, error: String?) { events += "done:$tokens:$promptTokens:$stopped" }
        }
        assertTrue(c.sendPrompt("measure me")); c.awaitIdle()
        assertEquals(listOf("send", "first", "tok1", "tok2", "tok3", "done:3:10:false"), events)
        assertEquals("power", c.state.value.runs.last().mode)
        assertEquals("measure me", c.state.value.turns.first { it.role == TurnRole.USER }.text)
    }

    @Test
    fun sendPromptRefusesWithoutALoadedModel() {
        val c = controller(); c.selectFamily("fake-npu")
        assertFalse(c.sendPrompt("x"))
        assertEquals("Load a model first.", c.state.value.status)
        assertFalse(c.sendPrompt("   "))
    }

    @Test
    fun powerChatFacadeReadsTheState() {
        val c = controller(); c.selectFamily("gemma4"); c.selectRuntime(RuntimeKind.LITERT)
        assertTrue(c.isLiteRt); assertFalse(c.isLoaded)
        c.setDraft("hello"); assertEquals("hello", c.composerText())
    }

    @Test
    fun observerHearsOnlyPowerGenerations() {
        val c = controller(); c.selectFamily("fake-npu"); c.setBasePath("/m"); c.load(); c.awaitIdle()
        val events = mutableListOf<String>()
        c.observer = object : GenerationObserver {
            override fun onSend(tMs: Long) { events += "send" }
            override fun onFirstToken(tMs: Long) { events += "first" }
            override fun onToken(count: Int, tMs: Long) { events += "tok" }
            override fun onDone(tMs: Long, tokens: Int, promptTokens: Int?, stopped: Boolean, error: String?) { events += "done" }
        }
        c.setDraft("hello"); c.send(); c.awaitIdle()                  // the user's own chat turn
        assertTrue(events.toString(), events.isEmpty())
        assertTrue(c.sendPrompt("measure")); c.awaitIdle()
        assertEquals("send", events.first())
    }

    @Test
    fun sendPromptRefusesWhenTheSelectionDiffersFromTheLoadedModel() {
        val c = controller(); c.selectFamily("fake-npu"); c.setBasePath("/m"); c.load(); c.awaitIdle()
        assertEquals("fake-npu", c.loadedKey?.substringBefore(':'))
        c.selectFamily("qwen3-0.6b")
        assertFalse(c.sendPrompt("x"))
        assertEquals("Load the selected model first.", c.state.value.status)
        assertEquals(1, engines.size)                                 // nothing was loaded by the refusal
    }

    @Test
    fun theHtpPresetIsSelectedFirstWhenTheCatalogHasIt() {
        catalog = listOf(qwen, npu, htp, litert)
        val c = controller()
        assertEquals(htp, c.state.value.descriptor)
    }

    /** The HTP packs hold 1,280 positions and truncate a long prompt's tail silently (nntrainer #4474):
     *  the chat sends only the newest user turn, like the View sample did, while the transcript keeps everything. */
    @Test
    fun anHtpChatSendsOnlyTheLastUserTurn() {
        catalog = listOf(qwen, htp)
        existing += "/m/gemma4-26b-a4b-htp-w4a32/nntr_config.json"
        val c = controller()
        c.selectFamily("gemma4-26b-a4b"); c.setBasePath("/m"); c.load(); c.awaitIdle()
        assertTrue(c.state.value.status, c.state.value.load is LoadState.Loaded)
        c.setDraft("first"); c.send(); c.awaitIdle()
        c.setDraft("second"); c.send(); c.awaitIdle()
        val e = engines.single()
        assertEquals(listOf(QuickAiChatRole.USER to "second"),
            e.lastMessages!!.map { it.role to (it.parts.single() as PromptPart.Text).text })
        assertEquals(4, c.state.value.turns.size)
    }
}
