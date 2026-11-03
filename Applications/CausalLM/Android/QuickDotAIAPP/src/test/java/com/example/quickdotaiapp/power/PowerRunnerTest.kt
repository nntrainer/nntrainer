// SPDX-License-Identifier: Apache-2.0
package com.example.quickdotaiapp.power

import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class PowerRunnerTest {
    private val meta = SessionMeta("2026-11-02T20:00:00+09:00", "SM-S948N", "16", "fake-npu", "Quick.AI", "NPU")
    private val settings = PowerSettings(PromptSource.Fixed(256), windowTokens = 64, repeats = 2, baselineSec = 2, cooldownSec = 1)

    /** One harness per test: the fake chat, a sampler that draws 4 W while generating and 0.8 W otherwise, the runner. */
    private inner class Harness(scope: TestScope, prompt: String? = "prompt text", settings: PowerSettings = this.settings) {
        val now = { scope.testScheduler.currentTime }
        val chat = FakePowerChat(scope, now)
        var plugged = false
        var status = 0
        val logs = mutableListOf<PowerSessionLog>()
        val runner = PowerRunner(
            scope = scope.backgroundScope, chat = chat,
            sampler = { PowerSample(now(), if (chat.generating) -1_000_000 else -200_000, 4000, 280, plugged, status, Float.NaN, null, null, null) },
            prompts = { prompt },
            onSessionDone = { logs += it; "/x/runs.json" },
            now = now,
        ).also { it.setSettings(settings) }
        val state get() = runner.state.value
    }

    @Test fun fullSessionProducesTwoRunsAndWritesTheLog() = runTest {
        val h = Harness(this)
        h.runner.start(meta)
        advanceTimeBy(30_000); runCurrent()
        assertEquals(PowerPhase.DONE, h.state.phase)
        assertEquals(2, h.state.runs.size)
        for (run in h.state.runs) {
            assertNull(run.skipped)
            // (4.0 − 0.8) W × 63 × 50 ms ÷ 64 tok ≈ 157 mJ/tok
            assertEquals(160.0, run.mJPerToken!!, 15.0)
            assertEquals(64, run.marks!!.generatedTokens)
            assertNotNull(run.marks!!.windowEndMs)
            assertTrue(run.marks!!.stopped)
        }
        assertEquals(2, h.chat.cancels); assertEquals(2, h.chat.clears); assertEquals(2, h.chat.sends)
        assertEquals(1, h.logs.size); assertEquals(2, h.logs[0].runs.size); assertEquals(meta, h.logs[0].meta)
        assertEquals("/x/runs.json", h.state.savedPath)
        assertNull(h.chat.observer)
    }

    @Test fun liteRtRunsToTheEndButTheWindowClosesAtN() = runTest {
        val h = Harness(this, settings = settings.copy(repeats = 1))
        h.chat.isLiteRt = true
        h.runner.start(meta)
        advanceTimeBy(30_000); runCurrent()
        val run = h.state.runs.single()
        assertEquals(0, h.chat.cancels)
        assertEquals(100, run.marks!!.generatedTokens)
        assertTrue(run.marks!!.windowEndMs!! < run.marks!!.doneMs!!)
        assertEquals(20.3, run.tokensPerSec!!, 1.0)
        assertTrue(!run.marks!!.stopped)
    }

    @Test fun windowClosesOnceEvenIfDoneArrivesFirst() = runTest {
        val h = Harness(this, settings = settings.copy(repeats = 1))
        h.chat.tokens = 10
        h.chat.promptTokens = 0                                       // LiteRT reports no prompt count
        h.runner.start(meta)
        advanceTimeBy(30_000); runCurrent()
        val run = h.state.runs.single()
        assertEquals(run.marks!!.doneMs, run.marks!!.windowEndMs)
        assertEquals(0, h.chat.cancels)
        assertEquals(10, run.marks!!.generatedTokens)
        assertNull(run.marks!!.promptTokens)
    }

    @Test fun thermalGateWaitsThenSkipsAHotRun() = runTest {
        val h = Harness(this)
        h.status = 2
        h.runner.start(meta)
        advanceTimeBy(3_000 + 50_000); runCurrent()
        assertEquals(PowerPhase.THERMAL_GATE, h.state.phase)
        assertEquals(0, h.chat.sends)
        advanceTimeBy(45_000); runCurrent()
        assertEquals("hot", h.state.runs.first().skipped)
        h.status = 0
        advanceTimeBy(20_000); runCurrent()
        assertEquals(2, h.state.runs.size)
        assertNull(h.state.runs[1].skipped)
        assertEquals(PowerPhase.DONE, h.state.phase)
    }

    @Test fun waitsForTheCableUnlessAllowed() = runTest {
        val h = Harness(this)
        h.plugged = true
        h.runner.start(meta)
        advanceTimeBy(30_000); runCurrent()
        assertEquals(PowerPhase.WAIT_UNPLUG, h.state.phase)
        assertTrue(h.state.plugged)
        h.plugged = false
        advanceTimeBy(5_000); runCurrent()
        assertEquals(PowerPhase.BASELINE, h.state.phase)

        val allowed = Harness(this, settings = settings.copy(allowPlugged = true))
        allowed.plugged = true
        allowed.runner.start(meta)
        advanceTimeBy(4_000); runCurrent()
        assertEquals(PowerPhase.BASELINE, allowed.state.phase)
    }

    @Test fun repluggingSkipsTheRunAndEndsTheSession() = runTest {
        val h = Harness(this)
        h.runner.start(meta)
        advanceTimeBy(3_000 + 2_000 + 500 + 1_000); runCurrent()      // settle, baseline, ttft, 20 tokens in
        assertTrue(h.chat.generating)
        h.plugged = true
        advanceTimeBy(30_000); runCurrent()
        assertEquals(1, h.state.runs.size)
        assertEquals("plugged", h.state.runs[0].skipped)
        assertEquals(PowerPhase.DONE, h.state.phase)
        assertEquals(1, h.logs.size)
    }

    @Test fun stopDuringARunDiscardsItAndKeepsEarlierRuns() = runTest {
        val h = Harness(this)
        h.runner.start(meta)
        advanceTimeBy(13_000); runCurrent()                            // run 1 done (~8.7 s), run 2 generating
        assertEquals(1, h.state.runs.size)
        assertTrue(h.chat.generating)
        h.runner.stop()
        advanceTimeBy(1_000); runCurrent()
        assertEquals(1, h.state.runs.size)
        assertEquals(2, h.chat.cancels)
        assertNull(h.chat.observer)
        assertEquals(PowerPhase.DONE, h.state.phase)
        assertEquals(1, h.logs.size)
    }

    @Test fun emptyPromptNeverStarts() = runTest {
        val h = Harness(this, prompt = null, settings = settings.copy(prompt = PromptSource.Composer))
        h.runner.start(meta)
        advanceTimeBy(10_000); runCurrent()
        assertEquals("Composer is empty", h.state.error)
        assertEquals(PowerPhase.IDLE, h.state.phase)
        assertEquals(0, h.chat.sends)
    }

    @Test fun livePointsAreCapped() = runTest {
        val h = Harness(this, settings = settings.copy(baselineSec = 60, repeats = 1))
        h.runner.start(meta)
        advanceTimeBy(3_000 + 40_000); runCurrent()
        assertTrue(h.state.livePoints.size in 100..PowerRunner.LIVE_POINTS)
        assertEquals(0.8, h.state.liveW!!, 1e-9)
        assertEquals(0.2, h.state.gaugeHz!!, 1e-9)                   // one distinct reading over the 5 s probe
    }

    @Test fun aForeignGenerationDuringTheBaselineIsIgnored() = runTest {
        val h = Harness(this, settings = settings.copy(repeats = 1))
        h.runner.start(meta)
        advanceTimeBy(3_500); runCurrent()                            // in the baseline
        h.chat.observer!!.run { onSend(h.now()); onFirstToken(h.now()); onToken(64, h.now()); onDone(h.now(), 64, null, false, null) }
        advanceTimeBy(30_000); runCurrent()
        val m = h.state.runs.single().marks!!
        assertTrue(m.sendMs >= 3_000 + 2_000)                          // the runner's own send, after the baseline
        assertTrue(m.firstTokenMs!! > m.sendMs)
        assertEquals(64, m.generatedTokens)
        assertEquals(1, h.chat.sends)
    }

    @Test fun anExternalStopSkipsTheRun() = runTest {
        val h = Harness(this, settings = settings.copy(repeats = 1))
        h.chat.isLiteRt = true                                        // no cancel from the window
        h.runner.start(meta)
        advanceTimeBy(3_000 + 2_000 + 500 + 1_000); runCurrent()      // 20 tokens in
        assertTrue(h.chat.generating)
        h.chat.cancel()                                               // Chat's Stop button
        advanceTimeBy(30_000); runCurrent()
        assertEquals("stopped", h.state.runs.single().skipped)
    }

    @Test fun unloadingEndsTheSession() = runTest {
        val h = Harness(this)
        h.runner.start(meta)
        advanceTimeBy(9_500); runCurrent()                            // run 1 done, cool-down
        assertEquals(1, h.state.runs.size)
        h.chat.isLoaded = false; h.chat.loadedKey = null
        advanceTimeBy(30_000); runCurrent()
        assertEquals(1, h.state.runs.size)
        assertEquals(PowerPhase.DONE, h.state.phase)
        assertTrue(h.state.error!!, h.state.error!!.contains("unloaded"))
        assertEquals(1, h.chat.sends)
    }

    @Test fun aModelChangeEndsTheSession() = runTest {
        val h = Harness(this)
        h.runner.start(meta)
        advanceTimeBy(3_000 + 2_000 + 500 + 1_000); runCurrent()      // run 1 generating
        h.chat.loadedKey = "other-model"
        advanceTimeBy(30_000); runCurrent()
        assertEquals(PowerPhase.DONE, h.state.phase)
        assertTrue(h.state.runs.size <= 1)
        assertTrue(h.state.error!!, h.state.error!!.contains("changed"))
        assertEquals(1, h.chat.sends)
    }

    @Test fun aSilentSendIsSkippedAfterTheTimeout() = runTest {
        val h = Harness(this)
        h.chat.silent = true
        h.runner.start(meta)
        advanceTimeBy(3_000 + 2 * (2_000 + PowerRunner.SEND_TIMEOUT_MS + 1_000) + 5_000); runCurrent()
        assertEquals(PowerPhase.DONE, h.state.phase)
        assertEquals(listOf("send lost", "send lost"), h.state.runs.map { it.skipped })
    }
}
