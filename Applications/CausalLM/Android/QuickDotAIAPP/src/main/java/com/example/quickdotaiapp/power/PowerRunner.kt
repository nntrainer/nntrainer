// SPDX-License-Identifier: Apache-2.0
/**
 * @file  PowerRunner.kt
 * @brief The measurement session (spec §3): wait for the cable, then per
 *        run a thermal gate, a baseline, one prompt through the chat,
 *        the decode window, a cool-down; samples all the while; the
 *        numbers of each run from PowerMath; the log to a callback at the
 *        end. Pure Kotlin over an injected clock, delay and sampler.
 */
package com.example.quickdotaiapp.power

import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeoutOrNull

class PowerRunner(
    private val scope: CoroutineScope,
    private val chat: PowerChat,
    private val sampler: () -> PowerSample,
    private val prompts: (PromptSource) -> String?,
    private val onSessionDone: (PowerSessionLog) -> String?,
    private val now: () -> Long = System::currentTimeMillis,
    private val samplePeriodMs: Long = 250,
) : GenerationObserver {
    private val _state = MutableStateFlow(PowerUi())
    val state: StateFlow<PowerUi> = _state.asStateFlow()

    private val lock = Any()
    private val samples = mutableListOf<PowerSample>()      // guarded by lock
    private var run: RunState? = null                        // guarded by lock
    private var session: Job? = null
    private var sampling: Job? = null
    private var meta: SessionMeta? = null
    private var sessionKey: String? = null
    /** Why the session ended early, shown as the error of the result. */
    @Volatile private var endReason: String? = null

    /** The run in progress; the engine thread writes the marks, the session coroutine reads them. */
    private class RunState(val windowTokens: Int, val baselineStartMs: Long) {
        var sendMs: Long? = null
        var firstTokenMs: Long? = null
        var windowEndMs: Long? = null
        var doneMs: Long? = null
        var generated = 0
        var promptTokens: Int? = null
        var stopped = false
        var error: String? = null
        var cancelSent = false
        /** Set right before the runner's own sendPrompt; onSend is accepted only while this is true. */
        var awaitingSend = false
        val done = CompletableDeferred<Unit>()
    }

    fun setSettings(settings: PowerSettings) = _state.update { it.copy(settings = settings) }

    fun start(meta: SessionMeta) {
        if (session?.isActive == true) return
        val settings = state.value.settings
        val prompt = prompts(settings.prompt)?.takeIf { it.isNotBlank() }
        if (prompt == null) {
            _state.update { it.copy(error = if (settings.prompt is PromptSource.Composer) "Composer is empty" else "Prompt preset missing") }
            return
        }
        if (!chat.isLoaded) { _state.update { it.copy(error = "Load a model first") }; return }
        this.meta = meta
        sessionKey = chat.loadedKey
        endReason = null
        synchronized(lock) { samples.clear(); run = null }
        _state.update {
            PowerUi(settings = it.settings, phase = PowerPhase.WAIT_UNPLUG, plugged = it.plugged, thermalStatus = it.thermalStatus)
        }
        chat.observer = this
        sampling = scope.launch { sampleLoop() }
        session = scope.launch {
            try { runSession(settings, prompt) } finally { finish() }
        }
    }

    /** Ends the session now: the run in progress is dropped, earlier runs are kept and written. */
    fun stop() {
        session?.cancel()
    }

    // ----- sampling -----------------------------------------------------------

    private suspend fun sampleLoop() {
        val probeStart = now()
        val distinct = HashSet<Int>()
        var probed = false
        while (true) {
            val s = sampler()
            val points = synchronized(lock) {
                samples += s
                samples.subList(maxOf(0, samples.size - LIVE_POINTS), samples.size).map { it.watts }
            }
            if (!probed) {
                distinct += s.microAmps
                if (now() - probeStart >= GAUGE_PROBE_MS) {
                    probed = true
                    _state.update { it.copy(gaugeHz = distinct.size / (GAUGE_PROBE_MS / 1000.0)) }
                }
            }
            _state.update { it.copy(liveW = s.watts, livePoints = points, plugged = s.plugged, thermalStatus = s.thermalStatus) }
            delay(samplePeriodMs)
        }
    }

    private fun latest(): PowerSample? = synchronized(lock) { samples.lastOrNull() }
    private fun snapshot(): List<PowerSample> = synchronized(lock) { samples.toList() }

    /** False once the model was unloaded or another one loaded; [whyNotLoaded] says which. */
    private fun stillLoaded(): Boolean = chat.isLoaded && chat.loadedKey == sessionKey
    private fun whyNotLoaded(): String = if (!chat.isLoaded) "Model unloaded; session ended" else "Model changed; session ended"

    private fun cool(): Boolean {
        val s = latest() ?: return false
        return s.thermalStatus == 0 && (s.headroom.isNaN() || s.headroom < GATE_HEADROOM)
    }

    // ----- the session ----------------------------------------------------------

    private suspend fun runSession(settings: PowerSettings, prompt: String) {
        while (!settings.allowPlugged && latest()?.plugged != false) delay(samplePeriodMs)
        delay(SETTLE_MS)
        val runs = mutableListOf<PowerRun>()
        for (i in 1..settings.repeats) {
            _state.update {
                it.copy(runIndex = i, phase = PowerPhase.THERMAL_GATE, liveBaselineW = null, livePrefillW = null,
                    liveDecodeW = null, liveTokens = 0, liveMs = 0)
            }
            val gateStart = now()
            while (!cool() && now() - gateStart < GATE_MAX_MS) delay(GATE_POLL_MS)
            if (!cool()) {
                runs += skipped(i, "hot")
                _state.update { it.copy(runs = runs.toList()) }
                continue
            }
            if (!stillLoaded()) { endReason = whyNotLoaded(); return }
            chat.clearConversation()
            val rs = RunState(settings.windowTokens, now())
            synchronized(lock) { run = rs }
            _state.update { it.copy(phase = PowerPhase.BASELINE) }
            delay(settings.baselineSec * 1000L)
            val baselineW = PowerMath.median(snapshot().filter { it.tMs > rs.baselineStartMs }.map { it.watts })
            _state.update { it.copy(liveBaselineW = baselineW) }
            if (!stillLoaded()) { endReason = whyNotLoaded(); return }

            synchronized(lock) { rs.awaitingSend = true }
            val sentAt = now()
            if (!chat.sendPrompt(prompt)) {
                synchronized(lock) { run = null }
                runs += skipped(i, "send refused")
                _state.update { it.copy(runs = runs.toList()) }
                continue
            }
            // Wait for the generation. A cable plugged back in or a model change ends the run and the
            // session; a send that never reports onSend is treated as lost after SEND_TIMEOUT_MS.
            var replugged = false
            var ended: String? = null
            var lost = false
            while (!rs.done.isCompleted) {
                if (!settings.allowPlugged && latest()?.plugged == true) { replugged = true; chat.cancel(); break }
                if (!stillLoaded()) { ended = whyNotLoaded(); chat.cancel(); break }
                if (synchronized(lock) { rs.sendMs } == null && now() - sentAt >= SEND_TIMEOUT_MS) { lost = true; break }
                delay(samplePeriodMs)
            }
            if (lost) {
                synchronized(lock) { run = null }
                runs += skipped(i, "send lost")
                _state.update { it.copy(runs = runs.toList()) }
                continue
            }
            withTimeoutOrNull(DONE_WAIT_MS) { rs.done.await() }
            synchronized(lock) { run = null }
            val marks = RunMarks(
                baselineStartMs = rs.baselineStartMs, sendMs = rs.sendMs ?: now(), firstTokenMs = rs.firstTokenMs,
                windowEndMs = rs.windowEndMs, doneMs = rs.doneMs, windowTokens = settings.windowTokens,
                generatedTokens = rs.generated, promptTokens = rs.promptTokens, stopped = rs.stopped,
            )
            var result = PowerMath.evaluate(i, marks, snapshot())
            when {
                replugged -> result = result.copy(skipped = "plugged")
                ended != null -> result = result.copy(skipped = if (chat.isLoaded) "model changed" else "model unloaded")
                rs.error != null -> result = result.copy(skipped = "error: ${rs.error}")
                // Stopped by something other than the window (Chat's Stop, an unload): not a measurement.
                rs.stopped && !rs.cancelSent -> result = result.copy(skipped = "stopped")
            }
            runs += result
            _state.update { it.copy(runs = runs.toList(), livePrefillW = result.prefillW, liveDecodeW = result.decodeW) }
            if (replugged) return
            if (ended != null) { endReason = ended; return }
            if (i < settings.repeats) {
                _state.update { it.copy(phase = PowerPhase.COOLDOWN) }
                delay(settings.cooldownSec * 1000L)
            }
        }
    }

    private fun skipped(index: Int, reason: String) =
        PowerRun(index, null, reason, null, null, null, null, null, null, null, null, null, null, null, null, 0, false)

    /** Runs when the session ends for any reason: cancel a generation in flight, stop sampling, write the log. */
    private fun finish() {
        val inFlight = synchronized(lock) { run.also { run = null } }
        if (inFlight != null && !inFlight.done.isCompleted) chat.cancel()
        chat.observer = null
        sampling?.cancel()
        val runs = state.value.runs
        val m = meta
        val log = if (runs.isNotEmpty() && m != null) PowerSessionLog(m, state.value.settings, state.value.gaugeHz, runs) else null
        val path = log?.let { l -> runCatching { onSessionDone(l) }.getOrNull() }
        _state.update { it.copy(phase = PowerPhase.DONE, savedPath = path, log = log, error = endReason) }
    }

    // ----- generation events (engine thread) ------------------------------------

    // A generation the runner did not send (a chat turn during the baseline) is ignored: onSend is
    // accepted only right after the runner's own sendPrompt, the rest only once sendMs is known.
    override fun onSend(tMs: Long) {
        synchronized(lock) {
            val rs = run ?: return
            if (!rs.awaitingSend) return
            rs.awaitingSend = false
            rs.sendMs = tMs
        }
        _state.update { it.copy(phase = PowerPhase.PREFILL) }
    }

    override fun onFirstToken(tMs: Long) {
        synchronized(lock) {
            val rs = run ?: return
            if (rs.sendMs == null) return
            rs.firstTokenMs = tMs
        }
        _state.update { it.copy(phase = PowerPhase.DECODE) }
    }

    override fun onToken(count: Int, tMs: Long) {
        var cancel = false
        var first: Long? = null
        synchronized(lock) {
            val rs = run ?: return
            if (rs.sendMs == null) return
            rs.generated = count
            first = rs.firstTokenMs
            if (count >= rs.windowTokens && rs.windowEndMs == null) {
                rs.windowEndMs = tMs
                if (!chat.isLiteRt && !rs.cancelSent) { rs.cancelSent = true; cancel = true }
            }
        }
        if (cancel) {
            chat.cancel()
            _state.update { it.copy(phase = PowerPhase.TAIL, liveTokens = count, liveMs = tMs - (first ?: tMs)) }
        } else {
            _state.update { it.copy(liveTokens = count, liveMs = tMs - (first ?: tMs)) }
        }
    }

    override fun onDone(tMs: Long, tokens: Int, promptTokens: Int?, stopped: Boolean, error: String?) {
        val rs = synchronized(lock) {
            val rs = run ?: return
            if (rs.sendMs == null) return
            rs.doneMs = tMs
            if (tokens > rs.generated) rs.generated = tokens
            if (rs.windowEndMs == null) rs.windowEndMs = tMs
            rs.promptTokens = promptTokens?.takeIf { it > 0 }     // LiteRT reports 0: unknown, not zero
            rs.stopped = stopped
            rs.error = error
            rs
        }
        rs.done.complete(Unit)
    }

    companion object {
        const val SETTLE_MS = 3_000L
        const val GATE_MAX_MS = 90_000L
        const val GATE_POLL_MS = 500L
        /** A send that produces no onSend within this time is treated as lost. */
        const val SEND_TIMEOUT_MS = 10_000L
        /** How long to wait for the engine after a cancel before the run is evaluated anyway. */
        const val DONE_WAIT_MS = 10_000L
        const val GATE_HEADROOM = 0.80f
        const val GAUGE_PROBE_MS = 5_000L
        const val LIVE_POINTS = 120
    }
}
