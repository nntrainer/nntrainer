// SPDX-License-Identifier: Apache-2.0
/**
 * @file  PowerTypes.kt
 * @brief Value types of the Power tab: one gauge sample, the settings of a
 *        session, the marks and numbers of a run, the series and the UI
 *        state. Pure Kotlin; what goes into the session JSON is serializable.
 */
package com.example.quickdotaiapp.power

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlin.math.abs

/** PowerManager.THERMAL_STATUS_MODERATE: from here the OS throttles noticeably. */
const val THERMAL_MODERATE = 2

enum class PowerPhase { IDLE, WAIT_UNPLUG, THERMAL_GATE, BASELINE, PREFILL, DECODE, TAIL, COOLDOWN, DONE }

/** One reading of the battery gauge plus the thermal state at that moment. */
@Serializable
data class PowerSample(
    val tMs: Long,
    val microAmps: Int,
    val milliVolts: Int,
    val batteryTenthsC: Int,
    val plugged: Boolean,
    val thermalStatus: Int,
    val headroom: Float,
    val npuC: Float?,
    val cpuC: Float?,
    val gpuC: Float?,
) {
    /** Whole-device draw; the sign of the current only says charging or discharging. */
    val watts: Double get() = abs(microAmps.toDouble()) * milliVolts / 1e9
}

@Serializable
sealed interface PromptSource {
    @Serializable @SerialName("composer") data object Composer : PromptSource
    @Serializable @SerialName("fixed") data class Fixed(val tokens: Int) : PromptSource
    /** The validated HTP summarization prompt (res/raw/gemma4_moe_messages.json, about 1K tokens). */
    @Serializable @SerialName("htp-summary") data object HtpSummary : PromptSource
}

fun PromptSource.label(): String = when (this) {
    PromptSource.Composer -> "composer"
    is PromptSource.Fixed -> "fixed-$tokens"
    PromptSource.HtpSummary -> "htp-summary"
}

@Serializable
data class PowerSettings(
    val prompt: PromptSource = PromptSource.Fixed(1024),
    val windowTokens: Int = 512,
    val repeats: Int = 3,
    val baselineSec: Int = 10,
    val cooldownSec: Int = 5,
    /** Debug only: measure with the cable attached; the numbers include charging current. */
    val allowPlugged: Boolean = false,
)

/** The moments of one run on the sample clock. */
@Serializable
data class RunMarks(
    val baselineStartMs: Long,
    val sendMs: Long,
    val firstTokenMs: Long?,
    val windowEndMs: Long?,
    val doneMs: Long?,
    val windowTokens: Int,
    val generatedTokens: Int,
    val promptTokens: Int?,
    val stopped: Boolean,
)

/** One run's numbers (spec §4); null where the gauge gave too little to compute. */
@Serializable
data class PowerRun(
    val index: Int,
    val marks: RunMarks?,
    val skipped: String?,
    val baselineW: Double?,
    val prefillW: Double?,
    val decodeW: Double?,
    val peakW: Double?,
    val mJPerToken: Double?,
    val runEnergyJ: Double?,
    val tokensPerSec: Double?,
    val batteryStartC: Double?,
    val batteryEndC: Double?,
    val npuMaxC: Float?,
    val cpuMaxC: Float?,
    val gpuMaxC: Float?,
    val thermalStatusMax: Int,
    val throttled: Boolean,
    val samples: List<PowerSample> = emptyList(),
)

data class PowerSeries(val runs: List<PowerRun>) {
    val valid: List<PowerRun> get() = runs.filter { it.skipped == null && it.mJPerToken != null }
    val medianMJPerToken: Double? get() = PowerMath.median(valid.map { it.mJPerToken!! })
    val throttledRuns: Int get() = runs.count { it.throttled }
    val skippedRuns: Int get() = runs.count { it.skipped != null }
}

/** What the Power screen renders. */
data class PowerUi(
    val settings: PowerSettings = PowerSettings(),
    val phase: PowerPhase = PowerPhase.IDLE,
    /** 1-based index of the run in progress while measuring. */
    val runIndex: Int = 0,
    val plugged: Boolean = true,
    val thermalStatus: Int = 0,
    val liveW: Double? = null,
    val liveTokens: Int = 0,
    val liveMs: Long = 0,
    /** The last few watts for the sparkline. */
    val livePoints: List<Double> = emptyList(),
    val liveBaselineW: Double? = null,
    val livePrefillW: Double? = null,
    val liveDecodeW: Double? = null,
    val runs: List<PowerRun> = emptyList(),
    val savedPath: String? = null,
    /** Distinct current readings per second, measured once at the start of a session. */
    val gaugeHz: Double? = null,
    /** The finished session, for Copy CSV. */
    val log: PowerSessionLog? = null,
    val error: String? = null,
)
