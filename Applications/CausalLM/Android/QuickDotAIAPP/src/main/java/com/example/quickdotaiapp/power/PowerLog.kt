// SPDX-License-Identifier: Apache-2.0
/**
 * @file  PowerLog.kt
 * @brief What a measurement session leaves behind: one CSV row per run
 *        (spec §6 columns) and the whole session as JSON.
 */
package com.example.quickdotaiapp.power

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.util.Locale

@Serializable
data class SessionMeta(
    val startedAt: String,
    val device: String,
    val android: String,
    val modelId: String,
    val runtime: String,
    val backend: String,
)

@Serializable
data class PowerSessionLog(
    val meta: SessionMeta,
    val settings: PowerSettings,
    val gaugeHz: Double?,
    val runs: List<PowerRun>,
)

object PowerLog {
    const val CSV_HEADER = "time,model_id,runtime,backend,prompt,prompt_tokens,window_tokens,gen_tokens,tok_s," +
        "prefill_w,decode_w,baseline_w,peak_w,mj_per_token,run_energy_j,bat_c_start,bat_c_end," +
        "npu_c_max,cpu_c_max,gpu_c_max,thermal_status_max,throttled,skipped"

    private val json = Json { encodeDefaults = true; classDiscriminator = "type"; allowSpecialFloatingPointValues = true }

    private fun num(v: Double?, pattern: String): String = v?.let { String.format(Locale.US, pattern, it) } ?: ""
    private fun num(v: Float?, pattern: String): String = v?.let { String.format(Locale.US, pattern, it) } ?: ""

    fun csvRow(log: PowerSessionLog, run: PowerRun): String {
        val m = run.marks
        return listOf(
            log.meta.startedAt, log.meta.modelId, log.meta.runtime, log.meta.backend, log.settings.prompt.label(),
            m?.promptTokens?.toString() ?: "", m?.windowTokens?.toString() ?: "", m?.generatedTokens?.toString() ?: "",
            num(run.tokensPerSec, "%.2f"),
            num(run.prefillW, "%.3f"), num(run.decodeW, "%.3f"), num(run.baselineW, "%.3f"), num(run.peakW, "%.3f"),
            num(run.mJPerToken, "%.1f"), num(run.runEnergyJ, "%.2f"),
            num(run.batteryStartC, "%.1f"), num(run.batteryEndC, "%.1f"),
            num(run.npuMaxC, "%.1f"), num(run.cpuMaxC, "%.1f"), num(run.gpuMaxC, "%.1f"),
            run.thermalStatusMax.toString(), run.throttled.toString(),
            run.skipped?.replace(',', ';')?.replace('\n', ' ') ?: "",
        ).joinToString(",")
    }

    fun csv(log: PowerSessionLog): String =
        (listOf(CSV_HEADER) + log.runs.map { csvRow(log, it) }).joinToString("\n") + "\n"

    fun toJson(log: PowerSessionLog): String = json.encodeToString(PowerSessionLog.serializer(), log)
    fun fromJson(text: String): PowerSessionLog = json.decodeFromString(PowerSessionLog.serializer(), text)
}
