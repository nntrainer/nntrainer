// SPDX-License-Identifier: Apache-2.0
/**
 * @file  MetricsScreen.kt
 * @brief The speed benchmark: pick a prompt and an answer length, run the
 *        bundled text a few times, read the median TTFT, prefill and
 *        decode speeds and each run's real token counts. Lengths are
 *        nominal (four characters per token); bench.csv gets every run.
 */
package com.example.quickdotaiapp.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.quickdotaiapp.LoadState
import com.example.quickdotaiapp.UiState
import com.example.quickdotaiapp.power.PackLimits
import com.example.quickdotaiapp.power.PowerPhase
import com.example.quickdotaiapp.power.PowerPrompts
import com.example.quickdotaiapp.power.PowerRun
import com.example.quickdotaiapp.power.PowerRunner
import com.example.quickdotaiapp.power.PowerSeries
import com.example.quickdotaiapp.power.PowerSettings
import com.example.quickdotaiapp.power.PowerUi
import com.example.quickdotaiapp.power.PromptSource

private fun tokens(n: Int): String = if (n < 1024) "$n" else "${n / 1024}K"

@Composable
fun MetricsScreen(
    chat: UiState,
    bench: PowerUi,
    runner: PowerRunner,
    limits: PackLimits?,
    onRun: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val s = bench.settings
    fun edit(change: PowerSettings.() -> PowerSettings) = runner.setSettings(runner.state.value.settings.change())
    val prefill = (s.prompt as? PromptSource.Bench)?.tokens ?: 1024
    val measuring = bench.phase.measuring
    val loaded = chat.load is LoadState.Loaded
    val fits = limits == null || (prefill <= limits.promptTokens && s.windowTokens <= limits.decodeTokens)
    Column(
        modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 16.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        Text("Benchmark", style = MaterialTheme.typography.titleLarge)
        Text(
            "Sends a bundled text of about the chosen length and stops the answer after the chosen number of tokens. " +
                "Lengths are nominal; the runs show what the engine counted.",
            style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Column(
            Modifier.fillMaxWidth().clip(RoundedCornerShape(16.dp)).background(MaterialTheme.colorScheme.surfaceContainer).padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Choice("PREFILL TOKENS", PowerPrompts.benchPrefill.map { it to tokens(it) }, prefill,
                enabled = { !measuring && (limits == null || it <= limits.promptTokens) }) {
                edit { copy(prompt = PromptSource.Bench(it)) }
            }
            Choice("DECODE TOKENS", PowerPrompts.benchDecode.map { it to "$it" }, s.windowTokens,
                enabled = { !measuring && (limits == null || it <= limits.decodeTokens) }) {
                edit { copy(windowTokens = it) }
            }
            Choice("REPEATS", listOf(1, 3, 5).map { it to "$it" }, s.repeats, enabled = { !measuring }) {
                edit { copy(repeats = it) }
            }
            limits?.let {
                Text("This pack: prompts up to ${it.promptTokens} tokens, answers up to ${it.decodeTokens}",
                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            HorizontalDivider()
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("Wait for a cool phone", style = MaterialTheme.typography.bodyMedium)
                    Text("Up to 90 s before each run until the thermal status is normal", style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                Switch(checked = s.thermalGate, enabled = !measuring, onCheckedChange = { on -> edit { copy(thermalGate = on) } })
            }
        }
        if (measuring) {
            OutlinedButton(onClick = runner::stop, modifier = Modifier.fillMaxWidth().height(52.dp).testTag(Tags.BENCH_STOP)) {
                Text("Stop", style = MaterialTheme.typography.titleMedium)
            }
            val phase = when (bench.phase) {
                PowerPhase.THERMAL_GATE -> if (s.thermalGate) "waiting for the phone to cool" else "starting"
                PowerPhase.PREFILL -> "prefill"
                PowerPhase.DECODE -> "decode · ${bench.liveTokens} tokens"
                PowerPhase.TAIL -> "stopping the answer"
                else -> "starting"
            }
            Text("Run ${bench.runIndex} of ${s.repeats} · $phase", style = MaterialTheme.typography.bodyMedium)
        } else {
            Button(onClick = onRun, enabled = loaded && !chat.busy && fits,
                modifier = Modifier.fillMaxWidth().height(52.dp).testTag(Tags.BENCH_RUN)) {
                Text("Run", style = MaterialTheme.typography.titleMedium)
            }
            if (!loaded) Text("Load a model first", style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        bench.error?.let { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodyMedium) }
        if (bench.runs.isNotEmpty()) Results(bench)
    }
}

@Composable
private fun Results(bench: PowerUi) {
    val series = PowerSeries(bench.runs)
    Text("Median of ${series.speedRuns.size} run(s)", style = MaterialTheme.typography.titleSmall)
    Tile("TIME TO FIRST TOKEN", fmt(series.medianTtftMs?.div(1000.0), "%.2f"), "s", "send → first token in the app", big = true,
        bg = MaterialTheme.colorScheme.primaryContainer, fg = MaterialTheme.colorScheme.onPrimaryContainer)
    Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
        Tile("PREFILL", fmt(series.medianEnginePrefillTps, "%.0f"), "tok/s", "engine timing", modifier = Modifier.weight(1f))
        Tile("DECODE", fmt(series.medianTokensPerSec, "%.1f"), "tok/s", "first token → last, wall clock", modifier = Modifier.weight(1f))
    }
    Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp)).background(MaterialTheme.colorScheme.surfaceContainer)) {
        RunCells(listOf("#", "prompt", "answer", "TTFT", "prefill", "decode"), bold = true)
        bench.runs.forEach { r ->
            HorizontalDivider()
            RunCells(listOf(
                r.index.toString(), r.marks?.promptTokens?.toString() ?: "—", r.marks?.generatedTokens?.toString() ?: "—",
                fmt(r.ttftMs?.div(1000.0), "%.2f s"), fmt(r.enginePrefillTps, "%.0f"), fmt(r.tokensPerSec, "%.1f"),
            ))
            note(r)?.let {
                Text(it, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.error,
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp))
            }
        }
    }
    bench.savedPath?.let {
        Text("Saved to $it", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

/** Why a run's numbers are not what was asked: skipped, or the answer ended before the window. */
private fun note(r: PowerRun): String? = when {
    r.skipped != null -> "skipped: ${r.skipped}"
    r.marks != null && r.marks.generatedTokens < r.marks.windowTokens ->
        "the answer ended after ${r.marks.generatedTokens} of ${r.marks.windowTokens} tokens"
    r.tokensPerSec == null -> "decode under a second: no speed"
    else -> null
}

@Composable
private fun RunCells(cells: List<String>, bold: Boolean = false) {
    Row(Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 6.dp)) {
        cells.forEachIndexed { i, t ->
            Text(t, style = MaterialTheme.typography.labelSmall, fontFamily = FontFamily.Monospace,
                fontWeight = if (bold) FontWeight.Bold else FontWeight.Normal, maxLines = 1,
                modifier = Modifier.weight(if (i == 0) 0.08f else 0.184f))
        }
    }
}

@Composable
private fun Tile(
    label: String, value: String, unit: String, caption: String,
    big: Boolean = false, modifier: Modifier = Modifier,
    bg: Color = MaterialTheme.colorScheme.surfaceContainer, fg: Color = MaterialTheme.colorScheme.onSurface,
) {
    Column(modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(bg).padding(16.dp)) {
        Text(label, style = MaterialTheme.typography.labelSmall, color = fg.copy(alpha = 0.8f))
        Row(verticalAlignment = Alignment.Bottom) {
            Text(value, fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold, fontSize = if (big) 36.sp else 24.sp, color = fg)
            Text(" $unit", style = MaterialTheme.typography.labelMedium, color = fg, modifier = Modifier.padding(bottom = 6.dp))
        }
        Text(caption, style = MaterialTheme.typography.labelSmall, color = fg.copy(alpha = 0.85f))
    }
}
