// SPDX-License-Identifier: Apache-2.0
/**
 * @file  PowerScreen.kt
 * @brief The Power tab in its three states: setup (prompt, window,
 *        repeats, baseline, Measure), measuring (phase steps, live watts,
 *        sparkline, Stop) and result (energy per token, the four numbers
 *        it is made of, the runs, where the files went, Copy CSV).
 */
package com.example.quickdotaiapp.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.quickdotaiapp.LoadState
import com.example.quickdotaiapp.R
import com.example.quickdotaiapp.UiState
import com.example.quickdotaiapp.power.PowerLog
import com.example.quickdotaiapp.power.PowerMath
import com.example.quickdotaiapp.power.PowerPhase
import com.example.quickdotaiapp.power.PowerPrompts
import com.example.quickdotaiapp.power.PowerRun
import com.example.quickdotaiapp.power.PowerRunner
import com.example.quickdotaiapp.power.PowerSeries
import com.example.quickdotaiapp.power.PowerUi
import com.example.quickdotaiapp.power.PromptSource
import java.util.Locale

private fun fmt(v: Double?, pattern: String): String = v?.let { String.format(Locale.US, pattern, it) } ?: "—"

internal val PowerPhase.measuring: Boolean get() = this != PowerPhase.IDLE && this != PowerPhase.DONE

@Composable
fun PowerScreen(
    chat: UiState,
    power: PowerUi,
    runner: PowerRunner,
    onMeasure: () -> Unit,
    onCopy: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    var showSetup by rememberSaveable { mutableStateOf(false) }
    val measuring = power.phase.measuring
    // Back to the result only once a session really started; a refused start stays on the setup with its error.
    LaunchedEffect(measuring) { if (measuring) showSetup = false }
    when {
        measuring -> Measuring(power, runner, modifier)
        power.phase == PowerPhase.DONE && power.runs.isNotEmpty() && !showSetup ->
            Result(power, onAgain = { showSetup = true }, onCopy = onCopy, modifier = modifier)
        else -> Setup(chat, power, runner, onMeasure = onMeasure, modifier = modifier)
    }
}

// ----- setup --------------------------------------------------------------------

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun Setup(chat: UiState, power: PowerUi, runner: PowerRunner, onMeasure: () -> Unit, modifier: Modifier) {
    val s = power.settings
    fun edit(change: com.example.quickdotaiapp.power.PowerSettings.() -> com.example.quickdotaiapp.power.PowerSettings) =
        runner.setSettings(runner.state.value.settings.change())
    val loaded = chat.load is LoadState.Loaded
    Column(
        modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 16.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        Text("Power", style = MaterialTheme.typography.titleLarge)
        Text(
            "Measures whole-device power from the battery gauge while the model answers. You will be asked to unplug the cable.",
            style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Column(
            Modifier.fillMaxWidth().clip(RoundedCornerShape(16.dp)).background(MaterialTheme.colorScheme.surfaceContainer).padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Choice("PROMPT", listOf<Pair<PromptSource, String>>(PromptSource.Composer to "Composer") +
                PowerPrompts.sizes.map { PromptSource.Fixed(it) to (if (it < 1024) "$it" else "${it / 1024}K") },
                s.prompt) { edit { copy(prompt = it) } }
            Choice("WINDOW TOKENS", listOf(64, 512, 1024).map { it to "$it" }, s.windowTokens) { edit { copy(windowTokens = it) } }
            Choice("REPEATS", listOf(1, 2, 3, 5).map { it to "$it" }, s.repeats) { edit { copy(repeats = it) } }
            Choice("BASELINE", listOf(5, 10, 20).map { it to "$it s" }, s.baselineSec) { edit { copy(baselineSec = it) } }
            HorizontalDivider()
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("Measure while plugged", style = MaterialTheme.typography.bodyMedium)
                    Text("Debug only: charging current makes the numbers invalid", style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                Switch(checked = s.allowPlugged, onCheckedChange = { on -> edit { copy(allowPlugged = on) } },
                    modifier = Modifier.testTag(Tags.POWER_ALLOW_PLUGGED))
            }
        }
        Button(
            onClick = onMeasure,
            enabled = loaded && !chat.busy,
            modifier = Modifier.fillMaxWidth().height(52.dp).testTag(Tags.POWER_MEASURE),
        ) {
            Icon(painterResource(R.drawable.ic_bolt), contentDescription = null)
            Spacer(Modifier.width(8.dp))
            Text("Measure", style = MaterialTheme.typography.titleMedium)
        }
        val last = PowerSeries(power.runs).medianMJPerToken
        Text(
            when {
                !loaded -> "Load a model first"
                last != null -> "About a minute per run · last result ${fmt(last, "%.0f")} mJ/tok"
                else -> "About a minute per run"
            },
            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.fillMaxWidth(), textAlign = androidx.compose.ui.text.style.TextAlign.Center,
        )
        power.error?.let { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodyMedium) }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun <T> Choice(label: String, options: List<Pair<T, String>>, selected: T, onSelect: (T) -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text(label, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            options.forEach { (value, text) ->
                FilterChip(selected = value == selected, onClick = { onSelect(value) }, label = { Text(text, maxLines = 1) })
            }
        }
    }
}

// ----- measuring ------------------------------------------------------------------

private fun rank(phase: PowerPhase): Int = when (phase) {
    PowerPhase.BASELINE -> 0
    PowerPhase.PREFILL -> 1
    PowerPhase.DECODE, PowerPhase.TAIL -> 2
    PowerPhase.COOLDOWN -> 3
    else -> -1
}

@Composable
private fun Measuring(power: PowerUi, runner: PowerRunner, modifier: Modifier) {
    val title = when (power.phase) {
        PowerPhase.WAIT_UNPLUG -> "Unplug the cable"
        PowerPhase.THERMAL_GATE -> "Waiting for the phone to cool"
        else -> "Run ${power.runIndex} of ${power.settings.repeats}"
    }
    val current = rank(power.phase)
    val tps = if (power.liveMs > 0) power.liveTokens * 1000.0 / power.liveMs else 0.0
    Column(modifier.fillMaxSize().padding(horizontal = 16.dp, vertical = 12.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
        Text(title, style = MaterialTheme.typography.titleLarge)
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            listOf("Baseline", "Prefill", "Decode", "Cool-down").forEachIndexed { i, name ->
                val now = i == current
                Text(
                    (if (i <= current) "● " else "○ ") + name,
                    style = MaterialTheme.typography.bodySmall,
                    fontWeight = if (now) FontWeight.Bold else FontWeight.Normal,
                    color = if (now) MaterialTheme.colorScheme.onSurface else MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
        Column(
            Modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(MaterialTheme.colorScheme.primaryContainer).padding(18.dp, 16.dp, 18.dp, 14.dp),
        ) {
            val fg = MaterialTheme.colorScheme.onPrimaryContainer
            Text("NOW", style = MaterialTheme.typography.labelSmall, color = fg.copy(alpha = 0.8f))
            Row(verticalAlignment = Alignment.Bottom) {
                Text(fmt(power.liveW, "%.1f"), fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold, fontSize = 48.sp, color = fg, lineHeight = 52.sp)
                Text(" W", style = MaterialTheme.typography.titleMedium, color = fg, modifier = Modifier.padding(bottom = 8.dp))
            }
            Text(
                when (power.phase) {
                    PowerPhase.WAIT_UNPLUG -> if (power.plugged) "cable attached · starts 3 s after it is unplugged" else "starting…"
                    PowerPhase.THERMAL_GATE -> "thermal status ${power.thermalStatus} · waiting up to 90 s"
                    PowerPhase.BASELINE -> "idle with the model loaded"
                    PowerPhase.PREFILL -> "prompt sent · waiting for the first token"
                    PowerPhase.DECODE -> "${power.liveTokens} tokens · ${fmt(tps, "%.1f")} tok/s · ${fmt(power.liveMs / 1000.0, "%.1f")} s"
                    PowerPhase.TAIL -> "window closed at ${power.liveTokens} tokens · engine finishing"
                    PowerPhase.COOLDOWN -> "cooling down"
                    else -> ""
                },
                style = MaterialTheme.typography.bodyMedium, color = fg,
            )
        }
        Sparkline(power.livePoints, power.liveBaselineW)
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            listOf("baseline" to power.liveBaselineW, "prefill" to power.livePrefillW, "decode" to power.liveDecodeW).forEach { (n, w) ->
                Text("$n ${fmt(w, "%.1f")} W", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
        Spacer(Modifier.weight(1f))
        Button(
            onClick = runner::stop,
            colors = ButtonDefaults.buttonColors(
                containerColor = MaterialTheme.colorScheme.errorContainer, contentColor = MaterialTheme.colorScheme.onErrorContainer),
            modifier = Modifier.fillMaxWidth().height(52.dp).testTag(Tags.POWER_STOP),
        ) { Icon(painterResource(R.drawable.ic_stop), contentDescription = null); Spacer(Modifier.width(8.dp)); Text("Stop", style = MaterialTheme.typography.titleMedium) }
    }
}

@Composable
private fun Sparkline(points: List<Double>, baseline: Double?) {
    val line = MaterialTheme.colorScheme.primary
    val guide = MaterialTheme.colorScheme.outlineVariant
    val bg = MaterialTheme.colorScheme.surfaceContainer
    Canvas(Modifier.fillMaxWidth().height(84.dp).clip(RoundedCornerShape(12.dp)).background(bg)) {
        if (points.size < 2) return@Canvas
        val maxW = maxOf(6.0, points.max())
        val pad = 6.dp.toPx()
        fun y(w: Double): Float = (size.height - pad) - (w / maxW * (size.height - 2 * pad)).toFloat()
        baseline?.let { drawLine(guide, Offset(0f, y(it)), Offset(size.width, y(it)), strokeWidth = 1.dp.toPx()) }
        val stepX = size.width / (points.size - 1)
        val path = Path().apply { points.forEachIndexed { i, w -> if (i == 0) moveTo(0f, y(w)) else lineTo(i * stepX, y(w)) } }
        drawPath(path, line, style = Stroke(width = 2.5.dp.toPx(), join = StrokeJoin.Round))
    }
}

// ----- result ----------------------------------------------------------------------

@Composable
private fun Result(power: PowerUi, onAgain: () -> Unit, onCopy: (String) -> Unit, modifier: Modifier) {
    val series = PowerSeries(power.runs)
    val valid = series.valid
    fun med(pick: (PowerRun) -> Double?) = PowerMath.median(valid.mapNotNull(pick))
    val tps = med { it.tokensPerSec }
    val runSec = med { r -> r.marks?.let { m -> m.windowEndMs?.let { (it - m.sendMs) / 1000.0 } } }
    Column(
        modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 16.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Result", style = MaterialTheme.typography.titleLarge, modifier = Modifier.weight(1f))
            TextButton(onClick = onAgain, modifier = Modifier.testTag(Tags.POWER_AGAIN)) { Text("Measure again") }
        }
        Tile("ENERGY PER TOKEN", fmt(series.medianMJPerToken, "%.0f"), "mJ/tok",
            "decode power above baseline ÷ ${fmt(tps, "%.1f")} tok/s · median of ${valid.size} run${if (valid.size == 1) "" else "s"}" +
                (if (series.throttledRuns > 0) " · ${series.throttledRuns} throttled" else ""),
            big = true, bg = MaterialTheme.colorScheme.primaryContainer, fg = MaterialTheme.colorScheme.onPrimaryContainer)
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            Tile("DECODE", fmt(med { it.decodeW }, "%.1f"), "W", null, modifier = Modifier.weight(1f))
            Tile("PREFILL", fmt(med { it.prefillW }, "%.1f"), "W", null, modifier = Modifier.weight(1f))
        }
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            Tile("BASELINE", fmt(med { it.baselineW }, "%.1f"), "W", null, modifier = Modifier.weight(1f))
            Tile("SPEED", fmt(tps, "%.1f"), "tok/s", null, modifier = Modifier.weight(1f))
        }
        Text(
            "Run energy ${fmt(med { it.runEnergyJ }, "%.0f")} J · battery ${fmt(valid.firstOrNull()?.batteryStartC, "%.1f")} → " +
                "${fmt(valid.lastOrNull()?.batteryEndC, "%.1f")} °C · ${fmt(runSec, "%.1f")} s per run" +
                (valid.mapNotNull { it.npuMaxC }.maxOrNull()?.let { " · NPU ${fmt(it.toDouble(), "%.0f")} °C" } ?: "") +
                (if (power.gaugeHz != null) " · gauge ${fmt(power.gaugeHz, "%.1f")} Hz" else ""),
            style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(16.dp)).background(MaterialTheme.colorScheme.surfaceContainer).padding(horizontal = 14.dp)) {
            power.runs.asReversed().forEachIndexed { i, r ->
                Row(Modifier.fillMaxWidth().padding(vertical = 10.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text("Run ${r.index}", style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(0.75f))
                    if (r.skipped != null) {
                        Text(r.skipped, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.error, modifier = Modifier.weight(3.45f))
                    } else {
                        Text("${fmt(r.mJPerToken, "%.0f")} mJ/tok", style = MaterialTheme.typography.bodyMedium, fontFamily = FontFamily.Monospace, maxLines = 1, modifier = Modifier.weight(1.3f))
                        Text("${fmt(r.decodeW, "%.1f")} W", style = MaterialTheme.typography.bodyMedium, fontFamily = FontFamily.Monospace, maxLines = 1, modifier = Modifier.weight(0.85f))
                        Text("${fmt(r.tokensPerSec, "%.1f")} tok/s", style = MaterialTheme.typography.bodyMedium, fontFamily = FontFamily.Monospace, maxLines = 1, modifier = Modifier.weight(1.3f))
                    }
                }
                if (i < power.runs.lastIndex) HorizontalDivider()
            }
        }
        Text(
            power.savedPath?.let { "saved to $it" } ?: "not saved: the power folder could not be written",
            style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace, color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        power.error?.let { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodyMedium) }
        val log = power.log
        OutlinedButton(
            onClick = { log?.let { onCopy(PowerLog.csv(it)) } },
            enabled = log != null,
            modifier = Modifier.fillMaxWidth().testTag(Tags.POWER_COPY),
        ) { Text("Copy CSV") }
    }
}

@Composable
private fun Tile(
    label: String, value: String, unit: String, caption: String?,
    big: Boolean = false, modifier: Modifier = Modifier,
    bg: Color = MaterialTheme.colorScheme.surfaceContainer, fg: Color = MaterialTheme.colorScheme.onSurface,
) {
    Column(modifier.fillMaxWidth().clip(RoundedCornerShape(if (big) 20.dp else 16.dp)).background(bg).padding(if (big) 16.dp else 12.dp)) {
        Text(label, style = MaterialTheme.typography.labelSmall, color = fg.copy(alpha = 0.8f))
        Row(verticalAlignment = Alignment.Bottom) {
            Text(value, fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold, fontSize = if (big) 44.sp else 24.sp, color = fg, lineHeight = if (big) 48.sp else 28.sp)
            Text(" $unit", style = MaterialTheme.typography.labelMedium, color = fg, modifier = Modifier.padding(bottom = 5.dp))
        }
        if (caption != null) Text(caption, style = MaterialTheme.typography.bodySmall, color = fg.copy(alpha = 0.85f))
    }
}
