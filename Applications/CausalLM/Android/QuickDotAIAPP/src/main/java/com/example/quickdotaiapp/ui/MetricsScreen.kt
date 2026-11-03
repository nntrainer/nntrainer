// SPDX-License-Identifier: Apache-2.0
/**
 * @file  MetricsScreen.kt
 * @brief Last-run tiles (real TTFT, steady decode, tok/s, prefill, peak
 *        memory), the run history and the raw output of the last run.
 */
package com.example.quickdotaiapp.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.quickdotai.PerformanceMetrics
import com.example.quickdotaiapp.ChatController
import com.example.quickdotaiapp.RunRecord
import com.example.quickdotaiapp.UiState

@Composable
fun MetricsScreen(state: UiState, controller: ChatController, modifier: Modifier = Modifier) {
    var rawOpen by remember { mutableStateOf(false) }
    LazyColumn(modifier.fillMaxSize(), contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item {
            Row {
                Text("Last run", style = MaterialTheme.typography.titleLarge, modifier = Modifier.weight(1f))
                state.runs.lastOrNull()?.let {
                    Text("${it.mode} · ${it.prefillTokens} in · ${it.generationTokens} out",
                        style = MaterialTheme.typography.labelSmall, fontFamily = FontFamily.Monospace)
                }
            }
        }
        val m = state.lastMetrics
        if (m == null) {
            item {
                Text("No run yet. Send a message or a raw request; the engine's counters appear here.",
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        } else {
            item { Tiles(m) }
        }
        item {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Recent runs", style = MaterialTheme.typography.titleSmall, modifier = Modifier.weight(1f))
                TextButton(onClick = controller::clearRuns, enabled = state.runs.isNotEmpty()) { Text("Clear") }
            }
        }
        item { RunHeader() }
        items(state.runs.asReversed(), key = { it.index }) { RunRow(it) }
        item {
            HorizontalDivider()
            Text(
                if (rawOpen) "▾ Raw output of the last run" else "▸ Raw output of the last run",
                style = MaterialTheme.typography.titleSmall,
                modifier = Modifier.fillMaxWidth().clickable { rawOpen = !rawOpen }.padding(vertical = 8.dp),
            )
            if (rawOpen) {
                SelectionContainer {
                    Text(
                        state.runs.lastOrNull()?.output ?: "—",
                        fontFamily = FontFamily.Monospace,
                        style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp))
                            .background(MaterialTheme.colorScheme.surfaceContainer).padding(12.dp),
                    )
                }
            }
        }
    }
}

@Composable
private fun Tiles(m: PerformanceMetrics) {
    val tps = if (m.generationDurationMs > 0) m.generationTokens * 1000.0 / m.generationDurationMs else 0.0
    val prefillTps = if (m.prefillDurationMs > 0) m.prefillTokens * 1000.0 / m.prefillDurationMs else 0.0
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Tile("TIME TO FIRST TOKEN", "%.2f".format(m.ttftMs / 1000.0), "s", "prompt submit → first token to the app", big = true,
            bg = MaterialTheme.colorScheme.primaryContainer, fg = MaterialTheme.colorScheme.onPrimaryContainer)
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            Tile("STEADY DECODE", "%.1f".format(m.decodeSteadyMs), "ms/tok", "median of tokens 2..N", modifier = Modifier.weight(1f))
            Tile("GENERATION", "%.1f".format(tps), "tok/s",
                "${m.generationTokens} tok · ${"%.0f".format(m.generationDurationMs)} ms", modifier = Modifier.weight(1f))
        }
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            Tile("PREFILL", "%.0f".format(prefillTps), "tok/s",
                "${m.prefillTokens} tok · ${"%.0f".format(m.prefillDurationMs)} ms", modifier = Modifier.weight(1f))
            Tile("PEAK MEMORY", "%.1f".format(m.peakMemoryKb / 1024.0 / 1024.0), "GB",
                "init ${"%.1f".format(m.initializationDurationMs / 1000.0)} s", modifier = Modifier.weight(1f))
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

@Composable
private fun RunHeader() {
    Row(Modifier.fillMaxWidth().background(MaterialTheme.colorScheme.surfaceContainer).padding(horizontal = 8.dp, vertical = 6.dp)) {
        listOf("#" to 0.1f, "mode" to 0.15f, "model" to 0.35f, "in" to 0.12f, "TTFT" to 0.14f, "tok/s" to 0.14f).forEach { (t, w) ->
            Text(t, style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.Bold, modifier = Modifier.weight(w))
        }
    }
}

@Composable
private fun RunRow(r: RunRecord) {
    Row(Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 6.dp)) {
        val cells = listOf(
            r.index.toString() to 0.1f, r.mode to 0.15f, r.modelId to 0.35f, r.prefillTokens.toString() to 0.12f,
            "%.2f s".format(r.ttftMs / 1000.0) to 0.14f, "%.1f".format(r.tokensPerSec) to 0.14f,
        )
        cells.forEach { (t, w) ->
            Text(t, style = MaterialTheme.typography.labelSmall, fontFamily = FontFamily.Monospace, maxLines = 1, modifier = Modifier.weight(w))
        }
    }
}
