// SPDX-License-Identifier: Apache-2.0
/**
 * @file  ModelSheet.kt
 * @brief Model picker, base path, load controls and (LiteRT only) the
 *        session settings, as a modal bottom sheet.
 */
package com.example.quickdotaiapp.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import com.example.quickdotai.RuntimeKind
import com.example.quickdotaiapp.ChatController
import com.example.quickdotaiapp.DEFAULT_BASE_PATH
import com.example.quickdotaiapp.Examples
import com.example.quickdotaiapp.LoadState
import com.example.quickdotaiapp.R
import com.example.quickdotaiapp.SessionSettings
import com.example.quickdotaiapp.ThinkingChoice
import com.example.quickdotaiapp.UiState
import java.io.File

@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
fun ModelSheet(state: UiState, controller: ChatController, onDismiss: () -> Unit) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    ModalBottomSheet(onDismissRequest = onDismiss, sheetState = sheetState) {
        Column(
            Modifier.fillMaxWidth().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp).padding(bottom = 24.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Model", style = MaterialTheme.typography.titleLarge, modifier = Modifier.weight(1f))
                LoadBadge(state.load)
                IconButton(onClick = onDismiss, modifier = Modifier.testTag(Tags.SHEET_CLOSE)) {
                    Icon(painterResource(R.drawable.ic_close), contentDescription = "Close")
                }
            }

            Label("FAMILY")
            FamilyDropdown(state, controller)

            // Each chip row takes the full width and wraps whole chips, so a label such as
            // "LiteRT" never breaks inside its chip.
            Label("RUNTIME")
            FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                state.runtimesFor(state.selection.family).forEach { rt ->
                    FilterChip(
                        selected = rt == state.selection.runtime,
                        onClick = { controller.selectRuntime(rt) },
                        label = { Text(Examples.runtimeLabel(rt), maxLines = 1) },
                    )
                }
            }
            Label("BACKEND")
            FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                state.backendsFor(state.selection.family, state.selection.runtime).forEach { be ->
                    FilterChip(
                        selected = be == state.selection.backend,
                        onClick = { controller.selectBackend(be) },
                        label = { Text(be.name, maxLines = 1) },
                    )
                }
            }

            Label("BASE PATH")
            var basePath by rememberLocalText(Unit) { controller.state.value.selection.basePath }
            OutlinedTextField(
                value = basePath,
                onValueChange = { basePath = it; controller.setBasePath(it) },
                singleLine = true,
                textStyle = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace),
                modifier = Modifier.fillMaxWidth().testTag(Tags.BASE_PATH),
            )
            val descriptor = state.descriptor
            if (descriptor != null) {
                val base = state.selection.basePath.ifBlank { DEFAULT_BASE_PATH }
                val required = Examples.requiredPath(descriptor, base)
                // ponytail: a stat() during composition; fine for a test app, a flow from the
                // ViewModel is the upgrade path if the base path ever points at slow storage.
                val exists = remember(required) { File(required).exists() }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        painterResource(if (exists) R.drawable.ic_check else R.drawable.ic_warning),
                        contentDescription = null,
                        tint = if (exists) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error,
                    )
                    Spacer(Modifier.width(8.dp))
                    Text(
                        required.removePrefix(base.trimEnd('/') + "/") + if (exists) "  found" else "  not found",
                        style = MaterialTheme.typography.bodySmall,
                        fontFamily = FontFamily.Monospace,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                AssistChip(onClick = {}, label = { Text("${Examples.badge(descriptor)} · W4A32 · ${descriptor.displayName}") })
            }

            if (Examples.isSdCapable(descriptor)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Speculative decoding", style = MaterialTheme.typography.titleSmall, modifier = Modifier.weight(1f))
                    Switch(checked = state.selection.sdEnabled, onCheckedChange = controller::setSdEnabled)
                }
            }

            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                val loaded = state.load is LoadState.Loaded
                Button(
                    onClick = { controller.load(force = loaded) },
                    enabled = state.load !is LoadState.Loading,
                    modifier = Modifier.weight(1f).testTag(Tags.LOAD),
                ) { Text(if (loaded) "Reload model" else "Load model") }
                OutlinedButton(
                    onClick = controller::unload,
                    enabled = loaded,
                    modifier = Modifier.weight(1f).testTag(Tags.UNLOAD),
                ) { Text("Unload") }
            }

            HorizontalDivider()

            Text("Session settings", style = MaterialTheme.typography.titleSmall)
            if (state.selection.runtime == RuntimeKind.LITERT) {
                SessionSettingsFields(state, controller)
            } else {
                Text(
                    if (Examples.isOneShot(state.descriptor))
                        "This engine does not support session settings. Only the newest message is sent with the model's own template: the HTP pack holds 1,280 positions."
                    else
                        "This engine does not support session settings. Each turn is sent with the full conversation and the model's own template.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
private fun LoadBadge(load: LoadState) {
    val (text, color) = when (load) {
        is LoadState.Loaded -> "Loaded" to MaterialTheme.colorScheme.primary
        is LoadState.Loading -> "Loading…" to MaterialTheme.colorScheme.tertiary
        LoadState.Idle -> "Not loaded" to MaterialTheme.colorScheme.onSurfaceVariant
    }
    Text(text, style = MaterialTheme.typography.labelMedium, color = color, modifier = Modifier.padding(end = 8.dp))
}

@Composable
private fun Label(text: String) {
    Text(text, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
}

@Composable
private fun FamilyDropdown(state: UiState, controller: ChatController) {
    var open by remember { mutableStateOf(false) }
    Box {
        OutlinedButton(onClick = { open = true }, modifier = Modifier.fillMaxWidth()) {
            Text(state.selection.family.ifEmpty { "—" }, modifier = Modifier.weight(1f))
            Icon(painterResource(R.drawable.ic_arrow_drop_down), contentDescription = null)
        }
        DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
            state.families.forEach { family ->
                DropdownMenuItem(
                    text = { Text(family) },
                    onClick = { open = false; controller.selectFamily(family) },
                )
            }
        }
    }
}

@Composable
private fun SessionSettingsFields(state: UiState, controller: ChatController) {
    val s = state.session
    // Each edit copies the controller's latest settings, never this composition's snapshot.
    fun edit(change: SessionSettings.() -> SessionSettings) = controller.setSession(controller.state.value.session.change())
    var system by rememberLocalText(Unit) { controller.state.value.session.systemPrompt }
    var temperature by rememberLocalText(Unit) { controller.state.value.session.temperature }
    var topK by rememberLocalText(Unit) { controller.state.value.session.topK }
    var topP by rememberLocalText(Unit) { controller.state.value.session.topP }
    var seed by rememberLocalText(Unit) { controller.state.value.session.seed }
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        OutlinedTextField(value = system, onValueChange = { system = it; edit { copy(systemPrompt = it) } },
            label = { Text("System prompt") }, minLines = 2, modifier = Modifier.fillMaxWidth())
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedTextField(value = temperature, onValueChange = { temperature = it; edit { copy(temperature = it) } },
                label = { Text("temperature") }, singleLine = true, modifier = Modifier.weight(1f))
            OutlinedTextField(value = topK, onValueChange = { topK = it; edit { copy(topK = it) } },
                label = { Text("top_k") }, singleLine = true, modifier = Modifier.weight(1f))
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedTextField(value = topP, onValueChange = { topP = it; edit { copy(topP = it) } },
                label = { Text("top_p") }, singleLine = true, modifier = Modifier.weight(1f))
            OutlinedTextField(value = seed, onValueChange = { seed = it; edit { copy(seed = it) } },
                label = { Text("seed") }, singleLine = true, modifier = Modifier.weight(1f))
        }
        Label("THINKING")
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            ThinkingChoice.entries.forEach { choice ->
                FilterChip(
                    selected = choice == s.thinking,
                    onClick = { edit { copy(thinking = choice) } },
                    label = { Text(choice.name.lowercase()) },
                )
            }
        }
        Spacer(Modifier.height(4.dp))
    }
}
