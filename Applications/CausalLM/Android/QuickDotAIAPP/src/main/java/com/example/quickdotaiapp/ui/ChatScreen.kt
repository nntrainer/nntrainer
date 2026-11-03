// SPDX-License-Identifier: Apache-2.0
/**
 * @file  ChatScreen.kt
 * @brief The transcript (bubbles, notes, per-turn metrics, follow-the-
 *        stream scrolling) and the composer with attach, send and stop.
 *        The composer has two modes: a message, or OpenAI JSON sent as-is
 *        (the only path that forwards tools).
 */
package com.example.quickdotaiapp.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.scrollBy
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyListState
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.FilledIconToggleButton
import androidx.compose.material3.FilledTonalIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.SmallFloatingActionButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.material3.TextButton
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.sp
import androidx.compose.ui.unit.dp
import com.example.quickdotaiapp.ChatController
import com.example.quickdotaiapp.ChatTurn
import com.example.quickdotai.Capability
import com.example.quickdotai.RuntimeKind
import com.example.quickdotaiapp.R
import com.example.quickdotaiapp.RawRequest
import com.example.quickdotaiapp.TurnRole
import com.example.quickdotaiapp.UiState
import kotlinx.coroutines.launch

/** The bubble hides only a trailing end-of-turn marker; the state keeps the raw text. */
fun displayText(turn: ChatTurn): String = turn.text.removeSuffix("<turn|>").trimEnd('\n')

@Composable
fun ChatScreen(
    state: UiState,
    controller: ChatController,
    onAttach: () -> Unit,
    onLoadJson: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val listState = rememberLazyListState()
    Column(modifier.fillMaxSize()) {
        Box(Modifier.weight(1f).fillMaxWidth()) {
            Transcript(state, listState)
        }
        Composer(state, controller, onAttach, onLoadJson)
    }
}

@Composable
private fun BoxScope.Transcript(state: UiState, listState: LazyListState) {
    var follow by remember { mutableStateOf(true) }
    val scope = rememberCoroutineScope()
    // A drag that leaves the bottom stops following; reaching the bottom resumes it.
    LaunchedEffect(listState) {
        snapshotFlow { listState.isScrollInProgress to listState.canScrollForward }
            .collect { (scrolling, canScrollForward) -> if (scrolling) follow = !canScrollForward }
    }
    val lastText = state.turns.lastOrNull()?.text
    LaunchedEffect(state.turns.size, lastText?.length) {
        if (follow && state.turns.isNotEmpty()) {
            listState.scrollToItem(state.turns.lastIndex)
            listState.scrollBy(100_000f)
        }
    }
    LazyColumn(
        state = listState,
        contentPadding = PaddingValues(horizontal = 16.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
        modifier = Modifier.fillMaxSize().testTag(Tags.TRANSCRIPT),
    ) {
        if (state.turns.isEmpty()) {
            item {
                NoteRow(
                    if (state.loadedRequest != null) "Model loaded · start a conversation"
                    else "Load a model from the chip above, then send a message"
                )
            }
        }
        items(state.turns, key = { it.id }) { turn ->
            when (turn.role) {
                TurnRole.NOTE -> NoteRow(turn.text)
                TurnRole.USER -> UserBubble(turn)
                TurnRole.ASSISTANT -> AssistantBubble(turn)
            }
        }
    }
    if (!follow && state.streaming) {
        SmallFloatingActionButton(
            onClick = {
                follow = true
                scope.launch { listState.scrollToItem(state.turns.lastIndex); listState.scrollBy(100_000f) }
            },
            modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 8.dp),
        ) { Icon(painterResource(R.drawable.ic_arrow_down), contentDescription = "Scroll to the latest token") }
    }
}

@Composable
private fun NoteRow(text: String) {
    Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
        Text(
            text,
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier
                .clip(RoundedCornerShape(10.dp))
                .background(MaterialTheme.colorScheme.surfaceContainer)
                .padding(horizontal = 12.dp, vertical = 4.dp),
        )
    }
}

@Composable
private fun UserBubble(turn: ChatTurn) {
    Column(Modifier.fillMaxWidth(), horizontalAlignment = Alignment.End) {
        if (turn.images.isNotEmpty()) {
            Text(
                "${turn.images.size} image${if (turn.images.size > 1) "s" else ""} attached",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(bottom = 4.dp, end = 6.dp),
            )
        }
        SelectionContainer {
            Text(
                turn.text,
                color = MaterialTheme.colorScheme.onPrimaryContainer,
                fontFamily = if (turn.raw) FontFamily.Monospace else null,
                style = if (turn.raw) MaterialTheme.typography.bodySmall else MaterialTheme.typography.bodyLarge,
                modifier = Modifier
                    .widthIn(max = 300.dp)
                    .clip(RoundedCornerShape(20.dp, 20.dp, 6.dp, 20.dp))
                    .background(MaterialTheme.colorScheme.primaryContainer)
                    .padding(horizontal = 16.dp, vertical = 12.dp),
            )
        }
    }
}

@Composable
private fun AssistantBubble(turn: ChatTurn) {
    Column(Modifier.fillMaxWidth(), horizontalAlignment = Alignment.Start) {
        SelectionContainer {
            Text(
                text = displayText(turn) + if (turn.streaming) " ▍" else "",
                color = MaterialTheme.colorScheme.onSurface,
                style = MaterialTheme.typography.bodyLarge,
                modifier = Modifier
                    .widthIn(max = 320.dp)
                    .clip(RoundedCornerShape(20.dp, 20.dp, 20.dp, 6.dp))
                    .background(MaterialTheme.colorScheme.surfaceContainerHigh)
                    .padding(horizontal = 16.dp, vertical = 12.dp)
                    .testTag(Tags.ASSISTANT_TEXT),
            )
        }
        val m = turn.metrics
        val line = when {
            turn.streaming -> null
            m != null && m.generationTokens > 0 ->
                "TTFT %.2f s · %.1f tok/s · %d tok".format(m.ttftMs / 1000.0, m.tokensPerSec, m.generationTokens)
            m != null -> "%s · total %.1f s".format(m.engineLabel, m.totalMs / 1000.0)
            else -> null
        }
        val suffix = if (turn.stopped) " · stopped" else ""
        if (line != null || turn.stopped) {
            Text(
                (line ?: "") + suffix,
                style = MaterialTheme.typography.labelSmall,
                fontFamily = FontFamily.Monospace,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(start = 6.dp, top = 4.dp),
            )
        }
    }
}

@Composable
private fun Composer(state: UiState, controller: ChatController, onAttach: () -> Unit, onLoadJson: () -> Unit) {
    var draft by rememberLocalText(state.draftRevision) { controller.state.value.draft }
    var raw by rememberLocalText(state.rawRevision) { controller.state.value.rawJson }
    val rawMode = state.rawMode
    val parsed = remember(raw, rawMode) { if (rawMode) RawRequest.parse(raw) else null }
    Column(Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 8.dp)) {
        if (state.draftImages.isNotEmpty()) {
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(bottom = 8.dp)) {
                state.draftImages.forEachIndexed { i, bytes ->
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier
                            .padding(end = 8.dp)
                            .clip(RoundedCornerShape(14.dp))
                            .background(MaterialTheme.colorScheme.surfaceContainer)
                            .padding(start = 10.dp),
                    ) {
                        Icon(painterResource(R.drawable.ic_image), null,
                            tint = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(6.dp))
                        Text("${bytes.size / 1024} KB", style = MaterialTheme.typography.labelSmall)
                        IconButton(onClick = { controller.removeDraftImage(i) }, modifier = Modifier.size(32.dp)) {
                            Icon(painterResource(R.drawable.ic_close), contentDescription = "Remove image", modifier = Modifier.size(16.dp))
                        }
                    }
                }
            }
        }
        if (parsed != null) JsonHeader(parsed, state, onLoadJson, onReset = controller::resetRawJson)
        Row(verticalAlignment = Alignment.Bottom) {
            FilledTonalIconButton(onClick = onAttach, modifier = Modifier.size(48.dp).testTag(Tags.ATTACH)) {
                Icon(painterResource(R.drawable.ic_attach), contentDescription = "Attach a file")
            }
            Spacer(Modifier.width(8.dp))
            val fieldColors = TextFieldDefaults.colors(
                focusedIndicatorColor = Color.Transparent,
                unfocusedIndicatorColor = Color.Transparent,
                disabledIndicatorColor = Color.Transparent,
            )
            // The mode switch sits beside attach: braces lit = the field is the request itself.
            FilledIconToggleButton(
                checked = rawMode,
                onCheckedChange = controller::setRawMode,
                modifier = Modifier.size(48.dp).testTag(Tags.RAW_MODE),
            ) {
                Icon(painterResource(R.drawable.ic_braces),
                    contentDescription = if (rawMode) "Back to a message" else "Send OpenAI JSON instead")
            }
            Spacer(Modifier.width(8.dp))
            if (rawMode) {
                TextField(
                    value = raw,
                    onValueChange = { raw = it; controller.setRawJson(it) },
                    placeholder = { Text("OpenAI JSON: {\"messages\": […], \"tools\": […]}", style = MaterialTheme.typography.bodySmall) },
                    maxLines = 10,
                    textStyle = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace, lineHeight = 18.sp),
                    shape = RoundedCornerShape(24.dp),
                    colors = fieldColors,
                    modifier = Modifier.weight(1f).testTag(Tags.RAW_JSON),
                )
            } else {
                TextField(
                    value = draft,
                    onValueChange = { draft = it; controller.setDraft(it) },
                    placeholder = { Text("Type a message…") },
                    maxLines = 6,
                    shape = RoundedCornerShape(24.dp),
                    colors = fieldColors,
                    modifier = Modifier.weight(1f).testTag(Tags.COMPOSER),
                )
            }
            Spacer(Modifier.width(8.dp))
            if (state.streaming) {
                FilledIconButton(
                    onClick = controller::cancel,
                    colors = IconButtonDefaults.filledIconButtonColors(
                        containerColor = MaterialTheme.colorScheme.errorContainer,
                        contentColor = MaterialTheme.colorScheme.onErrorContainer,
                    ),
                    modifier = Modifier.size(48.dp).testTag(Tags.STOP),
                ) { Icon(painterResource(R.drawable.ic_stop), contentDescription = "Stop generating") }
            } else {
                val ready = if (rawMode) parsed !is RawRequest.Parsed.Invalid && raw.isNotBlank()
                            else draft.isNotBlank() || state.draftImages.isNotEmpty()
                FilledIconButton(
                    onClick = if (rawMode) controller::sendRaw else controller::send,
                    enabled = !state.busy && ready,
                    modifier = Modifier.size(48.dp).testTag(Tags.SEND),
                ) { Icon(painterResource(R.drawable.ic_send), contentDescription = if (rawMode) "Send the JSON request" else "Send") }
            }
        }
    }
}

/** Above the JSON field: whether it parses, how it will be routed, and the load / reset actions. */
@Composable
private fun JsonHeader(parsed: RawRequest.Parsed, state: UiState, onLoadJson: () -> Unit, onReset: () -> Unit) {
    val d = state.descriptor
    val path = when {
        parsed is RawRequest.Parsed.Messages && d != null &&
            (Capability.MESSAGES_API in d.capabilities || d.runtime == RuntimeKind.LITERT) -> "messages API"
        else -> "JSON as-is"
    }
    val (ok, line) = when (parsed) {
        is RawRequest.Parsed.Messages -> true to "${parsed.messages.size} messages${if (parsed.hasTools) " · tools" else ""} · $path"
        RawRequest.Parsed.Other -> true to "Valid JSON, not a messages request · $path"
        is RawRequest.Parsed.Invalid -> false to "Invalid JSON: ${parsed.reason}"
    }
    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth().padding(start = 4.dp, bottom = 4.dp)) {
        Icon(painterResource(if (ok) R.drawable.ic_check else R.drawable.ic_warning), null,
            tint = if (ok) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error, modifier = Modifier.size(16.dp))
        Spacer(Modifier.width(6.dp))
        Text(line, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
            maxLines = 2, modifier = Modifier.weight(1f))
        TextButton(onClick = onLoadJson, contentPadding = PaddingValues(horizontal = 8.dp), modifier = Modifier.testTag(Tags.RAW_LOAD)) { Text("Load .json") }
        TextButton(onClick = onReset, contentPadding = PaddingValues(horizontal = 8.dp), modifier = Modifier.testTag(Tags.RAW_RESET)) { Text("Reset") }
    }
}
