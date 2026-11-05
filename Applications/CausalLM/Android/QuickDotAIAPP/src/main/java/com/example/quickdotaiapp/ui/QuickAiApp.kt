// SPDX-License-Identifier: Apache-2.0
/**
 * @file  QuickAiApp.kt
 * @brief The scaffold: splash, top bar with the model chip, status strip,
 *        the three screens on a bottom bar, the sheets, the file pickers
 *        and the snackbar.
 */
package com.example.quickdotaiapp.ui

import android.content.ClipData
import android.content.ClipboardManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.Crossfade
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.AssistChip
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.runtime.DisposableEffect
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.example.quickdotai.Capability
import com.example.quickdotai.ModelDescriptor
import com.example.quickdotaiapp.AppViewModel
import com.example.quickdotaiapp.Examples
import com.example.quickdotaiapp.LoadState
import com.example.quickdotaiapp.R
import com.example.quickdotaiapp.Screen
import com.example.quickdotaiapp.ThemeMode
import com.example.quickdotaiapp.UiState

@Composable
fun QuickAiApp(vm: AppViewModel) {
    val controller = vm.controller
    val state by controller.state.collectAsStateWithLifecycle()
    val power by vm.power.state.collectAsStateWithLifecycle()
    QuickAiTheme(state.theme) {
        // The splash cross-fades into the app instead of being swapped out.
        Crossfade(targetState = state.splashShown, animationSpec = tween(350), label = "splash") { shown ->
            if (shown) AppScaffold(state, power, vm) else Splash(onDone = controller::splashDone)
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun AppScaffold(state: UiState, power: com.example.quickdotaiapp.power.PowerUi, vm: AppViewModel) {
    val controller = vm.controller
    val context = LocalContext.current
    // The gauge must see the screen on for the whole Power session, whichever tab is shown.
    val view = LocalView.current
    val measuring = power.phase.measuring
    DisposableEffect(measuring) {
        view.keepScreenOn = measuring
        onDispose { view.keepScreenOn = false }
    }
    var modelSheetOpen by rememberSaveable { mutableStateOf(false) }
    var attachSheetOpen by rememberSaveable { mutableStateOf(false) }
    val snackbar = remember { SnackbarHostState() }
    val descriptor = state.descriptor
    val multiImage = descriptor?.capabilities?.contains(Capability.MULTI_IMAGE) == true
    val multimodal = descriptor?.capabilities?.contains(Capability.MULTIMODAL) == true

    val pickImage = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        vm.attachImages(listOfNotNull(uri))
    }
    val pickImages = rememberLauncherForActivityResult(ActivityResultContracts.PickMultipleVisualMedia()) { uris ->
        vm.attachImages(uris)
    }
    val pickText = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let(vm::attachTextFile)
    }
    val pickJson = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let(vm::attachJsonFile)
    }

    LaunchedEffect(state.notice) {
        val n = state.notice ?: return@LaunchedEffect
        // Show first: consuming changes this effect's key and would cancel the snackbar.
        snackbar.showSnackbar(n)
        controller.consumeNotice()
    }

    Scaffold(
        topBar = {
            Column {
                TopAppBar(
                    title = { Wordmark(size = 20.sp) },
                    actions = {
                        IconButton(onClick = controller::clearConversation, modifier = Modifier.testTag(Tags.CLEAR)) {
                            Icon(painterResource(R.drawable.ic_delete), contentDescription = "Clear conversation")
                        }
                        IconButton(onClick = { vm.setTheme(nextTheme(state.theme)) }) {
                            Icon(
                                painterResource(if (state.theme == ThemeMode.DARK) R.drawable.ic_light_mode else R.drawable.ic_dark_mode),
                                contentDescription = "Theme: ${state.theme.name.lowercase()}",
                            )
                        }
                    },
                )
                // The model chip gets its own row so the name above it never squeezes it.
                Row(Modifier.fillMaxWidth().padding(start = 12.dp, end = 12.dp, bottom = 6.dp)) {
                    ModelChip(descriptor, state.load) { modelSheetOpen = true }
                }
            }
        },
        bottomBar = {
            NavigationBar {
                NavigationBarItem(selected = state.screen == Screen.CHAT, onClick = { controller.setScreen(Screen.CHAT) },
                    icon = { Icon(painterResource(R.drawable.ic_chat), null) }, label = { Text("Chat") },
                    modifier = Modifier.testTag(Tags.NAV_CHAT))
                NavigationBarItem(selected = state.screen == Screen.METRICS, onClick = { controller.setScreen(Screen.METRICS) },
                    icon = { Icon(painterResource(R.drawable.ic_chart), null) }, label = { Text("Metrics") },
                    modifier = Modifier.testTag(Tags.NAV_METRICS))
                NavigationBarItem(selected = state.screen == Screen.POWER, onClick = { controller.setScreen(Screen.POWER) },
                    icon = { Icon(painterResource(R.drawable.ic_bolt), null) }, label = { Text("Power") },
                    modifier = Modifier.testTag(Tags.NAV_POWER))
            }
        },
        snackbarHost = { SnackbarHost(snackbar) },
    ) { padding ->
        Column(Modifier.padding(padding).fillMaxSize()) {
            StatusStrip(state)
            when (state.screen) {
                Screen.CHAT -> ChatScreen(state, controller, onAttach = { attachSheetOpen = true },
                    onLoadJson = { pickJson.launch(arrayOf("application/json", "text/*")) }, modifier = Modifier.weight(1f))
                Screen.METRICS -> MetricsScreen(state, controller, modifier = Modifier.weight(1f))
                Screen.POWER -> PowerScreen(
                    state, power, vm.power, onMeasure = vm::startPower,
                    onCopy = { text ->
                        context.getSystemService(ClipboardManager::class.java)?.setPrimaryClip(ClipData.newPlainText("runs.csv", text))
                        controller.notify("CSV copied")
                    },
                    modifier = Modifier.weight(1f),
                )
            }
        }
    }

    if (modelSheetOpen) ModelSheet(state, controller, onDismiss = { modelSheetOpen = false })
    if (attachSheetOpen) {
        AttachSheet(
            canAttachImages = multimodal,
            modelName = descriptor?.displayName ?: "This model",
            onImage = {
                if (multiImage) pickImages.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
                else pickImage.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
            },
            onText = { pickText.launch(arrayOf("text/*")) },
            onJson = { pickJson.launch(arrayOf("application/json", "text/*")) },
            onDismiss = { attachSheetOpen = false },
        )
    }
}

private fun nextTheme(t: ThemeMode): ThemeMode = when (t) {
    ThemeMode.SYSTEM -> ThemeMode.LIGHT
    ThemeMode.LIGHT -> ThemeMode.DARK
    ThemeMode.DARK -> ThemeMode.SYSTEM
}

@Composable
private fun ModelChip(descriptor: ModelDescriptor?, load: LoadState, onClick: () -> Unit) {
    val label = descriptor?.let { "${it.displayName} · ${Examples.runtimeLabel(it.runtime)}" } ?: "Pick a model"
    val dot = when (load) {
        is LoadState.Loaded -> MaterialTheme.colorScheme.primary
        is LoadState.Loading -> MaterialTheme.colorScheme.tertiary
        LoadState.Idle -> MaterialTheme.colorScheme.outline
    }
    AssistChip(
        onClick = onClick,
        leadingIcon = { Icon(painterResource(R.drawable.ic_model), null) },
        trailingIcon = { Box(Modifier.size(8.dp).clip(CircleShape).background(dot)) },
        label = { Text(label, maxLines = 1) },
        modifier = Modifier.testTag(Tags.MODEL_CHIP),
    )
}

@Composable
private fun StatusStrip(state: UiState) {
    val live = if (state.streaming && state.liveElapsedMs > 0)
        " · %.1f s · %d tokens · %.1f tok/s".format(
            state.liveElapsedMs / 1000.0, state.liveTokens, state.liveTokens * 1000.0 / state.liveElapsedMs)
    else ""
    val busy = state.busy
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier.fillMaxWidth()
            .background(if (busy) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceContainerLow)
            .padding(horizontal = 16.dp, vertical = 6.dp),
    ) {
        Box(Modifier.size(8.dp).clip(CircleShape)
            .background(if (busy) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outline))
        Spacer(Modifier.width(8.dp))
        Text(
            state.status + live,
            style = MaterialTheme.typography.labelMedium,
            fontFamily = FontFamily.Monospace,
            maxLines = 1,
            color = if (busy) MaterialTheme.colorScheme.onPrimaryContainer else MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.testTag(Tags.STATUS),
        )
    }
}
