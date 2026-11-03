// SPDX-License-Identifier: Apache-2.0
/**
 * @file  UiState.kt
 * @brief The one immutable state the Compose screens render and the
 *        value types inside it. No Android types: this is unit-tested
 *        on the JVM.
 */
package com.example.quickdotaiapp

import com.example.quickdotai.BackendType
import com.example.quickdotai.LoadModelRequest
import com.example.quickdotai.ModelDescriptor
import com.example.quickdotai.PerformanceMetrics
import com.example.quickdotai.RuntimeKind

enum class Screen { CHAT, METRICS, POWER }

enum class ThemeMode { SYSTEM, LIGHT, DARK }

enum class TurnRole { USER, ASSISTANT, NOTE }

enum class ThinkingChoice { DEFAULT, ON, OFF }

/** What the user picked in the model sheet. */
data class ModelSelection(
    val family: String = "",
    val runtime: RuntimeKind = RuntimeKind.NATIVE,
    val backend: BackendType = BackendType.CPU,
    val sdEnabled: Boolean = false,
    val basePath: String = DEFAULT_BASE_PATH,
)

sealed interface LoadState {
    data object Idle : LoadState
    data class Loading(val key: String) : LoadState
    data class Loaded(val request: LoadModelRequest, val kind: String, val architecture: String?) : LoadState
}

/** Per-turn numbers shown under an assistant bubble. */
data class TurnMetrics(
    val ttftMs: Double,
    val tokensPerSec: Double,
    val generationTokens: Int,
    val prefillTokens: Int,
    val totalMs: Double,
    val engineLabel: String,
)

data class ChatTurn(
    val id: Long,
    val role: TurnRole,
    val text: String,
    val images: List<ByteArray> = emptyList(),
    val streaming: Boolean = false,
    val stopped: Boolean = false,
    /** Sent as OpenAI JSON from the composer; kept out of the chat history. */
    val raw: Boolean = false,
    val metrics: TurnMetrics? = null,
)

/** One row of the Metrics screen's history. */
data class RunRecord(
    val index: Int,
    val modelId: String,
    val mode: String,
    val prefillTokens: Int,
    val generationTokens: Int,
    val ttftMs: Double,
    val tokensPerSec: Double,
    val totalMs: Double,
    val output: String,
)

/** LiteRT session settings; strings because they come straight from text fields. */
data class SessionSettings(
    val systemPrompt: String = "",
    val temperature: String = "",
    val topK: String = "",
    val topP: String = "",
    val seed: String = "",
    val thinking: ThinkingChoice = ThinkingChoice.DEFAULT,
)

data class UiState(
    val catalog: List<ModelDescriptor> = emptyList(),
    val selection: ModelSelection = ModelSelection(),
    val load: LoadState = LoadState.Idle,
    val turns: List<ChatTurn> = emptyList(),
    val streaming: Boolean = false,
    val draft: String = "",
    val draftImages: List<ByteArray> = emptyList(),
    val session: SessionSettings = SessionSettings(),
    val rawJson: String = "",
    /** The composer takes OpenAI JSON ([rawJson]) instead of a message ([draft]). */
    val rawMode: Boolean = false,
    val lastMetrics: PerformanceMetrics? = null,
    val runs: List<RunRecord> = emptyList(),
    val status: String = "Idle.",
    val screen: Screen = Screen.CHAT,
    val theme: ThemeMode = ThemeMode.SYSTEM,
    val splashShown: Boolean = false,
    /** One-shot snackbar text; the UI clears it with consumeNotice(). */
    val notice: String? = null,
    val liveTokens: Int = 0,
    val liveElapsedMs: Long = 0,
    /** Bumped when the controller rewrites [draft] / [rawJson] itself (seed, file, send, reset),
     *  never on typing: the text fields keep their own text and resync only on these. */
    val draftRevision: Int = 0,
    val rawRevision: Int = 0,
) {
    val descriptor: ModelDescriptor?
        get() = catalog.firstOrNull {
            it.family == selection.family && it.runtime == selection.runtime && selection.backend in it.backends
        }
    val families: List<String> get() = catalog.map { it.family }.distinct()
    fun runtimesFor(family: String): List<RuntimeKind> =
        catalog.filter { it.family == family }.map { it.runtime }.distinct()
    fun backendsFor(family: String, runtime: RuntimeKind): List<BackendType> =
        catalog.filter { it.family == family && it.runtime == runtime }.flatMap { it.backends }.distinct()
    val loadedRequest: LoadModelRequest? get() = (load as? LoadState.Loaded)?.request
    val busy: Boolean get() = streaming || load is LoadState.Loading
    val lastAssistantText: String get() = turns.lastOrNull { it.role == TurnRole.ASSISTANT }?.text ?: ""
}
