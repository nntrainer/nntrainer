// SPDX-License-Identifier: Apache-2.0
/**
 * @file  HtpAppLifecycleTest.kt
 * @brief Opt-in device test of the HTP presets through the Compose sample:
 *        load, generate, stop, repeat, reload, active unload, invalid path,
 *        the Gemma4 prompt format and the LFM2 default-path restart. Needs
 *        the v81 HTP build, a staged pack and the -e arguments; without
 *        them every test is skipped. Inference always starts through the
 *        same controls as a user; the controller is read, never driven,
 *        except to pick the model (the chips carry no test tags).
 */
package com.example.quickdotaiapp

import android.graphics.Bitmap
import android.view.WindowManager
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.junit4.createEmptyComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.semantics.SemanticsActions
import androidx.compose.ui.test.performSemanticsAction
import androidx.compose.ui.test.performTextReplacement
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.quickdotai.BackendType
import com.example.quickdotai.ModelCatalog
import com.example.quickdotai.ModelIds
import com.example.quickdotaiapp.ui.Tags
import java.io.File
import java.util.concurrent.TimeUnit
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class HtpAppLifecycleTest {
    @get:Rule
    val rule = createEmptyComposeRule()

    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val args = InstrumentationRegistry.getArguments()
    private val context get() = instrumentation.targetContext
    private lateinit var scenario: ActivityScenario<MainActivity>
    private lateinit var activity: MainActivity
    private lateinit var report: File
    private var traceConfig: File? = null
    private var originalConfig: ByteArray? = null
    private lateinit var liveTrace: File

    private val controller: ChatController get() = activity.viewModel.controller
    private val state: UiState get() = controller.state.value
    private fun status() = state.status
    private fun output() = state.lastAssistantText

    // ----- the app -----------------------------------------------------------

    /** A normal launch: the splash runs, the catalog is read, nothing is injected. */
    private fun launch() {
        scenario = ActivityScenario.launch(MainActivity::class.java)
        scenario.onActivity { activity = it; it.window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON) }
        await("splash and catalog", timeoutMinutes = 1) { state.splashShown && state.catalog.isNotEmpty() }
        rule.waitForIdle()
    }

    /** Polls [condition]. The test owns the Compose frame clock: delays and animations in composition (the
     *  splash, the transcript) only advance when the clock is moved, so each round advances it and idles. */
    private fun await(label: String, timeoutMinutes: Long = 4, condition: () -> Boolean) {
        val deadline = System.nanoTime() + TimeUnit.MINUTES.toNanos(timeoutMinutes)
        while (!condition()) {
            if (System.nanoTime() > deadline) fail("Timeout: $label; ${status()}; splash=${state.splashShown} catalog=${state.catalog.size}")
            rule.mainClock.advanceTimeBy(250)
            rule.waitForIdle()
            Thread.sleep(50)
        }
    }

    /** Waits until no load or generation is in flight, then lets Compose settle. */
    private fun awaitIdle() {
        await("idle") { !state.busy }
        rule.waitForIdle()
    }

    /** Clicks through semantics, as the View test did with its buttons: a snackbar (the "Not found"
     *  notice) or a sheet scrim over the control must not swallow the step. */
    private fun tap(tag: String) {
        rule.onNodeWithTag(tag).assertIsEnabled().performSemanticsAction(SemanticsActions.OnClick)
        rule.waitForIdle()
    }
    private fun type(tag: String, text: String) { rule.onNodeWithTag(tag).performTextReplacement(text); rule.waitForIdle() }

    /** The model pickers are chips without test tags: select through the controller, then check the descriptor. */
    private fun selectModel(modelId: String) {
        val descriptor = requireNotNull(ModelCatalog.byId(modelId)) { "unknown model $modelId" }
        instrumentation.runOnMainSync {
            controller.selectFamily(descriptor.family)
            controller.selectRuntime(descriptor.runtime)
            controller.selectBackend(BackendType.NPU)
        }
        rule.waitForIdle()
        assertEquals(modelId, state.descriptor?.id)
        assertTrue("Model folder must match the public API",
            Examples.modelPath(descriptor, "/b").endsWith("/$modelId-w4a32"))
    }

    /** Types into the sheet's base path field and checks the controller took it. */
    private fun typeBasePath(base: String) {
        type(Tags.BASE_PATH, base)
        assertEquals("Base path field did not reach the controller", base, state.selection.basePath)
    }

    private fun loadFromSheet(base: String? = null) {
        tap(Tags.MODEL_CHIP)
        if (base != null) typeBasePath(base)
        tap(Tags.LOAD)
        awaitIdle()
        tap(Tags.SHEET_CLOSE)
    }

    /** The unload runs on the engine thread while busy stays false: wait for the handle to be gone. */
    private fun unloadFromSheet() {
        tap(Tags.MODEL_CHIP); tap(Tags.UNLOAD)
        await("unloaded") { state.load is LoadState.Idle && !state.busy }
        rule.waitForIdle()
        tap(Tags.SHEET_CLOSE)
        assertEquals("Unloaded.", status())
    }

    private fun setRawMode(on: Boolean) {
        if (state.rawMode != on) tap(Tags.RAW_MODE)
        assertEquals(on, state.rawMode)
    }

    /** Sends [messages] through the JSON composer and returns the answer; optionally stops or unloads mid-stream. */
    private fun runJson(name: String, messages: String, stop: Boolean = false, unload: Boolean = false): String {
        liveTrace.delete()
        tap(Tags.CLEAR)
        setRawMode(true)
        type(Tags.RAW_JSON, messages)
        tap(Tags.SEND)
        if (stop || unload) {
            await("first output") { output().length >= 24 || status().contains("failed", true) }
            assertFalse(status(), status().contains("failed", true))
            if (unload) unloadFromSheet() else tap(Tags.STOP)
        }
        awaitIdle()
        assertEquals(if (unload) "Unloaded." else "Done.", status())
        val text = output()
        assertTrue("Empty output", text.isNotBlank())
        File(report, "$name.txt").writeText(text)
        captureTrace(name)
        return text
    }

    /** One chat turn from a cleared transcript; the HTP presets are one-shot anyway (Examples.oneShotIds). */
    private fun runChat(name: String, prompt: String): String {
        liveTrace.delete()
        tap(Tags.CLEAR)
        setRawMode(false)
        File(report, "$name-input.txt").writeText(prompt)
        type(Tags.COMPOSER, prompt)
        tap(Tags.SEND)
        awaitIdle()
        File(report, "$name-status.txt").writeText(status())
        File(report, "$name-output.txt").writeText(output())
        assertEquals("Done.", status())
        assertTrue("Empty chat output", output().isNotBlank())
        captureTrace(name)
        return output()
    }

    private fun screenshot(name: String) {
        instrumentation.waitForIdleSync()
        File(report, "$name.png").outputStream().use {
            instrumentation.uiAutomation.takeScreenshot().compress(Bitmap.CompressFormat.PNG, 100, it)
        }
    }

    // ----- the pack's token trace --------------------------------------------

    // Runtime options are immutable after model load. Restore the pack after
    // every test, including failures, and copy each completed generation trace.
    private fun prepareTrace(base: String, modelId: String) {
        check(traceConfig == null)
        val file = File(base, "$modelId-w4a32/nntr_config.json")
        val original = file.readBytes()
        val config = JSONObject(String(original, Charsets.UTF_8))
        val htp = config.optJSONObject("htp") ?: JSONObject().also { config.put("htp", it) }
        liveTrace = File(report, "live-trace.json")
        htp.put("token_trace", liveTrace.path)
        traceConfig = file
        originalConfig = original
        file.writeText(config.toString(2))
    }

    private fun restoreTrace() {
        traceConfig?.writeBytes(requireNotNull(originalConfig))
        traceConfig = null
        originalConfig = null
    }

    private fun captureTrace(name: String) {
        assertTrue("Missing native token trace for $name", liveTrace.isFile)
        liveTrace.copyTo(File(report, "trace-$name.json"), overwrite = true)
    }

    private fun finish() {
        try {
            if (::scenario.isInitialized) scenario.close()
            instrumentation.waitForIdleSync()
        } finally {
            restoreTrace()
        }
    }

    // ----- scenarios ---------------------------------------------------------

    @Test
    fun lfm2ChatOpensFromDefaultPathAfterActivityRestart() {
        assumeTrue("Stage LFM2 under the app's default model base path first",
            args.getString("htpDefaultPathCheck") == "true")
        report = File(context.filesDir, "htp-default-path-validation").apply { mkdirs() }
        val prompt = "Explain three practical ways to conserve water at home. " +
            "Use a numbered list with two sentences per item."
        val messages = JSONArray().put(JSONObject().put("role", "user").put("content", prompt)).toString()
        var firstOutput: String? = null
        repeat(2) { attempt ->
            launch()
            try {
                selectModel(ModelIds.LFM2_24BA2B_HTP)
                // Do not inject a model path: this must work after a normal launch.
                val base = state.selection.basePath
                assertTrue("Missing model at the app's default path: $base",
                    File(base, "lfm2-24ba2b-htp-w4a32/nntr_config.json").isFile)
                prepareTrace(base, ModelIds.LFM2_24BA2B_HTP)
                loadFromSheet()
                assertTrue(status(), status().startsWith("Loaded "))
                val chatOutput = runChat("chat-$attempt", prompt)
                if (firstOutput == null) firstOutput = chatOutput else assertEquals(firstOutput, chatOutput)
                screenshot("chat-$attempt")
                assertEquals("JSON and Chat must render the same prompt", chatOutput, runJson("openai-$attempt", messages))
                unloadFromSheet()
                File(report, "base-path.txt").writeText(base)
            } finally {
                finish()
            }
        }
        File(report, "result.json").writeText(JSONObject()
            .put("status", "passed").put("activityLaunches", 2)
            .put("pathInjected", false).put("chatRuns", 2).put("openAiRuns", 2).toString(2))
    }

    @Test
    fun chatUsesGemma4PromptFormat() {
        val base = args.getString("htpModelBase")
        val messagesPath = args.getString("htpMessages")
        assumeTrue("Supply htpModelBase and htpMessages for the v81 device test", base != null && messagesPath != null)
        report = File(context.filesDir, "htp-chat-validation").apply { mkdirs() }
        val messages = File(messagesPath!!).readText()
        val summaryPrompt = JSONArray(messages).getJSONObject(0).getString("content")
        launch()
        try {
            selectModel(ModelIds.GEMMA4_26B_A4B_HTP)
            prepareTrace(base!!, ModelIds.GEMMA4_26B_A4B_HTP)
            loadFromSheet(base)
            assertTrue(status(), status().startsWith("Loaded "))
            val shortPrompt = args.getString("htpChatPrompt") ?: "I bought a red car yesterday"
            for ((name, prompt) in listOf("short" to shortPrompt, "repeat" to shortPrompt, "summary" to summaryPrompt)) {
                val text = runChat(name, prompt)
                assertFalse("Unexpected thought channel in $name", text.contains("<|channel>"))
                assertFalse("Incomplete thought prefix in $name", text.contains("<channel|>"))
                assertTrue("Missing end of turn in $name", text.endsWith("<turn|>"))
                assertEquals("Generation continued past the first end of turn in $name",
                    text.length - "<turn|>".length, text.indexOf("<turn|>"))
            }
            val chatSummary = File(report, "summary-output.txt").readText()
            assertEquals("JSON and Chat must render the same prompt", chatSummary, runJson("openai-summary", messages))
            unloadFromSheet()
            File(report, "result.json").writeText(JSONObject().put("status", "passed").toString())
        } finally {
            finish()
        }
    }

    @Test
    fun loadGenerateStopRepeatReloadAndReportInvalidPath() {
        val modelId = args.getString("htpModelId") ?: ModelIds.GEMMA4_26B_A4B_HTP
        val base = args.getString("htpModelBase")
        val messagesPath = args.getString("htpMessages")
        assumeTrue("Supply htpModelBase and htpMessages for the v81 device test", base != null && messagesPath != null)
        report = File(context.filesDir, "htp-app-validation").apply { mkdirs() }
        val messages = File(messagesPath!!).readText()
        launch()
        try {
            selectModel(modelId)
            prepareTrace(base!!, modelId)
            loadFromSheet(base)
            assertTrue(status(), status().startsWith("Loaded "))
            assertEquals(modelId, state.loadedRequest?.modelId)
            val first = runJson("first", messages)
            assertEquals(first, runJson("repeat", messages))
            val partial = runJson("cancel", messages, stop = true)
            assertTrue("Cancel must stop before completion", partial.length < first.length)
            assertTrue("Cancelled output differs", first.startsWith(partial))
            assertEquals(first, runJson("after-cancel", messages))

            unloadFromSheet()
            loadFromSheet()
            assertTrue(status(), status().startsWith("Loaded "))
            assertEquals(first, runJson("reload", messages))
            loadFromSheet()                                              // "Reload model" replaces the handle
            assertTrue("Reload reused the previous handle: ${status()}", status().startsWith("Loaded "))
            assertEquals(first, runJson("explicit-reload", messages))

            val partialUnload = runJson("active-unload", messages, unload = true)
            assertTrue(partialUnload.length < first.length && first.startsWith(partialUnload))
            loadFromSheet()
            assertTrue(status(), status().startsWith("Loaded "))
            assertEquals(first, runJson("after-active-unload", messages))

            // A changed path must not cost the loaded model: the folder check runs before the teardown.
            tap(Tags.MODEL_CHIP); typeBasePath("$base/missing-htp-model"); tap(Tags.SHEET_CLOSE)
            tap(Tags.CLEAR); setRawMode(true); type(Tags.RAW_JSON, messages); tap(Tags.SEND); awaitIdle()
            assertTrue("Lost load diagnostic: ${status()}", status().startsWith("Load failed:"))
            val invalidPathError = status()
            assertTrue("The loaded model was torn down", state.load is LoadState.Loaded)
            tap(Tags.MODEL_CHIP); typeBasePath(base); tap(Tags.SHEET_CLOSE)
            assertEquals(first, runJson("after-failed-load", messages))
            screenshot("screen")
            unloadFromSheet()
            File(report, "result.json").writeText(JSONObject().apply {
                put("status", "passed")
                put("package", context.packageName)
                put("modelId", modelId)
                put("nativeLibraryDir", context.applicationInfo.nativeLibraryDir)
                put("modelBasePath", base)
                put("invalidPathError", invalidPathError)
                put("fullRuns", 7)
                put("cancelledRuns", 2)
            }.toString(2))
        } finally {
            finish()
        }
    }
}
