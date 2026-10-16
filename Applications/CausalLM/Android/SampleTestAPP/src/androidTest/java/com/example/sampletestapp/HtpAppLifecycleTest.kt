// SPDX-License-Identifier: Apache-2.0
package com.example.sampletestapp

import android.content.Intent
import android.graphics.Bitmap
import android.system.Os
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.EditText
import android.widget.TextView
import androidx.core.widget.NestedScrollView
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executor
import java.util.concurrent.TimeUnit
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Opt-in device test: requires the v81 HTP build and a staged Gemma4 model. */
@RunWith(AndroidJUnit4::class)
class HtpAppLifecycleTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private lateinit var activity: MainActivity
    private lateinit var report: File

    private fun <T> ui(block: () -> T): T {
        var result: T? = null
        instrumentation.runOnMainSync { result = block() }
        @Suppress("UNCHECKED_CAST")
        return result as T
    }

    // Access view references/state only; inference always starts through the
    // same visible controls as a user. No test entry point in the application.
    @Suppress("UNCHECKED_CAST")
    private fun <T> field(name: String): T = activity.javaClass.getDeclaredField(name).run {
        isAccessible = true
        get(activity) as T
    }

    private fun views(root: View): Sequence<View> = sequence {
        yield(root)
        if (root is ViewGroup) for (i in 0 until root.childCount) yieldAll(views(root.getChildAt(i)))
    }

    private fun click(label: String) = ui {
        val button = views(activity.window.decorView).filterIsInstance<TextView>()
            .first { it.isClickable && it.text.toString().contains(label) }
        assertTrue("Disabled control: $label", button.isEnabled)
        assertTrue(button.performClick())
    }

    private fun status(): String = ui { field<String>("statusText") }
    private fun output(): String = ui { field<String>("outputText") }

    private fun await(label: String, condition: () -> Boolean) {
        val deadline = System.nanoTime() + TimeUnit.MINUTES.toNanos(4)
        while (!condition()) {
            if (System.nanoTime() > deadline) fail("Timeout: $label; ${status()}")
            Thread.sleep(100)
        }
    }

    private fun drainWorker() {
        val done = CountDownLatch(1)
        ui { field<Executor>("engineExecutor").execute { done.countDown() } }
        assertTrue("Engine worker did not finish: ${status()}", done.await(4, TimeUnit.MINUTES))
        instrumentation.waitForIdleSync()
    }

    private fun run(name: String, messages: String, stop: Boolean = false, unload: Boolean = false): String {
        Os.setenv("NNTR_HTP_TOKEN_TRACE", File(report, "trace-$name.json").path, true)
        ui { field<EditText>("openAIMessagesField").setText(messages) }
        click("Run (streaming)")
        if (stop || unload) {
            await("first output") { output().length >= 24 || status().contains("failed", true) }
            assertFalse(status(), status().contains("failed", true))
            click(if (unload) "Unload" else "Stop")
        }
        drainWorker()
        assertEquals(if (unload) "Unloaded." else "Done.", status())
        val text = output()
        assertTrue("Empty output", text.isNotBlank())
        File(report, "$name.txt").writeText(text)
        assertTrue("Missing native token trace", File(report, "trace-$name.json").isFile)
        return text
    }

    @Test
    fun chatUsesGemma4PromptFormat() {
        val args = InstrumentationRegistry.getArguments()
        val base = args.getString("htpModelBase")
        val messagesPath = args.getString("htpMessages")
        assumeTrue(base != null && messagesPath != null)
        val context = instrumentation.targetContext
        report = File(context.filesDir, "htp-chat-validation").apply { mkdirs() }
        val messages = File(messagesPath!!).readText()
        activity = instrumentation.startActivitySync(
            Intent(context, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        ) as MainActivity
        try {
            ui { activity.window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON) }
            val defaultMessages = ui { field<String>("openAiJsonText") }
            assertEquals(org.json.JSONArray(messages).toString(),
                org.json.JSONArray(defaultMessages).toString())
            File(report, "default-openai.json").writeText(defaultMessages)
            click("Chat")
            val summaryPrompt = org.json.JSONArray(messages).getJSONObject(0).getString("content")
            val defaultChatPrompt = ui { field<EditText>("chatPromptField").text.toString() }
            assertEquals(summaryPrompt, defaultChatPrompt)
            File(report, "default-chat.txt").writeText(defaultChatPrompt)
            ui { field<EditText>("chatModelBasePathField").setText(base) }
            click("+ Open")
            drainWorker()
            assertTrue(status(), status().startsWith("Chat session opened:"))
            val shortPrompt = args.getString("htpChatPrompt") ?: "I bought a red car yesterday"
            val prompts = listOf(
                "short" to shortPrompt,
                "repeat" to shortPrompt,
                "summary" to summaryPrompt
            )
            for ((name, prompt) in prompts) {
                Os.setenv("NNTR_HTP_TOKEN_TRACE", File(report, "trace-$name.json").path, true)
                File(report, "$name-input.txt").writeText(prompt)
                ui { field<EditText>("chatPromptField").setText(prompt) }
                click("▶")
                drainWorker()
                File(report, "$name-status.txt").writeText(status())
                File(report, "$name-output.txt").writeText(output())
                assertTrue(status(), status().startsWith("Chat done."))
                assertTrue(output().isNotBlank())
                assertFalse("Unexpected thought channel in $name", output().contains("<|channel>"))
                assertFalse("Incomplete thought prefix in $name", output().contains("<channel|>"))
                assertTrue("Missing end of turn in $name", output().endsWith("<turn|>"))
                assertEquals("Generation continued past the first end of turn in $name",
                    output().length - "<turn|>".length, output().indexOf("<turn|>"))
            }
            click("Close")
            drainWorker()
            val chatSummary = File(report, "summary-output.txt").readText()
            click("OpenAI")
            ui { field<EditText>("modelBasePathField").setText(base) }
            assertEquals(org.json.JSONArray(messages).toString(),
                org.json.JSONArray(ui { field<EditText>("openAIMessagesField").text.toString() }).toString())
            assertEquals("OpenAI and Chat must render the same prompt", chatSummary,
                run("openai-summary", messages))
            click("Unload")
            drainWorker()
            File(report, "result.json").writeText(JSONObject().put("status", "passed").toString())
        } finally {
            ui { activity.finish() }
        }
    }

    @Test
    fun loadGenerateStopRepeatReloadAndReportInvalidPath() {
        val args = InstrumentationRegistry.getArguments()
        val base = args.getString("htpModelBase")
        val messagesPath = args.getString("htpMessages")
        assumeTrue("Supply htpModelBase and htpMessages for the v81 device test",
            base != null && messagesPath != null)
        val context = instrumentation.targetContext
        report = File(context.filesDir, "htp-app-validation").apply { mkdirs() }
        val messages = File(messagesPath!!).readText()
        activity = instrumentation.startActivitySync(
            Intent(context, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        ) as MainActivity
        try {
            ui { activity.window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON) }
            assertEquals("gemma4-26b-a4b", ui { field<String>("selFamily") })
            ui { field<EditText>("modelBasePathField").setText(base) }
            click("Load model")
            drainWorker()
            assertTrue(status(), status().startsWith("Loaded gemma4-26b-a4b-htp"))
            val first = run("first", messages)
            assertEquals(first, run("repeat", messages))
            val partial = run("cancel", messages, stop = true)
            assertTrue("Cancel must stop before completion", partial.length < first.length)
            assertTrue("Cancelled output differs", first.startsWith(partial))
            assertEquals(first, run("after-cancel", messages))
            click("Unload")
            drainWorker()
            assertEquals("Unloaded.", status())
            click("Load model")
            drainWorker()
            assertTrue(status(), status().startsWith("Loaded "))
            assertEquals(first, run("reload", messages))
            click("Reload model")
            drainWorker()
            assertTrue("Reload reused the previous handle: ${status()}", status().startsWith("Loaded "))
            assertEquals(first, run("explicit-reload", messages))

            val partialUnload = run("active-unload", messages, unload = true)
            assertTrue(partialUnload.length < first.length && first.startsWith(partialUnload))
            click("Load model")
            drainWorker()
            assertTrue(status(), status().startsWith("Loaded "))
            assertEquals(first, run("after-active-unload", messages))

            // A changed path must not reuse the previously loaded model.
            ui { field<EditText>("modelBasePathField").setText("$base/missing-htp-model") }
            click("Run (streaming)")
            drainWorker()
            assertTrue("Lost load diagnostic: ${status()}", status().startsWith("Load failed:"))
            val invalidPathError = status()
            ui { field<EditText>("modelBasePathField").setText(base) }
            click("Load model")
            drainWorker()
            assertTrue(status(), status().startsWith("Loaded "))
            assertEquals(first, run("after-failed-load", messages))
            ui { field<NestedScrollView>("mainScrollView").fullScroll(View.FOCUS_DOWN) }
            instrumentation.waitForIdleSync()
            File(report, "screen.png").outputStream().use {
                instrumentation.uiAutomation.takeScreenshot().compress(Bitmap.CompressFormat.PNG, 100, it)
            }
            click("Unload")
            drainWorker()
            assertEquals("Unloaded.", status())
            File(report, "result.json").writeText(JSONObject().apply {
                put("status", "passed")
                put("package", context.packageName)
                put("nativeLibraryDir", context.applicationInfo.nativeLibraryDir)
                put("modelBasePath", base)
                put("invalidPathError", invalidPathError)
                put("fullRuns", 7)
                put("cancelledRuns", 2)
            }.toString(2))
        } finally {
            ui { activity.finish() }
        }
    }
}
