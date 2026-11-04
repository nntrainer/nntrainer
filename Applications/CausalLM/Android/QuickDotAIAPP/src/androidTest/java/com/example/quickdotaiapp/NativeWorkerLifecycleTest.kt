// SPDX-License-Identifier: Apache-2.0
package com.example.quickdotaiapp

import android.os.IBinder
import android.os.Parcel
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.quickdotai.PromptPart
import com.example.quickdotai.QuickAiError
import com.example.quickdotai.StreamSink
import kotlinx.serialization.json.*
import java.io.File
import java.io.FileOutputStream
import java.util.UUID
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

/** No model, NativeQuickDotAI instance, or DSP call is needed by these tests. */
@RunWith(AndroidJUnit4::class)
class NativeWorkerLifecycleTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private fun checkpoint(phase: String) {
        FileOutputStream(File(context.filesDir, "s26u_unified_app_worker_model_free_checkpoints.jsonl"), true).use {
            it.write((buildJsonObject { put("phase", phase); put("timeMs", System.currentTimeMillis()) }.toString() + "\n").toByteArray())
            it.fd.sync()
        }
    }
    private fun connect(engine: SampleNativeEngine): IBinder =
        engine.javaClass.getDeclaredMethod("connect").run {
            isAccessible = true; invoke(engine) as IBinder
        }

    @Test fun oldWorkerMustExitBeforeReplacementCanConnect() {
        val first = SampleNativeEngine(context)
        val second = SampleNativeEngine(context)
        var passed = false
        checkpoint("worker-lifecycle-start")
        try {
            connect(first)
            assertTrue(first.workerPid > 0)
            assertNotEquals(android.os.Process.myPid(), first.workerPid)
            assertTrue("Two owners must not share a native process", runCatching { connect(second) }.isFailure)
            assertEquals(0, second.workerPid)
            first.close()
            assertTrue(first.workerExitConfirmed)
            assertFalse(first.ipcDirectory.exists())
            connect(second)
            assertNotEquals(first.workerPid, second.workerPid)
            second.close()
            assertTrue(second.workerExitConfirmed)
            assertFalse(second.ipcDirectory.exists())
            passed = true
        } finally {
            runCatching { first.close() }; runCatching { second.close() }
            checkpoint("worker-lifecycle-end-passed=$passed")
        }
    }

    @Test fun largePayloadImageStreamAndErrorPreserveValuesAndCleanFiles() {
        val engine = SampleNativeEngine(context)
        val files = mutableListOf<File>()
        var passed = false
        checkpoint("worker-transport-start")
        try {
            val remote = connect(engine)
            val large = "한글 Paris ".repeat(40000)
            val request = buildJsonObject { put("text", large) }
            val callDir = File(engine.ipcDirectory, UUID.randomUUID().toString()).apply { mkdir() }
            val bundle = NativeWire.pack(context, request, files, callDir).apply {
                putString("reply_dir", callDir.path)
            }
            assertNotNull("Large text must leave the Binder parcel", bundle.getString("file"))
            assertEquals(request, NativeWire.unpack(context, bundle, files))
            val data = Parcel.obtain(); val reply = Parcel.obtain()
            try {
                data.writeInterfaceToken(NativeWire.DESCRIPTOR)
                data.writeBundle(bundle); data.writeStrongBinder(null); data.writeStrongBinder(null)
                assertTrue(remote.transact(NativeWire.METRICS, data, reply, 0))
                reply.readException()
                val error = NativeWire.unpack(context, checkNotNull(reply.readBundle()), files)
                assertFalse(error.getValue("ok").jsonPrimitive.boolean)
                assertEquals(QuickAiError.NOT_INITIALIZED, NativeWire.error(error))
            } finally { data.recycle(); reply.recycle() }
            val image = ByteArray(2 * 1024 * 1024) { (it % 251).toByte() }
            val parts = NativeWire.parts(context, listOf(PromptPart.ImageBytes(image), PromptPart.Text("describe")), files, engine.ipcDirectory)
            val restored = NativeWire.parts(parts)
            assertArrayEquals(image, File((restored[0] as PromptPart.ImageFile).absolutePath).readBytes())
            assertEquals(PromptPart.Text("describe"), restored[1])
            val output = StringBuilder(); val reasoning = StringBuilder(); val errors = mutableListOf<String?>()
            var done = 0
            val sink = NativeWire.sink(NativeWire.Callback(NativeWire.TerminalSink(object : StreamSink {
                override fun onDelta(text: String) { output.append(text) }
                override fun onReasoningDelta(text: String) { reasoning.append(text) }
                override fun onDone() { done++ }
                override fun onError(error: QuickAiError, message: String?) { errors.add(message) }
            })))
            sink.onReasoningDelta("plan"); sink.onDelta(large)
            sink.onError(QuickAiError.INFERENCE_FAILED, "test error"); sink.onDone()
            assertEquals(large, output.toString()); assertEquals("plan", reasoning.toString())
            assertEquals(listOf("test error"), errors); assertEquals(0, done)
            engine.cancel(); engine.chatCancel()
            files.forEach { assertTrue(it.delete() || !it.exists()) }
            engine.close()
            assertTrue(engine.workerExitConfirmed)
            assertFalse(engine.ipcDirectory.exists())
            passed = true
        } finally {
            files.forEach { it.delete() }; runCatching { engine.close() }
            checkpoint("worker-transport-end-passed=$passed")
        }
    }
}
