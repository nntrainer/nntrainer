// SPDX-License-Identifier: Apache-2.0
package com.example.quickdotaiapp

import android.app.Service
import android.content.Intent
import android.os.Binder
import android.os.IBinder
import android.os.Parcel
import android.os.Process
import android.system.Os
import com.example.quickdotai.*
import kotlinx.serialization.json.*
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/** One same-UID worker, recycled before another native model is initialized. */
class SampleNativeWorkerService : Service() {
    private val executor = Executors.newSingleThreadExecutor()
    private val closing = AtomicBoolean(false)
    @Volatile private var engine: NativeQuickDotAI? = null
    @Volatile private var loaded = false
    private var owner: IBinder? = null
    private var ownerDirectory: File? = null
    @Volatile private var ownerDied = false
    private val ownerDeath = IBinder.DeathRecipient { ownerDied = true; recycle() }

    override fun onCreate() {
        super.onCreate()
        // Set before constructing an engine or entering JNI. QNN models initialize explicitly.
        Os.setenv("NNTR_QNN_EAGER_CTX", "0", true)
    }

    private val binder = object : Binder() {
        override fun onTransact(code: Int, data: Parcel, reply: Parcel?, flags: Int): Boolean {
            if (code == INTERFACE_TRANSACTION) { reply?.writeString(NativeWire.DESCRIPTOR); return true }
            data.enforceInterface(NativeWire.DESCRIPTOR)
            val temporary = mutableListOf<File>()
            val bundle = checkNotNull(data.readBundle())
            val request = NativeWire.unpack(this@SampleNativeWorkerService, bundle, temporary)
            val token = data.readStrongBinder()
            val callback = data.readStrongBinder()
            val result = try {
                when (code) {
                    NativeWire.CONNECT -> synchronized(this) {
                        check(!closing.get() && owner == null) { "Native worker is already owned or closing" }
                        val directory = File(request.getValue("directory").jsonPrimitive.content).canonicalFile
                        check(directory.parentFile == NativeWire.directory(this@SampleNativeWorkerService).canonicalFile)
                        ownerDirectory = directory
                        owner = checkNotNull(token)
                        token.linkToDeath(ownerDeath, 0)
                        buildJsonObject { put("ok", true); put("pid", Process.myPid()) }
                    }
                    NativeWire.CANCEL -> { engine?.cancel(); success(Unit) }
                    NativeWire.CHAT_CANCEL -> { engine?.chatCancel(); success(Unit) }
                    NativeWire.CLOSE -> { recycle(); success(Unit) }
                    else -> {
                        check(!closing.get()) { "Native worker is closing" }
                        executor.submit<JsonObject> {
                            try { dispatch(code, request, callback) }
                            catch (t: Exception) { NativeWire.failure(QuickAiError.UNKNOWN, t.cause?.message ?: t.message) }
                        }.get()
                    }
                }
            } catch (t: Exception) {
                NativeWire.failure(QuickAiError.UNKNOWN, t.cause?.message ?: t.message)
            }
            reply?.writeNoException()
            reply?.writeBundle(NativeWire.pack(this@SampleNativeWorkerService, result, mutableListOf(),
                NativeWire.replyDirectory(this@SampleNativeWorkerService, bundle)))
            return true
        }
    }

    override fun onBind(intent: Intent?): IBinder = binder

    private fun dispatch(op: Int, request: JsonObject, callback: IBinder?): JsonObject {
        val e = if (op == NativeWire.LOAD) {
            engine ?: NativeQuickDotAI(applicationContext).also { engine = it }
        } else engine ?: return NativeWire.failure(QuickAiError.NOT_INITIALIZED, "No model loaded")
        fun sink() = NativeWire.sink(checkNotNull(callback) { "Missing stream callback" })
        val result: BackendResult<*> = when (op) {
            NativeWire.LOAD -> e.load(Json.decodeFromJsonElement(LoadModelRequest.serializer(), request.getValue("request")))
                .also { loaded = it is BackendResult.Ok }
            NativeWire.UNLOAD -> e.unload().also { if (it is BackendResult.Ok) loaded = false }
            NativeWire.METRICS -> e.metrics()
            NativeWire.OPEN_CHAT -> e.openChatSession(request["config"]?.takeUnless { it is JsonNull }?.let {
                Json.decodeFromJsonElement(QuickAiChatSessionConfig.serializer(), it)
            })
            NativeWire.CLOSE_CHAT -> e.closeChatSession()
            NativeWire.REBUILD -> e.chatRebuild(NativeWire.messages(request.getValue("messages")))
            NativeWire.CHAT -> e.runChatModelHandleStreaming(request.getValue("text").jsonPrimitive.content, sink())
            NativeWire.MULTIMODAL -> e.runMultimodalHandleStreaming(NativeWire.parts(request.getValue("parts")), sink())
            NativeWire.MESSAGES -> e.runModelHandleWithMessagesStreaming(NativeWire.messages(request.getValue("messages")), sink())
            NativeWire.MULTIMODAL_MESSAGES -> e.runMultimodalHandleWithMessagesStreaming(NativeWire.messages(request.getValue("messages")), sink())
            NativeWire.JSON -> e.runModelHandleWithJsonStreaming(request.getValue("text").jsonPrimitive.content, sink())
            else -> BackendResult.Err(QuickAiError.UNSUPPORTED, "Unsupported sample worker operation $op")
        }
        val response = when (result) {
            is BackendResult.Ok -> success(result.value)
            is BackendResult.Err -> NativeWire.failure(result.error, result.message)
        }
        return JsonObject(response + mapOf(
            "architecture" to (e.architecture?.let(::JsonPrimitive) ?: JsonNull),
            "session_id" to (e.chatSessionId?.let(::JsonPrimitive) ?: JsonNull),
            "loaded" to JsonPrimitive(loaded)
        ))
    }

    private fun success(value: Any?) = buildJsonObject {
        put("ok", true)
        put("value", when (value) {
            is String -> JsonPrimitive(value)
            is PerformanceMetrics -> Json.encodeToJsonElement(PerformanceMetrics.serializer(), value)
            is QuickAiChatResult -> buildJsonObject {
                put("content", value.content)
                put("reasoning", value.reasoning?.let(::JsonPrimitive) ?: JsonNull)
                put("metrics", value.metrics?.let { Json.encodeToJsonElement(PerformanceMetrics.serializer(), it) } ?: JsonNull)
            }
            else -> JsonNull
        })
    }

    private fun recycle() {
        if (!closing.compareAndSet(false, true)) return
        runCatching { engine?.chatCancel(); engine?.cancel() }
        executor.execute {
            try { engine?.close() }
            finally {
                engine = null
                if (ownerDied) ownerDirectory?.deleteRecursively()
                runCatching { owner?.unlinkToDeath(ownerDeath, 0) }
                stopSelf()
                // Exit releases process-lifetime QNN/HTP contexts. The client waits for Binder death.
                kotlin.system.exitProcess(0)
            }
        }
    }

    override fun onDestroy() {
        recycle()
        super.onDestroy()
    }
}
