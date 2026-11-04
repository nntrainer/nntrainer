// SPDX-License-Identifier: Apache-2.0
package com.example.quickdotaiapp

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.Binder
import android.os.Bundle
import android.os.IBinder
import android.os.Parcel
import android.os.RemoteException
import com.example.quickdotai.*
import kotlinx.serialization.json.*
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.UUID

/** The sample's native engine lives in one disposable process; the public AAR stays in-process. */
internal class SampleNativeEngine(private val context: Context) : QuickDotAI {
    internal val ipcDirectory = File(NativeWire.directory(context), UUID.randomUUID().toString()).apply { mkdirs() }
    override val kind = "native"
    @Volatile override var architecture: String? = null
        private set
    @Volatile override var chatSessionId: String? = null
        private set
    @Volatile internal var workerPid: Int = 0
        private set
    @Volatile internal var workerExitConfirmed = false
        private set
    @Volatile internal var isLoaded = false
        private set
    @Volatile private var closed = false
    private var bound = false
    @Volatile private var binder: IBinder? = null
    @Volatile private var connectionError: String? = null
    private val connected = CountDownLatch(1)
    private val died = CountDownLatch(1)
    private val clientToken = Binder()
    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            try {
                service?.linkToDeath(death, 0)
                binder = service
            } catch (t: RemoteException) {
                connectionError = t.message ?: "Native worker exited while connecting"
                if (service != null && !service.isBinderAlive) death.binderDied()
            } finally { connected.countDown() }
            if (closed && service != null) {
                Thread({
                    runCatching { transaction(service, NativeWire.CLOSE, buildJsonObject {}) }
                }, "late-native-cleanup").start()
            }
        }
        override fun onServiceDisconnected(name: ComponentName?) = Unit
        override fun onNullBinding(name: ComponentName?) { connected.countDown() }
    }
    private val death = IBinder.DeathRecipient {
        workerExitConfirmed = true
        isLoaded = false
        chatSessionId = null
        ipcDirectory.deleteRecursively()
        synchronized(ownerLock) { if (owner === this) owner = null }
        died.countDown()
    }

    private fun connect(): IBinder {
        check(!closed) { "Native engine is closed" }
        binder?.let { check(it.isBinderAlive) { "Native worker exited" }; return it }
        synchronized(ownerLock) {
            check(owner == null || owner === this) {
                "Previous native worker exit is unconfirmed; wait for cleanup or reopen the app before loading another model"
            }
            owner = this
        }
        bound = context.bindService(
            Intent(context, SampleNativeWorkerService::class.java), connection, Context.BIND_AUTO_CREATE
        )
        if (!bound) {
            synchronized(ownerLock) { if (owner === this) owner = null }
            error("Could not bind native worker")
        }
        check(connected.await(15, TimeUnit.SECONDS)) { "Timed out connecting to native worker" }
        check(!closed) { "Native engine closed while connecting" }
        check(connectionError == null) { connectionError.orEmpty() }
        val remote = checkNotNull(binder) { "Native worker returned no binder" }
        check(remote.isBinderAlive) { "Native worker exited while connecting" }
        val hello = transaction(remote, NativeWire.CONNECT, buildJsonObject {
            put("directory", ipcDirectory.absolutePath)
        }, clientToken)
        check(hello["ok"]?.jsonPrimitive?.boolean == true) {
            hello["message"]?.jsonPrimitive?.contentOrNull ?: "Native worker is already closing"
        }
        workerPid = hello.getValue("pid").jsonPrimitive.int
        return remote
    }

    private fun transaction(
        remote: IBinder, op: Int, request: JsonObject, token: IBinder? = null,
        sink: StreamSink? = null, files: MutableList<File> = mutableListOf()
    ): JsonObject {
        val callDirectory = File(ipcDirectory, UUID.randomUUID().toString()).apply {
            check(mkdir()) { "Could not create IPC request directory" }
        }
        val data = Parcel.obtain()
        val reply = Parcel.obtain()
        try {
            data.writeInterfaceToken(NativeWire.DESCRIPTOR)
            data.writeBundle(NativeWire.pack(context, request, files, callDirectory).apply {
                putString("reply_dir", callDirectory.absolutePath)
            })
            data.writeStrongBinder(token)
            data.writeStrongBinder(sink?.let { NativeWire.Callback(it) })
            check(remote.transact(op, data, reply, 0)) { "Native worker rejected operation $op" }
            reply.readException()
            return NativeWire.unpack(context, checkNotNull(reply.readBundle()), files)
        } finally {
            reply.recycle()
            data.recycle()
            files.forEach { it.delete() }
            callDirectory.deleteRecursively()
        }
    }

    private fun call(op: Int, request: JsonObject = buildJsonObject {}, sink: StreamSink? = null,
                     files: MutableList<File> = mutableListOf()): JsonObject {
        val terminal = sink?.let { NativeWire.TerminalSink(it) }
        return try {
            val result = transaction(connect(), op, request, sink = terminal, files = files)
            if (!closed) {
                architecture = result["architecture"]?.jsonPrimitive?.contentOrNull
                chatSessionId = result["session_id"]?.jsonPrimitive?.contentOrNull
                isLoaded = result["loaded"]?.jsonPrimitive?.booleanOrNull ?: false
            }
            if (terminal != null) {
                if (result.getValue("ok").jsonPrimitive.boolean) terminal.onDone()
                else terminal.onError(NativeWire.error(result), result["message"]?.jsonPrimitive?.contentOrNull)
            }
            result
        } catch (t: Exception) {
            val result = NativeWire.failure(QuickAiError.UNKNOWN, t.message ?: "Native worker failed")
            terminal?.onError(NativeWire.error(result), t.message)
            result
        } finally {
            files.forEach { it.delete() }
        }
    }

    private fun unit(op: Int, request: JsonObject = buildJsonObject {}, sink: StreamSink? = null,
                     files: MutableList<File> = mutableListOf()): BackendResult<Unit> =
        NativeWire.result(call(op, request, sink, files)) { Unit }

    override fun load(req: LoadModelRequest): BackendResult<Unit> = unit(NativeWire.LOAD,
        buildJsonObject { put("request", Json.encodeToJsonElement(LoadModelRequest.serializer(), req)) })
    override fun encode(text: String): BackendResult<FloatArray> =
        BackendResult.Err(QuickAiError.UNSUPPORTED, "The sample worker does not expose embedding inference")
    override fun unload(): BackendResult<Unit> = if (!isLoaded) BackendResult.Ok(Unit) else unit(NativeWire.UNLOAD)
    override fun metrics(): BackendResult<PerformanceMetrics> = if (!isLoaded) notLoaded() else
        NativeWire.result(call(NativeWire.METRICS)) {
            Json.decodeFromJsonElement(PerformanceMetrics.serializer(), it)
        }
    override fun openChatSession(config: QuickAiChatSessionConfig?): BackendResult<String> =
        if (!isLoaded) notLoaded() else NativeWire.result(call(NativeWire.OPEN_CHAT, buildJsonObject {
            put("config", config?.let { Json.encodeToJsonElement(QuickAiChatSessionConfig.serializer(), it) } ?: JsonNull)
        })) { it.jsonPrimitive.content }
    override fun closeChatSession(): BackendResult<Unit> = unit(NativeWire.CLOSE_CHAT)
    override fun chatRebuild(messages: List<QuickAiChatMessage>): BackendResult<Unit> =
        withMessages(messages) { request, files -> unit(NativeWire.REBUILD, request, files = files) }
    override fun runChatModelHandleStreaming(text: String, sink: StreamSink): BackendResult<QuickAiChatResult> =
        NativeWire.result(call(NativeWire.CHAT, buildJsonObject { put("text", text) }, sink)) {
            val value = it.jsonObject
            QuickAiChatResult(value.getValue("content").jsonPrimitive.content,
                value["reasoning"]?.jsonPrimitive?.contentOrNull,
                value["metrics"]?.takeUnless { m -> m is JsonNull }?.let { m ->
                    Json.decodeFromJsonElement(PerformanceMetrics.serializer(), m)
                })
        }
    override fun runMultimodalHandleStreaming(parts: List<PromptPart>, sink: StreamSink): BackendResult<Unit> {
        val files = mutableListOf<File>()
        return try {
            unit(NativeWire.MULTIMODAL, buildJsonObject { put("parts", NativeWire.parts(context, parts, files, ipcDirectory)) }, sink, files)
        } finally { files.forEach { it.delete() } }
    }
    override fun runModelHandleWithMessagesStreaming(messages: List<QuickAiChatMessage>, sink: StreamSink): BackendResult<Unit> =
        withMessages(messages) { request, files -> unit(NativeWire.MESSAGES, request, sink, files) }
    override fun runMultimodalHandleWithMessagesStreaming(messages: List<QuickAiChatMessage>, sink: StreamSink): BackendResult<Unit> =
        withMessages(messages) { request, files -> unit(NativeWire.MULTIMODAL_MESSAGES, request, sink, files) }
    override fun runModelHandleWithJsonStreaming(jsonRequest: String, sink: StreamSink): BackendResult<Unit> =
        unit(NativeWire.JSON, buildJsonObject { put("text", jsonRequest) }, sink)

    private fun <T> withMessages(messages: List<QuickAiChatMessage>, block: (JsonObject, MutableList<File>) -> T): T {
        val files = mutableListOf<File>()
        try {
            return block(buildJsonObject { put("messages", buildJsonArray {
                messages.forEach { message -> add(buildJsonObject {
                    put("role", message.role.name); put("parts", NativeWire.parts(context, message.parts, files, ipcDirectory))
                }) }
            }) }, files)
        } finally { files.forEach { it.delete() } }
    }

    private fun cancel(op: Int) {
        binder?.takeIf { it.isBinderAlive }?.let { remote ->
            runCatching { transaction(remote, op, buildJsonObject {}) }
        }
    }
    override fun cancel() = cancel(NativeWire.CANCEL)
    override fun chatCancel() = cancel(NativeWire.CHAT_CANCEL)

    override fun close() {
        if (closed && workerExitConfirmed) return
        closed = true
        cancel()
        val remote = binder
        if (remote == null && !bound) { ipcDirectory.deleteRecursively(); return }
        if (remote != null && remote.isBinderAlive) {
            // CLOSE acknowledges queuing cleanup. The exit notification, not the reply, is the barrier.
            runCatching { transaction(remote, NativeWire.CLOSE, buildJsonObject {}) }
        }
        if (bound) { context.unbindService(connection); bound = false }
        if (remote != null && !remote.isBinderAlive) death.binderDied()
        check(died.await(5, TimeUnit.SECONDS)) {
            "Native worker $workerPid did not exit; another model will not be initialized"
        }
        runCatching { remote?.unlinkToDeath(death, 0) }
        binder = null
    }

    private fun <T> notLoaded(): BackendResult<T> = BackendResult.Err(QuickAiError.NOT_INITIALIZED, "No model loaded")

    companion object {
        private val ownerLock = Any()
        private var owner: SampleNativeEngine? = null
    }
}

/** Only the sample's used operations cross this private, same-UID Binder boundary. */
internal object NativeWire {
    const val DESCRIPTOR = "com.example.quickdotaiapp.NativeWorker"
    const val CONNECT = IBinder.FIRST_CALL_TRANSACTION
    const val LOAD = CONNECT + 1
    const val UNLOAD = CONNECT + 2
    const val METRICS = CONNECT + 3
    const val OPEN_CHAT = CONNECT + 4
    const val CLOSE_CHAT = CONNECT + 5
    const val REBUILD = CONNECT + 6
    const val CHAT = CONNECT + 7
    const val MULTIMODAL = CONNECT + 8
    const val MESSAGES = CONNECT + 9
    const val MULTIMODAL_MESSAGES = CONNECT + 10
    const val JSON = CONNECT + 11
    const val CANCEL = CONNECT + 12
    const val CLOSE = CONNECT + 13
    const val CHAT_CANCEL = CONNECT + 14
    private const val INLINE_BYTES = 128 * 1024
    fun directory(context: Context) = File(context.cacheDir, "sample_native_ipc").apply { mkdirs() }
    fun pack(context: Context, value: JsonObject, files: MutableList<File>,
             target: File = directory(context)): Bundle = Bundle().apply {
        val text = value.toString()
        if (text.toByteArray(Charsets.UTF_8).size <= INLINE_BYTES) putString("json", text)
        else {
            val file = File.createTempFile("payload_", ".json", target)
            files.add(file); file.writeText(text); putString("file", file.absolutePath)
        }
    }
    fun unpack(context: Context, bundle: Bundle, files: MutableList<File>): JsonObject {
        val text = bundle.getString("json") ?: run {
            val file = File(checkNotNull(bundle.getString("file"))).canonicalFile
            check(file.path.startsWith(directory(context).canonicalPath + File.separator)) { "Invalid IPC payload path" }
            files.add(file); file.readText()
        }
        return Json.parseToJsonElement(text).jsonObject
    }
    fun replyDirectory(context: Context, bundle: Bundle): File {
        val path = File(checkNotNull(bundle.getString("reply_dir"))).canonicalFile
        check(path.path.startsWith(directory(context).canonicalPath + File.separator) && path.isDirectory) { "Invalid IPC reply directory" }
        return path
    }
    fun parts(context: Context, parts: List<PromptPart>, files: MutableList<File>,
              target: File = directory(context)) = buildJsonArray {
        parts.forEach { part -> add(buildJsonObject {
            when (part) {
                is PromptPart.Text -> put("text", part.text)
                is PromptPart.ImageFile -> put("file", part.absolutePath)
                is PromptPart.ImageBytes -> {
                    val file = File.createTempFile("image_", ".bin", target)
                    files.add(file); file.writeBytes(part.bytes); put("file", file.absolutePath)
                }
                is PromptPart.PreprocessedPixels -> error("Sample worker does not accept PreprocessedPixels")
            }
        }) }
    }
    fun parts(value: JsonElement): List<PromptPart> = value.jsonArray.map { part ->
        part.jsonObject["text"]?.let { PromptPart.Text(it.jsonPrimitive.content) }
            ?: PromptPart.ImageFile(part.jsonObject.getValue("file").jsonPrimitive.content)
    }
    fun messages(value: JsonElement): List<QuickAiChatMessage> = value.jsonArray.map { m ->
        QuickAiChatMessage(QuickAiChatRole.valueOf(m.jsonObject.getValue("role").jsonPrimitive.content),
            parts(m.jsonObject.getValue("parts")))
    }
    fun failure(error: QuickAiError, message: String?) = buildJsonObject {
        put("ok", false); put("error", error.code); put("message", message?.let(::JsonPrimitive) ?: JsonNull)
    }
    fun error(value: JsonObject) = QuickAiError.fromNativeCode(value.getValue("error").jsonPrimitive.int)
    fun <T> result(value: JsonObject, decode: (JsonElement) -> T): BackendResult<T> =
        if (value.getValue("ok").jsonPrimitive.boolean) BackendResult.Ok(decode(value["value"] ?: JsonNull))
        else BackendResult.Err(error(value), value["message"]?.jsonPrimitive?.contentOrNull)

    class TerminalSink(private val sink: StreamSink) : StreamSink {
        private val terminal = AtomicBoolean(false)
        @Synchronized override fun onDelta(text: String) { if (!terminal.get()) sink.onDelta(text) }
        @Synchronized override fun onReasoningDelta(text: String) { if (!terminal.get()) sink.onReasoningDelta(text) }
        @Synchronized override fun onDone() { if (terminal.compareAndSet(false, true)) sink.onDone() }
        @Synchronized override fun onError(error: QuickAiError, message: String?) {
            if (terminal.compareAndSet(false, true)) sink.onError(error, message)
        }
    }
    class Callback(private val sink: StreamSink) : Binder() {
        override fun onTransact(code: Int, data: Parcel, reply: Parcel?, flags: Int): Boolean {
            data.enforceInterface(DESCRIPTOR)
            when (code) {
                1 -> sink.onDelta(data.readString().orEmpty())
                2 -> sink.onReasoningDelta(data.readString().orEmpty())
                3 -> sink.onDone()
                4 -> sink.onError(QuickAiError.fromNativeCode(data.readInt()), data.readString())
                else -> return false
            }
            reply?.writeNoException(); return true
        }
    }
    fun sink(callback: IBinder): StreamSink = object : StreamSink {
        private fun send(code: Int, text: String? = null, error: QuickAiError? = null) {
            val data = Parcel.obtain(); val reply = Parcel.obtain()
            try {
                data.writeInterfaceToken(DESCRIPTOR)
                error?.let { data.writeInt(it.code) }; data.writeString(text)
                check(callback.transact(code, data, reply, 0)) { "Stream callback failed" }
                reply.readException()
            } finally { data.recycle(); reply.recycle() }
        }
        override fun onDelta(text: String) { text.chunked(16 * 1024).forEach { send(1, it) } }
        override fun onReasoningDelta(text: String) { text.chunked(16 * 1024).forEach { send(2, it) } }
        override fun onDone() = send(3)
        override fun onError(error: QuickAiError, message: String?) = send(4, message?.take(16 * 1024), error)
    }
}
