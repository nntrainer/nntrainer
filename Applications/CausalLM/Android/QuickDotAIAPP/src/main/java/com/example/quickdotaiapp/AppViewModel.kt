// SPDX-License-Identifier: Apache-2.0
/**
 * @file  AppViewModel.kt
 * @brief Android glue around ChatController: the engine executor, the
 *        native catalog, file reading through ContentResolver and the
 *        theme preference. Survives rotation; owns the only executor.
 */
package com.example.quickdotaiapp

import android.app.Application
import android.net.Uri
import android.provider.OpenableColumns
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.quickdotai.ModelCatalog
import com.example.quickdotai.createEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.launch
import java.util.concurrent.Executors

class AppViewModel(app: Application) : AndroidViewModel(app) {
    private val engineExecutor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "QuickDotAIAPP-Engine").apply { isDaemon = true }
    }
    private val prefs = app.getSharedPreferences("quickdotaiapp", 0)

    val controller = ChatController(
        scope = viewModelScope,
        engineDispatcher = engineExecutor.asCoroutineDispatcher(),
        catalogProvider = { ModelCatalog.selectable() },   // first access dlopens the native libraries
        engineFactory = { descriptor, basePath -> createEngine(app, descriptor, basePath) },
        nativeLibDir = app.applicationInfo.nativeLibraryDir,
        examples = Examples(),
        initialTheme = ThemeMode.entries.firstOrNull { it.name == prefs.getString("theme", null) } ?: ThemeMode.SYSTEM,
    )

    fun setTheme(mode: ThemeMode) {
        prefs.edit().putString("theme", mode.name).apply()
        controller.setTheme(mode)
    }

    fun attachImages(uris: List<Uri>) {
        if (uris.isEmpty()) return
        viewModelScope.launch(Dispatchers.IO) {
            val bytes = uris.mapNotNull { uri -> readBytes(uri, MAX_IMAGE_BYTES)?.second }
            if (bytes.isNotEmpty()) controller.addDraftImages(bytes)
        }
    }

    fun attachTextFile(uri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            readBytes(uri, MAX_TEXT_BYTES)?.let { (name, bytes) -> controller.insertTextFile(name, bytes.toString(Charsets.UTF_8)) }
        }
    }

    fun attachJsonFile(uri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            readBytes(uri, MAX_TEXT_BYTES)?.let { (name, bytes) -> controller.loadJsonFile(name, bytes.toString(Charsets.UTF_8)) }
        }
    }

    /** Display name and bytes, or null with a notice when the file is too big or unreadable. */
    private fun readBytes(uri: Uri, limit: Long): Pair<String, ByteArray>? {
        val resolver = getApplication<Application>().contentResolver
        var name = uri.lastPathSegment ?: "file"
        var size = -1L
        runCatching {
            resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE), null, null, null)?.use { c ->
                if (c.moveToFirst()) {
                    c.getColumnIndex(OpenableColumns.DISPLAY_NAME).takeIf { it >= 0 }?.let { name = c.getString(it) ?: name }
                    c.getColumnIndex(OpenableColumns.SIZE).takeIf { it >= 0 }?.let { if (!c.isNull(it)) size = c.getLong(it) }
                }
            }
        }
        if (size > limit) {
            controller.notify("$name is larger than ${limit / 1024 / 1024} MB")
            return null
        }
        val bytes = try {
            resolver.openInputStream(uri)?.use { it.readBytes() }
        } catch (t: Throwable) {
            null
        }
        if (bytes == null || bytes.isEmpty()) { controller.notify("Could not read $name"); return null }
        if (bytes.size > limit) { controller.notify("$name is larger than ${limit / 1024 / 1024} MB"); return null }
        return name to bytes
    }

    override fun onCleared() {
        controller.shutdown()?.let { e -> engineExecutor.execute { runCatching { e.close() } } }
        engineExecutor.shutdown()
    }

    companion object {
        /** ponytail: 1 MB covers the 1,028-token summary prompt many times over; a streaming reader is the upgrade path. */
        const val MAX_TEXT_BYTES = 1L * 1024 * 1024
        const val MAX_IMAGE_BYTES = 32L * 1024 * 1024
    }
}
