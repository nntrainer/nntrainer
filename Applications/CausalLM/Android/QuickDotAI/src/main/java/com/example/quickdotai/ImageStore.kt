// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    ImageStore.kt
 * @brief   SHA-256-based in-memory image cache for structured chat sessions.
 *
 * Identical images that arrive via different temporary file paths are
 * identified by content hash, so the conversation history can reference
 * them stably across turns without depending on filesystem paths.
 *
 * Each chat session owns its own [ImageStore]. The store is
 * cleared when the session is closed or the owning engine is unloaded.
 */
package com.example.quickdotai

import android.util.Log
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.ConcurrentHashMap

/**
 * @brief In-memory, hash-addressed image cache.
 *
 * Thread safety: all public methods are safe to call concurrently
 * (backed by [ConcurrentHashMap]). In practice the session drives
 * the store from a single worker thread, but defensive safety costs
 * almost nothing here.
 */
internal class ImageStore {

    private val cache = ConcurrentHashMap<String, ByteArray>()

    /**
     * Store raw image [bytes] and return the SHA-256 hex digest.
     * If the same content was already stored, no duplicate is created.
     */
    fun store(bytes: ByteArray): String {
        val hash = sha256Hex(bytes)
        cache.putIfAbsent(hash, bytes)
        return hash
    }

    /**
     * Read the image at [absolutePath], store its bytes, and return
     * the SHA-256 hex digest.
     *
     * @throws IllegalArgumentException if the file does not exist or
     *         is not readable.
     */
    fun store(absolutePath: String): String {
        val f = File(absolutePath)
        require(f.exists() && f.canRead()) {
            "ImageStore: file not readable: $absolutePath"
        }
        return store(f.readBytes())
    }

    /** Retrieve previously-stored bytes by their hash, or null. */
    fun get(hash: String): ByteArray? = cache[hash]

    /** Check whether a hash is present in the store. */
    fun contains(hash: String): Boolean = cache.containsKey(hash)

    /** Number of images currently cached. */
    val size: Int get() = cache.size

    /**
     * Remove all cached images and free memory. Called when the owning
     * session is closed or the engine is unloaded.
     */
    fun clear() {
        val n = cache.size
        cache.clear()
        if (n > 0) {
            Log.i(TAG, "clear(): removed $n cached image(s)")
        }
    }

    /**
     * Remove images whose hashes are NOT in [retainHashes]. Used by
     * [LiteRTLmChatSession.rebuild] to prune images that are no longer
     * referenced by the new history.
     */
    fun retainOnly(retainHashes: Set<String>) {
        val iter = cache.keys.iterator()
        var removed = 0
        while (iter.hasNext()) {
            if (iter.next() !in retainHashes) {
                iter.remove()
                removed++
            }
        }
        if (removed > 0) {
            Log.i(TAG, "retainOnly(): pruned $removed unreferenced image(s)")
        }
    }

    companion object {
        private const val TAG = "ImageStore"

        fun sha256Hex(data: ByteArray): String {
            val digest = MessageDigest.getInstance("SHA-256").digest(data)
            return digest.joinToString("") { "%02x".format(it) }
        }
    }
}
