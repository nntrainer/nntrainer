// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    ModelCatalog.kt
 * @brief   Kotlin model catalog types and the ModelCatalog singleton.
 *
 * The catalog is seeded from the native C registry via
 * [NativeCausalLm.nativeQueryCatalog] (returns a JSON array). LiteRT-only
 * descriptors are appended locally because they are never registered in the
 * C layer. A static fallback is used when the native call fails (e.g.
 * emulator / missing .so).
 */
package com.example.quickdotai

import android.util.Log
import org.json.JSONArray

enum class RuntimeKind { NATIVE, LITERT }
enum class Capability { STREAMING, MESSAGES_API, MULTIMODAL, TOOL_USE, EMBEDDING, MULTI_IMAGE, SPECULATIVE }

data class ModelDescriptor(
    val id: String,
    val family: String,
    val displayName: String,
    val runtime: RuntimeKind,
    val backends: Set<BackendType>,
    val capabilities: Set<Capability>,
    val sdVariantId: String? = null,
)

/** String constants for public model ids. */
object ModelIds {
    const val QWEN3_0_6B        = "qwen3-0.6b"
    const val QWEN3_1_7B_Q40    = "qwen3-1.7b-q40"
    const val TINY_BERT         = "tiny-bert"
    const val FUNCTION_GEMMA    = "function-gemma"
    const val GEMMA4            = "gemma4"       // LiteRT only
    const val GEMMA4_CPU        = "gemma4-cpu"
    const val GEMMA4_E2B_QNN    = "gemma4-e2b-qnn"
    const val VJEPA_QNN         = "vjepa2-qnn"   // V-JEPA 2 multi-image (QNN)
    const val VJEPA_LFM2        = "vjepa-lfm2"        // V-JEPA 2 + LFM2 video (CPU)
}

object ModelCatalog {
    private const val TAG = "ModelCatalog"

    // LiteRT-only descriptor (not registered in C).
    private val liteRtDescriptors = listOf(
        ModelDescriptor(
            id = ModelIds.GEMMA4,
            family = "gemma4",
            displayName = "Gemma4 (LiteRT)",
            runtime = RuntimeKind.LITERT,
            backends = setOf(BackendType.GPU),
            capabilities = setOf(Capability.MULTIMODAL, Capability.MESSAGES_API, Capability.STREAMING),
        )
    )

    // Fallback when native query fails (public NATIVE subset).
    private val nativeFallback = listOf(
        ModelDescriptor(ModelIds.QWEN3_0_6B, "qwen3-0.6b", "Qwen3 0.6B",
            RuntimeKind.NATIVE, setOf(BackendType.CPU, BackendType.GPU),
            setOf(Capability.STREAMING, Capability.TOOL_USE)),
        ModelDescriptor(ModelIds.GEMMA4_CPU, "gemma4", "Gemma4 (CPU)",
            RuntimeKind.NATIVE, setOf(BackendType.CPU), setOf(Capability.STREAMING)),
    )

    private val catalog: List<ModelDescriptor> by lazy(LazyThreadSafetyMode.SYNCHRONIZED) { build() }

    fun all(): List<ModelDescriptor> = catalog

    private fun build(): List<ModelDescriptor> {
        val native = if (NativeCausalLm.ensureLoaded()) {
            try {
                parse(NativeCausalLm.nativeQueryCatalog())
            } catch (t: Throwable) {
                Log.e(TAG, "nativeQueryCatalog failed; using fallback", t)
                nativeFallback
            }
        } else {
            Log.w(TAG, "native library not loaded; using fallback catalog")
            nativeFallback
        }
        return native + liteRtDescriptors
    }

    private fun parse(json: String): List<ModelDescriptor> {
        val arr = JSONArray(json)
        return (0 until arr.length()).map { i ->
            val o = arr.getJSONObject(i)
            ModelDescriptor(
                id = o.getString("id"),
                family = o.getString("family"),
                displayName = o.optString("display_name", o.getString("id")),
                runtime = if (o.getInt("runtime") == 1) RuntimeKind.LITERT else RuntimeKind.NATIVE,
                backends = decodeBackends(o.getInt("backend_mask")),
                capabilities = decodeCaps(o.getInt("capabilities")),
                sdVariantId = if (o.has("sd_variant_id")) o.getString("sd_variant_id") else null,
            )
        }
    }

    private fun decodeBackends(mask: Int): Set<BackendType> =
        BackendType.values().filter { (mask shr it.ordinal) and 1 == 1 }.toSet()

    private fun decodeCaps(bits: Int): Set<Capability> = buildSet {
        if (bits and 0b000001 != 0) add(Capability.STREAMING)
        if (bits and 0b000010 != 0) add(Capability.MESSAGES_API)
        if (bits and 0b000100 != 0) add(Capability.MULTIMODAL)
        if (bits and 0b001000 != 0) add(Capability.TOOL_USE)
        if (bits and 0b010000 != 0) add(Capability.EMBEDDING)
        if (bits and 0b100000 != 0) add(Capability.MULTI_IMAGE)
        if (bits and 0b10000000 != 0) add(Capability.SPECULATIVE)
    }

    fun byId(id: String): ModelDescriptor? = all().firstOrNull { it.id == id }
    fun families(): List<String> = all().map { it.family }.distinct()
    fun runtimesFor(family: String): Set<RuntimeKind> =
        all().filter { it.family == family }.map { it.runtime }.toSet()
    fun backendsFor(family: String, rt: RuntimeKind): Set<BackendType> =
        all().filter { it.family == family && it.runtime == rt }
            .flatMap { it.backends }.toSet()
    fun resolve(family: String, rt: RuntimeKind, backend: BackendType): ModelDescriptor? =
        all().firstOrNull { it.family == family && it.runtime == rt && backend in it.backends }

    /** 사용자 선택(생성/실행) 가능 capability. 하나라도 있으면 피커에 노출. */
    private val SELECTABLE_CAPS = setOf(
        Capability.STREAMING, Capability.MESSAGES_API,
        Capability.MULTIMODAL, Capability.TOOL_USE
    )

    /** 생성/실행 가능한 모델인지(EMBEDDING 전용 모델은 false). */
    fun isSelectable(d: ModelDescriptor): Boolean =
        d.capabilities.any { it in SELECTABLE_CAPS }

    /** 피커에 노출할 모델만. all()은 전체를 그대로 유지. */
    fun selectable(): List<ModelDescriptor> = all().filter { isSelectable(it) }

    /** selectable()에서 파생한 family 목록(families()와 동일 distinct 규칙). */
    fun selectableFamilies(): List<String> = selectable().map { it.family }.distinct()
}
