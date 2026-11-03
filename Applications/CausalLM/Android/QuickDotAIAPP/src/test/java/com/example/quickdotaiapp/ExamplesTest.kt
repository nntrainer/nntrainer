// SPDX-License-Identifier: Apache-2.0
package com.example.quickdotaiapp

import com.example.quickdotai.BackendType
import com.example.quickdotai.Capability
import com.example.quickdotai.ModelDescriptor
import com.example.quickdotai.ModelIds
import com.example.quickdotai.RuntimeKind
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ExamplesTest {
    private val npu = ModelDescriptor("fake-npu", "fake-npu", "Fake NPU",
        RuntimeKind.NATIVE, setOf(BackendType.NPU), setOf(Capability.STREAMING, Capability.MESSAGES_API))
    private val litert = ModelDescriptor(ModelIds.GEMMA4, "gemma4", "Gemma4 (LiteRT)",
        RuntimeKind.LITERT, setOf(BackendType.GPU), setOf(Capability.MULTIMODAL, Capability.MESSAGES_API, Capability.STREAMING))
    private val qwen = ModelDescriptor(ModelIds.QWEN3_0_6B, "qwen3-0.6b", "Qwen3 0.6B",
        RuntimeKind.NATIVE, setOf(BackendType.CPU), setOf(Capability.STREAMING, Capability.TOOL_USE))
    private val htp = ModelDescriptor(ModelIds.GEMMA4_26B_A4B_HTP, "gemma4-26b-a4b", "Gemma4 26B A4B (HTP)",
        RuntimeKind.NATIVE, setOf(BackendType.NPU), setOf(Capability.STREAMING, Capability.MESSAGES_API))
    private val lfm2Htp = ModelDescriptor(ModelIds.LFM2_24BA2B_HTP, "lfm2", "LFM2 24B A2B (HTP)",
        RuntimeKind.NATIVE, setOf(BackendType.NPU), setOf(Capability.STREAMING, Capability.MESSAGES_API))
    private val examples = Examples()

    @Test
    fun rawJsonFollowsTheMessagesApiCapability() {
        assertEquals(examples.rawJsonFor(null), examples.rawJsonFor(npu))   // unknown id + MESSAGES_API
        assertTrue(examples.rawJsonFor(npu).trimStart().startsWith("["))     // messages array
        assertTrue(examples.rawJsonFor(qwen).trimStart().startsWith("{"))    // full request object (tools go here)
    }

    @Test
    fun requiredPathIsTheFolderForNativeAndTheFileForLiteRt() {
        assertEquals("/m/fake-npu/nntr_config.json", Examples.requiredPath(npu, "/m/"))
        assertEquals("/m/gemma-4-E2B-it/gemma-4-E2B-it.litertlm", Examples.requiredPath(litert, "/m"))
        assertEquals("/m/gemma-4-E2B-it/gemma-4-E2B-it.litertlm", Examples.modelPath(litert, "/m"))
    }

    @Test
    fun runtimeLabels() {
        assertEquals("Quick.AI", Examples.runtimeLabel(RuntimeKind.NATIVE))
        assertEquals("LiteRT", Examples.runtimeLabel(RuntimeKind.LITERT))
    }

    /** The two HTP presets: the pack folders the public API names, the HTP badge, and a one-shot chat. */
    @Test
    fun htpPresetsHaveTheirFoldersBadgeAndOneShotChat() {
        assertEquals("/m/gemma4-26b-a4b-htp-w4a32/nntr_config.json", Examples.requiredPath(htp, "/m"))
        assertEquals("/m/lfm2-24ba2b-htp-w4a32", Examples.modelPath(lfm2Htp, "/m/"))
        assertEquals("HTP", Examples.badge(htp))
        assertEquals("NPU", Examples.badge(npu))
        assertTrue(Examples.isOneShot(htp))
        assertTrue(Examples.isOneShot(lfm2Htp))
        assertFalse(Examples.isOneShot(npu))
        assertFalse(Examples.isOneShot(null))
    }
}
