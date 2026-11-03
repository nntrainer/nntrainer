// SPDX-License-Identifier: Apache-2.0
package com.example.quickdotaiapp

import com.example.quickdotai.PromptPart
import com.example.quickdotai.QuickAiChatRole
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class RawRequestTest {
    @Test
    fun parsesTopLevelArray() {
        val p = RawRequest.parse("""[{"role":"system","content":"S"},{"role":"user","content":"U"}]""")
        p as RawRequest.Parsed.Messages
        assertEquals(listOf(QuickAiChatRole.SYSTEM, QuickAiChatRole.USER), p.messages.map { it.role })
        assertEquals("U", (p.messages[1].parts.single() as PromptPart.Text).text)
        assertEquals(false, p.hasTools)
    }

    @Test
    fun parsesMessagesObjectWithToolsAndContentParts() {
        val p = RawRequest.parse(
            """{"messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"image_url","image_url":{"url":"x"}}]}],
               "tools":[{"type":"function","function":{"name":"f"}}]}"""
        )
        p as RawRequest.Parsed.Messages
        assertEquals(true, p.hasTools)
        assertEquals(listOf(PromptPart.Text("a")), p.messages.single().parts)
    }

    @Test
    fun parseAcceptsSystemOnlyMessages() {
        val p = RawRequest.parse("""[{"role":"system","content":"only"}]""")
        p as RawRequest.Parsed.Messages
        assertEquals(1, p.messages.size)
    }

    @Test
    fun unknownRolesAreSkippedAndNonRequestJsonIsOther() {
        val p = RawRequest.parse("""[{"role":"tool","content":"x"},{"role":"user","content":"u"}]""")
        assertEquals(1, (p as RawRequest.Parsed.Messages).messages.size)
        assertTrue(RawRequest.parse("""{"a":1}""") is RawRequest.Parsed.Other)
        assertTrue(RawRequest.parse("not json") is RawRequest.Parsed.Invalid)
    }

    /** The Raw screen parses on every keystroke: odd shapes must not throw out of composition. */
    @Test
    fun contentObjectIsInvalidNotACrash() {
        assertTrue(RawRequest.parse("""[{"role":"user","content":{"x":1}}]""") is RawRequest.Parsed.Invalid)
        assertTrue(RawRequest.parse("""[{"role":{"r":1},"content":"u"}]""") is RawRequest.Parsed.Invalid)
    }

    @Test
    fun withImagesPrependsToLastUserMessageOnce() {
        val base = (RawRequest.parse("""[{"role":"user","content":"u1"},{"role":"assistant","content":"a"},{"role":"user","content":"u2"}]""")
            as RawRequest.Parsed.Messages).messages
        val img = byteArrayOf(1, 2, 3)
        val out = RawRequest.withImages(base, listOf(img))
        assertEquals(listOf(PromptPart.ImageBytes(img), PromptPart.Text("u2")), out[2].parts)
        assertEquals(out, RawRequest.withImages(out, listOf(img)))
    }
}
