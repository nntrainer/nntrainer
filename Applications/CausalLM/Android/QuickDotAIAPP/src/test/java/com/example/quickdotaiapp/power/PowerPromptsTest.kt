// SPDX-License-Identifier: Apache-2.0
package com.example.quickdotaiapp.power

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class PowerPromptsTest {
    @Test fun resourceNamesAndSizes() {
        assertEquals(listOf(256, 1024, 2048, 4096), PowerPrompts.sizes)
        assertEquals("power_prompt_256", PowerPrompts.resourceName(256))
        assertEquals("power_prompt_1k", PowerPrompts.resourceName(1024))
        assertEquals("power_prompt_2k", PowerPrompts.resourceName(2048))
        assertEquals("power_prompt_4k", PowerPrompts.resourceName(4096))
        assertEquals(4096, PowerPrompts.charsFor(1024))
    }

    /** The bundled texts are sized by 4 characters per token and end with the instruction. Unit tests run in the module directory. */
    @Test fun bundledFilesHaveTheSizeAndTheInstruction() {
        for (tokens in PowerPrompts.sizes) {
            val file = File("src/main/res/raw/${PowerPrompts.resourceName(tokens)}.txt")
            assertTrue(file.path, file.exists())
            val text = file.readText()
            val want = PowerPrompts.charsFor(tokens)
            assertTrue("${file.path}: ${text.length} chars for $want", text.length in (want * 0.9).toInt()..(want * 1.1).toInt())
            assertTrue(file.path, text.trimEnd().endsWith(PowerPrompts.INSTRUCTION))
        }
    }

    /** The Power preset for the HTP packs is the validated summarization prompt: the first user message of the bundled file. */
    @Test fun theHtpSummaryPromptIsTheFirstUserMessageOfTheBundledFile() {
        val file = File("src/main/res/raw/gemma4_moe_messages.json")
        assertTrue(file.path, file.exists())
        val prompt = PowerPrompts.firstUserContent(file.readText())!!
        assertTrue(prompt.startsWith("You are a summarization expert."))
        assertTrue(prompt.trimEnd().endsWith("</Text>"))
        assertNull(PowerPrompts.firstUserContent("not json"))
        assertNull(PowerPrompts.firstUserContent("[]"))
        assertNull(PowerPrompts.firstUserContent("""[{"role": "assistant", "content": "x"}]"""))
    }

    /** Every benchmark size is cut from the 4K text, about four characters per token, and asks for a long answer. */
    @Test fun benchPromptsAreCutFromTheFourKTextAndAskForALongAnswer() {
        val source = File("src/main/res/raw/${PowerPrompts.resourceName(4096)}.txt").readText()
        for (tokens in PowerPrompts.benchPrefill) {
            val p = PowerPrompts.bench(source, tokens)
            val want = PowerPrompts.charsFor(tokens)
            assertTrue("$tokens: ${p.length} chars for $want", p.length in (want * 0.9).toInt()..want)
            assertTrue(p.endsWith("\n\n" + PowerPrompts.BENCH_INSTRUCTION))
            assertTrue("$tokens: the summary instruction must be gone", !p.contains(PowerPrompts.INSTRUCTION))
        }
        assertEquals("a b\n\n" + PowerPrompts.BENCH_INSTRUCTION, PowerPrompts.bench("a b", 1024))
    }

    /** The engine's ceiling: max_seq_len - num_to_generate prompt tokens, num_to_generate answer tokens. */
    @Test fun packLimitsFollowTheEngineBudget() {
        assertEquals(PackLimits(2048, 256),
            PowerPrompts.packLimits("""{"init_seq_len": 2048, "max_seq_len": 2304, "num_to_generate": 256}"""))
        assertEquals(PackLimits(1153, 127),
            PowerPrompts.packLimits("""{"init_seq_len": 1280, "max_seq_len": 1280, "num_to_generate": 127}"""))
        assertEquals(PackLimits(8576, 128),        // LFM2 HTP: the prefill is chunked, init_seq_len is no bound
            PowerPrompts.packLimits("""{"init_seq_len": 64, "max_seq_len": 8704, "num_to_generate": 128}"""))
        assertNull(PowerPrompts.packLimits("""{"init_seq_len": 64, "max_seq_len": 8704}"""))
        assertNull(PowerPrompts.packLimits("not json"))
        val preset = File("../../res/gemma4/htp_chat/nntr_config.json")
        assertTrue(preset.path, preset.exists())
        assertEquals(PackLimits(2048, 256), PowerPrompts.packLimits(preset.readText()))
    }
}
