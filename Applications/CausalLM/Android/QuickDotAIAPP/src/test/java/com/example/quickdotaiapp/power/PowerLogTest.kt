// SPDX-License-Identifier: Apache-2.0
package com.example.quickdotaiapp.power

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PowerLogTest {
    private val meta = SessionMeta("2026-11-02T19:30:00+09:00", "SM-S948N", "16", "gemma4", "LiteRT", "GPU")
    private val run = PowerRun(1, RunMarks(0, 10_000, 11_000, 15_000, 15_000, 512, 512, 1017, false), null,
        0.9, 5.2, 4.6, 5.3, 181.25, 65.0, 20.4, 27.0, 28.4, 41.5f, 55.0f, null, 1, false)

    @Test fun csvRowFollowsTheHeader() {
        val log = PowerSessionLog(meta, PowerSettings(), 4.0, listOf(run))
        assertEquals(PowerLog.CSV_HEADER.split(',').size, PowerLog.csvRow(log, run).split(',').size)
        assertEquals(
            "2026-11-02T19:30:00+09:00,gemma4,LiteRT,GPU,fixed-1024,1017,512,512,20.40,5.200,4.600,0.900,5.300,181.3,65.00,27.0,28.4,41.5,55.0,,1,false,",
            PowerLog.csvRow(log, run),
        )
        assertTrue(PowerLog.csv(log).startsWith(PowerLog.CSV_HEADER + "\n"))
    }

    @Test fun skippedRunRowHasEmptyNumbersAndTheReason() {
        val skipped = PowerRun(2, null, "hot, 95 s", null, null, null, null, null, null, null, null, null, null, null, null, 0, false)
        val row = PowerLog.csvRow(PowerSessionLog(meta, PowerSettings(), null, listOf(skipped)), skipped)
        assertTrue(row, row.endsWith(",0,false,hot; 95 s"))          // commas in the reason become semicolons
        assertEquals(PowerLog.CSV_HEADER.split(',').size, row.split(',').size)
    }

    @Test fun jsonRoundTripsTheWholeSession() {
        val sample = PowerSample(100, -1_050_000, 4310, 284, false, 1, 0.83f, 41.5f, 55.0f, null)
        val log = PowerSessionLog(meta, PowerSettings(prompt = PromptSource.Composer, repeats = 1), 4.0, listOf(run.copy(samples = listOf(sample))))
        val text = PowerLog.toJson(log)
        assertTrue(text, text.contains("\"composer\"") && text.contains("\"microAmps\":-1050000"))
        assertEquals(log, PowerLog.fromJson(text))
    }

    @Test fun jsonAcceptsAnUnsupportedHeadroom() {
        val sample = PowerSample(100, -500_000, 4300, 280, false, 0, Float.NaN, null, null, null)
        val log = PowerSessionLog(meta, PowerSettings(), null, listOf(run.copy(samples = listOf(sample))))
        val back = PowerLog.fromJson(PowerLog.toJson(log))
        assertTrue(back.runs.single().samples.single().headroom.isNaN())
    }

    @Test fun theHtpSummaryPromptSourceRoundTripsAndLabels() {
        val log = PowerSessionLog(meta, PowerSettings(prompt = PromptSource.HtpSummary), null, listOf(run))
        val text = PowerLog.toJson(log)
        assertTrue(text, text.contains("\"htp-summary\""))
        assertEquals(log, PowerLog.fromJson(text))
        assertEquals("htp-summary", PromptSource.HtpSummary.label())
        assertEquals("htp-summary", PowerLog.csvRow(log, run).split(",")[4])
    }
}
