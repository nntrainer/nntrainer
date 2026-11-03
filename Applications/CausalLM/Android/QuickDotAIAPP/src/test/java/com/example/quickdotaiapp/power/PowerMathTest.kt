// SPDX-License-Identifier: Apache-2.0
package com.example.quickdotaiapp.power

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class PowerMathTest {
    private fun s(t: Long, amps: Double, volts: Double = 4.0, bat: Int = 280, status: Int = 0, npu: Float? = null) =
        PowerSample(t, (amps * 1e6).toInt(), (volts * 1000).toInt(), bat, false, status, Float.NaN, npu, null, null)

    @Test fun wattsUseTheMagnitudeOfTheCurrent() {
        assertEquals(4.0, s(0, -1.0).watts, 1e-9)
        assertEquals(4.0, s(0, 1.0).watts, 1e-9)
    }

    @Test fun medianOfEvenAndOddLists() {
        assertEquals(2.5, PowerMath.median(listOf(4.0, 1.0, 2.0, 3.0))!!, 1e-9)
        assertEquals(2.0, PowerMath.median(listOf(3.0, 1.0, 2.0))!!, 1e-9)
        assertNull(PowerMath.median(emptyList()))
    }

    @Test fun trapezoidIntegralOverASpan() {
        // 1 W for 2 s then 3 W for 2 s, sampled every second: trapezoids 1, 1, 2, 3 = 7 J
        val samples = listOf(s(0, 0.25), s(1000, 0.25), s(2000, 0.25), s(3000, 0.75), s(4000, 0.75))
        assertEquals(7.0, PowerMath.integrateJ(samples, 0, 4000)!!, 1e-9)
        assertEquals(3.0, PowerMath.integrateJ(samples, 0, 4000, baselineW = 1.0)!!, 1e-9)
        assertNull(PowerMath.integrateJ(samples, 0, 500))              // one sample inside
    }

    @Test fun evaluateProducesTheSpecNumbers() {
        // baseline 0.8 W for 10 s, prefill 5 W for 1 s, decode 4.0 W for 4 s, 64 tokens
        val samples = buildList {
            for (t in 0..10_000 step 250) add(s(t.toLong(), 0.2, bat = 270))
            for (t in 10_250..11_000 step 250) add(s(t.toLong(), 1.25))
            for (t in 11_250..15_000 step 250) add(s(t.toLong(), 1.0, bat = 284, status = 1, npu = 41.5f))
        }
        val marks = RunMarks(baselineStartMs = 0, sendMs = 10_000, firstTokenMs = 11_000, windowEndMs = 15_000, doneMs = 15_000,
            windowTokens = 64, generatedTokens = 64, promptTokens = 1000, stopped = false)
        val run = PowerMath.evaluate(1, marks, samples)
        assertEquals(0.8, run.baselineW!!, 1e-9)
        assertEquals(4.0, run.decodeW!!, 0.05)
        assertEquals(5.0, run.prefillW!!, 0.3)
        assertEquals(16.0, run.tokensPerSec!!, 1e-9)
        // (4.0 − 0.8) W × 4 s ÷ 64 tok = 0.2 J = 200 mJ/tok (the window's edges average in neighbours)
        assertEquals(200.0, run.mJPerToken!!, 12.0)
        assertEquals(27.0, run.batteryStartC!!, 1e-9); assertEquals(28.4, run.batteryEndC!!, 1e-9)
        assertEquals(41.5f, run.npuMaxC); assertEquals(1, run.thermalStatusMax); assertTrue(!run.throttled)
        assertTrue(run.runEnergyJ!! > 20.0 && run.runEnergyJ!! < 22.0)
        assertEquals(samples.size, run.samples.size)
        assertNull(run.skipped)
    }

    @Test fun evaluateWithoutGaugeDataGivesNulls() {
        val marks = RunMarks(0, 10_000, 11_000, 15_000, 15_000, 64, 64, null, false)
        val run = PowerMath.evaluate(1, marks, listOf(s(0, 0.2), s(12_000, 0.2)))
        assertNull(run.mJPerToken); assertNull(run.baselineW); assertNull(run.runEnergyJ)
        assertEquals("no gauge data", run.skipped)
    }

    @Test fun aShortSpanWithOneSampleStillHasAMean() {
        // prefill of 300 ms at a 250 ms cadence: one sample inside (send, first] is the only reading there is
        val samples = (0..12_000 step 250).map { s(it.toLong(), if (it in 10_001..10_300) 1.25 else if (it > 10_300) 1.0 else 0.2) }
        val run = PowerMath.evaluate(1, RunMarks(0, 10_000, 10_300, 12_000, 12_000, 64, 64, null, false), samples)
        assertEquals(5.0, run.prefillW!!, 1e-9)
        assertNull(run.skipped)
    }

    @Test fun windowShorterThanASecondOrWithoutTokensGivesNoEnergyPerToken() {
        val samples = (0..12_000 step 250).map { s(it.toLong(), 1.0) }
        val short = PowerMath.evaluate(1, RunMarks(0, 10_000, 11_000, 11_500, 11_500, 64, 5, null, false), samples)
        assertNull(short.mJPerToken)
        val none = PowerMath.evaluate(1, RunMarks(0, 10_000, null, null, 12_000, 64, 0, null, false), samples)
        assertNull(none.mJPerToken); assertNull(none.tokensPerSec)
    }

    @Test fun seriesMedianSkipsInvalidRuns() {
        fun run(i: Int, mj: Double?, skipped: String? = null, throttled: Boolean = false) =
            PowerRun(i, null, skipped, null, null, null, null, mj, null, null, null, null, null, null, null, 0, throttled)
        val series = PowerSeries(listOf(run(1, 180.0), run(2, 200.0, throttled = true), run(3, null, "hot"), run(4, 190.0)))
        assertEquals(190.0, series.medianMJPerToken!!, 1e-9)
        assertEquals(1, series.throttledRuns); assertEquals(1, series.skippedRuns)
    }

    @Test fun evaluateWithAConstantGaugeGivesNoGaugeData() {
        // a gauge that never moves (unsupported property, or 0 µA throughout) cannot give an energy per token
        val samples = (0..15_000 step 250).map { s(it.toLong(), 0.0) }
        val run = PowerMath.evaluate(1, RunMarks(0, 10_000, 11_000, 15_000, 15_000, 64, 64, null, false), samples)
        assertEquals("no gauge data", run.skipped)
        assertNull(run.mJPerToken)
    }
}
