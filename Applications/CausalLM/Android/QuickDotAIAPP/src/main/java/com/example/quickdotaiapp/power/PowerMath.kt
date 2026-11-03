// SPDX-License-Identifier: Apache-2.0
/**
 * @file  PowerMath.kt
 * @brief The numbers of a run from its samples and marks (spec §4).
 *        A sample at t describes the gauge over the interval that ends at
 *        t, so averages take the samples in (from, to]; integrals take
 *        [from, to] because the trapezoids need both edges.
 */
package com.example.quickdotaiapp.power

object PowerMath {
    fun median(values: List<Double>): Double? {
        if (values.isEmpty()) return null
        val s = values.sorted()
        val n = s.size
        return if (n % 2 == 1) s[n / 2] else (s[n / 2 - 1] + s[n / 2]) / 2
    }

    /** Samples inside [fromMs, toMs]. */
    fun within(samples: List<PowerSample>, fromMs: Long, toMs: Long): List<PowerSample> =
        samples.filter { it.tMs in fromMs..toMs }

    private fun ending(samples: List<PowerSample>, fromMs: Long, toMs: Long): List<PowerSample> =
        samples.filter { it.tMs > fromMs && it.tMs <= toMs }

    /** Trapezoid integral of (P − baselineW) over [fromMs, toMs] in joules; null with fewer than two samples. */
    fun integrateJ(samples: List<PowerSample>, fromMs: Long, toMs: Long, baselineW: Double = 0.0): Double? {
        val s = within(samples, fromMs, toMs)
        if (s.size < 2) return null
        var joules = 0.0
        for (i in 1 until s.size) {
            val dt = (s[i].tMs - s[i - 1].tMs) / 1000.0
            joules += dt * ((s[i].watts - baselineW) + (s[i - 1].watts - baselineW)) / 2
        }
        return joules
    }

    /** Mean watts over (fromMs, toMs]; null without a sample. A short prefill may hold a single reading. */
    fun meanW(samples: List<PowerSample>, fromMs: Long, toMs: Long): Double? =
        ending(samples, fromMs, toMs).takeIf { it.isNotEmpty() }?.map { it.watts }?.average()

    fun maxW(samples: List<PowerSample>, fromMs: Long, toMs: Long): Double? =
        within(samples, fromMs, toMs).maxOfOrNull { it.watts }

    fun evaluate(index: Int, marks: RunMarks, samples: List<PowerSample>): PowerRun {
        val first = marks.firstTokenMs
        val end = marks.windowEndMs ?: marks.doneMs
        val baselineW = ending(samples, marks.baselineStartMs, marks.sendMs).takeIf { it.size >= 2 }
            ?.let { median(it.map { s -> s.watts }) }
        val prefillW = first?.let { meanW(samples, marks.sendMs, it) }
        val decodeW = if (first != null && end != null) meanW(samples, first, end) else null
        val tokens = minOf(marks.generatedTokens, marks.windowTokens)
        val windowMs = if (first != null && end != null) end - first else 0L
        val windowOk = tokens >= 1 && windowMs >= 1000
        val tokensPerSec = if (windowOk) tokens / (windowMs / 1000.0) else null
        val runEnd = end ?: marks.sendMs
        val whole = within(samples, marks.baselineStartMs, runEnd)
        // A reading that never changes (unsupported property, 0 µA) is no measurement at all.
        val gaugeMoves = whole.map { it.microAmps }.distinct().size > 1
        val mJPerToken = if (windowOk && baselineW != null && gaugeMoves)
            integrateJ(samples, first!!, end!!, baselineW)?.let { it / tokens * 1000 } else null
        val runEnergyJ = integrateJ(samples, marks.sendMs, runEnd)
        val runSpan = within(samples, marks.sendMs, runEnd)
        val thermalMax = runSpan.maxOfOrNull { it.thermalStatus } ?: 0
        val skipped = when {
            first == null -> "no tokens"
            baselineW == null || decodeW == null || !gaugeMoves -> "no gauge data"
            else -> null
        }
        return PowerRun(
            index = index,
            marks = marks,
            skipped = skipped,
            baselineW = baselineW,
            prefillW = prefillW,
            decodeW = decodeW,
            peakW = maxW(samples, marks.sendMs, runEnd),
            mJPerToken = mJPerToken,
            runEnergyJ = runEnergyJ,
            tokensPerSec = tokensPerSec,
            batteryStartC = samples.firstOrNull { it.tMs >= marks.baselineStartMs }?.let { it.batteryTenthsC / 10.0 },
            batteryEndC = samples.lastOrNull { it.tMs <= runEnd }?.let { it.batteryTenthsC / 10.0 },
            npuMaxC = whole.mapNotNull { it.npuC }.maxOrNull(),
            cpuMaxC = whole.mapNotNull { it.cpuC }.maxOrNull(),
            gpuMaxC = whole.mapNotNull { it.gpuC }.maxOrNull(),
            thermalStatusMax = thermalMax,
            throttled = thermalMax >= THERMAL_MODERATE,
            samples = within(samples, marks.baselineStartMs, marks.doneMs ?: runEnd),
        )
    }
}
